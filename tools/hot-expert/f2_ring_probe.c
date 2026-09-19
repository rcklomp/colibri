/* f2_ring_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, the
 * microbenchmark that has to run BEFORE the engine is touched
 * (tools/hot-expert/F2-STREAM-PREFILL-DESIGN-2026-09-19.md §4).
 *
 * The question this answers, and nothing else: with GLM-5.3's REAL expert
 * shapes and the engine's OWN ring + dispatch code, what does one wave of
 * streamed experts cost -- fill alone, compute alone, and the two in the
 * order the engine will run them -- per card and on three cards at once?
 *
 * Why it is not staging_fill_probe.c. That probe measured a two-leg pipeline of
 * its own construction: CPU memcpy into a HOST_VISIBLE|HOST_COHERENT staging
 * buffer, then a compute shader reading that staging buffer over PCIe (38.6
 * GB/s aggregate, three cards). This one drives the ENGINE's own ring and the
 * ENGINE's own expert-group dispatch -- it links c/backend_vulkan.c and calls
 * coli_vk_ring_* and coli_vk_expert_group_issue*, so what it times is the code
 * that will run, not a model of it.
 *
 * Shapes, from the served model's config.json (text_config): D (hidden) 4096,
 * I (moe_intermediate) 2048, int4 group-64 -> gate/up [2048,4096], down
 * [4096,2048], 14.16 MB per expert. Rows per expert is swept because the
 * design's schedule gives a streamed expert only 2.7-6.0 rows in a 512-row
 * chunk (10.66 % of 512*8 routed calls spread over 73-163 distinct experts).
 *
 * Source bytes are the served model's own shard files, mmap'd read-only, at
 * fixed-seed random 4 KiB-aligned offsets -- the scatter pattern a
 * non-resident miss set actually has, and page-cache-resident DRAM, which is
 * where the engine reads its experts from. The bytes are NOT a real expert:
 * weights and scale planes are raw shard bytes, so the arithmetic is
 * representative and the outputs are meaningless. Nothing is written to the
 * model directory or anywhere else.
 *
 * Output: one `ROW ` line per configuration, `INFO `/`WARN ` for machine
 * facts. GB = 1e9 bytes throughout, matching every other probe in this
 * directory.
 *
 * Run 1 (2026-09-19, `where=vram`) refuted the design's own assumption: a CPU
 * memcpy into ReBAR write-combined VRAM reaches only 11.9-13.1 GB/s per card
 * and 16.3 GB/s on three, against the 38.6 GB/s a staging ring reached. CPU
 * stores over PCIe are not GPU DMA reads over PCIe. So the probe now sweeps
 * BOTH placements -- `where=vram` (CPU writes VRAM, shader reads VRAM) and
 * `where=host` (CPU writes cached DRAM, shader reads it over PCIe) -- and the
 * engine takes whichever the rig says is faster.
 *
 *   f2_ring_probe <shard_dir> <qmatmul.spv> [slots] [min_seconds] [reps] [where]
 *       where: 0 = vram only, 1 = host only, 2 = both (default)
 *
 * F1_DECODE_PROBE=1 (env, checked before the sweeps above): the falsifier
 * for FRANKEN-ENGINE-PLAN item F1 (record sec F1-STEP0). F1's whole
 * question is a batch-1 latency this file's existing sweeps do not answer
 * (they measure sustained bandwidth over >= 1 s of back-to-back waves, not
 * one window's wall latency from "routing known" to "outputs on host"). At
 * k = 1, 2, 3, 4, 6 experts per window, two device splits (round-robin
 * across dev0/dev2/dev3, and all k on dev2, the card with its own PCIe
 * link), >= F1_DECODE_WINDOWS (default 2000) independent windows, each a
 * fresh random expert per slot from the same page-cache-resident pool
 * shard_open() already built (>= 2 GB, printed as INFO pool_gb=) so nothing
 * is CPU-cache-hot across windows: fill (parallel over OMP threads) + submit
 * (issue) + compute-and-readback (take), timed separately and as one total.
 * Compared, in the probe's own output, against record sec F1-STEP0's
 * measured CPU cost: 1.747 ms/window at 3.12 experts/window. Env:
 * F1_DECODE_KS="1,2,3,4,6", F1_DECODE_WINDOWS=2000, F1_DECODE_WHERE=1 (host
 * ring, matching F2's shipped placement -- record sec F2a: a CPU write into
 * ReBAR VRAM is 16.3 GB/s on three cards against 58.6 for a host-RAM ring
 * the GPU reads over PCIe).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../c/backend_vulkan.h"

#define DIM_D 4096
#define DIM_I 2048
#define GS    64
#define FMT   4

/* plane sizes for one int4-g64 expert at these shapes */
#define GU_W ((size_t)DIM_I * (DIM_D / 2))          /* 4 194 304 */
#define GU_S ((size_t)DIM_I * (DIM_D / GS) * 4)     /*   524 288 */
#define DN_W ((size_t)DIM_D * (DIM_I / 2))          /* 4 194 304 */
#define DN_S ((size_t)DIM_D * (DIM_I / GS) * 4)     /*   524 288 */
#define EXPERT_BYTES (2 * (GU_W + GU_S) + DN_W + DN_S)

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}
static double median(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }
/* v must already be sorted ascending (nearest-rank on a linear interpolation
 * between the two bracketing samples -- fine for n in the thousands). */
static double percentile_sorted(const double *v, int n, double p) {
    if (n <= 0) return 0.0;
    if (n == 1) return v[0];
    double idx = p * (n - 1);
    int lo = (int)idx; int hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - lo;
    return v[lo] + (v[hi] - v[lo]) * frac;
}
/* "1,2,3,4,6" -> out[]={1,2,3,4,6}, returns count (capped at cap). */
static int parse_int_list(const char *s, int *out, int cap) {
    char buf[256]; int n = 0;
    strncpy(buf, s, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    for (char *tok = strtok(buf, ","); tok && n < cap; tok = strtok(NULL, ","))
        out[n++] = atoi(tok);
    return n;
}

/* ---- the shard pool: every .safetensors in the dir, mmap'd read-only ---- */
#define MAX_SHARDS 128
static struct { void *base; size_t len; } g_sh[MAX_SHARDS];
static int g_nsh;
/* a "block" is one expert's worth of contiguous bytes we can carve six planes from */
static struct { int sh; size_t off; } *g_blk;
static size_t g_nblk;

static int shard_open(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "cannot open %s\n", dir); return 0; }
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d)) && g_nsh < MAX_SHARDS) {
        const char *dot = strrchr(e->d_name, '.');
        if (!dot || strcmp(dot, ".safetensors")) continue;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        int fd = open(path, O_RDONLY);
        if (fd < 0) continue;
        struct stat st;
        if (fstat(fd, &st) || (size_t)st.st_size < EXPERT_BYTES * 2) { close(fd); continue; }
        void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (m == MAP_FAILED) continue;
        g_sh[g_nsh].base = m; g_sh[g_nsh].len = (size_t)st.st_size; g_nsh++;
    }
    closedir(d);
    if (!g_nsh) { fprintf(stderr, "no usable shards in %s\n", dir); return 0; }
    /* block list: every EXPERT_BYTES-sized window, 4 KiB aligned */
    size_t cap = 0;
    for (int i = 0; i < g_nsh; i++) cap += g_sh[i].len / EXPERT_BYTES;
    g_blk = malloc(cap * sizeof(*g_blk));
    if (!g_blk) return 0;
    for (int i = 0; i < g_nsh; i++)
        for (size_t o = 0; o + EXPERT_BYTES <= g_sh[i].len; o += EXPERT_BYTES) {
            g_blk[g_nblk].sh = i;
            g_blk[g_nblk].off = o & ~(size_t)4095;
            if (g_blk[g_nblk].off + EXPERT_BYTES <= g_sh[i].len) g_nblk++;
        }
    printf("INFO shards=%d blocks=%zu expert_bytes=%zu pool_gb=%.2f\n",
           g_nsh, g_nblk, EXPERT_BYTES, (double)g_nblk * EXPERT_BYTES / 1e9);
    return g_nblk > 0;
}

/* six plane pointers carved out of block b */
static void block_planes(size_t b, const uint8_t **gw, const float **gs,
                         const uint8_t **uw, const float **us,
                         const uint8_t **dw, const float **ds) {
    const uint8_t *p = (const uint8_t *)g_sh[g_blk[b].sh].base + g_blk[b].off;
    *gw = p;                       p += GU_W;
    *gs = (const float *)p;        p += GU_S;
    *uw = p;                       p += GU_W;
    *us = (const float *)p;        p += GU_S;
    *dw = p;                       p += DN_W;
    *ds = (const float *)p;
}

static uint64_t rng_s = 42;
static uint64_t rnd(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s;
}

/* ---- the fill task list: (dev, slot, block), flat across the active cards --- */
typedef struct { int dev, slot; size_t blk; } Task;

static double fill_tasks(const Task *t, int n) {
    const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < n; i++) {
        const uint8_t *gw, *uw, *dw; const float *gs, *us, *ds;
        block_planes(t[i].blk, &gw, &gs, &uw, &us, &dw, &ds);
        coli_vk_ring_fill(t[i].dev, t[i].slot, gw, gs, uw, us, dw, ds);
    }
    return now_s() - t0;
}

static const char *WHERE_NAME[2] = {"vram", "host"};

static int run_where(int where, int slots, double mins, int reps, int has2, int has3) {
    const char *wn = WHERE_NAME[where];
    const int devs[3] = {0, 2, 3};
    int got[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++) {
        if (k == 1 && !has2) continue;
        if (k == 2 && !has3) continue;
        got[k] = coli_vk_ring_init(devs[k], slots, FMT, DIM_D, DIM_I, GS, where);
        printf("INFO ring where=%s dev%d slots=%d bytes=%.3f GB slot_bytes=%zu\n",
               wn, devs[k], got[k], (double)coli_vk_ring_bytes(devs[k]) / 1e9,
               coli_vk_ring_slot_bytes(devs[k]));
        if (got[k] < slots)
            printf("WARN ring where=%s dev%d got %d of %d slots -- memory short\n", wn, devs[k], got[k], slots);
    }
    { double u, b;
      if (coli_vk_mem_budget(&u, &b))  printf("INFO budget-after where=%s dev0 free=%.2f GB\n", wn, b - u);
      if (has2 && coli_vk_mem_budget2(&u, &b)) printf("INFO budget-after where=%s dev2 free=%.2f GB\n", wn, b - u);
      if (has3 && coli_vk_mem_budget3(&u, &b)) printf("INFO budget-after where=%s dev3 free=%.2f GB\n", wn, b - u); }
    if (!got[0]) { printf("WARN where=%s: no ring on dev0, skipping\n", wn); coli_vk_ring_free(); return 1; }

    /* -------- (a) FILL ONLY: one card at a time, then all three together ----- */
    for (int cfg = 0; cfg < 4; cfg++) {
        /* cfg 0,1,2 = dev0/dev2/dev3 alone; cfg 3 = all three concurrently */
        int use[3] = {0, 0, 0};
        if (cfg < 3) { if (!got[cfg]) continue; use[cfg] = 1; }
        else for (int k = 0; k < 3; k++) use[k] = got[k] > 0;
        int nuse = use[0] + use[1] + use[2];
        if (!nuse) continue;

        int ntask = 0;
        for (int k = 0; k < 3; k++) if (use[k]) ntask += got[k];
        Task *t = malloc((size_t)ntask * sizeof(Task));
        double *v = malloc((size_t)reps * sizeof(double));
        for (int r = 0; r < reps; r++) {
            double acc = 0.0, bytes = 0.0;
            int it = 0;
            do {
                int n = 0;
                for (int k = 0; k < 3; k++)
                    if (use[k]) for (int s = 0; s < got[k]; s++)
                        t[n].dev = devs[k], t[n].slot = s, t[n].blk = rnd() % g_nblk, n++;
                acc += fill_tasks(t, n);
                bytes += (double)n * EXPERT_BYTES;
                it++;
            } while (acc < mins);
            v[r] = bytes / acc / 1e9;
        }
        double gbps = median(v, reps);
        if (cfg < 3)
            printf("ROW kind=fill where=%s slots=%d cards=1 dev=dev%d gbps=%.3f\n", wn, got[cfg], devs[cfg], gbps);
        else
            printf("ROW kind=fill where=%s slots=%d cards=%d dev=all gbps=%.3f\n", wn, ntask, nuse, gbps);
        free(t); free(v);
    }

    /* -------- (b) COMPUTE ONLY, and (c) FILL->COMPUTE, over rows per expert -- */
    const int rowsweep[] = {1, 3, 6, 12};
    for (unsigned ri = 0; ri < sizeof(rowsweep) / sizeof(rowsweep[0]); ri++) {
        const int R = rowsweep[ri];
        const int total = slots * R;
        float *x = malloc((size_t)total * DIM_D * sizeof(float));
        float *y = malloc((size_t)total * DIM_D * sizeof(float));
        int *rows = malloc((size_t)slots * sizeof(int));
        ColiVkTensor **gt = malloc((size_t)slots * sizeof(void *));
        ColiVkTensor **ut = malloc((size_t)slots * sizeof(void *));
        ColiVkTensor **dt = malloc((size_t)slots * sizeof(void *));
        if (!x || !y || !rows || !gt || !ut || !dt) { fprintf(stderr, "OOM\n"); return 2; }
        for (size_t i = 0; i < (size_t)total * DIM_D; i++) x[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
        for (int s = 0; s < slots; s++) rows[s] = R;

        /* one fill so the slots hold real bytes before the compute is timed */
        { Task *t = malloc((size_t)(3 * slots) * sizeof(Task)); int n = 0;
          for (int k = 0; k < 3; k++) if (got[k] > 0)
              for (int s = 0; s < got[k]; s++) t[n].dev = devs[k], t[n].slot = s, t[n].blk = rnd() % g_nblk, n++;
          fill_tasks(t, n); free(t); }

        /* (b) compute only: issue all active cards, then join all -- the shape
         *     ffn_moe already uses (c/glm53.c, the issue/issue2/issue3 round). */
        double *v = malloc((size_t)reps * sizeof(double));
        for (int r = 0; r < reps; r++) {
            double acc = 0.0; int it = 0;
            do {
                const double t0 = now_s();
                int i0 = 0, i2 = 0, i3 = 0;
                if (got[0]) { for (int s = 0; s < got[0]; s++) coli_vk_ring_tensors(0, s, &gt[s], &ut[s], &dt[s]);
                              i0 = coli_vk_expert_group_issue(gt, ut, dt, rows, got[0], x); }
                if (got[1]) { for (int s = 0; s < got[1]; s++) coli_vk_ring_tensors(2, s, &gt[s], &ut[s], &dt[s]);
                              i2 = coli_vk_expert_group_issue2(gt, ut, dt, rows, got[1], x); }
                if (got[2]) { for (int s = 0; s < got[2]; s++) coli_vk_ring_tensors(3, s, &gt[s], &ut[s], &dt[s]);
                              i3 = coli_vk_expert_group_issue3(gt, ut, dt, rows, got[2], x); }
                if (i0) coli_vk_expert_group_take(y);
                if (i2) coli_vk_expert_group_take2(y);
                if (i3) coli_vk_expert_group_take3(y);
                acc += now_s() - t0; it++;
            } while (acc < mins);
            v[r] = acc / it * 1e3;
        }
        printf("ROW kind=compute where=%s slots=%d rows=%d cards=%d dev=all ms=%.3f\n",
               wn, slots, R, (got[0] > 0) + (got[1] > 0) + (got[2] > 0), median(v, reps));

        /* (c) the engine's order: fill a wave on every card, then issue+take. */
        Task *t = malloc((size_t)(3 * slots) * sizeof(Task));
        for (int r = 0; r < reps; r++) {
            double acc = 0.0, bytes = 0.0; int it = 0;
            do {
                int n = 0;
                for (int k = 0; k < 3; k++) if (got[k] > 0)
                    for (int s = 0; s < got[k]; s++) t[n].dev = devs[k], t[n].slot = s, t[n].blk = rnd() % g_nblk, n++;
                const double t0 = now_s();
                fill_tasks(t, n);
                int i0 = 0, i2 = 0, i3 = 0;
                if (got[0]) { for (int s = 0; s < got[0]; s++) coli_vk_ring_tensors(0, s, &gt[s], &ut[s], &dt[s]);
                              i0 = coli_vk_expert_group_issue(gt, ut, dt, rows, got[0], x); }
                if (got[1]) { for (int s = 0; s < got[1]; s++) coli_vk_ring_tensors(2, s, &gt[s], &ut[s], &dt[s]);
                              i2 = coli_vk_expert_group_issue2(gt, ut, dt, rows, got[1], x); }
                if (got[2]) { for (int s = 0; s < got[2]; s++) coli_vk_ring_tensors(3, s, &gt[s], &ut[s], &dt[s]);
                              i3 = coli_vk_expert_group_issue3(gt, ut, dt, rows, got[2], x); }
                if (i0) coli_vk_expert_group_take(y);
                if (i2) coli_vk_expert_group_take2(y);
                if (i3) coli_vk_expert_group_take3(y);
                acc += now_s() - t0;
                bytes += (double)n * EXPERT_BYTES;
                it++;
            } while (acc < mins);
            v[r] = acc / it * 1e3;                 /* ms per wave */
        }
        { double ms = median(v, reps);
          int n = 0; for (int k = 0; k < 3; k++) n += got[k];
          printf("ROW kind=wave where=%s slots=%d rows=%d cards=%d dev=all experts=%d ms=%.3f gbps=%.3f\n",
                 wn, slots, R, (got[0] > 0) + (got[1] > 0) + (got[2] > 0), n, ms,
                 (double)n * EXPERT_BYTES / (ms / 1e3) / 1e9); }
        free(t); free(v); free(x); free(y); free(rows); free(gt); free(ut); free(dt);
    }

    coli_vk_ring_free();
    return 0;
}

#define F1_MAX_K 16

/* least-squares fit of ms = a + b*k over the (k, median_ms) points collected
 * for one split -- b is the marginal ms/expert, a is what is left at k=0,
 * i.e. the fixed per-window overhead (fence wait, command buffer record,
 * dispatch) the ring API charges no matter how many experts ride along.
 * Two points give an exact line; more than two average out window-to-window
 * jitter in the underlying medians. Prints the fit AND says outright when
 * the fixed term dominates over the range actually tested -- the "read the
 * ring API's per-wave overhead honestly" instruction, not left to whoever
 * reads the ROW lines afterward. */
static void print_linear_fit(const char *label, const double *ks, const double *ms, int n) {
    if (n < 2) return;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) { sx += ks[i]; sy += ms[i]; sxx += ks[i] * ks[i]; sxy += ks[i] * ms[i]; }
    const double denom = n * sxx - sx * sx;
    if (denom == 0.0) { printf("INFO f1decode %s: fit refused (all k identical)\n", label); return; }
    const double b = (n * sxy - sx * sy) / denom;   /* ms per extra expert */
    const double a = (sy - b * sx) / n;              /* ms at k=0: the fixed part */
    const double kmax = ks[n - 1];
    const double fixed_share = a / (a + b * kmax) * 100.0;
    printf("INFO f1decode %s: linear fit ms ~= %.4f + %.4f*k (fixed=%.4f ms/window, "
           "marginal=%.4f ms/expert) -- fixed is %.1f%% of the total at k=%.0f%s\n",
           label, a, b, a, b, fixed_share, kmax,
           fixed_share > 50.0 ? " -- FIXED-COST DOMINATED" : "");
}

/* F1's falsifier (record sec F1-STEP0, plan rev 20 task B): at batch 1, what
 * does an on-demand streamed miss cost end to end -- fill, submit, compute,
 * readback -- per window, against the 1.747 ms/window the CPU path pays
 * today at ~3.12 experts/window? See the file header comment
 * (F1_DECODE_PROBE) for the full design. Runs AFTER shard_open() and the
 * device inits in main(), so g_blk/g_nblk (the >= 2 GB pool) already exist. */
static int run_f1_decode_probe(int has2, int has3) {
    const char *ks_env = getenv("F1_DECODE_KS");
    int ks[F1_MAX_K];
    const int nk = parse_int_list(ks_env ? ks_env : "1,2,3,4,6", ks, F1_MAX_K);
    const int windows = getenv("F1_DECODE_WINDOWS") ? atoi(getenv("F1_DECODE_WINDOWS")) : 2000;
    const int where = getenv("F1_DECODE_WHERE") ? atoi(getenv("F1_DECODE_WHERE")) : 1;
    const int slots_per_dev = 8;   /* >= max k tested here, headroom for split=alldev2 */

    printf("INFO f1decode ks=%s windows=%d where=%s slots_per_dev=%d\n",
           ks_env ? ks_env : "1,2,3,4,6", windows, WHERE_NAME[where], slots_per_dev);
    printf("INFO f1decode reference: CPU path today = 1.747 ms/window at "
           "k~3.12 experts/window (record sec F1-STEP0, req 8, 18669-token decode, "
           "chain f1s009191720)\n");

    const int devs[3] = {0, 2, 3};
    int got[3] = {0, 0, 0};
    got[0] = coli_vk_ring_init(devs[0], slots_per_dev, FMT, DIM_D, DIM_I, GS, where);
    if (has2) got[1] = coli_vk_ring_init(devs[1], slots_per_dev, FMT, DIM_D, DIM_I, GS, where);
    if (has3) got[2] = coli_vk_ring_init(devs[2], slots_per_dev, FMT, DIM_D, DIM_I, GS, where);
    for (int k = 0; k < 3; k++)
        printf("INFO f1decode ring dev%d slots=%d\n", devs[k], got[k]);
    if (got[0] < 1) { fprintf(stderr, "FATAL f1decode: no ring on dev0\n"); coli_vk_ring_free(); return 2; }

    float *x = malloc((size_t)F1_MAX_K * DIM_D * sizeof(float));
    float *y = malloc((size_t)F1_MAX_K * DIM_D * sizeof(float));
    if (!x || !y) { fprintf(stderr, "OOM\n"); return 2; }
    for (int i = 0; i < F1_MAX_K * DIM_D; i++)
        x[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    int rowsbuf[F1_MAX_K]; for (int i = 0; i < F1_MAX_K; i++) rowsbuf[i] = 1;   /* S = 1 row/expert */

    for (int splitmode = 0; splitmode < 2; splitmode++) {
        const char *splitname = splitmode == 0 ? "roundrobin" : "alldev2";
        if (splitmode == 1 && got[1] < 1) {
            printf("WARN f1decode split=alldev2: no dev2 ring, skipping\n");
            continue;
        }
        double fit_k[F1_MAX_K], fit_ms[F1_MAX_K]; int fit_n = 0;

        for (int ik = 0; ik < nk; ik++) {
            const int k = ks[ik];
            if (k < 1 || k > F1_MAX_K) { fprintf(stderr, "f1decode: k=%d out of range, skipping\n", k); continue; }

            /* the (dev, slot) assignment for this k/split is the same every
             * window -- only the block (which expert) is re-rolled per
             * window -- so build it once. */
            int adev[F1_MAX_K], aslot[F1_MAX_K];
            int cnt[3] = {0, 0, 0};
            if (splitmode == 0) {
                for (int i = 0; i < k; i++) {
                    int d = i % 3, tries = 0;
                    while (got[d] < 1 && tries < 3) { d = (d + 1) % 3; tries++; }
                    adev[i] = d; aslot[i] = cnt[d]++;
                }
            } else {
                for (int i = 0; i < k; i++) { adev[i] = 1; aslot[i] = i; cnt[1]++; }
            }
            int maxcnt = cnt[0]; if (cnt[1] > maxcnt) maxcnt = cnt[1]; if (cnt[2] > maxcnt) maxcnt = cnt[2];
            if (maxcnt > slots_per_dev) {
                printf("WARN f1decode split=%s k=%d: needs %d slots on one device, have %d -- skipping\n",
                       splitname, k, maxcnt, slots_per_dev);
                continue;
            }

            double *tot = malloc((size_t)windows * sizeof(double));
            double *fil = malloc((size_t)windows * sizeof(double));
            double *sub = malloc((size_t)windows * sizeof(double));
            double *cmp = malloc((size_t)windows * sizeof(double));
            if (!tot || !fil || !sub || !cmp) { fprintf(stderr, "OOM\n"); return 2; }

            Task tasks[F1_MAX_K];
            ColiVkTensor *g0[F1_MAX_K], *u0[F1_MAX_K], *d0[F1_MAX_K];
            ColiVkTensor *g1[F1_MAX_K], *u1[F1_MAX_K], *d1[F1_MAX_K];
            ColiVkTensor *g2[F1_MAX_K], *u2[F1_MAX_K], *d2[F1_MAX_K];

            for (int w = 0; w < windows; w++) {
                int n0 = 0, n1 = 0, n2 = 0;
                for (int i = 0; i < k; i++) {
                    tasks[i].dev = devs[adev[i]]; tasks[i].slot = aslot[i]; tasks[i].blk = rnd() % g_nblk;
                    if (adev[i] == 0) n0++; else if (adev[i] == 1) n1++; else n2++;
                }

                const double t0 = now_s();
                fill_tasks(tasks, k);                              /* fill: 8 OMP threads, random blocks */
                const double t1 = now_s();

                int i0 = 0, i1 = 0, i2 = 0;
                if (n0) { for (int s = 0; s < n0; s++) coli_vk_ring_tensors(devs[0], s, &g0[s], &u0[s], &d0[s]);
                          i0 = coli_vk_expert_group_issue(g0, u0, d0, rowsbuf, n0, x); }
                if (n1) { for (int s = 0; s < n1; s++) coli_vk_ring_tensors(devs[1], s, &g1[s], &u1[s], &d1[s]);
                          i1 = coli_vk_expert_group_issue2(g1, u1, d1, rowsbuf, n1, x); }
                if (n2) { for (int s = 0; s < n2; s++) coli_vk_ring_tensors(devs[2], s, &g2[s], &u2[s], &d2[s]);
                          i2 = coli_vk_expert_group_issue3(g2, u2, d2, rowsbuf, n2, x); }
                const double t2 = now_s();                          /* submit: issue only, no wait */

                if (i0) coli_vk_expert_group_take(y);
                if (i1) coli_vk_expert_group_take2(y);
                if (i2) coli_vk_expert_group_take3(y);
                const double t3 = now_s();                          /* compute+readback: take blocks on the fence */

                fil[w] = (t1 - t0) * 1e3;
                sub[w] = (t2 - t1) * 1e3;
                cmp[w] = (t3 - t2) * 1e3;
                tot[w] = (t3 - t0) * 1e3;
            }

            qsort(tot, windows, sizeof(double), cmpd);
            qsort(fil, windows, sizeof(double), cmpd);
            qsort(sub, windows, sizeof(double), cmpd);
            qsort(cmp, windows, sizeof(double), cmpd);
            double mean_tot = 0; for (int w = 0; w < windows; w++) mean_tot += tot[w]; mean_tot /= windows;

            printf("ROW kind=f1decode split=%s k=%d windows=%d "
                   "total_ms_median=%.4f total_ms_p90=%.4f total_ms_mean=%.4f "
                   "fill_ms_median=%.4f submit_ms_median=%.4f compute_ms_median=%.4f "
                   "vs_cpu_ms=1.747\n",
                   splitname, k, windows,
                   tot[windows / 2], percentile_sorted(tot, windows, 0.90), mean_tot,
                   fil[windows / 2], sub[windows / 2], cmp[windows / 2]);

            if (fit_n < F1_MAX_K) { fit_k[fit_n] = (double)k; fit_ms[fit_n] = tot[windows / 2]; fit_n++; }
            free(tot); free(fil); free(sub); free(cmp);
        }
        char lbl[64]; snprintf(lbl, sizeof(lbl), "split=%s", splitname);
        print_linear_fit(lbl, fit_k, fit_ms, fit_n);
    }

    free(x); free(y);
    coli_vk_ring_free();
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/home/ronald/models/GLM-5.3-Flash-colibri-int4-g64";
    const char *spv = argc > 2 ? argv[2] : "/home/ronald/src/colibri/c/shaders/qmatmul.spv";
    const int slots   = argc > 3 ? atoi(argv[3]) : 48;
    const double mins = argc > 4 ? atof(argv[4]) : 1.0;
    const int reps    = argc > 5 ? atoi(argv[5]) : 5;
    const int wsel    = argc > 6 ? atoi(argv[6]) : 2;   /* 0=vram 1=host 2=both */
    if (slots < 1 || slots > 64) { fprintf(stderr, "slots must be 1..64 (the submit cap)\n"); return 2; }

    if (!coli_vk_init(spv)) { fprintf(stderr, "coli_vk_init failed (%s)\n", spv); return 2; }
    const int has2 = coli_vk_init_dev2(spv, -1);
    const int has3 = coli_vk_init_dev3(spv, -1);
    printf("INFO devices dev0=1 dev2=%d dev3=%d\n", has2, has3);
    { double u, b;
      if (coli_vk_mem_budget(&u, &b))  printf("INFO budget dev0 used=%.2f budget=%.2f free=%.2f GB\n", u, b, b - u);
      if (has2 && coli_vk_mem_budget2(&u, &b)) printf("INFO budget dev2 used=%.2f budget=%.2f free=%.2f GB\n", u, b, b - u);
      if (has3 && coli_vk_mem_budget3(&u, &b)) printf("INFO budget dev3 used=%.2f budget=%.2f free=%.2f GB\n", u, b, b - u); }
    if (!shard_open(dir)) return 2;
#ifdef _OPENMP
    printf("INFO omp_max_threads=%d\n", omp_get_max_threads());
#endif

    /* Step 2a: a model-free check of the CLAMPED gate_up kernels, so a broken
     * pipeline, layout or push-constant block is found in a minute here
     * instead of an hour into a gateway outage.
     *
     * Two properties, neither of which needs a CPU dequant reference:
     *   (a) the clamped pipeline with an enormous limit must reproduce the
     *       unclamped one -- same SPIR-V source, same expression, the clamp
     *       never binds. This is what proves the second pipeline, its layout
     *       and its seventh push constant are all wired correctly.
     *   (b) the clamped pipeline at the model's real limit (10.0) must
     *       actually change the result -- otherwise the knob is a no-op and
     *       would pass (a) while doing nothing.
     * Both the per-row (R=1) and the tiled (R=8) pipeline are exercised. */
    if (getenv("F2_CLAMP_CHECK")) {
        const int devs[3] = {0, 2, 3};
        int ok = coli_vk_ring_init(0, 2, FMT, DIM_D, DIM_I, GS, 1);
        if (ok < 1) { printf("WARN clamp-check: no ring on dev0\n"); coli_vk_shutdown(); return 2; }
        /* The bandwidth sweeps above feed the shaders raw shard bytes for BOTH
         * the weights and the scale planes, which is fine for timing and
         * useless for numerics: reinterpreted as f32 those bytes are mostly
         * huge or NaN, the kernel returns NaN, and every |difference| compares
         * false so a broken clamp would read as "no difference". (That is
         * exactly what the first run of this check printed.) Here the int4
         * WEIGHT nibbles stay real shard bytes and the scales are synthetic
         * and sane -- 0.1, chosen so the pre-activation lands around +-25 and
         * the model's real limit of 10 actually binds. */
        const size_t sfl = (size_t)DIM_I * (DIM_D / GS);   /* == DIM_D * (DIM_I/GS) */
        float *sc = malloc(sfl * sizeof(float));
        if (!sc) { printf("WARN clamp-check: OOM\n"); return 2; }
        for (size_t i = 0; i < sfl; i++) sc[i] = 0.1f;
        {
            const uint8_t *gw, *uw, *dw; const float *gs_, *us_, *ds_;
            block_planes(rnd() % g_nblk, &gw, &gs_, &uw, &us_, &dw, &ds_);
            (void)gs_; (void)us_; (void)ds_;
            coli_vk_ring_fill(devs[0], 0, gw, sc, uw, sc, dw, sc);
        }
        /* Step 2c's question, answered for free while we are here: the engine
         * picks the TILED gate_up/down pipeline for an expert with more than
         * one row and the PER-ROW one for a single row (vk_tile_ok4, keyed on
         * rows[c]). A prefill chunk of 128 gives a cold expert 1 row and a
         * chunk of 512 gives it 2-6, so the chunk silently switches kernels
         * for the same (token, expert) pair. Do the two kernels agree bit for
         * bit on the SAME row? x below is generated by the same formula for
         * both R, so row 0's input is identical and row 0's output is
         * comparable. */
        float *row0_perrow = malloc((size_t)DIM_D * sizeof(float));
        for (int ri = 0; ri < 2; ri++) {
            const int R = ri ? 8 : 1;
            float *x = malloc((size_t)R * DIM_D * sizeof(float));
            float *y0 = calloc((size_t)R * DIM_D, sizeof(float));
            float *y1 = calloc((size_t)R * DIM_D, sizeof(float));
            float *y2 = calloc((size_t)R * DIM_D, sizeof(float));
            int rows[1] = {R};
            ColiVkTensor *g = NULL, *u = NULL, *d = NULL;
            coli_vk_ring_tensors(0, 0, &g, &u, &d);
            for (int i = 0; i < R * DIM_D; i++) x[i] = (float)((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
            int i0 = 0, i1 = 0, i2 = 0, n1 = 0, n2 = 0;
            coli_vk_set_swiglu_limit(0.0f);
            i0 = coli_vk_expert_group_issue(&g, &u, &d, rows, 1, x);
            if (i0) i0 = coli_vk_expert_group_take(y0);
            n1 = coli_vk_set_swiglu_limit(1e30f);
            i1 = coli_vk_expert_group_issue(&g, &u, &d, rows, 1, x);
            if (i1) i1 = coli_vk_expert_group_take(y1);
            n2 = coli_vk_set_swiglu_limit(10.0f);
            i2 = coli_vk_expert_group_issue(&g, &u, &d, rows, 1, x);
            if (i2) i2 = coli_vk_expert_group_take(y2);
            coli_vk_set_swiglu_limit(0.0f);
            printf("INFO clamp-check rows=%d take0=%d take1=%d take2=%d devs_clamped=%d/%d "
                   "g=%p ring_where=%d\n", R, i0, i1, i2, n1, n2, (void *)g, coli_vk_ring_where(0));
            double m01 = 0, m02 = 0, ref = 0;
            long bad = 0;
            for (int i = 0; i < R * DIM_D; i++) {
                if (!isfinite(y0[i]) || !isfinite(y1[i]) || !isfinite(y2[i])) { bad++; continue; }
                double a = fabs((double)y0[i] - y1[i]); if (a > m01) m01 = a;
                double b = fabs((double)y0[i] - y2[i]); if (b > m02) m02 = b;
                double r = fabs((double)y0[i]);         if (r > ref) ref = r;
            }
            printf("ROW kind=clamp-check rows=%d pipeline=%s nonfinite=%ld/%d "
                   "max|y_unclamped-y_limit1e30|=%.6g max|y_unclamped-y_limit10|=%.6g "
                   "max|y_unclamped|=%.6g verdict=%s\n",
                   R, R > 1 ? "tiled" : "per-row", bad, R * DIM_D, m01, m02, ref,
                   bad ? "FAIL(non-finite outputs)"
                   : ref == 0.0 ? "FAIL(all-zero output -- the check proves nothing)"
                   : (m01 == 0.0 && m02 > 0.0) ? "PASS"
                   : (m01 != 0.0) ? "FAIL(huge-limit not inert)"
                                  : "FAIL(limit 10 changed nothing)");
            if (R == 1) {
                memcpy(row0_perrow, y0, (size_t)DIM_D * sizeof(float));
            } else {
                double mr = 0; long nf = 0;
                for (int i = 0; i < DIM_D; i++) {
                    if (!isfinite(y0[i]) || !isfinite(row0_perrow[i])) { nf++; continue; }
                    double dd = fabs((double)y0[i] - row0_perrow[i]); if (dd > mr) mr = dd;
                }
                printf("ROW kind=tile-vs-perrow rows_tiled=%d nonfinite=%ld "
                       "max|y_tiled_row0 - y_perrow_row0|=%.6g max|y|=%.6g verdict=%s\n",
                       R, nf, mr, ref,
                       mr == 0.0 ? "IDENTICAL (the tile threshold is not a numerics change)"
                                 : "DIFFERS (the chunk switches kernels and the kernels disagree)");
            }
            free(x); free(y0); free(y1); free(y2);
        }
        free(row0_perrow);
        free(sc);
        coli_vk_ring_free();
        coli_vk_shutdown();
        printf("INFO done (clamp-check)\n");
        return 0;
    }

    /* F1's falsifier (record sec F1-STEP0, plan rev 20 task B) -- mutually
     * exclusive with F2_CLAMP_CHECK above and with the wsel sweeps below;
     * neither existing mode is touched. */
    if (getenv("F1_DECODE_PROBE")) {
        int rc = run_f1_decode_probe(has2, has3);
        coli_vk_shutdown();
        printf("INFO done (f1decode)\n");
        return rc;
    }

    if (wsel == 0 || wsel == 2) run_where(0, slots, mins, reps, has2, has3);
    if (wsel == 1 || wsel == 2) run_where(1, slots, mins, reps, has2, has3);

    coli_vk_shutdown();
    printf("INFO done\n");
    return 0;
}
