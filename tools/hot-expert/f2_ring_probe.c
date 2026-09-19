/* f2_ring_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, the
 * microbenchmark that has to run BEFORE the engine is touched
 * (tools/hot-expert/F2-STREAM-PREFILL-DESIGN-2026-09-19.md §4).
 *
 * The question this answers, and nothing else: with GLM-5.3's REAL expert
 * shapes and the engine's OWN ring + dispatch code, what does one wave of
 * streamed experts cost -- fill alone, compute alone, and the two in the
 * order the engine will run them -- per card and on three cards at once?
 *
 * Why it is not staging_fill_probe.c. That probe measured a two-leg pipeline:
 * CPU memcpy into a HOST_VISIBLE|HOST_COHERENT staging buffer, then a compute
 * shader reading that staging buffer over PCIe (38.6 GB/s aggregate, three
 * cards). The engine's weight path has only ONE leg: pick_memtype() picks
 * HOST_VISIBLE|HOST_COHERENT|DEVICE_LOCAL (ReBAR write-combined VRAM) and
 * upload_tensor memcpys straight into it, so for a ring slot the fill IS the
 * upload and the shader then reads local VRAM at ~900 GB/s, not PCIe. That is
 * a different measurement and it may be faster or slower; assuming either way
 * is what this probe exists to avoid. It links c/backend_vulkan.c and calls
 * coli_vk_ring_* and coli_vk_expert_group_issue*, so what it times is the code
 * the engine will run, not a model of it.
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
 *   f2_ring_probe <shard_dir> <qmatmul.spv> [slots] [min_seconds] [reps]
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

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/home/ronald/models/GLM-5.3-Flash-colibri-int4-g64";
    const char *spv = argc > 2 ? argv[2] : "/home/ronald/src/colibri/c/shaders/qmatmul.spv";
    const int slots  = argc > 3 ? atoi(argv[3]) : 48;
    const double mins = argc > 4 ? atof(argv[4]) : 1.0;
    const int reps   = argc > 5 ? atoi(argv[5]) : 5;
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

    const int devs[3] = {0, 2, 3};
    int got[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++) {
        if (k == 1 && !has2) continue;
        if (k == 2 && !has3) continue;
        got[k] = coli_vk_ring_init(devs[k], slots, FMT, DIM_D, DIM_I, GS);
        printf("INFO ring dev%d slots=%d bytes=%.3f GB slot_bytes=%zu\n",
               devs[k], got[k], (double)coli_vk_ring_bytes(devs[k]) / 1e9,
               coli_vk_ring_slot_bytes(devs[k]));
        if (got[k] < slots)
            printf("WARN ring dev%d got %d of %d slots -- VRAM short\n", devs[k], got[k], slots);
    }
    { double u, b;
      if (coli_vk_mem_budget(&u, &b))  printf("INFO budget-after dev0 free=%.2f GB\n", b - u);
      if (has2 && coli_vk_mem_budget2(&u, &b)) printf("INFO budget-after dev2 free=%.2f GB\n", b - u);
      if (has3 && coli_vk_mem_budget3(&u, &b)) printf("INFO budget-after dev3 free=%.2f GB\n", b - u); }
    if (!got[0]) { fprintf(stderr, "no ring on dev0, nothing to measure\n"); return 2; }

#ifdef _OPENMP
    printf("INFO omp_max_threads=%d\n", omp_get_max_threads());
#endif

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
            printf("ROW kind=fill slots=%d cards=1 dev=dev%d gbps=%.3f\n", got[cfg], devs[cfg], gbps);
        else
            printf("ROW kind=fill slots=%d cards=%d dev=all gbps=%.3f\n", ntask, nuse, gbps);
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
        printf("ROW kind=compute slots=%d rows=%d cards=%d dev=all ms=%.3f\n",
               slots, R, (got[0] > 0) + (got[1] > 0) + (got[2] > 0), median(v, reps));

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
          printf("ROW kind=wave slots=%d rows=%d cards=%d dev=all experts=%d ms=%.3f gbps=%.3f\n",
                 slots, R, (got[0] > 0) + (got[1] > 0) + (got[2] > 0), n, ms,
                 (double)n * EXPERT_BYTES / (ms / 1e3) / 1e9); }
        free(t); free(v); free(x); free(y); free(rows); free(gt); free(ut); free(dt);
    }

    coli_vk_ring_free();
    coli_vk_shutdown();
    printf("INFO done\n");
    return 0;
}
