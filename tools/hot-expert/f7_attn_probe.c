/* f7_attn_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F7, the
 * microbenchmark that has to run BEFORE the engine's numbers are believed
 * (tools/hot-expert/F7-MLA-ATTN-GPU-DESIGN-2026-09-19.md, deliverable B).
 *
 * The question it answers, and nothing else: at GLM-5.3's REAL attention
 * shapes, what does one layer-chunk of the batched DSA attention core cost on
 * dev0 through the ENGINE'S OWN backend (it links c/backend_vulkan.c and calls
 * coli_vk_mla_attn, the function glm53.c calls), and how far is its output from
 * a scalar CPU reference?
 *
 * Shapes, from the served model's config.json text_config: H = 64 heads,
 * L (kv_lora) = 512, V (v_head) = 256, qk_nope = 256 -> scale = 1/16,
 * width = index_topk 2048 / kpool 4 expanded + tail = 2051, 11 DSA layers.
 * Swept: S (rows per chunk) 128 and 512, seen (prefix length) 2 000 / 9 000 /
 * 18 439 -- 2 000 is BELOW width, which is the dense-selection regime the
 * shallow half of the gate exercises.
 *
 * Data is fixed-seed random: latents and absorbed queries ~U(-1,1)/sqrt(L),
 * kvb_v raw int4-g64 nibbles with ~U(0,1)*2^-6 group scales, selection a
 * fixed-seed random subset of [0, seen) per row (sorted, so the gather pattern
 * has the same locality the indexer's output has). Nothing is read from or
 * written to the model directory.
 *
 * The CPU reference is a plain scalar rewrite of glm53.c's CPU nest -- same
 * slots in the same order, max-subtraction, `double` softmax total, pool, and
 * a scalar int4-g64 dequant. It is NOT bit-identical to the engine's
 * vectorised CPU kernels and is not meant to be: it is there to catch a wrong
 * nibble order, a wrong scale plane, a transposed index or a dropped slot,
 * which are all O(1) errors, not ulp errors. Bit-identity is not available
 * here by construction -- this whole path reorders summation, which is why it
 * ships behind GLM53_MLA_ATTN_GPU and is gated by KL and not by memcmp.
 *
 *   f7_attn_probe <shaders_dir> [reps]
 *
 * Output: one `ROW ` line per configuration, `INFO `/`WARN ` for machine facts.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "../../c/backend_vulkan.h"

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static uint32_t S_RND = 12345u;
static float rnd(void) { S_RND = S_RND * 1103515245u + 12345u;
    return (float)((S_RND >> 9) & 0x7fffff) / 4194304.0f - 1.0f; }
static uint32_t rndu(void) { S_RND = S_RND * 1103515245u + 12345u; return S_RND >> 8; }

#define H 64
#define L 512
#define V 256
#define QKN 256
#define WIDTH 2051
#define GS 64

/* scalar reference for ONE row: score -> softmax -> pool -> kvb_v value rows */
static void ref_row(float *ctx, const float *absorbed_row, const float *lat,
                    const int *slot, int used, int seen,
                    const uint8_t *vw, const float *vsc, float scale,
                    float *score, float *pooled) {
    (void)seen;
    const int ng = L / GS, packed = L / 2;
    for (int h = 0; h < H; h++) {
        const float *q = absorbed_row + (size_t)h * L;
        float top = -INFINITY;
        for (int u = 0; u < used; u++) {
            const float *cj = lat + (size_t)slot[u] * L;
            float dot = 0.0f;
            for (int d = 0; d < L; d++) dot += q[d] * cj[d];
            score[u] = dot * scale;
            if (score[u] > top) top = score[u];
        }
        float *result = ctx + (size_t)h * V;
        memset(result, 0, (size_t)V * sizeof(float));
        if (!used) continue;
        double total = 0.0;
        for (int u = 0; u < used; u++) { score[u] = expf(score[u] - top); total += score[u]; }
        memset(pooled, 0, (size_t)L * sizeof(float));
        for (int u = 0; u < used; u++) {
            const float w = (float)(score[u] / total);
            const float *cj = lat + (size_t)slot[u] * L;
            for (int d = 0; d < L; d++) pooled[d] += w * cj[d];
        }
        for (int v = 0; v < V; v++) {
            const int row = h * V + v;
            const uint8_t *w = vw + (size_t)row * packed;
            const float *sc = vsc + (size_t)row * ng;
            float a = 0.0f;
            for (int g = 0; g < ng; g++) {
                float part = 0.0f;
                for (int i = g * GS; i < (g + 1) * GS; i++) {
                    const uint8_t byte = w[i >> 1];
                    const int nib = (i & 1) ? (int)(byte >> 4) : (int)(byte & 0xF);
                    part += pooled[i] * (float)(nib - 8);
                }
                a += part * sc[g];
            }
            result[v] = a;
        }
    }
}

/* ---- F7_REPLAY: one REAL layer-chunk from the engine, against float64 -------
 *
 * The sweep above runs on random data whose softmax is nearly uniform (dots of
 * ~1e-3), and Fable's float64 arbitration showed that hides the defect: the CPU
 * fp32 path is within 1.7e-05 mean KL / 100 % top-1 of float64 on the shallow
 * packet while the GPU path is ~160x further away. So this mode replays a chunk
 * the ENGINE actually computed -- real absorbed queries, real latents, the real
 * per-row `selected` lists and the real kvb_v -- written by
 * GLM53_MLA_ATTN_DUMP, and compares `context` three ways:
 *
 *   engine CPU fp32 (in the file) vs float64   <- the yardstick
 *   coli_vk_mla_attn (this run)     vs float64   <- the thing under test
 *
 * per ROW and per HEAD, not aggregated, so the answer is "from row X, head Y"
 * and not "something is off". The float64 reference here is the same nest as
 * glm53.c's GLM53_MLA_ATTN_REF64 arm, including the kvb_v value rows. */
struct Replay {
    int tokens, nh, nl, nv, width, seen, base, vfmt, vgs, vrows, vpacked, vng;
    float scale;
    float *absorbed, *latent, *cpuctx, *vsc;
    int *selected;
    uint8_t *vw;
};

static int replay_load(const char *path, struct Replay *r) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "replay: cannot open %s\n", path); return 0; }
    int hdr[13];
    if (fread(hdr, sizeof hdr, 1, f) != 1 || hdr[0] != 0x50443746 || hdr[1] != 1) {
        fprintf(stderr, "replay: bad header in %s\n", path); fclose(f); return 0; }
    r->tokens = hdr[2]; r->nh = hdr[3]; r->nl = hdr[4]; r->nv = hdr[5];
    r->width = hdr[6]; r->seen = hdr[7]; r->base = hdr[8];
    r->vfmt = hdr[9]; r->vgs = hdr[10]; r->vrows = hdr[11]; r->vpacked = hdr[12];
    if (fread(&r->vng, 4, 1, f) != 1 || fread(&r->scale, 4, 1, f) != 1) { fclose(f); return 0; }
    size_t na = (size_t)r->tokens * r->nh * r->nl, nl = (size_t)r->seen * r->nl;
    size_t ns = (size_t)r->tokens * r->width, nc = (size_t)r->tokens * r->nh * r->nv;
    size_t nw = (size_t)r->vrows * r->vpacked, nsc = (size_t)r->vrows * r->vng;
    r->absorbed = malloc(na * 4); r->latent = malloc(nl * 4);
    r->selected = malloc(ns * 4); r->cpuctx = malloc(nc * 4);
    r->vw = malloc(nw);           r->vsc = malloc(nsc * 4);
    if (!r->absorbed || !r->latent || !r->selected || !r->cpuctx || !r->vw || !r->vsc) {
        fprintf(stderr, "replay: OOM\n"); fclose(f); return 0; }
    int ok = fread(r->absorbed, 4, na, f) == na && fread(r->latent, 4, nl, f) == nl
          && fread(r->selected, 4, ns, f) == ns && fread(r->cpuctx, 4, nc, f) == nc
          && fread(r->vw, 1, nw, f) == nw && fread(r->vsc, 4, nsc, f) == nsc;
    fclose(f);
    if (!ok) { fprintf(stderr, "replay: short read\n"); return 0; }
    return 1;
}

/* float64 context for one row, all heads: the GLM53_MLA_ATTN_REF64 nest. */
static void replay_ref64(const struct Replay *r, int t, const int *slot, int used, double *ctx64) {
    const int nh = r->nh, nl = r->nl, nv = r->nv, gs = r->vgs, ng = r->vng, packed = r->vpacked;
    const double scale = r->scale;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < nh; h++) {
        double *score = malloc((size_t)(used > 0 ? used : 1) * sizeof(double));
        double *pooled = malloc((size_t)nl * sizeof(double));
        const float *q = r->absorbed + ((size_t)t * nh + h) * nl;
        double *res = ctx64 + (size_t)h * nv;
        for (int v = 0; v < nv; v++) res[v] = 0.0;
        if (used > 0 && score && pooled) {
            double top = -HUGE_VAL;
            for (int u = 0; u < used; u++) {
                const float *cj = r->latent + (size_t)slot[u] * nl;
                double dot = 0.0;
                for (int d = 0; d < nl; d++) dot += (double)q[d] * (double)cj[d];
                score[u] = dot * scale;
                if (score[u] > top) top = score[u];
            }
            double total = 0.0;
            for (int u = 0; u < used; u++) { score[u] = exp(score[u] - top); total += score[u]; }
            for (int d = 0; d < nl; d++) pooled[d] = 0.0;
            for (int u = 0; u < used; u++) {
                const double w = score[u] / total;
                const float *cj = r->latent + (size_t)slot[u] * nl;
                for (int d = 0; d < nl; d++) pooled[d] += w * (double)cj[d];
            }
            for (int v = 0; v < nv; v++) {
                const int row = h * nv + v;
                const uint8_t *w = r->vw + (size_t)row * packed;
                const float *sc = r->vsc + (size_t)row * ng;
                double a = 0.0;
                for (int g = 0; g < ng; g++) {
                    double part = 0.0;
                    for (int i = g * gs; i < (g + 1) * gs && i < nl; i++) {
                        const uint8_t by = w[i >> 1];
                        const int nib = (i & 1) ? (int)(by >> 4) : (int)(by & 0xF);
                        part += pooled[i] * (double)(nib - 8);
                    }
                    a += part * (double)sc[g];
                }
                res[v] = a;
            }
        }
        free(pooled); free(score);
    }
}

static int replay_main(const char *path, const char *shaders) {
    (void)shaders;
    struct Replay r;
    if (!replay_load(path, &r)) return 2;
    printf("INFO replay %s: tokens=%d base=%d seen=%d width=%d H=%d L=%d V=%d fmt=%d gs=%d\n",
           path, r.tokens, r.base, r.seen, r.width, r.nh, r.nl, r.nv, r.vfmt, r.vgs);

    /* compact `selected` exactly as glm53.c's mla_attn_gpu_try does */
    int *slot = malloc((size_t)r.tokens * r.width * sizeof(int));
    int *used = malloc((size_t)r.tokens * sizeof(int));
    for (int t = 0; t < r.tokens; t++) {
        const int *ch = r.selected + (size_t)t * r.width;
        int *d = slot + (size_t)t * r.width, n = 0;
        for (int i = 0; i < r.width; i++) {
            const int at = ch[i];
            if (at < 0 || at >= r.seen) continue;
            d[n++] = at;
        }
        used[t] = n;
    }
    printf("INFO used[]: first=%d last=%d  (dense regime iff used[t] == base+t+1)\n",
           used[0], used[r.tokens - 1]);
    int dense_ok = 1;
    for (int t = 0; t < r.tokens; t++)
        if (used[t] != r.base + t + 1 && r.base + t + 1 <= r.width) { dense_ok = 0; break; }
    printf("INFO per-row causal used[] matches base+t+1 below width: %s\n", dense_ok ? "yes" : "NO");

    float *gpuctx = malloc((size_t)r.tokens * r.nh * r.nv * sizeof(float));
    ColiVkTensor *vt = NULL;
    const int sb = getenv("GLM53_MLA_ATTN_SB") ? atoi(getenv("GLM53_MLA_ATTN_SB")) : 128;
    const double t0 = now_ms();
    if (!coli_vk_mla_attn(&vt, r.vw, r.vsc, r.vfmt, r.vgs, gpuctx, r.absorbed, r.latent,
                          slot, used, r.tokens, r.nh, r.nl, r.nv, r.width, r.seen, sb, r.scale)) {
        printf("WARN coli_vk_mla_attn returned 0 on the replay\n"); return 2;
    }
    printf("INFO replay gpu call %.1f ms (sb=%d)\n", now_ms() - t0, sb);

    /* Triage over EVERY row first, GPU against the engine's own CPU fp32 -- no
     * float64 needed, so it is cheap enough to cover all 512 rows and all 64
     * heads and catch a rare bad row that a 25-row sample would miss. The
     * float64 reference then runs only on the worst rows. */
    const int nall = r.tokens * r.nh * r.nv;
    (void)nall;
    double *rowrel = malloc((size_t)r.tokens * sizeof(double));
    int *rowhead = malloc((size_t)r.tokens * sizeof(int));
    double sumrel = 0, maxrel = 0; int maxrow = -1;
    for (int t = 0; t < r.tokens; t++) {
        double mc = 0, mg = 0; int hw = -1;
        for (int i = 0; i < r.nh * r.nv; i++) {
            const double a = r.cpuctx[(size_t)t * r.nh * r.nv + i];
            const double b = gpuctx[(size_t)t * r.nh * r.nv + i];
            if (fabs(a) > mc) mc = fabs(a);
            const double e = fabs(a - b);
            if (e > mg) { mg = e; hw = i / r.nv; }
        }
        rowrel[t] = mg / (mc > 0 ? mc : 1); rowhead[t] = hw;
        sumrel += rowrel[t];
        if (rowrel[t] > maxrel) { maxrel = rowrel[t]; maxrow = t; }
    }
    {   /* p50 / p99 of the per-row relative gap */
        double *srt = malloc((size_t)r.tokens * sizeof(double));
        memcpy(srt, rowrel, (size_t)r.tokens * sizeof(double));
        for (int i = 1; i < r.tokens; i++) {          /* insertion sort, 512 rows */
            double v = srt[i]; int j = i - 1;
            while (j >= 0 && srt[j] > v) { srt[j + 1] = srt[j]; j--; }
            srt[j + 1] = v;
        }
        printf("ROW triage gpu-vs-cpu over ALL %d rows: mean=%.3e p50=%.3e p99=%.3e max=%.3e "
               "(row %d, head %d)\n", r.tokens, sumrel / r.tokens, srt[r.tokens / 2],
               srt[(int)(r.tokens * 0.99)], maxrel, r.base + maxrow, maxrow >= 0 ? rowhead[maxrow] : -1);
        free(srt);
    }

    /* float64 on the worst rows the triage found, plus the focus row. */
    const int nworst = getenv("F7_REPLAY_ROWS") ? atoi(getenv("F7_REPLAY_ROWS")) : 8;
    const int focus = getenv("F7_REPLAY_FOCUS") ? atoi(getenv("F7_REPLAY_FOCUS")) : 1420;
    int *pick = malloc((size_t)(nworst + 1) * sizeof(int));
    int npick = 0;
    for (int k = 0; k < nworst; k++) {
        int best = -1;
        for (int t = 0; t < r.tokens; t++) {
            int dup = 0;
            for (int j = 0; j < npick; j++) if (pick[j] == t) dup = 1;
            if (!dup && (best < 0 || rowrel[t] > rowrel[best])) best = t;
        }
        if (best >= 0) pick[npick++] = best;
    }
    if (focus - r.base >= 0 && focus - r.base < r.tokens) pick[npick++] = focus - r.base;

    double *ctx64 = malloc((size_t)r.nh * r.nv * sizeof(double));
    double wc_cpu = 0, wc_gpu = 0;
    int wr_cpu = -1, wr_gpu = -1;
    printf("INFO per-row context error vs float64 on the worst rows (rel = maxabs / max|ref64|)\n");
    for (int k = 0; k < npick; k++) {
        const int t = pick[k];
        replay_ref64(&r, t, slot + (size_t)t * r.width, used[t], ctx64);
        double mref = 0, mc = 0, mg = 0; int hc = -1, hg = -1;
        for (int i = 0; i < r.nh * r.nv; i++) {
            const double ref = ctx64[i];
            if (fabs(ref) > mref) mref = fabs(ref);
            const double ec = fabs(ref - (double)r.cpuctx[(size_t)t * r.nh * r.nv + i]);
            const double eg = fabs(ref - (double)gpuctx[(size_t)t * r.nh * r.nv + i]);
            if (ec > mc) { mc = ec; hc = i / r.nv; }
            if (eg > mg) { mg = eg; hg = i / r.nv; }
        }
        const double rc = mc / (mref > 0 ? mref : 1), rg = mg / (mref > 0 ? mref : 1);
        if (rc > wc_cpu) { wc_cpu = rc; wr_cpu = r.base + t; }
        if (rg > wc_gpu) { wc_gpu = rg; wr_gpu = r.base + t; }
        printf("RROW t=%-6d used=%-5d ref_max=%.4e | cpu_rel=%.3e h=%-3d | gpu_rel=%.3e h=%-3d "
               "| gpu/cpu=%.2f%s\n", r.base + t, used[t], mref, rc, hc, rg, hg,
               rc > 0 ? rg / rc : 0.0, (r.base + t == focus) ? "   <- focus" : "");
    }
    printf("ROW replay worst_rel_vs_ref64 cpu=%.3e (t=%d) gpu=%.3e (t=%d) ratio=%.2f\n",
           wc_cpu, wr_cpu, wc_gpu, wr_gpu, wc_cpu > 0 ? wc_gpu / wc_cpu : 0.0);
    printf("INFO done\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <shaders_dir> [reps]\n", argv[0]); return 2; }
    char spv[512]; snprintf(spv, sizeof(spv), "%s/qmatmul.spv", argv[1]);
    const int reps = argc > 2 ? atoi(argv[2]) : 3;

    if (!coli_vk_init(spv)) { fprintf(stderr, "coli_vk_init failed (%s)\n", spv); return 2; }
    if (!coli_vk_mla_attn_ready()) {
        fprintf(stderr, "FATAL: the four mla_attn_*.spv are not next to %s\n", spv);
        return 2;
    }
    double used_gb = 0, budget_gb = 0;
    if (coli_vk_mem_budget(&used_gb, &budget_gb))
        printf("INFO dev0 heap used=%.2f GB budget=%.2f GB free=%.2f GB\n",
               used_gb, budget_gb, budget_gb - used_gb);
    printf("INFO shapes H=%d L=%d V=%d width=%d scale=1/sqrt(%d) gs=%d\n", H, L, V, WIDTH, QKN, GS);

    if (getenv("F7_REPLAY")) return replay_main(getenv("F7_REPLAY"), argv[1]);

    /* kvb_v: raw int4-g64 nibbles, one f32 scale per (row, 64-group) */
    const int packed = L / 2, ng = L / GS, vrows = H * V;
    uint8_t *vw = malloc((size_t)vrows * packed);
    float   *vsc = malloc((size_t)vrows * ng * sizeof(float));
    for (size_t i = 0; i < (size_t)vrows * packed; i++) vw[i] = (uint8_t)(rndu() & 0xFF);
    for (size_t i = 0; i < (size_t)vrows * ng; i++) vsc[i] = (rnd() * 0.5f + 0.5f) * 0.015625f;

    const int seens[3] = {2000, 9000, 18439};
    const int Ss[2] = {128, 512};
    const float scale = 1.0f / sqrtf((float)QKN);
    ColiVkTensor *vt = NULL;

    for (int si = 0; si < 3; si++) {
        const int seen = seens[si];
        float *lat = malloc((size_t)seen * L * sizeof(float));
        for (size_t i = 0; i < (size_t)seen * L; i++) lat[i] = rnd() * 0.044194174f; /* 1/sqrt(512) */
        for (int Si = 0; Si < 2; Si++) {
            const int S = Ss[Si];
            const int width = seen < WIDTH ? seen : WIDTH;
            float *absorbed = malloc((size_t)S * H * L * sizeof(float));
            float *ctx = malloc((size_t)S * H * V * sizeof(float));
            int *slot = malloc((size_t)S * width * sizeof(int));
            int *used = malloc((size_t)S * sizeof(int));
            for (size_t i = 0; i < (size_t)S * H * L; i++) absorbed[i] = rnd() * 0.044194174f;
            /* one selection per row: a sorted random subset of [0, seen), the
             * shape the indexer emits (dense when seen <= width). */
            for (int t = 0; t < S; t++) {
                int *d = slot + (size_t)t * width;
                if (seen <= width) { for (int i = 0; i < seen; i++) d[i] = i; used[t] = seen; }
                else {
                    int n = 0, step = seen / width;
                    for (int i = 0; i < width; i++) {
                        int v = i * step + (int)(rndu() % (unsigned)step);
                        if (v >= seen) v = seen - 1;
                        d[n++] = v;
                    }
                    used[t] = n;
                }
            }
            double best = 1e30, sum = 0;
            for (int r = 0; r < reps; r++) {
                const double t0 = now_ms();
                if (!coli_vk_mla_attn(&vt, vw, vsc, 4, GS, ctx, absorbed, lat, slot, used,
                                      S, H, L, V, width, seen, 128, scale)) {
                    printf("WARN coli_vk_mla_attn returned 0 at seen=%d S=%d\n", seen, S);
                    goto next;
                }
                const double dt = now_ms() - t0;
                if (dt < best) best = dt;
                sum += dt;
            }
            {
                /* numerical check on a handful of rows (the reference is O(S*268 MFLOP)) */
                const int nchk = 4;
                float *rctx = malloc((size_t)H * V * sizeof(float));
                float *score = malloc((size_t)width * sizeof(float));
                float *pooled = malloc((size_t)L * sizeof(float));
                double maxabs = 0, maxrel = 0, refmax = 0;
                for (int k = 0; k < nchk; k++) {
                    const int t = (int)((size_t)k * (size_t)S / nchk);
                    ref_row(rctx, absorbed + (size_t)t * H * L, lat,
                            slot + (size_t)t * width, used[t], seen, vw, vsc, scale, score, pooled);
                    for (int i = 0; i < H * V; i++) {
                        const double a = rctx[i], b = ctx[(size_t)t * H * V + i];
                        const double e = fabs(a - b);
                        if (e > maxabs) maxabs = e;
                        if (fabs(a) > refmax) refmax = fabs(a);
                        const double den = fabs(a) > 1e-6 ? fabs(a) : 1e-6;
                        if (e / den > maxrel) maxrel = e / den;
                    }
                }
                const double gflop = 2.0 * (double)S * (double)used[0] * H * L * 2.0 / 1e9
                                   + 2.0 * (double)S * H * V * L / 1e9;
                printf("ROW seen=%-6d S=%-4d width=%-5d used=%-5d best_ms=%8.2f mean_ms=%8.2f "
                       "gflop=%7.2f tflops=%5.2f max_abs=%.3e max_rel=%.3e ref_max=%.3e\n",
                       seen, S, width, used[0], best, sum / reps, gflop, gflop / (best / 1e3) / 1e3,
                       maxabs, maxrel, refmax);
                free(pooled); free(score); free(rctx);
            }
            {
                double cp = 0, gp = 0; long ca = 0, su = 0;
                coli_vk_mla_attn_prof(&cp, &gp, &ca, &su);
                printf("INFO   cumulative copy=%.1f ms gpu=%.1f ms calls=%ld submits=%ld\n",
                       cp, gp, ca, su);
            }
next:
            free(used); free(slot); free(ctx); free(absorbed);
        }
        free(lat);
    }
    printf("INFO done\n");
    return 0;
}
