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
