/* rome_mlaattn.c -- P5.1 microbenchmark: GLM-5.3's MLA attention core at the
 * engine's real shapes, on the CPU, isolated from the engine.
 *
 * The op, per (token, layer), as c/glm53.c mla_layer() runs it today:
 *
 *   for h in 0..H-1                       (an OpenMP team per token)
 *       for i in 0..width-1               (the indexer's slots, -1 skipped)
 *           dot = sum_d q[h][d] * latent[chosen[i]][d]      <-- SCALAR, one
 *           score[used++] = dot * scale                         accumulator
 *       softmax(score[0..used))
 *       for i: pooled[d] += weight * latent[chosen[i]][d]   <-- auto-vectorised
 *       result[h] = kvb_v[h] . pooled
 *
 * Shapes from the model's own config.json (GLM-5.3-Flash):
 *   H = 64 heads, L = kv_lora = 512, v_head = 256, index_topk = 2048,
 *   index_kpool = 4 (+tail) -> width = 2051, 11 DSA layers of 45.
 *
 * The candidate puts the 64 HEADS in the SIMD lanes and broadcasts the latent
 * row, over a transposed query slab qT[d][h]. Lane h then accumulates over d
 * in ascending order with one FMA per step -- which is exactly what the
 * compiler contracts the scalar `dot += q[d]*c[d]` into on this box
 * (-O3 -march=native, -ffp-contract=fast) -- so every score is bit-identical.
 * The bench asserts that, exhaustively, over every (head, slot) score.
 *
 * Working set, deliberately: the engine's latent for ONE layer at 3 462 tokens
 * is 3 462 x 512 x 4 = 7.1 MB and all 11 layers together are 78 MB, so this op
 * is genuinely L3-resident in the engine (L3 is 128 MB). Faking a >128 MB pool
 * here would measure a regime the engine never runs in. Instead both are
 * reported: `L1` cycles one layer's latent (the engine's per-layer locality)
 * and `L11` rotates 11 layers' worth (78 MB, the engine's per-token footprint),
 * so an L3-only win shows up as a gap between the two.
 *
 * Build:  gcc -O3 -march=native -fopenmp -o rome_mlaattn rome_mlaattn.c -lm
 * Run:    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close ./rome_mlaattn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MLAATTN_AVX2 1
/* The reference is `dot += q[d]*c[d]`, which gcc 15 -O3 -march=native compiles
 * to vmulps for the products plus an IN-ORDER vaddss chain: every product is
 * rounded to float before it is accumulated, and the sum is strictly
 * sequential. To be bit-identical the vector kernel must do the same -- but
 * gcc contracts add(a, mul(b,c)) written with intrinsics straight back into
 * vfmadd231ps (and ignores `#pragma STDC FP_CONTRACT OFF`), which is 1 ulp off
 * and the oracle catches it. The empty asm forces the product into a register
 * first; it costs no instruction. */
#define MULADD(acc, x, y) do { __m256 m_ = _mm256_mul_ps((x), (y)); \
                               __asm__("" : "+x"(m_)); \
                               (acc) = _mm256_add_ps((acc), m_); } while (0)
#endif

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define H 64
#define L 512
#define WIDTH 2051

/* ---------------- the engine's kernel, verbatim in shape ---------------- */
static void base_scores(float *score, const float *q, const float *latent,
                        const int *chosen, int width, int seen, float scale,
                        int *used_out, float *top_out) {
    float top = -INFINITY;
    int used = 0;
    for (int i = 0; i < width; i++) {
        const int at = chosen[i];
        if (at < 0 || at >= seen) continue;
        const float *c_j = latent + (size_t)at * L;
        float dot = 0.0f;
        for (int d = 0; d < L; d++) dot += q[d] * c_j[d];
        score[used] = dot * scale;
        if (score[used] > top) top = score[used];
        used++;
    }
    *used_out = used; *top_out = top;
}

/* ---------------- the candidate: 64 heads in the lanes ----------------- */
/* qT[d*H + h] == q_of_head_h[d] */
static void cand_score_row(float *dst, const float *qT, const float *c_j, float scale) {
#ifdef MLAATTN_AVX2
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
    const float *p = qT;
    for (int d = 0; d < L; d++, p += H) {
        const __m256 b = _mm256_broadcast_ss(c_j + d);
        MULADD(a0, b, _mm256_loadu_ps(p));
        MULADD(a1, b, _mm256_loadu_ps(p + 8));
        MULADD(a2, b, _mm256_loadu_ps(p + 16));
        MULADD(a3, b, _mm256_loadu_ps(p + 24));
        MULADD(a4, b, _mm256_loadu_ps(p + 32));
        MULADD(a5, b, _mm256_loadu_ps(p + 40));
        MULADD(a6, b, _mm256_loadu_ps(p + 48));
        MULADD(a7, b, _mm256_loadu_ps(p + 56));
    }
    const __m256 s = _mm256_set1_ps(scale);
    _mm256_storeu_ps(dst,      _mm256_mul_ps(a0, s));
    _mm256_storeu_ps(dst + 8,  _mm256_mul_ps(a1, s));
    _mm256_storeu_ps(dst + 16, _mm256_mul_ps(a2, s));
    _mm256_storeu_ps(dst + 24, _mm256_mul_ps(a3, s));
    _mm256_storeu_ps(dst + 32, _mm256_mul_ps(a4, s));
    _mm256_storeu_ps(dst + 40, _mm256_mul_ps(a5, s));
    _mm256_storeu_ps(dst + 48, _mm256_mul_ps(a6, s));
    _mm256_storeu_ps(dst + 56, _mm256_mul_ps(a7, s));
#else
    for (int h = 0; h < H; h++) {
        float dot = 0.0f;
        for (int d = 0; d < L; d++) dot += qT[(size_t)d * H + h] * c_j[d];
        dst[h] = dot * scale;
    }
#endif
}

static void transpose_q(float *qT, const float *q) {   /* [H][L] -> [L][H] */
    for (int h = 0; h < H; h += 8)
        for (int d = 0; d < L; d += 8)
            for (int hh = 0; hh < 8; hh++)
                for (int dd = 0; dd < 8; dd++)
                    qT[(size_t)(d + dd) * H + h + hh] = q[(size_t)(h + hh) * L + d + dd];
}

int main(int argc, char **argv) {
    const int layers = 11;
    int seen = argc > 1 ? atoi(argv[1]) : 1442;      /* mean `used` at 3 462 tok */
    int reps  = argc > 2 ? atoi(argv[2]) : 20;
    int pool_layers = argc > 3 ? atoi(argv[3]) : 1;  /* 1 = one layer, 11 = 78 MB */
    if (seen > WIDTH) seen = WIDTH;

    const int cap = 3600;
    size_t lat_n = (size_t)pool_layers * cap * L;
    float *latent = aligned_alloc(64, lat_n * sizeof(float));
    float *q      = aligned_alloc(64, (size_t)H * L * sizeof(float));
    float *qT     = aligned_alloc(64, (size_t)L * H * sizeof(float));
    int   *chosen = malloc((size_t)WIDTH * sizeof(int));
    if (!latent || !q || !qT || !chosen) { fprintf(stderr, "OOM\n"); return 1; }

    unsigned s = 12345u;
#define RND ((s = s * 1103515245u + 12345u), ((float)((s >> 9) & 0x7fffff) / 4194304.0f - 1.0f))
    for (size_t i = 0; i < lat_n; i++) latent[i] = RND * 0.5f;
    for (size_t i = 0; i < (size_t)H * L; i++) q[i] = RND * 0.5f;
    transpose_q(qT, q);
    /* the indexer emits runs of index_kpool=4 consecutive positions, in
     * selection order, with -1 in unused slots and a tail run at the end. */
    for (int i = 0; i < WIDTH; i++) chosen[i] = -1;
    {
        int run = 0;
        for (int i = 0; i + 4 <= 2048 && run * 4 < seen; i += 4, run++) {
            int base = (int)(((size_t)run * 2654435761u) % (size_t)(seen / 4 ? seen / 4 : 1)) * 4;
            for (int j = 0; j < 4; j++) chosen[i + j] = base + j;
        }
    }
    const float scale = 1.0f / sqrtf(256.0f);

    /* ---------------- oracle: every score, bit for bit ---------------- */
    float *sb = malloc((size_t)WIDTH * sizeof(float));
    float *sc = malloc((size_t)WIDTH * H * sizeof(float));
    int used = 0, bad = 0; float topb;
    int *idx = malloc((size_t)WIDTH * sizeof(int));
    for (int i = 0; i < WIDTH; i++)
        if (chosen[i] >= 0 && chosen[i] < cap) idx[used++] = chosen[i];
    for (int u = 0; u < used; u++)
        cand_score_row(sc + (size_t)u * H, qT, latent + (size_t)idx[u] * L, scale);
    for (int h = 0; h < H; h++) {
        int ub; base_scores(sb, q + (size_t)h * L, latent, chosen, WIDTH, cap, scale, &ub, &topb);
        if (ub != used) { printf("ORACLE: used mismatch %d vs %d\n", ub, used); bad++; break; }
        for (int u = 0; u < used; u++) {
            unsigned a, b;
            memcpy(&a, &sb[u], 4); memcpy(&b, &sc[(size_t)u * H + h], 4);
            if (a != b) { if (bad < 5) printf("ORACLE: h=%d u=%d %.9g vs %.9g (%08x %08x)\n",
                                              h, u, sb[u], sc[(size_t)u*H+h], a, b); bad++; }
        }
    }
    printf("oracle: %d scores compared (%d heads x %d slots), mismatches %d -> %s\n",
           used * H, H, used, bad, bad ? "FAIL" : "BIT-IDENTICAL");

    /* ---------------- timing ---------------- */
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    double macs = (double)H * used * L;      /* per token per layer */
    float *out_base = malloc((size_t)nthr * WIDTH * sizeof(float));
    float *out_cand = malloc((size_t)nthr * WIDTH * H * sizeof(float));

    /* baseline: one token's 64 heads, parallel over heads, as the engine does */
    double t0 = now();
    for (int r = 0; r < reps; r++) {
        for (int lay = 0; lay < layers; lay++) {
            const float *lat = latent + (size_t)(lay % pool_layers) * cap * L;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int h = 0; h < H; h++) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                int ub; float tp;
                base_scores(out_base + (size_t)tid * WIDTH, q + (size_t)h * L, lat,
                            chosen, WIDTH, cap, scale, &ub, &tp);
            }
        }
    }
    double tb = (now() - t0) / reps;

    /* candidate: parallel over the indexer's slots, 64 heads in the lanes */
    t0 = now();
    for (int r = 0; r < reps; r++) {
        for (int lay = 0; lay < layers; lay++) {
            const float *lat = latent + (size_t)(lay % pool_layers) * cap * L;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int u = 0; u < used; u++)
                cand_score_row(out_cand + (size_t)u * H, qT, lat + (size_t)idx[u] * L, scale);
        }
    }
    double tc = (now() - t0) / reps;

    /* the transpose the candidate has to pay, per token per layer */
    t0 = now();
    for (int r = 0; r < reps; r++)
        for (int lay = 0; lay < layers; lay++) transpose_q(qT, q);
    double tt = (now() - t0) / reps;

    printf("threads=%d used=%d layers=%d pool=%d layer(s) = %.1f MB\n",
           nthr, used, layers, pool_layers,
           (double)lat_n * 4 / 1048576.0);
    printf("  baseline  %8.3f ms/token   %6.2f GMAC/s\n", tb * 1e3, macs * layers / tb / 1e9);
    printf("  candidate %8.3f ms/token   %6.2f GMAC/s   speedup %.2fx\n",
           tc * 1e3, macs * layers / tc / 1e9, tb / tc);
    printf("  transpose %8.3f ms/token (%.1f%% of the candidate)\n",
           tt * 1e3, 100.0 * tt / tc);
    printf("  net       %8.3f ms/token   speedup %.2fx\n", (tc + tt) * 1e3, tb / (tc + tt));
    return bad ? 1 : 0;
}
