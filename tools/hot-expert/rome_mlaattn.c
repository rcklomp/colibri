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
#define DTILE 8           /* P5b: d per accumulator tile (measured best of 8/16/32/64) */

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

/* ---------------- P5b: the weighted pool, blocked over head x d -----------
 *
 * The reference, per head, is
 *
 *     for u:  w = (float)(score[u]/total);
 *             for d:  pooled[d] += w * latent[idx[u]][d];
 *
 * and gcc 15 -O3 -march=native CONTRACTS that into vfmadd213ps (checked in the
 * objdump; -ffp-contract=fast is the GNU default), so the reference term is
 * fmaf(w, c, acc) -- ONE rounding. That is the opposite of the score pass,
 * where the reduction cannot vectorise and each product is rounded before an
 * in-order scalar add. A bit-identical blocked pool therefore has to USE fma,
 * and to keep u ascending for every (h, d).
 *
 * As written the pool walks the layer's selected latent ONCE PER HEAD:
 * 64 x used x L x 4 = 180 MB per token per layer at used = 1444, 2.0 GB per
 * token over the 11 DSA layers, of which only 2.8 MB is distinct. Blocking it
 * over (d-tile x head group) reads each latent byte once. The d-tiles are the
 * parallel axis (L = 512 -> 16 tiles of 32 for 8 threads), each tile keeps its
 * own accumulator plane acc[dt][H] (h contiguous, 8 KB at dt = 32) in L1, and
 * eight slots are folded at a time so one accumulator load+store serves eight
 * FMAs. Every (h, d) still accumulates u = 0, 1, 2, ... in order. */
static void pool_blocked(float *pooled_all, const float *wT, const float *lat,
                         const int *idx, int used, int dtile, float *accbuf) {
    const int ntiles = (L + dtile - 1) / dtile;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int tb = 0; tb < ntiles; tb++) {
        const int d0 = tb * dtile;
        const int dt = (L - d0 < dtile) ? (L - d0) : dtile;
        float *acc = accbuf + (size_t)tb * dtile * H;
        memset(acc, 0, (size_t)dt * H * sizeof(float));
        int u = 0;
#ifdef MLAATTN_AVX2
        for (; u + 8 <= used; u += 8) {
            const float *r[8], *w[8];
            for (int k = 0; k < 8; k++) {
                r[k] = lat + (size_t)idx[u + k] * L + d0;
                w[k] = wT + (size_t)(u + k) * H;
            }
            for (int hg = 0; hg + 8 <= H; hg += 8) {
                const __m256 v0=_mm256_loadu_ps(w[0]+hg), v1=_mm256_loadu_ps(w[1]+hg);
                const __m256 v2=_mm256_loadu_ps(w[2]+hg), v3=_mm256_loadu_ps(w[3]+hg);
                const __m256 v4=_mm256_loadu_ps(w[4]+hg), v5=_mm256_loadu_ps(w[5]+hg);
                const __m256 v6=_mm256_loadu_ps(w[6]+hg), v7=_mm256_loadu_ps(w[7]+hg);
                float *a = acc + hg;
                for (int dd = 0; dd < dt; dd++, a += H) {
                    __m256 x = _mm256_loadu_ps(a);
                    x = _mm256_fmadd_ps(v0, _mm256_broadcast_ss(r[0]+dd), x);
                    x = _mm256_fmadd_ps(v1, _mm256_broadcast_ss(r[1]+dd), x);
                    x = _mm256_fmadd_ps(v2, _mm256_broadcast_ss(r[2]+dd), x);
                    x = _mm256_fmadd_ps(v3, _mm256_broadcast_ss(r[3]+dd), x);
                    x = _mm256_fmadd_ps(v4, _mm256_broadcast_ss(r[4]+dd), x);
                    x = _mm256_fmadd_ps(v5, _mm256_broadcast_ss(r[5]+dd), x);
                    x = _mm256_fmadd_ps(v6, _mm256_broadcast_ss(r[6]+dd), x);
                    x = _mm256_fmadd_ps(v7, _mm256_broadcast_ss(r[7]+dd), x);
                    _mm256_storeu_ps(a, x);
                }
            }
        }
        for (; u < used; u++) {
            const float *r = lat + (size_t)idx[u] * L + d0;
            const float *w = wT + (size_t)u * H;
            for (int hg = 0; hg + 8 <= H; hg += 8) {
                const __m256 v = _mm256_loadu_ps(w + hg);
                float *a = acc + hg;
                for (int dd = 0; dd < dt; dd++, a += H)
                    _mm256_storeu_ps(a, _mm256_fmadd_ps(v, _mm256_broadcast_ss(r+dd),
                                                        _mm256_loadu_ps(a)));
            }
        }
#else
        for (; u < used; u++) {
            const float *r = lat + (size_t)idx[u] * L + d0;
            const float *w = wT + (size_t)u * H;
            for (int h = 0; h < H; h++)
                for (int dd = 0; dd < dt; dd++)
                    acc[(size_t)dd*H+h] = fmaf(w[h], r[dd], acc[(size_t)dd*H+h]);
        }
#endif
        for (int h = 0; h < H; h++)
            for (int dd = 0; dd < dt; dd++)
                pooled_all[(size_t)h * L + d0 + dd] = acc[(size_t)dd * H + h];
    }
}

int main(int argc, char **argv) {
    const int layers = 11;
    int seen = argc > 1 ? atoi(argv[1]) : 1442;      /* mean `used` at 3 462 tok */
    int reps  = argc > 2 ? atoi(argv[2]) : 20;
    int pool_layers = argc > 3 ? atoi(argv[3]) : 1;  /* 1 = one layer, 11 = 78 MB */
    /* P5b: the per-thread `pooled`/`score` slices are L and width floats wide
     * in the engine -- 2 048 and 8 204 bytes, NEITHER a multiple of 64 -- so
     * adjacent threads share the boundary cache lines and ping-pong them for
     * the whole pool. pad=1 rounds each slice up to a cache line, which is
     * the engine's P5b.1. Run the bench both ways: the difference is the
     * artefact, and it is most of what P5 recorded as "softmax+pool 17->39". */
    int pad = argc > 4 ? atoi(argv[4]) : 1;
    int dtile = argc > 5 ? atoi(argv[5]) : DTILE;  /* P5b accumulator tile */
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

    /* ---------------- oracle 2: every pooled value, bit for bit -------------
     * The pool is what P5b replaces, so the check is the same shape as the
     * score one: the reference nest (per head: softmax, then the rank-1
     * accumulate over slots) against the blocked nest, all H x L values. */
    float *pooled_ref = malloc((size_t)H * L * sizeof(float));
    float *pooled_c1  = malloc((size_t)H * L * sizeof(float));
    float *pooled_c2  = malloc((size_t)H * L * sizeof(float));
    float *wT_o       = malloc((size_t)WIDTH * H * sizeof(float));
    float *accbuf     = aligned_alloc(64, (size_t)L * H * sizeof(float));
    float *sco        = malloc((size_t)WIDTH * sizeof(float));
    if (!pooled_ref || !pooled_c1 || !pooled_c2 || !wT_o || !accbuf || !sco) {
        fprintf(stderr, "OOM\n"); return 1;
    }
    for (int h = 0; h < H; h++) {
        int ub; float top;
        base_scores(sco, q + (size_t)h * L, latent, chosen, WIDTH, cap, scale, &ub, &top);
        double total = 0.0;
        for (int u = 0; u < ub; u++) { sco[u] = expf(sco[u] - top); total += sco[u]; }
        float *pooled = pooled_ref + (size_t)h * L;
        memset(pooled, 0, (size_t)L * sizeof(float));
        for (int u = 0; u < ub; u++) {
            const float w = (float)(sco[u] / total);
            const float *c_j = latent + (size_t)idx[u] * L;
            for (int d = 0; d < L; d++) pooled[d] += w * c_j[d];
        }
    }
    /* candidate: the scores are already in sc[u][h]; the softmax turns each
     * head's column into its weights IN PLACE, then the blocked pool runs. */
    for (int h = 0; h < H; h++) {
        float top = -INFINITY;
        for (int u = 0; u < used; u++) {
            sco[u] = sc[(size_t)u * H + h];
            if (sco[u] > top) top = sco[u];
        }
        double total = 0.0;
        for (int u = 0; u < used; u++) { sco[u] = expf(sco[u] - top); total += sco[u]; }
        for (int u = 0; u < used; u++) wT_o[(size_t)u * H + h] = (float)(sco[u] / total);
    }
    pool_blocked(pooled_c1, wT_o, latent, idx, used, dtile, accbuf);
    pool_blocked(pooled_c2, wT_o, latent, idx, used, 8, accbuf);
    int badp1 = 0, badp2 = 0;
    for (int i = 0; i < H * L; i++) {
        unsigned a, b, c2;
        memcpy(&a, &pooled_ref[i], 4); memcpy(&b, &pooled_c1[i], 4); memcpy(&c2, &pooled_c2[i], 4);
        if (a != b) { if (badp1 < 3) printf("ORACLE pool dt=32: i=%d %.9g vs %.9g\n", i, pooled_ref[i], pooled_c1[i]); badp1++; }
        if (a != c2) { if (badp2 < 3) printf("ORACLE pool dt=8:  i=%d %.9g vs %.9g\n", i, pooled_ref[i], pooled_c2[i]); badp2++; }
    }
    printf("oracle: %d pooled values (%d heads x %d), dtile=32 mismatches %d -> %s, dtile=8 mismatches %d -> %s\n",
           H * L, H, L, badp1, badp1 ? "FAIL" : "BIT-IDENTICAL",
           badp2, badp2 ? "FAIL" : "BIT-IDENTICAL");
    bad += badp1 + badp2;

    /* ---------------- timing ---------------- */
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    double macs = (double)H * used * L;      /* per token per layer */
    /* pad=0 reproduces the engine's `malloc(nthreads * L * sizeof(float))`:
     * glibc hands back a 16-byte-aligned pointer, so the per-thread slices are
     * NOT cache-line aligned and every pair of neighbouring threads shares the
     * two boundary lines -- which they then write `used` times per head. The
     * +4 float offset below makes that misalignment deterministic instead of
     * whatever the allocator happened to do; pad=1 is the fix (64-byte base,
     * stride rounded up to a cache line). */
    const int sstride = pad ? ((WIDTH + 15) & ~15) : WIDTH;
    const int pstride = pad ? ((L + 15) & ~15) : L;
    float *out_base = (float *)aligned_alloc(64, (size_t)nthr * sstride * sizeof(float) + 64) + (pad ? 0 : 4);
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
                base_scores(out_base + (size_t)tid * sstride, q + (size_t)h * L, lat,
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

    /* ---- the WHOLE core, both ways, so the residual is attributed ----
     * The score pass is only part of `mla.attn`: the softmax, the weighted
     * pool over the same rows, and the kvb_v projection are the rest, and the
     * candidate additionally reads its 64 scores back out of sc[u][h]. Running
     * both whole cores says how much of the bucket the score pass ever was. */
    float *pool_pool = (float *)aligned_alloc(64, (size_t)nthr * pstride * sizeof(float) + 64) + (pad ? 0 : 4);
    float *pooled_big = malloc((size_t)H * L * sizeof(float));
    double tfb = 0, tfc = 0, tft = 0, tf1 = 0, tf2 = 0;
    double tpb = 0, tp1 = 0, tp2 = 0;   /* the pool phase alone */
    float *scT = malloc((size_t)WIDTH * H * sizeof(float));
    for (int pass = 0; pass < 5; pass++) {
        t0 = now();
        for (int r = 0; r < reps; r++) for (int lay = 0; lay < layers; lay++) {
            const float *lat = latent + (size_t)(lay % pool_layers) * cap * L;
            if (pass >= 1) {
                transpose_q(qT, q);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int u = 0; u < used; u++)
                    cand_score_row(out_cand + (size_t)u * H, qT, lat + (size_t)idx[u] * L, scale);
            }
            /* pass 2: hand the h loop a CONTIGUOUS score row per head, by
             * transposing sc[u][h] -> scT[h][u] once. If the candidate's extra
             * cost is the strided read-back, this pass gets it back. */
            if (pass == 2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int h = 0; h < H; h++)
                    for (int u = 0; u < used; u++)
                        scT[(size_t)h * WIDTH + u] = out_cand[(size_t)u * H + h];
            }
            if (pass >= 3) {
                /* P5b: softmax per head turns sc[u][h] into the weights in
                 * place, then ONE walk of the latent serves all 64 heads. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int h = 0; h < H; h++) {
                    int tid = 0;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
#endif
                    float *score = out_base + (size_t)tid * sstride;
                    float top = -INFINITY;
                    for (int u = 0; u < used; u++) {
                        score[u] = out_cand[(size_t)u * H + h];
                        if (score[u] > top) top = score[u];
                    }
                    double total = 0.0;
                    for (int u = 0; u < used; u++) { score[u] = expf(score[u] - top); total += score[u]; }
                    for (int u = 0; u < used; u++)
                        out_cand[(size_t)u * H + h] = (float)(score[u] / total);
                }
                double tp0 = now();
                pool_blocked(pooled_big, out_cand, lat, idx, used,
                             pass == 3 ? dtile : 8, accbuf);
                if (pass == 3) tp1 += now() - tp0; else tp2 += now() - tp0;
                continue;
            }
            double tpool0 = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int h = 0; h < H; h++) {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float *score = out_base + (size_t)tid * sstride;
                float *pooled = pool_pool + (size_t)tid * pstride;
                float top = -INFINITY;
                if (pass == 0) {
                    int ub; base_scores(score, q + (size_t)h * L, lat, chosen, WIDTH, cap,
                                        scale, &ub, &top);
                } else if (pass == 2) {
                    for (int u = 0; u < used; u++) {
                        score[u] = scT[(size_t)h * WIDTH + u];
                        if (score[u] > top) top = score[u];
                    }
                } else {
                    for (int u = 0; u < used; u++) {
                        score[u] = out_cand[(size_t)u * H + h];
                        if (score[u] > top) top = score[u];
                    }
                }
                double total = 0.0;
                for (int u = 0; u < used; u++) { score[u] = expf(score[u] - top); total += score[u]; }
                memset(pooled, 0, (size_t)L * sizeof(float));
                for (int u = 0; u < used; u++) {
                    const float w = (float)(score[u] / total);
                    const float *c_j = lat + (size_t)idx[u] * L;
                    for (int d = 0; d < L; d++) pooled[d] += w * c_j[d];
                }
                memcpy(pooled_big + (size_t)h * L, pooled, (size_t)L * sizeof(float));
            }
            (void)tpool0;
        }
        double el = (now() - t0) / reps;
        if (pass == 0) tfb = el; else if (pass == 1) tfc = el;
        else if (pass == 2) tft = el; else if (pass == 3) tf1 = el; else tf2 = el;
    }
    tp1 /= reps; tp2 /= reps;

    printf("threads=%d used=%d layers=%d pool=%d layer(s) = %.1f MB  per-thread scratch %s\n",
           nthr, used, layers, pool_layers,
           (double)lat_n * 4 / 1048576.0, pad ? "CACHE-LINE PADDED" : "as the engine has it");
    printf("  baseline  %8.3f ms/token   %6.2f GMAC/s\n", tb * 1e3, macs * layers / tb / 1e9);
    printf("  candidate %8.3f ms/token   %6.2f GMAC/s   speedup %.2fx\n",
           tc * 1e3, macs * layers / tc / 1e9, tb / tc);
    printf("  transpose %8.3f ms/token (%.1f%% of the candidate)\n",
           tt * 1e3, 100.0 * tt / tc);
    printf("  net       %8.3f ms/token   speedup %.2fx\n", (tc + tt) * 1e3, tb / (tc + tt));
    printf("  --- whole core (scores + softmax + weighted pool; no kvb_v, no o) ---\n");
    printf("  core base %8.3f ms/token   (scores %5.1f%% of it)\n", tfb * 1e3, 100.0 * tb / tfb);
    printf("  core cand %8.3f ms/token   speedup %.2fx   (softmax+pool floor %.3f ms)\n",
           tfc * 1e3, tfb / tfc, (tfb - tb) * 1e3);
    printf("  core cand+T %6.3f ms/token   speedup %.2fx  (sc transposed to [h][u] first)\n",
           tft * 1e3, tfb / tft);
    printf("  core P5b    %6.3f ms/token   speedup %.2fx  (blocked pool, dtile %d; pool alone %.3f ms)\n",
           tf1 * 1e3, tfb / tf1, dtile, tp1 * 1e3);
    printf("  core P5b/8  %6.3f ms/token   speedup %.2fx  (blocked pool, dtile 8;  pool alone %.3f ms)\n",
           tf2 * 1e3, tfb / tf2, tp2 * 1e3);
    printf("  softmax+pool: P5 %.3f ms/token, P5b %.3f (of which the pool %.3f)\n",
           (tfc - tc - tt) * 1e3, (tf1 - tc - tt) * 1e3, tp1 * 1e3);
    free(scT); free(pooled_big);
    free(pooled_ref); free(pooled_c1); free(pooled_c2); free(wT_o); free(accbuf); free(sco);
    return bad ? 1 : 0;
}
