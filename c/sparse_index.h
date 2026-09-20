#ifndef COLIBRI_SPARSE_INDEX_H
#define COLIBRI_SPARSE_INDEX_H

/*
 * DeepSeek-style sparse attention with k-pooling, as GLM-5.3-Flash runs it.
 *
 * The plain lightning indexer scores every key and keeps the top ones. With
 * k-pooling the keys are first grouped into pools of `pool` tokens; the pools
 * are scored and selected, and the winners expand back into their token
 * indices. Selecting 512 pools of 4 instead of 2048 tokens is the same budget
 * with a quarter of the scoring, and the pooled key is a per-channel softmax
 * mixture of the pool's keys rather than an average, so a pool is not forced to
 * describe itself by its mean.
 *
 * Three rules decide which pools may be chosen, and all three matter:
 *
 *   - a pool is selectable only if it is COMPLETE (every member present) and
 *     its last token is causally visible to the query;
 *   - the incomplete tail - the tokens after the last complete pool - is
 *     appended unconditionally when the model asks for it, which is why the
 *     output is `topk + pool - 1` wide rather than `topk`;
 *   - ties go to the lower pool index, so selection is deterministic.
 *
 * The scores carry a ReLU. It is written here as `if (dot > 0)` before scaling,
 * which is the same function as the reference's relu(dot * scale): ReLU
 * commutes with a positive scale.
 *
 * Unselected slots are -1, and the attention below skips them. Callers must not
 * treat the width as a count.
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Width of one query's index row: topk expanded tokens plus the tail. */
static inline int coli_sparse_index_width(int topk, int pool, int with_tail) {
    return with_tail ? topk + pool - 1 : topk;
}

/* F6a: GLM53_INDEX_SCALAR=1 restores the original serial, O(wanted*pools)
 * scan below byte-for-byte -- the A/B against the parallel/heap path, named
 * after the file's own GLM53_NO_INDEX_CACHE convention (glm53.c). Default
 * (0) is the new path: it changes no arithmetic and no selection outcome,
 * only how many threads compute it and how the top-`wanted` is found, so
 * default-on is the same footing G4/G5/G7/G8 shipped on. */
static inline int coli_sparse_index_scalar_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GLM53_INDEX_SCALAR");
        cached = (e && atoi(e)) ? 1 : 0;
    }
    return cached;
}

/* F9a: head-lane SIMD for the score pass's per-pool dot, ported from P5.1's
 * MLA attention core (glm53.c's glm_transpose/glm_lane_dots). The per-pool,
 * per-head dot `for d: dot += query[h][d] * pv[d]` below is exactly the
 * shape P5.1 fixed: a scalar float reduction with a strictly sequential add
 * chain (no -ffast-math, so the compiler may not reorder it), latency- not
 * throughput-bound. The fix is the same: transpose the row's queries once to
 * qT[d][h], then compute all heads' dots against one pool vector at a time,
 * heads in the SIMD lanes, each lane still summing over d in ascending order
 * with the SAME rounding as the scalar loop -- so every dot_h is
 * bit-identical, and the ReLU / head_w / scale accumulation over h stays the
 * untouched scalar loop it always was (its order is part of the score, and
 * this does not vectorize it).
 *
 * Checked on the rig (objdump -d of this build's own
 * coli_sparse_index_score_row, and a matching -O3 -march=native -fopenmp
 * probe TU): the compiled scalar dot loop is NOT a bare sequential scalar
 * chain -- GCC auto-vectorizes the eight products of a block with a single
 * vmulps, but then adds those eight products back into the running scalar
 * accumulator one at a time, in strict ascending order (vaddss, lane 0, 1,
 * 2, ... 7), never vfmadd. That is bit-identical to the naive sequential
 * `dot=0; for d: dot+=query[d]*pv[d]` with separate multiply and add and no
 * FMA contraction -- multiplication is elementwise and rounds the same
 * regardless of which instruction performs it, and the add order the
 * compiler chose is exactly left-to-right. The lane kernel below reproduces
 * that: a plain multiply-then-add per d, across all head lanes at once, with
 * FMA contraction blocked by the same empty-asm trick P5.1 uses (needed
 * because the compiler DOES contract a same-width intrinsic mul+add into
 * vfmadd, unlike the shuffle-heavy auto-vectorized reduction above it does
 * not contract).
 *
 * GLM53_INDEX_LANES=0 restores the scalar dot (read once, cached); default
 * is on -- FOR GCC. The bit-identical claim above is about what GCC (the
 * engine's only production compiler, `c/Makefile`'s CC=gcc) emits for this
 * exact loop shape; it was checked once, not derived from a portable rule.
 * A clang build of this same source was found, while writing this item, to
 * contract the reference's OWN scalar dot into vfmadd in cases GCC does not
 * (e.g. a short head count's remainder), which would silently break the
 * match -- so on anything other than real GCC the default is off, same as
 * P5.1's own COLI_MLA_HEADVEC gate defaults off without AVX2/FMA. The env
 * var still forces it on any compiler for whoever explicitly wants that
 * (their choice to make, not this default's). GLM53_INDEX_SCALAR=1 (the
 * original single-threaded O(wanted*pools) scan, F6a) never reaches this
 * code at all, so it implies lanes off regardless. */
static int g_coli_index_lanes_cached = -1;
static inline int coli_sparse_index_lanes_on(void) {
    if (g_coli_index_lanes_cached < 0) {
        const char *e = getenv("GLM53_INDEX_LANES");
#if defined(__GNUC__) && !defined(__clang__)
        g_coli_index_lanes_cached = e ? atoi(e) : 1;
#else
        g_coli_index_lanes_cached = e ? atoi(e) : 0;
#endif
    }
    return g_coli_index_lanes_cached;
}

/* Test hook only: force the cached decision directly, bypassing getenv, so
 * one test process can flip the lane path on and off between calls (the
 * env var is read once and cached, same as GLM53_INDEX_SCALAR's own
 * cache -- a real process picks one mode for its life; a test comparing
 * both needs to override that). Not used by the engine. */
static inline void coli_sparse_index_lanes_set(int v) { g_coli_index_lanes_cached = v; }

/* The lane kernel matches the reference dot bit-for-bit only where the
 * reference's OWN compiled loop has no odd remainder: gcc vectorises the
 * per-head dot 8-wide (no FMA, verified by objdump -- see the header
 * comment above and the F9a record), then for `dim % 8` left over
 * vectorises a further 4-wide block (again no FMA) if at least 4 remain,
 * and finally falls to `vfmadd231ss`, ONE HEAD ELEMENT AT A TIME, for
 * whatever is still left -- which happens exactly when `dim % 4 != 0`.
 * The lane kernel never does that last step (every d it processes is a
 * plain multiply-then-add, uniformly), so it would quietly stop matching
 * for such a dim. GLM-5.3's index_head_dim is 128 (dim % 4 == 0, always
 * clear of this), so this guard is a safety net for a dim this engine does
 * not use today, not a correctness fix for one it does: it degrades to the
 * scalar dot instead of drifting off it. */
static inline int coli_sparse_index_lanes_usable(int dim) {
    return coli_sparse_index_lanes_on() && dim % 4 == 0;
}

#if defined(__AVX2__) && defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define COLI_INDEX_HEADVEC 1
#define COLI_IDX_MULADD(acc, x, y) do { __m256 m_ = _mm256_mul_ps((x), (y)); \
                                        __asm__("" : "+x"(m_)); \
                                        (acc) = _mm256_add_ps((acc), m_); } while (0)
#endif

/* a[N][K] -> aT[K][N], 8x8 blocked -- verbatim shape of glm53.c's
 * glm_transpose (P5.1), copied rather than shared because this header must
 * stay self-contained (other engines and the unit tests include it
 * directly). N = heads, K = dim; called once per row, not once per pool. */
static inline void coli_index_transpose(float *aT, const float *a, int N, int K) {
    const int nb = N & ~7, kb = K & ~7;
    for (int n = 0; n < nb; n += 8)
        for (int k = 0; k < kb; k += 8)
            for (int nn = 0; nn < 8; nn++)
                for (int kk = 0; kk < 8; kk++)
                    aT[(size_t)(k + kk) * N + n + nn] = a[(size_t)(n + nn) * K + k + kk];
    for (int n = 0; n < N; n++)
        for (int k = (n < nb) ? kb : 0; k < K; k++)
            aT[(size_t)k * N + n] = a[(size_t)n * K + k];
}

/* dst[h] = sum_d aT[d][h] * b[d], for all N=heads lanes at once, each lane
 * accumulating over d in ascending order with the product rounded before it
 * is added -- no scale multiply here (unlike glm_lane_dots): the score pass
 * applies `head_w * dot * scale` in that exact left-to-right order after the
 * ReLU test, and folding scale in here would change which multiply happens
 * first. The 64- and 32-lane blocks are what break the add-chain latency;
 * the 8-lane block and the scalar tail cover any head count. */
static inline void coli_index_lane_dots(float *dst, const float *aT, const float *b,
                                        int N, int K) {
    const int H = N, L = K;
    const float *qT = aT, *c_j = b;
    int h = 0;
#ifdef COLI_INDEX_HEADVEC
    for (; h + 64 <= H; h += 64) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b8 = _mm256_broadcast_ss(c_j + d);
            COLI_IDX_MULADD(a0, b8, _mm256_loadu_ps(p));
            COLI_IDX_MULADD(a1, b8, _mm256_loadu_ps(p + 8));
            COLI_IDX_MULADD(a2, b8, _mm256_loadu_ps(p + 16));
            COLI_IDX_MULADD(a3, b8, _mm256_loadu_ps(p + 24));
            COLI_IDX_MULADD(a4, b8, _mm256_loadu_ps(p + 32));
            COLI_IDX_MULADD(a5, b8, _mm256_loadu_ps(p + 40));
            COLI_IDX_MULADD(a6, b8, _mm256_loadu_ps(p + 48));
            COLI_IDX_MULADD(a7, b8, _mm256_loadu_ps(p + 56));
        }
        _mm256_storeu_ps(dst + h,      a0);
        _mm256_storeu_ps(dst + h + 8,  a1);
        _mm256_storeu_ps(dst + h + 16, a2);
        _mm256_storeu_ps(dst + h + 24, a3);
        _mm256_storeu_ps(dst + h + 32, a4);
        _mm256_storeu_ps(dst + h + 40, a5);
        _mm256_storeu_ps(dst + h + 48, a6);
        _mm256_storeu_ps(dst + h + 56, a7);
    }
    for (; h + 32 <= H; h += 32) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b8 = _mm256_broadcast_ss(c_j + d);
            COLI_IDX_MULADD(a0, b8, _mm256_loadu_ps(p));
            COLI_IDX_MULADD(a1, b8, _mm256_loadu_ps(p + 8));
            COLI_IDX_MULADD(a2, b8, _mm256_loadu_ps(p + 16));
            COLI_IDX_MULADD(a3, b8, _mm256_loadu_ps(p + 24));
        }
        _mm256_storeu_ps(dst + h,      a0);
        _mm256_storeu_ps(dst + h + 8,  a1);
        _mm256_storeu_ps(dst + h + 16, a2);
        _mm256_storeu_ps(dst + h + 24, a3);
    }
    for (; h + 8 <= H; h += 8) {
        __m256 a0 = _mm256_setzero_ps();
        const float *p = qT + h;
        for (int d = 0; d < L; d++, p += H) {
            const __m256 b8 = _mm256_broadcast_ss(c_j + d);
            COLI_IDX_MULADD(a0, b8, _mm256_loadu_ps(p));
        }
        _mm256_storeu_ps(dst + h, a0);
    }
#endif
    /* Portable tail for a head count not a multiple of 8 (and the whole
     * dot, on a build without AVX2/FMA): plain per-d multiply-then-add, in
     * the SAME order the reference computes. `#pragma STDC FP_CONTRACT OFF`
     * -- unlike the empty-asm trick above, needed only where the compiler
     * fuses a same-width intrinsic mul+add it issued itself -- reliably
     * blocks contraction for a plain scalar reduction like this one on both
     * GCC and Clang; verified against a mismatch this exact loop produced
     * without it (F9a record). */
    {
#pragma STDC FP_CONTRACT OFF
        for (; h < H; h++) {
            float dot = 0.0f;
            for (int d = 0; d < L; d++) dot += qT[(size_t)d * H + h] * c_j[d];
            dst[h] = dot;
        }
    }
}

/* A candidate pool for the top-`wanted` selection: its score and its own
 * pool index, so ties can still be broken to the lower index after the
 * score alone stops distinguishing two entries. */
typedef struct { float score; int idx; } coli_index_cand_t;

/* Is `a` strictly worse than `b` in the selection order -- lower score, or
 * (tied score) the higher pool index? This is the total order the original
 * greedy scan's `scores[p] > scores[best]` (strict) already imposes by
 * scanning pools ascending and only ever replacing `best` on a strict
 * improvement: the first pool (lowest index) at a given score is never
 * displaced by a later one at the same score. */
static inline int coli_index_cand_worse(const coli_index_cand_t *a, const coli_index_cand_t *b) {
    if (a->score != b->score) return a->score < b->score;
    return a->idx > b->idx;
}

static inline void coli_index_heap_sift_down(coli_index_cand_t *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, m = i;
        if (l < n && coli_index_cand_worse(&h[l], &h[m])) m = l;
        if (r < n && coli_index_cand_worse(&h[r], &h[m])) m = r;
        if (m == i) break;
        coli_index_cand_t t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static inline void coli_index_heap_sift_up(coli_index_cand_t *h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!coli_index_cand_worse(&h[i], &h[p])) break;
        coli_index_cand_t t = h[i]; h[i] = h[p]; h[p] = t;
        i = p;
    }
}

/* Descending by score, ties ascending by index -- rank order, lowest index
 * first among ties, exactly what the original scan emits. */
static inline int coli_index_cand_cmp_desc(const void *pa, const void *pb) {
    const coli_index_cand_t *a = pa, *b = pb;
    if (a->score != b->score) return a->score < b->score ? 1 : -1;
    return a->idx - b->idx;
}

/* Exact equivalent of "scan `wanted` times, each time take the strict
 * maximum of the remaining pools, ties to the lower index" -- computed with
 * a bounded min-heap of size `wanted` (O(pools log wanted)) instead of the
 * O(wanted * pools) repeated scan. `heap` is caller-owned scratch of at
 * least `wanted` entries. Pools at exactly -FLT_MAX (incomplete, or not yet
 * causally visible -- see the callers) are excluded, matching the
 * `scores[p] > -FLT_MAX` filter the scan applies before ever comparing a
 * pool to `best`. Returns how many entries were selected (< wanted when
 * fewer than `wanted` pools were eligible); the caller's row was already
 * initialised to -1 and is left that way past this count, same as the
 * scan's own `break`. */
static inline int coli_index_select_topk(int *row, const float *scores, int pools,
                                          int pool, int first, int wanted,
                                          coli_index_cand_t *heap) {
    int hn = 0;
    for (int p = 0; p < pools; p++) {
        if (scores[p] == -FLT_MAX) continue;
        coli_index_cand_t cand = { scores[p], p };
        if (hn < wanted) {
            heap[hn] = cand;
            coli_index_heap_sift_up(heap, hn);
            hn++;
        } else if (coli_index_cand_worse(&heap[0], &cand)) {
            heap[0] = cand;
            coli_index_heap_sift_down(heap, hn, 0);
        }
    }
    qsort(heap, (size_t)hn, sizeof(*heap), coli_index_cand_cmp_desc);
    for (int rank = 0; rank < hn; rank++)
        for (int j = 0; j < pool; j++)
            row[rank * pool + j] = first + heap[rank].idx * pool + j;
    return hn;
}

/* One row's score pass, pools as the parallel axis (G7's pattern, verbatim:
 * each iteration writes only its own scores[p], reads pool_base/queries/
 * head_w/complete read-only, no shared accumulator -- no score's own
 * summation order changes, only the order pools are computed in). Used for
 * the few-rows case (decode): a single row has no other axis to give the
 * other cores. */
static inline void coli_sparse_index_score_row(float *scores, const float *pool_base,
        const float *queries, const float *head_w, const unsigned char *complete,
        int q, int q_from, int first, int pools, int heads, int dim, int pool, float scale) {
    /* F9a: this row is fixed for the whole call, so its transpose is done
     * ONCE here, not per pool -- every one of the (parallel) pool iterations
     * below reads the same qT read-only. One dots[] scratch per thread
     * (the parallel axis here is pools, not rows), 64-byte strided so two
     * threads never share a cache line (P5b.1's false-sharing lesson). */
    const int lanes = coli_sparse_index_lanes_usable(dim);
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const size_t dstride = (((size_t)heads * sizeof(float) + 63) / 64) * 64 / sizeof(float);
    float *qT = NULL, *dots_pool = NULL;
    if (lanes) {
        qT = malloc((size_t)dim * heads * sizeof(float));
        dots_pool = malloc((size_t)nthreads * dstride * sizeof(float));
        if (qT && dots_pool)
            coli_index_transpose(qT, queries + (size_t)(q - q_from) * heads * dim, heads, dim);
        else { free(qT); free(dots_pool); qT = NULL; dots_pool = NULL; }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < pools; p++) {
        const int last = first + (p + 1) * pool - 1;
        if (!complete[p] || last > q) { scores[p] = -FLT_MAX; continue; }
        const float *pv = pool_base + (size_t)p * dim;
        float score = 0.0f;
        if (qT) {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            float *dots = dots_pool + (size_t)tid * dstride;
            coli_index_lane_dots(dots, qT, pv, heads, dim);
            for (int h = 0; h < heads; h++) {
                const float dot = dots[h];
                if (dot > 0.0f)                              /* ReLU */
                    score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
            }
        } else {
            for (int h = 0; h < heads; h++) {
                const float *query = queries + ((size_t)(q - q_from) * heads + h) * dim;
                float dot = 0.0f;
                for (int d = 0; d < dim; d++) dot += query[d] * pv[d];
                if (dot > 0.0f)                              /* ReLU */
                    score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
            }
        }
        scores[p] = score;
    }
    free(qT);
    free(dots_pool);
}

/* The score pass and the top-`wanted` selection, shared by the plain and
 * cached pooled-key paths below -- their score/select/tail loops were
 * byte-for-byte identical, differing only in which pooled-key buffer they
 * read (`pooled` vs `pool_cache`), which is `pool_base` here.
 *
 * Two axes, chosen once per call:
 *   - many rows (prefill, q_to-q_from >= the thread count): parallel over
 *     ROWS, one thread per row end to end, its own scratch -- fewer and
 *     cheaper barriers than one per row;
 *   - few rows (decode): parallel over POOLS within each row's score pass
 *     (the axis above), one row at a time.
 * Both compute exactly the serial code's own numbers, only in a different
 * order (§G7's argument): no score's own summation order changes, and the
 * selection is the same exact top-`wanted` by construction (see
 * coli_index_select_topk). GLM53_INDEX_SCALAR=1 forces the original
 * single-threaded O(wanted*pools) scan, unchanged, for the A/B.
 *
 * Returns 0, or -1 on allocation failure (the caller frees its own buffers
 * and returns -1 in turn, same as every other failure path here). */
static inline int coli_sparse_index_score_select(int *out, const float *pool_base,
        const float *queries, const float *head_w, const unsigned char *complete,
        const unsigned char *valid, int heads, int dim, int pool,
        int wanted, int with_tail, int topk, int width,
        int q_from, int q_to, int first, int pools) {
    const float scale = 1.0f / sqrtf((float)dim);
    const int scalar = coli_sparse_index_scalar_on();
#ifdef _OPENMP
    const int nthreads = scalar ? 1 : omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    const int rows = q_to - q_from;

    if (scalar) {
        float *scores = malloc((size_t)pools * sizeof(float));
        unsigned char *taken = calloc((size_t)pools, 1);
        if (!scores || !taken) { free(scores); free(taken); return -1; }
        for (int q = q_from; q < q_to; q++) {
            int *row = out + (size_t)(q - q_from) * width;
            for (int i = 0; i < width; i++) row[i] = -1;
            if (valid[q]) {
                for (int p = 0; p < pools; p++) {
                    const int last = first + (p + 1) * pool - 1;
                    if (!complete[p] || last > q) { scores[p] = -FLT_MAX; continue; }
                    const float *pv = pool_base + (size_t)p * dim;
                    float score = 0.0f;
                    for (int h = 0; h < heads; h++) {
                        const float *query = queries + ((size_t)(q - q_from) * heads + h) * dim;
                        float dot = 0.0f;
                        for (int d = 0; d < dim; d++) dot += query[d] * pv[d];
                        if (dot > 0.0f)
                            score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
                    }
                    scores[p] = score;
                }
                for (int rank = 0; rank < wanted; rank++) {
                    int best = -1;
                    for (int p = 0; p < pools; p++)
                        if (!taken[p] && scores[p] > -FLT_MAX &&
                            (best < 0 || scores[p] > scores[best])) best = p;
                    if (best < 0) break;
                    taken[best] = 1;
                    for (int j = 0; j < pool; j++) row[rank * pool + j] = first + best * pool + j;
                }
                memset(taken, 0, (size_t)pools);
            }
            if (with_tail) {
                int visible = 0;
                for (int i = first; i <= q; i++) if (valid[i]) visible++;
                const int tail = visible % pool;
                const int tail_start = first + visible - tail;
                for (int j = 0; j < tail && j < pool - 1; j++)
                    if (tail_start + j <= q && valid[tail_start + j])
                        row[topk + j] = tail_start + j;
            }
        }
        free(scores); free(taken);
        return 0;
    }

    if (nthreads > 1 && rows >= nthreads) {
        /* Per-thread scratch, 64-byte aligned and cache-line strided
         * (P5b.1's false-sharing lesson) so two threads' buffers never
         * share a line. F9a: qt_pool/dots_pool are the same idea for the
         * head-lane path -- allocated ONCE per call per thread here, not
         * once per row: each thread re-transposes into its own qT slice at
         * the start of every row it owns (dim*heads flops, once), then every
         * one of that row's ~pools pool iterations does an O(heads) lane
         * dot against it instead of an O(heads*dim) scalar one. */
        const size_t sstride = (((size_t)pools * sizeof(float) + 63) / 64) * 64 / sizeof(float);
        const size_t hstride = (((size_t)wanted * sizeof(coli_index_cand_t) + 63) / 64) * 64
                                / sizeof(coli_index_cand_t);
        const size_t qtstride = (((size_t)dim * heads * sizeof(float) + 63) / 64) * 64
                                / sizeof(float);
        const size_t dstride = (((size_t)heads * sizeof(float) + 63) / 64) * 64 / sizeof(float);
        float *scores_pool = malloc((size_t)nthreads * sstride * sizeof(float));
        coli_index_cand_t *heap_pool =
            malloc((size_t)nthreads * hstride * sizeof(coli_index_cand_t));
        if (!scores_pool || !heap_pool) { free(scores_pool); free(heap_pool); return -1; }
        const int lanes = coli_sparse_index_lanes_usable(dim);
        float *qt_pool = lanes ? malloc((size_t)nthreads * qtstride * sizeof(float)) : NULL;
        float *dots_pool = lanes ? malloc((size_t)nthreads * dstride * sizeof(float)) : NULL;
        const int lanes_ok = lanes && qt_pool && dots_pool;
        if (lanes && !lanes_ok) { free(qt_pool); free(dots_pool); qt_pool = NULL; dots_pool = NULL; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int q = q_from; q < q_to; q++) {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            float *scores = scores_pool + (size_t)tid * sstride;
            coli_index_cand_t *heap = heap_pool + (size_t)tid * hstride;
            float *qT = lanes_ok ? qt_pool + (size_t)tid * qtstride : NULL;
            float *dots = lanes_ok ? dots_pool + (size_t)tid * dstride : NULL;
            int *row = out + (size_t)(q - q_from) * width;
            for (int i = 0; i < width; i++) row[i] = -1;
            if (valid[q]) {
                if (qT)
                    coli_index_transpose(qT, queries + (size_t)(q - q_from) * heads * dim,
                                         heads, dim);
                for (int p = 0; p < pools; p++) {
                    const int last = first + (p + 1) * pool - 1;
                    if (!complete[p] || last > q) { scores[p] = -FLT_MAX; continue; }
                    const float *pv = pool_base + (size_t)p * dim;
                    float score = 0.0f;
                    if (qT) {
                        coli_index_lane_dots(dots, qT, pv, heads, dim);
                        for (int h = 0; h < heads; h++) {
                            const float dot = dots[h];
                            if (dot > 0.0f)
                                score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
                        }
                    } else {
                        for (int h = 0; h < heads; h++) {
                            const float *query = queries + ((size_t)(q - q_from) * heads + h) * dim;
                            float dot = 0.0f;
                            for (int d = 0; d < dim; d++) dot += query[d] * pv[d];
                            if (dot > 0.0f)
                                score += head_w[(size_t)(q - q_from) * heads + h] * dot * scale;
                        }
                    }
                    scores[p] = score;
                }
                coli_index_select_topk(row, scores, pools, pool, first, wanted, heap);
            }
            if (with_tail) {
                int visible = 0;
                for (int i = first; i <= q; i++) if (valid[i]) visible++;
                const int tail = visible % pool;
                const int tail_start = first + visible - tail;
                for (int j = 0; j < tail && j < pool - 1; j++)
                    if (tail_start + j <= q && valid[tail_start + j])
                        row[topk + j] = tail_start + j;
            }
        }
        free(scores_pool); free(heap_pool); free(qt_pool); free(dots_pool);
        return 0;
    }

    /* Decode: one row at a time, parallel over pools within the score pass
     * (the router's own axis, G7). */
    {
        float *scores = malloc((size_t)pools * sizeof(float));
        coli_index_cand_t *heap = malloc((size_t)wanted * sizeof(coli_index_cand_t));
        if (!scores || !heap) { free(scores); free(heap); return -1; }
        for (int q = q_from; q < q_to; q++) {
            int *row = out + (size_t)(q - q_from) * width;
            for (int i = 0; i < width; i++) row[i] = -1;
            if (valid[q]) {
                coli_sparse_index_score_row(scores, pool_base, queries, head_w, complete,
                                             q, q_from, first, pools, heads, dim, pool, scale);
                coli_index_select_topk(row, scores, pools, pool, first, wanted, heap);
            }
            if (with_tail) {
                int visible = 0;
                for (int i = first; i <= q; i++) if (valid[i]) visible++;
                const int tail = visible % pool;
                const int tail_start = first + visible - tail;
                for (int j = 0; j < tail && j < pool - 1; j++)
                    if (tail_start + j <= q && valid[tail_start + j])
                        row[topk + j] = tail_start + j;
            }
        }
        free(scores); free(heap);
    }
    return 0;
}

/* Select the key blocks each query may attend.
 *
 * In the range form, `keys`, `gates` and `valid` cover the whole sequence,
 * because a query may attend anywhere behind it. `queries`, `head_w` and `out`
 * cover only [q_from, q_to) and are indexed from zero: during decode the whole
 * prefix is already in the cache and only the new token has a query, so making
 * the caller allocate prefix-length query arrays would be asking it to build
 * what it deliberately no longer computes.
 *
 *   out       [sequence * width]  token indices, -1 where unused
 *   queries   [sequence * heads * dim]
 *   keys      [sequence * dim]     one indexer key per token (shared by heads)
 *   gates     [sequence * dim]     pooling logits, per channel
 *   head_w    [sequence * heads]   already scaled by heads^-0.5 by the caller
 *   ape       [pool * dim]         per-position bias inside a pool
 *   valid     [sequence]           0 marks padding
 *
 * Returns 0, or -1 on bad arguments or allocation failure. */
static inline int coli_sparse_index_select_range(int *out, const float *queries,
                                                 const float *keys, const float *gates,
                                                 const float *head_w, const float *ape,
                                                 const unsigned char *valid,
                                                 int sequence, int heads, int dim,
                                                 int pool, int topk, int with_tail,
                                                 int q_from, int q_to) {
    if (!out || !queries || !keys || !gates || !head_w || !ape || !valid ||
        q_from < 0 || q_to > sequence || q_from > q_to ||
        sequence < 1 || heads < 1 || dim < 1 || pool < 1 || topk < pool || topk % pool)
        return -1;

    const int width = coli_sparse_index_width(topk, pool, with_tail);
    const int pools = (sequence + pool - 1) / pool;
    const int wanted = topk / pool;

    float *pooled = calloc((size_t)pools * dim, sizeof(*pooled));
    unsigned char *complete = calloc((size_t)pools, 1);
    if (!pooled || !complete) {
        free(complete); free(pooled);
        return -1;
    }

    int first = 0;
    while (first < sequence && !valid[first]) first++;

    /* Pool the keys: per channel, a softmax over the pool's members using the
     * gate logits plus the positional bias. */
    for (int p = 0; p < pools; p++) {
        const int start = first + p * pool;
        complete[p] = start + pool <= sequence;
        for (int j = 0; complete[p] && j < pool; j++)
            if (!valid[start + j]) complete[p] = 0;
        if (!complete[p]) continue;
        for (int d = 0; d < dim; d++) {
            float maximum = -FLT_MAX;
            for (int j = 0; j < pool; j++) {
                const float logit = gates[(size_t)(start + j) * dim + d] +
                                    ape[(size_t)j * dim + d];
                if (logit > maximum) maximum = logit;
            }
            float total = 0.0f;
            for (int j = 0; j < pool; j++)
                total += expf(gates[(size_t)(start + j) * dim + d] +
                              ape[(size_t)j * dim + d] - maximum);
            float mixed = 0.0f;
            for (int j = 0; j < pool; j++) {
                const float weight = expf(gates[(size_t)(start + j) * dim + d] +
                                          ape[(size_t)j * dim + d] - maximum) / total;
                mixed += weight * keys[(size_t)(start + j) * dim + d];
            }
            pooled[(size_t)p * dim + d] = mixed;
        }
    }

    if (coli_sparse_index_score_select(out, pooled, queries, head_w, complete, valid,
            heads, dim, pool, wanted, with_tail, topk, width,
            q_from, q_to, first, pools)) {
        free(complete);
        free(pooled);
        return -1;
    }

    free(complete);
    free(pooled);
    return 0;
}

/* G5: same selection, but a pool's mixed key is cached instead of rebuilt on
 * every call.
 *
 * The pooling loop above computes pooled[p] from keys/gates/ape of pool p's
 * own members only -- never from the query -- so once a pool is COMPLETE
 * (all `pool` members present and causally behind every future query, which
 * for an append-only KV cache means "behind the current sequence length")
 * its mixed key never changes again. The plain function above still redoes
 * that mix for every pool on every call, which is what makes it O(context)
 * per decode token and O(context^2) per generation -- ported here from
 * Qwen's block-key cache (2d3cf7e, qwen38_core.h's q38_attention/IK_pooled),
 * whose commit message is the derivation: q38_rope() there and the softmax
 * mix here both use only the block's/pool's own fixed position, so the
 * result is identical no matter which later token scores it.
 *
 * `pool_cache` is a caller-owned buffer sized to hold every pool this layer
 * will ever have ([pools_cap][dim]); `pool_cache_count` is how many LEADING
 * pools it already holds a valid mix for. Both live for the life of one
 * session (GLM's GLayerState, one per DSA layer) and must be reset to count
 * 0 -- pool_cache itself does not need clearing, only unreachable pools are
 * ever read -- whenever a session starts fresh or rewinds to a shorter
 * prefix than it last held: a fresh session_open, or a checkpoint/segment
 * restore into one. session_open already calloc's GLayerState, which zeroes
 * the count for the ordinary "new session" case for free; restore sites are
 * the ones that must do it explicitly, exactly as Q38's four
 * q38_ik_pool_reset() call sites did for the same reason.
 *
 * Passing pool_cache == NULL or pool_cache_count == NULL falls back to the
 * uncached function above unconditionally -- the two are bit-identical on
 * every input, this only changes how much repeated work is thrown away. */
static inline int coli_sparse_index_select_range_cached(int *out, const float *queries,
                                                 const float *keys, const float *gates,
                                                 const float *head_w, const float *ape,
                                                 const unsigned char *valid,
                                                 int sequence, int heads, int dim,
                                                 int pool, int topk, int with_tail,
                                                 int q_from, int q_to,
                                                 float *pool_cache, int *pool_cache_count) {
    if (!pool_cache || !pool_cache_count)
        return coli_sparse_index_select_range(out, queries, keys, gates, head_w, ape,
                                              valid, sequence, heads, dim, pool, topk,
                                              with_tail, q_from, q_to);
    if (!out || !queries || !keys || !gates || !head_w || !ape || !valid ||
        q_from < 0 || q_to > sequence || q_from > q_to ||
        sequence < 1 || heads < 1 || dim < 1 || pool < 1 || topk < pool || topk % pool)
        return -1;

    const int width = coli_sparse_index_width(topk, pool, with_tail);
    const int pools = (sequence + pool - 1) / pool;
    const int wanted = topk / pool;

    unsigned char *complete = calloc((size_t)pools, 1);
    if (!complete) {
        free(complete);
        return -1;
    }

    int first = 0;
    while (first < sequence && !valid[first]) first++;

    int cached = *pool_cache_count;
    if (cached > pools) cached = pools;   /* defensive: sequence should only grow */
    for (int p = 0; p < cached; p++) complete[p] = 1;

    /* Extend the cached prefix by exactly as many pools as are newly
     * complete and contiguous with what is already cached -- the same
     * "blocks only grow" invariant Qwen's cache relies on. A pool found
     * incomplete stops the *count* from advancing past it (so a later call
     * still recomputes it once it does close) but the loop keeps going so
     * every complete pool still gets a mix this call, cached or not. */
    int frontier = cached;
    for (int p = cached; p < pools; p++) {
        const int start = first + p * pool;
        complete[p] = start + pool <= sequence;
        for (int j = 0; complete[p] && j < pool; j++)
            if (!valid[start + j]) complete[p] = 0;
        if (!complete[p]) continue;
        float *dst = pool_cache + (size_t)p * dim;
        for (int d = 0; d < dim; d++) {
            float maximum = -FLT_MAX;
            for (int j = 0; j < pool; j++) {
                const float logit = gates[(size_t)(start + j) * dim + d] +
                                    ape[(size_t)j * dim + d];
                if (logit > maximum) maximum = logit;
            }
            float total = 0.0f;
            for (int j = 0; j < pool; j++)
                total += expf(gates[(size_t)(start + j) * dim + d] +
                              ape[(size_t)j * dim + d] - maximum);
            float mixed = 0.0f;
            for (int j = 0; j < pool; j++) {
                const float weight = expf(gates[(size_t)(start + j) * dim + d] +
                                          ape[(size_t)j * dim + d] - maximum) / total;
                mixed += weight * keys[(size_t)(start + j) * dim + d];
            }
            dst[d] = mixed;
        }
        if (p == frontier) frontier = p + 1;
    }
    *pool_cache_count = frontier;

    if (coli_sparse_index_score_select(out, pool_cache, queries, head_w, complete, valid,
            heads, dim, pool, wanted, with_tail, topk, width,
            q_from, q_to, first, pools)) {
        free(complete);
        return -1;
    }

    free(complete);
    return 0;
}

/* Tutte le query, che e' il caso del prefill. */
static inline int coli_sparse_index_select(int *out, const float *queries,
                                           const float *keys, const float *gates,
                                           const float *head_w, const float *ape,
                                           const unsigned char *valid,
                                           int sequence, int heads, int dim,
                                           int pool, int topk, int with_tail) {
    return coli_sparse_index_select_range(out, queries, keys, gates, head_w, ape,
                                          valid, sequence, heads, dim, pool, topk,
                                          with_tail, 0, sequence);
}

/* Softmax attention restricted to the selected indices.
 *
 *   out      [sequence * heads * value_dim]
 *   indices  [sequence * width], -1 entries skipped
 *
 * Queries, keys and values are laid out [position][head][dim]. */
static inline int coli_sparse_attention_range(float *out, const float *queries,
                                             const float *keys, const float *values,
                                             const int *indices, int sequence, int width,
                                             int heads, int key_dim, int value_dim,
                                             int q_from, int q_to) {
    if (!out || !queries || !keys || !values || !indices || sequence < 1 ||
        width < 1 || heads < 1 || key_dim < 1 || value_dim < 1 ||
        q_from < 0 || q_to > sequence || q_from > q_to) return -1;
    float *scores = malloc((size_t)width * sizeof(*scores));
    if (!scores) return -1;
    const float scale = 1.0f / sqrtf((float)key_dim);
    for (int q = q_from; q < q_to; q++) {
        const int *selected = indices + (size_t)(q - q_from) * width;
        for (int h = 0; h < heads; h++) {
            const float *query = queries + ((size_t)(q - q_from) * heads + h) * key_dim;
            float maximum = -FLT_MAX;
            int used = 0;
            for (int slot = 0; slot < width; slot++) {
                const int key_position = selected[slot];
                if (key_position < 0 || key_position >= sequence) continue;
                const float *key = keys + ((size_t)key_position * heads + h) * key_dim;
                float dot = 0.0f;
                for (int d = 0; d < key_dim; d++) dot += query[d] * key[d];
                scores[used] = dot * scale;
                if (scores[used] > maximum) maximum = scores[used];
                used++;
            }
            float *result = out + ((size_t)(q - q_from) * heads + h) * value_dim;
            memset(result, 0, (size_t)value_dim * sizeof(*result));
            if (!used) continue;
            float total = 0.0f;
            for (int i = 0; i < used; i++) {
                scores[i] = expf(scores[i] - maximum);
                total += scores[i];
            }
            int index = 0;
            for (int slot = 0; slot < width; slot++) {
                const int key_position = selected[slot];
                if (key_position < 0 || key_position >= sequence) continue;
                const float weight = scores[index++] / total;
                const float *value = values + ((size_t)key_position * heads + h) * value_dim;
                for (int d = 0; d < value_dim; d++) result[d] += weight * value[d];
            }
        }
    }
    free(scores);
    return 0;
}

/* Tutte le query, che e' il caso del prefill. */
static inline int coli_sparse_attention(float *out, const float *queries,
                                        const float *keys, const float *values,
                                        const int *indices, int sequence, int width,
                                        int heads, int key_dim, int value_dim) {
    return coli_sparse_attention_range(out, queries, keys, values, indices, sequence,
                                       width, heads, key_dim, value_dim, 0, sequence);
}

#endif /* COLIBRI_SPARSE_INDEX_H */
