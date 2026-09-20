/* f8_cpu_expert_bench.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F8, step 0.
 *
 * The question this answers, and nothing else: at batch 1 (decode), where
 * does the 0.56 ms/expert of a CPU-computed GLM-5.3 expert go -- compute,
 * OpenMP fork/join, or memory bandwidth? (record sec F1-STEP0: 3.12
 * non-resident experts x 0.560 ms = 1.747 ms/window, 73 ms of a 243 ms
 * decode token, the largest decode bucket and the one on the critical path).
 *
 * This links quant.h directly and reproduces glm53.c's own CPU expert path
 * (mlp3_cpu's default plain branch, bit-identical to mlp3_cpu_rows at
 * nr=1 -- the source comment at mlp3_cpu_rows says so, and this file's own
 * checksum check at startup proves it against a from-scratch rebuild of the
 * same three stages) so what is timed is the code that runs, not a model of
 * it. siluf_/swiglu_clamped are copied verbatim from c/glm53.c (they are
 * static there, not in a header) -- 4 lines, unchanged.
 *
 * Real GLM-5.3 shapes (rig's model config.json, text_config): hidden=4096,
 * moe_intermediate_size=2048, group size 64 -> gate/up [O=2048,I=4096],
 * down [O=4096,I=2048], fmt=4 (int4 g64). 14.16 MB/expert: 3 x (O*(I+1)/2
 * q4 bytes + O*ceil(I/64)*4 scale bytes) = 12 582 912 + 1 572 864.
 *
 * Fidelity: L3 is 128 MB, one expert is 14.16 MB -- a pool of a handful of
 * experts lies (record has a kernel that was fast in L3 and slow in the
 * engine, sec G11/i4_kernel_bench.c). This pool defaults to 350 distinct
 * experts (4.95 GB), visited in fixed-seed random order, so a revisit is
 * separated by ~4.9 GB of other experts, never an L3 hit.
 *
 * Weights are random int4 nibbles and scales ~U(0,0.06) (plausible g64
 * dequant scale magnitude; the exact value does not affect timing, only
 * numerics, and numerics are not the point here -- outputs are meaningless).
 * Nothing is read from or written to the model directory.
 *
 * Every timed variant except "expert-parallel" (item 5, a different row/
 * thread partition by construction) is checked against the baseline
 * (mlp3_like, called 3x serially) for bit-exactness on one window before
 * either is timed, the way i4_kernel_bench.c (G11) does it -- a nonzero
 * diff there is reported, not hidden.
 *
 *   f8_cpu_expert_bench [n_windows] [pool_experts] [seed]
 *
 * Output: one `ROW kind=... median_ms=... p90_ms=... gmac_s=... gbytes_s=...`
 * line per configuration, `INFO`/`WARN` for machine facts, one closing
 * `INFO` line naming the best variant's ms/window against 1.747.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../c/quant.h"

/* ---------------- real GLM-5.3 shapes (rig config.json) ------------------ */
#define HIDDEN 4096
#define INTER  2048
#define GS     64
#define LIMIT  10.0f            /* swiglu_limit, matches CLAUDE.md / glm53.c */

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* xorshift32, fixed seed unless overridden -- same generator shape as the
 * other tools/hot-expert probes (f7_attn_probe.c, i4_kernel_bench.c). */
static uint32_t S_RND = 12345u;
static uint32_t rndu(void) { S_RND ^= S_RND << 13; S_RND ^= S_RND >> 17; S_RND ^= S_RND << 5; return S_RND; }
static float rnd01(void) { return (float)(rndu() >> 8) / 16777216.0f; }

/* ---- copied verbatim from c/glm53.c (static there, not in a header) ----- */
static float siluf_(float x) { return x / (1.0f + expf(-x)); }
static void swiglu_clamped(float *gate, const float *up, int n, float limit) {
    for (int i = 0; i < n; i++) {
        float g = gate[i] > limit ? limit : gate[i];
        float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
        gate[i] = siluf_(g) * u;
    }
}

/* ---------------------------------------------------------------------- */
typedef struct { int O, I, gs; uint8_t *q4; float *s; } Mat3;

static void mat3_alloc(Mat3 *m, int O, int I, int gs) {
    m->O = O; m->I = I; m->gs = gs;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    if (posix_memalign((void **)&m->q4, 64, (size_t)O * rb) != 0) { perror("memalign q4"); exit(1); }
    if (posix_memalign((void **)&m->s, 64, (size_t)O * ng * sizeof(float)) != 0) { perror("memalign s"); exit(1); }
    for (size_t i = 0; i < (size_t)O * rb; i++) m->q4[i] = (uint8_t)rndu();
    for (int i = 0; i < O * ng; i++) m->s[i] = rnd01() * 0.06f;
}

typedef struct { Mat3 gate, up, down; } Expert;

static size_t expert_bytes(void) {
    int grb = (HIDDEN + 1) / 2, gng = (HIDDEN + GS - 1) / GS;   /* gate/up: O=INTER,I=HIDDEN */
    int drb = (INTER + 1) / 2, dng = (INTER + GS - 1) / GS;     /* down:    O=HIDDEN,I=INTER */
    size_t gu = (size_t)INTER * grb + (size_t)INTER * gng * 4;
    size_t dn = (size_t)HIDDEN * drb + (size_t)HIDDEN * dng * 4;
    return 2 * gu + dn;
}

static void expert_alloc(Expert *e) {
    mat3_alloc(&e->gate, INTER, HIDDEN, GS);
    mat3_alloc(&e->up,   INTER, HIDDEN, GS);
    mat3_alloc(&e->down, HIDDEN, INTER, GS);
}

/* ---- the engine's own path: mlp3_cpu's default plain branch (glm53.c
 * lines ~3530-3561), bit-identical to mlp3_cpu_rows at nr=1 per that
 * function's own comment ("Row r of out is bit-identical to mlp3_cpu on
 * row r of x"). x: HIDDEN floats in. out: HIDDEN floats out. sg/su: INTER
 * scratch floats each, caller-owned like the engine's. ---------------- */
static void mlp3_like(float *out, const float *x, const Expert *e, float *sg, float *su) {
    const int Ig = e->gate.I, Og = e->gate.O, Id = e->down.I, Od = e->down.O;
    const int grb = (Ig + 1) / 2, gng = (Ig + e->gate.gs - 1) / e->gate.gs;
    const int urb = (e->up.I + 1) / 2, ung = (e->up.I + e->up.gs - 1) / e->up.gs;
    const int drb = (Id + 1) / 2, dng = (Id + e->down.gs - 1) / e->down.gs;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int z = 0; z < 2 * Og; z++) {
            if (z < Og)
                sg[z] = coli_i4_row(e->gate.q4 + (int64_t)z * grb, e->gate.s + (int64_t)z * gng, x, Ig, e->gate.gs);
            else {
                int o = z - Og;
                su[o] = coli_i4_row(e->up.q4 + (int64_t)o * urb, e->up.s + (int64_t)o * ung, x, e->up.I, e->up.gs);
            }
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < Og; i++) {
            float gv = sg[i] > LIMIT ? LIMIT : sg[i];
            float uv = su[i] < -LIMIT ? -LIMIT : (su[i] > LIMIT ? LIMIT : su[i]);
            sg[i] = siluf_(gv) * uv;
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int o = 0; o < Od; o++)
            out[o] = coli_i4_row(e->down.q4 + (int64_t)o * drb, e->down.s + (int64_t)o * dng, sg, Id, e->down.gs);
    }
}

/* independent rebuild of the same three stages (separate helper, so the
 * bit-exactness check is not "a function compared to itself"), used only
 * once at startup to cross-check mlp3_like -- not part of any timed path. */
static void mlp3_check(float *out, const float *x, const Expert *e, float *sg, float *su) {
    matmul_i4_grouped(sg, x, e->gate.q4, e->gate.s, 1, e->gate.I, e->gate.O, e->gate.gs);
    matmul_i4_grouped(su, x, e->up.q4, e->up.s, 1, e->up.I, e->up.O, e->up.gs);
    swiglu_clamped(sg, su, e->gate.O, LIMIT);
    matmul_i4_grouped(out, sg, e->down.q4, e->down.s, 1, e->down.I, e->down.O, e->down.gs);
}

/* item 4: the SAME 9 omp-for regions of 3 serial mlp3_like calls, but under
 * ONE outer #pragma omp parallel (one team spawn) instead of three. Same
 * per-row coli_i4_row calls in the same z/i/o order per expert -> bit-
 * identical to calling mlp3_like three times; only the number of team
 * spawns changes. */
static void mlp3_window_fused(float *out3, const float *x, const Expert *e3, float *sg3, float *su3) {
    const int Og = INTER, Od = HIDDEN, Ig = HIDDEN, Id = INTER;
    const int grb = (Ig + 1) / 2, gng = (Ig + GS - 1) / GS;
    const int drb = (Id + 1) / 2, dng = (Id + GS - 1) / GS;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (int w = 0; w < 3; w++) {
            const Expert *e = &e3[w];
            float *sg = sg3 + (size_t)w * Og, *su = su3 + (size_t)w * Og, *out = out3 + (size_t)w * Od;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int z = 0; z < 2 * Og; z++) {
                if (z < Og)
                    sg[z] = coli_i4_row(e->gate.q4 + (int64_t)z * grb, e->gate.s + (int64_t)z * gng, x, Ig, GS);
                else {
                    int o = z - Og;
                    su[o] = coli_i4_row(e->up.q4 + (int64_t)o * grb, e->up.s + (int64_t)o * gng, x, Ig, GS);
                }
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < Og; i++) {
                float gv = sg[i] > LIMIT ? LIMIT : sg[i];
                float uv = su[i] < -LIMIT ? -LIMIT : (su[i] > LIMIT ? LIMIT : su[i]);
                sg[i] = siluf_(gv) * uv;
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int o = 0; o < Od; o++)
                out[o] = coli_i4_row(e->down.q4 + (int64_t)o * drb, e->down.s + (int64_t)o * dng, sg, Id, GS);
        }
    }
}

/* ================= F8 arbitration round: reassociating row kernels =====
 * Fable's read of the first pass: perfectly linear 1->8 thread scaling is
 * the signature of a LATENCY-bound serial reduction (coli_i4_row's single
 * `acc` chained through `_mm256_fmadd_ps` inside each 64-wide group, 5-cycle
 * Zen2 FMA latency), not of a bandwidth ceiling -- G14 (record) diagnosed
 * and fixed exactly this with `coli_i4_row_f4` (4 independent accumulators,
 * shipped as `GLM53_I4_FAST`, isolated 1.26-1.65x, in-engine 1.095x). Tested
 * here at nr=1 (decode's own shape), engine structure, against the
 * mlp3_like/mlp3_window_fused baseline. coli_i4_row_f4 itself is quant.h's,
 * called directly (not copied) -- only the two callers below are new.
 *
 * coli_i4_row_f8: NOT a shipped kernel. A mechanical width-doubling of f4's
 * own documented technique (8 independent accumulators instead of 4, one
 * `_mm256_fmadd_ps` each per 64-wide group instead of two chained
 * per accumulator) -- written here only to see whether f4 already captured
 * most of the latency-hiding or whether more independent accumulators keep
 * paying off. Bit-trick decode identical to f4's (same magic-number nibble
 * unpack), so the only change from f4 is accumulator count / combine order
 * -- reassociation noise only, same class of change as f4 itself. */
static inline float coli_i4_row_f8(const uint8_t *w, const float *scl,
                                   const float *xs, int I, int gs) {
    float a = 0;
    const int ng = (I + gs - 1) / gs;
    for (int g = 0; g < ng; g++) {
        int base = g * gs, glen = gs; if (base + glen > I) glen = I - base;
        float sc = scl[g];
        int i = base;
#ifdef __AVX2__
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i magic_i = _mm256_set1_epi32(0x4B000000);
        const __m256  magic_f = _mm256_set1_ps(8388608.0f + 8.0f);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        for (; i + 64 <= base + glen; i += 64) {
            __m128i by0 = _mm_loadu_si128((const __m128i *)(w + (i >> 1)));
            __m128i by1 = _mm_loadu_si128((const __m128i *)(w + (i >> 1) + 16));
            __m128i lo0 = _mm_and_si128(by0, m4), hi0 = _mm_and_si128(_mm_srli_epi16(by0, 4), m4);
            __m128i lo1 = _mm_and_si128(by1, m4), hi1 = _mm_and_si128(_mm_srli_epi16(by1, 4), m4);
            __m128i n0 = _mm_unpacklo_epi8(lo0, hi0), n1 = _mm_unpackhi_epi8(lo0, hi0);
            __m128i n2 = _mm_unpacklo_epi8(lo1, hi1), n3 = _mm_unpackhi_epi8(lo1, hi1);
            __m256 w0 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(n0), magic_i)), magic_f);
            __m256 w1 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(n0, 8)), magic_i)), magic_f);
            __m256 w2 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(n1), magic_i)), magic_f);
            __m256 w3 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(n1, 8)), magic_i)), magic_f);
            __m256 w4 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(n2), magic_i)), magic_f);
            __m256 w5 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(n2, 8)), magic_i)), magic_f);
            __m256 w6 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(n3), magic_i)), magic_f);
            __m256 w7 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(n3, 8)), magic_i)), magic_f);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),      w0, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8),  w1, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 16), w2, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 24), w3, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 32), w4, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 40), w5, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 48), w6, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 56), w7, a7);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
                                    _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
        a = fmaf(hsum256(acc), sc, a);
#endif
        for (; i < base + glen; i += 2) {
            if (i + 1 < base + glen) { uint8_t byte = w[i >> 1];
                a += (xs[i] * (float)((int)(byte & 0xF) - 8) + xs[i + 1] * (float)((int)(byte >> 4) - 8)) * sc; }
            else { uint8_t byte = w[i >> 1]; a += xs[i] * (float)((int)(byte & 0xF) - 8) * sc; }
        }
    }
    return a;
}

typedef float (*RowFn)(const uint8_t *, const float *, const float *, int, int);

/* Generic single-expert path, same 3-stage/1-team structure as mlp3_like,
 * parametrised on the row kernel -- used for f4/f8, never for coli_i4_row
 * (mlp3_like stays the untouched, already-checked baseline). */
static void mlp3_kern(float *out, const float *x, const Expert *e, float *sg, float *su, RowFn row) {
    const int Ig = e->gate.I, Og = e->gate.O, Id = e->down.I, Od = e->down.O;
    const int grb = (Ig + 1) / 2, gng = (Ig + e->gate.gs - 1) / e->gate.gs;
    const int urb = (e->up.I + 1) / 2, ung = (e->up.I + e->up.gs - 1) / e->up.gs;
    const int drb = (Id + 1) / 2, dng = (Id + e->down.gs - 1) / e->down.gs;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int z = 0; z < 2 * Og; z++) {
            if (z < Og)
                sg[z] = row(e->gate.q4 + (int64_t)z * grb, e->gate.s + (int64_t)z * gng, x, Ig, e->gate.gs);
            else {
                int o = z - Og;
                su[o] = row(e->up.q4 + (int64_t)o * urb, e->up.s + (int64_t)o * ung, x, e->up.I, e->up.gs);
            }
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < Og; i++) {
            float gv = sg[i] > LIMIT ? LIMIT : sg[i];
            float uv = su[i] < -LIMIT ? -LIMIT : (su[i] > LIMIT ? LIMIT : su[i]);
            sg[i] = siluf_(gv) * uv;
        }
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int o = 0; o < Od; o++)
            out[o] = row(e->down.q4 + (int64_t)o * drb, e->down.s + (int64_t)o * dng, sg, Id, e->down.gs);
    }
}

/* Generic fused-window path (one team, 3 experts, same row kernel). */
static void mlp3_window_fused_kern(float *out3, const float *x, const Expert *e3, float *sg3, float *su3, RowFn row) {
    const int Og = INTER, Od = HIDDEN, Ig = HIDDEN, Id = INTER;
    const int grb = (Ig + 1) / 2, gng = (Ig + GS - 1) / GS;
    const int drb = (Id + 1) / 2, dng = (Id + GS - 1) / GS;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (int w = 0; w < 3; w++) {
            const Expert *e = &e3[w];
            float *sg = sg3 + (size_t)w * Og, *su = su3 + (size_t)w * Og, *out = out3 + (size_t)w * Od;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int z = 0; z < 2 * Og; z++) {
                if (z < Og)
                    sg[z] = row(e->gate.q4 + (int64_t)z * grb, e->gate.s + (int64_t)z * gng, x, Ig, GS);
                else {
                    int o = z - Og;
                    su[o] = row(e->up.q4 + (int64_t)o * grb, e->up.s + (int64_t)o * gng, x, Ig, GS);
                }
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < Og; i++) {
                float gv = sg[i] > LIMIT ? LIMIT : sg[i];
                float uv = su[i] < -LIMIT ? -LIMIT : (su[i] > LIMIT ? LIMIT : su[i]);
                sg[i] = siluf_(gv) * uv;
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int o = 0; o < Od; o++)
                out[o] = row(e->down.q4 + (int64_t)o * drb, e->down.s + (int64_t)o * dng, sg, Id, GS);
        }
    }
}

/* ---------------------------------------------------------------------- */
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static void stats(double *v, int n, double *median, double *p90) {
    qsort(v, n, sizeof(double), cmp_double);
    *median = v[n / 2];
    *p90 = v[(int)(n * 0.9)];
}
/* gmacs/gbytes are already in giga-units (caller divides by 1e9 once when
 * computing them from element/byte counts) -- do not divide again here. */
static void report(const char *kind, double *ms, int n, double gmacs, double gbytes) {
    double med, p90; stats(ms, n, &med, &p90);
    double gmac_s = gmacs / (med / 1000.0);
    double gbytes_s = gbytes / (med / 1000.0);
    printf("ROW kind=%-28s median_ms=%.4f p90_ms=%.4f gmac_s=%.2f gbytes_s=%.2f\n",
           kind, med, p90, gmac_s, gbytes_s);
}

/* random distinct triple of pool indices */
static void pick3(int pool_n, int *a, int *b, int *c) {
    *a = rndu() % pool_n;
    do { *b = rndu() % pool_n; } while (*b == *a);
    do { *c = rndu() % pool_n; } while (*c == *b || *c == *a);
}

static double window_gmacs(void) {
    /* one expert's MACs: gate + up + down, each O*I */
    double g = (double)INTER * HIDDEN, u = (double)INTER * HIDDEN, d = (double)HIDDEN * INTER;
    return (g + u + d) * 3.0 / 1e9;   /* x3 experts per window, /1e9 -> giga */
}

/* ================= item 5: expert-parallel, disjoint thread groups ===== */
#define NGROUPS 3
static const int GROUP_SIZE[NGROUPS] = {3, 3, 2};
static const int GROUP_CPU0[NGROUPS] = {0, 3, 6};

typedef struct {
    int gid, tid;                 /* group id, thread index within group */
    pthread_barrier_t *bar;       /* this group's barrier (GROUP_SIZE[gid] parties) */
    const Expert *e;              /* this group's one expert for the whole run */
    const float *x;
    float *out, *sg, *su;
    int n_windows;
    double *elapsed_ms;           /* [0] written by tid==0 */
} GThread;

static void *gworker(void *arg) {
    GThread *g = (GThread *)arg;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(GROUP_CPU0[g->gid] + g->tid, &set);
    sched_setaffinity(0, sizeof(set), &set);
    const int T = GROUP_SIZE[g->gid];
    const int Og = INTER, Od = HIDDEN, Ig = HIDDEN, Id = INTER;
    const int grb = (Ig + 1) / 2, gng = (Ig + GS - 1) / GS;
    const int drb = (Id + 1) / 2, dng = (Id + GS - 1) / GS;
    /* contiguous block per thread, like gcc/libgomp's default static
     * schedule (ceil(N/T) per thread, last thread gets the remainder). */
    int gchunk = (2 * Og + T - 1) / T;
    int gz0 = g->tid * gchunk, gz1 = gz0 + gchunk; if (gz1 > 2 * Og) gz1 = 2 * Og;
    int schunk = (Og + T - 1) / T;
    int s0 = g->tid * schunk, s1 = s0 + schunk; if (s1 > Og) s1 = Og;
    int dchunk = (Od + T - 1) / T;
    int d0 = g->tid * dchunk, d1 = d0 + dchunk; if (d1 > Od) d1 = Od;

    double t0 = 0;
    if (g->tid == 0) t0 = now_ms();
    pthread_barrier_wait(g->bar);
    for (int w = 0; w < g->n_windows; w++) {
        for (int z = gz0; z < gz1; z++) {
            if (z < Og)
                g->sg[z] = coli_i4_row(g->e->gate.q4 + (int64_t)z * grb, g->e->gate.s + (int64_t)z * gng, g->x, Ig, GS);
            else {
                int o = z - Og;
                g->su[o] = coli_i4_row(g->e->up.q4 + (int64_t)o * grb, g->e->up.s + (int64_t)o * gng, g->x, Ig, GS);
            }
        }
        pthread_barrier_wait(g->bar);
        for (int i = s0; i < s1; i++) {
            float gv = g->sg[i] > LIMIT ? LIMIT : g->sg[i];
            float uv = g->su[i] < -LIMIT ? -LIMIT : (g->su[i] > LIMIT ? LIMIT : g->su[i]);
            g->sg[i] = siluf_(gv) * uv;
        }
        pthread_barrier_wait(g->bar);
        for (int o = d0; o < d1; o++)
            g->out[o] = coli_i4_row(g->e->down.q4 + (int64_t)o * drb, g->e->down.s + (int64_t)o * dng, g->sg, Id, GS);
        pthread_barrier_wait(g->bar);
    }
    if (g->tid == 0) *g->elapsed_ms = now_ms() - t0;
    return NULL;
}

/* Runs NGROUPS experts concurrently, n_windows times each (so total expert-
 * instances = NGROUPS * n_windows, same order as the pool RNG gives us),
 * returns ms for ONE "window" = the slowest group's total / n_windows (the
 * three groups finish together only if perfectly balanced; they are not,
 * since group sizes differ -- the max is what a real 3-expert window would
 * wait for if issued this way). */
static double run_expert_parallel(const Expert *pool, int pool_n, int n_windows,
                                  float *scratch_out, float *scratch_sg, float *scratch_su) {
    pthread_barrier_t bar[NGROUPS];
    for (int gi = 0; gi < NGROUPS; gi++) pthread_barrier_init(&bar[gi], NULL, GROUP_SIZE[gi]);
    pthread_t th[8]; GThread gt[8];
    double elapsed[NGROUPS] = {0, 0, 0};
    static float x[HIDDEN];
    static int xinit = 0;
    if (!xinit) { for (int i = 0; i < HIDDEN; i++) x[i] = rnd01() - 0.5f; xinit = 1; }
    int tn = 0;
    for (int gi = 0; gi < NGROUPS; gi++) {
        int eid = rndu() % pool_n;
        for (int ti = 0; ti < GROUP_SIZE[gi]; ti++) {
            gt[tn].gid = gi; gt[tn].tid = ti; gt[tn].bar = &bar[gi];
            gt[tn].e = &pool[eid]; gt[tn].x = x;
            gt[tn].out = scratch_out + (size_t)gi * HIDDEN;
            gt[tn].sg = scratch_sg + (size_t)gi * INTER;
            gt[tn].su = scratch_su + (size_t)gi * INTER;
            gt[tn].n_windows = n_windows;
            gt[tn].elapsed_ms = &elapsed[gi];
            pthread_create(&th[tn], NULL, gworker, &gt[tn]);
            tn++;
        }
    }
    for (int i = 0; i < tn; i++) pthread_join(th[i], NULL);
    for (int gi = 0; gi < NGROUPS; gi++) pthread_barrier_destroy(&bar[gi]);
    double worst = elapsed[0];
    for (int gi = 1; gi < NGROUPS; gi++) if (elapsed[gi] > worst) worst = elapsed[gi];
    return worst / n_windows;
}

/* ================= item 6: bandwidth ceiling, plain memcpy/sum ========= */
/* Reads EVERY 8-byte word of a buffer once (no stride-skipping -- a sampled
 * touch measures TLB/page-walk cost on a huge stride, not bandwidth, and
 * would understate the true byte count by >100x). 8-way OMP split.
 *
 * F8 arbitration: a single OpenMP `reduction(+:total)` accumulator gives
 * each thread its own serial add chain over its slice -- a 1-cycle-latency
 * integer add is not the FMA case (5-cycle latency on Zen2), but to remove
 * ANY doubt that this row is itself measuring a latency-bound reduction
 * rather than the memory system, it now unrolls 4 INDEPENDENT partial sums
 * per thread (no accumulator sees two dependent adds back to back inside a
 * cache line's worth of words) before the reduction combines them. */
static uint64_t touch_all(const uint8_t *p, size_t n) {
    size_t nu64 = n / 8;
    const uint64_t *pu = (const uint64_t *)p;
    long nu4 = (long)(nu64 / 4);
    uint64_t total = 0;
    #pragma omp parallel reduction(+:total)
    {
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        #pragma omp for schedule(static) nowait
        for (long b = 0; b < nu4; b++) {
            long i = b * 4;
            s0 += pu[i]; s1 += pu[i + 1]; s2 += pu[i + 2]; s3 += pu[i + 3];
        }
        total += s0 + s1 + s2 + s3;
    }
    for (size_t i = (size_t)nu4 * 4; i < nu64; i++) total += pu[i];
    for (size_t i = nu64 * 8; i < n; i++) total += p[i];
    return total;
}
static double run_bandwidth_ceiling(const Expert *pool, int pool_n, int n_windows, size_t bytes_per_expert) {
    /* 8-thread sum over the same random 14 MB blocks the kernel touches
     * (q4 + scale of one expert's gate/up/down), reading every byte once,
     * same as any int4 kernel that dequantises every weight. */
    volatile uint64_t sink = 0;
    double t0 = now_ms();
    for (int w = 0; w < n_windows; w++) {
        int e0, e1, e2; pick3(pool_n, &e0, &e1, &e2);
        int ids[3] = {e0, e1, e2};
        for (int k = 0; k < 3; k++) {
            const Expert *e = &pool[ids[k]];
            const uint8_t *bufs[6] = { e->gate.q4, (const uint8_t *)e->gate.s,
                                       e->up.q4,   (const uint8_t *)e->up.s,
                                       e->down.q4, (const uint8_t *)e->down.s };
            size_t lens[6] = { (size_t)e->gate.O * ((e->gate.I + 1) / 2),
                               (size_t)e->gate.O * ((e->gate.I + GS - 1) / GS) * sizeof(float),
                               (size_t)e->up.O * ((e->up.I + 1) / 2),
                               (size_t)e->up.O * ((e->up.I + GS - 1) / GS) * sizeof(float),
                               (size_t)e->down.O * ((e->down.I + 1) / 2),
                               (size_t)e->down.O * ((e->down.I + GS - 1) / GS) * sizeof(float) };
            for (int b = 0; b < 6; b++) sink += touch_all(bufs[b], lens[b]);
        }
    }
    double ms = now_ms() - t0;
    (void)sink;
    double gb = (double)bytes_per_expert * 3.0 * n_windows / 1e9;
    printf("ROW kind=%-28s median_ms=%.4f p90_ms=%.4f gmac_s=%.2f gbytes_s=%.2f\n",
           "bandwidth_ceiling_touch", ms / n_windows, ms / n_windows, 0.0, gb / (ms / 1000.0));
    return ms / n_windows;
}

int main(int argc, char **argv) {
    int n_windows = argc > 1 ? atoi(argv[1]) : 3000;
    int pool_n = argc > 2 ? atoi(argv[2]) : 350;
    if (argc > 3) S_RND = (uint32_t)strtoul(argv[3], NULL, 10);
    if (n_windows < 100) n_windows = 100;

    size_t ebytes = expert_bytes();
    double pool_gb = (double)ebytes * pool_n / 1e9;
    printf("INFO expert_bytes=%zu pool_experts=%d pool_gb=%.2f n_windows=%d\n", ebytes, pool_n, pool_gb, n_windows);
    if (pool_gb < 4.0) { fprintf(stderr, "FATAL: pool %.2f GB < 4 GB floor, increase pool_experts\n", pool_gb); return 1; }

#ifdef _OPENMP
    printf("INFO omp_max_threads=%d\n", omp_get_max_threads());
#else
    printf("WARN built without OpenMP -- baseline/thread-scaling numbers are not comparable to the engine\n");
#endif
#ifdef __AVX2__
    printf("INFO AVX2 build\n");
#else
    printf("WARN no AVX2 -- coli_i4_row falls back to scalar, not representative of the engine on this box\n");
#endif

    printf("INFO allocating pool...\n");
    Expert *pool = calloc(pool_n, sizeof(Expert));
    if (!pool) { perror("calloc pool"); return 1; }
    for (int i = 0; i < pool_n; i++) expert_alloc(&pool[i]);
    static float x[HIDDEN];
    for (int i = 0; i < HIDDEN; i++) x[i] = rnd01() - 0.5f;

    /* ---- bit-exactness check: mlp3_like vs an independently-built rebuild
     * of the same three stages (matmul_i4_grouped-based), one expert. ---- */
    {
        float out_a[HIDDEN], out_b[HIDDEN], sg[INTER], su[INTER];
        mlp3_like(out_a, x, &pool[0], sg, su);
        mlp3_check(out_b, x, &pool[0], sg, su);
        double maxabs = 0, sumabs = 0;
        for (int i = 0; i < HIDDEN; i++) { double d = fabs(out_a[i] - out_b[i]); if (d > maxabs) maxabs = d; sumabs += fabs(out_a[i]); }
        printf("INFO bit-exactness check mlp3_like vs matmul_i4_grouped rebuild: max_abs_diff=%.3e (sum_abs=%.3e)\n", maxabs, sumabs);
    }

    /* ================= item 1: baseline, engine-shaped ================= */
    {
        double *ms = malloc(sizeof(double) * n_windows);
        float out[HIDDEN], sg[INTER], su[INTER];
        for (int w = 0; w < n_windows; w++) {
            int a, b, c; pick3(pool_n, &a, &b, &c);
            double t0 = now_ms();
            mlp3_like(out, x, &pool[a], sg, su);
            mlp3_like(out, x, &pool[b], sg, su);
            mlp3_like(out, x, &pool[c], sg, su);
            ms[w] = (now_ms() - t0) / 3.0;   /* ms PER EXPERT */
        }
        double gmacs_per_expert = ((double)INTER * HIDDEN * 2 + (double)HIDDEN * INTER) / 1e9;
        double gbytes_per_expert = (double)ebytes / 1e9;
        report("baseline_engine_shaped", ms, n_windows, gmacs_per_expert, gbytes_per_expert);
        free(ms);
    }

    /* ================= item 2: per-matmul split ========================= */
    {
        double *ms_g = malloc(sizeof(double) * n_windows);
        double *ms_u = malloc(sizeof(double) * n_windows);
        double *ms_d = malloc(sizeof(double) * n_windows);
        float sg[INTER], su[INTER], out[HIDDEN];
        for (int w = 0; w < n_windows; w++) {
            int eid = rndu() % pool_n;
            const Expert *e = &pool[eid];
            double t0 = now_ms();
            matmul_i4_grouped(sg, x, e->gate.q4, e->gate.s, 1, e->gate.I, e->gate.O, e->gate.gs);
            ms_g[w] = now_ms() - t0;
            t0 = now_ms();
            matmul_i4_grouped(su, x, e->up.q4, e->up.s, 1, e->up.I, e->up.O, e->up.gs);
            ms_u[w] = now_ms() - t0;
            swiglu_clamped(sg, su, e->gate.O, LIMIT);
            t0 = now_ms();
            matmul_i4_grouped(out, sg, e->down.q4, e->down.s, 1, e->down.I, e->down.O, e->down.gs);
            ms_d[w] = now_ms() - t0;
        }
        double gmac_gu = (double)INTER * HIDDEN / 1e9, gbytes_gu = ((double)INTER * ((HIDDEN + 1) / 2) + (double)INTER * (HIDDEN / GS) * 4) / 1e9;
        double gmac_d = (double)HIDDEN * INTER / 1e9, gbytes_d = ((double)HIDDEN * ((INTER + 1) / 2) + (double)HIDDEN * (INTER / GS) * 4) / 1e9;
        report("matmul_gate_alone", ms_g, n_windows, gmac_gu, gbytes_gu);
        report("matmul_up_alone", ms_u, n_windows, gmac_gu, gbytes_gu);
        report("matmul_down_alone", ms_d, n_windows, gmac_d, gbytes_d);
        free(ms_g); free(ms_u); free(ms_d);
    }

    /* ================= item 3: thread scaling of the baseline =========== */
#ifdef _OPENMP
    {
        int tcounts[4] = {1, 2, 4, 8};
        int nw3 = n_windows / 2; if (nw3 < 200) nw3 = 200;
        for (int ti = 0; ti < 4; ti++) {
            int T = tcounts[ti];
            omp_set_num_threads(T);
            double *ms = malloc(sizeof(double) * nw3);
            float out[HIDDEN], sg[INTER], su[INTER];
            for (int w = 0; w < nw3; w++) {
                int a, b, c; pick3(pool_n, &a, &b, &c);
                double t0 = now_ms();
                mlp3_like(out, x, &pool[a], sg, su);
                mlp3_like(out, x, &pool[b], sg, su);
                mlp3_like(out, x, &pool[c], sg, su);
                ms[w] = (now_ms() - t0) / 3.0;
            }
            char kind[64]; snprintf(kind, sizeof(kind), "thread_scaling_T%d", T);
            double gmacs_per_expert = ((double)INTER * HIDDEN * 2 + (double)HIDDEN * INTER) / 1e9;
            double gbytes_per_expert = (double)ebytes / 1e9;
            report(kind, ms, nw3, gmacs_per_expert, gbytes_per_expert);
            free(ms);
        }
        omp_set_num_threads(8);
    }
#else
    printf("WARN no OpenMP -- thread-scaling item skipped\n");
#endif

    /* ================= item 4: fork/join cost ============================ */
    {
        /* bit-exactness: fused window vs 3 serial mlp3_like calls, one window */
        int a, b, c; pick3(pool_n, &a, &b, &c);
        Expert e3[3] = {pool[a], pool[b], pool[c]};
        float out_serial[3][HIDDEN], sg1[INTER], su1[INTER];
        mlp3_like(out_serial[0], x, &e3[0], sg1, su1);
        mlp3_like(out_serial[1], x, &e3[1], sg1, su1);
        mlp3_like(out_serial[2], x, &e3[2], sg1, su1);
        float out_fused[3 * HIDDEN], sg3[3 * INTER], su3[3 * INTER];
        mlp3_window_fused(out_fused, x, e3, sg3, su3);
        double maxabs = 0;
        for (int w = 0; w < 3; w++)
            for (int i = 0; i < HIDDEN; i++) {
                double d = fabs((double)out_serial[w][i] - (double)out_fused[w * HIDDEN + i]);
                if (d > maxabs) maxabs = d;
            }
        printf("INFO bit-exactness check fused-window vs 3x serial mlp3_like: max_abs_diff=%.3e (expect 0.0, same coli_i4_row calls same order)\n", maxabs);

        double *ms_3team = malloc(sizeof(double) * n_windows);
        double *ms_1team = malloc(sizeof(double) * n_windows);
        float out[HIDDEN], sg[INTER], su[INTER];
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            double t0 = now_ms();
            mlp3_like(out, x, &pool[p], sg, su);
            mlp3_like(out, x, &pool[q], sg, su);
            mlp3_like(out, x, &pool[r], sg, su);
            ms_3team[w] = now_ms() - t0;
        }
        float outF[3 * HIDDEN], sgF[3 * INTER], suF[3 * INTER];
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            Expert ew[3] = {pool[p], pool[q], pool[r]};
            double t0 = now_ms();
            mlp3_window_fused(outF, x, ew, sgF, suF);
            ms_1team[w] = now_ms() - t0;
        }
        double gmacs_win = window_gmacs();
        double gbytes_win = (double)ebytes * 3.0 / 1e9;
        report("forkjoin_3_teams_per_window", ms_3team, n_windows, gmacs_win, gbytes_win);
        report("forkjoin_1_team_per_window", ms_1team, n_windows, gmacs_win, gbytes_win);
        free(ms_3team); free(ms_1team);
    }

    /* ================= item 7 (F8 arbitration): reassociating kernels ==== */
    {
        int a, b, c; pick3(pool_n, &a, &b, &c);
        Expert e3[3] = {pool[a], pool[b], pool[c]};
        float sg1[INTER], su1[INTER];
        float out_base[HIDDEN], out_f4[HIDDEN], out_f8[HIDDEN];
        mlp3_kern(out_base, x, &e3[0], sg1, su1, coli_i4_row);
        mlp3_kern(out_f4, x, &e3[0], sg1, su1, coli_i4_row_f4);
        mlp3_kern(out_f8, x, &e3[0], sg1, su1, coli_i4_row_f8);
        double maxrel_f4 = 0, maxrel_f8 = 0;
        for (int i = 0; i < HIDDEN; i++) {
            double denom = fabs((double)out_base[i]) > 1e-12 ? fabs((double)out_base[i]) : 1e-12;
            double r4 = fabs((double)out_f4[i] - (double)out_base[i]) / denom;
            double r8 = fabs((double)out_f8[i] - (double)out_base[i]) / denom;
            if (r4 > maxrel_f4) maxrel_f4 = r4;
            if (r8 > maxrel_f8) maxrel_f8 = r8;
        }
        printf("INFO reassociation check (one expert, one row kernel each): coli_i4_row_f4 max_rel_diff=%.3e, coli_i4_row_f8 max_rel_diff=%.3e (both expect ~1e-7, G14's own figure for f4)\n",
               maxrel_f4, maxrel_f8);

        double *ms_f4 = malloc(sizeof(double) * n_windows);
        double *ms_f4_fused = malloc(sizeof(double) * n_windows);
        double *ms_f8 = malloc(sizeof(double) * n_windows);
        double *ms_f8_fused = malloc(sizeof(double) * n_windows);
        float out[HIDDEN], sg[INTER], su[INTER];
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            double t0 = now_ms();
            mlp3_kern(out, x, &pool[p], sg, su, coli_i4_row_f4);
            mlp3_kern(out, x, &pool[q], sg, su, coli_i4_row_f4);
            mlp3_kern(out, x, &pool[r], sg, su, coli_i4_row_f4);
            ms_f4[w] = now_ms() - t0;
        }
        float outF[3 * HIDDEN], sgF[3 * INTER], suF[3 * INTER];
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            Expert ew[3] = {pool[p], pool[q], pool[r]};
            double t0 = now_ms();
            mlp3_window_fused_kern(outF, x, ew, sgF, suF, coli_i4_row_f4);
            ms_f4_fused[w] = now_ms() - t0;
        }
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            double t0 = now_ms();
            mlp3_kern(out, x, &pool[p], sg, su, coli_i4_row_f8);
            mlp3_kern(out, x, &pool[q], sg, su, coli_i4_row_f8);
            mlp3_kern(out, x, &pool[r], sg, su, coli_i4_row_f8);
            ms_f8[w] = now_ms() - t0;
        }
        for (int w = 0; w < n_windows; w++) {
            int p, q, r; pick3(pool_n, &p, &q, &r);
            Expert ew[3] = {pool[p], pool[q], pool[r]};
            double t0 = now_ms();
            mlp3_window_fused_kern(outF, x, ew, sgF, suF, coli_i4_row_f8);
            ms_f8_fused[w] = now_ms() - t0;
        }
        double gmacs_win = window_gmacs();
        double gbytes_win = (double)ebytes * 3.0 / 1e9;
        report("f4_3_teams_per_window", ms_f4, n_windows, gmacs_win, gbytes_win);
        report("f4_1_team_per_window", ms_f4_fused, n_windows, gmacs_win, gbytes_win);
        report("f8_3_teams_per_window", ms_f8, n_windows, gmacs_win, gbytes_win);
        report("f8_1_team_per_window", ms_f8_fused, n_windows, gmacs_win, gbytes_win);
        free(ms_f4); free(ms_f4_fused); free(ms_f8); free(ms_f8_fused);
    }

    /* ================= item 5: expert-parallel, disjoint groups ========= */
    {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpu < 8) {
            printf("WARN only %ld CPUs visible -- skipping expert-parallel item (needs 8 distinct physical cores, cpu 0-7 per record's staging_fill_probe topology note)\n", ncpu);
        } else {
            float scratch_out[NGROUPS * HIDDEN], scratch_sg[NGROUPS * INTER], scratch_su[NGROUPS * INTER];
            int nw5 = n_windows;
            /* warm-up window (thread creation cost excluded from steady state
             * would need a persistent pool; this measures create+run+join for
             * the whole nw5 loop once, i.e. thread lifetime spans all windows) */
            double ms_per_window = run_expert_parallel(pool, pool_n, nw5, scratch_out, scratch_sg, scratch_su);
            double one[1] = {ms_per_window};
            double gmacs_win = window_gmacs(), gbytes_win = (double)ebytes * 3.0 / 1e9;
            report("expert_parallel_groups_3_3_2", one, 1, gmacs_win, gbytes_win);
        }
    }

    /* ================= item 6: bandwidth ceiling ========================= */
    run_bandwidth_ceiling(pool, pool_n, n_windows, ebytes / 3);

    printf("INFO F1-STEP0 reference: 0.560 ms/expert, 1.747 ms/window (3.12 experts) on the live engine\n");
    return 0;
}
