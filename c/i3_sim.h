#ifndef COLI_I3_SIM_H
#define COLI_I3_SIM_H
/* i3_sim.h -- G15 / roadmap item 4h: int3-g64 SIMULATED on int4-g64 weights.
 *
 * Probe code, not a shipping kernel. Included by c/glm53.c (behind the
 * GLM53_I3_SIM knob, off by default) and by tools/hot-expert/rome_i3sim.c,
 * which is the oracle: it checks this simulation against quant.h's own
 * pack_int3_g64 + matmul_i3, i.e. against what the real fmt=5 pipeline would
 * compute on real converter output. Nothing else includes it, so no other
 * engine is perturbed.
 *
 * Requires quant.h (I3_GROUP, hsum256, coli_i4_row, coli_i4_rows4).
 *
 * VERDICT (record §G15, 2026-09-11): GLM-5.3 does NOT survive int3 experts.
 * Against an identically-placed int4 control, 13 of 42 short-prompt and 16 of
 * 1232 long-prompt teacher_forcing predictions change and last_logits cosine is
 * 0.9726 / 0.8778, an order of magnitude past the line §G14 rejected at. This
 * file is kept so the measurement can be reproduced and so the transform stays
 * checkable, not because the direction is still open.
 */
#include <math.h>
#include <stdint.h>

/* THE SIMULATION
 *
 * §G14 ended with "the constraint is bytes, not ALU": the engine streams
 * 985 MB/token of cold expert weights, so the lever is fewer bytes. fmt=5
 * (int3-g64, 24 B per 64-group) takes an expert slot 13.5 -> 10.5 MiB, which
 * both cuts the bytes a CPU expert streams by 22% and fits ~28% more experts
 * in the same VRAM budget. Before any of that is worth building, ONE question
 * has to be answered: does GLM-5.3 survive 3-bit expert weights at all? int3
 * is a real precision drop across EVERY expert, including the ~79% the GPU
 * tier serves -- categorically bigger than anything in §G14, which only ever
 * moved summation order or quantised activations.
 *
 * So this simulates int3 on the int4 weights already on disk. No converter,
 * no shader, no fmt=5 kernel. Per 64-input subgroup of an int4-g64 row:
 *
 *   v      = nibble - 8                        (the stored int4 level)
 *   w      = v * sc                            (dequantised, sc = int4 group scale)
 *   amax   = sc * max|v|                       (exactly, since sc > 0)
 *   s3     = max(amax / 3, 1e-8)               <- FRESH per-group scale
 *   q      = clip(rint(w / s3), -4, 3)         = clip(rint(3v/max|v|), -4, 3)
 *   w'     = q * s3
 *
 * which is quant_int3_g64() in c/tools/convert_fp8_to_int4.py, element for
 * element (symmetric absmax, qmax=3, clamp [-4,3]), applied to the same
 * 64-wide groups fmt=5 uses. The accumulation shape mirrors matmul_i3: the
 * group partial is a float dot against the integer levels, and the group scale
 * multiplies the partial once, added to the row accumulator in scalar order.
 *
 * TWO KNOWN DEVIATIONS FROM A REAL CONVERSION, both stated rather than hidden:
 *  - This is a DOUBLE quantisation (fp8 -> int4 -> int3), where the converter
 *    would go fp8 -> int3 once. The extra step is strictly pessimistic and its
 *    size is computable: the int4 rounding error has variance s4^2/12 with
 *    s4 = amax/7, on top of int3's s3^2/12 with s3 = amax/3, i.e. (3/7)^2 =
 *    18% more error variance, ~9% more RMS. If the model survives THIS it
 *    survives a real conversion.
 *  - w/s3 is evaluated as v * (3/m) rather than numpy's true division of two
 *    floats. Same value mathematically; the two can differ by an ulp and so
 *    disagree on exact rounding ties. Ties are measure-zero here and the
 *    disagreement, where it happens, is one level -- far below the thing being
 *    measured.
 *
 * GLM53_I3_SIM=1 turns it on. It is paired with GLM53_EXPERTS_CPU=1 (below),
 * which forces every routed expert onto the CPU path so the GPU tier -- which
 * serves ~79% of activations from unmodified int4 -- cannot mask the effect.
 * Both are OFF by default and both are probe knobs, not shipping ones. The
 * shared expert and the dense layers are deliberately untouched: item 4h
 * converts the ROUTED expert slots, nothing else. */
/* max|v| over a run of packed int4 nibbles, v = nibble - 8. Equivalently
 * max(8 - min_nibble, max_nibble - 8), so one unsigned min/max pass over the
 * bytes answers it. The shift-reduce leaves garbage in the high lanes (srli
 * shifts zeros in) and that is fine: only lane 0 is read, and every lane that
 * feeds lane 0 is a real value. */
static inline int i4_sub_absmax(const uint8_t *w, int base, int n) {
    int mn = 15, mx = 0, i = base;
#ifdef __AVX2__
    if (!(base & 1) && n >= 32) {
        const __m128i m4 = _mm_set1_epi8(0x0F);
        __m128i vmn = _mm_set1_epi8(0x0F), vmx = _mm_setzero_si128();
        for (; i + 32 <= base + n; i += 32) {
            __m128i by = _mm_loadu_si128((const __m128i *)(w + (i >> 1)));
            __m128i lo = _mm_and_si128(by, m4);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
            vmn = _mm_min_epu8(vmn, _mm_min_epu8(lo, hi));
            vmx = _mm_max_epu8(vmx, _mm_max_epu8(lo, hi));
        }
        vmn = _mm_min_epu8(vmn, _mm_srli_si128(vmn, 8));
        vmx = _mm_max_epu8(vmx, _mm_srli_si128(vmx, 8));
        vmn = _mm_min_epu8(vmn, _mm_srli_si128(vmn, 4));
        vmn = _mm_min_epu8(vmn, _mm_srli_si128(vmn, 2));
        vmn = _mm_min_epu8(vmn, _mm_srli_si128(vmn, 1));
        vmx = _mm_max_epu8(vmx, _mm_srli_si128(vmx, 4));
        vmx = _mm_max_epu8(vmx, _mm_srli_si128(vmx, 2));
        vmx = _mm_max_epu8(vmx, _mm_srli_si128(vmx, 1));
        mn = _mm_cvtsi128_si32(vmn) & 0xFF;
        mx = _mm_cvtsi128_si32(vmx) & 0xFF;
    }
#endif
    for (; i < base + n; i++) {
        const int nb = (w[i >> 1] >> ((i & 1) * 4)) & 0xF;
        if (nb < mn) mn = nb;
        if (nb > mx) mx = nb;
    }
    const int a = 8 - mn, b = mx - 8;
    return a > b ? a : b;
}

/* One output element: coli_i4_row's loop with the int3 re-quantisation folded
 * into the decode. The int4 group is walked in 64-wide subgroups because that
 * is fmt=5's group, whatever gs the int4 tensor uses. */
static inline float coli_i4_row_sim3(const uint8_t *w, const float *scl,
                                     const float *xs, int I, int gs) {
    float a = 0;
    for (int g = 0; g * gs < I; g++) {
        int gbase = g * gs, glen = gs;
        if (gbase + glen > I) glen = I - gbase;
        const float sc = scl[g];
        for (int sb = 0; sb < glen; sb += I3_GROUP) {
            const int base = gbase + sb;
            int n = glen - sb; if (n > I3_GROUP) n = I3_GROUP;
            const int m = i4_sub_absmax(w, base, n);
            if (m <= 0) continue;                    /* group is all zeros: q == 0 */
            float s3 = sc * (float)m / 3.0f;
            if (s3 < 1e-8f) s3 = 1e-8f;
            const float k = 3.0f / (float)m;
            float p = 0;
            int i = base;
#ifdef __AVX2__
            {
                const __m128i m4 = _mm_set1_epi8(0x0F);
                const __m256i b8 = _mm256_set1_epi32(8);
                const __m256 kk = _mm256_set1_ps(k);
                const __m256 c4 = _mm256_set1_ps(-4.0f), c3 = _mm256_set1_ps(3.0f);
                __m256 acc = _mm256_setzero_ps();
                for (; i + 16 <= base + n; i += 16) {
                    __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
                    __m128i lo = _mm_and_si128(by, m4);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);
                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                                    _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), b8));
                    w0 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(w0, kk),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), c4), c3);
                    w1 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(w1, kk),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), c4), c3);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),     w0, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8), w1, acc);
                }
                p = hsum256(acc);
            }
#endif
            for (; i < base + n; i++) {
                const int nb = (w[i >> 1] >> ((i & 1) * 4)) & 0xF;
                float q = nearbyintf((float)(nb - 8) * k);
                if (q < -4.0f) q = -4.0f; else if (q > 3.0f) q = 3.0f;
                p += xs[i] * q;
            }
            a += p * s3;
        }
    }
    return a;
}

/* Four activation rows against one weight row: the sim3 decode is paid once
 * per 16 weights and applied to four accumulators, exactly as coli_i4_rows4
 * does for the plain int4 decode. Needed for the prefill (rows) path, or the
 * teacher_forcing oracle over 1232 positions would re-decode every weight per
 * row and take hours instead of minutes. */
static inline void coli_i4_rows4_sim3(const uint8_t *w, const float *scl, const float *xs,
                                      int64_t ldx, int I, int gs, float out[4]) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const float *x0 = xs, *x1 = xs + ldx, *x2 = xs + 2 * ldx, *x3 = xs + 3 * ldx;
    for (int g = 0; g * gs < I; g++) {
        int gbase = g * gs, glen = gs;
        if (gbase + glen > I) glen = I - gbase;
        const float sc = scl[g];
        for (int sb = 0; sb < glen; sb += I3_GROUP) {
            const int base = gbase + sb;
            int n = glen - sb; if (n > I3_GROUP) n = I3_GROUP;
            const int m = i4_sub_absmax(w, base, n);
            if (m <= 0) continue;
            float s3 = sc * (float)m / 3.0f;
            if (s3 < 1e-8f) s3 = 1e-8f;
            const float k = 3.0f / (float)m;
            float p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            int i = base;
#ifdef __AVX2__
            {
                const __m128i m4 = _mm_set1_epi8(0x0F);
                const __m256i b8 = _mm256_set1_epi32(8);
                const __m256 kk = _mm256_set1_ps(k);
                const __m256 c4 = _mm256_set1_ps(-4.0f), c3 = _mm256_set1_ps(3.0f);
                __m256 q0 = _mm256_setzero_ps(), q1 = _mm256_setzero_ps();
                __m256 q2 = _mm256_setzero_ps(), q3 = _mm256_setzero_ps();
                for (; i + 16 <= base + n; i += 16) {
                    __m128i by = _mm_loadl_epi64((const __m128i *)(w + (i >> 1)));
                    __m128i lo = _mm_and_si128(by, m4);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);
                    __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), b8));
                    __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                                    _mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), b8));
                    w0 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(w0, kk),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), c4), c3);
                    w1 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(w1, kk),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), c4), c3);
                    q0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i),     w0, q0);
                    q0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i + 8), w1, q0);
                    q1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i),     w0, q1);
                    q1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + 8), w1, q1);
                    q2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + i),     w0, q2);
                    q2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + i + 8), w1, q2);
                    q3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + i),     w0, q3);
                    q3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + i + 8), w1, q3);
                }
                p0 = hsum256(q0); p1 = hsum256(q1); p2 = hsum256(q2); p3 = hsum256(q3);
            }
#endif
            for (; i < base + n; i++) {
                const int nb = (w[i >> 1] >> ((i & 1) * 4)) & 0xF;
                float q = nearbyintf((float)(nb - 8) * k);
                if (q < -4.0f) q = -4.0f; else if (q > 3.0f) q = 3.0f;
                p0 += x0[i] * q; p1 += x1[i] * q; p2 += x2[i] * q; p3 += x3[i] * q;
            }
            a0 += p0 * s3; a1 += p1 * s3; a2 += p2 * s3; a3 += p3 * s3;
        }
    }
    out[0] = a0; out[1] = a1; out[2] = a2; out[3] = a3;
}

/* The roadmap's named probe entry point: matmul_i4_grouped's signature and
 * loop shape, with the int3 re-quantisation in the row kernel. */
static void matmul_i4_sim3(float *y, const float *x, const uint8_t *q4, const float *scale,
                           int S, int I, int O, int gs) {
    const int64_t rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int o = 0; o < O; o++)
        for (int s = 0; s < S; s++)
            y[(int64_t)s * O + o] = coli_i4_row_sim3(q4 + (int64_t)o * rb, scale + (int64_t)o * ng,
                                                     x + (int64_t)s * I, I, gs);
}

/* Selectors for the rows (prefill) path, so the probe does not need a second
 * copy of mlp3_cpu_rows. sim3 is a hoisted local there, 0 in every shipped
 * configuration. */
static inline float i4_row_sel(int sim3, const uint8_t *w, const float *scl,
                               const float *xs, int I, int gs) {
    return sim3 ? coli_i4_row_sim3(w, scl, xs, I, gs) : coli_i4_row(w, scl, xs, I, gs);
}
static inline void i4_rows4_sel(int sim3, const uint8_t *w, const float *scl, const float *xs,
                                int64_t ldx, int I, int gs, float out[4]) {
    if (sim3) coli_i4_rows4_sim3(w, scl, xs, ldx, I, gs, out);
    else      coli_i4_rows4(w, scl, xs, ldx, I, gs, out);
}


#endif /* COLI_I3_SIM_H */
