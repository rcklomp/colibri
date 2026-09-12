#ifndef COLI_Q38_SIM_H
#define COLI_Q38_SIM_H
/* q38_sim.h -- roadmap item QP: the three weight-format SIMULATIONS for
 * Qwen3.8-Flash-Next, plus the two placement/VRAM instruments they need.
 * EVERYTHING HERE IS OFF BY DEFAULT and nothing in it is on any shipped path;
 * with the knobs unset the engine is bit-identical to the binary before this
 * file existed (proved per item, see record §QP).
 *
 * WHY A SIMULATION AND NOT A CONVERTER (§G15 is the precedent): QP's whole
 * point is to answer "does this format survive" BEFORE anyone writes the
 * converter, the shader and the kernel it would need. §G15 did exactly this for
 * GLM-5.3's int3 experts and killed the item at a cost of one day instead of
 * three-plus. So each format is computed on the fly from the weights already on
 * disk, and no second checkpoint is written.
 *
 *   Q38_I4_SIM=1        every routed expert's FP8 weight is computed as if it
 *                       had been converted to int4-g64 (fmt=4). Pair it with
 *                       Q38_EXPERTS_CPU=1 -- otherwise the ~80 % of routed
 *                       experts the GPU tier serves from UNMODIFIED fp8 would
 *                       mask the divergence (this is the mistake §G15's item
 *                       text forbids by name).
 *   Q38_EXPERTS_CPU=1   the GPU tier declines every routed expert, so the CPU
 *                       path serves all of them. Placement only: no arithmetic
 *                       changes. It is the CONTROL for Q38_I4_SIM, and it must
 *                       be on in BOTH arms of the comparison or placement
 *                       itself is a confound.
 *   Q38_I8_DENSE=1|2    the resident BF16 dense projections as int8, 1 = one
 *                       scale per output row, 2 = one per 64 inputs.
 *   Q38_I8_HEAD=1|2     the LM head as int8, same encoding. Reported SEPARATELY
 *                       from the projections: QP(c) does not assume the two
 *                       move together.
 *   Q38_TF=1            the teacher-forcing oracle (see qwen38.c).
 *   Q38_VK_BALLAST_GB=N N GB of idle VRAM on dev0 before the expert preload.
 *
 * THE DOUBLE-QUANTISATION QUESTION, WHICH §G15 HAD AND THIS ITEM DOES NOT.
 * §G15 simulated int3 on weights that were ALREADY int4, so its probe was
 * pessimistic by a measured 1.08x and it had to price that. Here every
 * simulation's source is the format the converter's source would be:
 *   (a) the experts are FP8 e4m3 on disk and a converter would read FP8, so
 *       fp8 -> int4-g64 IS the conversion, not an extra step;
 *   (b)/(c) the dense tensors and the head are BF16 on disk and resident as
 *       BF16, so bf16 -> int8 IS the conversion.
 * There is therefore NO pessimism to discount in either direction, and a
 * verdict here is the verdict for the real thing. tools/hot-expert/rome_q4sim.c
 * proves the transforms are the target formats before any model is run.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "quant.h"

/* st.h's bf16_to_f32, restated here ONLY so this header can be included by the
 * standalone oracle harness without dragging in st.h's shard machinery.
 * rome_q4sim.c checks it against st.h's own function exhaustively over all
 * 65 536 bf16 bit patterns, so "restated" does not mean "assumed". */
static inline float q38_bf16_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}

/* int4-g64: the group size fmt=4 ships with for GLM's experts (vk_tile_ok4,
 * matmul_i4_grouped) and the one Q8 would use. FP8_BLOCK is 128 and a multiple
 * of it, which is what makes q38_i4sim_row's single scale lookup per group
 * correct -- a group of 64 that starts at a multiple of 64 never straddles two
 * fp8 128x128 blocks. Checked at run time in q38_sim_init_knobs. */
#define Q38_I4_GS 64
/* Widest input dimension any simulated weight has: Qwen3.8's is 2560 (hidden).
 * The row scratch lives on the stack inside an omp region, so this bounds it. */
#define Q38_SIM_MAXI 4096

static int   g_q38_i4_sim      = 0;
static int   g_q38_experts_cpu = 0;
static int   g_q38_i8_dense    = 0;
static int   g_q38_i8_head     = 0;
static double g_q38_vk_ballast_gb = 0.0;
/* The teacher_forcing oracle's state. Declared here with the other knobs
 * because model_init_range resolves it and `step` uses it, and those sit on
 * opposite sides of qwen38_core.h. */
static int g_q38_tf = 0, g_q38_tf_done = 0;
static const char *g_q38_tf_dump = NULL;
/* One NaN fp8 byte (0x7F/0xFF) would make a group's absmax and every level in
 * it undefined. quant.h's fmt=8 NaN POLICY is "propagate and let the sampler's
 * existing net catch it", which an integer format cannot do, so the simulation
 * maps a NaN weight to level 0 and COUNTS it -- a silent substitution inside an
 * accuracy probe is exactly the kind of thing that makes a verdict worthless. */
static uint64_t g_q38_sim_nan = 0;

static int q38_sim_env_int(const char *name) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : 0;
}

/* Resolve every knob ONCE, on the main thread, before any omp region can run:
 * a lazily-initialised `static int cached` inside a kernel called from
 * `#pragma omp parallel for` is a data race (quant.h's own E4M3_LUT comment
 * makes the same point about lazy tables). */
static void q38_sim_init_knobs(void) {
    g_q38_i4_sim      = q38_sim_env_int("Q38_I4_SIM");
    g_q38_experts_cpu = q38_sim_env_int("Q38_EXPERTS_CPU");
    g_q38_i8_dense    = q38_sim_env_int("Q38_I8_DENSE");
    g_q38_i8_head     = q38_sim_env_int("Q38_I8_HEAD");
    g_q38_tf          = q38_sim_env_int("Q38_TF");
    g_q38_tf_dump     = getenv("Q38_TF_DUMP");
    {
        const char *b = getenv("Q38_VK_BALLAST_GB");
        g_q38_vk_ballast_gb = (b && *b) ? atof(b) : 0.0;
    }
    if (FP8_BLOCK % Q38_I4_GS) {
        fprintf(stderr, "q38_sim: FP8_BLOCK %d is not a multiple of the int4 group %d\n",
                FP8_BLOCK, Q38_I4_GS);
        exit(1);
    }
    if (g_q38_i4_sim && !g_q38_experts_cpu)
        fprintf(stderr, "[q38sim] WARNING: Q38_I4_SIM=1 without Q38_EXPERTS_CPU=1 -- the GPU "
                        "tier still serves most routed experts from unmodified fp8 and will "
                        "mask the divergence (record §G15)\n");
    if (g_q38_i4_sim || g_q38_experts_cpu || g_q38_i8_dense || g_q38_i8_head || g_q38_vk_ballast_gb > 0)
        fprintf(stderr, "[q38sim] QP probe knobs: I4_SIM=%d EXPERTS_CPU=%d I8_DENSE=%d "
                        "I8_HEAD=%d BALLAST_GB=%.2f\n",
                g_q38_i4_sim, g_q38_experts_cpu, g_q38_i8_dense, g_q38_i8_head,
                g_q38_vk_ballast_gb);
}

/* ==================== (a) experts: fp8 -> int4-g64 ======================== */

/* One output row of a block-scaled FP8 weight, re-expressed as int4-g64.
 * `lev[i]` receives the int4 LEVEL (an integer in [-8,7]) as a float and
 * `gsc[g]` the group scale.
 *
 * The arithmetic is quantize_i4_grouped()'s in glm53.c -- itself pinned to
 * quant_int4_grouped() in c/tools/convert_fp8_to_int4.py, the authoritative
 * converter-side math -- applied to the DEQUANTISED fp8 row. The dequant is
 * quant.h's fmt=8 rule, w[o,i] = e4m3_decode(byte) * scale[o/128][i/128]
 * (MULTIPLY). Both halves are checked against their originals rather than
 * restated: rome_q4sim.c compares the levels this produces, byte for byte,
 * against the converter's own quant_int4_grouped(gs=64) run through numpy on
 * the same dequantised floats.
 *
 * The group's fp8 block scale is looked up ONCE because a 64-group cannot
 * straddle two 128-blocks (see Q38_I4_GS). */
static void q38_i4sim_row(const uint8_t *w, const float *scl, int I,
                          float *lev, float *gsc) {
    int g = 0;
    for (int base = 0; base < I; base += Q38_I4_GS, g++) {
        const int glen = (I - base < Q38_I4_GS) ? I - base : Q38_I4_GS;
        const float sc = scl[base / FP8_BLOCK];
        float deq[Q38_I4_GS];
        float amax = 0.0f;
        int nan_here = 0;
        for (int k = 0; k < glen; k++) {
            const float v = e4m3_decode(w[base + k]) * sc;
            deq[k] = v;
            if (v != v) { nan_here = 1; continue; }
            const float a = fabsf(v);
            if (a > amax) amax = a;
        }
        float step = amax / 7.0f;
        if (step < 1e-8f) step = 1e-8f;
        gsc[g] = step;
        for (int k = 0; k < glen; k++) {
            const float t = deq[k] / step;
            int q;
            if (t != t) q = 0;                       /* NaN: see g_q38_sim_nan */
            else { q = (int)lrintf(t); if (q < -8) q = -8; else if (q > 7) q = 7; }
            lev[base + k] = (float)q;
        }
        if (nan_here) {
            #pragma omp atomic
            g_q38_sim_nan++;
        }
    }
}

/* y[S,O] = x[S,I] @ W'^T where W' is W quantised to int4-g64.
 *
 * Drop-in for matmul_fp8 (same signature, same layout). The accumulation after
 * the levels exist is matmul_i4_grouped's, INSTRUCTION FOR INSTRUCTION: eight
 * lanes times two per 16 inputs into one __m256, hsum256, then the pinned fmaf
 * with the group scale, then the two-at-a-time scalar tail. matmul_i4_grouped
 * builds its weight vectors as (float)(nibble-8) via
 * cvtepi32_ps(sub(cvtepu8_epi32(unpacklo(lo,hi)), 8)), which is elements
 * i..i+7 then i+8..i+15 in order -- exactly what loadu_ps(lev+i) gives here.
 * So this is bit-identical to packing the row and calling the shipping kernel,
 * and rome_q4sim.c test C measures that it is (float reassociation only).
 *
 * Quantisation is per WEIGHT ROW, not per (row, activation row): the levels are
 * computed once and reused across all S, which is both cheaper and the only
 * thing a converter could mean. */
static void q38_matmul_fp8_sim_i4(float *y, const float *x, const uint8_t *q8,
                                  const float *bscale, int S, int I, int O) {
    const int64_t nblkI = fp8_nblk(I);
    if (I > Q38_SIM_MAXI) {
        fprintf(stderr, "q38_sim: I=%d exceeds Q38_SIM_MAXI=%d\n", I, Q38_SIM_MAXI);
        exit(1);
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        float lev[Q38_SIM_MAXI];
        float gsc[Q38_SIM_MAXI / Q38_I4_GS + 1];
        const uint8_t *w = q8 + (int64_t)o * I;
        const float *scl = bscale + (int64_t)(o / FP8_BLOCK) * nblkI;
        q38_i4sim_row(w, scl, I, lev, gsc);
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0.0f;
            for (int g2 = 0; g2 * Q38_I4_GS < I; g2++) {
                int base = g2 * Q38_I4_GS, glen = Q38_I4_GS;
                if (base + glen > I) glen = I - base;
                const float sc = gsc[g2];
                int i = base;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 acc = _mm256_setzero_ps();
                for (; i + 16 <= base + glen; i += 16) {
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i),
                                          _mm256_loadu_ps(lev + i), acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i + 8),
                                          _mm256_loadu_ps(lev + i + 8), acc);
                }
                a = fmaf(hsum256(acc), sc, a);
#endif
                for (; i < base + glen; i += 2) {
                    if (i + 1 < base + glen) a += (xs[i] * lev[i] + xs[i + 1] * lev[i + 1]) * sc;
                    else a += xs[i] * lev[i] * sc;
                }
            }
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* ============== (b)/(c) dense and head: bf16 -> int8 ====================== */

static inline int q38_i8sim_groups(int I, int gs) { return gs > 0 ? (I + gs - 1) / gs : 1; }

/* BF16 [O,I] -> int8 [O,I] + f32 scales [O,ngroups]. gs <= 0 means one scale
 * per output row, which is the int8 the tree already ships: with gs<=0 this is
 * quantize_rows(w, q, s, O, I, 8) from quant.h element for element (qmax=127,
 * step = max(amax/127, 1e-8), level = clamp(lrintf(w/step), -128, 127)), and
 * rome_q4sim.c test D checks that against quantize_rows itself rather than
 * against a restatement. gs=64 is the same arithmetic per 64-input group, i.e.
 * quant_int4_grouped's grouping with int8's qmax.
 *
 * The source is read with bf16_to_f32, so the quantiser sees exactly the values
 * q38_matmul_bf16 multiplies today: no f32 staging copy, no second rounding. */
static void q38_i8sim_pack_bf16(const uint16_t *W, int O, int I, int gs,
                                int8_t *q, float *s) {
    const int ng = q38_i8sim_groups(I, gs);
    const int span = gs > 0 ? gs : I;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *wr = W + (int64_t)o * I;
        int8_t *qr = q + (int64_t)o * I;
        float *sr = s + (int64_t)o * ng;
        for (int g = 0; g < ng; g++) {
            const int base = g * span;
            const int stop = (base + span > I) ? I : base + span;
            float amax = 0.0f;
            for (int i = base; i < stop; i++) {
                const float a = fabsf(q38_bf16_f32(wr[i]));
                if (a > amax) amax = a;
            }
            float step = amax / 127.0f;
            if (step < 1e-8f) step = 1e-8f;
            sr[g] = step;
            for (int i = base; i < stop; i++) {
                int v = (int)lrintf(q38_bf16_f32(wr[i]) / step);
                if (v > 127) v = 127; else if (v < -128) v = -128;
                qr[i] = (int8_t)v;
            }
        }
    }
}

/* y[S,O] = x[S,I] @ W'^T with W' the int8 above.
 *
 * The reduction is q38_matmul_bf16's, TERM FOR TERM: one eight-lane fmadd
 * accumulator, then buf[0]+buf[1]+...+buf[7] in that order, then the scalar
 * tail. Only two things differ from the BF16 kernel, and both are the format
 * rather than the kernel: the weight lane comes from cvtepi8_epi32+cvtepi32_ps
 * instead of a bf16 shift, and the group scale is applied with one pinned fmaf
 * per group at the end (for gs<=0 that is one multiply for the whole row).
 * Keeping the summation order identical is deliberate: QP(b) must measure the
 * FORMAT, and a different reduction order would put §G14's reassociation noise
 * in the same number. */
static void q38_matmul_i8sim(float *y, const float *x, const int8_t *q,
                             const float *s, int S, int I, int O, int gs) {
    const int ng = q38_i8sim_groups(I, gs);
    const int span = gs > 0 ? gs : I;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sr = s + (int64_t)o * ng;
        for (int r = 0; r < S; r++) {
            const float *xs = x + (int64_t)r * I;
            float a = 0.0f;
            for (int g = 0; g < ng; g++) {
                const int base = g * span;
                const int stop = (base + span > I) ? I : base + span;
                const float sc = sr[g];
                int i = base;
                float acc;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 vacc = _mm256_setzero_ps();
                for (; i + 8 <= stop; i += 8) {
                    __m256i wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i)));
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), _mm256_cvtepi32_ps(wi), vacc);
                }
                float buf[8]; _mm256_storeu_ps(buf, vacc);
                acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#else
                acc = 0.0f;
#endif
                for (; i < stop; i++) acc += xs[i] * (float)w[i];
                a = fmaf(acc, sc, a);
            }
            y[(int64_t)r * O + o] = a;
        }
    }
}

#endif /* COLI_Q38_SIM_H */
