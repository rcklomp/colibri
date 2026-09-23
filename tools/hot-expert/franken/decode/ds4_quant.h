// tools/hot-expert/franken/decode/ds4_quant.h
//
// The per-lane decode of the six weight formats DeepSeek-V4-Flash
// UD-IQ2_M carries that decode_quant.h / m1_native_decode.h do not:
//
//   format    where in this file (read off the GGUF header, 2026-09-23)
//   IQ2_XXS   ffn_{gate,up}_exps on 42 of 43 layers           (66 B / 256 w)
//   IQ3_XXS   ffn_down_exps on 41 of 43 layers                (98 B / 256 w)
//   IQ2_S     ffn_{gate,up}_exps on layer 26                  (82 B / 256 w)
//   MXFP4     ffn_down_exps on layers 26 and 42               (17 B /  32 w)
//   Q5_K      attn_q_a (42 layers), ffn_{gate,up}_shexp (42), token_embd
//                                                             (176 B / 256 w)
//   Q4_K      output.weight (lm_head)                         (144 B / 256 w)
//
// Written ONCE and meant to compile into both a HIP kernel and the host
// reference, exactly as decode_quant.h is (same two macros, FK_QUAL and
// FK_HALF_TO_F32). Ported from ~/src/llama-glm53/ggml/src/ggml-quants.c:
// dequantize_row_q4_K, _q5_K, _iq2_xxs, _iq2_s, _iq3_xxs, _mxfp4, with the
// block layouts of ggml-common.h. ds4_quant_emul.cpp proves every one of
// them BIT-EXACT against ggml's own to_float (one-hot extraction, every
// weight of real and random blocks) -- that check, not this comment, is
// what "ported" means.
//
// Lane mapping: the same as decode_quant.h. For a 256-weight format lane
// `tid` in [0,32) owns weights [8*tid, 8*tid+8) of the block and `xb` points
// at x + block*256 + 8*tid. For MXFP4 (32-weight blocks) lane `tid` owns
// weight `tid`, like Q8_0.
//
// The lookup tables (iq2xxs_grid, iq2s_grid, iq3xxs_grid, ksigns_iq2xs,
// kmask_iq2xs, kvalues_mxfp4) are PARAMETERS, as m1_native_decode.h does
// with iq3s_grid: the host passes ggml-common.h's arrays, a kernel passes
// its __constant__ copy.
//
// ROUNDING. Every weight is formed with the SAME float operations, in the
// same order, as ggml's dequantiser, and ggml-base is compiled without FMA
// (no -march in its flags; checked in build-hip's flags.make). Q4_K/Q5_K's
// `d1*q - m1` must therefore NOT be contracted into an fma, which hipcc
// would do by default on the device (-ffp-contract=fast): the one place it
// matters is fenced with `#pragma clang fp contract(off)`. For the IQ types
// the weight is a product chain (db * grid * sign), which contraction cannot
// change; only the accumulate into `a` may fuse, and that is a summation
// order question, not a decode one.

#pragma once

#include <cstddef>
#include <cstdint>

#ifndef FK_QUAL
#define FK_QUAL inline
#endif
#ifndef FK_HALF_TO_F32
#error "define FK_HALF_TO_F32(bits) before including ds4_quant.h"
#endif

#if defined(__clang__)
#define FK_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define FK_NO_CONTRACT
#endif

// ------------------------------------------------------------ geometry ----
#define FK_Q4K_BLOCK_WEIGHTS    256
#define FK_Q4K_BLOCK_BYTES      144   // d, dmin, scales[12], qs[128]
#define FK_Q4K_OFF_D              0
#define FK_Q4K_OFF_DMIN           2
#define FK_Q4K_OFF_SCALES         4
#define FK_Q4K_OFF_QS            16

#define FK_Q5K_BLOCK_WEIGHTS    256
#define FK_Q5K_BLOCK_BYTES      176   // d, dmin, scales[12], qh[32], qs[128]
#define FK_Q5K_OFF_D              0
#define FK_Q5K_OFF_DMIN           2
#define FK_Q5K_OFF_SCALES         4
#define FK_Q5K_OFF_QH            16
#define FK_Q5K_OFF_QS            48

#define FK_IQ2XXS_BLOCK_WEIGHTS 256
#define FK_IQ2XXS_BLOCK_BYTES    66   // d, qs[32] as uint16 (64 bytes)
#define FK_IQ2XXS_OFF_D           0
#define FK_IQ2XXS_OFF_QS          2

#define FK_IQ2S_BLOCK_WEIGHTS   256
#define FK_IQ2S_BLOCK_BYTES      82   // d, qs[64] (32 grid lo + 32 signs), qh[8], scales[8]
#define FK_IQ2S_OFF_D             0
#define FK_IQ2S_OFF_QS            2
#define FK_IQ2S_OFF_QH           66
#define FK_IQ2S_OFF_SCALES       74

#define FK_IQ3XXS_BLOCK_WEIGHTS 256
#define FK_IQ3XXS_BLOCK_BYTES    98   // d, qs[96] (64 grid + 32 scales_and_signs)
#define FK_IQ3XXS_OFF_D           0
#define FK_IQ3XXS_OFF_QS          2

#define FK_MXFP4_BLOCK_WEIGHTS   32
#define FK_MXFP4_BLOCK_BYTES     17   // e (E8M0), qs[16]
#define FK_MXFP4_OFF_E            0
#define FK_MXFP4_OFF_QS           1

FK_QUAL unsigned int fk_ld_u16(const unsigned char * p) {
    return (unsigned int) p[0] | ((unsigned int) p[1] << 8);
}
FK_QUAL unsigned int fk_ld_u32(const unsigned char * p) {
    return (unsigned int) p[0] | ((unsigned int) p[1] << 8) |
           ((unsigned int) p[2] << 16) | ((unsigned int) p[3] << 24);
}

// ggml_e8m0_to_fp32_half (ggml-impl.h:477): 0.5 * 2^(x-127), with the two
// denormal patterns for x < 2. Built from bits, so identical on any target.
FK_QUAL float fk_e8m0_to_f32_half(unsigned int x) {
    union { float f; unsigned int u; } c;
    c.u = (x < 2u) ? (0x00200000u << x) : ((x - 1u) << 23);
    return c.f;
}

// get_scale_min_k4 (ggml-quants.c:880), the 6-bit scale/min of sub-block j.
FK_QUAL void fk_k4_scale_min(int j, const unsigned char * q, int * d, int * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------- Q4_K ----
//
// dequantize_row_q4_K: four chunks of 64 weights; chunk c's first 32 are the
// LOW nibbles of qs[32c .. 32c+31] at sub-block scale 2c, the next 32 the HIGH
// nibbles at scale 2c+1:  y = (d*sc) * nib - (dmin*m).
// Lane tid: chunk c = tid>>3, offset o = (tid&7)*8 in the chunk, half = o>>5,
// l0 = o&31; its eight weights share one scale and one nibble half.
FK_QUAL float fk_q4k_block_dot(const unsigned char * bp, const float * xb, int tid) {
    FK_NO_CONTRACT
    const float d   = FK_HALF_TO_F32(fk_ld_u16(bp + FK_Q4K_OFF_D));
    const float min = FK_HALF_TO_F32(fk_ld_u16(bp + FK_Q4K_OFF_DMIN));
    const int c    = tid >> 3;
    const int o    = (tid & 7) << 3;
    const int half = o >> 5;
    const int l0   = o & 31;
    int sc, m;
    fk_k4_scale_min(2 * c + half, bp + FK_Q4K_OFF_SCALES, &sc, &m);
    const float d1 = d * (float) sc;
    const float m1 = min * (float) m;
    const unsigned char * q = bp + FK_Q4K_OFF_QS + 32 * c + l0;
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        const int n0 = half ? (q[j]     >> 4) : (q[j]     & 0xF);
        const int n1 = half ? (q[j + 1] >> 4) : (q[j + 1] & 0xF);
        const float w0 = d1 * (float) n0 - m1;
        const float w1 = d1 * (float) n1 - m1;
        a0 += w0 * xb[j];
        a1 += w1 * xb[j + 1];
    }
    return a0 + a1;
}

// ---------------------------------------------------------------- Q5_K ----
//
// dequantize_row_q5_K: as Q4_K plus a fifth bit from qh[l] (l in 0..31, qh is
// NOT advanced per chunk): chunk c's low half tests bit 2c, its high half
// bit 2c+1:  y = (d*sc) * (nib + (bit ? 16 : 0)) - (dmin*m).
FK_QUAL float fk_q5k_block_dot(const unsigned char * bp, const float * xb, int tid) {
    FK_NO_CONTRACT
    const float d   = FK_HALF_TO_F32(fk_ld_u16(bp + FK_Q5K_OFF_D));
    const float min = FK_HALF_TO_F32(fk_ld_u16(bp + FK_Q5K_OFF_DMIN));
    const int c    = tid >> 3;
    const int o    = (tid & 7) << 3;
    const int half = o >> 5;
    const int l0   = o & 31;
    int sc, m;
    fk_k4_scale_min(2 * c + half, bp + FK_Q5K_OFF_SCALES, &sc, &m);
    const float d1 = d * (float) sc;
    const float m1 = min * (float) m;
    const unsigned char * ql = bp + FK_Q5K_OFF_QS + 32 * c + l0;
    const unsigned char * qh = bp + FK_Q5K_OFF_QH + l0;
    const unsigned int ubit = 1u << (2 * c + half);
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        const int n0 = (half ? (ql[j]     >> 4) : (ql[j]     & 0xF)) + ((qh[j]     & ubit) ? 16 : 0);
        const int n1 = (half ? (ql[j + 1] >> 4) : (ql[j + 1] & 0xF)) + ((qh[j + 1] & ubit) ? 16 : 0);
        const float w0 = d1 * (float) n0 - m1;
        const float w1 = d1 * (float) n1 - m1;
        a0 += w0 * xb[j];
        a1 += w1 * xb[j + 1];
    }
    return a0 + a1;
}

// ------------------------------------------------------------- IQ2_XXS ----
//
// dequantize_row_iq2_xxs: per 32-weight sub-block ib32, two uint32 words at
// qs + 8*ib32: aux32[0] holds four grid indices (one byte each), aux32[1]
// four 7-bit sign indices and the 4-bit scale in its top nibble.
//   db = d * (0.5f + (aux32[1] >> 28)) * 0.25f
//   y[8l+j] = db * grid[aux8[l]][j] * (ksigns[(aux32[1] >> 7l) & 127] & kmask[j] ? -1 : 1)
// Lane tid: ib32 = tid>>2, l = tid&3 -- exactly one grid entry, one sign
// byte, one scale.
FK_QUAL float fk_iq2xxs_block_dot(const unsigned char * bp, const uint64_t * grid_tab,
                                  const unsigned char * ksigns, const unsigned char * kmask,
                                  const float * xb, int tid) {
    const float d = FK_HALF_TO_F32(fk_ld_u16(bp + FK_IQ2XXS_OFF_D));
    const int ib32 = tid >> 2, l = tid & 3;
    const unsigned char * w = bp + FK_IQ2XXS_OFF_QS + 8 * ib32;
    const unsigned int aux1 = fk_ld_u32(w + 4);
    const float db = d * (0.5f + (float) (aux1 >> 28)) * 0.25f;
    const uint64_t g = grid_tab[w[l]];
    const unsigned int signs = ksigns[(aux1 >> (7 * l)) & 127u];
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        const float w0 = db * (float) (unsigned int) ((g >> (8 * j))       & 0xFFu) * ((signs & kmask[j])     ? -1.f : 1.f);
        const float w1 = db * (float) (unsigned int) ((g >> (8 * (j + 1))) & 0xFFu) * ((signs & kmask[j + 1]) ? -1.f : 1.f);
        a0 += w0 * xb[j];
        a1 += w1 * xb[j + 1];
    }
    return a0 + a1;
}

// --------------------------------------------------------------- IQ2_S ----
//
// dequantize_row_iq2_s: qs[0..31] the grid index low bytes (4 per ib32),
// qs[32..63] the sign bytes (4 per ib32), qh[ib32] two high index bits per
// grid entry, scales[ib32] two 4-bit scales (low nibble for l = 0,1, high for
// l = 2,3).
//   db[h] = d * (0.5f + nibble_h) * 0.25f
//   idx   = qs[4ib32+l] | ((qh[ib32] << (8-2l)) & 0x300)
//   y     = db[l/2] * grid[idx][j] * (signs[4ib32+l] & kmask[j] ? -1 : 1)
FK_QUAL float fk_iq2s_block_dot(const unsigned char * bp, const uint64_t * grid_tab,
                                const unsigned char * kmask, const float * xb, int tid) {
    const float d = FK_HALF_TO_F32(fk_ld_u16(bp + FK_IQ2S_OFF_D));
    const int ib32 = tid >> 2, l = tid & 3;
    const unsigned char * qs = bp + FK_IQ2S_OFF_QS;
    const unsigned int sc = bp[FK_IQ2S_OFF_SCALES + ib32];
    const unsigned int nib = (l >> 1) ? (sc >> 4) : (sc & 0xFu);
    const float dl = d * (0.5f + (float) nib) * 0.25f;
    const unsigned int qh = bp[FK_IQ2S_OFF_QH + ib32];
    const unsigned int idx = (unsigned int) qs[4 * ib32 + l] | ((qh << (8 - 2 * l)) & 0x300u);
    const uint64_t g = grid_tab[idx];
    const unsigned int signs = qs[32 + 4 * ib32 + l];
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        const float w0 = dl * (float) (unsigned int) ((g >> (8 * j))       & 0xFFu) * ((signs & kmask[j])     ? -1.f : 1.f);
        const float w1 = dl * (float) (unsigned int) ((g >> (8 * (j + 1))) & 0xFFu) * ((signs & kmask[j + 1]) ? -1.f : 1.f);
        a0 += w0 * xb[j];
        a1 += w1 * xb[j + 1];
    }
    return a0 + a1;
}

// ------------------------------------------------------------- IQ3_XXS ----
//
// dequantize_row_iq3_xxs: qs[0..63] grid indices (8 per ib32, two per group
// of 8 weights), qs[64 + 4*ib32] one uint32 of four 7-bit sign indices and
// the 4-bit scale in the top nibble.
//   db = d * (0.5f + (aux32 >> 28)) * 0.5f
//   y[8l+j]   = db * grid[qs[2l]][j]   * (ksigns[(aux32>>7l)&127] & kmask[j]   ? -1 : 1)   j<4
//   y[8l+4+j] = db * grid[qs[2l+1]][j] * (....................... & kmask[j+4] ? -1 : 1)
FK_QUAL float fk_iq3xxs_block_dot(const unsigned char * bp, const uint32_t * grid_tab,
                                  const unsigned char * ksigns, const unsigned char * kmask,
                                  const float * xb, int tid) {
    const float d = FK_HALF_TO_F32(fk_ld_u16(bp + FK_IQ3XXS_OFF_D));
    const int ib32 = tid >> 2, l = tid & 3;
    const unsigned char * qs  = bp + FK_IQ3XXS_OFF_QS + 8 * ib32;
    const unsigned int aux32  = fk_ld_u32(bp + FK_IQ3XXS_OFF_QS + 64 + 4 * ib32);
    const float db = d * (0.5f + (float) (aux32 >> 28)) * 0.5f;
    const unsigned int signs = ksigns[(aux32 >> (7 * l)) & 127u];
    const uint32_t g1 = grid_tab[qs[2 * l + 0]];
    const uint32_t g2 = grid_tab[qs[2 * l + 1]];
    float a0 = 0.0f, a1 = 0.0f;
    for (int j = 0; j < 4; j += 2) {
        const float w0 = db * (float) ((g1 >> (8 * j))       & 0xFFu) * ((signs & kmask[j])     ? -1.f : 1.f);
        const float w1 = db * (float) ((g1 >> (8 * (j + 1))) & 0xFFu) * ((signs & kmask[j + 1]) ? -1.f : 1.f);
        a0 += w0 * xb[j];
        a1 += w1 * xb[j + 1];
    }
    for (int j = 0; j < 4; j += 2) {
        const float w0 = db * (float) ((g2 >> (8 * j))       & 0xFFu) * ((signs & kmask[j + 4]) ? -1.f : 1.f);
        const float w1 = db * (float) ((g2 >> (8 * (j + 1))) & 0xFFu) * ((signs & kmask[j + 5]) ? -1.f : 1.f);
        a0 += w0 * xb[4 + j];
        a1 += w1 * xb[5 + j];
    }
    return a0 + a1;
}

// --------------------------------------------------------------- MXFP4 ----
//
// dequantize_row_mxfp4: d = e8m0_half(e); qs[j] holds weight j (low nibble)
// and weight j+16 (high nibble);  y = (float) kvalues_mxfp4[nib] * d.
// One weight a lane, like Q8_0.
FK_QUAL float fk_mxfp4_lane_dot(const unsigned char * bp, const signed char * kvals,
                                const float * xb, int lane) {
    const float d = fk_e8m0_to_f32_half(bp[FK_MXFP4_OFF_E]);
    const unsigned int q = bp[FK_MXFP4_OFF_QS + (lane & 15)];
    const int nib = (lane >> 4) ? (int) (q >> 4) : (int) (q & 0xFu);
    const float w = (float) kvals[nib] * d;
    return w * xb[lane];
}
