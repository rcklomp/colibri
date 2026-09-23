// tools/hot-expert/franken/decode/decode_quant.h
//
// The per-lane decode of every weight format layers 0-15 of
// Qwen3.8-Flash-Next UD-IQ4_XS actually carry, written ONCE and compiled
// into both the HIP kernels (decode_gpu.hip) and the host reference
// (decode_cpu.cpp).
//
// This is the same discipline as tools/hot-expert/m1/m1_native_decode.h, and
// for the same reason stated there: a host "emulation" that is a
// faithful-LOOKING retype of a kernel proves nothing once the two drift.
// IQ3_S and IQ4_NL are not re-derived here at all -- m1_native_decode.h is
// included and its two primitives are used verbatim, so the expert kernels
// this file feeds are bit-identical to the ones §M1 candidate 3 measured
// (record §M1: 385 GB/s at batch 1, oracle cos 1.000000).
//
// Which format appears where (read off the GGUF header, 2026-09-22; an
// Unsloth "UD" dynamic quant is NOT uniform across layers, and assuming it
// is would have silently mis-decoded layers 2 and 4):
//
//   trunk, every layer 0-15          Q8_0  (AoS block_q8_0: f16 d + 32 int8)
//   blk.N.indexer.{q,k}_proj         BF16
//   ffn_gate_inp, ssm_{a,alpha,beta,dt,conv1d,norm}, hc_*_norm/inject, F32
//   ffn_{gate,up}_exps               IQ3_S  on layers 0,1,3,4..15;  IQ4_XS on layer 2
//   ffn_down_exps                    IQ4_NL on layers 0,1,3,5..15;  Q8_0   on layers 2 and 4
//   per_layer_token_embd (host)      IQ4_NL  (dequantised by ../ple.cpp, not here)
//
// Lane mapping convention, shared by every primitive below: a 32-lane wave
// covers one block, lane `tid` owning a contiguous run of weights. For the
// 256-weight formats (IQ3_S, IQ4_XS) that is 8 weights at 8*tid; for the
// 32-weight formats (Q8_0) it is one weight at tid; IQ4_NL keeps
// m1_native_decode.h's chunk form (8 weights per 4 qs bytes). The host
// reference walks tid 0..31 in a loop and sums, so it executes the SAME
// expressions as a wave does -- only the reduction order differs, which is a
// reassociation and not a decode difference.
//
// Two macros parameterise the target, exactly as m1_native_decode.h does:
//   FK_QUAL          function qualifier (host: inline; device: __device__ __forceinline__)
//   FK_HALF_TO_F32   f16 bit pattern -> float
// Define both before including.

#pragma once

#include <cstdint>

#include "decode_shapes.h"   // FkQuantType

#ifndef FK_QUAL
#define FK_QUAL inline
#endif

#ifndef FK_HALF_TO_F32
#error "define FK_HALF_TO_F32(bits) before including decode_quant.h"
#endif

// m1_native_decode.h's two macros are the same idea under different names.
#ifndef M1N_QUAL
#define M1N_QUAL FK_QUAL
#endif
#ifndef M1N_HALF_TO_F32
#define M1N_HALF_TO_F32(bits) FK_HALF_TO_F32(bits)
#endif
#include "../../m1/m1_native_decode.h"   // m1n_iq3s_block_dot, m1n_iq4nl_chunk_dot
#include "ds4_quant.h"                    // the DeepSeek-V4 formats (Q4_K ... MXFP4)

// ---------------------------------------------------------------- Q8_0 ----
//
// ggml's AoS block_q8_0 { ggml_half d; int8_t qs[32]; } == 34 bytes.
// NOT m5_bench.hip's planar/SoA convention: that harness deliberately
// simplified the layout ("the brief says the format does not matter for
// M5"), and this engine reads the file as it is. The 34-byte stride leaves
// `d` only 2-aligned, so it is read as two bytes, as in m1_native_decode.h.
#define FK_Q8_0_BLOCK_WEIGHTS 32
#define FK_Q8_0_BLOCK_BYTES   34
#define FK_Q8_0_OFF_D          0
#define FK_Q8_0_OFF_QS         2

FK_QUAL float fk_q8_0_block_d(const unsigned char * bp) {
    const unsigned int dbits = (unsigned int)bp[FK_Q8_0_OFF_D] |
                               ((unsigned int)bp[FK_Q8_0_OFF_D + 1] << 8);
    return FK_HALF_TO_F32(dbits);
}

// One weight of one Q8_0 block times its activation. `lane` in [0, 32).
// The whole wave reads qs[0..31] -- 32 contiguous bytes, one coalesced
// request -- and every lane recomputes `d` from the same two bytes (a
// broadcast, not 32 loads).
FK_QUAL float fk_q8_0_lane_dot(const unsigned char * bp, const float * xb, int lane) {
    const signed char * qs = (const signed char *)(bp + FK_Q8_0_OFF_QS);
    return fk_q8_0_block_d(bp) * (float)qs[lane] * xb[lane];
}

// --------------------------------------------------------------- IQ4_XS ---
//
// ggml's block_iq4_xs { ggml_half d; uint16_t scales_h; uint8_t scales_l[4];
// uint8_t qs[128]; } == 136 bytes for 256 weights, decoded by
// dequantize_row_iq4_xs (ggml-quants.c:2743):
//
//   for ib in [0,8):                                 // 8 sub-blocks of 32
//     ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)
//     dl = d * (ls - 32)
//     for j in [0,16): y[j] = dl*kv[qs[j]&0xf];  y[j+16] = dl*kv[qs[j]>>4]
//     qs += 16
//
// THE INTERLEAVE IS THE SAME TRAP AS IQ4_NL'S: qs[j] carries weights j and
// j+16 of the sub-block, not two adjacent weights.
//
// Lane mapping: lane `tid` owns weights [8*tid, 8*tid+8) of the 256. Those
// eight always lie inside sub-block ib = tid>>2 at local offset
// j0 = 8*(tid&3) in {0,8,16,24}; j0 < 16 means the LOW nibbles of
// qs[ib*16 + j0 .. +7], j0 >= 16 the HIGH nibbles of qs[ib*16 + j0-16 .. +7].
#define FK_IQ4XS_BLOCK_WEIGHTS 256
#define FK_IQ4XS_BLOCK_BYTES   136
#define FK_IQ4XS_OFF_D           0
#define FK_IQ4XS_OFF_SCALES_H    2
#define FK_IQ4XS_OFF_SCALES_L    4
#define FK_IQ4XS_OFF_QS          8

FK_QUAL float fk_iq4xs_block_dot(
    const unsigned char * bp,      // block base
    const signed char   * kvals,   // kvalues_iq4nl, 16 entries
    const float         * xb,      // x + block*256 + 8*tid
    int tid)
{
    const unsigned int dbits = (unsigned int)bp[FK_IQ4XS_OFF_D] |
                               ((unsigned int)bp[FK_IQ4XS_OFF_D + 1] << 8);
    const float d = FK_HALF_TO_F32(dbits);

    const unsigned int scales_h = (unsigned int)bp[FK_IQ4XS_OFF_SCALES_H] |
                                  ((unsigned int)bp[FK_IQ4XS_OFF_SCALES_H + 1] << 8);
    const unsigned char * scales_l = bp + FK_IQ4XS_OFF_SCALES_L;
    const unsigned char * qs       = bp + FK_IQ4XS_OFF_QS;

    const int ib = tid >> 2;                       // 0..7
    const int ls = (int)((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xf) |
                   (int)(((scales_h >> (2 * ib)) & 3u) << 4);
    const float dl = d * (float)(ls - 32);

    const int j0    = (tid & 3) << 3;              // 0, 8, 16, 24
    const int base  = j0 & 15;                     // 0 or 8
    const int high  = j0 >> 4;                     // 0 = low nibbles, 1 = high
    const unsigned char * q = qs + (ib << 4) + base;

    float a0 = 0.0f, a1 = 0.0f;
#ifdef __HIP_DEVICE_COMPILE__
    #pragma unroll
#endif
    for (int j = 0; j < 8; j += 2) {
        const int q0 = high ? (q[j]     >> 4) : (q[j]     & 0xf);
        const int q1 = high ? (q[j + 1] >> 4) : (q[j + 1] & 0xf);
        a0 += (dl * (float)kvals[q0]) * xb[j];
        a1 += (dl * (float)kvals[q1]) * xb[j + 1];
    }
    return a0 + a1;
}

// ---------------------------------------------------------------- Q6_K ----
//
// ggml's block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
// ggml_half d; } == 210 bytes for 256 weights. Only `output.weight`
// (lm_head, [2560, 248320]) uses it, and only step 3 needs it.
//
// dequantize_row_q6_K (ggml-quants.c:1939) walks the super-block in two
// GROUPS of 128 weights, and inside a group the four quarters are strided,
// not contiguous:
//
//   for n in {0,128}:
//     for l in 0..31:  is = l/16
//       y[l+ 0] = d*sc[is+0]*(((ql[l   ] & 0xF) | ((qh[l]>>0 & 3)<<4)) - 32)
//       y[l+32] = d*sc[is+2]*(((ql[l+32] & 0xF) | ((qh[l]>>2 & 3)<<4)) - 32)
//       y[l+64] = d*sc[is+4]*(((ql[l   ] >>  4) | ((qh[l]>>4 & 3)<<4)) - 32)
//       y[l+96] = d*sc[is+6]*(((ql[l+32] >>  4) | ((qh[l]>>6 & 3)<<4)) - 32)
//     ql += 64; qh += 32; sc += 8
//
// Lane mapping: lane `tid` owns weights [8*tid, 8*tid+8). Because 8 divides
// 32, those eight always land in ONE quarter of ONE group, so the strided
// layout costs no branching inside the loop:
//   group   n = tid>>4          (0 or 1)
//   offset  o = (tid&15)*8      (0..120, a multiple of 8)
//   quarter q = o>>5, base l0 = o&31 in {0,8,16,24}
#define FK_Q6K_BLOCK_WEIGHTS 256
#define FK_Q6K_BLOCK_BYTES   210
#define FK_Q6K_OFF_QL          0
#define FK_Q6K_OFF_QH        128
#define FK_Q6K_OFF_SCALES    192
#define FK_Q6K_OFF_D         208

FK_QUAL float fk_q6k_block_dot(const unsigned char * bp, const float * xb, int tid) {
    const unsigned int dbits = (unsigned int)bp[FK_Q6K_OFF_D] |
                               ((unsigned int)bp[FK_Q6K_OFF_D + 1] << 8);
    const float d = FK_HALF_TO_F32(dbits);

    const int n  = tid >> 4;
    const int o  = (tid & 15) << 3;
    const int qt = o >> 5;              // 0..3
    const int l0 = o & 31;              // 0, 8, 16, 24

    const unsigned char * ql = bp + FK_Q6K_OFF_QL     + n * 64;
    const unsigned char * qh = bp + FK_Q6K_OFF_QH     + n * 32;
    const signed char   * sc = (const signed char *)(bp + FK_Q6K_OFF_SCALES) + n * 8;

    // within a quarter the low/high nibble and the qh shift are fixed
    const int lo_off  = (qt & 1) ? 32 : 0;     // quarters 1 and 3 read ql[l+32]
    const int hi_nib  = (qt >= 2);             // quarters 2 and 3 take the high nibble
    const int qh_shift = qt << 1;              // 0, 2, 4, 6
    const int sc_base  = qt << 1;              // sc[is + 0/2/4/6]

    float a0 = 0.0f, a1 = 0.0f;
#ifdef __HIP_DEVICE_COMPILE__
    #pragma unroll
#endif
    for (int j = 0; j < 8; j += 2) {
        const int lA = l0 + j, lB = l0 + j + 1;
        const int vA = (int)(hi_nib ? (ql[lA + lo_off] >> 4) : (ql[lA + lo_off] & 0xF))
                     | (int)(((qh[lA] >> qh_shift) & 3) << 4);
        const int vB = (int)(hi_nib ? (ql[lB + lo_off] >> 4) : (ql[lB + lo_off] & 0xF))
                     | (int)(((qh[lB] >> qh_shift) & 3) << 4);
        a0 += (d * (float) sc[(lA >> 4) + sc_base]) * (float)(vA - 32) * xb[j];
        a1 += (d * (float) sc[(lB >> 4) + sc_base]) * (float)(vB - 32) * xb[j + 1];
    }
    return a0 + a1;
}

// ----------------------------------------------------------------- BF16 ---
//
// A bf16 value is the top 16 bits of an IEEE-754 f32, so widening is a
// shift. Only blk.N.indexer.{q,k}_proj are stored this way.
FK_QUAL float fk_bf16_to_f32(unsigned short b) {
    union { float f; unsigned int u; } c;
    c.u = ((unsigned int)b) << 16;
    return c.f;
}
FK_QUAL unsigned short fk_f32_to_bf16(float f) {
    union { float f; unsigned int u; } c; c.f = f;
    const unsigned int lsb  = (c.u >> 16) & 1u;
    const unsigned int bias = 0x7fffu + lsb;      // round to nearest even
    c.u += bias;
    return (unsigned short)(c.u >> 16);
}

// ------------------------------------------------------ row byte sizes ----
//
// ggml_row_size(type, K) for the four formats a GEMV here can meet. Kept as
// a function of K so decode_model.cpp can assert it against the tensor's own
// nbytes/ne1 rather than trusting either side.
FK_QUAL size_t fk_row_bytes(int type, long long K) {
    switch (type) {
        case FK_Q_F32:    return (size_t)K * 4;
        case FK_Q_BF16:   return (size_t)K * 2;
        case FK_Q_Q8_0:   return (size_t)(K / FK_Q8_0_BLOCK_WEIGHTS)   * FK_Q8_0_BLOCK_BYTES;
        case FK_Q_IQ4_NL: return (size_t)(K / IQ4NL_BLOCK_WEIGHTS)     * IQ4NL_BLOCK_BYTES;
        case FK_Q_IQ4_XS: return (size_t)(K / FK_IQ4XS_BLOCK_WEIGHTS)  * FK_IQ4XS_BLOCK_BYTES;
        case FK_Q_IQ3_S:  return (size_t)(K / IQ3S_BLOCK_WEIGHTS)      * IQ3S_BLOCK_BYTES;
        case FK_Q_Q6_K:   return (size_t)(K / FK_Q6K_BLOCK_WEIGHTS)   * FK_Q6K_BLOCK_BYTES;
        case FK_Q_Q4_K:    return (size_t)(K / FK_Q4K_BLOCK_WEIGHTS)    * FK_Q4K_BLOCK_BYTES;
        case FK_Q_Q5_K:    return (size_t)(K / FK_Q5K_BLOCK_WEIGHTS)    * FK_Q5K_BLOCK_BYTES;
        case FK_Q_IQ2_XXS: return (size_t)(K / FK_IQ2XXS_BLOCK_WEIGHTS) * FK_IQ2XXS_BLOCK_BYTES;
        case FK_Q_IQ2_S:   return (size_t)(K / FK_IQ2S_BLOCK_WEIGHTS)   * FK_IQ2S_BLOCK_BYTES;
        case FK_Q_IQ3_XXS: return (size_t)(K / FK_IQ3XXS_BLOCK_WEIGHTS) * FK_IQ3XXS_BLOCK_BYTES;
        case FK_Q_MXFP4:   return (size_t)(K / FK_MXFP4_BLOCK_WEIGHTS)  * FK_MXFP4_BLOCK_BYTES;
        default:          return 0;
    }
}
