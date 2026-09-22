// tools/hot-expert/m1/m1_native_decode.h
//
// The IQ3_S / IQ4_NL per-lane decode, written ONCE and compiled into both
// m1_native.hip's device kernels and m1_native_emul.cpp's host emulation of
// them. Sharing the source is the point: an emulation that is merely a
// faithful-looking retype of the kernel proves nothing when the two drift,
// which is the same failure mode gate_lib.sh's gate_compare exists to stop
// (two greps that both matched nothing, diffed, reported IDENTICAL).
//
// Ported from the CPU dequantisers in
// ~/src/llama-glm53/ggml/src/ggml-quants.c -- `dequantize_row_iq3_s` (line
// 2607) and `dequantize_row_iq4_nl` (line 2725) -- with the block layouts
// from ggml/src/ggml-common.h. See m1_native.hip's file header for the
// layout tables and for why every payload read is a byte load.
//
// Two macros parameterise the target; both have host defaults:
//   M1N_QUAL         function qualifier (host: inline; device:
//                    __device__ __forceinline__)
//   M1N_HALF_TO_F32  f16 bit pattern -> float (host: ggml_fp16_to_fp32;
//                    device: __half2float(__ushort_as_half(...)))
// Define them BEFORE including this header. They are IEEE-identical
// conversions of the same 16 bits, so the two builds decode the same values.

#pragma once

#include <cstdint>

#ifndef M1N_QUAL
#define M1N_QUAL inline
#endif

#ifndef M1N_HALF_TO_F32
#error "define M1N_HALF_TO_F32(bits) before including m1_native_decode.h"
#endif

// --- block geometry (asserted against ggml's structs by both includers) ---
#define IQ3S_BLOCK_WEIGHTS 256
#define IQ3S_BLOCK_BYTES   110
#define IQ3S_OFF_D           0
#define IQ3S_OFF_QS          2
#define IQ3S_OFF_QH         66
#define IQ3S_OFF_SIGNS      74
#define IQ3S_OFF_SCALES    106

#define IQ4NL_BLOCK_WEIGHTS 32
#define IQ4NL_BLOCK_BYTES   18
#define IQ4NL_OFF_D          0
#define IQ4NL_OFF_QS         2

// ---------------------------------------------------------------------
// One IQ3_S block (256 weights) dotted against 8 elements of x by ONE lane.
//
// Lane-to-weight mapping: 32 lanes x 8 weights = 256. Lane `tid` owns
// weights [8*tid, 8*tid+8), which is exactly sub-block ib32 = tid>>2, pair
// l = tid&3 of the CPU dequantiser's own (ib32, l) loop nest. Every payload
// read is then coalesced across the wave and every header index is
// lane-derivable:
//   qs[2*tid], qs[2*tid+1]     -> lanes sweep qs[0..63] contiguously
//   signs[tid]                 -> lanes sweep signs[0..31] contiguously
//   qh[tid>>2], scales[tid>>3] -> 4- and 8-lane broadcasts
// `sh1`/`sh2` (= 8-2l, 7-2l) and `xb` (= x + block*256 + 8*tid) are hoisted
// by the caller: they are loop-invariant over the blocks of a row.
// ---------------------------------------------------------------------
M1N_QUAL float m1n_iq3s_block_dot(
    const unsigned char * bp,     // block base
    const uint32_t      * grid,   // iq3s_grid, 512 entries
    const float         * xb,     // x + block*256 + 8*tid
    int tid, int ib32, int sh1, int sh2)
{
    // Two byte loads, not a ushort load: a packed row of 110 B blocks gives
    // `d` no alignment guarantee beyond 2, and the row base is itself
    // 110-strided. Little-endian, matching GGML_FP16_TO_FP32 on these bytes.
    const unsigned int dbits = (unsigned int)bp[IQ3S_OFF_D] |
                               ((unsigned int)bp[IQ3S_OFF_D + 1] << 8);
    const float d = M1N_HALF_TO_F32(dbits);

    const unsigned char * qs     = bp + IQ3S_OFF_QS;
    const unsigned char * qh     = bp + IQ3S_OFF_QH;
    const unsigned char * signs  = bp + IQ3S_OFF_SIGNS;
    const unsigned char * scales = bp + IQ3S_OFF_SCALES;

    // db = d * (1 + 2*scale_nibble): low nibble for even ib32, high for odd
    // (ggml-quants.c: db1 from `& 0xf`, db2 from `>> 4`, on scales[ib32/2]).
    const int sc  = scales[ib32 >> 1];
    const int nib = (ib32 & 1) ? (sc >> 4) : (sc & 0xf);
    const float db = d * (float)(1 + 2 * nib);

    // Grid indices: low 8 bits from qs, bit 8 from qh[ib32] -- bit 2*l for
    // grid1 and 2*l+1 for grid2, i.e. the CPU code's `(qh << (8-2l)) & 256`
    // and `(qh << (7-2l)) & 256` with the shifts hoisted into sh1/sh2.
    const int qhb = qh[ib32];
    const int i1  = (int)qs[2 * tid]     | ((qhb << sh1) & 256);
    const int i2  = (int)qs[2 * tid + 1] | ((qhb << sh2) & 256);
    const uint32_t g1 = grid[i1];
    const uint32_t g2 = grid[i2];

    // signs[4*ib32 + l] == signs[tid] under this mapping. Bit m is the sign
    // of output element m (kmask_iq2xs[m] == 1u<<m; the includers assert it).
    const int sg = signs[tid];

    float a0 = 0.0f, a1 = 0.0f;
#ifdef __HIP_DEVICE_COMPILE__
    #pragma unroll
#endif
    for (int j = 0; j < 4; j++) {
        // A grid entry is one uint32 read as 4 uint8 by the CPU code
        // (`(const uint8_t *)(iq3s_grid + idx)`): little-endian byte j.
        float w1 = db * (float)((g1 >> (8 * j)) & 0xffu);
        if ((sg >> j) & 1)       w1 = -w1;
        a0 += w1 * xb[j];
        float w2 = db * (float)((g2 >> (8 * j)) & 0xffu);
        if ((sg >> (j + 4)) & 1) w2 = -w2;
        a1 += w2 * xb[j + 4];
    }
    return a0 + a1;
}

// ---------------------------------------------------------------------
// One IQ4_NL "chunk" -- 4 qs bytes = 8 weights -- by ONE lane.
//
// A 32-weight block's payload is 16 qs bytes = 4 chunks, so a row of
// (K/32) blocks has (K/32)*4 chunks and a lane walks them with a stride-32
// loop. `w` is the chunk index: block = w>>2, byte offset = (w&3)*4.
//
// The low and high nibble streams get their own accumulator so the two
// dependent FMA chains can dual-issue; they are summed by the caller.
//
// The INTERLEAVE is the trap: qs[j]'s two nibbles are weights j and j+16 of
// the block, not two adjacent weights (dequantize_row_iq4_nl writes
// y[j] and y[j+QK4_NL/2]). Getting it wrong is a silent 50 % scramble that
// still produces plausible-looking numbers.
// ---------------------------------------------------------------------
M1N_QUAL void m1n_iq4nl_chunk_dot(
    const unsigned char * row_ptr,  // first block of this output row
    const int8_t        * kvals,    // kvalues_iq4nl, 16 entries
    const float         * h_row,    // activations for this (token, krank)
    int w,                          // chunk index
    float * acc_lo, float * acc_hi)
{
    const int blk = w >> 2;
    const int off = (w & 3) << 2;   // 0, 4, 8, 12

    const unsigned char * bp = row_ptr + (long long)blk * IQ4NL_BLOCK_BYTES;
    const unsigned int dbits = (unsigned int)bp[IQ4NL_OFF_D] |
                               ((unsigned int)bp[IQ4NL_OFF_D + 1] << 8);
    const float d = M1N_HALF_TO_F32(dbits);

    const unsigned char * qs = bp + IQ4NL_OFF_QS + off;
    const float         * hb = h_row + (blk << 5) + off;

    float lo = *acc_lo, hi = *acc_hi;
#ifdef __HIP_DEVICE_COMPILE__
    #pragma unroll
#endif
    for (int c = 0; c < 4; c++) {
        const int q = qs[c];
        lo += (d * (float)kvals[q & 0xf]) * hb[c];
        hi += (d * (float)kvals[q >> 4])  * hb[c + 16];
    }
    *acc_lo = lo;
    *acc_hi = hi;
}
