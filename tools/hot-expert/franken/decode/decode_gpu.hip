// tools/hot-expert/franken/decode/decode_gpu.hip
//
// L0 step 2's device-0 kernels (L0-STEP2-BRIEF-2026-09-22.md, deliverable B).
// BUILT ONLY -- the agent that wrote this file was not allowed to touch a GPU
// (no --device flags on the docker run, no rocminfo, no engine start), so
// nothing here has been executed. The orchestrator runs it under the rig lock.
//
// Every kernel is plain and bounded-safe, and there is no fusion beyond what
// the source kernels already had. That is a measured decision, not laziness:
// §M5 found 2.97 us per asynchronous launch and a synthetic 48-layer token
// running AT the 800 GB/s bandwidth bound as 240 separate launches, so the
// ~35 launches a layer here cost ~1.7 ms of CPU issue time a token, hidden
// behind the GPU. Fusing would trade readability for nothing measurable.
//
// Provenance, kernel by kernel:
//   k_moe_gate_up_iq3s / k_moe_down_iq4nl   m1_native.hip (§M1 candidate 3,
//       385 GB/s at batch 1, oracle cos 1.000000), with the ONE change the
//       file forces: this GGUF stores ffn_gate_exps and ffn_up_exps as
//       SEPARATE tensors, so the "M = 2*n_ff, gate rows first" slab split
//       becomes two base pointers on grid.z.
//   k_moe_gate_up_iq4xs / k_moe_down_q8_0   new, same shape. Layers 2 and 4
//       of this UD quant are IQ4_XS gate/up and Q8_0 down; assuming the file
//       is uniformly IQ3_S/IQ4_NL would silently mis-decode them.
//   k_attn_flash_split / k_attn_combine     m3_attn.hip's v2 attention
//       (§M3: 37.7 us over 2 048 selected tokens at 256k), with the sigmoid
//       output gate REMOVED from the combine -- the reference gates after
//       cb("kqv_out"), and step 2 needs kqv_out as its own oracle point.
//   k_topk_radix_select                     m3_attn.hip's v2 top-k
//       (radix_select_11_7_7_7_1wg), with ratio = 1: the reference selects
//       over TOKENS (`ggml_top_k(expanded, indexer_top_k + r - 1)`), not over
//       blocks, and a block-level select is an optimisation that changes the
//       set at the cut.
//   k_idx_scan                              NEW, not m3's. m3's scan assumed
//       FOUR indexer key heads and no rectification; this model has ONE key
//       head (indexer.k_proj is [2560, 128]) and the reference rectifies each
//       of the four QUERY heads' dot products before summing
//       (qwen4exp.cpp:576-591). The pooled-key cache is therefore 128 bf16 a
//       block, a QUARTER of what §M3 charged the depth term.
//   k_gdn_step                              NEW. m3's k_gdn_step predicts
//       from the UNDECAYED state; ops.cpp:11012 decays first. It is also
//       transposed: ggml stores the state with the KEY axis contiguous
//       (`s_out[j*S_v + i]`).
//   everything else                         new, small, one formula each.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define FK_QUAL __device__ __forceinline__
#define FK_HALF_TO_F32(bits) __half2float(__ushort_as_half((unsigned short)(bits)))

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "decode_quant.h"
#include "decode_backend.h"
#include <stdexcept>
#include "ds4_ops.h"      // DeepSeek-V4 (ds4_gpu.inc, included at the end)
#include "ds4_shapes.h"

#define HIP_CHECK(x) do {                                                      \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__,      \
                     __LINE__, hipGetErrorString(_e));                         \
        std::abort();                                                          \
    }                                                                          \
} while (0)

// Teardown paths: the result genuinely does not matter and hipError_t is
// [[nodiscard]].
#define HIP_IGNORE(x) do { (void) (x); } while (0)

namespace fk {
namespace {

__constant__ uint32_t c_iq3s_grid[512];
__constant__ int8_t   c_kvals_iq4nl[16];

constexpr int WAVE = 32;          // gfx1100 wave32
constexpr int BLK  = 256;         // the generic elementwise / reduction block

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, WAVE);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float * lds) {
    const int t = threadIdx.x;
    lds[t] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) lds[t] += lds[t + s];
        __syncthreads();
    }
    return lds[0];
}

__device__ __forceinline__ float d_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float d_silu(float x)    { return x * d_sigmoid(x); }
// ggml-cpu/unary-ops.cpp:79's shortcut, kept so a large alpha cannot overflow
__device__ __forceinline__ float d_softplus(float x){ return x > 20.0f ? x : logf(1.0f + expf(x)); }

// ============================================================== GEMV ======
//
// GEOMETRY, and why it is not one wave per row any more.
//
// The first cut launched `grid = rows, block = 32`: one wave per output row.
// That is fine for the two wide projections and starves the card on every
// other one, because this model's trunk is mostly SHORT and FAT or SHORT and
// THIN. Counted on a 7900 XTX (96 CUs, 192 SIMD32), per GDN layer:
//
//   tensor            rows    K     bytes   workgroups (old)   waves/SIMD
//   attn_qkv         10240   2560   27.9M      10240              53
//   ssm_out           2560   6144   16.7M       2560              13
//   hc_*_down          320  10240    3.5M        320             1.7   <-- x4
//   ffn_gate_inp       512   2560    5.2M        512             2.7   <-- x1
//   ssm_alpha/beta      48   2560    0.5M         48             0.25  <-- x2
//   hc_*_inject          4  10240    0.16M         4             0.02  <-- x4
//   ffn_gate_inp_shexp   1   2560    0.01M         1             0.005 <-- x1
//
// A memory-bound kernel needs many waves in flight to cover latency; four
// workgroups cannot, however few bytes they move. `hc_attn_inject` and
// `hc_ffn_inject` run 32 times a token between them and each was four waves
// walking 40 KB apiece.
//
// So: a workgroup is now 256 threads = GEMV_WAVES waves, each wave on its own
// ROW, and K is SPLIT across `nsplit` workgroups chosen by the host so that
// `row_groups * nsplit` reaches GEMV_TARGET_WGS. hc_inject goes from 4
// workgroups to 256, ssm_alpha from 48 to 384, ffn_gate_inp from 512 to 2048.
// With nsplit > 1 the partials land in a scratch [nsplit][rows] and
// k_reduce_splits sums them in a FIXED order, so the result stays
// reproducible run to run (an atomicAdd would not, and the oracle depends on
// it).
//
// The Q8_0 inner loop keeps decode_quant.h's lane mapping -- the wave reads
// one block's 32 contiguous qs bytes per step, which is the widest request
// ggml's 34-byte AoS stride allows without an unaligned dword load -- but
// unrolls FOUR blocks instead of two, so four independent requests are in
// flight per wave rather than two.

#define GEMV_WAVES      8                     // waves per workgroup, one row each
#define GEMV_THREADS    (GEMV_WAVES * WAVE)   // 256
#define GEMV_TARGET_WGS 2048
// A GEMV that already fills the card is left alone: splitting attn_qkv
// (1 280 row-groups) would only add a reduce launch.
#define GEMV_MIN_WGS    512

// Block range [b0, b1) this split owns, for a K split on block boundaries.
__device__ __forceinline__ void split_range(int n_units, int nsplit, int split,
                                            int * u0, int * u1) {
    const int per = (n_units + nsplit - 1) / nsplit;
    *u0 = split * per;
    *u1 = min(n_units, *u0 + per);
}

// ---------------------------------------------------- batched GEMV -------
//
// Up to GEMV_BATCH_MAX matrices over one activation vector, each with its own
// format, output pointer and epilogue; see decode_backend.h for which
// projections share an input and why that is the only way to cut the launch
// count. The grid is (row_groups over the CONCATENATED rows, nsplit).
//
// `stage_x` optionally copies this workgroup's slice of the activation into
// LDS so the inner loop reads it from there instead of L1. Whether that helps
// is a question for the profile, not for an assumption -- the kernel is
// weight-bandwidth-bound and x is small enough to live in L1 anyway -- so it
// is a RUNTIME switch (`--gemv-lds 0|1`) and both arms can be measured with
// one binary. It is skipped whenever the slice would not fit in
// GEMV_LDS_MAX_FLOATS, because a 40 KB workgroup allocation would cost more
// occupancy than the L1 traffic is worth.
//
// On 16-byte loads: two consecutive Q8_0 blocks are 68 bytes and `qs` sits at
// offset 34b+2, so a block's payload is 4-byte aligned only for even b and
// NEVER 16-byte aligned. A dwordx4 load from such an address is not something
// this file will do on the strength of a guess; the widest safe request is
// the 32 contiguous bytes one wave already pulls per block. What IS tunable
// is how many of those are in flight, so the unroll depth is a compile-time
// knob (GEMV_UNROLL, default 4) that the profile can move.
// The largest batch is the QSA projection's five (wq, wk, wv, indexer q/k).
// The descriptor travels as a BY-VALUE kernel argument, so its size is host
// work on every GEMV launch: eight slots cost ~404 bytes a launch, five ~260.
#define GEMV_BATCH_MAX     5
#define GEMV_LDS_MAX_FLOATS 4096
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif
// How many GEMV_UNROLL groups of blocks a decode wave has in flight at once.
// It is NOT the accumulator count and must never become it: see the burst loop
// in k_gemm_batch. Instantiated for 1, 2 and 4; --gemv-burst picks one.
#ifndef GEMV_BURST
#define GEMV_BURST 4
#endif
// A GEMV launch reading fewer weight bytes than this is charged to
// prof_gemv_small_*. On the decode token that set is exactly hc_down+inject,
// hc_up, ffn_router_shexp and shexp_down -- the four groups measured at
// 128-245 GB/s in record SS-L0-STEP2, against 434-497 for the rest.
#define GEMV_SMALL_BYTES ((size_t) 10 * 1024 * 1024)

struct GemvDesc {
    const void * base;
    float *      out;
    size_t       row_stride;   // bytes for a quantised row, elements otherwise
    int          type;
    int          rows;
    int          row_off;      // first row of this matrix in the concatenation
    int          epi;
    float        arg;
};
struct GemvBatchD {
    GemvDesc d[GEMV_BATCH_MAX];
    int n;
    int K;
    int total_rows;
    int x_mode;
    int has_q6k;
};

__device__ __forceinline__ float apply_epi(float v, int epi, float arg) {
    switch (epi) {
        case GE_SIGMOID:    return d_sigmoid(v);
        case GE_SCALE_SILU: return d_silu(v * arg);
        default:            return v;
    }
}

// Which matrix owns a concatenated row. n <= 8, so a linear scan is cheaper
// than any lookup table.
__device__ __forceinline__ int desc_of(const GemvBatchD & b, int row) {
    int j = b.n - 1;
#pragma unroll
    for (int i = GEMV_BATCH_MAX - 1; i >= 0; --i)
        if (i < b.n && row < b.d[i].row_off + b.d[i].rows && row >= b.d[i].row_off) j = i;
    return j;
}

// THE ONE GEMV/GEMM PATH (design 9.4): T activation columns per launch, a
// wave loading each weight block ONCE and dotting it against GEMM_TILE of
// them. That reuse is the whole point of batching a prompt -- the trunk is
// ~5 GB a token, so a tile of 8 reads it an eighth as often.
//
// Activations are token-major (x[t*K + i]), so a row is contiguous and the
// per-token loads inside the tile stay coalesced across the wave. Outputs are
// token-major too: out[t*rows + row].
//
// `t0` is the first token of this tile; grid.z selects the tile. A tile short
// of GEMM_TILE at the end of a chunk is handled by the `nt` bound, not by
// padding, so no garbage column is ever read.
//
// WHY THERE IS NO SEPARATE T=1 KERNEL, and what that costs. This kernel
// replaced k_gemv_batch, whose one-column Q8_0 loop kept FOUR accumulators
// (a0..a3, four blocks in flight per wave) and closed with (a0+a1)+(a2+a3).
// Keeping a second kernel for T = 1 would mean the decode path and the
// prefill path could disagree in the last bit and the `--chunk 6` vs
// `--chunk 1` oracle would be testing two different things. So the tile form
// carries the SAME per-column accumulator structure -- four for Q8_0, two for
// Q6_K, one for BF16/F32, plain `+` where the old kernel had `+` and fmaf
// where it had fmaf -- and every column of every tile therefore lands on the
// bit pattern the old one-column kernel produced. The split-K count is chosen
// from the MATRIX alone and never from T, for the same reason: nsplit changes
// the summation order.
//
// `acc` is indexed only by fully unrolled constants, or it would be scratch
// rather than registers.
//
// TILE IS A TEMPLATE PARAMETER, and T == 1 instantiates TILE = 1. It has to
// be: measured 2026-09-22, the TILE = 8 kernel run at T = 1 costs the decode
// token 26.98 ms against the 22.70 ms of the one-column kernel it replaced.
// The cause is not the arithmetic -- at nt = 1 seven of the eight columns are
// predicated off -- it is that `float acc[8][4]` is live whatever nt is, so
// the kernel asks for 135 VGPRs and gets 10 waves/SIMD where the one-column
// kernel got 16. TILE = 1 gives back `acc[1][4]`, and it is the SAME SOURCE:
// same accumulator structure, same operation order, same `+` and `fmaf` in
// the same places, so T = 1 through TILE = 1 and T = 6 through TILE = 8 still
// land on the same bits per column. The LDS staging rule follows TILE too,
// so a decode token stages exactly what it staged before.
#define GEMM_TILE 8
#define GEMM_ACC  4                       // the Q8_0 unroll, and the array's width

// The activation of tile column j at global element index i. With `stage_x`
// it comes from LDS (already silu-multiplied); without, the GX_SILU_MUL
// activation is formed inline so both arms compute the same value.
#define GEMM_X(j, i)                                                                     \
    (stage_x ? sx[(size_t)(j) * slice + ((i) - e0)]                                      \
             : (b.x_mode == GX_SILU_MUL                                                  \
                ? d_silu(x[(size_t)(t0 + (j)) * b.K + (i)]) * x2[(size_t)(t0 + (j)) * b.K + (i)] \
                : x[(size_t)(t0 + (j)) * b.K + (i)]))

//
// THREE THINGS THE DECODE TOKEN (TILE == 1) DOES AND THE PROMPT CHUNK DOES NOT.
// All three are compile-time branches on TILE, so the T > 1 path below is the
// same instruction stream it was, and none of them touches the arithmetic:
//
//  (a) THE SPLIT RIDES ON grid.x, NOT grid.y. Workgroups are dispatched with x
//      fastest, so with the split on y the 50 workgroups that between them read
//      one row of hc_down were dispatched 41 apart and the card walked the
//      tensor in 238-byte pieces at 50 scattered offsets. Put the split on x
//      and consecutive workgroups sweep the SAME eight rows along K, in order.
//      That is the whole of the difference between the measured 128 GB/s of
//      hc_down+inject (nsplit 50, 238 B a wave) and the 461-497 GB/s of the
//      groups that run unsplit (2 720-6 528 B a wave) -- achieved bandwidth on
//      this card tracks the contiguous run a wave reads, and nothing else in
//      that table explains it. `split` and `rgrp` are the only readers of
//      blockIdx, so this is an ADDRESS change and not an arithmetic one:
//      split s still owns block range s of split_range and still writes
//      partial s, so the reduce sums the same numbers in the same order.
//
//  (b) THE SPLIT REDUCE HAPPENS IN THIS KERNEL. `cnt` is one counter per output
//      row; the wave that finds itself last for its row sums the nsplit
//      partials in s order -- literally k_reduce_splits_gemm's loop -- and
//      writes the destination. Same order, same bits, one launch instead of
//      two. On the decode token hc_down+inject is the ONLY split GEMV, so this
//      is 32 launches a card a token.
//
//  (c) THE hc COLLAPSE RIDES ON hc_up'S EPILOGUE. See the `collapse` block at
//      the bottom.
template <int TILE, int BURST>
__global__ __launch_bounds__(GEMV_THREADS)
void k_gemm_batch(GemvBatchD b, const float * __restrict__ x,
                  const float * __restrict__ x2, int nsplit,
                  float * __restrict__ part, int stage_x, int T,
                  int * __restrict__ cnt, float * __restrict__ collapse_out,
                  const float * __restrict__ collapse_xn) {
    extern __shared__ float sx[];
    // One float a wave, written only on the fused-collapse path.
    __shared__ float sgate[GEMV_WAVES];

    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int split = (TILE == 1) ? (int) blockIdx.x : (int) blockIdx.y;
    const int rgrp  = (TILE == 1) ? (int) blockIdx.y : (int) blockIdx.x;
    // hc_up's rows are HC streams of N_EMBD and hc_collapse needs all HC of
    // them for one embedding index. A workgroup of GEMV_WAVES == 2*HC waves
    // therefore takes TWO indices and every stream of each, instead of eight
    // consecutive rows. Same rows, same per-row dot, different wave -> row map.
    const bool collapse = (TILE == 1) && (collapse_out != nullptr);
    const int grow = collapse
                   ? ((wave & (HC - 1)) * N_EMBD + rgrp * (GEMV_WAVES / HC) + (wave / HC))
                   : (rgrp * GEMV_WAVES + wave);
    const int t0   = blockIdx.z * TILE;
    const int nt   = min(TILE, T - t0);
    if (nt <= 0) return;

    // Element range of this split, in units of 32 so every lane stays in
    // step. Q6_K super-blocks are 256 weights, so a batch containing one
    // splits on 256-element boundaries; everything else splits on 32.
    const int gran = (b.has_q6k ? 256 : WAVE);
    int c0, c1;
    split_range((b.K + gran - 1) / gran, nsplit, split, &c0, &c1);
    const int e0 = c0 * gran;
    const int e1 = min(b.K, c1 * gran);
    const int slice = e1 - e0;

    // Stage this tile's slice of every column: sx[j*slice + i]. Shared by all
    // GEMV_WAVES rows of the workgroup, which is the second reuse.
    if (stage_x) {
        for (int j = 0; j < nt; ++j) {
            const float * src  = x  + (size_t)(t0 + j) * b.K + e0;
            const float * src2 = x2 ? x2 + (size_t)(t0 + j) * b.K + e0 : nullptr;
            float * dst = sx + (size_t) j * slice;
            for (int i = threadIdx.x; i < slice; i += GEMV_THREADS)
                dst[i] = (b.x_mode == GX_SILU_MUL) ? d_silu(src[i]) * src2[i] : src[i];
        }
        __syncthreads();
    }

    if (grow >= b.total_rows) return;
    const int  di  = desc_of(b, grow);
    const GemvDesc d = b.d[di];
    const int  row = grow - d.row_off;

    float acc[TILE][GEMM_ACC];
#pragma unroll
    for (int j = 0; j < TILE; ++j)
#pragma unroll
        for (int u = 0; u < GEMM_ACC; ++u) acc[j][u] = 0.0f;

    if (d.type == FK_Q_Q8_0) {
        // the split is on 32-element boundaries, which is exactly a Q8_0
        // block; one block's scale*quant is hoisted out of the column loop
        const unsigned char * rp = (const unsigned char *) d.base + (size_t) row * d.row_stride;
        int bl = e0 >> 5;
        const int bend = e1 >> 5;
        // MORE LOADS IN FLIGHT, SAME SUMMATION ORDER. The loop below issues
        // GEMV_UNROLL weight loads and consumes them; the compiler pipelines
        // about one extra iteration ahead (the baseline ISA waits on vmcnt(6)
        // of eight outstanding loads), so a wave whose row is only ten blocks
        // long -- hc_up at K = 320, shexp_down at K = 640 after its split --
        // never gets more than two rounds of memory latency in flight and runs
        // at 141-199 GB/s against the 461-497 of the long rows.
        //
        // This loop issues BURST*GEMV_UNROLL of them BEFORE the first multiply.
        // It is not a deeper unroll: the accumulator count stays GEMV_UNROLL
        // and block g still lands in acc[g % GEMV_UNROLL], in increasing g --
        // which is exactly what the loop below does across its iterations. It
        // consumes a multiple of GEMV_UNROLL blocks, so the loops after it see
        // the same alignment and the same tail. Bit-identical by construction,
        // and checked the way the down-gather divergence had to be checked
        // (PREFILL.md section 8): the -ffp-contract=fast decision is read out
        // of the ISA, not assumed -- `acc += w*x` must stay v_fmac/v_fma here
        // as it is at BURST 1, or the bits move even though the order did not.
        if constexpr (TILE == 1 && BURST > 1) {
            constexpr int NB = GEMV_UNROLL * BURST;
            for (; bl + NB - 1 < bend; bl += NB) {
                float w[NB], xv[NB];
#pragma unroll
                for (int u = 0; u < NB; ++u) {
                    const unsigned char * bp = rp + (size_t)(bl + u) * FK_Q8_0_BLOCK_BYTES;
                    w[u] = fk_q8_0_block_d(bp) *
                           (float)((const signed char *)(bp + FK_Q8_0_OFF_QS))[lane];
                }
#pragma unroll
                for (int u = 0; u < NB; ++u) xv[u] = GEMM_X(0, ((bl + u) << 5) + lane);
#pragma unroll
                for (int u = 0; u < NB; ++u)
                    acc[0][u & (GEMV_UNROLL - 1)] += w[u] * xv[u];
            }
        }
        for (; bl + GEMV_UNROLL - 1 < bend; bl += GEMV_UNROLL) {
            float w[GEMV_UNROLL];
#pragma unroll
            for (int u = 0; u < GEMV_UNROLL; ++u) {
                const unsigned char * bp = rp + (size_t)(bl + u) * FK_Q8_0_BLOCK_BYTES;
                w[u] = fk_q8_0_block_d(bp) *
                       (float)((const signed char *)(bp + FK_Q8_0_OFF_QS))[lane];
            }
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= nt) continue;
#pragma unroll
                for (int u = 0; u < GEMV_UNROLL; ++u)
                    acc[j][u] += w[u] * GEMM_X(j, ((bl + u) << 5) + lane);
            }
        }
        for (; bl < bend; ++bl) {
            const unsigned char * bp = rp + (size_t) bl * FK_Q8_0_BLOCK_BYTES;
            const float w = fk_q8_0_block_d(bp) *
                            (float)((const signed char *)(bp + FK_Q8_0_OFF_QS))[lane];
#pragma unroll
            for (int j = 0; j < TILE; ++j)
                if (j < nt) acc[j][0] += w * GEMM_X(j, (bl << 5) + lane);
        }
    } else if (d.type == FK_Q_Q6_K) {
        // Q6_K decodes a whole eight-weight lane run at once, so there is no
        // single weight to hoist: the primitive is called per column, with
        // the old kernel's two accumulators.
        const unsigned char * rp = (const unsigned char *) d.base + (size_t) row * d.row_stride;
        const int xoff = lane << 3;
        int bl = e0 >> 8;
        const int bend = e1 >> 8;
        for (; bl + 1 < bend; bl += 2) {
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= nt) continue;
                const float * xt = stage_x ? sx + (size_t) j * slice - e0
                                           : x + (size_t)(t0 + j) * b.K;
                acc[j][0] += fk_q6k_block_dot(rp + (size_t)(bl+0) * FK_Q6K_BLOCK_BYTES,
                                              xt + ((bl+0) << 8) + xoff, lane);
                acc[j][1] += fk_q6k_block_dot(rp + (size_t)(bl+1) * FK_Q6K_BLOCK_BYTES,
                                              xt + ((bl+1) << 8) + xoff, lane);
            }
        }
        for (; bl < bend; ++bl) {
#pragma unroll
            for (int j = 0; j < TILE; ++j) {
                if (j >= nt) continue;
                const float * xt = stage_x ? sx + (size_t) j * slice - e0
                                           : x + (size_t)(t0 + j) * b.K;
                acc[j][0] += fk_q6k_block_dot(rp + (size_t) bl * FK_Q6K_BLOCK_BYTES,
                                              xt + (bl << 8) + xoff, lane);
            }
        }
    } else if (d.type == FK_Q_BF16) {
        const unsigned short * rp = (const unsigned short *) d.base + (size_t) row * d.row_stride;
        for (int i = e0 + lane; i < e1; i += WAVE) {
            const float w = fk_bf16_to_f32(rp[i]);
#pragma unroll
            for (int j = 0; j < TILE; ++j)
                if (j < nt) acc[j][0] = fmaf(w, GEMM_X(j, i), acc[j][0]);
        }
    } else {
        const float * rp = (const float *) d.base + (size_t) row * d.row_stride;
        for (int i = e0 + lane; i < e1; i += WAVE) {
            const float w = rp[i];
#pragma unroll
            for (int j = 0; j < TILE; ++j)
                if (j < nt) acc[j][0] = fmaf(w, GEMM_X(j, i), acc[j][0]);
        }
    }

    // The closing reassociation is the old one-column kernel's, per format:
    // (a0+a1)+(a2+a3) for Q8_0, a0+a1 for Q6_K, a0 alone otherwise -- the
    // unused slots are zero, so one expression covers all three.
    float my_gate = 0.0f;                  // the collapse path's own epilogue value
#pragma unroll
    for (int j = 0; j < TILE; ++j) {
        if (j >= nt) break;
        const float sum = wave_sum((acc[j][0] + acc[j][1]) + (acc[j][2] + acc[j][3]));
        if (lane != 0) continue;
        if (nsplit > 1) {
            float * pp = part + (((size_t) split * T + (t0 + j)) * b.total_rows + grow);
            // The wave that closes this row reads it back, so the store must
            // leave the write-through L0 behind it; the fence below orders it.
            if (cnt) *(volatile float *) pp = sum;
            else     *pp = sum;
        } else {
            const float v = apply_epi(sum, d.epi, d.arg);
            d.out[(size_t)(t0 + j) * d.rows + row] = v;
            my_gate = v;
        }
    }

    // ---- (b) the split reduce, without a second launch -------------------
    // Each row's nsplit workgroups arrive in any order; the last one to
    // increment this row's counter has, by the fence pairing, every partial
    // visible and sums them s = 0 .. nsplit-1 -- k_reduce_splits_gemm's loop,
    // copied rather than approximated, because the oracle is bit-for-bit. The
    // counter goes back to 0 so the next launch finds it as this one did.
    if (TILE == 1 && nsplit > 1 && cnt) {
        if (lane == 0) {
            __threadfence();
            if (atomicAdd(&cnt[grow], 1) == nsplit - 1) {
                __threadfence();
                const volatile float * p = (const volatile float *) part;
                float a = 0.0f;
                for (int s = 0; s < nsplit; ++s)
                    a += p[((size_t) s * T + t0) * b.total_rows + grow];
                d.out[(size_t) t0 * d.rows + row] = apply_epi(a, d.epi, d.arg);
                cnt[grow] = 0;
            }
        }
        return;
    }

    // ---- (c) build_hc_mix's collapse, on the gate GEMV's own workgroup ----
    //   mixed[i] = mean over the HC streams of xn[c][i] * sigmoid(gate[c][i])
    // The wave -> row map above put all HC streams of two indices in this
    // workgroup, so the gate values it just produced are all the collapse
    // needs. Same expression, same order (c = 0 first, then 1..HC-1), same
    // scale, so it is k_hc_collapse's arithmetic without k_hc_collapse's
    // launch -- and `gate` is still written, so hc_gate stays an oracle tap.
    if (collapse) {
        if (lane == 0) sgate[wave] = my_gate;
        __syncthreads();
        constexpr int PER = GEMV_WAVES / HC;
        if ((int) threadIdx.x < PER) {
            const int i = rgrp * PER + (int) threadIdx.x;
            if (i < N_EMBD) {
                const int w0 = (int) threadIdx.x * HC;        // wave == sub*HC + c
                float a = collapse_xn[i] * sgate[w0];
#pragma unroll
                for (int c = 1; c < HC; ++c)
                    a += collapse_xn[(size_t) c * N_EMBD + i] * sgate[w0 + c];
                collapse_out[i] = a * (1.0f / (float) HC);
            }
        }
    }
}

// =============================================== LDS-tiled trunk GEMM =====
//
// WHY A SECOND GEMM EXISTS AT ALL, when PREFILL.md section 9 says there is
// exactly one. Because the profile said so, and the invariant it is protecting
// is "decode and prefill cannot drift", which a knob that is OFF by default
// does not break: at --gemm-lds 0 nothing below is reachable.
//
// Measured (record §L0-PREFILL, `g.txt`, 8 192 tokens at chunk 256, per prompt
// token per card): the trunk GEMM is 1.69-1.76 ms. Its weights are ~1.7 GB a
// card and k_gemm_batch<8> reads all of them once per TILE of 8 token columns,
// so at T = 256 that is 32 passes = 54 GB a chunk = 0.27 ms a token at the
// 800 GB/s bound. It takes 1.7. So it is neither bandwidth- nor compute-bound:
// at 135 VGPR it gets 10 waves/SIMD, and every wave re-decodes the same Q8_0
// block for each group of 8 columns.
//
// This kernel attacks both halves of that:
//
//   * a workgroup owns a 64 x 64 tile (rows x tokens) and stages K in steps of
//     32 through LDS, so a weight element is read from VRAM and decoded ONCE
//     per 64 columns instead of once per 8: 4 passes at T = 256, 6.8 GB a
//     chunk, 0.033 ms a token at the bound.
//   * the activation tile is re-read once per 64 ROWS, which looks like a lot
//     (10 240 rows / 64 = 160 passes over 2.6 MB) and is not: 2.6 MB lives in
//     the 96 MB Infinity Cache, so those passes never reach VRAM.
//
// NUMERICS, and why this is a knob. Each thread accumulates its own (row,
// column) over the WHOLE of K in one f32 register, in K order. k_gemm_batch
// accumulates across a wave's 32 lanes and closes with a butterfly, and the
// decode token does the same. Those are different summation orders, so the
// results differ in the last bits -- legitimately, the way F7's reordered
// attention kernel did -- and the oracle's meaning is preserved by keeping
// this OFF by default and measuring the divergence with it on.
//
// LDS layout: `ws[k][row]` and `as[k][col]`, stride 68 floats. k is the
// staging axis and (row, col) the reading axis, so the inner loop reads four
// CONSECUTIVE floats per operand -- one 16-byte LDS request, and the 16
// distinct columns of a wave cover 256 contiguous bytes, which is
// conflict-free. The 4-element padding keeps the float4s 16-byte aligned.
// 2 x 32 x 68 floats = 17.4 KB, so three workgroups fit a CU's 64 KB.

#define GEMM_LDS_R       64     // output rows a workgroup owns
#define GEMM_LDS_C       64     // activation columns (tokens) a workgroup owns
#define GEMM_LDS_KB      32     // K staged per step == exactly one Q8_0 block
#define GEMM_LDS_THREADS 256
#define GEMM_LDS_RT      4      // rows a thread owns    (64 / 16)
#define GEMM_LDS_CT      4      // columns a thread owns (64 / 16)
#define GEMM_LDS_STRIDE  (GEMM_LDS_R + 4)   // == GEMM_LDS_C + 4; 16 B aligned
// Below this many rows the old split-K path is better (a 64-row tile wastes
// most of its lanes on a 4-row or 48-row matrix) AND it is bit-identical, so
// small matrices keep it whatever the knob says.
#define GEMM_LDS_MIN_ROWS 256

// The row's weight element at global index `e`, decoded to f32. `e` is in
// [0, K) and, for Q8_0, kbase is a multiple of 32 so `e >> 5` is the block and
// `e & 31` the lane -- the SAME decode expression k_gemm_batch uses.
__device__ __forceinline__ float gemm_lds_w(const GemvDesc & d, int row, int e) {
    if (d.type == FK_Q_Q8_0) {
        const unsigned char * bp = (const unsigned char *) d.base +
                                   (size_t) row * d.row_stride +
                                   (size_t)(e >> 5) * FK_Q8_0_BLOCK_BYTES;
        return fk_q8_0_block_d(bp) *
               (float)((const signed char *)(bp + FK_Q8_0_OFF_QS))[e & 31];
    }
    if (d.type == FK_Q_BF16) {
        const unsigned short * rp = (const unsigned short *) d.base + (size_t) row * d.row_stride;
        return fk_bf16_to_f32(rp[e]);
    }
    const float * rp = (const float *) d.base + (size_t) row * d.row_stride;
    return rp[e];
}

// The activation of column `t` at element `e`, with the GX_SILU_MUL form
// written out exactly as GEMM_X does it so both kernels see the same value.
__device__ __forceinline__ float gemm_lds_x(const GemvBatchD & b, const float * x,
                                            const float * x2, int t, int e) {
    const float v = x[(size_t) t * b.K + e];
    return (b.x_mode == GX_SILU_MUL) ? d_silu(v) * x2[(size_t) t * b.K + e] : v;
}

__global__ __launch_bounds__(GEMM_LDS_THREADS)
void k_gemm_lds(GemvBatchD b, const float * __restrict__ x,
                const float * __restrict__ x2, int T) {
    __shared__ __align__(16) float ws[GEMM_LDS_KB * GEMM_LDS_STRIDE];
    __shared__ __align__(16) float as[GEMM_LDS_KB * GEMM_LDS_STRIDE];

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * GEMM_LDS_R;
    const int col0 = blockIdx.y * GEMM_LDS_C;
    if (row0 >= b.total_rows || col0 >= T) return;

    const int tr = tid / 16;                 // 16 row groups of GEMM_LDS_RT
    const int tc = tid % 16;                 // 16 column groups of GEMM_LDS_CT
    const int sr = tid / 32;                 // staging: 8 rows/cols at a time,
    const int sl = tid % 32;                 // one 32-lane wave each

    float acc[GEMM_LDS_RT][GEMM_LDS_CT];
#pragma unroll
    for (int i = 0; i < GEMM_LDS_RT; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_LDS_CT; ++j) acc[i][j] = 0.0f;

    for (int kbase = 0; kbase < b.K; kbase += GEMM_LDS_KB) {
        __syncthreads();                     // the previous step's reads are done
        const int e = kbase + sl;
        const bool in_k = (e < b.K);
        // Weights: one wave per row, 32 contiguous elements -- for Q8_0 that
        // is one block's 32 qs bytes in one request, the widest the 34-byte
        // AoS stride allows (see k_gemm_batch's header).
        for (int rr = sr; rr < GEMM_LDS_R; rr += GEMM_LDS_THREADS / 32) {
            const int grow = row0 + rr;
            float w = 0.0f;
            if (grow < b.total_rows && in_k) {
                const GemvDesc & d = b.d[desc_of(b, grow)];
                w = gemm_lds_w(d, grow - d.row_off, e);
            }
            ws[sl * GEMM_LDS_STRIDE + rr] = w;
        }
        for (int cc = sr; cc < GEMM_LDS_C; cc += GEMM_LDS_THREADS / 32) {
            const int t = col0 + cc;
            float a = 0.0f;
            if (t < T && in_k) a = gemm_lds_x(b, x, x2, t, e);
            as[sl * GEMM_LDS_STRIDE + cc] = a;
        }
        __syncthreads();

#pragma unroll 4
        for (int k = 0; k < GEMM_LDS_KB; ++k) {
            const float * wp = ws + k * GEMM_LDS_STRIDE + tr * GEMM_LDS_RT;
            const float * ap = as + k * GEMM_LDS_STRIDE + tc * GEMM_LDS_CT;
            float wv[GEMM_LDS_RT], av[GEMM_LDS_CT];
#pragma unroll
            for (int i = 0; i < GEMM_LDS_RT; ++i) wv[i] = wp[i];
#pragma unroll
            for (int j = 0; j < GEMM_LDS_CT; ++j) av[j] = ap[j];
#pragma unroll
            for (int i = 0; i < GEMM_LDS_RT; ++i)
#pragma unroll
                for (int j = 0; j < GEMM_LDS_CT; ++j)
                    acc[i][j] = fmaf(wv[i], av[j], acc[i][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < GEMM_LDS_RT; ++i) {
        const int grow = row0 + tr * GEMM_LDS_RT + i;
        if (grow >= b.total_rows) break;
        const GemvDesc d = b.d[desc_of(b, grow)];
        const int row = grow - d.row_off;
#pragma unroll
        for (int j = 0; j < GEMM_LDS_CT; ++j) {
            const int t = col0 + tc * GEMM_LDS_CT + j;
            if (t >= T) break;
            d.out[(size_t) t * d.rows + row] = apply_epi(acc[i][j], d.epi, d.arg);
        }
    }
}

// ------------------------------- the int8 arm (--gemm-lds 2) --------------
//
// RDNA3 HAS a packed int8 dot product and this toolchain DOES expose it for
// gfx1100 -- just not under the name the Makefile's note tried:
// `__builtin_amdgcn_sdot4` wants the `dot1-insts` feature and is rejected,
// while `__builtin_amdgcn_sudot4(signed_a, a, signed_b, b, acc, clamp)`
// compiles and emits `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` (verified by reading
// the ISA out of hipcc -S, not by hoping).
//
// The WEIGHTS are already int8: Q8_0 is 32 int8 and one f16 scale, and this
// kernel's K step is exactly that block, so the weight side is EXACT -- no
// requantisation at all. The ACTIVATION tile is quantised per column per
// 32-block, which is ggml's own Q8_1 recipe and is what llama.cpp's CPU
// mul_mat does to every activation it dots against a quantised weight
// (decode_backend.h set_quant_act). It is a real accuracy change, not a
// reassociation, which is why it is a SEPARATE knob value from mode 1 and why
// the gate has to report its divergence separately.
//
// Per 32-block: eight v_dot4 into an int accumulator, then one f32 multiply
// by (weight scale * activation scale) into the f32 accumulator. So the f32
// accumulation granularity is the same block the T = 1 kernel uses.
//
// LDS: int8 tiles at stride GEMM_LDS_KB+4 = 36 bytes (a multiple of 4, so the
// packed int loads are aligned) plus the two scale vectors -- 5.1 KB, which is
// why this arm can keep far more workgroups in flight than the f32 one.
#define GEMM_LDS_I8_STRIDE 36

__global__ __launch_bounds__(GEMM_LDS_THREADS)
void k_gemm_lds_i8(GemvBatchD b, const float * __restrict__ x,
                   const float * __restrict__ x2, int T) {
    // DECLARED as int and written through a char view, not the other way
    // round: v_dot4 wants four packed bytes as one int, so the int reads are
    // the ones that must be the declared type, and a char* may alias anything.
    // The stride is 9 ints = 36 bytes, so every packed read is 4-aligned.
    __shared__ int         wq_i[GEMM_LDS_R * (GEMM_LDS_I8_STRIDE / 4)];
    __shared__ int         aq_i[GEMM_LDS_C * (GEMM_LDS_I8_STRIDE / 4)];
    __shared__ float       wd[GEMM_LDS_R];
    __shared__ float       ad[GEMM_LDS_C];
    signed char * const wq = (signed char *) wq_i;
    signed char * const aq = (signed char *) aq_i;

    const int tid  = threadIdx.x;
    const int row0 = blockIdx.x * GEMM_LDS_R;
    const int col0 = blockIdx.y * GEMM_LDS_C;
    if (row0 >= b.total_rows || col0 >= T) return;

    const int tr = tid / 16, tc = tid % 16;
    const int sr = tid / 32, sl = tid % 32;

    float acc[GEMM_LDS_RT][GEMM_LDS_CT];
#pragma unroll
    for (int i = 0; i < GEMM_LDS_RT; ++i)
#pragma unroll
        for (int j = 0; j < GEMM_LDS_CT; ++j) acc[i][j] = 0.0f;

    for (int kbase = 0; kbase < b.K; kbase += GEMM_LDS_KB) {
        __syncthreads();
        const int e = kbase + sl;
        const bool in_k = (e < b.K);
        for (int rr = sr; rr < GEMM_LDS_R; rr += GEMM_LDS_THREADS / 32) {
            const int grow = row0 + rr;
            signed char q = 0;
            float d0 = 0.0f;
            if (grow < b.total_rows && in_k) {
                const GemvDesc & d = b.d[desc_of(b, grow)];
                const unsigned char * bp = (const unsigned char *) d.base +
                                           (size_t)(grow - d.row_off) * d.row_stride +
                                           (size_t)(e >> 5) * FK_Q8_0_BLOCK_BYTES;
                q  = ((const signed char *)(bp + FK_Q8_0_OFF_QS))[e & 31];
                d0 = fk_q8_0_block_d(bp);
            }
            wq[rr * GEMM_LDS_I8_STRIDE + sl] = q;
            if (sl == 0) wd[rr] = d0;
        }
        for (int cc = sr; cc < GEMM_LDS_C; cc += GEMM_LDS_THREADS / 32) {
            const int t = col0 + cc;
            float a = 0.0f;
            if (t < T && in_k) a = gemm_lds_x(b, x, x2, t, e);
            // ggml's Q8_1 amax over exactly these 32 values. The 32 lanes
            // staging this column ARE one wave, so the butterfly is the whole
            // block's max and costs no LDS.
            float m = fabsf(a);
#pragma unroll
            for (int off = WAVE / 2; off > 0; off >>= 1)
                m = fmaxf(m, __shfl_xor(m, off, WAVE));
            const float inv = (m > 0.0f) ? 127.0f / m : 0.0f;
            aq[cc * GEMM_LDS_I8_STRIDE + sl] = (signed char) __float2int_rn(a * inv);
            if (sl == 0) ad[cc] = m * (1.0f / 127.0f);
        }
        __syncthreads();

        int iacc[GEMM_LDS_RT][GEMM_LDS_CT];
#pragma unroll
        for (int i = 0; i < GEMM_LDS_RT; ++i)
#pragma unroll
            for (int j = 0; j < GEMM_LDS_CT; ++j) iacc[i][j] = 0;
        const int wstep = GEMM_LDS_I8_STRIDE / 4;      // 9 ints a row
        const int * wi4 = wq_i + (tr * GEMM_LDS_RT) * wstep;
        const int * ai4 = aq_i + (tc * GEMM_LDS_CT) * wstep;
#pragma unroll
        for (int kg = 0; kg < GEMM_LDS_KB / 4; ++kg) {
            int wv[GEMM_LDS_RT], av[GEMM_LDS_CT];
#pragma unroll
            for (int i = 0; i < GEMM_LDS_RT; ++i) wv[i] = wi4[i * wstep + kg];
#pragma unroll
            for (int j = 0; j < GEMM_LDS_CT; ++j) av[j] = ai4[j * wstep + kg];
#pragma unroll
            for (int i = 0; i < GEMM_LDS_RT; ++i)
#pragma unroll
                for (int j = 0; j < GEMM_LDS_CT; ++j)
                    iacc[i][j] = __builtin_amdgcn_sudot4(true, wv[i], true, av[j],
                                                         iacc[i][j], false);
        }
#pragma unroll
        for (int i = 0; i < GEMM_LDS_RT; ++i) {
            const float dwi = wd[tr * GEMM_LDS_RT + i];
#pragma unroll
            for (int j = 0; j < GEMM_LDS_CT; ++j)
                acc[i][j] = fmaf((float) iacc[i][j] * dwi, ad[tc * GEMM_LDS_CT + j], acc[i][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < GEMM_LDS_RT; ++i) {
        const int grow = row0 + tr * GEMM_LDS_RT + i;
        if (grow >= b.total_rows) break;
        const GemvDesc d = b.d[desc_of(b, grow)];
        const int row = grow - d.row_off;
#pragma unroll
        for (int j = 0; j < GEMM_LDS_CT; ++j) {
            const int t = col0 + tc * GEMM_LDS_CT + j;
            if (t >= T) break;
            d.out[(size_t) t * d.rows + row] = apply_epi(acc[i][j], d.epi, d.arg);
        }
    }
}

// Fixed-order sum of the split partials, then each matrix's epilogue and its
// own destination. Reproducible, unlike an atomicAdd -- the oracle needs it.
__global__ void k_reduce_splits_gemm(GemvBatchD b, const float * __restrict__ part,
                                     int nsplit, int T) {
    const int grow = blockIdx.x * blockDim.x + threadIdx.x;
    const int t    = blockIdx.y;
    if (grow >= b.total_rows || t >= T) return;
    float a = 0.0f;
    for (int s = 0; s < nsplit; ++s)
        a += part[((size_t) s * T + t) * b.total_rows + grow];
    const int di = desc_of(b, grow);
    b.d[di].out[(size_t) t * b.d[di].rows + grow - b.d[di].row_off] =
        apply_epi(a, b.d[di].epi, b.d[di].arg);
}

// ======================================================= routed experts ===
//
// grid = (rows, K_TOP*T, 2): z == 0 is gate, z == 1 is up. One wave per row.
// Assignment (token t, rank k) is column t*K_TOP + k of `ids` and of the
// outputs, so the per-token work is the decode kernel with a longer grid.y.
// Item 5 of design 9.4 then sorts those columns by expert so a weight row is
// read once for all the tokens that chose it; until then each assignment
// reads its own.

__global__ __launch_bounds__(WAVE, 16)
void k_moe_gate_up_iq3s(const unsigned char * __restrict__ gate_base,
                        const unsigned char * __restrict__ up_base,
                        size_t expert_stride, size_t row_bytes,
                        const int * __restrict__ ids, const float * __restrict__ x,
                        float * __restrict__ y_gate, float * __restrict__ y_up,
                        int rows, int K, int k_top, int T) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int col = blockIdx.y;
    if (col >= k_top * T) return;
    if (K & (IQ3S_BLOCK_WEIGHTS - 1)) return;            // bounded-safe
    const int which = blockIdx.z;
    const int tid   = threadIdx.x;
    x += (size_t)(col / k_top) * K;

    const unsigned char * A = (which ? up_base : gate_base)
                            + (size_t) ids[col] * expert_stride + (size_t) row * row_bytes;

    const int ib32 = tid >> 2, l = tid & 3;
    const int sh1 = 8 - 2 * l, sh2 = 7 - 2 * l, xoff = tid << 3;
    const int nblk = K >> 8;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int b = 0;
    for (; b + 3 < nblk; b += 4) {
        a0 += m1n_iq3s_block_dot(A + (size_t)(b+0)*IQ3S_BLOCK_BYTES, c_iq3s_grid, x + ((b+0)<<8) + xoff, tid, ib32, sh1, sh2);
        a1 += m1n_iq3s_block_dot(A + (size_t)(b+1)*IQ3S_BLOCK_BYTES, c_iq3s_grid, x + ((b+1)<<8) + xoff, tid, ib32, sh1, sh2);
        a2 += m1n_iq3s_block_dot(A + (size_t)(b+2)*IQ3S_BLOCK_BYTES, c_iq3s_grid, x + ((b+2)<<8) + xoff, tid, ib32, sh1, sh2);
        a3 += m1n_iq3s_block_dot(A + (size_t)(b+3)*IQ3S_BLOCK_BYTES, c_iq3s_grid, x + ((b+3)<<8) + xoff, tid, ib32, sh1, sh2);
    }
    for (; b < nblk; ++b)
        a0 += m1n_iq3s_block_dot(A + (size_t)b*IQ3S_BLOCK_BYTES, c_iq3s_grid, x + (b<<8) + xoff, tid, ib32, sh1, sh2);

    const float s = wave_sum((a0 + a1) + (a2 + a3));
    if (tid == 0) (which ? y_up : y_gate)[(size_t) col * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_gate_up_iq4xs(const unsigned char * __restrict__ gate_base,
                         const unsigned char * __restrict__ up_base,
                         size_t expert_stride, size_t row_bytes,
                         const int * __restrict__ ids, const float * __restrict__ x,
                         float * __restrict__ y_gate, float * __restrict__ y_up,
                         int rows, int K, int k_top, int T) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int col = blockIdx.y;
    if (col >= k_top * T) return;
    if (K & (FK_IQ4XS_BLOCK_WEIGHTS - 1)) return;        // bounded-safe
    const int which = blockIdx.z;
    const int tid   = threadIdx.x;
    x += (size_t)(col / k_top) * K;

    const unsigned char * A = (which ? up_base : gate_base)
                            + (size_t) ids[col] * expert_stride + (size_t) row * row_bytes;
    const int nblk = K >> 8;
    const int xoff = tid << 3;

    float a0 = 0.0f, a1 = 0.0f;
    int b = 0;
    for (; b + 1 < nblk; b += 2) {
        a0 += fk_iq4xs_block_dot(A + (size_t)(b+0)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, x + ((b+0)<<8) + xoff, tid);
        a1 += fk_iq4xs_block_dot(A + (size_t)(b+1)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, x + ((b+1)<<8) + xoff, tid);
    }
    for (; b < nblk; ++b)
        a0 += fk_iq4xs_block_dot(A + (size_t)b*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, x + (b<<8) + xoff, tid);

    const float s = wave_sum(a0 + a1);
    if (tid == 0) (which ? y_up : y_gate)[(size_t) col * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_down_iq4nl(const unsigned char * __restrict__ base, size_t expert_stride,
                      size_t row_bytes, const int * __restrict__ ids,
                      const float * __restrict__ h, float * __restrict__ eo,
                      int rows, int K, int k_top, int T) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int col = blockIdx.y;
    if (col >= k_top * T) return;
    if (K & (IQ4NL_BLOCK_WEIGHTS - 1)) return;           // bounded-safe
    const int tid = threadIdx.x;

    const unsigned char * A = base + (size_t) ids[col] * expert_stride + (size_t) row * row_bytes;
    const float * hr = h + (size_t) col * K;
    const int nchunk = (K >> 5) << 2;

    float lo = 0.0f, hi = 0.0f;
    for (int w = tid; w < nchunk; w += WAVE) m1n_iq4nl_chunk_dot(A, c_kvals_iq4nl, hr, w, &lo, &hi);

    const float s = wave_sum(lo + hi);
    if (tid == 0) eo[(size_t) col * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_down_q8_0(const unsigned char * __restrict__ base, size_t expert_stride,
                     size_t row_bytes, const int * __restrict__ ids,
                     const float * __restrict__ h, float * __restrict__ eo,
                     int rows, int K, int k_top, int T) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int col = blockIdx.y;
    if (col >= k_top * T) return;
    if (K % FK_Q8_0_BLOCK_WEIGHTS) return;               // bounded-safe
    const int tid = threadIdx.x;

    const unsigned char * A = base + (size_t) ids[col] * expert_stride + (size_t) row * row_bytes;
    const float * hr = h + (size_t) col * K;
    const int nblk = K / FK_Q8_0_BLOCK_WEIGHTS;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b)
        acc += fk_q8_0_lane_dot(A + (size_t) b * FK_Q8_0_BLOCK_BYTES,
                                hr + b * FK_Q8_0_BLOCK_WEIGHTS, tid);

    const float s = wave_sum(acc);
    if (tid == 0) eo[(size_t) col * rows + row] = s;
}

// ------------------------------------------- the row-gather (9.4 item 5) --
//
// At T = 1 the ten assignments are ten DIFFERENT experts, so there is
// nothing to share and the kernels above are the whole story. At T = 256
// there are 2 560 assignments over at most 512 experts -- five a card on
// average -- and the kernels above read each chosen expert's row once PER
// ASSIGNMENT. This sorts the T x 10 assignments by expert on the device and
// hands a wave a TILE of up to MOE_TILE_E assignments that share one, so the
// row is walked once for all of them.
//
// What it deliberately does NOT do is decode the weights once and reuse the
// decoded values: that would mean a second copy of the IQ3_S / IQ4_XS /
// IQ4_NL decode next to decode_quant.h's, and a "faithful-looking retype" of
// a kernel is exactly what that header exists to prevent (and what the two
// paths would silently drift apart on). The tile calls the SAME primitive per
// column, so the block's bytes are read from VRAM once and then served from
// L1 for the rest of the tile, and every column accumulates in exactly the
// order the one-assignment kernel used -- which is what keeps a chunk
// bit-identical to T decode steps.
// The DEFAULT tile. It is a template parameter rather than a constant because
// what it should be is a measurement, and the number that makes it one is the
// AVERAGE ASSIGNMENTS AN EXPERT: at T = 256 a chunk has 2 560 assignments over
// at most 512 experts, so a tile of 8 is on average five-eighths full and its
// `a[MOE_TILE_E][4]` accumulators are paid for in registers whether they are
// used or not. k_moe_gate_up_iq3s_gather is at 213 VGPR, which is 7 waves a
// SIMD on a kernel that is supposed to be weight-bandwidth bound. Smaller
// tiles trade amortisation for occupancy and the profile is what should pick,
// so --moe-tile does (4, 8 or 16; anything else falls back to 8).
//
// The tile size CANNOT move a bit: an assignment's column is accumulated in
// the same order whatever tile it lands in, which is the whole design of
// these kernels. Checked the way PREFILL.md section 8.1 checks everything
// now -- the ISA op multiset of each instantiation against the
// per-assignment kernel's, times the tile.
#define MOE_TILE_E 8

// Counting sort of the assignments by expert, plus the tile table, in one
// workgroup of N_EXPERT threads. The scatter order inside an expert is
// whatever the atomics give, which does not matter: every assignment writes
// its own output column.
template <int TE>
__global__ __launch_bounds__(N_EXPERT)
void k_moe_sort(const int * __restrict__ ids, int n,
                int * __restrict__ order, int * __restrict__ tile_exp,
                int * __restrict__ tile_off, int * __restrict__ tile_cnt,
                int n_tiles_max) {
    __shared__ int hist[N_EXPERT];
    __shared__ int base[N_EXPERT];
    __shared__ int tbase[N_EXPERT];
    __shared__ int cur[N_EXPERT];
    const int t = threadIdx.x;

    hist[t] = 0;
    for (int i = t; i < n_tiles_max; i += N_EXPERT) tile_cnt[i] = 0;
    __syncthreads();
    for (int i = t; i < n; i += N_EXPERT) atomicAdd(&hist[ids[i]], 1);
    __syncthreads();

    const int c  = hist[t];
    const int tc = (c + TE - 1) / TE;
    base[t]  = c;
    tbase[t] = tc;
    __syncthreads();
    for (int off = 1; off < N_EXPERT; off <<= 1) {          // inclusive scans
        const int a = (t >= off) ? base[t - off]  : 0;
        const int b = (t >= off) ? tbase[t - off] : 0;
        __syncthreads();
        base[t]  += a;
        tbase[t] += b;
        __syncthreads();
    }
    const int start = base[t]  - c;                          // exclusive
    const int t0    = tbase[t] - tc;
    cur[t] = start;
    __syncthreads();

    for (int i = t; i < n; i += N_EXPERT) {
        const int e = ids[i];
        order[atomicAdd(&cur[e], 1)] = i;
    }

    for (int k = 0; k * TE < c; ++k) {
        const int idx = t0 + k;
        if (idx >= n_tiles_max) break;
        tile_exp[idx] = t;
        tile_off[idx] = start + k * TE;
        tile_cnt[idx] = min(TE, c - k * TE);
    }
}

// grid = (rows, n_tiles_max, 2). A tile past the end has tile_cnt 0.
#define MOE_GATHER_PROLOGUE(nbits)                                                    \
    const int row = blockIdx.x;                                                       \
    if (row >= rows) return;                                                          \
    const int cnt = tile_cnt[blockIdx.y];                                             \
    if (cnt <= 0) return;                                                             \
    const int tid = threadIdx.x;                                                      \
    const int toff = tile_off[blockIdx.y];                                            \
    int cols[TE], xb[TE];

// SER: the m loop OUTER and four scalar accumulators, which is the shape the
// down gather had to take to be bit-identical (section 8.1 of PREFILL.md) and
// which here is about REGISTERS instead. The tiled shape holds TE*4 live
// accumulators on top of the IQ3_S decode's own working set and lands at 213
// VGPR -- 7 waves a SIMD on a kernel that is supposed to be waiting on VRAM --
// and it stays at 189 even with TE cut to 4, because the tile is not where
// most of them go. The serial shape is literally k_moe_gate_up_iq3s's body
// with a different x row, so it is that kernel's 94 VGPR and 16 waves, and it
// reads the expert row TE times from L1 instead of once from registers.
//
// Which of those two wins is a measurement (--expert-gather-serial), not a
// judgement. Neither can move a bit: the four accumulators, the block order
// and the closing (a0+a1)+(a2+a3) are the same in both.
template <int TE, bool SER>
__global__ __launch_bounds__(WAVE)
void k_moe_gate_up_iq3s_gather(const unsigned char * __restrict__ gate_base,
                               const unsigned char * __restrict__ up_base,
                               size_t expert_stride, size_t row_bytes,
                               const int * __restrict__ order,
                               const int * __restrict__ tile_exp,
                               const int * __restrict__ tile_off,
                               const int * __restrict__ tile_cnt,
                               const float * __restrict__ x,
                               float * __restrict__ y_gate, float * __restrict__ y_up,
                               int rows, int K, int k_top) {
    MOE_GATHER_PROLOGUE(0)
    if (K & (IQ3S_BLOCK_WEIGHTS - 1)) return;            // bounded-safe
#pragma unroll
    for (int m = 0; m < TE; ++m) {
        cols[m] = (m < cnt) ? order[toff + m] : 0;
        xb[m]   = (cols[m] / k_top) * K;
    }
    const unsigned char * A = (blockIdx.z ? up_base : gate_base)
                            + (size_t) tile_exp[blockIdx.y] * expert_stride
                            + (size_t) row * row_bytes;
    const int ib32 = tid >> 2, l = tid & 3;
    const int sh1 = 8 - 2 * l, sh2 = 7 - 2 * l, xoff = tid << 3;
    const int nblk = K >> 8;

    float * out = blockIdx.z ? y_up : y_gate;

    if constexpr (SER) {
#pragma clang loop unroll(disable)
        for (int m = 0; m < cnt; ++m) {
            const float * xm = x + xb[m] + xoff;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int b = 0;
            for (; b + 3 < nblk; b += 4) {
                a0 += m1n_iq3s_block_dot(A + (size_t)(b+0)*IQ3S_BLOCK_BYTES, c_iq3s_grid, xm + ((b+0)<<8), tid, ib32, sh1, sh2);
                a1 += m1n_iq3s_block_dot(A + (size_t)(b+1)*IQ3S_BLOCK_BYTES, c_iq3s_grid, xm + ((b+1)<<8), tid, ib32, sh1, sh2);
                a2 += m1n_iq3s_block_dot(A + (size_t)(b+2)*IQ3S_BLOCK_BYTES, c_iq3s_grid, xm + ((b+2)<<8), tid, ib32, sh1, sh2);
                a3 += m1n_iq3s_block_dot(A + (size_t)(b+3)*IQ3S_BLOCK_BYTES, c_iq3s_grid, xm + ((b+3)<<8), tid, ib32, sh1, sh2);
            }
            for (; b < nblk; ++b)
                a0 += m1n_iq3s_block_dot(A + (size_t)b*IQ3S_BLOCK_BYTES, c_iq3s_grid, xm + (b<<8), tid, ib32, sh1, sh2);
            const float s = wave_sum((a0 + a1) + (a2 + a3));
            if (tid == 0) out[(size_t) cols[m] * rows + row] = s;
        }
        return;
    }

    float a[TE][4];
#pragma unroll
    for (int m = 0; m < TE; ++m)
#pragma unroll
        for (int u = 0; u < 4; ++u) a[m][u] = 0.0f;

    int b = 0;
    for (; b + 3 < nblk; b += 4) {
#pragma unroll
        for (int m = 0; m < TE; ++m) {
            if (m >= cnt) continue;
            const float * xm = x + xb[m] + xoff;
#pragma unroll
            for (int u = 0; u < 4; ++u)
                a[m][u] += m1n_iq3s_block_dot(A + (size_t)(b+u)*IQ3S_BLOCK_BYTES, c_iq3s_grid,
                                              xm + ((b+u) << 8), tid, ib32, sh1, sh2);
        }
    }
    for (; b < nblk; ++b) {
#pragma unroll
        for (int m = 0; m < TE; ++m) {
            if (m >= cnt) continue;
            a[m][0] += m1n_iq3s_block_dot(A + (size_t)b*IQ3S_BLOCK_BYTES, c_iq3s_grid,
                                          x + xb[m] + xoff + (b << 8), tid, ib32, sh1, sh2);
        }
    }

#pragma unroll
    for (int m = 0; m < TE; ++m) {
        if (m >= cnt) continue;
        const float s = wave_sum((a[m][0] + a[m][1]) + (a[m][2] + a[m][3]));
        if (tid == 0) out[(size_t) cols[m] * rows + row] = s;
    }
}

template <int TE, bool SER>
__global__ __launch_bounds__(WAVE)
void k_moe_gate_up_iq4xs_gather(const unsigned char * __restrict__ gate_base,
                                const unsigned char * __restrict__ up_base,
                                size_t expert_stride, size_t row_bytes,
                                const int * __restrict__ order,
                                const int * __restrict__ tile_exp,
                                const int * __restrict__ tile_off,
                                const int * __restrict__ tile_cnt,
                                const float * __restrict__ x,
                                float * __restrict__ y_gate, float * __restrict__ y_up,
                                int rows, int K, int k_top) {
    MOE_GATHER_PROLOGUE(0)
    if (K & (FK_IQ4XS_BLOCK_WEIGHTS - 1)) return;        // bounded-safe
#pragma unroll
    for (int m = 0; m < TE; ++m) {
        cols[m] = (m < cnt) ? order[toff + m] : 0;
        xb[m]   = (cols[m] / k_top) * K;
    }
    const unsigned char * A = (blockIdx.z ? up_base : gate_base)
                            + (size_t) tile_exp[blockIdx.y] * expert_stride
                            + (size_t) row * row_bytes;
    const int nblk = K >> 8;
    const int xoff = tid << 3;

    float * out = blockIdx.z ? y_up : y_gate;

    if constexpr (SER) {
#pragma clang loop unroll(disable)
        for (int m = 0; m < cnt; ++m) {
            const float * xm = x + xb[m] + xoff;
            float a0 = 0.0f, a1 = 0.0f;
            int b = 0;
            for (; b + 1 < nblk; b += 2) {
                a0 += fk_iq4xs_block_dot(A + (size_t)(b+0)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, xm + ((b+0)<<8), tid);
                a1 += fk_iq4xs_block_dot(A + (size_t)(b+1)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, xm + ((b+1)<<8), tid);
            }
            for (; b < nblk; ++b)
                a0 += fk_iq4xs_block_dot(A + (size_t)b*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, xm + (b<<8), tid);
            const float s = wave_sum(a0 + a1);
            if (tid == 0) out[(size_t) cols[m] * rows + row] = s;
        }
        return;
    }

    float a[TE][2];
#pragma unroll
    for (int m = 0; m < TE; ++m) { a[m][0] = 0.0f; a[m][1] = 0.0f; }

    int b = 0;
    for (; b + 1 < nblk; b += 2) {
#pragma unroll
        for (int m = 0; m < TE; ++m) {
            if (m >= cnt) continue;
            const float * xm = x + xb[m] + xoff;
            a[m][0] += fk_iq4xs_block_dot(A + (size_t)(b+0)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl,
                                          xm + ((b+0) << 8), tid);
            a[m][1] += fk_iq4xs_block_dot(A + (size_t)(b+1)*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl,
                                          xm + ((b+1) << 8), tid);
        }
    }
    for (; b < nblk; ++b) {
#pragma unroll
        for (int m = 0; m < TE; ++m) {
            if (m >= cnt) continue;
            a[m][0] += fk_iq4xs_block_dot(A + (size_t)b*FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl,
                                          x + xb[m] + xoff + (b << 8), tid);
        }
    }

#pragma unroll
    for (int m = 0; m < TE; ++m) {
        if (m >= cnt) continue;
        const float s = wave_sum(a[m][0] + a[m][1]);
        if (tid == 0) out[(size_t) cols[m] * rows + row] = s;
    }
}

template <int TE>
__global__ __launch_bounds__(WAVE)
void k_moe_down_iq4nl_gather(const unsigned char * __restrict__ base, size_t expert_stride,
                             size_t row_bytes, const int * __restrict__ order,
                             const int * __restrict__ tile_exp,
                             const int * __restrict__ tile_off,
                             const int * __restrict__ tile_cnt,
                             const float * __restrict__ h, float * __restrict__ eo,
                             int rows, int K, int k_top) {
    (void) k_top;
    MOE_GATHER_PROLOGUE(0)
    if (K & (IQ4NL_BLOCK_WEIGHTS - 1)) return;           // bounded-safe
#pragma unroll
    for (int m = 0; m < TE; ++m) {
        cols[m] = (m < cnt) ? order[toff + m] : 0;
        xb[m]   = cols[m] * K;                           // h is [col][K]
    }
    const unsigned char * A = base + (size_t) tile_exp[blockIdx.y] * expert_stride
                            + (size_t) row * row_bytes;
    const int nchunk = (K >> 5) << 2;

    // THE M LOOP IS OUTER AND THE ACCUMULATORS ARE SCALARS, and that shape is
    // the answer to PREFILL.md section 8. With the w loop outside and eight
    // independent accumulator pairs inside, the backend has ILP to spare and
    // keeps every `lo += (d*kv)*hb` of m1n_iq4nl_chunk_dot contracted into an
    // FMA; with ONE pair the dependency chain is the critical path, so it
    // breaks all but one of them into v_mul_f32 + v_add_f32 to shorten it.
    // Same source, same algebra, one rounding apart -- which is exactly the
    // divergence that survived the sort, the indexing, the accumulator
    // structure and the reduction being cleared on the host.
    //
    // Measured on the ISA (gfx1100, ROCm 7.14, -O3, hipcc --offload-device-only
    // -S): this body and k_moe_down_iq4nl now emit the SAME multiset of float
    // ops -- 8 v_fma_mix_f32, 1 v_fmac_f32, 7 v_mul_f32, 13 v_add_f32 -- where
    // the tiled form emitted 8 v_fma_mix_f32, 16 v_fma_f32, 48 v_fmac_f32 and
    // 48 v_add_f32, i.e. every term fused. VGPRs 43 -> 31, occupancy 16 either
    // way, no spills.
    //
    // The amortisation survives, which is the point: the expert row is
    // row_bytes = 360 B at K = 640, so the second and later passes over it are
    // L1 hits and the VRAM read is still ONCE A TILE rather than once an
    // assignment. Only the order of the reads changes, not their count.
    //
    // unroll(disable) is load-bearing: an unrolled m loop is a different
    // scheduling problem and the backend starts contracting again.
#pragma clang loop unroll(disable)
    for (int m = 0; m < cnt; ++m) {
        const float * hr = h + xb[m];
        float lo = 0.0f, hi = 0.0f;
        for (int w = tid; w < nchunk; w += WAVE) m1n_iq4nl_chunk_dot(A, c_kvals_iq4nl, hr, w, &lo, &hi);

        const float s = wave_sum(lo + hi);
        if (tid == 0) eo[(size_t) cols[m] * rows + row] = s;
    }
}

template <int TE>
__global__ __launch_bounds__(WAVE)
void k_moe_down_q8_0_gather(const unsigned char * __restrict__ base, size_t expert_stride,
                            size_t row_bytes, const int * __restrict__ order,
                            const int * __restrict__ tile_exp,
                            const int * __restrict__ tile_off,
                            const int * __restrict__ tile_cnt,
                            const float * __restrict__ h, float * __restrict__ eo,
                            int rows, int K, int k_top) {
    (void) k_top;
    MOE_GATHER_PROLOGUE(0)
    if (K % FK_Q8_0_BLOCK_WEIGHTS) return;               // bounded-safe
#pragma unroll
    for (int m = 0; m < TE; ++m) {
        cols[m] = (m < cnt) ? order[toff + m] : 0;
        xb[m]   = cols[m] * K;
    }
    const unsigned char * A = base + (size_t) tile_exp[blockIdx.y] * expert_stride
                            + (size_t) row * row_bytes;
    const int nblk = K / FK_Q8_0_BLOCK_WEIGHTS;

    float acc[TE];
#pragma unroll
    for (int m = 0; m < TE; ++m) acc[m] = 0.0f;

    // The inner statement is TEXTUALLY the per-assignment kernel's, down to
    // the primitive call. It used to hoist `d * qs[tid]` out of the column
    // loop -- algebraically the same thing, and the host test agrees to the
    // bit -- but the device did not, so the hoist is gone: when two paths
    // must produce the same bits, the cheapest guarantee is that they are
    // the same source. The decode is then recomputed per column, which for
    // Q8_0 is one f16 widen and one multiply off L1.
    //
    // THIS ONE KEEPS ITS TILE, and the ISA says why it may. Against
    // k_moe_down_q8_0 it is the per-assignment body times eight, op for op:
    // 8 v_fma_f32 against 1 v_fmac_f32, 40 v_add_f32 against 5 (the wave
    // reduction), and ONE v_fma_mix_f32 either way because `d * qs[lane]`
    // does not depend on m and is common-subexpression-eliminated. Every
    // term stays FUSED here, so there is no rounding to disagree about --
    // unlike IQ4_NL above, whose accumulate lives inside the shared
    // primitive and which the backend de-contracts when it has only one
    // accumulator chain to schedule. This model's down projection is
    // IQ4_NL; Q8_0 is here for a container that is not.
    for (int b = 0; b < nblk; ++b) {
#pragma unroll
        for (int m = 0; m < TE; ++m)
            if (m < cnt) acc[m] += fk_q8_0_lane_dot(A + (size_t) b * FK_Q8_0_BLOCK_BYTES,
                                                    h + xb[m] + b * FK_Q8_0_BLOCK_WEIGHTS, tid);
    }

#pragma unroll
    for (int m = 0; m < TE; ++m) {
        if (m >= cnt) continue;
        const float s = wave_sum(acc[m]);
        if (tid == 0) eo[(size_t) cols[m] * rows + row] = s;
    }
}

__global__ void k_silu_mul(const float * __restrict__ g, const float * __restrict__ u,
                           float * __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = d_silu(g[i]) * u[i];
}

// build_moe_ffn's gating, one workgroup: softmax over all 512, then ten
// argmax passes (index-tiebroken, so it is the same set std::partial_sort
// gives the CPU path), then the clamped normalisation.
__global__ __launch_bounds__(N_EXPERT)
void k_router(const float * __restrict__ logits, int * __restrict__ ids,
              float * __restrict__ w, int * __restrict__ ids_log,
              float * __restrict__ w_log) {
    __shared__ float sv[N_EXPERT];
    __shared__ int   si[N_EXPERT];
    __shared__ float sp[N_EXPERT];
    __shared__ int   taken[N_EXPERT_USED];
    const int t = threadIdx.x;
    // one workgroup per token of the chunk
    logits += (size_t) blockIdx.x * N_EXPERT;
    ids    += (size_t) blockIdx.x * N_EXPERT_USED;
    w      += (size_t) blockIdx.x * N_EXPERT_USED;
    if (ids_log) ids_log += (size_t) blockIdx.x * N_EXPERT_USED;
    if (w_log)   w_log   += (size_t) blockIdx.x * N_EXPERT_USED;

    sv[t] = logits[t];
    __syncthreads();
    for (int s = N_EXPERT / 2; s > 0; s >>= 1) {
        if (t < s) sv[t] = fmaxf(sv[t], sv[t + s]);
        __syncthreads();
    }
    const float mx = sv[0];
    __syncthreads();

    sp[t] = __expf(logits[t] - mx);
    sv[t] = sp[t];
    __syncthreads();
    for (int s = N_EXPERT / 2; s > 0; s >>= 1) {
        if (t < s) sv[t] += sv[t + s];
        __syncthreads();
    }
    const float den = sv[0];
    __syncthreads();
    sp[t] /= den;

    for (int k = 0; k < N_EXPERT_USED; ++k) {
        bool used = false;
        for (int j = 0; j < k; ++j) if (taken[j] == t) used = true;
        sv[t] = used ? -1.0f : sp[t];
        si[t] = t;
        __syncthreads();
        for (int s = N_EXPERT / 2; s > 0; s >>= 1) {
            if (t < s) {
                const bool better = sv[t + s] > sv[t] || (sv[t + s] == sv[t] && si[t + s] < si[t]);
                if (better) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
            }
            __syncthreads();
        }
        if (t == 0) taken[k] = si[0];
        __syncthreads();
    }

    if (t < N_EXPERT_USED) {
        ids[t] = taken[t];
        if (ids_log) ids_log[t] = taken[t];      // the routing oracle's copy,
    }                                            // written here so capture
    __syncthreads();                             // costs no extra launch
    if (t == 0) {
        float s = 0.0f;
        for (int k = 0; k < N_EXPERT_USED; ++k) s += sp[taken[k]];
        s = fmaxf(s, 6.103515625e-5f);           // build_moe_ffn's f16 clamp
        for (int k = 0; k < N_EXPERT_USED; ++k) {
            const float v = sp[taken[k]] / s;
            w[k] = v;
            if (w_log) w_log[k] = v;
        }
    }
}

// ====================================================== elementwise ======

__global__ void k_unary(int op, const float * __restrict__ x, float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    switch (op) {
        case FK_SILU:     y[i] = d_silu(v);     break;
        case FK_SIGMOID:  y[i] = d_sigmoid(v);  break;
        case FK_SOFTPLUS: y[i] = d_softplus(v); break;
        case FK_EXP:      y[i] = expf(v);       break;
        default:          y[i] = v;             break;
    }
}

__global__ void k_binary(int op, const float * __restrict__ a, const float * __restrict__ b,
                         float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (op) {
        case FK_ADD: y[i] = a[i] + b[i]; break;
        case FK_MUL: y[i] = a[i] * b[i]; break;
        default:     y[i] = a[i] - b[i]; break;
    }
}

__global__ void k_binary3_add(const float * __restrict__ a, const float * __restrict__ b,
                              const float * __restrict__ c, float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i] + c[i];
}

__global__ void k_scale(const float * __restrict__ x, float s, float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}

__global__ void k_mul_tiled(const float * __restrict__ x, const float * __restrict__ w,
                            float * __restrict__ y, int n, int ne0_w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * w[i % ne0_w];
}

__global__ void k_gather_strided(const float * __restrict__ src, float * __restrict__ dst,
                                 int n_rows, int row_len, int src_stride, int src_off) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * row_len) return;
    const int r = i / row_len, c = i - r * row_len;
    dst[i] = src[(size_t) r * src_stride + src_off + c];
}

// One workgroup per group. The gamma is PERIODIC in the group index with
// period `w_period` groups: 1 for a [ne0] gamma reused by every group
// (build_norm), HC for build_hc_mix's grouped RMSNorm over a token-major
// [T][HC][ne0] buffer. That replaced a `w_tiled` flag, which could only say
// "all groups" or "one group each" and therefore could not describe a chunk.
// The hc modules normalise HC groups a token, so at T = 1 the grid is four
// workgroups whatever the block size: the only lever left is making each
// workgroup wider, which shortens the strided walk over ne0=2560 from ten
// steps to two or three. Block size is chosen by the host and the LDS is
// dynamic. At T = 256 the same launch has 1 024 workgroups.
__global__ void k_rms_norm_mul(const float * __restrict__ x, const float * __restrict__ w,
                    float * __restrict__ y, int ne0, float eps, int w_period, int has_w) {
    extern __shared__ float lds[];
    const float * xg = x + (size_t) blockIdx.x * ne0;
    float *       yg = y + (size_t) blockIdx.x * ne0;

    float s = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) s = fmaf(xg[i], xg[i], s);
    s = block_sum(s, lds);
    const float sc = rsqrtf(s / (float) ne0 + eps);
    const float * wg = has_w ? w + (size_t)(blockIdx.x % w_period) * ne0 : nullptr;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) yg[i] = xg[i] * sc * (has_w ? wg[i] : 1.0f);
}

// ggml_l2_norm: 1/max(sqrt(sum), eps). NOT the rms form.
// q_conv and k_conv are adjacent halves of the conv output, so one launch of
// 2*GDN_K_HEADS groups normalises both.
__global__ void k_l2_norm(const float * __restrict__ x, float * __restrict__ y, int ne0, float eps) {
    extern __shared__ float lds[];
    const float * xg = x + (size_t) blockIdx.x * ne0;
    float *       yg = y + (size_t) blockIdx.x * ne0;
    float s = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) s = fmaf(xg[i], xg[i], s);
    s = block_sum(s, lds);
    const float sc = 1.0f / fmaxf(sqrtf(s), eps);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) yg[i] = xg[i] * sc;
}

__global__ void k_gated_rms_norm(const float * __restrict__ x, const float * __restrict__ w,
                      const float * __restrict__ z, float * __restrict__ y,
                      int ne0, float eps) {
    extern __shared__ float lds[];
    const size_t o = (size_t) blockIdx.x * ne0;
    float s = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) s = fmaf(x[o + i], x[o + i], s);
    s = block_sum(s, lds);
    const float sc = rsqrtf(s / (float) ne0 + eps);
    for (int i = threadIdx.x; i < ne0; i += blockDim.x)
        y[o + i] = x[o + i] * sc * w[i] * d_sigmoid(z[o + i]);
}

// ================================================ hyper-connections ======

// The grid is n*T: a token is a row, and a row's arithmetic does not know
// how many rows travel with it.
__global__ void k_hc_broadcast(const float * __restrict__ x, float * __restrict__ res, int T) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N_EMBD * T) return;
    const int t = g / N_EMBD, i = g - t * N_EMBD;
    const float v = x[g];
    float * r = res + (size_t) t * HC_DIM;
#pragma unroll
    for (int c = 0; c < HC; ++c) r[c * N_EMBD + i] = v;
}

__global__ void k_hc_collapse(const float * __restrict__ xn, const float * __restrict__ gate,
                              float * __restrict__ out, int T) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N_EMBD * T) return;
    const int t = g / N_EMBD, i = g - t * N_EMBD;
    const float * xt = xn   + (size_t) t * HC_DIM;
    const float * gt = gate + (size_t) t * HC_DIM;
    float a = xt[i] * gt[i];
#pragma unroll
    for (int c = 1; c < HC; ++c) a += xt[c * N_EMBD + i] * gt[c * N_EMBD + i];
    out[g] = a * (1.0f / (float) HC);
}

// ============================================================== PLE ======

// One workgroup per (token, hc stream): blockIdx.x = t*HC + c.
__global__ void k_ple_gate(const float * __restrict__ key, const float * __restrict__ query,
                const float * __restrict__ value, float * __restrict__ gated,
                float * __restrict__ gate_out) {
    extern __shared__ float lds[];
    const int t = blockIdx.x / HC;
    const int c = blockIdx.x - t * HC;
    const size_t o = (size_t) t * HC_DIM + (size_t) c * N_EMBD;
    const size_t vo = (size_t) t * N_EMBD;
    float s = 0.0f;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) s = fmaf(key[o + i], query[o + i], s);
    s = block_sum(s, lds);
    const float sf  = s * rsqrtf((float) N_EMBD);
    const float mag = sqrtf(fmaxf(fabsf(sf), 1e-6f));      // ggml_sqrt(ggml_clamp(|s|,1e-6,inf))
    const float sgn = sf > 0.0f ? 1.0f : (sf < 0.0f ? -1.0f : 0.0f);
    const float g   = d_sigmoid(sgn * mag);
    if (threadIdx.x == 0) gate_out[blockIdx.x] = g;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) gated[o + i] = value[vo + i] * g;
}

// ggml_ssm_conv generalised by a dilation, over a WINDOW rather than a ring:
// token t of the chunk sits at slot (kern-1)*dil + t and its tap k -- which
// reads (kern-1-k)*dil positions back -- is at slot t + k*dil. No modulus,
// and a chunk's first tokens read the previous chunk's tail with no special
// case. The grid is channels*T.
__global__ void k_conv_win(const float * __restrict__ win, int kern, int dil, int channels,
                           const float * __restrict__ w, float * __restrict__ out, int T) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= channels * T) return;
    const int t = g / channels, c = g - t * channels;
    float a = 0.0f;
    for (int k = 0; k < kern; ++k)
        a = fmaf(win[(size_t)(t + k * dil) * channels + c], w[(size_t) c * kern + k], a);
    out[g] = d_silu(a);
}

// Carry the window's tail into the next chunk. Source [T, T+hist) and
// destination [0, hist) OVERLAP whenever T < hist (chunk 1 against the PLE
// conv's nine history slots is the everyday case), so each thread reads its
// whole channel column into registers before writing any of it -- a thread
// touches only its own channel, so there is no cross-thread hazard.
#define CONV_SLIDE_MAX_HIST 16
__global__ void k_conv_slide(float * __restrict__ win, int hist, int channels, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // unrolled over a compile-time bound, or `col` would be scratch
    float col[CONV_SLIDE_MAX_HIST];
#pragma unroll
    for (int s = 0; s < CONV_SLIDE_MAX_HIST; ++s)
        if (s < hist) col[s] = win[(size_t)(T + s) * channels + c];
#pragma unroll
    for (int s = 0; s < CONV_SLIDE_MAX_HIST; ++s)
        if (s < hist) win[(size_t) s * channels + c] = col[s];
}

// ===================================================== Gated DeltaNet ====
//
// One wave per (v-head, value row j). The row is the 128 KEY entries of the
// state, contiguous in ggml's layout, so the whole step -- decay, the
// prediction dot, the rank-1 update and the output dot -- happens in four
// registers per lane with two wave reductions and no LDS.
// The chunk is walked IN ORDER inside the kernel with the state row held in
// registers: the state token t sees is the one token t-1 left, so T tokens
// in one launch are the same sequence of updates as T launches of one token
// and the recurrence is bit-identical at any chunk size. The state is read
// once and written once for the whole chunk instead of once per token, which
// is also the only reason a sequential kernel is not a disaster here -- it
// is ~0.6 ms of a 22.7 ms token (PREFILL.md section 4).
__global__ __launch_bounds__(WAVE)
void k_gdn_step(float * __restrict__ S, const float * __restrict__ q,
                const float * __restrict__ k, const float * __restrict__ v,
                const float * __restrict__ g_exp, const float * __restrict__ beta,
                float * __restrict__ out, int T, int qk_stride, int v_stride) {
    constexpr int PER = GDN_STATE / WAVE;                 // 4
    const int h    = blockIdx.x / GDN_STATE;
    const int j    = blockIdx.x - h * GDN_STATE;
    const int lane = threadIdx.x;
    const int hk   = h % GDN_K_HEADS;                     // ggml_repeat / `iv1 % nek1`

    float * row = S + ((size_t) h * GDN_STATE + j) * GDN_STATE;

    float r[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) r[p] = row[lane + p * WAVE];

    for (int t = 0; t < T; ++t) {
        const float * kh = k + (size_t) t * qk_stride + (size_t) hk * GDN_STATE;
        const float * qh = q + (size_t) t * qk_stride + (size_t) hk * GDN_STATE;
        const float ge = g_exp[(size_t) t * GDN_V_HEADS + h];

        float kk[PER], qq[PER];
        float sk = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int i = lane + p * WAVE;
            r[p]  = r[p] * ge;                            // decay FIRST (ops.cpp:11012)
            kk[p] = kh[i];
            qq[p] = qh[i];
            sk    = fmaf(r[p], kk[p], sk);
        }
        sk = wave_sum(sk);

        const float d = beta[(size_t) t * GDN_V_HEADS + h] *
                        (v[(size_t) t * v_stride + (size_t) h * GDN_STATE + j] - sk);
        float o = 0.0f;
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            r[p] = fmaf(kk[p], d, r[p]);
            o = fmaf(r[p], qq[p], o);
        }
        o = wave_sum(o);
        if (lane == 0)
            out[(size_t) t * GDN_VAL_DIM + (size_t) h * GDN_STATE + j] =
                o * rsqrtf((float) GDN_STATE);
    }

#pragma unroll
    for (int p = 0; p < PER; ++p) row[lane + p * WAVE] = r[p];
}

// =================================================== fused small kernels ==
//
// Every one of these replaces a run of dependent elementwise launches whose
// data would fit in a cache line or two. They exist because a dependent chain
// pays a kernel latency per link whatever its size -- the measured
// prof_elem_hc_us was 2.0 ms of an 11.7 ms token for work worth a few
// microseconds of bandwidth. All of them still WRITE the reference's named
// intermediates, so every oracle tap is unchanged.

// The Gated DeltaNet conv (ggml_ssm_conv + silu over GDN_CONV_DIM channels)
// and the gate chain of qwen4exp.cpp:816-834 (sigmoid(beta),
// softplus(alpha+dt), *ssm_a, exp) in one launch. The gate work is 48
// elements; it rides along on threads past the channel count for free.
// grid.x covers GDN_CONV_DIM + GDN_V_HEADS lanes of work, grid.y the chunk.
// The conv output is written twice: conv_out in the reference's contiguous
// [q|k|v] row (the oracle tap, and where the recurrence reads v) and
// conv_qk with q and k alone, so l2_norm's 32 groups a token are contiguous
// across the whole chunk.
__global__ void k_gdn_conv_gate(const float * __restrict__ win,
                                const float * __restrict__ w,
                                const float * __restrict__ beta_raw,
                                const float * __restrict__ alpha_raw,
                                const float * __restrict__ dt,
                                const float * __restrict__ ssm_a,
                                float * __restrict__ conv_out,
                                float * __restrict__ conv_qk,
                                float * __restrict__ beta_sig,
                                float * __restrict__ a_softplus,
                                float * __restrict__ gate,
                                float * __restrict__ g_exp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i < GDN_CONV_DIM) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < GDN_CONV_K; ++k)
            a = fmaf(win[(size_t)(t + k) * GDN_CONV_DIM + i], w[(size_t) i * GDN_CONV_K + k], a);
        const float v = d_silu(a);
        conv_out[(size_t) t * GDN_CONV_DIM + i] = v;
        if (i < 2 * GDN_KEY_DIM) conv_qk[(size_t) t * 2 * GDN_KEY_DIM + i] = v;
        return;
    }
    const int h = i - GDN_CONV_DIM;
    if (h >= GDN_V_HEADS) return;
    const size_t o = (size_t) t * GDN_V_HEADS + h;
    beta_sig[o]   = d_sigmoid(beta_raw[o]);
    const float sp = d_softplus(alpha_raw[o] + dt[h]);
    a_softplus[o] = sp;
    const float g = sp * ssm_a[h];          // ssm_a already holds -exp(A_log)
    gate[o]       = g;
    g_exp[o]      = expf(g);
}

// build_moe_ffn's tail: the weighted expert sum, the shared expert's sigmoid
// gate and the add, in one launch instead of four. ffn_moe_out and
// ffn_shexp_gated are written because both are oracle taps.
__global__ void k_moe_finish(const float * __restrict__ eo,
                             const float * __restrict__ w,
                             const float * __restrict__ sh_raw,
                             const float * __restrict__ sh_gate,   // RAW logit
                             float * __restrict__ sh_gate_sig,
                             float * __restrict__ moe_out,
                             float * __restrict__ sh_gated,
                             float * __restrict__ out,
                             int rows, int k_top, int T) {
    const int g_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_ >= rows * T) return;
    const int t = g_ / rows, i = g_ - t * rows;
    const float g0 = d_sigmoid(sh_gate[t]);
    if (i == 0) sh_gate_sig[t] = g0;
    const float * eot = eo + (size_t) t * k_top * rows;
    const float * wt  = w  + (size_t) t * k_top;
    float a = 0.0f;
    for (int k = 0; k < k_top; ++k) a += wt[k] * eot[(size_t) k * rows + i];
    moe_out[g_] = a;
    const float g = sh_raw[g_] * g0;
    sh_gated[g_] = g;
    out[g_] = a + g;
}

// hc_combine, optionally followed by the grouped RMSNorm the NEXT hc_mix
// would have launched on its own. One workgroup per hc stream: the combine is
// elementwise over that stream's 2 560 elements and the norm reduces over
// exactly the same range, so the two fuse with one __syncthreads and no
// extra traffic. `w_norm == nullptr` does the combine alone.
// blockIdx.x = t*HC + c: one workgroup per (token, hc stream).
__global__ void k_hc_combine_norm(float * __restrict__ res, const float * __restrict__ blk,
                                  const float * __restrict__ inject,
                                  const float * __restrict__ w_norm,
                                  float * __restrict__ xn, float eps) {
    extern __shared__ float lds[];
    const int t = blockIdx.x / HC;
    const int c = blockIdx.x - t * HC;
    float * rc = res + (size_t) t * HC_DIM + (size_t) c * N_EMBD;
    const float * bt = blk + (size_t) t * N_EMBD;
    const float wgt = 2.0f * d_sigmoid(inject[(size_t) t * HC + c] * (1.0f / (float) HC));

    float ss = 0.0f;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) {
        const float v = rc[i] + bt[i] * wgt;
        rc[i] = v;
        ss = fmaf(v, v, ss);
    }
    if (!w_norm) return;
    ss = block_sum(ss, lds);
    const float sc = rsqrtf(ss / (float) N_EMBD + eps);
    const float * wg = w_norm + (size_t) c * N_EMBD;
    float * xt = xn + (size_t) t * HC_DIM + (size_t) c * N_EMBD;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x)
        xt[i] = rc[i] * sc * wg[i];
}

// build_layer_attn's per-head post-processing, all of it, in one launch:
// the [q|gate] split of attn_q's interleaved rows, the q/k QK-norms, IMRoPE
// on q, k and the indexer query, and the output gate's sigmoid. It replaces
// two gathers, three norms, three ropes and a sigmoid -- nine launches whose
// grids were as small as four workgroups of one wave.
//
//   blockIdx.x <  N_Q_HEADS                      a query head (256 dims)
//              <  N_Q_HEADS+N_KV_HEADS           a key head   (256 dims)
//              else                              an indexer query head (128)
#define QKPOST_THREADS 256
__global__ __launch_bounds__(QKPOST_THREADS)
void k_qsa_qk_post(const float * __restrict__ qfull,   // [N_Q_HEADS][2*HEAD_DIM]
                   const float * __restrict__ kraw,    // [N_KV_HEADS][HEAD_DIM]
                   const float * __restrict__ idxraw,  // [IDX_N_HEADS][IDX_DIM]
                   const float * __restrict__ q_norm, const float * __restrict__ k_norm,
                   const float * __restrict__ iq_norm,
                   float * __restrict__ qcur, float * __restrict__ gate,
                   float * __restrict__ gsig, float * __restrict__ kcur,
                   float * __restrict__ idxq,
                   int pos0, float eps, const int * __restrict__ dpos) {
    __shared__ float lds[QKPOST_THREADS];
    __shared__ float sv[HEAD_DIM];
    // --hip-graph: the position is DEVICE-RESIDENT (one int the replayed
    // graph advances itself), so the captured launch never changes. Same
    // integer either way, so the same arithmetic.
    if (dpos) pos0 = *dpos;

    // grid.y is the chunk: a row's post-processing is independent of every
    // other row's, so this is the decode kernel with one more grid axis.
    const int tk  = blockIdx.y;
    const int pos = pos0 + tk;
    qfull  += (size_t) tk * N_Q_HEADS   * HEAD_DIM * 2;
    kraw   += (size_t) tk * N_KV_HEADS  * HEAD_DIM;
    idxraw += (size_t) tk * IDX_N_HEADS * IDX_DIM;
    qcur   += (size_t) tk * N_Q_HEADS   * HEAD_DIM;
    gate   += (size_t) tk * N_Q_HEADS   * HEAD_DIM;
    gsig   += (size_t) tk * N_Q_HEADS   * HEAD_DIM;
    kcur   += (size_t) tk * N_KV_HEADS  * HEAD_DIM;
    idxq   += (size_t) tk * IDX_N_HEADS * IDX_DIM;

    const int b = blockIdx.x;
    const bool is_q   = b < N_Q_HEADS;
    const bool is_k   = !is_q && b < N_Q_HEADS + N_KV_HEADS;
    const int  ne0    = (is_q || is_k) ? HEAD_DIM : IDX_DIM;
    const int  t      = threadIdx.x;

    const float * src;
    const float * wn;
    if      (is_q) { src = qfull + (size_t) b * HEAD_DIM * 2;                    wn = q_norm;  }
    else if (is_k) { src = kraw  + (size_t)(b - N_Q_HEADS) * HEAD_DIM;           wn = k_norm;  }
    else           { src = idxraw+ (size_t)(b - N_Q_HEADS - N_KV_HEADS) * IDX_DIM; wn = iq_norm; }

    // the gate half of attn_q's interleaved row needs no norm and no rotation
    if (is_q) {
        const float * g = src + HEAD_DIM;
        for (int i = t; i < HEAD_DIM; i += QKPOST_THREADS) {
            const float v = g[i];
            gate[(size_t) b * HEAD_DIM + i] = v;
            gsig[(size_t) b * HEAD_DIM + i] = d_sigmoid(v);
        }
    }

    float ss = 0.0f;
    for (int i = t; i < ne0; i += QKPOST_THREADS) ss = fmaf(src[i], src[i], ss);
    ss = block_sum(ss, lds);
    const float sc = rsqrtf(ss / (float) ne0 + eps);
    for (int i = t; i < ne0; i += QKPOST_THREADS) sv[i] = src[i] * sc * wn[i];
    __syncthreads();

    // IMRoPE over the first N_ROT dims, NeoX pairing (j, j + N_ROT/2)
    if (t < N_ROT / 2) {
        const int sect = ROPE_SECTIONS[0] + ROPE_SECTIONS[1] + ROPE_SECTIONS[2] + ROPE_SECTIONS[3];
        const int sector = t % sect;
        const bool is_e = !((sector % 3 == 1 && sector < 3 * ROPE_SECTIONS[1]) ||
                            (sector % 3 == 2 && sector < 3 * ROPE_SECTIONS[2]) ||
                            (sector % 3 == 0 && sector < 3 * ROPE_SECTIONS[0]));
        const float base = is_e ? 0.0f : (float) pos;
        const float ts   = __powf(ROPE_FREQ_BASE, -2.0f / (float) N_ROT);
        const float th   = base * __powf(ts, (float) t);
        const float cc = __cosf(th), sn = __sinf(th);
        const float x0 = sv[t], x1 = sv[t + N_ROT / 2];
        sv[t]              = x0 * cc - x1 * sn;
        sv[t + N_ROT / 2]  = x0 * sn + x1 * cc;
    }
    __syncthreads();

    float * dst = is_q ? qcur + (size_t) b * HEAD_DIM
                : is_k ? kcur + (size_t)(b - N_Q_HEADS) * HEAD_DIM
                       : idxq + (size_t)(b - N_Q_HEADS - N_KV_HEADS) * IDX_DIM;
    for (int i = t; i < ne0; i += QKPOST_THREADS) dst[i] = sv[i];
}

// ============================================================ IMRoPE =====
//
// mode GGML_ROPE_TYPE_IMROPE with a TEXT batch: p_t = p_h = p_w = pos and
// p_e = 0 (llama-graph.cpp:132-144), and with sections [11,11,10,0] every one
// of the 32 sectors maps to t, h or w -- so all three thetas being equal
// makes this an ordinary NeoX rope over n_rot dims. The sections are still
// evaluated, so an image batch does not need a different kernel.
// theta is taken as pos * theta_scale^(i0/2) with a powf, as llama.cpp's own
// device rope does, rather than the CPU's iterative multiply: the two differ
// by f32 rounding only.
__global__ void k_rope_imrope(float * __restrict__ x, int head_dim, int n_rot, int pos) {
    const int j = threadIdx.x;                            // 0 .. n_rot/2-1
    if (j >= n_rot / 2) return;
    const int h = blockIdx.x;

    const int sect_dims = ROPE_SECTIONS[0] + ROPE_SECTIONS[1] + ROPE_SECTIONS[2] + ROPE_SECTIONS[3];
    const int sector = j % sect_dims;
    const bool is_e = !((sector % 3 == 1 && sector < 3 * ROPE_SECTIONS[1]) ||
                        (sector % 3 == 2 && sector < 3 * ROPE_SECTIONS[2]) ||
                        (sector % 3 == 0 && sector < 3 * ROPE_SECTIONS[0]));
    const float base = is_e ? 0.0f : (float) pos;         // p_e is 0 for a text batch

    const float ts    = __powf(ROPE_FREQ_BASE, -2.0f / (float) n_rot);
    const float theta = base * __powf(ts, (float) j);
    const float c = __cosf(theta), s = __sinf(theta);

    float * row = x + (size_t) h * head_dim;
    const int half = n_rot / 2;
    const float x0 = row[j], x1 = row[j + half];
    row[j]        = x0 * c - x1 * s;
    row[j + half] = x0 * s + x1 * c;
}

// ============================================================= QSA =======

__global__ void k_kv_store_q8_0(const float * __restrict__ k, const float * __restrict__ v,
                                int8_t * __restrict__ kqs, __half * __restrict__ ksc,
                                int8_t * __restrict__ vqs, __half * __restrict__ vsc,
                                int cell0, int T, const int * __restrict__ dpos) {
    // one workgroup per (tensor, head, token, 32-element block); 32 lanes
    if (dpos) cell0 = *dpos;                              // --hip-graph
    const int which = blockIdx.z;                         // 0 = K, 1 = V
    const int h     = blockIdx.y;
    const int t     = blockIdx.x / (HEAD_DIM / 32);
    const int b     = blockIdx.x - t * (HEAD_DIM / 32);
    const int lane  = threadIdx.x;
    if (t >= T) return;
    const int cell  = cell0 + t;

    const float * src = (which ? v : k) + (size_t) t * N_KV_HEADS * HEAD_DIM
                      + (size_t) h * HEAD_DIM + b * 32;
    float a = fabsf(src[lane]);
#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor(a, off, WAVE));
    const __half sc = __float2half(a > 0.0f ? a / 127.0f : 1.0f);
    const float inv = 1.0f / __half2float(sc);            // quantise against the STORED scale
    int q = (int) rintf(src[lane] * inv);
    q = max(-127, min(127, q));

    const size_t e  = ((size_t) cell * N_KV_HEADS + h) * HEAD_DIM + b * 32 + lane;
    const size_t sb = ((size_t) cell * N_KV_HEADS + h) * (HEAD_DIM / 32) + b;
    if (which) { vqs[e] = (int8_t) q; if (lane == 0) vsc[sb] = sc; }
    else       { kqs[e] = (int8_t) q; if (lane == 0) ksc[sb] = sc; }
}

// One workgroup of IDX_DIM threads. A block's members arrive in order, so a
// running SUM plus cell 0's key reproduces set_input_qsa's formula exactly --
// unset slots read cell 0 (llama-memory-hybrid-idx.cpp:395) -- without keeping
// every token's raw key. That f32 cache was 512 B a token a layer, eight times
// what design 9.1 budgets for the pooled bf16 key and 134 MB a layer at 256k.
// One workgroup, the whole chunk, IN ORDER: a block's members arrive in
// order, so the running sum is a sequence and cannot be parallelised over
// tokens -- and must not be, or a chunk would not reproduce T single-token
// calls. It also means `sum` and `raw0` are read and written once for the
// chunk rather than once a token.
__global__ __launch_bounds__(IDX_DIM)
void k_idx_pool_chunk(const float * __restrict__ k_new, float * __restrict__ sum,
                      float * __restrict__ raw0, int pos0, int T,
                      const float * __restrict__ k_norm, float eps,
                      unsigned short * __restrict__ pooled,
                      float * __restrict__ dbg_pooled, float * __restrict__ dbg_roped,
                      const int * __restrict__ dpos) {
    __shared__ float s_v[IDX_DIM];
    __shared__ float s_r[IDX_DIM];
    const int i = threadIdx.x;
    if (dpos) pos0 = *dpos;                               // --hip-graph

    float run  = sum[i];
    float fill = raw0[i];

    for (int t = 0; t < T; ++t) {
        const int p        = pos0 + t;
        const int blk      = p / QSA_RATIO;
        const int n_filled = (p + 1) - blk * QSA_RATIO;
        const bool last    = (t == T - 1);

        const float kv = k_new[(size_t) t * IDX_DIM + i];
        if (blk == 0 && n_filled == 1) fill = kv;     // cell 0 is the fill value
        run = (n_filled == 1) ? kv : run + kv;
        float p_v = (run + (float)(QSA_RATIO - n_filled) * fill) * (1.0f / (float) QSA_RATIO);
        if (last && dbg_pooled) dbg_pooled[i] = p_v;

        s_v[i] = p_v * p_v;
        __syncthreads();
        for (int s = IDX_DIM / 2; s > 0; s >>= 1) {
            if (i < s) s_v[i] += s_v[i + s];
            __syncthreads();
        }
        p_v = p_v * rsqrtf(s_v[0] / (float) IDX_DIM + eps) * k_norm[i];
        s_r[i] = p_v;
        __syncthreads();

        // IMRoPE over the first N_ROT of IDX_DIM dims, position blk*QSA_RATIO
        if (i < N_ROT / 2) {
            const int sect_dims = ROPE_SECTIONS[0] + ROPE_SECTIONS[1] + ROPE_SECTIONS[2] + ROPE_SECTIONS[3];
            const int sector = i % sect_dims;
            const bool is_e = !((sector % 3 == 1 && sector < 3 * ROPE_SECTIONS[1]) ||
                                (sector % 3 == 2 && sector < 3 * ROPE_SECTIONS[2]) ||
                                (sector % 3 == 0 && sector < 3 * ROPE_SECTIONS[0]));
            const float base = is_e ? 0.0f : (float)(blk * QSA_RATIO);
            const float ts   = __powf(ROPE_FREQ_BASE, -2.0f / (float) N_ROT);
            const float th   = base * __powf(ts, (float) i);
            const float c = __cosf(th), s = __sinf(th);
            const float x0 = s_r[i], x1 = s_r[i + N_ROT / 2];
            s_r[i]              = x0 * c - x1 * s;
            s_r[i + N_ROT / 2]  = x0 * s + x1 * c;
        }
        __syncthreads();

        if (last && dbg_roped) dbg_roped[i] = s_r[i];
        pooled[(size_t) blk * IDX_DIM + i] = fk_f32_to_bf16(s_r[i]);
        __syncthreads();
    }

    sum[i]  = run;
    raw0[i] = fill;
}

// scores[b] = sum over the FOUR QUERY heads of relu(q_h . kpool[b]).
// One 32-lane wave per pooled block; a lane owns 4 of the 128 dims (one
// 8-byte load, fully coalesced across the wave) and keeps its four query
// elements per head in registers, so the pooled-key array is streamed once
// and nothing else is touched.
#define IDXS_THREADS 256
// ------------------------------------------- the QSA row block (item 3) --
//
// The four kernels below used to be launched once per query row from the host
// loop in decode_graph.cpp: ~1 000 launches a QSA layer at T = 256, ~12 000 a
// chunk, 0.19 ms a prompt token a card of pure issue cost. They now take a ROW
// AXIS. Every per-row quantity is a closed form of the row's POSITION -- n_kv,
// n_blocks, the incomplete tail, the budget, whether the top-k is the identity
// -- so nothing is uploaded per row and the host enqueues one launch a stage
// for the whole block.
//
// NUMERICS ARE UNTOUCHED BY CONSTRUCTION: a row reads and writes its own slice
// of every buffer, the arithmetic per row is the code that was there before,
// and no reduction crosses rows. What changes is the grid.
struct QsaRowsD {
    int    pos0;         // position of row 0 of this block
    int    ratio;        // the layer's compress ratio
    int    budget;       // IDX_TOP_K + ratio - 1
    size_t blk_stride;   // row pitch of blk_scores
    size_t cell_stride;  // row pitch of cell_scores
    size_t sel_stride;   // row pitch of the selection
    size_t cand_stride;  // row pitch of the radix candidate lists
    size_t part_stride;  // row pitch of the attention partials
    size_t ml_stride;    // row pitch of mbuf / lbuf
    // --hip-graph: when set, pos0 is READ from here at the top of every
    // kernel instead of taken from the launch, so a captured launch is
    // position-free. The grids are then sized for the position CLASS's
    // largest n_kv and every kernel below already returns (or strides) past
    // its own row's extent -- the same guards a row block relies on.
    const int * dpos;
};
__device__ __forceinline__ void qsa_resolve(QsaRowsD & j) { if (j.dpos) j.pos0 = *j.dpos; }
__device__ __forceinline__ int qsa_n_kv(const QsaRowsD & j, int t) { return j.pos0 + t + 1; }
__device__ __forceinline__ int qsa_n_blocks(const QsaRowsD & j, int t) {
    return (qsa_n_kv(j, t) + j.ratio - 1) / j.ratio;
}
// "the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus
// the tail" (qwen4exp.cpp:613). min() of that and the cache is BOTH the width
// the radix select is asked for and the count the attention reads, and when it
// equals n_kv the selection is the identity and the select does not run.
__device__ __forceinline__ int qsa_n_sel(const QsaRowsD & j, int t) {
    const int n = qsa_n_kv(j, t);
    return n < j.budget ? n : j.budget;
}

__global__ __launch_bounds__(IDXS_THREADS)
void k_idx_scan(const unsigned short * __restrict__ pooled, const float * __restrict__ q,
                float * __restrict__ scores, QsaRowsD j) {
    qsa_resolve(j);
    const int row  = blockIdx.y;
    const int n_blocks = qsa_n_blocks(j, row);
    q      += (size_t) row * IDX_N_HEADS * IDX_DIM;
    scores += (size_t) row * j.blk_stride;
    const int lane = threadIdx.x & (WAVE - 1);
    const int sub  = threadIdx.x / WAVE;
    const int nsub = IDXS_THREADS / WAVE;

    float qr[IDX_N_HEADS][4];
#pragma unroll
    for (int h = 0; h < IDX_N_HEADS; ++h) {
        const float4 a = *reinterpret_cast<const float4 *>(q + h * IDX_DIM + lane * 4);
        qr[h][0] = a.x; qr[h][1] = a.y; qr[h][2] = a.z; qr[h][3] = a.w;
    }

    for (int b = blockIdx.x * nsub + sub; b < n_blocks; b += gridDim.x * nsub) {
        const uint2 raw = *reinterpret_cast<const uint2 *>(pooled + (size_t) b * IDX_DIM + lane * 4);
        float kv[4];
        kv[0] = fk_bf16_to_f32((unsigned short)(raw.x & 0xffffu));
        kv[1] = fk_bf16_to_f32((unsigned short)(raw.x >> 16));
        kv[2] = fk_bf16_to_f32((unsigned short)(raw.y & 0xffffu));
        kv[3] = fk_bf16_to_f32((unsigned short)(raw.y >> 16));

        float total = 0.0f;
#pragma unroll
        for (int h = 0; h < IDX_N_HEADS; ++h) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) acc = fmaf(qr[h][j], kv[j], acc);
            acc = wave_sum(acc);
            total += fmaxf(acc, 0.0f);                    // ggml_relu, per head
        }
        if (lane == 0) scores[b] = total;
    }
}

// set_input_qsa's per-block bias, inline: the incomplete tail is always
// visible (+1e9), a block that could not be pooled is -inf, everything else
// 0. Computing it here removed the host->device copy the layer body used to
// do once per QSA layer.
// `sel_identity`: when the budget (indexer_top_k + r - 1) is at least the
// whole cache, the top-k is the identity -- every visible cell is selected --
// so the selection is written here and the radix select is not launched at
// all. At the depths L0 step 2 runs (ctx 512 against a 2 051-cell budget)
// that is ALWAYS the case, and the radix select was a 1 024-thread workgroup
// doing four passes of LDS histograms and scans over 70 scores.
__global__ void k_qsa_expand(const float * __restrict__ blk_scores,
                             float * __restrict__ cell, int * __restrict__ sel,
                             QsaRowsD jr) {
    qsa_resolve(jr);
    const int row = blockIdx.y;
    const int j   = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_kv      = qsa_n_kv(jr, row);
    if (j >= n_kv) return;
    const int ratio     = jr.ratio;
    const int q_pos     = jr.pos0 + row;
    const int tail_start = ((q_pos + 1) / ratio) * ratio;
    blk_scores += (size_t) row * jr.blk_stride;
    cell       += (size_t) row * jr.cell_stride;
    sel        += (size_t) row * jr.sel_stride;
    if (qsa_n_sel(jr, row) >= n_kv) sel[j] = j;       // the identity selection
    const int b      = j / ratio;
    const int filled = min(ratio, n_kv - b * ratio);
    const float bias = (b * ratio >= tail_start) ? 1e9f
                     : (filled < ratio ? -INFINITY : 0.0f);
    cell[j] = (j <= q_pos) ? blk_scores[b] + bias : -INFINITY;
}

// m3_attn.hip's radix_select_11_7_7_7_1wg, ratio fixed at 1 (the reference
// selects over TOKENS). One launch, one workgroup, no host round trip: the
// kernel it replaced in M3 bisected from the host and its 2 ms was 40
// synchronisations, not data.
#define TOPK_THREADS   1024
#define TOPK_HI_BITS   11
#define TOPK_HI_BINS   (1 << TOPK_HI_BITS)
#define TOPK_HI_SHIFT  (32 - TOPK_HI_BITS)
#define TOPK_LO_BITS   7
#define TOPK_LO_BINS   (1 << TOPK_LO_BITS)
#define TOPK_LO_PASSES 3
static_assert(TOPK_HI_BINS == 2 * TOPK_THREADS, "hi-bin scan assumes 2 bins per thread");
static_assert(TOPK_LO_BITS * TOPK_LO_PASSES + TOPK_HI_BITS == 32, "radix passes must cover 32 bits");

__device__ __forceinline__ uint32_t ord_key(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    return (c.u & 0x80000000u) ? ~c.u : (c.u | 0x80000000u);
}

__global__ __launch_bounds__(TOPK_THREADS)
void k_topk_radix_select(const float * __restrict__ scores,
                         uint32_t * __restrict__ cand_key, int * __restrict__ cand_idx,
                         int * __restrict__ out, QsaRowsD jr) {
    __shared__ uint32_t s_hist[TOPK_HI_BINS];
    __shared__ uint32_t s_scan[TOPK_THREADS];
    __shared__ int      s_found;
    __shared__ uint32_t s_above, s_prefix;
    __shared__ int      s_need, s_cand, s_out, s_tie;

    // One workgroup a row. A row whose budget covers its whole cache already
    // has its selection -- k_qsa_expand wrote the identity -- and returns
    // before the first __syncthreads(), which is uniform across the workgroup
    // and therefore safe.
    qsa_resolve(jr);
    const int row  = blockIdx.x;
    const int n    = qsa_n_kv(jr, row);
    const int kw   = qsa_n_sel(jr, row);
    if (kw >= n) return;
    scores   += (size_t) row * jr.cell_stride;
    out      += (size_t) row * jr.sel_stride;
    cand_key += (size_t) row * jr.cand_stride;
    cand_idx += (size_t) row * jr.cand_stride;

    const int t = threadIdx.x;
    const int k = kw < n ? kw : n;

    for (int i = t; i < TOPK_HI_BINS; i += TOPK_THREADS) s_hist[i] = 0u;
    if (t == 0) { s_cand = 0; s_out = 0; s_tie = 0; s_found = 0; s_above = 0u; }
    __syncthreads();

    for (int i = t; i < n; i += TOPK_THREADS) atomicAdd(&s_hist[ord_key(scores[i]) >> TOPK_HI_SHIFT], 1u);
    __syncthreads();

    s_scan[t] = s_hist[TOPK_HI_BINS - 1 - 2 * t] + s_hist[TOPK_HI_BINS - 2 - 2 * t];
    __syncthreads();
    for (int off = 1; off < TOPK_THREADS; off <<= 1) {
        uint32_t v = (t >= off) ? s_scan[t - off] : 0u;
        __syncthreads();
        s_scan[t] += v;
        __syncthreads();
    }
    {
        const uint32_t incl = s_scan[t];
        const uint32_t excl = (t == 0) ? 0u : s_scan[t - 1];
        if (incl >= (uint32_t) k && excl < (uint32_t) k) { s_found = t; s_above = excl; }
    }
    __syncthreads();
    if (t == 0) {
        uint32_t above = s_above;
        int bin = TOPK_HI_BINS - 1 - 2 * s_found;
        const uint32_t c0 = s_hist[bin];
        if (above + c0 < (uint32_t) k) { above += c0; bin -= 1; }
        s_prefix = ((uint32_t) bin) << TOPK_HI_SHIFT;
        s_need   = k - (int) above;
    }
    __syncthreads();

    const uint32_t hi = s_prefix >> TOPK_HI_SHIFT;
    for (int i = t; i < n; i += TOPK_THREADS) {
        const uint32_t u = ord_key(scores[i]);
        const uint32_t b = u >> TOPK_HI_SHIFT;
        if (b > hi) {
            const int pos = atomicAdd(&s_out, 1);
            if (pos < k) out[pos] = i;
        } else if (b == hi) {
            const int c = atomicAdd(&s_cand, 1);
            if (c < n) { cand_key[c] = u; cand_idx[c] = i; }
        }
    }
    __syncthreads();

    for (int p = 0; p < TOPK_LO_PASSES; ++p) {
        const int shift = TOPK_LO_BITS * (TOPK_LO_PASSES - 1 - p);
        const uint32_t himask = ~((1u << (shift + TOPK_LO_BITS)) - 1u);
        for (int i = t; i < TOPK_LO_BINS; i += TOPK_THREADS) s_hist[i] = 0u;
        __syncthreads();
        const int ncand = s_cand;
        const uint32_t pref = s_prefix;
        for (int i = t; i < ncand; i += TOPK_THREADS) {
            const uint32_t u = cand_key[i];
            if ((u & himask) == (pref & himask)) atomicAdd(&s_hist[(u >> shift) & (TOPK_LO_BINS - 1)], 1u);
        }
        __syncthreads();
        if (t < TOPK_LO_BINS) s_scan[t] = s_hist[TOPK_LO_BINS - 1 - t];
        __syncthreads();
        for (int off = 1; off < TOPK_LO_BINS; off <<= 1) {
            const uint32_t v = (t < TOPK_LO_BINS && t >= off) ? s_scan[t - off] : 0u;
            __syncthreads();
            if (t < TOPK_LO_BINS) s_scan[t] += v;
            __syncthreads();
        }
        if (t < TOPK_LO_BINS) {
            const uint32_t incl = s_scan[t];
            const uint32_t excl = (t == 0) ? 0u : s_scan[t - 1];
            if (incl >= (uint32_t) s_need && excl < (uint32_t) s_need) { s_found = t; s_above = excl; }
        }
        __syncthreads();
        if (t == 0) {
            const int bin = TOPK_LO_BINS - 1 - s_found;
            s_prefix |= ((uint32_t) bin) << shift;
            s_need   -= (int) s_above;
        }
        __syncthreads();
    }

    {
        const uint32_t thr = s_prefix;
        const int need = s_need, ncand = s_cand;
        for (int i = t; i < ncand; i += TOPK_THREADS) {
            const uint32_t u = cand_key[i];
            bool take = (u > thr);
            if (!take && u == thr) take = (atomicAdd(&s_tie, 1) < need);
            if (take) {
                const int pos = atomicAdd(&s_out, 1);
                if (pos < k) out[pos] = cand_idx[i];
            }
        }
    }
}

// m3_attn.hip's flash_split_wave_per_head, with the gate taken out of the
// combine (the reference gates after cb("kqv_out")).
#define ATTN_CHUNK   64
#define ATTN_DPL     (HEAD_DIM / WAVE)                    // 8 dims per lane
#define ATTN_WAVES   GQA_GROUP                            // 12 query heads share a KV head
#define ATTN_THREADS (ATTN_WAVES * WAVE)                  // 384
#define ATTN_UNR     4
#define ATTN_NEG_INF (-3.0e38f)
static_assert(ATTN_CHUNK % ATTN_UNR == 0, "chunk must divide into unrolled groups");
static_assert(HEAD_DIM % 32 == 0, "q8_0 blocks must tile the head dim");

__device__ __forceinline__ void q8_bytes(int lo, int hi, float sc, float * out8) {
#pragma unroll
    for (int i = 0; i < 4; ++i) out8[i]     = sc * (float)(int8_t)((lo >> (8 * i)) & 0xff);
#pragma unroll
    for (int i = 0; i < 4; ++i) out8[4 + i] = sc * (float)(int8_t)((hi >> (8 * i)) & 0xff);
}

__global__ __launch_bounds__(ATTN_THREADS)
void k_attn_flash_split(const int8_t * __restrict__ Kqs, const __half * __restrict__ Ksc,
                        const int8_t * __restrict__ Vqs, const __half * __restrict__ Vsc,
                        const float * __restrict__ Q, const int * __restrict__ sel,
                        float * __restrict__ partial, float * __restrict__ mbuf,
                        float * __restrict__ lbuf, QsaRowsD jr) {
    __shared__ float sh_p[ATTN_WAVES][ATTN_CHUNK];
    __shared__ int   sh_tok[ATTN_CHUNK];

    // grid.z is the query row. Rows of a block have different n_sel -- the
    // budget is min(n_kv, indexer_top_k + r - 1) and n_kv grows by one a row
    // -- so the grid is sized for the LAST row and a workgroup past its own
    // row's chunk count returns. The return is uniform over the workgroup.
    qsa_resolve(jr);
    const int row   = blockIdx.z;
    const int n_sel = qsa_n_sel(jr, row);
    if (blockIdx.x * ATTN_CHUNK >= n_sel) return;
    sel     += (size_t) row * jr.sel_stride;
    Q       += (size_t) row * N_Q_HEADS * HEAD_DIM;
    partial += (size_t) row * jr.part_stride;
    mbuf    += (size_t) row * jr.ml_stride;
    lbuf    += (size_t) row * jr.ml_stride;

    const int chunk = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int w     = threadIdx.x / WAVE;
    const int lane  = threadIdx.x % WAVE;
    const int h     = kvh * GQA_GROUP + w;
    const int base  = chunk * ATTN_CHUNK;
    const int d0    = lane * ATTN_DPL;

    for (int j = threadIdx.x; j < ATTN_CHUNK; j += ATTN_THREADS) {
        const int jj = base + j;
        sh_tok[j] = (jj < n_sel) ? sel[jj] : 0;           // token 0: a safe dummy
    }
    __syncthreads();

    float qr[ATTN_DPL];
    {
        const float4 * q4 = reinterpret_cast<const float4 *>(Q + (size_t) h * HEAD_DIM + d0);
        const float4 a = q4[0], b = q4[1];
        qr[0]=a.x; qr[1]=a.y; qr[2]=a.z; qr[3]=a.w;
        qr[4]=b.x; qr[5]=b.y; qr[6]=b.z; qr[7]=b.w;
    }
    const float qscale = rsqrtf((float) HEAD_DIM);
    const int sc_off = d0 >> 5;

    for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR) {
        int2  raw[ATTN_UNR];
        float sc[ATTN_UNR];
#pragma unroll
        for (int u = 0; u < ATTN_UNR; ++u) {
            const size_t row = ((size_t) sh_tok[j0 + u] * N_KV_HEADS + kvh);
            raw[u] = *reinterpret_cast<const int2 *>(Kqs + row * HEAD_DIM + d0);
            sc[u]  = __half2float(Ksc[row * (HEAD_DIM / 32) + sc_off]);
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNR; ++u) {
            float kd[ATTN_DPL];
            q8_bytes(raw[u].x, raw[u].y, sc[u], kd);
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < ATTN_DPL; ++i) acc = fmaf(qr[i], kd[i], acc);
            acc = wave_sum(acc);
            if (lane == 0)
                sh_p[w][j0 + u] = (base + j0 + u < n_sel) ? acc * qscale : ATTN_NEG_INF;
        }
    }
    __syncthreads();

    float mx = ATTN_NEG_INF, sum = 0.0f;
    for (int j = lane; j < ATTN_CHUNK; j += WAVE) mx = fmaxf(mx, sh_p[w][j]);
#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, WAVE));
    for (int j = lane; j < ATTN_CHUNK; j += WAVE) {
        const float s = sh_p[w][j];
        const float e = (s <= ATTN_NEG_INF) ? 0.0f : __expf(s - mx);
        sh_p[w][j] = e;
        sum += e;
    }
    sum = wave_sum(sum);
    if (lane == 0) {
        mbuf[(size_t) chunk * N_Q_HEADS + h] = mx;
        lbuf[(size_t) chunk * N_Q_HEADS + h] = sum;
    }
    __syncthreads();

    float acc[ATTN_DPL];
#pragma unroll
    for (int i = 0; i < ATTN_DPL; ++i) acc[i] = 0.0f;
    for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR) {
        int2  raw[ATTN_UNR];
        float sc[ATTN_UNR];
#pragma unroll
        for (int u = 0; u < ATTN_UNR; ++u) {
            const size_t row = ((size_t) sh_tok[j0 + u] * N_KV_HEADS + kvh);
            raw[u] = *reinterpret_cast<const int2 *>(Vqs + row * HEAD_DIM + d0);
            sc[u]  = __half2float(Vsc[row * (HEAD_DIM / 32) + sc_off]);
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNR; ++u) {
            float vd[ATTN_DPL];
            q8_bytes(raw[u].x, raw[u].y, sc[u], vd);
            const float p = sh_p[w][j0 + u];
#pragma unroll
            for (int i = 0; i < ATTN_DPL; ++i) acc[i] = fmaf(p, vd[i], acc[i]);
        }
    }
    float * dst = partial + ((size_t) chunk * N_Q_HEADS + h) * HEAD_DIM + d0;
    reinterpret_cast<float4 *>(dst)[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    reinterpret_cast<float4 *>(dst)[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

// The combine now also applies the sigmoid output gate, writing BOTH the
// ungated result (the reference's kqv_out / attn_pregate) and the gated one
// (attn_gated). That removes the separate sigmoid and multiply launches
// without losing either tap.
__global__ __launch_bounds__(HEAD_DIM)
void k_attn_combine(const float * __restrict__ partial, const float * __restrict__ mbuf,
                    const float * __restrict__ lbuf, const float * __restrict__ gsig,
                    float * __restrict__ out, float * __restrict__ out_gated,
                    QsaRowsD jr) {
    qsa_resolve(jr);
    const int row       = blockIdx.y;
    const int n_chunks  = (qsa_n_sel(jr, row) + ATTN_CHUNK - 1) / ATTN_CHUNK;
    partial   += (size_t) row * jr.part_stride;
    mbuf      += (size_t) row * jr.ml_stride;
    lbuf      += (size_t) row * jr.ml_stride;
    gsig      += (size_t) row * N_Q_HEADS * HEAD_DIM;
    out       += (size_t) row * N_Q_HEADS * HEAD_DIM;
    out_gated += (size_t) row * N_Q_HEADS * HEAD_DIM;
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float M = ATTN_NEG_INF;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, mbuf[(size_t) c * N_Q_HEADS + h]);
    float acc = 0.0f, den = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = mbuf[(size_t) c * N_Q_HEADS + h];
        if (m <= ATTN_NEG_INF) continue;
        const float s = __expf(m - M);
        den += s * lbuf[(size_t) c * N_Q_HEADS + h];
        acc += s * partial[((size_t) c * N_Q_HEADS + h) * HEAD_DIM + d];
    }
    const float v = den > 0.0f ? acc / den : 0.0f;
    out[(size_t) h * HEAD_DIM + d]       = v;
    out_gated[(size_t) h * HEAD_DIM + d] = v * gsig[(size_t) h * HEAD_DIM + d];
}

// Greedy sampling on the device. Two stages so the 248 320 logits are read
// once by a full grid: each workgroup reduces its slice to one (value, index)
// pair, then one workgroup reduces those. Ties go to the LOWEST index, which
// is what a scan of ggml's logits with `>` does.
#define ARGMAX_THREADS 256
#define ARGMAX_BLOCKS  256

__device__ __forceinline__ void argmax_reduce(float * sv, int * si, int t, int n) {
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            const bool better = sv[t + stride] > sv[t] ||
                                (sv[t + stride] == sv[t] && si[t + stride] < si[t]);
            if (better) { sv[t] = sv[t + stride]; si[t] = si[t + stride]; }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(ARGMAX_THREADS)
void k_argmax_partial(const float * __restrict__ x, int n,
                      float * __restrict__ pv, int * __restrict__ pi) {
    __shared__ float sv[ARGMAX_THREADS];
    __shared__ int   si[ARGMAX_THREADS];
    const int t = threadIdx.x;
    float best = -INFINITY; int bi = 0;
    for (int i = blockIdx.x * ARGMAX_THREADS + t; i < n; i += ARGMAX_BLOCKS * ARGMAX_THREADS) {
        const float v = x[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    sv[t] = best; si[t] = bi;
    __syncthreads();
    argmax_reduce(sv, si, t, ARGMAX_THREADS);
    if (t == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
}

__global__ __launch_bounds__(ARGMAX_BLOCKS)
void k_argmax_final(const float * __restrict__ pv, const int * __restrict__ pi,
                    int * __restrict__ out) {
    __shared__ float sv[ARGMAX_BLOCKS];
    __shared__ int   si[ARGMAX_BLOCKS];
    const int t = threadIdx.x;
    sv[t] = pv[t]; si[t] = pi[t];
    __syncthreads();
    argmax_reduce(sv, si, t, ARGMAX_BLOCKS);
    if (t == 0) out[0] = si[0];
}

// ================================================ --hip-graph helpers =====
//
// The last node of every captured per-card body: the device-resident position
// moves on by one, so the NEXT replay of the same graph reads the next
// position without a host write. The host only uploads the position when its
// shadow says the device copy is stale (a prefill chunk ran eagerly, a serving
// rollback moved pos) -- see DecodeRunner::seg_begin.
__global__ void k_pos_advance(int * __restrict__ pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pos[0] += 1;
}

// --prequeue-gate: a diagnostic, not an engine path. The first card is the
// only one whose token is NOT pre-queued (cards 1 and 2 wait on a boundary
// event while the host enqueues their whole body), so its per-op profile
// intervals can be the HOST's issue period rather than the kernels' time
// (README "Step 3b"). This kernel holds the card's stream until the host has
// enqueued the whole token, making card 0 pre-queued like the others; if its
// small-GEMV bandwidth then reads like cards 1-2, the gap was issue-bound.
// `flag` is host-mapped coherent memory the host bumps to `want`; the wall
// clock bounds the wait so a full hardware queue (the host blocking on an
// enqueue while the card waits on the host) can only cost `limit` ticks,
// never a hang -- `timed_out` then says the number is not what it claims.
__global__ void k_prequeue_gate(const volatile unsigned * flag, unsigned want,
                                volatile unsigned * timed_out,
                                unsigned long long limit) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned long long t0 = wall_clock64();
    while (*flag < want) {
        if (wall_clock64() - t0 > limit) { *timed_out = 1u; break; }
        __builtin_amdgcn_s_sleep(8);
    }
}

// ========================================================== backend ======

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// THE STEP-3 CRASH, and why it looked like a cache-sizing bug.
//
// hipMalloc, hipMemcpy and a kernel launch all act on the CURRENT device, and
// the current device is process-wide state -- not something a stream carries.
// GpuBackend set it once in its constructor, so after `make_gpu_backend(0..2)`
// the current device was 2 and EVERY later allocation went there, whichever
// backend was asked. Placement then filled one card with all 48 layers and
// died at layer 18: 24 GB / ~1.34 GB a layer is 17.9. Step 2 never saw it
// because with one backend the constructor's hipSetDevice was also the right
// one for everything after it.
//
// So every entry point that allocates, copies or launches now ensures its own
// device first. The check is a compare against a process-global, so the cost
// is one predictable branch on the ~1 160 launches a token that do not switch.
inline int & cur_device_slot() { static int d = -1; return d; }
inline void dev_ensure(int d) {
    int & cur = cur_device_slot();
    if (cur != d) { HIP_CHECK(hipSetDevice(d)); cur = d; }
}

// A reduction workgroup wide enough to walk ne0 in a few steps, capped at the
// hardware maximum; the LDS is sized to match.
inline int red_threads(int ne0) {
    int t = WAVE;
    while (t < ne0 && t < 1024) t <<= 1;
    return t;
}

// ------------------------------------------------------------ profiling --
//
// One hipEvent recorded BEFORE each op plus one after the body: on a single
// stream the interval between consecutive events is that op's wall time on
// the device timeline INCLUDING the bubble in front of it, which is what
// separates "this kernel is slow" from "the GPU was waiting for the host".
// Nothing here synchronises until the token is over.
enum ProfClass {
    PC_TRUNK = 0,   // trunk GEMVs (+ their split reduce)
    PC_NORM,        // rms / l2 / gated rms
    PC_ELEM,        // elementwise, hc mix/combine, conv, PLE gate
    PC_GDN,         // the delta-rule step
    PC_QSA,         // indexer scan / top-k / cache writes / attention
    PC_ROUTER,
    PC_EXP_GATE_UP,
    PC_EXP_DOWN,
    // Only at T > 1, and only for the stages --expert-gather has turned on:
    // the counting sort plus the row-gather kernels, charged apart from the
    // per-assignment path above so the profile says how much of the expert
    // time the amortisation is actually carrying.
    PC_EXP_GATHER,
    PC_EXP_COMBINE,
    // The batched query-row stage of a QSA layer: scan, expand, radix select,
    // attention split and combine for a whole block of rows. Apart from
    // PC_QSA, which keeps the per-chunk work (cache writes, pooling,
    // qk_post, rope) that is not per query row.
    PC_QSA_ROWS,
    PC_COPY,        // device-to-device copies
    PC_BOUNDARY,    // the hyper-connection residual crossing to the next card
    PC_ARGMAX,      // greedy sampling on the device
    // --hip-graph: one replayed per-card body, launch to the next event --
    // i.e. the card's whole layer range (and the head on the last card) as
    // ONE interval. prof_graph_replay_us is this class.
    PC_GRAPH,
    // Only with a deferred profile (--prefill-pipeline): the interval from
    // the end of one chunk's enqueued work on this card to the first op of
    // the next. It IS this card's idle time, so `total - gap` is its busy
    // time and three cards whose busy times add up to more than the wall
    // clock are overlapping.
    PC_GAP,
    // DeepSeek-V4 (ds4_gpu.inc). Appended, and prof_report prints them only
    // when non-zero, so a qwen4exp profile reads exactly as it did.
    PC_DS4_TRUNK_L0,    // Q8_0 / Q6_K / BF16 / F32 trunk GEMVs through k_gemm_batch
    PC_DS4_TRUNK_KQ,    // Q5_K / Q4_K trunk and lm_head through k_ds4_gemv
    PC_DS4_EXP_GU,      // routed experts, gate + up
    PC_DS4_EXP_DOWN,    // routed experts, down
    PC_DS4_EXP_MISS,    // (--miss-stage 0 --profile) the slots read over PCIe, split out
    PC_DS4_MISS_WAIT,   // the main stream waiting on the side stream's miss copy
    PC_DS4_LID,         // the lightning indexer scan
    PC_DS4_TOPK,
    PC_DS4_ATTN,        // attention split + combine
    PC_DS4_ATTN_MISC,   // rope / de-rope / FWHT / f16 stores / compressor pools and rings
    PC_DS4_HC,          // the four-stream mix: split + Sinkhorn, weighted sum, post
    PC_DS4_MOE_TAIL,    // swiglu clamp, the weighted sum of the six
    PC_DS4_UPSTREAM,    // a card waiting for the previous card's share of the token
    PC_DS4_ADAPT,       // --adapt: counter read-back, table edits, the copy join (between tokens)
    PC_N
};
const char * prof_name(int c) {
    switch (c) {
        case PC_TRUNK:        return "trunk_gemv";
        case PC_NORM:         return "norm";
        case PC_ELEM:         return "elem_hc";
        case PC_GDN:          return "gdn_step";
        case PC_QSA:          return "qsa";
        case PC_ROUTER:       return "router";
        case PC_EXP_GATE_UP:  return "expert_gate_up";
        case PC_EXP_DOWN:     return "expert_down";
        case PC_EXP_GATHER:   return "expert_gather";
        case PC_EXP_COMBINE:  return "expert_combine";
        case PC_QSA_ROWS:     return "qsa_rows";
        case PC_COPY:         return "copy";
        case PC_BOUNDARY:     return "boundary_p2p";
        case PC_ARGMAX:       return "argmax";
        case PC_GRAPH:        return "graph_replay";
        case PC_GAP:          return "gap_idle";
        case PC_DS4_TRUNK_L0:  return "ds4_trunk_l0_kernel";
        case PC_DS4_TRUNK_KQ:  return "ds4_trunk_kquant";
        case PC_DS4_EXP_GU:    return "ds4_expert_gate_up";
        case PC_DS4_EXP_DOWN:  return "ds4_expert_down";
        case PC_DS4_EXP_MISS:  return "ds4_expert_host_mapped";
        case PC_DS4_MISS_WAIT: return "ds4_miss_wait";
        case PC_DS4_LID:       return "ds4_indexer_scan";
        case PC_DS4_TOPK:      return "ds4_topk";
        case PC_DS4_ATTN:      return "ds4_attention";
        case PC_DS4_ATTN_MISC: return "ds4_rope_pool_store";
        case PC_DS4_HC:        return "ds4_hc_mix";
        case PC_DS4_MOE_TAIL:  return "ds4_moe_tail";
        case PC_DS4_UPSTREAM:  return "ds4_upstream_wait";
        case PC_DS4_ADAPT:     return "ds4_adapt";
        default:              return "copy";
    }
}
constexpr int PROF_MAX_EVENTS = 8192;

class GpuBackend : public Backend {
    // ds4_gpu.inc's ops launch on this backend's stream through mark().
    friend class Ds4GpuOps;
public:
    explicit GpuBackend(int dev) : dev_(dev) {
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipStreamCreate(&stream_));
        std::snprintf(name_, sizeof(name_), "gpu(device %d)", dev_);
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_iq3s_grid),  iq3s_grid,      sizeof(uint32_t) * 512));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_kvals_iq4nl), kvalues_iq4nl, sizeof(int8_t) * 16));
        HIP_CHECK(hipEventCreate(&ev0_));
        HIP_CHECK(hipEventCreate(&ev1_));
        // A pinned staging ring so a per-token upload is an ASYNC copy on the
        // engine's own stream rather than a blocking hipMemcpy: a pageable
        // hipMemcpyAsync degrades to a synchronous one.
        for (int i = 0; i < STAGE_SLOTS; ++i)
            HIP_CHECK(hipHostMalloc(&stage_[i], stage_bytes_, hipHostMallocDefault));
        // the attention/top-k scratch is sized for the worst case the runner
        // can present; both are small and allocated once
        // One row's worth here; reserve_qsa_rows() grows it to the row block
        // the runner asks for. A row is 811 KB of partials at MAX_SEL, which
        // is why the block is bounded at all.
        alloc_attn_rows(1);
        HIP_CHECK(hipMalloc(&gemv_part_, GEMV_PART_FLOATS * sizeof(float)));
        part_floats_ = GEMV_PART_FLOATS;
        HIP_CHECK(hipMalloc(&gemv_cnt_, GEMV_CNT_ROWS * sizeof(int)));
        HIP_CHECK(hipMemset(gemv_cnt_, 0, GEMV_CNT_ROWS * sizeof(int)));
        cnt_rows_ = GEMV_CNT_ROWS;
        HIP_CHECK(hipMalloc(&amax_v_, ARGMAX_BLOCKS * sizeof(float)));
        HIP_CHECK(hipMalloc(&amax_i_, ARGMAX_BLOCKS * sizeof(int)));
        HIP_CHECK(hipEventCreateWithFlags(&bnd_, hipEventDisableTiming));
        for (int i = 0; i < N_BANKS; ++i)
            HIP_CHECK(hipEventCreateWithFlags(&drained_[i], hipEventDisableTiming));
        for (int i = 0; i < STAGE_SLOTS; ++i)
            HIP_CHECK(hipEventCreateWithFlags(&stage_ev_[i], hipEventDisableTiming));
        // the candidate lists are sized by the CACHE depth, not the budget,
        // and grown on demand by topk_select()
    }
    ~GpuBackend() override {
        dev_ensure(dev_);
        HIP_IGNORE(hipFree(partial_)); HIP_IGNORE(hipFree(mbuf_)); HIP_IGNORE(hipFree(lbuf_));
        HIP_IGNORE(hipFree(cand_key_)); HIP_IGNORE(hipFree(cand_idx_));
        HIP_IGNORE(hipFree(gemv_part_)); HIP_IGNORE(hipFree(gemv_cnt_));
        HIP_IGNORE(hipFree(moe_order_));    HIP_IGNORE(hipFree(moe_tile_exp_));
        HIP_IGNORE(hipFree(moe_tile_off_)); HIP_IGNORE(hipFree(moe_tile_cnt_));
        for (int i = 0; i < STAGE_SLOTS; ++i) {
            if (stage_[i]) HIP_IGNORE(hipHostFree(stage_[i]));
            HIP_IGNORE(hipEventDestroy(stage_ev_[i]));
        }
        HIP_IGNORE(hipFree(amax_v_)); HIP_IGNORE(hipFree(amax_i_));
        if (gate_host_) HIP_IGNORE(hipHostFree((void *) gate_host_));
        for (int i = 0; i < N_BANKS; ++i) HIP_IGNORE(hipEventDestroy(drained_[i]));
        HIP_IGNORE(hipEventDestroy(bnd_)); HIP_IGNORE(hipStreamDestroy(stream_));
        for (auto & e : ev_) HIP_IGNORE(hipEventDestroy(e));
        HIP_IGNORE(hipEventDestroy(ev0_)); HIP_IGNORE(hipEventDestroy(ev1_));
    }

    const char * name() const override { return name_; }
    int device() const override { return dev_; }

    // The destination records nothing of its own: it waits on an event the
    // SOURCE stream recorded, then pulls the bytes on its own stream. No host
    // call, so a boundary costs the P2P time M5 measured and nothing else.
    //
    // The pipeline adds the OTHER half of the handshake. After the copy this
    // device records `drained_[bank]` on its own stream and registers itself
    // as the source's consumer of that bank; the source then waits on that
    // event (boundary_wait_free) before it overwrites the bank for a later
    // chunk. Both halves are stream events -- the host never waits, which is
    // the whole point -- and together they bound the pipeline at two chunks
    // of run-ahead per boundary without any explicit depth counter.
    void boundary_recv(void * dst, Backend & src, const void * src_ptr, size_t bytes,
                       int bank) override {
        GpuBackend & s = static_cast<GpuBackend &>(src);
        if (bank < 0 || bank >= N_BANKS) bank = 0;
        // The event belongs to the SOURCE's device, so record it with that
        // device current; the wait and the copy belong to the destination.
        dev_ensure(s.dev_);
        s.debug_check("boundary source drain");
        HIP_CHECK(hipEventRecord(s.bnd_, s.stream_));
        mark(PC_BOUNDARY);                 // ensures THIS device
        last_op_ = "boundary_recv";
        if (sync_debug_) {
            std::fprintf(stderr, "sync-debug boundary: dev%d[%p .. %p) <- dev%d[%p .. %p) %zu B bank %d\n",
                         dev_, dst, (const void *)((const char *) dst + bytes),
                         s.dev_, src_ptr, (const void *)((const char *) src_ptr + bytes),
                         bytes, bank);
        }
        HIP_CHECK(hipStreamWaitEvent(stream_, s.bnd_, 0));
        HIP_CHECK(hipMemcpyPeerAsync(dst, dev_, src_ptr, s.dev_, bytes, stream_));
        // "this bank has been read out of you": recorded on the DESTINATION's
        // stream, after the copy, and handed to the source.
        HIP_CHECK(hipEventRecord(drained_[bank], stream_));
        s.sink_[bank] = this;
    }

    void boundary_wait_free(int bank) override {
        if (bank < 0 || bank >= N_BANKS) return;
        GpuBackend * c = sink_[bank];
        if (!c) return;                    // nobody has ever consumed it
        dev_ensure(dev_);
        // hipStreamWaitEvent binds to the event's MOST RECENT record at the
        // time of this host call, and the host issues chunks in order, so
        // this is the drain of the last chunk that used this bank.
        HIP_CHECK(hipStreamWaitEvent(stream_, c->drained_[bank], 0));
    }

    void argmax(const float * logits, int n, int * out_id) override {
        mark(PC_ARGMAX);
        last_op_ = "argmax";
        k_argmax_partial<<<ARGMAX_BLOCKS, ARGMAX_THREADS, 0, stream_>>>(logits, n, amax_v_, amax_i_);
        ++launches_;
        k_argmax_final<<<1, ARGMAX_BLOCKS, 0, stream_>>>(amax_v_, amax_i_, out_id);
    }

    void vram_report(FILE * out, const char * tag) override {
        size_t freeb = 0, totalb = 0;
        dev_ensure(dev_);
        HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
        std::fprintf(out, "vram_dev%d_used_gb=%.2f free_gb=%.2f total_gb=%.2f placed_gb=%.2f  (%s)\n",
                     dev_, (totalb - freeb) / 1e9, freeb / 1e9, totalb / 1e9,
                     placed_ / 1e9, tag);
    }

    // Fails EARLY and loudly when a device holds far less than it was asked to
    // place -- the signature of the bug above, where every card but one came
    // out near-empty and the run died a dozen layers later with a message
    // about the wrong thing.
    bool verify_placement(FILE * out) override {
        size_t freeb = 0, totalb = 0;
        dev_ensure(dev_);
        HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
        const double used = (totalb - freeb) / 1e9, want = placed_ / 1e9;
        if (want > 1.0 && used < want * 0.8) {
            std::fprintf(out, "PLACEMENT ERROR dev%d: %.2f GB was placed here but the card "
                              "reports only %.2f GB used -- the allocations went to another "
                              "device (check hipSetDevice before hipMalloc)\n",
                         dev_, want, used);
            return false;
        }
        return true;
    }
    bool is_gpu() const override { return true; }

    float * alloc_f32(size_t n) override { return (float *) alloc_raw(n * sizeof(float)); }
    int *   alloc_i32(size_t n) override { return (int   *) alloc_raw(n * sizeof(int)); }
    void *  alloc_raw(size_t b) override {
        dev_ensure(dev_);
        void * p = nullptr;
        if (b == 0) b = 4;
        HIP_CHECK(hipMalloc(&p, b));
        HIP_CHECK(hipMemset(p, 0, b));
        return p;
    }
    void free_buf(void * p) override { dev_ensure(dev_); HIP_IGNORE(hipFree(p)); }
    // THE STAGING RING GROWS, and that is what lets a chunk be pipelined.
    //
    // A chunk uploads the embedding gather and the PLE gather: T x 2 560 f32,
    // which is 2.6 MB at T = 256 and 5.2 MB at 512. Against a FIXED 64 KB
    // ring both fell through to the blocking `hipMemcpy` below -- and a
    // blocking copy on the default stream drains a blocking stream, so every
    // chunk started by emptying card 0's queue. Nothing would have overlapped.
    // So the ring is sized to the largest upload it is ever handed, once, and
    // a slot is only reused after the event that says its previous copy has
    // left it (with STAGE_SLOTS slots and two uploads a chunk that is four
    // chunks of slack, so the wait is normally already satisfied).
    void ensure_stage(size_t bytes) {
        if (bytes <= stage_bytes_) return;
        HIP_CHECK(hipStreamSynchronize(stream_));   // once, before the swap
        size_t want = stage_bytes_;
        while (want < bytes) want <<= 1;
        for (int i = 0; i < STAGE_SLOTS; ++i) {
            if (stage_[i]) HIP_IGNORE(hipHostFree(stage_[i]));
            HIP_CHECK(hipHostMalloc(&stage_[i], want, hipHostMallocDefault));
            stage_used_[i] = false;
        }
        stage_bytes_ = want;
    }
    void upload(void * d, const void * s, size_t b) override {
        dev_ensure(dev_);
        ensure_stage(b);
        const int sl = stage_next_;
        stage_next_ = (stage_next_ + 1) % STAGE_SLOTS;
        if (stage_used_[sl]) HIP_CHECK(hipEventSynchronize(stage_ev_[sl]));
        std::memcpy(stage_[sl], s, b);
        HIP_CHECK(hipMemcpyAsync(d, stage_[sl], b, hipMemcpyHostToDevice, stream_));
        HIP_CHECK(hipEventRecord(stage_ev_[sl], stream_));
        stage_used_[sl] = true;            // stream-ordered, does NOT block the host
    }
    void download(void * d, const void * s, size_t b) override {
        dev_ensure(dev_);
        debug_check("download()");
        ++blocking_;                       // the host reads it: this always blocks
        HIP_CHECK(hipStreamSynchronize(stream_));
        HIP_CHECK(hipMemcpy(d, s, b, hipMemcpyDeviceToHost));
    }
    // A serving checkpoint's copy: stream-ordered, never waited on (see
    // decode_backend.h). It is NOT counted in `blocking_` because it does not
    // block -- prof_host_syncs_per_token must stay the number it measures.
    void download_async(void * d, const void * s, size_t b) override {
        dev_ensure(dev_);
        debug_check("download_async()");
        HIP_CHECK(hipMemcpyAsync(d, s, b, hipMemcpyDeviceToHost, stream_));
    }
    // Pinned, or the copy above silently becomes a synchronous one. A failure
    // is not fatal: the caller falls back to pageable memory and pays the
    // blocking copy, which is what this engine did before there was a choice.
    void * alloc_pinned(size_t bytes) override {
        dev_ensure(dev_);
        void * p = nullptr;
        if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess) return nullptr;
        return p;
    }
    void free_pinned(void * p) override { if (p) HIP_IGNORE(hipHostFree(p)); }
    void sync() override {
        dev_ensure(dev_);
        debug_check("sync()");
        ++blocking_;
        HIP_CHECK(hipStreamSynchronize(stream_));
    }

    const void * place(const void * host, size_t bytes, const char * what) override {
        dev_ensure(dev_);                  // or the bytes land on another card
        void * p = nullptr;
        const hipError_t e = hipMalloc(&p, bytes);
        if (e != hipSuccess) {
            size_t fb = 0, tb = 0;
            HIP_IGNORE(hipMemGetInfo(&fb, &tb));
            std::fprintf(stderr, "out of VRAM placing %s (%.2f MB) on device %d: %s\n"
                                 "  this card: %.2f GB free of %.2f GB, %.2f GB placed here so far\n",
                         what, bytes / 1e6, dev_, hipGetErrorString(e),
                         fb / 1e9, tb / 1e9, placed_ / 1e9);
            std::abort();
        }
        HIP_CHECK(hipMemcpy(p, host, bytes, hipMemcpyHostToDevice));
        placed_ += bytes;
        return p;
    }
    size_t placed_bytes() const override { return placed_; }

    // There is exactly ONE GEMV path on the device: everything goes through
    // k_gemm_batch, so a single-matrix call is a batch of one and a decode
    // token is a chunk of one.
    void gemv(const Mat & W, const float * x, float * y) override {
        GemvJob j; j.W = &W; j.out = y;
        gemv_batch(&j, 1, x, -1, GX_PLAIN, nullptr, 1);
    }

    void gemv_batch(const GemvJob * jobs, int n, const float * x, int group,
                    int x_mode, const float * x2, int T) override {
        gemv_batch_impl(jobs, n, x, group, x_mode, x2, T, nullptr, nullptr);
    }

    // build_hc_mix's gate projection and the collapse that consumes it. The
    // fused form needs the decode token's geometry (T == 1, no split, hc_up's
    // exact row count); anything else falls back to the two launches, which is
    // what the base class does and what the CPU backend always does.
    void gemv_hc_gate_collapse(const GemvJob & up, const float * lo, const float * xn,
                               float * out_mixed, int group, int T) override {
        const bool fused = gemv_batch_impl(&up, 1, lo, group, GX_PLAIN, nullptr, T,
                                           gemv_fuse_collapse_ ? out_mixed : nullptr, xn);
        if (!fused) hc_collapse(xn, up.out, out_mixed, T);
    }

    // Returns true when `collapse_out` was folded into the launch.
    bool gemv_batch_impl(const GemvJob * jobs, int n, const float * x, int group,
                         int x_mode, const float * x2, int T,
                         float * collapse_out, const float * collapse_xn) {
        GemvBatchD b{};
        b.n = n < GEMV_BATCH_MAX ? n : GEMV_BATCH_MAX;
        b.K = (int) jobs[0].W->K;
        b.x_mode = x_mode;
        size_t bytes = 0;
        int off = 0;
        b.has_q6k = 0;
        for (int i = 0; i < b.n; ++i) {
            const Mat & W = *jobs[i].W;
            if ((int) W.K != b.K) { std::fprintf(stderr, "gemv_batch: %s has a different K\n", W.name); std::abort(); }
            b.d[i].base       = W.base;
            b.d[i].out        = jobs[i].out;
            // A QUANTISED row is addressed in BYTES, an f32/bf16 row in
            // ELEMENTS -- see the two pointer casts in k_gemv_batch. This
            // used to test only for Q8_0, which was accidentally right while
            // Q8_0, BF16 and F32 were the only formats a batch ever saw.
            // Q6_K (lm_head) broke it: its rows are 2 100 B but W.K is 2 560,
            // so every row past 203 700 of 248 320 addressed beyond the
            // 521 MB allocation, up to 114 MB past its end -- the "page not
            // present" fault on the card that holds lm_head.
            b.d[i].row_stride = (W.type == FK_Q_F32 || W.type == FK_Q_BF16)
                              ? (size_t) W.K : W.row_bytes;
            b.d[i].type       = W.type;
            if (W.type == FK_Q_Q6_K) b.has_q6k = 1;
            b.d[i].rows       = (int) W.rows;
            b.d[i].row_off    = off;
            b.d[i].epi        = jobs[i].epi;
            b.d[i].arg        = jobs[i].arg;
            off   += (int) W.rows;
            bytes += W.row_bytes * (size_t) W.rows;
        }
        b.total_rows = off;

        // ---- the LDS-tiled arm (--gemm-lds), T > 1 only -------------------
        //
        // ALL-OR-NOTHING PER BATCH, and that is not fastidiousness: nsplit
        // below is a function of the CONCATENATION's row count, so routing
        // half a batch here and half to k_gemm_batch would silently change
        // the other half's split -- and therefore its summation order -- for
        // a reason that has nothing to do with this kernel. A batch either
        // qualifies whole or is untouched.
        //
        // It has to be wide enough to fill the card from the ROW axis (a
        // 64-row tile on a 4-row inject matrix is 60 idle lanes, and those
        // matrices are the ones split-K exists for), it must carry no Q6_K
        // (lm_head, which only ever runs at T = 1 anyway) and no expert
        // format, and mode 2 additionally needs every matrix to be Q8_0
        // because v_dot4 has nothing to chew on otherwise.
        int lds_mode = 0;
        if (gemm_lds_ > 0 && T > 1 && !b.has_q6k && b.total_rows >= GEMM_LDS_MIN_ROWS) {
            bool ok = true, all_q8 = true;
            for (int i = 0; i < b.n; ++i) {
                const int ty = b.d[i].type;
                if (ty != FK_Q_Q8_0 && ty != FK_Q_BF16 && ty != FK_Q_F32) ok = false;
                if (ty != FK_Q_Q8_0) all_q8 = false;
            }
            if (ok) lds_mode = (gemm_lds_ >= 2 && all_q8) ? 2 : 1;
        }
        if (lds_mode) {
            mark(PC_TRUNK, group, bytes * (size_t) T);
            last_op_ = "gemm_lds";
            ++gemm_lds_calls_;
            const dim3 grid((unsigned) ceil_div(b.total_rows, GEMM_LDS_R),
                            (unsigned) ceil_div(T, GEMM_LDS_C));
            if (lds_mode == 2)
                k_gemm_lds_i8<<<grid, GEMM_LDS_THREADS, 0, stream_>>>(b, x, x2, T);
            else
                k_gemm_lds<<<grid, GEMM_LDS_THREADS, 0, stream_>>>(b, x, x2, T);
            return false;
        }
        if (T > 1) ++gemm_tile_calls_;

        // total waves == total rows, so wider workgroups never add
        // parallelism; splitting K is the only thing that does.
        const int row_groups = ceil_div(b.total_rows, GEMV_WAVES);
        const int gran  = b.has_q6k ? 256 : WAVE;
        const int units = ceil_div(b.K, gran);
        // total waves == total rows, so the test is on ROWS: a matrix with
        // enough rows already fills the card and a split would only add a
        // reduce launch.
        // nsplit is a function of the MATRIX ONLY. It must not see T: it
        // decides the summation order, so a chunk whose split differed from
        // a decode token's would not be bit-identical to it and the
        // `--chunk 6` oracle would fail for a reason that is not a bug.
        int nsplit = (b.total_rows >= gemv_min_rows_) ? 1 : ceil_div(GEMV_TARGET_WGS, row_groups);
        if (nsplit > units) nsplit = units;
        if (nsplit < 1) nsplit = 1;
        if (T < 1) T = 1;
        // TILE = 1 for a decode token: see k_gemm_batch's header. The tile
        // width is the ONLY thing that changes, and it changes nothing a
        // column sees.
        const int tile    = (T == 1) ? 1 : GEMM_TILE;
        const int n_tiles = ceil_div(T, tile);
        // The partial buffer is what grows with T instead; it is grown ONCE
        // (chunk sizes do not change mid-run) rather than capped, because
        // capping would make nsplit depend on T through the back door.
        if (nsplit > 1) ensure_part((size_t) nsplit * (size_t) T * (size_t) b.total_rows);

        // `units` counts granules, so the slice is granules * gran -- not
        // * WAVE, which under-sized the LDS 8x for a Q6_K batch and let the
        // staging loop write past the allocation. A tile stages `nt` columns
        // of it, so the LDS request is nt_max times bigger than at T = 1 --
        // and staging is dropped rather than shrunk when it will not fit,
        // because both arms compute the same value (GEMM_X).
        const int slice  = ceil_div(units, nsplit) * gran;
        const int nt_max = T < tile ? T : tile;
        const int stage  = (gemv_lds_ && (size_t) nt_max * slice <= GEMV_LDS_MAX_FLOATS) ? 1 : 0;
        const size_t lds = stage ? (size_t) nt_max * slice * sizeof(float) : 0;

        // The decode token reduces its splits inside the kernel (one counter
        // per output row) and, for hc_up, collapses inside it too. Both are
        // knobs because both are new, not because either can move a bit.
        int * cnt = nullptr;
        if (tile == 1 && nsplit > 1 && gemv_fused_reduce_) {
            ensure_cnt((size_t) b.total_rows);
            cnt = gemv_cnt_;
        }
        float *       col_out = nullptr;
        const float * col_xn  = nullptr;
        if (collapse_out && tile == 1 && nsplit == 1 && b.n == 1 && T == 1 &&
            b.total_rows == HC * N_EMBD && (GEMV_WAVES % HC) == 0 &&
            row_groups == N_EMBD / (GEMV_WAVES / HC)) {
            col_out = collapse_out; col_xn = collapse_xn;
        }

        mark(PC_TRUNK, group, bytes * (size_t) T);
        last_op_ = "gemv_batch";
        // grid.x is the SPLIT at TILE == 1 and the row group at TILE > 1; see
        // the kernel's header (a). n_tiles is 1 whenever tile == 1.
        const dim3 grid = (tile == 1)
                        ? dim3((unsigned) nsplit, (unsigned) row_groups, 1u)
                        : dim3((unsigned) row_groups, (unsigned) nsplit, (unsigned) n_tiles);
        if (tile == 1) {
            switch (gemv_burst_) {
                case 1:  k_gemm_batch<1, 1><<<grid, GEMV_THREADS, lds, stream_>>>(
                             b, x, x2, nsplit, gemv_part_, stage, T, cnt, col_out, col_xn); break;
                case 2:  k_gemm_batch<1, 2><<<grid, GEMV_THREADS, lds, stream_>>>(
                             b, x, x2, nsplit, gemv_part_, stage, T, cnt, col_out, col_xn); break;
                default: k_gemm_batch<1, 4><<<grid, GEMV_THREADS, lds, stream_>>>(
                             b, x, x2, nsplit, gemv_part_, stage, T, cnt, col_out, col_xn); break;
            }
        } else {
            k_gemm_batch<GEMM_TILE, 1><<<grid, GEMV_THREADS, lds, stream_>>>(
                b, x, x2, nsplit, gemv_part_, stage, T, nullptr, nullptr, nullptr);
        }
        if (nsplit > 1 && !cnt) {
            ++launches_;
            k_reduce_splits_gemm<<<dim3((unsigned) ceil_div(b.total_rows, BLK), (unsigned) T),
                                   BLK, 0, stream_>>>(b, gemv_part_, nsplit, T);
        }
        return col_out != nullptr;
    }

    // Sort the chunk's T*10 expert assignments and build the tile table.
    // Returns the tile-grid bound: at most one tile per expert that appears,
    // plus one more for every MOE_TILE_E assignments. Tiles past the real
    // count have tile_cnt 0 and their workgroups return immediately.
    // moe_gate_up and moe_down of one layer are handed the SAME ids, so the
    // second call reuses the first's sort rather than launching again.
    // --moe-tile: the assignments a wave takes at once. Three instantiations,
    // and the dispatch is a switch rather than a runtime loop bound because
    // the accumulator array has to stay in registers.
    int moe_tile() const { return moe_tile_ == 4 || moe_tile_ == 16 ? moe_tile_ : 8; }

    int moe_sort(const int * ids, int T) {
        const int te = moe_tile();
        const int n  = N_EXPERT_USED * T;
        const int nt = (n < N_EXPERT ? n : N_EXPERT) + ceil_div(n, te);
        if (moe_ids_ == ids && moe_n_ == n && moe_sorted_tile_ == te) return nt;
        if (n > moe_cap_ || te != moe_alloc_tile_) {
            dev_ensure(dev_);
            HIP_IGNORE(hipFree(moe_order_));    HIP_IGNORE(hipFree(moe_tile_exp_));
            HIP_IGNORE(hipFree(moe_tile_off_)); HIP_IGNORE(hipFree(moe_tile_cnt_));
            if (n > moe_cap_) moe_cap_ = n;
            moe_alloc_tile_ = te;
            // the smallest tile gives the most tiles, so size for it and the
            // allocation survives a knob change mid-run
            const int ntc = (moe_cap_ < N_EXPERT ? moe_cap_ : N_EXPERT) + ceil_div(moe_cap_, 4);
            HIP_CHECK(hipMalloc(&moe_order_,    (size_t) moe_cap_ * sizeof(int)));
            HIP_CHECK(hipMalloc(&moe_tile_exp_, (size_t) ntc * sizeof(int)));
            HIP_CHECK(hipMalloc(&moe_tile_off_, (size_t) ntc * sizeof(int)));
            HIP_CHECK(hipMalloc(&moe_tile_cnt_, (size_t) ntc * sizeof(int)));
        }
        mark(PC_EXP_GATHER);
        last_op_ = "moe_sort";
        switch (te) {
            case 4:  k_moe_sort<4><<<1, N_EXPERT, 0, stream_>>>(ids, n, moe_order_,
                         moe_tile_exp_, moe_tile_off_, moe_tile_cnt_, nt); break;
            case 16: k_moe_sort<16><<<1, N_EXPERT, 0, stream_>>>(ids, n, moe_order_,
                         moe_tile_exp_, moe_tile_off_, moe_tile_cnt_, nt); break;
            default: k_moe_sort<8><<<1, N_EXPERT, 0, stream_>>>(ids, n, moe_order_,
                         moe_tile_exp_, moe_tile_off_, moe_tile_cnt_, nt); break;
        }
        moe_ids_ = ids;
        moe_n_   = n;
        moe_sorted_tile_ = te;
        return nt;
    }

    const int * moe_ids_ = nullptr;
    int   moe_n_   = 0;
    int   moe_cap_ = 0;
    int * moe_order_    = nullptr;
    int * moe_tile_exp_ = nullptr;
    int * moe_tile_off_ = nullptr;
    int * moe_tile_cnt_ = nullptr;

    // The split-K partials, [nsplit][T][total_rows]. nsplit*total_rows stays
    // near GEMV_TARGET_WGS*GEMV_WAVES by construction (nsplit rises only as
    // rows falls), so this is ~32k floats a token; T multiplies it. Grown
    // geometrically and only when a chunk first needs it -- hipFree
    // SYNCHRONISES THE DEVICE, so this must never be a per-token event.
    void ensure_part(size_t floats) {
        if (floats <= part_floats_) return;
        dev_ensure(dev_);
        size_t want = part_floats_ ? part_floats_ : GEMV_PART_FLOATS;
        while (want < floats) want <<= 1;
        HIP_IGNORE(hipFree(gemv_part_));
        HIP_CHECK(hipMalloc(&gemv_part_, want * sizeof(float)));
        part_floats_ = want;
        ++epoch_;
    }
    // One counter per output row for the in-kernel split reduce. It is
    // allocated once and is ALWAYS zero between launches -- the workgroup that
    // closes a row puts its counter back -- so nothing here is per token.
    void ensure_cnt(size_t rows) {
        if (rows <= cnt_rows_) return;
        dev_ensure(dev_);
        size_t want = cnt_rows_ ? cnt_rows_ : GEMV_CNT_ROWS;
        while (want < rows) want <<= 1;
        HIP_IGNORE(hipFree(gemv_cnt_));
        HIP_CHECK(hipMalloc(&gemv_cnt_, want * sizeof(int)));
        HIP_CHECK(hipMemset(gemv_cnt_, 0, want * sizeof(int)));
        cnt_rows_ = want;
        ++epoch_;
    }
    void set_gemv_lds(int on) override { gemv_lds_ = on; }
    void set_gemv_fused_reduce(int on) override { gemv_fused_reduce_ = on; }
    void set_gemv_fuse_collapse(int on) override { gemv_fuse_collapse_ = on; }
    void set_gemv_burst(int n) override {
        gemv_burst_ = (n == 1 || n == 2 || n == 4) ? n : GEMV_BURST;
    }
    void set_sync_debug(bool on) override { sync_debug_ = on; }
    void set_debug_context(const char * phase, int layer) override {
        dbg_phase_ = phase; dbg_layer_ = layer;
    }
    void set_gemv_min_rows(int r) override { gemv_min_rows_ = r > 0 ? r : 1; }
    void set_gemm_lds(int m) override { gemm_lds_ = m; }
    void set_expert_gather(int mask) override { expert_gather_ = mask; }
    void set_moe_tile(int te) override {
        if (te != moe_tile_) moe_ids_ = nullptr;     // the tile table is stale
        moe_tile_ = te;
    }
    void set_gather_serial(int on) override { gather_serial_ = on; }
    void gemv_expert(const Mat & W, int e, const float * x, float * y) override {
        Mat slice = W;
        slice.base = (const unsigned char *) W.base + (size_t) e * W.expert_stride;
        slice.expert_stride = 0;
        gemv(slice, x, y);
    }

    void copy(float * d, const float * s, size_t n) override {
        mark(PC_COPY);
        last_op_ = "copy";
        HIP_CHECK(hipMemcpyAsync(d, s, n * sizeof(float), hipMemcpyDeviceToDevice, 0));
    }
    void unary(int op, const float * x, float * y, size_t n) override {
        mark(PC_ELEM);
        last_op_ = "unary";
        k_unary<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(op, x, y, (int) n);
    }
    void binary(int op, const float * a, const float * b, float * y, size_t n) override {
        mark(PC_ELEM);
        last_op_ = "binary";
        k_binary<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(op, a, b, y, (int) n);
    }
    void scale(const float * x, float s, float * y, size_t n) override {
        mark(PC_ELEM);
        last_op_ = "scale";
        k_scale<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(x, s, y, (int) n);
    }
    void binary3_add(const float * a, const float * b, const float * c,
                     float * y, size_t n) override {
        mark(PC_ELEM);
        last_op_ = "binary3_add";
        k_binary3_add<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(a, b, c, y, (int) n);
    }
    void mul_tiled(const float * x, const float * w, float * y, size_t n, size_t ne0_w) override {
        mark(PC_ELEM);
        last_op_ = "mul_tiled";
        k_mul_tiled<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(x, w, y, (int) n, (int) ne0_w);
    }
    void gather_strided(const float * src, float * dst, int n_rows, int row_len,
                        int src_stride, int src_off) override {
        mark(PC_ELEM);
        last_op_ = "gather_strided";
        k_gather_strided<<<ceil_div(n_rows * row_len, BLK), BLK, 0, stream_>>>(src, dst, n_rows, row_len, src_stride, src_off);
    }
    void rms_norm_mul(const float * x, const float * w, float * y,
                      int ne0, int n_groups, size_t w_len, float eps) override {
        mark(PC_NORM);
        last_op_ = "rms_norm_mul";
        const int t = red_threads(ne0);
        const int period = (int) std::max<size_t>(1, w_len / (size_t) ne0);
        k_rms_norm_mul<<<n_groups, t, t * sizeof(float), stream_>>>(x, w, y, ne0, eps,
                                          period, w ? 1 : 0);
    }
    void l2_norm(const float * x, float * y, int ne0, int n_groups, float eps) override {
        mark(PC_NORM);
        last_op_ = "l2_norm";
        const int t = red_threads(ne0);
        k_l2_norm<<<n_groups, t, t * sizeof(float), stream_>>>(x, y, ne0, eps);
    }
    void hc_broadcast(const float * x, float * r, int T) override {
        mark(PC_ELEM);
        last_op_ = "hc_broadcast";
        k_hc_broadcast<<<ceil_div(N_EMBD * T, BLK), BLK, 0, stream_>>>(x, r, T);
    }
    void hc_collapse(const float * xn, const float * g, float * o, int T) override {
        mark(PC_ELEM);
        last_op_ = "hc_collapse";
        k_hc_collapse<<<ceil_div(N_EMBD * T, BLK), BLK, 0, stream_>>>(xn, g, o, T);
    }
    void hc_combine(float * r, const float * b, const float * inj, int T) override {
        hc_combine_norm(r, b, inj, nullptr, nullptr, 0.0f, T);
    }
    void hc_combine_norm(float * r, const float * b, const float * inj,
                         const float * w_norm, float * xn, float eps, int T) override {
        mark(PC_ELEM);
        last_op_ = "hc_combine_norm";
        const int t = red_threads(N_EMBD);
        k_hc_combine_norm<<<HC * T, t, t * sizeof(float), stream_>>>(r, b, inj, w_norm, xn, eps);
    }
    void ple_gate(const float * k, const float * q, const float * v, float * g, float * go,
                  int T) override {
        mark(PC_ELEM);
        last_op_ = "ple_gate";
        const int t = red_threads(N_EMBD);
        k_ple_gate<<<HC * T, t, t * sizeof(float), stream_>>>(k, q, v, g, go);
    }
    void ple_conv_win(const float * win, const float * w, float * out, int T) override {
        mark(PC_ELEM);
        last_op_ = "ple_conv_win";
        k_conv_win<<<ceil_div(HC_DIM * T, BLK), BLK, 0, stream_>>>(
            win, PLE_CONV_K, PLE_CONV_DIL, HC_DIM, w, out, T);
    }
    void conv_slide(float * win, int hist, int channels, int T) override {
        mark(PC_ELEM);
        last_op_ = "conv_slide";
        k_conv_slide<<<ceil_div(channels, BLK), BLK, 0, stream_>>>(win, hist, channels, T);
    }
    void gdn_conv_gate(const float * win, const float * w,
                       const float * beta_raw, const float * alpha_raw,
                       const float * dt, const float * ssm_a, float * conv_out,
                       float * conv_qk, float * beta_sig, float * a_softplus,
                       float * gate, float * g_exp, int T) override {
        mark(PC_ELEM);
        last_op_ = "gdn_conv_gate";
        k_gdn_conv_gate<<<dim3(ceil_div(GDN_CONV_DIM + GDN_V_HEADS, BLK), (unsigned) T),
                          BLK, 0, stream_>>>(
            win, w, beta_raw, alpha_raw, dt, ssm_a,
            conv_out, conv_qk, beta_sig, a_softplus, gate, g_exp);
    }
    void gdn_step(float * S, const float * q, const float * k, const float * v,
                  const float * g, const float * b, float * o,
                  int T, int qk_stride, int v_stride) override {
        mark(PC_GDN);
        last_op_ = "gdn_step";
        k_gdn_step<<<GDN_V_HEADS * GDN_STATE, WAVE, 0, stream_>>>(
            S, q, k, v, g, b, o, T, qk_stride, v_stride);
    }
    void gated_rms_norm(const float * x, const float * w, const float * z, float * y,
                        int ne0, int n_groups, float eps) override {
        mark(PC_NORM);
        last_op_ = "gated_rms_norm";
        const int t = red_threads(ne0);
        k_gated_rms_norm<<<n_groups, t, t * sizeof(float), stream_>>>(x, w, z, y, ne0, eps);
    }
    void router(const float * logits, int * ids, float * w,
                int * ids_log, float * w_log, int T) override {
        mark(PC_ROUTER);
        last_op_ = "router";
        k_router<<<T, N_EXPERT, 0, stream_>>>(logits, ids, w, ids_log, w_log);
        // Every layer routes into the SAME scratch buffer, so pointer
        // identity alone would make moe_sort reuse the previous layer's sort.
        // The router is the one thing that invalidates it.
        moe_ids_ = nullptr;
    }
    void moe_gate_up(const Mat & g, const Mat & u, const int * ids, const float * x,
                     float * yg, float * yu, int T) override {
        if (T > 1 && (expert_gather_ & 1)) {
            const int nt = moe_sort(ids, T);
            const dim3 grid((unsigned) g.rows, (unsigned) nt, 2);
            mark(PC_EXP_GATHER);
            last_op_ = "moe_gate_up_gather";
#define FK_GU_GATHER(KER, TE, SER) KER<TE, SER><<<grid, WAVE, 0, stream_>>>(                      \
                (const unsigned char *) g.base, (const unsigned char *) u.base,      \
                g.expert_stride, g.row_bytes, moe_order_, moe_tile_exp_,             \
                moe_tile_off_, moe_tile_cnt_, x, yg, yu, (int) g.rows, (int) g.K,    \
                N_EXPERT_USED)
            const int sel = moe_tile() * 2 + (gather_serial_ ? 1 : 0);
            if (g.type == FK_Q_IQ3_S) {
                switch (sel) {
                    case  8: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 4,  false); break;
                    case  9: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 4,  true);  break;
                    case 32: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 16, false); break;
                    case 33: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 16, true);  break;
                    case 17: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 8,  true);  break;
                    default: FK_GU_GATHER(k_moe_gate_up_iq3s_gather, 8,  false); break;
                }
            } else {
                switch (sel) {
                    case  8: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 4,  false); break;
                    case  9: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 4,  true);  break;
                    case 32: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 16, false); break;
                    case 33: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 16, true);  break;
                    case 17: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 8,  true);  break;
                    default: FK_GU_GATHER(k_moe_gate_up_iq4xs_gather, 8,  false); break;
                }
            }
#undef FK_GU_GATHER
            return;
        }
        const dim3 grid((unsigned) g.rows, (unsigned)(N_EXPERT_USED * T), 2);
        mark(PC_EXP_GATE_UP);
        last_op_ = "moe_gate_up";
        if (g.type == FK_Q_IQ3_S) {
            k_moe_gate_up_iq3s<<<grid, WAVE, 0, stream_>>>((const unsigned char *) g.base,
                (const unsigned char *) u.base, g.expert_stride, g.row_bytes, ids, x, yg, yu,
                (int) g.rows, (int) g.K, N_EXPERT_USED, T);
        } else {
            k_moe_gate_up_iq4xs<<<grid, WAVE, 0, stream_>>>((const unsigned char *) g.base,
                (const unsigned char *) u.base, g.expert_stride, g.row_bytes, ids, x, yg, yu,
                (int) g.rows, (int) g.K, N_EXPERT_USED, T);
        }
    }
    void silu_mul(const float * g, const float * u, float * h, size_t n) override {
        mark(PC_ELEM);
        last_op_ = "silu_mul";
        k_silu_mul<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(g, u, h, (int) n);
    }
    void moe_down(const Mat & d, const int * ids, const float * h, float * eo, int T) override {
        if (T > 1 && (expert_gather_ & 2)) {
            // the layer's moe_gate_up may already have sorted these very ids; the
            // sort is reused unless something unexpected changed
            const int nt = moe_sort(ids, T);
            const dim3 grid((unsigned) d.rows, (unsigned) nt, 1);
            mark(PC_EXP_GATHER);
            last_op_ = "moe_down_gather";
#define FK_DN_GATHER(KER, TE) KER<TE><<<grid, WAVE, 0, stream_>>>(                     \
                (const unsigned char *) d.base, d.expert_stride, d.row_bytes,       \
                moe_order_, moe_tile_exp_, moe_tile_off_, moe_tile_cnt_,            \
                h, eo, (int) d.rows, (int) d.K, N_EXPERT_USED)
            if (d.type == FK_Q_IQ4_NL) {
                switch (moe_tile()) {
                    case 4:  FK_DN_GATHER(k_moe_down_iq4nl_gather, 4);  break;
                    case 16: FK_DN_GATHER(k_moe_down_iq4nl_gather, 16); break;
                    default: FK_DN_GATHER(k_moe_down_iq4nl_gather, 8);  break;
                }
            } else {
                switch (moe_tile()) {
                    case 4:  FK_DN_GATHER(k_moe_down_q8_0_gather, 4);  break;
                    case 16: FK_DN_GATHER(k_moe_down_q8_0_gather, 16); break;
                    default: FK_DN_GATHER(k_moe_down_q8_0_gather, 8);  break;
                }
            }
#undef FK_DN_GATHER
            return;
        }
        const dim3 grid((unsigned) d.rows, (unsigned)(N_EXPERT_USED * T), 1);
        mark(PC_EXP_DOWN);
        last_op_ = "moe_down";
        if (d.type == FK_Q_IQ4_NL) {
            k_moe_down_iq4nl<<<grid, WAVE, 0, stream_>>>((const unsigned char *) d.base, d.expert_stride,
                d.row_bytes, ids, h, eo, (int) d.rows, (int) d.K, N_EXPERT_USED, T);
        } else {
            k_moe_down_q8_0<<<grid, WAVE, 0, stream_>>>((const unsigned char *) d.base, d.expert_stride,
                d.row_bytes, ids, h, eo, (int) d.rows, (int) d.K, N_EXPERT_USED, T);
        }
    }
    void moe_finish(const float * eo, const float * w, const float * sh_raw,
                    const float * sh_gate, float * sh_gate_sig, float * moe_out,
                    float * sh_gated, float * out, int T) override {
        mark(PC_EXP_COMBINE);
        last_op_ = "moe_finish";
        k_moe_finish<<<ceil_div(N_EMBD * T, BLK), BLK, 0, stream_>>>(eo, w, sh_raw, sh_gate,
                                                     sh_gate_sig, moe_out, sh_gated,
                                                     out, N_EMBD, N_EXPERT_USED, T);
    }
    void rope_imrope(float * x, int n_heads, int head_dim, int n_rot, int pos) override {
        mark(PC_QSA);
        last_op_ = "rope_imrope";
        k_rope_imrope<<<n_heads, n_rot / 2, 0, stream_>>>(x, head_dim, n_rot, pos);
    }
    void kv_store_q8_0(const float * k, const float * v, int8_t * kqs, uint16_t * ksc,
                       int8_t * vqs, uint16_t * vsc, int cell0, int T) override {
        const dim3 grid((unsigned)(HEAD_DIM / 32 * T), N_KV_HEADS, 2);
        mark(PC_QSA);
        last_op_ = "kv_store_q8_0";
        k_kv_store_q8_0<<<grid, WAVE, 0, stream_>>>(k, v, kqs, (__half *) ksc, vqs,
                                                    (__half *) vsc, cell0, T, gpos());
    }
    void idx_pool_chunk(const float * k_new, float * sum, float * raw0, int pos0,
                        int T, const float * kn, float eps, uint16_t * pooled,
                        float * dbg_p, float * dbg_r) override {
        mark(PC_QSA);
        last_op_ = "idx_pool_chunk";
        k_idx_pool_chunk<<<1, IDX_DIM, 0, stream_>>>(k_new, sum, raw0, pos0, T,
                                         kn, eps, (unsigned short *) pooled, dbg_p, dbg_r,
                                         gpos());
    }
    void idx_scan(const uint16_t * pooled, const float * q, float * s, int n_blocks) override {
        const int nsub = IDXS_THREADS / WAVE;
        int wgs = ceil_div(n_blocks, nsub);
        if (wgs > 4096) wgs = 4096;
        if (wgs < 1) wgs = 1;
        // n_blocks == ceil(n_kv / ratio), so a position that reproduces it is
        // n_blocks*ratio - 1: the kernel derives the row's block count the
        // same way for a block of one as for a block of 256.
        QsaRowsD d = one_row(n_blocks * QSA_RATIO - 1, QSA_RATIO, MAX_SEL);
        mark(PC_QSA);
        last_op_ = "idx_scan";
        k_idx_scan<<<dim3((unsigned) wgs, 1), IDXS_THREADS, 0, stream_>>>(
            (const unsigned short *) pooled, q, s, d);
    }
    void qsa_qk_post(const float * qfull, const float * kraw, const float * idxraw,
                     const float * q_norm, const float * k_norm, const float * iq_norm,
                     float * qcur, float * gate, float * gsig, float * kcur,
                     float * idxq, int pos0, float eps, int T) override {
        mark(PC_QSA);
        last_op_ = "qsa_qk_post";
        k_qsa_qk_post<<<dim3(N_Q_HEADS + N_KV_HEADS + IDX_N_HEADS, (unsigned) T),
                        QKPOST_THREADS, 0, stream_>>>(
            qfull, kraw, idxraw, q_norm, k_norm, iq_norm,
            qcur, gate, gsig, kcur, idxq, pos0, eps, gpos());
    }
    void qsa_expand(const float * bs, float * c, int * sel, int n_kv, int q, int r,
                    int tail, int sel_identity) override {
        (void) tail;                 // the kernel computes it from the position
        // The cache is contiguous (cell j == position j), so n_kv == q + 1 and
        // a budget of n_kv forces the identity the caller asked for.
        QsaRowsD d = one_row(q, r, sel_identity ? n_kv : (IDX_TOP_K + r - 1));
        mark(PC_QSA);
        last_op_ = "qsa_expand";
        k_qsa_expand<<<dim3((unsigned) ceil_div(n_kv, BLK), 1), BLK, 0, stream_>>>(bs, c, sel, d);
    }
    void reserve_topk(int n) override {
        if (n <= cand_cap_ && qsa_rows_ <= cand_rows_) return;
        dev_ensure(dev_);
        HIP_IGNORE(hipFree(cand_key_));
        HIP_IGNORE(hipFree(cand_idx_));
        if (n > cand_cap_) cand_cap_ = n;
        cand_rows_ = qsa_rows_;
        const size_t nc = (size_t) cand_cap_ * (size_t) cand_rows_;
        HIP_CHECK(hipMalloc(&cand_key_, nc * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&cand_idx_, nc * sizeof(int)));
        ++epoch_;
    }

    // How many query rows of a chunk this card has scratch for. Everything a
    // row needs that is NOT shared -- its cell scores, its radix candidates,
    // its attention partials -- is counted, and the block is whatever fits in
    // the budget (FRANKEN_QSA_ROW_MB, default 256). At ctx 262 144 a row is
    // 1.0 MB of cell scores + 2.1 MB of candidates + 0.81 MB of partials, so
    // the block is ~64 rows; at 8 192 it is the whole chunk. The runner walks
    // the chunk in blocks of the answer, and a backend returning 1 gets the
    // old row-at-a-time loop back.
    int reserve_qsa_rows(int rows_max, int ctx) override {
        dev_ensure(dev_);
        const size_t per_row =
              (size_t) ctx * sizeof(float)                              // cell scores
            + (size_t) ctx * (sizeof(uint32_t) + sizeof(int))           // radix candidates
            + (size_t) ceil_div(MAX_SEL, ATTN_CHUNK) * N_Q_HEADS * (HEAD_DIM + 2) * sizeof(float)
            + (size_t) ceil_div(ctx, QSA_RATIO) * sizeof(float)         // block scores
            + (size_t) MAX_SEL * sizeof(int);                           // the selection
        size_t budget = (size_t) qsa_row_mb_ * 1024u * 1024u;
        int rows = per_row ? (int)(budget / per_row) : rows_max;
        if (rows < 1)        rows = 1;
        if (rows > rows_max) rows = rows_max;
        qsa_rows_ = rows;
        alloc_attn_rows(rows);
        return rows;
    }

    // One launch a stage for a whole block of rows. This is the only caller
    // of the five QSA kernels in a T > 1 run, and at T = 1 it is a block of
    // one -- same kernels, gridDim.y == 1, so the decode path cannot drift
    // from the prefill path here any more than k_gemm_batch<1> lets it drift
    // in the trunk (PREFILL.md section 9).
    void qsa_rows(const QsaRows & j) override {
        dev_ensure(dev_);
        QsaRowsD d;
        d.pos0 = j.pos0; d.ratio = j.ratio; d.budget = j.budget;
        d.blk_stride  = j.blk_stride;
        d.cell_stride = j.cell_stride;
        d.sel_stride  = j.sel_stride;
        d.cand_stride = (size_t) cand_cap_;
        d.part_stride = part_stride_;
        d.ml_stride   = ml_stride_;
        d.dpos        = gpos();
        const int rows     = j.rows;
        const int last     = rows - 1;                      // the widest row
        // Captured for --hip-graph, the launch must serve EVERY position of
        // the class, so the grids are sized for the class's largest n_kv and
        // the kernels bound themselves by the position they read. Whether the
        // radix select is a node is decided on that bound too: a class that
        // can exceed the budget carries it, and it returns for a row whose
        // budget still covers its cache -- the device-side condition the
        // kernel already had for a block of rows.
        int n_kv_max = j.pos0 + rows;                       // == n_kv of `last`
        if (gpos()) {
            if (rows != 1 || graph_bound_ < n_kv_max) {
                std::fprintf(stderr, "qsa_rows: captured with rows=%d bound=%d < n_kv=%d\n",
                             rows, graph_bound_, n_kv_max);
                std::abort();
            }
            n_kv_max = graph_bound_;
        }
        const int nblk_max = (n_kv_max + j.ratio - 1) / j.ratio;
        const int nsel_max = n_kv_max < j.budget ? n_kv_max : j.budget;
        const bool ident   = nsel_max >= n_kv_max;          // uniform over a block?

        {   // the lightning indexer's scan over the pooled keys
            const int nsub = IDXS_THREADS / WAVE;
            int wgs = ceil_div(nblk_max, nsub);
            if (wgs > 4096) wgs = 4096;
            if (wgs < 1)    wgs = 1;
            mark(PC_QSA_ROWS);
            last_op_ = "idx_scan_rows";
            k_idx_scan<<<dim3((unsigned) wgs, (unsigned) rows), IDXS_THREADS, 0, stream_>>>(
                (const unsigned short *) j.pooled, j.idxq, j.blk, d);
        }
        {   // the per-cell expansion, the bias and (when it is) the identity
            mark(PC_QSA_ROWS);
            last_op_ = "qsa_expand_rows";
            k_qsa_expand<<<dim3((unsigned) ceil_div(n_kv_max, BLK), (unsigned) rows),
                           BLK, 0, stream_>>>(j.blk, j.cell, j.sel, d);
        }
        // The radix select runs only for the rows whose budget does NOT cover
        // their cache. n_kv grows by one a row, so within a block the two
        // cases can only split once -- and the kernel returns early for an
        // identity row anyway, so one launch covers both.
        if (!ident) {
            if (n_kv_max > cand_cap_ || qsa_rows_ > cand_rows_) {
                std::fprintf(stderr, "qsa_rows: growing the candidate list to %d x %d mid-run "
                                     "(reserve_topk/reserve_qsa_rows were not called with --ctx)\n",
                             n_kv_max, qsa_rows_);
                reserve_topk(n_kv_max > 2 * cand_cap_ ? n_kv_max : 2 * cand_cap_);
                d.cand_stride = (size_t) cand_cap_;
            }
            mark(PC_QSA_ROWS);
            last_op_ = "topk_select_rows";
            k_topk_radix_select<<<dim3((unsigned) rows), TOPK_THREADS, 0, stream_>>>(
                j.cell, cand_key_, cand_idx_, j.sel, d);
        }
        {   // flash attention over the selection, split then combined
            const int n_chunks = ceil_div(nsel_max, ATTN_CHUNK);
            mark(PC_QSA_ROWS);
            last_op_ = "attn_qsa_rows";
            k_attn_flash_split<<<dim3((unsigned) n_chunks, N_KV_HEADS, (unsigned) rows),
                                 ATTN_THREADS, 0, stream_>>>(
                j.kqs, (const __half *) j.ksc, j.vqs, (const __half *) j.vsc,
                j.q, j.sel, partial_, mbuf_, lbuf_, d);
            mark(PC_QSA_ROWS);
            last_op_ = "attn_qsa_rows";
            k_attn_combine<<<dim3(N_Q_HEADS, (unsigned) rows), HEAD_DIM, 0, stream_>>>(
                partial_, mbuf_, lbuf_, j.gsig, j.out, j.out_gated, d);
        }
        (void) last;
    }

    // A block of one. Kept because Backend's default qsa_rows() -- what the
    // CPU arm runs -- is written against these, and because a caller with a
    // single row should not have to build a job.
    int topk_select(const float * scores, int n, int width, int * out) override {
        dev_ensure(dev_);
        // Must never fire: the runner reserves --ctx at construction. Kept as
        // a safety net, and GEOMETRIC so that even if it did it could not
        // happen again on the next token.
        if (n > cand_cap_) {
            std::fprintf(stderr, "topk_select: growing the candidate list to %d mid-run "
                                 "(reserve_topk was not called with a large enough --ctx)\n", n);
            reserve_topk(n > 2 * cand_cap_ ? n : 2 * cand_cap_);
        }
        const int k = width < n ? width : n;
        if (k >= n) return k;                     // the identity; nothing to select
        QsaRowsD d = one_row(n - 1, QSA_RATIO, width);
        mark(PC_QSA);
        last_op_ = "topk_select";
        k_topk_radix_select<<<1, TOPK_THREADS, 0, stream_>>>(scores, cand_key_, cand_idx_, out, d);
        return k;
    }
    void attn_qsa(const int8_t * kqs, const uint16_t * ksc, const int8_t * vqs,
                  const uint16_t * vsc, const float * q, const int * sel, int n_sel,
                  const float * gsig, float * out, float * out_gated) override {
        const int n_chunks = ceil_div(n_sel, ATTN_CHUNK);
        // n_sel is min(n_kv, budget); pos = n_sel - 1 with budget = n_sel
        // reproduces exactly that for a block of one.
        QsaRowsD d = one_row(n_sel - 1, QSA_RATIO, n_sel);
        mark(PC_QSA);
        last_op_ = "attn_qsa";
        k_attn_flash_split<<<dim3(n_chunks, N_KV_HEADS, 1), ATTN_THREADS, 0, stream_>>>(
            kqs, (const __half *) ksc, vqs, (const __half *) vsc, q, sel,
            partial_, mbuf_, lbuf_, d);
        mark(PC_QSA);
        last_op_ = "attn_qsa";
        k_attn_combine<<<dim3(N_Q_HEADS, 1), HEAD_DIM, 0, stream_>>>(
            partial_, mbuf_, lbuf_, gsig, out, out_gated, d);
    }

    void set_profile(bool on) override {
        prof_ = on;
        // ------------------------------------------------------------------
        // DO NOT DELETE THIS LINE. It has been lost three times now (c265941,
        // then a rebase, then ab8c9cd) and the symptom is always the same:
        // "invalid resource handle" from mark(), because hipEventCreate acts
        // on the CURRENT device -- process-wide state, not something a stream
        // or a backend object carries -- and make_gpu_backend(0..2) leaves it
        // at 2, so devices 0 and 1 got device-2 events and recording them on
        // their own streams is an error. It is a rule in CLAUDE.md.
        // Anything in this class that allocates, copies, creates an event or
        // launches must call dev_ensure(dev_) first.
        // ------------------------------------------------------------------
        dev_ensure(dev_);
        if (on && ev_.empty()) {
            ev_.resize(PROF_MAX_EVENTS + 1);
            ev_cls_.resize(PROF_MAX_EVENTS + 1);
            ev_grp_.resize(PROF_MAX_EVENTS + 1, -1);
            ev_small_.resize(PROF_MAX_EVENTS + 1, 0);
            for (auto & e : ev_) HIP_CHECK(hipEventCreate(&e));
        }
    }
    void prof_reset() override {
        // Anything still pending belongs to the chunks being discarded.
        if (prof_defer_ && n_ev_ > 0) prof_flush();
        for (int i = 0; i < PC_N; ++i) prof_us_[i] = 0.0;
        for (int i = 0; i < GG_N; ++i) { gemv_us_[i] = 0.0; gemv_bytes_[i] = 0.0; }
        gemv_small_us_ = 0.0; gemv_small_bytes_ = 0.0; gemv_small_launches_ = 0;
        tokens_profiled_ = 0;
        blocking_in_body_ = 0;
        launches_in_body_ = 0;
        gemm_lds_calls_ = 0;
        gemm_tile_calls_ = 0;
    }
    void prof_defer(bool on) override { prof_defer_ = on; }
    // Records the closing event, waits for it ONCE, and reduces every interval
    // still in the pool. The only synchronisation the deferred profile does.
    void prof_flush() override {
        if (!prof_ || n_ev_ == 0) { n_ev_ = 0; return; }
        dev_ensure(dev_);
        HIP_CHECK(hipEventRecord(ev_[n_ev_], stream_));
        HIP_CHECK(hipEventSynchronize(ev_[n_ev_]));
        for (int i = 0; i < n_ev_; ++i) {
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev_[i], ev_[i + 1]));
            prof_us_[ev_cls_[i]] += (double) ms * 1000.0;
            if (ev_grp_[i] >= 0 && ev_grp_[i] < GG_N) gemv_us_[ev_grp_[i]] += (double) ms * 1000.0;
            if (ev_small_[i]) gemv_small_us_ += (double) ms * 1000.0;
        }
        n_ev_ = 0;
    }
    void prof_end_token() override {
        dev_ensure(dev_);
        debug_check("end of token");
        blocking_in_body_ += blocking_ - blocking_at_token_start_;
        launches_last_ = launches_ - launches_at_token_start_;
        launches_in_body_ += launches_last_;
        ++tokens_profiled_;
        if (!prof_ || n_ev_ == 0) { n_ev_ = 0; return; }
        if (!prof_defer_) { prof_flush(); return; }
        // DEFERRED: close the chunk with an event of class PC_GAP and keep
        // going. The interval from it to the next chunk's first mark() is
        // this card's idle time, and nothing here waits -- a synchronise
        // would serialise the pipeline the profile exists to show.
        HIP_CHECK(hipEventRecord(ev_[n_ev_], stream_));
        ev_cls_[n_ev_] = PC_GAP;
        ev_grp_[n_ev_] = -1;
        ev_small_[n_ev_] = 0;
        ++n_ev_;
        // Reducing costs one sync; at ~400 events a chunk this is roughly
        // once every twenty chunks, which is why it is cheaper than the
        // per-chunk sync it replaced and still cannot exhaust the pool.
        if (n_ev_ >= PROF_MAX_EVENTS - 1024) prof_flush();
    }
    void prof_report(FILE * out, int n_tokens) override {
        prof_flush();                      // anything a pipelined run left pending
        const double n = n_tokens > 0 ? (double) n_tokens : 1.0;
        double total = 0.0;
        for (int c = 0; c < PC_N; ++c) total += prof_us_[c];
        for (int c = 0; c < PC_N; ++c) {
            if (c >= PC_DS4_TRUNK_L0 && prof_us_[c] == 0.0) continue;   // a qwen4exp run
            std::fprintf(out, "prof_%s_us=%.2f\n", prof_name(c), prof_us_[c] / n);
        }
        std::fprintf(out, "prof_total_us=%.2f\n", total / n);
        // THE OVERLAP NUMBER. `busy` is this card's own work; `total` is the
        // span it covered. Three cards each busy ~2.5 ms a token inside a
        // 7.6 ms token is the sequential picture; the same three busy ~2.5 ms
        // inside a 2.6 ms token is the pipelined one.
        std::fprintf(out, "prof_busy_us=%.2f prof_gap_us_explicit=%.2f prof_busy_frac=%.3f\n",
                     (total - prof_us_[PC_GAP]) / n, prof_us_[PC_GAP] / n,
                     total > 0.0 ? (total - prof_us_[PC_GAP]) / total : 0.0);
        std::fprintf(out, "prof_gemm_lds_calls=%lld prof_gemm_tile_calls=%lld\n",
                     gemm_lds_calls_, gemm_tile_calls_);
        // Achieved bandwidth per projection class: the bytes are the weight
        // bytes those launches actually read, so this is the number to hold
        // against the 800 GB/s of M5, not the lumped trunk total.
        for (int g = 0; g < GG_N; ++g) {
            if (gemv_bytes_[g] <= 0.0) continue;
            const double us = gemv_us_[g] / n;
            const double mb = gemv_bytes_[g] / n / 1e6;
            std::fprintf(out, "prof_gemv_%s_us=%.2f prof_gemv_%s_mb=%.2f prof_gemv_%s_gbs=%.1f\n",
                         gemv_group_name(g), us, gemv_group_name(g), mb,
                         gemv_group_name(g), us > 0.0 ? mb / us * 1e3 : 0.0);
        }
        {   // the sub-10 MB set, across groups
            const double us = gemv_small_us_ / n;
            const double mb = gemv_small_bytes_ / n / 1e6;
            std::fprintf(out, "prof_gemv_small_us=%.2f prof_gemv_small_mb=%.2f "
                              "prof_gemv_small_gbs=%.1f prof_gemv_small_launches=%.2f\n",
                         us, mb, us > 0.0 ? mb / us * 1e3 : 0.0,
                         (double) gemv_small_launches_ / n);
        }
        std::fprintf(out, "prof_host_syncs_per_token=%.2f\n", (double) blocking_in_body_ / n);
        std::fprintf(out, "prof_launches_per_token=%.2f\n", (double) launches_in_body_ / n);
        std::fprintf(out, "prof_events_merged_per_token=%.2f  (class runs sharing one event)\n",
                     (double) ev_merged_ / n);
        if (n_ev_ >= PROF_MAX_EVENTS)
            std::fprintf(out, "prof_WARNING=event pool exhausted, times are a lower bound\n");
    }

    long long launches_last_token() const override { return launches_last_; }

    void   timer_start() override {
        dev_ensure(dev_);
        blocking_at_token_start_ = blocking_;
        launches_at_token_start_ = launches_;
        HIP_CHECK(hipEventRecord(ev0_, stream_));
    }
    double timer_stop_ms() override {
        dev_ensure(dev_);
        HIP_CHECK(hipEventRecord(ev1_, stream_));
        HIP_CHECK(hipEventSynchronize(ev1_));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0_, ev1_));
        return (double) ms;
    }

    // ---- --hip-graph: the decode token's per-card body as one hipGraph ----
    //
    // Captured from THIS card's own stream only, so a graph never spans a
    // device: the boundary's event record/wait and the P2P copy stay plain
    // stream operations issued by the host BETWEEN the cards' graph launches,
    // exactly where they were (decode_graph.cpp step()). A graph launch is
    // stream-ordered like any other op, so the bank-drain handshake is
    // unchanged.
    bool graph_capable() const override { return true; }

    void graph_capture_begin(const int * dpos, int n_kv_bound) override {
        dev_ensure(dev_);
        if (capturing_) { std::fprintf(stderr, "graph_capture_begin: already capturing on dev%d\n", dev_); std::abort(); }
        gpos_ = dpos;
        graph_bound_ = n_kv_bound;
        capture_launch0_ = launches_;
        capture_epoch_ = epoch_;
        capture_t0_ = std::chrono::steady_clock::now();
        // Relaxed: nothing in the body calls a stream-unsafe API (the ensure_*
        // growths cannot fire at T = 1 after construction, and the epoch check
        // below catches it if one ever does), and hipSetDevice from
        // dev_ensure must stay legal while a stream captures.
        HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed));
        capturing_ = true;
    }

    void graph_capture_end(void ** exec) override {
        dev_ensure(dev_);
        // the last node: the device-resident position moves on by one
        k_pos_advance<<<1, 1, 0, stream_>>>(const_cast<int *>(gpos_));
        hipGraph_t g = nullptr;
        HIP_CHECK(hipStreamEndCapture(stream_, &g));
        capturing_ = false;
        gpos_ = nullptr;
        graph_nodes_last_ = (int)(launches_ - capture_launch0_) + 1;
        // what the capture enqueued is not this token's launches: the replay
        // below is ONE launch, and prof_launches_per_token must say so
        launches_ = capture_launch0_;
        if (epoch_ != capture_epoch_) {
            std::fprintf(stderr, "graph_capture_end: a scratch buffer was reallocated DURING the "
                                 "capture on dev%d; the graph would hold a freed pointer\n", dev_);
            std::abort();
        }
        hipGraphExec_t ex = (hipGraphExec_t) *exec;
        if (ex) {
            // Same topology (a new position class of the same kind): only the
            // kernel parameters -- the grids -- moved, so update in place,
            // which is far cheaper than a fresh instantiate.
            hipGraphNode_t err_node = nullptr;
            hipGraphExecUpdateResult r = hipGraphExecUpdateError;
            const hipError_t e = hipGraphExecUpdate(ex, g, &err_node, &r);
            if (e == hipSuccess && r == hipGraphExecUpdateSuccess) {
                ++graph_updates_;
            } else {
                (void) hipGetLastError();          // not sticky; clear it
                HIP_IGNORE(hipGraphExecDestroy(ex));
                ex = nullptr;
            }
        }
        if (!ex) {
            HIP_CHECK(hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
            ++graph_instantiates_;
        }
        HIP_IGNORE(hipGraphDestroy(g));
        *exec = (void *) ex;
        ++graph_captures_;
        graph_capture_ms_ += std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - capture_t0_).count();
    }

    void graph_launch(void * exec) override {
        mark(PC_GRAPH);                    // ensures THIS device
        last_op_ = "graph_launch";
        HIP_CHECK(hipGraphLaunch((hipGraphExec_t) exec, stream_));
        ++graph_replays_;
    }

    void graph_destroy(void * exec) override {
        if (!exec) return;
        dev_ensure(dev_);
        HIP_IGNORE(hipGraphExecDestroy((hipGraphExec_t) exec));
    }

    long long graph_epoch() const override { return epoch_; }

    void graph_report(FILE * out) override {
        std::fprintf(out, "hip_graph_dev%d nodes=%d captures=%lld instantiates=%lld updates=%lld "
                          "replays=%lld capture_ms_total=%.2f",
                     dev_, graph_nodes_last_, graph_captures_, graph_instantiates_,
                     graph_updates_, graph_replays_, graph_capture_ms_);
        if (gate_host_) std::fprintf(out, " prequeue_holds=%u prequeue_timed_out=%u",
                                     gate_seq_, gate_host_[16]);
        std::fprintf(out, "\n");
    }

    // ---- --prequeue-gate (diagnostic): see k_prequeue_gate ----
    void prequeue_hold() override {
        dev_ensure(dev_);
        if (!gate_host_) {
            HIP_CHECK(hipHostMalloc((void **) &gate_host_, 128,
                                    hipHostMallocMapped | hipHostMallocCoherent));
            std::memset((void *) gate_host_, 0, 128);
            HIP_CHECK(hipHostGetDevicePointer((void **) &gate_dev_, (void *) gate_host_, 0));
            int khz = 0;
            if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev_) != hipSuccess || khz <= 0)
                khz = 100000;              // gfx11's 100 MHz constant clock
            gate_limit_ = (unsigned long long) khz * 1000ull * 2ull;   // 2 s
        }
        ++gate_seq_;
        k_prequeue_gate<<<1, 1, 0, stream_>>>(gate_dev_, gate_seq_, gate_dev_ + 16, gate_limit_);
    }
    void prequeue_release() override {
        if (!gate_host_) return;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&gate_host_[0], gate_seq_, __ATOMIC_SEQ_CST);
    }

private:
    // captured launches read the position from here (nullptr = eager path)
    const int * gpos() const { return capturing_ ? gpos_ : nullptr; }
    bool        capturing_ = false;
    const int * gpos_ = nullptr;
    int         graph_bound_ = 0;
    long long   capture_launch0_ = 0;
    long long   capture_epoch_ = 0;
    std::chrono::steady_clock::time_point capture_t0_{};
    // Bumped by every reallocation of a buffer a T = 1 kernel is handed
    // (split-K partials and counters, attention partials, radix candidates):
    // a graph captured before it holds a freed pointer, so the runner
    // recaptures whenever this moved.
    long long   epoch_ = 0;
    int         graph_nodes_last_ = 0;
    long long   graph_captures_ = 0, graph_instantiates_ = 0, graph_updates_ = 0, graph_replays_ = 0;
    double      graph_capture_ms_ = 0.0;
    volatile unsigned * gate_host_ = nullptr;   // [0] the flag, [16] timed-out
    unsigned *          gate_dev_  = nullptr;
    unsigned            gate_seq_  = 0;
    unsigned long long  gate_limit_ = 0;

    // ---- profiling state ----
    // --sync-debug: validate everything enqueued since the PREVIOUS mark, so
    // the op named in the message is the one that actually faulted rather
    // than whichever launch happened to be in flight when the queue drained.
    void debug_check(const char * where) {
        if (!sync_debug_) return;
        const hipError_t es = hipStreamSynchronize(stream_);
        const hipError_t el = hipGetLastError();
        if (es == hipSuccess && el == hipSuccess) return;
        std::fprintf(stderr,
            "\nSYNC-DEBUG FAULT on device %d\n"
            "  failing op   : %s (class %s)\n"
            "  reached at   : %s\n"
            "  context      : %s layer %d\n"
            "  sync says    : %s\n"
            "  lasterr says : %s\n",
            dev_, last_op_ ? last_op_ : "(none)", prof_name(last_cls_), where,
            dbg_phase_ ? dbg_phase_ : "(none)", dbg_layer_,
            hipGetErrorString(es), hipGetErrorString(el));
        std::fflush(stderr);
        std::abort();
    }

    void mark(int cls, int gemv_group = -1, size_t bytes = 0) {
        dev_ensure(dev_);                  // every op runs on its OWN card
        if (cls == PC_TRUNK && trunk_cls_override_ >= 0) cls = trunk_cls_override_;   // ds4_gpu.inc
        debug_check("the next op's mark()");
        last_cls_ = cls;
        ++launches_;                       // one op == one launch, except where noted
        // A capture is not a token's work: nothing it enqueues runs now, its
        // bytes are charged by the replays' interval instead, and an event
        // recorded into a capturing stream would become a graph node. The
        // launch count above is kept -- it is the graph's node count.
        if (capturing_) return;
        if (gemv_group >= 0 && gemv_group < GG_N) gemv_bytes_[gemv_group] += (double) bytes;
        // The sub-GEMV_SMALL_BYTES set, across groups: the launches that cannot
        // fill the card from their own bytes and therefore carry the launch
        // floor. It is reported separately so a change aimed at them is
        // attributable to them.
        const bool small = (gemv_group >= 0 && bytes > 0 && bytes < GEMV_SMALL_BYTES);
        if (small) { gemv_small_bytes_ += (double) bytes; ++gemv_small_launches_; }
        if (!prof_ || n_ev_ >= PROF_MAX_EVENTS) return;
        // Record only when the CLASS CHANGES. The interval from event i to
        // event i+1 is charged to ev_cls_[i], so a run of ops in one class
        // needs one event, not one each -- and hipEventRecord is host work in
        // the issue path, which is exactly what --profile was perturbing. A
        // GEMV always takes its own event, because the per-projection
        // bandwidths are per launch group.
        if (n_ev_ > 0 && cls == ev_cls_[n_ev_ - 1] && gemv_group < 0 &&
            ev_grp_[n_ev_ - 1] < 0) {
            ++ev_merged_;
            return;
        }
        HIP_CHECK(hipEventRecord(ev_[n_ev_], stream_));
        ev_cls_[n_ev_] = cls;
        ev_grp_[n_ev_] = gemv_group;
        ev_small_[n_ev_] = small ? 1 : 0;
        ++n_ev_;
    }

    bool       prof_ = false;
    bool       prof_defer_ = false;
    long long  gemm_lds_calls_ = 0;
    long long  gemm_tile_calls_ = 0;
    bool       sync_debug_ = false;
    const char * last_op_   = nullptr;
    const char * dbg_phase_ = nullptr;
    int        last_cls_ = 0;
    int        dbg_layer_ = -1;
    int        n_ev_ = 0;
    long long  ev_merged_ = 0;
    long long  launches_ = 0;
    long long  blocking_ = 0;              // host-blocking calls, counted always
    long long  blocking_at_token_start_ = 0;
    long long  launches_at_token_start_ = 0;
    int        tokens_profiled_ = 0;
    long long  blocking_in_body_ = 0;
    long long  launches_in_body_ = 0;
    long long  launches_last_ = 0;
    double     prof_us_[PC_N] = {0};
    int        trunk_cls_override_ = -1;   // set around a DeepSeek-V4 call into gemv()
    double     gemv_us_[GG_N]  = {0};
    double     gemv_bytes_[GG_N] = {0};
    double     gemv_small_us_ = 0.0;
    double     gemv_small_bytes_ = 0.0;
    long long  gemv_small_launches_ = 0;
    std::vector<hipEvent_t> ev_;
    std::vector<int>        ev_cls_;
    std::vector<int>        ev_grp_;
    std::vector<int>        ev_small_;

    hipEvent_t ev0_{}, ev1_{};
    float * partial_ = nullptr;
    float * mbuf_    = nullptr;
    float * lbuf_    = nullptr;
    size_t  part_stride_ = 0, ml_stride_ = 0;
    int     qsa_rows_ = 1;         // rows of a chunk this card has scratch for
    int     attn_rows_ = 0;        // rows the attention split scratch is sized for
    uint32_t * cand_key_ = nullptr;
    int      * cand_idx_ = nullptr;
    int        cand_cap_ = 0;
    int        cand_rows_ = 0;

    // A job describing ONE row at position `pos`, so the five QSA kernels
    // have a single code path whatever the caller.
    QsaRowsD one_row(int pos, int ratio, int budget) const {
        QsaRowsD d;
        d.pos0 = pos < 0 ? 0 : pos;
        d.ratio = ratio; d.budget = budget;
        d.blk_stride = d.cell_stride = d.sel_stride = 0;
        d.cand_stride = 0; d.part_stride = 0; d.ml_stride = 0;
        d.dpos = nullptr;
        return d;
    }
    void alloc_attn_rows(int rows) {
        // Idempotent: franken_serve builds one DecodeRunner a SLOT and they
        // all reserve the same row count from the same backend, so without
        // this the second slot frees the first slot buffers and hands back
        // identical ones.
        if (rows == attn_rows_ && partial_) return;
        attn_rows_ = rows;
        const size_t max_chunks = (size_t) ceil_div(MAX_SEL, ATTN_CHUNK);
        part_stride_ = max_chunks * N_Q_HEADS * HEAD_DIM;
        ml_stride_   = max_chunks * N_Q_HEADS;
        HIP_IGNORE(hipFree(partial_)); HIP_IGNORE(hipFree(mbuf_)); HIP_IGNORE(hipFree(lbuf_));
        HIP_CHECK(hipMalloc(&partial_, part_stride_ * (size_t) rows * sizeof(float)));
        HIP_CHECK(hipMalloc(&mbuf_,    ml_stride_   * (size_t) rows * sizeof(float)));
        HIP_CHECK(hipMalloc(&lbuf_,    ml_stride_   * (size_t) rows * sizeof(float)));
        ++epoch_;
    }
    int        gemv_lds_ = 1;
    int        gemv_min_rows_ = 1024;
    // 0 = k_gemm_batch only, the ONE path that is bit-identical to decode.
    int        gemm_lds_ = 0;
    // Bit 0 (gate/up) is MEASURED bit-exact on three cards; bit 1 (down) is
    // measured NOT to be and is off until it is. See decode_backend.h.
    int        expert_gather_ = 3;
    int        moe_tile_ = MOE_TILE_E;   // --moe-tile: 4, 8 or 16
    int        gather_serial_ = 0;       // --expert-gather-serial
    int        moe_sorted_tile_ = 0;     // the tile the current sort was built for
    int        moe_alloc_tile_  = 0;
    int        dev_ = 0;
    char       name_[24] = {0};
    hipStream_t stream_{};
    hipEvent_t  bnd_{};
    float *     amax_v_ = nullptr;
    int *       amax_i_ = nullptr;
    // split-K partials: nsplit*rows stays near GEMV_TARGET_WGS*GEMV_WAVES by
    // construction (nsplit rises only as rows falls), so this is small at
    // T = 1 and grows with the chunk (ensure_part).
    float *    gemv_part_ = nullptr;
    size_t     part_floats_ = 0;
    static constexpr size_t GEMV_PART_FLOATS = 1u << 17;   // 128k floats = 512 KB
    // A GEMV only splits when its row count is under --gemv-min-rows (1024 by
    // default, and the only decode batch under it is hc_down+inject's 324), so
    // 16k rows of counters is four times any setting that splits at all.
    int *      gemv_cnt_ = nullptr;
    size_t     cnt_rows_ = 0;
    static constexpr size_t GEMV_CNT_ROWS = 1u << 14;
    int        gemv_fused_reduce_  = 1;
    int        gemv_fuse_collapse_ = 1;
    int        gemv_burst_         = GEMV_BURST;
    static constexpr int    STAGE_SLOTS = 8;
    void *     stage_[STAGE_SLOTS] = {nullptr};
    hipEvent_t stage_ev_[STAGE_SLOTS] = {};
    bool       stage_used_[STAGE_SLOTS] = {false};
    size_t     stage_bytes_ = 1u << 16;                    // 64 KB, grown on demand
    int        stage_next_ = 0;
    // The residual banks of the pipeline (decode_graph.h N_RES_BANKS).
    // `drained_[b]` is recorded on THIS device's stream once it has copied
    // bank b out of some other card; `sink_[b]` is the card that did that to
    // US, so boundary_wait_free knows whose event to wait on.
    static constexpr int N_BANKS = 2;
    hipEvent_t   drained_[N_BANKS] = {};
    GpuBackend * sink_[N_BANKS] = {nullptr, nullptr};
    size_t placed_ = 0;
};

// DeepSeek-V4 (L5 step 2): its kernels and Ds4GpuOps. Appended AFTER every
// qwen4exp kernel and GpuBackend, and nothing above reads anything in it.
#include "ds4_gpu.inc"

} // namespace

Backend * make_gpu_backend(int device) { return new GpuBackend(device); }

namespace ds4 {
Ds4Ops * make_ds4_gpu_ops(Backend & be) {
    if (!be.is_gpu()) throw std::runtime_error("make_ds4_gpu_ops: not a GPU backend");
    return new Ds4GpuOps(static_cast<GpuBackend &>(be));
}
} // namespace ds4

// P2P between every pair, once, before anything is placed. All six pairs can
// and do on this box (M5: 16-24 GB/s, 28-32 us for a 10 KB hidden vector).
void enable_peer_access(int n_devices) {
    for (int a = 0; a < n_devices; ++a) {
        HIP_CHECK(hipSetDevice(a));
        for (int b = 0; b < n_devices; ++b) {
            if (a == b) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, a, b));
            if (!can) {
                std::fprintf(stderr, "device %d cannot peer-access %d; boundaries would "
                                     "stage through the host\n", a, b);
                continue;
            }
            const hipError_t e = hipDeviceEnablePeerAccess(b, 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                std::fprintf(stderr, "hipDeviceEnablePeerAccess(%d->%d): %s\n",
                             a, b, hipGetErrorString(e));
            }
        }
    }
}

} // namespace fk
