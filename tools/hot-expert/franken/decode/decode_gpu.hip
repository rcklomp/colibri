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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstring>

#define FK_QUAL __device__ __forceinline__
#define FK_HALF_TO_F32(bits) __half2float(__ushort_as_half((unsigned short)(bits)))

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "decode_quant.h"
#include "decode_backend.h"

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
#define GEMV_BATCH_MAX     8
#define GEMV_LDS_MAX_FLOATS 4096
#ifndef GEMV_UNROLL
#define GEMV_UNROLL 4
#endif

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

__global__ __launch_bounds__(GEMV_THREADS)
void k_gemv_batch(GemvBatchD b, const float * __restrict__ x,
                  const float * __restrict__ x2, int nsplit,
                  float * __restrict__ part, int stage_x) {
    extern __shared__ float sx[];

    const int wave = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int grow = blockIdx.x * GEMV_WAVES + wave;   // concatenated row index

    // Element range of this split, in units of 32 so every lane stays in step.
    // Q6_K super-blocks are 256 weights, so a batch containing one splits on
    // 256-element boundaries; everything else splits on 32.
    const int gran = (b.has_q6k ? 256 : WAVE);
    int c0, c1;
    split_range((b.K + gran - 1) / gran, nsplit, blockIdx.y, &c0, &c1);
    const int e0 = c0 * gran;
    const int e1 = min(b.K, c1 * gran);

    const float * xs = x + e0;
    if (stage_x) {
        for (int i = threadIdx.x; i < e1 - e0; i += GEMV_THREADS) {
            const float v = x[e0 + i];
            sx[i] = (b.x_mode == GX_SILU_MUL) ? d_silu(v) * x2[e0 + i] : v;
        }
        __syncthreads();
        xs = sx;
    }

    if (grow >= b.total_rows) return;
    const int  di  = desc_of(b, grow);
    const GemvDesc d = b.d[di];
    const int  row = grow - d.row_off;

    float acc = 0.0f;
    if (d.type == FK_Q_Q8_0) {
        // the split is on 32-element boundaries, which is exactly a Q8_0 block
        const unsigned char * rp = (const unsigned char *) d.base + (size_t) row * d.row_stride;
        int bl = e0 >> 5, bend = e1 >> 5;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (; bl + GEMV_UNROLL - 1 < bend; bl += GEMV_UNROLL) {
            a0 += fk_q8_0_lane_dot(rp + (size_t)(bl+0) * FK_Q8_0_BLOCK_BYTES, xs + ((bl+0) << 5) - e0, lane);
            a1 += fk_q8_0_lane_dot(rp + (size_t)(bl+1) * FK_Q8_0_BLOCK_BYTES, xs + ((bl+1) << 5) - e0, lane);
            a2 += fk_q8_0_lane_dot(rp + (size_t)(bl+2) * FK_Q8_0_BLOCK_BYTES, xs + ((bl+2) << 5) - e0, lane);
            a3 += fk_q8_0_lane_dot(rp + (size_t)(bl+3) * FK_Q8_0_BLOCK_BYTES, xs + ((bl+3) << 5) - e0, lane);
        }
        for (; bl < bend; ++bl)
            a0 += fk_q8_0_lane_dot(rp + (size_t) bl * FK_Q8_0_BLOCK_BYTES, xs + (bl << 5) - e0, lane);
        acc = (a0 + a1) + (a2 + a3);
    } else if (d.type == FK_Q_Q6_K) {
        const unsigned char * rp = (const unsigned char *) d.base + (size_t) row * d.row_stride;
        int bl = e0 >> 8, bend = e1 >> 8;          // 256 weights a super-block
        const int xoff = lane << 3;
        float a0 = 0.0f, a1 = 0.0f;
        for (; bl + 1 < bend; bl += 2) {
            a0 += fk_q6k_block_dot(rp + (size_t)(bl+0) * FK_Q6K_BLOCK_BYTES,
                                   xs + ((bl+0) << 8) - e0 + xoff, lane);
            a1 += fk_q6k_block_dot(rp + (size_t)(bl+1) * FK_Q6K_BLOCK_BYTES,
                                   xs + ((bl+1) << 8) - e0 + xoff, lane);
        }
        for (; bl < bend; ++bl)
            a0 += fk_q6k_block_dot(rp + (size_t) bl * FK_Q6K_BLOCK_BYTES,
                                   xs + (bl << 8) - e0 + xoff, lane);
        acc = a0 + a1;
    } else if (d.type == FK_Q_BF16) {
        const unsigned short * rp = (const unsigned short *) d.base + (size_t) row * d.row_stride;
        for (int i = e0 + lane; i < e1; i += WAVE)
            acc = fmaf(fk_bf16_to_f32(rp[i]), xs[i - e0], acc);
    } else {
        const float * rp = (const float *) d.base + (size_t) row * d.row_stride;
        for (int i = e0 + lane; i < e1; i += WAVE)
            acc = fmaf(rp[i], xs[i - e0], acc);
    }

    const float sum = wave_sum(acc);
    if (lane != 0) return;
    if (nsplit > 1) part[(size_t) blockIdx.y * b.total_rows + grow] = sum;
    else            d.out[row] = apply_epi(sum, d.epi, d.arg);
}

// Fixed-order sum of the split partials, then each matrix's epilogue and its
// own destination. Reproducible, unlike an atomicAdd -- the oracle needs it.
__global__ void k_reduce_splits_batch(GemvBatchD b, const float * __restrict__ part,
                                      int nsplit) {
    const int grow = blockIdx.x * blockDim.x + threadIdx.x;
    if (grow >= b.total_rows) return;
    float a = 0.0f;
    for (int s = 0; s < nsplit; ++s) a += part[(size_t) s * b.total_rows + grow];
    const int di = desc_of(b, grow);
    b.d[di].out[grow - b.d[di].row_off] = apply_epi(a, b.d[di].epi, b.d[di].arg);
}

// ======================================================= routed experts ===
//
// grid = (rows, K_TOP, 2): z == 0 is gate, z == 1 is up. One wave per row.

__global__ __launch_bounds__(WAVE, 16)
void k_moe_gate_up_iq3s(const unsigned char * __restrict__ gate_base,
                        const unsigned char * __restrict__ up_base,
                        size_t expert_stride, size_t row_bytes,
                        const int * __restrict__ ids, const float * __restrict__ x,
                        float * __restrict__ y_gate, float * __restrict__ y_up,
                        int rows, int K, int k_top) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int rank = blockIdx.y;
    if (rank >= k_top) return;
    if (K & (IQ3S_BLOCK_WEIGHTS - 1)) return;            // bounded-safe
    const int which = blockIdx.z;
    const int tid   = threadIdx.x;

    const unsigned char * A = (which ? up_base : gate_base)
                            + (size_t) ids[rank] * expert_stride + (size_t) row * row_bytes;

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
    if (tid == 0) (which ? y_up : y_gate)[(size_t) rank * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_gate_up_iq4xs(const unsigned char * __restrict__ gate_base,
                         const unsigned char * __restrict__ up_base,
                         size_t expert_stride, size_t row_bytes,
                         const int * __restrict__ ids, const float * __restrict__ x,
                         float * __restrict__ y_gate, float * __restrict__ y_up,
                         int rows, int K, int k_top) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int rank = blockIdx.y;
    if (rank >= k_top) return;
    if (K & (FK_IQ4XS_BLOCK_WEIGHTS - 1)) return;        // bounded-safe
    const int which = blockIdx.z;
    const int tid   = threadIdx.x;

    const unsigned char * A = (which ? up_base : gate_base)
                            + (size_t) ids[rank] * expert_stride + (size_t) row * row_bytes;
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
    if (tid == 0) (which ? y_up : y_gate)[(size_t) rank * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_down_iq4nl(const unsigned char * __restrict__ base, size_t expert_stride,
                      size_t row_bytes, const int * __restrict__ ids,
                      const float * __restrict__ h, float * __restrict__ eo,
                      int rows, int K, int k_top) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int rank = blockIdx.y;
    if (rank >= k_top) return;
    if (K & (IQ4NL_BLOCK_WEIGHTS - 1)) return;           // bounded-safe
    const int tid = threadIdx.x;

    const unsigned char * A = base + (size_t) ids[rank] * expert_stride + (size_t) row * row_bytes;
    const float * hr = h + (size_t) rank * K;
    const int nchunk = (K >> 5) << 2;

    float lo = 0.0f, hi = 0.0f;
    for (int w = tid; w < nchunk; w += WAVE) m1n_iq4nl_chunk_dot(A, c_kvals_iq4nl, hr, w, &lo, &hi);

    const float s = wave_sum(lo + hi);
    if (tid == 0) eo[(size_t) rank * rows + row] = s;
}

__global__ __launch_bounds__(WAVE, 16)
void k_moe_down_q8_0(const unsigned char * __restrict__ base, size_t expert_stride,
                     size_t row_bytes, const int * __restrict__ ids,
                     const float * __restrict__ h, float * __restrict__ eo,
                     int rows, int K, int k_top) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int rank = blockIdx.y;
    if (rank >= k_top) return;
    if (K % FK_Q8_0_BLOCK_WEIGHTS) return;               // bounded-safe
    const int tid = threadIdx.x;

    const unsigned char * A = base + (size_t) ids[rank] * expert_stride + (size_t) row * row_bytes;
    const float * hr = h + (size_t) rank * K;
    const int nblk = K / FK_Q8_0_BLOCK_WEIGHTS;

    float acc = 0.0f;
    for (int b = 0; b < nblk; ++b)
        acc += fk_q8_0_lane_dot(A + (size_t) b * FK_Q8_0_BLOCK_BYTES,
                                hr + b * FK_Q8_0_BLOCK_WEIGHTS, tid);

    const float s = wave_sum(acc);
    if (tid == 0) eo[(size_t) rank * rows + row] = s;
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
              float * __restrict__ w) {
    __shared__ float sv[N_EXPERT];
    __shared__ int   si[N_EXPERT];
    __shared__ float sp[N_EXPERT];
    __shared__ int   taken[N_EXPERT_USED];
    const int t = threadIdx.x;

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
    }
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int k = 0; k < N_EXPERT_USED; ++k) s += sp[taken[k]];
        s = fmaxf(s, 6.103515625e-5f);           // build_moe_ffn's f16 clamp
        for (int k = 0; k < N_EXPERT_USED; ++k) w[k] = sp[taken[k]] / s;
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

// One workgroup per group; `w_tiled` picks between a [ne0] gamma reused by
// every group (build_norm) and a [n_groups*ne0] one (build_hc_mix's grouped
// RMSNorm over the hc streams).
// The hc modules normalise only HC=4 groups, so the grid is four workgroups
// whatever the block size: the only lever left is making each workgroup
// wider, which shortens the strided walk over ne0=2560 from ten steps to
// two or three. Block size is chosen by the host and the LDS is dynamic.
__global__ void k_rms_norm_mul(const float * __restrict__ x, const float * __restrict__ w,
                    float * __restrict__ y, int ne0, float eps, int w_tiled, int has_w) {
    extern __shared__ float lds[];
    const float * xg = x + (size_t) blockIdx.x * ne0;
    float *       yg = y + (size_t) blockIdx.x * ne0;

    float s = 0.0f;
    for (int i = threadIdx.x; i < ne0; i += blockDim.x) s = fmaf(xg[i], xg[i], s);
    s = block_sum(s, lds);
    const float sc = rsqrtf(s / (float) ne0 + eps);
    const float * wg = has_w ? (w_tiled ? w : w + (size_t) blockIdx.x * ne0) : nullptr;
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

__global__ void k_hc_broadcast(const float * __restrict__ x, float * __restrict__ res) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_EMBD) return;
    const float v = x[i];
#pragma unroll
    for (int c = 0; c < HC; ++c) res[c * N_EMBD + i] = v;
}

__global__ void k_hc_collapse(const float * __restrict__ xn, const float * __restrict__ gate,
                              float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_EMBD) return;
    float a = xn[i] * gate[i];
#pragma unroll
    for (int c = 1; c < HC; ++c) a += xn[c * N_EMBD + i] * gate[c * N_EMBD + i];
    out[i] = a * (1.0f / (float) HC);
}

// ============================================================== PLE ======

__global__ void k_ple_gate(const float * __restrict__ key, const float * __restrict__ query,
                const float * __restrict__ value, float * __restrict__ gated,
                float * __restrict__ gate_out) {
    extern __shared__ float lds[];
    const int c = blockIdx.x;
    const size_t o = (size_t) c * N_EMBD;
    float s = 0.0f;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) s = fmaf(key[o + i], query[o + i], s);
    s = block_sum(s, lds);
    const float sf  = s * rsqrtf((float) N_EMBD);
    const float mag = sqrtf(fmaxf(fabsf(sf), 1e-6f));      // ggml_sqrt(ggml_clamp(|s|,1e-6,inf))
    const float sgn = sf > 0.0f ? 1.0f : (sf < 0.0f ? -1.0f : 0.0f);
    const float g   = d_sigmoid(sgn * mag);
    if (threadIdx.x == 0) gate_out[c] = g;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) gated[o + i] = value[i] * g;
}

// ggml_ssm_conv generalised by a dilation: tap k reads (kern-1-k)*dil
// positions back, which is build_ple's shifted-copy sum and, at dil = 1, the
// GDN conv1d. The ring slot is computed with a positive modulus.
__global__ void k_conv_taps(const float * __restrict__ ring, int head, int ring_slots,
                            int kern, int dil, int channels,
                            const float * __restrict__ w, float * __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float a = 0.0f;
    for (int k = 0; k < kern; ++k) {
        const int back = (kern - 1 - k) * dil;
        int slot = (head - back) % ring_slots;
        if (slot < 0) slot += ring_slots;
        a = fmaf(ring[(size_t) slot * channels + c], w[(size_t) c * kern + k], a);
    }
    out[c] = d_silu(a);
}

// ===================================================== Gated DeltaNet ====
//
// One wave per (v-head, value row j). The row is the 128 KEY entries of the
// state, contiguous in ggml's layout, so the whole step -- decay, the
// prediction dot, the rank-1 update and the output dot -- happens in four
// registers per lane with two wave reductions and no LDS.
__global__ __launch_bounds__(WAVE)
void k_gdn_step(float * __restrict__ S, const float * __restrict__ q,
                const float * __restrict__ k, const float * __restrict__ v,
                const float * __restrict__ g_exp, const float * __restrict__ beta,
                float * __restrict__ out) {
    constexpr int PER = GDN_STATE / WAVE;                 // 4
    const int h    = blockIdx.x / GDN_STATE;
    const int j    = blockIdx.x - h * GDN_STATE;
    const int lane = threadIdx.x;
    const int hk   = h % GDN_K_HEADS;                     // ggml_repeat / `iv1 % nek1`

    float * row = S + ((size_t) h * GDN_STATE + j) * GDN_STATE;
    const float * kh = k + (size_t) hk * GDN_STATE;
    const float * qh = q + (size_t) hk * GDN_STATE;

    const float ge = g_exp[h];
    float r[PER], kk[PER], qq[PER];
    float sk = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int i = lane + p * WAVE;
        r[p]  = row[i] * ge;                              // decay FIRST (ops.cpp:11012)
        kk[p] = kh[i];
        qq[p] = qh[i];
        sk    = fmaf(r[p], kk[p], sk);
    }
    sk = wave_sum(sk);

    const float d = beta[h] * (v[(size_t) h * GDN_STATE + j] - sk);
    float o = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        r[p] = fmaf(kk[p], d, r[p]);
        row[lane + p * WAVE] = r[p];
        o = fmaf(r[p], qq[p], o);
    }
    o = wave_sum(o);
    if (lane == 0) out[(size_t) h * GDN_STATE + j] = o * rsqrtf((float) GDN_STATE);
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
__global__ void k_gdn_conv_gate(const float * __restrict__ ring, int head,
                                const float * __restrict__ w,
                                const float * __restrict__ beta_raw,
                                const float * __restrict__ alpha_raw,
                                const float * __restrict__ dt,
                                const float * __restrict__ ssm_a,
                                float * __restrict__ conv_out,
                                float * __restrict__ beta_sig,
                                float * __restrict__ a_softplus,
                                float * __restrict__ gate,
                                float * __restrict__ g_exp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < GDN_CONV_DIM) {
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < GDN_CONV_K; ++k) {
            int slot = (head - (GDN_CONV_K - 1 - k)) % GDN_CONV_K;
            if (slot < 0) slot += GDN_CONV_K;
            a = fmaf(ring[(size_t) slot * GDN_CONV_DIM + i], w[(size_t) i * GDN_CONV_K + k], a);
        }
        conv_out[i] = d_silu(a);
        return;
    }
    const int h = i - GDN_CONV_DIM;
    if (h >= GDN_V_HEADS) return;
    beta_sig[h]   = d_sigmoid(beta_raw[h]);
    const float sp = d_softplus(alpha_raw[h] + dt[h]);
    a_softplus[h] = sp;
    const float g = sp * ssm_a[h];          // ssm_a already holds -exp(A_log)
    gate[h]       = g;
    g_exp[h]      = expf(g);
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
                             int rows, int k_top) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float g0 = d_sigmoid(sh_gate[0]);
    if (i == 0) sh_gate_sig[0] = g0;
    if (i >= rows) return;
    float a = 0.0f;
    for (int k = 0; k < k_top; ++k) a += w[k] * eo[(size_t) k * rows + i];
    moe_out[i] = a;
    const float g = sh_raw[i] * g0;
    sh_gated[i] = g;
    out[i] = a + g;
}

// hc_combine, optionally followed by the grouped RMSNorm the NEXT hc_mix
// would have launched on its own. One workgroup per hc stream: the combine is
// elementwise over that stream's 2 560 elements and the norm reduces over
// exactly the same range, so the two fuse with one __syncthreads and no
// extra traffic. `w_norm == nullptr` does the combine alone.
__global__ void k_hc_combine_norm(float * __restrict__ res, const float * __restrict__ blk,
                                  const float * __restrict__ inject,
                                  const float * __restrict__ w_norm,
                                  float * __restrict__ xn, float eps) {
    extern __shared__ float lds[];
    const int c = blockIdx.x;
    float * rc = res + (size_t) c * N_EMBD;
    const float wgt = 2.0f * d_sigmoid(inject[c] * (1.0f / (float) HC));

    float ss = 0.0f;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x) {
        const float v = rc[i] + blk[i] * wgt;
        rc[i] = v;
        ss = fmaf(v, v, ss);
    }
    if (!w_norm) return;
    ss = block_sum(ss, lds);
    const float sc = rsqrtf(ss / (float) N_EMBD + eps);
    const float * wg = w_norm + (size_t) c * N_EMBD;
    for (int i = threadIdx.x; i < N_EMBD; i += blockDim.x)
        xn[(size_t) c * N_EMBD + i] = rc[i] * sc * wg[i];
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
                   int pos, float eps) {
    __shared__ float lds[QKPOST_THREADS];
    __shared__ float sv[HEAD_DIM];

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
                                int cell) {
    // one workgroup per (tensor, head, 32-element block); 32 lanes
    const int which = blockIdx.z;                         // 0 = K, 1 = V
    const int h     = blockIdx.y;
    const int b     = blockIdx.x;
    const int lane  = threadIdx.x;

    const float * src = (which ? v : k) + (size_t) h * HEAD_DIM + b * 32;
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

__global__ void k_idx_raw_store(const float * __restrict__ kr, float * __restrict__ raw, int cell) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < IDX_DIM) raw[(size_t) cell * IDX_DIM + i] = kr[i];
}

// One workgroup of IDX_DIM threads: mean-pool the block's members, rms-norm
// with index_k_norm, IMRoPE at the block's first token position, store bf16.
// Missing members read cell 0 -- set_input_qsa zero-fills blk_cells
// (llama-memory-hybrid-idx.cpp:395) and never revisits an incomplete block's
// pooled value, because the tail block's +1e9 bias selects it regardless.
__global__ __launch_bounds__(IDX_DIM)
void k_idx_pool_block(const float * __restrict__ raw, int blk, int n_filled,
                      const float * __restrict__ k_norm, float eps,
                      unsigned short * __restrict__ pooled,
                      float * __restrict__ dbg_pooled, float * __restrict__ dbg_roped) {
    __shared__ float s_v[IDX_DIM];
    __shared__ float s_r[IDX_DIM];
    const int i = threadIdx.x;

    float p = 0.0f;
    for (int m = 0; m < QSA_RATIO; ++m) {
        const int cell = (m < n_filled) ? blk * QSA_RATIO + m : 0;
        p += raw[(size_t) cell * IDX_DIM + i];
    }
    p *= 1.0f / (float) QSA_RATIO;
    if (dbg_pooled) dbg_pooled[i] = p;

    s_v[i] = p * p;
    __syncthreads();
    for (int s = IDX_DIM / 2; s > 0; s >>= 1) {
        if (i < s) s_v[i] += s_v[i + s];
        __syncthreads();
    }
    p = p * rsqrtf(s_v[0] / (float) IDX_DIM + eps) * k_norm[i];
    s_r[i] = p;
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

    if (dbg_roped) dbg_roped[i] = s_r[i];
    pooled[(size_t) blk * IDX_DIM + i] = fk_f32_to_bf16(s_r[i]);
}

// scores[b] = sum over the FOUR QUERY heads of relu(q_h . kpool[b]).
// One 32-lane wave per pooled block; a lane owns 4 of the 128 dims (one
// 8-byte load, fully coalesced across the wave) and keeps its four query
// elements per head in registers, so the pooled-key array is streamed once
// and nothing else is touched.
#define IDXS_THREADS 256
__global__ __launch_bounds__(IDXS_THREADS)
void k_idx_scan(const unsigned short * __restrict__ pooled, const float * __restrict__ q,
                float * __restrict__ scores, int n_blocks) {
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
                             int n_kv, int q_pos, int ratio, int tail_start,
                             int sel_identity) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_kv) return;
    if (sel_identity) sel[j] = j;
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
void k_topk_radix_select(const float * __restrict__ scores, int n, int k_want,
                         uint32_t * __restrict__ cand_key, int * __restrict__ cand_idx,
                         int * __restrict__ out) {
    __shared__ uint32_t s_hist[TOPK_HI_BINS];
    __shared__ uint32_t s_scan[TOPK_THREADS];
    __shared__ int      s_found;
    __shared__ uint32_t s_above, s_prefix;
    __shared__ int      s_need, s_cand, s_out, s_tie;

    const int t = threadIdx.x;
    const int k = k_want < n ? k_want : n;

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
                        float * __restrict__ lbuf, int n_sel) {
    __shared__ float sh_p[ATTN_WAVES][ATTN_CHUNK];
    __shared__ int   sh_tok[ATTN_CHUNK];

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
                    int n_chunks) {
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

// ========================================================== backend ======

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
    PC_EXP_COMBINE,
    PC_COPY,        // device-to-device copies
    PC_BOUNDARY,    // the hyper-connection residual crossing to the next card
    PC_ARGMAX,      // greedy sampling on the device
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
        case PC_EXP_COMBINE:  return "expert_combine";
        case PC_COPY:         return "copy";
        case PC_BOUNDARY:     return "boundary_p2p";
        case PC_ARGMAX:       return "argmax";
        default:              return "copy";
    }
}
constexpr int PROF_MAX_EVENTS = 8192;

class GpuBackend : public Backend {
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
            HIP_CHECK(hipHostMalloc(&stage_[i], STAGE_BYTES, hipHostMallocDefault));
        // the attention/top-k scratch is sized for the worst case the runner
        // can present; both are small and allocated once
        const int max_chunks = ceil_div(MAX_SEL, ATTN_CHUNK);
        HIP_CHECK(hipMalloc(&partial_, (size_t) max_chunks * N_Q_HEADS * HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMalloc(&mbuf_,    (size_t) max_chunks * N_Q_HEADS * sizeof(float)));
        HIP_CHECK(hipMalloc(&lbuf_,    (size_t) max_chunks * N_Q_HEADS * sizeof(float)));
        HIP_CHECK(hipMalloc(&gemv_part_, GEMV_PART_FLOATS * sizeof(float)));
        HIP_CHECK(hipMalloc(&amax_v_, ARGMAX_BLOCKS * sizeof(float)));
        HIP_CHECK(hipMalloc(&amax_i_, ARGMAX_BLOCKS * sizeof(int)));
        HIP_CHECK(hipEventCreateWithFlags(&bnd_, hipEventDisableTiming));
        // the candidate lists are sized by the CACHE depth, not the budget,
        // and grown on demand by topk_select()
    }
    ~GpuBackend() override {
        HIP_IGNORE(hipFree(partial_)); HIP_IGNORE(hipFree(mbuf_)); HIP_IGNORE(hipFree(lbuf_));
        HIP_IGNORE(hipFree(cand_key_)); HIP_IGNORE(hipFree(cand_idx_));
        HIP_IGNORE(hipFree(gemv_part_));
        for (int i = 0; i < STAGE_SLOTS; ++i) if (stage_[i]) HIP_IGNORE(hipHostFree(stage_[i]));
        HIP_IGNORE(hipFree(amax_v_)); HIP_IGNORE(hipFree(amax_i_));
        HIP_IGNORE(hipEventDestroy(bnd_)); HIP_IGNORE(hipStreamDestroy(stream_));
        for (auto & e : ev_) HIP_IGNORE(hipEventDestroy(e));
        HIP_IGNORE(hipEventDestroy(ev0_)); HIP_IGNORE(hipEventDestroy(ev1_));
    }

    const char * name() const override { return name_; }
    int device() const override { return dev_; }

    // The destination records nothing of its own: it waits on an event the
    // SOURCE stream recorded, then pulls the bytes on its own stream. No host
    // call, so a boundary costs the P2P time M5 measured and nothing else.
    void boundary_recv(void * dst, Backend & src, const void * src_ptr, size_t bytes) override {
        GpuBackend & s = static_cast<GpuBackend &>(src);
        mark(PC_BOUNDARY);
        HIP_CHECK(hipEventRecord(s.bnd_, s.stream_));
        HIP_CHECK(hipStreamWaitEvent(stream_, s.bnd_, 0));
        HIP_CHECK(hipMemcpyPeerAsync(dst, dev_, src_ptr, s.dev_, bytes, stream_));
    }

    void argmax(const float * logits, int n, int * out_id) override {
        mark(PC_ARGMAX);
        k_argmax_partial<<<ARGMAX_BLOCKS, ARGMAX_THREADS, 0, stream_>>>(logits, n, amax_v_, amax_i_);
        ++launches_;
        k_argmax_final<<<1, ARGMAX_BLOCKS, 0, stream_>>>(amax_v_, amax_i_, out_id);
    }

    void vram_report(FILE * out, const char * tag) override {
        size_t freeb = 0, totalb = 0;
        HIP_CHECK(hipSetDevice(dev_));
        HIP_CHECK(hipMemGetInfo(&freeb, &totalb));
        std::fprintf(out, "vram dev%d %s: used=%.2f GB free=%.2f GB total=%.2f GB placed=%.2f GB\n",
                     dev_, tag, (totalb - freeb) / 1e9, freeb / 1e9, totalb / 1e9, placed_ / 1e9);
    }
    bool is_gpu() const override { return true; }

    float * alloc_f32(size_t n) override { return (float *) alloc_raw(n * sizeof(float)); }
    int *   alloc_i32(size_t n) override { return (int   *) alloc_raw(n * sizeof(int)); }
    void *  alloc_raw(size_t b) override {
        void * p = nullptr;
        if (b == 0) b = 4;
        HIP_CHECK(hipMalloc(&p, b));
        HIP_CHECK(hipMemset(p, 0, b));
        return p;
    }
    void free_buf(void * p) override { HIP_IGNORE(hipFree(p)); }
    void upload(void * d, const void * s, size_t b) override {
        if (b <= STAGE_BYTES) {
            void * st = stage_[stage_next_];
            stage_next_ = (stage_next_ + 1) % STAGE_SLOTS;
            std::memcpy(st, s, b);
            HIP_CHECK(hipMemcpyAsync(d, st, b, hipMemcpyHostToDevice, stream_));
            return;                        // stream-ordered, does NOT block the host
        }
        ++blocking_;
        HIP_CHECK(hipMemcpy(d, s, b, hipMemcpyHostToDevice));
    }
    void download(void * d, const void * s, size_t b) override {
        ++blocking_;                       // the host reads it: this always blocks
        HIP_CHECK(hipStreamSynchronize(stream_));
        HIP_CHECK(hipMemcpy(d, s, b, hipMemcpyDeviceToHost));
    }
    void sync() override { ++blocking_; HIP_CHECK(hipStreamSynchronize(stream_)); }

    const void * place(const void * host, size_t bytes, const char * what) override {
        void * p = nullptr;
        const hipError_t e = hipMalloc(&p, bytes);
        if (e != hipSuccess) {
            std::fprintf(stderr, "out of VRAM placing %s (%.2f MB): %s\n",
                         what, bytes / 1e6, hipGetErrorString(e));
            std::abort();
        }
        HIP_CHECK(hipMemcpy(p, host, bytes, hipMemcpyHostToDevice));
        placed_ += bytes;
        return p;
    }
    size_t placed_bytes() const override { return placed_; }

    // There is exactly ONE GEMV path on the device now: everything goes
    // through k_gemv_batch, so a single-matrix call is a batch of one.
    void gemv(const Mat & W, const float * x, float * y) override {
        GemvJob j; j.W = &W; j.out = y;
        gemv_batch(&j, 1, x, -1, GX_PLAIN, nullptr);
    }

    void gemv_batch(const GemvJob * jobs, int n, const float * x, int group,
                    int x_mode, const float * x2) override {
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
            b.d[i].row_stride = (W.type == FK_Q_Q8_0) ? W.row_bytes : (size_t) W.K;
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

        // total waves == total rows, so wider workgroups never add
        // parallelism; splitting K is the only thing that does.
        const int row_groups = ceil_div(b.total_rows, GEMV_WAVES);
        const int gran  = b.has_q6k ? 256 : WAVE;
        const int units = ceil_div(b.K, gran);
        // total waves == total rows, so the test is on ROWS: a matrix with
        // enough rows already fills the card and a split would only add a
        // reduce launch.
        int nsplit = (b.total_rows >= gemv_min_rows_) ? 1 : ceil_div(GEMV_TARGET_WGS, row_groups);
        if (nsplit > units) nsplit = units;
        if (nsplit < 1) nsplit = 1;
        while (nsplit > 1 && (size_t) nsplit * b.total_rows > GEMV_PART_FLOATS) nsplit >>= 1;

        const int slice = ceil_div(units, nsplit) * WAVE;
        const int stage = (gemv_lds_ && slice <= GEMV_LDS_MAX_FLOATS) ? 1 : 0;
        const size_t lds = stage ? (size_t) slice * sizeof(float) : 0;

        mark(PC_TRUNK, group, bytes);
        k_gemv_batch<<<dim3((unsigned) row_groups, (unsigned) nsplit), GEMV_THREADS, lds, stream_>>>(
            b, x, x2, nsplit, gemv_part_, stage);
        if (nsplit > 1) {
            ++launches_;
            k_reduce_splits_batch<<<ceil_div(b.total_rows, BLK), BLK, 0, stream_>>>(b, gemv_part_, nsplit);
        }
    }
    void set_gemv_lds(int on) override { gemv_lds_ = on; }
    void set_gemv_min_rows(int r) override { gemv_min_rows_ = r > 0 ? r : 1; }
    void gemv_expert(const Mat & W, int e, const float * x, float * y) override {
        Mat slice = W;
        slice.base = (const unsigned char *) W.base + (size_t) e * W.expert_stride;
        slice.expert_stride = 0;
        gemv(slice, x, y);
    }

    void copy(float * d, const float * s, size_t n) override {
        mark(PC_COPY);
        HIP_CHECK(hipMemcpyAsync(d, s, n * sizeof(float), hipMemcpyDeviceToDevice, 0));
    }
    void unary(int op, const float * x, float * y, size_t n) override {
        mark(PC_ELEM);
        k_unary<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(op, x, y, (int) n);
    }
    void binary(int op, const float * a, const float * b, float * y, size_t n) override {
        mark(PC_ELEM);
        k_binary<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(op, a, b, y, (int) n);
    }
    void scale(const float * x, float s, float * y, size_t n) override {
        mark(PC_ELEM);
        k_scale<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(x, s, y, (int) n);
    }
    void binary3_add(const float * a, const float * b, const float * c,
                     float * y, size_t n) override {
        mark(PC_ELEM);
        k_binary3_add<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(a, b, c, y, (int) n);
    }
    void mul_tiled(const float * x, const float * w, float * y, size_t n, size_t ne0_w) override {
        mark(PC_ELEM);
        k_mul_tiled<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(x, w, y, (int) n, (int) ne0_w);
    }
    void gather_strided(const float * src, float * dst, int n_rows, int row_len,
                        int src_stride, int src_off) override {
        mark(PC_ELEM);
        k_gather_strided<<<ceil_div(n_rows * row_len, BLK), BLK, 0, stream_>>>(src, dst, n_rows, row_len, src_stride, src_off);
    }
    void rms_norm_mul(const float * x, const float * w, float * y,
                      int ne0, int n_groups, size_t w_len, float eps) override {
        mark(PC_NORM);
        const int t = red_threads(ne0);
        k_rms_norm_mul<<<n_groups, t, t * sizeof(float), stream_>>>(x, w, y, ne0, eps,
                                          w_len == (size_t) ne0 ? 1 : 0, w ? 1 : 0);
    }
    void l2_norm(const float * x, float * y, int ne0, int n_groups, float eps) override {
        mark(PC_NORM);
        const int t = red_threads(ne0);
        k_l2_norm<<<n_groups, t, t * sizeof(float), stream_>>>(x, y, ne0, eps);
    }
    void hc_broadcast(const float * x, float * r) override {
        mark(PC_ELEM);
        k_hc_broadcast<<<ceil_div(N_EMBD, BLK), BLK, 0, stream_>>>(x, r);
    }
    void hc_collapse(const float * xn, const float * g, float * o) override {
        mark(PC_ELEM);
        k_hc_collapse<<<ceil_div(N_EMBD, BLK), BLK, 0, stream_>>>(xn, g, o);
    }
    void hc_combine(float * r, const float * b, const float * inj) override {
        hc_combine_norm(r, b, inj, nullptr, nullptr, 0.0f);
    }
    void hc_combine_norm(float * r, const float * b, const float * inj,
                         const float * w_norm, float * xn, float eps) override {
        mark(PC_ELEM);
        const int t = red_threads(N_EMBD);
        k_hc_combine_norm<<<HC, t, t * sizeof(float), stream_>>>(r, b, inj, w_norm, xn, eps);
    }
    void ple_gate(const float * k, const float * q, const float * v, float * g, float * go) override {
        mark(PC_ELEM);
        const int t = red_threads(N_EMBD);
        k_ple_gate<<<HC, t, t * sizeof(float), stream_>>>(k, q, v, g, go);
    }
    void ple_conv_silu(const float * ring, int head, const float * w, float * out) override {
        mark(PC_ELEM);
        k_conv_taps<<<ceil_div(HC_DIM, BLK), BLK, 0, stream_>>>(ring, head, PLE_CONV_HIST + 1,
                                                    PLE_CONV_K, PLE_CONV_DIL, HC_DIM, w, out);
    }
    void gdn_conv_gate(const float * ring, int head, const float * w,
                       const float * beta_raw, const float * alpha_raw,
                       const float * dt, const float * ssm_a, float * conv_out,
                       float * beta_sig, float * a_softplus, float * gate,
                       float * g_exp) override {
        mark(PC_ELEM);
        k_gdn_conv_gate<<<ceil_div(GDN_CONV_DIM + GDN_V_HEADS, BLK), BLK, 0, stream_>>>(
            ring, head, w, beta_raw, alpha_raw, dt, ssm_a,
            conv_out, beta_sig, a_softplus, gate, g_exp);
    }
    void gdn_step(float * S, const float * q, const float * k, const float * v,
                  const float * g, const float * b, float * o) override {
        mark(PC_GDN);
        k_gdn_step<<<GDN_V_HEADS * GDN_STATE, WAVE, 0, stream_>>>(S, q, k, v, g, b, o);
    }
    void gated_rms_norm(const float * x, const float * w, const float * z, float * y,
                        int ne0, int n_groups, float eps) override {
        mark(PC_NORM);
        const int t = red_threads(ne0);
        k_gated_rms_norm<<<n_groups, t, t * sizeof(float), stream_>>>(x, w, z, y, ne0, eps);
    }
    void router(const float * logits, int * ids, float * w) override {
        mark(PC_ROUTER);
        k_router<<<1, N_EXPERT, 0, stream_>>>(logits, ids, w);
    }
    void moe_gate_up(const Mat & g, const Mat & u, const int * ids, const float * x,
                     float * yg, float * yu) override {
        const dim3 grid((unsigned) g.rows, N_EXPERT_USED, 2);
        mark(PC_EXP_GATE_UP);
        if (g.type == FK_Q_IQ3_S) {
            k_moe_gate_up_iq3s<<<grid, WAVE, 0, stream_>>>((const unsigned char *) g.base,
                (const unsigned char *) u.base, g.expert_stride, g.row_bytes, ids, x, yg, yu,
                (int) g.rows, (int) g.K, N_EXPERT_USED);
        } else {
            k_moe_gate_up_iq4xs<<<grid, WAVE, 0, stream_>>>((const unsigned char *) g.base,
                (const unsigned char *) u.base, g.expert_stride, g.row_bytes, ids, x, yg, yu,
                (int) g.rows, (int) g.K, N_EXPERT_USED);
        }
    }
    void silu_mul(const float * g, const float * u, float * h, size_t n) override {
        mark(PC_ELEM);
        k_silu_mul<<<ceil_div((int) n, BLK), BLK, 0, stream_>>>(g, u, h, (int) n);
    }
    void moe_down(const Mat & d, const int * ids, const float * h, float * eo) override {
        const dim3 grid((unsigned) d.rows, N_EXPERT_USED, 1);
        mark(PC_EXP_DOWN);
        if (d.type == FK_Q_IQ4_NL) {
            k_moe_down_iq4nl<<<grid, WAVE, 0, stream_>>>((const unsigned char *) d.base, d.expert_stride,
                d.row_bytes, ids, h, eo, (int) d.rows, (int) d.K, N_EXPERT_USED);
        } else {
            k_moe_down_q8_0<<<grid, WAVE, 0, stream_>>>((const unsigned char *) d.base, d.expert_stride,
                d.row_bytes, ids, h, eo, (int) d.rows, (int) d.K, N_EXPERT_USED);
        }
    }
    void moe_finish(const float * eo, const float * w, const float * sh_raw,
                    const float * sh_gate, float * sh_gate_sig, float * moe_out,
                    float * sh_gated, float * out) override {
        mark(PC_EXP_COMBINE);
        k_moe_finish<<<ceil_div(N_EMBD, BLK), BLK, 0, stream_>>>(eo, w, sh_raw, sh_gate,
                                                     sh_gate_sig, moe_out, sh_gated,
                                                     out, N_EMBD, N_EXPERT_USED);
    }
    void rope_imrope(float * x, int n_heads, int head_dim, int n_rot, int pos) override {
        mark(PC_QSA);
        k_rope_imrope<<<n_heads, n_rot / 2, 0, stream_>>>(x, head_dim, n_rot, pos);
    }
    void kv_store_q8_0(const float * k, const float * v, int8_t * kqs, uint16_t * ksc,
                       int8_t * vqs, uint16_t * vsc, int cell) override {
        const dim3 grid(HEAD_DIM / 32, N_KV_HEADS, 2);
        mark(PC_QSA);
        k_kv_store_q8_0<<<grid, WAVE, 0, stream_>>>(k, v, kqs, (__half *) ksc, vqs, (__half *) vsc, cell);
    }
    void idx_raw_store(const float * kr, float * raw, int cell) override {
        mark(PC_QSA);
        k_idx_raw_store<<<ceil_div(IDX_DIM, BLK), BLK, 0, stream_>>>(kr, raw, cell);
    }
    void idx_pool_block(const float * raw, int blk, int n_filled, const float * kn,
                        float eps, uint16_t * pooled, float * dbg_p, float * dbg_r) override {
        mark(PC_QSA);
        k_idx_pool_block<<<1, IDX_DIM, 0, stream_>>>(raw, blk, n_filled, kn, eps,
                                         (unsigned short *) pooled, dbg_p, dbg_r);
    }
    void idx_scan(const uint16_t * pooled, const float * q, float * s, int n_blocks) override {
        const int nsub = IDXS_THREADS / WAVE;
        int wgs = ceil_div(n_blocks, nsub);
        if (wgs > 4096) wgs = 4096;
        if (wgs < 1) wgs = 1;
        mark(PC_QSA);
        k_idx_scan<<<wgs, IDXS_THREADS, 0, stream_>>>((const unsigned short *) pooled, q, s, n_blocks);
    }
    void qsa_qk_post(const float * qfull, const float * kraw, const float * idxraw,
                     const float * q_norm, const float * k_norm, const float * iq_norm,
                     float * qcur, float * gate, float * gsig, float * kcur,
                     float * idxq, int pos, float eps) override {
        mark(PC_QSA);
        k_qsa_qk_post<<<N_Q_HEADS + N_KV_HEADS + IDX_N_HEADS, QKPOST_THREADS, 0, stream_>>>(
            qfull, kraw, idxraw, q_norm, k_norm, iq_norm,
            qcur, gate, gsig, kcur, idxq, pos, eps);
    }
    void qsa_expand(const float * bs, float * c, int * sel, int n_kv, int q, int r,
                    int tail, int sel_identity) override {
        mark(PC_QSA);
        k_qsa_expand<<<ceil_div(n_kv, BLK), BLK, 0, stream_>>>(bs, c, sel, n_kv, q, r, tail, sel_identity);
    }
    int topk_select(const float * scores, int n, int width, int * out) override {
        if (n > cand_cap_) {
            HIP_IGNORE(hipFree(cand_key_)); HIP_IGNORE(hipFree(cand_idx_));
            cand_cap_ = n;
            HIP_CHECK(hipMalloc(&cand_key_, (size_t) cand_cap_ * sizeof(uint32_t)));
            HIP_CHECK(hipMalloc(&cand_idx_, (size_t) cand_cap_ * sizeof(int)));
        }
        const int k = width < n ? width : n;
        mark(PC_QSA);
        k_topk_radix_select<<<1, TOPK_THREADS, 0, stream_>>>(scores, n, k, cand_key_, cand_idx_, out);
        return k;
    }
    void attn_qsa(const int8_t * kqs, const uint16_t * ksc, const int8_t * vqs,
                  const uint16_t * vsc, const float * q, const int * sel, int n_sel,
                  const float * gsig, float * out, float * out_gated) override {
        const int n_chunks = ceil_div(n_sel, ATTN_CHUNK);
        mark(PC_QSA);
        k_attn_flash_split<<<dim3(n_chunks, N_KV_HEADS), ATTN_THREADS, 0, stream_>>>(
            kqs, (const __half *) ksc, vqs, (const __half *) vsc, q, sel,
            partial_, mbuf_, lbuf_, n_sel);
        mark(PC_QSA);
        k_attn_combine<<<N_Q_HEADS, HEAD_DIM, 0, stream_>>>(partial_, mbuf_, lbuf_, gsig,
                                                out, out_gated, n_chunks);
    }

    void set_profile(bool on) override {
        prof_ = on;
        if (on && ev_.empty()) {
            ev_.resize(PROF_MAX_EVENTS + 1);
            ev_cls_.resize(PROF_MAX_EVENTS + 1);
            ev_grp_.resize(PROF_MAX_EVENTS + 1, -1);
            for (auto & e : ev_) HIP_CHECK(hipEventCreate(&e));
        }
    }
    void prof_reset() override {
        for (int i = 0; i < PC_N; ++i) prof_us_[i] = 0.0;
        for (int i = 0; i < GG_N; ++i) { gemv_us_[i] = 0.0; gemv_bytes_[i] = 0.0; }
        tokens_profiled_ = 0;
        blocking_in_body_ = 0;
        launches_in_body_ = 0;
    }
    void prof_end_token() override {
        blocking_in_body_ += blocking_ - blocking_at_token_start_;
        launches_in_body_ += launches_ - launches_at_token_start_;
        ++tokens_profiled_;
        if (!prof_ || n_ev_ == 0) { n_ev_ = 0; return; }
        // one closing event, then ONE synchronisation for the whole token
        HIP_CHECK(hipEventRecord(ev_[n_ev_], stream_));
        HIP_CHECK(hipEventSynchronize(ev_[n_ev_]));
        for (int i = 0; i < n_ev_; ++i) {
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev_[i], ev_[i + 1]));
            prof_us_[ev_cls_[i]] += (double) ms * 1000.0;
            if (ev_grp_[i] >= 0 && ev_grp_[i] < GG_N) gemv_us_[ev_grp_[i]] += (double) ms * 1000.0;
        }
        n_ev_ = 0;
    }
    void prof_report(FILE * out, int n_tokens) override {
        const double n = n_tokens > 0 ? (double) n_tokens : 1.0;
        double total = 0.0;
        for (int c = 0; c < PC_N; ++c) total += prof_us_[c];
        for (int c = 0; c < PC_N; ++c)
            std::fprintf(out, "prof_%s_us=%.2f\n", prof_name(c), prof_us_[c] / n);
        std::fprintf(out, "prof_total_us=%.2f\n", total / n);
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
        std::fprintf(out, "prof_host_syncs_per_token=%.2f\n", (double) blocking_in_body_ / n);
        std::fprintf(out, "prof_launches_per_token=%.2f\n", (double) launches_in_body_ / n);
        if (n_ev_ >= PROF_MAX_EVENTS)
            std::fprintf(out, "prof_WARNING=event pool exhausted, times are a lower bound\n");
    }

    void   timer_start() override {
        blocking_at_token_start_ = blocking_;
        launches_at_token_start_ = launches_;
        HIP_CHECK(hipEventRecord(ev0_, stream_));
    }
    double timer_stop_ms() override {
        HIP_CHECK(hipEventRecord(ev1_, stream_));
        HIP_CHECK(hipEventSynchronize(ev1_));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev0_, ev1_));
        return (double) ms;
    }

private:
    // ---- profiling state ----
    void mark(int cls, int gemv_group = -1, size_t bytes = 0) {
        ++launches_;                       // one op == one launch, except where noted
        if (gemv_group >= 0 && gemv_group < GG_N) gemv_bytes_[gemv_group] += (double) bytes;
        if (!prof_ || n_ev_ >= PROF_MAX_EVENTS) return;
        HIP_CHECK(hipEventRecord(ev_[n_ev_], stream_));
        ev_cls_[n_ev_] = cls;
        ev_grp_[n_ev_] = gemv_group;
        ++n_ev_;
    }

    bool       prof_ = false;
    int        n_ev_ = 0;
    long long  launches_ = 0;
    long long  blocking_ = 0;              // host-blocking calls, counted always
    long long  blocking_at_token_start_ = 0;
    long long  launches_at_token_start_ = 0;
    int        tokens_profiled_ = 0;
    long long  blocking_in_body_ = 0;
    long long  launches_in_body_ = 0;
    double     prof_us_[PC_N] = {0};
    double     gemv_us_[GG_N]  = {0};
    double     gemv_bytes_[GG_N] = {0};
    std::vector<hipEvent_t> ev_;
    std::vector<int>        ev_cls_;
    std::vector<int>        ev_grp_;

    hipEvent_t ev0_{}, ev1_{};
    float * partial_ = nullptr;
    float * mbuf_    = nullptr;
    float * lbuf_    = nullptr;
    uint32_t * cand_key_ = nullptr;
    int      * cand_idx_ = nullptr;
    int        cand_cap_ = 0;
    int        gemv_lds_ = 1;
    int        gemv_min_rows_ = 1024;
    int        dev_ = 0;
    char       name_[24] = {0};
    hipStream_t stream_{};
    hipEvent_t  bnd_{};
    float *     amax_v_ = nullptr;
    int *       amax_i_ = nullptr;
    // split-K partials: nsplit*rows stays near GEMV_TARGET_WGS*GEMV_WAVES by
    // construction (nsplit rises only as rows falls), so this is small.
    float *    gemv_part_ = nullptr;
    static constexpr size_t GEMV_PART_FLOATS = 1u << 17;   // 128k floats = 512 KB
    static constexpr int    STAGE_SLOTS = 8;
    static constexpr size_t STAGE_BYTES = 1u << 16;        // 64 KB
    void *     stage_[STAGE_SLOTS] = {nullptr};
    int        stage_next_ = 0;
    size_t placed_ = 0;
};

} // namespace

Backend * make_gpu_backend(int device) { return new GpuBackend(device); }

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
