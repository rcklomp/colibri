// tools/hot-expert/m1/m1_native.hip
//
// M1 candidate 3: a routed-expert MoE decode step that reads the experts in
// their NATIVE GGUF formats -- gate/up in IQ3_S, down in IQ4_NL -- with
// hipFire's kernel SHAPE (Franken-engine design rev 7 section 9.3, L0 step 1).
//
// Why this exists: candidate 2 (m1_hipfire.hip) measured 452 GB/s at batch 1
// but only by REQUANTISING the experts into hipFire's own MQ4G256V2 4.25 bpw
// container, which (a) costs a quality question nobody has answered and (b)
// reads 19 % more bytes than the shipped IQ3_S/IQ4_NL weights. This kernel
// removes both: it decodes the ggml blocks in place, so L0 can serve the
// GGUF's own bytes.
//
// The shape copied from hipFire (see hipfire_src/gemv_mq4g256v2_moe_gate_up_
// k8_indexed_batched.hip, which is the file this is modelled on line for
// line where the format allows):
//   - one wave32 (block = 32 threads) per output row
//   - grid = (M, K_TOP, T): output row on grid.x, top-k rank on grid.y,
//     token on grid.z
//   - the expert id is read ON THE DEVICE from topk_indices[bid*K_TOP+krank]
//   - expert weights are addressed through a per-expert device pointer table
//   - gate and up are ONE launch (rows [0,MI) are gate, [MI,M) are up)
//   - f32 accumulation, four independent accumulators, __shfl_down reduction
//   - no atomics: down writes [T x K_TOP x M] and a separate combine kernel
//     folds the K_TOP slots with the router weights
//
// Differences from hipFire's kernels, all deliberate and all reported:
//   1. K_TOP is a RUNTIME parameter here (grid.y extent + kernel arg), used
//      at the model's real top-10; hipFire's `_k8_` family is contractually
//      K_TOP=8 and m1_hipfire.hip runs 8 experts, not 10. Nothing in the
//      shape needed K_TOP to be constant -- the combine kernel's hard-coded
//      `#pragma unroll for (k = 0; k < 8; k++) if (k < K_TOP)` was the only
//      place, and a plain runtime loop replaces it.
//   2. No padding anywhere. hipFire's down kernel needs K to be a multiple
//      of its 256-weight group, so m1_hipfire pads down's K=640 up to 768
//      (+20 % bytes on the down expert, hipfire_down_pad_overhead=0.2) and
//      pays a hipMemcpy2DAsync per step to widen h. IQ4_NL's block is 32
//      weights and 640 = 20 x 32 exactly, so the pad, its bytes and its copy
//      all disappear: down's h feeds the kernel directly.
//   3. x is NOT staged in LDS. The brief's shape summary says "x staged once
//      per workgroup", but hipFire's kernel does not stage it (it reads
//      `x_row` straight from global; its own header comment says as much --
//      "The staging is absent there (global x_row)"), and staging would be
//      pure loss here: with 32 lanes x 8 weights per 256-block, the workgroup
//      reads each element of x_row EXACTLY ONCE over the whole row. Staging
//      would turn one global read per element into one global read plus an
//      LDS write, a barrier, and an LDS read, and would cost 10 KB of LDS
//      per workgroup (K=2560 floats), capping occupancy at 6 workgroups per
//      64 KB. So it is not implemented, and this is why.
//   4. m1_hipfire launches the down kernel with grid.x = M = 2560 while that
//      kernel's body starts `row0 = blockIdx.x * 4` and returns when
//      `row0 >= M` -- correct, but 1920 of its 2560 blocks exit immediately
//      (hipFire's own launcher on gfx1100 defaults to that over-launch).
//      This kernel is one row per block, so grid.x = M is exact. Noted as an
//      observation about candidate 2, not a measurement claim about it.
//
// FORMAT DECODE -- ported bit-exactly from the CPU dequantisers in
// ~/src/llama-glm53/ggml/src/ggml-quants.c (`dequantize_row_iq3_s` line 2607,
// `dequantize_row_iq4_nl` line 2725), structs from ggml/src/ggml-common.h:
//
//   block_iq3_s, 110 B for 256 weights (3.4375 bpw):
//     off   0  ggml_half d           (f16)
//     off   2  uint8_t qs[64]        grid index low 8 bits
//     off  66  uint8_t qh[8]         grid index bit 8, one byte per 32-weight
//                                    sub-block, two indices per byte-pair
//     off  74  uint8_t signs[32]     one byte per 8 weights, bit m = sign of
//                                    weight m (kmask_iq2xs is {1,2,4,...,128},
//                                    i.e. exactly bit m -- asserted on the host)
//     off 106  uint8_t scales[4]     4-bit per 32-weight sub-block, low nibble
//                                    for even ib32, high nibble for odd
//     value(w) = d*(1 + 2*scale_nibble) * iq3s_grid_byte * (sign ? -1 : +1)
//
//   block_iq4_nl, 18 B for 32 weights (4.5 bpw):
//     off   0  ggml_half d
//     off   2  uint8_t qs[16]
//     value(j)      = d * kvalues_iq4nl[qs[j] & 0xf]     for j in [0,16)
//     value(j + 16) = d * kvalues_iq4nl[qs[j] >> 4]
//     NOTE the INTERLEAVE: qs[j]'s two nibbles are weights j and j+16, not
//     two adjacent weights. Getting this wrong is a silent 50 % scramble.
//
// The two tables (iq3s_grid, 512 x uint32; kvalues_iq4nl, 16 x int8) are not
// transcribed: ggml-common.h is included here with GGML_COMMON_IMPL_CPP and
// the host copies are hipMemcpyToSymbol'd into __constant__ arrays, so they
// are the same bytes llama.cpp uses, by construction.
//
// The per-lane decode itself lives in m1_native_decode.h and is compiled
// BOTH into these kernels and into m1_native_emul.cpp, a host-only
// emulation that checks the decode and the lane mapping against ggml's own
// to_float() on a CPU, with no GPU and no rig lock (`make check-m1-native-emul`).
// One source, two targets -- so the check cannot silently drift from the
// thing it checks.
//
// Neither block struct is 4-byte aligned in a packed row (110 and 18 are both
// only 2-aligned), so every payload read is a byte load; a `*(const uint32_t*)`
// on qs -- which is what hipFire can do with its 136 B group stride -- would
// be a misaligned dword load here. Only `d` is read wide-ish, and it is
// assembled from two byte loads rather than assuming even alignment.
//
// Building this program does NOT touch the GPUs (offline hipcc codegen for
// --offload-arch=gfx1100, docker with no --device flags). It is not run here;
// the orchestrating session runs it under the rig lock via run_m1.sh.
//
// HOST SIDE: weights are quantised with ggml_quantize_chunk() and the oracle
// dequantises with ggml_get_type_traits(type)->to_float(), exactly as
// m1_ggml.cpp does, so the stored bytes are what llama.cpp would hold.
// Moreover the RNG draw ORDER here is identical to m1_ggml.cpp's (per expert:
// gate 640x2560, up 640x2560, down 2560x640; then per T: ids, weights, x),
// and m1_common.h's generator is deterministic -- so candidate 1 and this
// candidate quantise the SAME pre-quantisation floats and route the SAME
// ids/weights/x. That makes the two candidates' check numbers comparable
// without any cross-process plumbing. (The brief's optional "cosine against
// m1_ggml's ggml path with the same ids" is NOT computed here: it would need
// the two binaries in one process, or a dumped tensor exchange. Not done.)
//
// Startup is dominated by host-side IQ3_S quantisation (512 experts x 2 x 640
// rows x 2560 weights, single-threaded inside ggml) -- the same cost
// m1_ggml.cpp already pays, minutes not seconds, before any timing begins.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ggml public API: ggml_quantize_chunk / ggml_get_type_traits / ggml_row_size.
#include "ggml.h"

// ggml internal: the block structs and the decode tables, verbatim.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

// The per-lane IQ3_S / IQ4_NL decode, shared verbatim with the host
// emulation in m1_native_emul.cpp (see that file and the note above).
#define M1N_QUAL __device__ __forceinline__
#define M1N_HALF_TO_F32(bits) __half2float(__ushort_as_half((unsigned short)(bits)))
#include "m1_native_decode.h"

#include "m1_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(_e)); \
        std::abort(); \
    } \
} while (0)

// ---------------------------------------------------------------------
// Block geometry (the IQ3S_*/IQ4NL_* macros come from m1_native_decode.h),
// asserted against ggml's own structs so a change in ggml-common.h breaks
// the build instead of silently mis-decoding.
// ---------------------------------------------------------------------
static_assert(sizeof(block_iq3_s)  == IQ3S_BLOCK_BYTES,  "block_iq3_s size moved");
static_assert(sizeof(block_iq4_nl) == IQ4NL_BLOCK_BYTES, "block_iq4_nl size moved");
static_assert(offsetof(block_iq3_s, qs)     == IQ3S_OFF_QS,     "iq3_s qs offset moved");
static_assert(offsetof(block_iq3_s, qh)     == IQ3S_OFF_QH,     "iq3_s qh offset moved");
static_assert(offsetof(block_iq3_s, signs)  == IQ3S_OFF_SIGNS,  "iq3_s signs offset moved");
static_assert(offsetof(block_iq3_s, scales) == IQ3S_OFF_SCALES, "iq3_s scales offset moved");
static_assert(offsetof(block_iq4_nl, qs)    == IQ4NL_OFF_QS,    "iq4_nl qs offset moved");

// ---------------------------------------------------------------------
// Decode tables on the device.
//
// NATIVE_IQ3S_GRID_LDS is the compile-time switch the brief asks for:
//   0 (default) -- iq3s_grid lives in __constant__ memory. 2 KB, read with a
//                  divergent index so it is served by the vector path through
//                  L1; it is the hottest 2 KB in the kernel and costs no LDS,
//                  so occupancy is untouched.
//   1           -- each workgroup copies the 512 entries into LDS at entry
//                  and reads from there. Costs 2 KB of LDS per 32-thread
//                  workgroup (capping a 64 KB CU at 32 resident workgroups)
//                  plus one __syncthreads, for a table that is already
//                  L1-resident. Provided so the choice can be measured, not
//                  because it is expected to win.
// Both variants read the SAME bytes; only the storage differs.
// ---------------------------------------------------------------------
#ifndef NATIVE_IQ3S_GRID_LDS
#define NATIVE_IQ3S_GRID_LDS 0
#endif

__constant__ uint32_t c_iq3s_grid[512];
__constant__ int8_t   c_kvalues_iq4nl[16];

// ---------------------------------------------------------------------
// gate + up, one launch. grid = (M, K_TOP, T), block = 32.
//   rows [0, M/2)  -> y_gate[row]
//   rows [M/2, M)  -> y_up[row - M/2]
// (hipFire's `mi = M >> 1` split, kept so the two candidates' output layouts
// and launch counts line up.)
//
// expert_ptrs[e] points at expert e's gate+up slab: M rows of (K/256) IQ3_S
// blocks, gate rows first.
// ---------------------------------------------------------------------
__launch_bounds__(32, 16)
extern "C" __global__ void m1_native_moe_gate_up_iq3s(
    const unsigned long long * __restrict__ expert_ptrs,   // [n_exp]
    const int                * __restrict__ topk_indices,  // [T x K_TOP]
    const float              * __restrict__ x,             // [T x K]
    float                    * __restrict__ y_gate,        // [T x K_TOP x M/2]
    float                    * __restrict__ y_up,          // [T x K_TOP x M/2]
    int M, int K, int K_TOP)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    if (krank >= K_TOP) return;
    // Bounded-safe: refuse a K the block geometry cannot tile rather than
    // walking off the slab. (Host asserts this too; belt and braces.)
    if ((K & (IQ3S_BLOCK_WEIGHTS - 1)) != 0) return;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

#if NATIVE_IQ3S_GRID_LDS
    __shared__ uint32_t s_grid[512];
    for (int i = tid; i < 512; i += 32) s_grid[i] = c_iq3s_grid[i];
    __syncthreads();
    const uint32_t * __restrict__ grid = s_grid;
#else
    const uint32_t * __restrict__ grid = c_iq3s_grid;
#endif

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char * __restrict__ A =
        reinterpret_cast<const unsigned char *>(expert_ptrs[expert_id]);
    const float * __restrict__ x_row = x + (long long)bid * K;

    const int nblk = K >> 8;                       // K / 256
    const unsigned char * row_ptr = A + (long long)row * nblk * IQ3S_BLOCK_BYTES;

    const int ib32 = tid >> 2;                     // 0..7
    const int l    = tid & 3;                      // 0..3
    const int sh1  = 8 - 2 * l;
    const int sh2  = 7 - 2 * l;
    const int xoff = tid << 3;                     // 8 weights per lane

    // Four independent accumulators, as in hipFire's quad-unrolled body: the
    // dependent FMA chain is the limiter once the loads are in flight.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int b = 0;
    for (; b + 3 < nblk; b += 4) {
        acc0 += m1n_iq3s_block_dot(row_ptr + (long long)(b + 0) * IQ3S_BLOCK_BYTES, grid,
                               x_row + ((b + 0) << 8) + xoff, tid, ib32, sh1, sh2);
        acc1 += m1n_iq3s_block_dot(row_ptr + (long long)(b + 1) * IQ3S_BLOCK_BYTES, grid,
                               x_row + ((b + 1) << 8) + xoff, tid, ib32, sh1, sh2);
        acc2 += m1n_iq3s_block_dot(row_ptr + (long long)(b + 2) * IQ3S_BLOCK_BYTES, grid,
                               x_row + ((b + 2) << 8) + xoff, tid, ib32, sh1, sh2);
        acc3 += m1n_iq3s_block_dot(row_ptr + (long long)(b + 3) * IQ3S_BLOCK_BYTES, grid,
                               x_row + ((b + 3) << 8) + xoff, tid, ib32, sh1, sh2);
    }
    for (; b < nblk; b++) {
        acc0 += m1n_iq3s_block_dot(row_ptr + (long long)b * IQ3S_BLOCK_BYTES, grid,
                               x_row + (b << 8) + xoff, tid, ib32, sh1, sh2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) y_gate[base_out + row]      = acc;
        else          y_up  [base_out + row - mi] = acc;
    }
}

// ---------------------------------------------------------------------
// down, IQ4_NL. grid = (M, K_TOP, T), block = 32. No atomics: each
// (token, krank, row) is written by exactly one workgroup.
//
// Lane-to-weight mapping: a 32-weight block's payload is 16 qs bytes = 4
// 4-byte chunks; the row therefore has (K/32)*4 chunks and lane `tid` walks
// them with a stride-32 loop, 8 weights per chunk. For K=640 that is 80
// chunks: lanes 0..15 do 3 iterations, 16..31 do 2. The four bytes of a
// chunk are read individually -- an 18 B block stride leaves qs only
// 2-aligned for every other block, so a dword load would be misaligned.
//
// Each element of h_row is read exactly once by the workgroup, same as x in
// the gate/up kernel, so LDS staging would again be pure overhead.
// ---------------------------------------------------------------------
__launch_bounds__(32, 16)
extern "C" __global__ void m1_native_moe_down_iq4nl(
    const unsigned long long * __restrict__ expert_ptrs,    // [n_exp]
    const int                * __restrict__ topk_indices,   // [T x K_TOP]
    const float              * __restrict__ h,              // [T x K_TOP x K]
    float                    * __restrict__ expert_outputs, // [T x K_TOP x M]
    int M, int K, int K_TOP)
{
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    if (krank >= K_TOP) return;
    if ((K & (IQ4NL_BLOCK_WEIGHTS - 1)) != 0) return;   // bounded-safe
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char * __restrict__ A =
        reinterpret_cast<const unsigned char *>(expert_ptrs[expert_id]);
    const float * __restrict__ h_row = h + ((long long)bid * K_TOP + krank) * K;

    const int nblk = K >> 5;                        // K / 32
    const unsigned char * row_ptr = A + (long long)row * nblk * IQ4NL_BLOCK_BYTES;

    const int nchunk = nblk << 2;                   // 4 chunks of 4 qs bytes

    float acc_lo = 0.0f, acc_hi = 0.0f;
    for (int w = tid; w < nchunk; w += 32) {
        m1n_iq4nl_chunk_dot(row_ptr, c_kvalues_iq4nl, h_row, w, &acc_lo, &acc_hi);
    }

    float acc = acc_lo + acc_hi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        expert_outputs[((long long)bid * K_TOP + krank) * M + row] = acc;
    }
}

// silu(gate) * up, elementwise. Body identical to hipFire's silu_mul_f32,
// including plain `expf` rather than the fast `__expf` intrinsic -- this
// kernel is not on the critical path (one pass over T*K_TOP*640 floats) and
// keeping it bit-for-bit the same as candidate 2's makes any difference
// between the two candidates' oracle numbers attributable to the GEMVs.
extern "C" __global__ void m1_native_silu_mul_f32(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float       * __restrict__ out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = gate[i];
        out[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

// Fold the K_TOP per-rank down outputs with the router weights.
//
// Unlike hipFire's moe_down_combine_k8_batched this (a) takes K_TOP as a
// runtime loop bound rather than an unrolled 8, and (b) STORES rather than
// `+=` into the destination -- there is no residual to accumulate into in
// this microbenchmark, and a store removes the hipMemsetAsync that the
// accumulate form needs before every step. That is one launch fewer per
// step than candidate 2 spends.
extern "C" __global__ void m1_native_moe_combine(
    const float * __restrict__ expert_outputs,  // [T x K_TOP x M]
    const float * __restrict__ topk_weights,    // [T x K_TOP]
    float       * __restrict__ out,             // [T x M]
    int M, int K_TOP)
{
    const int bid = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;

    const long long base = (long long)bid * K_TOP * M;
    const int wbase = bid * K_TOP;
    float acc = 0.0f;
    for (int k = 0; k < K_TOP; k++) {
        acc += topk_weights[wbase + k] * expert_outputs[base + (long long)k * M + col];
    }
    out[(long long)bid * M + col] = acc;
}

namespace {

constexpr int N_EXPERT = 512;
constexpr int K_TOP    = 10;    // the model's real top-k, runtime on grid.y
constexpr int D_MODEL  = 2560;  // gate/up K; down's output rows
constexpr int N_FF     = 640;   // gate/up output rows; down's K
constexpr int GATEUP_M = 2 * N_FF;  // 1280: gate rows then up rows

constexpr int IQ3S_BLOCKS_PER_GATEUP_ROW = D_MODEL / IQ3S_BLOCK_WEIGHTS;   // 10
constexpr int IQ4NL_BLOCKS_PER_DOWN_ROW  = N_FF    / IQ4NL_BLOCK_WEIGHTS;  // 20

constexpr size_t GATEUP_ROW_BYTES = (size_t)IQ3S_BLOCKS_PER_GATEUP_ROW * IQ3S_BLOCK_BYTES;  // 1100
constexpr size_t DOWN_ROW_BYTES   = (size_t)IQ4NL_BLOCKS_PER_DOWN_ROW  * IQ4NL_BLOCK_BYTES; // 360
constexpr size_t GATEUP_EXPERT_BYTES = (size_t)GATEUP_M * GATEUP_ROW_BYTES;  // 1,408,000
constexpr size_t DOWN_EXPERT_BYTES   = (size_t)D_MODEL  * DOWN_ROW_BYTES;    //   921,600

// 10 experts x (1,408,000 + 921,600) = 23,296,000 -- the brief's number.
constexpr size_t BYTES_PER_STEP = (size_t)K_TOP * (GATEUP_EXPERT_BYTES + DOWN_EXPERT_BYTES);

constexpr int LAUNCHES_PER_STEP = 4;   // gate_up, silu, down, combine
constexpr int WARMUP_ITERS = 100;
constexpr int TIMED_ITERS  = 1000;

struct TimedStep {
    const unsigned long long *d_ptrs_gateup = nullptr;
    const unsigned long long *d_ptrs_down   = nullptr;
    int32_t *d_ids      = nullptr;
    float   *d_weights  = nullptr;
    float   *d_x        = nullptr;
    float   *d_y_gate   = nullptr;
    float   *d_y_up     = nullptr;
    float   *d_h        = nullptr;
    float   *d_expert_outputs = nullptr;
    float   *d_out      = nullptr;
    int T = 0;
};

void run_step(TimedStep &s)
{
    dim3 block32(32, 1, 1);

    dim3 grid_gateup((unsigned)GATEUP_M, (unsigned)K_TOP, (unsigned)s.T);
    m1_native_moe_gate_up_iq3s<<<grid_gateup, block32>>>(
        s.d_ptrs_gateup, s.d_ids, s.d_x, s.d_y_gate, s.d_y_up,
        GATEUP_M, D_MODEL, K_TOP);

    const int n_h = s.T * K_TOP * N_FF;
    m1_native_silu_mul_f32<<<dim3((unsigned)((n_h + 255) / 256), 1, 1), dim3(256, 1, 1)>>>(
        s.d_y_gate, s.d_y_up, s.d_h, n_h);

    dim3 grid_down((unsigned)D_MODEL, (unsigned)K_TOP, (unsigned)s.T);
    m1_native_moe_down_iq4nl<<<grid_down, block32>>>(
        s.d_ptrs_down, s.d_ids, s.d_h, s.d_expert_outputs,
        D_MODEL, N_FF, K_TOP);

    m1_native_moe_combine<<<dim3((unsigned)((D_MODEL + 255) / 256), (unsigned)s.T, 1), dim3(256, 1, 1)>>>(
        s.d_expert_outputs, s.d_weights, s.d_out, D_MODEL, K_TOP);
}

// Quantise `nrows` x `n_per_row` fp32 into `type`, into `dst`. One draw of
// the whole tensor, in the same order and the same sizes as
// m1_ggml.cpp::quantize_upload_expert, so the shared m1::rng() stream stays
// in lockstep between the two candidates (see the file header).
void quantize_rows(ggml_type type, int64_t nrows, int64_t n_per_row, uint8_t *dst, size_t dst_bytes)
{
    std::vector<float> src = m1::randf_vec((size_t)nrows * n_per_row, -1.0f, 1.0f);
    const size_t written = ggml_quantize_chunk(type, src.data(), dst,
                                                /*start=*/0, nrows, n_per_row, /*imatrix=*/nullptr);
    if (written != dst_bytes) {
        std::fprintf(stderr, "quantize_chunk wrote %zu, expected %zu\n", written, dst_bytes);
        std::abort();
    }
}

} // namespace

int main()
{
    // ---- Tables: ggml's own bytes, copied into __constant__. ----
    // kmask_iq2xs must be {1,2,4,...,128} for the "bit m of signs[tid]"
    // shortcut in iq3s_block_dot to be the same test the CPU code makes.
    for (int i = 0; i < 8; i++) {
        if (kmask_iq2xs[i] != (uint8_t)(1u << i)) {
            std::fprintf(stderr, "kmask_iq2xs is not 1<<m; the sign decode shortcut is invalid\n");
            return 2;
        }
    }
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_iq3s_grid), iq3s_grid, sizeof(uint32_t) * 512));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_kvalues_iq4nl), kvalues_iq4nl, sizeof(int8_t) * 16));

    // ---- Row sizes must match ggml's own idea of them. ----
    if (ggml_row_size(GGML_TYPE_IQ3_S, D_MODEL) != GATEUP_ROW_BYTES ||
        ggml_row_size(GGML_TYPE_IQ4_NL, N_FF)   != DOWN_ROW_BYTES) {
        std::fprintf(stderr, "row size mismatch: iq3_s(%d)=%zu (want %zu), iq4_nl(%d)=%zu (want %zu)\n",
                     D_MODEL, ggml_row_size(GGML_TYPE_IQ3_S, D_MODEL), GATEUP_ROW_BYTES,
                     N_FF, ggml_row_size(GGML_TYPE_IQ4_NL, N_FF), DOWN_ROW_BYTES);
        return 2;
    }

    // ---- Build 512 experts in the native formats, host side. ----
    std::vector<uint8_t> gateup_pool((size_t)N_EXPERT * GATEUP_EXPERT_BYTES); // ~721 MB
    std::vector<uint8_t> down_pool((size_t)N_EXPERT * DOWN_EXPERT_BYTES);     // ~472 MB

    std::fprintf(stderr, "quantising %d experts (IQ3_S gate+up, IQ4_NL down)...\n", N_EXPERT);
    for (int e = 0; e < N_EXPERT; e++) {
        uint8_t *gu = gateup_pool.data() + (size_t)e * GATEUP_EXPERT_BYTES;
        // gate rows [0, N_FF), then up rows [N_FF, 2*N_FF) -- two separate
        // quantise calls, matching m1_ggml.cpp's separate gate_exps/up_exps
        // tensors and its draw order.
        quantize_rows(GGML_TYPE_IQ3_S, N_FF, D_MODEL, gu, (size_t)N_FF * GATEUP_ROW_BYTES);
        quantize_rows(GGML_TYPE_IQ3_S, N_FF, D_MODEL, gu + (size_t)N_FF * GATEUP_ROW_BYTES,
                      (size_t)N_FF * GATEUP_ROW_BYTES);
        uint8_t *dn = down_pool.data() + (size_t)e * DOWN_EXPERT_BYTES;
        quantize_rows(GGML_TYPE_IQ4_NL, D_MODEL, N_FF, dn, (size_t)D_MODEL * DOWN_ROW_BYTES);
    }
    std::fprintf(stderr, "quantised.\n");

    uint8_t *d_gateup_pool = nullptr, *d_down_pool = nullptr;
    HIP_CHECK(hipMalloc(&d_gateup_pool, gateup_pool.size()));
    HIP_CHECK(hipMalloc(&d_down_pool,   down_pool.size()));
    HIP_CHECK(hipMemcpy(d_gateup_pool, gateup_pool.data(), gateup_pool.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_down_pool,   down_pool.data(),   down_pool.size(),   hipMemcpyHostToDevice));

    std::vector<unsigned long long> ptrs_gateup_host(N_EXPERT), ptrs_down_host(N_EXPERT);
    for (int e = 0; e < N_EXPERT; e++) {
        ptrs_gateup_host[e] = (unsigned long long)(uintptr_t)(d_gateup_pool + (size_t)e * GATEUP_EXPERT_BYTES);
        ptrs_down_host[e]   = (unsigned long long)(uintptr_t)(d_down_pool   + (size_t)e * DOWN_EXPERT_BYTES);
    }
    unsigned long long *d_ptrs_gateup = nullptr, *d_ptrs_down = nullptr;
    HIP_CHECK(hipMalloc(&d_ptrs_gateup, N_EXPERT * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_ptrs_down,   N_EXPERT * sizeof(unsigned long long)));
    HIP_CHECK(hipMemcpy(d_ptrs_gateup, ptrs_gateup_host.data(), N_EXPERT * sizeof(unsigned long long), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptrs_down,   ptrs_down_host.data(),   N_EXPERT * sizeof(unsigned long long), hipMemcpyHostToDevice));

    std::printf("native_k_top=%d\n", K_TOP);
    std::printf("native_bytes_per_step=%zu\n", BYTES_PER_STEP);
    std::printf("native_launches_per_step=%d\n", LAUNCHES_PER_STEP);
    std::printf("native_iq3s_grid_placement=%s\n", NATIVE_IQ3S_GRID_LDS ? "lds" : "constant");
    std::printf("native_gateup_expert_bytes=%zu\n", GATEUP_EXPERT_BYTES);
    std::printf("native_down_expert_bytes=%zu\n", DOWN_EXPERT_BYTES);
    std::printf("native_down_pad_overhead=%.4f\n", 0.0);   // none: 640 = 20 x 32 exactly

    bool any_check_fail = false;

    for (int T : {1, 32}) {
        TimedStep s;
        s.T = T;
        s.d_ptrs_gateup = d_ptrs_gateup;
        s.d_ptrs_down   = d_ptrs_down;

        // Same draws, same order as m1_ggml.cpp's per-T inputs.
        std::vector<int32_t> ids_host     = m1::random_distinct_ids(N_EXPERT, K_TOP, T);
        std::vector<float>   weights_host = m1::random_norm_weights(K_TOP, T);
        std::vector<float>   x_host       = m1::randf_vec((size_t)D_MODEL * T, -1.0f, 1.0f);

        HIP_CHECK(hipMalloc(&s.d_ids,     ids_host.size() * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&s.d_weights, weights_host.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_x,       x_host.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_y_gate,  (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_y_up,    (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_h,       (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_expert_outputs, (size_t)T * K_TOP * D_MODEL * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_out,     (size_t)T * D_MODEL * sizeof(float)));

        HIP_CHECK(hipMemcpy(s.d_ids, ids_host.data(), ids_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(s.d_weights, weights_host.data(), weights_host.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(s.d_x, x_host.data(), x_host.size() * sizeof(float), hipMemcpyHostToDevice));

        // ---- Oracle (T=1), before any warmup: fp32 CPU reference on the
        // weights dequantised with ggml's own to_float, same ids/weights/x.
        if (T == 1) {
            run_step(s);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipGetLastError());

            std::vector<float> gpu_out(D_MODEL);
            HIP_CHECK(hipMemcpy(gpu_out.data(), s.d_out, D_MODEL * sizeof(float), hipMemcpyDeviceToHost));

            const ggml_type_traits *tt_gu = ggml_get_type_traits(GGML_TYPE_IQ3_S);
            const ggml_type_traits *tt_dn = ggml_get_type_traits(GGML_TYPE_IQ4_NL);

            std::vector<float> ref_out(D_MODEL, 0.0f);
            std::vector<float> row_f(D_MODEL), down_row_f(N_FF);
            std::vector<float> y_gate(N_FF), y_up(N_FF), h_vec(N_FF);

            for (int k = 0; k < K_TOP; k++) {
                const int e = ids_host[k];
                const uint8_t *gu = gateup_pool.data() + (size_t)e * GATEUP_EXPERT_BYTES;
                const uint8_t *dn = down_pool.data()   + (size_t)e * DOWN_EXPERT_BYTES;

                for (int r = 0; r < N_FF; r++) {
                    tt_gu->to_float(gu + (size_t)r * GATEUP_ROW_BYTES, row_f.data(), D_MODEL);
                    double acc = 0.0;
                    for (int c = 0; c < D_MODEL; c++) acc += (double)row_f[c] * x_host[c];
                    y_gate[r] = (float)acc;

                    tt_gu->to_float(gu + (size_t)(N_FF + r) * GATEUP_ROW_BYTES, row_f.data(), D_MODEL);
                    acc = 0.0;
                    for (int c = 0; c < D_MODEL; c++) acc += (double)row_f[c] * x_host[c];
                    y_up[r] = (float)acc;

                    const float g = y_gate[r];
                    h_vec[r] = (g / (1.0f + std::exp(-g))) * y_up[r];
                }
                for (int r = 0; r < D_MODEL; r++) {
                    tt_dn->to_float(dn + (size_t)r * DOWN_ROW_BYTES, down_row_f.data(), N_FF);
                    double acc = 0.0;
                    for (int c = 0; c < N_FF; c++) acc += (double)down_row_f[c] * h_vec[c];
                    ref_out[r] += weights_host[k] * (float)acc;
                }
            }

            const double gpu_l1 = m1::l1_norm(gpu_out.data(), D_MODEL);
            const double ref_l1 = m1::l1_norm(ref_out.data(), D_MODEL);
            std::printf("native_gpu_out_l1=%.6g\n", gpu_l1);
            std::printf("native_ref_out_l1=%.6g\n", ref_l1);
            m1::print_first_n("native_gpu_out", gpu_out.data(), D_MODEL);
            m1::print_first_n("native_ref_out", ref_out.data(), D_MODEL);

            // A zero on either side is CHECK_FAIL outright: cos(0,0) is NaN,
            // which would not pass the >= 0.999 test, but a one-sided
            // near-zero could still cosine-pass by accident. Same guard
            // m1_ggml.cpp grew after its all-zero oracle on 2026-09-22.
            if (gpu_l1 == 0.0 || ref_l1 == 0.0) {
                std::fprintf(stderr, "CHECK_FAIL zero output (gpu_l1=%.6g ref_l1=%.6g)\n", gpu_l1, ref_l1);
                any_check_fail = true;
            } else {
                const double cos    = m1::cosine_similarity(gpu_out.data(), ref_out.data(), D_MODEL);
                const double maxabs = m1::max_abs_diff(gpu_out.data(), ref_out.data(), D_MODEL);
                std::printf("native_check_maxabs=%.6g\n", maxabs);
                std::printf("native_check_cos=%.6g\n", cos);
                if (!(cos >= 0.999)) any_check_fail = true;
            }
        }

        // ---- Timing: hipEvents around the whole 4-launch step. ----
        hipEvent_t ev_start, ev_stop;
        HIP_CHECK(hipEventCreate(&ev_start));
        HIP_CHECK(hipEventCreate(&ev_stop));

        for (int i = 0; i < WARMUP_ITERS; i++) run_step(s);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipGetLastError());

        std::vector<double> samples_us;
        samples_us.reserve(TIMED_ITERS);
        for (int i = 0; i < TIMED_ITERS; i++) {
            HIP_CHECK(hipEventRecord(ev_start));
            run_step(s);
            HIP_CHECK(hipEventRecord(ev_stop));
            HIP_CHECK(hipEventSynchronize(ev_stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
            samples_us.push_back((double)ms * 1000.0);
        }
        const m1::TimingStats ts = m1::stats_from_samples_us(samples_us);
        const char *tag = (T == 1) ? "T1" : "T32";
        std::printf("native_moe_%s_us_median=%.3f\n", tag, ts.median_us);
        std::printf("native_moe_%s_us_p10=%.3f\n", tag, ts.p10_us);
        std::printf("native_moe_%s_us_p90=%.3f\n", tag, ts.p90_us);
        if (T == 1) {
            std::printf("native_gbs_T1=%.3f\n", (double)BYTES_PER_STEP / (ts.median_us * 1e-6) / 1e9);
        } else {
            std::printf("native_gbs_T32=%.3f\n", (double)BYTES_PER_STEP * T / (ts.median_us * 1e-6) / 1e9);
            std::printf("native_gbs_T32_assumption=32x_independent_token_footprints_no_reuse_credited\n");
        }

        HIP_CHECK(hipEventDestroy(ev_start));
        HIP_CHECK(hipEventDestroy(ev_stop));

        (void)hipFree(s.d_ids); (void)hipFree(s.d_weights); (void)hipFree(s.d_x);
        (void)hipFree(s.d_y_gate); (void)hipFree(s.d_y_up); (void)hipFree(s.d_h);
        (void)hipFree(s.d_expert_outputs); (void)hipFree(s.d_out);
    }

    (void)hipFree(d_ptrs_gateup); (void)hipFree(d_ptrs_down);
    (void)hipFree(d_gateup_pool); (void)hipFree(d_down_pool);

    if (any_check_fail) {
        std::fprintf(stderr, "CHECK_FAIL\n");
        return 1;
    }
    return 0;
}
