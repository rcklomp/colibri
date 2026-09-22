// m5_bench.hip — M5: the launch and sync floor on gfx1100 (HIP, ROCm 7.14)
//
// Implements tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, section "M5", items
// 1-5, plus item 5's Q8_0 GEMV as a simple *correct* int8-dot kernel (not a
// port of ggml's or hipFire's kernel).
//
// Q8_0 layout used here (deliberately simplified vs. ggml's AoS
// block_q8_0 struct — the brief says the format does not matter for M5):
// planar/SoA per matrix: one int8_t[rows*K] array of quantised values,
// contiguous in K within a row (so a "block" is just 32 consecutive columns
// — no reordering needed), and one separate half[rows*(K/32)] array of
// per-block scales. Activations use the same scheme over their own K.
//
// Units: every *_gb / *_gbs value uses decimal GB = 1e9 bytes (not GiB).
// This box's RAM/VRAM units have bitten this project before (CLAUDE.md); the
// convention is stated once here and used consistently throughout.
//
// Dot product: block of 32 int8 x int8 -> int32, accumulated across blocks
// in fp32 (scale = half(w)*half(x) per block). The inner 4-wide dot is
// PLAIN SCALAR int accumulation (default, -DM5_DOT_SCALAR=1 set by the
// Makefile) — not __builtin_amdgcn_sdot4. That intrinsic was tried first
// (it would be the dp4a-equivalent on RDNA3, v_dot4_i32_i8) but does not
// build for gfx1100 with this hipcc/ROCm 7.14 toolchain: it requires Clang
// target feature "dot1-insts" as a Sema-level gate, and gfx1100 here does
// not recognize dot1-insts through dot12-insts as valid features at all
// ("not a recognized feature for this target"); forcing it via
// -Xclang -target-feature only satisfies Sema; the backend silently drops
// the same feature with the same message, so the resulting codegen would
// be unverified, and this cannot be checked without running on the GPU
// (forbidden here). See the Makefile for how to rebuild the sdot4 arm to
// reproduce the failure. The scalar fallback is correctness-first, not
// tuned for occupancy, exactly as the task allows. The binary prints
// which one it was built with as `dot_intrinsic=`.
//
// This file is BUILT ONLY here (docker hipcc, no --device flags, no GPU
// access). It is not run by the agent that wrote it — the orchestrating
// session runs it later under the rig lock.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>

// ---------------------------------------------------------------- utility --

#define HIP_CHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", #cmd, __FILE__, __LINE__, hipGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)floor(idx), hi = (size_t)ceil(idx);
    if (lo == hi) return v[lo];
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
static double median_of(std::vector<double> v) { return percentile(v, 0.5); }

// Collected key=value pairs, in order, for the JSON dump.
static std::vector<std::pair<std::string, std::string>> g_kv;

static void emit_num(const std::string& key, double val) {
    printf("%s=%.6f\n", key.c_str(), val);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", val);
    g_kv.emplace_back(key, std::string(buf));
}
static void emit_int(const std::string& key, long long val) {
    printf("%s=%lld\n", key.c_str(), val);
    g_kv.emplace_back(key, std::to_string(val));
}
static void emit_str(const std::string& key, const std::string& val) {
    printf("%s=%s\n", key.c_str(), val.c_str());
    g_kv.emplace_back(key, std::string("\"") + val + "\"");
}

// Prints <name>_us / <name>_us_p10 / <name>_us_p90 from a sample vector
// already in microseconds.
static void emit_stats_us(const std::string& name, std::vector<double>& samples_us) {
    emit_num(name + "_us", median_of(samples_us));
    emit_num(name + "_us_p10", percentile(samples_us, 0.10));
    emit_num(name + "_us_p90", percentile(samples_us, 0.90));
}
static void emit_stats_ms(const std::string& name, std::vector<double>& samples_ms) {
    emit_num(name + "_ms", median_of(samples_ms));
    emit_num(name + "_ms_p10", percentile(samples_ms, 0.10));
    emit_num(name + "_ms_p90", percentile(samples_ms, 0.90));
}

static float event_elapsed_us(hipEvent_t a, hipEvent_t b) {
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms * 1000.f;
}

// ------------------------------------------------------------- empty kernel

__global__ void empty_kernel() {}

// --------------------------------------------------------- Q8_0-ish GEMV --
//
// Planar Q8_0: qs[row*K + k] int8, d[row*nblk + b] half (nblk = K/32).
// One thread block per row; THREADS threads stride over the row's blocks;
// shared-memory tree reduction to one fp32 result per row.

#define THREADS 128

__device__ __forceinline__ int32_t dot4_i8(int32_t a, int32_t b, int32_t c) {
#if defined(M5_DOT_SCALAR) && M5_DOT_SCALAR
    // Plain scalar int8 dot, 4 lanes, unpacked by hand. Correctness-first
    // fallback if __builtin_amdgcn_sdot4 is unavailable for the target.
    int8_t a0 = (int8_t)(a & 0xff), a1 = (int8_t)((a >> 8) & 0xff);
    int8_t a2 = (int8_t)((a >> 16) & 0xff), a3 = (int8_t)((a >> 24) & 0xff);
    int8_t b0 = (int8_t)(b & 0xff), b1 = (int8_t)((b >> 8) & 0xff);
    int8_t b2 = (int8_t)((b >> 16) & 0xff), b3 = (int8_t)((b >> 24) & 0xff);
    return c + (int32_t)a0 * b0 + (int32_t)a1 * b1 + (int32_t)a2 * b2 + (int32_t)a3 * b3;
#else
    return __builtin_amdgcn_sdot4(a, b, c, false);
#endif
}

// Generic Q8_0 GEMV: out[row] = sum_k W[row,k] * x[k], row in [0, rows).
// W is a "matrix slab": qs_base/d_base point at the *first row of this
// matrix instance* (caller has already added any per-expert/per-matrix
// offset). x is a single shared activation vector (K, nblk = K/32).
__global__ void q8_0_gemv_kernel(
    const int8_t* __restrict__ w_qs, const __half* __restrict__ w_d,
    const int8_t* __restrict__ x_qs, const __half* __restrict__ x_d,
    float* __restrict__ out, int rows, int K, int nblk) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const int8_t* wrow_qs = w_qs + (size_t)row * K;
    const __half* wrow_d = w_d + (size_t)row * nblk;

    float partial = 0.f;
    for (int b = threadIdx.x; b < nblk; b += THREADS) {
        const int8_t* wq = wrow_qs + b * 32;
        const int8_t* xq = x_qs + b * 32;
        int32_t dot = 0;
        #pragma unroll
        for (int k = 0; k < 32; k += 4) {
            int32_t wpack, xpack;
            memcpy(&wpack, wq + k, 4);
            memcpy(&xpack, xq + k, 4);
            dot = dot4_i8(wpack, xpack, dot);
        }
        float scale = __half2float(wrow_d[b]) * __half2float(x_d[b]);
        partial += scale * (float)dot;
    }

    __shared__ float sdata[THREADS];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = sdata[0];
}

// MoE gate+up GEMV for TOPK experts in one launch: grid = (rows, topk, 2)
// (z==0 -> gate, z==1 -> up). Shares the same x activation across all
// experts and both matrices (one MoE step, one token position).
__global__ void moe_gate_up_kernel(
    const int8_t* __restrict__ gate_qs, const __half* __restrict__ gate_d,
    const int8_t* __restrict__ up_qs, const __half* __restrict__ up_d,
    size_t qs_stride_per_expert, size_t d_stride_per_expert,
    const int* __restrict__ ids,
    const int8_t* __restrict__ x_qs, const __half* __restrict__ x_d,
    float* __restrict__ out_gate, float* __restrict__ out_up,
    int rows, int K, int nblk) {
    int row = blockIdx.x;
    int rank = blockIdx.y;
    int which = blockIdx.z; // 0 = gate, 1 = up
    if (row >= rows) return;
    int expert_id = ids[rank];

    const int8_t* base_qs = (which == 0) ? gate_qs : up_qs;
    const __half* base_d = (which == 0) ? gate_d : up_d;
    const int8_t* wrow_qs = base_qs + (size_t)expert_id * qs_stride_per_expert + (size_t)row * K;
    const __half* wrow_d = base_d + (size_t)expert_id * d_stride_per_expert + (size_t)row * nblk;

    float partial = 0.f;
    for (int b = threadIdx.x; b < nblk; b += THREADS) {
        const int8_t* wq = wrow_qs + b * 32;
        const int8_t* xq = x_qs + b * 32;
        int32_t dot = 0;
        #pragma unroll
        for (int k = 0; k < 32; k += 4) {
            int32_t wpack, xpack;
            memcpy(&wpack, wq + k, 4);
            memcpy(&xpack, xq + k, 4);
            dot = dot4_i8(wpack, xpack, dot);
        }
        float scale = __half2float(wrow_d[b]) * __half2float(x_d[b]);
        partial += scale * (float)dot;
    }

    __shared__ float sdata[THREADS];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float* out = (which == 0) ? out_gate : out_up;
        out[(size_t)rank * rows + row] = sdata[0];
    }
}

// SiLU(gate)*up, then quantise the result into Q8_0 blocks of 32 in one
// launch (block == one 32-wide group, one thread per element -> exact
// wavefront-sized reduction via shared memory, no cross-block traffic).
// grid = (topk, nblk_h), block = 32.
__global__ void moe_silu_quant_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    int rows_per_mat, int8_t* __restrict__ h_qs, __half* __restrict__ h_d, int nblk) {
    int rank = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x; // 0..31
    int idx = blk * 32 + tid;
    if (idx >= rows_per_mat) return;

    float g = gate[(size_t)rank * rows_per_mat + idx];
    float u = up[(size_t)rank * rows_per_mat + idx];
    float silu = g / (1.f + expf(-g));
    float val = silu * u;

    __shared__ float sh[32];
    sh[tid] = fabsf(val);
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }
    float amax = sh[0];
    float scale = amax > 0.f ? amax / 127.f : 1.f;
    int q = (int)lroundf(val / scale);
    q = max(-127, min(127, q));
    h_qs[(size_t)rank * rows_per_mat + idx] = (int8_t)q;
    if (tid == 0) h_d[(size_t)rank * nblk + blk] = __float2half(scale);
}

// Down GEMV for TOPK experts, one launch, writes TOPK rows (each of length
// `rows`). grid = (rows, topk).
__global__ void moe_down_kernel(
    const int8_t* __restrict__ down_qs, const __half* __restrict__ down_d,
    size_t qs_stride_per_expert, size_t d_stride_per_expert,
    const int* __restrict__ ids,
    const int8_t* __restrict__ h_qs, const __half* __restrict__ h_d,
    float* __restrict__ down_out,
    int rows, int K, int nblk) {
    int row = blockIdx.x;
    int rank = blockIdx.y;
    if (row >= rows) return;
    int expert_id = ids[rank];

    const int8_t* wrow_qs = down_qs + (size_t)expert_id * qs_stride_per_expert + (size_t)row * K;
    const __half* wrow_d = down_d + (size_t)expert_id * d_stride_per_expert + (size_t)row * nblk;
    const int8_t* xq_base = h_qs + (size_t)rank * K;
    const __half* xd_base = h_d + (size_t)rank * nblk;

    float partial = 0.f;
    for (int b = threadIdx.x; b < nblk; b += THREADS) {
        const int8_t* wq = wrow_qs + b * 32;
        const int8_t* xq = xq_base + b * 32;
        int32_t dot = 0;
        #pragma unroll
        for (int k = 0; k < 32; k += 4) {
            int32_t wpack, xpack;
            memcpy(&wpack, wq + k, 4);
            memcpy(&xpack, xq + k, 4);
            dot = dot4_i8(wpack, xpack, dot);
        }
        float scale = __half2float(wrow_d[b]) * __half2float(xd_base[b]);
        partial += scale * (float)dot;
    }

    __shared__ float sdata[THREADS];
    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(size_t)rank * rows + row] = sdata[0];
}

__global__ void moe_sum_kernel(const float* __restrict__ down_out,
                                const float* __restrict__ weights,
                                float* __restrict__ y, int rows, int topk) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= rows) return;
    float acc = 0.f;
    for (int r = 0; r < topk; r++) acc += weights[r] * down_out[(size_t)r * rows + elem];
    y[elem] = acc;
}

// Plain read-reduce bandwidth reference: grid-stride sum over a buffer.
__global__ void read_reduce_kernel(const float* __restrict__ buf, size_t n, float* __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += buf[i];
    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

// ------------------------------------------------------- host-side quant --

static void quantize_row_host(const float* src, int K, int8_t* qs_out, uint16_t* d_out_bits) {
    int nblk = K / 32;
    for (int b = 0; b < nblk; b++) {
        float amax = 0.f;
        for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(src[b * 32 + k]));
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        __half hd = __float2half(scale);
        memcpy(&d_out_bits[b], &hd, 2);
        for (int k = 0; k < 32; k++) {
            float v = src[b * 32 + k] / scale;
            int q = (int)lroundf(v);
            q = std::max(-127, std::min(127, q));
            qs_out[b * 32 + k] = (int8_t)q;
        }
    }
}

// fp64 reference dot of two quantised rows using the SAME blockwise formula
// the kernels use (scale-product times int dot, summed) — this is the
// higher-precision version of the exact arithmetic the GPU performs, not a
// separately-derived elementwise dequant-multiply; it catches kernel bugs
// (wrong indexing, wrong reduction) while only absorbing fp32-vs-fp64
// rounding, which is far under the 1e-2 relative tolerance.
static double cpu_dot_q8(const int8_t* wq, const uint16_t* wd_bits,
                          const int8_t* xq, const uint16_t* xd_bits, int nblk) {
    double acc = 0.0;
    for (int b = 0; b < nblk; b++) {
        __half hw, hx;
        memcpy(&hw, &wd_bits[b], 2);
        memcpy(&hx, &xd_bits[b], 2);
        double ds = (double)__half2float(hw) * (double)__half2float(hx);
        int64_t idot = 0;
        for (int k = 0; k < 32; k++) idot += (int64_t)wq[b * 32 + k] * (int64_t)xq[b * 32 + k];
        acc += ds * (double)idot;
    }
    return acc;
}

// ------------------------------------------------------- correctness: gemv

static double g_check_gemv_maxabs = 0.0;
static double g_check_moe_maxabs = 0.0;

static void run_gemv_correctness_check() {
    const int rows = 17, K = 96; // 3 blocks, deliberately not power-of-2
    const int nblk = K / 32;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> w_f(rows * K), x_f(K);
    for (auto& v : w_f) v = dist(rng);
    for (auto& v : x_f) v = dist(rng);

    std::vector<int8_t> w_qs(rows * K), x_qs(K);
    std::vector<uint16_t> w_d(rows * nblk), x_d(nblk);
    for (int r = 0; r < rows; r++)
        quantize_row_host(&w_f[r * K], K, &w_qs[r * K], &w_d[r * nblk]);
    quantize_row_host(x_f.data(), K, x_qs.data(), x_d.data());

    // CPU reference
    std::vector<double> cpu_y(rows);
    for (int r = 0; r < rows; r++)
        cpu_y[r] = cpu_dot_q8(&w_qs[r * K], &w_d[r * nblk], x_qs.data(), x_d.data(), nblk);

    // GPU
    int8_t *d_wqs, *d_xqs; __half *d_wd, *d_xd; float* d_y;
    HIP_CHECK(hipMalloc(&d_wqs, rows * K));
    HIP_CHECK(hipMalloc(&d_wd, rows * nblk * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_xqs, K));
    HIP_CHECK(hipMalloc(&d_xd, nblk * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_y, rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_wqs, w_qs.data(), rows * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wd, w_d.data(), rows * nblk * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xqs, x_qs.data(), K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xd, x_d.data(), nblk * sizeof(__half), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(q8_0_gemv_kernel, dim3(rows), dim3(THREADS), 0, 0,
                        d_wqs, d_wd, d_xqs, d_xd, d_y, rows, K, nblk);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> gpu_y(rows);
    HIP_CHECK(hipMemcpy(gpu_y.data(), d_y, rows * sizeof(float), hipMemcpyDeviceToHost));

    double maxabs_diff = 0.0, maxabs_ref = 1e-30;
    for (int r = 0; r < rows; r++) {
        maxabs_diff = std::max(maxabs_diff, std::fabs((double)gpu_y[r] - cpu_y[r]));
        maxabs_ref = std::max(maxabs_ref, std::fabs(cpu_y[r]));
    }
    g_check_gemv_maxabs = maxabs_diff / maxabs_ref;

    hipFree(d_wqs); hipFree(d_wd); hipFree(d_xqs); hipFree(d_xd); hipFree(d_y);
}

// -------------------------------------------------------- correctness: moe

static void run_moe_correctness_check() {
    const int GU_ROWS = 32, GU_K = 64;       // 1 down-block, 2 gate/up-blocks
    const int DOWN_ROWS = 8, DOWN_K = GU_ROWS; // 32 == 1 block
    const int N_EXPERTS = 4, TOPK = 2;
    const int gu_nblk = GU_K / 32;   // 2
    const int down_nblk = DOWN_K / 32; // 1

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // host fp32 weights
    std::vector<float> gate_f(N_EXPERTS * GU_ROWS * GU_K), up_f(N_EXPERTS * GU_ROWS * GU_K);
    std::vector<float> down_f(N_EXPERTS * DOWN_ROWS * DOWN_K);
    std::vector<float> x_f(GU_K);
    for (auto& v : gate_f) v = dist(rng);
    for (auto& v : up_f) v = dist(rng);
    for (auto& v : down_f) v = dist(rng);
    for (auto& v : x_f) v = dist(rng);

    std::vector<int8_t> gate_qs(gate_f.size()), up_qs(up_f.size()), down_qs(down_f.size()), x_qs(GU_K);
    std::vector<uint16_t> gate_d(N_EXPERTS * GU_ROWS * gu_nblk), up_d(N_EXPERTS * GU_ROWS * gu_nblk);
    std::vector<uint16_t> down_d(N_EXPERTS * DOWN_ROWS * down_nblk), x_d(gu_nblk);
    for (int e = 0; e < N_EXPERTS; e++) {
        for (int r = 0; r < GU_ROWS; r++) {
            size_t off = (size_t)e * GU_ROWS * GU_K + (size_t)r * GU_K;
            size_t doff = (size_t)e * GU_ROWS * gu_nblk + (size_t)r * gu_nblk;
            quantize_row_host(&gate_f[off], GU_K, &gate_qs[off], &gate_d[doff]);
            quantize_row_host(&up_f[off], GU_K, &up_qs[off], &up_d[doff]);
        }
        for (int r = 0; r < DOWN_ROWS; r++) {
            size_t off = (size_t)e * DOWN_ROWS * DOWN_K + (size_t)r * DOWN_K;
            size_t doff = (size_t)e * DOWN_ROWS * down_nblk + (size_t)r * down_nblk;
            quantize_row_host(&down_f[off], DOWN_K, &down_qs[off], &down_d[doff]);
        }
    }
    quantize_row_host(x_f.data(), GU_K, x_qs.data(), x_d.data());

    int ids[TOPK] = {0, 2};
    float weights[TOPK] = {0.6f, 0.4f};

    // ---- CPU reference, mirroring the GPU pipeline exactly (fp64) ----
    std::vector<double> gate_cpu(TOPK * GU_ROWS), up_cpu(TOPK * GU_ROWS);
    for (int r = 0; r < TOPK; r++) {
        int e = ids[r];
        for (int row = 0; row < GU_ROWS; row++) {
            size_t off = (size_t)e * GU_ROWS * GU_K + (size_t)row * GU_K;
            size_t doff = (size_t)e * GU_ROWS * gu_nblk + (size_t)row * gu_nblk;
            gate_cpu[r * GU_ROWS + row] = cpu_dot_q8(&gate_qs[off], &gate_d[doff], x_qs.data(), x_d.data(), gu_nblk);
            up_cpu[r * GU_ROWS + row] = cpu_dot_q8(&up_qs[off], &up_d[doff], x_qs.data(), x_d.data(), gu_nblk);
        }
    }
    std::vector<float> h_f(TOPK * GU_ROWS);
    for (size_t i = 0; i < h_f.size(); i++) {
        double g = gate_cpu[i], u = up_cpu[i];
        double silu = g / (1.0 + exp(-g));
        h_f[i] = (float)(silu * u);
    }
    std::vector<int8_t> h_qs_cpu(TOPK * DOWN_K);
    std::vector<uint16_t> h_d_cpu(TOPK * down_nblk);
    for (int r = 0; r < TOPK; r++)
        quantize_row_host(&h_f[r * GU_ROWS], DOWN_K, &h_qs_cpu[r * DOWN_K], &h_d_cpu[r * down_nblk]);

    std::vector<double> down_cpu(TOPK * DOWN_ROWS);
    for (int r = 0; r < TOPK; r++) {
        int e = ids[r];
        for (int row = 0; row < DOWN_ROWS; row++) {
            size_t off = (size_t)e * DOWN_ROWS * DOWN_K + (size_t)row * DOWN_K;
            size_t doff = (size_t)e * DOWN_ROWS * down_nblk + (size_t)row * down_nblk;
            down_cpu[r * DOWN_ROWS + row] = cpu_dot_q8(&down_qs[off], &down_d[doff],
                                                        &h_qs_cpu[r * DOWN_K], &h_d_cpu[r * down_nblk], down_nblk);
        }
    }
    std::vector<double> y_cpu(DOWN_ROWS, 0.0);
    for (int elem = 0; elem < DOWN_ROWS; elem++)
        for (int r = 0; r < TOPK; r++)
            y_cpu[elem] += weights[r] * down_cpu[r * DOWN_ROWS + elem];

    // ---- GPU pipeline, same 5 kernels used by the full-scale run ----
    int8_t *d_gate_qs, *d_up_qs, *d_down_qs, *d_x_qs, *d_h_qs;
    __half *d_gate_d, *d_up_d, *d_down_d, *d_x_d, *d_h_d;
    int* d_ids; float* d_weights;
    float *d_gate_out, *d_up_out, *d_down_out, *d_y;

    HIP_CHECK(hipMalloc(&d_gate_qs, gate_qs.size()));
    HIP_CHECK(hipMalloc(&d_up_qs, up_qs.size()));
    HIP_CHECK(hipMalloc(&d_down_qs, down_qs.size()));
    HIP_CHECK(hipMalloc(&d_gate_d, gate_d.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_up_d, up_d.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_down_d, down_d.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_x_qs, x_qs.size()));
    HIP_CHECK(hipMalloc(&d_x_d, x_d.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_h_qs, TOPK * DOWN_K));
    HIP_CHECK(hipMalloc(&d_h_d, TOPK * down_nblk * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_ids, TOPK * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_weights, TOPK * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate_out, TOPK * GU_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_up_out, TOPK * GU_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_down_out, TOPK * DOWN_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, DOWN_ROWS * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_gate_qs, gate_qs.data(), gate_qs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up_qs, up_qs.data(), up_qs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_down_qs, down_qs.data(), down_qs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate_d, gate_d.data(), gate_d.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_up_d, up_d.data(), up_d.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_down_d, down_d.data(), down_d.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_qs, x_qs.data(), x_qs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_d, x_d.data(), x_d.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ids, ids, TOPK * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, weights, TOPK * sizeof(float), hipMemcpyHostToDevice));

    size_t gu_qs_stride = (size_t)GU_ROWS * GU_K;
    size_t gu_d_stride = (size_t)GU_ROWS * gu_nblk;
    size_t down_qs_stride = (size_t)DOWN_ROWS * DOWN_K;
    size_t down_d_stride = (size_t)DOWN_ROWS * down_nblk;

    hipLaunchKernelGGL(moe_gate_up_kernel, dim3(GU_ROWS, TOPK, 2), dim3(THREADS), 0, 0,
                        d_gate_qs, d_gate_d, d_up_qs, d_up_d, gu_qs_stride, gu_d_stride,
                        d_ids, d_x_qs, d_x_d, d_gate_out, d_up_out, GU_ROWS, GU_K, gu_nblk);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(moe_silu_quant_kernel, dim3(TOPK, down_nblk), dim3(32), 0, 0,
                        d_gate_out, d_up_out, GU_ROWS, d_h_qs, d_h_d, down_nblk);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(moe_down_kernel, dim3(DOWN_ROWS, TOPK), dim3(THREADS), 0, 0,
                        d_down_qs, d_down_d, down_qs_stride, down_d_stride, d_ids,
                        d_h_qs, d_h_d, d_down_out, DOWN_ROWS, DOWN_K, down_nblk);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(moe_sum_kernel, dim3((DOWN_ROWS + 255) / 256), dim3(256), 0, 0,
                        d_down_out, d_weights, d_y, DOWN_ROWS, TOPK);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> y_gpu(DOWN_ROWS);
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, DOWN_ROWS * sizeof(float), hipMemcpyDeviceToHost));

    double maxabs_diff = 0.0, maxabs_ref = 1e-30;
    for (int i = 0; i < DOWN_ROWS; i++) {
        maxabs_diff = std::max(maxabs_diff, std::fabs((double)y_gpu[i] - y_cpu[i]));
        maxabs_ref = std::max(maxabs_ref, std::fabs(y_cpu[i]));
    }
    g_check_moe_maxabs = maxabs_diff / maxabs_ref;

    hipFree(d_gate_qs); hipFree(d_up_qs); hipFree(d_down_qs);
    hipFree(d_gate_d); hipFree(d_up_d); hipFree(d_down_d);
    hipFree(d_x_qs); hipFree(d_x_d); hipFree(d_h_qs); hipFree(d_h_d);
    hipFree(d_ids); hipFree(d_weights);
    hipFree(d_gate_out); hipFree(d_up_out); hipFree(d_down_out); hipFree(d_y);
}

// ------------------------------------------------------------- M5 item 1 --

static void bench_launch() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const int CHUNKS = 100, PER_CHUNK = 100; // 10 000 launches total
    std::vector<hipEvent_t> starts(CHUNKS), ends(CHUNKS);
    for (int c = 0; c < CHUNKS; c++) {
        HIP_CHECK(hipEventCreate(&starts[c]));
        HIP_CHECK(hipEventCreate(&ends[c]));
    }
    // warm-up
    for (int i = 0; i < 100; i++) hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int c = 0; c < CHUNKS; c++) {
        HIP_CHECK(hipEventRecord(starts[c], stream));
        for (int k = 0; k < PER_CHUNK; k++)
            hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipEventRecord(ends[c], stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> per_launch_us(CHUNKS);
    for (int c = 0; c < CHUNKS; c++)
        per_launch_us[c] = event_elapsed_us(starts[c], ends[c]) / PER_CHUNK;
    emit_stats_us("launch", per_launch_us);

    for (int c = 0; c < CHUNKS; c++) { hipEventDestroy(starts[c]); hipEventDestroy(ends[c]); }
    hipStreamDestroy(stream);
}

static void bench_launch_sync() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const int ITERS = 2000;
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    std::vector<double> samples(ITERS);
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        samples[i] = now_us() - t0;
    }
    emit_stats_us("launch_sync", samples);
    hipStreamDestroy(stream);
}

// ------------------------------------------------------------- M5 item 2 --

static void bench_graph() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    const int NODES = 2000, LAUNCHES = 100;

    hipGraph_t graph;
    hipGraphExec_t graphExec;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < NODES; i++)
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, stream);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // warm-up
    for (int i = 0; i < 10; i++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<hipEvent_t> starts(LAUNCHES), ends(LAUNCHES);
    for (int i = 0; i < LAUNCHES; i++) {
        HIP_CHECK(hipEventCreate(&starts[i]));
        HIP_CHECK(hipEventCreate(&ends[i]));
    }
    for (int i = 0; i < LAUNCHES; i++) {
        HIP_CHECK(hipEventRecord(starts[i], stream));
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        HIP_CHECK(hipEventRecord(ends[i], stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> whole_us(LAUNCHES), node_us(LAUNCHES);
    for (int i = 0; i < LAUNCHES; i++) {
        double us = event_elapsed_us(starts[i], ends[i]);
        whole_us[i] = us;
        node_us[i] = us / NODES;
    }
    emit_stats_us("graph_launch", whole_us);
    emit_stats_us("graph_node", node_us);

    for (int i = 0; i < LAUNCHES; i++) { hipEventDestroy(starts[i]); hipEventDestroy(ends[i]); }
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
}

// ------------------------------------------------------------- M5 item 3 --

static void bench_event_wait() {
    hipStream_t s1, s2;
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    hipEvent_t e, e2;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventCreate(&e2));

    const int ITERS = 5000;
    std::vector<hipEvent_t> tick(ITERS);
    for (int i = 0; i < ITERS; i++) HIP_CHECK(hipEventCreate(&tick[i]));
    hipEvent_t start;
    HIP_CHECK(hipEventCreate(&start));

    // warm-up (untimed) round trips
    for (int i = 0; i < 100; i++) {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s1);
        HIP_CHECK(hipEventRecord(e, s1));
        HIP_CHECK(hipStreamWaitEvent(s2, e, 0));
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s2);
        HIP_CHECK(hipEventRecord(e2, s2));
        HIP_CHECK(hipStreamWaitEvent(s1, e2, 0));
    }
    HIP_CHECK(hipStreamSynchronize(s1));
    HIP_CHECK(hipStreamSynchronize(s2));

    HIP_CHECK(hipEventRecord(start, s1));
    for (int i = 0; i < ITERS; i++) {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s1);
        HIP_CHECK(hipEventRecord(e, s1));
        HIP_CHECK(hipStreamWaitEvent(s2, e, 0));
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s2);
        HIP_CHECK(hipEventRecord(e2, s2));
        HIP_CHECK(hipStreamWaitEvent(s1, e2, 0));
        HIP_CHECK(hipEventRecord(tick[i], s1)); // trailing marker, end of round trip i
    }
    HIP_CHECK(hipStreamSynchronize(s1));
    HIP_CHECK(hipStreamSynchronize(s2));

    std::vector<double> rt_us(ITERS);
    rt_us[0] = event_elapsed_us(start, tick[0]);
    for (int i = 1; i < ITERS; i++) rt_us[i] = event_elapsed_us(tick[i - 1], tick[i]);
    emit_stats_us("event_wait", rt_us);

    for (int i = 0; i < ITERS; i++) hipEventDestroy(tick[i]);
    hipEventDestroy(start); hipEventDestroy(e); hipEventDestroy(e2);
    hipStreamDestroy(s1); hipStreamDestroy(s2);
}

// ------------------------------------------------------------- M5 item 4 --

static void bench_p2p(int ndev) {
    std::vector<std::string> pci(ndev);
    for (int d = 0; d < ndev; d++) {
        HIP_CHECK(hipSetDevice(d));
        char buf[32];
        HIP_CHECK(hipDeviceGetPCIBusId(buf, sizeof(buf), d));
        pci[d] = buf;
        emit_str("dev_" + std::to_string(d) + "_pci", pci[d]);
    }

    void* h_pinned;
    const size_t SMALL_BYTES = 10240; // 2560 f32
    HIP_CHECK(hipHostMalloc(&h_pinned, SMALL_BYTES, hipHostMallocDefault));

    for (int i = 0; i < ndev; i++) {
        for (int j = 0; j < ndev; j++) {
            if (i == j) continue;
            std::string tag = std::to_string(i) + "_" + std::to_string(j);

            int can_access = 0;
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipDeviceCanAccessPeer(&can_access, i, j));
            emit_int("p2p_" + tag + "_can_access", can_access);

            int enabled = 0;
            if (can_access) {
                hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                enabled = (e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) ? 1 : 0;
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                    // clear the sticky error so it doesn't leak into later checks
                    hipGetLastError();
                }
            }
            emit_int("p2p_" + tag + "_enabled", enabled);

            void *d_src, *d_dst;
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&d_src, SMALL_BYTES));
            HIP_CHECK(hipSetDevice(j));
            HIP_CHECK(hipMalloc(&d_dst, SMALL_BYTES));
            hipStream_t stream;
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamCreate(&stream));

            if (enabled) {
                const int ITERS = 5000;
                for (int w = 0; w < 100; w++) {
                    HIP_CHECK(hipMemcpyPeerAsync(d_dst, j, d_src, i, SMALL_BYTES, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                }
                std::vector<double> samples(ITERS);
                for (int it = 0; it < ITERS; it++) {
                    double t0 = now_us();
                    HIP_CHECK(hipMemcpyPeerAsync(d_dst, j, d_src, i, SMALL_BYTES, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                    samples[it] = now_us() - t0;
                }
                emit_stats_us("p2p_" + tag, samples);

                // 64 MB bandwidth
                const size_t BIG = 64ULL * 1024 * 1024;
                void *d_src_big, *d_dst_big;
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipMalloc(&d_src_big, BIG));
                HIP_CHECK(hipSetDevice(j));
                HIP_CHECK(hipMalloc(&d_dst_big, BIG));
                hipEvent_t bstart, bend;
                HIP_CHECK(hipEventCreate(&bstart));
                HIP_CHECK(hipEventCreate(&bend));
                std::vector<double> bw_times_us(10);
                for (int r = 0; r < 10; r++) {
                    HIP_CHECK(hipEventRecord(bstart, stream));
                    HIP_CHECK(hipMemcpyPeerAsync(d_dst_big, j, d_src_big, i, BIG, stream));
                    HIP_CHECK(hipEventRecord(bend, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                    bw_times_us[r] = event_elapsed_us(bstart, bend);
                }
                double med_us = median_of(bw_times_us);
                double bw_gbs = (double)BIG / (med_us * 1e-6) / 1e9;
                emit_num("p2p_" + tag + "_bw_gbs", bw_gbs);
                hipEventDestroy(bstart); hipEventDestroy(bend);
                hipFree(d_src_big); hipFree(d_dst_big);
            }

            // host bounce: device i -> pinned host -> device j, synchronous
            {
                const int ITERS = 2000;
                for (int w = 0; w < 50; w++) {
                    HIP_CHECK(hipSetDevice(i));
                    HIP_CHECK(hipMemcpy(h_pinned, d_src, SMALL_BYTES, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(j));
                    HIP_CHECK(hipMemcpy(d_dst, h_pinned, SMALL_BYTES, hipMemcpyHostToDevice));
                }
                std::vector<double> samples(ITERS);
                for (int it = 0; it < ITERS; it++) {
                    double t0 = now_us();
                    HIP_CHECK(hipSetDevice(i));
                    HIP_CHECK(hipMemcpy(h_pinned, d_src, SMALL_BYTES, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(j));
                    HIP_CHECK(hipMemcpy(d_dst, h_pinned, SMALL_BYTES, hipMemcpyHostToDevice));
                    samples[it] = now_us() - t0;
                }
                emit_stats_us("host_bounce_" + tag, samples);
            }

            hipStreamDestroy(stream);
            hipFree(d_src);
            hipFree(d_dst);
        }
    }
    hipHostFree(h_pinned);
}

// ------------------------------------------------------------- M5 item 5 --

struct TokenLayerBuffers {
    // trunk
    int8_t* trunk_qs; __half* trunk_d;
    int trunk_rows, trunk_K, trunk_nblk;
    float* y_trunk;

    // experts (one layer's worth, reused for all 48 layers)
    int8_t *gate_qs, *up_qs, *down_qs;
    __half *gate_d, *up_d, *down_d;
    int n_experts, gu_rows, gu_K, gu_nblk;
    int down_rows, down_K, down_nblk;
    size_t gu_qs_stride, gu_d_stride, down_qs_stride, down_d_stride;

    // shared activation (fixed across layers/tokens, cheap)
    int8_t* x_qs; __half* x_d;

    // per-layer scratch (overwritten every layer -- correct because same
    // stream serialises dependent kernels in program order)
    int* ids;         // TOPK
    float* weights;   // TOPK
    float *gate_out, *up_out; // TOPK * gu_rows
    int8_t* h_qs; __half* h_d; // TOPK * down_K, TOPK * down_nblk
    float* down_out;  // TOPK * down_rows
    float* y_moe;     // down_rows

    int topk;
};

static void run_one_layer(hipStream_t stream, const TokenLayerBuffers& b) {
    hipLaunchKernelGGL(q8_0_gemv_kernel, dim3(b.trunk_rows), dim3(THREADS), 0, stream,
                        b.trunk_qs, b.trunk_d, b.x_qs, b.x_d, b.y_trunk,
                        b.trunk_rows, b.trunk_K, b.trunk_nblk);

    hipLaunchKernelGGL(moe_gate_up_kernel, dim3(b.gu_rows, b.topk, 2), dim3(THREADS), 0, stream,
                        b.gate_qs, b.gate_d, b.up_qs, b.up_d, b.gu_qs_stride, b.gu_d_stride,
                        b.ids, b.x_qs, b.x_d, b.gate_out, b.up_out, b.gu_rows, b.gu_K, b.gu_nblk);

    hipLaunchKernelGGL(moe_silu_quant_kernel, dim3(b.topk, b.down_nblk), dim3(32), 0, stream,
                        b.gate_out, b.up_out, b.gu_rows, b.h_qs, b.h_d, b.down_nblk);

    hipLaunchKernelGGL(moe_down_kernel, dim3(b.down_rows, b.topk), dim3(THREADS), 0, stream,
                        b.down_qs, b.down_d, b.down_qs_stride, b.down_d_stride, b.ids,
                        b.h_qs, b.h_d, b.down_out, b.down_rows, b.down_K, b.down_nblk);

    hipLaunchKernelGGL(moe_sum_kernel, dim3((b.down_rows + 255) / 256), dim3(256), 0, stream,
                        b.down_out, b.weights, b.y_moe, b.down_rows, b.topk);
}

static void bench_token() {
    const int N_LAYERS = 48;
    const int TRUNK_ROWS = 10240, TRUNK_K = 2560;
    const int GU_ROWS = 640, GU_K = 2560;
    const int DOWN_ROWS = 2560, DOWN_K = GU_ROWS; // 640
    const int N_EXPERTS = 512, TOPK = 10;
    const int trunk_nblk = TRUNK_K / 32;
    const int gu_nblk = GU_K / 32;
    const int down_nblk = DOWN_K / 32;

    TokenLayerBuffers b{};
    b.trunk_rows = TRUNK_ROWS; b.trunk_K = TRUNK_K; b.trunk_nblk = trunk_nblk;
    b.n_experts = N_EXPERTS; b.gu_rows = GU_ROWS; b.gu_K = GU_K; b.gu_nblk = gu_nblk;
    b.down_rows = DOWN_ROWS; b.down_K = DOWN_K; b.down_nblk = down_nblk;
    b.topk = TOPK;
    b.gu_qs_stride = (size_t)GU_ROWS * GU_K;
    b.gu_d_stride = (size_t)GU_ROWS * gu_nblk;
    b.down_qs_stride = (size_t)DOWN_ROWS * DOWN_K;
    b.down_d_stride = (size_t)DOWN_ROWS * down_nblk;

    // ---- allocate + fill trunk weights ----
    size_t trunk_qs_bytes = (size_t)TRUNK_ROWS * TRUNK_K;
    size_t trunk_d_bytes = (size_t)TRUNK_ROWS * trunk_nblk * sizeof(__half);
    HIP_CHECK(hipMalloc(&b.trunk_qs, trunk_qs_bytes));
    HIP_CHECK(hipMalloc(&b.trunk_d, trunk_d_bytes));
    HIP_CHECK(hipMalloc(&b.y_trunk, TRUNK_ROWS * sizeof(float)));
    {
        std::vector<int8_t> tmp_qs(trunk_qs_bytes);
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> di(-127, 127);
        for (auto& v : tmp_qs) v = (int8_t)di(rng);
        HIP_CHECK(hipMemcpy(b.trunk_qs, tmp_qs.data(), trunk_qs_bytes, hipMemcpyHostToDevice));
        std::vector<uint16_t> tmp_d(TRUNK_ROWS * trunk_nblk);
        __half one = __float2half(0.01f);
        uint16_t one_bits; memcpy(&one_bits, &one, 2);
        for (auto& v : tmp_d) v = one_bits;
        HIP_CHECK(hipMemcpy(b.trunk_d, tmp_d.data(), trunk_d_bytes, hipMemcpyHostToDevice));
    }

    // ---- allocate + fill expert weights (one layer's worth, reused) ----
    size_t gate_qs_bytes = (size_t)N_EXPERTS * GU_ROWS * GU_K;
    size_t gate_d_bytes = (size_t)N_EXPERTS * GU_ROWS * gu_nblk * sizeof(__half);
    size_t down_qs_bytes = (size_t)N_EXPERTS * DOWN_ROWS * DOWN_K;
    size_t down_d_bytes = (size_t)N_EXPERTS * DOWN_ROWS * down_nblk * sizeof(__half);

    HIP_CHECK(hipMalloc(&b.gate_qs, gate_qs_bytes));
    HIP_CHECK(hipMalloc(&b.up_qs, gate_qs_bytes));
    HIP_CHECK(hipMalloc(&b.down_qs, down_qs_bytes));
    HIP_CHECK(hipMalloc(&b.gate_d, gate_d_bytes));
    HIP_CHECK(hipMalloc(&b.up_d, gate_d_bytes));
    HIP_CHECK(hipMalloc(&b.down_d, down_d_bytes));

    {
        // Fill with a constant pattern via hipMemset for qs (values don't
        // matter for a bandwidth/launch-floor measurement) and a small host
        // buffer broadcast for scales (avoid a multi-GB host allocation).
        HIP_CHECK(hipMemset(b.gate_qs, 3, gate_qs_bytes));
        HIP_CHECK(hipMemset(b.up_qs, 5, gate_qs_bytes));
        HIP_CHECK(hipMemset(b.down_qs, 7, down_qs_bytes));

        __half hs = __float2half(0.01f);
        uint16_t hs_bits; memcpy(&hs_bits, &hs, 2);
        size_t max_halfs = std::max(gate_d_bytes, down_d_bytes) / sizeof(__half);
        std::vector<uint16_t> tmp_d(max_halfs, hs_bits);
        HIP_CHECK(hipMemcpy(b.gate_d, tmp_d.data(), gate_d_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b.up_d, tmp_d.data(), gate_d_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b.down_d, tmp_d.data(), down_d_bytes, hipMemcpyHostToDevice));
    }

    // ---- shared activation ----
    HIP_CHECK(hipMalloc(&b.x_qs, GU_K));
    HIP_CHECK(hipMalloc(&b.x_d, gu_nblk * sizeof(__half)));
    HIP_CHECK(hipMemset(b.x_qs, 11, GU_K));
    {
        __half hs = __float2half(0.02f);
        uint16_t hs_bits; memcpy(&hs_bits, &hs, 2);
        std::vector<uint16_t> tmp(gu_nblk, hs_bits);
        HIP_CHECK(hipMemcpy(b.x_d, tmp.data(), gu_nblk * sizeof(__half), hipMemcpyHostToDevice));
    }

    // ---- per-layer scratch + ids/weights (reused every layer) ----
    HIP_CHECK(hipMalloc(&b.ids, TOPK * sizeof(int)));
    HIP_CHECK(hipMalloc(&b.weights, TOPK * sizeof(float)));
    {
        std::vector<int> ids(TOPK);
        std::vector<float> w(TOPK, 1.0f / TOPK);
        for (int i = 0; i < TOPK; i++) ids[i] = i * (N_EXPERTS / TOPK); // 10 spread-out experts
        HIP_CHECK(hipMemcpy(b.ids, ids.data(), TOPK * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b.weights, w.data(), TOPK * sizeof(float), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&b.gate_out, TOPK * GU_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&b.up_out, TOPK * GU_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&b.h_qs, TOPK * DOWN_K));
    HIP_CHECK(hipMalloc(&b.h_d, TOPK * down_nblk * sizeof(__half)));
    HIP_CHECK(hipMalloc(&b.down_out, TOPK * DOWN_ROWS * sizeof(float)));
    HIP_CHECK(hipMalloc(&b.y_moe, DOWN_ROWS * sizeof(float)));

    // ---- bytes-per-token accounting (decimal GB) ----
    double trunk_bytes_per_layer = (double)trunk_qs_bytes + (double)(TRUNK_ROWS * trunk_nblk * sizeof(__half));
    double gu_bytes_per_expert_touch = (double)(GU_ROWS * GU_K) + (double)(GU_ROWS * gu_nblk * sizeof(__half));
    double down_bytes_per_expert_touch = (double)(DOWN_ROWS * DOWN_K) + (double)(DOWN_ROWS * down_nblk * sizeof(__half));
    double moe_bytes_per_layer = TOPK * (2.0 * gu_bytes_per_expert_touch + down_bytes_per_expert_touch);
    double bytes_per_layer = trunk_bytes_per_layer + moe_bytes_per_layer;
    double bytes_per_token = bytes_per_layer * N_LAYERS;
    double token_bytes_gb = bytes_per_token / 1e9;
    emit_num("token_bytes_gb", token_bytes_gb);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ---- separate launches ----
    const int TOKENS = 200;
    for (int i = 0; i < 5; i++) { // warm-up tokens
        for (int L = 0; L < N_LAYERS; L++) run_one_layer(stream, b);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<hipEvent_t> tick(TOKENS);
    for (int i = 0; i < TOKENS; i++) HIP_CHECK(hipEventCreate(&tick[i]));
    hipEvent_t start_sep;
    HIP_CHECK(hipEventCreate(&start_sep));
    HIP_CHECK(hipEventRecord(start_sep, stream));
    for (int t = 0; t < TOKENS; t++) {
        for (int L = 0; L < N_LAYERS; L++) run_one_layer(stream, b);
        HIP_CHECK(hipEventRecord(tick[t], stream));
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> sep_ms(TOKENS);
    sep_ms[0] = event_elapsed_us(start_sep, tick[0]) / 1000.0;
    for (int t = 1; t < TOKENS; t++) sep_ms[t] = event_elapsed_us(tick[t - 1], tick[t]) / 1000.0;
    emit_stats_ms("token_separate", sep_ms);
    double sep_median_s = median_of(sep_ms) / 1000.0;
    emit_num("token_bw_gbs_separate", token_bytes_gb / sep_median_s);

    for (int i = 0; i < TOKENS; i++) hipEventDestroy(tick[i]);
    hipEventDestroy(start_sep);

    // ---- graph-captured token ----
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int L = 0; L < N_LAYERS; L++) run_one_layer(stream, b);
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < 5; i++) HIP_CHECK(hipGraphLaunch(graphExec, stream)); // warm-up
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<hipEvent_t> gtick(TOKENS);
    for (int i = 0; i < TOKENS; i++) HIP_CHECK(hipEventCreate(&gtick[i]));
    hipEvent_t start_graph;
    HIP_CHECK(hipEventCreate(&start_graph));
    HIP_CHECK(hipEventRecord(start_graph, stream));
    for (int t = 0; t < TOKENS; t++) {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        HIP_CHECK(hipEventRecord(gtick[t], stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> graph_ms(TOKENS);
    graph_ms[0] = event_elapsed_us(start_graph, gtick[0]) / 1000.0;
    for (int t = 1; t < TOKENS; t++) graph_ms[t] = event_elapsed_us(gtick[t - 1], gtick[t]) / 1000.0;
    emit_stats_ms("token_graph", graph_ms);
    double graph_median_s = median_of(graph_ms) / 1000.0;
    emit_num("token_bw_gbs_graph", token_bytes_gb / graph_median_s);

    for (int i = 0; i < TOKENS; i++) hipEventDestroy(gtick[i]);
    hipEventDestroy(start_graph);
    hipGraphExecDestroy(graphExec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);

    hipFree(b.trunk_qs); hipFree(b.trunk_d); hipFree(b.y_trunk);
    hipFree(b.gate_qs); hipFree(b.up_qs); hipFree(b.down_qs);
    hipFree(b.gate_d); hipFree(b.up_d); hipFree(b.down_d);
    hipFree(b.x_qs); hipFree(b.x_d);
    hipFree(b.ids); hipFree(b.weights);
    hipFree(b.gate_out); hipFree(b.up_out);
    hipFree(b.h_qs); hipFree(b.h_d);
    hipFree(b.down_out); hipFree(b.y_moe);
}

static void bench_d2d_read() {
    const size_t BYTES = 2000000000ULL; // 2 GB decimal
    size_t n = BYTES / sizeof(float);
    float* d_buf;
    float* d_out;
    HIP_CHECK(hipMalloc(&d_buf, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_buf, 1, n * sizeof(float))); // pattern, value irrelevant
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int threads = 256;
    int blocks = 4096; // grid-stride loop covers the rest

    // warm-up
    hipLaunchKernelGGL(read_reduce_kernel, dim3(blocks), dim3(threads), 0, 0, d_buf, n, d_out);
    HIP_CHECK(hipDeviceSynchronize());

    const int REPS = 10;
    std::vector<double> times_us(REPS);
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));
    for (int r = 0; r < REPS; r++) {
        HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
        HIP_CHECK(hipEventRecord(s));
        hipLaunchKernelGGL(read_reduce_kernel, dim3(blocks), dim3(threads), 0, 0, d_buf, n, d_out);
        HIP_CHECK(hipEventRecord(e));
        HIP_CHECK(hipEventSynchronize(e));
        times_us[r] = event_elapsed_us(s, e);
    }
    double med_us = median_of(times_us);
    double bw_gbs = (double)BYTES / (med_us * 1e-6) / 1e9;
    emit_num("d2d_read_bw_gbs", bw_gbs);

    hipEventDestroy(s); hipEventDestroy(e);
    hipFree(d_buf); hipFree(d_out);
}

// ------------------------------------------------------------------ main --

static void write_json(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "could not open %s for write\n", path.c_str()); return; }
    fprintf(f, "{\n");
    for (size_t i = 0; i < g_kv.size(); i++) {
        fprintf(f, "  \"%s\": %s%s\n", g_kv[i].first.c_str(), g_kv[i].second.c_str(),
                (i + 1 < g_kv.size()) ? "," : "");
    }
    fprintf(f, "}\n");
    fclose(f);
}

int main(int argc, char** argv) {
    std::string json_path;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--json" && i + 1 < argc) json_path = argv[++i];
    }

#if defined(M5_DOT_SCALAR) && M5_DOT_SCALAR
    emit_str("dot_intrinsic", "scalar");
#else
    emit_str("dot_intrinsic", "amdgcn_sdot4");
#endif

    HIP_CHECK(hipSetDevice(0));

    // ---- correctness gate, must run before any timing is trusted ----
    run_gemv_correctness_check();
    run_moe_correctness_check();
    emit_num("check_gemv_maxabs", g_check_gemv_maxabs);
    emit_num("check_moe_maxabs", g_check_moe_maxabs);
    if (g_check_gemv_maxabs > 1e-2 || g_check_moe_maxabs > 1e-2) {
        printf("CHECK_FAIL\n");
        if (!json_path.empty()) write_json(json_path);
        return 1;
    }

    // ---- item 1 ----
    bench_launch();
    bench_launch_sync();

    // ---- item 2 ----
    bench_graph();

    // ---- item 3 ----
    bench_event_wait();

    // ---- item 4 ----
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    emit_int("device_count", ndev);
    HIP_CHECK(hipSetDevice(0));
    bench_p2p(ndev);
    HIP_CHECK(hipSetDevice(0));

    // ---- item 5 ----
    bench_token();
    bench_d2d_read();

    if (!json_path.empty()) write_json(json_path);
    return 0;
}
