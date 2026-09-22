// tools/hot-expert/m1/m1_hipfire.hip
//
// M1 candidate 2: hipFire's MoE kernels head-to-head against candidate 1
// (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, section M1).
//
// Kernels used, copied verbatim into hipfire_src/ (each file's own header
// names its exact source path and commit -- all at hipFire commit 2774aa2):
//   - gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip  (gate+up, dual output)
//   - gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded.hip (down, per-krank
//     output, no atomics -- HIPFIRE_MOE_DOWN_LAST_COMBINE left at its default
//     0, so the fold is the separate combine kernel below)
//   - moe_down_combine_k8_batched.hip (folds K_TOP down outputs, weighted,
//     into the residual)
//   - silu_mul.hip (silu(gate)*up, generic elementwise)
// plus gfx12_weight_cache_policy.inc, the preamble kernels.rs concatenates
// ahead of the down kernel's source on gfx1100/gfx1201 (confirmed by
// reading crates/rdna-compute/src/kernels.rs:2392 on the rig: the down
// kernel's SRC constant is `concat!("#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
// include_str!(".../gfx12_weight_cache_policy.inc"), include_str!(".../gemv_...down....hip"))`
// -- the gate_up/combine/silu_mul kernels get no such preamble in kernels.rs).
// Building the down kernel WITHOUT this preamble would compile (the ifndef
// guards default it off) but would NOT be the kernel hipFire actually ships
// on gfx1100, which this box is -- the preamble's condition matches
// `__gfx1100__` and turns on the buffer-resource load path by default.
//
// Argument lists and grid/block below were read from the Rust launchers in
// ~/src/hipfire/crates/rdna-compute/src/gemv.rs (gate_up batched ~line 11619,
// down ~line 11773) and src/moe.rs (combine ~line 18), not guessed from the
// kernel bodies alone -- summarised here for the orchestrator to check
// against those launchers directly:
//   gate_up: grid=(M, K_TOP, batch), block=(32,1,1)
//            args: expert_ptrs, topk_indices, x, y_gate, y_up, M(i32), K(i32), K_TOP(i32)
//            M = 2*mi = 1280 (mi=640 output rows each of gate/up), K = 2560
//            (grid.x = M is the gemv.rs DEFAULT on gfx1100 -- the tight-grid
//            M/2 contraction is opt-in only on gfx1151/gfx1201 there)
//   down:    grid=(M, K_TOP, batch), block=(32,1,1)
//            args: expert_ptrs, topk_indices, rot_batch, expert_outputs, M(i32), K(i32), K_TOP(i32)
//            M = 2560 (n_embd, down's output rows), K = 768 (padded, see below)
//            (grid.x = M is ALSO the gemv.rs DEFAULT here -- tight M/4 is
//            opt-in behind its own unmeasured gate; this box has none)
//   combine: grid=(ceil(M/256), batch, 1), block=(256,1,1)
//            args: expert_outputs, topk_weights, x_residual, M(i32), K_TOP(i32)
//
// Two misfits, reported not hidden (the brief's own words):
//   1. K_TOP is fixed at 8 by these kernels' contract (grid.y / the ABI's
//      K_TOP param): run with 8 experts, not the model's real top-10.
//   2. down's K=640 is not a multiple of 256. Padded to K=768 (3 groups)
//      with the extra 128 weight-columns AND the matching 128 activation
//      elements zeroed -- the zeroed weight columns quantise to a
//      degenerate all-zero MQ4G256V2 half-group (scale=0, zero=0, per the
//      down kernel's own header comment on degenerate groups), so they
//      contribute exactly 0 regardless of the (also zero) activation.
//      Extra bytes actually read from VRAM: 3 groups vs an ideal 640/256=2.5,
//      i.e. +20% on the down expert's bytes -- hipfire_down_pad_overhead=0.2.
//
// `x` here is plain random fp32, NOT "FWHT-rotated post-RMSNorm" as the
// down kernel's own shape comment describes hipFire's real pipeline input --
// this harness only exercises the GEMV kernels' bytes/cycles, not hipFire's
// activation-rotation step, so the values are semantically arbitrary (the
// kernel does not care).
//
// Building this program does NOT touch the GPUs (offline hipcc codegen for
// --offload-arch=gfx1100, no docker --device flags). It is not run here.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "m1_common.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------
// Kernel sources, included verbatim (see hipfire_src/*.hip's own header
// comments for exact provenance). The down kernel needs the weight-buffer
// preamble defined ahead of its #include -- gate_up/combine/silu_mul do not
// (confirmed against kernels.rs, see the file-level comment above).
// ---------------------------------------------------------------------
#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1
#include "hipfire_src/gfx12_weight_cache_policy.inc"
#include "hipfire_src/gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded.hip"
#include "hipfire_src/gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip"
#include "hipfire_src/moe_down_combine_k8_batched.hip"
#include "hipfire_src/silu_mul.hip"

#define HIP_CHECK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(_e)); \
        std::abort(); \
    } \
} while (0)

namespace {

constexpr int    N_EXPERT      = 512;
constexpr int    K_TOP         = 8;    // fixed by the kernels' contract (misfit #1)
constexpr int    D_MODEL       = 2560; // gate/up K, and down's M (output rows)
constexpr int    N_FF          = 640;  // gate/up output rows (mi); down's REAL K
constexpr int    DOWN_K_PAD    = 768;  // down's padded K (misfit #2): 3 groups of 256
constexpr int    GROUP         = 256;
constexpr int    GROUP_BYTES   = 136;
constexpr int    GATEUP_M      = 2 * N_FF;               // 1280: gate rows 0..639, up rows 640..1279
constexpr size_t GATEUP_ROW_BYTES = (D_MODEL / GROUP) * GROUP_BYTES;     // 10 groups * 136 = 1360
constexpr size_t DOWN_ROW_BYTES   = (DOWN_K_PAD / GROUP) * GROUP_BYTES; // 3 groups * 136 = 408
constexpr size_t GATEUP_EXPERT_BYTES = (size_t)GATEUP_M * GATEUP_ROW_BYTES;   // 1,740,800
constexpr size_t DOWN_EXPERT_BYTES   = (size_t)D_MODEL  * DOWN_ROW_BYTES;    // 1,044,480
constexpr int    WARMUP_ITERS  = 100;
constexpr int    TIMED_ITERS   = 1000;

// ---- IEEE-754 binary16 <-> float32, used only on the host to build/check
// the MQ4G256V2 group headers (the device kernels use __half2float on the
// raw bits, which is the same format). Not performance-critical: this runs
// once per expert at setup and once per checked row in the numerics ref. ----
uint16_t f32_to_f16_bits(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t abse = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    if (abse == 0xFFu) { // inf/nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    int32_t exp = (int32_t)abse - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u); // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflow -> 0
        mant |= 0x800000u;
        int shift = 14 - exp; // 14..24
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) { half_mant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            int e = -1; uint32_t m = mant;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            m &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 - 15 - e);
            f = sign | (fexp << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t fexp = exp - 15 + 127;
        f = sign | (fexp << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, 4);
    return out;
}

// ---- MQ4G256V2 quantiser/dequantiser, host side. 136 B / 256-weight group:
// 2 x (f16 scale, f16 zero) -- one pair per 128-weight half -- then 128 B of
// packed nibbles (2/byte, low nibble first), covering weights 0..255 in
// order. See hipfire_src/gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip's
// header comment for the wire format and the DOG_X8/DOG macros in both
// kernel files for the exact decode this must invert. ----
void quantize_half128(const float *v, float &scale, float &zero, uint8_t nibbles[64]) {
    float lo = v[0], hi = v[0];
    for (int i = 1; i < 128; i++) { lo = std::min(lo, v[i]); hi = std::max(hi, v[i]); }
    if (hi <= lo) {
        // Degenerate group (e.g. the zero-padded down tail): encoder writes
        // scale=0, zero=0 (not `lo`, so an all-zero padded half stays 0 even
        // if `lo` were some nonzero constant elsewhere) and all nibbles 0.
        scale = 0.0f; zero = 0.0f;
        std::memset(nibbles, 0, 64);
        return;
    }
    scale = (hi - lo) / 15.0f;
    zero  = lo;
    for (int i = 0; i < 128; i += 2) {
        auto q_of = [&](float x) -> uint8_t {
            float qf = (x - zero) / scale;
            int q = (int)std::lround(qf);
            q = std::max(0, std::min(15, q));
            return (uint8_t)q;
        };
        uint8_t q0 = q_of(v[i]);
        uint8_t q1 = q_of(v[i + 1]);
        nibbles[i / 2] = (uint8_t)(q0 | (q1 << 4));
    }
}

// Quantise one row of `k` floats (k a multiple of 256) into MQ4G256V2,
// writing `(k/256)*136` bytes to `out`.
void quantize_row_mq4g256v2(const float *row, int k, uint8_t *out) {
    int groups = k / GROUP;
    for (int g = 0; g < groups; g++) {
        float sc0, zp0, sc1, zp1;
        uint8_t nib0[64], nib1[64];
        quantize_half128(row + g * GROUP, sc0, zp0, nib0);
        quantize_half128(row + g * GROUP + 128, sc1, zp1, nib1);
        uint8_t *gp = out + (size_t)g * GROUP_BYTES;
        uint16_t sc0b = f32_to_f16_bits(sc0), zp0b = f32_to_f16_bits(zp0);
        uint16_t sc1b = f32_to_f16_bits(sc1), zp1b = f32_to_f16_bits(zp1);
        std::memcpy(gp + 0, &sc0b, 2);
        std::memcpy(gp + 2, &zp0b, 2);
        std::memcpy(gp + 4, &sc1b, 2);
        std::memcpy(gp + 6, &zp1b, 2);
        std::memcpy(gp + 8, nib0, 64);
        std::memcpy(gp + 8 + 64, nib1, 64);
    }
}

// Dequantise one MQ4G256V2 row back to `k` floats -- inverts
// quantize_row_mq4g256v2 exactly the way the kernels' DOG macros do
// (value = scale*nibble + zero), for the CPU numerics reference.
void dequantize_row_mq4g256v2(const uint8_t *row_bytes, int k, float *out) {
    int groups = k / GROUP;
    for (int g = 0; g < groups; g++) {
        const uint8_t *gp = row_bytes + (size_t)g * GROUP_BYTES;
        uint16_t sc0b, zp0b, sc1b, zp1b;
        std::memcpy(&sc0b, gp + 0, 2);
        std::memcpy(&zp0b, gp + 2, 2);
        std::memcpy(&sc1b, gp + 4, 2);
        std::memcpy(&zp1b, gp + 6, 2);
        float sc0 = f16_bits_to_f32(sc0b), zp0 = f16_bits_to_f32(zp0b);
        float sc1 = f16_bits_to_f32(sc1b), zp1 = f16_bits_to_f32(zp1b);
        const uint8_t *nib = gp + 8;
        for (int i = 0; i < 128; i++) {
            uint8_t byte = nib[i / 2];
            uint8_t q = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            out[g * GROUP + i] = sc0 * (float)q + zp0;
        }
        for (int i = 0; i < 128; i++) {
            uint8_t byte = nib[64 + i / 2];
            uint8_t q = (i % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            out[g * GROUP + 128 + i] = sc1 * (float)q + zp1;
        }
    }
}

struct TimedStep {
    // Device buffers reused across warmup + timed iterations for one T.
    unsigned long long *d_expert_ptrs_gateup = nullptr;
    unsigned long long *d_expert_ptrs_down   = nullptr;
    int32_t *d_ids = nullptr;
    float   *d_weights = nullptr;
    float   *d_x = nullptr;
    float   *d_y_gate = nullptr;
    float   *d_y_up = nullptr;
    float   *d_h = nullptr;
    float   *d_rot_padded = nullptr;
    float   *d_expert_outputs = nullptr;
    float   *d_residual = nullptr;
    int T = 0;
};

void run_step(TimedStep &s) {
    dim3 block32(32, 1, 1);
    dim3 grid_gateup((unsigned)GATEUP_M, (unsigned)K_TOP, (unsigned)s.T);
    gemv_mq4g256v2_moe_gate_up_k8_indexed_batched<<<grid_gateup, block32>>>(
        s.d_expert_ptrs_gateup, s.d_ids, s.d_x, s.d_y_gate, s.d_y_up,
        GATEUP_M, D_MODEL, K_TOP);

    int n_h = s.T * K_TOP * N_FF;
    dim3 block_silu(256, 1, 1);
    dim3 grid_silu((unsigned)((n_h + 255) / 256), 1, 1);
    silu_mul_f32<<<grid_silu, block_silu>>>(s.d_y_gate, s.d_y_up, s.d_h, n_h);

    // Pad-overhead copy (misfit #2): h [T*K_TOP, 640] -> rot_padded
    // [T*K_TOP, 768] contiguous rows, tail 128 columns left at the zero
    // this buffer was memset to once at setup.
    HIP_CHECK(hipMemcpy2DAsync(
        s.d_rot_padded, (size_t)DOWN_K_PAD * sizeof(float),
        s.d_h, (size_t)N_FF * sizeof(float),
        (size_t)N_FF * sizeof(float), (size_t)s.T * K_TOP,
        hipMemcpyDeviceToDevice));

    dim3 grid_down((unsigned)D_MODEL, (unsigned)K_TOP, (unsigned)s.T);
    gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded<<<grid_down, block32>>>(
        s.d_expert_ptrs_down, s.d_ids, s.d_rot_padded, s.d_expert_outputs,
        D_MODEL, DOWN_K_PAD, K_TOP);

    HIP_CHECK(hipMemsetAsync(s.d_residual, 0, (size_t)s.T * D_MODEL * sizeof(float)));

    dim3 block_combine(256, 1, 1);
    dim3 grid_combine((unsigned)((D_MODEL + 255) / 256), (unsigned)s.T, 1);
    moe_down_combine_k8_batched<<<grid_combine, block_combine>>>(
        s.d_expert_outputs, s.d_weights, s.d_residual, D_MODEL, K_TOP);
}

} // namespace

int main() {
    // ---- Build 512 experts' weights on the host, upload once. ----
    std::vector<uint8_t> gateup_pool((size_t)N_EXPERT * GATEUP_EXPERT_BYTES);
    std::vector<uint8_t> down_pool((size_t)N_EXPERT * DOWN_EXPERT_BYTES);

    std::vector<float> row_buf(D_MODEL);
    std::vector<float> down_row_buf(DOWN_K_PAD, 0.0f); // tail 128 stay 0 (pad)

    for (int e = 0; e < N_EXPERT; e++) {
        uint8_t *gu = gateup_pool.data() + (size_t)e * GATEUP_EXPERT_BYTES;
        for (int r = 0; r < N_FF; r++) { // gate rows
            std::vector<float> row = m1::randf_vec(D_MODEL, -1.0f, 1.0f);
            quantize_row_mq4g256v2(row.data(), D_MODEL, gu + (size_t)r * GATEUP_ROW_BYTES);
        }
        for (int r = 0; r < N_FF; r++) { // up rows, at row index N_FF+r
            std::vector<float> row = m1::randf_vec(D_MODEL, -1.0f, 1.0f);
            quantize_row_mq4g256v2(row.data(), D_MODEL, gu + (size_t)(N_FF + r) * GATEUP_ROW_BYTES);
        }
        uint8_t *dn = down_pool.data() + (size_t)e * DOWN_EXPERT_BYTES;
        for (int r = 0; r < D_MODEL; r++) {
            std::vector<float> real = m1::randf_vec(N_FF, -1.0f, 1.0f);
            std::copy(real.begin(), real.end(), down_row_buf.begin());
            // down_row_buf[N_FF..DOWN_K_PAD) stays 0.0f from initialisation.
            quantize_row_mq4g256v2(down_row_buf.data(), DOWN_K_PAD, dn + (size_t)r * DOWN_ROW_BYTES);
        }
    }

    uint8_t *d_gateup_pool = nullptr, *d_down_pool = nullptr;
    HIP_CHECK(hipMalloc(&d_gateup_pool, gateup_pool.size()));
    HIP_CHECK(hipMalloc(&d_down_pool, down_pool.size()));
    HIP_CHECK(hipMemcpy(d_gateup_pool, gateup_pool.data(), gateup_pool.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_down_pool, down_pool.data(), down_pool.size(), hipMemcpyHostToDevice));

    std::vector<unsigned long long> ptrs_gateup_host(N_EXPERT), ptrs_down_host(N_EXPERT);
    for (int e = 0; e < N_EXPERT; e++) {
        ptrs_gateup_host[e] = (unsigned long long)(uintptr_t)(d_gateup_pool + (size_t)e * GATEUP_EXPERT_BYTES);
        ptrs_down_host[e]   = (unsigned long long)(uintptr_t)(d_down_pool   + (size_t)e * DOWN_EXPERT_BYTES);
    }
    unsigned long long *d_ptrs_gateup = nullptr, *d_ptrs_down = nullptr;
    HIP_CHECK(hipMalloc(&d_ptrs_gateup, N_EXPERT * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc(&d_ptrs_down, N_EXPERT * sizeof(unsigned long long)));
    HIP_CHECK(hipMemcpy(d_ptrs_gateup, ptrs_gateup_host.data(), N_EXPERT * sizeof(unsigned long long), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ptrs_down, ptrs_down_host.data(), N_EXPERT * sizeof(unsigned long long), hipMemcpyHostToDevice));

    std::printf("hipfire_k_top=%d\n", K_TOP);
    std::printf("hipfire_down_padded_k=%d\n", DOWN_K_PAD);
    std::printf("hipfire_down_pad_overhead=%.4f\n", (double)DOWN_K_PAD / N_FF - 1.0);

    size_t bytes_per_step = (size_t)K_TOP * (GATEUP_EXPERT_BYTES + DOWN_EXPERT_BYTES);
    std::printf("hipfire_bytes_per_step=%zu\n", bytes_per_step);

    bool any_check_fail = false;

    for (int T : {1, 32}) {
        TimedStep s;
        s.T = T;
        std::vector<int32_t> ids_host = m1::random_distinct_ids(N_EXPERT, K_TOP, T);
        std::vector<float> weights_host = m1::random_norm_weights(K_TOP, T);
        std::vector<float> x_host = m1::randf_vec((size_t)D_MODEL * T, -1.0f, 1.0f);

        s.d_expert_ptrs_gateup = d_ptrs_gateup;
        s.d_expert_ptrs_down   = d_ptrs_down;

        HIP_CHECK(hipMalloc(&s.d_ids, ids_host.size() * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&s.d_weights, weights_host.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_x, x_host.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_y_gate, (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_y_up, (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_h, (size_t)T * K_TOP * N_FF * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_rot_padded, (size_t)T * K_TOP * DOWN_K_PAD * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_expert_outputs, (size_t)T * K_TOP * D_MODEL * sizeof(float)));
        HIP_CHECK(hipMalloc(&s.d_residual, (size_t)T * D_MODEL * sizeof(float)));

        HIP_CHECK(hipMemcpy(s.d_ids, ids_host.data(), ids_host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(s.d_weights, weights_host.data(), weights_host.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(s.d_x, x_host.data(), x_host.size() * sizeof(float), hipMemcpyHostToDevice));
        // Zeroed once: only the [0,640) prefix of each 768-wide row is ever
        // written again (the pad-overhead copy in run_step), so the tail
        // 128 columns stay this initial zero for every iteration.
        HIP_CHECK(hipMemset(s.d_rot_padded, 0, (size_t)T * K_TOP * DOWN_K_PAD * sizeof(float)));

        // ---- One untimed verification pass (T=1 only), before warmup. ----
        if (T == 1) {
            run_step(s);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> gpu_out(D_MODEL);
            HIP_CHECK(hipMemcpy(gpu_out.data(), s.d_residual, D_MODEL * sizeof(float), hipMemcpyDeviceToHost));

            std::vector<float> ref_out(D_MODEL, 0.0f);
            std::vector<float> gate_row(D_MODEL), up_row(D_MODEL);
            std::vector<float> y_gate(N_FF), y_up(N_FF), h_pad(DOWN_K_PAD, 0.0f);
            std::vector<float> down_row(DOWN_K_PAD);
            for (int k = 0; k < K_TOP; k++) {
                int e = ids_host[k];
                const uint8_t *gu = gateup_pool.data() + (size_t)e * GATEUP_EXPERT_BYTES;
                const uint8_t *dn = down_pool.data() + (size_t)e * DOWN_EXPERT_BYTES;
                for (int r = 0; r < N_FF; r++) {
                    dequantize_row_mq4g256v2(gu + (size_t)r * GATEUP_ROW_BYTES, D_MODEL, gate_row.data());
                    double acc = 0.0;
                    for (int c = 0; c < D_MODEL; c++) acc += (double)gate_row[c] * x_host[c];
                    y_gate[r] = (float)acc;

                    dequantize_row_mq4g256v2(gu + (size_t)(N_FF + r) * GATEUP_ROW_BYTES, D_MODEL, up_row.data());
                    acc = 0.0;
                    for (int c = 0; c < D_MODEL; c++) acc += (double)up_row[c] * x_host[c];
                    y_up[r] = (float)acc;

                    float g = y_gate[r];
                    float silu = g / (1.0f + std::exp(-g));
                    h_pad[r] = silu * y_up[r];
                }
                // h_pad[N_FF..DOWN_K_PAD) stays 0 from initialisation.
                for (int r = 0; r < D_MODEL; r++) {
                    dequantize_row_mq4g256v2(dn + (size_t)r * DOWN_ROW_BYTES, DOWN_K_PAD, down_row.data());
                    double acc = 0.0;
                    for (int c = 0; c < DOWN_K_PAD; c++) acc += (double)down_row[c] * h_pad[c];
                    ref_out[r] += weights_host[k] * (float)acc;
                }
            }
            double cos = m1::cosine_similarity(gpu_out.data(), ref_out.data(), D_MODEL);
            double maxabs = m1::max_abs_diff(gpu_out.data(), ref_out.data(), D_MODEL);
            std::printf("hipfire_check_maxabs=%.6g\n", maxabs);
            std::printf("hipfire_check_cos=%.6g\n", cos);
            if (cos < 0.999) any_check_fail = true;
        }

        // ---- Timing: hipEvents around the whole step, 100 warmup + 1000. ----
        hipEvent_t ev_start, ev_stop;
        HIP_CHECK(hipEventCreate(&ev_start));
        HIP_CHECK(hipEventCreate(&ev_stop));

        for (int i = 0; i < WARMUP_ITERS; i++) run_step(s);
        HIP_CHECK(hipDeviceSynchronize());

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
        m1::TimingStats ts = m1::stats_from_samples_us(samples_us);
        const char *tag = (T == 1) ? "T1" : "T32";
        std::printf("hipfire_moe_%s_us_median=%.3f\n", tag, ts.median_us);
        std::printf("hipfire_moe_%s_us_p10=%.3f\n", tag, ts.p10_us);
        std::printf("hipfire_moe_%s_us_p90=%.3f\n", tag, ts.p90_us);
        if (T == 1) {
            double gbs = (double)bytes_per_step / (ts.median_us * 1e-6) / 1e9;
            std::printf("hipfire_gbs_T1=%.3f\n", gbs);
        } else {
            double gbs32 = (double)bytes_per_step * T / (ts.median_us * 1e-6) / 1e9;
            std::printf("hipfire_gbs_T32=%.3f\n", gbs32);
            std::printf("hipfire_gbs_T32_assumption=32x_independent_token_footprints_no_reuse_credited\n");
        }

        HIP_CHECK(hipEventDestroy(ev_start));
        HIP_CHECK(hipEventDestroy(ev_stop));

        (void)hipFree(s.d_ids); (void)hipFree(s.d_weights); (void)hipFree(s.d_x);
        (void)hipFree(s.d_y_gate); (void)hipFree(s.d_y_up); (void)hipFree(s.d_h);
        (void)hipFree(s.d_rot_padded); (void)hipFree(s.d_expert_outputs); (void)hipFree(s.d_residual);
    }

    (void)hipFree(d_ptrs_gateup); (void)hipFree(d_ptrs_down);
    (void)hipFree(d_gateup_pool); (void)hipFree(d_down_pool);

    if (any_check_fail) {
        std::fprintf(stderr, "CHECK_FAIL\n");
        return 1;
    }
    return 0;
}
