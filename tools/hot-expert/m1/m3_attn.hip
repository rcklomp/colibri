// tools/hot-expert/m1/m3_attn.hip — M3: QSA block-sparse attention + Gated
// DeltaNet decode-step microbenchmark for Qwen3.8-Flash-Next (qwen4exp), per
// tools/hot-expert/M3-BRIEF-2026-09-22.md.
//
// Shapes come from the model's own HF config.json
// (~/models/Qwen3.8-Flash-Next-FP8/config.json on the rig), NOT from the
// served GGUF: llama.cpp's conversion of this model carries no
// `attention.compress_ratios` key, and llama-model-loader.cpp's
// get_key_or_arr(..., required=false) just returns false and leaves
// hparams.dsv4_compress_ratios untouched when the key is absent — there is
// no per-architecture fallback for it (unlike is_recr_impl's
// full_attention_interval fallback a few lines below it in
// llama_model_qwen4exp::load_arch_hparams, qwen4exp.cpp:92-98). Per
// qwen4exp.cpp:718, `qsa = ... && hparams.dsv4_compress_ratios[il] > 0`, so
// an all-zero ratio array means the served llama.cpp build runs QSA's 12
// layers as DENSE full attention over the whole growing cache instead of
// the model's designed block-sparse form — consistent with F11-DEPTH's
// roughly-linear-in-depth 66 -> 165 ms/token growth. This file measures
// what the model's OWN design (QSA engaged) would cost, from:
//   indexer_compress_ratio = 4      -> qsa_ratio
//   indexer_budget         = 2048   -> qsa_top_k (tokens)
//   indexer_n_heads = 4, indexer_head_dim = 128, indexer_kv_heads = 1
//   full_attention_interval = 4  -> 12 of 48 layers are QSA, 36 are GDN
//   num_attention_heads = 24, head_dim = 256, num_key_value_heads = 2 (GQA 12:1)
//   linear_num_key_heads = 16, linear_num_value_heads = 48,
//   linear_key_head_dim = linear_value_head_dim = 128 (== ssm_d_state)
//
// Reference read: ~/src/llama-glm53/src/models/qwen4exp.cpp —
// build_qsa_top_k (~477: "the reference returns indexer_top_k +
// compress_ratio - 1, whole blocks plus the tail" — this synthetic bench
// always runs on a cache whose depth is a multiple of the ratio, so there is
// no partial tail block; the budget always selects a whole number of
// blocks), build_attn_qsa, build_layer_attn (707-789: GQA 24:2, IMRoPE
// skipped here as a synthetic/oracle-only test — rotation does not change
// this file's cosine check since both GPU and CPU reference skip it
// identically), build_layer_attn_linear (793-918, the GDN step: state
// reshape (head_v_dim, head_v_dim, num_v_heads, n_seqs) is exactly the
// "128 x 6144 per sequence" the brief names, 128*128*48 = 128*6144).
//
// Numerics discipline: every quantity a production kernel would read in a
// lossy cache format (Q8_0 K/V, bf16 pooled indexer keys) is quantised ONCE
// on the host into the exact byte layout the GPU kernel reads, and the CPU
// fp32 reference dequantises those SAME bytes before computing — so the
// check isolates "does the kernel's arithmetic match a plain summation",
// not "does quantisation lose precision" (out of scope here; the model's
// own indexer/attention quantisation error is a separate question this
// bench does not answer).
//
// Q8_0 layout (same simplified planar/SoA convention m5_bench.hip uses, and
// for the same reason: the exact byte format does not matter for a
// microbenchmark, only that GPU and CPU reference read the identical
// bytes): qs[token*heads*HEAD_DIM + head*HEAD_DIM + d] int8, one half scale
// per 32-element block, scale[token*heads*(HEAD_DIM/32) + head*(HEAD_DIM/32) + b].
//
// GDN step scope: this benchmarks the recurrent delta-rule state update and
// its output projection for ONE decode token, ONE layer — the part whose
// cost is constant per token regardless of depth. It does NOT model the
// conv1d/QKVZ projections build_layer_attn_linear also performs (those are
// ordinary matmuls, already covered by M1's routed-expert-kernel work and
// not depth-independent in the same sense). The recurrence implemented here
// (state S_h in R^[128x128] per head, v_pred = k^T S, delta = beta*(v -
// v_pred), S' = a*S + k (x) delta, out = q^T S') is a plain, dimensionally
// faithful gated delta rule, not a byte-for-byte port of ggml_ssm_scan's
// internals — the CPU reference implements the SAME formula, so the check
// validates the GPU kernel's arithmetic against it, not against the
// production kernel's exact numerics.
//
// This file is BUILT ONLY here (docker hipcc, no --device flags, no GPU
// access). It is not run by the agent that wrote it — the orchestrating
// session runs it later under the rig lock, GPUs idle, one engine at a time
// (CLAUDE.md, "One benchmark at a time").
//
// ---------------------------------------------------------------------------
// Second cut (2026-09-22): the three DEPTH kernels were replaced; the harness,
// its CPU references, its oracle semantics and its key=value/json output are
// unchanged, and the GDN kernel is untouched. The first cut measured (record
// §M3, 256k depth) indexer scan 281 us, top-k 2 070 us, attention 476 us,
// against a byte bound of ~90 / ~30 / ~20 us. What the three were doing:
//
//   idx_scan   one thread per pooled block, so each thread walked 1 KB of
//              bf16 with 2-byte scalar loads and neighbouring threads were
//              1 KB apart -- an uncoalesced read of a purely streaming array.
//              Now: one 32-lane sub-group per block, two 16-byte loads per
//              lane, the query hoisted into registers, one wave reduction.
//   topk       a HOST bisection: 40 rounds of (launch a counting kernel,
//              hipMemcpy the count back). Its 2 ms was 40 synchronisations,
//              not data. Now: one launch of an MSB-first radix select that
//              never leaves the device.
//   attn       three kernels with a 24 x 2 048 score matrix materialised in
//              VRAM between them, and the gathered K re-read once per QUERY
//              head (12x the bytes). Now: a two-launch split-K flash decode
//              whose workgroup holds all 12 query heads of a GQA group, with
//              the scores in LDS.
//
// Each kernel's own header comment carries its layout and why it is that
// layout. No measurement is claimed here: the numbers above are the first
// cut's, and the second cut had not been run when it was written.
//
// `--rotate N` (default 12) was added after the second cut ran: it reported
// 2 314 GB/s for the indexer scan at 262 144, which is three times the card's
// VRAM bandwidth, because a 67 MB array read a thousand times lives in the
// 96 MB Infinity Cache and a decode token never gets that luxury. See "cache
// rotation" below for what the option does and where its reach stops.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------- utility --

#define HIP_CHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", #cmd, __FILE__, __LINE__, hipGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

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

static void emit_stats_us(const std::string& name, std::vector<double>& samples_us) {
    emit_num(name + "_us", median_of(samples_us));
    emit_num(name + "_us_p10", percentile(samples_us, 0.10));
    emit_num(name + "_us_p90", percentile(samples_us, 0.90));
}

static float event_elapsed_us(hipEvent_t a, hipEvent_t b) {
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms * 1000.f;
}

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

// A zero output or cosine < 0.999 is a hard failure everywhere in this file
// (M3-BRIEF-2026-09-22.md, "Numerics"): print CHECK_FAIL <name> and exit 1,
// never a silent pass.
static void check_or_fail(const std::string& name, const float* gpu, const float* cpu, size_t n,
                           const std::string& json_path) {
    double dot = 0.0, na = 0.0, nb = 0.0, maxabs = 0.0;
    for (size_t i = 0; i < n; i++) {
        double a = gpu[i], b = cpu[i];
        dot += a * b; na += a * a; nb += b * b;
        maxabs = std::max(maxabs, std::fabs(a - b));
    }
    double cos = (na > 0.0 && nb > 0.0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    emit_num("check_" + name + "_maxabs", maxabs);
    emit_num("check_" + name + "_cos", cos);
    if (na <= 0.0 || cos < 0.999) {
        printf("CHECK_FAIL %s\n", name.c_str());
        if (!json_path.empty()) write_json(json_path);
        exit(1);
    }
}

// ------------------------------------------------------------- RNG / data --

static std::mt19937& rng() {
    static std::mt19937 gen(0xC0FFEE42u);
    return gen;
}
static float randf(float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng());
}
static std::vector<float> randf_vec(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    for (auto& x : v) x = randf(lo, hi);
    return v;
}

// bf16 <-> fp32, portable (no GPU-specific bf16 type needed): a bf16 value
// is exactly the top 16 bits of an IEEE-754 fp32, round-to-nearest-even.
__host__ __device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    uint32_t lsb = (c.u >> 16) & 1u;
    uint32_t bias = 0x7fffu + lsb;
    c.u += bias;
    return (uint16_t)(c.u >> 16);
}
__host__ __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    union { float f; uint32_t u; } c;
    c.u = ((uint32_t)b) << 16;
    return c.f;
}
__host__ __device__ __forceinline__ float bits_to_f32(uint32_t u) {
    union { float f; uint32_t u; } c;
    c.u = u;
    return c.f;
}
// The two bf16 values packed in one 32-bit word of a uint4 load, in memory
// order (little-endian: the LOW half is the lower element index). A bf16 is
// the top 16 bits of the fp32, so widening is a shift, not a conversion.
__device__ __forceinline__ float bf16_lo(uint32_t w) { return bits_to_f32(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return bits_to_f32(w & 0xffff0000u); }

// Q8_0: 32 elements/block, int8 qs + one fp16 scale/block.
__host__ __device__ __forceinline__ float dequant_q8_0(const int8_t* qs, const __half* scale, int d) {
    int b = d >> 5;
    return __half2float(scale[b]) * (float)qs[d];
}
static void quantize_q8_0_row(const float* x, int n, int8_t* qs_out, __half* scale_out) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        float amax = 0.f;
        for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        float scale = amax > 0.f ? amax / 127.0f : 1.0f;
        scale_out[b] = __float2half(scale);
        for (int i = 0; i < 32; i++) {
            int q = (int)std::lround(x[b * 32 + i] / scale);
            q = std::max(-127, std::min(127, q));
            qs_out[b * 32 + i] = (int8_t)q;
        }
    }
}

// --------------------------------------------------------------- shapes ---

#define QSA_RATIO         4     // indexer_compress_ratio
#define QSA_TOP_K_TOKENS  2048  // indexer_budget
#define N_SEL_BLOCKS      (QSA_TOP_K_TOKENS / QSA_RATIO)  // 512

#define IDX_N_HEADS  4
#define IDX_DIM      128

#define N_Q_HEADS    24
#define HEAD_DIM     256
#define N_KV_HEADS   2
#define GQA_GROUP    (N_Q_HEADS / N_KV_HEADS)  // 12

#define N_QSA_LAYERS 12
#define N_GDN_LAYERS 36

#define GDN_DIM      128   // ssm_d_state == linear_key_head_dim == linear_value_head_dim
#define GDN_V_HEADS  48    // linear_num_value_heads

static const int DEPTHS[] = {32768, 131072, 262144};

// --------------------------------------------------------- cache rotation --
//
// `--rotate N` (default below) allocates N INDEPENDENT device copies of every
// cache a depth kernel reads -- the pooled indexer keys, and the q8_0 K/V the
// attention gathers from -- and iteration i of every warm-up and timed loop
// reads copy i % N.
//
// It exists because the first run of these kernels reported 2 314 GB/s for the
// indexer scan at 262 144, which is three times this card's VRAM bandwidth and
// therefore not a VRAM number at all: the pooled-key array is 67 MB, the timed
// loop read the SAME 67 MB a thousand times, and 67 MB fits in the 96 MB
// Infinity Cache. A decode token does not get that. It streams the twelve QSA
// layers' caches (12 x 67 MB of pooled keys alone) plus several GB of weights
// between one visit to a given layer's cache and the next, so every one of
// those reads starts cold in IC. Rotating over N >= 12 copies reproduces that:
// by the time iteration i + N comes back to copy i, N-1 other copies have been
// streamed through the cache.
//
// The copies hold IDENTICAL data (one host-side generation, uploaded N times).
// That is deliberate and it is enough: caches are indexed by address, not by
// content, so N distinct allocations evict each other exactly as N distinct
// layers would -- and identical content keeps every copy valid against the one
// CPU reference, so the oracles are unchanged (they check copy 0).
//
// Its reach is not equal for the three kernels, and the difference matters
// when reading the numbers:
//   idx_scan  fully defeated: each iteration streams a whole 67 MB copy, so N
//             = 12 pushes 805 MB through the cache per cycle.
//   attn      only partly: the gather touches ~2.2 MB of its 285 MB copy, so
//             one cycle of 12 copies touches ~26 MB -- still inside a 96 MB
//             IC. Defeating it outright would need ~44 copies (12.5 GB). The
//             honest reading is that attn_us remains an upper bound on how
//             cached the gather is, and qsa_layer_us -- where the indexer
//             streams 805 MB per cycle -- is the one to trust.
//   topk      not rotated at all, on purpose: it reads the 256 KB score array
//             the indexer wrote microseconds earlier, which is exactly what
//             happens in service, cache and all.
//
// Memory at 262 144 with N = 12: 12 x 67 MB of pooled keys plus 12 x ~285 MB
// of K/V, about 4.3 GB on an empty 24 GB card. If a copy cannot be allocated
// the run does not die: it backs off to the copies it already has and emits
// the effective count as rotate_eff_n<depth>.
#define ROTATE_DEFAULT 12

// ------------------------------------------------------------- kernel 1: idx_scan --
//
// scores[b] = sum over 4 indexer heads of q_h . kpool_h[b] (bf16 pooled
// keys, f32 accumulate).
//
// idx_scan_method = "wave32_block_uint4". The whole scan is one streaming
// read of the pooled-key cache (1 KB per block, N/4 blocks, nothing else
// touched), so the only thing that matters is that the loads are wide and
// coalesced and that no LDS or scalar traffic sits in the inner loop:
//
//   * one 32-lane sub-group per pooled block; the block's 1 024 bytes are
//     exactly 64 uint4, i.e. TWO 16-byte loads per lane, and consecutive
//     lanes read consecutive 16-byte chunks (a fully coalesced 512-byte
//     request per load per sub-group; the 8 sub-groups of a 256-thread
//     workgroup cover 8 KB of contiguous cache).
//   * the 4 heads are contiguous in memory and the score is their SUM, so
//     the whole block is one flat 512-element dot product against the
//     512-element query -- no per-head bookkeeping at all.
//   * each lane's 16 query elements depend only on its lane id, never on
//     the block, so they are hoisted into registers ONCE per workgroup
//     (four float4 loads). Reading them from LDS instead would cost 16
//     8-way-bank-conflicting LDS reads per block, which on this shape is
//     more work than the arithmetic it feeds.
//   * one wave reduction (5 shuffles) per block, then a single f32 store.
//
// Numerics: the per-lane sum runs over a strided subset of the 512 terms
// and the wave reduction is a tree, so the summation ORDER differs from
// both the old kernel and the CPU reference (which accumulates in double).
// That is a reordering, not a loss -- the reference stays the oracle.

#define IDX_ELEMS       (IDX_N_HEADS * IDX_DIM)     // 512 bf16 per pooled block
#define IDX_U4_PER_BLK  (IDX_ELEMS * 2 / 16)        // 64 uint4 per pooled block
#define IDX_LANES       32
#define IDX_U4_PER_LANE (IDX_U4_PER_BLK / IDX_LANES) // 2
#define IDX_THREADS     256

__global__ __launch_bounds__(IDX_THREADS)
void k_idx_scan(const uint4* __restrict__ kpool4, const float* __restrict__ q_idx,
                float* __restrict__ scores, int n_blocks) {
    const int lane = threadIdx.x & (IDX_LANES - 1);
    const int sub  = threadIdx.x / IDX_LANES;
    const int nsub = IDX_THREADS / IDX_LANES;

    // This lane's 16 query elements: flat indices (lane + r*32)*8 + 0..7.
    const float4* q4 = reinterpret_cast<const float4*>(q_idx);
    float qr[IDX_U4_PER_LANE][8];
#pragma unroll
    for (int r = 0; r < IDX_U4_PER_LANE; r++) {
        const int e4 = (lane + r * IDX_LANES) * 2;   // float4 index of element base
        float4 a = q4[e4], b = q4[e4 + 1];
        qr[r][0] = a.x; qr[r][1] = a.y; qr[r][2] = a.z; qr[r][3] = a.w;
        qr[r][4] = b.x; qr[r][5] = b.y; qr[r][6] = b.z; qr[r][7] = b.w;
    }

    for (int b = blockIdx.x * nsub + sub; b < n_blocks; b += gridDim.x * nsub) {
        const uint4* src = kpool4 + (size_t)b * IDX_U4_PER_BLK;
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < IDX_U4_PER_LANE; r++) {
            uint4 v = src[lane + r * IDX_LANES];
            acc = fmaf(qr[r][0], bf16_lo(v.x), acc);
            acc = fmaf(qr[r][1], bf16_hi(v.x), acc);
            acc = fmaf(qr[r][2], bf16_lo(v.y), acc);
            acc = fmaf(qr[r][3], bf16_hi(v.y), acc);
            acc = fmaf(qr[r][4], bf16_lo(v.z), acc);
            acc = fmaf(qr[r][5], bf16_hi(v.z), acc);
            acc = fmaf(qr[r][6], bf16_lo(v.w), acc);
            acc = fmaf(qr[r][7], bf16_hi(v.w), acc);
        }
#pragma unroll
        for (int off = IDX_LANES / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, IDX_LANES);
        if (lane == 0) scores[b] = acc;
    }
}

// One launch of the indexer scan (grid capped so the grid-stride loop keeps
// every sub-group fed without an absurd grid at 256k).
static void run_idx_scan_once(const uint16_t* d_kpool, const float* d_q_idx,
                               float* d_scores, int n_blocks) {
    const int nsub = IDX_THREADS / IDX_LANES;
    int wgs = (n_blocks + nsub - 1) / nsub;
    if (wgs > 4096) wgs = 4096;
    if (wgs < 1) wgs = 1;
    hipLaunchKernelGGL(k_idx_scan, dim3(wgs), dim3(IDX_THREADS), 0, 0,
                        reinterpret_cast<const uint4*>(d_kpool), d_q_idx, d_scores, n_blocks);
}

static void idx_scan_cpu_ref(const std::vector<uint16_t>& kpool, const std::vector<float>& q_idx,
                              std::vector<float>& scores, int n_blocks) {
    scores.resize(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        double acc = 0.0;
        for (int h = 0; h < IDX_N_HEADS; h++) {
            const uint16_t* kb = &kpool[((size_t)b * IDX_N_HEADS + h) * IDX_DIM];
            const float* qh = &q_idx[h * IDX_DIM];
            double dot = 0.0;
            for (int d = 0; d < IDX_DIM; d++) dot += (double)qh[d] * bf16_to_f32(kb[d]);
            acc += dot;
        }
        scores[b] = (float)acc;
    }
}

// -------------------------------------------------------- kernel 2: topk --
//
// topk_method = "radix_select_11_7_7_7_1wg". ONE kernel launch, one
// workgroup of 1 024 threads, no host round trip anywhere -- that last part
// is half the point: the kernel it replaces bisected the score range from
// the HOST, 40 iterations each with a hipMemcpy of the count back, so its
// cost was 40 synchronisations (~2 ms) and had nothing to do with the data.
// A device-side select also lets the whole QSA layer be enqueued as a
// straight line of launches, which is what qsa_layer_us is supposed to
// measure.
//
// The select itself is an ordinary MSB-first radix select on the
// order-preserving uint32 image of the f32 score (negatives: flip all bits;
// non-negatives: set the sign bit -- after which unsigned order IS float
// order, ties included):
//
//   pass 0  histogram of the top 11 bits over all N/4 scores (2 048 LDS
//           bins), suffix-scan the bins from the largest key down, and take
//           the bin that contains the k-th largest. `above` = how many keys
//           are strictly above that bin, `need` = k - above.
//   pass 1  one more sweep of the scores: keys above the bin are SELECTED
//           outright (there are `above` < k of them); keys inside it are
//           appended to a candidate list (key, block index).
//   pass 2  three 7-bit refinement passes (bits 20..14, 13..7, 6..0) over
//           the candidate list only -- a few hundred entries in practice,
//           n_blocks in the pathological all-equal case -- which narrows
//           the prefix to the exact 32-bit key of the k-th largest.
//   pass 3  candidates above that key are selected; candidates EQUAL to it
//           are selected until `need` of them have been taken (an atomic
//           counter caps it), so exactly k blocks come out.
//
// Ties: with distinct scores the tie path takes exactly one element and the
// result is the unique top-k set. With duplicates at the cut, the k-th
// value's ties are taken in an arbitrary (atomic) order -- which is also
// what the CPU reference's std::partial_sort does, so neither side is a
// stricter oracle than the other. The harness's set-match check is exact
// for the distinct-score data it generates.
//
// Output is TOKEN indices, as before: block b contributes [b*ratio, b*ratio+ratio).
//
// Bounded-safe: the score sweeps are grid-strided with a scalar tail after
// the float4 body, every LDS scan runs over its full power-of-two width
// with all 1 024 threads reaching every __syncthreads, and both writers
// (`out` and `cand`) are capped.

#define TOPK_THREADS   1024
#define TOPK_HI_BITS   11
#define TOPK_HI_BINS   (1 << TOPK_HI_BITS)          // 2048
#define TOPK_HI_SHIFT  (32 - TOPK_HI_BITS)          // 21
#define TOPK_LO_BITS   7
#define TOPK_LO_BINS   (1 << TOPK_LO_BITS)          // 128
#define TOPK_LO_PASSES 3                            // 11 + 3*7 = 32 bits

static_assert(TOPK_HI_BINS == 2 * TOPK_THREADS, "hi-bin scan assumes 2 bins per thread");
static_assert(TOPK_LO_BITS * TOPK_LO_PASSES + TOPK_HI_BITS == 32, "radix passes must cover 32 bits");

// f32 -> uint32, order preserving (total order on finite floats).
__device__ __forceinline__ uint32_t ord_key(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    return (c.u & 0x80000000u) ? ~c.u : (c.u | 0x80000000u);
}

__global__ __launch_bounds__(TOPK_THREADS)
void k_topk_radix_select(const float* __restrict__ scores, int n_blocks, int k_blocks, int ratio,
                          uint32_t* __restrict__ cand_key, int* __restrict__ cand_idx,
                          int* __restrict__ out_tokens) {
    __shared__ uint32_t s_hist[TOPK_HI_BINS];
    __shared__ uint32_t s_scan[TOPK_THREADS];
    __shared__ int      s_found;
    __shared__ uint32_t s_above;
    __shared__ uint32_t s_prefix;
    __shared__ int      s_need;
    __shared__ int      s_cand;
    __shared__ int      s_out;
    __shared__ int      s_tie;

    const int t = threadIdx.x;
    const int k = (k_blocks < n_blocks) ? k_blocks : n_blocks;

    for (int i = t; i < TOPK_HI_BINS; i += TOPK_THREADS) s_hist[i] = 0u;
    if (t == 0) { s_cand = 0; s_out = 0; s_tie = 0; s_found = 0; s_above = 0u; }
    __syncthreads();

    // ---- pass 0: 11-bit histogram over every score ----
    const int n4 = n_blocks >> 2;
    const float4* s4 = reinterpret_cast<const float4*>(scores);
#pragma unroll 4
    for (int i = t; i < n4; i += TOPK_THREADS) {
        float4 v = s4[i];
        atomicAdd(&s_hist[ord_key(v.x) >> TOPK_HI_SHIFT], 1u);
        atomicAdd(&s_hist[ord_key(v.y) >> TOPK_HI_SHIFT], 1u);
        atomicAdd(&s_hist[ord_key(v.z) >> TOPK_HI_SHIFT], 1u);
        atomicAdd(&s_hist[ord_key(v.w) >> TOPK_HI_SHIFT], 1u);
    }
    for (int i = (n4 << 2) + t; i < n_blocks; i += TOPK_THREADS)
        atomicAdd(&s_hist[ord_key(scores[i]) >> TOPK_HI_SHIFT], 1u);
    __syncthreads();

    // ---- suffix scan (largest key first), 2 bins per thread ----
    s_scan[t] = s_hist[TOPK_HI_BINS - 1 - 2 * t] + s_hist[TOPK_HI_BINS - 2 - 2 * t];
    __syncthreads();
    for (int off = 1; off < TOPK_THREADS; off <<= 1) {
        uint32_t v = (t >= off) ? s_scan[t - off] : 0u;
        __syncthreads();
        s_scan[t] += v;
        __syncthreads();
    }
    {
        uint32_t incl = s_scan[t];
        uint32_t excl = (t == 0) ? 0u : s_scan[t - 1];
        if (incl >= (uint32_t)k && excl < (uint32_t)k) { s_found = t; s_above = excl; }
    }
    __syncthreads();
    if (t == 0) {
        uint32_t above = s_above;
        int bin = TOPK_HI_BINS - 1 - 2 * s_found;         // higher of the thread's two bins
        uint32_t c0 = s_hist[bin];
        if (above + c0 < (uint32_t)k) { above += c0; bin -= 1; }
        s_prefix = ((uint32_t)bin) << TOPK_HI_SHIFT;
        s_need   = k - (int)above;
    }
    __syncthreads();

    // ---- pass 1: select what is above the bin, collect what is inside it ----
    const uint32_t hi = s_prefix >> TOPK_HI_SHIFT;
#pragma unroll 4
    for (int i = t; i < n_blocks; i += TOPK_THREADS) {
        uint32_t u = ord_key(scores[i]);
        uint32_t b = u >> TOPK_HI_SHIFT;
        if (b > hi) {
            int pos = atomicAdd(&s_out, 1);
            if (pos < k)
                for (int r = 0; r < ratio; r++) out_tokens[pos * ratio + r] = i * ratio + r;
        } else if (b == hi) {
            int c = atomicAdd(&s_cand, 1);
            if (c < n_blocks) { cand_key[c] = u; cand_idx[c] = i; }
        }
    }
    __syncthreads();

    // ---- pass 2: three 7-bit refinements over the candidates ----
    for (int p = 0; p < TOPK_LO_PASSES; p++) {
        const int shift = TOPK_LO_BITS * (TOPK_LO_PASSES - 1 - p);   // 14, 7, 0
        const uint32_t himask = ~((1u << (shift + TOPK_LO_BITS)) - 1u);
        for (int i = t; i < TOPK_LO_BINS; i += TOPK_THREADS) s_hist[i] = 0u;
        __syncthreads();
        const int ncand = s_cand;
        const uint32_t pref = s_prefix;
        for (int i = t; i < ncand; i += TOPK_THREADS) {
            uint32_t u = cand_key[i];
            if ((u & himask) == (pref & himask))
                atomicAdd(&s_hist[(u >> shift) & (TOPK_LO_BINS - 1)], 1u);
        }
        __syncthreads();
        if (t < TOPK_LO_BINS) s_scan[t] = s_hist[TOPK_LO_BINS - 1 - t];
        __syncthreads();
        for (int off = 1; off < TOPK_LO_BINS; off <<= 1) {
            uint32_t v = (t < TOPK_LO_BINS && t >= off) ? s_scan[t - off] : 0u;
            __syncthreads();
            if (t < TOPK_LO_BINS) s_scan[t] += v;
            __syncthreads();
        }
        if (t < TOPK_LO_BINS) {
            uint32_t incl = s_scan[t];
            uint32_t excl = (t == 0) ? 0u : s_scan[t - 1];
            if (incl >= (uint32_t)s_need && excl < (uint32_t)s_need) { s_found = t; s_above = excl; }
        }
        __syncthreads();
        if (t == 0) {
            int bin = TOPK_LO_BINS - 1 - s_found;
            s_prefix |= ((uint32_t)bin) << shift;
            s_need   -= (int)s_above;
        }
        __syncthreads();
    }

    // ---- pass 3: final selection among the candidates ----
    {
        const uint32_t thr = s_prefix;
        const int need = s_need;
        const int ncand = s_cand;
        for (int i = t; i < ncand; i += TOPK_THREADS) {
            uint32_t u = cand_key[i];
            bool take = (u > thr);
            if (!take && u == thr) take = (atomicAdd(&s_tie, 1) < need);
            if (take) {
                int pos = atomicAdd(&s_out, 1);
                if (pos < k) {
                    int b = cand_idx[i];
                    for (int r = 0; r < ratio; r++) out_tokens[pos * ratio + r] = b * ratio + r;
                }
            }
        }
    }
}

// One launch, nothing else: no memset, no host synchronisation. Every piece
// of state the kernel carries is LDS, initialised on entry, so back-to-back
// invocations in the timing loop are independent.
static void run_topk_once(const float* d_scores, int n_blocks, uint32_t* d_cand_key,
                           int* d_cand_idx, int* d_out_tokens) {
    hipLaunchKernelGGL(k_topk_radix_select, dim3(1), dim3(TOPK_THREADS), 0, 0,
                        d_scores, n_blocks, N_SEL_BLOCKS, QSA_RATIO,
                        d_cand_key, d_cand_idx, d_out_tokens);
}

// CPU reference: exact partial_sort on the block scores, taking the top
// N_SEL_BLOCKS by score (ties broken by index, though with continuous
// random fp32 data ties are not expected in practice).
static void topk_cpu_ref(const std::vector<float>& scores, int n_blocks, std::vector<int>& sel_tokens_sorted) {
    std::vector<int> idx(n_blocks);
    for (int i = 0; i < n_blocks; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + N_SEL_BLOCKS, idx.end(),
                       [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(N_SEL_BLOCKS);
    sel_tokens_sorted.clear();
    sel_tokens_sorted.reserve(QSA_TOP_K_TOKENS);
    for (int b : idx) {
        for (int r = 0; r < QSA_RATIO; r++) sel_tokens_sorted.push_back(b * QSA_RATIO + r);
    }
    std::sort(sel_tokens_sorted.begin(), sel_tokens_sorted.end());
}

// -------------------------------------------------------- kernel 3: attn --
//
// attn_method = "flash_split_wave_per_head". Two launches: a split-K
// flash-decode pass over chunks of the selected token list, then a combine.
//
// The shape the kernel is built around: 2 048 selected tokens x 2 KV heads
// x 256 dims x (K and V) in q8_0 is 2.2 MB, and 24 query heads read it --
// so the ONLY thing that matters is that the gathered K/V bytes cross the
// memory system once per KV head, and that enough workgroups are in flight
// to keep 2.2 MB moving. The kernel it replaces re-read the whole gathered
// K once per QUERY head (12x the traffic), materialised a 24 x 2 048 score
// matrix in VRAM, and re-read it in a third kernel.
//
//   * grid is (chunks of 64 selected tokens) x (2 KV heads); a workgroup is
//     12 waves, one per query head of the GQA group, so the 12 heads that
//     share a KV head consume the same gathered K/V from the same
//     workgroup -- one VRAM trip, the other 11 reads served by L1.
//   * a lane owns 8 of the 256 dims, so each gathered K (or V) row is one
//     8-byte load per lane, 256 contiguous bytes per wave, and the q8_0
//     scale for those 8 dims is a single half (8 dims never straddle a
//     32-element quant block). Four tokens are loaded per iteration so the
//     loads are in flight while the previous four are reduced.
//   * scores stay in LDS (12 x 64 floats), never in VRAM. Softmax is
//     flash-style: each chunk emits its own max and sum next to its partial
//     output, and the combine kernel rescales -- so nothing of size
//     n_sel is ever materialised.
//   * the V accumulation needs no cross-lane reduction at all: the weight
//     is an LDS broadcast and each lane owns its own 8 output dims.
//
// Numerics are the reference's, reassociated: score = (sum_d q_d * s*qs_d)
// * rsqrt(256) with a tree reduction over lanes, softmax over the same set
// of keys via the flash identity (sum_c e^(m_c - M) * partial_c), and the
// same sigmoid gate on the f32 result. No fp16 accumulation anywhere: the
// q8_0 scale is widened to f32 before it is used.

#define ATTN_CHUNK   64                              // selected tokens per workgroup
#define ATTN_LANES   32
#define ATTN_DPL     (HEAD_DIM / ATTN_LANES)         // 8 dims per lane
#define ATTN_WAVES   GQA_GROUP                       // 12 query heads share a KV head
#define ATTN_THREADS (ATTN_WAVES * ATTN_LANES)       // 384
#define ATTN_UNR     4                               // tokens loaded per iteration
#define ATTN_NEG_INF (-3.0e38f)

static_assert(ATTN_CHUNK % ATTN_UNR == 0, "chunk must divide into unrolled groups");
static_assert(HEAD_DIM % 32 == 0, "q8_0 blocks must tile the head dim");

// 8 signed bytes packed in two 32-bit words, scaled to f32.
__device__ __forceinline__ void q8_bytes(int lo, int hi, float sc, float* out8) {
#pragma unroll
    for (int i = 0; i < 4; i++) out8[i]     = sc * (float)(int8_t)((lo >> (8 * i)) & 0xff);
#pragma unroll
    for (int i = 0; i < 4; i++) out8[4 + i] = sc * (float)(int8_t)((hi >> (8 * i)) & 0xff);
}

__global__ __launch_bounds__(ATTN_THREADS)
void k_attn_flash_split(const int8_t* __restrict__ Kqs, const __half* __restrict__ Kscale,
                         const int8_t* __restrict__ Vqs, const __half* __restrict__ Vscale,
                         const float* __restrict__ Q, const int* __restrict__ sel_tokens,
                         float* __restrict__ partial, float* __restrict__ mbuf,
                         float* __restrict__ lbuf, int n_sel_tokens) {
    __shared__ float sh_p[ATTN_WAVES][ATTN_CHUNK];
    __shared__ int   sh_tok[ATTN_CHUNK];

    const int chunk = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int w     = threadIdx.x / ATTN_LANES;          // query head within the group
    const int lane  = threadIdx.x % ATTN_LANES;
    const int h     = kvh * GQA_GROUP + w;               // global query head
    const int base  = chunk * ATTN_CHUNK;
    const int d0    = lane * ATTN_DPL;

    for (int j = threadIdx.x; j < ATTN_CHUNK; j += ATTN_THREADS) {
        int jj = base + j;
        sh_tok[j] = (jj < n_sel_tokens) ? sel_tokens[jj] : 0;   // token 0: a safe dummy
    }
    __syncthreads();

    // this lane's 8 query elements, once
    float qr[ATTN_DPL];
    {
        const float4* q4 = reinterpret_cast<const float4*>(Q + (size_t)h * HEAD_DIM + d0);
        float4 a = q4[0], b = q4[1];
        qr[0] = a.x; qr[1] = a.y; qr[2] = a.z; qr[3] = a.w;
        qr[4] = b.x; qr[5] = b.y; qr[6] = b.z; qr[7] = b.w;
    }
    const float qscale = rsqrtf((float)HEAD_DIM);
    const int sc_off = d0 >> 5;                          // which q8_0 scale covers these 8 dims

    // ---- phase A: scores for this head over the chunk ----
    for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR) {
        int2  raw[ATTN_UNR];
        float sc[ATTN_UNR];
#pragma unroll
        for (int u = 0; u < ATTN_UNR; u++) {
            size_t row = ((size_t)sh_tok[j0 + u] * N_KV_HEADS + kvh);
            raw[u] = *reinterpret_cast<const int2*>(Kqs + row * HEAD_DIM + d0);
            sc[u]  = __half2float(Kscale[row * (HEAD_DIM / 32) + sc_off]);
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNR; u++) {
            float kd[ATTN_DPL];
            q8_bytes(raw[u].x, raw[u].y, sc[u], kd);
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < ATTN_DPL; i++) acc = fmaf(qr[i], kd[i], acc);
#pragma unroll
            for (int off = ATTN_LANES / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, ATTN_LANES);
            if (lane == 0)
                sh_p[w][j0 + u] = (base + j0 + u < n_sel_tokens) ? acc * qscale : ATTN_NEG_INF;
        }
    }
    __syncthreads();

    // ---- phase A2: chunk-local softmax of this head's row ----
    float mx = ATTN_NEG_INF, sum = 0.f;
    for (int j = lane; j < ATTN_CHUNK; j += ATTN_LANES) mx = fmaxf(mx, sh_p[w][j]);
#pragma unroll
    for (int off = ATTN_LANES / 2; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, ATTN_LANES));
    for (int j = lane; j < ATTN_CHUNK; j += ATTN_LANES) {
        float s = sh_p[w][j];
        float e = (s <= ATTN_NEG_INF) ? 0.f : expf(s - mx);
        sh_p[w][j] = e;
        sum += e;
    }
#pragma unroll
    for (int off = ATTN_LANES / 2; off > 0; off >>= 1) sum += __shfl_xor(sum, off, ATTN_LANES);
    if (lane == 0) {
        mbuf[(size_t)chunk * N_Q_HEADS + h] = mx;
        lbuf[(size_t)chunk * N_Q_HEADS + h] = sum;
    }
    __syncthreads();

    // ---- phase B: weighted V sum for this head's 8 dims per lane ----
    float acc[ATTN_DPL];
#pragma unroll
    for (int i = 0; i < ATTN_DPL; i++) acc[i] = 0.f;
    for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR) {
        int2  raw[ATTN_UNR];
        float sc[ATTN_UNR];
#pragma unroll
        for (int u = 0; u < ATTN_UNR; u++) {
            size_t row = ((size_t)sh_tok[j0 + u] * N_KV_HEADS + kvh);
            raw[u] = *reinterpret_cast<const int2*>(Vqs + row * HEAD_DIM + d0);
            sc[u]  = __half2float(Vscale[row * (HEAD_DIM / 32) + sc_off]);
        }
#pragma unroll
        for (int u = 0; u < ATTN_UNR; u++) {
            float vd[ATTN_DPL];
            q8_bytes(raw[u].x, raw[u].y, sc[u], vd);
            float p = sh_p[w][j0 + u];                   // LDS broadcast, zero for a padded slot
#pragma unroll
            for (int i = 0; i < ATTN_DPL; i++) acc[i] = fmaf(p, vd[i], acc[i]);
        }
    }
    float* dst = partial + ((size_t)chunk * N_Q_HEADS + h) * HEAD_DIM + d0;
    reinterpret_cast<float4*>(dst)[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    reinterpret_cast<float4*>(dst)[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

// Rescale the per-chunk partials onto one global max, normalise, gate.
__global__ __launch_bounds__(HEAD_DIM)
void k_attn_combine(const float* __restrict__ partial, const float* __restrict__ mbuf,
                     const float* __restrict__ lbuf, const float* __restrict__ gate,
                     float* __restrict__ out, int n_chunks) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float M = ATTN_NEG_INF;
    for (int c = 0; c < n_chunks; c++) M = fmaxf(M, mbuf[(size_t)c * N_Q_HEADS + h]);
    float acc = 0.f, denom = 0.f;
    for (int c = 0; c < n_chunks; c++) {
        float m = mbuf[(size_t)c * N_Q_HEADS + h];
        if (m <= ATTN_NEG_INF) continue;
        float s = expf(m - M);
        denom += s * lbuf[(size_t)c * N_Q_HEADS + h];
        acc   += s * partial[((size_t)c * N_Q_HEADS + h) * HEAD_DIM + d];
    }
    float g = 1.0f / (1.0f + expf(-gate[(size_t)h * HEAD_DIM + d]));
    out[(size_t)h * HEAD_DIM + d] = (denom > 0.f ? acc / denom : 0.f) * g;
}

static int attn_chunks(int n_sel_tokens) { return (n_sel_tokens + ATTN_CHUNK - 1) / ATTN_CHUNK; }

static void run_attn_once(const int8_t* d_Kqs, const __half* d_Ksc, const int8_t* d_Vqs,
                           const __half* d_Vsc, const float* d_Q, const int* d_sel,
                           const float* d_gate, float* d_partial, float* d_m, float* d_l,
                           float* d_out, int n_sel_tokens) {
    int n_chunks = attn_chunks(n_sel_tokens);
    hipLaunchKernelGGL(k_attn_flash_split, dim3(n_chunks, N_KV_HEADS), dim3(ATTN_THREADS), 0, 0,
                        d_Kqs, d_Ksc, d_Vqs, d_Vsc, d_Q, d_sel, d_partial, d_m, d_l, n_sel_tokens);
    hipLaunchKernelGGL(k_attn_combine, dim3(N_Q_HEADS), dim3(HEAD_DIM), 0, 0,
                        d_partial, d_m, d_l, d_gate, d_out, n_chunks);
}

static void attn_cpu_ref(const std::vector<int8_t>& Kqs, const std::vector<__half>& Kscale,
                          const std::vector<int8_t>& Vqs, const std::vector<__half>& Vscale,
                          const std::vector<float>& Q, const std::vector<float>& gate,
                          const std::vector<int>& sel_tokens, std::vector<float>& out) {
    int n_sel = (int)sel_tokens.size();
    std::vector<float> scores((size_t)N_Q_HEADS * n_sel);
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kvh = h / GQA_GROUP;
        for (int j = 0; j < n_sel; j++) {
            int tok = sel_tokens[j];
            const int8_t* qs = &Kqs[((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM];
            const __half* sc = &Kscale[((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32)];
            const float* q = &Q[(size_t)h * HEAD_DIM];
            double acc = 0.0;
            for (int d = 0; d < HEAD_DIM; d++) acc += (double)q[d] * dequant_q8_0(qs, sc, d);
            scores[(size_t)h * n_sel + j] = (float)(acc / std::sqrt((double)HEAD_DIM));
        }
        float* row = &scores[(size_t)h * n_sel];
        float maxv = *std::max_element(row, row + n_sel);
        double sum = 0.0;
        for (int j = 0; j < n_sel; j++) { row[j] = std::exp(row[j] - maxv); sum += row[j]; }
        for (int j = 0; j < n_sel; j++) row[j] = (float)(row[j] / sum);
    }
    out.assign((size_t)N_Q_HEADS * HEAD_DIM, 0.f);
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kvh = h / GQA_GROUP;
        const float* w = &scores[(size_t)h * n_sel];
        for (int d = 0; d < HEAD_DIM; d++) {
            double acc = 0.0;
            for (int j = 0; j < n_sel; j++) {
                int tok = sel_tokens[j];
                const int8_t* qs = &Vqs[((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM];
                const __half* sc = &Vscale[((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32)];
                acc += (double)w[j] * dequant_q8_0(qs, sc, d);
            }
            double g = 1.0 / (1.0 + std::exp(-(double)gate[(size_t)h * HEAD_DIM + d]));
            out[(size_t)h * HEAD_DIM + d] = (float)(acc * g);
        }
    }
}

// --------------------------------------------------------- kernel 4: GDN --
//
// One head per block, one thread per state column d (0..127). Gated delta
// rule: v_pred = k^T S, delta = beta*(v - v_pred), S' = a*S + k(x)delta,
// out = q^T S'. See the file header for scope (this is the recurrent step
// only, not the QKVZ/conv projections around it).

__global__ void k_gdn_step(const float* __restrict__ S_in, const float* __restrict__ q,
                            const float* __restrict__ k, const float* __restrict__ v,
                            const float* __restrict__ a, const float* __restrict__ beta,
                            float* __restrict__ S_out, float* __restrict__ out) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    __shared__ float sh_delta[GDN_DIM];
    const float* Sh = S_in + (size_t)h * GDN_DIM * GDN_DIM;
    float* Sh_out = S_out + (size_t)h * GDN_DIM * GDN_DIM;
    const float* qh = q + (size_t)h * GDN_DIM;
    const float* kh = k + (size_t)h * GDN_DIM;
    const float* vh = v + (size_t)h * GDN_DIM;
    float ah = a[h], bh = beta[h];

    float vpred = 0.f;
    for (int i = 0; i < GDN_DIM; i++) vpred += Sh[i * GDN_DIM + d] * kh[i];
    float delta = bh * (vh[d] - vpred);
    sh_delta[d] = delta;
    __syncthreads();

    for (int i = 0; i < GDN_DIM; i++) Sh_out[i * GDN_DIM + d] = ah * Sh[i * GDN_DIM + d] + kh[i] * sh_delta[d];
    __syncthreads();

    float o = 0.f;
    for (int i = 0; i < GDN_DIM; i++) o += Sh_out[i * GDN_DIM + d] * qh[i];
    out[h * GDN_DIM + d] = o;
}

static void gdn_cpu_ref(const std::vector<float>& S_in, const std::vector<float>& q,
                         const std::vector<float>& k, const std::vector<float>& v,
                         const std::vector<float>& a, const std::vector<float>& beta,
                         std::vector<float>& out) {
    out.assign((size_t)GDN_V_HEADS * GDN_DIM, 0.f);
    std::vector<double> Sh_new(GDN_DIM * GDN_DIM);
    for (int h = 0; h < GDN_V_HEADS; h++) {
        const float* Sh = &S_in[(size_t)h * GDN_DIM * GDN_DIM];
        const float* qh = &q[(size_t)h * GDN_DIM];
        const float* kh = &k[(size_t)h * GDN_DIM];
        const float* vh = &v[(size_t)h * GDN_DIM];
        double ah = a[h], bh = beta[h];
        std::vector<double> vpred(GDN_DIM, 0.0), delta(GDN_DIM, 0.0);
        for (int d = 0; d < GDN_DIM; d++) {
            double acc = 0.0;
            for (int i = 0; i < GDN_DIM; i++) acc += (double)Sh[i * GDN_DIM + d] * kh[i];
            vpred[d] = acc;
            delta[d] = bh * ((double)vh[d] - acc);
        }
        for (int i = 0; i < GDN_DIM; i++)
            for (int d = 0; d < GDN_DIM; d++)
                Sh_new[i * GDN_DIM + d] = ah * Sh[i * GDN_DIM + d] + (double)kh[i] * delta[d];
        for (int d = 0; d < GDN_DIM; d++) {
            double acc = 0.0;
            for (int i = 0; i < GDN_DIM; i++) acc += Sh_new[i * GDN_DIM + d] * (double)qh[i];
            out[(size_t)h * GDN_DIM + d] = (float)acc;
        }
    }
}

// ------------------------------------------------------------------ main --

int main(int argc, char** argv) {
    std::string json_path;
    int rotate_req = ROTATE_DEFAULT;
    for (int i = 1; i < argc; i++) {
        std::string a(argv[i]);
        if (a == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (a == "--rotate" && i + 1 < argc) rotate_req = atoi(argv[++i]);
    }
    if (rotate_req < 1) rotate_req = 1;

    HIP_CHECK(hipSetDevice(0));

    emit_int("rotate", rotate_req);
    emit_str("idx_scan_method", "wave32_block_uint4");
    emit_str("topk_method", "radix_select_11_7_7_7_1wg");
    emit_str("attn_method", "flash_split_wave_per_head");
    emit_int("qsa_ratio", QSA_RATIO);
    printf("qsa_ratio=%d\n", QSA_RATIO);
    emit_int("qsa_top_k", QSA_TOP_K_TOKENS);
    printf("qsa_top_k=%d\n", QSA_TOP_K_TOKENS);

    // Per the brief's own formula: "2 x 2 x 256 q8_0 + 128 bf16 ... times
    // 12 layers" -- K+V bytes/token/layer in Q8_0 (2 heads x 256 dims each,
    // 34/32 bytes/elem: one fp16 scale + 32 int8 per 32-elem block) plus
    // the pooled indexer key's 128 dims charged at bf16 (2 bytes/elem) per
    // token, even though one pooled key actually amortises over `ratio`
    // tokens in the real cache -- this is the brief's literal, and
    // conservative (larger), reading of "128 bf16 ... per token".
    const double Q8_0_BYTES_PER_ELEM = 34.0 / 32.0;
    const double kv_bytes_per_token =
        (2.0 * N_KV_HEADS * HEAD_DIM * Q8_0_BYTES_PER_ELEM + (double)IDX_DIM * 2.0) * N_QSA_LAYERS;
    emit_num("kv_bytes_per_token", kv_bytes_per_token);
    printf("kv_bytes_per_token=%.2f (M0d service measurement: 17.9 KB/token)\n", kv_bytes_per_token);

    const int WARMUP = 100, ITERS = 1000;

    for (int di = 0; di < 3; di++) {
        int N = DEPTHS[di];
        std::string tag = "_n" + std::to_string(N);
        int n_blocks = N / QSA_RATIO;
        printf("# --- depth=%d n_blocks=%d ---\n", N, n_blocks);
        emit_int("depth" + tag, N);

        // ---- synthetic data, host side ----
        std::vector<float> q_idx_f = randf_vec((size_t)IDX_N_HEADS * IDX_DIM);
        std::vector<float> kpool_f = randf_vec((size_t)n_blocks * IDX_N_HEADS * IDX_DIM);
        std::vector<uint16_t> kpool_bf16(kpool_f.size());
        for (size_t i = 0; i < kpool_f.size(); i++) kpool_bf16[i] = f32_to_bf16(kpool_f[i]);
        // CPU reference reads the SAME bf16-rounded values back (not the
        // pre-rounding fp32), so the check isolates kernel arithmetic.
        std::vector<uint16_t> kpool_for_ref = kpool_bf16;

        std::vector<float> Q_attn_f = randf_vec((size_t)N_Q_HEADS * HEAD_DIM);
        std::vector<float> gate_f = randf_vec((size_t)N_Q_HEADS * HEAD_DIM);

        std::vector<float> K_f = randf_vec((size_t)N * N_KV_HEADS * HEAD_DIM);
        std::vector<float> V_f = randf_vec((size_t)N * N_KV_HEADS * HEAD_DIM);
        std::vector<int8_t> K_qs((size_t)N * N_KV_HEADS * HEAD_DIM);
        std::vector<__half> K_sc((size_t)N * N_KV_HEADS * (HEAD_DIM / 32));
        std::vector<int8_t> V_qs((size_t)N * N_KV_HEADS * HEAD_DIM);
        std::vector<__half> V_sc((size_t)N * N_KV_HEADS * (HEAD_DIM / 32));
        for (int t = 0; t < N; t++) {
            for (int h = 0; h < N_KV_HEADS; h++) {
                size_t off_e = ((size_t)t * N_KV_HEADS + h) * HEAD_DIM;
                size_t off_s = ((size_t)t * N_KV_HEADS + h) * (HEAD_DIM / 32);
                quantize_q8_0_row(&K_f[off_e], HEAD_DIM, &K_qs[off_e], &K_sc[off_s]);
                quantize_q8_0_row(&V_f[off_e], HEAD_DIM, &V_qs[off_e], &V_sc[off_s]);
            }
        }

        // ---- device buffers ----
        float* d_q_idx; float* d_scores;
        HIP_CHECK(hipMalloc(&d_q_idx, q_idx_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scores, (size_t)n_blocks * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_q_idx, q_idx_f.data(), q_idx_f.size() * sizeof(float), hipMemcpyHostToDevice));

        // Radix-select scratch: worst case (every score in the same 11-bit
        // bucket) every block is a candidate, so both arrays are n_blocks
        // long and the kernel can never run off the end of them.
        int* d_sel_tokens; uint32_t* d_cand_key; int* d_cand_idx;
        HIP_CHECK(hipMalloc(&d_sel_tokens, QSA_TOP_K_TOKENS * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_cand_key, (size_t)n_blocks * sizeof(uint32_t)));
        HIP_CHECK(hipMalloc(&d_cand_idx, (size_t)n_blocks * sizeof(int)));

        // ---- the rotating copies of everything a depth kernel READS ----
        // (see "cache rotation" above). Identical bytes, N distinct
        // allocations; copy 0 is the one every oracle checks. A copy that
        // cannot be allocated is not fatal: we keep what we have, back the
        // rotation off, and say so in rotate_eff.
        std::vector<uint16_t*> r_kpool;
        std::vector<int8_t*>   r_Kqs, r_Vqs;
        std::vector<__half*>   r_Ksc, r_Vsc;
        for (int c = 0; c < rotate_req; c++) {
            uint16_t* p_kpool = nullptr; int8_t* p_Kqs = nullptr; int8_t* p_Vqs = nullptr;
            __half* p_Ksc = nullptr; __half* p_Vsc = nullptr;
            bool ok = hipMalloc(&p_kpool, kpool_bf16.size() * sizeof(uint16_t)) == hipSuccess
                   && hipMalloc(&p_Kqs, K_qs.size()) == hipSuccess
                   && hipMalloc(&p_Vqs, V_qs.size()) == hipSuccess
                   && hipMalloc(&p_Ksc, K_sc.size() * sizeof(__half)) == hipSuccess
                   && hipMalloc(&p_Vsc, V_sc.size() * sizeof(__half)) == hipSuccess;
            if (!ok) {
                if (p_kpool) (void)hipFree(p_kpool);
                if (p_Kqs) (void)hipFree(p_Kqs);
                if (p_Vqs) (void)hipFree(p_Vqs);
                if (p_Ksc) (void)hipFree(p_Ksc);
                if (p_Vsc) (void)hipFree(p_Vsc);
                (void)hipGetLastError();     // an OOM here is handled, not inherited
                if (c == 0) {
                    fprintf(stderr, "could not allocate even one cache copy at depth %d\n", N);
                    if (!json_path.empty()) write_json(json_path);
                    return 1;
                }
                fprintf(stderr, "rotate: only %d of %d copies fit at depth %d\n", c, rotate_req, N);
                break;
            }
            HIP_CHECK(hipMemcpy(p_kpool, kpool_bf16.data(), kpool_bf16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p_Kqs, K_qs.data(), K_qs.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p_Vqs, V_qs.data(), V_qs.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p_Ksc, K_sc.data(), K_sc.size() * sizeof(__half), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p_Vsc, V_sc.data(), V_sc.size() * sizeof(__half), hipMemcpyHostToDevice));
            r_kpool.push_back(p_kpool); r_Kqs.push_back(p_Kqs); r_Vqs.push_back(p_Vqs);
            r_Ksc.push_back(p_Ksc); r_Vsc.push_back(p_Vsc);
        }
        const int ROT = (int)r_kpool.size();
        emit_int("rotate_eff" + tag, ROT);
        // Iteration i reads copy i % ROT everywhere below.
#define ROT_IDX(i) ((i) % ROT)

        const int n_attn_chunks = attn_chunks(QSA_TOP_K_TOKENS);
        float *d_Q_attn, *d_gate, *d_attn_partial, *d_attn_m, *d_attn_l, *d_attn_out;
        HIP_CHECK(hipMalloc(&d_Q_attn, Q_attn_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_gate, gate_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_partial, (size_t)n_attn_chunks * N_Q_HEADS * HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_m, (size_t)n_attn_chunks * N_Q_HEADS * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_l, (size_t)n_attn_chunks * N_Q_HEADS * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_out, (size_t)N_Q_HEADS * HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_Q_attn, Q_attn_f.data(), Q_attn_f.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gate, gate_f.data(), gate_f.size() * sizeof(float), hipMemcpyHostToDevice));

        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));

        // ---- 1. idx_scan: correctness (copy 0), then timing (rotating) ----
        {
            run_idx_scan_once(r_kpool[0], d_q_idx, d_scores, n_blocks);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> gpu_scores(n_blocks);
            HIP_CHECK(hipMemcpy(gpu_scores.data(), d_scores, (size_t)n_blocks * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> cpu_scores;
            idx_scan_cpu_ref(kpool_for_ref, q_idx_f, cpu_scores, n_blocks);
            check_or_fail("idx_scan" + tag, gpu_scores.data(), cpu_scores.data(), (size_t)n_blocks, json_path);

            for (int i = 0; i < WARMUP; i++)
                run_idx_scan_once(r_kpool[ROT_IDX(i)], d_q_idx, d_scores, n_blocks);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                run_idx_scan_once(r_kpool[ROT_IDX(i)], d_q_idx, d_scores, n_blocks);
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("idx_scan" + tag, samples);
            double bytes = (double)n_blocks * IDX_N_HEADS * IDX_DIM * 2.0; // bf16
            emit_num("idx_scan_gbs" + tag, bytes / 1e9 / (median_of(samples) * 1e-6));
        }

        // ---- 2. topk: correctness, then timing (uses the scores idx_scan just wrote) ----
        std::vector<int> gpu_sel_tokens_sorted;
        {
            run_topk_once(d_scores, n_blocks, d_cand_key, d_cand_idx, d_sel_tokens);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<int> gpu_sel(QSA_TOP_K_TOKENS);
            HIP_CHECK(hipMemcpy(gpu_sel.data(), d_sel_tokens, QSA_TOP_K_TOKENS * sizeof(int), hipMemcpyDeviceToHost));
            gpu_sel_tokens_sorted = gpu_sel;
            std::sort(gpu_sel_tokens_sorted.begin(), gpu_sel_tokens_sorted.end());

            std::vector<float> cpu_scores;
            idx_scan_cpu_ref(kpool_for_ref, q_idx_f, cpu_scores, n_blocks);
            std::vector<int> cpu_sel_tokens_sorted;
            topk_cpu_ref(cpu_scores, n_blocks, cpu_sel_tokens_sorted);

            int match = (gpu_sel_tokens_sorted == cpu_sel_tokens_sorted) ? 1 : 0;
            emit_int("check_topk_set_match" + tag, match);
            printf("check_topk_set_match%s=%d\n", tag.c_str(), match);
            if (!match) {
                printf("CHECK_FAIL topk%s\n", tag.c_str());
                if (!json_path.empty()) write_json(json_path);
                return 1;
            }

            for (int i = 0; i < WARMUP; i++) run_topk_once(d_scores, n_blocks, d_cand_key, d_cand_idx, d_sel_tokens);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                run_topk_once(d_scores, n_blocks, d_cand_key, d_cand_idx, d_sel_tokens);
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("topk" + tag, samples);
        }

        // ---- 3. attn: correctness, then timing (uses the token list topk just wrote) ----
        {
            int n_sel = QSA_TOP_K_TOKENS;
#define RUN_ATTN(c) run_attn_once(r_Kqs[c], r_Ksc[c], r_Vqs[c], r_Vsc[c], d_Q_attn, d_sel_tokens, \
                                   d_gate, d_attn_partial, d_attn_m, d_attn_l, d_attn_out, n_sel)

            RUN_ATTN(0);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> gpu_out((size_t)N_Q_HEADS * HEAD_DIM);
            HIP_CHECK(hipMemcpy(gpu_out.data(), d_attn_out, gpu_out.size() * sizeof(float), hipMemcpyDeviceToHost));

            std::vector<float> cpu_out;
            attn_cpu_ref(K_qs, K_sc, V_qs, V_sc, Q_attn_f, gate_f, gpu_sel_tokens_sorted, cpu_out);
            check_or_fail("attn" + tag, gpu_out.data(), cpu_out.data(), gpu_out.size(), json_path);

            for (int i = 0; i < WARMUP; i++) RUN_ATTN(ROT_IDX(i));
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                RUN_ATTN(ROT_IDX(i));
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("attn" + tag, samples);

            // ---- qsa_layer_us / qsa_token_ms: the whole layer in sequence, timed
            // as one unit. All three kernels of an iteration read the SAME copy,
            // the way one layer reads its own cache, and the iteration after it
            // moves to the next copy, the way the next token does. ----
            for (int i = 0; i < WARMUP; i++) {
                run_idx_scan_once(r_kpool[ROT_IDX(i)], d_q_idx, d_scores, n_blocks);
                run_topk_once(d_scores, n_blocks, d_cand_key, d_cand_idx, d_sel_tokens);
                RUN_ATTN(ROT_IDX(i));
            }
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> layer_samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                run_idx_scan_once(r_kpool[ROT_IDX(i)], d_q_idx, d_scores, n_blocks);
                run_topk_once(d_scores, n_blocks, d_cand_key, d_cand_idx, d_sel_tokens);
                RUN_ATTN(ROT_IDX(i));
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                layer_samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("qsa_layer" + tag, layer_samples);
            double qsa_token_ms = median_of(layer_samples) * N_QSA_LAYERS / 1000.0;
            emit_num("qsa_token_ms" + tag, qsa_token_ms);
            printf("qsa_token_ms%s=%.6f\n", tag.c_str(), qsa_token_ms);
#undef RUN_ATTN
        }

        hipEventDestroy(ev0); hipEventDestroy(ev1);
        // Every rotating copy goes back before the next depth allocates its
        // own -- at 262 144 the three depths together would not fit otherwise.
        for (int c = 0; c < ROT; c++) {
            hipFree(r_kpool[c]); hipFree(r_Kqs[c]); hipFree(r_Vqs[c]);
            hipFree(r_Ksc[c]); hipFree(r_Vsc[c]);
        }
#undef ROT_IDX
        hipFree(d_q_idx); hipFree(d_scores);
        hipFree(d_sel_tokens); hipFree(d_cand_key); hipFree(d_cand_idx);
        hipFree(d_Q_attn); hipFree(d_gate); hipFree(d_attn_out);
        hipFree(d_attn_partial); hipFree(d_attn_m); hipFree(d_attn_l);
    }

    // ---------------------------------------------------------------- GDN --
    // Depth-independent: one recurrent step, one layer, real sizes.
    {
        std::vector<float> S_in = randf_vec((size_t)GDN_V_HEADS * GDN_DIM * GDN_DIM, -0.1f, 0.1f);
        std::vector<float> q = randf_vec((size_t)GDN_V_HEADS * GDN_DIM);
        std::vector<float> k = randf_vec((size_t)GDN_V_HEADS * GDN_DIM);
        std::vector<float> v = randf_vec((size_t)GDN_V_HEADS * GDN_DIM);
        std::vector<float> a = randf_vec(GDN_V_HEADS, 0.80f, 0.999f);     // decay gate
        std::vector<float> beta = randf_vec(GDN_V_HEADS, 0.05f, 0.95f);   // sigmoid-range update gate

        float *d_S_in, *d_q, *d_k, *d_v, *d_a, *d_beta, *d_S_out, *d_out;
        HIP_CHECK(hipMalloc(&d_S_in, S_in.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_k, k.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_v, v.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_a, a.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_beta, beta.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_S_out, S_in.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_S_in, S_in.data(), S_in.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_k, k.data(), k.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_v, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_a, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(k_gdn_step, dim3(GDN_V_HEADS), dim3(GDN_DIM), 0, 0,
                            d_S_in, d_q, d_k, d_v, d_a, d_beta, d_S_out, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> gpu_out(q.size());
        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> cpu_out;
        gdn_cpu_ref(S_in, q, k, v, a, beta, cpu_out);
        check_or_fail("gdn", gpu_out.data(), cpu_out.data(), gpu_out.size(), json_path);

        const int WARMUP2 = 100, ITERS2 = 1000;
        for (int i = 0; i < WARMUP2; i++)
            hipLaunchKernelGGL(k_gdn_step, dim3(GDN_V_HEADS), dim3(GDN_DIM), 0, 0,
                                d_S_in, d_q, d_k, d_v, d_a, d_beta, d_S_out, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));
        std::vector<double> samples(ITERS2);
        for (int i = 0; i < ITERS2; i++) {
            HIP_CHECK(hipEventRecord(ev0));
            hipLaunchKernelGGL(k_gdn_step, dim3(GDN_V_HEADS), dim3(GDN_DIM), 0, 0,
                                d_S_in, d_q, d_k, d_v, d_a, d_beta, d_S_out, d_out);
            HIP_CHECK(hipEventRecord(ev1));
            HIP_CHECK(hipEventSynchronize(ev1));
            samples[i] = event_elapsed_us(ev0, ev1);
        }
        emit_stats_us("gdn_step", samples);
        double gdn_token_ms = median_of(samples) * N_GDN_LAYERS / 1000.0;
        emit_num("gdn_token_ms", gdn_token_ms);
        printf("gdn_token_ms=%.6f\n", gdn_token_ms);

        hipEventDestroy(ev0); hipEventDestroy(ev1);
        hipFree(d_S_in); hipFree(d_q); hipFree(d_k); hipFree(d_v);
        hipFree(d_a); hipFree(d_beta); hipFree(d_S_out); hipFree(d_out);
    }

    if (!json_path.empty()) write_json(json_path);
    return 0;
}
