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

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// ------------------------------------------------------------- kernel 1: idx_scan --
//
// scores[b] = sum over 4 indexer heads of q_h . kpool_h[b] (bf16 pooled
// keys, f32 accumulate). One thread per block b, grid-stride over n_blocks.

__global__ void k_idx_scan(const uint16_t* __restrict__ kpool, const float* __restrict__ q_idx,
                            float* __restrict__ scores, int n_blocks) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += gridDim.x * blockDim.x) {
        float acc = 0.f;
        for (int h = 0; h < IDX_N_HEADS; h++) {
            const uint16_t* kb = kpool + ((size_t)b * IDX_N_HEADS + h) * IDX_DIM;
            const float* qh = q_idx + h * IDX_DIM;
            float dot = 0.f;
            for (int d = 0; d < IDX_DIM; d++) dot += qh[d] * bf16_to_f32(kb[d]);
            acc += dot;
        }
        scores[b] = acc;
    }
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
// Two-pass threshold + compaction (brief: "correctness over elegance;
// report which" -> topk_method="threshold_bisection"). The score range is
// statically bounded (sum of 4*128 = 512 terms each in [-1,1]*[-1,1], so
// |score| <= 512) -- no device-side min/max reduction is needed, which
// keeps this to one kernel type run in a host bisection loop plus one
// compaction kernel. Output is TOKEN indices directly (not block indices):
// each accepted block b contributes tokens [b*ratio, b*ratio+ratio).

__global__ void k_count_ge(const float* __restrict__ scores, int n, float thresh, int* count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && scores[i] >= thresh) atomicAdd(count, 1);
}

__global__ void k_compact_topk_tokens(const float* __restrict__ scores, int n_blocks, float thresh,
                                       int* __restrict__ out_tokens, int* counter, int max_out, int ratio) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    if (scores[b] >= thresh) {
        int pos = atomicAdd(counter, ratio);
        for (int r = 0; r < ratio; r++) {
            int p = pos + r;
            if (p < max_out) out_tokens[p] = b * ratio + r;
        }
    }
}

// Runs the full bisection + compaction pipeline once on [start,stop]
// events already recorded by the caller around it (so callers can time
// warm-up runs identically to timed ones).
static void run_topk_once(const float* d_scores, int n_blocks, int* d_count, int* d_out_tokens,
                           int* d_counter, int threads) {
    float lo = -600.f, hi = 600.f;
    int zero = 0;
    for (int iter = 0; iter < 40; iter++) {
        float mid = 0.5f * (lo + hi);
        HIP_CHECK(hipMemcpy(d_count, &zero, sizeof(int), hipMemcpyHostToDevice));
        int blocks = (n_blocks + threads - 1) / threads;
        hipLaunchKernelGGL(k_count_ge, dim3(blocks), dim3(threads), 0, 0, d_scores, n_blocks, mid, d_count);
        int count = 0;
        HIP_CHECK(hipMemcpy(&count, d_count, sizeof(int), hipMemcpyDeviceToHost));
        if (count >= N_SEL_BLOCKS) lo = mid; else hi = mid;
    }
    HIP_CHECK(hipMemcpy(d_counter, &zero, sizeof(int), hipMemcpyHostToDevice));
    int blocks = (n_blocks + threads - 1) / threads;
    hipLaunchKernelGGL(k_compact_topk_tokens, dim3(blocks), dim3(threads), 0, 0,
                        d_scores, n_blocks, lo, d_out_tokens, d_counter, QSA_TOP_K_TOKENS, QSA_RATIO);
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
// Gather (by index, no separate materialisation pass) K/V for the selected
// tokens from the full Q8_0 cache, score 24 query heads (GQA 12:1 against 2
// KV heads) against them, softmax, weighted V sum, f32 out, sigmoid gate.

__global__ void k_attn_scores(const int8_t* __restrict__ Kqs, const __half* __restrict__ Kscale,
                               const float* __restrict__ Q, const int* __restrict__ sel_tokens,
                               float* __restrict__ scores_out, int n_sel_tokens) {
    int h = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sel_tokens) return;
    int kvh = h / GQA_GROUP;
    int tok = sel_tokens[j];
    const int8_t* qs = Kqs + ((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM;
    const __half* sc = Kscale + ((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32);
    const float* q = Q + h * HEAD_DIM;
    float acc = 0.f;
    for (int d = 0; d < HEAD_DIM; d++) acc += q[d] * dequant_q8_0(qs, sc, d);
    scores_out[(size_t)h * n_sel_tokens + j] = acc * rsqrtf((float)HEAD_DIM);
}

__global__ void k_softmax_rows(float* __restrict__ scores, int n_sel_tokens) {
    int h = blockIdx.x;
    float* row = scores + (size_t)h * n_sel_tokens;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float m = -1e30f;
    for (int j = tid; j < n_sel_tokens; j += blockDim.x) m = fmaxf(m, row[j]);
    red[tid] = m; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    float maxv = red[0]; __syncthreads();
    float sum = 0.f;
    for (int j = tid; j < n_sel_tokens; j += blockDim.x) { float e = expf(row[j] - maxv); row[j] = e; sum += e; }
    red[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float total = red[0]; __syncthreads();
    for (int j = tid; j < n_sel_tokens; j += blockDim.x) row[j] /= total;
}

__global__ void k_attn_output(const int8_t* __restrict__ Vqs, const __half* __restrict__ Vscale,
                               const float* __restrict__ weights, const int* __restrict__ sel_tokens,
                               const float* __restrict__ gate, float* __restrict__ out, int n_sel_tokens) {
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HEAD_DIM) return;
    int kvh = h / GQA_GROUP;
    const float* w = weights + (size_t)h * n_sel_tokens;
    float acc = 0.f;
    for (int j = 0; j < n_sel_tokens; j++) {
        int tok = sel_tokens[j];
        const int8_t* qs = Vqs + ((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM;
        const __half* sc = Vscale + ((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32);
        acc += w[j] * dequant_q8_0(qs, sc, d);
    }
    float g = 1.0f / (1.0f + expf(-gate[h * HEAD_DIM + d]));
    out[h * HEAD_DIM + d] = acc * g;
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
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--json" && i + 1 < argc) json_path = argv[++i];
    }

    HIP_CHECK(hipSetDevice(0));

    emit_str("topk_method", "threshold_bisection");
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
    const int THREADS = 256;

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
        uint16_t* d_kpool; float* d_q_idx; float* d_scores;
        HIP_CHECK(hipMalloc(&d_kpool, kpool_bf16.size() * sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&d_q_idx, q_idx_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scores, (size_t)n_blocks * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_kpool, kpool_bf16.data(), kpool_bf16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_q_idx, q_idx_f.data(), q_idx_f.size() * sizeof(float), hipMemcpyHostToDevice));

        int* d_count; int* d_counter; int* d_sel_tokens;
        HIP_CHECK(hipMalloc(&d_count, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
        HIP_CHECK(hipMalloc(&d_sel_tokens, QSA_TOP_K_TOKENS * sizeof(int)));

        int8_t *d_Kqs, *d_Vqs; __half *d_Ksc, *d_Vsc;
        HIP_CHECK(hipMalloc(&d_Kqs, K_qs.size()));
        HIP_CHECK(hipMalloc(&d_Vqs, V_qs.size()));
        HIP_CHECK(hipMalloc(&d_Ksc, K_sc.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&d_Vsc, V_sc.size() * sizeof(__half)));
        HIP_CHECK(hipMemcpy(d_Kqs, K_qs.data(), K_qs.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_Vqs, V_qs.data(), V_qs.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_Ksc, K_sc.data(), K_sc.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_Vsc, V_sc.data(), V_sc.size() * sizeof(__half), hipMemcpyHostToDevice));

        float *d_Q_attn, *d_gate, *d_attn_scores, *d_attn_out;
        HIP_CHECK(hipMalloc(&d_Q_attn, Q_attn_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_gate, gate_f.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_scores, (size_t)N_Q_HEADS * QSA_TOP_K_TOKENS * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_attn_out, (size_t)N_Q_HEADS * HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_Q_attn, Q_attn_f.data(), Q_attn_f.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gate, gate_f.data(), gate_f.size() * sizeof(float), hipMemcpyHostToDevice));

        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));

        // ---- 1. idx_scan: correctness, then timing ----
        {
            int blocks = (n_blocks + THREADS - 1) / THREADS;
            hipLaunchKernelGGL(k_idx_scan, dim3(blocks), dim3(THREADS), 0, 0, d_kpool, d_q_idx, d_scores, n_blocks);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> gpu_scores(n_blocks);
            HIP_CHECK(hipMemcpy(gpu_scores.data(), d_scores, (size_t)n_blocks * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> cpu_scores;
            idx_scan_cpu_ref(kpool_for_ref, q_idx_f, cpu_scores, n_blocks);
            check_or_fail("idx_scan" + tag, gpu_scores.data(), cpu_scores.data(), (size_t)n_blocks, json_path);

            for (int i = 0; i < WARMUP; i++)
                hipLaunchKernelGGL(k_idx_scan, dim3(blocks), dim3(THREADS), 0, 0, d_kpool, d_q_idx, d_scores, n_blocks);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                hipLaunchKernelGGL(k_idx_scan, dim3(blocks), dim3(THREADS), 0, 0, d_kpool, d_q_idx, d_scores, n_blocks);
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
            run_topk_once(d_scores, n_blocks, d_count, d_sel_tokens, d_counter, THREADS);
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

            for (int i = 0; i < WARMUP; i++) run_topk_once(d_scores, n_blocks, d_count, d_sel_tokens, d_counter, THREADS);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                run_topk_once(d_scores, n_blocks, d_count, d_sel_tokens, d_counter, THREADS);
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("topk" + tag, samples);
        }

        // ---- 3. attn: correctness, then timing (uses the token list topk just wrote) ----
        {
            int n_sel = QSA_TOP_K_TOKENS;
            dim3 grid_scores((n_sel + THREADS - 1) / THREADS, N_Q_HEADS);
            dim3 grid_out((HEAD_DIM + THREADS - 1) / THREADS, N_Q_HEADS);

            hipLaunchKernelGGL(k_attn_scores, grid_scores, dim3(THREADS), 0, 0,
                                d_Kqs, d_Ksc, d_Q_attn, d_sel_tokens, d_attn_scores, n_sel);
            hipLaunchKernelGGL(k_softmax_rows, dim3(N_Q_HEADS), dim3(256), 0, 0, d_attn_scores, n_sel);
            hipLaunchKernelGGL(k_attn_output, grid_out, dim3(THREADS), 0, 0,
                                d_Vqs, d_Vsc, d_attn_scores, d_sel_tokens, d_gate, d_attn_out, n_sel);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> gpu_out((size_t)N_Q_HEADS * HEAD_DIM);
            HIP_CHECK(hipMemcpy(gpu_out.data(), d_attn_out, gpu_out.size() * sizeof(float), hipMemcpyDeviceToHost));

            std::vector<float> cpu_out;
            attn_cpu_ref(K_qs, K_sc, V_qs, V_sc, Q_attn_f, gate_f, gpu_sel_tokens_sorted, cpu_out);
            check_or_fail("attn" + tag, gpu_out.data(), cpu_out.data(), gpu_out.size(), json_path);

            for (int i = 0; i < WARMUP; i++) {
                hipLaunchKernelGGL(k_attn_scores, grid_scores, dim3(THREADS), 0, 0,
                                    d_Kqs, d_Ksc, d_Q_attn, d_sel_tokens, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_softmax_rows, dim3(N_Q_HEADS), dim3(256), 0, 0, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_attn_output, grid_out, dim3(THREADS), 0, 0,
                                    d_Vqs, d_Vsc, d_attn_scores, d_sel_tokens, d_gate, d_attn_out, n_sel);
            }
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> samples(ITERS);
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                hipLaunchKernelGGL(k_attn_scores, grid_scores, dim3(THREADS), 0, 0,
                                    d_Kqs, d_Ksc, d_Q_attn, d_sel_tokens, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_softmax_rows, dim3(N_Q_HEADS), dim3(256), 0, 0, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_attn_output, grid_out, dim3(THREADS), 0, 0,
                                    d_Vqs, d_Vsc, d_attn_scores, d_sel_tokens, d_gate, d_attn_out, n_sel);
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("attn" + tag, samples);

            // ---- qsa_layer_us / qsa_token_ms: the three kernels in sequence, timed as one unit ----
            for (int i = 0; i < WARMUP; i++) {
                int blocks_idx = (n_blocks + THREADS - 1) / THREADS;
                hipLaunchKernelGGL(k_idx_scan, dim3(blocks_idx), dim3(THREADS), 0, 0, d_kpool, d_q_idx, d_scores, n_blocks);
                run_topk_once(d_scores, n_blocks, d_count, d_sel_tokens, d_counter, THREADS);
                hipLaunchKernelGGL(k_attn_scores, grid_scores, dim3(THREADS), 0, 0,
                                    d_Kqs, d_Ksc, d_Q_attn, d_sel_tokens, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_softmax_rows, dim3(N_Q_HEADS), dim3(256), 0, 0, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_attn_output, grid_out, dim3(THREADS), 0, 0,
                                    d_Vqs, d_Vsc, d_attn_scores, d_sel_tokens, d_gate, d_attn_out, n_sel);
            }
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<double> layer_samples(ITERS);
            int blocks_idx = (n_blocks + THREADS - 1) / THREADS;
            for (int i = 0; i < ITERS; i++) {
                HIP_CHECK(hipEventRecord(ev0));
                hipLaunchKernelGGL(k_idx_scan, dim3(blocks_idx), dim3(THREADS), 0, 0, d_kpool, d_q_idx, d_scores, n_blocks);
                run_topk_once(d_scores, n_blocks, d_count, d_sel_tokens, d_counter, THREADS);
                hipLaunchKernelGGL(k_attn_scores, grid_scores, dim3(THREADS), 0, 0,
                                    d_Kqs, d_Ksc, d_Q_attn, d_sel_tokens, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_softmax_rows, dim3(N_Q_HEADS), dim3(256), 0, 0, d_attn_scores, n_sel);
                hipLaunchKernelGGL(k_attn_output, grid_out, dim3(THREADS), 0, 0,
                                    d_Vqs, d_Vsc, d_attn_scores, d_sel_tokens, d_gate, d_attn_out, n_sel);
                HIP_CHECK(hipEventRecord(ev1));
                HIP_CHECK(hipEventSynchronize(ev1));
                layer_samples[i] = event_elapsed_us(ev0, ev1);
            }
            emit_stats_us("qsa_layer" + tag, layer_samples);
            double qsa_token_ms = median_of(layer_samples) * N_QSA_LAYERS / 1000.0;
            emit_num("qsa_token_ms" + tag, qsa_token_ms);
            printf("qsa_token_ms%s=%.6f\n", tag.c_str(), qsa_token_ms);
        }

        hipEventDestroy(ev0); hipEventDestroy(ev1);
        hipFree(d_kpool); hipFree(d_q_idx); hipFree(d_scores);
        hipFree(d_count); hipFree(d_counter); hipFree(d_sel_tokens);
        hipFree(d_Kqs); hipFree(d_Vqs); hipFree(d_Ksc); hipFree(d_Vsc);
        hipFree(d_Q_attn); hipFree(d_gate); hipFree(d_attn_scores); hipFree(d_attn_out);
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
