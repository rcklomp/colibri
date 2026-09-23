// tools/hot-expert/franken/decode/decode_cpu.cpp
//
// The `--cpu` backend of L0-STEP2-BRIEF-2026-09-22.md deliverable B: every
// operation of the decode loop in plain host C++, so the MATH can be checked
// against llama.cpp's dumped intermediates without a GPU minute being spent.
// The agent that wrote this could not run a GPU at all (task rule), so this
// is not a convenience -- it is the only oracle available before the run.
//
// What it does NOT do: it does not emulate the HIP kernels. It implements the
// same formulas from the same reference (qwen4exp.cpp, delta-net-base.cpp,
// ggml-cpu/ops.cpp) in the most obvious way, sharing only the weight DECODE
// with the kernels (decode_quant.h). A disagreement between this path and the
// GPU path against the same oracle therefore localises the fault: the graph
// (shared) or the kernels (not shared).
//
// Memory discipline (task rule: "mmap the GGUF, touch only what a test
// needs"): place() returns the mmap pointer unchanged, so a `--cpu` run pages
// in exactly the trunk rows and the 10-of-512 expert slabs it reads -- about
// 1.7 GB for layers 0-15 plus ~23 MB of experts per token, not the 94 GB
// file. Nothing here ever reads per_layer_token_embd wholesale; ../ple.cpp
// gathers its 16 rows.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) fk_cpu_half_to_f32(bits)
static inline float fk_cpu_half_to_f32(unsigned int bits);

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"       // block structs, iq3s_grid, kvalues_iq4nl

#include "decode_quant.h"
#include "decode_backend.h"

static inline float fk_cpu_half_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    const uint16_t u = (uint16_t) bits;
    std::memcpy(&h, &u, sizeof(u));
    return ggml_fp16_to_fp32(h);
}
static inline uint16_t fk_cpu_f32_to_half(float f) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t u; std::memcpy(&u, &h, sizeof(u));
    return u;
}

// The layouts the shared decoders assume must be ggml's own.
static_assert(sizeof(block_q8_0)   == FK_Q8_0_BLOCK_BYTES,   "block_q8_0 size moved");
static_assert(sizeof(block_iq4_xs) == FK_IQ4XS_BLOCK_BYTES,  "block_iq4_xs size moved");
static_assert(sizeof(block_iq3_s)  == IQ3S_BLOCK_BYTES,      "block_iq3_s size moved");
static_assert(sizeof(block_iq4_nl) == IQ4NL_BLOCK_BYTES,     "block_iq4_nl size moved");
static_assert(sizeof(block_q6_K)   == FK_Q6K_BLOCK_BYTES,    "block_q6_K size moved");
static_assert(offsetof(block_q6_K, qh)     == FK_Q6K_OFF_QH,     "q6_K qh moved");
static_assert(offsetof(block_q6_K, scales) == FK_Q6K_OFF_SCALES, "q6_K scales moved");
static_assert(offsetof(block_q6_K, d)      == FK_Q6K_OFF_D,      "q6_K d moved");
static_assert(offsetof(block_q8_0,   qs)       == FK_Q8_0_OFF_QS,        "q8_0 qs moved");
static_assert(offsetof(block_iq4_xs, scales_h) == FK_IQ4XS_OFF_SCALES_H, "iq4_xs scales_h moved");
static_assert(offsetof(block_iq4_xs, scales_l) == FK_IQ4XS_OFF_SCALES_L, "iq4_xs scales_l moved");
static_assert(offsetof(block_iq4_xs, qs)       == FK_IQ4XS_OFF_QS,       "iq4_xs qs moved");
// ds4_quant.h's layouts (DeepSeek-V4-Flash)
static_assert(sizeof(block_q4_K)    == FK_Q4K_BLOCK_BYTES,    "block_q4_K size moved");
static_assert(offsetof(block_q4_K, scales) == FK_Q4K_OFF_SCALES, "q4_K scales moved");
static_assert(offsetof(block_q4_K, qs)     == FK_Q4K_OFF_QS,     "q4_K qs moved");
static_assert(sizeof(block_q5_K)    == FK_Q5K_BLOCK_BYTES,    "block_q5_K size moved");
static_assert(offsetof(block_q5_K, scales) == FK_Q5K_OFF_SCALES, "q5_K scales moved");
static_assert(offsetof(block_q5_K, qh)     == FK_Q5K_OFF_QH,     "q5_K qh moved");
static_assert(offsetof(block_q5_K, qs)     == FK_Q5K_OFF_QS,     "q5_K qs moved");
static_assert(sizeof(block_iq2_xxs) == FK_IQ2XXS_BLOCK_BYTES, "block_iq2_xxs size moved");
static_assert(offsetof(block_iq2_xxs, qs)  == FK_IQ2XXS_OFF_QS,  "iq2_xxs qs moved");
static_assert(sizeof(block_iq2_s)   == FK_IQ2S_BLOCK_BYTES,   "block_iq2_s size moved");
static_assert(offsetof(block_iq2_s, qs)     == FK_IQ2S_OFF_QS,     "iq2_s qs moved");
static_assert(offsetof(block_iq2_s, qh)     == FK_IQ2S_OFF_QH,     "iq2_s qh moved");
static_assert(offsetof(block_iq2_s, scales) == FK_IQ2S_OFF_SCALES, "iq2_s scales moved");
static_assert(sizeof(block_iq3_xxs) == FK_IQ3XXS_BLOCK_BYTES, "block_iq3_xxs size moved");
static_assert(offsetof(block_iq3_xxs, qs)  == FK_IQ3XXS_OFF_QS,  "iq3_xxs qs moved");
static_assert(sizeof(block_mxfp4)   == FK_MXFP4_BLOCK_BYTES,  "block_mxfp4 size moved");
static_assert(offsetof(block_mxfp4, qs)    == FK_MXFP4_OFF_QS,   "mxfp4 qs moved");

namespace fk {

const char * gemv_group_name(int g) {
    switch (g) {
        case GG_HC_DOWN_INJECT: return "hc_down_inject";
        case GG_HC_UP:          return "hc_up";
        case GG_GDN_PROJ:       return "gdn_qkv_gate_beta_alpha";
        case GG_SSM_OUT:        return "ssm_out";
        case GG_FFN_PROJ:       return "ffn_router_shexp";
        case GG_SH_DOWN:        return "shexp_down";
        case GG_QSA_PROJ:       return "qsa_q_k_v_indexer";
        case GG_ATTN_OUT:       return "attn_output";
        case GG_LM_HEAD:        return "lm_head";
        default:                return "ple";
    }
}

namespace {

// ------------------------------------------------------------- threading --
// Rows of a GEMV split over a small pool. Deliberately small by default: the
// owner's GLM-5.3 gateway is serving on this box (CLAUDE.md), and a `--cpu`
// check is a correctness run, not a benchmark.
struct Pool {
    int n = 1;
    template <class F> void parallel_for(int64_t n_items, F && f) const {
        if (n <= 1 || n_items < 64) { f(0, n_items); return; }
        const int nt = (int) std::min<int64_t>(n, n_items);
        std::vector<std::thread> th;
        th.reserve(nt);
        const int64_t chunk = (n_items + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int64_t a = t * chunk, b = std::min<int64_t>(a + chunk, n_items);
            if (a >= b) break;
            th.emplace_back([&f, a, b] { f(a, b); });
        }
        for (auto & t : th) t.join();
    }
};

// --------------------------------------------------------------- row dot --
//
// One output row against the activation vector, in the row's own format.
// The lane loops reproduce the wave mapping the kernels use (decode_quant.h's
// header), so the two paths execute the same decode expressions.
// The Q8_0-weight x Q8_0-activation dot ggml's CPU path actually performs
// (ggml_vec_dot_q8_0_q8_0): per 32-element block, an int8 dot scaled by the
// product of the two block scales. Only used by the --quant-act arm.
float row_dot_q8_0_q8_0(const unsigned char * row, const int8_t * xq, const float * xd,
                        int64_t K) {
    const int64_t nblk = K / FK_Q8_0_BLOCK_WEIGHTS;
    double a = 0.0;
    for (int64_t b = 0; b < nblk; ++b) {
        const unsigned char * bp = row + b * FK_Q8_0_BLOCK_BYTES;
        const signed char * qw = (const signed char *)(bp + FK_Q8_0_OFF_QS);
        const int8_t * qx = xq + b * FK_Q8_0_BLOCK_WEIGHTS;
        int32_t s = 0;
        for (int i = 0; i < FK_Q8_0_BLOCK_WEIGHTS; ++i) s += (int32_t) qw[i] * (int32_t) qx[i];
        a += (double)(fk_q8_0_block_d(bp) * xd[b]) * (double) s;
    }
    return (float) a;
}

float row_dot(int type, const unsigned char * row, const float * x, int64_t K) {
    switch (type) {
        case FK_Q_F32: {
            const float * w = (const float *) row;
            double a = 0.0;
            for (int64_t i = 0; i < K; ++i) a += (double) w[i] * x[i];
            return (float) a;
        }
        case FK_Q_BF16: {
            const uint16_t * w = (const uint16_t *) row;
            double a = 0.0;
            for (int64_t i = 0; i < K; ++i) a += (double) fk_bf16_to_f32(w[i]) * x[i];
            return (float) a;
        }
        case FK_Q_Q8_0: {
            const int64_t nblk = K / FK_Q8_0_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_Q8_0_BLOCK_BYTES;
                const float * xb = x + b * FK_Q8_0_BLOCK_WEIGHTS;
                for (int lane = 0; lane < 32; ++lane) a += fk_q8_0_lane_dot(bp, xb, lane);
            }
            return (float) a;
        }
        case FK_Q_IQ4_NL: {
            const int64_t nchunk = (K / IQ4NL_BLOCK_WEIGHTS) * 4;
            float lo = 0.0f, hi = 0.0f;
            for (int64_t w = 0; w < nchunk; ++w) {
                m1n_iq4nl_chunk_dot(row, kvalues_iq4nl, x, (int) w, &lo, &hi);
            }
            return lo + hi;
        }
        case FK_Q_IQ4_XS: {
            const int64_t nblk = K / FK_IQ4XS_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_IQ4XS_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid) {
                    a += fk_iq4xs_block_dot(bp, kvalues_iq4nl,
                                            x + b * FK_IQ4XS_BLOCK_WEIGHTS + 8 * tid, tid);
                }
            }
            return (float) a;
        }
        case FK_Q_Q6_K: {
            const int64_t nblk = K / FK_Q6K_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_Q6K_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_q6k_block_dot(bp, x + b * FK_Q6K_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_IQ3_S: {
            const int64_t nblk = K / IQ3S_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * IQ3S_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid) {
                    const int ib32 = tid >> 2, l = tid & 3;
                    a += m1n_iq3s_block_dot(bp, iq3s_grid,
                                            x + b * IQ3S_BLOCK_WEIGHTS + 8 * tid,
                                            tid, ib32, 8 - 2 * l, 7 - 2 * l);
                }
            }
            return (float) a;
        }
        // ---- the DeepSeek-V4-Flash formats (ds4_quant.h) -------------------
        // Same shape as the cases above: walk the 32 lanes of every block,
        // each lane's partial from the shared decoder, summed in double.
        case FK_Q_Q4_K: {
            const int64_t nblk = K / FK_Q4K_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_Q4K_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_q4k_block_dot(bp, x + b * FK_Q4K_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_Q5_K: {
            const int64_t nblk = K / FK_Q5K_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_Q5K_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_q5k_block_dot(bp, x + b * FK_Q5K_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_IQ2_XXS: {
            const int64_t nblk = K / FK_IQ2XXS_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_IQ2XXS_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_iq2xxs_block_dot(bp, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs,
                                             x + b * FK_IQ2XXS_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_IQ2_S: {
            const int64_t nblk = K / FK_IQ2S_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_IQ2S_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_iq2s_block_dot(bp, iq2s_grid, kmask_iq2xs,
                                           x + b * FK_IQ2S_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_IQ3_XXS: {
            const int64_t nblk = K / FK_IQ3XXS_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_IQ3XXS_BLOCK_BYTES;
                for (int tid = 0; tid < 32; ++tid)
                    a += fk_iq3xxs_block_dot(bp, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs,
                                             x + b * FK_IQ3XXS_BLOCK_WEIGHTS + 8 * tid, tid);
            }
            return (float) a;
        }
        case FK_Q_MXFP4: {
            const int64_t nblk = K / FK_MXFP4_BLOCK_WEIGHTS;
            double a = 0.0;
            for (int64_t b = 0; b < nblk; ++b) {
                const unsigned char * bp = row + b * FK_MXFP4_BLOCK_BYTES;
                const float * xb = x + b * FK_MXFP4_BLOCK_WEIGHTS;
                for (int lane = 0; lane < 32; ++lane)
                    a += fk_mxfp4_lane_dot(bp, kvalues_mxfp4, xb, lane);
            }
            return (float) a;
        }
        default: return 0.0f;
    }
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf(float x)    { return x * sigmoidf(x); }
// ggml-cpu/unary-ops.cpp:79 -- the >20 shortcut matters for a large alpha
inline float softplusf(float x){ return x > 20.0f ? x : std::log1p(std::exp(x)); }
inline float sgnf(float x)     { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }

// ggml_mrope_cache_init with mode GGML_ROPE_TYPE_IMROPE, a TEXT batch
// (llama-graph.cpp:132-144 sets p_t = p_h = p_w = pos and p_e = 0) and this
// model's ext_factor 0 / attn_factor 1 / freq_scale 1.
// With the three positions equal, every sector maps to the same theta and the
// result is an ordinary NeoX rope over n_rot dims -- but the sections are
// applied literally here so that an image batch, where they differ, does not
// silently need a different kernel.
void rope_cache(int n_rot, int pos, float * cache) {
    const float ts = std::pow(ROPE_FREQ_BASE, -2.0f / (float) n_rot);
    const int sect_dims = ROPE_SECTIONS[0] + ROPE_SECTIONS[1] + ROPE_SECTIONS[2] + ROPE_SECTIONS[3];
    float th_t = (float) pos, th_h = (float) pos, th_w = (float) pos, th_e = 0.0f;
    for (int i0 = 0; i0 < n_rot; i0 += 2) {
        const int sector = (i0 / 2) % sect_dims;
        float theta;
        if      (sector % 3 == 1 && sector < 3 * ROPE_SECTIONS[1]) theta = th_h;
        else if (sector % 3 == 2 && sector < 3 * ROPE_SECTIONS[2]) theta = th_w;
        else if (sector % 3 == 0 && sector < 3 * ROPE_SECTIONS[0]) theta = th_t;
        else                                                      theta = th_e;
        cache[i0 + 0] = std::cos(theta);
        cache[i0 + 1] = std::sin(theta);
        th_t *= ts; th_h *= ts; th_w *= ts; th_e *= ts;
    }
}

// ---------------------------------------------------------------- backend --
class CpuBackend : public Backend {
public:
    explicit CpuBackend(int threads) { pool_.n = std::max(1, threads); }
    ~CpuBackend() override {}

    const char * name() const override { return "cpu"; }
    bool is_gpu() const override { return false; }

    float * alloc_f32(size_t n) override { return (float *) std::calloc(n ? n : 1, sizeof(float)); }
    void *  alloc_raw(size_t b) override { return std::calloc(b ? b : 1, 1); }
    int *   alloc_i32(size_t n) override { return (int *) std::calloc(n ? n : 1, sizeof(int)); }
    void    free_buf(void * p) override { std::free(p); }
    void    upload(void * d, const void * s, size_t b) override { std::memcpy(d, s, b); }
    void    download(void * d, const void * s, size_t b) override { std::memcpy(d, s, b); }
    void    sync() override {}
    int     device() const override { return -1; }
    void    boundary_recv(void * dst, Backend &, const void * src, size_t b, int) override {
        // One host backend serves every range, and it is synchronous, so the
        // residual bank the runner picked is only ever a different address --
        // which is exactly what makes the CPU arm a gate on the bank logic.
        std::memcpy(dst, src, b);
    }
    void    argmax(const float * logits, int n, int * out_id) override {
        int best = 0;
        for (int i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
        *out_id = best;
    }
    void    set_quant_act(bool on) override { quant_act_ = on; }

    const void * place(const void * host, size_t bytes, const char *) override {
        // No copy: the mmap IS the residency. Nothing is paged in until a row
        // is read, which is the whole point of running this check on a box
        // that is also serving a 149 GB model.
        placed_ += bytes;
        return host;
    }
    size_t placed_bytes() const override { return placed_; }

    // -- GEMV ---------------------------------------------------------------
    void gemv(const Mat & W, const float * x, float * y) override { gemv_at(W, (const unsigned char *) W.base, x, y); }

    // The host reference simply runs the jobs in turn: batching exists to cut
    // GPU launches, and reproducing it here would only make this path slower
    // and less obviously correct. The epilogues and the GX_SILU_MUL
    // activation ARE reproduced, because those change the arithmetic.
    //
    // A chunk is the same thing T times, one column at a time, so the host
    // path is bit-identical at any T by construction -- which is what makes
    // `franken_decode_cpu --chunk 6` a usable oracle for `--chunk 1`.
    void gemv_batch(const GemvJob * jobs, int n, const float * x, int group,
                    int x_mode, const float * x2, int T) override {
        (void) group;
        const int64_t K = jobs[0].W->K;
        std::vector<float> xbuf;
        if (x_mode == GX_SILU_MUL) xbuf.resize((size_t) K);
        for (int t = 0; t < (T > 0 ? T : 1); ++t) {
            const float * xt  = x + (size_t) t * K;
            const float * xin = xt;
            if (x_mode == GX_SILU_MUL) {
                const float * x2t = x2 + (size_t) t * K;
                for (int64_t i = 0; i < K; ++i) xbuf[i] = siluf(xt[i]) * x2t[i];
                xin = xbuf.data();
            }
            for (int i = 0; i < n; ++i) {
                const Mat & W = *jobs[i].W;
                float * out = jobs[i].out + (size_t) t * W.rows;
                gemv(W, xin, out);
                switch (jobs[i].epi) {
                    case GE_SIGMOID:
                        for (int64_t r = 0; r < W.rows; ++r) out[r] = sigmoidf(out[r]);
                        break;
                    case GE_SCALE_SILU:
                        for (int64_t r = 0; r < W.rows; ++r) out[r] = siluf(out[r] * jobs[i].arg);
                        break;
                    default: break;
                }
            }
        }
    }

    void gemv_expert(const Mat & W, int e, const float * x, float * y) override {
        gemv_at(W, (const unsigned char *) W.base + (size_t) e * W.expert_stride, x, y);
    }

    void copy(float * d, const float * s, size_t n) override { std::memcpy(d, s, n * sizeof(float)); }

    void unary(int op, const float * x, float * y, size_t n) override {
        for (size_t i = 0; i < n; ++i) {
            switch (op) {
                case FK_SILU:     y[i] = siluf(x[i]);      break;
                case FK_SIGMOID:  y[i] = sigmoidf(x[i]);   break;
                case FK_SOFTPLUS: y[i] = softplusf(x[i]);  break;
                case FK_EXP:      y[i] = std::exp(x[i]);   break;
                default:          y[i] = x[i];             break;
            }
        }
    }
    void binary(int op, const float * a, const float * b, float * y, size_t n) override {
        for (size_t i = 0; i < n; ++i) {
            switch (op) {
                case FK_ADD: y[i] = a[i] + b[i]; break;
                case FK_MUL: y[i] = a[i] * b[i]; break;
                default:     y[i] = a[i] - b[i]; break;
            }
        }
    }
    void scale(const float * x, float s, float * y, size_t n) override {
        for (size_t i = 0; i < n; ++i) y[i] = x[i] * s;
    }
    void binary3_add(const float * a, const float * b, const float * c,
                     float * y, size_t n) override {
        for (size_t i = 0; i < n; ++i) y[i] = a[i] + b[i] + c[i];
    }
    void mul_tiled(const float * x, const float * w, float * y, size_t n, size_t ne0_w) override {
        for (size_t i = 0; i < n; ++i) y[i] = x[i] * w[i % ne0_w];
    }
    void gather_strided(const float * src, float * dst, int n_rows, int row_len,
                        int src_stride, int src_off) override {
        for (int r = 0; r < n_rows; ++r)
            std::memcpy(dst + (size_t) r * row_len,
                        src + (size_t) r * src_stride + src_off,
                        (size_t) row_len * sizeof(float));
    }

    void rms_norm_mul(const float * x, const float * w, float * y,
                      int ne0, int n_groups, size_t w_len, float eps) override {
        // period in GROUPS: 1 for a [ne0] gamma, HC for a [HC*ne0] one. See
        // decode_backend.h -- this is the batched generalisation of the old
        // `w_tiled` flag and reduces to it at T == 1.
        const int period = (int) std::max<size_t>(1, w_len / (size_t) ne0);
        for (int g = 0; g < n_groups; ++g) {
            const float * xg = x + (size_t) g * ne0;
            float * yg = y + (size_t) g * ne0;
            double sum = 0.0;
            for (int i = 0; i < ne0; ++i) sum += (double) xg[i] * xg[i];
            const float sc = 1.0f / std::sqrt((float)(sum / ne0) + eps);
            const float * wg = w ? w + (size_t)(g % period) * ne0 : nullptr;
            for (int i = 0; i < ne0; ++i) yg[i] = xg[i] * sc * (wg ? wg[i] : 1.0f);
        }
    }
    void l2_norm(const float * x, float * y, int ne0, int n_groups, float eps) override {
        for (int g = 0; g < n_groups; ++g) {
            const float * xg = x + (size_t) g * ne0;
            float * yg = y + (size_t) g * ne0;
            double sum = 0.0;
            for (int i = 0; i < ne0; ++i) sum += (double) xg[i] * xg[i];
            const float sc = 1.0f / std::max(std::sqrt((float) sum), eps);
            for (int i = 0; i < ne0; ++i) yg[i] = xg[i] * sc;
        }
    }

    // -- hyper-connections ---------------------------------------------------
    void hc_broadcast(const float * x, float * res, int T) override {
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < HC; ++c)
                std::memcpy(res + (size_t) t * HC_DIM + (size_t) c * N_EMBD,
                            x + (size_t) t * N_EMBD, N_EMBD * sizeof(float));
    }
    void hc_collapse(const float * xn, const float * gate, float * out, int T) override {
        for (int t = 0; t < T; ++t) {
            const float * xt = xn   + (size_t) t * HC_DIM;
            const float * gt = gate + (size_t) t * HC_DIM;
            float * ot = out + (size_t) t * N_EMBD;
            for (int i = 0; i < N_EMBD; ++i) {
                float a = xt[i] * gt[i];                     // stream 0 first, as ggml_cont does
                for (int c = 1; c < HC; ++c)
                    a += xt[(size_t) c * N_EMBD + i] * gt[(size_t) c * N_EMBD + i];
                ot[i] = a * (1.0f / (float) HC);
            }
        }
    }
    void hc_combine(float * res, const float * blk, const float * inject, int T) override {
        hc_combine_norm(res, blk, inject, nullptr, nullptr, 0.0f, T);
    }
    void hc_combine_norm(float * res, const float * blk, const float * inject,
                         const float * w_norm, float * xn, float eps, int T) override {
        for (int t = 0; t < T; ++t) {
            const float * bt = blk + (size_t) t * N_EMBD;
            for (int c = 0; c < HC; ++c) {
                const float w = 2.0f * sigmoidf(inject[(size_t) t * HC + c] * (1.0f / (float) HC));
                float * rc = res + (size_t) t * HC_DIM + (size_t) c * N_EMBD;
                double ss = 0.0;
                for (int i = 0; i < N_EMBD; ++i) { rc[i] += bt[i] * w; ss += (double) rc[i] * rc[i]; }
                if (!w_norm) continue;
                const float sc = 1.0f / std::sqrt((float)(ss / N_EMBD) + eps);
                const float * wg = w_norm + (size_t) c * N_EMBD;
                float * xt = xn + (size_t) t * HC_DIM + (size_t) c * N_EMBD;
                for (int i = 0; i < N_EMBD; ++i) xt[i] = rc[i] * sc * wg[i];
            }
        }
    }

    // -- PLE -----------------------------------------------------------------
    void ple_gate(const float * key, const float * query, const float * value,
                  float * gated, float * gate_out, int T) override {
        for (int t = 0; t < T; ++t) {
            const float * kt = key   + (size_t) t * HC_DIM;
            const float * qt = query + (size_t) t * HC_DIM;
            const float * vt = value + (size_t) t * N_EMBD;
            for (int c = 0; c < HC; ++c) {
                double s = 0.0;
                for (int i = 0; i < N_EMBD; ++i)
                    s += (double) kt[(size_t) c * N_EMBD + i] * qt[(size_t) c * N_EMBD + i];
                const float sf  = (float)(s) * (1.0f / std::sqrt((float) N_EMBD));
                const float mag = std::sqrt(std::max(std::fabs(sf), 1e-6f));
                const float g   = sigmoidf(sgnf(sf) * mag);
                gate_out[(size_t) t * HC + c] = g;
                float * gt = gated + (size_t) t * HC_DIM + (size_t) c * N_EMBD;
                for (int i = 0; i < N_EMBD; ++i) gt[i] = vt[i] * g;
            }
        }
    }
    void ple_conv_win(const float * win, const float * w, float * out, int T) override {
        for (int t = 0; t < T; ++t)
            conv_win_taps(win, t, PLE_CONV_K, PLE_CONV_DIL, HC_DIM, w,
                          out + (size_t) t * HC_DIM);
    }
    void conv_slide(float * win, int hist, int channels, int T) override {
        // every channel column is read whole before it is written, so the
        // overlapping T < hist case is safe
        std::vector<float> col(hist);
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < hist; ++s) col[s] = win[(size_t)(T + s) * channels + c];
            for (int s = 0; s < hist; ++s) win[(size_t) s * channels + c] = col[s];
        }
    }

    // -- Gated DeltaNet ------------------------------------------------------
    void gdn_conv_gate(const float * win, const float * w,
                       const float * beta_raw, const float * alpha_raw,
                       const float * dt, const float * ssm_a, float * conv_out,
                       float * conv_qk, float * beta_sig, float * a_softplus,
                       float * gate, float * g_exp, int T) override {
        for (int t = 0; t < T; ++t) {
            float * co = conv_out + (size_t) t * GDN_CONV_DIM;
            conv_win_taps(win, t, GDN_CONV_K, 1, GDN_CONV_DIM, w, co);
            std::memcpy(conv_qk + (size_t) t * 2 * GDN_KEY_DIM, co,
                        (size_t) 2 * GDN_KEY_DIM * sizeof(float));
            for (int h = 0; h < GDN_V_HEADS; ++h) {
                const size_t o = (size_t) t * GDN_V_HEADS + h;
                beta_sig[o]   = sigmoidf(beta_raw[o]);
                const float sp = softplusf(alpha_raw[o] + dt[h]);
                a_softplus[o] = sp;
                const float g = sp * ssm_a[h];  // ssm_a already holds -exp(A_log)
                gate[o]       = g;
                g_exp[o]      = std::exp(g);
            }
        }
    }

    // ops.cpp:11003-11047, the K=1 / non-KDA branch of
    // ggml_compute_forward_gated_delta_net_one_chunk, one token:
    //   S *= exp(g);  d[j] = (v[j] - dot(S_row_j, k)) * beta;
    //   S_row_j += k * d[j];  out[j] = dot(S_row_j, q) / sqrt(S_v)
    // The state is [value][key] with the KEY contiguous (`s_out[j*S_v + i]`),
    // which is the transpose of m3_attn.hip's k_gdn_step layout -- and the
    // ORDER matters too: m3 computed the prediction from the UNDECAYED state,
    // this decays first, as the op does.
    void gdn_step(float * state, const float * q, const float * k, const float * v,
                  const float * g_exp, const float * beta, float * out,
                  int T, int qk_stride, int v_stride) override {
        const float scale = 1.0f / std::sqrt((float) GDN_STATE);
        pool_.parallel_for(GDN_V_HEADS, [&](int64_t h0, int64_t h1) {
            for (int64_t h = h0; h < h1; ++h) {
                const int hk = (int)(h % GDN_K_HEADS);     // ggml_repeat / `iv1 % nek1`
                float * S = state + (size_t) h * GDN_STATE * GDN_STATE;
                // the chunk in order: the state a token sees is the one the
                // token before it left, which is what makes T steps of one
                // token and one step of T tokens the same arithmetic
                for (int t = 0; t < T; ++t) {
                    const float * kh = k + (size_t) t * qk_stride + (size_t) hk * GDN_STATE;
                    const float * qh = q + (size_t) t * qk_stride + (size_t) hk * GDN_STATE;
                    const float * vh = v + (size_t) t * v_stride  + (size_t) h  * GDN_STATE;
                    const float ge = g_exp[(size_t) t * GDN_V_HEADS + h];
                    const float bh = beta [(size_t) t * GDN_V_HEADS + h];
                    float * ot = out + (size_t) t * GDN_VAL_DIM + (size_t) h * GDN_STATE;
                    for (int j = 0; j < GDN_STATE; ++j) {
                        float * row = S + (size_t) j * GDN_STATE;
                        double sk = 0.0;
                        for (int i = 0; i < GDN_STATE; ++i) { row[i] *= ge; sk += (double) row[i] * kh[i]; }
                        const float d = bh * (vh[j] - (float) sk);
                        double o = 0.0;
                        for (int i = 0; i < GDN_STATE; ++i) { row[i] += kh[i] * d; o += (double) row[i] * qh[i]; }
                        ot[j] = (float) o * scale;
                    }
                }
            }
        });
    }

    void gated_rms_norm(const float * x, const float * w, const float * z, float * y,
                        int ne0, int n_groups, float eps) override {
        for (int g = 0; g < n_groups; ++g) {
            const float * xg = x + (size_t) g * ne0;
            const float * zg = z + (size_t) g * ne0;
            float * yg = y + (size_t) g * ne0;
            double sum = 0.0;
            for (int i = 0; i < ne0; ++i) sum += (double) xg[i] * xg[i];
            const float sc = 1.0f / std::sqrt((float)(sum / ne0) + eps);
            for (int i = 0; i < ne0; ++i) yg[i] = xg[i] * sc * w[i] * sigmoidf(zg[i]);
        }
    }

    // -- MoE ------------------------------------------------------------------
    void router(const float * logits, int * ids, float * weights,
                int * ids_log, float * w_log, int T) override {
        for (int t = 0; t < T; ++t)
            router_one(logits + (size_t) t * N_EXPERT,
                       ids + (size_t) t * N_EXPERT_USED,
                       weights + (size_t) t * N_EXPERT_USED,
                       ids_log ? ids_log + (size_t) t * N_EXPERT_USED : nullptr,
                       w_log   ? w_log   + (size_t) t * N_EXPERT_USED : nullptr);
    }
    void router_one(const float * logits, int * ids, float * weights,
                    int * ids_log, float * w_log) {
        // ggml_soft_max over all N_EXPERT, then ggml_top_k on the same probs
        float mx = logits[0];
        for (int i = 1; i < N_EXPERT; ++i) mx = std::max(mx, logits[i]);
        std::vector<float> p(N_EXPERT);
        double sum = 0.0;
        for (int i = 0; i < N_EXPERT; ++i) { p[i] = std::exp(logits[i] - mx); sum += p[i]; }
        for (int i = 0; i < N_EXPERT; ++i) p[i] = (float)(p[i] / sum);

        std::vector<int> idx(N_EXPERT);
        for (int i = 0; i < N_EXPERT; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + N_EXPERT_USED, idx.end(),
                          [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        double wsum = 0.0;
        for (int k = 0; k < N_EXPERT_USED; ++k) { ids[k] = idx[k]; weights[k] = p[idx[k]]; wsum += p[idx[k]]; }
        // build_moe_ffn: clamp to the smallest f16 before the divide
        const float den = std::max((float) wsum, 6.103515625e-5f);
        for (int k = 0; k < N_EXPERT_USED; ++k) weights[k] /= den;
        if (ids_log) for (int k = 0; k < N_EXPERT_USED; ++k) ids_log[k] = ids[k];
        if (w_log)   for (int k = 0; k < N_EXPERT_USED; ++k) w_log[k]   = weights[k];
        // expert_weights_scale is 0 for this file, so no scale step (see decode_shapes.h)
    }

    void moe_gate_up(const Mat & gate, const Mat & up, const int * ids,
                     const float * x, float * y_gate, float * y_up, int T) override {
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < N_EXPERT_USED; ++k) {
                const int e = ids[(size_t) t * N_EXPERT_USED + k];
                const size_t col = (size_t) t * N_EXPERT_USED + k;
                const float * xt = x + (size_t) t * gate.K;
                gemv_expert(gate, e, xt, y_gate + col * N_FF_EXP);
                gemv_expert(up,   e, xt, y_up   + col * N_FF_EXP);
            }
    }
    void silu_mul(const float * g, const float * u, float * h, size_t n) override {
        for (size_t i = 0; i < n; ++i) h[i] = siluf(g[i]) * u[i];
    }
    void moe_down(const Mat & down, const int * ids, const float * h, float * eo, int T) override {
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < N_EXPERT_USED; ++k) {
                const size_t col = (size_t) t * N_EXPERT_USED + k;
                gemv_expert(down, ids[col], h + col * N_FF_EXP, eo + col * N_EMBD);
            }
    }
    void moe_finish(const float * eo, const float * w, const float * sh_raw,
                    const float * sh_gate, float * sh_gate_sig, float * moe_out,
                    float * sh_gated, float * out, int T) override {
        for (int t = 0; t < T; ++t) {
            const float g0 = sigmoidf(sh_gate[t]);
            sh_gate_sig[t] = g0;
            const float * eot = eo + (size_t) t * N_EXPERT_USED * N_EMBD;
            const float * wt  = w  + (size_t) t * N_EXPERT_USED;
            const size_t o = (size_t) t * N_EMBD;
            for (int i = 0; i < N_EMBD; ++i) {
                float a = 0.0f;
                for (int k = 0; k < N_EXPERT_USED; ++k) a += wt[k] * eot[(size_t) k * N_EMBD + i];
                moe_out[o + i]  = a;
                sh_gated[o + i] = sh_raw[o + i] * g0;
                out[o + i]      = a + sh_gated[o + i];
            }
        }
    }

    // -- IMRoPE ----------------------------------------------------------------
    void rope_imrope(float * x, int n_heads, int head_dim, int n_rot, int pos) override {
        std::vector<float> cache(n_rot);
        rope_cache(n_rot, pos, cache.data());
        const int half = n_rot / 2;
        for (int h = 0; h < n_heads; ++h) {
            float * row = x + (size_t) h * head_dim;
            for (int i0 = 0; i0 < n_rot; i0 += 2) {
                const int ic = i0 / 2;                      // NeoX pairing (ic, ic+n_rot/2)
                const float c = cache[i0], s = cache[i0 + 1];
                const float x0 = row[ic], x1 = row[ic + half];
                row[ic]        = x0 * c - x1 * s;
                row[ic + half] = x0 * s + x1 * c;
            }
            // dims [n_rot, head_dim) are copied through untouched, as
            // ggml_compute_forward_rope_flt's "fill the remain channels" does.
        }
    }

    // -- QSA -------------------------------------------------------------------
    void kv_store_q8_0(const float * k, const float * v,
                       int8_t * kqs, uint16_t * ksc, int8_t * vqs, uint16_t * vsc,
                       int cell0, int T) override {
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < N_KV_HEADS; ++h) {
                const size_t e  = ((size_t)(cell0 + t) * N_KV_HEADS + h) * HEAD_DIM;
                const size_t sb = ((size_t)(cell0 + t) * N_KV_HEADS + h) * (HEAD_DIM / 32);
                const size_t s  = (size_t) t * N_KV_HEADS * HEAD_DIM + (size_t) h * HEAD_DIM;
                quant_row(k + s, kqs + e, ksc + sb);
                quant_row(v + s, vqs + e, vsc + sb);
            }
    }
    void idx_pool_chunk(const float * k_new, float * sum, float * raw0, int pos0,
                        int T, const float * k_norm, float eps, uint16_t * pooled,
                        float * dbg_pooled, float * dbg_roped) override {
        for (int t = 0; t < T; ++t) {
            const int p        = pos0 + t;
            const int blk      = p / QSA_RATIO;
            const int n_filled = (p + 1) - blk * QSA_RATIO;
            idx_pool_block(k_new + (size_t) t * IDX_DIM, sum, raw0, blk, n_filled,
                           k_norm, eps, pooled,
                           t + 1 == T ? dbg_pooled : nullptr,
                           t + 1 == T ? dbg_roped  : nullptr);
        }
    }
    void idx_pool_block(const float * k_new, float * sum, float * raw0, int blk,
                        int n_filled, const float * k_norm, float eps,
                        uint16_t * pooled, float * dbg_pooled, float * dbg_roped) {
        float p[IDX_DIM];
        for (int i = 0; i < IDX_DIM; ++i) {
            const float kv = k_new[i];
            if (blk == 0 && n_filled == 1) raw0[i] = kv;       // cell 0 is the fill value
            sum[i] = (n_filled == 1) ? kv : sum[i] + kv;       // members arrive in order
            p[i] = (sum[i] + (float)(QSA_RATIO - n_filled) * raw0[i]) * (1.0f / (float) QSA_RATIO);
        }
        if (dbg_pooled) std::memcpy(dbg_pooled, p, sizeof(p));

        rms_norm_mul(p, k_norm, p, IDX_DIM, 1, IDX_DIM, eps);
        rope_imrope(p, 1, IDX_DIM, N_ROT, blk * QSA_RATIO);   // block position = first token's
        if (dbg_roped) std::memcpy(dbg_roped, p, sizeof(p));

        for (int i = 0; i < IDX_DIM; ++i) pooled[(size_t) blk * IDX_DIM + i] = fk_f32_to_bf16(p[i]);
    }
    void idx_scan(const uint16_t * pooled, const float * q, float * scores, int n_blocks) override {
        for (int b = 0; b < n_blocks; ++b) {
            const uint16_t * kb = pooled + (size_t) b * IDX_DIM;
            float acc = 0.0f;
            for (int h = 0; h < IDX_N_HEADS; ++h) {
                const float * qh = q + (size_t) h * IDX_DIM;
                double d = 0.0;
                for (int i = 0; i < IDX_DIM; ++i) d += (double) qh[i] * fk_bf16_to_f32(kb[i]);
                acc += std::max(0.0f, (float) d);          // ggml_relu, per head, before the sum
            }
            scores[b] = acc;
        }
    }
    void qsa_qk_post(const float * qfull, const float * kraw, const float * idxraw,
                     const float * q_norm, const float * k_norm, const float * iq_norm,
                     float * qcur, float * gate, float * gsig, float * kcur,
                     float * idxq, int pos0, float eps, int T) override {
      for (int t = 0; t < T; ++t) {
        const float * qf = qfull  + (size_t) t * N_Q_HEADS   * HEAD_DIM * 2;
        const float * kr = kraw   + (size_t) t * N_KV_HEADS  * HEAD_DIM;
        const float * ir = idxraw + (size_t) t * IDX_N_HEADS * IDX_DIM;
        float * qc = qcur + (size_t) t * N_Q_HEADS   * HEAD_DIM;
        float * ga = gate + (size_t) t * N_Q_HEADS   * HEAD_DIM;
        float * gs = gsig + (size_t) t * N_Q_HEADS   * HEAD_DIM;
        float * kc = kcur + (size_t) t * N_KV_HEADS  * HEAD_DIM;
        float * iq = idxq + (size_t) t * IDX_N_HEADS * IDX_DIM;

        // the [q|gate] split of attn_q's interleaved rows (qwen4exp.cpp:726)
        for (int h = 0; h < N_Q_HEADS; ++h) {
            const float * row = qf + (size_t) h * HEAD_DIM * 2;
            std::memcpy(qc + (size_t) h * HEAD_DIM, row, HEAD_DIM * sizeof(float));
            for (int i = 0; i < HEAD_DIM; ++i) {
                const float v = row[HEAD_DIM + i];
                ga[(size_t) h * HEAD_DIM + i] = v;
                gs[(size_t) h * HEAD_DIM + i] = sigmoidf(v);
            }
        }
        std::memcpy(kc, kr, (size_t) N_KV_HEADS  * HEAD_DIM * sizeof(float));
        std::memcpy(iq, ir, (size_t) IDX_N_HEADS * IDX_DIM  * sizeof(float));

        rms_norm_mul(qc, q_norm,  qc, HEAD_DIM, N_Q_HEADS,   HEAD_DIM, eps);
        rms_norm_mul(kc, k_norm,  kc, HEAD_DIM, N_KV_HEADS,  HEAD_DIM, eps);
        rms_norm_mul(iq, iq_norm, iq, IDX_DIM,  IDX_N_HEADS, IDX_DIM,  eps);
        rope_imrope(qc, N_Q_HEADS,   HEAD_DIM, N_ROT, pos0 + t);
        rope_imrope(kc, N_KV_HEADS,  HEAD_DIM, N_ROT, pos0 + t);
        rope_imrope(iq, IDX_N_HEADS, IDX_DIM,  N_ROT, pos0 + t);
      }
    }
    void qsa_expand(const float * blk_scores, float * cell, int * sel, int n_kv,
                    int q_pos, int ratio, int tail_start, int sel_identity) override {
        const int n_blocks = (n_kv + ratio - 1) / ratio;
        for (int j = 0; j < n_kv; ++j) {
            const int b      = j / ratio;
            const int filled = std::min(ratio, n_kv - b * ratio);
            const float bias = (b * ratio >= tail_start) ? 1e9f
                             : (filled < ratio ? -INFINITY : 0.0f);
            cell[j] = (j <= q_pos) ? blk_scores[b] + bias : -INFINITY;
            if (sel_identity) sel[j] = j;
        }
        (void) n_blocks;
    }
    int topk_select(const float * scores, int n, int width, int * out) override {
        std::vector<int> idx;
        idx.reserve(n);
        for (int i = 0; i < n; ++i) if (scores[i] > -INFINITY) idx.push_back(i);
        const int k = (int) std::min<size_t>(idx.size(), (size_t) width);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
        for (int i = 0; i < k; ++i) out[i] = idx[i];
        return k;
    }
    void attn_qsa(const int8_t * kqs, const uint16_t * ksc, const int8_t * vqs,
                  const uint16_t * vsc, const float * q, const int * sel, int n_sel,
                  const float * gsig, float * out, float * out_gated) override {
        const float scale = 1.0f / std::sqrt((float) HEAD_DIM);
        std::vector<float> p(n_sel);
        for (int h = 0; h < N_Q_HEADS; ++h) {
            const int kvh = h / GQA_GROUP;
            const float * qh = q + (size_t) h * HEAD_DIM;
            float mx = -INFINITY;
            for (int j = 0; j < n_sel; ++j) {
                const size_t e  = ((size_t) sel[j] * N_KV_HEADS + kvh) * HEAD_DIM;
                const size_t sb = ((size_t) sel[j] * N_KV_HEADS + kvh) * (HEAD_DIM / 32);
                double a = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d)
                    a += (double) qh[d] * (fk_cpu_half_to_f32(ksc[sb + (d >> 5)]) * (float) kqs[e + d]);
                p[j] = (float)(a) * scale;
                mx = std::max(mx, p[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < n_sel; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
            for (int j = 0; j < n_sel; ++j) p[j] = (float)(p[j] / sum);
            for (int d = 0; d < HEAD_DIM; ++d) {
                double a = 0.0;
                for (int j = 0; j < n_sel; ++j) {
                    const size_t e  = ((size_t) sel[j] * N_KV_HEADS + kvh) * HEAD_DIM;
                    const size_t sb = ((size_t) sel[j] * N_KV_HEADS + kvh) * (HEAD_DIM / 32);
                    a += (double) p[j] * (fk_cpu_half_to_f32(vsc[sb + (d >> 5)]) * (float) vqs[e + d]);
                }
                const float v = (float) a;
                out[(size_t) h * HEAD_DIM + d]       = v;
                out_gated[(size_t) h * HEAD_DIM + d] = v * gsig[(size_t) h * HEAD_DIM + d];
            }
        }
    }

private:
    void gemv_at(const Mat & W, const unsigned char * base, const float * x, float * y) {
        if (quant_act_ && W.type == FK_Q_Q8_0) {
            // quantise the activation once per GEMV, as ggml does for src1
            const int64_t nblk = W.K / FK_Q8_0_BLOCK_WEIGHTS;
            std::vector<int8_t> xq((size_t) W.K);
            std::vector<float>  xd((size_t) nblk);
            for (int64_t b = 0; b < nblk; ++b) {
                float amax = 0.0f;
                for (int i = 0; i < FK_Q8_0_BLOCK_WEIGHTS; ++i)
                    amax = std::max(amax, std::fabs(x[b * FK_Q8_0_BLOCK_WEIGHTS + i]));
                const float d = amax / 127.0f;
                xd[b] = d;
                const float inv = d ? 1.0f / d : 0.0f;
                for (int i = 0; i < FK_Q8_0_BLOCK_WEIGHTS; ++i)
                    xq[b * FK_Q8_0_BLOCK_WEIGHTS + i] =
                        (int8_t) std::lround(x[b * FK_Q8_0_BLOCK_WEIGHTS + i] * inv);
            }
            pool_.parallel_for(W.rows, [&](int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; ++r)
                    y[r] = row_dot_q8_0_q8_0(base + (size_t) r * W.row_bytes,
                                             xq.data(), xd.data(), W.K);
            });
            return;
        }
        if (quant_act_ && W.type == FK_Q_BF16) {
            std::vector<float> xb((size_t) W.K);
            for (int64_t i = 0; i < W.K; ++i) xb[i] = fk_bf16_to_f32(fk_f32_to_bf16(x[i]));
            pool_.parallel_for(W.rows, [&](int64_t r0, int64_t r1) {
                for (int64_t r = r0; r < r1; ++r)
                    y[r] = row_dot(W.type, base + (size_t) r * W.row_bytes, xb.data(), W.K);
            });
            return;
        }
        pool_.parallel_for(W.rows, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r)
                y[r] = row_dot(W.type, base + (size_t) r * W.row_bytes, x, W.K);
        });
    }

    // ggml_ssm_conv (ops.cpp:9741-9750) generalised by a dilation, which is
    // how build_ple's tap k reads (kern-1-k)*dilation positions back.
    //
    // Window form: token `t` of the chunk lives at slot (kern-1)*dil + t, so
    // its tap k -- which reads (kern-1-k)*dil positions back -- is at slot
    // t + k*dil. No modulus, and the first tokens of a chunk read the
    // previous chunk's tail without any special case.
    void conv_win_taps(const float * win, int t, int kern, int dil,
                       int channels, const float * w, float * out) {
        for (int c = 0; c < channels; ++c) out[c] = 0.0f;
        for (int k = 0; k < kern; ++k) {
            const float * s = win + (size_t)(t + k * dil) * channels;
            for (int c = 0; c < channels; ++c) out[c] += s[c] * w[(size_t) c * kern + k];
        }
        for (int c = 0; c < channels; ++c) out[c] = siluf(out[c]);
    }

    static void quant_row(const float * x, int8_t * qs, uint16_t * sc) {
        for (int b = 0; b < HEAD_DIM / 32; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            sc[b] = fk_cpu_f32_to_half(s);
            const float inv = 1.0f / fk_cpu_half_to_f32(sc[b]);   // quantise against the STORED scale
            for (int i = 0; i < 32; ++i) {
                int q = (int) std::lround(x[b * 32 + i] * inv);
                q = std::max(-127, std::min(127, q));
                qs[b * 32 + i] = (int8_t) q;
            }
        }
    }

    Pool   pool_;
    bool   quant_act_ = false;
    size_t placed_ = 0;
};

} // namespace

Backend * make_cpu_backend(int n_threads) { return new CpuBackend(n_threads); }

} // namespace fk
