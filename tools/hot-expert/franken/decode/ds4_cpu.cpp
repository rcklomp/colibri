// tools/hot-expert/franken/decode/ds4_cpu.cpp
//
// Ds4CpuOps: the DeepSeek-V4 operations of ds4_ops.h in plain host C++, the
// arm that proves the math against llama.cpp's dumped intermediates before
// any kernel exists (L5 step 1). Each op follows the ggml CPU op it ports
// statement for statement where the rounding is cheap to follow (the rope
// cache, the Sinkhorn loop, the FWHT, the softmax's scale-by-reciprocal) and
// accumulates dot products in double where ggml uses SIMD float lanes -- the
// same policy as decode_cpu.cpp: this arm is the MORE accurate side.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "ggml.h"

#include "ds4_ops.h"
#include "ds4_shapes.h"

namespace fk {
namespace ds4 {
namespace {

inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
// ggml-cpu/unary-ops.cpp:80
inline float softplusf_(float x) { return (x > 20.0f) ? x : logf(1.0f + expf(x)); }

inline float h2f(uint16_t h) {
    ggml_fp16_t v; std::memcpy(&v, &h, 2);
    return ggml_fp16_to_fp32(v);
}

// ggml.c: ggml_rope_yarn_corr_dim(s)
float corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}
void corr_dims(int n_dims, int n_ctx_orig, float base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(corr_dim(n_dims, n_ctx_orig, beta_fast, base));
    const float end   =  ceilf(corr_dim(n_dims, n_ctx_orig, beta_slow, base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min((float) (n_dims - 1), end);
}
// ops.cpp: rope_yarn_ramp / rope_yarn
float yarn_ramp(float low, float high, int64_t i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1 - std::min(1.0f, std::max(0.0f, y));
}
void rope_yarn(float theta_extrap, float freq_scale, const float cd[2], int64_t i0,
               float ext_factor, float mscale, float * c, float * s) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = yarn_ramp(cd[0], cd[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

class Ds4CpuOps : public Ds4Ops {
public:
    void hc_split(const float * m, const float * s, const float * b,
                  float * pre, float * post, float * comb, float eps, int iters) override {
        for (int h = 0; h < HC; ++h) {
            // ggml_mul, ggml_add, ggml_sigmoid, ggml_scale_bias(1, eps)
            pre[h]  = sigmoidf_(m[h] * s[0] + b[h]) * 1.0f + eps;
            // ggml_mul, ggml_add, ggml_sigmoid, ggml_scale(2)
            post[h] = sigmoidf_(m[HC + h] * s[1] + b[HC + h]) * 2.0f;
        }
        // ops.cpp ggml_compute_forward_dsv4_hc_comb_f32, verbatim
        const float scale_comb = s[2];
        float c[HC * HC];
        for (int isrc = 0; isrc < HC; ++isrc) {
            float mx = -INFINITY;
            for (int idst = 0; idst < HC; ++idst) {
                const int idx = idst + HC * isrc;
                const float v = m[2 * HC + idx] * scale_comb + b[2 * HC + idx];
                c[idx] = v;
                mx = std::max(mx, v);
            }
            float sum = 0.0f;
            for (int idst = 0; idst < HC; ++idst) {
                const int idx = idst + HC * isrc;
                const float v = expf(c[idx] - mx);
                c[idx] = v;
                sum += v;
            }
            const float inv = 1.0f / sum;
            for (int idst = 0; idst < HC; ++idst) {
                const int idx = idst + HC * isrc;
                c[idx] = c[idx] * inv + eps;
            }
        }
        auto norm_cols = [&]() {
            for (int idst = 0; idst < HC; ++idst) {
                float sum = eps;
                for (int isrc = 0; isrc < HC; ++isrc) sum += c[idst + HC * isrc];
                const float inv = 1.0f / sum;
                for (int isrc = 0; isrc < HC; ++isrc) c[idst + HC * isrc] *= inv;
            }
        };
        auto norm_rows = [&]() {
            for (int isrc = 0; isrc < HC; ++isrc) {
                float sum = eps;
                for (int idst = 0; idst < HC; ++idst) sum += c[idst + HC * isrc];
                const float inv = 1.0f / sum;
                for (int idst = 0; idst < HC; ++idst) c[idst + HC * isrc] *= inv;
            }
        };
        norm_cols();
        for (int i = 1; i < iters; ++i) { norm_rows(); norm_cols(); }
        std::memcpy(comb, c, sizeof(c));
    }

    void hc_head_pre(const float * m, const float * s, const float * b, float * pre, float eps) override {
        for (int h = 0; h < HC; ++h) pre[h] = sigmoidf_(m[h] * s[0] + b[h]) * 1.0f + eps;
    }

    void hc_weighted_sum(const float * H, const float * w, float * out) override {
        for (int i = 0; i < N_EMBD; ++i) {
            float sum = 0.0f;
            for (int h = 0; h < HC; ++h) sum += H[(size_t) h * N_EMBD + i] * w[h];
            out[i] = sum;
        }
    }

    void hc_post(const float * x, const float * H, const float * post, const float * comb,
                 float * Hout) override {
        for (int d = 0; d < HC; ++d) {
            for (int i = 0; i < N_EMBD; ++i) {
                float sum = x[i] * post[d];
                for (int s = 0; s < HC; ++s) sum += H[(size_t) s * N_EMBD + i] * comb[d + HC * s];
                Hout[(size_t) d * N_EMBD + i] = sum;
            }
        }
    }

    void rope_tail(float * x, int n_rows, int row_len, int pos, const RopeParams & rp,
                   bool inverse) override {
        const int n_dims = N_ROT;
        const int offs   = row_len - N_ROT;
        const float theta_scale = powf(rp.freq_base, -2.0f / n_dims);
        float cd[2];
        corr_dims(n_dims, rp.n_ctx_orig, rp.freq_base, rp.beta_fast, rp.beta_slow, cd);
        const float sin_sign = inverse ? -1.0f : 1.0f;
        float cache[N_ROT];
        float theta = (float) pos;
        for (int i0 = 0; i0 < n_dims; i0 += 2) {
            rope_yarn(theta, rp.freq_scale, cd, i0, rp.ext_factor, rp.attn_factor,
                      &cache[i0], &cache[i0 + 1]);
            cache[i0 + 1] *= sin_sign;
            theta *= theta_scale;
        }
        for (int r = 0; r < n_rows; ++r) {
            float * v = x + (size_t) r * row_len + offs;
            for (int i0 = 0; i0 < n_dims; i0 += 2) {   // GGML_ROPE_TYPE_NORMAL: adjacent pairs
                const float c = cache[i0], s = cache[i0 + 1];
                const float x0 = v[i0], x1 = v[i0 + 1];
                v[i0]     = x0 * c - x1 * s;
                v[i0 + 1] = x0 * s + x1 * c;
            }
        }
    }

    void fwht(float * x, int n_rows, int n) override {
        const float scale = 1.0f / sqrtf((float) n);
        for (int r = 0; r < n_rows; ++r) {
            float * d = x + (size_t) r * n;
            for (int j = 0; j < n; ++j) d[j] = d[j] * scale;
            for (int len = 1; len < n; len <<= 1)
                for (int i = 0; i < n; i += 2 * len)
                    for (int j = 0; j < len; ++j) {
                        const float u = d[i + j], v = d[i + len + j];
                        d[i + j] = u + v;
                        d[i + len + j] = u - v;
                    }
        }
    }

    void to_f16(const float * x, uint16_t * y, int n) override {
        for (int i = 0; i < n; ++i) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(x[i]);
            std::memcpy(&y[i], &h, 2);
        }
    }

    // ggml_soft_max over one row of candidates, then sum_rows(values*weights):
    // max, exp(x-max) with a double sum, scale by (float)(1/sum), then the
    // weighted values summed in candidate order in double (ggml_vec_sum_f32).
    static float pool_one(const float * v, const float * sc, int n) {
        float mx = -INFINITY;
        for (int j = 0; j < n; ++j) mx = std::max(mx, sc[j]);
        double sum = 0.0;
        float e[2 * HCA_RATIO];
        for (int j = 0; j < n; ++j) { e[j] = expf(sc[j] - mx); sum += (double) e[j]; }
        const float inv = (float) (1.0 / sum);
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += (double) (v[j] * (e[j] * inv));
        return (float) acc;
    }

    void comp_pool(const float * rk, const float * rs, int ring_rows, int ratio, int d_out,
                   int blk, float * out) override {
        std::vector<float> v((size_t) ratio), sc((size_t) ratio);
        for (int d = 0; d < d_out; ++d) {
            for (int j = 0; j < ratio; ++j) {
                const int slot = (blk * ratio + j) % ring_rows;
                v[j]  = rk[(size_t) slot * d_out + d];
                sc[j] = rs[(size_t) slot * d_out + d];
            }
            out[d] = pool_one(v.data(), sc.data(), ratio);
        }
    }

    void comp_pool_overlap(const float * rk, const float * rs, int ring_rows, int ratio,
                           int d_out, int blk, float * out) override {
        const int width = 2 * d_out;
        std::vector<float> v((size_t) 2 * ratio), sc((size_t) 2 * ratio);
        for (int d = 0; d < d_out; ++d) {
            for (int j = 0; j < ratio; ++j) {
                if (blk == 0) {           // the synthetic row: kv 0, score -inf
                    v[j] = 0.0f; sc[j] = -INFINITY;
                } else {
                    const int slot = ((blk - 1) * ratio + j) % ring_rows;
                    v[j]  = rk[(size_t) slot * width + d];
                    sc[j] = rs[(size_t) slot * width + d];
                }
                const int slot = (blk * ratio + j) % ring_rows;
                v[ratio + j]  = rk[(size_t) slot * width + d_out + d];
                sc[ratio + j] = rs[(size_t) slot * width + d_out + d];
            }
            out[d] = pool_one(v.data(), sc.data(), 2 * ratio);
        }
    }

    void lid_scores(const float * q, const float * w, const uint16_t * keys, int n_blocks,
                    float * scores) override {
        float k[IDX_DIM];
        for (int b = 0; b < n_blocks; ++b) {
            for (int i = 0; i < IDX_DIM; ++i) k[i] = h2f(keys[(size_t) b * IDX_DIM + i]);
            float score = 0.0f;
            for (int h = 0; h < IDX_N_HEAD; ++h) {
                double qk = 0.0;
                const float * qh = q + (size_t) h * IDX_DIM;
                for (int i = 0; i < IDX_DIM; ++i) qk += (double) qh[i] * k[i];
                score += std::max((float) qk, 0.0f) * w[h];
            }
            scores[b] = score;   // + the visibility mask's 0: every b < n_blocks is visible
        }
    }

    int topk(const float * scores, int n, int k, int * out) override {
        std::vector<int> idx((size_t) n);
        std::iota(idx.begin(), idx.end(), 0);
        const int m = std::min(k, n);
        std::partial_sort(idx.begin(), idx.begin() + m, idx.end(),
                          [&](int a, int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
        for (int i = 0; i < m; ++i) out[i] = idx[i];
        return m;
    }

    void attn(const float * q, const uint16_t * raw_ring, int raw_pos0, int n_raw,
              const uint16_t * comp, const int * comp_ids, int n_comp,
              const float * sinks, float scale, float * out) override {
        const int n_keys = n_raw + n_comp;
        // K as f32 once for all 64 heads (it is shared: MQA)
        std::vector<float> K((size_t) n_keys * HEAD_DIM);
        for (int j = 0; j < n_raw; ++j) {
            const uint16_t * src = raw_ring + (size_t) ((raw_pos0 + j) % N_SWA) * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; ++i) K[(size_t) j * HEAD_DIM + i] = h2f(src[i]);
        }
        for (int j = 0; j < n_comp; ++j) {
            const int row = comp_ids ? comp_ids[j] : j;
            const uint16_t * src = comp + (size_t) row * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; ++i) K[(size_t) (n_raw + j) * HEAD_DIM + i] = h2f(src[i]);
        }
        std::vector<float> s((size_t) n_keys);
        std::vector<double> acc(HEAD_DIM);
        for (int h = 0; h < N_HEAD; ++h) {
            const float * qh = q + (size_t) h * HEAD_DIM;
            float mx = sinks[h];
            for (int j = 0; j < n_keys; ++j) {
                double d = 0.0;
                const float * kj = K.data() + (size_t) j * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM; ++i) d += (double) qh[i] * kj[i];
                s[j] = (float) d * scale;
                mx = std::max(mx, s[j]);
            }
            double den = (double) expf(sinks[h] - mx);
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < n_keys; ++j) {
                const float p = expf(s[j] - mx);
                den += p;
                const float * kj = K.data() + (size_t) j * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM; ++i) acc[i] += (double) p * kj[i];
            }
            float * oh = out + (size_t) h * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; ++i) oh[i] = (float) (acc[i] / den);
        }
    }

    void router(const float * logits, const float * bias, const int32_t * hash_ids,
                float * probs, float * probs_biased, int * ids, float * w_raw,
                float * w_norm, float * w_scaled) override {
        for (int e = 0; e < N_EXPERT; ++e) probs[e] = sqrtf(softplusf_(logits[e]));
        if (hash_ids) {
            for (int k = 0; k < N_EXPERT_USED; ++k) ids[k] = hash_ids[k];
            if (probs_biased) std::memcpy(probs_biased, probs, sizeof(float) * N_EXPERT);
        } else {
            for (int e = 0; e < N_EXPERT; ++e) probs_biased[e] = probs[e] + bias[e];
            topk(probs_biased, N_EXPERT, N_EXPERT_USED, ids);
        }
        double sum = 0.0;
        for (int k = 0; k < N_EXPERT_USED; ++k) { w_raw[k] = probs[ids[k]]; sum += (double) w_raw[k]; }
        float fs = (float) sum;
        fs = std::max(fs, 6.103515625e-5f);           // ggml_clamp(sum, 6.1e-5, inf)
        for (int k = 0; k < N_EXPERT_USED; ++k) {
            w_norm[k]   = w_raw[k] / fs;               // ggml_div
            w_scaled[k] = w_norm[k] * EXPERT_WEIGHTS_SCALE;   // ggml_scale
        }
    }

    void swiglu_clamp(const float * gate, const float * up, float * h, int n, float limit) override {
        for (int k = 0; k < n; ++k) {
            const float g = std::min(gate[k], limit);
            const float u = std::min(std::max(up[k], -limit), limit);
            h[k] = g / (1.f + expf(-g)) * u;
        }
    }

    void moe_accum(const float * y, const float * w, int n_used, int n, float * weighted,
                   float * out) override {
        for (int e = 0; e < n_used; ++e)
            for (int i = 0; i < n; ++i)
                weighted[(size_t) e * n + i] = y[(size_t) e * n + i] * w[e];
        for (int i = 0; i < n; ++i) {
            float a = weighted[i];
            for (int e = 1; e < n_used; ++e) a = a + weighted[(size_t) e * n + i];
            out[i] = a;
        }
    }
};

} // namespace

Ds4Ops * make_ds4_cpu_ops() { return new Ds4CpuOps(); }

} // namespace ds4
} // namespace fk
