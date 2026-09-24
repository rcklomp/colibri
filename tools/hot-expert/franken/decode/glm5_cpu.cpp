// tools/hot-expert/franken/decode/glm5_cpu.cpp
//
// Glm5CpuOps: the GLM-5.3-Flash operations of glm5_ops.h in plain host C++,
// the arm that proves the math against llama.cpp's dumped intermediates
// before any kernel exists (L5 GLM step 2). Each op follows the ggml CPU op
// it ports statement for statement where the rounding is cheap to follow
// (the softmax's scale-by-reciprocal, the conv's float accumulation, the
// gate's op order, the f16 cache rounding) and accumulates long dot
// products in double where ggml uses SIMD float lanes -- decode_cpu.cpp's
// policy: this arm is the MORE accurate side.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "ggml.h"

#include "glm5_ops.h"
#include "glm5_shapes.h"

namespace fk {
namespace glm5 {
namespace {

inline float h2f(uint16_t h) {
    ggml_fp16_t v; std::memcpy(&v, &h, 2);
    return ggml_fp16_to_fp32(v);
}
inline uint16_t f2h(float f) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t u; std::memcpy(&u, &h, 2);
    return u;
}
// ggml's scalar forms (ggml-cpu/vec.h ggml_silu_f32, the sigmoid unary)
inline float silu_(float x)    { return x / (1.0f + expf(-x)); }
inline float sigmoid_(float x) { return 1.0f / (1.0f + expf(-x)); }

class Glm5CpuOps : public Glm5Ops {
public:
    explicit Glm5CpuOps(Backend & be) : be_(be) {}

    // ------------------------------------------------------------ KDA --------
    void kda_conv(const float * qkv, float * state, const float * wq, const float * wk,
                  const float * wv, float * out, int T) override {
        constexpr int H = KDA_CONV - 1;                 // history rows
        // the window: H history rows then the T new ones (ggml_concat, dim 0)
        std::vector<float> win((size_t) (H + T) * KDA_QKV);
        std::memcpy(win.data(), state, (size_t) H * KDA_QKV * sizeof(float));
        std::memcpy(win.data() + (size_t) H * KDA_QKV, qkv, (size_t) T * KDA_QKV * sizeof(float));
        for (int t = 0; t < T; ++t) {
            float * o = out + (size_t) t * KDA_QKV;
            for (int c = 0; c < KDA_QKV; ++c) {
                const float * w = c < KDA_INNER ? wq + (size_t) c * KDA_CONV
                                : c < 2 * KDA_INNER ? wk + (size_t) (c - KDA_INNER) * KDA_CONV
                                : wv + (size_t) (c - 2 * KDA_INNER) * KDA_CONV;
                float sumf = 0.0f;                      // ops.cpp ssm_conv: float, tap order
                for (int k = 0; k < KDA_CONV; ++k) sumf += win[(size_t) (t + k) * KDA_QKV + c] * w[k];
                o[c] = silu_(sumf);
            }
        }
        // conv_state_last: the window's last H rows
        std::memcpy(state, win.data() + (size_t) T * KDA_QKV, (size_t) H * KDA_QKV * sizeof(float));
    }

    void kda_gate(const float * fb, const float * dt, const float * a, float lb, float * g, int T) override {
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < N_HEAD; ++h)
                for (int i = 0; i < KDA_DIM; ++i) {
                    const size_t c = (size_t) h * KDA_DIM + i;
                    float x = fb[(size_t) t * KDA_INNER + c] + dt[c];   // ggml_add(g, dt_b)
                    x = x * a[h];                                       // ggml_mul(g, ssm_a)
                    x = x * -1.0f;                                      // ggml_scale(g, -1)
                    x = sigmoid_(x);                                    // ggml_sigmoid
                    g[(size_t) t * KDA_INNER + c] = x * lb;             // ggml_scale(g, lower_bound)
                }
    }

    // ops.cpp ggml_compute_forward_gated_delta_net_one_chunk, kda branch; the
    // state row j (a VALUE index) holds the KEY axis contiguous. Every row j
    // is independent once decayed, so the per-row fusion below executes the
    // op's statements in its order for that row.
    void kda_step(float * state, const float * q, const float * k, const float * v,
                  const float * g, const float * beta, float * out, int stride, int T) override {
        const float scale = 1.0f / sqrtf((float) KDA_DIM);
        float dec[KDA_DIM];
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < N_HEAD; ++h) {
                float * S = state + (size_t) h * KDA_DIM * KDA_DIM;
                const float * qh = q + (size_t) t * stride + (size_t) h * KDA_DIM;
                const float * kh = k + (size_t) t * stride + (size_t) h * KDA_DIM;
                const float * vh = v + (size_t) t * stride + (size_t) h * KDA_DIM;
                const float * gh = g + (size_t) t * KDA_INNER + (size_t) h * KDA_DIM;
                const float bh = beta[(size_t) t * N_HEAD + h];
                float * oh = out + (size_t) t * KDA_INNER + (size_t) h * KDA_DIM;
                for (int i = 0; i < KDA_DIM; ++i) dec[i] = expf(gh[i]);
                for (int j = 0; j < KDA_DIM; ++j) {
                    float * row = S + (size_t) j * KDA_DIM;
                    double sk = 0.0;
                    for (int i = 0; i < KDA_DIM; ++i) { row[i] = row[i] * dec[i]; sk += (double) row[i] * kh[i]; }
                    const float d = (vh[j] - (float) sk) * bh;
                    double o = 0.0;
                    for (int i = 0; i < KDA_DIM; ++i) { row[i] = row[i] + kh[i] * d; o += (double) row[i] * qh[i]; }
                    oh[j] = (float) o * scale;
                }
            }
    }

    // ------------------------------------------------------- the indexer -----
    void layer_norm(const float * x, const float * w, const float * b, float * y,
                    int ne0, int n_rows, float eps) override {
        for (int r = 0; r < n_rows; ++r) {
            const float * xr = x + (size_t) r * ne0;
            float * yr = y + (size_t) r * ne0;
            double sum = 0.0;
            for (int i = 0; i < ne0; ++i) sum += (double) xr[i];
            const float mean = (float) (sum / ne0);
            double var = 0.0;
            for (int i = 0; i < ne0; ++i) { const float d = xr[i] - mean; yr[i] = d; var += (double) (d * d); }
            const float scale = 1.0f / sqrtf((float) (var / ne0) + eps);
            for (int i = 0; i < ne0; ++i) {
                float v = yr[i] * scale;                    // ggml_norm
                if (w) v = v * w[i];                        // build_norm: ggml_mul
                if (b) v = v + b[i];                        //             ggml_add
                yr[i] = v;
            }
        }
    }

    void store_f16(const float * src, int src_stride, uint16_t * dst, int dst_stride, int dst_off,
                   int width, int ring_rows, int pos, int T) override {
        for (int t = 0; t < T; ++t) {
            const int p = pos + t;
            const int row = ring_rows > 0 ? p % ring_rows : p;
            uint16_t * d = dst + (size_t) row * dst_stride + dst_off;
            const float * s = src + (size_t) t * src_stride;
            for (int i = 0; i < width; ++i) d[i] = f2h(s[i]);
        }
    }

    void idx_pool(const uint16_t * kg, int kg_rows, const float * ape, uint16_t * pooled,
                  int pos, int T) override {
        for (int t = 0; t < T; ++t) {
            const int p = pos + t;
            if ((p + 1) % KPOOL != 0) continue;
            const int b = p / KPOOL;
            for (int d = 0; d < IDX_DIM; ++d) {
                float key[KPOOL], sc[KPOOL], e[KPOOL];
                float mx = -INFINITY;
                for (int j = 0; j < KPOOL; ++j) {
                    const uint16_t * cell = kg + (size_t) ((b * KPOOL + j) % kg_rows) * (2 * IDX_DIM);
                    key[j] = h2f(cell[d]);
                    sc[j]  = h2f(cell[IDX_DIM + d]) + ape[(size_t) j * IDX_DIM + d];   // gate_t + ape
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0.0;                           // ggml_soft_max over the slot axis
                for (int j = 0; j < KPOOL; ++j) { e[j] = expf(sc[j] - mx); sum += (double) e[j]; }
                const float inv = (float) (1.0 / sum);
                double acc = 0.0;                           // sum_rows(keys * probs)
                for (int j = 0; j < KPOOL; ++j) acc += (double) (key[j] * (e[j] * inv));
                pooled[(size_t) b * IDX_DIM + d] = f2h((float) acc);
            }
        }
    }

    void idx_scores(const float * q, const float * w, const uint16_t * pooled, int pos,
                    float * scores, int stride, int T) override {
        for (int t = 0; t < T; ++t) {
            const int n = (pos + t + 1) / KPOOL;
            const float * qt = q + (size_t) t * IDX_N_HEAD * IDX_DIM;
            const float * wt = w + (size_t) t * IDX_N_HEAD;
            float k[IDX_DIM];
            for (int b = 0; b < n; ++b) {
                for (int i = 0; i < IDX_DIM; ++i) k[i] = h2f(pooled[(size_t) b * IDX_DIM + i]);
                double score = 0.0;
                for (int h = 0; h < IDX_N_HEAD; ++h) {
                    double qk = 0.0;
                    const float * qh = qt + (size_t) h * IDX_DIM;
                    for (int i = 0; i < IDX_DIM; ++i) qk += (double) qh[i] * k[i];
                    score += (double) (std::max((float) qk, 0.0f) * wt[h]);   // relu, THEN * w
                }
                scores[(size_t) t * stride + b] = (float) score;  // + pool_bias 0 (complete, visible)
            }
        }
    }

    void idx_topk(const float * scores, int stride, int pos, int * out, int T) override {
        for (int t = 0; t < T; ++t) {
            const int n = (pos + t + 1) / KPOOL;
            int * o = out + (size_t) t * IDX_TOP_POOLS;
            if (n <= IDX_TOP_POOLS) { for (int i = 0; i < n; ++i) o[i] = i; continue; }
            const float * s = scores + (size_t) t * stride;
            std::vector<int> idx((size_t) n);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + IDX_TOP_POOLS, idx.end(),
                              [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
            std::sort(idx.begin(), idx.begin() + IDX_TOP_POOLS);
            std::copy(idx.begin(), idx.begin() + IDX_TOP_POOLS, o);
        }
    }

    // ------------------------------------------------------- attention ------
    void mla_attn(const float * q, const uint16_t * kv, int pos, const int * sel, float scale,
                  float * out, int T) override {
        for (int t = 0; t < T; ++t) {
            const int p = pos + t;
            const int n_vis = (p + 1) / KPOOL;
            std::vector<int> cells;
            if (!sel || n_vis <= IDX_TOP_POOLS) {
                cells.resize((size_t) p + 1);
                std::iota(cells.begin(), cells.end(), 0);
            } else {
                const int * st = sel + (size_t) t * IDX_TOP_POOLS;
                for (int i = 0; i < IDX_TOP_POOLS; ++i)
                    for (int j = 0; j < KPOOL; ++j) cells.push_back(st[i] * KPOOL + j);
                for (int c = n_vis * KPOOL; c <= p; ++c) cells.push_back(c);   // the tail
            }
            attn_1(q + (size_t) t * N_HEAD * KV_LORA, kv, cells, scale, out + (size_t) t * N_HEAD * KV_LORA);
        }
    }
    void attn_1(const float * q, const uint16_t * kv, const std::vector<int> & cells, float scale, float * out) {
        const size_t n = cells.size();
        std::vector<float> K(n * KV_LORA);                // the shared latent rows, once for 64 heads
        for (size_t j = 0; j < n; ++j) {
            const uint16_t * src = kv + (size_t) cells[j] * KV_LORA;
            for (int i = 0; i < KV_LORA; ++i) K[j * KV_LORA + i] = h2f(src[i]);
        }
        std::vector<float> s(n);
        std::vector<double> acc(KV_LORA);
        for (int h = 0; h < N_HEAD; ++h) {
            const float * qh = q + (size_t) h * KV_LORA;
            float mx = -INFINITY;
            for (size_t j = 0; j < n; ++j) {
                double d = 0.0;
                const float * kj = K.data() + j * KV_LORA;
                for (int i = 0; i < KV_LORA; ++i) d += (double) qh[i] * kj[i];
                s[j] = (float) d * scale;
                mx = std::max(mx, s[j]);
            }
            double den = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (size_t j = 0; j < n; ++j) {
                const float pj = expf(s[j] - mx);
                den += pj;
                const float * kj = K.data() + j * KV_LORA;
                for (int i = 0; i < KV_LORA; ++i) acc[i] += (double) pj * kj[i];
            }
            float * oh = out + (size_t) h * KV_LORA;
            for (int i = 0; i < KV_LORA; ++i) oh[i] = (float) (acc[i] / den);
        }
    }

    // ------------------------------------------------------------ MoE -------
    void router(const float * logits, const float * bias, float * probs, float * probs_b, int * ids,
                float * w_raw, float * w_norm, float * w_scaled, const int * miss, uint32_t * stats,
                int T) override {
        for (int t = 0; t < T; ++t) {
            const float * lg = logits + (size_t) t * N_EXPERT;
            float * pr = probs + (size_t) t * N_EXPERT, * pb = probs_b + (size_t) t * N_EXPERT;
            int * id = ids + (size_t) t * N_EXPERT_USED;
            float * wr = w_raw + (size_t) t * N_EXPERT_USED, * wn = w_norm + (size_t) t * N_EXPERT_USED;
            float * ws = w_scaled + (size_t) t * N_EXPERT_USED;
            for (int e = 0; e < N_EXPERT; ++e) { pr[e] = sigmoid_(lg[e]); pb[e] = pr[e] + bias[e]; }
            std::vector<int> idx(N_EXPERT);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + N_EXPERT_USED, idx.end(),
                              [&](int a, int b) { return pb[a] > pb[b] || (pb[a] == pb[b] && a < b); });
            double sum = 0.0;
            for (int k = 0; k < N_EXPERT_USED; ++k) { id[k] = idx[k]; wr[k] = pr[id[k]]; sum += (double) wr[k]; }
            const float fs = std::max((float) sum, 6.103515625e-5f);   // ggml_clamp(sum, 6.1e-5, inf)
            for (int k = 0; k < N_EXPERT_USED; ++k) {
                wn[k] = wr[k] / fs;                                     // ggml_div
                ws[k] = wn[k] * EXPERT_WEIGHTS_SCALE;                   // ggml_scale
            }
            if (stats)                                                  // --adapt: the route counters
                for (int k = 0; k < N_EXPERT_USED; ++k) {
                    ++stats[id[k]];
                    stats[G_ADAPT_MISS] += miss[id[k]] != 0;
                    ++stats[G_ADAPT_PICKS];
                }
        }
    }

    static Mat expert_mat(const void * base, int type, size_t row_bytes, int K, int rows) {
        Mat m;
        m.base = base; m.type = type; m.row_bytes = row_bytes; m.K = K; m.rows = rows;
        m.name = "expert";
        return m;
    }
    // ---- the routed experts, split by a LinkPlan (see glm5_ops.h) -----------
    // On the CPU arm every "card" is this process: a helper's rows are the
    // same GEMVs on the same bytes, written into the owner's buffers, so the
    // three-link split is bit-identical to the owner-only path by construction.
    struct Packet { std::vector<float> x; int ids[N_EXPERT_USED], mk[N_EXPERT_USED], assign[N_EXPERT_USED]; };
    Packet pk_[2];
    void moe_plan(const ExpertTable & t, const int * ids, const float * x, int me, const LinkPlan & lp,
                  int salt, int bank) override {
        Packet & p = pk_[bank & 1];
        p.x.assign(x, x + t.K_gu);
        int j = 0;
        for (int k = 0; k < N_EXPERT_USED; ++k) {
            p.ids[k] = ids[k];
            p.mk[k] = t.miss_bytes[ids[k]] != 0;
            p.assign[k] = p.mk[k] ? glm_link_card(lp, me, j++, salt) : me;
        }
    }
    void rows_for(const ExpertTable & v, const Packet & p, int me, float * yg, float * yu, float * yh,
                  float * yd, float limit) {
        for (int k = 0; k < N_EXPERT_USED; ++k) {
            if (p.assign[k] != me) continue;
            const int e = p.ids[k];
            float * gk = yg + (size_t) k * v.rows_gu, * uk = yu + (size_t) k * v.rows_gu;
            float * hk = yh + (size_t) k * v.rows_gu;
            be_.gemv(expert_mat(v.up[e],   v.type_gu, v.row_gu, v.K_gu, v.rows_gu), p.x.data(), uk);
            be_.gemv(expert_mat(v.gate[e], v.type_gu, v.row_gu, v.K_gu, v.rows_gu), p.x.data(), gk);
            for (int i = 0; i < v.rows_gu; ++i) {                        // ds4 swiglu_clamp, verbatim
                const float g = std::min(gk[i], limit);
                const float u = std::min(std::max(uk[i], -limit), limit);
                hk[i] = g / (1.f + expf(-g)) * u;
            }
            be_.gemv(expert_mat(v.down[e], v.type_d, v.row_d, v.K_d, v.rows_d), hk, yd + (size_t) k * v.rows_d);
        }
    }
    void moe_help(Glm5Ops & owner, const ExpertTable & view, int me, int bank, float * yg, float * yu,
                  float * yh, float * yd, float limit) override {
        rows_for(view, static_cast<Glm5CpuOps &>(owner).pk_[bank & 1], me, yg, yu, yh, yd, limit);
    }
    void moe_stage_own(const ExpertTable &, int, int) override {}
    void moe_compute_own(const ExpertTable & t, int me, int bank, float * yg, float * yu, float * yh,
                         float * yd, float limit) override {
        rows_for(t, pk_[bank & 1], me, yg, yu, yh, yd, limit);
    }

    // per-head slices of one [K, rows, n_heads] tensor: the same GEMV per head
    void head_gemv(const Mat & W, int n_heads, const float * x, int x_stride, float * y) override {
        for (int h = 0; h < n_heads; ++h) {
            Mat m = W;
            m.base = (const unsigned char *) W.base + (size_t) h * W.rows * W.row_bytes;
            be_.gemv(m, x + (size_t) h * x_stride, y + (size_t) h * W.rows);
        }
    }

    // build_hc_mean: acc = H0; acc += H1; acc += H2; acc += H3; * (1/4)
    void hc_mean(const float * H, float * out, int T) override {
        for (int t = 0; t < T; ++t) {
            const float * Ht = H + (size_t) t * HC_DIM;
            float * o = out + (size_t) t * N_EMBD;
            for (int i = 0; i < N_EMBD; ++i) {
                float a = Ht[i];
                for (int s = 1; s < HC; ++s) a = a + Ht[(size_t) s * N_EMBD + i];
                o[i] = a * (1.0f / HC);
            }
        }
    }

private:
    Backend & be_;
};

} // namespace

Glm5Ops * make_glm5_cpu_ops(Backend & be) { return new Glm5CpuOps(be); }

} // namespace glm5
} // namespace fk
