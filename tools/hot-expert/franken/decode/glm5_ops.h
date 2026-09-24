// tools/hot-expert/franken/decode/glm5_ops.h
//
// The GLM-5.3-Flash-specific operations of the decode token, as an
// interface, in the discipline of ds4_ops.h: glm5_graph.cpp holds THE MATH
// and never dereferences a buffer; it calls these, the generic half of
// Backend (rms_norm_mul, l2_norm, gated_rms_norm, unary, binary, scale,
// upload/download, argmax) and -- for what GLM shares with DeepSeek-V4 bit
// for bit (the four-stream hyper-connection mix, the GEMV in any format,
// swiglu_clamp, the weighted expert sum) -- Ds4Ops.
//
//   Glm5CpuOps (glm5_cpu.cpp)   plain host C++, the oracle arm (L5 GLM step 2).
//   Glm5GpuOps                  not built yet (GLM5.md section 8 lists the kernels).
//
// Every pointer crossing this interface is a BACKEND pointer. Every count an
// op needs is a closed form of the position `pos` (the FIRST row's; row t is
// at pos + t), so a GPU implementation can read the position from a device
// int as Ds4GpuOps does and one captured graph serves a position class.
//
// ROWS: every per-token op takes T rows, token-major. Step 2 feeds the prompt
// one token at a time (T == 1), as DeepSeek step 1 did: both GLM attention
// kinds are causal per token, so that is exact.

#pragma once

#include <cstddef>
#include <cstdint>

#include "decode_backend.h"   // Mat
#include "ds4_ops.h"          // ExpertTable (per-expert addresses; N_EXPERT entries each)

namespace fk {
namespace glm5 {

using ds4::ExpertTable;

class Glm5Ops {
public:
    virtual ~Glm5Ops() {}

    // -- KDA (build_kda_layer) ------------------------------------------------
    // ggml_ssm_conv over [the layer's last KDA_CONV-1 inputs | the T new rows]
    // then SiLU: out[t][c] = silu(sum_k win[t+k][c] * w_c[k]), c over the
    // concatenated q|k|v channels (w_q, w_k, w_v: [KDA_INNER][KDA_CONV] each).
    // `state` ([KDA_CONV-1][KDA_QKV], oldest first) is advanced past the T rows.
    virtual void kda_conv(const float * qkv, float * state, const float * w_q, const float * w_k,
                          const float * w_v, float * out, int T = 1) = 0;
    // g = lower_bound * sigmoid(-(ssm_a[h] * (fb + dt_bias))), per channel:
    // fb [T][KDA_INNER], dt_bias [KDA_INNER], ssm_a [N_HEAD] (holds -exp(A_log))
    virtual void kda_gate(const float * fb, const float * dt_bias, const float * ssm_a,
                          float lower_bound, float * g, int T = 1) = 0;
    // The KDA recurrence, ggml_gated_delta_net's kda branch, token by token:
    // S[i][:] *= exp(g[i]); d[j] = (v[j] - sum_i S[i][j] k[i]) * beta;
    // S[i][j] += k[i] d[j]; out[j] = sum_i S[i][j] q[i] / sqrt(KDA_DIM).
    // state [N_HEAD][KDA_DIM (value j)][KDA_DIM (key i)]; q/k/v rows at
    // q + t*qkv_stride (head h at + h*KDA_DIM); g [T][KDA_INNER]; beta [T][N_HEAD];
    // out [T][KDA_INNER].
    virtual void kda_step(float * state, const float * q, const float * k, const float * v,
                          const float * g, const float * beta, float * out, int qkv_stride,
                          int T = 1) = 0;

    // -- the pooled lightning indexer (build_indexer) --------------------------
    // ggml_norm (LayerNorm) then * w + b, per row of ne0
    virtual void layer_norm(const float * x, const float * w, const float * b, float * y,
                            int ne0, int n_rows, float eps) = 0;
    // f16 row store: dst + ((pos + t) % ring_rows) * dst_stride + dst_off,
    // `width` values from src + t * src_stride. ring_rows 0 = no wrap.
    virtual void store_f16(const float * src, int src_stride, uint16_t * dst, int dst_stride,
                           int dst_off, int width, int ring_rows, int pos, int T = 1) = 0;
    // For every row whose position p completes a pool ((p + 1) % KPOOL == 0):
    // pool b = p / KPOOL from its KPOOL member cells' f16 key|gate rows
    // (kg ring, cell p at row p % kg_rows, [key 128 | gate 128]):
    // probs = softmax_j(gate_j + ape[j]) per dim, pooled = sum_j key_j * probs_j,
    // stored f16 at pooled[b][0..IDX_DIM). ape [KPOOL][IDX_DIM].
    virtual void idx_pool(const uint16_t * kg, int kg_rows, const float * ape, uint16_t * pooled,
                          int pos, int T = 1) = 0;
    // score_b = sum_h relu(q_h . pooled_b) * w_h over the n = (p + 1) / KPOOL
    // complete pools visible to row t. q [T][IDX_N_HEAD * IDX_DIM], w [T][IDX_N_HEAD]
    // (already scaled), scores [T][score_stride].
    virtual void idx_scores(const float * q, const float * w, const uint16_t * pooled, int pos,
                            float * scores, int score_stride, int T = 1) = 0;
    // The top IDX_TOP_POOLS of the n visible pools as a SET, written in
    // ascending pool order (ties at the cut: lowest index first); n <= k
    // writes the identity 0..n-1. out [T][IDX_TOP_POOLS].
    virtual void idx_topk(const float * scores, int score_stride, int pos, int * out, int T = 1) = 0;

    // -- absorbed MLA attention (build_dsa_layer / build_attn_sparse) ----------
    // N_HEAD query heads of KV_LORA against one shared latent K = V row a cell
    // (f16 [ctx][KV_LORA]). Keys of row t at position q: every cell <= q while
    // the visible pools are <= IDX_TOP_POOLS (or sel == nullptr); otherwise the
    // KPOOL cells of each selected pool (sel [T][IDX_TOP_POOLS], ascending)
    // then the TAIL cells (q+1)/KPOOL*KPOOL .. q. out [T][N_HEAD][KV_LORA].
    virtual void mla_attn(const float * q, const uint16_t * kv, int pos, const int * sel,
                          float scale, float * out, int T = 1) = 0;

    // -- MoE (build_moe_ffn: sigmoid gating, biased top-8, normalised, x2.5) --
    // rows: logits/probs/probs_biased [T][N_EXPERT], ids/w_* [T][N_EXPERT_USED]
    virtual void router(const float * logits, const float * bias, float * probs, float * probs_biased,
                        int * ids, float * w_raw, float * w_norm, float * w_scaled, int T = 1) = 0;
    // y_gate / y_up [T*8][rows_gu]; assignment a = t*8 + k reads x + t*K_gu
    virtual void moe_gate_up(const ExpertTable & t, const int * ids, const float * x,
                             float * y_gate, float * y_up, int T = 1) = 0;
    // y [T*8][rows_d]; assignment a reads h + a*K_d
    virtual void moe_down(const ExpertTable & t, const int * ids, const float * h, float * y,
                          int T = 1) = 0;

    // -- the head: the UNWEIGHTED mean of the four streams (build_hc_mean) -----
    virtual void hc_mean(const float * H, float * out, int T = 1) = 0;
};

Glm5Ops * make_glm5_cpu_ops(Backend & be);

} // namespace glm5
} // namespace fk
