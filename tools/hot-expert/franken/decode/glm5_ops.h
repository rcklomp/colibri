// tools/hot-expert/franken/decode/glm5_ops.h
//
// The GLM-5.3-Flash-specific operations of the decode token, as an
// interface, in the discipline of ds4_ops.h: glm5_graph.cpp holds THE MATH
// and never dereferences a buffer; it calls these, the generic half of
// Backend (rms_norm_mul, l2_norm, gated_rms_norm, unary, binary, scale,
// upload/download, argmax, boundary P2P) and -- for what GLM shares with
// DeepSeek-V4 bit for bit (the four-stream hyper-connection mix, the GEMV in
// any format, swiglu_clamp, the weighted expert sum, the adaptive-placement
// I/O: route-counter snapshots, table edits, swap copies) -- Ds4Ops.
//
//   Glm5CpuOps (glm5_cpu.cpp)   plain host C++, the oracle arm.
//   Glm5GpuOps (glm5_gpu.inc)   HIP kernels, compiled into decode_gpu.hip's
//                               translation unit so they run on the backend's
//                               own stream (GLM step 4). One instance a card.
//
// Every pointer crossing this interface is a BACKEND pointer. Every count an
// op needs is a closed form of the position `pos` (the FIRST row's; row t is
// at pos + t). The decode token is T == 1; the prompt is fed token by token
// (both GLM attention kinds are causal per token, so that is exact).

#pragma once

#include <cstddef>
#include <cstdint>

#include "decode_backend.h"   // Mat
#include "ds4_ops.h"          // ExpertTable (per-expert addresses; N_EXPERT entries each)

namespace fk {
namespace glm5 {

using ds4::ExpertTable;

// --adapt: the router's per-layer route counters. [e] how often expert e was
// chosen, [G_ADAPT_MISS] choices that found their expert host-side,
// [G_ADAPT_PICKS] every choice (the DeepSeek layout at 288 experts).
constexpr int G_ADAPT_STRIDE = 292;
constexpr int G_ADAPT_MISS   = 288;
constexpr int G_ADAPT_PICKS  = 289;

// The THREE-LINK miss path (GLM5.md section 9). After the router, the owner
// card of a layer writes a PLAN: x, the 8 ids, per slot whether its expert is
// host-side ("missed") and which card computes it -- a resident slot on the
// owner, a missed one on one of the `n_links` cards by a fixed 11-long
// pattern that gives the lone card 5 of every 11 missed slabs and each card
// of the shared-link pair 3 (45 / 27 / 27 %). Every card that got slots
// fetches ITS missed slabs over ITS OWN host link into its own VRAM and
// computes them there; only x (16 KB) goes out and the finished rows
// (gate, up, swiglu, down: 40 KB a slot) come back, by P2P. n_links == 1 is
// the owner-card path (DeepSeek step 3's staging ring). Every slot runs the
// same kernel on the same bytes wherever it runs, so the result is
// bit-identical to n_links == 1.
struct LinkPlan {
    int n_links = 1;        // 1: owner only; 3: the three links
    int cards[3] = {0, 1, 2};
    int lone = 2;           // the card on its own host link (48:00.0, HIP index 2)
};

// The 11-long assignment pattern of missed slots (index into {lone, A, B},
// A < B the other two cards): the lone card 5 of 11, A and B 3 each. The
// j-th missed slot of a layer goes to pattern[(salt + j) % 11], salt =
// (7 * layer) % 11, so the share holds over a token, not only a layer.
constexpr int GLM_LINK_PAT[11] = {0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 0};
inline int glm_link_card(const LinkPlan & lp, int owner, int j, int salt) {
    if (lp.n_links <= 1) return owner;
    int other[2], n = 0;
    for (int c = 0; c < 3; ++c) if (lp.cards[c] != lp.lone) other[n++] = lp.cards[c];
    const int w = GLM_LINK_PAT[(salt + j) % 11];
    return w == 0 ? lp.lone : other[w - 1];
}

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
    // g = lower_bound * sigmoid(-(ssm_a[h] * (fb + dt_bias))), per channel
    virtual void kda_gate(const float * fb, const float * dt_bias, const float * ssm_a,
                          float lower_bound, float * g, int T = 1) = 0;
    // The KDA recurrence (ggml_gated_delta_net's kda branch), token by token:
    // S[i][:] *= exp(g[i]); d[j] = (v[j] - sum_i S[i][j] k[i]) * beta;
    // S[i][j] += k[i] d[j]; out[j] = sum_i S[i][j] q[i] / sqrt(KDA_DIM).
    // state [N_HEAD][KDA_DIM (value j)][KDA_DIM (key i)]
    virtual void kda_step(float * state, const float * q, const float * k, const float * v,
                          const float * g, const float * beta, float * out, int qkv_stride,
                          int T = 1) = 0;

    // -- the pooled lightning indexer (build_indexer) --------------------------
    virtual void layer_norm(const float * x, const float * w, const float * b, float * y,
                            int ne0, int n_rows, float eps) = 0;
    // f16 row store: dst + ((pos + t) % ring_rows) * dst_stride + dst_off
    virtual void store_f16(const float * src, int src_stride, uint16_t * dst, int dst_stride,
                           int dst_off, int width, int ring_rows, int pos, int T = 1) = 0;
    // pool b = p / KPOOL when p completes it: softmax_j(gate_j + ape[j]) per dim
    virtual void idx_pool(const uint16_t * kg, int kg_rows, const float * ape, uint16_t * pooled,
                          int pos, int T = 1) = 0;
    // score_b = sum_h relu(q_h . pooled_b) * w_h over the (p + 1) / KPOOL visible pools
    virtual void idx_scores(const float * q, const float * w, const uint16_t * pooled, int pos,
                            float * scores, int score_stride, int T = 1) = 0;
    // the top IDX_TOP_POOLS pools as a SET in ascending order; identity when n <= k
    virtual void idx_topk(const float * scores, int score_stride, int pos, int * out, int T = 1) = 0;

    // -- absorbed MLA ----------------------------------------------------------
    // y[h*W.rows + r] = W_h[r] . x[h*x_stride ..], W_h = head h's slice (the
    // [K, rows, n_heads] tensor's heads are contiguous: one batched GEMV)
    virtual void head_gemv(const Mat & W_head0, int n_heads, const float * x, int x_stride, float * y) = 0;
    // Keys of the row at q: every cell <= q while the visible pools are <= 512
    // (or sel null); otherwise the cells of the selected pools then the tail.
    virtual void mla_attn(const float * q, const uint16_t * kv, int pos, const int * sel,
                          float scale, float * out, int T = 1) = 0;
    virtual void reserve(int ctx) { (void) ctx; }

    // -- MoE (build_moe_ffn: sigmoid gating, biased top-8, normalised, x2.5) --
    // `stats` (--adapt, else null): the layer's G_ADAPT_STRIDE route counters;
    // `miss` the table's per-expert miss bytes. Neither changes the routing.
    virtual void router(const float * logits, const float * bias, float * probs, float * probs_biased,
                        int * ids, float * w_raw, float * w_norm, float * w_scaled,
                        const int * miss, uint32_t * stats, int T = 1) = 0;

    // The routed experts of one layer, T == 1, split across cards (LinkPlan).
    // Call order a layer: plan (owner) -> help (every other card of the plan)
    // -> stage_own (owner) -> [the shared expert] -> compute_own (owner) ->
    // join (owner, each helper) -> moe_accum. yg/yu/yh [8][rows_gu], yd
    // [8][rows_d] are the OWNER's buffers; a helper writes its rows into them.
    // `view` is the layer's table as THIS card reads it (the owner: its own
    // table; a helper: the host mirror as mapped into this card). `bank`
    // alternates by layer, so a layer's rings are not reused by the next.
    virtual void moe_reserve(size_t max_slab_bytes) { (void) max_slab_bytes; }
    virtual void moe_plan(const ExpertTable & t, const int * ids, const float * x, int me,
                          const LinkPlan & lp, int salt, int bank) = 0;
    virtual void moe_help(Glm5Ops & owner, const ExpertTable & view, int me, int bank,
                          float * yg, float * yu, float * yh, float * yd, float limit) = 0;
    virtual void moe_stage_own(const ExpertTable & t, int me, int bank) = 0;
    virtual void moe_compute_own(const ExpertTable & t, int me, int bank, float * yg, float * yu,
                                 float * yh, float * yd, float limit) = 0;
    virtual void moe_join(Glm5Ops & helper) { (void) helper; }

    // -- the head: the UNWEIGHTED mean of the four streams (build_hc_mean) -----
    virtual void hc_mean(const float * H, float * out, int T = 1) = 0;

    // -- memory ------------------------------------------------------------------
    // The address THIS card reads a pinned, portable, mapped host allocation
    // at (the CPU arm: the host address itself).
    virtual const void * host_view(const void * host_base) { return host_base; }
    virtual void set_profile(bool on) { (void) on; }
};

Glm5Ops * make_glm5_cpu_ops(Backend & be);
// decode_gpu.hip (the HIP build) or decode_nogpu.cpp (throws)
Glm5Ops * make_glm5_gpu_ops(Backend & be);

} // namespace glm5
} // namespace fk
