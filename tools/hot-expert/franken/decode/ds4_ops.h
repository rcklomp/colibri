// tools/hot-expert/franken/decode/ds4_ops.h
//
// The DeepSeek-V4-specific operations of the decode token, as an interface,
// in the same discipline as decode_backend.h: ds4_graph.cpp holds THE MATH
// (the order of the ops, which buffer feeds which) and never dereferences a
// buffer; it calls these, plus the generic half of Backend (alloc, GEMV,
// expert GEMV, rms_norm_mul, copy, binary, download).
//
//   Ds4CpuOps (ds4_cpu.cpp)   plain host C++ -- the oracle arm, and the only
//                             implementation today (L5 step 1).
//   a GPU implementation      L5 step 2: HIP kernels behind the same calls.
//
// Every formula below is cited to the line of the reference it ports:
// ~/src/llama-glm53/src/models/deepseek4.cpp (DS4), the fused ops in
// ggml/src/ggml-cpu/ops.cpp, and llama-kv-cache-dsv4.cpp's compression plan.
// Buffers are token-major and one token wide (T == 1): the batched prefill
// is a later step, and nothing here is shaped against it.

#pragma once

#include <cstddef>
#include <cstdint>

namespace fk {
namespace ds4 {

// ggml_rope_ext's float parameters, as deepseek4.cpp passes them per layer
// (build_attention_impl: compress-rope layers use the yarn set, ratio-0
// layers base 10000 and no scaling).
struct RopeParams {
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;
    int   n_ctx_orig  = 0;
};

class Ds4Ops {
public:
    virtual ~Ds4Ops() {}

    // -- hyper-connections ---------------------------------------------------
    // build_hc_pre's split of mixes[24] (ops.cpp dsv4_hc_comb + deepseek4.cpp
    // build_hc_pre): pre[h] = sigmoid(m[h]*s[0] + b[h]) + eps,
    // post[h] = 2*sigmoid(m[4+h]*s[1] + b[4+h]), comb[dst + 4*src] = the
    // Sinkhorn-normalised softmax of m[8..24]*s[2] + b[8..24].
    virtual void hc_split(const float * mixes, const float * scale3, const float * base24,
                          float * pre, float * post, float * comb, float eps, int iters) = 0;
    // build_hc_head: pre[h] = sigmoid(m[h]*s[0] + b[h]) + eps (4 values).
    virtual void hc_head_pre(const float * mixes4, const float * scale1, const float * base4,
                             float * pre, float eps) = 0;
    // ggml_dsv4_hc_pre: out[i] = sum_h H[h][i] * w[h], h in order.
    virtual void hc_weighted_sum(const float * H, const float * w, float * out) = 0;
    // ggml_dsv4_hc_post: Hout[d][i] = x[i]*post[d] + sum_s H[s][i]*comb[d + 4*s].
    virtual void hc_post(const float * x, const float * H, const float * post,
                         const float * comb, float * Hout) = 0;

    // -- rope on the TAIL of each row (ggml_rope_ext + ggml_rope_set_offset) --
    // `n_rows` rows of `row_len`; dims [row_len-N_ROT, row_len) are rotated as
    // GGML_ROPE_TYPE_NORM adjacent pairs at position `pos`. `inverse` is
    // ggml_rope_ext_back (the de-rope of the attention output).
    virtual void rope_tail(float * x, int n_rows, int row_len, int pos,
                           const RopeParams & rp, bool inverse) = 0;
    // ggml's CPU FWHT (ops.cpp ggml_compute_forward_fwht_f32): the matmul by
    // the orthonormal Walsh-Hadamard matrix the lid cache applies (k_rot).
    virtual void fwht(float * x, int n_rows, int n) = 0;
    // f32 -> f16 (IEEE bits), the cache store; ggml_fp32_to_fp16 rounding.
    virtual void to_f16(const float * x, uint16_t * y, int n) = 0;

    // -- compressor pooling (build_*_compressed_kv_from_state) ---------------
    // HCA, ratio R: out[d] = sum_j softmax_j(sc[j][d]) * kv[j][d] over the R
    // tokens of block `blk`, read from ring slots (blk*R + j) % ring_rows of
    // rows `width` == d_out wide.
    virtual void comp_pool(const float * ring_kv, const float * ring_sc, int ring_rows,
                           int ratio, int d_out, int blk, float * out) = 0;
    // CSA / lid, ratio R, overlapped: 2R candidates per dimension -- the
    // PREVIOUS block's R tokens through the FIRST half of their 2*d_out-wide
    // rows, then this block's R tokens through the SECOND half; block 0's
    // previous half is the synthetic zero / -inf row.
    virtual void comp_pool_overlap(const float * ring_kv, const float * ring_sc, int ring_rows,
                                   int ratio, int d_out, int blk, float * out) = 0;

    // -- the lightning indexer (ggml_lightning_indexer) ----------------------
    // score[b] = sum_h max(q_h . k_b, 0) * w_h over b < n_blocks, keys f16.
    virtual void lid_scores(const float * q, const float * w, const uint16_t * keys,
                            int n_blocks, float * scores) = 0;
    // The top min(k, n) of scores[0..n) by value, descending (ggml_top_k);
    // returns the count.
    virtual int  topk(const float * scores, int n, int k, int * out) = 0;

    // -- attention: 64 query heads against one shared K=V head --------------
    // Keys: the raw window (ring slots of positions raw_pos0 .. raw_pos0 +
    // n_raw - 1, slot = pos % N_SWA) then `n_comp` compressed rows (ids, or
    // 0..n_comp-1 when `comp_ids` is null). scores * scale, the per-head sink
    // in the denominator (ggml flash_attn_ext's sinks), V = K.
    virtual void attn(const float * q, const uint16_t * raw_ring, int raw_pos0, int n_raw,
                      const uint16_t * comp, const int * comp_ids, int n_comp,
                      const float * sinks, float scale, float * out) = 0;

    // -- MoE (build_moe_ffn, sqrt-softplus gating) ---------------------------
    // probs = sqrt(softplus(logits)); ids = top-6 of probs + bias, or the
    // hash row when `hash_ids` is given; w = probs[ids] / max(sum, 6.1e-5)
    // * 1.5. Also returns probs and the biased probs for the taps.
    virtual void router(const float * logits, const float * bias, const int32_t * hash_ids,
                        float * probs, float * probs_biased, int * ids, float * w_raw,
                        float * w_norm, float * w_scaled) = 0;
    // ggml_swiglu_clamp: g = min(gate, L); u = clamp(up, -L, L); h = g*sigmoid(g)*u.
    virtual void swiglu_clamp(const float * gate, const float * up, float * h, int n,
                              float limit) = 0;
    // weighted[e][i] = y[e][i]*w[e]; out = ((w0 + w1) + w2) + ... in expert order.
    virtual void moe_accum(const float * y, const float * w, int n_used, int n,
                           float * weighted, float * out) = 0;
};

Ds4Ops * make_ds4_cpu_ops();

} // namespace ds4
} // namespace fk
