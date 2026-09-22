// tools/hot-expert/franken/decode/decode_graph.cpp -- see decode_graph.h.
//
// Line references below are to ~/src/llama-glm53/src/models/qwen4exp.cpp
// @ 39931761a unless another file is named.

#include "decode_graph.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace fk {

// -------------------------------------------------------------- recorder --

std::string Recorder::make_key(const char * name, int il, const char * suffix) {
    if (il < 0) return std::string(name);
    return std::string(name) + "-" + std::to_string(il) + suffix;
}

void Recorder::tap(Backend & be, const char * name, int il, const float * buf, size_t n,
                   const char * suffix) {
    if (!on_ || n == 0) return;
    TapValue v;
    v.name = name;
    v.key  = make_key(name, il, suffix);
    v.il   = il;
    v.data.resize(n);
    be.sync();
    be.download(v.data.data(), buf, n * sizeof(float));
    taps_[v.key] = std::move(v);
}

void Recorder::tap_ints(const char * name, int il, const std::vector<int> & v,
                        const char * suffix) {
    if (!on_) return;
    TapValue t;
    t.name = name;
    t.key  = make_key(name, il, suffix);
    t.il   = il;
    t.data.assign(v.begin(), v.end());
    taps_[t.key] = std::move(t);
}

const TapValue * Recorder::get(const std::string & key) const {
    auto it = taps_.find(key);
    return it == taps_.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------- runner --

DecodeRunner::DecodeRunner(Backend & be, DecodeModel & model, const DecodeConfig & cfg)
    : be_(be), model_(model), cfg_(cfg), eps_(model.rms_eps()) {

    auto A = [&](size_t n) { float * p = be_.alloc_f32(n); owned_.push_back(p); return p; };
    auto AI = [&](size_t n) { int * p = be_.alloc_i32(n); owned_.push_back(p); return p; };
    auto AR = [&](size_t b) { void * p = be_.alloc_raw(b); owned_.push_back(p); return p; };

    x_        = A(N_EMBD);
    res_hc_   = A(HC_DIM);
    xn_       = A(HC_DIM);
    lo_       = A(HC_LR);
    hgate_    = A(HC_DIM);
    mixed_    = A(N_EMBD);
    inject_   = A(HC);
    blk_      = A(N_EMBD);

    qkv_      = A(GDN_CONV_DIM);
    z_        = A(GDN_VAL_DIM);
    conv_     = A(GDN_CONV_DIM);
    qn_       = A(GDN_KEY_DIM);
    kn_       = A(GDN_KEY_DIM);
    alpha_    = A(GDN_V_HEADS);
    beta_     = A(GDN_V_HEADS);
    gexp_     = A(GDN_V_HEADS);
    abuf_     = A(GDN_V_HEADS);
    gdn_      = A(GDN_VAL_DIM);
    gnorm_    = A(GDN_VAL_DIM);

    qfull_    = A((size_t) HEAD_DIM * N_Q_HEADS * 2);
    qcur_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    gate_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    gsig_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    kcur_     = A((size_t) HEAD_DIM * N_KV_HEADS);
    vcur_     = A((size_t) HEAD_DIM * N_KV_HEADS);
    kqv_      = A((size_t) HEAD_DIM * N_Q_HEADS);

    const int max_blocks = (cfg_.ctx + QSA_RATIO - 1) / QSA_RATIO;
    idxq_      = A((size_t) IDX_N_HEADS * IDX_DIM);
    idxk_      = A(IDX_DIM);
    blkscore_  = A(max_blocks);
    blkbias_   = A(max_blocks);
    cellscore_ = A(cfg_.ctx);
    pool_raw_  = A((size_t) max_blocks * IDX_DIM);   // f32 shadow of the pooled cache,
    pool_rope_ = A((size_t) max_blocks * IDX_DIM);   // pre- and post-norm/rope (oracle taps)
    sel_       = AI(cfg_.ctx);

    logits_   = A(N_EXPERT);
    wts_      = A(N_EXPERT_USED);
    ids_      = AI(N_EXPERT_USED);
    ygate_    = A((size_t) N_EXPERT_USED * N_FF_EXP);
    yup_      = A((size_t) N_EXPERT_USED * N_FF_EXP);
    hmoe_     = A((size_t) N_EXPERT_USED * N_FF_EXP);
    eo_       = A((size_t) N_EXPERT_USED * N_EMBD);
    moeout_   = A(N_EMBD);
    shg_      = A(N_FF_SHEXP);
    shu_      = A(N_FF_SHEXP);
    shh_      = A(N_FF_SHEXP);
    shout_    = A(N_EMBD);
    shgate_   = A(1);

    plek_     = A(HC_DIM);
    plev_     = A(N_EMBD);
    pleq_     = A(HC_DIM);
    plegated_ = A(HC_DIM);
    plegate_  = A(HC);
    plenorm_  = A(HC_DIM);
    pleconv_  = A(HC_DIM);
    pleemb_   = A(N_EMBD);
    ple_ring_ = A((size_t)(PLE_CONV_HIST + 1) * HC_DIM);

    // Per-layer persistent state. alloc_f32 zeroes, which is exactly a fresh
    // sequence: llama.cpp clears the recurrent cells the same way.
    lstate_.resize(model_.il1() - model_.il0() + 1);
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & st = lstate_[il - model_.il0()];
        if (is_recurrent_layer(il)) {
            st.conv_ring = A((size_t) GDN_CONV_K * GDN_CONV_DIM);
            st.gdn_state = A((size_t) GDN_V_HEADS * GDN_STATE * GDN_STATE);
        } else {
            const size_t n_qs = (size_t) cfg_.ctx * N_KV_HEADS * HEAD_DIM;
            const size_t n_sc = (size_t) cfg_.ctx * N_KV_HEADS * (HEAD_DIM / 32);
            st.kqs = (int8_t   *) AR(n_qs);
            st.vqs = (int8_t   *) AR(n_qs);
            st.ksc = (uint16_t *) AR(n_sc * sizeof(uint16_t));
            st.vsc = (uint16_t *) AR(n_sc * sizeof(uint16_t));
            st.idx_raw = A((size_t) cfg_.ctx * IDX_DIM);
            st.idx_pooled = (uint16_t *) AR((size_t) max_blocks * IDX_DIM * sizeof(uint16_t));
        }
    }
}

DecodeRunner::~DecodeRunner() {
    for (void * p : owned_) be_.free_buf(p);
}

// --------------------------------------------------- build_hc_mix (218) ---
//
//   xn    = rms_norm(x over n_embd, per hc stream) * w_norm[hc_dim]
//   lo    = silu(w_down @ xn / hc)
//   gate  = sigmoid(w_up @ lo)
//   mixed = mean over the hc streams of (xn * gate)
//   inject= w_inject @ xn
void DecodeRunner::hc_mix(const LayerWeights & L, const float * w_norm, const Mat & down,
                          const Mat & up, const Mat & inj, const float * x,
                          float * out_mixed, float * out_inject, int il, Recorder & rec,
                          const char * suffix) {
    (void) L;
    be_.rms_norm_mul(x, w_norm, xn_, N_EMBD, HC, HC_DIM, eps_);
    rec.tap(be_, "hc_norm", il, xn_, HC_DIM, suffix);

    be_.gemv(down, xn_, lo_);
    be_.scale(lo_, 1.0f / (float) HC, lo_, HC_LR);
    be_.unary(FK_SILU, lo_, lo_, HC_LR);
    be_.gemv(up, lo_, hgate_);
    be_.unary(FK_SIGMOID, hgate_, hgate_, HC_DIM);
    rec.tap(be_, "hc_gate", il, hgate_, HC_DIM, suffix);

    be_.hc_collapse(xn_, hgate_, out_mixed);
    rec.tap(be_, "hc_mixed", il, out_mixed, N_EMBD, suffix);

    be_.gemv(inj, xn_, out_inject);
    rec.tap(be_, "hc_inject", il, out_inject, HC, suffix);

    if (cfg_.jitter > 0.0f) jitter(out_mixed, N_EMBD);
}

// Scales every element by (1 +- jitter), deterministically. The block input
// is the right place: it is what every projection of the layer reads, so a
// perturbation here is exactly what a different (equally valid) summation
// order upstream would have produced.
void DecodeRunner::jitter(float * buf, size_t n) {
    std::vector<float> h(n);
    be_.sync();
    be_.download(h.data(), buf, n * sizeof(float));
    std::uniform_real_distribution<float> d(-cfg_.jitter, cfg_.jitter);
    for (size_t i = 0; i < n; ++i) h[i] *= (1.0f + d(jrng_));
    be_.upload(buf, h.data(), n * sizeof(float));
}

// ------------------------------------------------------- build_ple (1137) --
void DecodeRunner::layer_ple(const LayerWeights & L, const float * ple_emb, Recorder & rec) {
    if (!ple_emb) throw std::runtime_error("the layer range includes the PLE layer but no PLE gather was given");
    be_.upload(pleemb_, ple_emb, N_EMBD * sizeof(float));
    rec.tap(be_, "ple_embd", -1, pleemb_, N_EMBD);

    be_.gemv(L.ple_key,   pleemb_, plek_);   // [hc_dim]
    be_.gemv(L.ple_value, pleemb_, plev_);   // [n_embd]

    // grouped_norm: rms over n_embd per hc stream, then the [hc_dim] gamma
    be_.rms_norm_mul(plek_,   L.ple_norm_key,   plek_, N_EMBD, HC, HC_DIM, eps_);
    be_.rms_norm_mul(res_hc_, L.ple_norm_query, pleq_, N_EMBD, HC, HC_DIM, eps_);

    be_.ple_gate(plek_, pleq_, plev_, plegated_, plegate_);
    rec.tap(be_, "ple_gate",         L.il, plegate_,  HC);
    rec.tap(be_, "ple_gated_value",  L.il, plegated_, HC_DIM);

    be_.rms_norm_mul(plegated_, L.ple_norm_conv, plenorm_, N_EMBD, HC, HC_DIM, eps_);

    ple_head_ = (ple_head_ + 1) % (PLE_CONV_HIST + 1);
    be_.copy(ple_ring_ + (size_t) ple_head_ * HC_DIM, plenorm_, HC_DIM);
    be_.ple_conv_silu(ple_ring_, ple_head_, L.ple_conv1d, pleconv_);
    rec.tap(be_, "ple_conv_out", L.il, pleconv_, HC_DIM);

    be_.binary(FK_ADD, res_hc_, plegated_, res_hc_, HC_DIM);
    be_.binary(FK_ADD, res_hc_, pleconv_,  res_hc_, HC_DIM);
}

// ---------------------------------------- build_layer_attn_linear (793) ---
void DecodeRunner::layer_gdn(const LayerWeights & L, LayerState & st, int il, Recorder & rec) {
    be_.gemv(L.ssm_qkv, mixed_, qkv_);
    rec.tap(be_, "linear_attn_qkv_mixed", il, qkv_, GDN_CONV_DIM);

    be_.gemv(L.ssm_gate, mixed_, z_);
    rec.tap(be_, "z", il, z_, GDN_VAL_DIM);

    be_.gemv(L.ssm_beta, mixed_, beta_);
    rec.tap(be_, "beta", il, beta_, GDN_V_HEADS);
    be_.unary(FK_SIGMOID, beta_, beta_, GDN_V_HEADS);
    rec.tap(be_, "beta_sigmoid", il, beta_, GDN_V_HEADS);

    be_.gemv(L.ssm_alpha, mixed_, alpha_);
    rec.tap(be_, "alpha", il, alpha_, GDN_V_HEADS);
    be_.binary(FK_ADD, alpha_, L.ssm_dt, abuf_, GDN_V_HEADS);
    be_.unary(FK_SOFTPLUS, abuf_, abuf_, GDN_V_HEADS);
    rec.tap(be_, "a_softplus", il, abuf_, GDN_V_HEADS);

    // gate = softplus(alpha + dt) * ssm_a, where ssm_a already holds
    // -exp(A_log) (qwen4exp.cpp:831's comment). The delta rule decays by
    // exp(gate) -- ops.cpp:11012's ggml_vec_scale_f32(.., expf(g_d[0])).
    be_.binary(FK_MUL, abuf_, L.ssm_a, gexp_, GDN_V_HEADS);
    rec.tap(be_, "gate", il, gexp_, GDN_V_HEADS);
    be_.unary(FK_EXP, gexp_, gexp_, GDN_V_HEADS);

    // cb(state, "state_predelta") is the state as read from the cache, i.e.
    // BEFORE this token's decay and update. Its ne2 is the HEAD axis, so the
    // dump's "last column" is head GDN_V_HEADS-1's 128x128 block, not a token.
    rec.tap(be_, "state_predelta", il,
            st.gdn_state + (size_t)(GDN_V_HEADS - 1) * GDN_STATE * GDN_STATE,
            (size_t) GDN_STATE * GDN_STATE);

    st.conv_head = (st.conv_head + 1) % GDN_CONV_K;
    be_.copy(st.conv_ring + (size_t) st.conv_head * GDN_CONV_DIM, qkv_, GDN_CONV_DIM);
    be_.gdn_conv_silu(st.conv_ring, st.conv_head, L.ssm_conv1d, conv_);
    rec.tap(be_, "conv_output_silu", il, conv_, GDN_CONV_DIM);

    // qwen4exp.cpp:863-879: q at 0, k at key_dim, v at 2*key_dim
    float * q_conv = conv_;
    float * k_conv = conv_ + GDN_KEY_DIM;
    float * v_conv = conv_ + 2 * GDN_KEY_DIM;
    rec.tap(be_, "q_conv", il, q_conv, GDN_KEY_DIM);
    rec.tap(be_, "k_conv", il, k_conv, GDN_KEY_DIM);
    rec.tap(be_, "v_conv", il, v_conv, GDN_VAL_DIM);

    be_.l2_norm(q_conv, qn_, GDN_STATE, GDN_K_HEADS, eps_);
    be_.l2_norm(k_conv, kn_, GDN_STATE, GDN_K_HEADS, eps_);
    rec.tap(be_, "q_conv_predelta", il, qn_, GDN_KEY_DIM);
    rec.tap(be_, "k_conv_predelta", il, kn_, GDN_KEY_DIM);

    // The 16 k-heads serve the 48 v-heads as h_k = h_v % 16 -- ggml_repeat's
    // tiling (qwen4exp.cpp:893) and, on the fused path, ops.cpp's
    // `ik1 = iv1 % nek1`. NOT h_v / 3.
    be_.gdn_step(st.gdn_state, qn_, kn_, v_conv, gexp_, beta_, gdn_);
    rec.tap(be_, "attn_output", il, gdn_, GDN_VAL_DIM);

    be_.gated_rms_norm(gdn_, L.ssm_norm, z_, gnorm_, GDN_STATE, GDN_V_HEADS, eps_);
    rec.tap(be_, "final_output", il, gnorm_, GDN_VAL_DIM);

    be_.gemv(L.ssm_out, gnorm_, blk_);
    rec.tap(be_, "linear_attn_out", il, blk_, N_EMBD);
}

// ----------------------------------------------- build_layer_attn (707) ---
void DecodeRunner::layer_qsa(const LayerWeights & L, LayerState & st, int il, Recorder & rec) {
    const int r        = L.compress_ratio;
    const int n_kv     = pos_ + 1;                       // contiguous cache: cell j == position j
    const int n_blocks = (n_kv + r - 1) / r;

    // ---- build_qsa_top_k (477) ------------------------------------------
    // The cached indexer key is RAW: pooling precedes norm and rotation, so
    // neither is applied before the store (qwen4exp.cpp:533).
    be_.gemv(L.idx_k, mixed_, idxk_);
    rec.tap(be_, "indexer_k_raw", il, idxk_, IDX_DIM);
    be_.idx_raw_store(idxk_, st.idx_raw, pos_);

    // Only the block this token joined can change; every earlier block is
    // already final. Missing members read cell 0, which is what
    // set_input_qsa's zero-filled blk_cells produces (llama-memory-hybrid-idx.cpp:395).
    const int blk      = pos_ / r;
    const int n_filled = n_kv - blk * r;
    be_.idx_pool_block(st.idx_raw, blk, n_filled, L.idx_k_norm, eps_, st.idx_pooled,
                       pool_raw_  + (size_t) blk * IDX_DIM,
                       pool_rope_ + (size_t) blk * IDX_DIM);
    // The dump's `indexer_k_pooled` / `indexer_k` are [idx_dim, n_blocks], so
    // they cover EVERY block, not this token's. Each block's value is final
    // once its last member arrives, so the f32 shadow of the pooled cache
    // that idx_pool_block fills as it goes is exactly that tensor.
    rec.tap(be_, "indexer_k_pooled", il, pool_raw_,  (size_t) n_blocks * IDX_DIM);
    rec.tap(be_, "indexer_k",        il, pool_rope_, (size_t) n_blocks * IDX_DIM);

    be_.gemv(L.idx_q, mixed_, idxq_);
    be_.rms_norm_mul(idxq_, L.idx_q_norm, idxq_, IDX_DIM, IDX_N_HEADS, IDX_DIM, eps_);
    be_.rope_imrope(idxq_, IDX_N_HEADS, IDX_DIM, N_ROT, pos_);
    rec.tap(be_, "indexer_q", il, idxq_, IDX_N_HEADS * IDX_DIM);

    // score[b] = sum_h relu(q_h . kpool[b]) -- the DeepSeek lightning
    // indexer's per-head rectification (qwen4exp.cpp:576-591). NOTE: this
    // model has ONE indexer key head (indexer.k_proj is [2560, 128]) and
    // four query heads; m3_attn.hip's scan kernel assumed four key heads and
    // no relu, so it is a cost model for this pass, not a port of it -- and
    // the real pass reads a QUARTER of the pooled-key bytes §M3 charged it.
    be_.idx_scan(st.idx_pooled, idxq_, blkscore_, n_blocks);
    rec.tap(be_, "indexer_score", il, blkscore_, n_blocks);

    // set_input_qsa's per-block bias (llama-memory-hybrid-idx.cpp:438-447):
    // the incomplete tail is always visible (+1e9); a block that could not be
    // pooled is -inf. In a contiguous cache only the last block can be
    // partial, and it IS the tail, so the second case never fires -- the
    // general formula is kept so a paged cache does not silently change it.
    {
        const int tail_start = ((pos_ + 1) / r) * r;
        std::vector<float> bias(n_blocks);
        for (int b = 0; b < n_blocks; ++b) {
            const int filled = std::min(r, n_kv - b * r);
            bias[b] = (b * r >= tail_start) ? 1e9f
                    : (filled < r ? -INFINITY : 0.0f);
        }
        be_.upload(blkbias_, bias.data(), bias.size() * sizeof(float));
    }
    be_.qsa_expand(blkscore_, blkbias_, cellscore_, n_kv, pos_, r);
    rec.tap(be_, "indexer_score_tokens", il, cellscore_, n_kv);

    // "the reference returns indexer_top_k + compress_ratio - 1: whole blocks
    // plus the tail" (qwen4exp.cpp:613-614)
    const int width = std::min(n_kv, IDX_TOP_K + r - 1);
    const int n_sel = be_.topk_select(cellscore_, n_kv, width, sel_);
    if (rec.enabled()) {
        std::vector<int> ids(n_sel);
        be_.sync();
        be_.download(ids.data(), sel_, n_sel * sizeof(int));
        rec.tap_ints("indexer_top_k", il, ids);
    }

    // ---- q / k / v ------------------------------------------------------
    be_.gemv(L.wq, mixed_, qfull_);   // [q|gate] interleaved per head
    be_.gather_strided(qfull_, qcur_, N_Q_HEADS, HEAD_DIM, HEAD_DIM * 2, 0);
    be_.gather_strided(qfull_, gate_, N_Q_HEADS, HEAD_DIM, HEAD_DIM * 2, HEAD_DIM);
    be_.rms_norm_mul(qcur_, L.attn_q_norm, qcur_, HEAD_DIM, N_Q_HEADS, HEAD_DIM, eps_);

    be_.gemv(L.wk, mixed_, kcur_);
    be_.rms_norm_mul(kcur_, L.attn_k_norm, kcur_, HEAD_DIM, N_KV_HEADS, HEAD_DIM, eps_);
    be_.gemv(L.wv, mixed_, vcur_);

    be_.rope_imrope(qcur_, N_Q_HEADS,  HEAD_DIM, N_ROT, pos_);
    be_.rope_imrope(kcur_, N_KV_HEADS, HEAD_DIM, N_ROT, pos_);
    rec.tap(be_, "Qcur", il, qcur_, HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "Kcur", il, kcur_, HEAD_DIM * N_KV_HEADS);
    rec.tap(be_, "Vcur", il, vcur_, HEAD_DIM * N_KV_HEADS);

    be_.kv_store_q8_0(kcur_, vcur_, st.kqs, st.ksc, st.vqs, st.vsc, pos_);
    be_.attn_qsa(st.kqs, st.ksc, st.vqs, st.vsc, qcur_, sel_, n_sel, kqv_);
    rec.tap(be_, "kqv_out",       il, kqv_, HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "attn_pregate",  il, kqv_, HEAD_DIM * N_Q_HEADS);

    be_.unary(FK_SIGMOID, gate_, gsig_, HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "gate_sigmoid", il, gsig_, HEAD_DIM * N_Q_HEADS);
    be_.binary(FK_MUL, kqv_, gsig_, kqv_, HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "attn_gated", il, kqv_, HEAD_DIM * N_Q_HEADS);

    be_.gemv(L.wo, kqv_, blk_);
    rec.tap(be_, "attn_output", il, blk_, N_EMBD);
}

// ------------------------------------------------ build_layer_ffn (919) ---
void DecodeRunner::layer_ffn(const LayerWeights & L, int il, Recorder & rec) {
    be_.gemv(L.ffn_gate_inp, mixed_, logits_);
    rec.tap(be_, "ffn_moe_logits", il, logits_, N_EXPERT);

    // softmax over all 512, top-10 of the same probabilities, then normalise
    // the ten (clamped at 6.103515625e-5). expert_weights_scale is absent
    // from this GGUF, so build_moe_ffn's scale step does not run.
    be_.router(logits_, ids_, wts_);
    rec.tap(be_, "ffn_moe_weights_norm", il, wts_, N_EXPERT_USED);

    // The top-10-of-512 selection is DISCONTINUOUS in its input: a change of
    // 1e-6 in `mixed_` can swap the tenth expert for the eleventh and move
    // ffn_moe_out by percent, which is why a plain cosine bar on ffn_moe_out
    // tests the routing, not the arithmetic. Printing the chosen ids makes
    // that visible instead of leaving it as an unexplained divergence.
    if (cfg_.verbose) {
        int ids[N_EXPERT_USED];
        float w[N_EXPERT_USED];
        be_.sync();
        be_.download(ids, ids_, sizeof(ids));
        be_.download(w,   wts_, sizeof(w));
        std::printf("moe_ids il=%d pos=%d:", il, pos_);
        for (int k = 0; k < N_EXPERT_USED; ++k) std::printf(" %d(%.4f)", ids[k], w[k]);
        std::printf("\n");
    }

    be_.moe_gate_up(L.exp_gate, L.exp_up, ids_, mixed_, ygate_, yup_);
    be_.silu_mul(ygate_, yup_, hmoe_, (size_t) N_EXPERT_USED * N_FF_EXP);
    be_.moe_down(L.exp_down, ids_, hmoe_, eo_);
    be_.moe_combine(eo_, wts_, moeout_);
    rec.tap(be_, "ffn_moe_out", il, moeout_, N_EMBD);

    // shared expert: build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)
    be_.gemv(L.sh_up,   mixed_, shu_);
    be_.gemv(L.sh_gate, mixed_, shg_);
    be_.silu_mul(shg_, shu_, shh_, N_FF_SHEXP);
    be_.gemv(L.sh_down, shh_, shout_);
    rec.tap(be_, "ffn_shexp", il, shout_, N_EMBD);

    be_.gemv(L.sh_gate_inp, mixed_, shgate_);
    rec.tap(be_, "shared_expert_gate", il, shgate_, 1);
    be_.unary(FK_SIGMOID, shgate_, shgate_, 1);
    rec.tap(be_, "shared_expert_gate_sigmoid", il, shgate_, 1);

    be_.mul_tiled(shout_, shgate_, shout_, N_EMBD, 1);
    rec.tap(be_, "ffn_shexp_gated", il, shout_, N_EMBD);

    be_.binary(FK_ADD, moeout_, shout_, blk_, N_EMBD);
    rec.tap(be_, "ffn_out", il, blk_, N_EMBD);
}

// ------------------------------------------------------------- one token --
void DecodeRunner::step(int32_t token, const float * ple_emb, Recorder & rec) {
    if (pos_ >= cfg_.ctx) throw std::runtime_error("sequence longer than --ctx");

    // Design 9.1: the embedding and the PLE gather are the only host work per
    // token -- both are per-token ROW gathers, so they live in host RAM.
    std::vector<float> emb(N_EMBD);
    model_.embed_row(token, emb.data());
    be_.upload(x_, emb.data(), N_EMBD * sizeof(float));
    rec.tap(be_, "model.input_embed", -1, x_, N_EMBD);

    // "the wide residual starts as hc identical copies of the embedding" (324)
    be_.hc_broadcast(x_, res_hc_);
    rec.tap(be_, "hc_init", -1, res_hc_, HC_DIM);

    be_.timer_start();
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        const LayerWeights & L = model_.layer(il);
        LayerState & st = lstate_[il - model_.il0()];

        if (L.is_ple) layer_ple(L, ple_emb, rec);

        hc_mix(L, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject,
               res_hc_, mixed_, inject_, il, rec, "");

        if (L.recurrent) layer_gdn(L, st, il, rec);
        else             layer_qsa(L, st, il, rec);

        be_.hc_combine(res_hc_, blk_, inject_);
        rec.tap(be_, "hc_combine", il, res_hc_, HC_DIM);

        hc_mix(L, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject,
               res_hc_, mixed_, inject_, il, rec, ".2");

        layer_ffn(L, il, rec);

        // the second hc_combine's node is renamed l_last a line later, so the
        // dump has no `hc_combine-<il>.2`; taping one would print MISSING
        be_.hc_combine(res_hc_, blk_, inject_);
        rec.tap(be_, "l_last", il, res_hc_, HC_DIM);
    }
    last_body_ms_ = be_.timer_stop_ms();

    ++pos_;
}

std::vector<float> DecodeRunner::residual_host() {
    std::vector<float> v(HC_DIM);
    be_.sync();
    be_.download(v.data(), res_hc_, HC_DIM * sizeof(float));
    return v;
}

} // namespace fk
