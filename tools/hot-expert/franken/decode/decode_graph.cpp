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
    qkn_      = A(2 * GDN_KEY_DIM);   // q_conv and k_conv normalised together
    qn_       = qkn_;
    kn_       = qkn_ + GDN_KEY_DIM;
    alpha_    = A(GDN_V_HEADS);
    beta_     = A(GDN_V_HEADS);
    gexp_     = A(GDN_V_HEADS);
    abuf_     = A(GDN_V_HEADS);
    bsig_     = A(GDN_V_HEADS);
    gate_raw_ = A(GDN_V_HEADS);
    gdn_      = A(GDN_VAL_DIM);
    gnorm_    = A(GDN_VAL_DIM);

    qfull_    = A((size_t) HEAD_DIM * N_Q_HEADS * 2);
    qcur_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    gate_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    gsig_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    kcur_     = A((size_t) HEAD_DIM * N_KV_HEADS);
    vcur_     = A((size_t) HEAD_DIM * N_KV_HEADS);
    kqv_      = A((size_t) HEAD_DIM * N_Q_HEADS);
    kqvg_     = A((size_t) HEAD_DIM * N_Q_HEADS);
    kraw_     = A((size_t) HEAD_DIM * N_KV_HEADS);
    idxraw_   = A((size_t) IDX_N_HEADS * IDX_DIM);

    const int max_blocks = (cfg_.ctx + QSA_RATIO - 1) / QSA_RATIO;
    idxq_      = A((size_t) IDX_N_HEADS * IDX_DIM);
    idxk_      = A(IDX_DIM);
    blkscore_  = A(max_blocks);
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
    shgated_  = A(N_EMBD);
    shgate_   = A(1);
    shgsig_   = A(1);

    plek_     = A(HC_DIM);
    plev_     = A(N_EMBD);
    pleq_     = A(HC_DIM);
    plegated_ = A(HC_DIM);
    plegate_  = A(HC);
    plenorm_  = A(HC_DIM);
    pleconv_  = A(HC_DIM);
    pleemb_   = A(N_EMBD);
    ple_ring_ = A((size_t)(PLE_CONV_HIST + 1) * HC_DIM);

    const int n_layers = model_.il1() - model_.il0() + 1;
    ids_log_ = AI((size_t) n_layers * N_EXPERT_USED);
    wts_log_ = A((size_t) n_layers * N_EXPERT_USED);
    need_ple_ = (PLE_LAYER >= model_.il0() && PLE_LAYER <= model_.il1());

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
                          const char * suffix, bool xn_ready) {
    (void) L;
    // Four launches, from nine. hc_down and hc_inject share xn, so they are
    // one batched GEMV, and the scale+silu is that batch's epilogue (it rides
    // on the split reduce a 320-row matrix needs anyway); hc_up applies its
    // sigmoid as its own epilogue. `xn_ready` means the preceding
    // hc_combine already produced xn, so the norm costs nothing here.
    if (!xn_ready) be_.rms_norm_mul(x, w_norm, xn_, N_EMBD, HC, HC_DIM, eps_);
    rec.tap(be_, "hc_norm", il, xn_, HC_DIM, suffix);

    GemvJob jobs[2];
    jobs[0].W = &down; jobs[0].out = lo_;        jobs[0].epi = GE_SCALE_SILU;
    jobs[0].arg = 1.0f / (float) HC;
    jobs[1].W = &inj;  jobs[1].out = out_inject; jobs[1].epi = GE_NONE;
    be_.gemv_batch(jobs, 2, xn_, GG_HC_DOWN_INJECT);
    rec.tap(be_, "hc_inject", il, out_inject, HC, suffix);

    GemvJob up_job;
    up_job.W = &up; up_job.out = hgate_; up_job.epi = GE_SIGMOID;
    be_.gemv_batch(&up_job, 1, lo_, GG_HC_UP);
    rec.tap(be_, "hc_gate", il, hgate_, HC_DIM, suffix);

    be_.hc_collapse(xn_, hgate_, out_mixed);
    rec.tap(be_, "hc_mixed", il, out_mixed, N_EMBD, suffix);

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
void DecodeRunner::layer_ple(const LayerWeights & L, Recorder & rec) {
    // pleemb_ was uploaded by step() BEFORE the layer body: a per-token host
    // gather must not become a host->device copy in the middle of the loop.
    rec.tap(be_, "ple_embd", -1, pleemb_, N_EMBD);

    {   // both read the PLE gather, so one kernel
        GemvJob jobs[2];
        jobs[0].W = &L.ple_key;   jobs[0].out = plek_;
        jobs[1].W = &L.ple_value; jobs[1].out = plev_;
        be_.gemv_batch(jobs, 2, pleemb_, GG_PLE);
    }

    // grouped_norm: rms over n_embd per hc stream, then the [hc_dim] gamma
    be_.rms_norm_mul(plek_,   L.ple_norm_key,   plek_, N_EMBD, HC, HC_DIM, eps_);
    be_.rms_norm_mul(res_hc_, L.ple_norm_query, pleq_, N_EMBD, HC, HC_DIM, eps_);

    be_.ple_gate(plek_, pleq_, plev_, plegated_, plegate_);
    rec.tap(be_, "ple_gate",         L.il, plegate_,  HC);
    rec.tap(be_, "ple_gated_value",  L.il, plegated_, HC_DIM);

    be_.rms_norm_mul(plegated_, L.ple_norm_conv,
                     ple_ring_ + (size_t)((ple_head_ + 1) % (PLE_CONV_HIST + 1)) * HC_DIM,
                     N_EMBD, HC, HC_DIM, eps_);

    // the norm writes straight into this token's ring slot, so the conv needs
    // no copy in front of it
    ple_head_ = (ple_head_ + 1) % (PLE_CONV_HIST + 1);
    be_.ple_conv_silu(ple_ring_, ple_head_, L.ple_conv1d, pleconv_);
    rec.tap(be_, "ple_conv_out", L.il, pleconv_, HC_DIM);

    be_.binary3_add(res_hc_, plegated_, pleconv_, res_hc_, HC_DIM);
}

// ---------------------------------------- build_layer_attn_linear (793) ---
void DecodeRunner::layer_gdn(const LayerWeights & L, LayerState & st, int il, Recorder & rec) {
    // All four read `mixed_`, so they are one kernel. qkv goes STRAIGHT into
    // this token's conv ring slot: the ring copy was a launch of its own.
    st.conv_head = (st.conv_head + 1) % GDN_CONV_K;
    float * ring_slot = st.conv_ring + (size_t) st.conv_head * GDN_CONV_DIM;
    {
        GemvJob jobs[4];
        jobs[0].W = &L.ssm_qkv;   jobs[0].out = ring_slot;
        jobs[1].W = &L.ssm_gate;  jobs[1].out = z_;
        jobs[2].W = &L.ssm_beta;  jobs[2].out = beta_;
        jobs[3].W = &L.ssm_alpha; jobs[3].out = alpha_;
        be_.gemv_batch(jobs, 4, mixed_, GG_GDN_PROJ);
    }
    rec.tap(be_, "linear_attn_qkv_mixed", il, ring_slot, GDN_CONV_DIM);
    rec.tap(be_, "z",     il, z_,     GDN_VAL_DIM);
    rec.tap(be_, "beta",  il, beta_,  GDN_V_HEADS);
    rec.tap(be_, "alpha", il, alpha_, GDN_V_HEADS);

    // cb(state, "state_predelta") is the state as read from the cache, i.e.
    // BEFORE this token's decay and update. Its ne2 is the HEAD axis, so the
    // dump's "last column" is head GDN_V_HEADS-1's 128x128 block, not a token.
    rec.tap(be_, "state_predelta", il,
            st.gdn_state + (size_t)(GDN_V_HEADS - 1) * GDN_STATE * GDN_STATE,
            (size_t) GDN_STATE * GDN_STATE);

    // The conv and the gate chain in one launch; the gate's 48 elements ride
    // along on threads past the channel count.
    be_.gdn_conv_gate(st.conv_ring, st.conv_head, L.ssm_conv1d,
                      beta_, alpha_, L.ssm_dt, L.ssm_a,
                      conv_, bsig_, abuf_, gate_raw_, gexp_);
    rec.tap(be_, "conv_output_silu", il, conv_, GDN_CONV_DIM);
    rec.tap(be_, "beta_sigmoid", il, bsig_,     GDN_V_HEADS);
    rec.tap(be_, "a_softplus",   il, abuf_,     GDN_V_HEADS);
    rec.tap(be_, "gate",         il, gate_raw_, GDN_V_HEADS);

    // qwen4exp.cpp:863-879: q at 0, k at key_dim, v at 2*key_dim
    float * q_conv = conv_;
    float * k_conv = conv_ + GDN_KEY_DIM;
    float * v_conv = conv_ + 2 * GDN_KEY_DIM;
    rec.tap(be_, "q_conv", il, q_conv, GDN_KEY_DIM);
    rec.tap(be_, "k_conv", il, k_conv, GDN_KEY_DIM);
    rec.tap(be_, "v_conv", il, v_conv, GDN_VAL_DIM);

    // q_conv and k_conv are adjacent halves of the conv output and qkn_ is
    // one buffer, so both normalise in a single launch of 32 groups.
    be_.l2_norm(q_conv, qkn_, GDN_STATE, 2 * GDN_K_HEADS, eps_);
    rec.tap(be_, "q_conv_predelta", il, qn_, GDN_KEY_DIM);
    rec.tap(be_, "k_conv_predelta", il, kn_, GDN_KEY_DIM);

    // The 16 k-heads serve the 48 v-heads as h_k = h_v % 16 -- ggml_repeat's
    // tiling (qwen4exp.cpp:893) and, on the fused path, ops.cpp's
    // `ik1 = iv1 % nek1`. NOT h_v / 3.
    be_.gdn_step(st.gdn_state, qn_, kn_, v_conv, gexp_, bsig_, gdn_);
    rec.tap(be_, "attn_output", il, gdn_, GDN_VAL_DIM);

    be_.gated_rms_norm(gdn_, L.ssm_norm, z_, gnorm_, GDN_STATE, GDN_V_HEADS, eps_);
    rec.tap(be_, "final_output", il, gnorm_, GDN_VAL_DIM);

    {
        GemvJob j; j.W = &L.ssm_out; j.out = blk_;
        be_.gemv_batch(&j, 1, gnorm_, GG_SSM_OUT);
    }
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
    // wq, wk, wv and the two indexer projections all read `mixed_`, so they
    // are ONE kernel -- and the indexer key is written STRAIGHT into its cache
    // cell, which removes the separate store. The batch mixes Q8_0 and BF16.
    float * idx_cell = st.idx_raw + (size_t) pos_ * IDX_DIM;
    {
        GemvJob jobs[5];
        jobs[0].W = &L.wq;    jobs[0].out = qfull_;
        jobs[1].W = &L.wk;    jobs[1].out = kraw_;
        jobs[2].W = &L.wv;    jobs[2].out = vcur_;
        jobs[3].W = &L.idx_q; jobs[3].out = idxraw_;
        jobs[4].W = &L.idx_k; jobs[4].out = idx_cell;
        be_.gemv_batch(jobs, 5, mixed_, GG_QSA_PROJ);
    }
    rec.tap(be_, "indexer_k_raw", il, idx_cell, IDX_DIM);

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

    // the [q|gate] split, the three QK-norms, the three IMRoPEs and the
    // output gate's sigmoid -- nine launches with grids as small as four
    // workgroups of one wave -- in one kernel of 30 workgroups.
    be_.qsa_qk_post(qfull_, kraw_, idxraw_,
                    L.attn_q_norm, L.attn_k_norm, L.idx_q_norm,
                    qcur_, gate_, gsig_, kcur_, idxq_, pos_, eps_);
    rec.tap(be_, "indexer_q",    il, idxq_, IDX_N_HEADS * IDX_DIM);
    rec.tap(be_, "Qcur",         il, qcur_, HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "Kcur",         il, kcur_, HEAD_DIM * N_KV_HEADS);
    rec.tap(be_, "Vcur",         il, vcur_, HEAD_DIM * N_KV_HEADS);
    rec.tap(be_, "gate_sigmoid", il, gsig_, HEAD_DIM * N_Q_HEADS);

    // score[b] = sum_h relu(q_h . kpool[b]) -- the DeepSeek lightning
    // indexer's per-head rectification (qwen4exp.cpp:576-591). NOTE: this
    // model has ONE indexer key head (indexer.k_proj is [2560, 128]) and
    // four query heads; m3_attn.hip's scan kernel assumed four key heads and
    // no relu, so it is a cost model for this pass, not a port of it -- and
    // the real pass reads a QUARTER of the pooled-key bytes §M3 charged it.
    be_.idx_scan(st.idx_pooled, idxq_, blkscore_, n_blocks);
    rec.tap(be_, "indexer_score", il, blkscore_, n_blocks);

    // set_input_qsa's per-block bias (llama-memory-hybrid-idx.cpp:438-447) is
    // a closed form of the position, so the kernel computes it: the incomplete
    // tail is always visible (+1e9), a block that could not be pooled is -inf,
    // everything else 0. It used to be built on the host and uploaded here,
    // which put a BLOCKING hipMemcpy inside the layer body once per QSA layer.
    // "the reference returns indexer_top_k + compress_ratio - 1: whole blocks
    // plus the tail" (qwen4exp.cpp:613-614). When that budget covers the whole
    // cache the top-k is the IDENTITY -- every visible cell is selected -- so
    // qsa_expand writes the selection and the radix select is not launched.
    // At the depths step 2 runs it is always this branch, and the select was a
    // 1 024-thread workgroup doing four LDS passes over 70 scores.
    const int tail_start = ((pos_ + 1) / r) * r;
    const int width   = std::min(n_kv, IDX_TOP_K + r - 1);
    const bool ident  = (width >= n_kv);
    be_.qsa_expand(blkscore_, cellscore_, sel_, n_kv, pos_, r, tail_start, ident ? 1 : 0);
    rec.tap(be_, "indexer_score_tokens", il, cellscore_, n_kv);
    const int n_sel = ident ? n_kv : be_.topk_select(cellscore_, n_kv, width, sel_);
    if (rec.enabled()) {
        std::vector<int> ids(n_sel);
        be_.sync();
        be_.download(ids.data(), sel_, n_sel * sizeof(int));
        rec.tap_ints("indexer_top_k", il, ids);
    }

    // ---- attention ------------------------------------------------------
    // The combine applies the sigmoid output gate and writes BOTH taps, so
    // neither the sigmoid nor the multiply costs a launch.
    be_.kv_store_q8_0(kcur_, vcur_, st.kqs, st.ksc, st.vqs, st.vsc, pos_);
    be_.attn_qsa(st.kqs, st.ksc, st.vqs, st.vsc, qcur_, sel_, n_sel,
                 gsig_, kqv_, kqvg_);
    rec.tap(be_, "kqv_out",      il, kqv_,  HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "attn_pregate", il, kqv_,  HEAD_DIM * N_Q_HEADS);
    rec.tap(be_, "attn_gated",   il, kqvg_, HEAD_DIM * N_Q_HEADS);

    {
        GemvJob j; j.W = &L.wo; j.out = blk_;
        be_.gemv_batch(&j, 1, kqvg_, GG_ATTN_OUT);
    }
    rec.tap(be_, "attn_output", il, blk_, N_EMBD);
}

// ------------------------------------------------ build_layer_ffn (919) ---
void DecodeRunner::layer_ffn(const LayerWeights & L, int il, Recorder & rec) {
    // the router logits, the shared expert's up/gate and its sigmoid gate all
    // read `mixed_`: one kernel, with the gate's sigmoid as its epilogue.
    {
        GemvJob jobs[4];
        jobs[0].W = &L.ffn_gate_inp; jobs[0].out = logits_;
        jobs[1].W = &L.sh_up;        jobs[1].out = shu_;
        jobs[2].W = &L.sh_gate;      jobs[2].out = shg_;
        jobs[3].W = &L.sh_gate_inp;  jobs[3].out = shgate_;
        be_.gemv_batch(jobs, 4, mixed_, GG_FFN_PROJ);
    }
    rec.tap(be_, "ffn_moe_logits",     il, logits_, N_EXPERT);
    // the sigmoid is moe_finish's, not this GEMV's epilogue, so the RAW
    // shared-expert logit stays a tap of its own at no extra launch
    rec.tap(be_, "shared_expert_gate", il, shgate_, 1);

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
        // device->device only: reading these back here would put a sync per
        // layer inside the body, which is exactly what --profile must be able
        // to report as zero. step() prints them AFTER the body.
        const int slot = il - model_.il0();
        be_.copy((float *) (ids_log_ + (size_t) slot * N_EXPERT_USED), (const float *) ids_, N_EXPERT_USED);
        be_.copy(wts_log_ + (size_t) slot * N_EXPERT_USED, wts_, N_EXPERT_USED);
    }

    be_.moe_gate_up(L.exp_gate, L.exp_up, ids_, mixed_, ygate_, yup_);
    be_.silu_mul(ygate_, yup_, hmoe_, (size_t) N_EXPERT_USED * N_FF_EXP);
    be_.moe_down(L.exp_down, ids_, hmoe_, eo_);

    // The shared expert's down-projection forms its own activation,
    // silu(gate)*up, inside the GEMV -- build_ffn's LLM_FFN_PAR without the
    // elementwise launch in front of it.
    {
        GemvJob j; j.W = &L.sh_down; j.out = shout_;
        be_.gemv_batch(&j, 1, shg_, GG_SH_DOWN, GX_SILU_MUL, shu_);
    }
    rec.tap(be_, "ffn_shexp", il, shout_, N_EMBD);

    // the weighted expert sum, the shared expert's gate and the add, in one
    // launch; both intermediates are written because both are oracle taps.
    be_.moe_finish(eo_, wts_, shout_, shgate_, shgsig_, moeout_, shgated_, blk_);
    rec.tap(be_, "shared_expert_gate_sigmoid", il, shgsig_,  1);
    rec.tap(be_, "ffn_moe_out",     il, moeout_,  N_EMBD);
    rec.tap(be_, "ffn_shexp_gated", il, shgated_, N_EMBD);
    rec.tap(be_, "ffn_out",         il, blk_,     N_EMBD);
}

// ------------------------------------------------------------- one token --
void DecodeRunner::step(int32_t token, const float * ple_emb, Recorder & rec) {
    if (pos_ >= cfg_.ctx) throw std::runtime_error("sequence longer than --ctx");

    // Design 9.1: the embedding and the PLE gather are the only host work per
    // token -- both are per-token ROW gathers, so they live in host RAM.
    std::vector<float> emb(N_EMBD);
    model_.embed_row(token, emb.data());
    be_.upload(x_, emb.data(), N_EMBD * sizeof(float));
    if (ple_emb) be_.upload(pleemb_, ple_emb, N_EMBD * sizeof(float));
    else if (need_ple_) throw std::runtime_error("the layer range includes the PLE layer but no PLE gather was given");
    rec.tap(be_, "model.input_embed", -1, x_, N_EMBD);

    // "the wide residual starts as hc identical copies of the embedding" (324)
    be_.hc_broadcast(x_, res_hc_);
    rec.tap(be_, "hc_init", -1, res_hc_, HC_DIM);
    xn_ready_ = false;

    be_.timer_start();
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        const LayerWeights & L = model_.layer(il);
        LayerState & st = lstate_[il - model_.il0()];

        if (L.is_ple) { layer_ple(L, rec); xn_ready_ = false; }

        hc_mix(L, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject,
               res_hc_, mixed_, inject_, il, rec, "", xn_ready_);

        if (L.recurrent) layer_gdn(L, st, il, rec);
        else             layer_qsa(L, st, il, rec);

        // The combine also produces the norm the FFN hc_mix would have
        // launched: both reduce over the same 2 560 elements of one stream.
        be_.hc_combine_norm(res_hc_, blk_, inject_, L.hc_ffn_norm, xn_, eps_);
        rec.tap(be_, "hc_combine", il, res_hc_, HC_DIM);

        hc_mix(L, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject,
               res_hc_, mixed_, inject_, il, rec, ".2", /*xn_ready=*/true);

        layer_ffn(L, il, rec);

        // Same fusion across the layer boundary: the closing combine produces
        // the NEXT layer's attention norm -- unless that layer is the PLE
        // layer, which rewrites res_hc before the norm would be read, or
        // there is no next layer in this range. The second hc_combine's node
        // is renamed l_last a line later, so the dump has no
        // `hc_combine-<il>.2` and taping one would print MISSING.
        const bool next_ple  = (il + 1 <= model_.il1()) && model_.layer(il + 1).is_ple;
        const bool fuse_next = (il + 1 <= model_.il1()) && !next_ple;
        be_.hc_combine_norm(res_hc_, blk_, inject_,
                            fuse_next ? model_.layer(il + 1).hc_attn_norm : nullptr,
                            xn_, eps_);
        xn_ready_ = fuse_next;
        rec.tap(be_, "l_last", il, res_hc_, HC_DIM);
    }
    last_body_ms_ = be_.timer_stop_ms();
    be_.prof_end_token();

    if (cfg_.verbose) {
        const int nl = model_.il1() - model_.il0() + 1;
        std::vector<int>   ids((size_t) nl * N_EXPERT_USED);
        std::vector<float> w((size_t) nl * N_EXPERT_USED);
        be_.sync();
        be_.download(ids.data(), ids_log_, ids.size() * sizeof(int));
        be_.download(w.data(),   wts_log_, w.size() * sizeof(float));
        for (int l = 0; l < nl; ++l) {
            std::printf("moe_ids il=%d pos=%d:", model_.il0() + l, pos_);
            for (int k = 0; k < N_EXPERT_USED; ++k)
                std::printf(" %d(%.4f)", ids[(size_t) l * N_EXPERT_USED + k],
                                          w[(size_t) l * N_EXPERT_USED + k]);
            std::printf("\n");
        }
    }

    ++pos_;
}

std::vector<float> DecodeRunner::residual_host() {
    std::vector<float> v(HC_DIM);
    be_.sync();
    be_.download(v.data(), res_hc_, HC_DIM * sizeof(float));
    return v;
}

} // namespace fk
