// tools/hot-expert/franken/decode/decode_graph.cpp -- see decode_graph.h.
//
// Line references below are to ~/src/llama-glm53/src/models/qwen4exp.cpp
// @ 39931761a unless another file is named.

#include "decode_graph.h"

#include <algorithm>
#include <chrono>
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

DecodeRunner::DecodeRunner(DecodeModel & model, const DecodeConfig & cfg)
    : model_(model), cfg_(cfg), eps_(model.rms_eps()) {

    const int n_dev      = model_.n_devices();
    const int max_blocks = (cfg_.ctx + QSA_RATIO - 1) / QSA_RATIO;
    const int n_layers   = model_.il1() - model_.il0() + 1;
    max_T_ = cfg_.max_tokens > 0 ? cfg_.max_tokens : 1;
    const size_t T = (size_t) max_T_;

    // One scratch set per device: a kernel may not read another card's
    // memory, so every buffer the body touches exists on every card. It is
    // small -- a few MB a device -- next to the ~22 GB of weights.
    //
    // Everything a TOKEN owns is sized T wide and laid out TOKEN-MAJOR
    // (design 9.4): row t of an n-wide activation is `buf + t*n`, so a row is
    // contiguous, a tiled GEMM's loads are coalesced and every row-wise
    // kernel is the decode kernel with a grid axis.
    //
    // The QSA per-query buffers are the exception, and the reason is the one
    // the first version of this comment gave: cell_scores is O(n_kv) a ROW,
    // so 256 rows of a 256k cache would be 268 MB of them (and 537 MB of
    // radix-select candidates behind them). That is why the stage used to
    // loop one row at a time. It no longer does -- ~12 000 launches a chunk
    // was 0.19 ms a prompt token a card of pure issue cost -- but it does not
    // widen to T either: the backend says how many rows it has scratch for
    // (Backend::reserve_qsa_rows) and the chunk is walked in blocks of that
    // many. blkscore / cellscore / sel are [qsa_rb_][stride].
    blk_stride_  = (size_t) max_blocks;
    cell_stride_ = (size_t) cfg_.ctx;
    sel_stride_  = (size_t) std::min(cfg_.ctx, MAX_SEL);
    qsa_rb_ = max_T_;
    for (int d = 0; d < n_dev; ++d) {
        const int r = model_.dev(d).reserve_qsa_rows(max_T_, cfg_.ctx);
        if (r < qsa_rb_) qsa_rb_ = r;
    }
    if (qsa_rb_ < 1)      qsa_rb_ = 1;
    if (qsa_rb_ > max_T_) qsa_rb_ = max_T_;

    pool_.resize(n_dev);
    for (int d = 0; d < n_dev; ++d) {
        Backend & be = model_.dev(d);
        auto A  = [&](size_t n) { float * p = be.alloc_f32(n); owned_.push_back(p); return p; };
        auto AI = [&](size_t n) { int   * p = be.alloc_i32(n); owned_.push_back(p); return p; };
        Scratch & S = pool_[d];

        S.x        = A(T * N_EMBD);
        // Two banks: see N_RES_BANKS. 40 KB a token a bank, so 10 MB a device
        // extra at T = 256 -- the price of the whole pipeline.
        for (int b = 0; b < N_RES_BANKS; ++b) S.res_hc[b] = A(T * HC_DIM);
        S.xn       = A(T * HC_DIM);
        S.lo       = A(T * HC_LR);
        S.hgate    = A(T * HC_DIM);
        S.mixed    = A(T * N_EMBD);
        S.inject   = A(T * HC);
        S.blk      = A(T * N_EMBD);

        S.z        = A(T * GDN_VAL_DIM);
        S.conv     = A(T * GDN_CONV_DIM);
        S.convqk   = A(T * 2 * GDN_KEY_DIM);
        S.qkn      = A(T * 2 * GDN_KEY_DIM);   // q_conv and k_conv normalised together
        S.alpha    = A(T * GDN_V_HEADS);
        S.beta     = A(T * GDN_V_HEADS);
        S.gexp     = A(T * GDN_V_HEADS);
        S.abuf     = A(T * GDN_V_HEADS);
        S.bsig     = A(T * GDN_V_HEADS);
        S.gate_raw = A(T * GDN_V_HEADS);
        S.gdn      = A(T * GDN_VAL_DIM);
        S.gnorm    = A(T * GDN_VAL_DIM);

        S.qfull    = A(T * HEAD_DIM * N_Q_HEADS * 2);
        S.qcur     = A(T * HEAD_DIM * N_Q_HEADS);
        S.gate     = A(T * HEAD_DIM * N_Q_HEADS);
        S.gsig     = A(T * HEAD_DIM * N_Q_HEADS);
        S.kcur     = A(T * HEAD_DIM * N_KV_HEADS);
        S.vcur     = A(T * HEAD_DIM * N_KV_HEADS);
        S.kqv      = A(T * HEAD_DIM * N_Q_HEADS);
        S.kqvg     = A(T * HEAD_DIM * N_Q_HEADS);
        S.kraw     = A(T * HEAD_DIM * N_KV_HEADS);
        S.idxraw   = A(T * IDX_N_HEADS * IDX_DIM);
        S.idxq     = A(T * IDX_N_HEADS * IDX_DIM);        // distinct from idxraw:
                                                          // qk_post reads one, writes the other

        S.blkscore  = A((size_t) qsa_rb_ * blk_stride_);
        S.cellscore = A((size_t) qsa_rb_ * cell_stride_);
        // This token's block only. They used to be max_blocks long (67 MB a
        // device at 256k) to tap the whole pooled tensor -- for two points
        // the comparator marks INCOMPARABLE at any other cache depth anyway.
        S.pool_raw  = A(IDX_DIM);
        S.pool_rope = A(IDX_DIM);
        // the selection is capped by the BUDGET, not the cache: the identity
        // path only fires while n_kv <= indexer_top_k + r - 1
        S.sel       = AI((size_t) qsa_rb_ * sel_stride_);

        S.logits   = A(T * N_EXPERT);
        S.wts      = A(T * N_EXPERT_USED);
        S.ids      = AI(T * N_EXPERT_USED);
        S.ygate    = A(T * N_EXPERT_USED * N_FF_EXP);
        S.yup      = A(T * N_EXPERT_USED * N_FF_EXP);
        S.hmoe     = A(T * N_EXPERT_USED * N_FF_EXP);
        S.eo       = A(T * N_EXPERT_USED * N_EMBD);
        S.moeout   = A(T * N_EMBD);
        S.shg      = A(T * N_FF_SHEXP);
        S.shu      = A(T * N_FF_SHEXP);
        S.shh      = A(T * N_FF_SHEXP);
        S.shout    = A(T * N_EMBD);
        S.shgated  = A(T * N_EMBD);
        S.shgate   = A(T);
        S.shgsig   = A(T);

        S.plek     = A(T * HC_DIM);
        S.plev     = A(T * N_EMBD);
        S.pleq     = A(T * HC_DIM);
        S.plegated = A(T * HC_DIM);
        S.plegate  = A(T * HC);
        S.plenorm  = A(T * HC_DIM);
        S.pleconv  = A(T * HC_DIM);
        S.pleemb   = A(T * N_EMBD);
        S.ple_win  = A(((size_t) PLE_CONV_HIST + T) * HC_DIM);

        S.ids_log  = AI((size_t) n_layers * T * N_EXPERT_USED);
        S.wts_log  = A((size_t) n_layers * T * N_EXPERT_USED);
    }

    // Once, here -- never per token. See Backend::reserve_topk.
    for (int d = 0; d < n_dev; ++d) model_.dev(d).reserve_topk(cfg_.ctx);

    need_ple_ = (PLE_LAYER >= model_.il0() && PLE_LAYER <= model_.il1());
    routed_ids_.assign((size_t) n_layers * N_EXPERT_USED, -1);

    // The head runs on the LAST ROW only -- a prompt needs one logit vector
    // -- so none of this is T wide.
    // The head's scratch belongs to the last device, with lm_head.
    if (model_.have_head()) {
        Backend & be = model_.dev(n_dev - 1);
        head_xn_    = be.alloc_f32(HC_DIM);      owned_.push_back(head_xn_);
        head_lo_    = be.alloc_f32(HC_LR);       owned_.push_back(head_lo_);
        head_gate_  = be.alloc_f32(HC_DIM);      owned_.push_back(head_gate_);
        head_out_   = be.alloc_f32(N_EMBD);      owned_.push_back(head_out_);
        logits_all_ = be.alloc_f32((size_t) model_.lm_head().rows);
        owned_.push_back(logits_all_);
        greedy_id_  = be.alloc_i32(1);           owned_.push_back(greedy_id_);
    }

    // Per-layer persistent state, on the layer's OWN device. alloc_f32
    // zeroes, which is exactly a fresh sequence: llama.cpp clears the
    // recurrent cells the same way.
    lstate_.resize(n_layers);
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & st = lstate_[il - model_.il0()];
        Backend & be = model_.dev_for(il);
        auto A  = [&](size_t n) { float * p = be.alloc_f32(n); owned_.push_back(p); return p; };
        auto AR = [&](size_t b) { void  * p = be.alloc_raw(b); owned_.push_back(p); return p; };
        if (is_recurrent_layer(il)) {
            st.conv_win  = A(((size_t) GDN_CONV_HIST + T) * GDN_CONV_DIM);
            st.gdn_state = A((size_t) GDN_V_HEADS * GDN_STATE * GDN_STATE);
        } else {
            const size_t n_qs = (size_t) cfg_.ctx * N_KV_HEADS * HEAD_DIM;
            const size_t n_sc = (size_t) cfg_.ctx * N_KV_HEADS * (HEAD_DIM / 32);
            st.kqs = (int8_t   *) AR(n_qs);
            st.vqs = (int8_t   *) AR(n_qs);
            st.ksc = (uint16_t *) AR(n_sc * sizeof(uint16_t));
            st.vsc = (uint16_t *) AR(n_sc * sizeof(uint16_t));
            st.idx_new    = A(T * IDX_DIM);
            st.idx_sum    = A(IDX_DIM);
            st.idx_raw0   = A(IDX_DIM);
            st.idx_pooled = (uint16_t *) AR((size_t) max_blocks * IDX_DIM * sizeof(uint16_t));
        }
    }
    bind(0);
}

// Points the active scratch members at one device's set. Called once per
// layer, so the body below never has to know which card it is running on.
void DecodeRunner::bind(int d) {
    if (d == cur_dev_) return;
    cur_dev_ = d;
    bep_ = &model_.dev(d);
    Scratch & S = pool_[d];
    x_ = S.x; res_hc_ = S.res_hc[bank_]; xn_ = S.xn; lo_ = S.lo; hgate_ = S.hgate;
    mixed_ = S.mixed; inject_ = S.inject; blk_ = S.blk;
    z_ = S.z; conv_ = S.conv; convqk_ = S.convqk;
    qkn_ = S.qkn; qn_ = S.qkn; kn_ = S.qkn + GDN_KEY_DIM;
    alpha_ = S.alpha; beta_ = S.beta; gexp_ = S.gexp; abuf_ = S.abuf;
    bsig_ = S.bsig; gate_raw_ = S.gate_raw; gdn_ = S.gdn; gnorm_ = S.gnorm;
    qfull_ = S.qfull; qcur_ = S.qcur; gate_ = S.gate; gsig_ = S.gsig;
    kcur_ = S.kcur; vcur_ = S.vcur; kqv_ = S.kqv; kqvg_ = S.kqvg;
    kraw_ = S.kraw; idxraw_ = S.idxraw; idxq_ = S.idxq;
    blkscore_ = S.blkscore; cellscore_ = S.cellscore;
    pool_raw_ = S.pool_raw; pool_rope_ = S.pool_rope; sel_ = S.sel;
    logits_ = S.logits; wts_ = S.wts; ids_ = S.ids;
    ygate_ = S.ygate; yup_ = S.yup; hmoe_ = S.hmoe; eo_ = S.eo; moeout_ = S.moeout;
    shg_ = S.shg; shu_ = S.shu; shh_ = S.shh; shout_ = S.shout;
    shgated_ = S.shgated; shgate_ = S.shgate; shgsig_ = S.shgsig;
    plek_ = S.plek; plev_ = S.plev; pleq_ = S.pleq; plegated_ = S.plegated;
    plegate_ = S.plegate; plenorm_ = S.plenorm; pleconv_ = S.pleconv;
    pleemb_ = S.pleemb; ple_win_ = S.ple_win;
    ids_log_ = S.ids_log; wts_log_ = S.wts_log;
}

DecodeRunner::~DecodeRunner() {
    for (void * p : owned_) bep_->free_buf(p);
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
    // The grouped norm sees HC*T groups of N_EMBD: with a token-major
    // [T][HC][N_EMBD] buffer, group g is stream g % HC of token g / HC, which
    // is exactly the periodic gamma rms_norm_mul takes.
    if (!xn_ready) bep_->rms_norm_mul(x, w_norm, xn_, N_EMBD, HC * T_, HC_DIM, eps_);
    rec.tap(*bep_, "hc_norm", il, xn_ + (size_t)(T_ - 1) * HC_DIM, HC_DIM, suffix);

    GemvJob jobs[2];
    jobs[0].W = &down; jobs[0].out = lo_;        jobs[0].epi = GE_SCALE_SILU;
    jobs[0].arg = 1.0f / (float) HC;
    jobs[1].W = &inj;  jobs[1].out = out_inject; jobs[1].epi = GE_NONE;
    bep_->gemv_batch(jobs, 2, xn_, GG_HC_DOWN_INJECT, GX_PLAIN, nullptr, T_);
    rec.tap(*bep_, "hc_inject", il, out_inject + (size_t)(T_ - 1) * HC, HC, suffix);

    // The gate projection and the collapse that eats it, as ONE op: the
    // backend folds them into one launch when it can (decode_backend.h), and
    // hgate_ is written either way, so hc_gate is still a tap.
    GemvJob up_job;
    up_job.W = &up; up_job.out = hgate_; up_job.epi = GE_SIGMOID;
    bep_->gemv_hc_gate_collapse(up_job, lo_, xn_, out_mixed, GG_HC_UP, T_);
    rec.tap(*bep_, "hc_gate", il, hgate_ + (size_t)(T_ - 1) * HC_DIM, HC_DIM, suffix);
    rec.tap(*bep_, "hc_mixed", il, out_mixed + (size_t)(T_ - 1) * N_EMBD, N_EMBD, suffix);

    if (cfg_.jitter > 0.0f) jitter(out_mixed, (size_t) T_ * N_EMBD);
}

// Scales every element by (1 +- jitter), deterministically. The block input
// is the right place: it is what every projection of the layer reads, so a
// perturbation here is exactly what a different (equally valid) summation
// order upstream would have produced.
void DecodeRunner::jitter(float * buf, size_t n) {
    std::vector<float> h(n);
    bep_->sync();
    bep_->download(h.data(), buf, n * sizeof(float));
    std::uniform_real_distribution<float> d(-cfg_.jitter, cfg_.jitter);
    for (size_t i = 0; i < n; ++i) h[i] *= (1.0f + d(jrng_));
    bep_->upload(buf, h.data(), n * sizeof(float));
}

// ------------------------------------------------------- build_ple (1137) --
void DecodeRunner::layer_ple(const LayerWeights & L, Recorder & rec) {
    const size_t tl = (size_t)(T_ - 1);
    // pleemb_ was uploaded by step() BEFORE the layer body: a per-token host
    // gather must not become a host->device copy in the middle of the loop.
    rec.tap(*bep_, "ple_embd", -1, pleemb_ + tl * N_EMBD, N_EMBD);

    {   // both read the PLE gather, so one kernel
        GemvJob jobs[2];
        jobs[0].W = &L.ple_key;   jobs[0].out = plek_;
        jobs[1].W = &L.ple_value; jobs[1].out = plev_;
        bep_->gemv_batch(jobs, 2, pleemb_, GG_PLE, GX_PLAIN, nullptr, T_);
    }

    // grouped_norm: rms over n_embd per hc stream, then the [hc_dim] gamma
    bep_->rms_norm_mul(plek_,   L.ple_norm_key,   plek_, N_EMBD, HC * T_, HC_DIM, eps_);
    bep_->rms_norm_mul(res_hc_, L.ple_norm_query, pleq_, N_EMBD, HC * T_, HC_DIM, eps_);

    bep_->ple_gate(plek_, pleq_, plev_, plegated_, plegate_, T_);
    rec.tap(*bep_, "ple_gate",         L.il, plegate_  + tl * HC,     HC);
    rec.tap(*bep_, "ple_gated_value",  L.il, plegated_ + tl * HC_DIM, HC_DIM);

    // The norm writes straight into the chunk's window slots -- token t at
    // slot PLE_CONV_HIST+t -- so the conv needs no copy in front of it and
    // the first tokens read the previous chunk's tail without a special case.
    bep_->rms_norm_mul(plegated_, L.ple_norm_conv,
                       ple_win_ + (size_t) PLE_CONV_HIST * HC_DIM,
                       N_EMBD, HC * T_, HC_DIM, eps_);

    bep_->ple_conv_win(ple_win_, L.ple_conv1d, pleconv_, T_);
    rec.tap(*bep_, "ple_conv_out", L.il, pleconv_ + tl * HC_DIM, HC_DIM);

    bep_->binary3_add(res_hc_, plegated_, pleconv_, res_hc_, (size_t) T_ * HC_DIM);
    bep_->conv_slide(ple_win_, PLE_CONV_HIST, HC_DIM, T_);
}

// ---------------------------------------- build_layer_attn_linear (793) ---
void DecodeRunner::layer_gdn(const LayerWeights & L, LayerState & st, int il, Recorder & rec) {
    const size_t tl = (size_t)(T_ - 1);
    // All four read `mixed_`, so they are one kernel. qkv goes STRAIGHT into
    // the chunk's conv WINDOW slots: the GEMM writes token-major with a row
    // of GDN_CONV_DIM, which is the window's own slot stride, so the copy
    // into the conv history is still not a launch of its own.
    float * win_slot = st.conv_win + (size_t) GDN_CONV_HIST * GDN_CONV_DIM;
    {
        GemvJob jobs[4];
        jobs[0].W = &L.ssm_qkv;   jobs[0].out = win_slot;
        jobs[1].W = &L.ssm_gate;  jobs[1].out = z_;
        jobs[2].W = &L.ssm_beta;  jobs[2].out = beta_;
        jobs[3].W = &L.ssm_alpha; jobs[3].out = alpha_;
        bep_->gemv_batch(jobs, 4, mixed_, GG_GDN_PROJ, GX_PLAIN, nullptr, T_);
    }
    rec.tap(*bep_, "linear_attn_qkv_mixed", il, win_slot + tl * GDN_CONV_DIM, GDN_CONV_DIM);
    rec.tap(*bep_, "z",     il, z_     + tl * GDN_VAL_DIM, GDN_VAL_DIM);
    rec.tap(*bep_, "beta",  il, beta_  + tl * GDN_V_HEADS, GDN_V_HEADS);
    rec.tap(*bep_, "alpha", il, alpha_ + tl * GDN_V_HEADS, GDN_V_HEADS);

    // cb(state, "state_predelta") is the state as read from the cache, i.e.
    // BEFORE this token's decay and update. Its ne2 is the HEAD axis, so the
    // dump's "last column" is head GDN_V_HEADS-1's 128x128 block, not a token.
    rec.tap(*bep_, "state_predelta", il,
            st.gdn_state + (size_t)(GDN_V_HEADS - 1) * GDN_STATE * GDN_STATE,
            (size_t) GDN_STATE * GDN_STATE);

    // The conv and the gate chain in one launch; the gate's 48 elements a
    // token ride along on threads past the channel count. It writes the
    // conv output twice: conv_ is the reference's contiguous [q|k|v] row,
    // convqk_ is q and k alone so l2_norm's groups are contiguous over the
    // whole chunk.
    bep_->gdn_conv_gate(st.conv_win, L.ssm_conv1d,
                      beta_, alpha_, L.ssm_dt, L.ssm_a,
                      conv_, convqk_, bsig_, abuf_, gate_raw_, gexp_, T_);
    rec.tap(*bep_, "conv_output_silu", il, conv_ + tl * GDN_CONV_DIM, GDN_CONV_DIM);
    rec.tap(*bep_, "beta_sigmoid", il, bsig_     + tl * GDN_V_HEADS, GDN_V_HEADS);
    rec.tap(*bep_, "a_softplus",   il, abuf_     + tl * GDN_V_HEADS, GDN_V_HEADS);
    rec.tap(*bep_, "gate",         il, gate_raw_ + tl * GDN_V_HEADS, GDN_V_HEADS);

    // qwen4exp.cpp:863-879: q at 0, k at key_dim, v at 2*key_dim
    float * q_conv = conv_ + tl * GDN_CONV_DIM;
    float * k_conv = q_conv + GDN_KEY_DIM;
    float * v_conv = q_conv + 2 * GDN_KEY_DIM;
    rec.tap(*bep_, "q_conv", il, q_conv, GDN_KEY_DIM);
    rec.tap(*bep_, "k_conv", il, k_conv, GDN_KEY_DIM);
    rec.tap(*bep_, "v_conv", il, v_conv, GDN_VAL_DIM);

    // q_conv and k_conv are adjacent halves of convqk_'s row, so both
    // normalise in a single launch of 32 groups a token.
    bep_->l2_norm(convqk_, qkn_, GDN_STATE, 2 * GDN_K_HEADS * T_, eps_);
    rec.tap(*bep_, "q_conv_predelta", il, qn_ + tl * 2 * GDN_KEY_DIM, GDN_KEY_DIM);
    rec.tap(*bep_, "k_conv_predelta", il, kn_ + tl * 2 * GDN_KEY_DIM, GDN_KEY_DIM);

    // The 16 k-heads serve the 48 v-heads as h_k = h_v % 16 -- ggml_repeat's
    // tiling (qwen4exp.cpp:893) and, on the fused path, ops.cpp's
    // `ik1 = iv1 % nek1`. NOT h_v / 3.
    // The chunk is walked in order INSIDE the kernel: token t sees the state
    // token t-1 left, which is the same sequence of updates T single-token
    // calls make.
    bep_->gdn_step(st.gdn_state, qn_, kn_, conv_ + 2 * GDN_KEY_DIM, gexp_, bsig_, gdn_,
                   T_, 2 * GDN_KEY_DIM, GDN_CONV_DIM);
    rec.tap(*bep_, "attn_output", il, gdn_ + tl * GDN_VAL_DIM, GDN_VAL_DIM);

    bep_->gated_rms_norm(gdn_, L.ssm_norm, z_, gnorm_, GDN_STATE, GDN_V_HEADS * T_, eps_);
    rec.tap(*bep_, "final_output", il, gnorm_ + tl * GDN_VAL_DIM, GDN_VAL_DIM);

    {
        GemvJob j; j.W = &L.ssm_out; j.out = blk_;
        bep_->gemv_batch(&j, 1, gnorm_, GG_SSM_OUT, GX_PLAIN, nullptr, T_);
    }
    rec.tap(*bep_, "linear_attn_out", il, blk_ + tl * N_EMBD, N_EMBD);

    // carry the conv tail into the next chunk
    bep_->conv_slide(st.conv_win, GDN_CONV_HIST, GDN_CONV_DIM, T_);
}

// ----------------------------------------------- build_layer_attn (707) ---
// The ORDER inside this function is the whole causal argument of PREFILL.md
// section 3, and it is not the order a single token needs:
//
//   1. project all T rows                    (one batched GEMM per group)
//   2. pool the indexer keys of all T cells, re-pooling every block touched
//   3. write K and V of all T cells
//   4. THEN, per query row p: scan, bias, expand, top-k, attend
//
// Steps 2 and 3 must precede 4 because a chunk's own later rows are in the
// cache while row p is scored -- exactly as the reference's apply_ubatch
// leaves them (qwen4exp.cpp:1026). That is safe, and the reason is
// arithmetic rather than luck: tail_start is a multiple of the ratio, so a
// block that is SCORED (bias 0 or -inf) cannot contain a position after p,
// and a block that could is in the +1e9 set where its score is irrelevant
// and whose future cells the per-cell mask removes anyway.
void DecodeRunner::layer_qsa(const LayerWeights & L, LayerState & st, int il, Recorder & rec) {
    const int    r  = L.compress_ratio;
    const size_t tl = (size_t)(T_ - 1);

    // ---- build_qsa_top_k (477) ------------------------------------------
    // The cached indexer key is RAW: pooling precedes norm and rotation, so
    // neither is applied before the store (qwen4exp.cpp:533).
    // wq, wk, wv and the two indexer projections all read `mixed_`, so they
    // are ONE kernel -- and the indexer key is written STRAIGHT into its cache
    // cell, which removes the separate store. The batch mixes Q8_0 and BF16.
    {
        GemvJob jobs[5];
        jobs[0].W = &L.wq;    jobs[0].out = qfull_;
        jobs[1].W = &L.wk;    jobs[1].out = kraw_;
        jobs[2].W = &L.wv;    jobs[2].out = vcur_;
        jobs[3].W = &L.idx_q; jobs[3].out = idxraw_;
        jobs[4].W = &L.idx_k; jobs[4].out = st.idx_new;
        bep_->gemv_batch(jobs, 5, mixed_, GG_QSA_PROJ, GX_PLAIN, nullptr, T_);
    }
    rec.tap(*bep_, "indexer_k_raw", il, st.idx_new + tl * IDX_DIM, IDX_DIM);

    // Only the blocks this chunk joined can change; every earlier block is
    // already final. Missing members read cell 0, which is what
    // set_input_qsa's zero-filled blk_cells produces (llama-memory-hybrid-idx.cpp:395).
    bep_->idx_pool_chunk(st.idx_new, st.idx_sum, st.idx_raw0, pos_, T_,
                         L.idx_k_norm, eps_, st.idx_pooled, pool_raw_, pool_rope_);
    // The dump's `indexer_k_pooled` / `indexer_k` are [idx_dim, n_blocks], so
    // they cover EVERY block, not this token's. Each block's value is final
    // once its last member arrives, so the f32 shadow of the pooled cache
    // that idx_pool_chunk fills as it goes is exactly that tensor.
    rec.tap(*bep_, "indexer_k_pooled", il, pool_raw_,  IDX_DIM);
    rec.tap(*bep_, "indexer_k",        il, pool_rope_, IDX_DIM);

    // the [q|gate] split, the three QK-norms, the three IMRoPEs and the
    // output gate's sigmoid -- nine launches with grids as small as four
    // workgroups of one wave -- in one kernel of 30 workgroups a row.
    bep_->qsa_qk_post(qfull_, kraw_, idxraw_,
                    L.attn_q_norm, L.attn_k_norm, L.idx_q_norm,
                    qcur_, gate_, gsig_, kcur_, idxq_, pos_, eps_, T_);
    rec.tap(*bep_, "indexer_q",    il, idxq_ + tl * IDX_N_HEADS * IDX_DIM, IDX_N_HEADS * IDX_DIM);
    rec.tap(*bep_, "Qcur",         il, qcur_ + tl * HEAD_DIM * N_Q_HEADS,  HEAD_DIM * N_Q_HEADS);
    rec.tap(*bep_, "Kcur",         il, kcur_ + tl * HEAD_DIM * N_KV_HEADS, HEAD_DIM * N_KV_HEADS);
    rec.tap(*bep_, "Vcur",         il, vcur_ + tl * HEAD_DIM * N_KV_HEADS, HEAD_DIM * N_KV_HEADS);
    rec.tap(*bep_, "gate_sigmoid", il, gsig_ + tl * HEAD_DIM * N_Q_HEADS,  HEAD_DIM * N_Q_HEADS);

    // every cell of the chunk in the cache BEFORE anything is scored
    bep_->kv_store_q8_0(kcur_, vcur_, st.kqs, st.ksc, st.vqs, st.vsc, pos_, T_);

    // ---- the query rows, in blocks of qsa_rb_ ----------------------------
    //
    // Every row does the same four things and differs only in its POSITION,
    // and n_kv, n_blocks, tail_start, the budget and whether the top-k is the
    // identity are all closed forms of it. So the rows are a grid axis
    // (Backend::qsa_rows), not a host loop: at T = 256 the loop was ~1 000
    // launches a QSA layer and ~12 000 a chunk, which the profile charged
    // 0.19 ms a prompt token a card -- all of it issue cost. The block is
    // bounded by scratch, not by T, because cell_scores is O(n_kv) a row.
    //
    // What each stage is, unchanged by the batching:
    //
    //  * idx_scan: score[b] = sum_h relu(q_h . kpool[b]), the DeepSeek
    //    lightning indexer's per-head rectification (qwen4exp.cpp:576-591).
    //    This model has ONE indexer key head (indexer.k_proj is [2560,128])
    //    and four query heads; m3_attn.hip's scan kernel assumed four key
    //    heads and no relu, so it is a cost model for this pass, not a port
    //    of it -- the real pass reads a QUARTER of the pooled-key bytes §M3
    //    charged it.
    //  * qsa_expand: set_input_qsa's per-block bias
    //    (llama-memory-hybrid-idx.cpp:438-447) computed IN the kernel from
    //    the position -- the incomplete tail always visible (+1e9), a block
    //    that could not be pooled -inf, everything else 0. It used to be
    //    built on the host and uploaded, which put a BLOCKING hipMemcpy
    //    inside the layer body once per QSA layer.
    //  * the top-k: "the reference returns indexer_top_k + compress_ratio - 1:
    //    whole blocks plus the tail" (qwen4exp.cpp:613-614). When that budget
    //    covers the whole cache the top-k is the IDENTITY -- every visible
    //    cell is selected -- so qsa_expand writes the selection and the radix
    //    select is not launched at all.
    //  * attention: the combine applies the sigmoid output gate and writes
    //    BOTH taps, so neither the sigmoid nor the multiply costs a launch.
    const int budget = IDX_TOP_K + r - 1;
    for (int t0 = 0; t0 < T_; t0 += qsa_rb_) {
        const int nr = std::min(qsa_rb_, T_ - t0);
        Backend::QsaRows j;
        j.pooled = st.idx_pooled;
        j.kqs = st.kqs; j.ksc = st.ksc; j.vqs = st.vqs; j.vsc = st.vsc;
        j.idxq      = idxq_ + (size_t) t0 * IDX_N_HEADS * IDX_DIM;
        j.q         = qcur_ + (size_t) t0 * HEAD_DIM * N_Q_HEADS;
        j.gsig      = gsig_ + (size_t) t0 * HEAD_DIM * N_Q_HEADS;
        j.blk       = blkscore_;
        j.cell      = cellscore_;
        j.sel       = sel_;
        j.out       = kqv_  + (size_t) t0 * HEAD_DIM * N_Q_HEADS;
        j.out_gated = kqvg_ + (size_t) t0 * HEAD_DIM * N_Q_HEADS;
        j.rows = nr; j.pos0 = pos_ + t0; j.ratio = r; j.budget = budget;
        j.blk_stride = blk_stride_; j.cell_stride = cell_stride_; j.sel_stride = sel_stride_;
        bep_->qsa_rows(j);

        // The three selection taps are the LAST row of the chunk, exactly as
        // they were when the loop taps fired on `last`. A row's slice is
        // written only by its own row, so the values are the ones the loop
        // produced; only the moment they are read back has moved, from
        // between that row's stages to after its block.
        if (t0 + nr == T_) {
            const size_t lt    = (size_t)(nr - 1);
            const int    p     = pos_ + T_ - 1;
            const int    n_kv  = p + 1;              // cell j == position j
            const int    n_blk = (n_kv + r - 1) / r;
            const int    n_sel = std::min(n_kv, budget);
            rec.tap(*bep_, "indexer_score", il, blkscore_ + lt * blk_stride_, n_blk);
            rec.tap(*bep_, "indexer_score_tokens", il, cellscore_ + lt * cell_stride_, n_kv);
            if (rec.enabled()) {
                std::vector<int> ids(n_sel);
                bep_->sync();
                bep_->download(ids.data(), sel_ + lt * sel_stride_, n_sel * sizeof(int));
                rec.tap_ints("indexer_top_k", il, ids);
            }
        }
    }
    rec.tap(*bep_, "kqv_out",      il, kqv_  + tl * HEAD_DIM * N_Q_HEADS, HEAD_DIM * N_Q_HEADS);
    rec.tap(*bep_, "attn_pregate", il, kqv_  + tl * HEAD_DIM * N_Q_HEADS, HEAD_DIM * N_Q_HEADS);
    rec.tap(*bep_, "attn_gated",   il, kqvg_ + tl * HEAD_DIM * N_Q_HEADS, HEAD_DIM * N_Q_HEADS);

    {
        GemvJob j; j.W = &L.wo; j.out = blk_;
        bep_->gemv_batch(&j, 1, kqvg_, GG_ATTN_OUT, GX_PLAIN, nullptr, T_);
    }
    rec.tap(*bep_, "attn_output", il, blk_ + tl * N_EMBD, N_EMBD);
}

// ------------------------------------------------ build_layer_ffn (919) ---
void DecodeRunner::layer_ffn(const LayerWeights & L, int il, Recorder & rec) {
    const size_t tl = (size_t)(T_ - 1);
    // the router logits, the shared expert's up/gate and its sigmoid gate all
    // read `mixed_`: one kernel, with the gate's sigmoid as its epilogue.
    {
        GemvJob jobs[4];
        jobs[0].W = &L.ffn_gate_inp; jobs[0].out = logits_;
        jobs[1].W = &L.sh_up;        jobs[1].out = shu_;
        jobs[2].W = &L.sh_gate;      jobs[2].out = shg_;
        jobs[3].W = &L.sh_gate_inp;  jobs[3].out = shgate_;
        bep_->gemv_batch(jobs, 4, mixed_, GG_FFN_PROJ, GX_PLAIN, nullptr, T_);
    }
    rec.tap(*bep_, "ffn_moe_logits",     il, logits_ + tl * N_EXPERT, N_EXPERT);
    // the sigmoid is moe_finish's, not this GEMV's epilogue, so the RAW
    // shared-expert logit stays a tap of its own at no extra launch
    rec.tap(*bep_, "shared_expert_gate", il, shgate_ + tl, 1);

    // softmax over all 512, top-10 of the same probabilities, then normalise
    // the ten (clamped at 6.103515625e-5). expert_weights_scale is absent
    // from this GGUF, so build_moe_ffn's scale step does not run.
    // The router writes the per-layer log itself, so capturing the routed
    // experts costs NO extra op. It used to be two device-to-device copies a
    // layer -- 32 a token, and they stalled the queue badly enough to make a
    // --routing run 1.8x slower on every card.
    const int slot = il - model_.il0();
    const bool cap = capture_ && (cfg_.verbose || cfg_.log_routing);
    const size_t lstride = (size_t) max_T_ * N_EXPERT_USED;
    bep_->router(logits_, ids_, wts_,
                 cap ? ids_log_ + (size_t) slot * lstride : nullptr,
                 cap ? wts_log_ + (size_t) slot * lstride : nullptr, T_);
    rec.tap(*bep_, "ffn_moe_weights_norm", il, wts_ + tl * N_EXPERT_USED, N_EXPERT_USED);

    // The top-10-of-512 selection is DISCONTINUOUS in its input: a change of
    // 1e-6 in `mixed_` can swap the tenth expert for the eleventh and move
    // ffn_moe_out by percent, which is why a plain cosine bar on ffn_moe_out
    // tests the routing, not the arithmetic. Printing the chosen ids makes
    // that visible instead of leaving it as an unexplained divergence.


    bep_->moe_gate_up(L.exp_gate, L.exp_up, ids_, mixed_, ygate_, yup_, T_);
    bep_->silu_mul(ygate_, yup_, hmoe_, (size_t) T_ * N_EXPERT_USED * N_FF_EXP);
    bep_->moe_down(L.exp_down, ids_, hmoe_, eo_, T_);

    // The shared expert's down-projection forms its own activation,
    // silu(gate)*up, inside the GEMV -- build_ffn's LLM_FFN_PAR without the
    // elementwise launch in front of it.
    {
        GemvJob j; j.W = &L.sh_down; j.out = shout_;
        bep_->gemv_batch(&j, 1, shg_, GG_SH_DOWN, GX_SILU_MUL, shu_, T_);
    }
    rec.tap(*bep_, "ffn_shexp", il, shout_ + tl * N_EMBD, N_EMBD);

    // the weighted expert sum, the shared expert's gate and the add, in one
    // launch; both intermediates are written because both are oracle taps.
    bep_->moe_finish(eo_, wts_, shout_, shgate_, shgsig_, moeout_, shgated_, blk_, T_);
    rec.tap(*bep_, "shared_expert_gate_sigmoid", il, shgsig_ + tl,          1);
    rec.tap(*bep_, "ffn_moe_out",     il, moeout_  + tl * N_EMBD, N_EMBD);
    rec.tap(*bep_, "ffn_shexp_gated", il, shgated_ + tl * N_EMBD, N_EMBD);
    rec.tap(*bep_, "ffn_out",         il, blk_     + tl * N_EMBD, N_EMBD);
}

// ------------------------------------------------------------- one token --
//
// Design 9.2's token: embed on device 0, the layer ranges in order, the
// hyper-connection residual across each boundary, then the final mixer,
// lm_head and a device-side argmax on the last card. The only host work is
// the two per-token row gathers (design 9.1) and the one int that comes back.
int DecodeRunner::step(const int32_t * tokens, int T, const float * ple_emb, Recorder & rec,
                       bool flush) {
    if (T < 1 || T > max_T_)
        throw std::runtime_error("chunk larger than the configured --chunk");
    if (pos_ + T > cfg_.ctx) throw std::runtime_error("sequence longer than --ctx");
    T_ = T;
    const size_t tl = (size_t)(T_ - 1);

    // A tap is a download and the routing log is a download, so a captured
    // chunk cannot run ahead of the device. Deciding it here rather than at
    // the call site means no caller can accidentally lose a tap by asking for
    // the pipeline.
    const bool capture_now = rec.enabled() ||
                             (capture_ && (cfg_.verbose || cfg_.log_routing));
    if (capture_now || cfg_.jitter > 0.0f || cfg_.sync_debug) flush = true;
    last_flushed_ = flush;

    bank_ = (int) (chunks_ & (N_RES_BANKS - 1));
    ++chunks_;

    cur_dev_ = -1;                       // the bank moved: rebind unconditionally
    bind(model_.dev_of(model_.il0()));

    // Nothing may overwrite this bank until the card downstream has copied
    // the PREVIOUS chunk that used it out. A stream wait, not a host one.
    bep_->boundary_wait_free(bank_);

    // One gather and ONE upload for the chunk: a per-token host copy in the
    // issue path is host time, and at T = 256 it would be 256 of them.
    if (emb_.size() != (size_t) T * N_EMBD) emb_.resize((size_t) T * N_EMBD);
    for (int t = 0; t < T; ++t) model_.embed_row(tokens[t], emb_.data() + (size_t) t * N_EMBD);
    bep_->upload(x_, emb_.data(), (size_t) T * N_EMBD * sizeof(float));
    if (ple_emb) {
        // the PLE layer may not be on device 0; the gather goes to its card
        const int pd = (PLE_LAYER >= model_.il0() && PLE_LAYER <= model_.il1())
                     ? model_.dev_of(PLE_LAYER) : cur_dev_;
        model_.dev(pd).upload(pool_[pd].pleemb, ple_emb, (size_t) T * N_EMBD * sizeof(float));
    } else if (need_ple_) {
        throw std::runtime_error("the layer range includes the PLE layer but no PLE gather was given");
    }
    rec.tap(*bep_, "model.input_embed", -1, x_ + tl * N_EMBD, N_EMBD);

    // "the wide residual starts as hc identical copies of the embedding" (324)
    bep_->hc_broadcast(x_, res_hc_, T_);
    rec.tap(*bep_, "hc_init", -1, res_hc_ + tl * HC_DIM, HC_DIM);
    xn_ready_ = false;

    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).timer_start();
    const auto t_issue0 = std::chrono::steady_clock::now();

    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        const LayerWeights & L = model_.layer(il);
        LayerState & st = lstate_[il - model_.il0()];

        // ---- device boundary -------------------------------------------
        // Everything the next range needs is the wide residual: 4 x 2 560
        // f32 = 40 KB. The destination waits on an event recorded on the
        // source's stream and pulls the bytes on its own -- no host call, so
        // this is M5's ~30 us of P2P and nothing else.
        if (L.dev != cur_dev_) {
            if (cfg_.progress && pos_ == 0)
                std::printf("  range done: dev%d up to layer %d, crossing to dev%d\n",
                            cur_dev_, il - 1, L.dev);
            std::fflush(stdout);
            const int src_dev = cur_dev_;
            float * src = pool_[src_dev].res_hc[bank_];
            bind(L.dev);
            // This card is about to have bank_ written under it by the peer
            // copy; wait for ITS consumer to be done with the bank first.
            bep_->boundary_wait_free(bank_);
            bep_->boundary_recv(res_hc_, model_.dev(src_dev), src,
                                (size_t) T_ * HC_DIM * sizeof(float), bank_);
            xn_ready_ = false;             // xn was computed on the other card
        }

        bep_->set_debug_context(L.recurrent ? "GDN layer" : "QSA layer", il);
        if (L.is_ple) { layer_ple(L, rec); xn_ready_ = false; }

        hc_mix(L, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject,
               res_hc_, mixed_, inject_, il, rec, "", xn_ready_);

        if (L.recurrent) layer_gdn(L, st, il, rec);
        else             layer_qsa(L, st, il, rec);

        // The combine also produces the norm the FFN hc_mix would have
        // launched: both reduce over the same 2 560 elements of one stream.
        bep_->hc_combine_norm(res_hc_, blk_, inject_, L.hc_ffn_norm, xn_, eps_, T_);
        rec.tap(*bep_, "hc_combine", il, res_hc_ + tl * HC_DIM, HC_DIM);

        hc_mix(L, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject,
               res_hc_, mixed_, inject_, il, rec, ".2", /*xn_ready=*/true);

        layer_ffn(L, il, rec);

        // Same fusion across the layer boundary: the closing combine produces
        // the NEXT layer's attention norm -- unless that layer is the PLE
        // layer (which rewrites res_hc first), sits on ANOTHER CARD, or does
        // not exist. The second hc_combine's node is renamed l_last a line
        // later, so the dump has no `hc_combine-<il>.2`.
        const bool has_next  = (il + 1 <= model_.il1());
        const bool fuse_next = has_next && !model_.layer(il + 1).is_ple &&
                               model_.layer(il + 1).dev == L.dev;
        bep_->hc_combine_norm(res_hc_, blk_, inject_,
                            fuse_next ? model_.layer(il + 1).hc_attn_norm : nullptr,
                            xn_, eps_, T_);
        xn_ready_ = fuse_next;
        rec.tap(*bep_, "l_last", il, res_hc_ + tl * HC_DIM, HC_DIM);
    }

    // ---- head: the final mixer IS the output norm (qwen4exp.cpp:380) ----
    int greedy = -1;
    if (cfg_.progress && pos_ == 0) {
        std::printf("  range done: dev%d up to layer %d (last)\n", cur_dev_, model_.il1());
        std::fflush(stdout);
    }
    // The head reads the LAST ROW only: a prompt needs one logit vector, and
    // running lm_head (248 320 rows of Q6_K, 521 MB) over a whole chunk would
    // be the single most expensive thing in a prefill for no use at all.
    if (model_.have_head()) {
        bep_->set_debug_context("head", -1);
        bep_->rms_norm_mul(res_hc_ + tl * HC_DIM, model_.head_norm(), head_xn_,
                           N_EMBD, HC, HC_DIM, eps_);
        GemvJob dn; dn.W = &model_.head_down(); dn.out = head_lo_;
        dn.epi = GE_SCALE_SILU; dn.arg = 1.0f / (float) HC;
        bep_->gemv_batch(&dn, 1, head_xn_, GG_HC_DOWN_INJECT);
        GemvJob up; up.W = &model_.head_up(); up.out = head_gate_; up.epi = GE_SIGMOID;
        bep_->gemv_hc_gate_collapse(up, head_lo_, head_xn_, head_out_, GG_HC_UP, 1);
        rec.tap(*bep_, "result_norm", -1, head_out_, N_EMBD);

        GemvJob lm; lm.W = &model_.lm_head(); lm.out = logits_all_;
        bep_->gemv_batch(&lm, 1, head_out_, GG_LM_HEAD);
        rec.tap(*bep_, "result_output", -1, logits_all_, (size_t) model_.lm_head().rows);

        bep_->argmax(logits_all_, (int) model_.lm_head().rows, greedy_id_);
    }

    // Everything above only ENQUEUED work; the first thing that waits is
    // below. This is the host cost of a token.
    last_issue_ms_ = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_issue0).count();

    if (cfg_.progress && pos_ == 0) { std::printf("  head done on dev%d\n", cur_dev_); std::fflush(stdout); }
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).prof_end_token();

    // EVERYTHING BELOW WAITS, and a pipelined chunk does none of it: the
    // timer read is an event synchronise and the greedy id is a download, and
    // either one would drain the queue that the next chunk is supposed to be
    // filling. `last_body_ms_` is left at 0 to say so rather than repeating
    // the previous chunk's number.
    if (!flush) { last_body_ms_ = 0.0; pos_ += T_; return -1; }

    last_body_ms_ = bep_->timer_stop_ms();

    if (model_.have_head()) {
        bep_->download(&greedy, greedy_id_, sizeof(int));   // the one int per token
    }

    // The routed ids are logged device-to-device inside the body and read
    // back HERE, after it, so no layer pays a sync for them.
    // `routed_ids()` reports the LAST row of the chunk -- with --chunk 1
    // (which is what the routing oracle is run at) that is every position.
    if (capture_ && (cfg_.verbose || cfg_.log_routing)) {
        const int nl = model_.il1() - model_.il0() + 1;
        const size_t ls = (size_t) max_T_ * N_EXPERT_USED;
        std::vector<float> w((size_t) nl * ls);
        for (int d = 0; d < model_.n_devices(); ++d) {
            model_.dev(d).sync();
            std::vector<int> ids((size_t) nl * ls);
            model_.dev(d).download(ids.data(), pool_[d].ids_log, ids.size() * sizeof(int));
            model_.dev(d).download(w.data(),   pool_[d].wts_log, w.size() * sizeof(float));
            for (int l = 0; l < nl; ++l) {
                if (model_.layer(model_.il0() + l).dev != d) continue;
                const size_t o = (size_t) l * ls + tl * N_EXPERT_USED;
                for (int k = 0; k < N_EXPERT_USED; ++k)
                    routed_ids_[(size_t) l * N_EXPERT_USED + k] = ids[o + k];
                if (!cfg_.verbose) continue;
                std::printf("moe_ids il=%d pos=%d:", model_.il0() + l, pos_ + T_ - 1);
                for (int k = 0; k < N_EXPERT_USED; ++k)
                    std::printf(" %d(%.4f)", ids[o + k], w[o + k]);
                std::printf("\n");
            }
        }
    }

    pos_ += T_;
    return greedy;
}

// Design 9.1's KV budget, as arithmetic rather than assertion: per QSA layer
// per cell, K and V at 2 heads x 256 dims in q8_0 (32 int8 plus one f16 a
// block = 34 B per 32 values) plus one pooled bf16 indexer key per `ratio`
// cells.
void DecodeRunner::report_cache_bytes(FILE * out) const {
    const int max_blocks = (cfg_.ctx + QSA_RATIO - 1) / QSA_RATIO;
    const size_t per_layer =
        2 * ((size_t) cfg_.ctx * N_KV_HEADS * HEAD_DIM)                       // K,V int8
      + 2 * ((size_t) cfg_.ctx * N_KV_HEADS * (HEAD_DIM / 32) * sizeof(uint16_t))
      + (size_t) max_blocks * IDX_DIM * sizeof(uint16_t)                      // pooled keys
      + 3 * IDX_DIM * sizeof(float);                                          // sum, raw0, new
    for (int d = 0; d < model_.n_devices(); ++d) {
        int qsa = 0, gdn = 0;
        for (int il = model_.il0(); il <= model_.il1(); ++il) {
            if (model_.dev_of(il) != d) continue;
            if (is_recurrent_layer(il)) ++gdn; else ++qsa;
        }
        const size_t gdn_bytes = (size_t) gdn *
            ((size_t) GDN_CONV_K * GDN_CONV_DIM +
             (size_t) GDN_V_HEADS * GDN_STATE * GDN_STATE) * sizeof(float);
        std::fprintf(out,
            "kv_bytes_dev%d=%.3f GB  (qsa_layers=%d x %.1f MB at ctx=%d, "
            "%.0f B/token/layer)  gdn_state_dev%d=%.3f GB (gdn_layers=%d)\n",
            d, (double)((size_t) qsa * per_layer) / 1e9, qsa, per_layer / 1e6, cfg_.ctx,
            (double) per_layer / cfg_.ctx, d, (double) gdn_bytes / 1e9, gdn);
    }
}

// ------------------------------------------- serving: state save/restore --
//
// The pieces, in the one order every one of the three functions below walks.
// Only what a chunk boundary CARRIES is here: the conv windows' chunk slots
// are written before they are read inside a chunk (conv_slide has already
// moved the tail into the history slots by the time step() returns), and the
// per-token scratch is dead between chunks by construction.
void DecodeRunner::state_pieces(std::vector<StatePiece> & out) {
    out.clear();
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & st = lstate_[il - model_.il0()];
        Backend * be = &model_.dev_for(il);
        if (is_recurrent_layer(il)) {
            out.push_back({ be, st.conv_win,
                            (size_t) GDN_CONV_HIST * GDN_CONV_DIM * sizeof(float) });
            out.push_back({ be, st.gdn_state,
                            (size_t) GDN_V_HEADS * GDN_STATE * GDN_STATE * sizeof(float) });
        } else {
            out.push_back({ be, st.idx_sum,  (size_t) IDX_DIM * sizeof(float) });
            out.push_back({ be, st.idx_raw0, (size_t) IDX_DIM * sizeof(float) });
        }
    }
    if (need_ple_) {
        const int pd = model_.dev_of(PLE_LAYER);
        out.push_back({ &model_.dev(pd), pool_[pd].ple_win,
                        (size_t) PLE_CONV_HIST * HC_DIM * sizeof(float) });
    }
}

// The positional pieces, in the one order save_kv and load_kv both walk: per
// QSA layer, K then V quants, K then V scales, then the pooled indexer keys.
// `layout_len` sets the strides of the host image, `copy_len` how many cells
// are actually moved -- an image of 4 916 cells restores a 4 096-cell prefix
// by copying less out of each run, never by re-packing it.
void DecodeRunner::kv_plan(std::vector<KvPiece> & out, int layout_len, int copy_len) {
    out.clear();
    if (copy_len > layout_len) copy_len = layout_len;
    if (copy_len <= 0) return;
    auto qs = [](int n) { return (size_t) n * N_KV_HEADS * HEAD_DIM; };
    auto sc = [](int n) { return (size_t) n * N_KV_HEADS * (HEAD_DIM / 32) * sizeof(uint16_t); };
    auto pl = [](int n) {
        return (size_t) ((n + QSA_RATIO - 1) / QSA_RATIO) * IDX_DIM * sizeof(uint16_t);
    };
    size_t off = 0;
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        if (is_recurrent_layer(il)) continue;
        LayerState & st = lstate_[il - model_.il0()];
        Backend * be = &model_.dev_for(il);
        out.push_back({ be, st.kqs, off, qs(copy_len) });          off += qs(layout_len);
        out.push_back({ be, st.vqs, off, qs(copy_len) });          off += qs(layout_len);
        out.push_back({ be, st.ksc, off, sc(copy_len) });          off += sc(layout_len);
        out.push_back({ be, st.vsc, off, sc(copy_len) });          off += sc(layout_len);
        out.push_back({ be, st.idx_pooled, off, pl(copy_len) });   off += pl(layout_len);
    }
}

size_t DecodeRunner::kv_bytes(int len) const {
    if (len <= 0) return 0;
    const size_t per_layer =
          2 * ((size_t) len * N_KV_HEADS * HEAD_DIM)
        + 2 * ((size_t) len * N_KV_HEADS * (HEAD_DIM / 32) * sizeof(uint16_t))
        + (size_t) ((len + QSA_RATIO - 1) / QSA_RATIO) * IDX_DIM * sizeof(uint16_t);
    size_t n = 0;
    for (int il = model_.il0(); il <= model_.il1(); ++il)
        if (!is_recurrent_layer(il)) n += per_layer;
    return n;
}

void DecodeRunner::sync_devices() {
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
}

void DecodeRunner::save_rec_async(void * dst) {
    std::vector<StatePiece> pieces;
    state_pieces(pieces);
    char * p = (char *) dst;
    for (const auto & s : pieces) { s.be->download_async(p, s.ptr, s.bytes); p += s.bytes; }
}

void DecodeRunner::save_kv_async(void * dst, int len) {
    std::vector<KvPiece> plan;
    kv_plan(plan, len, len);
    char * base = (char *) dst;
    for (const auto & p : plan) p.be->download_async(base + p.off, p.ptr, p.bytes);
}

void DecodeRunner::save_kv(void * dst, int len) {
    std::vector<KvPiece> plan;
    kv_plan(plan, len, len);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    char * base = (char *) dst;
    for (const auto & p : plan) p.be->download(base + p.off, p.ptr, p.bytes);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
}

void DecodeRunner::load_kv(const void * src, int layout_len, int copy_len, int pos) {
    std::vector<KvPiece> plan;
    kv_plan(plan, layout_len, copy_len);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    const char * base = (const char *) src;
    for (const auto & p : plan) p.be->upload(p.ptr, base + p.off, p.bytes);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    pos_ = pos;
}

size_t DecodeRunner::rec_bytes() const {
    // const, so it counts rather than walks -- same order, same arithmetic.
    size_t n = 0;
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        if (is_recurrent_layer(il))
            n += (size_t) GDN_CONV_HIST * GDN_CONV_DIM * sizeof(float)
               + (size_t) GDN_V_HEADS * GDN_STATE * GDN_STATE * sizeof(float);
        else
            n += 2 * (size_t) IDX_DIM * sizeof(float);
    }
    if (need_ple_) n += (size_t) PLE_CONV_HIST * HC_DIM * sizeof(float);
    return n;
}

void DecodeRunner::save_rec(void * dst) {
    std::vector<StatePiece> pieces;
    state_pieces(pieces);
    // Every device first: a download that races the chunk still in flight
    // would copy a half-written state, and the chunk before a snapshot is
    // exactly the one the pipeline was allowed not to wait for.
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    char * p = (char *) dst;
    for (const auto & s : pieces) { s.be->download(p, s.ptr, s.bytes); p += s.bytes; }
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
}

void DecodeRunner::load_rec(const void * src, int pos) {
    std::vector<StatePiece> pieces;
    state_pieces(pieces);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    const char * p = (const char *) src;
    for (const auto & s : pieces) { s.be->upload(s.ptr, p, s.bytes); p += s.bytes; }
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    pos_ = pos;
    xn_ready_ = false;
}

void DecodeRunner::reset_state() {
    std::vector<StatePiece> pieces;
    state_pieces(pieces);
    size_t big = 0;
    for (const auto & s : pieces) big = std::max(big, s.bytes);
    std::vector<char> zero(big, 0);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    for (const auto & s : pieces) s.be->upload(s.ptr, zero.data(), s.bytes);
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).sync();
    pos_ = 0;
    xn_ready_ = false;
}

std::vector<float> DecodeRunner::logits_host() {
    std::vector<float> v;
    if (!model_.have_head()) return v;
    v.resize((size_t) model_.lm_head().rows);
    bep_->sync();
    bep_->download(v.data(), logits_all_, v.size() * sizeof(float));
    return v;
}

std::vector<float> DecodeRunner::residual_host() {
    std::vector<float> v(HC_DIM);
    bep_->sync();
    bep_->download(v.data(), res_hc_ + (size_t)(T_ - 1) * HC_DIM, HC_DIM * sizeof(float));
    return v;
}

} // namespace fk
