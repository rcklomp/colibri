// tools/hot-expert/franken/decode/glm5_graph.cpp -- see glm5_graph.h and
// GLM5.md. Every block names the part of ~/src/llama-glm53/src/models/
// glm5next.cpp (GLM) or deepseek4.cpp (DS4) it ports.

#include "glm5_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "decode_oracle.h"
#include "ds4_shapes.h"

namespace fk {
namespace glm5 {

// What GLM borrows from DeepSeek-V4's ops must have DeepSeek-V4's shapes.
static_assert(ds4::HC == HC && ds4::N_EMBD == N_EMBD && ds4::HC_MIX == HC_MIX && ds4::HC_DIM == HC_DIM,
              "the hyper-connection ops of Ds4Ops are shaped for DeepSeek-V4; GLM's must match");

// ---------------------------------------------------------------- runner --

Glm5Runner::Glm5Runner(Glm5Model & model, std::vector<ds4::Ds4Ops *> dops, std::vector<Glm5Ops *> gops,
                       const Glm5Config & cfg)
    : model_(model), dops_(std::move(dops)), gops_(std::move(gops)), cfg_(cfg),
      eps_(model.rms_eps()), ln_eps_(model.ln_eps()), hc_eps_(model.hc_eps()) {
    const int n_dev = model_.n_devices();
    if ((int) dops_.size() != n_dev || (int) gops_.size() != n_dev)
        throw std::runtime_error("Glm5Runner: one Ds4Ops and one Glm5Ops per device");
    for (int d = 0; d < n_dev; ++d)
        if (model_.dev(d).is_gpu()) throw std::runtime_error("Glm5Runner: the GLM GPU path is not built (GLM5.md section 8)");
    const size_t n_pool = (size_t) (cfg_.ctx / KPOOL + 1);
    iscore_stride_ = n_pool;
    scr_.resize((size_t) n_dev);
    for (int d = 0; d < n_dev; ++d) {
        Backend & b = model_.dev(d);
        auto F = [&](size_t n) { float * p = b.alloc_f32(n); owned_.emplace_back(&b, p); return p; };
        auto I = [&](size_t n) { int * p = b.alloc_i32(n); owned_.emplace_back(&b, p); return p; };
        Scratch & s = scr_[(size_t) d];
        s.H = F(HC_DIM); s.Hn = F(HC_DIM); s.mixes = F(HC_MIX + HC); s.pre = F(HC); s.post = F(HC);
        s.comb = F(HC * HC); s.x = F(N_EMBD); s.xn = F(N_EMBD); s.ao = F(N_EMBD); s.fo = F(N_EMBD);
        s.qkv = F(KDA_QKV); s.conv = F(KDA_QKV); s.fa = F(KDA_DIM); s.fb = F(KDA_INNER); s.g = F(KDA_INNER);
        s.beta = F(N_HEAD); s.ga = F(KDA_DIM); s.gb = F(KDA_INNER); s.o = F(KDA_INNER); s.gated = F(KDA_INNER);
        s.qr = F(Q_LORA); s.q = F(Q_WIDTH); s.kv = F(KV_LORA); s.qabs = F((size_t) N_HEAD * KV_LORA);
        s.att = F((size_t) N_HEAD * KV_LORA); s.kqv = F(O_WIDTH); s.ik = F(IDX_DIM); s.igate = F(IDX_DIM);
        s.iq = F(IDX_N_HEAD * IDX_DIM); s.iw = F(IDX_N_HEAD); s.iscore = F(n_pool); s.isel = I(IDX_TOP_POOLS);
        s.dup = F(N_FF_DENSE); s.dgate = F(N_FF_DENSE); s.dh = F(N_FF_DENSE);
        s.rlog = F(N_EXPERT); s.probs = F(N_EXPERT); s.probs_b = F(N_EXPERT);
        s.wraw = F(N_EXPERT_USED); s.wnorm = F(N_EXPERT_USED); s.wsc = F(N_EXPERT_USED);
        s.yg = F((size_t) N_EXPERT_USED * N_FF_EXP); s.yu = F((size_t) N_EXPERT_USED * N_FF_EXP);
        s.yh = F((size_t) N_EXPERT_USED * N_FF_EXP); s.yd = F((size_t) N_EXPERT_USED * N_EMBD);
        s.ywt = F((size_t) N_EXPERT_USED * N_EMBD); s.moe = F(N_EMBD);
        s.sg = F(N_FF_EXP); s.su = F(N_FF_EXP); s.sh = F(N_FF_EXP); s.sd = F(N_EMBD);
        s.ids = I(N_EXPERT_USED);
    }
    if (model_.have_head()) {
        Backend & b = model_.dev(n_dev - 1);
        auto F = [&](size_t n) { float * p = b.alloc_f32(n); owned_.emplace_back(&b, p); return p; };
        hmean_ = F(N_EMBD); hn_ = F(N_EMBD); logits_ = F(N_VOCAB);
        greedy_ = b.alloc_i32(1); owned_.emplace_back(&b, greedy_);
    }
    // positional state, on the layer's own card, zeroed (KDA starts from a
    // zero conv window and a zero state, as the recurrent cache does)
    st_.resize((size_t) (model_.il1() - model_.il0() + 1));
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & s = st_[(size_t) (il - model_.il0())];
        Backend & b = be(il);
        auto R = [&](size_t bytes) {
            void * p = b.alloc_raw(bytes); owned_.emplace_back(&b, p);
            std::vector<unsigned char> z(bytes, 0); b.upload(p, z.data(), bytes);
            return p;
        };
        if (!model_.layer(il).dsa) {
            s.conv = (float *) R((size_t) (KDA_CONV - 1) * KDA_QKV * sizeof(float));
            s.ssm  = (float *) R((size_t) N_HEAD * KDA_DIM * KDA_DIM * sizeof(float));
        } else {
            s.kv      = (uint16_t *) R((size_t) cfg_.ctx * KV_LORA * 2);
            s.kg_rows = KPOOL;           // T = 1: the pool's four cells are the last four positions
            s.kg      = (uint16_t *) R((size_t) s.kg_rows * 2 * IDX_DIM * 2);
            s.pooled  = (uint16_t *) R(n_pool * IDX_DIM * 2);
        }
    }
    routed_.assign((size_t) (model_.il1() - model_.il0() + 1) * N_EXPERT_USED, -1);
}

Glm5Runner::~Glm5Runner() {
    for (auto & o : owned_) o.first->free_buf(o.second);
}

void Glm5Runner::report_cache_bytes(FILE * out) const {
    size_t fixed = 0, per_tok_num = 0;     // bytes per token * KPOOL, to stay integral
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        if (model_.layer(il).dsa) {
            per_tok_num += (size_t) KPOOL * KV_LORA * 2 + (size_t) IDX_DIM * 2;
            fixed += (size_t) KPOOL * 2 * IDX_DIM * 2;
        } else {
            fixed += (size_t) (KDA_CONV - 1) * KDA_QKV * 4 + (size_t) N_HEAD * KDA_DIM * KDA_DIM * 4;
        }
    }
    std::fprintf(out, "glm5_cache fixed_bytes=%zu (KDA state + conv windows, indexer rings) "
                      "per_token_bytes=%.1f allocated_for_ctx=%.3f GB (layers %d-%d, ctx %d) scoring=%d\n",
                 fixed, per_tok_num / (double) KPOOL, ((per_tok_num / (double) KPOOL) * cfg_.ctx + fixed) / 1e9,
                 model_.il0(), model_.il1(), cfg_.ctx, (int) scoring());
}

// build_hc_pre (DS4): flat RMS norm (no gamma, the model's rms eps), the
// 24-row mix, the pre/post/comb split (the hc eps), the pre-weighted sum.
void Glm5Runner::hc_pre(Scratch & s, const Mat & fn, const float * base, const float * scale, int il,
                        Recorder & rec, const char * sfx) {
    Backend & b = be(il);
    ds4::Ds4Ops & o = dop(il);
    b.rms_norm_mul(s.H, nullptr, s.Hn, HC_DIM, 1, HC_DIM, eps_);
    o.gemv(fn, s.Hn, s.mixes, 1);
    rec.tap(b, "hc_mixes", il, s.mixes, HC_MIX, sfx);
    o.hc_split(s.mixes, scale, base, s.pre, s.post, s.comb, hc_eps_, HC_SINKHORN_ITERS, 1);
    rec.tap(b, "hc_pre",  il, s.pre, HC, sfx);
    rec.tap(b, "hc_post", il, s.post, HC, sfx);
    rec.tap(b, "hc_comb", il, s.comb, HC * HC, sfx);
    o.hc_weighted_sum(s.H, s.pre, s.x, 1);
}

// build_kda_layer (GLM): q/k/v projections, one conv over q|k|v then SiLU,
// l2-normed q and k, the per-channel gate from the f low-rank pair, beta,
// the KDA recurrence, a gated RMS norm with the g low-rank pair, attn_output.
void Glm5Runner::kda(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    LayerState & st = st_[(size_t) (il - model_.il0())];
    Scratch & s = S(il);
    Backend & b = be(il);
    ds4::Ds4Ops & o = dop(il);
    Glm5Ops & g = gop(il);

    o.gemv(L.wq, s.xn, s.qkv, 1);
    o.gemv(L.wk, s.xn, s.qkv + KDA_INNER, 1);
    o.gemv(L.wv, s.xn, s.qkv + 2 * KDA_INNER, 1);
    rec.tap(b, "kda_qkv", il, s.qkv, KDA_QKV);
    g.kda_conv(s.qkv, st.conv, L.conv_q, L.conv_k, L.conv_v, s.conv, 1);
    rec.tap(b, "kda_conv", il, s.conv, KDA_QKV);
    float * q = s.conv, * k = s.conv + KDA_INNER, * v = s.conv + 2 * KDA_INNER;
    b.l2_norm(q, q, KDA_DIM, N_HEAD, KDA_L2_EPS);
    b.l2_norm(k, k, KDA_DIM, N_HEAD, KDA_L2_EPS);
    rec.tap(b, "kda_q_norm", il, q, KDA_INNER);
    rec.tap(b, "kda_k_norm", il, k, KDA_INNER);

    // g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)) -- f/g/beta read the
    // layer INPUT, not the convolved q/k/v
    o.gemv(L.f_a, s.xn, s.fa, 1);
    o.gemv(L.f_b, s.fa, s.fb, 1);
    g.kda_gate(s.fb, L.dt_b, L.ssm_a, KDA_GATE_LOWER_BOUND, s.g, 1);
    rec.tap(b, "kda_gate", il, s.g, KDA_INNER);
    o.gemv(L.beta, s.xn, s.beta, 1);
    b.unary(FK_SIGMOID, s.beta, s.beta, N_HEAD);
    rec.tap(b, "kda_beta", il, s.beta, N_HEAD);

    g.kda_step(st.ssm, q, k, v, s.g, s.beta, s.o, KDA_QKV, 1);
    rec.tap(b, "kda_scan_out", il, s.o, KDA_INNER);

    // RMS norm per head with ssm_norm, times a PLAIN sigmoid gate
    o.gemv(L.g_a, s.xn, s.ga, 1);
    o.gemv(L.g_b, s.ga, s.gb, 1);
    b.gated_rms_norm(s.o, L.o_norm, s.gb, s.gated, KDA_DIM, N_HEAD, eps_);
    rec.tap(b, "kda_normed", il, s.gated, KDA_INNER);
    o.gemv(L.wo, s.gated, s.ao, 1);
    rec.tap(b, "kda_out", il, s.ao, N_EMBD);
}

// build_dsa_layer + build_indexer (GLM): q_a and its norm; the indexer's
// key (LayerNorm) and gate stored f16 per cell, a pool of four cells pooled
// when it completes; with scoring, the pool scores and the top-512 pools;
// q_b, the latent kv and its norm, q absorbed through wk_b, the latent
// cached f16, attention (MQA: every head against the one latent row), the
// latent back out through wv_b, attn_output.
void Glm5Runner::dsa(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    LayerState & st = st_[(size_t) (il - model_.il0())];
    Scratch & s = S(il);
    Backend & b = be(il);
    ds4::Ds4Ops & o = dop(il);
    Glm5Ops & g = gop(il);
    const int p = pos_;

    o.gemv(L.wq_a, s.xn, s.qr, 1);
    b.rms_norm_mul(s.qr, L.q_a_norm, s.qr, Q_LORA, 1, Q_LORA, eps_);
    rec.tap(b, "dsa_q_a_norm", il, s.qr, Q_LORA);

    // the indexer's STORE runs whether or not it scores (glm5next.cpp)
    o.gemv(L.idx_k, s.xn, s.ik, 1);
    g.layer_norm(s.ik, L.idx_k_norm, L.idx_k_norm_b, s.ik, IDX_DIM, 1, ln_eps_);
    rec.tap(b, "indexer_k", il, s.ik, IDX_DIM);
    o.gemv(L.idx_gate, s.xn, s.igate, 1);
    rec.tap(b, "indexer_gate", il, s.igate, IDX_DIM);
    g.store_f16(s.ik,    IDX_DIM, st.kg, 2 * IDX_DIM, 0,       IDX_DIM, st.kg_rows, p, 1);
    g.store_f16(s.igate, IDX_DIM, st.kg, 2 * IDX_DIM, IDX_DIM, IDX_DIM, st.kg_rows, p, 1);
    g.idx_pool(st.kg, st.kg_rows, L.idx_ape, st.pooled, p, 1);

    const int n_vis = (p + 1) / KPOOL;
    const int * sel = nullptr;
    if (scoring()) {
        o.gemv(L.idx_q_b, s.qr, s.iq, 1);             // no rope: n_rot is 0 for the whole tower
        rec.tap(b, "indexer_q", il, s.iq, IDX_N_HEAD * IDX_DIM);
        o.gemv(L.idx_proj, s.xn, s.iw, 1);            // F32 weights, PREC_F32
        b.scale(s.iw, 1.0f / sqrtf((float) (IDX_DIM * IDX_N_HEAD)), s.iw, IDX_N_HEAD);
        rec.tap(b, "indexer_weights", il, s.iw, IDX_N_HEAD);
        if (n_vis > 0) {
            g.idx_scores(s.iq, s.iw, st.pooled, p, s.iscore, (int) iscore_stride_, 1);
            rec.tap(b, "indexer_pool_score", il, s.iscore, (size_t) n_vis);
            g.idx_topk(s.iscore, (int) iscore_stride_, p, s.isel, 1);
            sel = s.isel;
            if (rec.enabled()) {
                const int n_sel = std::min(n_vis, IDX_TOP_POOLS);
                std::vector<int> pools((size_t) n_sel), cells;
                b.download(pools.data(), s.isel, (size_t) n_sel * sizeof(int));
                for (int pl : pools) for (int j = 0; j < KPOOL; ++j) cells.push_back(pl * KPOOL + j);
                rec.tap_ints("indexer_top_k", il, cells);   // the pools' cells; the tail rides the mask
            }
        }
    }

    o.gemv(L.wq_b, s.qr, s.q, 1);
    rec.tap(b, "dsa_q_b", il, s.q, Q_WIDTH);
    o.gemv(L.wkv_a, s.xn, s.kv, 1);
    b.rms_norm_mul(s.kv, L.kv_a_norm, s.kv, KV_LORA, 1, KV_LORA, eps_);
    rec.tap(b, "dsa_kv_a_norm", il, s.kv, KV_LORA);
    for (int h = 0; h < N_HEAD; ++h)                 // q_nope through wk_b, head by head
        o.gemv(L.wk_b[h], s.q + (size_t) h * QK_HEAD, s.qabs + (size_t) h * KV_LORA, 1);
    rec.tap(b, "dsa_q_absorbed", il, s.qabs, (size_t) N_HEAD * KV_LORA);
    rec.tap(b, "dsa_kv_latent", il, s.kv, KV_LORA);

    g.store_f16(s.kv, KV_LORA, st.kv, KV_LORA, 0, KV_LORA, 0, p, 1);   // cpy_k, then attend
    g.mla_attn(s.qabs, st.kv, p, sel, 1.0f / sqrtf((float) QK_HEAD), s.att, 1);
    for (int h = 0; h < N_HEAD; ++h)                 // v_mla: the latent back to 256 a head
        o.gemv(L.wv_b[h], s.att + (size_t) h * KV_LORA, s.kqv + (size_t) h * V_HEAD, 1);
    rec.tap(b, "kqv_out", il, s.kqv, O_WIDTH);
    o.gemv(L.wo, s.kqv, s.ao, 1);
    rec.tap(b, "dsa_out", il, s.ao, N_EMBD);
}

// build_layer_ffn (GLM): layers 0-2 a dense SwiGLU clamped at 10 (build_ffn
// applies swiglu_clamp_shexp[il] to every PAR SiLU FFN of this arch); the
// rest build_moe_ffn (sigmoid, exp_probs_b for the selection only, top-8,
// normalised, x2.5, clamped) plus the clamped, UNSCALED shared expert.
void Glm5Runner::ffn(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Scratch & s = S(il);
    Backend & b = be(il);
    ds4::Ds4Ops & o = dop(il);
    Glm5Ops & g = gop(il);
    const size_t U = N_EXPERT_USED;

    if (!L.moe) {
        o.gemv(L.ffn_up, s.xn, s.dup, 1);
        rec.tap(b, "ffn_up", il, s.dup, N_FF_DENSE);
        o.gemv(L.ffn_gate, s.xn, s.dgate, 1);
        rec.tap(b, "ffn_gate", il, s.dgate, N_FF_DENSE);
        o.swiglu_clamp(s.dgate, s.dup, s.dh, N_FF_DENSE, SWIGLU_CLAMP);
        rec.tap(b, "ffn_swiglu_limited", il, s.dh, N_FF_DENSE);
        o.gemv(L.ffn_down, s.dh, s.fo, 1);
        rec.tap(b, "ffn_out", il, s.fo, N_EMBD);
        return;
    }

    o.gemv(L.gate_inp, s.xn, s.rlog, 1);
    rec.tap(b, "ffn_moe_logits", il, s.rlog, N_EXPERT);
    g.router(s.rlog, L.exp_probs_b, s.probs, s.probs_b, s.ids, s.wraw, s.wnorm, s.wsc, 1);
    rec.tap(b, "ffn_moe_probs", il, s.probs, N_EXPERT);
    rec.tap(b, "ffn_moe_probs_biased", il, s.probs_b, N_EXPERT);
    if (cfg_.log_routing || rec.enabled()) {
        std::vector<int> ids(U);
        b.download(ids.data(), s.ids, U * sizeof(int));
        std::copy(ids.begin(), ids.end(), routed_.begin() + (size_t) (il - model_.il0()) * U);
    }
    rec.tap(b, "ffn_moe_weights", il, s.wraw, U);
    rec.tap(b, "ffn_moe_weights_norm", il, s.wnorm, U);
    rec.tap(b, "ffn_moe_weights_scaled", il, s.wsc, U);

    o.gemv(L.sh_up, s.xn, s.su, 1);
    rec.tap(b, "ffn_up", il, s.su, N_FF_EXP);
    o.gemv(L.sh_gate, s.xn, s.sg, 1);
    rec.tap(b, "ffn_gate", il, s.sg, N_FF_EXP);
    o.swiglu_clamp(s.sg, s.su, s.sh, N_FF_EXP, SWIGLU_CLAMP);
    rec.tap(b, "ffn_swiglu_limited", il, s.sh, N_FF_EXP);
    o.gemv(L.sh_down, s.sh, s.sd, 1);
    rec.tap(b, "ffn_shexp", il, s.sd, N_EMBD);

    g.moe_gate_up(L.et, s.ids, s.xn, s.yg, s.yu, 1);
    rec.tap(b, "ffn_moe_up",   il, s.yu, U * N_FF_EXP);
    rec.tap(b, "ffn_moe_gate", il, s.yg, U * N_FF_EXP);
    o.swiglu_clamp(s.yg, s.yu, s.yh, (int) (U * N_FF_EXP), SWIGLU_CLAMP);
    rec.tap(b, "ffn_moe_swiglu_limited", il, s.yh, U * N_FF_EXP);
    g.moe_down(L.et, s.ids, s.yh, s.yd, 1);
    rec.tap(b, "ffn_moe_down", il, s.yd, U * N_EMBD);
    o.moe_accum(s.yd, s.wsc, N_EXPERT_USED, N_EMBD, s.ywt, s.moe, 1);
    rec.tap(b, "ffn_moe_weighted", il, s.ywt, U * N_EMBD);
    rec.tap(b, "ffn_moe_out", il, s.moe, N_EMBD);

    b.binary(FK_ADD, s.moe, s.sd, s.fo, N_EMBD);
    rec.tap(b, "ffn_out", il, s.fo, N_EMBD);
}

// One layer (GLM graph::graph, the loop body).
void Glm5Runner::layer(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Scratch & s = S(il);
    Backend & b = be(il);
    ds4::Ds4Ops & o = dop(il);

    hc_pre(s, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, il, rec, "");
    rec.tap(b, "hc_attn_pre", il, s.x, N_EMBD);
    b.rms_norm_mul(s.x, L.attn_norm, s.xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "attn_norm", il, s.xn, N_EMBD);
    if (L.dsa) dsa(il, rec); else kda(il, rec);
    o.hc_post(s.ao, s.H, s.post, s.comb, s.Hn, 1);
    std::swap(s.H, s.Hn);
    rec.tap(b, "hc_attn_post", il, s.H, HC_DIM);

    hc_pre(s, L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, il, rec, ".2");
    rec.tap(b, "hc_ffn_pre", il, s.x, N_EMBD);
    b.rms_norm_mul(s.x, L.ffn_norm, s.xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "ffn_norm", il, s.xn, N_EMBD);
    ffn(il, rec);
    o.hc_post(s.fo, s.H, s.post, s.comb, s.Hn, 1);
    std::swap(s.H, s.Hn);
    rec.tap(b, "l_last", il, s.H, HC_DIM);
}

int Glm5Runner::step(int32_t token, Recorder & rec) {
    const int il0 = model_.il0(), il1 = model_.il1();
    if (pos_ + 1 > cfg_.ctx) throw std::runtime_error("context full");
    Backend & b0 = be(il0);
    Scratch & s0 = S(il0);
    std::vector<float> e(N_EMBD);
    model_.embed_row(token, e.data());
    b0.upload(s0.x, e.data(), N_EMBD * sizeof(float));
    std::fill(routed_.begin(), routed_.end(), -1);

    const bool head = model_.have_head() && il1 == N_LAYER - 1;
    int il = il0;
    while (il <= il1) {
        const int d = dev_of(il);
        int last = il;
        while (last + 1 <= il1 && dev_of(last + 1) == d) ++last;
        if (il == il0) {
            dop(il0).hc_init(s0.x, s0.H, 1);             // exact copies into the four streams
            rec.tap(b0, "hc_init", -1, s0.H, HC_DIM);
        } else {
            be(il).boundary_recv(S(il).H, be(il - 1), S(il - 1).H, (size_t) HC_DIM * sizeof(float), 0);
        }
        for (int l = il; l <= last; ++l) layer(l, rec);
        il = last + 1;
    }
    int id = -1;
    if (head) {
        Backend & b = model_.dev(model_.n_devices() - 1);
        Scratch & s = scr_.back();
        gops_.back()->hc_mean(s.H, hmean_, 1);           // the unweighted stream mean, not DS4's hc_head
        rec.tap(b, "hc_mean", -1, hmean_, N_EMBD);
        b.rms_norm_mul(hmean_, model_.output_norm(), hn_, N_EMBD, 1, N_EMBD, eps_);
        rec.tap(b, "result_norm", -1, hn_, N_EMBD);
        dops_.back()->gemv(model_.output(), hn_, logits_, 1);
        rec.tap(b, "result_output", -1, logits_, N_VOCAB);
        b.argmax(logits_, N_VOCAB, greedy_);
        b.download(&id, greedy_, sizeof(int));
    }
    ++pos_;
    return id;
}

// ------------------------------------------------------------------- CLI --

namespace {

bool parse_range(const std::string & s, int & a, int & b) {
    const size_t d = s.find('-');
    if (d == std::string::npos) return false;
    a = std::atoi(s.substr(0, d).c_str());
    b = std::atoi(s.substr(d + 1).c_str());
    return a >= 0 && b >= a;
}

std::vector<double> parse_list_d(const std::string & s) {
    std::vector<double> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) v.push_back(std::atof(item.c_str()));
    return v;
}

// `<dir>/moe_ids.txt` from ../oracle_dump.cpp: lines "il pos: id id ...".
bool load_routing(const std::string & dir, std::map<std::pair<int, int>, std::vector<int>> & ref) {
    std::ifstream f(dir + "/moe_ids.txt");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        int il, pos; char colon;
        if (!(is >> il >> pos >> colon)) continue;
        std::vector<int> ids; int v;
        while (is >> v) ids.push_back(v);
        ref[{il, pos}] = ids;
    }
    return !ref.empty();
}

} // namespace

int glm5_main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_path, oracle_dir, dump_dir, routing_dir;
    std::vector<int32_t> tokens;
    std::vector<int> split;
    int il0 = 0, il1 = N_LAYER - 1, n_devices = 3, threads = 4, ctx = 512, greedy_n = 0, max_tokens = 0;
    bool with_head = true, use_cpu = false, plan_only = false, quant_act = false;
    double min_cos = 0.999;
    Placement pl;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--routing"&& i + 1 < argc) routing_dir = argv[++i];
        else if (a == "--dump"   && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--placement" && i + 1 < argc) pl.hist_dir = argv[++i];
        else if (a == "--expert-gb" && i + 1 < argc) pl.expert_gb = parse_list_d(argv[++i]);
        else if (a == "--split" && i + 1 < argc) {
            for (double v : parse_list_d(argv[++i])) split.push_back((int) v);
        }
        else if (a == "--cpu")                    use_cpu = true;
        else if (a == "--plan-only")              plan_only = true;
        else if (a == "--quant-act")              quant_act = true;   // Q8_0 activations, as ggml's CPU matmul
        else if (a == "--no-head")                with_head = false;
        else if (a == "--devices"&& i + 1 < argc) n_devices = std::atoi(argv[++i]);
        else if (a == "--threads"&& i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--greedy" && i + 1 < argc) greedy_n = std::atoi(argv[++i]);
        else if (a == "--min-cos"&& i + 1 < argc) min_cos = std::atof(argv[++i]);
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) {
            if (!parse_range(argv[++i], il0, il1)) { std::fprintf(stderr, "bad --layers\n"); return 2; }
        } else if (a == "--tokens") {
            while (i + 1 < argc && argv[i + 1][0] != '-') tokens.push_back((int32_t) std::atoll(argv[++i]));
        } else if (a == "--tokens-file" && i + 1 < argc) {
            std::ifstream f(argv[++i]);
            if (!f) { std::fprintf(stderr, "no --tokens-file %s\n", argv[i]); return 2; }
            long long v;
            while (f >> v) tokens.push_back((int32_t) v);
        }
        else { std::fprintf(stderr, "glm5: unrecognised argument: %s\n", a.c_str()); return 2; }
    }
#ifdef FRANKEN_NO_HIP
    use_cpu = true;
#endif
    if (model_path.empty() || (tokens.empty() && !plan_only)) {
        std::fprintf(stderr, "usage: franken_decode --model <glm5next shard> --tokens <ids> [--cpu]\n"
                             "  [--layers A-B] [--no-head] [--devices N] [--split L1,L2] [--ctx N] [--threads N]\n"
                             "  [--tokens-file F [--max-tokens N]] [--quant-act]\n"
                             "  [--oracle DIR] [--routing DIR] [--dump DIR] [--min-cos X] [--greedy N]\n"
                             "  [--plan-only --placement DIR --expert-gb X[,Y,Z]]   (the VRAM / miss plan; no model byte read)\n");
        return 2;
    }
    if (!use_cpu) {
        std::fprintf(stderr, "glm5: the GPU path is not built yet (GLM5.md section 8); run with --cpu\n");
        return 2;
    }
    if (max_tokens > 0 && (int) tokens.size() > max_tokens) tokens.resize((size_t) max_tokens);
    if (!plan_only && (int) tokens.size() + greedy_n > ctx) { std::fprintf(stderr, "--ctx too small\n"); return 2; }
    if (n_devices < 1) n_devices = 1;
    pl.ctx = ctx;

    try {
        std::unique_ptr<Backend> cpu(make_cpu_backend(threads));
        if (quant_act) cpu->set_quant_act(true);
        std::vector<Backend *> devs((size_t) n_devices, cpu.get());
        std::vector<std::unique_ptr<ds4::Ds4Ops>> own_d;
        std::vector<std::unique_ptr<Glm5Ops>> own_g;
        std::vector<ds4::Ds4Ops *> dops;
        std::vector<Glm5Ops *> gops;
        for (Backend * b : devs) {
            own_d.emplace_back(ds4::make_ds4_cpu_ops(*b)); dops.push_back(own_d.back().get());
            own_g.emplace_back(make_glm5_cpu_ops(*b));     gops.push_back(own_g.back().get());
        }
        const bool head = with_head && il1 == N_LAYER - 1;
        Glm5Model model(model_path, devs, il0, il1, head, split, pl);
        std::printf("arch=glm5next backend=%s devices=%d layers=%d-%d head=%d ctx=%d tokens=%zu placed=%.2f GB "
                    "rms_eps=%g ln_eps=%g hc_eps=%g quant_act=%d\n",
                    devs[0]->name(), n_devices, il0, il1, (int) head, ctx, tokens.size(),
                    model.placed_bytes() / 1e9, model.rms_eps(), model.ln_eps(), model.hc_eps(), (int) quant_act);
        if (plan_only) { std::printf("plan-only: stopping before the caches\n"); return 0; }

        Glm5Config cfg; cfg.ctx = ctx; cfg.log_routing = !routing_dir.empty();
        Glm5Runner run(model, dops, gops, cfg);
        run.report_cache_bytes(stdout);

        std::map<std::pair<int, int>, std::vector<int>> ref_routes;
        const bool have_routes = !routing_dir.empty() && load_routing(routing_dir, ref_routes);
        if (!routing_dir.empty() && !have_routes) std::printf("routing PENDING: no %s/moe_ids.txt\n", routing_dir.c_str());
        std::map<int, std::pair<double, int>> overlap;
        int exact_sets = 0, n_sets = 0;
        auto observe = [&](int pos) {
            if (!have_routes) return;
            const std::vector<int> & ids = run.routed_ids();
            for (int il = std::max(il0, N_LEAD_DENSE); il <= il1; ++il) {
                auto it = ref_routes.find({il, pos});
                if (it == ref_routes.end()) continue;
                std::set<int> a(ids.begin() + (size_t) (il - il0) * N_EXPERT_USED,
                                ids.begin() + (size_t) (il - il0 + 1) * N_EXPERT_USED);
                std::set<int> r(it->second.begin(), it->second.end());
                int inter = 0; for (int v : a) inter += (int) r.count(v);
                overlap[il].first += (double) inter / N_EXPERT_USED;
                overlap[il].second += 1;
                exact_sets += (inter == N_EXPERT_USED); ++n_sets;
            }
        };

        Recorder rec;
        int last = -1;
        for (size_t t = 0; t < tokens.size(); ++t) {
            rec.enable(t + 1 == tokens.size());           // the taps are the LAST prompt token's
            last = run.step(tokens[t], rec);
            observe(run.pos() - 1);
            if (t == 0) std::printf("token 0 done\n");
        }
        const std::vector<int> last_ids = run.routed_ids();
        rec.enable(false);
        if (greedy_n > 0 && last >= 0) {
            std::vector<int> greedy{last};
            for (int g = 1; g < greedy_n; ++g) { last = run.step(last, rec); observe(run.pos() - 1); greedy.push_back(last); }
            std::printf("greedy_ids:");
            for (int v : greedy) std::printf(" %d", v);
            std::printf("\n");
            rec.enable(true);
            rec.tap_host("greedy_ids", -1, std::vector<float>(greedy.begin(), greedy.end()));
            rec.enable(false);
        }

        bool ok = true;
        // SLOT ORDER (the DeepSeek CLI's rule): align the per-slot taps to the
        // reference's slot order only when both routers picked the same SET at
        // the last prompt position; unequal sets are left to fail.
        if (have_routes && !oracle_dir.empty()) {
            const int last_pos = (int) tokens.size() - 1;
            for (int il = std::max(il0, N_LEAD_DENSE); il <= il1; ++il) {
                auto it = ref_routes.find({il, last_pos});
                if (it == ref_routes.end() || it->second.size() != (size_t) N_EXPERT_USED) continue;
                const int * mine = last_ids.data() + (size_t) (il - il0) * N_EXPERT_USED;
                int perm[N_EXPERT_USED];
                bool same = true;
                for (int k = 0; k < N_EXPERT_USED && same; ++k) {
                    perm[k] = -1;
                    for (int j = 0; j < N_EXPERT_USED; ++j) if (mine[j] == it->second[k]) perm[k] = j;
                    same = perm[k] >= 0;
                }
                bool ident = true;
                for (int k = 0; k < N_EXPERT_USED; ++k) ident &= same && perm[k] == k;
                if (!same || ident) {
                    if (!same) std::printf("slot_order layer=%d: expert SETS differ at the last position; per-slot taps left as computed\n", il);
                    continue;
                }
                std::printf("slot_order layer=%d: same set, reordered to the reference's slot order:", il);
                for (int k = 0; k < N_EXPERT_USED; ++k) std::printf(" %d", mine[perm[k]]);
                std::printf("\n");
                for (const char * nm : {"ffn_moe_up", "ffn_moe_gate", "ffn_moe_swiglu_limited",
                                        "ffn_moe_down", "ffn_moe_weighted", "ffn_moe_weights",
                                        "ffn_moe_weights_norm", "ffn_moe_weights_scaled"}) {
                    TapValue * t = rec.get_mut(std::string(nm) + "-" + std::to_string(il));
                    if (!t || t->data.size() % N_EXPERT_USED) continue;
                    const size_t w = t->data.size() / N_EXPERT_USED;
                    std::vector<float> v(t->data.size());
                    for (int k = 0; k < N_EXPERT_USED; ++k)
                        std::copy(t->data.begin() + (size_t) perm[k] * w, t->data.begin() + (size_t) (perm[k] + 1) * w,
                                  v.begin() + (size_t) k * w);
                    t->data.swap(v);
                }
            }
        }
        if (have_routes) {
            for (auto & kv : overlap)
                std::printf("routing layer=%d overlap=%.4f positions=%d\n", kv.first,
                            kv.second.first / std::max(1, kv.second.second), kv.second.second);
            std::printf("routing exact_sets=%d/%d\n", exact_sets, n_sets);
        }
        if (!dump_dir.empty()) {
            std::string why;
            if (!dump_taps(rec, dump_dir, why)) { std::printf("dump FAILED: %s\n", why.c_str()); ok = false; }
            else std::printf("dumped %zu taps to %s\n", rec.all().size(), dump_dir.c_str());
        }
        if (!oracle_dir.empty()) {
            Oracle orc;
            std::string why;
            if (!orc.load(oracle_dir, why)) {
                std::printf("oracle PENDING: %s\n", why.c_str());
            } else {
                const bool cmp_ok = orc.compare(rec, {}, min_cos, stdout);
                bool req_ok = true;
                for (int il = il0; il <= il1; ++il) {
                    const std::string key = "l_last-" + std::to_string(il);
                    if (!rec.get(key) || !orc.has(key)) {
                        std::printf("oracle %s REQUIRED but %s\n", key.c_str(),
                                    rec.get(key) ? "absent from the dump" : "not computed");
                        req_ok = false;
                    }
                }
                ok &= cmp_ok && req_ok;
                std::printf("GLM5 CPU ORACLE %s (min_cos=%g)\n", (cmp_ok && req_ok) ? "PASS" : "FAIL", min_cos);
            }
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "glm5 error: %s\n", e.what());
        return 1;
    }
}

} // namespace glm5
} // namespace fk
