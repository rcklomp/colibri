// tools/hot-expert/franken/decode/ds4_graph.cpp -- see ds4_graph.h and
// DEEPSEEK4.md. Every block below names the part of
// ~/src/llama-glm53/src/models/deepseek4.cpp (DS4) it ports.

#include "ds4_graph.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

#include "decode_oracle.h"

namespace fk {
namespace ds4 {

// ------------------------------------------------------------------ rope --
//
// build_attention_impl: a compress-rope layer (ratio != 0) uses the yarn set
// -- base 160000, freq_scale 1/16, ext_factor 1 (llama_context: yarn ->
// 1.0), n_ctx_orig 65536, beta 32/1, attn_factor dsv4_rope_attn_factor() --
// and a ratio-0 layer base 10000 with no scaling at all. The lid query and
// every compressed row always take the compress set.
static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) return 1.0f;
    return 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
}

RopeParams Ds4Runner::rope_compress() const {
    RopeParams r;
    r.freq_base   = COMPRESS_ROPE_FREQ_BASE;
    r.freq_scale  = 1.0f / ROPE_SCALE_FACTOR;
    r.ext_factor  = 1.0f;
    r.attn_factor = dsv4_rope_attn_factor(r.freq_scale, r.ext_factor);
    r.beta_fast   = YARN_BETA_FAST;
    r.beta_slow   = YARN_BETA_SLOW;
    r.n_ctx_orig  = ROPE_N_CTX_ORIG;
    return r;
}

RopeParams Ds4Runner::rope_for(int il) const {
    if (model_.layer(il).ratio != 0) return rope_compress();
    RopeParams r;
    r.freq_base   = ROPE_FREQ_BASE;
    r.freq_scale  = 1.0f;
    r.ext_factor  = 0.0f;
    r.attn_factor = dsv4_rope_attn_factor(1.0f, 0.0f);
    r.n_ctx_orig  = 0;
    return r;
}

// ---------------------------------------------------------------- runner --

Ds4Runner::Ds4Runner(Ds4Model & model, std::vector<Ds4Ops *> ops, const Ds4Config & cfg)
    : model_(model), ops_(std::move(ops)), cfg_(cfg), eps_(model.rms_eps()), hc_eps_(model.hc_eps()) {
    if ((int) ops_.size() != model_.n_devices()) throw std::runtime_error("Ds4Runner: one Ds4Ops per device");
    const int n_dev = model_.n_devices();
    scr_.resize((size_t) n_dev);
    const size_t n_blk = (size_t) (cfg_.ctx / CSA_RATIO + 1);
    for (int d = 0; d < n_dev; ++d) {
        Backend & b = model_.dev(d);
        auto F = [&](size_t n) { float * p = b.alloc_f32(n); owned_.emplace_back(&b, p); return p; };
        auto I = [&](size_t n) { int * p = b.alloc_i32(n); owned_.emplace_back(&b, p); return p; };
        Scratch & s = scr_[(size_t) d];
        s.H = F(HC_DIM); s.Hn = F(HC_DIM); s.mixes = F(HC_MIX + HC); s.pre = F(HC); s.post = F(HC);
        s.comb = F(HC * HC); s.x = F(N_EMBD); s.xn = F(N_EMBD); s.qr = F(Q_LORA);
        s.q = F((size_t) N_HEAD * HEAD_DIM); s.kv = F(HEAD_DIM); s.att = F((size_t) N_HEAD * HEAD_DIM);
        s.oa = F((size_t) O_GROUPS * O_LORA); s.ao = F(N_EMBD);
        s.ckv = F(2 * HEAD_DIM); s.csc = F(2 * HEAD_DIM); s.lkv = F(2 * IDX_DIM); s.lsc = F(2 * IDX_DIM);
        s.cpool = F(HEAD_DIM); s.iq = F((size_t) IDX_N_HEAD * IDX_DIM); s.iw = F(IDX_N_HEAD);
        s.iscore = F(n_blk);
        s.rlog = F(N_EXPERT); s.probs = F(N_EXPERT); s.probs_b = F(N_EXPERT);
        s.wraw = F(N_EXPERT_USED); s.wnorm = F(N_EXPERT_USED); s.wsc = F(N_EXPERT_USED);
        s.yg = F((size_t) N_EXPERT_USED * N_FF_EXP); s.yu = F((size_t) N_EXPERT_USED * N_FF_EXP);
        s.yh = F((size_t) N_EXPERT_USED * N_FF_EXP); s.yd = F((size_t) N_EXPERT_USED * N_EMBD);
        s.ywt = F((size_t) N_EXPERT_USED * N_EMBD); s.moe = F(N_EMBD);
        s.sg = F(N_FF_EXP); s.su = F(N_FF_EXP); s.sh = F(N_FF_EXP); s.sd = F(N_EMBD); s.fo = F(N_EMBD);
        s.isel = I(std::max<size_t>(n_blk, IDX_TOP_K)); s.ids = I(N_EXPERT_USED);
        s.hash = I((size_t) HASH_LAYERS * N_EXPERT_USED);
    }
    if (model_.have_head()) {
        Backend & b = model_.dev(n_dev - 1);
        auto F = [&](size_t n) { float * p = b.alloc_f32(n); owned_.emplace_back(&b, p); return p; };
        hmix_ = F(HC); hpre_ = F(HC); hx_ = F(N_EMBD); hxn_ = F(N_EMBD); logits_ = F(N_VOCAB);
        greedy_ = b.alloc_i32(1); owned_.emplace_back(&b, greedy_);
    }

    // Positional state per layer, on the layer's own card, sized ONCE from ctx.
    st_.resize((size_t) (model_.il1() - model_.il0() + 1));
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & s = st_[(size_t) (il - model_.il0())];
        Backend & b = be(il);
        auto R = [&](size_t bytes) { void * p = b.alloc_raw(bytes); owned_.emplace_back(&b, p); return p; };
        auto G = [&](size_t n) { float * p = b.alloc_f32(n); owned_.emplace_back(&b, p); return p; };
        s.raw = (uint16_t *) R((size_t) N_SWA * HEAD_DIM * 2);
        const int ratio = model_.layer(il).ratio;
        if (ratio == CSA_RATIO) {
            s.ring = 2 * CSA_RATIO;
            s.ck = G((size_t) s.ring * 2 * HEAD_DIM); s.cs = G((size_t) s.ring * 2 * HEAD_DIM);
            s.lk = G((size_t) s.ring * 2 * IDX_DIM);  s.ls = G((size_t) s.ring * 2 * IDX_DIM);
            s.comp = (uint16_t *) R(n_blk * HEAD_DIM * 2);
            s.lid  = (uint16_t *) R(n_blk * IDX_DIM * 2);
        } else if (ratio == HCA_RATIO) {
            s.ring = HCA_RATIO;
            s.ck = G((size_t) s.ring * HEAD_DIM); s.cs = G((size_t) s.ring * HEAD_DIM);
            s.comp = (uint16_t *) R((size_t) (cfg_.ctx / HCA_RATIO + 1) * HEAD_DIM * 2);
        }
    }
    routed_.assign((size_t) (model_.il1() - model_.il0() + 1) * N_EXPERT_USED, -1);

    // --hip-graph: a device position a card, and the attention scratch sized
    // for the whole context now, so nothing allocates inside a capture.
    graphs_.assign((size_t) n_dev, GraphSlot());
    dpos_val_.assign((size_t) n_dev, -1);
    for (int d = 0; d < n_dev; ++d) {
        Backend & b = model_.dev(d);
        int * p = b.alloc_i32(1);
        owned_.emplace_back(&b, p);
        dpos_.push_back(p);
        ops_[(size_t) d]->reserve(cfg_.ctx);
    }

    // The miss path's staging ring, per card: slots 0/1 alternate over the
    // layers, 2..4 belong to the token-id-routed layers 0-2 (staged at embed).
    if (cfg_.miss_stage) {
        std::vector<size_t> max_slab((size_t) n_dev, 0);
        for (int il = model_.il0(); il <= model_.il1(); ++il) {
            const ExpertTable & t = model_.layer(il).et;
            size_t & m = max_slab[(size_t) dev_of(il)];
            m = std::max(m, t.sz_g + t.sz_u + t.sz_d);
        }
        for (int d = 0; d < n_dev; ++d)
            if (max_slab[(size_t) d]) ops_[(size_t) d]->reserve_moe(max_slab[(size_t) d], 2 + HASH_LAYERS);
    }
}

int Ds4Runner::stage_slot(int il) const {
    if (!cfg_.miss_stage) return -1;
    return (il < HASH_LAYERS) ? 2 + il : (il & 1);
}

Ds4Runner::~Ds4Runner() {
    for (size_t d = 0; d < graphs_.size(); ++d)
        if (graphs_[d].exec) model_.dev((int) d).graph_destroy(graphs_[d].exec);
    for (auto & o : owned_) o.first->free_buf(o.second);
}

void Ds4Runner::graph_report(FILE * out) {
    for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).graph_report(out);
}

// --hip-graph (the qwen4exp runner's mechanism, decode_graph.cpp seg_begin):
// a card's share of the token is captured the first time a position CLASS
// (pos / bucket) meets it and replayed after; its grids are sized for the
// class's last position and the kernels bound themselves by the position they
// read. The graph's last node advances the device position; the host
// re-uploads it only when its shadow disagrees (after an eager token).
bool Ds4Runner::seg_begin(int d) {
    seg_capturing_ = false;
    if (!graph_now_) return true;
    Backend & b = model_.dev(d);
    GraphSlot & g = graphs_[(size_t) d];
    if (dpos_val_[(size_t) d] != pos_) {
        b.upload(dpos_[(size_t) d], &pos_, sizeof(int));
        dpos_val_[(size_t) d] = pos_;
    }
    if (g.exec && g.cls == graph_cls_ && g.epoch == b.graph_epoch()) {
        b.graph_launch(g.exec);
        return false;
    }
    b.graph_capture_begin(dpos_[(size_t) d], graph_bound_);
    seg_capturing_ = true;
    return true;
}

void Ds4Runner::seg_end(int d) {
    if (!graph_now_) return;
    Backend & b = model_.dev(d);
    if (seg_capturing_) {
        GraphSlot & g = graphs_[(size_t) d];
        b.graph_capture_end(&g.exec);
        g.cls = graph_cls_;
        g.epoch = b.graph_epoch();
        b.graph_launch(g.exec);
    }
    dpos_val_[(size_t) d] = pos_ + 1;
    seg_capturing_ = false;
}

void Ds4Runner::report_cache_bytes(FILE * out) const {
    size_t fixed = 0, per_tok_num = 0;   // bytes per token * 128, to stay integral
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        fixed += (size_t) N_SWA * HEAD_DIM * 2;
        const int r = model_.layer(il).ratio;
        if (r == CSA_RATIO) per_tok_num += (size_t) 128 / CSA_RATIO * (HEAD_DIM + IDX_DIM) * 2;
        if (r == HCA_RATIO) per_tok_num += (size_t) 128 / HCA_RATIO * HEAD_DIM * 2;
    }
    std::fprintf(out, "ds4_cache raw_window_bytes=%zu compressed_bytes_per_token=%.3f "
                      "allocated_for_ctx=%.3f GB (f16, layers %d-%d, ctx %d)\n",
                 fixed, per_tok_num / 128.0, ((per_tok_num / 128.0) * cfg_.ctx + fixed) / 1e9,
                 model_.il0(), model_.il1(), cfg_.ctx);
}

// build_hc_pre (DS4): flat RMS norm (no gamma), the 24-row mix, the
// pre/post/comb split, then the pre-weighted sum of the four streams.
void Ds4Runner::hc_pre(Scratch & s, const Mat & fn, const float * base, const float * scale, int il,
                       Recorder & rec, const char * sfx) {
    Backend & b = be(il);
    Ds4Ops & o = op(il);
    b.rms_norm_mul(s.H, nullptr, s.Hn, HC_DIM, 1, HC_DIM, eps_);
    o.gemv(fn, s.Hn, s.mixes);
    rec.tap(b, "hc_mixes", il, s.mixes, HC_MIX, sfx);
    o.hc_split(s.mixes, scale, base, s.pre, s.post, s.comb, hc_eps_, HC_SINKHORN_ITERS);
    rec.tap(b, "hc_pre",  il, s.pre,  HC, sfx);
    rec.tap(b, "hc_post", il, s.post, HC, sfx);
    rec.tap(b, "hc_comb", il, s.comb, HC * HC, sfx);
    o.hc_weighted_sum(s.H, s.pre, s.x);
}

// build_attention_impl (DS4).
void Ds4Runner::attention(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    LayerState & st = st_[(size_t) (il - model_.il0())];
    Scratch & s = S(il);
    Backend & b = be(il);
    Ds4Ops & o = op(il);
    const int p = pos_;
    const RopeParams rp = rope_for(il);
    const RopeParams rc = rope_compress();
    const float kq_scale = 1.0f / sqrtf((float) HEAD_DIM);

    // q: q_a, its norm, q_b, a per-head RMS norm (no gamma), rope on the tail
    o.gemv(L.wq_a, s.xn, s.qr);
    rec.tap(b, "qr", il, s.qr, Q_LORA);
    b.rms_norm_mul(s.qr, L.q_a_norm, s.qr, Q_LORA, 1, Q_LORA, eps_);
    rec.tap(b, "qr_norm", il, s.qr, Q_LORA);
    o.gemv(L.wq_b, s.qr, s.q);
    b.rms_norm_mul(s.q, nullptr, s.q, HEAD_DIM, N_HEAD, HEAD_DIM, eps_);
    rec.tap(b, "q_norm", il, s.q, (size_t) N_HEAD * HEAD_DIM);
    o.rope_tail(s.q, N_HEAD, HEAD_DIM, p, 0, rp, false);
    rec.tap(b, "q", il, s.q, (size_t) N_HEAD * HEAD_DIM);

    // the single K=V head, into the f16 raw window (slot pos % 128)
    o.gemv(L.wkv, s.xn, s.kv);
    b.rms_norm_mul(s.kv, L.kv_norm, s.kv, HEAD_DIM, 1, HEAD_DIM, eps_);
    rec.tap(b, "kv_norm", il, s.kv, HEAD_DIM);
    o.rope_tail(s.kv, 1, HEAD_DIM, p, 0, rp, false);
    rec.tap(b, "kv", il, s.kv, HEAD_DIM);
    o.store_raw(s.kv, st.raw, p);

    // In a captured graph every position-dependent op is ISSUED whatever the
    // position -- the ops themselves decide on the device whether a block
    // completed (L5 step 3b) -- so the one graph fits every position of its
    // class. Eagerly, the host skips what it knows is a no-op.
    const bool all = graph_now_ || cfg_.all_ops;
    if (L.ratio == HCA_RATIO) {
        o.gemv(L.comp_wkv, s.xn, s.ckv);
        rec.tap(b, "hca_state_kv", il, s.ckv, HEAD_DIM);
        o.gemv(L.comp_wgate, s.xn, s.csc);
        rec.tap(b, "hca_state_score", il, s.csc, HEAD_DIM);
        o.add_row(s.csc, L.comp_ape, HEAD_DIM, HCA_RATIO, p);
        rec.tap(b, "hca_state_score_ape", il, s.csc, HEAD_DIM);
        o.ring_put(st.ck, st.ring, HEAD_DIM, p, s.ckv);
        o.ring_put(st.cs, st.ring, HEAD_DIM, p, s.csc);
        if (all || (p + 1) % HCA_RATIO == 0) {      // build_hca_compressed_kv_from_state
            o.comp_pool(st.ck, st.cs, st.ring, HCA_RATIO, HEAD_DIM, p, s.cpool);
            b.rms_norm_mul(s.cpool, L.comp_norm, s.cpool, HEAD_DIM, 1, HEAD_DIM, eps_);
            o.rope_tail(s.cpool, 1, HEAD_DIM, p, HCA_RATIO, rc, false);
            o.store_block(s.cpool, st.comp, HEAD_DIM, HCA_RATIO, p);
        }
        o.attn(s.q, st.raw, p, st.comp, HCA_RATIO, 1 << 30, nullptr, L.attn_sinks, kq_scale, s.att);
        rec.tap(b, "attn_hca", il, s.att, (size_t) N_HEAD * HEAD_DIM);
    } else if (L.ratio == CSA_RATIO) {
        o.gemv(L.comp_wkv, s.xn, s.ckv);
        rec.tap(b, "csa_state_kv", il, s.ckv, 2 * HEAD_DIM);
        o.gemv(L.comp_wgate, s.xn, s.csc);
        rec.tap(b, "csa_state_score", il, s.csc, 2 * HEAD_DIM);
        o.add_row(s.csc, L.comp_ape, 2 * HEAD_DIM, CSA_RATIO, p);
        rec.tap(b, "csa_state_score_ape", il, s.csc, 2 * HEAD_DIM);
        o.ring_put(st.ck, st.ring, 2 * HEAD_DIM, p, s.ckv);
        o.ring_put(st.cs, st.ring, 2 * HEAD_DIM, p, s.csc);
        o.gemv(L.idx_comp_wkv, s.xn, s.lkv);
        rec.tap(b, "lid_state_kv", il, s.lkv, 2 * IDX_DIM);
        o.gemv(L.idx_comp_wgate, s.xn, s.lsc);
        rec.tap(b, "lid_state_score", il, s.lsc, 2 * IDX_DIM);
        o.add_row(s.lsc, L.idx_comp_ape, 2 * IDX_DIM, CSA_RATIO, p);
        rec.tap(b, "lid_state_score_ape", il, s.lsc, 2 * IDX_DIM);
        o.ring_put(st.lk, st.ring, 2 * IDX_DIM, p, s.lkv);
        o.ring_put(st.ls, st.ring, 2 * IDX_DIM, p, s.lsc);
        if (all || (p + 1) % CSA_RATIO == 0) {      // build_overlap_compressed_kv_from_state, twice
            o.comp_pool_overlap(st.ck, st.cs, st.ring, CSA_RATIO, HEAD_DIM, p, s.cpool);
            b.rms_norm_mul(s.cpool, L.comp_norm, s.cpool, HEAD_DIM, 1, HEAD_DIM, eps_);
            o.rope_tail(s.cpool, 1, HEAD_DIM, p, CSA_RATIO, rc, false);
            o.store_block(s.cpool, st.comp, HEAD_DIM, CSA_RATIO, p);
            o.comp_pool_overlap(st.lk, st.ls, st.ring, CSA_RATIO, IDX_DIM, p, s.cpool);
            b.rms_norm_mul(s.cpool, L.idx_comp_norm, s.cpool, IDX_DIM, 1, IDX_DIM, eps_);
            o.rope_tail(s.cpool, 1, IDX_DIM, p, CSA_RATIO, rc, false);
            o.fwht(s.cpool, 1, IDX_DIM);                     // the lid cache's k_rot
            o.store_block(s.cpool, st.lid, IDX_DIM, CSA_RATIO, p);
        }
        // the lightning indexer's query (build_lid_top_k)
        o.gemv(L.idx_q_b, s.qr, s.iq);
        rec.tap(b, "lid_q", il, s.iq, (size_t) IDX_N_HEAD * IDX_DIM);
        o.rope_tail(s.iq, IDX_N_HEAD, IDX_DIM, p, 0, rc, false);
        rec.tap(b, "lid_q_rope", il, s.iq, (size_t) IDX_N_HEAD * IDX_DIM);
        o.fwht(s.iq, IDX_N_HEAD, IDX_DIM);
        rec.tap(b, "lid_q_rot", il, s.iq, (size_t) IDX_N_HEAD * IDX_DIM);
        o.gemv(L.idx_proj, s.xn, s.iw);
        b.scale(s.iw, 1.0f / sqrtf((float) (IDX_DIM * IDX_N_HEAD)), s.iw, IDX_N_HEAD);
        rec.tap(b, "lid_weights", il, s.iw, IDX_N_HEAD);
        const int n_vis = (p + 1) / CSA_RATIO;
        // Every visible block while there are no more than top_k of them:
        // then ggml_top_k's selection is the identity and nothing is ranked.
        const int n_sel = std::min(n_vis, IDX_TOP_K);
        const int * sel = nullptr;
        if (all || n_vis > 0) {
            o.lid_scores(s.iq, s.iw, st.lid, p, s.iscore);
            rec.tap(b, "lid_score_masked", il, s.iscore, (size_t) n_vis);
            if (all || n_vis > IDX_TOP_K) { o.topk(s.iscore, p, IDX_TOP_K, s.isel); sel = s.isel; }
            if (rec.enabled() && n_sel > 0) {
                std::vector<int> v((size_t) n_sel);
                if (sel) b.download(v.data(), sel, (size_t) n_sel * sizeof(int));
                else std::iota(v.begin(), v.end(), 0);
                rec.tap_ints("lid_top_k", il, v);
            }
        }
        o.attn(s.q, st.raw, p, st.comp, CSA_RATIO, IDX_TOP_K, sel, L.attn_sinks, kq_scale, s.att);
        rec.tap(b, "attn_csa_lid", il, s.att, (size_t) N_HEAD * HEAD_DIM);
    } else {
        o.attn(s.q, st.raw, p, nullptr, 0, 0, nullptr, L.attn_sinks, kq_scale, s.att);
        rec.tap(b, "attn_raw", il, s.att, (size_t) N_HEAD * HEAD_DIM);
    }

    // de-rope (ggml_rope_ext_back), the grouped output LoRA, wo_b
    o.rope_tail(s.att, N_HEAD, HEAD_DIM, p, 0, rp, true);
    rec.tap(b, "attn_derope", il, s.att, (size_t) N_HEAD * HEAD_DIM);
    for (int g = 0; g < O_GROUPS; ++g)
        o.gemv(L.wo_a[g], s.att + (size_t) g * O_GROUP_DIM, s.oa + (size_t) g * O_LORA);
    o.gemv(L.wo_b, s.oa, s.ao);
    rec.tap(b, "attn_out", il, s.ao, N_EMBD);
}

// build_moe_ffn (sqrt-softplus, hash or biased top-6, normalised, x1.5,
// swiglu clamped at 10) plus the clamped shared expert.
void Ds4Runner::ffn(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Scratch & s = S(il);
    Backend & b = be(il);
    Ds4Ops & o = op(il);
    o.gemv(L.gate_inp, s.xn, s.rlog);
    rec.tap(b, "ffn_moe_logits", il, s.rlog, N_EXPERT);
    const bool hash = L.tid2eid != nullptr;
    o.router(s.rlog, L.exp_probs_b, hash ? s.hash + (size_t) il * N_EXPERT_USED : nullptr,
             s.probs, s.probs_b, s.ids, s.wraw, s.wnorm, s.wsc);
    rec.tap(b, "ffn_moe_probs", il, s.probs, N_EXPERT);
    if (!hash) rec.tap(b, "ffn_moe_probs_biased", il, s.probs_b, N_EXPERT);
    if (cfg_.log_routing || rec.enabled()) {
        int ids[N_EXPERT_USED];
        b.download(ids, s.ids, sizeof(ids));
        for (int k = 0; k < N_EXPERT_USED; ++k) routed_[(size_t) (il - model_.il0()) * N_EXPERT_USED + k] = ids[k];
    }
    rec.tap(b, "ffn_moe_weights", il, s.wraw, N_EXPERT_USED);
    rec.tap(b, "ffn_moe_weights_norm", il, s.wnorm, N_EXPERT_USED);
    rec.tap(b, "ffn_moe_weights_scaled", il, s.wsc, N_EXPERT_USED);
    o.count_misses(L.et, s.ids);                   // --profile only: bytes over PCIe, on the device

    // The miss path (L5 step 3): the six experts' staging starts NOW, on a
    // side stream, and the shared expert -- which does not depend on it --
    // runs meanwhile. A token-id-routed layer was staged at embed time.
    const int slot = stage_slot(il);
    if (!hash) o.moe_stage(L.et, s.ids, slot);

    o.gemv(L.sh_up, s.xn, s.su);
    rec.tap(b, "ffn_up", il, s.su, N_FF_EXP);
    o.gemv(L.sh_gate, s.xn, s.sg);
    rec.tap(b, "ffn_gate", il, s.sg, N_FF_EXP);
    o.swiglu_clamp(s.sg, s.su, s.sh, N_FF_EXP, SWIGLU_CLAMP);
    rec.tap(b, "ffn_swiglu_limited", il, s.sh, N_FF_EXP);
    o.gemv(L.sh_down, s.sh, s.sd);
    rec.tap(b, "ffn_shexp", il, s.sd, N_EMBD);

    o.moe_gate_up(L.et, s.ids, s.xn, s.yg, s.yu, slot);
    rec.tap(b, "ffn_moe_up",   il, s.yu, (size_t) N_EXPERT_USED * N_FF_EXP);
    rec.tap(b, "ffn_moe_gate", il, s.yg, (size_t) N_EXPERT_USED * N_FF_EXP);
    o.swiglu_clamp(s.yg, s.yu, s.yh, N_EXPERT_USED * N_FF_EXP, SWIGLU_CLAMP);
    rec.tap(b, "ffn_moe_swiglu_limited", il, s.yh, (size_t) N_EXPERT_USED * N_FF_EXP);
    o.moe_down(L.et, s.ids, s.yh, s.yd, slot);
    rec.tap(b, "ffn_moe_down", il, s.yd, (size_t) N_EXPERT_USED * N_EMBD);
    o.moe_accum(s.yd, s.wsc, N_EXPERT_USED, N_EMBD, s.ywt, s.moe);
    rec.tap(b, "ffn_moe_weighted", il, s.ywt, (size_t) N_EXPERT_USED * N_EMBD);
    rec.tap(b, "ffn_moe_out", il, s.moe, N_EMBD);

    b.binary(FK_ADD, s.moe, s.sd, s.fo, N_EMBD);
    rec.tap(b, "ffn_out", il, s.fo, N_EMBD);
}

// One layer (DS4 graph::graph, the loop body).
void Ds4Runner::layer(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Scratch & s = S(il);
    Backend & b = be(il);
    Ds4Ops & o = op(il);

    hc_pre(s, L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, il, rec, "");
    rec.tap(b, "hc_attn_pre", il, s.x, N_EMBD);
    b.rms_norm_mul(s.x, L.attn_norm, s.xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "attn_norm", il, s.xn, N_EMBD);
    attention(il, rec);
    o.hc_post(s.ao, s.H, s.post, s.comb, s.Hn);
    std::swap(s.H, s.Hn);
    rec.tap(b, "hc_attn_post", il, s.H, HC_DIM);

    hc_pre(s, L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, il, rec, ".2");
    rec.tap(b, "hc_ffn_pre", il, s.x, N_EMBD);
    b.rms_norm_mul(s.x, L.ffn_norm, s.xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "ffn_norm", il, s.xn, N_EMBD);
    ffn(il, rec);
    o.hc_post(s.fo, s.H, s.post, s.comb, s.Hn);
    std::swap(s.H, s.Hn);
    rec.tap(b, "l_last", il, s.H, HC_DIM);
}

int Ds4Runner::step(int32_t token, Recorder & rec) {
    const int il0 = model_.il0(), il1 = model_.il1();
    Backend & b0 = be(il0);
    Scratch & s0 = S(il0);
    if (pos_ >= cfg_.ctx) throw std::runtime_error("context full");
    // A graph only for a token nobody inspects: a tap or a routing read-back
    // is a download, which a capture cannot hold.
    graph_now_ = cfg_.hip_graph && !rec.enabled() && !cfg_.log_routing && b0.graph_capable();
    if (graph_now_) {
        const int bucket = std::max(1, cfg_.hip_graph_bucket);
        graph_cls_   = pos_ / bucket;
        graph_bound_ = std::min(cfg_.ctx, (graph_cls_ + 1) * bucket);   // positions < bound
    }
    // host work of the token: the embedding row and the hash layers' ids,
    // uploaded (stream-ordered, pinned staging) BEFORE any card's segment
    std::vector<float> e(N_EMBD);
    model_.embed_row(token, e.data());
    // --profile: a fresh per-token baseline for launches and host syncs on
    // every card (the backend counts from its timer_start)
    if (profile_)
        for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).timer_start();
    b0.boundary_wait_free(0);     // the next card has read last token's H out of us
    b0.upload(s0.x, e.data(), N_EMBD * sizeof(float));
    for (int il = il0; il <= std::min(il1, HASH_LAYERS - 1); ++il)
        be(il).upload(S(il).hash + (size_t) il * N_EXPERT_USED, model_.hash_ids(il, token),
                      N_EXPERT_USED * sizeof(int32_t));

    const bool head = model_.have_head() && il1 == N_LAYER - 1;
    int il = il0;
    while (il <= il1) {
        const int d = dev_of(il);
        int last = il;
        while (last + 1 <= il1 && dev_of(last + 1) == d) ++last;
        if (il > il0) {
            // --profile: the wait for the previous card gets its own class, so
            // boundary_p2p is the copy alone
            if (profile_) op(il).upstream_wait(op(il - 1));
            // the only per-token traffic between cards: H, 64 KB (outside the graphs)
            be(il).boundary_recv(S(il).H, be(il - 1), S(il - 1).H, HC_DIM * sizeof(float), 0);
        }
        if (seg_begin(d)) {
            if (il == il0) {
                // the token-id-routed layers' experts are known: stage them now
                for (int h = il0; h <= std::min(il1, HASH_LAYERS - 1); ++h)
                    op(h).moe_stage(model_.layer(h).et, S(h).hash + (size_t) h * N_EXPERT_USED, stage_slot(h));
                op(il0).hc_init(s0.x, s0.H);                                  // hc_init
                rec.tap(b0, "hc_init", -1, s0.H, HC_DIM);
            }
            for (int l = il; l <= last; ++l) layer(l, rec);
            if (head && last == il1) {
                Backend & b = model_.dev(model_.n_devices() - 1);
                Ds4Ops & o = *ops_.back();
                Scratch & s = scr_.back();
                // build_hc_head: flat norm, 4-row mix, sigmoid+eps, the weighted sum
                b.rms_norm_mul(s.H, nullptr, s.Hn, HC_DIM, 1, HC_DIM, eps_);
                o.gemv(model_.head_fn(), s.Hn, hmix_);
                rec.tap(b, "hc_head_mixes", -1, hmix_, HC);
                o.hc_head_pre(hmix_, model_.head_scale(), model_.head_base(), hpre_, hc_eps_);
                rec.tap(b, "hc_head_pre", -1, hpre_, HC);
                o.hc_weighted_sum(s.H, hpre_, hx_);
                rec.tap(b, "hc_head", -1, hx_, N_EMBD);
                b.rms_norm_mul(hx_, model_.output_norm(), hxn_, N_EMBD, 1, N_EMBD, eps_);
                rec.tap(b, "result_norm", -1, hxn_, N_EMBD);
                o.gemv(model_.output(), hxn_, logits_);
                rec.tap(b, "result_output", -1, logits_, N_VOCAB);
                b.argmax(logits_, N_VOCAB, greedy_);
            }
        }
        seg_end(d);
        // --profile: this card's share is over; what follows is idle (gap_idle)
        if (profile_) op(last).prof_gap();
        il = last + 1;
    }

    int id = -1;
    if (head) model_.dev(model_.n_devices() - 1).download(&id, greedy_, sizeof(int));
    if (profile_)
        for (int d = 0; d < model_.n_devices(); ++d) model_.dev(d).prof_end_token();
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

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int ds4_main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_path, oracle_dir, dump_dir, routing_dir;
    std::vector<int32_t> tokens;
    int il0 = 0, il1 = N_LAYER - 1, n_devices = 3, threads = 4, ctx = 512, greedy_n = 0, time_n = 0;
    bool with_head = true, use_cpu = false, plan_only = false, sync_debug = false, profile = false;
    bool hit_report = false;
    int stage_wgs = 64;
    std::string hist_out;
    int miss_stage = 1, staged_loads = 1, hip_graph = 0, hip_graph_bucket = 1024, all_ops = 0;
    double min_cos = 0.999;
    Placement pl;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--routing"&& i + 1 < argc) routing_dir = argv[++i];
        else if (a == "--dump"   && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--placement" && i + 1 < argc) pl.hist_dir = argv[++i];
        else if (a == "--expert-gb" && i + 1 < argc) pl.expert_gb = std::atof(argv[++i]);
        else if (a == "--cpu")                    use_cpu = true;
        else if (a == "--plan-only")              plan_only = true;   // print placement, run nothing
        else if (a == "--sync-debug")             sync_debug = true;  // drain + check after every launch
        else if (a == "--profile")                profile = true;     // per-class device time, --time tokens
        else if (a == "--hit-report")             hit_report = true;  // per-layer resident hits (reads ids back)
        else if (a == "--stage-wgs" && i + 1 < argc) stage_wgs = std::atoi(argv[++i]);
        else if (a == "--hist-out" && i + 1 < argc) hist_out = argv[++i];   // M2-format layer_<il>.csv of this run's routing
        else if (a == "--miss-stage" && i + 1 < argc) miss_stage = std::atoi(argv[++i]);
        else if (a == "--staged-loads" && i + 1 < argc) staged_loads = std::atoi(argv[++i]);
        else if (a == "--hip-graph" && i + 1 < argc) hip_graph = std::atoi(argv[++i]);
        else if (a == "--hip-graph-bucket" && i + 1 < argc) hip_graph_bucket = std::atoi(argv[++i]);
        else if (a == "--all-ops" && i + 1 < argc) all_ops = std::atoi(argv[++i]);
        else if (a == "--no-head")                with_head = false;
        else if (a == "--devices"&& i + 1 < argc) n_devices = std::atoi(argv[++i]);
        else if (a == "--threads"&& i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--greedy" && i + 1 < argc) greedy_n = std::atoi(argv[++i]);
        else if (a == "--time"   && i + 1 < argc) time_n = std::atoi(argv[++i]);
        else if (a == "--min-cos"&& i + 1 < argc) min_cos = std::atof(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) {
            if (!parse_range(argv[++i], il0, il1)) { std::fprintf(stderr, "bad --layers\n"); return 2; }
        } else if (a == "--tokens") {
            while (i + 1 < argc && argv[i + 1][0] != '-') tokens.push_back((int32_t) std::atoll(argv[++i]));
        } else if (a == "--chunk" && i + 1 < argc) { ++i; /* T == 1 only: ignored */ }
        else { std::fprintf(stderr, "ds4: unrecognised argument: %s\n", a.c_str()); return 2; }
    }
#ifdef FRANKEN_NO_HIP
    use_cpu = true;
#endif
    if (model_path.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: franken_decode --model <deepseek4 shard> --tokens <ids> [--cpu]\n"
                             "  [--layers A-B] [--no-head] [--devices N] [--ctx N] [--threads N]\n"
                             "  [--placement DIR --expert-gb X]   (GPU: M2 histogram, VRAM for experts a card)\n"
                             "  [--oracle DIR] [--routing DIR] [--dump DIR] [--min-cos X]\n"
                             "  [--greedy N] [--time N] [--plan-only] [--sync-debug]\n"
                             "  [--profile] [--hit-report] [--hist-out DIR] [--stage-wgs N] [--miss-stage 0|1] [--staged-loads 0|1] [--hip-graph 0|1 [--hip-graph-bucket N]] [--all-ops 0|1]\n");
        return 2;
    }
    if ((int) tokens.size() + greedy_n + time_n > ctx) { std::fprintf(stderr, "--ctx too small\n"); return 2; }
    if (n_devices < 1) n_devices = 1;

    try {
        std::vector<std::unique_ptr<Backend>> owned_be;
        std::vector<Backend *> devs;
        std::vector<std::unique_ptr<Ds4Ops>> owned_ops;
        std::vector<Ds4Ops *> ops;
        if (use_cpu) {
            owned_be.emplace_back(make_cpu_backend(threads));
            for (int d = 0; d < n_devices; ++d) devs.push_back(owned_be[0].get());
        } else {
            enable_peer_access(n_devices);          // before anything is placed
            for (int d = 0; d < n_devices; ++d) {
                owned_be.emplace_back(make_gpu_backend(d));
                devs.push_back(owned_be.back().get());
            }
        }
        for (Backend * b : devs) {
            if (sync_debug) b->set_sync_debug(true);
            if (profile && !use_cpu) b->set_profile(true);
            owned_ops.emplace_back(use_cpu ? make_ds4_cpu_ops(*b) : make_ds4_gpu_ops(*b));
            ops.push_back(owned_ops.back().get());
        }
        const int n_report = use_cpu ? 1 : n_devices;
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "before placement");
        const bool head = with_head && il1 == N_LAYER - 1;
        Ds4Model model(model_path, devs, ops, il0, il1, head, pl);
        std::printf("arch=deepseek4 backend=%s devices=%d layers=%d-%d head=%d ctx=%d tokens=%zu "
                    "placed=%.2f GB host_experts=%.2f GB rms_eps=%g hc_eps=%g\n",
                    devs[0]->name(), n_devices, il0, il1, (int) head, ctx, tokens.size(),
                    model.placed_bytes() / 1e9, model.host_expert_bytes() / 1e9,
                    model.rms_eps(), model.hc_eps());
        if (plan_only) { std::printf("plan-only: stopping before the caches\n"); return 0; }
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "after placement");
        bool placement_ok = true;
        for (int d = 0; d < n_report; ++d) placement_ok &= devs[d]->verify_placement(stdout);
        if (!placement_ok) { std::printf("DS4 FAIL (placement)\n"); return 1; }

        for (Ds4Ops * o : ops) { o->set_profile(profile && !use_cpu); o->set_staged_loads(staged_loads); o->set_stage_wgs(stage_wgs); }
        Ds4Config cfg; cfg.ctx = ctx; cfg.log_routing = !routing_dir.empty() || hit_report || !hist_out.empty();
        cfg.miss_stage = miss_stage;
        // --sync-debug synchronises after every launch, which a capture cannot hold
        cfg.hip_graph = (use_cpu || sync_debug) ? 0 : hip_graph; cfg.hip_graph_bucket = hip_graph_bucket;
        cfg.all_ops = all_ops;
        std::printf("miss_stage=%d staged_loads=%d profile=%d hip_graph=%d hip_graph_bucket=%d%s\n",
                    miss_stage, staged_loads, (int) profile, cfg.hip_graph, hip_graph_bucket,
                    (use_cpu && hip_graph) ? " (ignored: the CPU backend has no graphs)"
                    : (sync_debug && hip_graph) ? " (off: --sync-debug)" : "");
        Ds4Runner run(model, ops, cfg);
        run.report_cache_bytes(stdout);
        // --profile: every token closes its profile interval from the first one
        // on, so the event pool never fills; the counts are reset before the
        // --time tokens, which are the ones reported.
        if (profile && !use_cpu) run.set_profile(true);
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "after caches+scratch");

        std::map<std::pair<int, int>, std::vector<int>> ref_routes;
        const bool have_routes = !routing_dir.empty() && load_routing(routing_dir, ref_routes);
        if (!routing_dir.empty() && !have_routes) std::printf("routing PENDING: no %s/moe_ids.txt\n", routing_dir.c_str());
        std::map<int, std::pair<double, int>> overlap;   // il -> (sum overlap, n)
        int exact_sets = 0, n_sets = 0;
        // --hit-report: per layer, the share of the routed experts that the
        // placement plan made resident, against the histogram's own coverage
        // of the same resident set (model.m2_coverage) -- the check that the
        // plan ranks the right experts.
        std::vector<long long> hits((size_t) N_LAYER, 0), picks((size_t) N_LAYER, 0);
        std::vector<std::vector<long long>> hist((size_t) N_LAYER, std::vector<long long>(N_EXPERT, 0));
        auto count_hits = [&]() {
            if (!hist_out.empty())
                for (int il = il0; il <= il1; ++il)
                    for (int k = 0; k < N_EXPERT_USED; ++k) {
                        const int e = run.routed_ids()[(size_t) (il - il0) * N_EXPERT_USED + k];
                        if (e >= 0 && e < N_EXPERT) ++hist[(size_t) il][(size_t) e];
                    }
            if (!hit_report || !model.planned()) return;
            for (int il = il0; il <= il1; ++il)
                for (int k = 0; k < N_EXPERT_USED; ++k) {
                    const int e = run.routed_ids()[(size_t) (il - il0) * N_EXPERT_USED + k];
                    if (e < 0) continue;
                    hits[(size_t) il] += model.resident(il, e); ++picks[(size_t) il];
                }
        };
        auto observe = [&](int pos) {
            count_hits();
            if (!have_routes) return;
            for (int il = il0; il <= il1; ++il) {
                auto it = ref_routes.find({il, pos});
                if (it == ref_routes.end()) continue;
                std::set<int> a(run.routed_ids().begin() + (size_t) (il - il0) * N_EXPERT_USED,
                                run.routed_ids().begin() + (size_t) (il - il0 + 1) * N_EXPERT_USED);
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
            rec.enable(t + 1 == tokens.size());
            last = run.step(tokens[t], rec);
            observe((int) t);
            if (t == 0) std::printf("token 0 done\n");
        }
        const std::vector<int> last_ids = run.routed_ids();
        rec.enable(false);
        std::vector<int> greedy;
        if (greedy_n > 0 && last >= 0) {
            greedy.push_back(last);
            for (int g = 1; g < greedy_n; ++g) {
                last = run.step(last, rec);
                observe(run.pos() - 1);
                greedy.push_back(last);
            }
            std::printf("greedy_ids:");
            for (int v : greedy) std::printf(" %d", v);
            std::printf("\n");
            // a tap, so a --dump carries it and an --oracle compares it EXACTLY
            // (decode_oracle.cpp's greedy_ids rule)
            rec.enable(true);
            rec.tap_host("greedy_ids", -1, std::vector<float>(greedy.begin(), greedy.end()));
            rec.enable(false);
        }
        if (time_n > 0 && last >= 0) {
            // decode timing: one token a step, the id read back each token
            // (it feeds the next), routing capture off
            run.set_log_routing(hit_report || !hist_out.empty());   // off unless the ids are wanted
            const bool prof = profile && !use_cpu;
            if (prof) {
                // the warm tokens (prompt, greedy) are not in the profile
                for (int d = 0; d < n_report; ++d) { devs[d]->sync(); devs[d]->prof_reset(); ops[d]->reset_miss_count(); }
            }
            std::vector<double> ms;
            for (int i = 0; i < time_n; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                last = run.step(last, rec);
                if (hit_report || !hist_out.empty()) count_hits();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            const double med = median(ms);
            std::printf("ds4_decode_ms_median=%.3f tok_s=%.2f tokens=%d depth_end=%d ctx=%d hip_graph=%d\n",
                        med, med > 0 ? 1000.0 / med : 0.0, time_n, run.pos(), ctx, cfg.hip_graph);
            if (cfg.hip_graph) run.graph_report(stdout);
            if (prof) {
                // Per card: device time by class (prof_*_us), the card's total,
                // launches and host syncs a token; then the miss bytes counted
                // on the device. Profiled tokens pay one event sync each, so
                // the median above is NOT the unprofiled speed. prof_busy_us
                // is the card's own work: total minus gap_idle (after its
                // share) -- and ds4_upstream_wait is the wait before it.
                run.set_profile(false);
                double miss_all = 0.0;
                for (int d = 0; d < n_report; ++d) {
                    std::printf("--- device %d (per token, %d tokens) ---\n", d, time_n);
                    devs[d]->prof_report(stdout, time_n);
                    const double mb = ops[d]->miss_bytes_total() / time_n / 1e6;
                    miss_all += mb;
                    std::printf("ds4_miss_mb_per_token_dev%d=%.2f\n", d, mb);
                }
                std::printf("ds4_miss_mb_per_token=%.2f (counted on the device from the routed ids)\n", miss_all);
            }
        }

        if (!hist_out.empty()) {
            // the M2 format (moe_hist.cpp write_layer_csv): expert_id,count, hottest first
            ::mkdir(hist_out.c_str(), 0755);
            for (int il = il0; il <= il1; ++il) {
                std::vector<int> order(N_EXPERT);
                std::iota(order.begin(), order.end(), 0);
                const auto & c = hist[(size_t) il];
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return c[(size_t) a] > c[(size_t) b]; });
                std::ofstream f(hist_out + "/layer_" + std::to_string(il) + ".csv");
                f << "expert_id,count\n";
                for (int e : order) f << e << "," << c[(size_t) e] << "\n";
            }
            std::printf("hist-out: wrote %s/layer_<il>.csv for layers %d-%d (%d positions)\n",
                        hist_out.c_str(), il0, il1, run.pos());
        }
        if (hit_report && model.planned()) {
            long long h = 0, n = 0;
            for (int il = il0; il <= il1; ++il) {
                if (!picks[(size_t) il]) continue;
                std::printf("hit layer=%d resident=%d measured_hit=%.3f m2_coverage=%.3f picks=%lld\n", il,
                            model.n_resident(il), (double) hits[(size_t) il] / picks[(size_t) il],
                            model.m2_coverage(il), picks[(size_t) il]);
                h += hits[(size_t) il]; n += picks[(size_t) il];
            }
            if (n) std::printf("hit all measured_hit=%.3f picks=%lld\n", (double) h / n, n);
        } else if (hit_report) {
            std::printf("hit-report: no --placement given, nothing planned\n");
        }
        bool ok = true;
        // SLOT ORDER (L5 step 1): align the per-expert-slot taps to the
        // reference's slot order only when the two routers picked the same
        // SET at the last prompt position; unequal sets are left to fail.
        if (have_routes && !oracle_dir.empty()) {
            const int last_pos = (int) tokens.size() - 1;
            for (int il = il0; il <= il1; ++il) {
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
                std::printf("DS4 %s ORACLE %s (min_cos=%g)\n", use_cpu ? "CPU" : "GPU",
                            (cmp_ok && req_ok) ? "PASS" : "FAIL", min_cos);
            }
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ds4 error: %s\n", e.what());
        return 1;
    }
}

} // namespace ds4
} // namespace fk
