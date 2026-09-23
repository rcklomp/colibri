// tools/hot-expert/franken/decode/ds4_graph.cpp -- see ds4_graph.h and
// DEEPSEEK4.md. Every block below names the lines of
// ~/src/llama-glm53/src/models/deepseek4.cpp (DS4:) it ports.

#include "ds4_graph.h"

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

namespace fk {
namespace ds4 {

// ------------------------------------------------------------------ rope --
//
// build_attention_impl: a compress-rope layer (ratio != 0) uses the yarn set
// -- base 160000, freq_scale 1/16, ext_factor 1 (llama_context: yarn ->
// 1.0), n_ctx_orig 65536, beta 32/1, attn_factor dsv4_rope_attn_factor() --
// and a ratio-0 layer base 10000 with no scaling at all. The lid query and
// every compressed row always take the compress set (DS4 build_lid_top_k,
// build_*_compressed_kv_from_state).
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
    r.beta_fast   = 0.0f;
    r.beta_slow   = 0.0f;
    r.n_ctx_orig  = 0;
    return r;
}

// ---------------------------------------------------------------- runner --

Ds4Runner::Ds4Runner(Ds4Model & model, Ds4Ops & ops, const Ds4Config & cfg)
    : model_(model), ops_(ops), cfg_(cfg), eps_(model.rms_eps()), hc_eps_(model.hc_eps()) {
    hb_ = &model_.dev(0);
    if (hb_->is_gpu()) throw std::runtime_error("the DeepSeek-V4 graph has no GPU ops yet (L5 step 2); use --cpu");
    auto F = [&](size_t n) { float * p = hb_->alloc_f32(n); owned_.push_back(p); return p; };
    H = F(HC_DIM); Hn = F(HC_DIM); mixes = F(HC_MIX + HC); pre = F(HC); post = F(HC); comb = F(HC * HC);
    x = F(N_EMBD); xn = F(N_EMBD); qr = F(Q_LORA); q = F((size_t) N_HEAD * HEAD_DIM); kv = F(HEAD_DIM);
    att = F((size_t) N_HEAD * HEAD_DIM); oa = F((size_t) O_GROUPS * O_LORA); ao = F(N_EMBD);
    ckv = F(2 * HEAD_DIM); csc = F(2 * HEAD_DIM); lkv = F(2 * IDX_DIM); lsc = F(2 * IDX_DIM); cpool = F(HEAD_DIM);
    iq = F((size_t) IDX_N_HEAD * IDX_DIM); iw = F(IDX_N_HEAD);
    iscore = F((size_t) (cfg_.ctx / CSA_RATIO + 1));
    rlog = F(N_EXPERT); probs = F(N_EXPERT); probs_b = F(N_EXPERT);
    wraw = F(N_EXPERT_USED); wnorm = F(N_EXPERT_USED); wsc = F(N_EXPERT_USED);
    yg = F((size_t) N_EXPERT_USED * N_FF_EXP); yu = F((size_t) N_EXPERT_USED * N_FF_EXP);
    yh = F((size_t) N_EXPERT_USED * N_FF_EXP); yd = F((size_t) N_EXPERT_USED * N_EMBD);
    ywt = F((size_t) N_EXPERT_USED * N_EMBD); moe = F(N_EMBD);
    sg = F(N_FF_EXP); su = F(N_FF_EXP); sh = F(N_FF_EXP); sd = F(N_EMBD); fo = F(N_EMBD);
    hmix = F(HC); hpre = F(HC); hx = F(N_EMBD); hxn = F(N_EMBD);
    if (model_.have_head()) logits = F(N_VOCAB);
    isel = hb_->alloc_i32((size_t) std::max(cfg_.ctx / CSA_RATIO + 1, N_EXPERT));
    owned_.push_back(isel);

    // Positional state per layer, sized ONCE from ctx (DEEPSEEK4.md section 1).
    st_.resize((size_t) (model_.il1() - model_.il0() + 1));
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        LayerState & s = st_[(size_t) (il - model_.il0())];
        Backend & b = be(il);
        auto R = [&](size_t bytes) { void * p = b.alloc_raw(bytes); owned_.push_back(p); return p; };
        auto G = [&](size_t n) { float * p = b.alloc_f32(n); owned_.push_back(p); return p; };
        s.raw = (uint16_t *) R((size_t) N_SWA * HEAD_DIM * 2);
        const int ratio = model_.layer(il).ratio;
        if (ratio == CSA_RATIO) {
            s.ring = 2 * CSA_RATIO;
            s.ck = G((size_t) s.ring * 2 * HEAD_DIM); s.cs = G((size_t) s.ring * 2 * HEAD_DIM);
            s.lk = G((size_t) s.ring * 2 * IDX_DIM);  s.ls = G((size_t) s.ring * 2 * IDX_DIM);
            s.comp = (uint16_t *) R((size_t) (cfg_.ctx / CSA_RATIO + 1) * HEAD_DIM * 2);
            s.lid  = (uint16_t *) R((size_t) (cfg_.ctx / CSA_RATIO + 1) * IDX_DIM * 2);
        } else if (ratio == HCA_RATIO) {
            s.ring = HCA_RATIO;
            s.ck = G((size_t) s.ring * HEAD_DIM); s.cs = G((size_t) s.ring * HEAD_DIM);
            s.comp = (uint16_t *) R((size_t) (cfg_.ctx / HCA_RATIO + 1) * HEAD_DIM * 2);
        }
    }
    routed_.assign((size_t) (model_.il1() - model_.il0() + 1) * N_EXPERT_USED, -1);
}

Ds4Runner::~Ds4Runner() {
    for (void * p : owned_) hb_->free_buf(p);
}

void Ds4Runner::report_cache_bytes(FILE * out) const {
    size_t fixed = 0, per_tok_num = 0;   // per_tok in bytes * 128 (to stay integral)
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        fixed += (size_t) N_SWA * HEAD_DIM * 2;
        const int r = model_.layer(il).ratio;
        if (r == CSA_RATIO) per_tok_num += (size_t) 128 / CSA_RATIO * (HEAD_DIM + IDX_DIM) * 2;
        if (r == HCA_RATIO) per_tok_num += (size_t) 128 / HCA_RATIO * HEAD_DIM * 2;
    }
    std::fprintf(out, "ds4_cache raw_window_bytes=%zu compressed_bytes_per_token=%.3f "
                      "at_256k=%.3f GB (f16, layers %d-%d)\n",
                 fixed, per_tok_num / 128.0, (per_tok_num / 128.0) * 262144.0 / 1e9 + fixed / 1e9,
                 model_.il0(), model_.il1());
}

// build_hc_pre (DS4 ~352-407): flat RMS norm (no gamma), the 24-row mix,
// the pre/post/comb split, then the pre-weighted sum of the four streams.
void Ds4Runner::hc_pre(const Mat & fn, const float * base, const float * scale, int il,
                       Recorder & rec, const char * sfx) {
    Backend & b = be(il);
    b.rms_norm_mul(H, nullptr, Hn, HC_DIM, 1, HC_DIM, eps_);
    b.gemv(fn, Hn, mixes);
    rec.tap(b, "hc_mixes", il, mixes, HC_MIX, sfx);
    ops_.hc_split(mixes, scale, base, pre, post, comb, hc_eps_, HC_SINKHORN_ITERS);
    rec.tap(b, "hc_pre",  il, pre,  HC, sfx);
    rec.tap(b, "hc_post", il, post, HC, sfx);
    rec.tap(b, "hc_comb", il, comb, HC * HC, sfx);
    ops_.hc_weighted_sum(H, pre, x);
}

// build_attention_impl (DS4 ~870-1215).
void Ds4Runner::attention(int il, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    LayerState & s = st_[(size_t) (il - model_.il0())];
    Backend & b = be(il);
    const int p = pos_;
    const RopeParams rp = rope_for(il);
    const RopeParams rc = rope_compress();
    const float kq_scale = 1.0f / sqrtf((float) HEAD_DIM);

    // q: q_a, its norm, q_b, a per-head RMS norm (no gamma), rope on the tail
    b.gemv(L.wq_a, xn, qr);
    rec.tap(b, "qr", il, qr, Q_LORA);
    b.rms_norm_mul(qr, L.q_a_norm, qr, Q_LORA, 1, Q_LORA, eps_);
    rec.tap(b, "qr_norm", il, qr, Q_LORA);
    b.gemv(L.wq_b, qr, q);
    b.rms_norm_mul(q, nullptr, q, HEAD_DIM, N_HEAD, HEAD_DIM, eps_);
    rec.tap(b, "q_norm", il, q, (size_t) N_HEAD * HEAD_DIM);
    ops_.rope_tail(q, N_HEAD, HEAD_DIM, p, rp, false);
    rec.tap(b, "q", il, q, (size_t) N_HEAD * HEAD_DIM);

    // the single K=V head
    b.gemv(L.wkv, xn, kv);
    b.rms_norm_mul(kv, L.kv_norm, kv, HEAD_DIM, 1, HEAD_DIM, eps_);
    rec.tap(b, "kv_norm", il, kv, HEAD_DIM);
    ops_.rope_tail(kv, 1, HEAD_DIM, p, rp, false);
    rec.tap(b, "kv", il, kv, HEAD_DIM);
    // raw sliding-window cache (llama_kv_cache_iswa, f16): slot pos % 128
    ops_.to_f16(kv, s.raw + (size_t) (p % N_SWA) * HEAD_DIM, HEAD_DIM);
    const int raw0 = std::max(0, p - N_SWA + 1);
    const int n_raw = p - raw0 + 1;

    if (L.ratio == HCA_RATIO) {
        // hca_state_kv / hca_state_score (+ ape[pos % 128]) into the ring
        b.gemv(L.comp_wkv, xn, ckv);
        rec.tap(b, "hca_state_kv", il, ckv, HEAD_DIM);
        b.gemv(L.comp_wgate, xn, csc);
        rec.tap(b, "hca_state_score", il, csc, HEAD_DIM);
        b.binary(FK_ADD, csc, L.comp_ape + (size_t) (p % HCA_RATIO) * HEAD_DIM, csc, HEAD_DIM);
        rec.tap(b, "hca_state_score_ape", il, csc, HEAD_DIM);
        const int slot = p % s.ring;
        b.copy(s.ck + (size_t) slot * HEAD_DIM, ckv, HEAD_DIM);
        b.copy(s.cs + (size_t) slot * HEAD_DIM, csc, HEAD_DIM);
        // a block completes at (p+1) % 128 == 0: pool, norm, rope at its first
        // position, store f16 (build_hca_compressed_kv_from_state)
        if ((p + 1) % HCA_RATIO == 0) {
            const int blk = p / HCA_RATIO;
            ops_.comp_pool(s.ck, s.cs, s.ring, HCA_RATIO, HEAD_DIM, blk, cpool);
            b.rms_norm_mul(cpool, L.comp_norm, cpool, HEAD_DIM, 1, HEAD_DIM, eps_);
            ops_.rope_tail(cpool, 1, HEAD_DIM, blk * HCA_RATIO, rc, false);
            ops_.to_f16(cpool, s.comp + (size_t) blk * HEAD_DIM, HEAD_DIM);
        }
        const int n_vis = (p + 1) / HCA_RATIO;
        ops_.attn(q, s.raw, raw0, n_raw, s.comp, nullptr, n_vis, L.attn_sinks, kq_scale, att);
        rec.tap(b, "attn_hca", il, att, (size_t) N_HEAD * HEAD_DIM);
    } else if (L.ratio == CSA_RATIO) {
        // csa_state_* (2x512 wide, + ape[pos % 4]) into the 8-row ring
        b.gemv(L.comp_wkv, xn, ckv);
        rec.tap(b, "csa_state_kv", il, ckv, 2 * HEAD_DIM);
        b.gemv(L.comp_wgate, xn, csc);
        rec.tap(b, "csa_state_score", il, csc, 2 * HEAD_DIM);
        b.binary(FK_ADD, csc, L.comp_ape + (size_t) (p % CSA_RATIO) * 2 * HEAD_DIM, csc, 2 * HEAD_DIM);
        rec.tap(b, "csa_state_score_ape", il, csc, 2 * HEAD_DIM);
        const int slot = p % s.ring;
        b.copy(s.ck + (size_t) slot * 2 * HEAD_DIM, ckv, 2 * HEAD_DIM);
        b.copy(s.cs + (size_t) slot * 2 * HEAD_DIM, csc, 2 * HEAD_DIM);
        // lid_state_* (2x128 wide, + ape[pos % 4]) into its own ring
        b.gemv(L.idx_comp_wkv, xn, lkv);
        rec.tap(b, "lid_state_kv", il, lkv, 2 * IDX_DIM);
        b.gemv(L.idx_comp_wgate, xn, lsc);
        rec.tap(b, "lid_state_score", il, lsc, 2 * IDX_DIM);
        b.binary(FK_ADD, lsc, L.idx_comp_ape + (size_t) (p % CSA_RATIO) * 2 * IDX_DIM, lsc, 2 * IDX_DIM);
        rec.tap(b, "lid_state_score_ape", il, lsc, 2 * IDX_DIM);
        b.copy(s.lk + (size_t) slot * 2 * IDX_DIM, lkv, 2 * IDX_DIM);
        b.copy(s.ls + (size_t) slot * 2 * IDX_DIM, lsc, 2 * IDX_DIM);
        if ((p + 1) % CSA_RATIO == 0) {
            const int blk = p / CSA_RATIO;
            // build_overlap_compressed_kv_from_state for the CSA keys ...
            ops_.comp_pool_overlap(s.ck, s.cs, s.ring, CSA_RATIO, HEAD_DIM, blk, cpool);
            b.rms_norm_mul(cpool, L.comp_norm, cpool, HEAD_DIM, 1, HEAD_DIM, eps_);
            ops_.rope_tail(cpool, 1, HEAD_DIM, blk * CSA_RATIO, rc, false);
            ops_.to_f16(cpool, s.comp + (size_t) blk * HEAD_DIM, HEAD_DIM);
            // ... and for the indexer keys, which the lid cache rotates (k_rot)
            ops_.comp_pool_overlap(s.lk, s.ls, s.ring, CSA_RATIO, IDX_DIM, blk, cpool);
            b.rms_norm_mul(cpool, L.idx_comp_norm, cpool, IDX_DIM, 1, IDX_DIM, eps_);
            ops_.rope_tail(cpool, 1, IDX_DIM, blk * CSA_RATIO, rc, false);
            ops_.fwht(cpool, 1, IDX_DIM);
            ops_.to_f16(cpool, s.lid + (size_t) blk * IDX_DIM, IDX_DIM);
        }
        // the lightning indexer's query (build_lid_top_k)
        b.gemv(L.idx_q_b, qr, iq);
        rec.tap(b, "lid_q", il, iq, (size_t) IDX_N_HEAD * IDX_DIM);
        ops_.rope_tail(iq, IDX_N_HEAD, IDX_DIM, p, rc, false);
        rec.tap(b, "lid_q_rope", il, iq, (size_t) IDX_N_HEAD * IDX_DIM);
        ops_.fwht(iq, IDX_N_HEAD, IDX_DIM);
        rec.tap(b, "lid_q_rot", il, iq, (size_t) IDX_N_HEAD * IDX_DIM);
        b.gemv(L.idx_proj, xn, iw);
        b.scale(iw, 1.0f / sqrtf((float) (IDX_DIM * IDX_N_HEAD)), iw, IDX_N_HEAD);
        rec.tap(b, "lid_weights", il, iw, IDX_N_HEAD);
        const int n_vis = (p + 1) / CSA_RATIO;
        int n_sel = 0;
        if (n_vis > 0) {
            ops_.lid_scores(iq, iw, s.lid, n_vis, iscore);
            rec.tap(b, "lid_score_masked", il, iscore, (size_t) n_vis);
            n_sel = ops_.topk(iscore, n_vis, IDX_TOP_K, isel);
            if (rec.enabled()) {
                std::vector<int> sel((size_t) n_sel);
                b.download(sel.data(), isel, (size_t) n_sel * sizeof(int));
                rec.tap_ints("lid_top_k", il, sel);
            }
        }
        ops_.attn(q, s.raw, raw0, n_raw, s.comp, isel, n_sel, L.attn_sinks, kq_scale, att);
        rec.tap(b, "attn_csa_lid", il, att, (size_t) N_HEAD * HEAD_DIM);
    } else {
        ops_.attn(q, s.raw, raw0, n_raw, nullptr, nullptr, 0, L.attn_sinks, kq_scale, att);
        rec.tap(b, "attn_raw", il, att, (size_t) N_HEAD * HEAD_DIM);
    }

    // de-rope (ggml_rope_ext_back), the grouped output LoRA, wo_b
    ops_.rope_tail(att, N_HEAD, HEAD_DIM, p, rp, true);
    rec.tap(b, "attn_derope", il, att, (size_t) N_HEAD * HEAD_DIM);
    for (int g = 0; g < O_GROUPS; ++g)
        b.gemv(L.wo_a[g], att + (size_t) g * O_GROUP_DIM, oa + (size_t) g * O_LORA);
    b.gemv(L.wo_b, oa, ao);
    rec.tap(b, "attn_out", il, ao, N_EMBD);
}

// DS4 ~1270-1305: build_moe_ffn (sqrt-softplus, hash or biased top-6,
// normalised, x1.5, swiglu clamped at 10) plus the clamped shared expert.
void Ds4Runner::ffn(int il, int32_t token, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Backend & b = be(il);
    b.gemv(L.gate_inp, xn, rlog);
    rec.tap(b, "ffn_moe_logits", il, rlog, N_EXPERT);
    const int32_t * hash = model_.hash_ids(il, token);
    ops_.router(rlog, L.exp_probs_b, hash, probs, probs_b, isel, wraw, wnorm, wsc);
    rec.tap(b, "ffn_moe_probs", il, probs, N_EXPERT);
    if (!hash) rec.tap(b, "ffn_moe_probs_biased", il, probs_b, N_EXPERT);
    int ids[N_EXPERT_USED];
    b.download(ids, isel, sizeof(ids));
    for (int k = 0; k < N_EXPERT_USED; ++k) routed_[(size_t) (il - model_.il0()) * N_EXPERT_USED + k] = ids[k];
    rec.tap(b, "ffn_moe_weights", il, wraw, N_EXPERT_USED);
    rec.tap(b, "ffn_moe_weights_norm", il, wnorm, N_EXPERT_USED);
    rec.tap(b, "ffn_moe_weights_scaled", il, wsc, N_EXPERT_USED);

    for (int k = 0; k < N_EXPERT_USED; ++k) {
        b.gemv_expert(L.exp_up,   ids[k], xn, yu + (size_t) k * N_FF_EXP);
        b.gemv_expert(L.exp_gate, ids[k], xn, yg + (size_t) k * N_FF_EXP);
    }
    rec.tap(b, "ffn_moe_up",   il, yu, (size_t) N_EXPERT_USED * N_FF_EXP);
    rec.tap(b, "ffn_moe_gate", il, yg, (size_t) N_EXPERT_USED * N_FF_EXP);
    ops_.swiglu_clamp(yg, yu, yh, N_EXPERT_USED * N_FF_EXP, SWIGLU_CLAMP);
    rec.tap(b, "ffn_moe_swiglu_limited", il, yh, (size_t) N_EXPERT_USED * N_FF_EXP);
    for (int k = 0; k < N_EXPERT_USED; ++k)
        b.gemv_expert(L.exp_down, ids[k], yh + (size_t) k * N_FF_EXP, yd + (size_t) k * N_EMBD);
    rec.tap(b, "ffn_moe_down", il, yd, (size_t) N_EXPERT_USED * N_EMBD);
    ops_.moe_accum(yd, wsc, N_EXPERT_USED, N_EMBD, ywt, moe);
    rec.tap(b, "ffn_moe_weighted", il, ywt, (size_t) N_EXPERT_USED * N_EMBD);
    rec.tap(b, "ffn_moe_out", il, moe, N_EMBD);

    b.gemv(L.sh_up, xn, su);
    rec.tap(b, "ffn_up", il, su, N_FF_EXP);
    b.gemv(L.sh_gate, xn, sg);
    rec.tap(b, "ffn_gate", il, sg, N_FF_EXP);
    ops_.swiglu_clamp(sg, su, sh, N_FF_EXP, SWIGLU_CLAMP);
    rec.tap(b, "ffn_swiglu_limited", il, sh, N_FF_EXP);
    b.gemv(L.sh_down, sh, sd);
    rec.tap(b, "ffn_shexp", il, sd, N_EMBD);
    b.binary(FK_ADD, moe, sd, fo, N_EMBD);
    rec.tap(b, "ffn_out", il, fo, N_EMBD);
}

// One layer (DS4 graph::graph, the loop body ~1240-1310).
void Ds4Runner::layer(int il, int32_t token, Recorder & rec) {
    const LayerWeights & L = model_.layer(il);
    Backend & b = be(il);

    hc_pre(L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, il, rec, "");
    rec.tap(b, "hc_attn_pre", il, x, N_EMBD);
    b.rms_norm_mul(x, L.attn_norm, xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "attn_norm", il, xn, N_EMBD);
    attention(il, rec);
    ops_.hc_post(ao, H, post, comb, Hn);
    std::swap(H, Hn);
    rec.tap(b, "hc_attn_post", il, H, HC_DIM);

    hc_pre(L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, il, rec, ".2");
    rec.tap(b, "hc_ffn_pre", il, x, N_EMBD);
    b.rms_norm_mul(x, L.ffn_norm, xn, N_EMBD, 1, N_EMBD, eps_);
    rec.tap(b, "ffn_norm", il, xn, N_EMBD);
    ffn(il, token, rec);
    ops_.hc_post(fo, H, post, comb, Hn);
    std::swap(H, Hn);
    rec.tap(b, "l_last", il, H, HC_DIM);
}

int Ds4Runner::step(int32_t token, Recorder & rec) {
    Backend & b0 = be(model_.il0());
    if (pos_ >= cfg_.ctx) throw std::runtime_error("context full");
    std::vector<float> e(N_EMBD);
    model_.embed_row(token, e.data());
    b0.upload(x, e.data(), N_EMBD * sizeof(float));
    for (int h = 0; h < HC; ++h) b0.copy(H + (size_t) h * N_EMBD, x, N_EMBD);   // hc_init: repeat
    rec.tap(b0, "hc_init", -1, H, HC_DIM);

    for (int il = model_.il0(); il <= model_.il1(); ++il) layer(il, token, rec);

    int id = -1;
    if (model_.have_head() && model_.il1() == N_LAYER - 1) {
        Backend & b = model_.dev(model_.n_devices() - 1);
        // build_hc_head: flat norm, 4-row mix, sigmoid+eps, the weighted sum
        b.rms_norm_mul(H, nullptr, Hn, HC_DIM, 1, HC_DIM, eps_);
        b.gemv(model_.head_fn(), Hn, hmix);
        rec.tap(b, "hc_head_mixes", -1, hmix, HC);
        ops_.hc_head_pre(hmix, model_.head_scale(), model_.head_base(), hpre, hc_eps_);
        rec.tap(b, "hc_head_pre", -1, hpre, HC);
        ops_.hc_weighted_sum(H, hpre, hx);
        rec.tap(b, "hc_head", -1, hx, N_EMBD);
        b.rms_norm_mul(hx, model_.output_norm(), hxn, N_EMBD, 1, N_EMBD, eps_);
        rec.tap(b, "result_norm", -1, hxn, N_EMBD);
        b.gemv(model_.output(), hxn, logits);
        rec.tap(b, "result_output", -1, logits, N_VOCAB);
        int best = 0;
        b.argmax(logits, N_VOCAB, &best);
        id = best;
    }
    ++pos_;
    return id;
}

std::vector<float> Ds4Runner::logits_host() {
    std::vector<float> v;
    if (!logits) return v;
    v.resize(N_VOCAB);
    model_.dev(model_.n_devices() - 1).download(v.data(), logits, N_VOCAB * sizeof(float));
    return v;
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

} // namespace

int ds4_main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model_path, oracle_dir, dump_dir, routing_dir;
    std::vector<int32_t> tokens;
    int il0 = 0, il1 = N_LAYER - 1, n_devices = 3, threads = 4, ctx = 512, greedy_n = 0;
    bool with_head = true, use_cpu = false;
    double min_cos = 0.999;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--routing"&& i + 1 < argc) routing_dir = argv[++i];
        else if (a == "--dump"   && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--cpu")                    use_cpu = true;
        else if (a == "--no-head")                with_head = false;
        else if (a == "--devices"&& i + 1 < argc) n_devices = std::atoi(argv[++i]);
        else if (a == "--threads"&& i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--greedy" && i + 1 < argc) greedy_n = std::atoi(argv[++i]);
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
        std::fprintf(stderr, "usage: franken_decode --model <deepseek4 shard> --tokens <ids> --cpu "
                             "[--layers A-B] [--no-head] [--oracle DIR] [--routing DIR] [--dump DIR] "
                             "[--greedy N] [--ctx N] [--threads N] [--min-cos X]\n");
        return 2;
    }
    if (!use_cpu) { std::fprintf(stderr, "ds4: the GPU ops are L5 step 2; pass --cpu\n"); return 2; }
    if ((int) tokens.size() + greedy_n > ctx) { std::fprintf(stderr, "--ctx too small\n"); return 2; }

    try {
        std::unique_ptr<Backend> cpu(make_cpu_backend(threads));
        std::vector<Backend *> devs((size_t) std::max(1, n_devices), cpu.get());
        const bool head = with_head && il1 == N_LAYER - 1;
        Ds4Model model(model_path, devs, il0, il1, head);
        std::printf("arch=deepseek4 backend=%s layers=%d-%d head=%d ctx=%d tokens=%zu placed=%.2f GB "
                    "rms_eps=%g hc_eps=%g\n", cpu->name(), il0, il1, (int) head, ctx, tokens.size(),
                    model.placed_bytes() / 1e9, model.rms_eps(), model.hc_eps());
        std::unique_ptr<Ds4Ops> ops(make_ds4_cpu_ops());
        Ds4Config cfg; cfg.ctx = ctx;
        Ds4Runner run(model, *ops, cfg);
        run.report_cache_bytes(stdout);

        std::map<std::pair<int, int>, std::vector<int>> ref_routes;
        const bool have_routes = !routing_dir.empty() && load_routing(routing_dir, ref_routes);
        if (!routing_dir.empty() && !have_routes) std::printf("routing PENDING: no %s/moe_ids.txt\n", routing_dir.c_str());
        std::map<int, std::pair<double, int>> overlap;   // il -> (sum overlap, n)
        int exact_sets = 0, n_sets = 0;
        auto observe = [&](int pos) {
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
        std::vector<int> last_ids;   // the prompt's last position's routed ids
        for (size_t t = 0; t < tokens.size(); ++t) {
            rec.enable(t + 1 == tokens.size());
            last = run.step(tokens[t], rec);
            observe((int) t);
            if (t == 0) std::printf("token 0 done\n");
        }
        last_ids = run.routed_ids();
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
        }

        // SLOT ORDER. The per-expert-slot taps (ffn_moe_up, _gate, _down, the
        // weights, ...) are laid out in the router's top-k ORDER, and the two
        // engines agree on the SET but may order a near-tie differently
        // (2026-09-23, 136 tokens, layer 3: experts 143 and 118 at 9.6574 /
        // 9.6534 in the reference and 9.6580 / 9.6579 here). The order is not
        // part of the math -- build_moe_ffn sums the slots -- so when the last
        // position's sets are EQUAL the taps are permuted into the reference's
        // order before comparing, and the line says so. When the sets differ
        // nothing is permuted and the per-slot taps fail honestly; that is a
        // routing divergence and the routing lines report it.
        bool ok = true;
        if (have_routes && !oracle_dir.empty()) {
            const int last_pos = (int) tokens.size() - 1;
            for (int il = il0; il <= il1; ++il) {
                auto it = ref_routes.find({il, last_pos});
                if (it == ref_routes.end() || it->second.size() != (size_t) N_EXPERT_USED) continue;
                const int * mine = last_ids.data() + (size_t) (il - il0) * N_EXPERT_USED;
                int perm[N_EXPERT_USED];     // reference slot k <- my slot perm[k]
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
                // the layer OUTPUT must be among the compared points, not just
                // the pointwise taps on the way there
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
                std::printf("DS4 CPU ORACLE %s (min_cos=%g)\n", (cmp_ok && req_ok) ? "PASS" : "FAIL", min_cos);
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
