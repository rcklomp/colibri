// tools/hot-expert/franken/decode/glm5_model.cpp -- see glm5_model.h and GLM5.md.

#include "glm5_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tuple>

#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) glm5_half_bits_to_f32(bits)
static inline float glm5_half_bits_to_f32(unsigned int bits);
#include "decode_quant.h"
#include "decode_model.h"      // fk_type_of

#include "ggml.h"
#include "gguf.h"

static inline float glm5_half_bits_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    const uint16_t u = (uint16_t) bits;
    std::memcpy(&h, &u, sizeof(u));
    return ggml_fp16_to_fp32(h);
}

namespace fk {
namespace glm5 {

namespace {

double kv_num(const gguf_context * ctx, const std::string & key, bool & found) {
    const int64_t k = gguf_find_key(ctx, key.c_str());
    found = k >= 0;
    if (!found) return 0.0;
    switch (gguf_get_kv_type(ctx, k)) {
        case GGUF_TYPE_UINT8:   return gguf_get_val_u8(ctx, k);
        case GGUF_TYPE_INT8:    return gguf_get_val_i8(ctx, k);
        case GGUF_TYPE_UINT16:  return gguf_get_val_u16(ctx, k);
        case GGUF_TYPE_INT16:   return gguf_get_val_i16(ctx, k);
        case GGUF_TYPE_UINT32:  return gguf_get_val_u32(ctx, k);
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(ctx, k);
        case GGUF_TYPE_UINT64:  return (double) gguf_get_val_u64(ctx, k);
        case GGUF_TYPE_INT64:   return (double) gguf_get_val_i64(ctx, k);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, k);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, k);
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, k) ? 1.0 : 0.0;
        default: found = false; return 0.0;
    }
}

// VRAM a card, and what a card keeps off the expert budget (GLM5.md 5)
constexpr double CARD_GB    = 25.769803776;   // 24 GiB
constexpr double RESERVE_GB = 1.0;            // HIP context + allocator slack (the Qwen3.8 run's)
constexpr double SCRATCH_GB = 0.5;            // a 512-row prefill chunk's intermediates + attention
constexpr double STAGING_GB = 0.3;            // miss staging ring: 2 layers x 8 slabs x <= 15.8 MB

} // namespace

void Glm5Model::check_hparams() {
    gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(model_->shard_path(0).c_str(), p);
    if (!ctx) throw std::runtime_error("cannot read the GGUF header of " + model_->shard_path(0));
    std::string err;
    auto want = [&](const char * suffix, double exp) {
        bool found = false;
        const double got = kv_num(ctx, std::string("glm5next.") + suffix, found);
        if (!found || std::fabs(got - exp) > 1e-6 * std::max(1.0, std::fabs(exp))) {
            char buf[240];
            std::snprintf(buf, sizeof(buf), "  glm5next.%s: file %s%g, glm5_shapes.h %g\n",
                          suffix, found ? "" : "(absent) ", got, exp);
            err += buf;
        }
    };
    want("block_count",                          N_BLOCK);
    want("nextn_predict_layers",                 N_BLOCK - N_LAYER);
    want("embedding_length",                     N_EMBD);
    want("feed_forward_length",                  N_FF_DENSE);
    want("leading_dense_block_count",            N_LEAD_DENSE);
    want("attention.head_count",                 N_HEAD);
    want("attention.q_lora_rank",                Q_LORA);
    want("attention.kv_lora_rank",               KV_LORA);
    want("attention.key_length_mla",             QK_HEAD);
    want("attention.value_length_mla",           V_HEAD);
    want("rope.dimension_count",                 0);
    want("ssm.conv_kernel",                      KDA_CONV);
    want("kda.head_dim",                         KDA_DIM);
    want("kda.gate_lower_bound",                 KDA_GATE_LOWER_BOUND);
    want("attention.indexer.head_count",         IDX_N_HEAD);
    want("attention.indexer.key_length",         IDX_DIM);
    want("attention.indexer.top_k",              IDX_TOP_K);
    want("attention.indexer.kpool",              KPOOL);
    want("hyper_connection.count",               HC);
    want("hyper_connection.sinkhorn_iterations", HC_SINKHORN_ITERS);
    want("expert_count",                         N_EXPERT);
    want("expert_used_count",                    N_EXPERT_USED);
    want("expert_feed_forward_length",           N_FF_EXP);
    want("expert_shared_feed_forward_length",    N_FF_EXP);
    want("expert_shared_count",                  1);
    want("expert_weights_scale",                 EXPERT_WEIGHTS_SCALE);
    want("expert_weights_norm",                  1);
    want("expert_gating_func",                   2);   // LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
    want("expert_group_count",                   1);
    {
        bool f = false;
        rms_eps_ = (float) kv_num(ctx, "glm5next.attention.layer_norm_rms_epsilon", f);
        if (!f) err += "  attention.layer_norm_rms_epsilon absent\n";
        ln_eps_ = (float) kv_num(ctx, "glm5next.attention.layer_norm_epsilon", f);
        if (!f) err += "  attention.layer_norm_epsilon absent\n";
        hc_eps_ = (float) kv_num(ctx, "glm5next.hyper_connection.epsilon", f);
        if (!f) err += "  hyper_connection.epsilon absent\n";
    }
    for (const char * key : {"glm5next.swiglu_clamp_exp", "glm5next.swiglu_clamp_shexp"}) {
        const int64_t k = gguf_find_key(ctx, key);
        if (k < 0 || gguf_get_kv_type(ctx, k) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx, k) != GGUF_TYPE_FLOAT32) {
            err += std::string("  ") + key + " is not an f32 array\n";
            continue;
        }
        const float * v = (const float *) gguf_get_arr_data(ctx, k);
        for (size_t i = 0; i < std::min<size_t>(gguf_get_arr_n(ctx, k), N_LAYER); ++i)
            if (v[i] != SWIGLU_CLAMP) { err += std::string("  ") + key + " is not 10 everywhere\n"; break; }
    }
    gguf_free(ctx);

    // the per-layer kind: head_count_kv 0 = KDA, 1 = DSA
    const auto & hp = model_->hparams();
    if (hp.head_count_kv_layers.size() < (size_t) N_BLOCK) err += "  attention.head_count_kv is not a per-layer array\n";
    else for (int il = 0; il < N_BLOCK; ++il)
        if ((int) hp.head_count_kv_layers[(size_t) il] != (is_dsa(il) ? 1 : 0)) {
            char buf[120];
            std::snprintf(buf, sizeof(buf), "  head_count_kv[%d] = %u, glm5_shapes.h is_dsa %d\n",
                          il, hp.head_count_kv_layers[(size_t) il], (int) is_dsa(il));
            err += buf;
        }
    if (!err.empty()) throw std::runtime_error("GLM-5.3 header disagrees with glm5_shapes.h:\n" + err);
}

const TensorInfo * Glm5Model::need(const std::string & name) const {
    const TensorInfo * t = model_->find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

// The format, row stride and slice stride of a tensor, asserted against the
// file's own byte size before anything is placed.
Mat Glm5Model::describe_mat(const std::string & name, size_t n_slices) {
    const TensorInfo * t = need(name);
    Mat m;
    m.name          = strdup(name.c_str());
    m.type          = fk_type_of(t->type);
    m.K             = t->ne0();
    m.rows          = t->ne1();
    m.row_bytes     = fk_row_bytes(m.type, m.K);
    m.expert_stride = n_slices > 1 ? t->slice_bytes((int64_t) n_slices) : 0;
    const size_t expect = m.row_bytes * (size_t) m.rows * (n_slices ? n_slices : 1);
    if (expect != t->nbytes || m.row_bytes != t->row_size()) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "%s: row/slice arithmetic disagrees with the file -- ne=[%lld,%lld,%lld] type=%s "
            "row_bytes=%zu slices=%zu => %zu, file says %zu",
            name.c_str(), (long long) t->ne0(), (long long) t->ne1(), (long long) t->ne2(),
            ggml_type_name(t->type), m.row_bytes, n_slices, expect, t->nbytes);
        throw std::runtime_error(buf);
    }
    return m;
}

Mat Glm5Model::place_mat(Backend & be, const std::string & name, size_t n_slices) {
    Mat m = describe_mat(name, n_slices);
    const TensorInfo * t = need(name);
    m.base = be.place(t->data, t->nbytes, m.name);
    placed_ += t->nbytes;
    return m;
}

const float * Glm5Model::place_f32(Backend & be, const std::string & name, int64_t n_expect) {
    const TensorInfo * t = need(name);
    if (t->type != GGML_TYPE_F32) throw std::runtime_error(name + ": expected F32");
    const int64_t n = t->ne0() * t->ne1() * t->ne2() * t->ne3();
    if (n != n_expect) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), "%s: expected %lld elements, file has %lld",
                      name.c_str(), (long long) n_expect, (long long) n);
        throw std::runtime_error(buf);
    }
    placed_ += t->nbytes;
    return (const float *) be.place(t->data, t->nbytes, name.c_str());
}

int Glm5Model::dev_of_layer(int il) const {
    const int n_dev = (int) devs_.size();
    if (!split_.empty()) {
        int d = 0;
        for (size_t i = 0; i < split_.size() && (int) i + 1 < n_dev; ++i) if (il >= split_[i]) d = (int) i + 1;
        return d;
    }
    const int n_layers = il1_ - il0_ + 1;
    return std::min(n_dev - 1, (il - il0_) * n_dev / n_layers);
}

// Every byte of layer il's tensors except the routed experts.
size_t Glm5Model::trunk_bytes(int il) const {
    const std::string pre = "blk." + std::to_string(il) + ".";
    size_t n = 0;
    for (const TensorInfo & t : model_->tensors())
        if (t.name.compare(0, pre.size(), pre) == 0 && t.name.find("_exps") == std::string::npos) n += t.nbytes;
    return n;
}

// Per card: every (layer, expert) of its layers ranked by histogram count,
// taken until the card's budget is spent (the DeepSeek rule: count alone is
// the saving-per-byte ratio). Prints, per card, the resident set, the
// in-sample expected miss bytes a token and the VRAM projection at pl.ctx;
// then the stream time a token two ways (GLM5.md section 5).
void Glm5Model::plan_placement(const Placement & pl) {
    resident_.assign(N_BLOCK, std::vector<char>(N_EXPERT, 1));
    if (pl.hist_dir.empty()) {
        if (devs_[0]->is_gpu())
            throw std::runtime_error("a GPU run needs --placement <dir with layer_<il>.csv> "
                                     "(the M2 histogram): 141 GB of experts do not fit");
        return;
    }
    const int n_dev = (int) devs_.size();
    std::vector<std::vector<long long>> cnt(N_BLOCK, std::vector<long long>(N_EXPERT, 0));
    std::vector<long long> tot(N_BLOCK, 0);
    for (int il = std::max(il0_, N_LEAD_DENSE); il <= il1_; ++il) {
        std::ifstream f(pl.hist_dir + "/layer_" + std::to_string(il) + ".csv");
        if (!f) throw std::runtime_error("placement: no " + pl.hist_dir + "/layer_" + std::to_string(il) + ".csv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            int e = -1; long long c = 0; char comma;
            std::istringstream is(line);
            if (is >> e >> comma >> c && e >= 0 && e < N_EXPERT) { cnt[il][e] = c; tot[il] += c; }
        }
    }
    auto slab = [&](int il) {
        const auto b = [&](const char * s) { return need("blk." + std::to_string(il) + "." + s)->slice_bytes(N_EXPERT); };
        return b("ffn_gate_exps.weight") + b("ffn_up_exps.weight") + b("ffn_down_exps.weight");
    };
    planned_ = true;
    double miss_all = 0.0, stream_serial_ms = 0.0;
    for (int d = 0; d < n_dev; ++d) {
        const double budget = (d < (int) pl.expert_gb.size() ? pl.expert_gb[(size_t) d] : pl.expert_gb.back()) * 1e9;
        std::vector<std::tuple<long long, int, int>> c;
        int la = -1, lb = -1, n_dsa = 0, n_moe = 0;
        double trunk = 0.0;
        for (int il = il0_; il <= il1_; ++il) {
            if (dev_of_layer(il) != d) continue;
            if (la < 0) la = il;
            lb = il;
            trunk += (double) trunk_bytes(il);
            n_dsa += is_dsa(il);
            if (il < N_LEAD_DENSE) continue;
            ++n_moe;
            for (int e = 0; e < N_EXPERT; ++e) { c.emplace_back(-cnt[il][e], il, e); resident_[il][e] = 0; }
        }
        if (la < 0) continue;
        const bool head = d == n_dev - 1 && il1_ == N_LAYER - 1;
        if (head) trunk += (double) need("output.weight")->nbytes + (double) need("output_norm.weight")->nbytes;
        std::sort(c.begin(), c.end());
        double used = 0.0, miss = 0.0, host = 0.0;
        int n_res = 0;
        for (auto & t : c) {
            const int il = std::get<1>(t), e = std::get<2>(t);
            const double sb = (double) slab(il);
            if (used + sb <= budget) { resident_[il][e] = 1; used += sb; ++n_res; }
            else {
                host += sb;
                // P(e chosen) = count / positions, positions = total / 8 -- IN-SAMPLE
                if (tot[il] > 0) miss += sb * (double) cnt[il][e] / ((double) tot[il] / N_EXPERT_USED);
            }
        }
        // positional state at pl.ctx: a DSA layer keeps the f16 latent (512) a
        // cell and the f16 pooled indexer key (128) a pool; a KDA layer a fixed
        // 64 x 128 x 128 f32 state plus 3 conv rows
        const double kv = (double) n_dsa * (double) pl.ctx * (KV_LORA * 2.0 + IDX_DIM * 2.0 / KPOOL);
        const double kda = (double) (lb - la + 1 - n_dsa) * (N_HEAD * KDA_DIM * KDA_DIM * 4.0 + (KDA_CONV - 1) * KDA_QKV * 4.0);
        const double room = CARD_GB * 1e9 - trunk - kv - kda - (RESERVE_GB + SCRATCH_GB + STAGING_GB) * 1e9;
        miss_all += miss;
        stream_serial_ms += miss / 28e9 * 1e3;
        std::printf("placement dev=%d layers=%d-%d moe_layers=%d dsa_layers=%d trunk_gb=%.2f%s kv_gb=%.2f "
                    "kda_state_gb=%.3f expert_room_gb=%.2f resident=%d experts_gb=%.2f host_gb=%.2f "
                    "est_miss_mb_per_token=%.1f (histogram %s, in-sample)\n",
                    d, la, lb, n_moe, n_dsa, trunk / 1e9, head ? " (+head)" : "", kv / 1e9, kda / 1e9,
                    room / 1e9, n_res, used / 1e9, host / 1e9, miss / 1e6, pl.hist_dir.c_str());
        if (used > room) std::printf("placement dev=%d OVER BUDGET: experts %.2f GB > room %.2f GB\n", d, used / 1e9, room / 1e9);
    }
    // Stream time a token, two ways (record §M4): each card copying its own
    // layers' misses while it runs (the cards take turns, one ~28 GB/s link at
    // a time), or every layer's misses split over the three links at once
    // (27.5 / 27.5 / 45 %: 17.2 + 17.2 + 28 GB/s, the lone card the largest
    // share) and computed where they land.
    std::printf("placement all est_miss_mb_per_token=%.1f stream_ms_owner_card=%.1f stream_ms_three_links=%.1f "
                "(28 GB/s one link at a time vs 62.4 GB/s the three together; projections, not measurements)\n",
                miss_all / 1e6, stream_serial_ms, miss_all / 62.4e9 * 1e3);
}

// The expert table of one layer. CPU arm: every entry is an mmap address.
void Glm5Model::place_experts(LayerWeights & L, Backend & be) {
    const int il = L.il;
    const TensorInfo * tg = need("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
    const TensorInfo * tu = need("blk." + std::to_string(il) + ".ffn_up_exps.weight");
    const TensorInfo * td = need("blk." + std::to_string(il) + ".ffn_down_exps.weight");
    const size_t sg = tg->slice_bytes(N_EXPERT), su = tu->slice_bytes(N_EXPERT), sd = td->slice_bytes(N_EXPERT);
    L.et.type_gu = L.exp_gate.type; L.et.type_d = L.exp_down.type;
    L.et.row_gu  = L.exp_gate.row_bytes; L.et.row_d = L.exp_down.row_bytes;
    L.et.K_gu = (int) L.exp_gate.K; L.et.rows_gu = (int) L.exp_gate.rows;
    L.et.K_d  = (int) L.exp_down.K; L.et.rows_d  = (int) L.exp_down.rows;
    if (L.exp_up.type != L.exp_gate.type) throw std::runtime_error("gate/up formats differ on layer " + std::to_string(il));
    L.et.sz_g = sg; L.et.sz_u = su; L.et.sz_d = sd;
    L.tab_host.assign(3 * N_EXPERT, nullptr);
    L.miss_host.assign(N_EXPERT, 0);
    if (be.is_gpu()) throw std::runtime_error("GLM GPU expert placement is not built (GLM5.md section 8)");
    for (int e = 0; e < N_EXPERT; ++e) {
        L.tab_host[e]                = tg->data + (size_t) e * sg;
        L.tab_host[N_EXPERT + e]     = tu->data + (size_t) e * su;
        L.tab_host[2 * N_EXPERT + e] = td->data + (size_t) e * sd;
    }
    L.et.miss_bytes = L.miss_host.data();
    L.et.gate = L.tab_host.data();
    L.et.up   = L.tab_host.data() + N_EXPERT;
    L.et.down = L.tab_host.data() + 2 * N_EXPERT;
}

Glm5Model::Glm5Model(const std::string & path, std::vector<Backend *> devs, int il0, int il1,
                     bool with_head, const std::vector<int> & split, const Placement & pl)
    : devs_(std::move(devs)), split_(split), il0_(il0), il1_(il1) {
    if (devs_.empty()) throw std::runtime_error("Glm5Model needs at least one backend");
    if (il0 < 0 || il1 >= N_LAYER || il1 < il0)
        throw std::runtime_error("layer span out of range (0-44: the NextN block 45 is not run)");
    model_ = GgufModel::open(path);
    if (model_->hparams().arch != "glm5next")
        throw std::runtime_error("arch is " + model_->hparams().arch + ", not glm5next");
    check_hparams();

    tok_embd_ = need("token_embd.weight");
    if (tok_embd_->type != GGML_TYPE_Q8_0 || tok_embd_->ne0() != N_EMBD || tok_embd_->ne1() != N_VOCAB)
        throw std::runtime_error("token_embd.weight is not Q8_0 [4096, 154880]");
    if (devs_[0]->is_gpu())
        embd_copy_.assign(tok_embd_->data, tok_embd_->data + tok_embd_->nbytes);
    plan_placement(pl);

    layers_.resize((size_t) (il1 - il0 + 1));
    for (int il = il0; il <= il1; ++il) {
        LayerWeights & L = layers_[(size_t) (il - il0)];
        L.il  = il;
        L.dsa = is_dsa(il);
        L.moe = il >= N_LEAD_DENSE;
        L.dev = dev_of_layer(il);
        Backend & be = *devs_[(size_t) L.dev];
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
        const std::string lname = "layer " + std::to_string(il);

        L.hc_attn_fn    = place_mat(be, b("hc_attn_fn.weight"), 0);
        L.hc_attn_base  = place_f32(be, b("hc_attn_base.weight"), HC_MIX);
        L.hc_attn_scale = place_f32(be, b("hc_attn_scale.weight"), 3);
        L.hc_ffn_fn     = place_mat(be, b("hc_ffn_fn.weight"), 0);
        L.hc_ffn_base   = place_f32(be, b("hc_ffn_base.weight"), HC_MIX);
        L.hc_ffn_scale  = place_f32(be, b("hc_ffn_scale.weight"), 3);
        if (L.hc_attn_fn.K != HC_DIM || L.hc_attn_fn.rows != HC_MIX || L.hc_ffn_fn.rows != HC_MIX)
            throw std::runtime_error(b("hc_attn_fn.weight") + " is not [16384, 24]");
        L.attn_norm = place_f32(be, b("attn_norm.weight"), N_EMBD);
        L.ffn_norm  = place_f32(be, b("ffn_norm.weight"), N_EMBD);

        if (!L.dsa) {
            L.wq = place_mat(be, b("attn_q.weight"), 0);
            L.wk = place_mat(be, b("attn_k.weight"), 0);
            L.wv = place_mat(be, b("attn_v.weight"), 0);
            L.conv_q = place_f32(be, b("ssm_conv1d_q.weight"), (int64_t) KDA_CONV * KDA_INNER);
            L.conv_k = place_f32(be, b("ssm_conv1d_k.weight"), (int64_t) KDA_CONV * KDA_INNER);
            L.conv_v = place_f32(be, b("ssm_conv1d_v.weight"), (int64_t) KDA_CONV * KDA_INNER);
            L.f_a  = place_mat(be, b("ssm_f_a.weight"), 0);
            L.f_b  = place_mat(be, b("ssm_f_b.weight"), 0);
            L.g_a  = place_mat(be, b("ssm_g_a.weight"), 0);
            L.g_b  = place_mat(be, b("ssm_g_b.weight"), 0);
            L.beta = place_mat(be, b("ssm_beta.weight"), 0);
            L.ssm_a  = place_f32(be, b("ssm_a"), N_HEAD);
            L.dt_b   = place_f32(be, b("ssm_dt.bias"), KDA_INNER);
            L.o_norm = place_f32(be, b("ssm_norm.weight"), KDA_DIM);
            L.wo     = place_mat(be, b("attn_output.weight"), 0);
            if (L.wq.K != N_EMBD || L.wq.rows != KDA_INNER || L.wk.rows != KDA_INNER || L.wv.rows != KDA_INNER ||
                L.f_a.rows != KDA_DIM || L.f_b.K != KDA_DIM || L.f_b.rows != KDA_INNER ||
                L.g_a.rows != KDA_DIM || L.g_b.rows != KDA_INNER || L.beta.rows != N_HEAD ||
                L.wo.K != KDA_INNER || L.wo.rows != N_EMBD)
                throw std::runtime_error(lname + ": KDA shapes moved");
        } else {
            L.wq_a      = place_mat(be, b("attn_q_a.weight"), 0);
            L.q_a_norm  = place_f32(be, b("attn_q_a_norm.weight"), Q_LORA);
            L.wq_b      = place_mat(be, b("attn_q_b.weight"), 0);
            L.wkv_a     = place_mat(be, b("attn_kv_a_mqa.weight"), 0);
            L.kv_a_norm = place_f32(be, b("attn_kv_a_norm.weight"), KV_LORA);
            {
                // [K, rows, 64]: one [rows x K] matrix a head, head h at h * slice
                const Mat kb = place_mat(be, b("attn_k_b.weight"), N_HEAD);
                const Mat vb = place_mat(be, b("attn_v_b.weight"), N_HEAD);
                if (kb.K != QK_HEAD || kb.rows != KV_LORA || vb.K != KV_LORA || vb.rows != V_HEAD)
                    throw std::runtime_error(lname + ": attn_k_b / attn_v_b are not [256,512,64] / [512,256,64]");
                for (int h = 0; h < N_HEAD; ++h) {
                    L.wk_b[h] = kb; L.wk_b[h].expert_stride = 0;
                    L.wk_b[h].base = (const unsigned char *) kb.base + (size_t) h * kb.expert_stride;
                    L.wv_b[h] = vb; L.wv_b[h].expert_stride = 0;
                    L.wv_b[h].base = (const unsigned char *) vb.base + (size_t) h * vb.expert_stride;
                }
            }
            L.wo           = place_mat(be, b("attn_output.weight"), 0);
            L.idx_k        = place_mat(be, b("indexer.attn_k.weight"), 0);
            L.idx_k_norm   = place_f32(be, b("indexer.k_norm.weight"), IDX_DIM);
            L.idx_k_norm_b = place_f32(be, b("indexer.k_norm.bias"), IDX_DIM);
            L.idx_gate     = place_mat(be, b("indexer_compressor_gate.weight"), 0);
            L.idx_ape      = place_f32(be, b("indexer_compressor_ape.weight"), (int64_t) KPOOL * IDX_DIM);
            L.idx_q_b      = place_mat(be, b("indexer.attn_q_b.weight"), 0);
            L.idx_proj     = place_mat(be, b("indexer.proj.weight"), 0);
            if (L.wq_a.rows != Q_LORA || L.wq_b.K != Q_LORA || L.wq_b.rows != Q_WIDTH ||
                L.wkv_a.rows != KV_LORA || L.wo.K != O_WIDTH || L.wo.rows != N_EMBD ||
                L.idx_k.rows != IDX_DIM || L.idx_gate.rows != IDX_DIM ||
                L.idx_q_b.K != Q_LORA || L.idx_q_b.rows != IDX_N_HEAD * IDX_DIM ||
                L.idx_proj.rows != IDX_N_HEAD || L.idx_proj.type != FK_Q_F32)
                throw std::runtime_error(lname + ": DSA shapes moved");
        }

        if (!L.moe) {
            L.ffn_gate = place_mat(be, b("ffn_gate.weight"), 0);
            L.ffn_up   = place_mat(be, b("ffn_up.weight"), 0);
            L.ffn_down = place_mat(be, b("ffn_down.weight"), 0);
            if (L.ffn_gate.rows != N_FF_DENSE || L.ffn_up.rows != N_FF_DENSE || L.ffn_down.K != N_FF_DENSE ||
                L.ffn_down.rows != N_EMBD)
                throw std::runtime_error(lname + ": dense FFN shapes moved");
        } else {
            L.gate_inp    = place_mat(be, b("ffn_gate_inp.weight"), 0);
            L.exp_probs_b = place_f32(be, b("exp_probs_b.bias"), N_EXPERT);
            L.exp_gate = describe_mat(b("ffn_gate_exps.weight"), N_EXPERT);
            L.exp_up   = describe_mat(b("ffn_up_exps.weight"),   N_EXPERT);
            L.exp_down = describe_mat(b("ffn_down_exps.weight"), N_EXPERT);
            L.sh_gate  = place_mat(be, b("ffn_gate_shexp.weight"), 0);
            L.sh_up    = place_mat(be, b("ffn_up_shexp.weight"),   0);
            L.sh_down  = place_mat(be, b("ffn_down_shexp.weight"), 0);
            if (L.exp_gate.K != N_EMBD || L.exp_gate.rows != N_FF_EXP || L.exp_down.K != N_FF_EXP ||
                L.exp_down.rows != N_EMBD || L.sh_up.rows != N_FF_EXP || L.sh_down.rows != N_EMBD ||
                L.gate_inp.rows != N_EXPERT || L.gate_inp.type != FK_Q_F32)
                throw std::runtime_error(lname + ": MoE shapes moved");
            place_experts(L, be);
        }
    }

    if (with_head) {
        Backend & be = *devs_.back();
        output_norm_ = place_f32(be, "output_norm.weight", N_EMBD);
        output_      = place_mat(be, "output.weight", 0);
        if (output_.K != N_EMBD || output_.rows != N_VOCAB) throw std::runtime_error("head shapes moved");
        have_head_ = true;
    }
}

void Glm5Model::embed_row(int32_t tok, float * out) const {
    if (tok < 0 || tok >= N_VOCAB) throw std::runtime_error("token id out of range");
    const size_t row = tok_embd_->row_size();
    const uint8_t * base = embd_copy_.empty() ? tok_embd_->data : (const uint8_t *) embd_copy_.data();
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(base + (size_t) tok * row, out, N_EMBD);
}

} // namespace glm5
} // namespace fk
