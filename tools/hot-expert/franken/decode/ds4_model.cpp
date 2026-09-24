// tools/hot-expert/franken/decode/ds4_model.cpp -- see ds4_model.h.

#include "ds4_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <tuple>
#include <cstring>
#include <stdexcept>

#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) ds4_half_bits_to_f32(bits)
static inline float ds4_half_bits_to_f32(unsigned int bits);
#include "decode_quant.h"
#include "decode_model.h"      // fk_type_of

#include "ggml.h"
#include "gguf.h"

static inline float ds4_half_bits_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    const uint16_t u = (uint16_t) bits;
    std::memcpy(&h, &u, sizeof(u));
    return ggml_fp16_to_fp32(h);
}

namespace fk {
namespace ds4 {

namespace {

// One numeric GGUF value, whatever scalar type the converter chose.
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

} // namespace

// Every constant ds4_shapes.h compiles in, read back off the header. The
// kernels (and this graph) are written around them; the file is not trusted.
void Ds4Model::check_hparams() {
    gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(model_->shard_path(0).c_str(), p);
    if (!ctx) throw std::runtime_error("cannot read the GGUF header of " + model_->shard_path(0));
    std::string err;
    auto want = [&](const char * suffix, double exp) {
        bool found = false;
        const double got = kv_num(ctx, std::string("deepseek4.") + suffix, found);
        if (!found || std::fabs(got - exp) > 1e-6 * std::max(1.0, std::fabs(exp))) {
            char buf[240];
            std::snprintf(buf, sizeof(buf), "  deepseek4.%s: file %s%g, ds4_shapes.h %g\n",
                          suffix, found ? "" : "(absent) ", got, exp);
            err += buf;
        }
    };
    want("block_count",                        N_LAYER);
    want("embedding_length",                   N_EMBD);
    want("attention.head_count",               N_HEAD);
    want("attention.head_count_kv",            1);
    want("attention.key_length",               HEAD_DIM);
    want("attention.value_length",             HEAD_DIM);
    want("rope.dimension_count",               N_ROT);
    want("attention.q_lora_rank",              Q_LORA);
    want("attention.output_group_count",       O_GROUPS);
    want("attention.output_lora_rank",         O_LORA);
    want("attention.sliding_window",           N_SWA);
    want("attention.indexer.head_count",       IDX_N_HEAD);
    want("attention.indexer.key_length",       IDX_DIM);
    want("attention.indexer.top_k",            IDX_TOP_K);
    want("attention.compress_rope_freq_base",  COMPRESS_ROPE_FREQ_BASE);
    want("hyper_connection.count",             HC);
    want("hyper_connection.sinkhorn_iterations", HC_SINKHORN_ITERS);
    want("hash_layer_count",                   HASH_LAYERS);
    want("expert_count",                       N_EXPERT);
    want("expert_used_count",                  N_EXPERT_USED);
    want("expert_feed_forward_length",         N_FF_EXP);
    want("expert_shared_count",                1);
    want("expert_weights_scale",               EXPERT_WEIGHTS_SCALE);
    want("expert_weights_norm",                1);
    want("expert_gating_func",                 4);   // LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS
    want("rope.freq_base",                     ROPE_FREQ_BASE);
    want("rope.scaling.factor",                ROPE_SCALE_FACTOR);
    want("rope.scaling.original_context_length", ROPE_N_CTX_ORIG);
    want("rope.scaling.yarn_beta_fast",        YARN_BETA_FAST);
    want("rope.scaling.yarn_beta_slow",        YARN_BETA_SLOW);
    {
        bool f = false;
        rms_eps_ = (float) kv_num(ctx, "deepseek4.attention.layer_norm_rms_epsilon", f);
        if (!f) err += "  attention.layer_norm_rms_epsilon absent\n";
        hc_eps_ = (float) kv_num(ctx, "deepseek4.hyper_connection.epsilon", f);
        if (!f) err += "  hyper_connection.epsilon absent\n";
    }
    // the per-layer arrays: swiglu clamps must all be SWIGLU_CLAMP
    for (const char * key : {"deepseek4.swiglu_clamp_exp", "deepseek4.swiglu_clamp_shexp"}) {
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

    const auto & cr = model_->hparams().compress_ratios;
    if (cr.size() < (size_t) N_LAYER) err += "  compress_ratios shorter than block_count\n";
    else for (int il = 0; il < N_LAYER; ++il)
        if ((int) cr[il] != compress_ratio(il)) {
            char buf[120];
            std::snprintf(buf, sizeof(buf), "  compress_ratios[%d] = %u, ds4_shapes.h %d\n",
                          il, cr[il], compress_ratio(il));
            err += buf;
        }
    if (!err.empty()) throw std::runtime_error("DeepSeek-V4 header disagrees with ds4_shapes.h:\n" + err);
}

const TensorInfo * Ds4Model::need(const std::string & name) const {
    const TensorInfo * t = model_->find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

// As DecodeModel::place_mat: the row stride the GEMVs will use is asserted
// against the tensor's own byte size before anything is placed.
// The format, row stride and slice stride of a tensor, asserted against the
// file, WITHOUT placing it (the expert tensors are placed slab by slab).
Mat Ds4Model::describe_mat(const std::string & name, size_t n_slices) {
    const TensorInfo * t = need(name);
    Mat m;
    m.name          = strdup(name.c_str());
    m.type          = fk_type_of(t->type);
    m.K             = t->ne0();
    m.rows          = t->ne1();
    m.row_bytes     = fk_row_bytes(m.type, m.K);
    m.expert_stride = n_slices > 1 ? t->slice_bytes((int64_t) n_slices) : 0;
    const size_t expect = m.row_bytes * (size_t) m.rows * (n_slices ? n_slices : 1);
    if (expect != t->nbytes || m.row_bytes != t->row_size())
        throw std::runtime_error(name + ": row/slice arithmetic disagrees with the file");
    return m;
}

Mat Ds4Model::place_mat(Backend & be, const std::string & name, size_t n_slices) {
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
    m.base = be.place(t->data, t->nbytes, m.name);
    placed_ += t->nbytes;
    return m;
}

const float * Ds4Model::place_f32(Backend & be, const std::string & name, int64_t n_expect) {
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

// Per card: every (layer, expert) of its layers ranked by the histogram
// count, taken in order until the budget is spent. Saving a slab saves
// count x slab bytes of misses for slab bytes of VRAM, so count alone is the
// ratio and a single ranking across the card's layers is the greedy optimum.
int Ds4Model::n_resident(int il) const {
    int n = 0;
    for (int e = 0; e < N_EXPERT; ++e) n += resident_[(size_t) il][(size_t) e] != 0;
    return n;
}

void Ds4Model::plan_placement(const Placement & pl) {
    resident_.assign(N_LAYER, std::vector<char>(N_EXPERT, 1));
    if (pl.hist_dir.empty()) {
        if (devs_[0]->is_gpu())
            throw std::runtime_error("a GPU run needs --placement <dir with layer_<il>.csv> "
                                     "(the M2 histogram): 83.9 GB of experts do not fit");
        return;
    }
    const int n_dev = (int) devs_.size();
    const int n_layers = il1_ - il0_ + 1;
    std::vector<std::vector<long long>> cnt(N_LAYER, std::vector<long long>(N_EXPERT, 0));
    std::vector<long long> tot(N_LAYER, 0);
    for (int il = il0_; il <= il1_; ++il) {
        std::ifstream f(pl.hist_dir + "/layer_" + std::to_string(il) + ".csv");
        if (!f) throw std::runtime_error("placement: no " + pl.hist_dir + "/layer_" + std::to_string(il) + ".csv");
        std::string line;
        std::getline(f, line);                      // header: expert_id,count
        while (std::getline(f, line)) {
            int e = -1; long long c = 0; char comma;
            std::istringstream is(line);
            if (is >> e >> comma >> c && e >= 0 && e < N_EXPERT) { cnt[il][e] = c; tot[il] += c; }
        }
    }
    auto slab = [&](int il) {
        const auto b = [&](const char * s) {
            return need("blk." + std::to_string(il) + "." + s)->slice_bytes(N_EXPERT);
        };
        return b("ffn_gate_exps.weight") + b("ffn_up_exps.weight") + b("ffn_down_exps.weight");
    };
    const double budget = pl.expert_gb * 1e9;
    planned_ = true;
    for (int d = 0; d < n_dev; ++d) {
        std::vector<std::tuple<long long, int, int>> c;   // (-count, il, e)
        int la = -1, lb = -1;
        for (int il = il0_; il <= il1_; ++il) {
            if (std::min(n_dev - 1, (il - il0_) * n_dev / n_layers) != d) continue;
            if (la < 0) la = il;
            lb = il;
            for (int e = 0; e < N_EXPERT; ++e) { c.emplace_back(-cnt[il][e], il, e); resident_[il][e] = 0; }
        }
        if (la < 0) continue;
        std::sort(c.begin(), c.end());
        double used = 0.0, miss = 0.0, host = 0.0;
        int n_res = 0;
        for (auto & t : c) {
            const int il = std::get<1>(t), e = std::get<2>(t);
            const double sb = (double) slab(il);
            if (used + sb <= budget) { resident_[il][e] = 1; used += sb; ++n_res; }
            else {
                host += sb;
                // expected bytes a token: P(e chosen) = count / positions,
                // positions = total / 6 -- IN-SAMPLE, see DEEPSEEK4.md 5
                if (tot[il] > 0) miss += sb * (double) cnt[il][e] / ((double) tot[il] / N_EXPERT_USED);
            }
        }
        for (int il = la; il <= lb; ++il) {
            long long covered = 0;
            for (int e = 0; e < N_EXPERT; ++e) if (resident_[il][e]) covered += cnt[il][e];
            m2_cov_[(size_t) il] = tot[il] > 0 ? (double) covered / (double) tot[il] : 1.0;
        }
        std::printf("placement dev=%d layers=%d-%d resident=%d experts_gb=%.2f host_gb=%.2f "
                    "est_miss_mb_per_token=%.1f (histogram %s, in-sample)\n",
                    d, la, lb, n_res, used / 1e9, host / 1e9, miss / 1e6, pl.hist_dir.c_str());
    }
}

// The expert table of one layer. CPU arm: every entry is an mmap address.
// A card: resident experts are copied into one VRAM block, the others into
// one pinned, device-mapped host block, and the 3 x 256 addresses go to VRAM.
void Ds4Model::place_experts(LayerWeights & L, Backend & be, Ds4Ops & ops) {
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
    if ((sg | su | sd) % 16) throw std::runtime_error("expert slices of layer " + std::to_string(il) +
                                                      " are not 16-byte multiples (the staged loads need it)");
    L.tab_host.assign(3 * N_EXPERT, nullptr);
    L.miss_host.assign(N_EXPERT, 0);
    if (!be.is_gpu()) {
        L.et.miss_bytes = L.miss_host.data();          // all resident on the CPU arm
        for (int e = 0; e < N_EXPERT; ++e) {
            L.tab_host[e]                = tg->data + (size_t) e * sg;
            L.tab_host[N_EXPERT + e]     = tu->data + (size_t) e * su;
            L.tab_host[2 * N_EXPERT + e] = td->data + (size_t) e * sd;
        }
        L.n_resident = N_EXPERT;
        L.et.gate = L.tab_host.data();
        L.et.up   = L.tab_host.data() + N_EXPERT;
        L.et.down = L.tab_host.data() + 2 * N_EXPERT;
        return;
    }
    const size_t slab = sg + su + sd;
    std::vector<int> res, miss;
    for (int e = 0; e < N_EXPERT; ++e) (resident_[il][e] ? res : miss).push_back(e);
    unsigned char * vram = res.empty() ? nullptr : (unsigned char *) be.alloc_raw(res.size() * slab);
    for (size_t i = 0; i < res.size(); ++i) {
        const int e = res[i];
        unsigned char * dst = vram + i * slab;
        be.upload(dst,           tg->data + (size_t) e * sg, sg);
        be.upload(dst + sg,      tu->data + (size_t) e * su, su);
        be.upload(dst + sg + su, td->data + (size_t) e * sd, sd);
        L.tab_host[e] = dst; L.tab_host[N_EXPERT + e] = dst + sg; L.tab_host[2 * N_EXPERT + e] = dst + sg + su;
    }
    if (!miss.empty()) {
        void * host = nullptr;
        const unsigned char * view = (const unsigned char *) ops.alloc_host_mapped(miss.size() * slab, &host);
        unsigned char * h = (unsigned char *) host;
        for (size_t j = 0; j < miss.size(); ++j) {
            const int e = miss[j];
            std::memcpy(h + j * slab,           tg->data + (size_t) e * sg, sg);
            std::memcpy(h + j * slab + sg,      tu->data + (size_t) e * su, su);
            std::memcpy(h + j * slab + sg + su, td->data + (size_t) e * sd, sd);
            L.tab_host[e] = view + j * slab;
            L.tab_host[N_EXPERT + e] = view + j * slab + sg;
            L.tab_host[2 * N_EXPERT + e] = view + j * slab + sg + su;
            L.miss_host[e] = (int) slab;
        }
        host_expert_bytes_ += miss.size() * slab;
    }
    const void ** dtab = (const void **) be.alloc_raw(3 * N_EXPERT * sizeof(void *));
    be.upload(dtab, L.tab_host.data(), 3 * N_EXPERT * sizeof(void *));
    L.et.gate = dtab; L.et.up = dtab + N_EXPERT; L.et.down = dtab + 2 * N_EXPERT;
    int * dmiss = (int *) be.alloc_raw(N_EXPERT * sizeof(int));
    be.upload(dmiss, L.miss_host.data(), N_EXPERT * sizeof(int));
    L.et.miss_bytes = dmiss;
    L.n_resident = (int) res.size();
    placed_ += res.size() * slab;
}

Ds4Model::Ds4Model(const std::string & path, std::vector<Backend *> devs, std::vector<Ds4Ops *> ops,
                   int il0, int il1, bool with_head, const Placement & pl)
    : devs_(std::move(devs)), ops_(std::move(ops)), il0_(il0), il1_(il1) {
    if (devs_.empty()) throw std::runtime_error("Ds4Model needs at least one backend");
    if (ops_.size() != devs_.size()) throw std::runtime_error("Ds4Model: one Ds4Ops per backend");
    if (il0 < 0 || il1 >= N_LAYER || il1 < il0) throw std::runtime_error("layer span out of range");
    model_ = GgufModel::open(path);
    if (model_->hparams().arch != "deepseek4")
        throw std::runtime_error("arch is " + model_->hparams().arch + ", not deepseek4");
    check_hparams();

    tok_embd_ = need("token_embd.weight");
    if (tok_embd_->type != GGML_TYPE_Q5_K || tok_embd_->ne0() != N_EMBD || tok_embd_->ne1() != N_VOCAB)
        throw std::runtime_error("token_embd.weight is not Q5_K [4096, 129280]");
    // design 9.1: a per-token gather table lives in host RAM, loaded at start
    // (not left to the page cache) -- on a GPU run; the CPU arm reads the mmap
    if (devs_[0]->is_gpu())
        embd_copy_.assign(tok_embd_->data, tok_embd_->data + tok_embd_->nbytes);
    plan_placement(pl);

    const int n_layers = il1 - il0 + 1;
    const int n_dev = (int) devs_.size();
    layers_.resize((size_t) n_layers);
    for (int il = il0; il <= il1; ++il) {
        LayerWeights & L = layers_[(size_t) (il - il0)];
        L.il    = il;
        L.ratio = compress_ratio(il);
        L.dev   = std::min(n_dev - 1, (il - il0) * n_dev / n_layers);
        Backend & be = *devs_[(size_t) L.dev];
        auto b = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };

        L.hc_attn_fn    = place_mat(be, b("hc_attn_fn.weight"), 0);
        L.hc_attn_base  = place_f32(be, b("hc_attn_base.weight"), HC_MIX);
        L.hc_attn_scale = place_f32(be, b("hc_attn_scale.weight"), 3);
        L.hc_ffn_fn     = place_mat(be, b("hc_ffn_fn.weight"), 0);
        L.hc_ffn_base   = place_f32(be, b("hc_ffn_base.weight"), HC_MIX);
        L.hc_ffn_scale  = place_f32(be, b("hc_ffn_scale.weight"), 3);
        if (L.hc_attn_fn.K != HC_DIM || L.hc_attn_fn.rows != HC_MIX)
            throw std::runtime_error(b("hc_attn_fn.weight") + " is not [16384, 24]");

        L.attn_norm  = place_f32(be, b("attn_norm.weight"), N_EMBD);
        L.attn_sinks = place_f32(be, b("attn_sinks.weight"), N_HEAD);
        L.wq_a       = place_mat(be, b("attn_q_a.weight"), 0);
        L.q_a_norm   = place_f32(be, b("attn_q_a_norm.weight"), Q_LORA);
        L.wq_b       = place_mat(be, b("attn_q_b.weight"), 0);
        L.wkv        = place_mat(be, b("attn_kv.weight"), 0);
        L.kv_norm    = place_f32(be, b("attn_kv_a_norm.weight"), HEAD_DIM);
        {
            // [4096, 8192] in the file, read as 8 groups of 1024 rows: the
            // loader's TENSOR_ALLOW_RESHAPE to {4096, 1024, 8} (deepseek4.cpp).
            const Mat full = place_mat(be, b("attn_output_a.weight"), 0);
            if (full.K != O_GROUP_DIM || full.rows != (int64_t) O_GROUPS * O_LORA)
                throw std::runtime_error(b("attn_output_a.weight") + " is not [4096, 8192]");
            for (int g = 0; g < O_GROUPS; ++g) {
                L.wo_a[g] = full;
                L.wo_a[g].rows = O_LORA;
                L.wo_a[g].base = (const unsigned char *) full.base + (size_t) g * O_LORA * full.row_bytes;
            }
        }
        L.wo_b = place_mat(be, b("attn_output_b.weight"), 0);
        if (L.wq_a.rows != Q_LORA || L.wq_b.rows != (int64_t) N_HEAD * HEAD_DIM ||
            L.wkv.rows != HEAD_DIM || L.wo_b.K != (int64_t) O_GROUPS * O_LORA || L.wo_b.rows != N_EMBD)
            throw std::runtime_error("layer " + std::to_string(il) + ": attention shapes moved");

        if (L.ratio != 0) {
            const int coff = L.ratio == CSA_RATIO ? 2 : 1;
            L.comp_wkv   = place_mat(be, b("attn_compressor_kv.weight"), 0);
            L.comp_wgate = place_mat(be, b("attn_compressor_gate.weight"), 0);
            L.comp_ape   = place_f32(be, b("attn_compressor_ape.weight"), (int64_t) coff * HEAD_DIM * L.ratio);
            L.comp_norm  = place_f32(be, b("attn_compressor_norm.weight"), HEAD_DIM);
            if (L.comp_wkv.rows != coff * HEAD_DIM)
                throw std::runtime_error(b("attn_compressor_kv.weight") + ": width is not coff*512");
            if (L.ratio == CSA_RATIO) {
                L.idx_q_b        = place_mat(be, b("indexer.attn_q_b.weight"), 0);
                L.idx_proj       = place_mat(be, b("indexer.proj.weight"), 0);
                L.idx_comp_wkv   = place_mat(be, b("indexer_compressor_kv.weight"), 0);
                L.idx_comp_wgate = place_mat(be, b("indexer_compressor_gate.weight"), 0);
                L.idx_comp_ape   = place_f32(be, b("indexer_compressor_ape.weight"), 2 * IDX_DIM * CSA_RATIO);
                L.idx_comp_norm  = place_f32(be, b("indexer_compressor_norm.weight"), IDX_DIM);
                if (L.idx_q_b.rows != IDX_N_HEAD * IDX_DIM || L.idx_proj.rows != IDX_N_HEAD ||
                    L.idx_comp_wkv.rows != 2 * IDX_DIM)
                    throw std::runtime_error("layer " + std::to_string(il) + ": indexer shapes moved");
            }
        }

        L.ffn_norm = place_f32(be, b("ffn_norm.weight"), N_EMBD);
        L.gate_inp = place_mat(be, b("ffn_gate_inp.weight"), 0);
        if (il < HASH_LAYERS) {
            const TensorInfo * t = need(b("ffn_gate_tid2eid.weight"));
            if (t->type != GGML_TYPE_I32 || t->ne0() != N_EXPERT_USED || t->ne1() != N_VOCAB)
                throw std::runtime_error(b("ffn_gate_tid2eid.weight") + " is not I32 [6, 129280]");
            L.tid2eid = (const int32_t *) t->data;   // HOST: a per-token row gather
            if (devs_[0]->is_gpu()) {                 // ... loaded into RAM on a GPU run
                tid_copy_.emplace_back((const int32_t *) t->data,
                                       (const int32_t *) t->data + (size_t) N_EXPERT_USED * N_VOCAB);
                L.tid2eid = tid_copy_.back().data();
            }
        } else {
            L.exp_probs_b = place_f32(be, b("exp_probs_b.bias"), N_EXPERT);
        }
        L.exp_gate = describe_mat(b("ffn_gate_exps.weight"), N_EXPERT);
        L.exp_up   = describe_mat(b("ffn_up_exps.weight"),   N_EXPERT);
        L.exp_down = describe_mat(b("ffn_down_exps.weight"), N_EXPERT);
        L.sh_gate  = place_mat(be, b("ffn_gate_shexp.weight"), 0);
        L.sh_up    = place_mat(be, b("ffn_up_shexp.weight"),   0);
        L.sh_down  = place_mat(be, b("ffn_down_shexp.weight"), 0);
        if (L.exp_gate.K != N_EMBD || L.exp_gate.rows != N_FF_EXP || L.exp_down.K != N_FF_EXP ||
            L.exp_down.rows != N_EMBD || L.sh_up.rows != N_FF_EXP || L.sh_down.rows != N_EMBD ||
            L.gate_inp.rows != N_EXPERT)
            throw std::runtime_error("layer " + std::to_string(il) + ": MoE shapes moved");
        place_experts(L, be, *ops_[(size_t) L.dev]);
    }

    if (with_head) {
        Backend & be = *devs_.back();
        head_fn_     = place_mat(be, "output_hc_fn.weight", 0);
        head_base_   = place_f32(be, "output_hc_base.weight", HC);
        head_scale_  = place_f32(be, "output_hc_scale.weight", 1);
        output_norm_ = place_f32(be, "output_norm.weight", N_EMBD);
        output_      = place_mat(be, "output.weight", 0);
        if (head_fn_.K != HC_DIM || head_fn_.rows != HC || output_.rows != N_VOCAB)
            throw std::runtime_error("head shapes moved");
        have_head_ = true;
    }
}

void Ds4Model::embed_row(int32_t tok, float * out) const {
    if (tok < 0 || tok >= N_VOCAB) throw std::runtime_error("token id out of range");
    const size_t row = tok_embd_->row_size();
    const uint8_t * base = embd_copy_.empty() ? tok_embd_->data : (const uint8_t *) embd_copy_.data();
    ggml_get_type_traits(GGML_TYPE_Q5_K)->to_float(base + (size_t) tok * row, out, N_EMBD);
}

const int32_t * Ds4Model::hash_ids(int il, int32_t tok) const {
    const LayerWeights & L = layer(il);
    if (!L.tid2eid) return nullptr;
    return L.tid2eid + (size_t) tok * N_EXPERT_USED;
}

} // namespace ds4
} // namespace fk
