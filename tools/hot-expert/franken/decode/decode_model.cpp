// tools/hot-expert/franken/decode/decode_model.cpp -- see decode_model.h.

#include "decode_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) fk_half_bits_to_f32(bits)
static inline float fk_half_bits_to_f32(unsigned int bits);
#include "decode_quant.h"

#include "ggml.h"

// ggml_fp16_to_fp32 is in libggml-base; used only here and in decode_cpu.cpp.
static inline float fk_half_bits_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    std::memcpy(&h, &bits, sizeof(uint16_t));
    return ggml_fp16_to_fp32(h);
}

namespace fk {

int fk_type_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:    return FK_Q_F32;
        case GGML_TYPE_BF16:   return FK_Q_BF16;
        case GGML_TYPE_Q8_0:   return FK_Q_Q8_0;
        case GGML_TYPE_IQ4_NL: return FK_Q_IQ4_NL;
        case GGML_TYPE_IQ4_XS: return FK_Q_IQ4_XS;
        case GGML_TYPE_IQ3_S:  return FK_Q_IQ3_S;
        case GGML_TYPE_Q6_K:   return FK_Q_Q6_K;
        default:
            throw std::runtime_error(std::string("unsupported tensor type for L0 step 2: ") +
                                     ggml_type_name(t));
    }
}

const TensorInfo * DecodeModel::need(const std::string & name) const {
    const TensorInfo * t = model_->find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

// Places a weight matrix and asserts the row stride the kernels will use
// against the tensor's own byte size. A silent disagreement here is the
// classic way a quantised GEMV produces plausible garbage.
Mat DecodeModel::place_mat(Backend & be, const std::string & name, size_t n_slices) {
    const TensorInfo * t = need(name);

    Mat m;
    m.name          = strdup(name.c_str());   // lives as long as the process
    m.type          = fk_type_of(t->type);
    m.K             = t->ne0();
    m.rows          = t->ne1();
    m.row_bytes     = fk_row_bytes(m.type, m.K);
    m.expert_stride = n_slices > 1 ? t->slice_bytes((int64_t) n_slices) : 0;

    const size_t expect = m.row_bytes * (size_t) m.rows * (n_slices ? n_slices : 1);
    if (expect != t->nbytes) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "%s: row/slice arithmetic disagrees with the file -- "
            "ne=[%lld,%lld,%lld] type=%s row_bytes=%zu slices=%zu => %zu, file says %zu",
            name.c_str(), (long long) t->ne0(), (long long) t->ne1(), (long long) t->ne2(),
            ggml_type_name(t->type), m.row_bytes, n_slices, expect, t->nbytes);
        throw std::runtime_error(buf);
    }
    if (m.row_bytes != t->row_size()) {
        throw std::runtime_error(name + ": row_bytes disagrees with ggml_row_size");
    }
    if (n_slices > 1 && m.expert_stride != m.row_bytes * (size_t) m.rows) {
        throw std::runtime_error(name + ": expert slice stride is not rows*row_bytes");
    }

    m.base = be.place(t->data, t->nbytes, m.name);
    placed_ += t->nbytes;
    return m;
}

const float * DecodeModel::place_f32(Backend & be, const std::string & name, int64_t n_expect) {
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

DecodeModel::DecodeModel(const std::string & path, std::vector<Backend *> devs,
                         int il0, int il1, bool with_head)
    : devs_(std::move(devs)), il0_(il0), il1_(il1) {
    if (devs_.empty()) throw std::runtime_error("DecodeModel needs at least one backend");
    model_ = GgufModel::open(path);
    const HParams & h = model_->hparams();

    // --- the file must be the model these shapes were written for ---------
    auto want = [&](const char * what, long long got, long long exp) {
        if (got != exp) {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "hparam %s: file says %lld, decode_shapes.h says %lld",
                          what, got, exp);
            throw std::runtime_error(buf);
        }
    };
    if (h.arch != "qwen4exp") throw std::runtime_error("arch is " + h.arch + ", not qwen4exp");
    want("block_count",        h.block_count,        N_LAYER);
    want("embedding_length",   h.embedding_length,   N_EMBD);
    want("head_count",         h.head_count,         N_Q_HEADS);
    want("head_count_kv",      h.head_count_kv,      N_KV_HEADS);
    want("expert_count",       h.expert_count,       N_EXPERT);
    want("expert_used_count",  h.expert_used_count,  N_EXPERT_USED);
    want("expert_ff_len",      h.expert_ff_len,      N_FF_EXP);
    want("shared_ff_len",      h.expert_shared_ff_len, N_FF_SHEXP);
    want("hc_count",           h.hc_count,           HC);
    want("hc_low_rank",        h.hc_low_rank,        HC_LR);
    want("ssm_conv_kernel",    h.ssm_conv_kernel,    GDN_CONV_K);
    want("ssm_state_size",     h.ssm_state_size,     GDN_STATE);
    want("ssm_group_count",    h.ssm_group_count,    GDN_K_HEADS);
    want("ssm_time_step_rank", h.ssm_time_step_rank, GDN_V_HEADS);
    want("ssm_inner_size",     h.ssm_inner_size,     GDN_INNER);
    want("indexer_head_count", h.indexer_head_count, IDX_N_HEADS);
    want("indexer_key_length", h.indexer_key_length, IDX_DIM);
    want("indexer_top_k",      h.indexer_top_k,      IDX_TOP_K);
    want("ple_ngram_size",     h.ple_ngram_size,     PLE_NGRAM);
    want("ple_heads_per_ngram",h.ple_heads_per_ngram,PLE_HEADS_PER_NG);
    want("ple_conv_kernel",    h.ple_conv_kernel,    PLE_CONV_K);
    want("n_embd_per_layer",   h.n_embd_per_layer,   PLE_HEAD_DIM);
    if (h.ple_layers.size() != 1 || (int) h.ple_layers[0] != PLE_LAYER) {
        throw std::runtime_error("this file's PLE layer is not layer 1");
    }
    rms_eps_ = h.rms_eps;

    if (il0 < 0 || il1 >= (int) h.block_count || il0 > il1) {
        throw std::runtime_error("layer range out of the model");
    }

    tok_embd_ = need("token_embd.weight");
    if (tok_embd_->type != GGML_TYPE_Q8_0 || tok_embd_->ne0() != N_EMBD) {
        throw std::runtime_error("token_embd.weight is not [2560, n_vocab] Q8_0");
    }

    // Even split of the span across the devices: 48 layers over 3 cards is
    // design 9.1's 0-15 / 16-31 / 32-47. The design calls the balance
    // PROJECTED and says L0 measures and rebalances it, so this is the
    // starting point, not a fixed law.
    const int n_span = il1 - il0 + 1;
    const int n_dev  = (int) devs_.size();
    const int per_dev = (n_span + n_dev - 1) / n_dev;

    layers_.resize(n_span);
    for (int il = il0; il <= il1; ++il) {
        LayerWeights & L = layers_[il - il0];
        const std::string p = "blk." + std::to_string(il) + ".";
        L.dev = std::min(n_dev - 1, (il - il0) / per_dev);
        Backend & be = *devs_[L.dev];

        L.il        = il;
        L.recurrent = is_recurrent_layer(il);
        L.is_ple    = h.is_ple_layer(il);
        // qwen4exp.cpp:718 -- QSA runs only where compress_ratios[il] > 0.
        L.compress_ratio = (il < (int) h.compress_ratios.size()) ? (int) h.compress_ratios[il] : 0;
        if (!L.recurrent && L.compress_ratio != QSA_RATIO) {
            // The record (§M3) had to establish this the hard way: a build
            // agent claimed the key was absent from the file. It is present.
            throw std::runtime_error(p + "compress_ratio is not 4 on a full-attention layer");
        }

        L.hc_attn_norm   = place_f32(be, p + "hc_attn_norm.weight", HC_DIM);
        L.hc_attn_down   = place_mat(be, p + "hc_attn_down.weight", 1);
        L.hc_attn_up     = place_mat(be, p + "hc_attn_up.weight",   1);
        L.hc_attn_inject = place_mat(be, p + "hc_attn_inject.weight", 1);
        L.hc_ffn_norm    = place_f32(be, p + "hc_ffn_norm.weight",  HC_DIM);
        L.hc_ffn_down    = place_mat(be, p + "hc_ffn_down.weight",  1);
        L.hc_ffn_up      = place_mat(be, p + "hc_ffn_up.weight",    1);
        L.hc_ffn_inject  = place_mat(be, p + "hc_ffn_inject.weight", 1);

        if (L.recurrent) {
            L.ssm_qkv    = place_mat(be, p + "attn_qkv.weight",   1);
            L.ssm_gate   = place_mat(be, p + "attn_gate.weight",  1);
            L.ssm_out    = place_mat(be, p + "ssm_out.weight",    1);
            L.ssm_alpha  = place_mat(be, p + "ssm_alpha.weight",  1);
            L.ssm_beta   = place_mat(be, p + "ssm_beta.weight",   1);
            L.ssm_conv1d = place_f32(be, p + "ssm_conv1d.weight", (int64_t) GDN_CONV_K * GDN_CONV_DIM);
            L.ssm_dt     = place_f32(be, p + "ssm_dt.bias",       GDN_V_HEADS);
            L.ssm_a      = place_f32(be, p + "ssm_a",             GDN_V_HEADS);
            L.ssm_norm   = place_f32(be, p + "ssm_norm.weight",   GDN_STATE);
            if (L.ssm_qkv.rows != GDN_CONV_DIM || L.ssm_gate.rows != GDN_VAL_DIM) {
                throw std::runtime_error(p + "GDN projection widths are not the expected ones");
            }
        } else {
            L.wq          = place_mat(be, p + "attn_q.weight",      1);
            L.wk          = place_mat(be, p + "attn_k.weight",      1);
            L.wv          = place_mat(be, p + "attn_v.weight",      1);
            L.wo          = place_mat(be, p + "attn_output.weight", 1);
            L.attn_q_norm = place_f32(be, p + "attn_q_norm.weight", HEAD_DIM);
            L.attn_k_norm = place_f32(be, p + "attn_k_norm.weight", HEAD_DIM);
            L.idx_q       = place_mat(be, p + "indexer.q_proj.weight", 1);
            L.idx_k       = place_mat(be, p + "indexer.k_proj.weight", 1);
            L.idx_q_norm  = place_f32(be, p + "indexer.q_norm.weight", IDX_DIM);
            L.idx_k_norm  = place_f32(be, p + "indexer.k_norm.weight", IDX_DIM);
            // wq carries [q|gate] interleaved per head (qwen4exp.cpp:169)
            if (L.wq.rows != (int64_t) HEAD_DIM * N_Q_HEADS * 2) {
                throw std::runtime_error(p + "attn_q.weight is not [q|gate] interleaved");
            }
        }

        L.ffn_gate_inp = place_mat(be, p + "ffn_gate_inp.weight", 1);
        L.exp_gate     = place_mat(be, p + "ffn_gate_exps.weight", N_EXPERT);
        L.exp_up       = place_mat(be, p + "ffn_up_exps.weight",   N_EXPERT);
        L.exp_down     = place_mat(be, p + "ffn_down_exps.weight", N_EXPERT);
        L.sh_gate      = place_mat(be, p + "ffn_gate_shexp.weight", 1);
        L.sh_up        = place_mat(be, p + "ffn_up_shexp.weight",   1);
        L.sh_down      = place_mat(be, p + "ffn_down_shexp.weight", 1);
        L.sh_gate_inp  = place_mat(be, p + "ffn_gate_inp_shexp.weight", 1);

        // The expert kernels are written for exactly these four combinations;
        // this UD quant uses IQ4_XS gate/up on layer 2 and Q8_0 down on 2 and
        // 4, so "the file is IQ3_S/IQ4_NL" is NOT true layer by layer.
        if (L.exp_gate.type != FK_Q_IQ3_S && L.exp_gate.type != FK_Q_IQ4_XS)
            throw std::runtime_error(p + "ffn_gate_exps: unsupported expert format");
        if (L.exp_up.type != L.exp_gate.type)
            throw std::runtime_error(p + "gate and up experts are in different formats");
        if (L.exp_down.type != FK_Q_IQ4_NL && L.exp_down.type != FK_Q_Q8_0)
            throw std::runtime_error(p + "ffn_down_exps: unsupported expert format");

        if (L.is_ple) {
            L.ple_key        = place_mat(be, p + "ple_key.weight",   1);
            L.ple_value      = place_mat(be, p + "ple_value.weight", 1);
            L.ple_norm_key   = place_f32(be, p + "ple_norm_key.weight",   HC_DIM);
            L.ple_norm_query = place_f32(be, p + "ple_norm_query.weight", HC_DIM);
            L.ple_norm_conv  = place_f32(be, p + "ple_norm_conv.weight",  HC_DIM);
            L.ple_conv1d     = place_f32(be, p + "ple_conv1d.weight", (int64_t) PLE_CONV_K * HC_DIM);
        }
    }

    // The head lives with the last layer range. There is no separate output
    // norm: build_hc_mix on output_hc_* IS it (qwen4exp.cpp:380-386), and it
    // takes no inject, so only down/up are needed.
    if (with_head) {
        Backend & be = *devs_.back();
        head_norm_ = place_f32(be, "output_hc_norm.weight", HC_DIM);
        head_down_ = place_mat(be, "output_hc_down.weight", 1);
        head_up_   = place_mat(be, "output_hc_up.weight",   1);
        lm_head_   = place_mat(be, "output.weight",         1);
        // n_vocab comes from the tokenizer array, which this loader does not
        // parse, so token_embd's own row count is the reference.
        if (lm_head_.K != N_EMBD || lm_head_.rows != tok_embd_->ne1())
            throw std::runtime_error("output.weight is not [n_embd, n_vocab]");
        have_head_ = true;
    }
}

// One Q8_0 row of token_embd, on the host: design 9.1's rule puts every
// per-token row gather in host RAM. Decoded with the same primitive the
// kernels use (decode_quant.h), lane by lane.
void DecodeModel::embed_row(int32_t tok, float * out) const {
    if (tok < 0 || tok >= tok_embd_->ne1()) throw std::runtime_error("token id out of vocabulary");
    const unsigned char * row = tok_embd_->data + (size_t) tok * tok_embd_->row_size();
    const int nblk = N_EMBD / FK_Q8_0_BLOCK_WEIGHTS;
    for (int b = 0; b < nblk; ++b) {
        const unsigned char * bp = row + (size_t) b * FK_Q8_0_BLOCK_BYTES;
        const float d  = fk_q8_0_block_d(bp);
        const signed char * qs = (const signed char *)(bp + FK_Q8_0_OFF_QS);
        for (int i = 0; i < FK_Q8_0_BLOCK_WEIGHTS; ++i) {
            out[b * FK_Q8_0_BLOCK_WEIGHTS + i] = d * (float) qs[i];
        }
    }
}

} // namespace fk
