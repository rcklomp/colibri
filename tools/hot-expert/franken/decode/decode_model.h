// tools/hot-expert/franken/decode/decode_model.h
//
// Layers 0-15 of Qwen3.8-Flash-Next as the decode loop needs them: every
// tensor of a layer looked up by name in the split GGUF (../gguf_model.h,
// L0 step 1), classified into a Mat with its format and row stride, and
// PLACED on the backend (VRAM for the GPU backend; the mmap itself for the
// CPU one, so a `--cpu` run pages in only the rows it reads).
//
// Scope, from L0-STEP2-BRIEF-2026-09-22.md: one card, layers 0-15, no
// lm_head, no sampling. `output.weight` (Q6_K, 521 MB) and the final
// hyper-connection mixer are deliberately NOT loaded -- step 3's business.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../gguf_model.h"
#include "decode_backend.h"

namespace fk {

using franken::GgufModel;
using franken::HParams;
using franken::TensorInfo;

struct LayerWeights {
    int  il          = -1;
    bool recurrent   = false;   // Gated DeltaNet; otherwise QSA full attention
    bool is_ple      = false;

    // hyper-connections, both modules
    Mat hc_attn_down, hc_attn_up, hc_attn_inject;  const float * hc_attn_norm = nullptr;
    Mat hc_ffn_down,  hc_ffn_up,  hc_ffn_inject;   const float * hc_ffn_norm  = nullptr;

    // Gated DeltaNet
    Mat ssm_qkv, ssm_gate, ssm_out, ssm_alpha, ssm_beta;
    const float * ssm_conv1d = nullptr;   // [GDN_CONV_DIM][4] (ne = [4, 10240])
    const float * ssm_dt     = nullptr;   // [48]
    const float * ssm_a      = nullptr;   // [48], already -exp(A_log)
    const float * ssm_norm   = nullptr;   // [128]

    // QSA full attention
    Mat wq, wk, wv, wo;
    const float * attn_q_norm = nullptr;  // [256]
    const float * attn_k_norm = nullptr;  // [256]
    Mat idx_q, idx_k;
    const float * idx_q_norm = nullptr;   // [128]
    const float * idx_k_norm = nullptr;   // [128]
    int compress_ratio = 0;

    // MoE
    Mat ffn_gate_inp;                      // [2560 -> 512] F32 router
    Mat exp_gate, exp_up, exp_down;        // 512-expert tensors
    Mat sh_gate, sh_up, sh_down;
    Mat sh_gate_inp;                       // [2560 -> 1]

    // PLE (layer 1 only)
    Mat ple_key, ple_value;
    const float * ple_norm_key   = nullptr;
    const float * ple_norm_query = nullptr;
    const float * ple_norm_conv  = nullptr;
    const float * ple_conv1d     = nullptr;   // [HC_DIM][4] (ne = [4, 10240])
};

class DecodeModel {
public:
    // Opens the split, checks its hparams against decode_shapes.h, and places
    // layers [il0, il1] plus token_embd's metadata on `be`.
    DecodeModel(const std::string & any_shard_path, Backend & be, int il0, int il1);

    const GgufModel & gguf() const { return *model_; }
    const HParams &   hp()   const { return model_->hparams(); }
    float             rms_eps() const { return rms_eps_; }

    int il0() const { return il0_; }
    int il1() const { return il1_; }
    const LayerWeights & layer(int il) const { return layers_.at(il - il0_); }

    // token_embd stays on the HOST: it is a per-token row gather, and design
    // 9.1's rule ("any tensor read by a per-token row gather lives in host
    // RAM") plus record §PLE-GATHER put it there. Dequantises row `tok` of
    // the Q8_0 table into out[N_EMBD].
    void embed_row(int32_t tok, float * out) const;

    size_t placed_bytes() const { return placed_; }

private:
    const TensorInfo * need(const std::string & name) const;
    Mat  place_mat(Backend & be, const std::string & name, size_t n_slices);
    const float * place_f32(Backend & be, const std::string & name, int64_t n_expect);

    std::unique_ptr<GgufModel> model_;
    std::vector<LayerWeights>  layers_;
    const TensorInfo *         tok_embd_ = nullptr;
    int    il0_ = 0, il1_ = 0;
    float  rms_eps_ = RMS_EPS_DEFAULT;
    size_t placed_ = 0;
};

int fk_type_of(ggml_type t);   // ggml_type -> FkQuantType, throws on anything else

} // namespace fk
