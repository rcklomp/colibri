// tools/hot-expert/franken/decode/glm5_model.h
//
// GLM-5.3-Flash as the decode loop needs it: every tensor of a layer span
// looked up by its name in the split GGUF (../gguf_model.h), classified into
// a Mat with its format and row stride, and PLACED on its layer range's
// backend -- for the CPU backend that is the mmap itself, so a `--cpu` run
// pages in only the rows and expert slabs it reads.
//
// Two layer kinds (attention.head_count_kv per layer, glm5_shapes.h is_dsa):
//   KDA  attn_q/k/v, ssm_conv1d_q/k/v, ssm_f_a/f_b, ssm_g_a/g_b, ssm_beta,
//        ssm_a, ssm_dt.bias, ssm_norm, attn_output
//   DSA  attn_q_a (+norm), attn_q_b, attn_kv_a_mqa (+norm), attn_k_b,
//        attn_v_b (per-head [K, rows, 64] slices), attn_output, and the
//        pooled indexer: indexer.attn_k, indexer.k_norm (LayerNorm, weight +
//        bias), indexer.proj (F32), indexer.attn_q_b,
//        indexer_compressor_gate, indexer_compressor_ape
// and two FFN kinds: layers 0-2 a dense clamped SwiGLU (ffn_gate/up/down,
// 12288 wide), the rest 288 routed experts (top-8, sigmoid gating with
// exp_probs_b) plus one shared expert.
//
// token_embd (Q8_0) is a per-token gather table and stays on the HOST
// (design 9.1's rule, record §PLE-GATHER).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../gguf_model.h"
#include "decode_backend.h"
#include "ds4_ops.h"          // ExpertTable
#include "glm5_shapes.h"

namespace fk {
namespace glm5 {

using franken::GgufModel;
using franken::TensorInfo;
using ds4::ExpertTable;

struct LayerWeights {
    int  il  = -1;
    bool dsa = false;          // DSA (MLA + indexer) or KDA
    bool moe = false;          // routed experts or the dense FFN
    int  dev = 0;

    // hyper-connections, both modules
    Mat hc_attn_fn;  const float * hc_attn_base = nullptr, * hc_attn_scale = nullptr;
    Mat hc_ffn_fn;   const float * hc_ffn_base  = nullptr, * hc_ffn_scale  = nullptr;
    const float * attn_norm = nullptr, * ffn_norm = nullptr;   // [4096]

    // KDA
    Mat wq, wk, wv;                                  // 4096 -> 8192 each
    const float * conv_q = nullptr, * conv_k = nullptr, * conv_v = nullptr;   // [8192][4]
    Mat f_a, f_b, g_a, g_b;                          // 4096 -> 128 -> 8192
    Mat beta;                                        // 4096 -> 64
    const float * ssm_a = nullptr;                   // [64], holds -exp(A_log)
    const float * dt_b  = nullptr;                   // [8192]
    const float * o_norm = nullptr;                  // [128]
    Mat wo;                                          // KDA 8192 -> 4096; DSA 16384 -> 4096

    // DSA (absorbed MLA)
    Mat wq_a;  const float * q_a_norm = nullptr;     // 4096 -> 1536
    Mat wq_b;                                        // 1536 -> 64*256
    Mat wkv_a; const float * kv_a_norm = nullptr;    // 4096 -> 512
    Mat wk_b[N_HEAD];                                // head h: 256 -> 512 (q_nope into the latent)
    Mat wv_b[N_HEAD];                                // head h: 512 -> 256 (the latent back out)
    // the pooled indexer
    Mat idx_k;   const float * idx_k_norm = nullptr, * idx_k_norm_b = nullptr;   // 4096 -> 128, LayerNorm
    Mat idx_gate;                                    // 4096 -> 128 (indexer_compressor_gate)
    const float * idx_ape = nullptr;                 // [4][128]
    Mat idx_q_b;                                     // 1536 -> 32*128
    Mat idx_proj;                                    // 4096 -> 32, F32

    // dense FFN (layers 0-2)
    Mat ffn_gate, ffn_up, ffn_down;                  // 4096 -> 12288 -> 4096

    // MoE
    Mat gate_inp;                                    // 4096 -> 288, F32
    const float * exp_probs_b = nullptr;             // [288]
    Mat exp_gate, exp_up, exp_down;                  // format + row stride only
    ExpertTable et;                                  // per-expert addresses, backend memory
    std::vector<const void *> tab_host;              // [3 * 288]
    std::vector<int> miss_host;                      // [288]: slab bytes if host-side, else 0
    Mat sh_gate, sh_up, sh_down;                     // the shared expert, 2048 wide
};

// Expert placement (GLM5.md section 5): per card, the experts of its layers
// with the highest usage counts in the M2 histogram (`<hist_dir>/layer_<il>.csv`,
// expert_id,count) until that card's budget is spent. `expert_gb` is one
// budget per card (a single value applies to every card).
struct Placement {
    std::string hist_dir;
    std::vector<double> expert_gb = {17.0};
    int ctx = 262144;                 // for the per-card VRAM projection
};

class Glm5Model {
public:
    // One backend per layer range, in order. `split` (optional) is the first
    // layer of each card after the first; empty = the span split evenly. The
    // head (hc_mean, output_norm, output.weight) goes on the last card.
    Glm5Model(const std::string & any_shard_path, std::vector<Backend *> devs, int il0, int il1,
              bool with_head, const std::vector<int> & split = {},
              const Placement & pl = Placement());

    const GgufModel & gguf() const { return *model_; }
    float rms_eps() const { return rms_eps_; }
    float ln_eps() const  { return ln_eps_; }
    float hc_eps() const  { return hc_eps_; }

    int il0() const { return il0_; }
    int il1() const { return il1_; }
    const LayerWeights & layer(int il) const { return layers_.at(il - il0_); }
    int n_devices() const { return (int) devs_.size(); }
    Backend & dev(int d) const { return *devs_.at(d); }
    Backend & dev_for(int il) const { return *devs_.at(layer(il).dev); }

    bool have_head() const { return have_head_; }
    const float * output_norm() const { return output_norm_; }
    const Mat & output() const { return output_; }

    void embed_row(int32_t tok, float * out) const;   // token_embd, Q8_0 -> f32 (host)

    size_t placed_bytes() const { return placed_; }
    bool   planned() const { return planned_; }
    bool   resident(int il, int e) const { return resident_[(size_t) il][(size_t) e] != 0; }

private:
    const TensorInfo * need(const std::string & name) const;
    Mat place_mat(Backend & be, const std::string & name, size_t n_slices);
    Mat describe_mat(const std::string & name, size_t n_slices);
    const float * place_f32(Backend & be, const std::string & name, int64_t n_expect);
    void check_hparams();
    int  dev_of_layer(int il) const;
    size_t trunk_bytes(int il) const;
    void plan_placement(const Placement & pl);
    void place_experts(LayerWeights & L, Backend & be);

    std::unique_ptr<GgufModel> model_;
    std::vector<Backend *> devs_;
    std::vector<int> split_;
    std::vector<std::vector<char>> resident_;
    std::vector<unsigned char> embd_copy_;
    bool planned_ = false;
    std::vector<LayerWeights> layers_;
    const TensorInfo * tok_embd_ = nullptr;
    bool have_head_ = false;
    Mat output_;
    const float * output_norm_ = nullptr;
    int il0_ = 0, il1_ = 0;
    float rms_eps_ = RMS_EPS_DEFAULT, ln_eps_ = LN_EPS_DEFAULT, hc_eps_ = HC_EPS_DEFAULT;
    size_t placed_ = 0;
};

} // namespace glm5
} // namespace fk
