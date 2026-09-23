// tools/hot-expert/franken/decode/ds4_model.h
//
// DeepSeek-V4-Flash as the decode loop needs it: every tensor of a layer
// span looked up by its name in the split GGUF (../gguf_model.h), classified
// into a Mat with its format and row stride, and PLACED on its layer range's
// backend -- for the CPU backend that is the mmap itself, so a `--cpu` run
// pages in only the rows and the expert slabs it reads.
//
// The per-token gather tables stay on the HOST, by design 9.1's rule (record
// §PLE-GATHER): token_embd (Q5_K, one row a token) and the three hash
// layers' ffn_gate_tid2eid (I32 [6, 129280], one row a token a layer).
//
// The tensor names are the FILE's (they differ from llama-arch.cpp's enum
// spellings in places: attn_kv_a_norm, attn_output_a/b, attn_compressor_*,
// indexer.proj, indexer_compressor_*, exp_probs_b.bias).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../gguf_model.h"
#include "decode_backend.h"
#include "ds4_ops.h"
#include "ds4_shapes.h"

namespace fk {
namespace ds4 {

using franken::GgufModel;
using franken::TensorInfo;

struct LayerWeights {
    int il    = -1;
    int ratio = 0;       // compress ratio: 0 raw, 4 CSA, 128 HCA
    int dev   = 0;

    // hyper-connections, both modules
    Mat hc_attn_fn;  const float * hc_attn_base = nullptr, * hc_attn_scale = nullptr;
    Mat hc_ffn_fn;   const float * hc_ffn_base  = nullptr, * hc_ffn_scale  = nullptr;

    // attention
    const float * attn_norm = nullptr;     // [4096]
    const float * attn_sinks = nullptr;    // [64]
    Mat wq_a;                              // 4096 -> 1024 (Q5_K; Q6_K on one layer)
    const float * q_a_norm = nullptr;      // [1024]
    Mat wq_b;                              // 1024 -> 32768
    Mat wkv;                               // 4096 -> 512
    const float * kv_norm = nullptr;       // [512]
    Mat wo_a[O_GROUPS];                    // group g: 4096 -> 1024 (rows g*1024.. of [4096 x 8192])
    Mat wo_b;                              // 8192 -> 4096

    // the compressor (ratio != 0)
    Mat comp_wkv, comp_wgate;              // 4096 -> coff*512 (coff 2 for CSA, 1 for HCA)
    const float * comp_ape  = nullptr;     // [ratio][coff*512]
    const float * comp_norm = nullptr;     // [512]
    // the lightning indexer (ratio == 4)
    Mat idx_q_b;                           // 1024 -> 64*128
    Mat idx_proj;                          // 4096 -> 64, F32
    Mat idx_comp_wkv, idx_comp_wgate;      // 4096 -> 256
    const float * idx_comp_ape  = nullptr; // [4][256]
    const float * idx_comp_norm = nullptr; // [128]

    // MoE
    const float * ffn_norm = nullptr;      // [4096]
    Mat gate_inp;                          // 4096 -> 256, BF16
    const float * exp_probs_b = nullptr;   // [256], non-hash layers
    const int32_t * tid2eid = nullptr;     // HOST [n_vocab][6], hash layers
    Mat exp_gate, exp_up, exp_down;        // 256-expert tensors: format + row stride only
                                           // (base is null: the TABLE addresses the experts)
    ExpertTable et;                        // per-expert addresses, backend memory
    std::vector<const void *> tab_host;    // the CPU arm's table (mmap addresses)
    std::vector<int> miss_host;            // per expert: slab bytes if host-mapped, else 0
    int n_resident = N_EXPERT;             // experts in VRAM (all, on the CPU arm)
    Mat sh_gate, sh_up, sh_down;           // the shared expert, 2048 wide
};

// Expert placement (L5 step 2, DEEPSEEK4.md section 5): per card, the
// experts of its layers with the highest usage counts in the M2 histogram
// (`<hist_dir>/layer_<il>.csv`, expert_id,count) until `expert_gb` of VRAM is
// spent; the rest go to pinned host memory, read through the table.
struct Placement {
    std::string hist_dir;
    double      expert_gb = 20.0;
};

class Ds4Model {
public:
    // One backend per layer range, in order; the span [il0, il1] is split
    // evenly across them (the Qwen3.8 rule, decode_model.h). The head (the
    // final hyper-connection mixer, output_norm, output.weight) goes on the
    // last one when `with_head`.
    Ds4Model(const std::string & any_shard_path, std::vector<Backend *> devs,
             std::vector<Ds4Ops *> ops, int il0, int il1, bool with_head,
             const Placement & pl = Placement());

    const GgufModel & gguf() const { return *model_; }
    float rms_eps() const { return rms_eps_; }
    float hc_eps() const  { return hc_eps_; }

    int il0() const { return il0_; }
    int il1() const { return il1_; }
    const LayerWeights & layer(int il) const { return layers_.at(il - il0_); }
    int n_devices() const { return (int) devs_.size(); }
    Backend & dev(int d) const { return *devs_.at(d); }
    Backend & dev_for(int il) const { return *devs_.at(layer(il).dev); }

    bool have_head() const { return have_head_; }
    const Mat & head_fn() const { return head_fn_; }
    const float * head_base() const { return head_base_; }
    const float * head_scale() const { return head_scale_; }
    const float * output_norm() const { return output_norm_; }
    const Mat & output() const { return output_; }

    // Host-side row gathers.
    void embed_row(int32_t tok, float * out) const;           // token_embd, Q5_K -> f32
    const int32_t * hash_ids(int il, int32_t tok) const;      // tid2eid row, or nullptr

    size_t placed_bytes() const { return placed_; }
    size_t host_expert_bytes() const { return host_expert_bytes_; }

private:
    const TensorInfo * need(const std::string & name) const;
    Mat place_mat(Backend & be, const std::string & name, size_t n_slices);
    Mat describe_mat(const std::string & name, size_t n_slices);
    const float * place_f32(Backend & be, const std::string & name, int64_t n_expect);
    void check_hparams();
    void plan_placement(const Placement & pl);
    void place_experts(LayerWeights & L, Backend & be, Ds4Ops & ops);

    std::unique_ptr<GgufModel> model_;
    std::vector<Backend *> devs_;
    std::vector<Ds4Ops *>  ops_;
    std::vector<std::vector<char>> resident_;   // [layer][expert], from plan_placement
    std::vector<unsigned char> embd_copy_;       // token_embd in host RAM (GPU runs)
    std::vector<std::vector<int32_t>> tid_copy_; // the hash layers' tid2eid, likewise
    size_t host_expert_bytes_ = 0;
    std::vector<LayerWeights> layers_;
    const TensorInfo * tok_embd_ = nullptr;
    bool have_head_ = false;
    Mat head_fn_, output_;
    const float * head_base_ = nullptr, * head_scale_ = nullptr, * output_norm_ = nullptr;
    int il0_ = 0, il1_ = 0;
    float rms_eps_ = RMS_EPS_DEFAULT, hc_eps_ = RMS_EPS_DEFAULT;
    size_t placed_ = 0;
};

} // namespace ds4
} // namespace fk
