// tools/hot-expert/franken/gguf_model.h
//
// L0 step 1's loader (FRANKEN-ENGINE-DESIGN-2026-09-22.md, section 9, the L0
// specification, and its build order 9.5 step 1: "Loader: GGUF tensors ->
// device buffers per card by layer range"). Opens the split GGUF the way
// llama-model-loader.cpp does -- gguf_init_from_file(no_alloc=true) on each
// shard with no ggml_context at all (params.ctx == nullptr; confirmed in
// ggml/src/gguf.cpp:798 that this skips tensor-data allocation entirely, so
// this is pure header/metadata parsing, no GPU and no full-file read), then
// mmap() each shard's data section. Every tensor is exposed by name with
// its shape, ggml type, byte size and a pointer into the mapping; the
// hparams the design needs are read directly off the gguf KV pairs -- no
// llama_model_loader / no libllama linkage, same footprint as
// tools/hot-expert/m1/m1_ggml.cpp (ggml only).
//
// Split enumeration follows llama.cpp's own convention
// (llama_split_path/llama_split_prefix, src/llama.cpp:543-595):
// "<prefix>-%05d-of-%05d.gguf", split_no 1-based in the filename. Per
// llama-model-loader.cpp:591-620, shard 0 (split.no == 0) is the one that
// must carry every KV pair, including split.count and split.tensors.count;
// shards 1..N-1 carry only their own split.no plus their own tensor list.
// Verified on the rig against the served split, 2026-09-22
// (~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-000NN-of-00003.gguf):
//   shard 00001: n_tensors=0  n_kv=67  (all metadata, no tensor data at all)
//   shard 00002: n_tensors=373 n_kv=3
//   shard 00003: n_tensors=851 n_kv=3
//   373 + 851 == split.tensors.count == 1224, and 0 + 373 + 851 == 1224.
// A caller may pass ANY shard's path; open() reads split.no/split.count off
// that file, derives the prefix, and opens shard 0 itself for the metadata.
//
// Nothing here touches a GPU: gguf_init_from_file(no_alloc=true, ctx=nullptr)
// is CPU-only header parsing, and mmap() only maps virtual address space --
// no page is read until something dereferences it. Never mmap-and-touch the
// whole model; see ple.cpp for the one thing in this directory that reads
// tensor bytes, and it reads only the rows a token's gather selects.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace franken {

constexpr int MAX_PLE_NGRAM = 8;   // mirrors llama-hparams.h LLAMA_MAX_PLE_NGRAM (qwen4exp)
constexpr int MAX_PLE_HEADS = 64;  // mirrors llama-hparams.h LLAMA_MAX_PLE_HEADS (qwen4exp)

// One GGUF tensor, wherever it physically lives.
struct TensorInfo {
    std::string name;
    int32_t     n_dims = 0;
    int64_t     ne[GGML_MAX_DIMS] = {1, 1, 1, 1}; // ggml order: ne[0] is the row (fastest-varying)
    ggml_type   type = GGML_TYPE_F32;
    size_t      nbytes = 0;        // gguf_get_tensor_size(): the tensor's total byte size in the file
    int         shard = -1;        // index into GgufModel::shard_path()
    size_t      file_offset = 0;   // absolute byte offset of the tensor's data within that shard's file
    const uint8_t * data = nullptr; // == shard mmap base + file_offset, filled in once every shard is mapped

    int64_t ne0() const { return ne[0]; }
    int64_t ne1() const { return ne[1]; }
    int64_t ne2() const { return ne[2]; }
    int64_t ne3() const { return ne[3]; }

    // ggml_row_size(type, ne0): bytes for one row of ne[0] elements, block-aligned
    // for a quantised type (e.g. IQ4_NL's 32-value blocks -- see ple.h).
    size_t row_size() const;

    // For a tensor whose slowest dimension (ne[ndims-1]) enumerates a fixed-size
    // sub-tensor per index (routed experts: ne2 = expert count), the byte size of
    // one such slice. Equivalent to nbytes / ne[last], which is robust to the
    // exact row/col layout of a quantised type (placement.cpp's expert pointer
    // table uses this, not ne0/ne1 arithmetic).
    size_t slice_bytes(int64_t n_slices) const { return n_slices > 0 ? nbytes / (size_t) n_slices : nbytes; }
};

// Only the fields the L0 design (design doc section 9) reads. Every key is
// looked up as "<arch>.<suffix>" per ggml's own convention (arch comes from
// general.architecture, read first). A key absent from the file leaves its
// field at the zero-init default; callers that require it should check.
struct HParams {
    std::string arch;

    uint32_t block_count             = 0;
    uint32_t embedding_length        = 0;
    uint32_t expert_count            = 0;
    uint32_t expert_used_count       = 0;
    uint32_t head_count              = 0;
    // attention.head_count_kv: llama.cpp convention is one scalar for the
    // whole model, but a hybrid model that mixes GDN and QSA/full-attention
    // layers with different KV head counts per layer may write it as a
    // GGUF ARRAY instead (observed 2026-09-23 on the real GLM-5.3-Flash
    // UD-IQ4_XS split: "gguf key 'glm5next.attention.head_count_kv' has an
    // unexpected scalar type" -- it is GGUF_TYPE_ARRAY, one u32 per block).
    // head_count_kv stays a plain scalar for every existing caller: when
    // the file wrote an array, gguf_model.cpp sets it to that array's MAX
    // (a safe upper bound for anything still sizing a single buffer from
    // it); when the file wrote a scalar (or nothing), it is that value
    // exactly, unchanged from before this field grew an array sibling.
    // head_count_kv_layers is empty in the plain-scalar case and holds one
    // entry per block when the file carried the array -- use
    // head_count_kv_at(il) to get the layer-accurate value either way.
    uint32_t head_count_kv           = 0;
    std::vector<uint32_t> head_count_kv_layers;
    uint32_t full_attention_interval = 4;
    uint32_t expert_ff_len           = 0;
    uint32_t expert_shared_ff_len    = 0;
    float    rms_eps                 = 0.0f;
    uint32_t vocab_size              = 0;
    uint32_t context_length          = 0;

    // Gated DeltaNet (SSM-shaped) state, present on every layer's tensors
    // whether or not that layer is a GDN layer (design 9.3's "GDN layers").
    uint32_t ssm_conv_kernel    = 0;
    uint32_t ssm_inner_size     = 0;
    uint32_t ssm_state_size     = 0;
    uint32_t ssm_time_step_rank = 0;
    uint32_t ssm_group_count    = 0;

    // Hyper-connections (design 9.3's "hc_*"): hc parallel residual streams.
    uint32_t hc_count    = 0;
    uint32_t hc_low_rank = 0;

    // QSA indexer (design 9.3's "the indexer q/k projections").
    uint32_t indexer_head_count = 0;
    uint32_t indexer_key_length = 0;
    uint32_t indexer_top_k      = 0;

    std::vector<uint32_t> compress_ratios; // length block_count; nonzero marks a full-attention (QSA) layer

    // PLE per-layer n-gram hash embedding (design 9.1's host-RAM row-gather rule; record §PLE-GATHER).
    std::vector<uint32_t> ple_layers;      // qwen4exp carries exactly one entry (this model: [1])
    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_conv_kernel     = 0;
    uint32_t ple_eos_token_id    = 0;
    uint32_t ple_image_token_id  = 0;      // 0 if the file predates this key (falls back to eos_token_id)
    uint32_t n_embd_per_layer    = 0;      // PLE row width (this model: 160)
    std::array<uint64_t, MAX_PLE_NGRAM> ple_layer_multipliers{};
    std::array<uint32_t, MAX_PLE_HEADS> ple_head_offsets{};
    std::array<uint32_t, MAX_PLE_HEADS> ple_head_vocab_sizes{};

    // The layer-accurate KV head count: head_count_kv_layers[il] when the
    // file carried a per-layer array and il is in range, else the scalar
    // head_count_kv (covers both "the file only ever had one scalar" and
    // "il is out of the array's range").
    uint32_t head_count_kv_at(int il) const {
        if (il >= 0 && (size_t) il < head_count_kv_layers.size()) return head_count_kv_layers[(size_t) il];
        return head_count_kv;
    }

    uint32_t ple_n_heads() const {
        return ple_ngram_size >= 2 ? (ple_ngram_size - 1) * ple_heads_per_ngram : 0;
    }
    bool is_ple_layer(int il) const {
        for (uint32_t l : ple_layers) if ((int) l == il) return true;
        return false;
    }
    // qwen4exp's default rule when the file carries no explicit recurrent_layers
    // array (this model doesn't): every layer is GDN except every
    // full_attention_interval-th, which is QSA full attention
    // (src/models/qwen4exp.cpp:93-100).
    bool is_recurrent_layer(int il) const {
        return (uint32_t) il < block_count && (((uint32_t) il + 1) % full_attention_interval != 0);
    }
};

class GgufModel {
public:
    // Opens the split whose member `any_shard_path` names -- any shard works,
    // not just split 0; the prefix/split_no/split_count are read from that
    // file and every other shard's path is derived from llama.cpp's own
    // naming convention. Throws std::runtime_error on any structural problem
    // (missing shard, tensor count mismatch, split.no mismatch, ...).
    static std::unique_ptr<GgufModel> open(const std::string & any_shard_path);
    ~GgufModel();

    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;

    const HParams & hparams() const { return hparams_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }

    const TensorInfo * find(const std::string & name) const;
    const TensorInfo * find_layer(int il, const std::string & suffix) const; // looks up "blk.<il>.<suffix>"

    size_t num_shards() const { return shard_paths_.size(); }
    const std::string & shard_path(size_t idx) const { return shard_paths_[idx]; }
    size_t shard_size(size_t idx) const { return shard_len_[idx]; }

private:
    GgufModel() = default;
    void collect_tensors(gguf_context * ctx, int shard_idx);
    void map_shard(size_t idx);

    HParams hparams_;
    std::vector<TensorInfo> tensors_;
    std::vector<std::string> shard_paths_;
    std::vector<void *> shard_map_; // mmap() base per shard, nullptr until map_shard() runs
    std::vector<size_t> shard_len_;
    std::unordered_map<std::string, size_t> name_index_; // name -> index into tensors_
};

} // namespace franken
