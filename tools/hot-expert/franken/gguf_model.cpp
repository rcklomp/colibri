// tools/hot-expert/franken/gguf_model.cpp -- see gguf_model.h for the design and the
// shard-layout facts this was written against.

#include "gguf_model.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace franken {

namespace {

std::string errno_str(const std::string & what) {
    return what + ": " + std::strerror(errno);
}

// --- small gguf KV readers -------------------------------------------------
// llama-model-loader.cpp's GGUFMeta::GKV templates do this generically for
// libllama; we only need scalar u32/f32/str and integer arrays of whatever
// width the writer chose (u32/i32/u64/i64 all show up across these keys --
// verified against the served file with tools/hot-expert/franken's peek
// script, see gguf_model.h's header comment), so a few free functions cover it.

uint32_t get_u32(const gguf_context * ctx, const std::string & key, uint32_t def = 0) {
    int64_t kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0) return def;
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, kid);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(ctx, kid);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(ctx, kid);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(ctx, kid);
        default: throw std::runtime_error("gguf key '" + key + "' has an unexpected scalar type");
    }
}

float get_f32(const gguf_context * ctx, const std::string & key, float def = 0.0f) {
    int64_t kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0) return def;
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, kid);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(ctx, kid);
        default: throw std::runtime_error("gguf key '" + key + "' has an unexpected scalar type");
    }
}

std::string get_str(const gguf_context * ctx, const std::string & key, const std::string & def = "") {
    int64_t kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0) return def;
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error("gguf key '" + key + "' is not a string");
    }
    return gguf_get_val_str(ctx, kid);
}

// Reads an integer KV array of whatever stored width into a uint64 vector.
// Returns empty if the key is absent.
std::vector<uint64_t> get_u64_arr(const gguf_context * ctx, const std::string & key) {
    std::vector<uint64_t> out;
    int64_t kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0) return out;
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error("gguf key '" + key + "' is not an array");
    }
    size_t n = gguf_get_arr_n(ctx, kid);
    const void * raw = gguf_get_arr_data(ctx, kid);
    out.resize(n);
    switch (gguf_get_arr_type(ctx, kid)) {
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; ++i) out[i] = ((const uint32_t *) raw)[i]; break;
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; ++i) out[i] = (uint64_t) (uint32_t) ((const int32_t *) raw)[i]; break;
        case GGUF_TYPE_UINT64: for (size_t i = 0; i < n; ++i) out[i] = ((const uint64_t *) raw)[i]; break;
        case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; ++i) out[i] = (uint64_t) ((const int64_t *) raw)[i]; break;
        default: throw std::runtime_error("gguf array key '" + key + "' has an unexpected element type");
    }
    return out;
}

void load_hparams(const gguf_context * ctx, HParams & hp) {
    const std::string & a = hp.arch;
    auto k = [&](const char * suf) { return a + "." + suf; };

    hp.block_count              = get_u32(ctx, k("block_count"));
    hp.embedding_length         = get_u32(ctx, k("embedding_length"));
    hp.expert_count             = get_u32(ctx, k("expert_count"));
    hp.expert_used_count        = get_u32(ctx, k("expert_used_count"));
    hp.head_count                = get_u32(ctx, k("attention.head_count"));
    hp.head_count_kv             = get_u32(ctx, k("attention.head_count_kv"));
    hp.full_attention_interval   = get_u32(ctx, k("full_attention_interval"), 4);
    hp.expert_ff_len             = get_u32(ctx, k("expert_feed_forward_length"));
    hp.expert_shared_ff_len      = get_u32(ctx, k("expert_shared_feed_forward_length"));
    hp.rms_eps                   = get_f32(ctx, k("attention.layer_norm_rms_epsilon"));
    hp.vocab_size                = get_u32(ctx, k("vocab_size"));
    hp.context_length            = get_u32(ctx, k("context_length"));

    hp.ssm_conv_kernel    = get_u32(ctx, k("ssm.conv_kernel"));
    hp.ssm_inner_size     = get_u32(ctx, k("ssm.inner_size"));
    hp.ssm_state_size     = get_u32(ctx, k("ssm.state_size"));
    hp.ssm_time_step_rank = get_u32(ctx, k("ssm.time_step_rank"));
    hp.ssm_group_count    = get_u32(ctx, k("ssm.group_count"));

    hp.hc_count    = get_u32(ctx, k("hyper_connection.count"));
    hp.hc_low_rank = get_u32(ctx, k("hyper_connection.low_rank"));

    hp.indexer_head_count = get_u32(ctx, k("attention.indexer.head_count"));
    hp.indexer_key_length = get_u32(ctx, k("attention.indexer.key_length"));
    hp.indexer_top_k      = get_u32(ctx, k("attention.indexer.top_k"));

    {
        auto v = get_u64_arr(ctx, k("attention.compress_ratios"));
        hp.compress_ratios.assign(v.begin(), v.end());
    }

    {
        auto v = get_u64_arr(ctx, k("ple.layers"));
        hp.ple_layers.assign(v.begin(), v.end());
    }
    hp.ple_ngram_size      = get_u32(ctx, k("ple.ngram_size"));
    hp.ple_heads_per_ngram = get_u32(ctx, k("ple.heads_per_ngram"));
    hp.ple_conv_kernel     = get_u32(ctx, k("ple.conv_kernel"));
    hp.ple_eos_token_id    = get_u32(ctx, k("ple.eos_token_id"));
    hp.ple_image_token_id  = get_u32(ctx, k("ple.image_token_id"), 0);
    hp.n_embd_per_layer    = get_u32(ctx, k("embedding_length_per_layer_input"));

    {
        auto v = get_u64_arr(ctx, k("ple.layer_multipliers"));
        for (size_t i = 0; i < v.size() && i < hp.ple_layer_multipliers.size(); ++i) {
            hp.ple_layer_multipliers[i] = v[i];
        }
    }
    {
        auto v = get_u64_arr(ctx, k("ple.head_offsets"));
        for (size_t i = 0; i < v.size() && i < hp.ple_head_offsets.size(); ++i) {
            hp.ple_head_offsets[i] = (uint32_t) v[i];
        }
    }
    {
        auto v = get_u64_arr(ctx, k("ple.head_vocab_sizes"));
        for (size_t i = 0; i < v.size() && i < hp.ple_head_vocab_sizes.size(); ++i) {
            hp.ple_head_vocab_sizes[i] = (uint32_t) v[i];
        }
    }
}

} // namespace

size_t TensorInfo::row_size() const {
    return ggml_row_size(type, ne0());
}

std::unique_ptr<GgufModel> GgufModel::open(const std::string & any_shard_path) {
    std::unique_ptr<GgufModel> model(new GgufModel());

    gguf_init_params params{ /*no_alloc=*/ true, /*ctx=*/ nullptr };

    gguf_context * ctx0 = gguf_init_from_file(any_shard_path.c_str(), params);
    if (!ctx0) {
        throw std::runtime_error("gguf_init_from_file failed on " + any_shard_path);
    }

    int64_t kid_no    = gguf_find_key(ctx0, "split.no");
    int64_t kid_count = gguf_find_key(ctx0, "split.count");
    uint16_t split_no    = kid_no    >= 0 ? gguf_get_val_u16(ctx0, kid_no)    : 0;
    uint16_t split_count = kid_count >= 0 ? gguf_get_val_u16(ctx0, kid_count) : 1;

    // Derive "<prefix>-%05d-of-%05d.gguf" from whichever shard was given
    // (llama.cpp's llama_split_prefix, src/llama.cpp:568, run in reverse).
    std::string prefix;
    {
        char postfix[32];
        std::snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, (int) split_count);
        std::string p(postfix);
        if (any_shard_path.size() > p.size() &&
            any_shard_path.compare(any_shard_path.size() - p.size(), p.size(), p) == 0) {
            prefix = any_shard_path.substr(0, any_shard_path.size() - p.size());
        } else {
            gguf_free(ctx0);
            throw std::runtime_error("path does not match llama.cpp's split naming convention: " + any_shard_path);
        }
    }

    model->shard_paths_.resize(split_count);
    for (int i = 0; i < split_count; ++i) {
        char buf[4096];
        std::snprintf(buf, sizeof(buf), "%s-%05d-of-%05d.gguf", prefix.c_str(), i + 1, (int) split_count);
        model->shard_paths_[i] = buf;
    }
    model->shard_map_.assign(split_count, nullptr);
    model->shard_len_.assign(split_count, 0);

    // Shard 0 must carry the full metadata (llama-model-loader.cpp:598-604
    // requires the model be "loaded with the first split"); re-open it by
    // path if the caller handed us a different shard, so the metadata read
    // below is always against split 0 regardless of which path was given.
    gguf_context * ctx_meta = ctx0;
    if (split_no != 0) {
        gguf_free(ctx0);
        ctx_meta = gguf_init_from_file(model->shard_paths_[0].c_str(), params);
        if (!ctx_meta) {
            throw std::runtime_error("failed to open shard 0: " + model->shard_paths_[0]);
        }
    }

    model->hparams_.arch = get_str(ctx_meta, "general.architecture");
    load_hparams(ctx_meta, model->hparams_);
    uint32_t expected_tensor_count = get_u32(ctx_meta, "split.tensors.count", 0);

    model->collect_tensors(ctx_meta, 0);
    gguf_free(ctx_meta);

    for (int i = 1; i < split_count; ++i) {
        gguf_context * ctx = gguf_init_from_file(model->shard_paths_[i].c_str(), params);
        if (!ctx) {
            throw std::runtime_error("failed to open shard: " + model->shard_paths_[i]);
        }
        int64_t kid = gguf_find_key(ctx, "split.no");
        if (kid < 0) {
            gguf_free(ctx);
            throw std::runtime_error("shard is missing split.no: " + model->shard_paths_[i]);
        }
        uint16_t idx_gguf = gguf_get_val_u16(ctx, kid);
        if (idx_gguf != (uint16_t) i) {
            gguf_free(ctx);
            throw std::runtime_error("split.no mismatch in " + model->shard_paths_[i] +
                                      ": expected " + std::to_string(i) + " got " + std::to_string(idx_gguf));
        }
        model->collect_tensors(ctx, i);
        gguf_free(ctx);
    }

    if (expected_tensor_count > 0 && model->tensors_.size() != expected_tensor_count) {
        throw std::runtime_error("tensor count mismatch: split.tensors.count=" +
                                  std::to_string(expected_tensor_count) +
                                  " but found " + std::to_string(model->tensors_.size()));
    }

    for (size_t i = 0; i < model->shard_paths_.size(); ++i) {
        model->map_shard(i);
    }
    for (auto & t : model->tensors_) {
        t.data = (const uint8_t *) model->shard_map_[t.shard] + t.file_offset;
    }

    return model;
}

GgufModel::~GgufModel() {
    for (size_t i = 0; i < shard_map_.size(); ++i) {
        if (shard_map_[i]) {
            munmap(shard_map_[i], shard_len_[i]);
        }
    }
}

void GgufModel::collect_tensors(gguf_context * ctx, int shard_idx) {
    // offs = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i), exactly
    // llama_model_loader.h's llama_tensor_weight constructor (llama-model-loader.h:40-49).
    size_t data_offset = gguf_get_data_offset(ctx);
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; ++i) {
        TensorInfo t;
        t.name = gguf_get_tensor_name(ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(ctx, i);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) t.ne[d] = ne[d];
        t.n_dims = 1;
        for (int d = GGML_MAX_DIMS - 1; d > 0; --d) {
            if (ne[d] > 1) { t.n_dims = d + 1; break; }
        }
        t.type        = gguf_get_tensor_type(ctx, i);
        t.nbytes      = gguf_get_tensor_size(ctx, i);
        t.shard       = shard_idx;
        t.file_offset = data_offset + gguf_get_tensor_offset(ctx, i);
        t.data        = nullptr; // filled in by open() once every shard is mapped

        if (name_index_.count(t.name)) {
            throw std::runtime_error("duplicate tensor name: " + t.name);
        }
        name_index_.emplace(t.name, tensors_.size());
        tensors_.push_back(std::move(t));
    }
}

void GgufModel::map_shard(size_t idx) {
    if (shard_map_[idx]) return;
    const std::string & path = shard_paths_[idx];

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error(errno_str("open failed: " + path));

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error(errno_str("fstat failed: " + path));
    }
    size_t len = (size_t) st.st_size;

    void * base = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        throw std::runtime_error(errno_str("mmap failed: " + path));
    }

    shard_map_[idx] = base;
    shard_len_[idx] = len;
}

const TensorInfo * GgufModel::find(const std::string & name) const {
    auto it = name_index_.find(name);
    return it == name_index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo * GgufModel::find_layer(int il, const std::string & suffix) const {
    return find("blk." + std::to_string(il) + "." + suffix);
}

} // namespace franken
