// tools/hot-expert/franken/ple.cpp -- see ple.h for the ported hash, its
// llama.cpp line numbers, and the scope this single-sequence port covers.

#include "ple.h"

#include <cstring>
#include <stdexcept>

// load_ple_table_pinned() is the only thing in this file that needs the HIP
// runtime, and nothing calls it (ple.h says so). FRANKEN_NO_HIP omits both,
// so a binary that must provably link no HIP -- decode/franken_decode_cpu,
// which is how L0 step 2 proves its math without touching a card -- can still
// reuse this gather unchanged.
#ifndef FRANKEN_NO_HIP
#include <hip/hip_runtime.h>
#endif

namespace franken {

std::vector<int32_t> compute_ple_indices(const HParams & hp, const std::vector<int32_t> & tokens) {
    const int64_t n_tokens = (int64_t) tokens.size();
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads();
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    if (n_gram < 2) throw std::runtime_error("compute_ple_indices: hp.ple_ngram_size < 2 (no PLE in this model?)");

    std::vector<int32_t> idx((size_t) (n_heads * n_tokens), 0);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // ctx[0] = the token itself; ctx[1..n_gram-1] = its 1..n_gram-1 predecessors,
        // EOS-cut as qwen4exp.cpp:1030-1036 does (ple.h's header comment).
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tokens[i];
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            int64_t pred_pos = i - s;
            int64_t t = cut ? -1 : (pred_pos >= 0 ? (int64_t) tokens[pred_pos] : -1); // -1 == LLAMA_TOKEN_NULL
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[(size_t) (i * n_heads + h_i)] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    return idx;
}

PleGatherResult ple_gather(const GgufModel & model, const std::vector<int32_t> & tokens,
                           const uint8_t * table_base) {
    const HParams & hp = model.hparams();
    if (hp.ple_layers.empty()) {
        throw std::runtime_error("ple_gather: model has no ple.layers entry");
    }

    const TensorInfo * table = model.find("per_layer_token_embd.weight");
    if (!table) throw std::runtime_error("ple_gather: per_layer_token_embd.weight not found");
    if (table->type != GGML_TYPE_IQ4_NL) {
        throw std::runtime_error(std::string("ple_gather: unexpected PLE table type ") +
                                  ggml_type_name(table->type) + " (expected IQ4_NL)");
    }

    const int64_t head_dim = hp.n_embd_per_layer;
    const int64_t n_heads  = hp.ple_n_heads();
    if (head_dim != table->ne0()) {
        throw std::runtime_error("ple_gather: n_embd_per_layer does not match the table's row width");
    }
    const size_t row_bytes = table->row_size();

    PleGatherResult r;
    r.idx = compute_ple_indices(hp, tokens);
    r.emb.assign((size_t) tokens.size() * (size_t) n_heads * (size_t) head_dim, 0.0f);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    if (!tt || !tt->to_float) throw std::runtime_error("ple_gather: no to_float for IQ4_NL in this ggml build");

    for (size_t i = 0; i < tokens.size(); ++i) {
        for (int64_t h = 0; h < n_heads; ++h) {
            int32_t row = r.idx[i * (size_t) n_heads + (size_t) h];
            if (row < 0 || (int64_t) row >= table->ne1()) {
                throw std::runtime_error("ple_gather: row index " + std::to_string(row) + " out of range");
            }
            // table_base is the resident copy when the engine has one; the
            // mmap otherwise. Same rows either way.
            const uint8_t * base = table_base ? table_base : table->data;
            const uint8_t * row_ptr = base + (size_t) row * row_bytes;
            float * out = &r.emb[(i * (size_t) n_heads + (size_t) h) * (size_t) head_dim];
            tt->to_float(row_ptr, out, head_dim);
        }
    }

    return r;
}

void print_ple_check(const HParams & hp, const std::vector<int32_t> & tokens,
                      const PleGatherResult & r, FILE * out) {
    const int64_t n_heads  = hp.ple_n_heads();
    const int64_t head_dim = hp.n_embd_per_layer;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::fprintf(out, "pos %zu token %d rows:", i, tokens[i]);
        for (int64_t h = 0; h < n_heads; ++h) {
            std::fprintf(out, " %d", r.idx[i * (size_t) n_heads + (size_t) h]);
        }
        std::fprintf(out, "\n");

        std::fprintf(out, "pos %zu head0_first8:", i);
        for (int64_t j = 0; j < 8 && j < head_dim; ++j) {
            std::fprintf(out, " %.6f", r.emb[i * (size_t) n_heads * (size_t) head_dim + (size_t) j]);
        }
        std::fprintf(out, "\n");
    }
}

#ifndef FRANKEN_NO_HIP
void * load_ple_table_pinned(const GgufModel & model, size_t & out_bytes) {
    // NEVER CALLED (see ple.h) -- kept behind its own function so nothing
    // else in this file has to link or touch the HIP runtime to build.
    const TensorInfo * table = model.find("per_layer_token_embd.weight");
    if (!table) throw std::runtime_error("load_ple_table_pinned: per_layer_token_embd.weight not found");

    void * pinned = nullptr;
    hipError_t rc = hipHostMalloc(&pinned, table->nbytes, hipHostMallocDefault);
    if (rc != hipSuccess) throw std::runtime_error("hipHostMalloc failed for the PLE table");

    std::memcpy(pinned, table->data, table->nbytes); // pulls the whole 28.8 GB off the NVMe/page cache
    out_bytes = table->nbytes;
    return pinned;
}

#endif // FRANKEN_NO_HIP

} // namespace franken
