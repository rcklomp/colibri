// tools/hot-expert/franken/ple.h
//
// L0's CPU-side per-layer n-gram embedding (PLE) gather (design doc section
// 9.1: "The per-layer n-gram embedding table ... stays in host RAM ... its
// per-token gather ... is done on the CPU and uploaded with the token"; the
// rule and its measurement are record §PLE-GATHER).
//
// Ported from llama.cpp's src/models/qwen4exp.cpp, commit 39931761a of
// ~/src/llama-glm53 (read on the rig, 2026-09-22):
//
//   qwen4exp.cpp:969-1052   llm_graph_input_ple::set_input() -- the hash
//                            that picks which table rows a token gathers.
//   qwen4exp.cpp:1116-1135  build_inp_ple() -- get_rows + the head-slowest
//                            flatten this file's ple_gather() reproduces.
//   qwen4exp.cpp:1137-1227  build_ple() -- what those rows feed into
//                            (ple_key/value projections, the gate, the
//                            causal depthwise conv) -- NOT ported here; that
//                            is device-side per-layer work, build order step
//                            2/3 (design 9.5), not the loader.
//
// THE HASH (qwen4exp.cpp:970-971, reproduced verbatim as the source comment
// there put it):
//
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
//
// For n-gram order n in [2, ngram_size], per_gram = heads_per_ngram heads
// are filled at head index (n-2)*per_gram + g, g in [0, per_gram). t[0] is
// the token itself; t[1..n-1] are its 1..n-1 predecessors. An EOS token
// anywhere in the window CUTS it: every slot at or before the cut (measured
// from the near end) reads as EOS too, and a missing predecessor (nothing
// there yet) counts as a cut (qwen4exp.cpp:1030-1036). The EOS-ness of the
// token being hashed itself does NOT cut its own window.
//
// llama.cpp gets predecessor tokens out of the attention KV cache's cells
// (llama_kv_cache::get_prev_tokens, llama-kv-cache.cpp:1865), because a
// served request may resume from a prefix checkpoint. compute_ple_indices()
// below is for a single, freshly-tokenised prompt with no cache at all: the
// predecessor s positions before token i is exactly tokens[i-s] for i-s>=0,
// and "missing" (i-s<0, i.e. before the sequence start) otherwise -- which
// is bit-for-bit what get_prev_tokens returns for a cell that was never
// written (llama-kv-cache.cpp:1904-1906's LLAMA_TOKEN_NULL fill), so this
// port matches the reference for every case that can occur before any
// prefix checkpoint exists. It does NOT reproduce prefix-checkpoint resume
// (a real served request's predecessors can predate this call's own token
// list); that needs the KV-cache-backed version, a later L0 step.
//
// The PLE table itself (per_layer_token_embd.weight, IQ4_NL, ne=[160,
// 320001536] on the served Qwen3.8-Flash-Next split) is dequantised a row
// at a time with ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float -- IQ4_NL
// blocks are 32 values each, and a row is 160 = 5*32 values, so
// to_float(row_ptr, out, 160) covers exactly one row, no partial block.
//
// Nothing in the default (mmap) path here reads more of the table than the
// handful of rows a token's gather actually selects -- see gguf_model.h's
// header comment on why that matters (record §PLE-GATHER: 9.5 µs/token
// page-cached, 4.2 ms/token if a row's page has to come from the NVMe).
// load_ple_table_pinned() is the OTHER mode design 9.1 calls for (the whole
// 28.8 GB table resident in a hipHostMalloc'd pinned buffer at start) --
// it is declared here, implemented in ple.cpp, and is NEVER CALLED by
// anything in this directory: it needs the HIP runtime, which this build is
// not allowed to touch.

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "gguf_model.h"

namespace franken {

// idx[i * n_heads + h] for i in [0, tokens.size()), h in [0, hp.ple_n_heads()).
std::vector<int32_t> compute_ple_indices(const HParams & hp, const std::vector<int32_t> & tokens);

struct PleGatherResult {
    std::vector<int32_t> idx; // [n_tokens * n_heads], the row index compute_ple_indices() chose
    std::vector<float>   emb; // [n_tokens * n_heads * head_dim], dequantised rows, heads concatenated
                               // slowest (get_rows' own layout, qwen4exp.cpp:1132-1133's reshape)
};

// Computes the indices, then dequantises exactly those rows of
// per_layer_token_embd.weight out of the model's mmap -- CPU only, touches
// only the pages those rows live on. Throws if the model has no PLE layer
// or the table isn't the expected IQ4_NL.
// `table_base`, when given, replaces the mmap as the table's base address:
// design 9.1's resident PLE table (franken_serve.cpp loads the whole 28.8 GB
// once at engine start). The row arithmetic is identical -- the only thing
// that changes is whether a row that has not been read for a while costs
// 9.5 us or a 4.2 ms page fault off the NVMe (record §PLE-GATHER).
PleGatherResult ple_gather(const GgufModel & model, const std::vector<int32_t> & tokens,
                           const uint8_t * table_base = nullptr);

// franken_load --ple-check's output: per position, the chosen row index per
// head, then the first 8 dequantised values of the concatenated per-layer
// input (head 0's row starts it, since get_rows lays heads out slowest --
// see PleGatherResult::emb above). Stable, greppable text so an orchestrator
// can diff it against llama.cpp's own llama-eval-callback dump of the same
// op later (design 9.5 step 1's oracle).
void print_ple_check(const HParams & hp, const std::vector<int32_t> & tokens,
                      const PleGatherResult & r, FILE * out);

// Design 9.1's OTHER PLE mode: reads the WHOLE 28.8 GB table into a
// hipHostMalloc'd pinned buffer at engine start, so every later gather is a
// plain host read with no page fault possible. Needs the HIP runtime
// (hipHostMalloc initialises it) -- DO NOT CALL. Declared for completeness
// of the loader's public surface; ple_check and --plan never call it, and
// franken_load.cpp has no command-line path that reaches it either.
void * load_ple_table_pinned(const GgufModel & model, size_t & out_bytes);

} // namespace franken
