// tools/hot-expert/franken/decode/ds4_shapes.h
//
// The shapes of DeepSeek-V4-Flash (arch `deepseek4`) as the decode loop needs
// them. Namespace fk::ds4, so nothing here can collide with decode_shapes.h's
// Qwen3.8 constants: the two architectures share the Backend, the GEMV
// decoders and the oracle, and nothing else.
//
// Every constant is read back from the GGUF header at load time and asserted
// (ds4_model.cpp, check_hparams()). Source of every number: the header of
// ~/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/*-00001-of-00003.gguf,
// read 2026-09-23 (tools/hot-expert/franken/decode/DEEPSEEK4.md section 1),
// and ~/src/llama-glm53/src/models/deepseek4.cpp @ 39931761a for how each is
// used.

#pragma once

#include <cstdint>

namespace fk {
namespace ds4 {

// ---- trunk ---------------------------------------------------------------
constexpr int N_EMBD   = 4096;     // deepseek4.embedding_length
constexpr int N_LAYER  = 43;       // deepseek4.block_count (no MTP block in this file)
constexpr int N_VOCAB  = 129280;

// ---- hyper-connections (build_hc_pre / build_hc_post / build_hc_head) -----
constexpr int HC        = 4;                    // hyper_connection.count
constexpr int HC_DIM    = HC * N_EMBD;          // 16384
constexpr int HC_MIX    = (2 + HC) * HC;        // 24: pre[4] | post[4] | comb[16]
constexpr int HC_SINKHORN_ITERS = 20;           // hyper_connection.sinkhorn_iterations

// ---- attention (MQA: one K=V head of 512 for 64 query heads) -------------
constexpr int N_HEAD      = 64;     // attention.head_count
constexpr int HEAD_DIM    = 512;    // attention.key_length == value_length
constexpr int N_ROT       = 64;     // rope.dimension_count: the LAST 64 dims of a head
constexpr int N_NOPE      = HEAD_DIM - N_ROT;   // 448: rope_set_offset(n_embd_head_nope)
constexpr int Q_LORA      = 1024;   // attention.q_lora_rank
constexpr int O_GROUPS    = 8;      // attention.output_group_count
constexpr int O_LORA      = 1024;   // attention.output_lora_rank
constexpr int O_GROUP_DIM = (N_HEAD / O_GROUPS) * HEAD_DIM;   // 4096
constexpr int N_SWA       = 128;    // attention.sliding_window: the raw K window

// Compressed attention (compress_ratios per layer):
//   0   raw sliding window only                 layers 0, 1
//   4   CSA: overlapped 4-token compression + the lightning indexer's top-k
//   128 HCA: plain 128-token compression, every block attended
constexpr int CSA_RATIO = 4;
constexpr int HCA_RATIO = 128;

// ---- the lightning indexer (CSA layers) ----------------------------------
constexpr int IDX_N_HEAD = 64;     // attention.indexer.head_count
constexpr int IDX_DIM    = 128;    // attention.indexer.key_length
constexpr int IDX_N_NOPE = IDX_DIM - N_ROT;    // 64
constexpr int IDX_TOP_K  = 512;    // attention.indexer.top_k (in BLOCKS of 4 tokens)

// ---- MoE -------------------------------------------------------------------
constexpr int N_EXPERT      = 256;   // expert_count
constexpr int N_EXPERT_USED = 6;     // expert_used_count
constexpr int N_FF_EXP      = 2048;  // expert_feed_forward_length (== shared expert width)
constexpr int HASH_LAYERS   = 3;     // hash_layer_count: layers 0-2 route by token id
constexpr float EXPERT_WEIGHTS_SCALE = 1.5f;   // expert_weights_scale
constexpr float SWIGLU_CLAMP = 10.0f;          // swiglu_clamp_exp == swiglu_clamp_shexp, every layer

// ---- rope --------------------------------------------------------------------
constexpr float ROPE_FREQ_BASE          = 10000.0f;    // rope.freq_base (ratio-0 layers)
constexpr float COMPRESS_ROPE_FREQ_BASE = 160000.0f;   // attention.compress_rope_freq_base
constexpr float ROPE_SCALE_FACTOR       = 16.0f;       // rope.scaling.factor (yarn)
constexpr int   ROPE_N_CTX_ORIG         = 65536;       // rope.scaling.original_context_length
constexpr float YARN_BETA_FAST          = 32.0f;
constexpr float YARN_BETA_SLOW          = 1.0f;

constexpr float RMS_EPS_DEFAULT = 9.999999974752427e-07f;   // == hyper_connection.epsilon

// Per-layer compress ratio of THIS file (compress_ratios[0..42]); read from
// the header and compared at load time.
constexpr int compress_ratio(int il) {
    return il < 2 ? 0 : (il == 42 ? 4 : ((il % 2 == 0) ? 4 : 128));
}

} // namespace ds4
} // namespace fk
