// tools/hot-expert/franken/decode/glm5_shapes.h
//
// The shapes of GLM-5.3-Flash (arch `glm5next`) as the decode loop needs
// them. Namespace fk::glm5, so nothing here collides with decode_shapes.h
// (Qwen3.8) or ds4_shapes.h (DeepSeek-V4): the architectures share the
// Backend, the GEMV decoders, the oracle and -- through Ds4Ops -- the four
// hyper-connection ops, and nothing else.
//
// Every constant is read back from the GGUF header at load time and asserted
// (glm5_model.cpp, check_hparams()). Source of every number: the header of
// ~/models/GLM-5.3-Flash/UD-IQ4_XS/*-00001-of-00005.gguf, read 2026-09-24
// (GLM5.md section 1), and ~/src/llama-glm53/src/models/glm5next.cpp
// @ 39931761a for how each is used.

#pragma once

#include <cstdint>

namespace fk {
namespace glm5 {

// ---- trunk ---------------------------------------------------------------
constexpr int N_EMBD       = 4096;    // glm5next.embedding_length
constexpr int N_BLOCK      = 46;      // glm5next.block_count, the NextN (MTP) block included
constexpr int N_LAYER      = 45;      // block_count - nextn_predict_layers: the text tower
constexpr int N_VOCAB      = 154880;
constexpr int N_LEAD_DENSE = 3;       // leading_dense_block_count: layers 0-2 have a dense FFN
constexpr int N_FF_DENSE   = 12288;   // feed_forward_length

// ---- hyper-connections (DeepSeek-V4's build_hc_pre / build_hc_post) -------
constexpr int HC        = 4;                    // hyper_connection.count
constexpr int HC_DIM    = HC * N_EMBD;          // 16384
constexpr int HC_MIX    = (2 + HC) * HC;        // 24: pre[4] | post[4] | comb[16]
constexpr int HC_SINKHORN_ITERS = 20;

// ---- layer kinds -----------------------------------------------------------
// attention.head_count_kv is a per-layer ARRAY: 0 = a KDA (Kimi delta
// attention, recurrent) layer, 1 = a DSA layer (MLA, absorbed, with the
// pooled lightning indexer). In this file: layers 3, 7, ..., 43 (every 4th)
// and the NextN block 45 are DSA; the other 34 of 45 are KDA.
constexpr bool is_dsa(int il) { return il == 45 || (il % 4) == 3; }

// ---- KDA (build_kda_layer) -------------------------------------------------
constexpr int N_HEAD      = 64;       // attention.head_count (KDA heads = MLA query heads)
constexpr int KDA_DIM     = 128;      // kda.head_dim: key = value = state width
constexpr int KDA_INNER   = N_HEAD * KDA_DIM;   // 8192 = q/k/v width
constexpr int KDA_CONV    = 4;        // ssm.conv_kernel
constexpr int KDA_QKV     = 3 * KDA_INNER;      // 24576 conv channels (q|k|v)
constexpr float KDA_GATE_LOWER_BOUND = -5.0f;   // kda.gate_lower_bound
constexpr float KDA_L2_EPS = 1e-6f;   // glm5next.cpp: "the reference's own constant"

// ---- DSA: MLA, absorbed, nope-only -----------------------------------------
constexpr int Q_LORA      = 1536;     // attention.q_lora_rank
constexpr int KV_LORA     = 512;      // attention.kv_lora_rank: the latent K = V row
constexpr int QK_HEAD     = 256;      // attention.key_length_mla (no rope: rope.dimension_count 0)
constexpr int V_HEAD      = 256;      // attention.value_length_mla
constexpr int Q_WIDTH     = N_HEAD * QK_HEAD;   // 16384
constexpr int O_WIDTH     = N_HEAD * V_HEAD;    // 16384

// ---- the pooled lightning indexer -------------------------------------------
constexpr int IDX_N_HEAD    = 32;     // attention.indexer.head_count
constexpr int IDX_DIM       = 128;    // attention.indexer.key_length
constexpr int IDX_TOP_K     = 2048;   // attention.indexer.top_k, in CELLS
constexpr int KPOOL         = 4;      // attention.indexer.kpool: 4 cells a pool
constexpr int IDX_TOP_POOLS = IDX_TOP_K / KPOOL;              // 512 pools selected
constexpr int IDX_N_SELECT  = IDX_TOP_K + KPOOL - 1;          // 2051: scoring runs when n_ctx > this

// ---- MoE -------------------------------------------------------------------
constexpr int N_EXPERT      = 288;    // expert_count
constexpr int N_EXPERT_USED = 8;      // expert_used_count
constexpr int N_FF_EXP      = 2048;   // expert_feed_forward_length == shared width
constexpr float EXPERT_WEIGHTS_SCALE = 2.5f;   // expert_weights_scale (routed only)
constexpr float SWIGLU_CLAMP = 10.0f; // swiglu_clamp_exp == _shexp, every layer (dense FFN too)

constexpr float RMS_EPS_DEFAULT = 9.999999747378752e-06f;   // attention.layer_norm_rms_epsilon
constexpr float LN_EPS_DEFAULT  = 9.999999974752427e-07f;   // attention.layer_norm_epsilon (indexer k_norm)
constexpr float HC_EPS_DEFAULT  = 9.999999974752427e-07f;   // hyper_connection.epsilon

} // namespace glm5
} // namespace fk
