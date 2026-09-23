// tools/hot-expert/franken/decode/decode_shapes.h
//
// L0 step 2 (L0-STEP2-BRIEF-2026-09-22.md, deliverable B; design rev 8
// section 9.5 step 2): the shapes of Qwen3.8-Flash-Next (arch `qwen4exp`) as
// the decode loop needs them.
//
// Every constant here is ALSO read back from the GGUF header at run time and
// asserted against these values (decode_model.cpp, check_shapes()). They are
// compile-time constants because the kernels are written around them (a wave
// of 32 lanes over a 128-wide GDN head, 12 query heads per GQA group, ...),
// not because the file is trusted.
//
// Source of every number:
//   GGUF header of ~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/...-00001-of-00003.gguf
//   (read 2026-09-22), cross-checked against
//   ~/models/Qwen3.8-Flash-Next-FP8/config.json and against
//   ~/src/llama-glm53/src/models/qwen4exp.cpp's use of them.

#pragma once

#include <cstddef>
#include <cstdint>

// The weight formats layers 0-15 of this file actually carry. Declared here,
// not in decode_quant.h, because decode_backend.h's Mat names them and that
// header must be includable without pulling in the block decoders.
enum FkQuantType : int {
    FK_Q_F32    = 0,
    FK_Q_BF16   = 1,
    FK_Q_Q8_0   = 2,
    FK_Q_IQ4_NL = 3,
    FK_Q_IQ4_XS = 4,
    FK_Q_IQ3_S  = 5,
    FK_Q_Q6_K   = 6,   // lm_head only
    // DeepSeek-V4-Flash UD-IQ2_M (ds4_quant.h, DEEPSEEK4.md). Appended, so
    // every value above is unchanged.
    FK_Q_Q4_K    = 7,  // output.weight
    FK_Q_Q5_K    = 8,  // attn_q_a, ffn_{gate,up}_shexp, token_embd
    FK_Q_IQ2_XXS = 9,  // routed gate/up, 42 layers
    FK_Q_IQ2_S   = 10, // routed gate/up, layer 26
    FK_Q_IQ3_XXS = 11, // routed down, 41 layers
    FK_Q_MXFP4   = 12, // routed down, layers 26 and 42
};

namespace fk {

// ---- trunk ---------------------------------------------------------------
constexpr int N_EMBD   = 2560;   // qwen4exp.embedding_length
constexpr int N_LAYER  = 48;     // qwen4exp.block_count
constexpr int N_VOCAB  = 248320;

// ---- hyper-connections (qwen4exp.cpp build_hc_mix / build_hc_combine) ----
constexpr int HC       = 4;      // qwen4exp.hyper_connection.count
constexpr int HC_DIM   = HC * N_EMBD;   // 10240
constexpr int HC_LR    = 320;    // qwen4exp.hyper_connection.low_rank

// ---- QSA full attention (every 4th layer: 3, 7, 11, ...) ----------------
constexpr int N_Q_HEADS  = 24;   // qwen4exp.attention.head_count
constexpr int N_KV_HEADS = 2;    // qwen4exp.attention.head_count_kv
constexpr int HEAD_DIM   = 256;  // qwen4exp.attention.key_length == value_length
constexpr int GQA_GROUP  = N_Q_HEADS / N_KV_HEADS;  // 12
constexpr int N_ROT      = 64;   // qwen4exp.rope.dimension_count (partial_rotary_factor 0.25)
constexpr int ROPE_SECTIONS[4] = {11, 11, 10, 0};   // qwen4exp.rope.dimension_sections
constexpr float ROPE_FREQ_BASE = 10000000.0f;       // qwen4exp.rope.freq_base
constexpr int FULL_ATTN_INTERVAL = 4;

// ---- the lightning indexer that drives QSA's block selection ------------
constexpr int IDX_N_HEADS = 4;    // qwen4exp.attention.indexer.head_count
constexpr int IDX_DIM     = 128;  // qwen4exp.attention.indexer.key_length
constexpr int IDX_TOP_K   = 2048; // qwen4exp.attention.indexer.top_k (in TOKENS)
constexpr int QSA_RATIO   = 4;    // qwen4exp.attention.compress_ratios[il] on QSA layers

// ---- Gated DeltaNet (the other 36 layers) -------------------------------
constexpr int GDN_CONV_K  = 4;    // qwen4exp.ssm.conv_kernel
constexpr int GDN_STATE   = 128;  // qwen4exp.ssm.state_size  (== head_k_dim == head_v_dim)
constexpr int GDN_K_HEADS = 16;   // qwen4exp.ssm.group_count
constexpr int GDN_V_HEADS = 48;   // qwen4exp.ssm.time_step_rank
constexpr int GDN_INNER   = 6144; // qwen4exp.ssm.inner_size == GDN_STATE*GDN_V_HEADS
constexpr int GDN_KEY_DIM = GDN_STATE * GDN_K_HEADS;   // 2048
constexpr int GDN_VAL_DIM = GDN_STATE * GDN_V_HEADS;   // 6144
// load_arch_tensors: conv_dim = key_dim*2 + value_dim, and that is attn_qkv's ne1
constexpr int GDN_CONV_DIM = GDN_KEY_DIM * 2 + GDN_VAL_DIM;  // 10240
// The GDN conv is undilated, so its window needs GDN_CONV_K-1 history slots
// in front of the chunk (decode_backend.h, ple_conv_win).
constexpr int GDN_CONV_HIST = GDN_CONV_K - 1;                // 3

// ---- MoE ----------------------------------------------------------------
constexpr int N_EXPERT      = 512;  // qwen4exp.expert_count
constexpr int N_EXPERT_USED = 10;   // qwen4exp.expert_used_count
constexpr int N_FF_EXP      = 640;  // qwen4exp.expert_feed_forward_length
constexpr int N_FF_SHEXP    = 640;  // qwen4exp.expert_shared_feed_forward_length
// hparams.expert_weights_scale: the key is ABSENT from this GGUF, so
// llama-hparams.h's 0.0f default stands and build_moe_ffn's
// `if (w_scale != 0.0f && w_scale != 1.0f)` skips the scale entirely.
// Kept explicit so a future file that DOES carry the key is not silently
// mis-read (decode_model.cpp reads the key and refuses a non-zero value).
constexpr float EXPERT_WEIGHTS_SCALE = 0.0f;
// hparams.swiglu_clamp_exp[il] is 0 for this arch (llama-model.cpp:1285
// fills it with 0 and qwen4exp never sets it), so the expert FFN is a plain
// ggml_swiglu_split: silu(gate)*up, no clamp. (Contrast GLM-5.3, where the
// missing clamp was a real bug -- CLAUDE.md.)
constexpr float SWIGLU_CLAMP_EXP = 0.0f;

// ---- PLE (per-layer n-gram hash embedding), layer 1 ---------------------
constexpr int PLE_LAYER        = 1;    // qwen4exp.ple.layers = [1]
constexpr int PLE_NGRAM        = 3;    // qwen4exp.ple.ngram_size
constexpr int PLE_HEADS_PER_NG = 8;    // qwen4exp.ple.heads_per_ngram
constexpr int PLE_N_HEADS      = (PLE_NGRAM - 1) * PLE_HEADS_PER_NG;  // 16
constexpr int PLE_HEAD_DIM     = 160;  // qwen4exp.embedding_length_per_layer_input
constexpr int PLE_CONV_K       = 4;    // qwen4exp.ple.conv_kernel
// build_ple: the depthwise causal conv is DILATED by the n-gram size, so its
// history is (kern-1)*ngram positions deep.
constexpr int PLE_CONV_DIL     = PLE_NGRAM;               // 3
constexpr int PLE_CONV_HIST    = (PLE_CONV_K - 1) * PLE_CONV_DIL;  // 9

static_assert(PLE_N_HEADS * PLE_HEAD_DIM == N_EMBD, "PLE gather must produce one n_embd row");
static_assert(GDN_V_HEADS % GDN_K_HEADS == 0, "GDN k-heads must divide v-heads");

// ---- epsilons -----------------------------------------------------------
// qwen4exp.attention.layer_norm_rms_epsilon = 9.999999974752427e-07, i.e.
// the f32 nearest to 1e-6. Read from the file at run time; this is the value
// the CPU reference path uses when the caller does not override it.
constexpr float RMS_EPS_DEFAULT = 9.999999974752427e-07f;

// The most cells QSA can select for one query: "the reference returns
// indexer_top_k + compress_ratio - 1: whole blocks plus the tail"
// (qwen4exp.cpp:613). Sizes the attention split/combine scratch.
constexpr int MAX_SEL = IDX_TOP_K + QSA_RATIO - 1;   // 2051

// A layer is Gated DeltaNet unless it is the FULL_ATTN_INTERVAL-th
// (qwen4exp.cpp:93-100's fallback rule; this GGUF carries no explicit
// recurrent_layers array). compress_ratios[il] > 0 on exactly those layers,
// which is what turns QSA on (qwen4exp.cpp:718).
constexpr bool is_recurrent_layer(int il) {
    return ((il + 1) % FULL_ATTN_INTERVAL) != 0;
}

} // namespace fk
