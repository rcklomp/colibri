// tools/hot-expert/franken/decode/decode_backend.h
//
// The one interface the decode loop is written against, and the reason the
// `--cpu` path in L0-STEP2-BRIEF-2026-09-22.md is not optional.
//
// decode_graph.cpp holds THE MATH -- the order of the operations, the tensor
// layouts, the epsilons, which activation feeds which projection -- ported op
// for op from ~/src/llama-glm53/src/models/qwen4exp.cpp. It never touches a
// float: it only calls the methods below and hands around opaque pointers.
// Two backends implement them:
//
//   CpuBackend (decode_cpu.cpp)  plain C++ on host buffers, reading the GGUF
//                                mmap directly. Runs anywhere, needs no GPU,
//                                and is what proves the MATH against
//                                llama.cpp's dumped intermediates before a
//                                GPU minute is spent.
//   GpuBackend (decode_gpu.hip)  HIP kernels on device 0 buffers.
//
// Both are checked against the SAME oracle points, so a disagreement is
// localised immediately: `--cpu` failing is a port bug in decode_graph.cpp;
// `--cpu` passing and the GPU run failing is a kernel bug.
//
// Pointer discipline: every `float *` crossing this interface is a BACKEND
// pointer. For CpuBackend it happens to be host memory; for GpuBackend it is
// a device address. decode_graph.cpp must never dereference one -- it reads a
// buffer only through download(), which both backends implement.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "decode_shapes.h"

namespace fk {

// ------------------------------------------------------------- matrices --
//
// One weight matrix as the GEMVs read it: `rows` output rows of `K` inputs,
// in the file's own format, no requantisation (design rev 8 section 9.3, and
// record §M1 candidate 3 for why that is the decided choice).
//
// For a routed-expert tensor (ne = [K, rows, 512]) `expert_stride` is the
// byte distance between expert e and e+1; `base` points at expert 0.
struct Mat {
    const void * base          = nullptr;  // backend-resident, never dereferenced by the graph
    int          type          = FK_Q_F32; // FkQuantType
    int64_t      K             = 0;        // ne0, the reduction length
    int64_t      rows          = 0;        // ne1
    size_t       row_bytes     = 0;
    size_t       expert_stride = 0;        // 0 unless this is a 512-expert tensor
    const char * name          = "";       // for error messages only
    bool ok() const { return base != nullptr; }
};

// ------------------------------------------------------------- unary ops --
enum FkUnary  { FK_SILU = 0, FK_SIGMOID, FK_SOFTPLUS, FK_EXP, FK_NEG_NOP };
enum FkBinary { FK_ADD = 0, FK_MUL, FK_SUB };

// ------------------------------------------------------------- backend ----
class Backend {
public:
    virtual ~Backend() {}
    virtual const char * name() const = 0;
    virtual bool is_gpu() const = 0;

    // -- memory ------------------------------------------------------------
    // alloc() returns a ZEROED buffer (the GDN state and the conv rings rely
    // on it: a fresh sequence starts from a zero state, which is what
    // llama.cpp's recurrent cache clear does).
    virtual float * alloc_f32(size_t n) = 0;
    virtual void *  alloc_raw(size_t bytes) = 0;
    virtual int *   alloc_i32(size_t n) = 0;
    virtual void    free_buf(void * p) = 0;
    virtual void    upload(void * dst, const void * src, size_t bytes) = 0;
    virtual void    download(void * dst, const void * src, size_t bytes) = 0;
    virtual void    sync() = 0;

    // Make a host (mmap) tensor readable by this backend's kernels. CpuBackend
    // returns the pointer unchanged (nothing is copied, nothing is paged in
    // until a kernel reads a row); GpuBackend hipMallocs and copies.
    virtual const void * place(const void * host, size_t bytes, const char * what) = 0;
    virtual size_t placed_bytes() const = 0;

    // -- generic vector ops -------------------------------------------------
    virtual void gemv(const Mat & W, const float * x, float * y) = 0;
    // W with expert_stride != 0, one expert chosen by `expert`.
    virtual void gemv_expert(const Mat & W, int expert, const float * x, float * y) = 0;

    virtual void copy(float * dst, const float * src, size_t n) = 0;
    virtual void unary(int op, const float * x, float * y, size_t n) = 0;
    virtual void binary(int op, const float * a, const float * b, float * y, size_t n) = 0;
    virtual void scale(const float * x, float s, float * y, size_t n) = 0;
    // build_hc_mix's silu(scale(lo, 1/hc)) as one op. Every elementwise op
    // costs a dependent kernel latency whatever its size, and the hc modules
    // run 32 times a token.
    virtual void scale_silu(const float * x, float s, float * y, size_t n) = 0;
    // y[i] = x[i] * w[i % ne0_w]; ne0_w == n means a plain elementwise mul.
    virtual void mul_tiled(const float * x, const float * w, float * y, size_t n, size_t ne0_w) = 0;
    // dst[r*row_len + i] = src[r*src_stride + src_off + i] -- the strided view
    // qwen4exp.cpp:726 takes to split attn_q's [q|gate] interleaved rows.
    virtual void gather_strided(const float * src, float * dst,
                                int n_rows, int row_len, int src_stride, int src_off) = 0;

    // ggml_rms_norm over ne0, per group, then (optionally) * w[n_groups*ne0]
    // or * w[ne0] tiled across groups -- build_norm(LLM_NORM_RMS) and
    // build_hc_mix's "grouped RMSNorm" are both this call.
    virtual void rms_norm_mul(const float * x, const float * w, float * y,
                              int ne0, int n_groups, size_t w_len, float eps) = 0;
    // ggml_l2_norm: scale = 1/max(sqrt(sum x^2), eps)  (ops.cpp:4333 -- NOT
    // the rms form; using rms here is a silent 1/sqrt(ne0) error)
    virtual void l2_norm(const float * x, float * y, int ne0, int n_groups, float eps) = 0;

    // -- hyper-connections (qwen4exp.cpp:218-286) ---------------------------
    virtual void hc_broadcast(const float * x, float * res_hc) = 0;              // [2560] -> [4][2560]
    virtual void hc_collapse(const float * xn, const float * gate, float * out) = 0;
    virtual void hc_combine(float * res_hc, const float * blk, const float * inject) = 0;

    // -- PLE (qwen4exp.cpp:1137-1227) ---------------------------------------
    // s[c] = sum_i key[c][i]*query[c][i] / sqrt(n_embd);
    // gate[c] = sigmoid(sgn(s)*sqrt(clamp(|s|,1e-6,inf)));
    // gated[c][i] = value[i]*gate[c]
    virtual void ple_gate(const float * key, const float * query, const float * value,
                          float * gated, float * gate_out) = 0;
    // Dilated depthwise causal conv over the ring, then silu.
    // ring holds PLE_CONV_HIST+1 slots of HC_DIM; slot `head` is this token.
    virtual void ple_conv_silu(const float * ring, int head, const float * w, float * out) = 0;

    // -- Gated DeltaNet (qwen4exp.cpp:793-918 + the fused op's formula) -----
    // ggml_ssm_conv over GDN_CONV_K taps of the ring, then silu.
    virtual void gdn_conv_silu(const float * ring, int head, const float * w, float * out) = 0;
    // qwen4exp.cpp:816-834's gate chain -- sigmoid(beta), softplus(alpha+dt),
    // *ssm_a, exp -- as ONE op over 48 elements instead of five. It still
    // writes a_softplus and gate, so the oracle taps are unchanged.
    virtual void gdn_gate(const float * beta_raw, const float * alpha_raw,
                          const float * dt, const float * ssm_a,
                          float * beta_sig, float * a_softplus,
                          float * gate, float * g_exp, int n) = 0;
    // One delta-rule step. `g_exp` is already exp(g) per v-head.
    virtual void gdn_step(float * state, const float * q, const float * k, const float * v,
                          const float * g_exp, const float * beta, float * out) = 0;
    // rms_norm(x)*w * sigmoid(z), per head (build_norm_gated).
    virtual void gated_rms_norm(const float * x, const float * w, const float * z,
                                float * y, int ne0, int n_groups, float eps) = 0;

    // -- MoE (build_moe_ffn) -------------------------------------------------
    // softmax over N_EXPERT, top-N_EXPERT_USED, gather, normalise (clamped).
    virtual void router(const float * logits, int * ids, float * weights) = 0;
    virtual void moe_gate_up(const Mat & gate, const Mat & up, const int * ids,
                             const float * x, float * y_gate, float * y_up) = 0;
    virtual void silu_mul(const float * gate, const float * up, float * h, size_t n) = 0;
    virtual void moe_down(const Mat & down, const int * ids, const float * h, float * expert_out) = 0;
    virtual void moe_combine(const float * expert_out, const float * weights, float * out) = 0;

    // -- IMRoPE (ggml_rope_multi, mode GGML_ROPE_TYPE_IMROPE) ---------------
    virtual void rope_imrope(float * x, int n_heads, int head_dim, int n_rot, int pos) = 0;

    // -- QSA cache and attention --------------------------------------------
    // The q8_0 KV cache is laid out exactly as tools/hot-expert/m1/m3_attn.hip
    // lays it out (L0-STEP2-BRIEF, "exactly as the M3 harness lays them out"):
    // PLANAR, qs[(cell*N_KV_HEADS + h)*HEAD_DIM + d] int8 with one f16 scale
    // per 32-element block at sc[(cell*N_KV_HEADS + h)*(HEAD_DIM/32) + b].
    // That is NOT ggml's AoS block_q8_0 -- it is this engine's own cache, and
    // the planar form is what lets an attention lane pull 8 dims in one
    // aligned load. Scales are raw f16 bit patterns (uint16_t) so the CPU
    // backend needs no half type.
    virtual void kv_store_q8_0(const float * k, const float * v,
                               int8_t * kqs, uint16_t * ksc,
                               int8_t * vqs, uint16_t * vsc, int cell) = 0;
    virtual void idx_raw_store(const float * k_raw, float * raw_cache, int cell) = 0;
    // Mean-pool cells [blk*r, blk*r+r) (missing slots read cell 0, as
    // set_input_qsa's zero-filled blk_cells does), rms-norm with k_norm,
    // IMRoPE at position blk*r, store bf16 into the pooled cache.
    // `dbg_pooled` / `dbg_roped` receive this block's value before and after
    // the norm+rotation, which are the reference's `indexer_k_pooled` and
    // `indexer_k` taps; pass nullptr to skip them.
    virtual void idx_pool_block(const float * raw_cache, int blk, int n_filled,
                                const float * k_norm, float eps, uint16_t * pooled,
                                float * dbg_pooled, float * dbg_roped) = 0;
    virtual void idx_scan(const uint16_t * pooled, const float * q, float * scores, int n_blocks) = 0;
    // cell_scores[j] = (j <= q_pos) ? blk_scores[j/r] + bias(j/r) : -inf, with
    // set_input_qsa's per-block bias computed IN the kernel from tail_start:
    // +1e9 for a block at or past the incomplete tail, -inf for a block that
    // could not be pooled, 0 otherwise (llama-memory-hybrid-idx.cpp:438-447).
    // It used to be a host-side array uploaded per QSA layer, which put a
    // BLOCKING hipMemcpy inside the 16-layer body -- see decode_graph.cpp.
    virtual void qsa_expand(const float * blk_scores, float * cell_scores,
                            int n_kv, int q_pos, int ratio, int tail_start) = 0;
    // Returns how many were selected (min(width, visible)); fills `out`.
    virtual int  topk_select(const float * scores, int n, int width, int * out) = 0;
    virtual void attn_qsa(const int8_t * kqs, const uint16_t * ksc,
                          const int8_t * vqs, const uint16_t * vsc,
                          const float * q, const int * sel, int n_sel, float * out) = 0;

    // -- timing (GPU only; the CPU backend returns 0) -----------------------
    // Diagnostic arm, CPU backend only (--quant-act). llama.cpp's CPU
    // mul_mat does NOT dot f32 activations against quantised weights: it
    // converts the activation to the weight type's `vec_dot_type` first --
    // Q8_0 for a Q8_0 weight, BF16 for a BF16 one. Turning this on makes the
    // reference's rounding OURS, so a cosine that jumps when it is enabled
    // says the gap was the reference's own activation quantisation and not a
    // port bug. It is never on by default: this engine is the more accurate
    // side and should stay that way.
    virtual void set_quant_act(bool) {}

    virtual void   timer_start() {}
    virtual double timer_stop_ms() { return 0.0; }

    // -- profiling (--profile; GPU backend only) ----------------------------
    //
    // Per-kernel-class DEVICE time for one token, plus the two counts that say
    // whether the GPU is starved rather than slow: how many host-blocking
    // calls happen inside the layer body (it must be zero) and how many
    // kernels a token launches. One hipEvent is recorded BEFORE each op and
    // one after the body, so the interval between consecutive events is that
    // op's wall time on the device timeline INCLUDING any bubble in front of
    // it -- which is the number that distinguishes "this kernel is slow" from
    // "the GPU was idle waiting for the host".
    virtual void set_profile(bool) {}
    virtual void prof_reset() {}
    virtual void prof_end_token() {}
    virtual void prof_report(FILE * out, int n_tokens) { (void) out; (void) n_tokens; }
};

Backend * make_cpu_backend(int n_threads);
Backend * make_gpu_backend(int device);

} // namespace fk
