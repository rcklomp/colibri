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
//
// ------------------------------------------------------------------------
// BATCHED PREFILL (design 9.4, PREFILL.md). Every op below that can see more
// than one token takes a `T` and reads/writes TOKEN-MAJOR buffers: token t's
// row of an `n`-wide activation is `buf + t*n`, so a row is contiguous and a
// tiled GEMM's loads stay coalesced. T == 1 is the decode path and must stay
// BIT-IDENTICAL to it -- every kernel here is written so that the arithmetic
// of one row does not depend on how many rows travel with it:
//
//   * the GEMM accumulates each column in exactly the order the one-column
//     GEMV did (four Q8_0 accumulators, two Q6_K, one otherwise), and its
//     split-K count is chosen from the MATRIX alone, never from T;
//   * every row-wise kernel is the same kernel with a grid axis over T;
//   * the GDN recurrence walks the chunk in order with the state carried
//     in registers, which is the same sequence of updates T steps make;
//   * the QSA per-query stage loops, because its scratch is O(n_kv).
//
// That is what makes `--chunk 6` against `--chunk 1` a bit-for-bit oracle
// rather than a tolerance test.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ----------------------------------------------------- batched GEMV ------
//
// Step 2b. 931 launches a token (58 a layer) at a device floor of a few us
// each was ~4 ms of an 11.7 ms token, and 17 of those 58 were projections --
// one kernel each, however small. Most of them SHARE an activation vector,
// which is the only thing that lets them share a kernel:
//
//   xn (attn/ffn)  -> hc_down + hc_inject            2 -> 1, twice a layer
//   mixed (attn)   -> ssm_qkv, gate, beta, alpha     4 -> 1  (GDN layer)
//   mixed (attn)   -> wq, wk, wv, idx_q, idx_k       5 -> 1  (QSA layer)
//   mixed (ffn)    -> ffn_gate_inp, sh_up, sh_gate, sh_gate_inp  4 -> 1
//
// A batch is one kernel over the CONCATENATED row space of up to
// GEMV_BATCH_MAX matrices, which may differ in format (the QSA batch mixes
// Q8_0 and BF16) and each of which gets its own output pointer and epilogue.
// Batching also raises the row count, which IS the wave count -- grouping
// rows into wider workgroups never does, because total waves == total rows.
enum GemvEpi {
    GE_NONE = 0,
    GE_SIGMOID,     // build_moe_ffn's shared-expert gate, build_hc_mix's gate
    GE_SCALE_SILU,  // build_hc_mix's silu(scale(lo, 1/hc))
};

// How the activation is formed. GX_SILU_MUL lets the shared expert's
// down-projection consume silu(gate)*up without a kernel of its own.
enum GemvX { GX_PLAIN = 0, GX_SILU_MUL };

struct GemvJob {
    const Mat * W    = nullptr;
    float *     out  = nullptr;
    int         epi  = GE_NONE;
    float       arg  = 0.0f;
};

// Profiling groups, so --profile can report achieved GB/s per projection
// class rather than one lumped trunk number.
enum GemvGroup {
    GG_HC_DOWN_INJECT = 0,
    GG_HC_UP,
    GG_GDN_PROJ,        // ssm_qkv + gate + beta + alpha
    GG_SSM_OUT,
    GG_FFN_PROJ,        // ffn_gate_inp + shexp up/gate/gate_inp
    GG_SH_DOWN,
    GG_QSA_PROJ,        // wq + wk + wv + indexer q/k
    GG_ATTN_OUT,
    GG_PLE,
    GG_LM_HEAD,         // output.weight, Q6_K, 248 320 rows
    GG_N
};
const char * gemv_group_name(int g);

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

    // -- a copy the HOST does not wait for (L0 step 4, 2026-09-23) ----------
    //
    // `download()` above is a stream synchronise plus a blocking hipMemcpy,
    // because the host is about to READ what it asked for. A serving
    // checkpoint is the other case: the bytes are wanted, nobody reads them
    // now, and waiting for them costs the three-card prefill pipeline. On the
    // engine's own stream a copy is ordered after everything already queued
    // (so it sees the chunk that has just run) and before everything queued
    // after it (so the next chunk cannot overwrite the state under it) --
    // which is exactly the ordering a checkpoint needs, with no host wait at
    // all. The destination is safe to read only after a later sync() of this
    // device.
    //
    // It must be PINNED host memory or hipMemcpyAsync degrades to a
    // synchronous copy (the same trap the upload staging ring exists for),
    // hence alloc_pinned. The base implementations here are the CPU
    // backend's: plain malloc and the blocking copy, both already correct.
    virtual void *  alloc_pinned(size_t bytes) { return std::malloc(bytes); }
    virtual void    free_pinned(void * p) { std::free(p); }
    virtual void    download_async(void * dst, const void * src, size_t bytes) {
        download(dst, src, bytes);
    }

    // -- device boundaries (step 3, design 9.1/9.2) -------------------------
    //
    // Everything the next layer range needs is the wide residual: the
    // hyper-connection streams, hc_count * n_embd f32 = 40 KB. Nothing else
    // survives a layer, so that is the whole boundary payload.
    //
    // `boundary_recv` is a method of the DESTINATION: it records an event on
    // the source's stream, makes its own stream wait on it, and issues the
    // peer copy on its own stream. No host call is involved, so a boundary
    // costs the ~30 us of P2P that M5 measured and nothing else.
    //
    // PIPELINED PREFILL (design 3.5). `bank` names which of the two residual
    // buffers is crossing. The destination ALSO records, after its peer copy,
    // an event that says "bank `bank` has been drained out of the source", and
    // hands it to the source; the source waits on it (on its own stream, no
    // host call) before it overwrites that bank for a later chunk. That pair
    // -- two banks plus a drain event -- is the whole reason chunk n+1 may run
    // on card 0 while chunk n is still on card 1: without it card 0's writes
    // would race card 1's read, and with a host sync instead of an event there
    // would be no pipeline at all.
    virtual int  device() const { return -1; }
    virtual void boundary_recv(void * dst, Backend & src, const void * src_ptr,
                               size_t bytes, int bank) = 0;
    // Called on a device that is about to WRITE residual bank `bank` (card 0
    // before hc_broadcast, every other card before the peer copy lands in it).
    // A no-op until some destination has actually consumed that bank.
    virtual void boundary_wait_free(int bank) { (void) bank; }
    // Greedy sampling on the device: a reduction over the 248 320 logits, so
    // the only thing that crosses to the host per token is one int.
    virtual void argmax(const float * logits, int n, int * out_id) = 0;
    virtual void vram_report(FILE * out, const char * tag) { (void) out; (void) tag; }

    // Make a host (mmap) tensor readable by this backend's kernels. CpuBackend
    // returns the pointer unchanged (nothing is copied, nothing is paged in
    // until a kernel reads a row); GpuBackend hipMallocs and copies.
    virtual const void * place(const void * host, size_t bytes, const char * what) = 0;
    virtual size_t placed_bytes() const = 0;

    // -- generic vector ops -------------------------------------------------
    virtual void gemv(const Mat & W, const float * x, float * y) = 0;
    // One kernel for `n` matrices sharing `x`. `x2` is used only by
    // GX_SILU_MUL, where the activation is silu(x[i])*x2[i].
    //
    // With T > 1 this is a GEMM: `x` is [T][K] token-major and every job's
    // `out` is [T][rows] token-major. The kernel loads a weight block once
    // per tile of GEMM_TILE columns, which is the whole point of batching a
    // prompt -- the trunk is ~5 GB a token and a tile of 8 reads it an
    // eighth as often.
    virtual void gemv_batch(const GemvJob * jobs, int n, const float * x,
                            int group, int x_mode = GX_PLAIN, const float * x2 = nullptr,
                            int T = 1) = 0;
    // W with expert_stride != 0, one expert chosen by `expert`.
    virtual void gemv_expert(const Mat & W, int expert, const float * x, float * y) = 0;

    // build_hc_mix's gate projection and the collapse that consumes it.
    //
    // They are two ops and this is one call because on the GPU they can be one
    // LAUNCH: hc_up's rows are HC streams of n_embd and the collapse wants all
    // HC of them for one embedding index, so a workgroup that takes its rows
    // stream-wise instead of consecutively already holds everything the
    // collapse reads. The hc modules run 32 times a decode token, so that is
    // 32 launches at a device floor of several microseconds each.
    //
    // The default below is the two ops, which is what the CPU backend runs and
    // what any backend that has not implemented the fusion runs; `up.out` is
    // written either way, so hc_gate stays an oracle tap.
    virtual void gemv_hc_gate_collapse(const GemvJob & up, const float * lo,
                                       const float * xn, float * out_mixed,
                                       int group, int T = 1) {
        gemv_batch(&up, 1, lo, group, GX_PLAIN, nullptr, T);
        hc_collapse(xn, up.out, out_mixed, T);
    }

    virtual void copy(float * dst, const float * src, size_t n) = 0;
    virtual void unary(int op, const float * x, float * y, size_t n) = 0;
    virtual void binary(int op, const float * a, const float * b, float * y, size_t n) = 0;
    virtual void scale(const float * x, float s, float * y, size_t n) = 0;
    // y = a + b + c, build_ple's closing `hidden + gated + conv_out`
    virtual void binary3_add(const float * a, const float * b, const float * c,
                             float * y, size_t n) = 0;
    // build_hc_mix's silu(scale(lo, 1/hc)) as one op. Every elementwise op
    // costs a dependent kernel latency whatever its size, and the hc modules
    // run 32 times a token.
    // y[i] = x[i] * w[i % ne0_w]; ne0_w == n means a plain elementwise mul.
    virtual void mul_tiled(const float * x, const float * w, float * y, size_t n, size_t ne0_w) = 0;
    // dst[r*row_len + i] = src[r*src_stride + src_off + i] -- the strided view
    // qwen4exp.cpp:726 takes to split attn_q's [q|gate] interleaved rows.
    virtual void gather_strided(const float * src, float * dst,
                                int n_rows, int row_len, int src_stride, int src_off) = 0;

    // ggml_rms_norm over ne0, per group, then (optionally) * w -- both
    // build_norm(LLM_NORM_RMS) and build_hc_mix's "grouped RMSNorm".
    //
    // The gamma is PERIODIC with period `w_len / ne0` groups: group g reads
    // `w + (g % (w_len/ne0))*ne0`. That is the batched-prefill
    // generalisation of what used to be a `w_tiled` flag, and it is the same
    // thing at T == 1: w_len == ne0 gives period 1 (one gamma reused by
    // every group) and w_len == HC*ne0 with n_groups == HC*T gives period HC,
    // which is exactly "hc stream g % HC of token g / HC" for a token-major
    // [T][HC][ne0] buffer.
    virtual void rms_norm_mul(const float * x, const float * w, float * y,
                              int ne0, int n_groups, size_t w_len, float eps) = 0;
    // ggml_l2_norm: scale = 1/max(sqrt(sum x^2), eps)  (ops.cpp:4333 -- NOT
    // the rms form; using rms here is a silent 1/sqrt(ne0) error)
    virtual void l2_norm(const float * x, float * y, int ne0, int n_groups, float eps) = 0;

    // -- hyper-connections (qwen4exp.cpp:218-286) ---------------------------
    // Token-major throughout: x is [T][2560], res_hc / xn are [T][4][2560],
    // inject is [T][4].
    virtual void hc_broadcast(const float * x, float * res_hc, int T = 1) = 0;   // [2560] -> [4][2560]
    virtual void hc_collapse(const float * xn, const float * gate, float * out, int T = 1) = 0;
    virtual void hc_combine(float * res_hc, const float * blk, const float * inject, int T = 1) = 0;
    // hc_combine followed by the grouped RMSNorm the next hc_mix would have
    // launched on its own; `w_norm == nullptr` does the combine alone. Both
    // reduce over the same 2 560 elements of one hc stream, so they fuse for
    // free.
    virtual void hc_combine_norm(float * res_hc, const float * blk, const float * inject,
                                 const float * w_norm, float * xn, float eps, int T = 1) = 0;

    // -- PLE (qwen4exp.cpp:1137-1227) ---------------------------------------
    // s[c] = sum_i key[c][i]*query[c][i] / sqrt(n_embd);
    // gate[c] = sigmoid(sgn(s)*sqrt(clamp(|s|,1e-6,inf)));
    // gated[c][i] = value[i]*gate[c]
    virtual void ple_gate(const float * key, const float * query, const float * value,
                          float * gated, float * gate_out, int T = 1) = 0;
    // Dilated depthwise causal conv over a WINDOW, then silu.
    //
    // A ring with a moving head cannot serve a chunk: T tokens are produced
    // at once and each needs its own history. The window is the ring
    // straightened out -- PLE_CONV_HIST history slots followed by the
    // chunk's T slots, so token t lives at slot PLE_CONV_HIST+t and its tap
    // k reads slot t + k*PLE_CONV_DIL (the dilation is folded in: at k =
    // PLE_CONV_K-1 that is the token itself). `conv_slide` then carries the
    // tail into the next chunk. Out is [T][HC_DIM].
    virtual void ple_conv_win(const float * win, const float * w, float * out, int T = 1) = 0;
    // Copy the last `hist` occupied slots of a conv window down to slots
    // [0, hist), so the next chunk sees its history. Source and destination
    // overlap when T < hist, so each thread reads its whole channel column
    // before writing any of it.
    virtual void conv_slide(float * win, int hist, int channels, int T) = 0;

    // -- Gated DeltaNet (qwen4exp.cpp:793-918 + the fused op's formula) -----
    // ggml_ssm_conv over GDN_CONV_K taps of the window (dilation 1), then
    // silu -- together with the gate chain, whose 48 elements a token ride
    // along for free.
    //
    // It writes the conv output TWICE: `conv_out` is [T][GDN_CONV_DIM], the
    // reference's contiguous [q|k|v] row (the oracle tap, and where v is
    // read from), and `conv_qk` is [T][2*GDN_KEY_DIM], q and k alone. The
    // second copy costs 4 096 floats a token and buys l2_norm a buffer whose
    // 32*T groups of 128 are contiguous -- in the [q|k|v] layout the k of
    // token t and the q of token t+1 are 6 144 floats apart.
    virtual void gdn_conv_gate(const float * win, const float * w,
                               const float * beta_raw, const float * alpha_raw,
                               const float * dt, const float * ssm_a,
                               float * conv_out, float * conv_qk, float * beta_sig,
                               float * a_softplus, float * gate, float * g_exp,
                               int T = 1) = 0;
    // qwen4exp.cpp:816-834's gate chain -- sigmoid(beta), softplus(alpha+dt),
    // *ssm_a, exp -- as ONE op over 48 elements instead of five. It still
    // writes a_softplus and gate, so the oracle taps are unchanged.
    // The delta-rule recurrence over the chunk. `g_exp` is already exp(g)
    // per v-head per token. `q`/`k` are strided by `qk_stride` a token and
    // `v` by `v_stride`; `out` is [T][GDN_VAL_DIM]. The chunk is walked in
    // order inside the kernel with the state row in registers, so T steps
    // produce exactly what T calls at T = 1 produce.
    virtual void gdn_step(float * state, const float * q, const float * k, const float * v,
                          const float * g_exp, const float * beta, float * out,
                          int T = 1, int qk_stride = 0, int v_stride = 0) = 0;
    // rms_norm(x)*w * sigmoid(z), per head (build_norm_gated).
    virtual void gated_rms_norm(const float * x, const float * w, const float * z,
                                float * y, int ne0, int n_groups, float eps) = 0;

    // -- MoE (build_moe_ffn) -------------------------------------------------
    // softmax over N_EXPERT, top-N_EXPERT_USED, gather, normalise (clamped).
    // `ids_log` / `w_log` (nullable) receive a copy for the routing oracle,
    // written by this kernel so capture costs no extra launch.
    virtual void router(const float * logits, int * ids, float * weights,
                        int * ids_log, float * w_log, int T = 1) = 0;
    // `ids` is [T][N_EXPERT_USED], `x` is [T][K] and the outputs are
    // [T][N_EXPERT_USED][rows]: assignment (t,k) owns column t*K_TOP+k.
    virtual void moe_gate_up(const Mat & gate, const Mat & up, const int * ids,
                             const float * x, float * y_gate, float * y_up, int T = 1) = 0;
    virtual void silu_mul(const float * gate, const float * up, float * h, size_t n) = 0;
    virtual void moe_down(const Mat & down, const int * ids, const float * h,
                          float * expert_out, int T = 1) = 0;
    // build_moe_ffn's tail in one op: the weighted expert sum, the shared
    // expert's gate and the add. Writes ffn_moe_out and ffn_shexp_gated
    // because both are oracle taps.
    // `sh_gate` is the RAW shared-expert logit; the sigmoid is applied here,
    // so build_moe_ffn's shared_expert_gate and shared_expert_gate_sigmoid
    // are both taps and neither costs a launch.
    virtual void moe_finish(const float * expert_out, const float * weights,
                            const float * sh_raw, const float * sh_gate,
                            float * sh_gate_sig, float * moe_out,
                            float * sh_gated, float * out, int T = 1) = 0;

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
    // `k`/`v` are [T][N_KV_HEADS*HEAD_DIM]; row t lands in cell `cell0 + t`.
    // PREFILL.md section 3: EVERY row of the chunk is in the cache before
    // anything is scored, which is what makes a chunk's selection identical
    // to T single-token steps' (and is what the reference does too).
    virtual void kv_store_q8_0(const float * k, const float * v,
                               int8_t * kqs, uint16_t * ksc,
                               int8_t * vqs, uint16_t * vsc, int cell0, int T = 1) = 0;
    virtual bool verify_placement(FILE *) { return true; }
    // Mean-pool cells [blk*r, blk*r+r) (missing slots read cell 0, as
    // set_input_qsa's zero-filled blk_cells does), rms-norm with k_norm,
    // IMRoPE at position blk*r, store bf16 into the pooled cache.
    // Mean-pool the block this token joined, norm, IMRoPE at the block's
    // first position, store bf16.
    //
    // It used to keep every token's RAW indexer key in f32 to re-pool from --
    // 512 B a token a layer, EIGHT TIMES the 64 B design 9.1 budgets for the
    // pooled bf16 key, and 134 MB a layer at 256k. It does not need them: a
    // block's members arrive in order, so a running SUM plus cell 0's key (the
    // value set_input_qsa's zero-filled blk_cells gives every unset slot) is
    // all the reference's formula reads:
    //
    //   pooled = (sum_of_members_so_far + (r - n_filled) * key_of_cell_0) / r
    //
    // `sum` and `raw0` are IDX_DIM floats a layer, not a cache.
    // `dbg_pooled` / `dbg_roped` receive this block's value before and after
    // the norm+rotation -- the reference's `indexer_k_pooled` and `indexer_k`
    // taps, both of which are incomparable across cache depths anyway.
    //
    // `k_new` is [T][IDX_DIM], the chunk's raw indexer keys, and `pos0` is
    // the position of its first row. The T cells are folded into the running
    // sum IN ORDER inside one call, so every block the chunk touched is
    // re-pooled and the arithmetic is the same sequence T single-token calls
    // perform -- `sum` and `raw0` are what carry a partial block across a
    // chunk boundary. `dbg_*` receive the LAST row's block.
    virtual void idx_pool_chunk(const float * k_new, float * sum, float * raw0,
                                int pos0, int T, const float * k_norm,
                                float eps, uint16_t * pooled,
                                float * dbg_pooled, float * dbg_roped) = 0;
    virtual void idx_scan(const uint16_t * pooled, const float * q, float * scores, int n_blocks) = 0;
    // cell_scores[j] = (j <= q_pos) ? blk_scores[j/r] + bias(j/r) : -inf, with
    // set_input_qsa's per-block bias computed IN the kernel from tail_start:
    // +1e9 for a block at or past the incomplete tail, -inf for a block that
    // could not be pooled, 0 otherwise (llama-memory-hybrid-idx.cpp:438-447).
    // It used to be a host-side array uploaded per QSA layer, which put a
    // BLOCKING hipMemcpy inside the 16-layer body -- see decode_graph.cpp.
    // Also writes the identity selection when the budget covers the whole
    // cache, which is when the reference's top-k is the identity too -- then
    // topk_select is not called at all.
    virtual void qsa_expand(const float * blk_scores, float * cell_scores, int * sel,
                            int n_kv, int q_pos, int ratio, int tail_start,
                            int sel_identity) = 0;
    // Returns how many were selected (min(width, visible)); fills `out`.
    virtual int  topk_select(const float * scores, int n, int width, int * out) = 0;
    // Size the top-k candidate lists ONCE, from --ctx. They are indexed by
    // cache depth, which grows by one cell a token, so growing them on demand
    // meant a hipMalloc AND a hipFree per QSA layer per token -- and hipFree
    // SYNCHRONISES THE DEVICE, so each one drained the queue and the host sat
    // waiting for whatever was outstanding. That is why the cost appeared only
    // once the cache passed the budget (below it the selection is the identity
    // and topk_select is never called) and why it grew with depth: the deeper
    // the cache, the more queued work each of those 48 syncs a token waited on.
    virtual void reserve_topk(int n) { (void) n; }
    // build_layer_attn's per-head post-processing in one op: the [q|gate]
    // split of attn_q's interleaved rows, the QK-norms, IMRoPE on q, k and
    // the indexer query, and the output gate's sigmoid.
    // All buffers token-major; row t is rotated at position `pos0 + t`.
    virtual void qsa_qk_post(const float * qfull, const float * kraw, const float * idxraw,
                             const float * q_norm, const float * k_norm, const float * iq_norm,
                             float * qcur, float * gate, float * gsig,
                             float * kcur, float * idxq, int pos0, float eps, int T = 1) = 0;
    // `out` is the reference's kqv_out / attn_pregate, `out_gated` is
    // attn_gated: the sigmoid gate is applied in the combine, so neither tap
    // costs a launch of its own.
    virtual void attn_qsa(const int8_t * kqs, const uint16_t * ksc,
                          const int8_t * vqs, const uint16_t * vsc,
                          const float * q, const int * sel, int n_sel,
                          const float * gsig, float * out, float * out_gated) = 0;

    // ---- the chunk's query rows, batched (PREFILL.md section 11) ----------
    //
    // The four calls above used to be driven by a host loop of T rows in
    // decode_graph.cpp: at T = 256 that is ~1 000 launches a QSA layer and
    // ~12 000 a chunk, which the profile charged 0.19 ms a prompt token a
    // card -- all issue cost, no work. Row t differs from row t+1 only in its
    // POSITION, and every quantity the loop computed from it (n_kv, n_blocks,
    // tail_start, the budget, whether the top-k is the identity) is a closed
    // form of that position. So the rows can be a grid axis.
    //
    // The rows of a block are INDEPENDENT: same kernels, same per-row
    // arithmetic, more workgroups. Nothing here changes a summation order.
    //
    // The default implementation below IS the loop the runner used to write.
    // A backend that does not care -- the CPU one, which is what the
    // chunk-vs-decode oracle runs on -- is unchanged by construction, so the
    // oracle keeps its meaning and the GPU override is the only thing under
    // test.
    //
    // A block is bounded by SCRATCH, not by T: cell_scores is O(n_kv) a row,
    // so 256 rows of a 256k cache would be 268 MB of them and 537 MB of
    // radix-select candidates. reserve_qsa_rows() says how many rows a
    // backend has room for and the runner walks the chunk in blocks of that
    // many.
    struct QsaRows {
        const uint16_t * pooled;                      // pooled indexer keys
        const int8_t   * kqs;  const uint16_t * ksc;
        const int8_t   * vqs;  const uint16_t * vsc;
        const float    * idxq;        // [rows][IDX_N_HEADS][IDX_DIM]
        const float    * q;           // [rows][N_Q_HEADS][HEAD_DIM]
        const float    * gsig;        // [rows][N_Q_HEADS][HEAD_DIM]
        float * blk;                  // [rows][blk_stride]   scratch, and the
        float * cell;                 // [rows][cell_stride]  last row of the
        int   * sel;                  // [rows][sel_stride]   chunk is tapped
        float * out;                  // [rows][N_Q_HEADS][HEAD_DIM]
        float * out_gated;
        int    rows;                  // rows in THIS block
        int    pos0;                  // position of row 0 of THIS block
        int    ratio;                 // the layer's compress ratio
        int    budget;                // IDX_TOP_K + ratio - 1
        size_t blk_stride, cell_stride, sel_stride;
    };
    virtual void qsa_rows(const QsaRows & j) {
        for (int t = 0; t < j.rows; ++t) {
            const int p          = j.pos0 + t;
            const int n_kv       = p + 1;
            const int n_blocks   = (n_kv + j.ratio - 1) / j.ratio;
            const int tail_start = ((p + 1) / j.ratio) * j.ratio;
            const int width      = n_kv < j.budget ? n_kv : j.budget;
            const int ident      = (width >= n_kv) ? 1 : 0;
            float * blk  = j.blk  + (size_t) t * j.blk_stride;
            float * cell = j.cell + (size_t) t * j.cell_stride;
            int   * sel  = j.sel  + (size_t) t * j.sel_stride;
            idx_scan(j.pooled, j.idxq + (size_t) t * IDX_N_HEADS * IDX_DIM, blk, n_blocks);
            qsa_expand(blk, cell, sel, n_kv, p, j.ratio, tail_start, ident);
            const int n_sel = ident ? n_kv : topk_select(cell, n_kv, width, sel);
            const size_t qo = (size_t) t * HEAD_DIM * N_Q_HEADS;
            attn_qsa(j.kqs, j.ksc, j.vqs, j.vsc, j.q + qo, sel, n_sel,
                     j.gsig + qo, j.out + qo, j.out_gated + qo);
        }
    }
    // How many rows of a chunk this backend has scratch for. Called ONCE from
    // the runner's constructor with the widest chunk and the context, and it
    // is where a backend allocates whatever it has to size by the row count
    // (the attention split's partials, the radix select's candidate lists).
    // May return anything in [1, rows_max]; the runner walks the chunk in
    // blocks of the answer, so a backend that returns 1 gets today's loop.
    virtual int reserve_qsa_rows(int rows_max, int ctx) {
        // The default runs the loop above, so only the runner's own per-row
        // scratch has to fit: cell scores, block scores, the selection. At
        // the oracle's ctx that is ~11 KB a row and the whole chunk batches,
        // which is what makes the chunk-vs-decode gate cover this code at
        // all; at 256k it is 1.3 MB a row and the block is ~48.
        const size_t per_row = (size_t) ctx * sizeof(float)
                             + (size_t) ((ctx + QSA_RATIO - 1) / QSA_RATIO) * sizeof(float)
                             + (size_t) MAX_SEL * sizeof(int);
        const size_t budget = (size_t) qsa_row_mb_ * 1024u * 1024u;
        int rows = per_row ? (int)(budget / per_row) : rows_max;
        if (rows < 1)        rows = 1;
        if (rows > rows_max) rows = rows_max;
        return rows;
    }
    // --qsa-row-mb N / FRANKEN_QSA_ROW_MB: the scratch budget, per device,
    // for one block of query rows. It is what bounds the block, because
    // cell_scores is O(n_kv) a row: 256 rows of a 256k cache would be 268 MB
    // of them and 537 MB of radix candidates behind them. Raising it buys
    // fewer launches at 256k and nothing at all below ~2k, where the whole
    // chunk already fits.
    virtual void set_qsa_row_mb(int mb) { if (mb > 0) qsa_row_mb_ = mb; }

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
    // --gemv-lds 0|1: stage the activation slice in LDS or read it from L1.
    // A runtime switch so both arms are measurable with one binary -- which
    // of them wins on a weight-bandwidth-bound kernel is a question for the
    // profile, not an assumption.
    virtual void set_gemv_lds(int) {}
    // Below this many output ROWS a GEMV splits K to raise the wave count
    // (total waves == total rows, so wider workgroups never help). The
    // threshold trades a reduce launch against occupancy, and the profile's
    // per-group GB/s is what should set it, so it is a runtime knob.
    virtual void set_gemv_min_rows(int) {}
    // --gemv-fused-reduce 0|1: sum a split GEMV's partials in the GEMV kernel
    // (one counter per output row, the last workgroup reduces) instead of in a
    // second launch. The sum is k_reduce_splits_gemm's, in s order, so the
    // result is bit-identical; the knob exists because the pairing of
    // __threadfence with the counter is new here, not because a bit can move.
    virtual void set_gemv_fused_reduce(int) {}
    // --gemv-fuse-collapse 0|1: fold build_hc_mix's collapse into the gate
    // GEMV (gemv_hc_gate_collapse above).
    virtual void set_gemv_fuse_collapse(int) {}
    // --gemv-burst 1|2|4: how many GEMV_UNROLL groups of Q8_0 blocks a decode
    // wave loads before it multiplies any of them. The ACCUMULATOR count does
    // not move with it -- block g stays in acc[g % GEMV_UNROLL] and the order
    // within each accumulator stays increasing g -- so this is memory-level
    // parallelism at constant summation order. Only the T == 1 instantiation
    // takes it; the prompt chunk's TILE = 8 kernel is already at 135 VGPR.
    virtual void set_gemv_burst(int) {}
    // --gemm-lds MODE: which kernel serves a T > 1 trunk GEMM.
    //   0  k_gemm_batch<GEMM_TILE> -- the wave-per-row kernel, bit-identical
    //      to the decode token's TILE = 1 instantiation per column. DEFAULT.
    //   1  k_gemm_lds -- an LDS-tiled GEMM: a 64x64 (row x token) tile per
    //      workgroup, the weight block decoded ONCE per 64 activation
    //      columns instead of once per 8, f32 accumulation per thread over K.
    //   2  k_gemm_lds_i8 -- the same tiling with the activation tile
    //      quantised to int8 per 32-block and RDNA3's v_dot4_i32_iu8
    //      (__builtin_amdgcn_sudot4, which DOES compile for gfx1100 where
    //      __builtin_amdgcn_sdot4 does not). Falls back to mode 1 for a batch
    //      that is not all-Q8_0.
    // Modes 1 and 2 accumulate K linearly per thread instead of across a
    // wave's lanes, so they are a DIFFERENT SUMMATION ORDER and cannot be
    // bit-identical to decode. That is why they are a knob and the default is
    // 0: the chunk-vs-decode oracle keeps its meaning unless the knob is on,
    // and with it on the orchestrator measures how far the last bits moved.
    virtual void set_gemm_lds(int) {}
    // --expert-gather MASK: which expert stages take design 9.4 item 5's
    // device-side sort and row-gather at T > 1. Bit 0 is gate/up, bit 1 is
    // down. A stage that is off runs every assignment on its own, exactly as
    // a decode token does.
    //
    // THE DEFAULT IS 3 SINCE 2026-09-23, AND EVERY BIT OF THAT IS A
    // MEASUREMENT, not a judgement. On three cards, 2026-09-22, --chunk 6
    // against --chunk 1:
    //
    //   mask 0  bit-identical, greedy 12/16 as decode
    //   mask 1  bit-identical            <- gate/up is exact
    //   mask 2  diverged from layer 1 on (Kcur-11 1.7e-6 ... Kcur-27 1.0e-3)
    //   mask 3  diverged the same way
    //
    // so the DOWN gather alone was at fault, and the mask is per STAGE
    // because that is what turned "the gather is wrong" into "the down
    // gather is wrong" in two runs. Everything else was cleared by the same
    // bisection: --chunk 1 against its own dump is identical (the engine is
    // deterministic) and --gemv-lds 0 on both arms changes nothing.
    //
    // WHY THE DOWN GATHER WAS WRONG, found 2026-09-23 in the ISA rather than
    // on a card. moe_gather_test.cpp had already cleared the algorithm, the
    // indexing, the accumulator structure and the reduction by transcribing
    // both paths on the host over the SAME shared primitives and getting bit
    // equality; the sort is independently proved on the device by the mask-1
    // arm, which shares it. What was left was code generation, and it is:
    // `lo += (d * kv) * hb[c]` inside m1n_iq4nl_chunk_dot is CONTRACTED into
    // an FMA in one kernel and split into v_mul_f32 + v_add_f32 in the other.
    // Both are legal under -ffp-contract=fast and they round differently. The
    // backend contracts when it has ILP to spare (the tiled gather had eight
    // independent accumulator pairs) and de-contracts to shorten the
    // dependency chain when it has one (the per-assignment kernel). Giving
    // the gather ONE accumulator pair -- its m loop outer, its w loop inner,
    // scalars -- makes the two emit the same multiset of float ops:
    // 8 v_fma_mix_f32, 1 v_fmac_f32, 7 v_mul_f32, 13 v_add_f32, against the
    // tiled form's 8 / 16 v_fma_f32 / 48 v_fmac_f32 / 48 v_add_f32. The VRAM
    // read is still once a tile (the row is 360 B and the re-reads are L1),
    // so the amortisation is kept.
    //
    // The Q8_0 gather never needed this and still tiles: against its twin it
    // is the per-assignment body times eight op for op, every term fused.
    virtual void set_expert_gather(int) {}
    // --moe-tile N / FRANKEN_MOE_TILE: how many assignments of one expert a
    // wave of the row-gather takes at once. 4, 8 (the default) or 16; the
    // gather kernels are templated on it.
    //
    // What it trades. At T = 256 a chunk has 2 560 assignments over at most
    // 512 experts, so an expert averages FIVE and a tile of 8 is five-eighths
    // full -- but its `a[TILE][4]` accumulators cost registers whether they
    // are used or not, and k_moe_gate_up_iq3s_gather is at 213 VGPR, which is
    // 7 waves a SIMD on a kernel that should be weight-bandwidth bound. A
    // smaller tile reads the expert row more times (from L1/L2, not VRAM) and
    // gets more waves in flight to hide the latency of the reads that do go
    // to VRAM. Which way that lands is a measurement, and this is the knob
    // that takes it.
    //
    // It CANNOT move a bit, whatever it is set to: an assignment's column
    // accumulates in the same order whichever tile it lands in. That is the
    // design of these kernels and, since 2026-09-23, something the ISA is
    // checked for rather than asserted (PREFILL.md section 8.1).
    virtual void set_moe_tile(int) {}
    // --expert-gather-serial 0|1 / FRANKEN_GATHER_SERIAL: which SHAPE the
    // gate/up row-gather takes for the assignments of a tile.
    //
    //   0 (default)  the tile: the w/block loop outside, TE accumulator sets
    //                inside, so a decoded weight block is reused from
    //                REGISTERS across the tile. Measured bit-identical on the
    //                device, 2026-09-22, and 213 VGPR / 7 waves a SIMD for
    //                IQ3_S -- which is the problem, on a kernel that should be
    //                waiting for VRAM.
    //   1            the m loop outside, four (IQ3_S) or two (IQ4_XS) scalar
    //                accumulators inside: literally the per-assignment
    //                kernel's body with a different activation row, so its
    //                94 / 65 VGPR and 16 waves a SIMD. The expert row is then
    //                re-read once a tile member, from L1 rather than VRAM.
    //
    // This is the same trade the down gather has no choice about (PREFILL.md
    // section 8.1 -- there the tiled shape was not bit-identical, because the
    // accumulate lives inside a shared primitive and the backend contracted it
    // differently under the two register pressures). Here both shapes are
    // exact, and which is FASTER is a measurement nobody has taken.
    virtual void set_gather_serial(int) {}
    // --sync-debug: drain and check after every launch and copy, so the op
    // named in a fault message is the one that faulted rather than whichever
    // launch was in flight when the queue drained.
    virtual void set_sync_debug(bool) {}
    virtual void set_debug_context(const char * phase, int layer) { (void) phase; (void) layer; }
    virtual void prof_reset() {}
    virtual void prof_end_token() {}
    // --prefill-pipeline: prof_end_token must NOT synchronise, or the profile
    // would serialise the very overlap it is there to measure. Deferred, the
    // closing event of a chunk stays in the pool and the interval from it to
    // the next chunk's first op is charged to PC_GAP -- which is exactly this
    // card's idle time, so `busy = total - gap` per device is the overlap.
    // prof_flush() resolves everything outstanding; it is the only sync.
    virtual void prof_defer(bool on) { (void) on; }
    virtual void prof_flush() {}
    virtual void prof_report(FILE * out, int n_tokens) { (void) out; (void) n_tokens; }
    // Launches this device enqueued inside the last token's body. The runner
    // divides its own host wall time by this to get the per-launch issue cost,
    // which is the number that says whether a card is compute-bound or waiting
    // for the host to feed it.
    virtual long long launches_last_token() const { return 0; }

protected:
    int qsa_row_mb_ = 256;
};

Backend * make_cpu_backend(int n_threads);
Backend * make_gpu_backend(int device);
// Must run before anything is placed; a no-op in the CPU-only build.
void enable_peer_access(int n_devices);

} // namespace fk
