// tools/hot-expert/franken/decode/ds4_ops.h
//
// The DeepSeek-V4-specific operations of the decode token, as an interface,
// in the same discipline as decode_backend.h: ds4_graph.cpp holds THE MATH
// (the order of the ops, which buffer feeds which) and never dereferences a
// buffer; it calls these, plus the generic half of Backend (alloc, copy,
// rms_norm_mul, binary, scale, upload/download, argmax, boundary copies).
//
//   Ds4CpuOps (ds4_cpu.cpp)   plain host C++ -- the oracle arm (L5 step 1).
//   Ds4GpuOps (ds4_gpu.inc)   HIP kernels, compiled into decode_gpu.hip's
//                             translation unit so they run on the backend's
//                             own stream (L5 step 2). One instance per card.
//
// Every pointer crossing this interface is a BACKEND pointer (host memory for
// the CPU arm, a device address on a card), including the expert ids, the
// hash-routing ids and the top-k output: nothing here makes the host wait.
// Every count the kernels need is a closed form of the position the host
// already knows (raw window length, visible blocks, min(top_k, visible)).
//
// Buffers are one token wide (T == 1): the batched prefill is a later step.

#pragma once

#include <cstddef>
#include <cstdint>

#include "decode_backend.h"   // Mat

namespace fk {
namespace ds4 {

// ggml_rope_ext's float parameters, as deepseek4.cpp passes them per layer.
struct RopeParams {
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;
    int   n_ctx_orig  = 0;
};

// One layer's routed experts as the kernels read them: for each of the 256
// experts, the base address of its gate, up and down slice. On a card an
// entry points into VRAM (resident) or into the device-mapped view of a
// pinned host copy (a miss) -- the SAME kernel reads either, so a miss costs
// a PCIe read and no host round trip (DEEPSEEK4.md section 5). The three
// arrays are backend memory, 256 pointers each.
struct ExpertTable {
    const void * const * gate = nullptr;
    const void * const * up   = nullptr;
    const void * const * down = nullptr;
    int    type_gu = 0, type_d = 0;          // FkQuantType
    size_t row_gu  = 0, row_d  = 0;          // bytes per row
    int    K_gu    = 0, rows_gu = 0;         // 4096 -> 2048
    int    K_d     = 0, rows_d  = 0;         // 2048 -> 4096
    // The miss path (L5 step 3): per expert, the bytes a use of it has to
    // bring over PCIe -- its slab when it lives in pinned host memory, 0 when
    // it is resident (backend memory, 256 ints) -- and the three slice sizes,
    // which is how a staging copy lays a slab out: gate | up | down.
    const int * miss_bytes = nullptr;
    size_t sz_g = 0, sz_u = 0, sz_d = 0;
};

class Ds4Ops {
public:
    virtual ~Ds4Ops() {}

    // -- GEMV in any format the file carries (the six new ones included) ----
    virtual void gemv(const Mat & W, const float * x, float * y) = 0;

    // -- hyper-connections ---------------------------------------------------
    virtual void hc_split(const float * mixes, const float * scale3, const float * base24,
                          float * pre, float * post, float * comb, float eps, int iters) = 0;
    virtual void hc_head_pre(const float * mixes4, const float * scale1, const float * base4,
                             float * pre, float eps) = 0;
    virtual void hc_weighted_sum(const float * H, const float * w, float * out) = 0;
    // hc_init: the embedding row repeated into the four streams. An op of its
    // own because Backend::copy on a card is a NULL-stream memcpy, which a
    // stream capture cannot contain.
    virtual void hc_init(const float * x, float * H) = 0;
    virtual void hc_post(const float * x, const float * H, const float * post,
                         const float * comb, float * Hout) = 0;

    // -- POSITION-SEMANTIC OPS (L5 step 3b) -------------------------------------
    // Every op below takes the token's position `pos` and derives from it
    // whatever it needs -- the raw-window slot, the compressor ring slot, the
    // ape row, whether a block completed and which, how many blocks are
    // visible, the attention extents. On a card, while a hipGraph is being
    // captured, the kernels read the position from a device int instead of
    // the argument (Ds4GpuOps: GpuBackend's gpos), so ONE captured graph is
    // right for every position of its class; eagerly they take `pos` as is.
    // The block-completing ops are no-ops when no block completed at `pos`,
    // which is what lets a graph always contain them.

    // Rotate at `pos`, or at the first position of the block that ends at
    // `pos` when block_ratio > 0 (a compressed row). NORM pairs, last N_ROT
    // dims of every row; `inverse` is ggml_rope_ext_back.
    virtual void rope_tail(float * x, int n_rows, int row_len, int pos, int block_ratio,
                           const RopeParams & rp, bool inverse) = 0;
    virtual void fwht(float * x, int n_rows, int n) = 0;
    // f16 into the raw window, slot pos % N_SWA
    virtual void store_raw(const float * kv, uint16_t * raw_ring, int pos) = 0;
    // f16 into cache row pos / ratio -- only when (pos + 1) % ratio == 0
    virtual void store_block(const float * x, uint16_t * cache, int width, int ratio, int pos) = 0;
    // ring[pos % ring_rows][0..width) = src
    virtual void ring_put(float * ring, int ring_rows, int width, int pos, const float * src) = 0;
    // x[0..width) += table[pos % ratio][0..width)   (the compressor's ape row)
    virtual void add_row(float * x, const float * table, int width, int ratio, int pos) = 0;

    // -- compressor pooling: the block that ends at `pos`, if one does ---------
    virtual void comp_pool(const float * ring_kv, const float * ring_sc, int ring_rows,
                           int ratio, int d_out, int pos, float * out) = 0;
    virtual void comp_pool_overlap(const float * ring_kv, const float * ring_sc, int ring_rows,
                                   int ratio, int d_out, int pos, float * out) = 0;

    // -- the lightning indexer: the (pos+1)/4 visible blocks ---------------------
    virtual void lid_scores(const float * q, const float * w, const uint16_t * keys, int pos,
                            float * scores) = 0;
    // The top min(k, n) of the n = (pos+1)/4 scores as a SET (order unspecified,
    // as ggml_top_k's); n <= k writes the identity 0..n-1.
    virtual void topk(const float * scores, int pos, int k, int * out) = 0;

    // -- attention: 64 query heads against one shared K=V head ------------------
    // Keys: the raw window of `pos` (positions max(0, pos-127) .. pos) then
    // n_comp = min((pos+1)/ratio, cap) compressed rows (ids, or 0.. when
    // `comp_ids` is null); ratio 0 = the raw window alone.
    virtual void attn(const float * q, const uint16_t * raw_ring, int pos, const uint16_t * comp,
                      int ratio, int cap, const int * comp_ids, const float * sinks, float scale,
                      float * out) = 0;
    // Size whatever the attention needs for positions < ctx, once (a card must
    // not allocate inside a captured graph).
    virtual void reserve(int ctx) { (void) ctx; }

    // -- MoE ------------------------------------------------------------------
    // `hash_ids` (6 ids, backend memory) replaces the top-6 on hash layers.
    virtual void router(const float * logits, const float * bias, const int32_t * hash_ids,
                        float * probs, float * probs_biased, int * ids, float * w_raw,
                        float * w_norm, float * w_scaled) = 0;
    // THE MISS PATH (L5 step 3). moe_stage() starts bringing layer t's six
    // chosen experts to the card for staging `slot` and returns at once: a
    // resident expert is used where it is, a missed one is copied into a VRAM
    // staging ring on a side stream, overlapping whatever the caller enqueues
    // next (the shared expert; for the token-id-routed layers, everything
    // from the embedding on). moe_gate_up / moe_down with the same `slot`
    // wait for that copy and read the staged addresses; slot -1 reads every
    // expert through the table in place (a miss then is a PCIe read inside
    // the GEMV -- step 2's path, --miss-stage 0). The bytes are the same
    // either way, so the numerics are too.
    virtual void reserve_moe(size_t max_slab_bytes, int n_slots) { (void) max_slab_bytes; (void) n_slots; }
    virtual void moe_stage(const ExpertTable & t, const int * ids, int slot) { (void) t; (void) ids; (void) slot; }
    // y_gate / y_up [6][rows_gu] for the six experts `ids` names.
    virtual void moe_gate_up(const ExpertTable & t, const int * ids, const float * x,
                             float * y_gate, float * y_up, int slot) = 0;
    // y [6][rows_d]; slot k reads h + k*K_d.
    virtual void moe_down(const ExpertTable & t, const int * ids, const float * h, float * y,
                          int slot) = 0;
    virtual void swiglu_clamp(const float * gate, const float * up, float * h, int n,
                              float limit) = 0;
    virtual void moe_accum(const float * y, const float * w, int n_used, int n,
                           float * weighted, float * out) = 0;

    // -- profiling (--profile) ------------------------------------------------
    // Miss bytes counted ON THE DEVICE from the ids the router chose (no host
    // round trip): count_misses() adds the six experts' miss_bytes to a
    // device counter; miss_bytes_total() reads it (a sync -- report time only).
    virtual void set_profile(bool on) { (void) on; }
    // --staged-loads 0|1 (GPU): a GEMV row through LDS with 16-byte loads (1,
    // default) or read in place with the decoders' byte loads (0). Same
    // decoder on the same bytes: the pair is a bit-identity check.
    virtual void set_staged_loads(int on) { (void) on; }
    virtual void count_misses(const ExpertTable & t, const int * ids) { (void) t; (void) ids; }
    virtual double miss_bytes_total() { return 0.0; }
    virtual void reset_miss_count() {}

    // -- memory the kernels read over PCIe -------------------------------------
    // Pinned, device-mapped host memory; returns the view a kernel on THIS
    // card reads, and the host address in *host. The CPU arm returns plain
    // host memory for both.
    virtual const void * alloc_host_mapped(size_t bytes, void ** host) = 0;
};

Ds4Ops * make_ds4_cpu_ops(Backend & be);
// decode_gpu.hip (the HIP build) or decode_nogpu.cpp (throws)
Ds4Ops * make_ds4_gpu_ops(Backend & be);

} // namespace ds4
} // namespace fk
