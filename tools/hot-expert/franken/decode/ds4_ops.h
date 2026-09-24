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
// ROWS (L5 step 5, batched prefill; DEEPSEEK4.md section 13). Every per-token
// op takes `T`, the rows of a chunk, token-major (row t at `buf + t * width`),
// and every position-semantic op's `pos` is the FIRST row's position: row t
// is at pos + t. T == 1 is the decode token, and the kernels are the same ones
// (one kernel per op, the rows a grid axis or a non-unrolled loop), so a
// chunk's row t computes exactly what a decode step at pos + t computes.

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

// -- adaptive placement (design ladder L2; DEEPSEEK4.md section 12) ------------
// ROUTE STATISTICS, written by the router kernel itself (no host sync): per
// layer of a card, ADAPT_STRIDE uint32 -- [e] how often expert e was chosen,
// [ADAPT_MISS] how many choices found their expert host-mapped (a miss, by
// the table as it stood when the router ran), [ADAPT_PICKS] every choice.
// Cumulative from zero; the host reads them asynchronously every N tokens and
// keeps the decayed average itself (ds4_adapt.cpp).
constexpr int ADAPT_STRIDE = 260;
constexpr int ADAPT_MISS   = 256;
constexpr int ADAPT_PICKS  = 257;
// A batch of expert-table entries to rewrite, passed BY VALUE to one tiny
// kernel on the main stream, so no host buffer has to outlive the launch.
constexpr int ADAPT_MAX_EDITS = 24;
struct TableEdit {
    int n = 0;
    int e[ADAPT_MAX_EDITS];
    const void * g[ADAPT_MAX_EDITS];
    const void * u[ADAPT_MAX_EDITS];
    const void * d[ADAPT_MAX_EDITS];
    int miss[ADAPT_MAX_EDITS];
};

class Ds4Ops {
public:
    virtual ~Ds4Ops() {}

    // -- GEMV in any format the file carries (the six new ones included) ----
    // T > 1: x [T][K], y [T][rows] -- a GEMM whose column t is the T == 1 GEMV
    virtual void gemv(const Mat & W, const float * x, float * y, int T = 1) = 0;

    // -- hyper-connections ---------------------------------------------------
    // rows: mixes [T][HC_MIX], pre/post [T][HC], comb [T][HC*HC]
    virtual void hc_split(const float * mixes, const float * scale3, const float * base24,
                          float * pre, float * post, float * comb, float eps, int iters,
                          int T = 1) = 0;
    virtual void hc_head_pre(const float * mixes4, const float * scale1, const float * base4,
                             float * pre, float eps) = 0;
    // rows: H [T][HC_DIM], w [T][HC], out [T][N_EMBD]
    virtual void hc_weighted_sum(const float * H, const float * w, float * out, int T = 1) = 0;
    // hc_init: the embedding row repeated into the four streams. An op of its
    // own because Backend::copy on a card is a NULL-stream memcpy, which a
    // stream capture cannot contain.
    virtual void hc_init(const float * x, float * H, int T = 1) = 0;
    virtual void hc_post(const float * x, const float * H, const float * post,
                         const float * comb, float * Hout, int T = 1) = 0;

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
    // dims of every row; `inverse` is ggml_rope_ext_back. rows_per_pos > 0
    // (a chunk): row r is at pos + r / rows_per_pos; 0: every row at pos.
    virtual void rope_tail(float * x, int n_rows, int row_len, int pos, int block_ratio,
                           const RopeParams & rp, bool inverse, int rows_per_pos = 0) = 0;
    virtual void fwht(float * x, int n_rows, int n) = 0;
    // f16 into the raw window, slot pos % raw_rows (raw_rows = N_SWA + the
    // largest chunk - 1: a chunk's later rows must not overwrite a slot an
    // earlier row still attends; values are addressed by POSITION, so the
    // ring's size changes no number)
    virtual void store_raw(const float * kv, uint16_t * raw_ring, int raw_rows, int pos, int T = 1) = 0;
    // f16 into cache row p / ratio -- only for the rows with (p + 1) % ratio == 0
    virtual void store_block(const float * x, uint16_t * cache, int width, int ratio, int pos,
                             int T = 1) = 0;
    // ring[p % ring_rows][0..width) = src row, p = pos + t
    virtual void ring_put(float * ring, int ring_rows, int width, int pos, const float * src,
                          int T = 1) = 0;
    // x[t][0..width) += table[p % ratio][0..width)   (the compressor's ape row)
    virtual void add_row(float * x, const float * table, int width, int ratio, int pos,
                         int T = 1) = 0;
    // dst[r * dst_stride + i] = src[r * src_stride + i], i < row_len: the
    // grouped output projection's per-group column gather/scatter in a chunk
    virtual void copy_rows(const float * src, size_t src_stride, float * dst, size_t dst_stride,
                           int n_rows, int row_len) = 0;

    // -- compressor pooling: the block that ends at `p`, if one does ---------
    // (rows: out [T][d_out], row t written only when a block ends at pos + t)
    virtual void comp_pool(const float * ring_kv, const float * ring_sc, int ring_rows,
                           int ratio, int d_out, int pos, float * out, int T = 1) = 0;
    virtual void comp_pool_overlap(const float * ring_kv, const float * ring_sc, int ring_rows,
                                   int ratio, int d_out, int pos, float * out, int T = 1) = 0;

    // -- the lightning indexer: the (p+1)/4 visible blocks ---------------------
    // rows: q [T][64*128], w [T][64], scores [T][score_stride]
    virtual void lid_scores(const float * q, const float * w, const uint16_t * keys, int pos,
                            float * scores, int T = 1, int score_stride = 0) = 0;
    // The top min(k, n) of the n = (p+1)/4 scores as a SET (order unspecified,
    // as ggml_top_k's); n <= k writes the identity 0..n-1. rows: out [T][k].
    virtual void topk(const float * scores, int pos, int k, int * out, int T = 1,
                      int score_stride = 0) = 0;

    // -- attention: 64 query heads against one shared K=V head ------------------
    // Keys: the raw window of p (positions max(0, p-127) .. p, slot q % raw_rows)
    // then n_comp = min((p+1)/ratio, cap) compressed rows (ids, or 0.. when
    // `comp_ids` is null); ratio 0 = the raw window alone. rows: q/out
    // [T][64*512], comp_ids [T][IDX_TOP_K].
    virtual void attn(const float * q, const uint16_t * raw_ring, int raw_rows, int pos,
                      const uint16_t * comp, int ratio, int cap, const int * comp_ids,
                      const float * sinks, float scale, float * out, int T = 1) = 0;
    // Size whatever the attention needs for positions < ctx and chunks of up
    // to t_max rows, once (a card must not allocate inside a captured graph).
    virtual void reserve(int ctx, int t_max = 1) { (void) ctx; (void) t_max; }

    // -- MoE ------------------------------------------------------------------
    // `hash_ids` (6 ids, backend memory) replaces the top-6 on hash layers.
    // `stats` (--adapt 1; null otherwise): the layer's ADAPT_STRIDE route
    // counters, which the router bumps for its six choices, reading `miss`
    // (the table's per-expert miss bytes) to count the misses. Neither
    // changes what the router computes.
    // rows: logits/probs [T][256], hash_ids/ids/w_* [T][6]
    virtual void router(const float * logits, const float * bias, const int32_t * hash_ids,
                        float * probs, float * probs_biased, int * ids, float * w_raw,
                        float * w_norm, float * w_scaled, const int * miss, uint32_t * stats,
                        int T = 1) = 0;
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
    // y_gate / y_up [6][rows_gu] for the six experts `ids` names. Rows (T > 1,
    // slot -1 only): ids [T][6], x [T][K_gu], y [T*6][rows_gu] -- assignment
    // a = t*6 + k. A chunk may run them EXPERT-MAJOR (--ds4-expert-gather,
    // the GPU: each expert's row read once for all its assignments); every
    // assignment's dot is the same code either way.
    virtual void moe_gate_up(const ExpertTable & t, const int * ids, const float * x,
                             float * y_gate, float * y_up, int slot, int T = 1) = 0;
    // y [T*6][rows_d]; assignment a reads h + a*K_d.
    virtual void moe_down(const ExpertTable & t, const int * ids, const float * h, float * y,
                          int slot, int T = 1) = 0;
    virtual void swiglu_clamp(const float * gate, const float * up, float * h, int n,
                              float limit) = 0;
    // rows: y/weighted [T][n_used][n], w [T][n_used], out [T][n]
    virtual void moe_accum(const float * y, const float * w, int n_used, int n,
                           float * weighted, float * out, int T = 1) = 0;

    // -- profiling (--profile) ------------------------------------------------
    // Miss bytes counted ON THE DEVICE from the ids the router chose (no host
    // round trip): count_misses() adds the six experts' miss_bytes to a
    // device counter; miss_bytes_total() reads it (a sync -- report time only).
    virtual void set_profile(bool on) { (void) on; }
    // --staged-loads 0|1 (GPU): a GEMV row through LDS with 16-byte loads (1,
    // default) or read in place with the decoders' byte loads (0). Same
    // decoder on the same bytes: the pair is a bit-identity check.
    virtual void set_staged_loads(int on) { (void) on; }
    // --profile bookkeeping (no-ops unless profiling): prof_gap() closes this
    // card's share of the token, so the idle time after it is charged to
    // gap_idle rather than to the last op; upstream_wait(prev) makes this
    // card's stream wait on everything `prev`'s stream has queued and charges
    // that wait to ds4_upstream_wait (the boundary copy then finds it done).
    virtual void prof_gap() {}
    virtual void upstream_wait(Ds4Ops & prev) { (void) prev; }
    // --stage-wgs N: workgroups the staging copy uses (grid-stride). A bounded
    // grid leaves the CUs to the main stream's kernels it should overlap.
    virtual void set_stage_wgs(int n) { (void) n; }
    virtual void count_misses(const ExpertTable & t, const int * ids, int T = 1) { (void) t; (void) ids; (void) T; }
    // --ds4-expert-gather 0|1 (GPU, chunks only): expert-major routed GEMVs
    virtual void set_expert_gather(int on) { (void) on; }
    // --ds4-attn-mb N (GPU): attention partials a card holds for a block of a
    // chunk's query rows (the rows run in blocks under it; before reserve())
    virtual void set_attn_mb(int mb) { (void) mb; }
    virtual double miss_bytes_total() { return 0.0; }
    virtual void reset_miss_count() {}

    // -- adaptive placement (--adapt 1; DEEPSEEK4.md section 12) -----------------
    // Every call below is made by ds4_adapt.cpp at a TOKEN BOUNDARY: after the
    // previous token's work was queued on every card and before any of the
    // next token's, never inside a captured graph.
    // Zeroed backend memory for the route counters.
    virtual uint32_t * adapt_stats_alloc(size_t n_words) { (void) n_words; return nullptr; }
    // Queue a read-back of the counters behind everything queued on the main
    // stream so far; adapt_snapshot_poll() returns the host copy once it has
    // landed and nullptr until then. It never waits.
    virtual void adapt_snapshot(const uint32_t * stats, size_t n_words) { (void) stats; (void) n_words; }
    virtual const uint32_t * adapt_snapshot_poll() { return nullptr; }
    // Rewrite table entries, on the main stream: in order after every kernel
    // queued before it, before every kernel queued after it.
    virtual void adapt_table_edit(const ExpertTable & t, const TableEdit & ed) { (void) t; (void) ed; }
    // The swap copies, host -> card, on a low-priority copy stream:
    // adapt_copy_begin() fences them behind everything queued on the main
    // stream so far (including the eviction edits); adapt_copy_end() closes
    // the batch; adapt_copy_landed() asks, without waiting, whether it has;
    // adapt_copy_join() makes the main stream wait for it (queued work only:
    // it is called once the batch has landed, so nothing stalls).
    virtual void adapt_copy_begin() {}
    virtual void adapt_copy(void * dst, const void * src_host, size_t bytes) { (void) dst; (void) src_host; (void) bytes; }
    virtual void adapt_copy_end() {}
    virtual bool adapt_copy_landed() { return true; }
    virtual void adapt_copy_join() {}
    // --adapt-verify: bytes at `a` (card) that differ from `b` (the device view
    // of the host mirror), after everything queued. A sync: end of run only.
    virtual long long adapt_compare(const void * a, const void * b, size_t bytes) { (void) a; (void) b; (void) bytes; return 0; }

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
