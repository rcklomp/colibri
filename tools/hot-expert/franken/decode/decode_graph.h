// tools/hot-expert/franken/decode/decode_graph.h
//
// THE MATH of a Qwen3.8-Flash-Next decode token over layers [il0, il1],
// ported op for op from ~/src/llama-glm53/src/models/qwen4exp.cpp @ 39931761a
// and written against Backend (decode_backend.h) so the identical code runs
// on the CPU and on device 0.
//
// Every `cb(t, "name", il)` of the reference that the brief lists is a tap()
// here, keyed by the string the dump tool writes, because the reference's own
// naming has three traps and every one of them would read as a numerics
// failure while being a bookkeeping one:
//
//   * build_hc_mix runs TWICE a layer and cb's the same four names both
//     times. The dump keeps both, disambiguating the second as
//     `hc_norm-3.2`, `hc_gate-3.2`, `hc_mixed-3.2`, `hc_inject-3.2` -- so
//     the attention module's value is the unsuffixed file and the FFN
//     module's is `.2`.
//   * build_hc_combine also runs twice, but the SECOND call's tensor is
//     immediately cb'd again as `l_last`, and ggml_set_name overwrites, so
//     `hc_combine-3` is the POST-ATTENTION residual and `l_last-3` is the
//     post-FFN one. (Verified against the dump: the two files differ.) The
//     same overwrite is why `kqv_out` never appears -- `attn_pregate` is
//     cb'd on the same node a line later.
//   * the dump writes ne0*ne1 floats of the LAST ne2 slice. For nearly every
//     tensor ne2 is the token axis, so that is "the last token". For
//     `state_predelta` ne2 is the HEAD axis, so the file holds head 47's
//     128x128 state and nothing else.

#pragma once

#include <map>
#include <random>
#include <string>
#include <vector>

#include "decode_backend.h"
#include "decode_model.h"

namespace fk {

// One recorded intermediate. `data` is already on the host.
struct TapValue {
    std::string        name;     // the reference's cb() name
    std::string        key;      // the dump's file key: "<name>-<il><suffix>", or "<name>" at il < 0
    int                il = -1;
    std::vector<float> data;
};

class Recorder {
public:
    void enable(bool on) { on_ = on; }
    bool enabled() const { return on_; }
    void clear() { taps_.clear(); }

    void tap(Backend & be, const char * name, int il, const float * buf, size_t n,
             const char * suffix = "");
    void tap_ints(const char * name, int il, const std::vector<int> & v,
                  const char * suffix = "");

    const TapValue * get(const std::string & key) const;
    const std::map<std::string, TapValue> & all() const { return taps_; }

    static std::string make_key(const char * name, int il, const char * suffix);

private:
    bool on_ = false;
    std::map<std::string, TapValue> taps_;
};

struct DecodeConfig {
    int   ctx     = 512;   // cells the QSA caches are sized for
    // The largest chunk step() will be handed (design 9.4's --chunk). Every
    // per-token scratch buffer is sized for it ONCE, at construction: the
    // MoE intermediates dominate (T x 10 x 3 840 floats) and at T = 512 that
    // is 78 MB a device, which is why the cap is 512 and the default 256.
    int   max_tokens = 1;
    bool  verbose = false;
    // A random fp32-rounding-sized perturbation of every block input, oracle
    // only (CLAUDE.md, F7: "a float64 arm is not the yardstick for a
    // reordered kernel, the jitter arm is"). It answers the one question a
    // raw cosine bar cannot: how far apart do two EQUALLY CORRECT summation
    // orders of this architecture end up? Off by default.
    float jitter  = 0.0f;
    // read the routed ids back after the body even without --verbose (the
    // routing oracle needs them; it is one download a device a token)
    bool  log_routing = false;
    bool  sync_debug   = false;
    // print a line per layer range on the first token, so an abort says how
    // far it got (stdout is line-buffered by main for the same reason)
    bool  progress     = true;
};

// Step 3: one runner over all the devices the model was placed on. Every
// scratch buffer exists ONCE PER DEVICE (a layer may not read another card's
// memory), and `bind()` swaps the active set at a layer. The only thing that
// actually crosses a boundary is the hyper-connection residual.
class DecodeRunner {
public:
    DecodeRunner(DecodeModel & model, const DecodeConfig & cfg);
    ~DecodeRunner();

    // Feeds a CHUNK of `T` tokens starting at the next position (T <= the
    // configured max_tokens). `ple_emb` is the host-side PLE gather for the
    // chunk, T*N_EMBD floats token-major, or nullptr if the layer range does
    // not include the PLE layer. With a head placed, returns the greedy id
    // of the LAST row -- the prompt needs one logit vector, so the head and
    // the argmax run on that row alone.
    //
    // A chunk produces exactly what T single-token steps produce (PREFILL.md
    // section 2), so this is the decode path at T = 1 and nothing else.
    //
    // `flush` false ENQUEUES the chunk and returns without waiting for it: no
    // timer read, no greedy download, no routing read-back. That is design
    // 3.5's pipeline -- the host runs ahead and chunk n+1's work reaches card
    // 0 while chunk n is still on cards 1 and 2 -- and it is why the return
    // value is -1 there. The cross-card safety is NOT the host's doing: it is
    // the two residual banks and the drain events (Backend::boundary_recv).
    // A chunk that the recorder is capturing always flushes, because a tap is
    // a download.
    int  step(const int32_t * tokens, int T, const float * ple_emb, Recorder & rec,
              bool flush = true);
    int  step(int32_t token, const float * ple_emb, Recorder & rec) {
        return step(&token, 1, ple_emb, rec, true);
    }

    // The last step's logits, on the host (only with a head placed).
    std::vector<float> logits_host();

    // Per-layer routed expert ids of the last step, [n_layers][N_EXPERT_USED].
    const std::vector<int> & routed_ids() const { return routed_ids_; }

    // The wide residual after the last layer of the range, on the host.
    std::vector<float> residual_host();

    int pos() const { return pos_; }

    // Wall/GPU time of the last step's layer body, milliseconds. ZERO after a
    // step that did not flush: there is nothing to read without waiting.
    double last_body_ms() const { return last_body_ms_; }

    // Whether the last step actually waited. step() can OVERRIDE a caller's
    // `flush=false` (a captured chunk, --jitter, --sync-debug), so a caller
    // that wants the greedy id or the routed ids must ask this rather than
    // re-deriving the condition -- which is how the two would drift apart.
    bool last_flushed() const { return last_flushed_; }

    // HOST wall time to ENQUEUE one token's whole body -- no waits, no syncs,
    // nothing read back. A card with its work pre-queued runs at its own
    // speed; a card that is being fed one op at a time runs at this rate, so
    // `issue_ms` against `last_body_ms` says which of the two is happening.
    double last_issue_ms() const { return last_issue_ms_; }

    // Turns the --verbose / --routing capture off, so a timing loop measures
    // the engine rather than the instrumentation.
    void set_capture(bool on) { capture_ = on; }

    // ---- serving: the state a prefix-reuse rollback has to move (L0 step 4) --
    //
    // What a slot holds after `pos` tokens splits in two:
    //
    //   POSITIONAL, and therefore still valid after a rollback to any p <= pos:
    //     the q8_0 K/V cache (cell j IS position j) and the pooled indexer keys
    //     of the blocks that are already closed. Nothing below touches them.
    //
    //   RECURRENT, and therefore only valid AT pos: the GDN state of every
    //     recurrent layer, the GDN conv window's GDN_CONV_HIST history slots,
    //     the PLE conv window's PLE_CONV_HIST history slots, and the indexer's
    //     running block sum (idx_sum) with its fill value (idx_raw0) on every
    //     QSA layer. That is what these three move.
    //
    // So: save_state() at a chunk boundary, load_state(blob, p) to roll back to
    // it, and re-feed the tokens from p on. The re-fed chunk rewrites the K/V
    // cells and the pooled keys of every block it joins, exactly as the first
    // run did -- idx_sum is part of the blob, so a block that straddles p is
    // re-pooled from the same partial sum and lands on the same bits.
    //
    // Sizes at the served geometry (48 layers, 36 of them recurrent): 3.1 MB of
    // GDN state a layer dominates, 118 MB a snapshot in total. This is a HOST
    // copy: it costs one download a device per snapshot and nothing in VRAM.
    size_t rec_bytes() const;
    void   save_rec(void * dst);
    void   load_rec(const void * src, int pos);

    // The same copy with NOTHING waited on: stream-ordered behind the chunk
    // that has just run and in front of the chunk that comes next
    // (Backend::download_async). This is what keeps a serving checkpoint out
    // of the prefill's critical path -- the first served run took a snapshot
    // per chunk with the blocking save above and prefilled at 7.4 ms a token,
    // the UNPIPELINED rate, because every chunk ended in a full device drain
    // (record L0-PREFILL-2: 7.7 unpipelined, 3.0 pipelined, 1.63 with
    // --gemm-lds 1). `dst` must come from Backend::alloc_pinned, and is only
    // safe to read after sync_devices().
    void   save_rec_async(void * dst);
    void   save_kv_async(void * dst, int len);
    // Drains every device. One call ends the "in flight" state of every
    // async save issued before it.
    void   sync_devices();

    // ---- the POSITIONAL half of a slot, cells [0, len) ---------------------
    //
    // Restoring the recurrent state alone is only enough while the slot's own
    // K/V cells still hold the tokens the snapshot was taken over. One
    // interleaved request on the same slot ends that: it re-prefills from its
    // own longest common prefix and overwrites the cells from there on, and a
    // rollback afterwards would then report reuse over a cache that holds
    // somebody else's keys. With one KV slot and a gateway that routes every
    // conversation to it, that is the normal case, not a corner (2026-09-22:
    // a 58-token request between two UI chats is what made the second one
    // reuse nothing).
    //
    // So a checkpoint may also carry the cells. The q8_0 K/V cache and the
    // pooled indexer keys are laid out cell-major (decode_backend.h:
    // `qs[(cell*N_KV_HEADS + h)*HEAD_DIM + d]`, `pooled[blk*IDX_DIM + i]`), so
    // cells [0, len) are a contiguous PREFIX of every one of those buffers and
    // an image of `layout_len` cells can be restored to any `copy_len <=
    // layout_len` without re-packing. ~1152 B a token a QSA layer, i.e. 13.8 kB
    // a token over the 12 QSA layers of this model.
    size_t kv_bytes(int len) const;
    void   save_kv(void * dst, int len);
    void   load_kv(const void * src, int layout_len, int copy_len, int pos);
    // A fresh sequence: every recurrent buffer zeroed and pos back to 0, which
    // is the state the constructor left (alloc_f32 zeroes). The positional
    // caches are deliberately NOT cleared -- every cell they hold is about to
    // be overwritten before it can be read, since a read is bounded by pos.
    void   reset_state();

    int    ctx()       const { return cfg_.ctx; }
    int    max_chunk() const { return max_T_; }
    // stdout is the wire in SERVE mode: nothing may print there but a frame.
    void   set_progress(bool on) { cfg_.progress = on; }

    // Bytes of QSA cache this device holds, and the per-token rate, so the
    // sizing is a printed number rather than a belief (`kv_bytes_dev<i>=`).
    void report_cache_bytes(FILE * out) const;

private:
    struct LayerState {
        // GDN conv WINDOW: (GDN_CONV_K-1) history slots + max_tokens chunk
        // slots of GDN_CONV_DIM. Token t of the chunk is at slot
        // GDN_CONV_HIST+t and its tap k at slot t+k; conv_slide() carries the
        // tail across the chunk boundary.
        float *    conv_win   = nullptr;
        float *    gdn_state  = nullptr;  // [GDN_V_HEADS][GDN_STATE][GDN_STATE], ggml order
        int8_t *   kqs        = nullptr;  // q8_0 K cache, M3's planar layout
        uint16_t * ksc        = nullptr;
        int8_t *   vqs        = nullptr;
        uint16_t * vsc        = nullptr;
        // No raw-key cache: a running sum plus cell 0's key is all the
        // reference's pooling formula reads (decode_backend.h), so the
        // indexer costs the 64 B a token design 9.1 budgets, not 576.
        float *    idx_new    = nullptr;  // [T][IDX_DIM] the chunk's raw keys
        float *    idx_sum    = nullptr;  // [IDX_DIM] members of the current block
        float *    idx_raw0   = nullptr;  // [IDX_DIM] cell 0's key, the fill value
        uint16_t * idx_pooled = nullptr;  // [ctx/ratio][IDX_DIM] bf16 pooled keys
    };

    // One contiguous run of recurrent state on one device. The list is built
    // in a fixed order (layers in index order, then the PLE window), so a blob
    // written by save_state() is read back by load_state() piece for piece.
    struct StatePiece { Backend * be; void * ptr; size_t bytes; };
    void state_pieces(std::vector<StatePiece> & out);
    // One run of a positional cache. `off` is where it sits in a host image
    // laid out for `layout_len` cells; `bytes` is how much of it to move.
    struct KvPiece { Backend * be; void * ptr; size_t off; size_t bytes; };
    void kv_plan(std::vector<KvPiece> & out, int layout_len, int copy_len);

    void hc_mix(const LayerWeights & L, const float * w_norm, const Mat & down,
                const Mat & up, const Mat & inj, const float * x,
                float * out_mixed, float * out_inject, int il, Recorder & rec,
                const char * suffix, bool xn_ready);
    void layer_ple(const LayerWeights & L, Recorder & rec);
    void layer_gdn(const LayerWeights & L, LayerState & st, int il, Recorder & rec);
    void layer_qsa(const LayerWeights & L, LayerState & st, int il, Recorder & rec);
    void layer_ffn(const LayerWeights & L, int il, Recorder & rec);
    void jitter(float * buf, size_t n);

    // One set of scratch per device; `bind()` points the active members at
    // one of them. The body code below is written against the active set and
    // never has to know which card it is on.
    // Two residual banks a device, picked by chunk parity. The ONLY buffer
    // that crosses a card, so the only one a pipelined chunk n+1 could race:
    // card 0 writes bank (n+1)&1 while card 1 still reads bank n&1 out of it.
    // Two is enough and self-limiting -- writing bank (n+2)&1 == n&1 waits on
    // the drain event of the copy that read it at chunk n, which in the steady
    // state (card 0 on n+2, card 1 on n+1, card 2 on n) has long finished.
    static constexpr int N_RES_BANKS = 2;

    struct Scratch {
        float *x, *res_hc[N_RES_BANKS], *xn, *lo, *hgate, *mixed, *inject, *blk;
        float *z, *conv, *convqk, *qkn, *alpha, *beta, *gexp, *abuf, *bsig, *gate_raw, *gdn, *gnorm;
        float *qfull, *qcur, *gate, *gsig, *kcur, *vcur, *kqv, *kqvg, *kraw, *idxraw;
        float *idxq, *blkscore, *cellscore, *pool_raw, *pool_rope;
        int   *sel;
        float *logits, *wts, *ygate, *yup, *hmoe, *eo, *moeout;
        float *shg, *shu, *shh, *shout, *shgated, *shgate, *shgsig;
        float *plek, *plev, *pleq, *plegated, *plegate, *plenorm, *pleconv, *pleemb, *ple_win;
        int   *ids, *ids_log;
        float *wts_log;
    };
    void bind(int dev);

    std::vector<Scratch> pool_;
    int                  cur_dev_ = -1;
    Backend *            bep_ = nullptr;

    // head scratch, on the last device only
    float * head_xn_   = nullptr;
    float * head_lo_   = nullptr;
    float * head_gate_ = nullptr;
    float * head_out_  = nullptr;
    float * logits_all_ = nullptr;
    int   * greedy_id_  = nullptr;
    std::vector<int> routed_ids_;

    DecodeModel & model_;
    DecodeConfig  cfg_;
    float         eps_;
    int           pos_ = 0;
    double        last_body_ms_ = 0.0;
    double        last_issue_ms_ = 0.0;
    bool          last_flushed_ = true;
    bool          capture_ = true;
    std::vector<float> emb_;          // hoisted: a per-token heap allocation
                                      // in the issue path is host time too

    std::mt19937            jrng_{0xC0FFEE42u};
    std::vector<LayerState> lstate_;
    std::vector<void *>     owned_;

    // PLE conv WINDOW, one for the whole range (there is exactly one PLE
    // layer): PLE_CONV_HIST history slots + max_tokens chunk slots.
    float * ple_win_ = nullptr;

    int     T_ = 1;        // tokens in the chunk being run
    int     max_T_ = 1;
    int     bank_ = 0;     // this chunk's residual bank, chunk index & 1
    long long chunks_ = 0; // chunks fed since construction

    // the ACTIVE scratch set, rebound by bind() at every layer
    float *x_, *res_hc_, *xn_, *lo_, *hgate_, *mixed_, *inject_, *blk_;
    float *z_, *conv_, *convqk_, *qkn_, *qn_, *kn_, *alpha_, *beta_, *gexp_, *abuf_, *bsig_, *gate_raw_, *gdn_, *gnorm_;
    float *qfull_, *qcur_, *gate_, *gsig_, *kcur_, *vcur_, *kqv_, *kqvg_;
    float *kraw_, *idxraw_;
    float *idxq_, *blkscore_, *cellscore_, *pool_raw_, *pool_rope_;
    int   *sel_;
    float *logits_, *wts_, *ygate_, *yup_, *hmoe_, *eo_, *moeout_;
    float *shg_, *shu_, *shh_, *shout_, *shgated_, *shgate_, *shgsig_;
    float *plek_, *plev_, *pleq_, *plegated_, *plegate_, *plenorm_, *pleconv_, *pleemb_;
    int   *ids_;
    int   *ids_log_ = nullptr;   // per-layer routed ids, read back AFTER the body
    float *wts_log_ = nullptr;
    bool   need_ple_ = false;
    // set when the preceding hc_combine already produced xn for the next mix
    bool   xn_ready_ = false;
};

} // namespace fk
