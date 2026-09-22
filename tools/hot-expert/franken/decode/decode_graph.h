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
    bool  verbose = false;
    // A random fp32-rounding-sized perturbation of every block input, oracle
    // only (CLAUDE.md, F7: "a float64 arm is not the yardstick for a
    // reordered kernel, the jitter arm is"). It answers the one question a
    // raw cosine bar cannot: how far apart do two EQUALLY CORRECT summation
    // orders of this architecture end up? Off by default.
    float jitter  = 0.0f;
};

class DecodeRunner {
public:
    DecodeRunner(Backend & be, DecodeModel & model, const DecodeConfig & cfg);
    ~DecodeRunner();

    // Feeds one token at position `pos` (which must be the next position).
    // `ple_emb` is the host-side PLE gather for this token (N_EMBD floats),
    // or nullptr if the layer range does not include the PLE layer.
    void step(int32_t token, const float * ple_emb, Recorder & rec);

    // The wide residual after the last layer of the range, on the host.
    std::vector<float> residual_host();

    int pos() const { return pos_; }

    // Wall/GPU time of the last step's layer body, milliseconds.
    double last_body_ms() const { return last_body_ms_; }

private:
    struct LayerState {
        float *    conv_ring  = nullptr;  // GDN: [GDN_CONV_K][GDN_CONV_DIM]
        int        conv_head  = 0;
        float *    gdn_state  = nullptr;  // [GDN_V_HEADS][GDN_STATE][GDN_STATE], ggml order
        int8_t *   kqs        = nullptr;  // q8_0 K cache, M3's planar layout
        uint16_t * ksc        = nullptr;
        int8_t *   vqs        = nullptr;
        uint16_t * vsc        = nullptr;
        float *    idx_raw    = nullptr;  // [ctx][IDX_DIM] raw indexer keys
        uint16_t * idx_pooled = nullptr;  // [ctx/ratio][IDX_DIM] bf16 pooled keys
    };

    void hc_mix(const LayerWeights & L, const float * w_norm, const Mat & down,
                const Mat & up, const Mat & inj, const float * x,
                float * out_mixed, float * out_inject, int il, Recorder & rec,
                const char * suffix);
    void layer_ple(const LayerWeights & L, const float * ple_emb, Recorder & rec);
    void layer_gdn(const LayerWeights & L, LayerState & st, int il, Recorder & rec);
    void layer_qsa(const LayerWeights & L, LayerState & st, int il, Recorder & rec);
    void layer_ffn(const LayerWeights & L, int il, Recorder & rec);
    void jitter(float * buf, size_t n);

    Backend &     be_;
    DecodeModel & model_;
    DecodeConfig  cfg_;
    float         eps_;
    int           pos_ = 0;
    double        last_body_ms_ = 0.0;

    std::mt19937            jrng_{0xC0FFEE42u};
    std::vector<LayerState> lstate_;
    std::vector<void *>     owned_;

    // PLE conv ring, one for the whole range (there is exactly one PLE layer)
    float * ple_ring_ = nullptr;
    int     ple_head_ = 0;

    // scratch (all backend buffers)
    float *x_, *res_hc_, *xn_, *lo_, *hgate_, *mixed_, *inject_, *blk_;
    float *qkv_, *z_, *conv_, *qn_, *kn_, *alpha_, *beta_, *gexp_, *abuf_, *gdn_, *gnorm_;
    float *qfull_, *qcur_, *gate_, *gsig_, *kcur_, *vcur_, *kqv_;
    float *idxq_, *idxk_, *blkscore_, *blkbias_, *cellscore_, *pool_raw_, *pool_rope_;
    int   *sel_;
    float *logits_, *wts_, *ygate_, *yup_, *hmoe_, *eo_, *moeout_;
    float *shg_, *shu_, *shh_, *shout_, *shgate_;
    float *plek_, *plev_, *pleq_, *plegated_, *plegate_, *plenorm_, *pleconv_, *pleemb_;
    int   *ids_;
};

} // namespace fk
