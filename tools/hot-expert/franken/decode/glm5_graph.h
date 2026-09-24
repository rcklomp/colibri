// tools/hot-expert/franken/decode/glm5_graph.h
//
// THE MATH of a GLM-5.3-Flash decode token over layers [il0, il1], ported op
// for op from ~/src/llama-glm53/src/models/glm5next.cpp (the layer loop,
// build_kda_layer, build_indexer, build_dsa_layer, build_layer_ffn) and the
// DeepSeek-V4 hyper-connection builders it inherits (build_hc_pre,
// build_hc_post, build_hc_mean), written against Backend (generic half),
// Ds4Ops (what GLM shares with DeepSeek-V4: the HC mix, GEMV, swiglu_clamp,
// the weighted expert sum, the adaptive-placement I/O) and Glm5Ops
// (glm5_ops.h, the GLM-specific half). GLM5.md has the op list per layer.
//
// T = 1: the prompt is fed token by token (exact: KDA is a recurrence and
// DSA attends causally). The span is split over the backends by layer range;
// each card has its own scratch and positional state, and only the
// hyper-connection residual H (64 KB) crosses a boundary -- plus, with the
// three-link miss path, x out to the helper cards and their finished expert
// rows back (glm5_ops.h LinkPlan).
//
// Taps use the reference's cb() names, so ../oracle_dump.cpp's dump of
// llama.cpp -- or this binary's own --dump from another backend -- is
// compared key by key by decode_oracle.cpp.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "decode_graph.h"     // Recorder
#include "ds4_ops.h"
#include "glm5_adapt.h"
#include "glm5_model.h"
#include "glm5_ops.h"

namespace fk {
namespace glm5 {

struct Glm5Config {
    int  ctx = 512;           // positions the DSA caches are sized for
    bool log_routing = true;  // read the routed ids back every step (a sync on a card)
    LinkPlan links;           // the three-link miss path (n_links 1 = owner card only)
    AdaptConfig adapt;        // --adapt (GLM5.md section 9)
};

class Glm5Runner {
public:
    // one Ds4Ops and one Glm5Ops per model device, in the model's device order
    Glm5Runner(Glm5Model & model, std::vector<ds4::Ds4Ops *> dops, std::vector<Glm5Ops *> gops,
               const Glm5Config & cfg);
    ~Glm5Runner();

    // One token at the next position. With the head placed, returns the
    // greedy id; otherwise -1.
    int step(int32_t token, Recorder & rec);
    // Back to position 0 for --probe-at: the recurrent state (KDA state and
    // conv windows, the indexer's open pool) is zeroed; the positional caches
    // are rewritten as the prompt is fed again.
    void rewind();

    int pos() const { return pos_; }
    void set_log_routing(bool on) { cfg_.log_routing = on; }
    void set_profile(bool on) { profile_ = on; }
    const std::vector<int> & routed_ids() const { return routed_; }
    bool scoring() const { return cfg_.ctx > IDX_N_SELECT; }
    void report_cache_bytes(FILE * out) const;
    Adapter * adapter() { return adapt_.get(); }

private:
    struct LayerState {
        float * conv = nullptr;          // [KDA_CONV-1][KDA_QKV]
        float * ssm  = nullptr;          // [N_HEAD][KDA_DIM][KDA_DIM]
        uint16_t * kv = nullptr;         // f16 [ctx][KV_LORA]: the latent K = V
        uint16_t * kg = nullptr;         // f16 [kg_rows][2*IDX_DIM]: indexer key | gate, ring
        uint16_t * pooled = nullptr;     // f16 [ctx/KPOOL + 1][IDX_DIM]
        int kg_rows = KPOOL;
    };
    struct Scratch {
        float *H, *Hn, *mixes, *pre, *post, *comb, *x, *xn, *ao, *fo;
        float *qkv, *conv, *fa, *fb, *g, *beta, *ga, *gb, *o, *gated;
        float *qr, *q, *kv, *qabs, *att, *kqv, *ik, *igate, *iq, *iw, *iscore;
        int   *isel;
        float *dup, *dgate, *dh;
        float *rlog, *probs, *probs_b, *wraw, *wnorm, *wsc;
        float *yg, *yu, *yh, *yd, *ywt, *moe, *sg, *su, *sh, *sd;
        int   *ids;
    };

    void layer(int il, Recorder & rec);
    void hc_pre(Scratch & S, const Mat & fn, const float * base, const float * scale, int il,
                Recorder & rec, const char * sfx);
    void kda(int il, Recorder & rec);
    void dsa(int il, Recorder & rec);
    void ffn(int il, Recorder & rec);
    int dev_of(int il) const { return model_.layer(il).dev; }
    Backend & be(int il) { return model_.dev_for(il); }
    ds4::Ds4Ops & dop(int il) { return *dops_[(size_t) dev_of(il)]; }
    Glm5Ops & gop(int il) { return *gops_[(size_t) dev_of(il)]; }
    Scratch & S(int il) { return scr_[(size_t) dev_of(il)]; }

    Glm5Model & model_;
    std::vector<ds4::Ds4Ops *> dops_;
    std::vector<Glm5Ops *> gops_;
    Glm5Config cfg_;
    float eps_, ln_eps_, hc_eps_;
    int pos_ = 0;
    bool profile_ = false;
    size_t iscore_stride_ = 0;
    std::vector<LayerState> st_;
    std::vector<Scratch> scr_;
    std::vector<int> routed_;
    float *hmean_ = nullptr, *hn_ = nullptr, *logits_ = nullptr;
    int   *greedy_ = nullptr;
    std::vector<std::pair<Backend *, void *>> owned_;
    std::unique_ptr<Adapter> adapt_;
};

// The CLI of `franken_decode --model <glm5next gguf> ...` (dispatched from
// franken_decode.cpp's main on general.architecture).
int glm5_main(int argc, char ** argv);

} // namespace glm5
} // namespace fk
