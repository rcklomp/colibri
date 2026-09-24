// tools/hot-expert/franken/decode/glm5_graph.h
//
// THE MATH of a GLM-5.3-Flash decode token over layers [il0, il1], ported op
// for op from ~/src/llama-glm53/src/models/glm5next.cpp (the layer loop,
// build_kda_layer, build_indexer, build_dsa_layer, build_layer_ffn) and the
// DeepSeek-V4 hyper-connection builders it inherits (build_hc_pre,
// build_hc_post, build_hc_mean), written against Backend (generic half),
// Ds4Ops (what GLM shares with DeepSeek-V4: the HC mix, GEMV, swiglu_clamp,
// the weighted expert sum) and Glm5Ops (glm5_ops.h, the GLM-specific half).
// GLM5.md has the op list per layer and what each tap is.
//
// Step 2 (L5 GLM): T = 1 -- the prompt is fed token by token, which is exact
// because a KDA layer is a recurrence and a DSA layer attends causally; the
// CPU arm only (the runner refuses a GPU backend).
//
// Taps use the reference's cb() names, so ../oracle_dump.cpp's dump of
// llama.cpp is compared key by key by decode_oracle.cpp. build_hc_pre's
// four names occur twice a layer; the FFN module's carry the ".2" suffix.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "decode_graph.h"     // Recorder
#include "ds4_ops.h"
#include "glm5_model.h"
#include "glm5_ops.h"

namespace fk {
namespace glm5 {

struct Glm5Config {
    int  ctx = 512;           // positions the DSA caches are sized for
    bool log_routing = true;  // read the routed ids back every step
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

    int pos() const { return pos_; }
    // Per-layer routed ids of the last step, [n_layers][8] (-1 on dense layers).
    const std::vector<int> & routed_ids() const { return routed_; }
    // The indexer runs its scoring only when ctx > IDX_N_SELECT (glm5next.cpp:
    // "gated on n_ctx"); below that every visible cell is attended.
    bool scoring() const { return cfg_.ctx > IDX_N_SELECT; }
    void report_cache_bytes(FILE * out) const;

private:
    struct LayerState {
        // KDA
        float * conv = nullptr;          // [KDA_CONV-1][KDA_QKV]
        float * ssm  = nullptr;          // [N_HEAD][KDA_DIM][KDA_DIM]
        // DSA
        uint16_t * kv = nullptr;         // f16 [ctx][KV_LORA]: the latent K = V
        uint16_t * kg = nullptr;         // f16 [kg_rows][2*IDX_DIM]: indexer key | gate, ring
        uint16_t * pooled = nullptr;     // f16 [ctx/KPOOL + 1][IDX_DIM]
        int kg_rows = KPOOL;
    };
    struct Scratch {
        float *H, *Hn, *mixes, *pre, *post, *comb, *x, *xn, *ao, *fo;
        // KDA
        float *qkv, *conv, *fa, *fb, *g, *beta, *ga, *gb, *o, *gated;
        // DSA
        float *qr, *q, *kv, *qabs, *att, *kqv, *ik, *igate, *iq, *iw, *iscore;
        int   *isel;
        // FFN
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
    size_t iscore_stride_ = 0;
    std::vector<LayerState> st_;
    std::vector<Scratch> scr_;
    std::vector<int> routed_;
    float *hmean_ = nullptr, *hn_ = nullptr, *logits_ = nullptr;
    int   *greedy_ = nullptr;
    std::vector<std::pair<Backend *, void *>> owned_;
};

// The CLI of `franken_decode --model <glm5next gguf> ...` (dispatched from
// franken_decode.cpp's main on general.architecture).
int glm5_main(int argc, char ** argv);

} // namespace glm5
} // namespace fk
