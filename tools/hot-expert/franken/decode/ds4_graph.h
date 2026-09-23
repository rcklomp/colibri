// tools/hot-expert/franken/decode/ds4_graph.h
//
// THE MATH of a DeepSeek-V4-Flash decode token over layers [il0, il1],
// ported op for op from ~/src/llama-glm53/src/models/deepseek4.cpp and the
// compression plan of src/llama-kv-cache-dsv4.cpp, written against Backend
// (generic half) and Ds4Ops (ds4_ops.h, the DeepSeek-specific half). See
// DEEPSEEK4.md for the op list per layer and what each tap is.
//
// L5 step 1 scope: one token a step (the prompt is fed token by token -- the
// compressed attention is causal per token, so that is the reference's
// ubatch result by construction), CPU backend. Taps use the reference's cb()
// names, so ../oracle_dump.cpp's dump of llama.cpp is compared key by key by
// decode_oracle.cpp. The reference cb's four names twice a layer
// (build_hc_pre in the attention module, then in the FFN module): the second
// occurrence carries the ".2" suffix, as the dump writes it.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "decode_graph.h"     // Recorder
#include "ds4_model.h"
#include "ds4_ops.h"

namespace fk {
namespace ds4 {

struct Ds4Config {
    int  ctx = 512;          // positions the compressed caches are sized for
    bool verbose = false;
};

class Ds4Runner {
public:
    Ds4Runner(Ds4Model & model, Ds4Ops & ops, const Ds4Config & cfg);
    ~Ds4Runner();

    // One token at the next position. With the head placed, returns the
    // greedy id; otherwise -1.
    int step(int32_t token, Recorder & rec);

    int pos() const { return pos_; }
    // Per-layer routed ids of the last step, [n_layers][6].
    const std::vector<int> & routed_ids() const { return routed_; }
    std::vector<float> logits_host();
    // Bytes of positional cache (raw window + compressed rows) the span holds
    // at `ctx`, and the rate per token.
    void report_cache_bytes(FILE * out) const;

private:
    struct LayerState {
        uint16_t * raw = nullptr;       // f16 [N_SWA][512], slot = pos % N_SWA
        // compressor state rings (f32): CSA/lid 2R rows, HCA R rows
        float * ck = nullptr, * cs = nullptr;         // [ring][coff*512]
        float * lk = nullptr, * ls = nullptr;         // [8][256] (CSA only)
        uint16_t * comp = nullptr;      // f16 [ctx/ratio][512] compressed K
        uint16_t * lid  = nullptr;      // f16 [ctx/4][128] lid keys (CSA only)
        int ring = 0;
    };

    void layer(int il, int32_t token, Recorder & rec);
    void hc_pre(const Mat & fn, const float * base, const float * scale, int il,
                Recorder & rec, const char * sfx);
    void attention(int il, Recorder & rec);
    void ffn(int il, int32_t token, Recorder & rec);
    RopeParams rope_for(int il) const;
    RopeParams rope_compress() const;
    Backend & be(int il) { return model_.dev_for(il); }

    Ds4Model & model_;
    Ds4Ops & ops_;
    Ds4Config cfg_;
    float eps_, hc_eps_;
    int pos_ = 0;
    std::vector<LayerState> st_;
    std::vector<int> routed_;

    // scratch (host for the CPU backend; one set, the CPU arm has one device)
    float *H = nullptr, *Hn = nullptr, *mixes = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float *x = nullptr, *xn = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr;
    float *att = nullptr, *oa = nullptr, *ao = nullptr;
    float *ckv = nullptr, *csc = nullptr, *lkv = nullptr, *lsc = nullptr, *cpool = nullptr;
    float *iq = nullptr, *iw = nullptr, *iscore = nullptr;
    float *rlog = nullptr, *probs = nullptr, *probs_b = nullptr;
    float *wraw = nullptr, *wnorm = nullptr, *wsc = nullptr;
    float *yg = nullptr, *yu = nullptr, *yh = nullptr, *yd = nullptr, *ywt = nullptr, *moe = nullptr;
    float *sg = nullptr, *su = nullptr, *sh = nullptr, *sd = nullptr, *fo = nullptr;
    float *hmix = nullptr, *hpre = nullptr, *hx = nullptr, *hxn = nullptr, *logits = nullptr;
    int   *isel = nullptr;
    std::vector<void *> owned_;
    Backend * hb_ = nullptr;          // the backend the scratch lives on
};

// The CLI of `franken_decode --model <deepseek4 gguf> ...` (dispatched from
// franken_decode.cpp's main on general.architecture).
int ds4_main(int argc, char ** argv);

} // namespace ds4
} // namespace fk
