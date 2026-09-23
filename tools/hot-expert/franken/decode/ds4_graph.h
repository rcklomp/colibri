// tools/hot-expert/franken/decode/ds4_graph.h
//
// THE MATH of a DeepSeek-V4-Flash decode token over layers [il0, il1],
// ported op for op from ~/src/llama-glm53/src/models/deepseek4.cpp and the
// compression plan of src/llama-kv-cache-dsv4.cpp, written against Backend
// (generic half) and Ds4Ops (ds4_ops.h, the DeepSeek-specific half). See
// DEEPSEEK4.md for the op list per layer and what each tap is.
//
// One token a step (T == 1): every DS4 attention path is causal per token,
// so feeding a prompt token by token IS the reference's ubatch result. The
// span is split over the backends by layer range; each device has its own
// scratch set and positional state, and the only thing that crosses a card
// is the hyper-connection residual H (4 x 4096 f32), by Backend::boundary_recv.
//
// Taps use the reference's cb() names, so ../oracle_dump.cpp's dump of
// llama.cpp -- or this binary's own --dump from another backend -- is
// compared key by key by decode_oracle.cpp. build_hc_pre's four names occur
// twice a layer; the FFN module's carry the ".2" suffix, as the dump writes.

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
    bool log_routing = true; // read the routed ids back every step (a sync on a card)
    // --miss-stage 1 (default): a missed expert is copied into a VRAM staging
    // ring on a side stream while the shared expert runs, and the token-id
    // routed layers 0-2 are staged at embed time. 0: step 2's path, read in
    // place over PCIe inside the GEMV. Same bytes either way.
    int  miss_stage = 1;
    // --hip-graph 0|1 (L5 step 3b, default 0): each card's share of a decode
    // token is captured once per position class as a hipGraph and replayed;
    // the ops read the position from a device int the graph advances. The
    // same kernels in the same order with the same arguments -- except the
    // position, which the kernel now reads -- so a replay is the eager token.
    int  hip_graph = 0;
    int  hip_graph_bucket = 1024;   // positions a captured graph serves
    // --all-ops 1 (a check, not a mode): issue every position-guarded op
    // eagerly too, exactly as a captured graph contains them -- so the CPU arm
    // can prove the graph's op sequence gives the eager result.
    int  all_ops = 0;
};

class Ds4Runner {
public:
    // `ops` holds one Ds4Ops per model device, in the model's device order.
    Ds4Runner(Ds4Model & model, std::vector<Ds4Ops *> ops, const Ds4Config & cfg);
    ~Ds4Runner();
    void graph_report(FILE * out);

    // One token at the next position. With the head placed, returns the
    // greedy id; otherwise -1.
    int step(int32_t token, Recorder & rec);

    int pos() const { return pos_; }
    void set_log_routing(bool on) { cfg_.log_routing = on; }
    // --profile: every card closes its profile interval at the end of a step
    void set_profile(bool on) { profile_ = on; }
    // Per-layer routed ids of the last step, [n_layers][6] (with log_routing).
    const std::vector<int> & routed_ids() const { return routed_; }
    void report_cache_bytes(FILE * out) const;

private:
    struct LayerState {
        uint16_t * raw = nullptr;       // f16 [N_SWA][512], slot = pos % N_SWA
        float * ck = nullptr, * cs = nullptr;         // compressor ring [ring][coff*512]
        float * lk = nullptr, * ls = nullptr;         // lid ring [8][256] (CSA only)
        uint16_t * comp = nullptr;      // f16 [ctx/ratio + 1][512] compressed K
        uint16_t * lid  = nullptr;      // f16 [ctx/4 + 1][128] lid keys (CSA only)
        int ring = 0;
    };
    // One scratch set per device: a kernel never reads another card's memory.
    struct Scratch {
        float *H, *Hn, *mixes, *pre, *post, *comb;
        float *x, *xn, *qr, *q, *kv, *att, *oa, *ao;
        float *ckv, *csc, *lkv, *lsc, *cpool, *iq, *iw, *iscore;
        float *rlog, *probs, *probs_b, *wraw, *wnorm, *wsc;
        float *yg, *yu, *yh, *yd, *ywt, *moe, *sg, *su, *sh, *sd, *fo;
        int   *isel, *ids, *hash;
    };

    void layer(int il, Recorder & rec);
    void hc_pre(Scratch & S, const Mat & fn, const float * base, const float * scale, int il,
                Recorder & rec, const char * sfx);
    void attention(int il, Recorder & rec);
    void ffn(int il, Recorder & rec);
    RopeParams rope_for(int il) const;
    RopeParams rope_compress() const;
    int dev_of(int il) const { return model_.layer(il).dev; }
    Backend & be(int il) { return model_.dev_for(il); }
    Ds4Ops & op(int il) { return *ops_[(size_t) dev_of(il)]; }
    Scratch & S(int il) { return scr_[(size_t) dev_of(il)]; }

    Ds4Model & model_;
    std::vector<Ds4Ops *> ops_;
    Ds4Config cfg_;
    float eps_, hc_eps_;
    int pos_ = 0;
    bool profile_ = false;
    // --hip-graph
    struct GraphSlot { void * exec = nullptr; int cls = -1; long long epoch = -1; };
    bool seg_begin(int d);             // true: issue the body (eager or capturing)
    void seg_end(int d);
    bool graph_now_ = false, seg_capturing_ = false;
    int  graph_cls_ = 0, graph_bound_ = 0;
    std::vector<GraphSlot> graphs_;    // per card
    std::vector<int *>     dpos_;      // per card: the device-resident position
    std::vector<int>       dpos_val_;  // what it will hold once queued work has run (-1 unknown)
    int stage_slot(int il) const;
    std::vector<LayerState> st_;
    std::vector<Scratch> scr_;
    std::vector<int> routed_;
    // head, on the last device
    float *hmix_ = nullptr, *hpre_ = nullptr, *hx_ = nullptr, *hxn_ = nullptr, *logits_ = nullptr;
    int   *greedy_ = nullptr;
    std::vector<std::pair<Backend *, void *>> owned_;
};

// The CLI of `franken_decode --model <deepseek4 gguf> ...` (dispatched from
// franken_decode.cpp's main on general.architecture).
int ds4_main(int argc, char ** argv);

} // namespace ds4
} // namespace fk
