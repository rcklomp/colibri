// tools/hot-expert/franken/decode/ds4_adapt.h
//
// ADAPTIVE EXPERT PLACEMENT for DeepSeek-V4-Flash (design ladder L2;
// DEEPSEEK4.md section 12). The placement a run starts from is a histogram of
// some corpus; record §L5-DS4-STEP3b showed the hot set depends on the kind of
// text (the technical record's hot set is no better than random on prose). So
// the engine learns it from its own routing while it runs:
//
//   counters  the router kernel bumps, per layer, how often each expert was
//             chosen and how many choices missed (Ds4Ops::router `stats`);
//   average   every `every` tokens the host reads them back asynchronously
//             and keeps a decayed average per expert (half-life a knob);
//   swaps     per card and slab size, the hottest non-resident experts
//             replace the coldest resident ones, `mb_per_token` x `every`
//             bytes a round at most, copied host -> VRAM on a low-priority
//             stream OFF the critical path; the table is edited between
//             tokens so that no kernel ever sees a half-copied slot.
//
// Everything here runs on the host at a token boundary (Ds4Runner::step,
// before any card's work of the new token); it never waits on a card.
//
// The ordering argument (why the routing and every tap stay bit-identical and
// no slot is read while it is written) is DEEPSEEK4.md section 12.

#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#include "ds4_model.h"
#include "ds4_ops.h"

namespace fk {
namespace ds4 {

struct AdaptConfig {
    int    on = 0;              // --adapt 0|1
    int    every = 16;          // --adapt-every N: tokens between re-placements
    double mb_per_token = 64.0; // --adapt-mb-per-token X: swap bytes a card may move, per token
    double halflife = 2048.0;   // --adapt-halflife T: tokens for a count to weigh half
    double margin = 1.0;        // --adapt-margin M: an incoming expert must beat the outgoing one
    double hyst = 0.25;         //   by M selections AND by the factor (1 + --adapt-hyst H)
    int    verify = 0;          // --adapt-verify 0|1: at the end, every table entry against the mirror
};

class Adapter {
public:
    Adapter(Ds4Model & model, std::vector<Ds4Ops *> ops, const AdaptConfig & cfg);
    // The router's counters for layer il (backend memory on il's card).
    uint32_t * stats(int il) const;
    // A token boundary: the token at `pos` is next, nothing of it is queued.
    void tick(int pos);
    // End of run: totals (and, with --adapt-verify, the check; false on a mismatch).
    bool finish(int pos, FILE * out);

private:
    struct Swap { int il_out, e_out, il_in, e_in; unsigned char * slot; };
    struct Card {
        int dev = 0;
        std::vector<int> layers;                // il, ascending
        uint32_t * dstats = nullptr;            // [layers][ADAPT_STRIDE] on the card
        std::vector<uint32_t> prev;             // the last snapshot ingested
        int last_pos = 0;                       // its position
        int snap_pos = -1;                      // a snapshot in flight, taken at this position
        bool inflight = false;                  // a swap batch copying
        std::vector<Swap> batch;
        long long swaps = 0, rounds = 0;
        double swap_bytes = 0.0, miss_bytes = 0.0;
        long long picks = 0, misses = 0, tokens = 0;
        long long win_swaps = 0; double win_swap_bytes = 0.0;   // installed since the last line
    };
    void ingest(Card & c, const uint32_t * snap, int pos);
    void plan(Card & c, int tokens);
    void install(Card & c);
    void edit(Card & c, int il, const std::vector<int> & experts, const std::vector<const void *> & addr,
              const std::vector<int> & miss);
    int  local(int il) const { return il_local_[(size_t) il]; }

    Ds4Model & model_;
    std::vector<Ds4Ops *> ops_;
    AdaptConfig cfg_;
    std::vector<Card> cards_;
    std::vector<int> il_local_;                 // il -> index within its card's layers
    std::vector<std::vector<double>> ema_;      // [il][e]
    // one summary line per snapshot position once every card has reported it
    struct Round { int n = 0; double miss_b = 0.0; long long picks = 0, misses = 0; int tokens = 0;
                   long long swaps = 0; double swap_b = 0.0; };
    std::map<int, Round> rounds_;
};

} // namespace ds4
} // namespace fk
