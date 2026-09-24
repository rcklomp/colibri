// tools/hot-expert/franken/decode/glm5_adapt.cpp -- see ds4_adapt.h and
// DEEPSEEK4.md section 12 (the ordering argument).

#include "glm5_adapt.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace fk {
namespace glm5 {

namespace {
size_t slab_of(const LayerWeights & L) { return L.et.sz_g + L.et.sz_u + L.et.sz_d; }
}

Adapter::Adapter(Glm5Model & model, std::vector<ds4::Ds4Ops *> ops, const AdaptConfig & cfg)
    : model_(model), ops_(std::move(ops)), cfg_(cfg) {
    if (cfg_.every < 1) cfg_.every = 1;
    if (cfg_.halflife <= 0.0) cfg_.halflife = 1.0;
    il_local_.assign(N_LAYER, -1);
    ema_.assign(N_LAYER, std::vector<double>(N_EXPERT, 0.0));
    cards_.resize((size_t) model_.n_devices());
    for (int d = 0; d < model_.n_devices(); ++d) cards_[(size_t) d].dev = d;
    for (int il = model_.il0(); il <= model_.il1(); ++il) {
        const LayerWeights & L = model_.layer(il);
        if (!L.moe) continue;                     // GLM: layers 0-2 are dense
        if (L.mir_view.size() != (size_t) 3 * N_EXPERT || L.mir_src.size() != (size_t) 3 * N_EXPERT)
            throw std::runtime_error("--adapt 1 needs the host mirror: give --placement (layer " +
                                     std::to_string(il) + " has none)");
        Card & c = cards_[(size_t) L.dev];
        il_local_[(size_t) il] = (int) c.layers.size();
        c.layers.push_back(il);
    }
    for (Card & c : cards_) {
        if (c.layers.empty()) continue;
        const size_t n = c.layers.size() * (size_t) G_ADAPT_STRIDE;
        c.dstats = ops_[(size_t) c.dev]->adapt_stats_alloc(n);
        if (!c.dstats) throw std::runtime_error("--adapt 1: this backend has no route counters");
        c.prev.assign(n, 0);
    }
}

uint32_t * Adapter::stats(int il) const {
    const Card & c = cards_[(size_t) model_.layer(il).dev];
    return c.dstats + (size_t) local(il) * G_ADAPT_STRIDE;
}

// Host shadow first (tab_host / miss_host are what the planner and --verify
// read), then the same entries on the card, one launch per ds4::ADAPT_MAX_EDITS.
void Adapter::edit(Card & c, int il, const std::vector<int> & experts,
                   const std::vector<const void *> & addr, const std::vector<int> & miss) {
    LayerWeights & L = model_.layer_mut(il);
    ds4::TableEdit ed;
    for (size_t i = 0; i < experts.size(); ++i) {
        const int e = experts[i];
        L.tab_host[(size_t) e]                = addr[3 * i];
        L.tab_host[(size_t) (N_EXPERT + e)]   = addr[3 * i + 1];
        L.tab_host[(size_t) (2 * N_EXPERT + e)] = addr[3 * i + 2];
        L.miss_host[(size_t) e] = miss[i];
        ed.e[ed.n] = e; ed.g[ed.n] = addr[3 * i]; ed.u[ed.n] = addr[3 * i + 1]; ed.d[ed.n] = addr[3 * i + 2];
        ed.miss[ed.n] = miss[i];
        if (++ed.n == ds4::ADAPT_MAX_EDITS) { ops_[(size_t) c.dev]->adapt_table_edit(L.et, ed); ed.n = 0; }
    }
    if (ed.n) ops_[(size_t) c.dev]->adapt_table_edit(L.et, ed);
    int n_res = 0;
    for (int e = 0; e < N_EXPERT; ++e) n_res += L.miss_host[(size_t) e] == 0;
    L.n_resident = n_res;
}

void Adapter::tick(int pos) {
    bool any_snap = false;
    int last = 0;
    for (Card & c : cards_) {
        if (c.layers.empty()) continue;
        ds4::Ds4Ops & o = *ops_[(size_t) c.dev];
        // (1) a batch whose copies have landed: point the tables at it
        if (c.inflight && o.adapt_copy_landed()) install(c);
        // (2) a snapshot that has landed: fold it into the average, then plan
        if (c.snap_pos >= 0) {
            if (const uint32_t * s = o.adapt_snapshot_poll()) {
                const int sp = c.snap_pos;
                c.snap_pos = -1;
                const int tokens = sp - c.last_pos;
                ingest(c, s, sp);
                if (!c.inflight) plan(c, tokens);
            }
        }
        any_snap |= c.snap_pos >= 0;
        last = c.last_pos;
    }
    // (3) every `every` tokens, a new snapshot on every card at once (so each
    // round reports all cards), once the previous one has been ingested
    if (!any_snap && pos - last >= cfg_.every)
        for (Card & c : cards_) {
            if (c.layers.empty()) continue;
            ops_[(size_t) c.dev]->adapt_snapshot(c.dstats, c.prev.size());
            c.snap_pos = pos;
        }
}

void Adapter::ingest(Card & c, const uint32_t * snap, int sp) {
    const int tokens = std::max(1, sp - c.last_pos);
    const double decay = std::pow(0.5, (double) tokens / cfg_.halflife);
    double miss_b = 0.0;
    long long picks = 0, misses = 0;
    for (size_t j = 0; j < c.layers.size(); ++j) {
        const int il = c.layers[j];
        const uint32_t * a = snap + j * G_ADAPT_STRIDE;
        const uint32_t * b = c.prev.data() + j * G_ADAPT_STRIDE;
        std::vector<double> & m = ema_[(size_t) il];
        for (int e = 0; e < N_EXPERT; ++e) m[(size_t) e] = m[(size_t) e] * decay + (double) (uint32_t) (a[e] - b[e]);
        const uint32_t dm = a[G_ADAPT_MISS] - b[G_ADAPT_MISS], dp = a[G_ADAPT_PICKS] - b[G_ADAPT_PICKS];
        miss_b += (double) dm * (double) slab_of(model_.layer(il));
        misses += dm; picks += dp;
    }
    c.prev.assign(snap, snap + c.prev.size());
    c.last_pos = sp;
    c.miss_bytes += miss_b; c.picks += picks; c.misses += misses; c.tokens += tokens;
    int n_res = 0;
    for (int il : c.layers) n_res += model_.layer(il).n_resident;
    std::printf("adapt dev=%d pos=%d tokens=%d hit=%.4f miss_mb_per_token=%.1f swaps=%lld swap_mb=%.1f "
                "resident=%d inflight=%d\n", c.dev, sp, tokens, picks ? 1.0 - (double) misses / picks : 1.0,
                miss_b / tokens / 1e6, c.win_swaps, c.win_swap_bytes / 1e6, n_res, (int) c.inflight);
    Round & r = rounds_[sp];
    r.n++; r.miss_b += miss_b; r.picks += picks; r.misses += misses; r.tokens = tokens;
    r.swaps += c.win_swaps; r.swap_b += c.win_swap_bytes;
    c.win_swaps = 0; c.win_swap_bytes = 0.0;
    int active = 0;
    for (const Card & k : cards_) active += !k.layers.empty();
    if (r.n == active) {
        std::printf("adapt_all pos=%d tokens=%d hit=%.4f miss_mb_per_token=%.1f swaps=%lld swap_mb=%.1f\n",
                    sp, r.tokens, r.picks ? 1.0 - (double) r.misses / r.picks : 1.0, r.miss_b / r.tokens / 1e6,
                    r.swaps, r.swap_b / 1e6);
        std::fflush(stdout);
        rounds_.erase(sp);
    }
}

// Per card and slab layout (a slot can only take an expert of its own size:
// layers 26 and 42 carry other formats), pair the hottest non-resident
// experts with the coldest resident ones while the incoming one is clearly
// hotter; take the pairs by gain until the round's byte budget is spent.
// Evict at once (table edit on the main stream), copy on the copy stream.
void Adapter::plan(Card & c, int tokens) {
    struct Cand { double ema; int il, e; };
    struct Pair { double gain; int il_out, e_out, il_in, e_in; size_t slab; };
    std::map<std::tuple<size_t, size_t, size_t>, std::pair<std::vector<Cand>, std::vector<Cand>>> cls;
    for (int il : c.layers) {
        const LayerWeights & L = model_.layer(il);
        auto & io = cls[std::make_tuple(L.et.sz_g, L.et.sz_u, L.et.sz_d)];
        for (int e = 0; e < N_EXPERT; ++e) {
            const double v = ema_[(size_t) il][(size_t) e];
            if (L.miss_host[(size_t) e] == 0) io.second.push_back({v, il, e});
            else if (v > cfg_.margin) io.first.push_back({v, il, e});
        }
    }
    std::vector<Pair> pairs;
    for (auto & kv : cls) {
        auto & in = kv.second.first;
        auto & out = kv.second.second;
        std::sort(in.begin(), in.end(), [](const Cand & a, const Cand & b) {
            return a.ema != b.ema ? a.ema > b.ema : std::make_pair(a.il, a.e) < std::make_pair(b.il, b.e); });
        std::sort(out.begin(), out.end(), [](const Cand & a, const Cand & b) {
            return a.ema != b.ema ? a.ema < b.ema : std::make_pair(a.il, a.e) < std::make_pair(b.il, b.e); });
        const size_t slab = std::get<0>(kv.first) + std::get<1>(kv.first) + std::get<2>(kv.first);
        for (size_t i = 0; i < in.size() && i < out.size(); ++i) {
            if (!(in[i].ema > out[i].ema * (1.0 + cfg_.hyst) + cfg_.margin)) break;
            pairs.push_back({in[i].ema - out[i].ema, out[i].il, out[i].e, in[i].il, in[i].e, slab});
        }
    }
    if (pairs.empty()) return;
    std::stable_sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) { return a.gain > b.gain; });
    const double budget = cfg_.mb_per_token * 1e6 * std::max(1, tokens);
    double bytes = 0.0;
    std::vector<Swap> batch;
    for (const Pair & p : pairs) {
        if (bytes + (double) p.slab > budget) continue;
        bytes += (double) p.slab;
        const LayerWeights & L = model_.layer(p.il_out);
        batch.push_back({p.il_out, p.e_out, p.il_in, p.e_in,
                         (unsigned char *) const_cast<void *>(L.tab_host[(size_t) p.e_out])});
    }
    if (batch.empty()) return;
    ds4::Ds4Ops & o = *ops_[(size_t) c.dev];
    // EVICT: each outgoing expert's entry now points at its mirror copy (miss).
    std::map<int, std::vector<const Swap *>> by_out;
    for (const Swap & s : batch) by_out[s.il_out].push_back(&s);
    for (auto & kv : by_out) {
        const LayerWeights & L = model_.layer(kv.first);
        std::vector<int> ex, ms;
        std::vector<const void *> ad;
        for (const Swap * s : kv.second) {
            ex.push_back(s->e_out);
            for (int w = 0; w < 3; ++w) ad.push_back(L.mir_view[(size_t) (3 * s->e_out + w)]);
            ms.push_back((int) slab_of(L));
            model_.set_resident(kv.first, s->e_out, false);
        }
        edit(c, kv.first, ex, ad, ms);
    }
    // COPY: behind the evictions and everything before them, on the copy stream
    o.adapt_copy_begin();
    for (const Swap & s : batch) {
        const LayerWeights & L = model_.layer(s.il_in);
        const size_t sg = L.et.sz_g, su = L.et.sz_u, sd = L.et.sz_d;
        o.adapt_copy(s.slot,           L.mir_src[(size_t) (3 * s.e_in)],     sg);
        o.adapt_copy(s.slot + sg,      L.mir_src[(size_t) (3 * s.e_in + 1)], su);
        o.adapt_copy(s.slot + sg + su, L.mir_src[(size_t) (3 * s.e_in + 2)], sd);
    }
    o.adapt_copy_end();
    c.batch = std::move(batch);
    c.inflight = true;
    ++c.rounds;
}

// INSTALL: the copies have landed; the main stream waits on them (free now)
// and the incoming experts' entries are pointed at their slots.
void Adapter::install(Card & c) {
    ds4::Ds4Ops & o = *ops_[(size_t) c.dev];
    o.adapt_copy_join();
    std::map<int, std::vector<const Swap *>> by_in;
    for (const Swap & s : c.batch) by_in[s.il_in].push_back(&s);
    for (auto & kv : by_in) {
        const LayerWeights & L = model_.layer(kv.first);
        std::vector<int> ex, ms;
        std::vector<const void *> ad;
        for (const Swap * s : kv.second) {
            ex.push_back(s->e_in);
            ad.push_back(s->slot); ad.push_back(s->slot + L.et.sz_g); ad.push_back(s->slot + L.et.sz_g + L.et.sz_u);
            ms.push_back(0);
            model_.set_resident(kv.first, s->e_in, true);
            c.win_swap_bytes += (double) slab_of(L);
            c.swap_bytes += (double) slab_of(L);
        }
        edit(c, kv.first, ex, ad, ms);
    }
    c.win_swaps += (long long) c.batch.size();
    c.swaps += (long long) c.batch.size();
    c.batch.clear();
    c.inflight = false;
}

bool Adapter::finish(int pos, FILE * out) {
    bool ok = true;
    for (Card & c : cards_) {
        if (c.layers.empty()) continue;
        std::fprintf(out, "adapt_total dev=%d pos=%d tokens=%lld hit=%.4f miss_mb_per_token=%.1f swaps=%lld "
                          "swap_gb=%.2f rounds=%lld inflight=%d\n", c.dev, pos, c.tokens,
                     c.picks ? 1.0 - (double) c.misses / c.picks : 1.0,
                     c.tokens ? c.miss_bytes / c.tokens / 1e6 : 0.0, c.swaps, c.swap_bytes / 1e9, c.rounds,
                     (int) c.inflight);
        if (!cfg_.verify) continue;
        // --adapt-verify: (1) the card's table equals the host shadow;
        // (2) every entry's bytes equal the expert's mirror copy -- a resident
        // expert's VRAM slot compared on the card with its pinned host copy.
        ds4::Ds4Ops & o = *ops_[(size_t) c.dev];
        Backend & b = model_.dev(c.dev);
        long long bad_entries = 0, bad_bytes = 0, checked = 0;
        for (int il : c.layers) {
            const LayerWeights & L = model_.layer(il);
            std::vector<const void *> tab((size_t) 3 * N_EXPERT);
            std::vector<int> miss((size_t) N_EXPERT);
            b.download(tab.data(), L.et.gate, (size_t) N_EXPERT * sizeof(void *));
            b.download(tab.data() + N_EXPERT, L.et.up, (size_t) N_EXPERT * sizeof(void *));
            b.download(tab.data() + 2 * N_EXPERT, L.et.down, (size_t) N_EXPERT * sizeof(void *));
            b.download(miss.data(), L.et.miss_bytes, (size_t) N_EXPERT * sizeof(int));
            const size_t sz[3] = {L.et.sz_g, L.et.sz_u, L.et.sz_d};
            for (int e = 0; e < N_EXPERT; ++e) {
                for (int w = 0; w < 3; ++w)
                    bad_entries += tab[(size_t) (w * N_EXPERT + e)] != L.tab_host[(size_t) (w * N_EXPERT + e)];
                bad_entries += miss[(size_t) e] != L.miss_host[(size_t) e];
                bad_entries += (L.miss_host[(size_t) e] == 0) != model_.resident(il, e);
                if (L.miss_host[(size_t) e] != 0) {
                    // host-mapped: it must be the mirror itself
                    for (int w = 0; w < 3; ++w)
                        bad_entries += L.tab_host[(size_t) (w * N_EXPERT + e)] != L.mir_view[(size_t) (3 * e + w)];
                    continue;
                }
                for (int w = 0; w < 3; ++w)
                    bad_bytes += o.adapt_compare(L.tab_host[(size_t) (w * N_EXPERT + e)],
                                                 L.mir_view[(size_t) (3 * e + w)], sz[w]);
                ++checked;
            }
        }
        std::fprintf(out, "adapt_verify dev=%d resident_checked=%lld bad_entries=%lld bad_bytes=%lld %s\n",
                     c.dev, checked, bad_entries, bad_bytes, (bad_entries || bad_bytes) ? "FAIL" : "PASS");
        ok &= !bad_entries && !bad_bytes;
    }
    return ok;
}

} // namespace glm5
} // namespace fk
