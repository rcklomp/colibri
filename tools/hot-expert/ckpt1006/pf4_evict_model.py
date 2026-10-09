#!/usr/bin/env python3
"""pf4_evict_model.py -- what fewer resident experts cost DECODE (PF4's price; offline, stdlib, no GPU).

    python3 -I pf4_evict_model.py ~/bench/franken/glm5/lookahead/prose8k.la.log [slab_ms]

Input: a lookahead probe log (D-phase R records: one per decode token x MoE layer x lag; `act` = the 8 experts the token used, `miss` = 288-bit residency mask,
printed as 72 hex digits with nibble j holding experts 4j..4j+3, bit set = NOT resident; same parsing as glm5_lookahead_report.mask_int). The two logs of
2026-10-06 (prose8k, chat) are 256 decode tokens x 42 MoE layers each; the model's miss rate reproduces the logged one (0.362 / 0.244) as a parsing check.
Per layer the 256 tokens are split in halves. The layer's resident set (size = resident_at_start of the H line) is ranked by hits in one half (residents never hit
are an unknown pool of count 0, evicted first); the coldest fraction f of the set is evicted, and the hits of the OTHER half that would have been misses are counted
(both directions, averaged). That is a hold-out: the ranking is made without seeing the scored tokens, as an adapter's history is. Upper bound: random eviction
(f x all hits). Output: extra missed experts a token over the 42 layers, and ms a token at `slab_ms` per missed expert on the critical path (default 0.26: the
decode trace's 35.2 ms critical fetch over ~135 missed slabs a token, §L5-GLM-DECODE-TRACE).
Fractions: the resident set is ~3 800 experts at --expert-gb 17; -1 / -2 / -3 GB a card = 5.9 / 11.8 / 17.6 % of it.
"""
import collections, math, re, sys

def load(path):
    resident, recs = {}, collections.defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            if line.startswith("H layer"):
                m = re.search(r"il=(\d+).*resident_at_start=(\d+)", line)
                resident[int(m.group(1))] = int(m.group(2))
            elif line.startswith("R ") and "lag=1 " in line and "ph=D" in line:
                pos = int(re.search(r" pos=(\d+)", line).group(1))
                il = int(re.search(r" il=(\d+)", line).group(1))
                act = [int(x) for x in re.search(r" act=([0-9,]+)", line).group(1).split(",")]
                miss = 0
                for j, c in enumerate(re.search(r" miss=([0-9a-f]+)", line).group(1)):
                    miss |= int(c, 16) << (4 * j)
                recs[il][pos] = (act, miss)
    return resident, recs

def analyse(path, fracs):
    resident, recs = load(path)
    out = {f: [0.0, 0.0] for f in fracs}      # [hold-out extra misses a token, random extra misses a token]
    hits_tot = miss_tot = 0.0
    ntok = 0
    for il, byp in sorted(recs.items()):
        order = sorted(byp)
        n = len(order)
        if n < 16:
            continue
        ntok = n
        hits = [[e for e in byp[p][0] if not (byp[p][1] >> e) & 1] for p in order]
        hits_tot += sum(len(h) for h in hits) / n
        miss_tot += sum(8 - len(h) for h in hits) / n
        R = resident.get(il, 0)
        for rank_half, score_half in ((hits[:n // 2], hits[n // 2:]), (hits[n // 2:], hits[:n // 2])):
            c1 = collections.Counter(e for h in rank_half for e in h)
            ranked = sorted(c1.items(), key=lambda kv: kv[1])          # coldest first
            unseen = max(R - len(c1), 0)
            for f in fracs:
                K = int(math.ceil(f * R))
                k_unseen = min(K, unseen)
                evicted = {e for e, _ in ranked[:min(K - k_unseen, len(ranked))]}
                lost = 0.0
                for h in score_half:
                    for e in h:
                        if e in c1:
                            lost += e in evicted
                        elif unseen:
                            lost += k_unseen / unseen                  # a resident the ranking half never saw: it is in the unknown pool
                out[f][0] += 0.5 * lost / len(score_half)
                out[f][1] += 0.5 * f * sum(len(h) for h in score_half) / len(score_half)
    return out, hits_tot, miss_tot, ntok

if __name__ == "__main__":
    path = sys.argv[1]
    slab_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 0.26
    fracs = [0.059, 0.118, 0.176]
    res, th, tm, n = analyse(path, fracs)
    print(f"{path}: {n} decode tokens, {th:.1f} hits / {tm:.1f} misses a token over the MoE layers (miss rate {tm / (th + tm):.3f})")
    for f in fracs:
        o, r = res[f]
        print(f"  evict {f * 100:4.1f} % of the resident set (~{f * 17:.0f} GB a card of 17): +{o:5.2f} missed experts a token hold-out, +{r:5.2f} random"
              f"  = +{o * slab_ms:4.1f} .. +{r * slab_ms:4.1f} ms/token at {slab_ms} ms a slab")
