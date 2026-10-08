#!/usr/bin/env python3
"""route_report.py -- PF0 (d): the in-place hit rate of a chunked prefill, by layer, from a `--debug-route` dump (build_pf0dbg.sh).

    python3 -I route_report.py PREFIX [--slab-mb 11.5] [--chunk 1024]

PREFIX is the --debug-route argument; the dump is PREFIX.dev0 / .dev1 / .dev2, one line per (MoE layer, chunk):
    R il=<layer> dev=<owner card> T=<rows> miss=<288 chars 0/1: 1 = the expert is NOT resident for this layer> ids=<T*8 router ids,>

What an in-place chunk does (glm5_gpu.inc chunk_slots): every expert that at least one row of the chunk picked and that is host-side ("missed") is read ONCE over a link,
whatever the number of rows that use it; a resident expert is read from VRAM. So the link bytes of a chunk are  (#used missed experts) x slab,  and the per-token cost falls as 1/T.
Reported per layer (full chunks only; a short tail chunk is listed apart):
  res        resident experts of the layer (miss flag 0)
  used       distinct experts the chunk used, mean over chunks
  umiss      of those, the host-side ones = slab reads over the links per chunk
  ehit       share of the used experts that were resident          (expert-level hit rate: what the links are spared)
  ahit       share of the (row, slot) assignments that hit a resident expert (what the DECODE-style hit rate would say)
  whatif     umiss with the best resident set of the same size chosen in hindsight FOR THIS PROMPT (the experts used by the most chunks): the ceiling of any prefill-specific placement
  split      the same with the set chosen on the first half of the chunks and tested on the second half (no hindsight)
Totals are sums over layers; MB/token uses --slab-mb (bytes of gate+up+down of one expert; calibrate it against the profile's link MB per token).
"""
import argparse, collections, re, sys

LINE = re.compile(r"R il=(\d+) dev=(\d+) T=(\d+) miss=([01]+) ids=([\d,]*)")


def load(prefix):
    rows = collections.defaultdict(list)       # il -> [(T, miss, ids)]
    dev_of = {}
    n = 0
    for d in range(8):
        try:
            f = open("%s.dev%d" % (prefix, d))
        except OSError:
            continue
        with f:
            for ln in f:
                m = LINE.match(ln)
                if not m:
                    continue
                il, dev, T = int(m.group(1)), int(m.group(2)), int(m.group(3))
                ids = [int(x) for x in m.group(5).split(",") if x]
                if len(ids) != T * 8:
                    print("WARN il=%d: %d ids for T=%d" % (il, len(ids), T), file=sys.stderr)
                rows[il].append((T, m.group(4), ids))
                dev_of[il] = dev
                n += 1
    return rows, dev_of, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--slab-mb", type=float, default=0.0, help="MB of one expert's gate+up+down (0: report slab reads only)")
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()
    rows, dev_of, n = load(a.prefix)
    if not rows:
        sys.exit("no route lines under %s.dev*" % a.prefix)
    layers = sorted(rows)
    n_chunks = max(len(v) for v in rows.values())
    print("route dump %s: %d lines, %d MoE layers, up to %d chunks of %d rows" % (a.prefix, n, len(layers), n_chunks, a.chunk))
    tot = collections.Counter()
    print("%4s %3s %4s %6s %6s %6s %6s %7s %7s %7s %7s" % ("il", "dev", "res", "used", "umiss", "ehit%", "ahit%", "whatif", "split", "w/in%", "s/in%"))
    miss_changes = 0
    chunk_used = collections.defaultdict(list)   # chunk index -> [umiss per layer]
    tails = []
    for il in layers:
        calls = rows[il]
        full = [c for c in calls if c[0] == a.chunk]
        tails += [(il, c[0]) for c in calls if c[0] != a.chunk]
        if not full:
            continue
        miss = full[0][1]
        if any(c[1] != miss for c in full):
            miss_changes += 1
        M = {e for e, ch in enumerate(miss) if ch == "1"}
        R = len(miss) - len(M)
        used_sets, umiss, uhit_a = [], [], []
        for k, (T, _m, ids) in enumerate(full):
            U = set(ids)
            used_sets.append(U)
            um = len(U & M)
            umiss.append(um)
            chunk_used[k].append(um)
            uhit_a.append(sum(1 for x in ids if x not in M) / float(len(ids)))
        nU = sum(len(u) for u in used_sets) / float(len(used_sets))
        mean_umiss = sum(umiss) / float(len(umiss))
        ehit = 100.0 * (1 - mean_umiss / nU) if nU else 0.0
        ahit = 100.0 * sum(uhit_a) / len(uhit_a)
        # hindsight ceiling: R experts used by the most chunks
        freq = collections.Counter()
        for U in used_sets:
            freq.update(U)
        best = {e for e, _ in freq.most_common(R)}
        wi = sum(len(U - best) for U in used_sets) / float(len(used_sets))
        # train on the first half, test on the second
        h = max(1, len(used_sets) // 2)
        fq = collections.Counter()
        for U in used_sets[:h]:
            fq.update(U)
        tr = {e for e, _ in fq.most_common(R)}
        test = used_sets[h:] or used_sets
        sp_now = sum(len(U & M) for U in test) / float(len(test))
        sp = sum(len(U - tr) for U in test) / float(len(test))
        print("%4d %3d %4d %6.1f %6.1f %6.1f %6.1f %7.1f %7.1f %7.1f %7.1f" % (
            il, dev_of[il], R, nU, mean_umiss, ehit, ahit, wi, sp, 100.0 * (mean_umiss - wi) / mean_umiss if mean_umiss else 0,
            100.0 * (sp_now - sp) / sp_now if sp_now else 0))
        tot["layers"] += 1; tot["res"] += R; tot["used"] += nU; tot["umiss"] += mean_umiss; tot["whatif"] += wi
        tot["sp_now"] += sp_now; tot["sp"] += sp
    print("-" * 78)
    if tot["layers"]:
        L = tot["layers"]
        print("sum over %d layers: resident %d, used/chunk %.0f, umiss/chunk %.1f (= slab reads over the links), whatif %.1f (-%.1f %%), "
              "split-half now %.1f -> %.1f (-%.1f %%)" % (L, tot["res"], tot["used"], tot["umiss"], tot["whatif"],
                                                          100.0 * (tot["umiss"] - tot["whatif"]) / tot["umiss"], tot["sp_now"], tot["sp"],
                                                          100.0 * (tot["sp_now"] - tot["sp"]) / tot["sp_now"] if tot["sp_now"] else 0))
        print("expert-level hit rate overall %.1f %%; reads per token %.4f slabs" % (100.0 * (1 - tot["umiss"] / tot["used"]), tot["umiss"] / a.chunk))
        if a.slab_mb > 0:
            print("link MB per token at %.2f MB a slab: %.1f (hindsight set %.1f, split-half set %.1f x %.2f of now)" % (
                a.slab_mb, tot["umiss"] * a.slab_mb / a.chunk, tot["whatif"] * a.slab_mb / a.chunk,
                tot["sp"] * a.slab_mb / a.chunk, tot["sp"] / tot["sp_now"] if tot["sp_now"] else 0))
    print("umiss per chunk (sum over layers), by chunk index: " + " ".join("%d:%.0f" % (k, sum(v)) for k, v in sorted(chunk_used.items())))
    if miss_changes:
        print("NOTE: the miss flags of %d layers changed between chunks (a placement swap during the prompt)" % miss_changes)
    if tails:
        print("short chunks (not in the table): " + ", ".join("il %d T=%d" % t for t in tails[:6]) + (" ..." if len(tails) > 6 else ""))


if __name__ == "__main__":
    main()
