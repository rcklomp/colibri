#!/usr/bin/env python3
"""union_k.py -- (record §L5-SPEC-REPRICE)  python3 -I union_k.py LOG   (LOG = a lookahead-probe log, e.g. ~/bench/franken/glm5/lookahead/prose8k.la.log; stdlib only)
What a k-row verify step would read over the links, from the lookahead probe's decode logs (offline, no GPU).
For every window of k consecutive decode tokens and every MoE layer: the union of the experts the k tokens actually routed to,
and the part of that union that is NOT resident (the miss mask logged at routing time). Bytes = distinct missed experts x slab.
Compare with k separate 1-row steps (sum of per-token missed experts) and with 1 row alone.
"""
import re, sys, collections

REC = re.compile(r"^R pos=(\d+) ph=D il=(\d+) lag=1 .*? act=([\d,]+) .*? miss=([0-9a-f]+) slab=(\d+)")

def load(path):
    rows = collections.defaultdict(dict)       # pos -> il -> (set(act), missset)
    slab = {}
    for ln in open(path, errors="replace"):
        m = REC.match(ln)
        if not m:
            continue
        pos, il = int(m.group(1)), int(m.group(2))
        act = {int(x) for x in m.group(3).split(",")}
        h = m.group(4)
        v = 0                                   # glm5_lookahead_report.py mask_int(): hex char j is nibble j (first char = lowest nibble); bit e SET = expert e is NOT resident
        for j, c in enumerate(h):
            v |= int(c, 16) << (4 * j)
        miss = {e for e in range(288) if (v >> e) & 1}
        rows[pos][il] = (act, miss)
        slab[il] = int(m.group(5))
    return rows, slab

def stats(rows, slab, ks):
    pos = sorted(rows)
    layers = sorted(rows[pos[0]])
    one = 0.0; n_miss = 0; n_act = 0
    for p in pos:
        for il in layers:
            a, ms = rows[p][il]
            one += len(a & ms) * slab[il]; n_miss += len(a & ms); n_act += len(a)
    mean1 = one / len(pos)                       # missed BYTES a token, summed over layers
    print("tokens %d (pos %d..%d), layers %d, miss rate %.3f, missed %.3f GB/token (report: 0.362, 1.451)" % (
        len(pos), pos[0], pos[-1], len(layers), n_miss / n_act, mean1 / 1e9))
    print("%3s %10s %10s %9s %9s %12s" % ("k", "distinct", "miss GB", "x 1 row", "x k rows", "union/k-sep"))
    for k in ks:
        tot_u = tot_um = tot_sep = 0.0; nw = 0
        for i in range(0, len(pos) - k + 1):
            w = pos[i:i + k]
            if w[-1] - w[0] != k - 1:
                continue
            nw += 1
            for il in layers:
                U = set(); UM = set(); sep = 0
                for p in w:
                    a, ms = rows[p][il]
                    U |= a; UM |= (a & ms); sep += len(a & ms)
                tot_u += len(U); tot_um += len(UM) * slab[il]; tot_sep += sep * slab[il]
        if not nw:
            continue
        print("%3d %10.1f %10.3f %9.2f %9.2f %12.2f" % (k, tot_u / nw, tot_um / nw / 1e9, (tot_um / nw) / mean1, (tot_um / nw) / (mean1 * k), (tot_um / nw) / (tot_sep / nw)))

if __name__ == "__main__":
    path = sys.argv[1]
    rows, slab = load(path)
    if not rows:
        sys.exit("no records")
    stats(rows, slab, [1, 2, 3, 4, 5, 6, 8])
