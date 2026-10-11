#!/usr/bin/env python3
"""v1_analyze.py -- V1 stage 4, offline (stdlib only): is the CPU-vs-GPU routing divergence rounding noise or a bug?

  v1_analyze.py <v1 dir>      (the rig's ~/bench/franken/glm5/v1, or a copy of its cpu41/, g41/, cpu2300/moe_ids.txt, r_c1/, r_c1024/)

(1) The first flip (position 40): for every MoE layer, the router's biased scores (`ffn_moe_probs_biased-L`, the selection key) of the CPU arm
    and the GPU at that position: the two top-8 sets, the 8th/9th gap on each side, the largest score difference between the arms.
    Rounding noise: the sets differ only where the gap is of the order of the arms' score difference.
(2) Agreement curves: the fraction of (position, layer) expert sets that are equal, by position bin, for CPU vs GPU chunk 1, CPU vs GPU
    chunk 1024 (shipped, f16 trunk GEMM) and GPU chunk 1 vs GPU chunk 1024 (a known-benign perturbation, PF2). A bug past the indexer's
    threshold shows as a step at 2 048 in the CPU-vs-GPU curve that the GPU-vs-GPU curve does not have.
"""
import os, struct, sys

def tap(d, key):
    p = os.path.join(d, key + ".f32")
    if not os.path.exists(p):
        return None
    b = open(p, "rb").read()
    return list(struct.unpack("<%df" % (len(b) // 4), b))

def routes(path):
    r = {}
    for line in open(path):
        head, _, ids = line.partition(":")
        il, pos = map(int, head.split())
        r[(il, pos)] = frozenset(int(x) for x in ids.split())
    return r

def flip(d):
    a, b = os.path.join(d, "cpu41"), os.path.join(d, "g41")
    print("== (1) position 40: router selection key per layer, CPU arm vs GPU")
    print("layer  sets  cpu_gap8_9   gpu_gap8_9   max|cpu-gpu|  ratio(gap/diff)")
    n_diff = 0
    for il in range(0, 80):
        x, y = tap(a, "ffn_moe_probs_biased-%d" % il), tap(b, "ffn_moe_probs_biased-%d" % il)
        if x is None or y is None:
            continue
        sx = sorted(range(len(x)), key=lambda i: -x[i]); sy = sorted(range(len(y)), key=lambda i: -y[i])
        same = set(sx[:8]) == set(sy[:8])
        gx, gy = x[sx[7]] - x[sx[8]], y[sy[7]] - y[sy[8]]
        md = max(abs(p - q) for p, q in zip(x, y))
        n_diff += not same
        if not same or il % 6 == 0:
            print("%5d  %-5s %.4e   %.4e   %.4e    %.1f%s" % (il, "same" if same else "DIFF", gx, gy, md, gx / md if md else float("inf"),
                  "   cpu %s / gpu %s" % (sorted(set(sx[:8]) - set(sy[:8])), sorted(set(sy[:8]) - set(sx[:8]))) if not same else ""))
    print("layers whose set differs at position 40: %d" % n_diff)

def curves(d):
    ref = os.path.join(d, "cpu2300", "moe_ids.txt")
    c1, c1024 = os.path.join(d, "r_c1", "moe_ids.txt"), os.path.join(d, "r_c1024", "moe_ids.txt")
    R = {k: routes(p) for k, p in (("cpu", ref), ("gpu_c1", c1), ("gpu_c1024", c1024)) if os.path.exists(p)}
    bins = [(0, 64), (64, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 2300), (2300, 2308)]
    print("\n== (2) equal expert sets, fraction of (position, layer) pairs, by position bin (2300-2307 = the greedy tokens)")
    pairs = [("cpu", "gpu_c1"), ("cpu", "gpu_c1024"), ("gpu_c1", "gpu_c1024")]
    print("%-22s" % "pair" + "".join("%12s" % ("%d-%d" % b) for b in bins) + "   first diff (pos, layer)")
    for p, q in pairs:
        if p not in R or q not in R:
            continue
        A, B = R[p], R[q]
        row, first = [], None
        for lo, hi in bins:
            n = e = 0
            for (il, pos), s in A.items():
                if lo <= pos < hi and (il, pos) in B:
                    n += 1; e += s == B[(il, pos)]
            row.append("%12s" % ("%.4f" % (e / n) if n else "-"))
        for k in sorted(A, key=lambda k: (k[1], k[0])):
            if k in B and A[k] != B[k]:
                first = (k[1], k[0]); break
        print("%-22s" % (p + " vs " + q) + "".join(row) + "   %s" % (first,))
    # by layer, the deep bin against the bin before it (a step at 2 048 per layer)
    if "cpu" in R and "gpu_c1" in R and "gpu_c1024" in R:
        print("\nper layer, positions 1536-2047 -> 2048-2299:  cpu vs gpu_c1  |  gpu_c1 vs gpu_c1024")
        layers = sorted({il for il, _ in R["cpu"]})
        def frac(A, B, il, lo, hi):
            n = e = 0
            for pos in range(lo, hi):
                k = (il, pos)
                if k in A and k in B:
                    n += 1; e += A[k] == B[k]
            return e / n if n else float("nan")
        for il in layers:
            print("  layer %2d   %.3f -> %.3f   |   %.3f -> %.3f" % (il, frac(R["cpu"], R["gpu_c1"], il, 1536, 2048), frac(R["cpu"], R["gpu_c1"], il, 2048, 2300),
                  frac(R["gpu_c1"], R["gpu_c1024"], il, 1536, 2048), frac(R["gpu_c1"], R["gpu_c1024"], il, 2048, 2300)))

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench/franken/glm5/v1")
    flip(d)
    curves(d)
