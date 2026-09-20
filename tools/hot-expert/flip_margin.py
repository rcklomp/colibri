#!/usr/bin/env python3
# flip_margin.py <ref.dump> <cand.dump>: for positions whose argmax differs, how decided was the reference?
import sys, struct, array, math
def hdr(f):
    m, v, n, V = struct.unpack("<4I", f.read(16)); assert m == 0x444b4c47; return n, V
def probs_top(row, idxs):
    mx = max(row); tot = math.fsum(math.exp(x - mx) for x in row)
    return [math.exp(row[i] - mx) / tot for i in idxs]
fa, fb = open(sys.argv[1], "rb"), open(sys.argv[2], "rb")
n, V = hdr(fa); n2, V2 = hdr(fb); assert (n, V) == (n2, V2) and n > 0
flips = []; margins_all = []
for p in range(n):
    a = array.array("f"); a.frombytes(fa.read(4 * V)); b = array.array("f"); b.frombytes(fb.read(4 * V))
    ia = a.index(max(a)); ib = b.index(max(b))
    if ia != ib:
        pa = probs_top(a, [ia, ib]); pb = probs_top(b, [ia, ib])
        flips.append((p, pa[0], pa[1], pb[0], pb[1]))
print(f"positions={n} flips={len(flips)} ({100*len(flips)/n:.2f}%)")
if not flips:
    print("  no flips: every argmax agrees, nothing to measure")
    sys.exit(0)
gaps = sorted(f[1] - f[2] for f in flips)
for thr in (0.01, 0.02, 0.05, 0.10, 0.20):
    print(f"  ref P(top1)-P(cand's top1) < {thr:.2f}: {sum(g < thr for g in gaps)} of {len(flips)}")
print(f"  median gap {gaps[len(gaps)//2]:.4f}  max gap {gaps[-1]:.4f}  median ref P(top1) among flips {sorted(f[1] for f in flips)[len(flips)//2]:.3f}")
print("  largest-gap flips (pos, refP1, refP(candtop), candP(reftop), candP1):")
for f in sorted(flips, key=lambda f: f[2] - f[1])[:0]: pass
for f in sorted(flips, key=lambda f: -(f[1] - f[2]))[:6]: print("   ", f[0], *(f"{x:.3f}" for x in f[1:]))
