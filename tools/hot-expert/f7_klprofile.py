#!/usr/bin/env python3
"""Per-position-bucket mean KL between two GLKD dumps.

F7: every component of the GPU path measures at fp32 parity per step, yet the
end-to-end mean KL is 166x the CPU's. This says WHERE along the sequence the
divergence appears -- flat-and-large from position 0 means a defect in the
first prefill chunk, a ramp means amplification of a small per-step difference.
"""
import sys, struct, array, math

def load(p):
    d = open(p, "rb").read()
    _, _, n, V = struct.unpack("<4I", d[:16])
    return n, V, d

def main(a, b, nb=12, lo=0, hi=0):
    na, Va, da = load(a)
    nbv, Vb, db = load(b)
    assert (na, Va) == (nbv, Vb), "shape mismatch"
    if hi <= 0 or hi > na: hi = na
    rng = hi - lo
    step = (rng + nb - 1) // nb
    print(f"positions={na} vocab={Va} buckets of {step}")
    print(f"{'range':>14} {'meanKL':>12} {'maxKL':>12} {'top1%':>8}")
    for s in range(lo, hi, step):
        e = min(s + step, hi)
        tot, mx, agree = 0.0, 0.0, 0
        for i in range(s, e):
            ra = array.array('f'); ra.frombytes(da[16 + i*Va*4: 16 + (i+1)*Va*4])
            rb = array.array('f'); rb.frombytes(db[16 + i*Va*4: 16 + (i+1)*Va*4])
            ma = max(ra); mb = max(rb)
            ea = array.array('f', (math.exp(x - ma) for x in ra)); za = math.fsum(ea)
            zb = math.fsum(math.exp(x - mb) for x in rb)
            lza = math.log(za) + ma; lzb = math.log(zb) + mb
            epa = sum(p*r for p, r in zip(ea, ra)) / za
            epb = sum(p*c for p, c in zip(ea, rb)) / za
            kl = epa - lza - epb + lzb
            tot += kl; mx = max(mx, kl)
            if ra.index(ma) == rb.index(mb): agree += 1
        n = e - s
        print(f"{s:6d}-{e-1:<7d} {tot/n:12.6g} {mx:12.6g} {100.0*agree/n:8.2f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0, int(sys.argv[5]) if len(sys.argv) > 5 else 0)
