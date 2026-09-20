#!/usr/bin/env python3
"""f7_ctxdiff.py <cpu_chunk.bin> <gpu_chunk.bin> -- diff the `context` a CPU-path
run and a GPU-path run produced for the SAME layer-chunk, row by row.

F7: every component of the GPU attention path measures at fp32 parity per step
(replay against float64, 11 layers, every row, both selection regimes) and the
batched o-projection is bit-identical to the per-row one, yet the knob-on engine
run tracks the knob-off one only to a point and then diverges. A replay feeds
the CPU run's inputs through the GPU core; this reads what the GPU core actually
wrote IN SITU, so a difference here is the core misbehaving inside the engine
and a null here says the divergence is not the core's output.

Needs no GPU and no rig lock. Files are written by GLM53_MLA_ATTN_DUMP.
"""
import sys, struct, array

def load(p):
    f = open(p, "rb")
    h = struct.unpack("<13i", f.read(52))
    magic, ver, tokens, H, L, V, width, seen, base, vfmt, vgs, vrows, vpacked = h
    assert magic == 0x50443746 and ver == 1, p
    vng = struct.unpack("<i", f.read(4))[0]
    scale = struct.unpack("<f", f.read(4))[0]
    f.seek(tokens * H * L * 4 + seen * L * 4 + tokens * width * 4, 1)
    ctx = array.array('f'); ctx.fromfile(f, tokens * H * V)
    f.close()
    return dict(tokens=tokens, H=H, V=V, base=base, seen=seen, width=width), ctx

def main(a, b):
    ma, ca = load(a)
    mb, cb = load(b)
    assert (ma["tokens"], ma["H"], ma["V"], ma["base"]) == (mb["tokens"], mb["H"], mb["V"], mb["base"]), \
        "different chunks: %s vs %s" % (ma, mb)
    T, H, V, base = ma["tokens"], ma["H"], ma["V"], ma["base"]
    print("chunk base=%d tokens=%d H=%d V=%d seen=%d" % (base, T, H, V, ma["seen"]))
    worst, wrow, whead = 0.0, -1, -1
    rels = []
    nz = 0
    for t in range(T):
        o = t * H * V
        mx, mref, hw = 0.0, 0.0, -1
        for i in range(H * V):
            x, y = ca[o + i], cb[o + i]
            if abs(x) > mref: mref = abs(x)
            d = abs(x - y)
            if d > mx: mx, hw = d, i // V
        rel = mx / mref if mref else 0.0
        if mx: nz += 1
        rels.append(rel)
        if rel > worst: worst, wrow, whead = rel, base + t, hw
        if t < 4 or rel > 1e-4 or (t % 64 == 0):
            print("  t=%-6d maxabs=%.4e rel=%.4e head=%d" % (base + t, mx, rel, hw))
    rels.sort()
    print("WORST rel=%.4e at row %d head %d ; rows differing at all: %d/%d"
          % (worst, wrow, whead, nz, T))
    print("STATS rel p50=%.4e p90=%.4e p99=%.4e max=%.4e   (p50 is the honest one: a"
          " per-row max is dominated by near-degenerate softmax ties)"
          % (rels[T // 2], rels[int(T * 0.9)], rels[int(T * 0.99)], rels[-1]))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
