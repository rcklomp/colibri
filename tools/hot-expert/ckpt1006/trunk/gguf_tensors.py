#!/usr/bin/env python3
"""gguf_tensors.py SHARD... -- list name, ggml type, dims, bytes of every tensor in GGUF shards (header only, stdlib)."""
import struct, sys

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17)}

def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

def rstr(f):
    n = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")

def skip_val(f, t):
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sz: f.read(sz[t]); return
    if t == 8: rstr(f); return
    if t == 9:
        at = rd(f, "I"); n = rd(f, "Q")
        if at in sz: f.read(sz[at] * n); return
        for _ in range(n): skip_val(f, at)
        return
    raise ValueError("kv type %d" % t)

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = rd(f, "I"); nt = rd(f, "Q"); nkv = rd(f, "Q")
        for _ in range(nkv):
            rstr(f); skip_val(f, rd(f, "I"))
        for _ in range(nt):
            name = rstr(f); nd = rd(f, "I"); dims = [rd(f, "Q") for _ in range(nd)]; ty = rd(f, "I"); rd(f, "Q")
            tn, bs, bb = TYPES.get(ty, ("T%d" % ty, 1, 0))
            ne = 1
            for d in dims: ne *= d
            print("%s\t%s\t%s\t%d" % (name, tn, "x".join(map(str, dims)), ne // bs * bb))
