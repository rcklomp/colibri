#!/usr/bin/env python3
"""gguf_tensors.py -- per-tensor types and sizes from the first bytes of GGUF shards (no download of the weights).

    python3 -I gguf_tensors.py QUANT_NAME shard_header_1.bin [shard_header_2.bin ...]

Prints, for the routed experts (blk.N.ffn_{gate,up,down}_exps.weight), the type of each projection by layer and the bytes of one expert slab, plus the byte
totals by tensor class (routed experts, shared experts, attention/trunk, embeddings/head). Block sizes (bytes per 256 weights unless noted) from ggml.
"""
import collections, re, struct, sys

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
         22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17)}

def parse(path):
    b = open(path, "rb").read()
    p = 0
    def u32():
        nonlocal p; v = struct.unpack_from("<I", b, p)[0]; p += 4; return v
    def u64():
        nonlocal p; v = struct.unpack_from("<Q", b, p)[0]; p += 8; return v
    def string():
        nonlocal p; n = u64(); s = b[p:p + n].decode("utf-8", "replace"); p += n; return s
    assert b[:4] == b"GGUF", path
    p = 4; ver = u32(); nt = u64(); nkv = u64()
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        nonlocal p
        if t in sizes: p += sizes[t]
        elif t == 8: string()
        elif t == 9:
            et = u32(); n = u64()
            for _ in range(n): skip(et)
        else: raise ValueError("kv type %d" % t)
    for _ in range(nkv):
        string(); skip(u32())
    out = []
    for _ in range(nt):
        name = string(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        n = 1
        for d in dims: n *= d
        tn, blk, bpb = TYPES[ty]
        out.append((name, dims, tn, n // blk * bpb))
    return out

if __name__ == "__main__":
    quant = sys.argv[1]
    tens = [t for f in sys.argv[2:] for t in parse(f)]
    cls = collections.Counter(); cnt = collections.Counter()
    exp = collections.defaultdict(dict)       # layer -> proj -> (type, bytes of the whole tensor, dims)
    for name, dims, tn, nb in tens:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if m:
            exp[int(m.group(1))][m.group(2)] = (tn, nb, dims); c = "routed experts"
        elif "_shexp" in name: c = "shared experts"
        elif name.startswith(("token_embd", "output", "per_layer", "rope")) or "embd" in name: c = "embeddings / head"
        elif name.startswith("blk."): c = "trunk (attention, KDA, indexer, dense FFN, router)"
        else: c = "other"
        cls[c] += nb; cnt[c] += 1
    print(f"== {quant}: {len(tens)} tensors in the shards given")
    for c, v in cls.most_common(): print(f"   {c:55s} {v / 1e9:8.2f} GB  ({cnt[c]} tensors)")
    kinds = collections.Counter()
    slab = {}
    for il, d in sorted(exp.items()):
        ne = d["gate"][2][-1] if d["gate"][2] else 0
        s = sum(d[k][1] for k in ("gate", "up", "down")) / ne
        slab[il] = s
        kinds[(d["gate"][0], d["up"][0], d["down"][0], round(s))] += 1
    print(f"   routed-expert layers {min(exp)}..{max(exp)}: {len(exp)}; slab = gate+up+down bytes of ONE expert")
    for (g, u, dn, s), n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"     {n:3d} layers  gate {g:8s} up {u:8s} down {dn:8s}  slab {s / 1e6:6.2f} MB")
    if slab:
        print(f"   mean slab {sum(slab.values()) / len(slab) / 1e6:.2f} MB, all routed experts {cls['routed experts'] / 1e9:.1f} GB")
