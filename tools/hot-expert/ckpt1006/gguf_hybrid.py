#!/usr/bin/env python3
"""gguf_hybrid.py -- one GGUF made of two: the TRUNK (everything that is not a routed expert) from one split GGUF set, the routed experts from another (stdlib only).

    python3 -I gguf_hybrid.py --trunk TRUNK_SHARD1.gguf --experts EXPERT_SHARD1.gguf --out OUT.gguf [--dry-run] [--keep-last-from-trunk]

PF14 (2026-10-09): our served UD-IQ4_XS has a Q8_0 trunk that the tuned decode kernels (q8fast, rowsplit, grouped GEMV) were built for; Unsloth's UD-IQ3_XXS has smaller experts
(IQ2_S gate/up, IQ3_S down: 8.98 MB against 11.67) but a Q6_K trunk. The hybrid keeps the first and takes the second's experts. Every `blk.N.ffn_{gate,up,down}_exps.weight` comes from
the expert set, except layer 45 (the NextN block, unused by the text tower, Q2_K/Q3_K in the candidate) when --keep-last-from-trunk is given; every other tensor and the whole
key-value section (tokenizer, template, hyper-parameters) from the trunk set, minus the split.* keys (the output is ONE file). NAME THE OUTPUT ...-00001-of-00001.gguf: the engine's loader (llama.cpp's split-naming check) refuses any other name. Tensor data is copied in 64 MB pieces, 32-byte aligned
as the GGUF spec asks (general.alignment is honoured). Shards are found from SHARD1's name (...-00001-of-0000N.gguf). --dry-run prints the plan and the output size only.
"""
import argparse, glob, os, re, struct, sys

TYPES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
         15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

class Shard:
    def __init__(self, path):
        self.path = path
        b = open(path, "rb").read(32 * 1024 * 1024)       # the header of a shard >= 2 is far smaller; shard 1 holds the KV (tokenizer) in ~10 MB
        self.p = 0
        def need(n):
            nonlocal b
            if self.p + n > len(b):
                b += open(path, "rb").read(len(b) * 2)[len(b):]
        def u32():
            need(4); v = struct.unpack_from("<I", b, self.p)[0]; self.p += 4; return v
        def u64():
            need(8); v = struct.unpack_from("<Q", b, self.p)[0]; self.p += 8; return v
        def string():
            n = u64(); need(n); s = b[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return s
        assert b[:4] == b"GGUF"
        self.p = 4; self.version = u32(); nt = u64(); nkv = u64()
        def skip(t):
            if t in KV_FIXED: self.p += KV_FIXED[t]
            elif t == 8: string()
            elif t == 9:
                et = u32(); n = u64()
                for _ in range(n): skip(et)
            else: raise ValueError("kv type %d" % t)
        self.kv = []                                       # (key, raw bytes of the whole entry)
        self.align = 32
        for _ in range(nkv):
            s0 = self.p; key = string(); t = u32(); v0 = self.p; skip(t); need(0)
            self.kv.append((key, b[s0:self.p]))
            if key == "general.alignment": self.align = struct.unpack_from("<I", b, v0)[0]
        self.tensors = []                                  # (name, dims, type, offset, nbytes)
        for _ in range(nt):
            name = string(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
            n = 1
            for d in dims: n *= d
            blk, bpb = TYPES[ty]
            self.tensors.append((name, dims, ty, off, n // blk * bpb))
        self.data_start = (self.p + self.align - 1) // self.align * self.align

def shards_of(first):
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first)
    if not m: return [first]
    return [f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]

def pack_string(s): e = s.encode(); return struct.pack("<Q", len(e)) + e

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", required=True); ap.add_argument("--experts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--keep-last-from-trunk", action="store_true")
    a = ap.parse_args()
    T = [Shard(f) for f in shards_of(a.trunk)]; E = [Shard(f) for f in shards_of(a.experts)]
    is_exp = lambda n: re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight$", n) is not None
    last = max(int(re.match(r"blk\.(\d+)\.", n).group(1)) for s in T for (n, *_r) in s.tensors if is_exp(n))
    plan = []                                               # (name, dims, type, nbytes, source shard, source offset)
    for s in T:
        for (n, d, ty, off, nb) in s.tensors:
            if is_exp(n) and not (a.keep_last_from_trunk and n.startswith(f"blk.{last}.")): continue
            plan.append((n, d, ty, nb, s, off))
    got = set(p[0] for p in plan)
    for s in E:
        for (n, d, ty, off, nb) in s.tensors:
            if is_exp(n) and n not in got: plan.append((n, d, ty, nb, s, off))
    names = [p[0] for p in plan]
    assert len(names) == len(set(names)), "duplicate tensor names"
    n_exp = sum(1 for p in plan if is_exp(p[0]))
    kv = [(k, raw) for (k, raw) in T[0].kv if not k.startswith("split.")]
    kv_bytes = b"".join(raw for _, raw in kv)
    align = T[0].align
    infos = b""; off = 0; offs = []
    for (n, d, ty, nb, s, so) in plan:
        offs.append(off)
        infos += pack_string(n) + struct.pack("<I", len(d)) + b"".join(struct.pack("<Q", x) for x in d) + struct.pack("<I", ty) + struct.pack("<Q", off)
        off = (off + nb + align - 1) // align * align
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(kv)) + kv_bytes + infos
    data_start = (len(header) + align - 1) // align * align
    total = data_start + off
    t_exp = sum(p[3] for p in plan if is_exp(p[0])); t_all = sum(p[3] for p in plan)
    print(f"{len(plan)} tensors ({n_exp} routed-expert tensors from {os.path.basename(a.experts)}), {len(kv)} kv entries, header {len(header)} B, data {t_all / 1e9:.2f} GB "
          f"(experts {t_exp / 1e9:.2f}), file {total / 1e9:.2f} GB, alignment {align}")
    if a.dry_run: sys.exit(0)
    with open(a.out, "wb") as out:
        out.write(header); out.write(b"\0" * (data_start - len(header)))
        handles = {}
        for (n, d, ty, nb, s, so), o in zip(plan, offs):
            fh = handles.get(s.path) or handles.setdefault(s.path, open(s.path, "rb"))
            fh.seek(s.data_start + so); out.seek(data_start + o)
            left = nb
            while left:
                chunk = fh.read(min(left, 64 << 20))
                assert chunk, "short read " + n
                out.write(chunk); left -= len(chunk)
        out.truncate(total)
    print("wrote", a.out)
