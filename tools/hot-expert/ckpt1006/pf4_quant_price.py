#!/usr/bin/env python3
"""pf4_quant_price.py -- what a smaller expert quant of GLM-5.3-Flash buys: VRAM, decode, prefill (offline model, stdlib, no GPU).

    python3 -I pf4_quant_price.py HDR_DIR_ROOT lookahead_log [lookahead_log ...]

HDR_DIR_ROOT holds one directory per quant (UD-IQ4_XS, UD-Q3_K_XL, UD-IQ3_XXS, UD-Q2_K_XL, UD-IQ2_XXS) with the first 6 MB of each shard >= 2 as sN.bin (curl -r 0-6291455 on
huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/<quant>/<shard>); gguf_tensor_types.py (same directory) reads the tensor types and sizes from them.
Per layer (3..44) the slab = bytes of one expert (gate+up+down). Decode: the lookahead log (256 tokens x 42 layers; each use is a hit or a miss by the residency mask at routing time)
is split in halves and scored on the other half, both directions averaged, ANCHORED to the placement that was really there: with k more resident experts a layer recovers the misses of
the k hottest non-resident experts of the ranking half (k > 0); with k fewer it loses the hits of the k coldest residents of the ranking half (residents never hit are an unknown pool
evicted first, as in pf4_evict_model.py). k = 0 reproduces the logged miss rate (0.362 prose, 0.244 chat) by construction. The capacity of a layer is the same BYTES as today's resident
set (`resident_at_start`) divided by the new slab. ms per missed byte: 0.26 ms a 11.67 MB slab (decode trace). Prefill: the link floor = non-resident bytes a token / 62 GB/s (three links
balanced), scaled to PF9's 1.4 ms/token for today; the prefill gain is 0.77 x the change of the floor (PF3 probe: about half of the expert time is flat in the link bytes).
"""
import collections, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_tensor_types as G

LAYERS = range(3, 45)
BASE = "UD-IQ4_XS"

def slabs(root, quant):
    tens = [t for f in sorted(glob.glob(os.path.join(root, quant, "*.bin"))) for t in G.parse(f)]
    part = collections.defaultdict(dict)
    other = 0
    for name, dims, tn, nb in tens:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if m: part[int(m.group(1))][m.group(2)] = (nb, dims[-1])
        elif not name.startswith("blk.") or "_exps" not in name: other += nb
    return {il: sum(part[il][k][0] for k in part[il]) / part[il]["gate"][1] for il in part}, other

def load_log(path):
    res, recs = {}, collections.defaultdict(dict)
    for line in open(path):
        if line.startswith("H layer"):
            m = re.search(r"il=(\d+).*resident_at_start=(\d+)", line); res[int(m.group(1))] = int(m.group(2))
        elif line.startswith("R ") and "lag=1 " in line and "ph=D" in line:
            pos = int(re.search(r" pos=(\d+)", line).group(1)); il = int(re.search(r" il=(\d+)", line).group(1))
            act = [int(x) for x in re.search(r" act=([0-9,]+)", line).group(1).split(",")]
            recs[il][pos] = act
    return res, recs

class Layer:
    """per layer: second-half miss counts after ranking on the other half; f(k) = expected missed experts a token for k more (k < 0: fewer) resident experts"""
    def __init__(self, recs_il, R, mask_recs):
        self.R = R; self.dirs = []
        order = sorted(recs_il); n = len(order)
        for a, b in ((order[:n // 2], order[n // 2:]), (order[n // 2:], order[:n // 2])):
            hit1, mis1 = collections.Counter(), collections.Counter()
            for p in a:
                for e, miss in recs_il[p]: (mis1 if miss else hit1)[e] += 1
            mis2, hit2 = collections.Counter(), collections.Counter()
            for p in b:
                for e, miss in recs_il[p]: (mis2 if miss else hit2)[e] += 1
            cand = [e for e, _ in mis1.most_common() if hit1[e] == 0]          # non-resident in the ranking half, hottest first
            res = sorted((e for e in hit1), key=lambda e: hit1[e])             # residents seen, coldest first
            self.dirs.append((len(b), sum(mis2.values()), cand, mis2, res, hit1, hit2))
    def f(self, k):
        tot = 0.0
        for nb, m_act, cand, mis2, res, hit1, hit2 in self.dirs:
            if k >= 0:
                rec = sum(mis2[e] for e in cand[:k])
                tot += (m_act - rec) / nb
            else:
                kk = -k; unseen = max(self.R - len(res), 0); k_un = min(kk, unseen)
                evicted = set(res[:max(kk - k_un, 0)])
                lost = sum(c for e, c in hit2.items() if e in evicted)
                lost += sum(c for e, c in hit2.items() if e not in hit1) * (k_un / unseen if unseen else 0)
                tot += (m_act + lost) / nb
        return tot / len(self.dirs)

def load_log(path):
    res, recs = {}, collections.defaultdict(dict)
    for line in open(path):
        if line.startswith("H layer"):
            m = re.search(r"il=(\d+).*resident_at_start=(\d+)", line); res[int(m.group(1))] = int(m.group(2))
        elif line.startswith("R ") and "lag=1 " in line and "ph=D" in line:
            pos = int(re.search(r" pos=(\d+)", line).group(1)); il = int(re.search(r" il=(\d+)", line).group(1))
            act = [int(x) for x in re.search(r" act=([0-9,]+)", line).group(1).split(",")]
            hx = re.search(r" miss=([0-9a-f]+)", line).group(1); mask = 0
            for j, c in enumerate(hx): mask |= int(c, 16) << (4 * j)
            recs[il][pos] = [(e, (mask >> e) & 1) for e in act]
    return res, recs

if __name__ == "__main__":
    root, logs = sys.argv[1], sys.argv[2:]
    S = {q: slabs(root, q) for q in sorted(os.listdir(root))}
    base_slab, base_other = S[BASE]
    MS_PER_BYTE = 0.26 / 11.665408e6
    for lp in logs:
        R, recs = load_log(lp)
        il_list = [il for il in LAYERS if il in recs and il in R]
        L = {il: Layer(recs[il], R[il], None) for il in il_list}
        m0 = sum(L[il].f(0) for il in il_list)
        resident_gb = sum(R[il] * base_slab[il] for il in il_list) / 1e9
        floor0 = sum((288 - R[il]) * base_slab[il] for il in il_list) / 1024 / 62e9 * 1e3
        base_missbytes = sum(L[il].f(0) * base_slab[il] for il in il_list)
        print(f"\n{os.path.basename(lp)}: miss rate {m0 / (8 * len(il_list)):.3f} (logged value reproduced); {sum(R[il] for il in il_list)} resident experts = {resident_gb:.1f} GB; "
              f"link floor {floor0:.2f} ms/token (scaled to 1.40); missed {base_missbytes / 1e9:.2f} GB/token = {base_missbytes * MS_PER_BYTE:.1f} ms of critical fetch")
        print(f"{'quant':11s} {'slab MB':>7s} {'file GB':>7s} | SAME expert-gb: {'+experts':>8s} {'decode ms/tok':>13s} {'dMiss GB':>8s} {'link floor':>10s} {'prefill est':>11s} | SAME decode as today: {'GB freed/card':>13s} {'(+trunk)':>8s}")
        for q in ("UD-IQ4_XS", "UD-Q3_K_XL", "UD-IQ3_XXS", "UD-Q2_K_XL", "UD-IQ2_XXS"):
            if q not in S: continue
            sl, other = S[q]
            k = {il: int(round(R[il] * base_slab[il] / sl[il])) - R[il] for il in il_list}
            mb = sum(L[il].f(k[il]) * sl[il] for il in il_list)
            nres = sum(R[il] + k[il] for il in il_list)
            floor = sum((288 - R[il] - k[il]) * sl[il] for il in il_list) / 1024 / 62e9 * 1e3 * (1.40 / floor0)
            # same decode: take capacity away (in experts, from the enlarged set) until the missed bytes are back to today's
            lo, hi = 0.0, 1.0       # fraction of the added experts to give up, then beyond them
            def mbytes(t):          # t >= 0 experts removed per layer relative to the enlarged set, as a fraction of the layer's capacity
                return sum(L[il].f(k[il] - int(round(t * (R[il] + k[il])))) * sl[il] for il in il_list)
            lo, hi = 0.0, 0.6
            for _ in range(16):
                mid = (lo + hi) / 2
                if mbytes(mid) > base_missbytes: hi = mid
                else: lo = mid
            same_gb = sum((R[il] + k[il] - int(round(lo * (R[il] + k[il])))) * sl[il] for il in il_list) / 1e9
            freed = (resident_gb - same_gb) / 3
            print(f"{q:11s} {sum(sl[il] for il in il_list) / len(il_list) / 1e6:7.2f} {sum(sl.values()) * 288 / 1e9 + other / 1e9:7.0f} |                 {nres - sum(R[il] for il in il_list):+8d} "
                  f"{mb * MS_PER_BYTE - base_missbytes * MS_PER_BYTE:+13.1f} {(mb - base_missbytes) / 1e9:+8.2f} {floor:10.2f} {0.77 * (1.40 - floor):+11.2f} | {'':21s}{freed:13.2f} {(base_other - other) / 1e9 / 3:+8.2f}")
