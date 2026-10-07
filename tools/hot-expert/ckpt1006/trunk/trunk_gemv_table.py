#!/usr/bin/env python3
"""trunk_gemv_table.py TRACE_CSV [NTOK=12] -- the GLM decode trunk GEMVs (k_gemm_batch) of the last NTOK decode tokens of a
rocprofv3 kernel trace, stdlib only.

Token segmentation as decode_trace_report.py: one k_glm5_hc_mean a token on the LAST card; token i spans
(end of head i-1, end of head i]; a kernel belongs to the token if start >= ta and end <= tb.

Prints
  A. per launch signature (template instance, grid X/Y/Z in work-items, WG size, VGPR, LDS): launches/token, mean/median/p90 us,
     ms/token, share of the k_gemm_batch total;
  B. per MATRIX (the signature + the launch's position in the layer, parsed with the decode graph's fixed launch order,
     glm5_graph.cpp layer_a/kda/dsa/moe_b): rows, K, format, bytes, nsplit, row groups, time, GB/s, % of 800 GB/s;
  C. a least-squares fit of time against bytes and nsplit over the matrix classes;
  D. the other kernels of the window (per token, summed over the cards) and the trunk-path small kernels;
  E. the idle gap in front of every k_gemm_batch on its card (start - latest end of any earlier kernel on that card),
     and how many k_gemm_batch launches overlap another kernel on the same card (a helper / staging stream).
"""
import csv, re, sys, statistics as st, collections, bisect

BW = 800.0  # GB/s, the record's achievable VRAM bandwidth a card (section M5)
Q8 = 34.0 / 32.0  # bytes per Q8_0 weight

# The GLM decode trunk GEMVs (glm5_shapes.h, the GGUF header): name -> (rows, K, format, bytes per weight)
MATS = {
    "hc_attn_fn": (24, 16384, "Q8_0", Q8), "hc_ffn_fn": (24, 16384, "Q8_0", Q8),
    "kda_q": (8192, 4096, "Q8_0", Q8), "kda_k": (8192, 4096, "Q8_0", Q8), "kda_v": (8192, 4096, "Q8_0", Q8),
    "kda_f_a": (128, 4096, "Q8_0", Q8), "kda_f_b": (8192, 128, "Q8_0", Q8), "kda_beta": (64, 4096, "Q8_0", Q8),
    "kda_g_a": (128, 4096, "Q8_0", Q8), "kda_g_b": (8192, 128, "Q8_0", Q8), "kda_o": (4096, 8192, "Q8_0", Q8),
    "dsa_q_a": (1536, 4096, "Q8_0", Q8), "idx_k": (128, 4096, "Q8_0", Q8), "idx_gate": (128, 4096, "Q8_0", Q8),
    "idx_q_b": (4096, 1536, "Q8_0", Q8), "idx_proj": (32, 4096, "F32", 4.0), "dsa_q_b": (16384, 1536, "Q8_0", Q8),
    "dsa_kv_a": (512, 4096, "Q8_0", Q8), "dsa_o": (4096, 16384, "Q8_0", Q8),
    "dense_up": (12288, 4096, "Q8_0", Q8), "dense_gate": (12288, 4096, "Q8_0", Q8), "dense_down": (4096, 12288, "Q8_0", Q8),
    "gate_inp": (288, 4096, "F32", 4.0), "sh_up": (2048, 4096, "Q8_0", Q8), "sh_gate": (2048, 4096, "Q8_0", Q8),
    "sh_down": (4096, 2048, "Q8_0", Q8), "head": (154880, 4096, "Q6_K", 210.0 / 256.0),
}
KDA_SEQ = ["kda_q", "kda_k", "kda_v", "kda_f_a", "kda_f_b", "kda_beta", "kda_g_a", "kda_g_b", "kda_o"]
DSA_SEQ = ["dsa_q_a", "idx_k", "idx_gate", "idx_q_b", "idx_proj", "dsa_q_b", "dsa_kv_a", "dsa_o"]
DSA_SEQ_NOSCORE = ["dsa_q_a", "idx_k", "idx_gate", "dsa_q_b", "dsa_kv_a", "dsa_o"]
DENSE_SEQ = ["dense_up", "dense_gate", "dense_down"]
MOE_SEQ = ["gate_inp", "sh_up", "sh_gate", "sh_down"]


def host_geometry(rows, K, fmt, min_rows=1024, target=2048, waves=8):
    """decode_gpu.hip gemv_batch_impl: row groups, nsplit (a batch of one)."""
    rg = -(-rows // waves)
    gran = 256 if fmt == "Q6_K" else 32
    units = -(-K // gran)
    ns = 1 if rows >= min_rows else -(-target // rg)
    ns = max(1, min(ns, units))
    return rg, ns


def short(n):
    m = re.search(r"(k_[A-Za-z0-9_]+(?:<[^>]*>)?)", n)
    return m.group(1).replace(" ", "") if m else n.split("(")[0][:40]


def pct(v, p):
    v = sorted(v)
    if not v: return float("nan")
    k = (len(v) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def load(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out.append(dict(s=int(r["Start_Timestamp"]), e=int(r["End_Timestamp"]), a=r["Agent_Id"],
                                q=r.get("Queue_Id", ""), n=short(r["Kernel_Name"]),
                                sig=(short(r["Kernel_Name"]), int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Grid_Size_Z"]),
                                     int(r["Workgroup_Size_X"]), int(r["VGPR_Count"]), int(r["LDS_Block_Size"]))))
            except (ValueError, KeyError):
                pass
    out.sort(key=lambda k: k["s"])
    return out


def parse_layers(gl, sig_of_name):
    """Assign a matrix name to each k_gemm_batch launch of ONE card in ONE token, by the graph's launch order."""
    names = [None] * len(gl)
    hc_sig = sig_of_name["hc"]
    i = 0; n_hc = 0
    while i < len(gl):
        g = gl[i]
        if g["sig"][1:3] == hc_sig:
            names[i] = "hc_attn_fn" if n_hc % 2 == 0 else "hc_ffn_fn"
            seq = None
            nxt = gl[i + 1]["sig"][1:3] if i + 1 < len(gl) else None
            if n_hc % 2 == 0:   # attention module follows
                if nxt == sig_of_name["kda_q"]: seq = KDA_SEQ
                elif nxt == sig_of_name["dsa_q_a"]:
                    seq = DSA_SEQ if (i + 4 < len(gl) and gl[i + 4]["sig"][1:3] == sig_of_name["idx_q_b"]
                                      and i + 5 < len(gl) and gl[i + 5]["sig"][1:3] == sig_of_name["idx_proj"]) else DSA_SEQ_NOSCORE
            else:               # FFN follows
                if nxt == sig_of_name["dense_up"]: seq = DENSE_SEQ
                elif nxt == sig_of_name["gate_inp"]: seq = MOE_SEQ
            n_hc += 1
            i += 1
            if seq:
                for nm in seq:
                    if i < len(gl) and gl[i]["sig"][1:3] == sig_of_name[nm]:
                        names[i] = nm; i += 1
                    else:
                        break
            continue
        if g["sig"][1:3] == sig_of_name["head"]:
            names[i] = "head"
        i += 1
    return names


def lsq(X, y):
    """least squares via normal equations (small, stdlib)."""
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(n)]
    for c in range(n):  # Gauss-Jordan
        p = max(range(c, n), key=lambda r: abs(A[r][c])); A[c], A[p] = A[p], A[c]; b[c], b[p] = b[p], b[c]
        for r in range(n):
            if r != c and A[c][c]:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * z for x, z in zip(A[r], A[c])]; b[r] -= f * b[c]
    return [b[i] / A[i][i] for i in range(n)]


def main():
    path = sys.argv[1]
    ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ks = load(path)
    agents = sorted(set(k["a"] for k in ks), key=lambda a: int(a.split()[-1]))
    by = {a: [k for k in ks if k["a"] == a] for a in agents}
    last = agents[-1]
    heads = [k for k in by[last] if k["n"] == "k_glm5_hc_mean"]
    ends = [h["e"] for h in heads]
    toks = list(range(max(1, len(ends) - ntok), len(ends)))
    T = len(toks)
    print("trace: %s\nagents %s; head kernels %d; tokens used %d (the last %d)" % (path, agents, len(heads), T, ntok))
    walls = [(ends[i] - ends[i - 1]) / 1e3 for i in toks]
    print("traced wall per token (us): median %.0f min %.0f max %.0f" % (st.median(walls), min(walls), max(walls)))

    # expected grids (work-items) per matrix from the host code
    sig_of_name = {}
    for nm, (rows, K, fmt, bpw) in MATS.items():
        rg, ns = host_geometry(rows, K, fmt)
        sig_of_name[nm] = (ns * 256, rg)
    sig_of_name["hc"] = sig_of_name["hc_attn_fn"]

    gemm = []          # (token, agent, launch dict, name)
    other = collections.defaultdict(lambda: [0, 0])
    gaps = []          # (name, gap_ns, prev kernel name)
    overl = collections.defaultdict(lambda: [0, 0, [], []])   # name -> [n, n_overlapped, dur_overlapped, dur_clean]
    for ti in toks:
        ta, tb = ends[ti - 1], ends[ti]
        for a in agents:
            win = [k for k in by[a] if k["s"] >= ta and k["e"] <= tb]
            gl = [k for k in win if k["n"].startswith("k_gemm_batch")]
            names = parse_layers(gl, sig_of_name)
            nm_of = {id(g): (nm or "UNMAPPED") for g, nm in zip(gl, names)}
            for g, nm in zip(gl, names):
                gemm.append((ti, a, g, nm or "UNMAPPED"))
            for k in win:
                if not k["n"].startswith("k_gemm_batch"):
                    o = other[k["n"]]; o[0] += 1; o[1] += k["e"] - k["s"]
            # gaps and overlap: all kernels of the card (incl. ones just before the window)
            allk = by[a]
            starts = [k["s"] for k in allk]
            for g in gl:
                j = bisect.bisect_left(starts, g["s"])
                prev_end = 0; prev_n = ""
                for k in allk[max(0, j - 40):j]:
                    if k["e"] > prev_end: prev_end, prev_n = k["e"], k["n"]
                nm = nm_of[id(g)]
                gaps.append((nm, g["s"] - prev_end, prev_n))
                # overlap with any other kernel on this card
                ov = any((k is not g) and k["s"] < g["e"] and k["e"] > g["s"] for k in allk[max(0, j - 40):j + 40])
                rec = overl[nm]; rec[0] += 1
                if ov: rec[1] += 1; rec[2].append(g["e"] - g["s"])
                else: rec[3].append(g["e"] - g["s"])

    tot_ns = sum(g["e"] - g["s"] for _, _, g, _ in gemm)
    tot_ms = tot_ns / 1e6 / T
    print("\nk_gemm_batch: %.1f launches/token, %.2f ms/token summed over the cards, %.1f us each" %
          (len(gemm) / T, tot_ms, tot_ns / 1e3 / max(1, len(gemm))))

    # ---- A. per signature
    sig = collections.defaultdict(list)
    sig_names = collections.defaultdict(collections.Counter)
    for _, _, g, nm in gemm:
        sig[g["sig"]].append((g["e"] - g["s"]) / 1e3); sig_names[g["sig"]][nm] += 1
    print("\nA. per launch signature (grid in work-items; split rides grid.x at TILE 1: nsplit = gridX/256, row groups = gridY)")
    print("%-20s %8s %6s %3s %4s %5s | %7s %7s %7s %7s | %7s %6s | matrices" %
          ("kernel", "gridX", "gridY", "WG", "VGPR", "LDS", "n/tok", "mean", "med", "p90", "ms/tok", "share"))
    for s, v in sorted(sig.items(), key=lambda x: -sum(x[1])):
        ms = sum(v) / 1e3 / T
        print("%-20s %8d %6d %3d %4d %5d | %7.1f %7.1f %7.1f %7.1f | %7.2f %5.1f%% | %s" %
              (s[0], s[1], s[2], s[4], s[5], s[6], len(v) / T, st.mean(v), st.median(v), pct(v, 90), ms, 100 * ms / tot_ms,
               ", ".join("%s:%d" % (n, c // T if c % T == 0 else c / T) for n, c in sig_names[s].most_common())))

    # ---- B. per matrix
    per = collections.defaultdict(list)
    for _, _, g, nm in gemm:
        per[nm].append((g["e"] - g["s"]) / 1e3)
    print("\nB. per matrix (bytes = weight bytes; GB/s = bytes / mean time)")
    print("%-11s %6s %6s %5s %9s %6s %5s | %6s %7s %7s %7s %7s %6s | %6s %5s" %
          ("matrix", "rows", "K", "fmt", "bytes", "nsplit", "rg", "n/tok", "mean", "med", "p90", "ms/tok", "share", "GB/s", "%bw"))
    rows_fit = []
    tot_bytes = 0.0
    for nm, v in sorted(per.items(), key=lambda x: -sum(x[1])):
        ms = sum(v) / 1e3 / T
        if nm in MATS:
            rows, K, fmt, bpw = MATS[nm]
            byt = rows * K * bpw
            rg, ns = host_geometry(rows, K, fmt)
            gbs = byt / (st.mean(v) * 1e3)
            tot_bytes += byt * len(v) / T
            rows_fit.append((nm, byt, ns, st.mean(v), st.median(v), len(v) / T))
            print("%-11s %6d %6d %5s %9.0f %6d %5d | %6.1f %7.1f %7.1f %7.1f %7.2f %5.1f%% | %6.0f %4.0f%%" %
                  (nm, rows, K, fmt, byt, ns, rg, len(v) / T, st.mean(v), st.median(v), pct(v, 90), ms, 100 * ms / tot_ms,
                   gbs, 100 * gbs / BW))
        else:
            print("%-11s %s launches/token, %.2f ms/token" % (nm, len(v) / T, ms))
    print("weight bytes a token through k_gemm_batch: %.3f GB (head %.3f GB); aggregate %.0f GB/s over %.2f ms" %
          (tot_bytes / 1e9, MATS["head"][0] * MATS["head"][1] * MATS["head"][3] / 1e9, tot_bytes / (tot_ms * 1e6), tot_ms))
    floor = tot_bytes / (BW * 1e9) * 1e3
    print("bandwidth floor at %.0f GB/s: %.2f ms; at 650 GB/s: %.2f ms" % (BW, floor, tot_bytes / 650e9 * 1e3))

    # ---- C. fits
    print("\nC. fits over the matrix classes, per-launch MEDIAN us (weighted by launches/token)")
    def fit(sel, cols, label):
        X, y = [], []
        for nm, byt, ns, mean, med, cnt in rows_fit:
            if not sel(nm, byt, ns): continue
            w = max(1, int(round(cnt)))
            for _ in range(w):
                X.append([1.0] + [c(byt, ns) for c in cols]); y.append(med)
        if len(set(tuple(r) for r in X)) < len(cols) + 1:
            print("  %s: too few distinct points" % label); return None
        co = lsq(X, y)
        res = [yy - sum(c * x for c, x in zip(co, r)) for r, yy in zip(X, y)]
        print("  %s: %s  (rms residual %.1f us, %d launches)" % (label, "  ".join("%.4g" % c for c in co),
                                                               (sum(r * r for r in res) / len(res)) ** 0.5, len(y)))
        return co
    c1 = fit(lambda n, b, s: s == 1 and n != "head", [lambda b, s: b / 1e6], "nsplit=1, no head: t = a + b*MB  [a us, b us/MB]")
    if c1: print("     -> per-byte part = %.0f GB/s effective" % (1e6 / c1[1] / 1e3))
    c2 = fit(lambda n, b, s: s > 1, [lambda b, s: s], "nsplit>1: t = a + c*nsplit  [a us, c us/split]")
    c3 = fit(lambda n, b, s: n != "head", [lambda b, s: b / 1e6, lambda b, s: (s if s > 1 else 0)],
             "all but head: t = a + b*MB + c*nsplit(if>1)")
    wgs = {}
    for nm, (rows, K, fmt, bpw) in MATS.items():
        rg, ns = host_geometry(rows, K, fmt); wgs[nm] = rg * ns
    X, y = [], []
    for nm, byt, ns, mean, med, cnt in rows_fit:
        if byt < 3e6:
            for _ in range(max(1, int(round(cnt)))): X.append([1.0, wgs[nm] / 1000.0]); y.append(med)
    co = lsq(X, y)
    res = [yy - co[0] - co[1] * r[1] for r, yy in zip(X, y)]
    print("  matrices < 3 MB (split ones + f_b/g_b): t = a + c*workgroups: a %.2f us, c %.2f us per 1000 WGs (rms residual %.1f us, %d launches)"
          % (co[0], co[1], (sum(r * r for r in res) / len(res)) ** 0.5, len(y)))
    print("   " + "  ".join("%s %dWG %.1fus" % (nm, wgs[nm], med) for nm, byt, ns, mean, med, cnt in sorted(rows_fit, key=lambda r: wgs[r[0]]) if byt < 3e6))
    print("  per-matrix: the fixed part implied if the bytes ran at 800 GB/s (median - bytes/800GB/s):")
    print("   " + "  ".join("%s %.1f" % (nm, med - byt / 800e3) for nm, byt, ns, mean, med, cnt in sorted(rows_fit, key=lambda r: r[1])))
    n_launch = len(gemm) / T
    for a0 in (5.0, 8.0, 10.0):
        print("  launch floor at %.0f us each: %.0f launches x %.0f = %.2f ms/token; bandwidth part at 800 GB/s %.2f ms" %
              (a0, n_launch, a0, n_launch * a0 / 1e3, floor))

    # ---- D. other kernels
    print("\nD. other kernels in the window (all cards, per token)")
    ot = sorted(other.items(), key=lambda x: -x[1][1])
    for n, (c, t) in ot[:30]:
        print("  %-34s %7.1f /tok %8.0f us/tok %7.1f us each" % (n, c / T, t / 1e3 / T, t / 1e3 / max(1, c)))
    for n, (c, t) in ot:
        if "reduce" in n:
            print("  [trunk path] %s: %.1f/token, %.2f ms/token = %.1f%% of the k_gemm_batch time" %
                  (n, c / T, t / 1e6 / T, 100 * t / 1e6 / T / tot_ms))

    # ---- E. gaps and overlap
    print("\nE. idle gap before each k_gemm_batch on its card (start - latest end of any earlier kernel on that card), us")
    gp = collections.defaultdict(list)
    for nm, gns, pn in gaps: gp[nm].append(gns / 1e3)
    allg = [g / 1e3 for _, g, _ in gaps]
    pos = [g for g in allg if g > 0]
    print("  all: n %d/token; overlapping (gap<0) %.0f%%; gap>0: median %.1f p90 %.1f p99 %.1f; sum of positive gaps %.2f ms/token" %
          (len(allg) / T, 100 * (len(allg) - len(pos)) / len(allg), st.median(pos), pct(pos, 90), pct(pos, 99), sum(pos) / 1e3 / T))
    for cap in (2.0, 5.0):
        sav = sum(max(0.0, g - cap) for g in pos) / 1e3 / T
        sav50 = sum(max(0.0, min(g, 50.0) - cap) for g in pos) / 1e3 / T
        print("  if every positive gap before a k_gemm_batch went to %.0f us: -%.2f ms/token (gaps capped at 50 us first: -%.2f)" % (cap, sav, sav50))
    print("  %-11s %6s %7s %7s %7s  prev kernel (most common)" % ("matrix", "n/tok", "med", "p90", "ovl%"))
    prevc = collections.defaultdict(collections.Counter)
    for nm, gns, pn in gaps: prevc[nm][pn] += 1
    for nm in sorted(gp, key=lambda n: -len(gp[n])):
        v = gp[nm]
        print("  %-11s %6.1f %7.1f %7.1f %6.0f%%  %s" % (nm, len(v) / T, st.median(v), pct(v, 90),
              100 * sum(1 for x in v if x < 0) / len(v), ", ".join("%s:%d" % kv for kv in prevc[nm].most_common(2))))
    print("\n  k_gemm_batch overlapping another kernel on the same card (helper/stage stream), duration us:")
    for nm, (n, no, d_o, d_c) in sorted(overl.items(), key=lambda x: -x[1][0]):
        print("  %-11s overlapped %5.1f%%  med overlapped %7.1f  med clean %7.1f" %
              (nm, 100 * no / n, st.median(d_o) / 1e3 if d_o else float("nan"), st.median(d_c) / 1e3 if d_c else float("nan")))

    # ---- per card gap stats over ALL kernels (the graph-capture question)
    print("\n  all kernels, per card: gap between consecutive kernels (start - latest earlier end), us, last %d tokens" % T)
    t0, t1 = ends[toks[0] - 1], ends[toks[-1]]
    for a in agents:
        ev = [k for k in by[a] if k["s"] >= t0 and k["e"] <= t1]
        le = 0; g = []
        for k in ev:
            if le: g.append((k["s"] - le) / 1e3)
            le = max(le, k["e"])
        p = [x for x in g if x > 0]
        small = [x for x in p if x < 50]
        print("  %s: %d kernels/token; positive gaps median %.1f p90 %.1f; gaps < 50 us: n %d/token, sum %.2f ms/token, "
              "saving if each went to 2 us %.2f ms/token" % (a, len(ev) / T, st.median(p), pct(p, 90), len(small) / T,
                                                            sum(small) / 1e3 / T, sum(max(0, x - 2) for x in small) / 1e3 / T))


if __name__ == "__main__":
    main()
