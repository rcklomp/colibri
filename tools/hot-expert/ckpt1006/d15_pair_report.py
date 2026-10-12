#!/usr/bin/env python3
"""d15_pair_report.py DIR_A1 DIR_B2 DIR_B3 DIR_A4 [--tokens N] [--lone 'Agent 3'] -- D15 (record §L5-D15): the critical expert fetch of a decode
token per arm of a paired kernel trace (order A B B A, one process each; glm_d15_chain.sh writes dtrace_v1d15_1 / newd15_2 / newd15_3 / v1d15_4),
and the link model checked layer by layer. Also takes two dirs (A B) for a single pair (the 2026-10-07 pattern / optimal traces).

Per decode token (boundaries: the once-a-token k_glm5_hc_mean on the last card) and per MoE layer (one k_glm5_plan each): the three cards'
k_glm5_stage kernels; the layer's critical fetch = the slowest stage kernel. The slab count of each card is read from its duration:
lone card d / AL; the shared pair, if both fetch, the smaller m = d_min / SH and the larger m + (d_max - m SH) / AL (the REFINED pair model:
the card with more slabs runs alone once its partner is done); one alone: d / AL. AL and SH are measured from this trace (single-slab
clusters), so a model with another slab size (the hybrid) needs no constants. The miss count n = the sum must agree between the arms layer
by layer (decode is deterministic below the swap-timing noise): that agreement is the non-circular check of the count reading.
Priced per layer with (a) the 2026-10-07 model (a pair card at SH a slab whenever its partner has any) and (b) the refined model.
"""
import csv, glob, sys, statistics as st, collections

KEYS = ("k_glm5_stage", "k_glm5_hc_mean", "k_glm5_plan")


def load(d, ntok):
    fs = glob.glob(d + "/**/*kernel_trace.csv", recursive=True) or [d]
    ks = []
    for f in fs:
        for r in csv.DictReader(open(f)):
            n = r["Kernel_Name"]
            for k in KEYS:
                if k in n and "stage_tab" not in n:
                    ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], k)); break
    ks.sort()
    agents = sorted({k[2] for k in ks if k[3] == "k_glm5_stage"}, key=lambda a: int(a.split()[-1]))
    last = max(agents, key=lambda a: sum(1 for k in ks if k[3] == "k_glm5_hc_mean" and k[2] == a))
    ends = sorted(k[1] for k in ks if k[3] == "k_glm5_hc_mean" and k[2] == last)[-ntok - 1:]
    toks = []
    for t in range(len(ends) - 1):
        lo, hi = ends[t], ends[t + 1]
        tk = [k for k in ks if lo < k[0] <= hi]
        plans = [k for k in tk if k[3] == "k_glm5_plan"]
        layers = []
        for i, p in enumerate(plans):
            nxt = plans[i + 1][0] if i + 1 < len(plans) else hi
            layers.append({k[2]: (k[1] - k[0]) / 1e3 for k in tk if k[3] == "k_glm5_stage" and p[0] <= k[0] < nxt})
        toks.append({"wall": (hi - lo) / 1e6, "layers": layers})
    return agents, toks


def units(runs, lone, pair):
    lone_d = sorted(l[lone] for r in runs for t in r for l in t["layers"] if l.get(lone, 0) > 50)
    p05 = lone_d[len(lone_d) // 20]
    al = st.median([x for x in lone_d if x < 1.4 * p05])
    sh_d = [min(l[pair[0]], l[pair[1]]) for r in runs for t in r for l in t["layers"]
            if l.get(pair[0], 0) > 50 and l.get(pair[1], 0) > 50 and min(l[pair[0]], l[pair[1]]) < 1.9 * al]
    return al, (st.median(sh_d) if sh_d else 1.626 * al)


def counts(l, lone, pair, al, sh):
    w = {a: (l.get(a, 0) if l.get(a, 0) > 50 else 0.0) for a in [lone] + pair}
    cl = round(w[lone] / al); a, b = w[pair[0]], w[pair[1]]
    if a and b:
        lo, hi = min(a, b), max(a, b)
        m = max(1, round(lo / sh)); M = m + max(0, round((hi - m * sh) / al))
        ca, cb = (m, M) if a <= b else (M, m)
    else:
        ca, cb = round(a / al), round(b / al)
    return (cl, ca, cb), max(w.values())


def model_a(c, al, sh):
    cl, ca, cb = c
    return max(cl * al, ca * (sh if cb else al), cb * (sh if ca else al))


def model_b(c, al, sh):
    cl, ca, cb = c
    if ca and cb:
        m, M = min(ca, cb), max(ca, cb); tp = m * sh + (M - m) * al
    else:
        tp = max(ca, cb) * al
    return max(cl * al, tp)


def main():
    args = [a for a in sys.argv[1:]]
    ntok = 24; lone = None
    if "--tokens" in args: i = args.index("--tokens"); ntok = int(args[i + 1]); del args[i:i + 2]
    if "--lone" in args: i = args.index("--lone"); lone = args[i + 1]; del args[i:i + 2]
    dirs = args
    loaded = [load(d, ntok) for d in dirs]
    agents = loaded[0][0]
    runs = [x[1] for x in loaded]
    nt = min(len(r) for r in runs)
    runs = [r[-nt:] for r in runs]
    if lone is None:   # the lone card: the one whose stage durations sit nearest whole multiples of its own smallest cluster
        lone = "Agent 3" if "Agent 3" in agents else agents[-1]
    pair = [a for a in agents if a != lone][:2]
    al, sh = units(runs, lone, pair)
    print("arms: %s | tokens read %d | lone %s, pair %s | per-slab: alone %.0f us, shared %.0f us (ratio %.3f; IQ4_XS 2026-10-07: 417 / 678)" % (
        " ".join(d.rstrip("/").split("/")[-1] for d in dirs), nt, lone, pair, al, sh, sh / al))
    per = []
    for r in runs:
        rows = []
        for t in r:
            for l in t["layers"]:
                c, meas = counts(l, lone, pair, al, sh); rows.append((c, meas))
        per.append(rows)
    nl = min(len(p) for p in per)
    nmis = sum(1 for i in range(nl) if len({sum(p[i][0]) for p in per}) > 1)
    print("MoE layers a run %d; the miss count n disagrees between the arms in %d (%.1f %%)" % (nl, nmis, 100.0 * nmis / nl))
    print("\narm         crit fetch ms/token   model(a) 10-07   model(b) refined   (b)/meas   token wall traced ms (median)")
    res = []
    for d, p, r in zip(dirs, per, runs):
        meas = sum(x[1] for x in p) / 1e3 / nt; ma = sum(model_a(x[0], al, sh) for x in p) / 1e3 / nt; mb = sum(model_b(x[0], al, sh) for x in p) / 1e3 / nt
        res.append((meas, ma, mb))
        print("%-12s %10.2f          %10.2f        %10.2f        %6.3f     %8.2f" % (d.rstrip("/").split("/")[-1], meas, ma, mb, mb / meas,
                                                                                  st.median(t["wall"] for t in r)))
    if len(dirs) == 4:
        a = (res[0][0] + res[3][0]) / 2; b = (res[1][0] + res[2][0]) / 2
        print("\nA (arms 1, 4) %.2f ms, B (arms 2, 3) %.2f ms: B - A = %+.2f ms a token of critical fetch; identical-arm spread A %.2f, B %.2f" % (
            a, b, b - a, abs(res[0][0] - res[3][0]), abs(res[1][0] - res[2][0])))
        pa = (res[0][2] + res[3][2]) / 2; pb = (res[1][2] + res[2][2]) / 2
        print("refined model predicts B - A = %+.2f ms; old model %+.2f" % (pb - pa, (res[1][1] + res[2][1]) / 2 - (res[0][1] + res[3][1]) / 2))
        wa = st.mean([st.median(t["wall"] for t in runs[0]), st.median(t["wall"] for t in runs[3])])
        wb = st.mean([st.median(t["wall"] for t in runs[1]), st.median(t["wall"] for t in runs[2])])
        print("token wall (traced, median): A %.2f, B %.2f ms (%+.2f)" % (wa, wb, wb - wa))
    elif len(dirs) == 2:
        print("\nB - A: measured %+.2f ms, model (a) %+.2f, model (b) %+.2f" % (res[1][0] - res[0][0], res[1][1] - res[0][1], res[1][2] - res[0][2]))
    # by n, the arms grouped A / B
    grp = [0, 1, 1, 0] if len(dirs) == 4 else list(range(len(dirs)))
    byn = collections.defaultdict(lambda: collections.defaultdict(list))
    for gi, p in zip(grp, per):
        for c, meas in p[:nl]:
            byn[sum(c)][gi].append(meas)
    tot = sum(len(v[0]) for v in byn.values())
    print("\nby miss count n: share of layers, mean critical fetch us per group (A, B), B - A in ms a token")
    for n in sorted(byn):
        g = byn[n]
        if len(g) >= 2 and g[0] and g[1]:
            share = len(g[0]) / tot
            print("  n=%d %5.1f %%   A %6.0f   B %6.0f   %+.2f" % (n, 100 * share, st.mean(g[0]), st.mean(g[1]),
                                                             (st.mean(g[1]) - st.mean(g[0])) * share * 42 / 1e3))
    split = collections.Counter()
    for gi, p in zip(grp, per):
        for c, _ in p:
            if sum(c) == 5: split[(gi, c)] += 1
    print("  n = 5 splits seen (group, (lone, A, B)):", dict(split))


if __name__ == "__main__":
    main()
