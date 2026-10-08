#!/usr/bin/env python3
"""hc_slow_launch_report.py TRACE_DIR [TRACE_DIR ...] [--slow 1.25] [--idle-ms 5] [--window-ms 30] [--skip N] [--tokens] [--sig NAME,GX,GY,GZ,WG]

Why some launches of the hyper-connection fn GEMV (24 rows, 512 splits, 90 launches a decode token: hc_attn_fn / hc_ffn_fn of every
layer) take ~2x the usual time (record §L5-GLM-TRUNK-GEMV "Open, unexplained"; decode open-items plan item D2).  Reads the rocprofv3
kernel trace + memory-copy trace of glm_decode_trace_chain.sh (the dtrace_* directories; python3 stdlib only, run it with `python3 -I`).

What it does, per trace directory:
  1. finds the tokens (one k_glm5_hc_mean a token on the last card; the first one is the prompt's last chunk) and the hc fn launches
     (k_gemv_rowsplit with grid = 24 rows x workgroup, or k_gemm_batch 512 splits x 24 rows in the older builds; 90 a token) and
     numbers them: card (Agent 1/2/3 = dev0/1/2), layer (cards run layers 0-14 / 15-29 / 30-44 in that order), site (attn, ffn),
     layer kind (DSA when layer % 4 == 3, else KDA; layers 0-2 have a dense FFN).
  2. per launch, the state it ran in: the longest IDLE gap of its card (no kernel of any stream) that ended within --window-ms before
     it (a power-state ramp follows idle), H2D copy overlap (the adapter's swap DMA into this card), overlap with kernels of other
     streams, overlap with peer copies, its producer kernel and the gap before it.
  3. a baseline per (card, layer kind, site) from the CLEAN launches (not after an idle gap, no DMA overlap), a ratio duration/baseline,
     "slow" = ratio > --slow.
Tables: [A] distribution  [B] causes (the split that answers the question)  [C] per card  [D] per site and layer kind (clean)
[E] ramp curve after an idle gap  [F] is it the card or the kernel (other kernels of the same card in the same window)
[G] predecessor and gap (clean)  [H] overlap tests  [I] launch itself vs the gap before it  [J] host gaps at token boundaries
[K] (with --tokens) one line per token.  Several directories: a summary table at the end; with two directories of the SAME flags the
cell-level agreement of the slow flags (is it deterministic by position or by token?).

--skip N restricts the tables to tokens after the first N (the first ~5 decode tokens after a prefill carry the adapter's swap burst);
--sig overrides the hc fn kernel signature (found by itself: 24 rows, 90 launches a token).
Under tracing every duration is inflated a little; the ratios and the structure are what counts (see decode_trace_report.py).

Findings it produced on dtrace_gr1_2/3 (2026-10-08, record §L5-GLM-HC-SLOW): the slow launches of the CURRENT kernel (k_gemv_rowsplit) are
a card power-state ramp after a host-side gap at a token boundary (a traced-run artefact), not a property of the kernel; the 22 % / 21 us
population belongs to the pre-rowsplit k_gemm_batch hc fn (dtrace_wr, dtrace_rs0_*) and is gone from the shipped build.
"""
import bisect, collections, csv, glob, re, statistics as st, sys

NS = 1000.0


def short(n):
    m = re.search(r"(k_[A-Za-z0-9_]+)", n)
    return m.group(1) if m else n.split("(")[0][:40]


def med(x):
    return st.median(x) if x else float("nan")


def pct(x, p):
    if not x:
        return float("nan")
    s = sorted(x)
    return s[min(len(s) - 1, int(p * len(s)))]


def load(d):
    kf = glob.glob(d + "/**/*kernel_trace.csv", recursive=True)
    mf = glob.glob(d + "/**/*memory_copy_trace.csv", recursive=True)
    K = []
    for f in kf:
        for r in csv.DictReader(open(f)):
            try:
                K.append(dict(s=int(r["Start_Timestamp"]), e=int(r["End_Timestamp"]), a=r["Agent_Id"], n=short(r["Kernel_Name"]),
                              q=int(r["Queue_Id"]), st=int(r["Stream_Id"]), gx=int(r["Grid_Size_X"]), gy=int(r["Grid_Size_Y"]),
                              gz=int(r["Grid_Size_Z"]), wg=int(r["Workgroup_Size_X"])))
            except (ValueError, KeyError):
                pass
    M = []
    for f in mf:
        for r in csv.DictReader(open(f)):
            try:
                M.append(dict(s=int(r["Start_Timestamp"]), e=int(r["End_Timestamp"]), dir=r["Direction"].replace("MEMORY_COPY_", ""),
                              src=r["Source_Agent_Id"], dst=r["Destination_Agent_Id"], st=int(r["Stream_Id"])))
            except (ValueError, KeyError):
                pass
    K.sort(key=lambda k: (k["s"], k["e"]))
    M.sort(key=lambda m: m["s"])
    return K, M


def find_hc(K, ntok, forced):
    """The hc fn signature: 24 rows; 90 launches a decode token."""
    cnt = collections.Counter()
    for k in K:
        if k["n"] == "k_gemv_rowsplit" and k["wg"] and k["gx"] == 24 * k["wg"]:
            cnt[(k["n"], k["gx"], k["gy"], k["gz"], k["wg"])] += 1
        elif k["n"] == "k_gemm_batch" and k["wg"] and k["gx"] == 512 * k["wg"] and k["gy"] == 3 and k["gz"] == 1:
            cnt[(k["n"], k["gx"], k["gy"], k["gz"], k["wg"])] += 1
    if forced:
        sig = forced
    else:
        good = [s for s, c in cnt.items() if c == 90 * ntok]
        if len(good) != 1:
            good = sorted(cnt, key=lambda s: -cnt[s])[:1]
        if not good:
            return None, cnt
        sig = good[0]
    return sig, cnt


class Intervals:
    """sorted intervals with an overlap-time query"""
    def __init__(self, iv):
        self.iv = sorted(iv)
        self.starts = [a for a, _ in self.iv]
        self.maxd = max([b - a for a, b in self.iv], default=0)

    def overlap(self, s, e):
        if not self.iv:
            return 0, 0
        i = bisect.bisect_left(self.starts, s - self.maxd)
        tot = 0
        n = 0
        while i < len(self.iv) and self.iv[i][0] < e:
            a, b = self.iv[i]
            o = min(b, e) - max(a, s)
            if o > 0:
                tot += o
                n += 1
            i += 1
        return tot, n


def stall_tok_any(tokgap, idle_ms):
    return set(t for t, g in tokgap.items() if g >= idle_ms)


def analyse(d, args):
    K, M = load(d)
    if not K:
        print("no kernel records under", d)
        return None
    agents = sorted(set(k["a"] for k in K if k["a"] != "Agent 0"), key=lambda a: int(a.split()[-1]))
    last = agents[-1]
    heads = sorted(k["e"] for k in K if k["a"] == last and k["n"] == "k_glm5_hc_mean")
    if len(heads) < 3:
        print("too few tokens traced")
        return None
    ntok = len(heads) - 1                      # token t (1..ntok) = (heads[t-1], heads[t]]
    sig, cnt = find_hc(K, ntok, args["sig"])
    if sig is None:
        print("no hc fn signature found; candidates:", dict(cnt))
        return None
    print("=" * 110)
    print("%s" % d)
    print("kernels %d, memory copies %d, cards %s, tokens %d (the first head is the prompt's last chunk)" % (len(K), len(M), ",".join(agents), ntok))
    print("hc fn signature %s x%d (expected 90 x %d tokens = %d)" % (sig, cnt[sig], ntok, 90 * ntok))
    t_lo, t_hi = heads[0] - 5_000_000, heads[-1]
    win = [k for k in K if k["e"] >= t_lo and k["s"] <= t_hi]
    # ---- per card interval structures
    cidx = {a: i for i, a in enumerate(agents)}
    busy, idle_gaps, other = {}, {}, {}
    for a in agents:
        ks = [k for k in win if k["a"] == a]
        cur_s = cur_e = None
        merged = []
        for k in sorted(ks, key=lambda k: k["s"]):
            if cur_e is None or k["s"] > cur_e:
                if cur_e is not None:
                    merged.append((cur_s, cur_e))
                cur_s, cur_e = k["s"], k["e"]
            else:
                cur_e = max(cur_e, k["e"])
        if cur_e is not None:
            merged.append((cur_s, cur_e))
        gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1) if merged[i + 1][0] - merged[i][1] >= args["idle_ms"] * 1e6]
        idle_gaps[a] = (gaps, [g[1] for g in gaps])
        busy[a] = merged
    h2d = {a: Intervals([(m["s"], m["e"]) for m in M if m["dst"] == a and m["dir"].startswith("HOST_TO_DEVICE") and m["e"] >= t_lo and m["s"] <= t_hi]) for a in agents}
    p2p = {a: Intervals([(m["s"], m["e"]) for m in M if "Agent 0" not in (m["src"], m["dst"]) and a in (m["src"], m["dst"]) and m["e"] >= t_lo and m["s"] <= t_hi]) for a in agents}
    # per (agent, stream) lists for predecessors; per agent, other-stream kernels computed per query stream
    bystream = collections.defaultdict(list)
    for k in win:
        bystream[(k["a"], k["st"])].append(k)
    stream_start = {key: [x["s"] for x in v] for key, v in bystream.items()}
    all_iv_obj = {a: Intervals([(k["s"], k["e"]) for k in win if k["a"] == a]) for a in agents}
    # ---- the hc launches
    hc = [k for k in K if (k["n"], k["gx"], k["gy"], k["gz"], k["wg"]) == sig]
    per = collections.defaultdict(list)
    for k in hc:
        t = bisect.bisect_left(heads, k["e"])          # number of heads that ended before it = token index
        if 1 <= t <= ntok:
            per[(t, k["a"])].append(k)
    ncard = collections.Counter(len(v) for v in per.values())
    nper = {a: max([len(v) for (t, aa), v in per.items() if aa == a] or [0]) for a in agents}
    il0 = {}
    acc = 0
    for a in agents:
        il0[a] = acc
        acc += nper[a] // 2
    print("hc launches per token and card: %s  -> layers per card %s" % (dict(ncard), {a: (il0[a], il0[a] + nper[a] // 2 - 1) for a in agents}))
    # host gap before each token (head end -> first kernel of the first card)
    first = agents[0]
    copyrows = [k for k in K if k["a"] == first and k["n"] == "k_ds4_copy_rows"]
    cr_start = [k["s"] for k in copyrows]
    tokgap = {}
    for t in range(0, ntok + 1):
        i = bisect.bisect_left(cr_start, heads[t - 1] if t >= 1 else 0)
        if t >= 1 and i < len(copyrows):
            # token t starts after heads[t-1]
            tokgap[t] = (copyrows[i]["s"] - heads[t - 1]) / 1e6
    L = []
    for (t, a), v in per.items():
        v.sort(key=lambda k: k["s"])
        for i, k in enumerate(v):
            il = il0[a] + i // 2
            site = "attn" if i % 2 == 0 else "ffn"
            kind = "DSA" if il % 4 == 3 else "KDA"
            dur = (k["e"] - k["s"]) / NS
            lst = bystream[(a, k["st"])]
            j = bisect.bisect_left(stream_start[(a, k["st"])], k["s"])
            prev = lst[j - 1] if j >= 1 else None
            prev2 = lst[j - 2] if j >= 2 else None
            gap = (k["s"] - prev["e"]) / NS if prev else float("nan")
            gaps, gends = idle_gaps[a]
            gi = bisect.bisect_right(gends, k["s"]) - 1
            if gi >= 0:
                since = (k["s"] - gaps[gi][1]) / 1e6
                idle_len = (gaps[gi][1] - gaps[gi][0]) / 1e6
            else:
                since, idle_len = None, 0.0
            post_idle = since is not None and since <= args["window_ms"]
            dma, _ = h2d[a].overlap(k["s"], k["e"])
            pp, _ = p2p[a].overlap(k["s"], k["e"])
            # kernels of the same card on OTHER streams (the launch overlaps itself fully; what is left over is other work)
            ov_all, _ = all_iv_obj[a].overlap(k["s"], k["e"])
            oth = max(0, ov_all - (k["e"] - k["s"]))        # the launch overlaps itself fully; what is left is other work
            L.append(dict(t=t, a=a, c=cidx[a], il=il, site=site, kind=kind, dur=dur, s=k["s"], q=k["q"], st=k["st"], gap=gap,
                          prev=prev["n"] if prev else "-", prev2=prev2["n"] if prev2 else "-", since=since, idle_len=idle_len,
                          post=post_idle, dma=dma / NS, p2p=pp / NS, oth=oth / NS, tokgap=tokgap.get(t, float("nan"))))
    if not L:
        print("no hc launches attributed")
        return None
    # --skip N: the tables below cover tokens > N only (the idle gaps and the DMA overlap were found on the whole trace)
    ntok_all = ntok
    if args["skip"]:
        L = [x for x in L if x["t"] > args["skip"]]
        ntok = ntok_all - args["skip"]
        print("(--skip %d: tables [A]-[I] cover the %d tokens after the first %d; baselines from their clean launches)" % (args["skip"], ntok, args["skip"]))
    clean = [x for x in L if not x["post"] and x["dma"] == 0]
    base = {}
    for key in set((x["c"], x["kind"], x["site"]) for x in L):
        v = [x["dur"] for x in clean if (x["c"], x["kind"], x["site"]) == key]
        if not v:
            v = [x["dur"] for x in L if (x["c"], x["kind"], x["site"]) == key]
        base[key] = med(v)
    thr = args["slow"]
    for x in L:
        x["base"] = base[(x["c"], x["kind"], x["site"])]
        x["ratio"] = x["dur"] / x["base"]
        x["slow"] = x["ratio"] > thr
        x["exc"] = max(0.0, x["dur"] - x["base"])
    nL = len(L)
    # ---- [A]
    du = [x["dur"] for x in L]
    print("\n[A] distribution of the %d launches (us, traced): min %.1f  p10 %.1f  p25 %.1f  median %.1f  p75 %.1f  p90 %.1f  p99 %.1f  max %.1f"
          % (nL, min(du), pct(du, .1), pct(du, .25), med(du), pct(du, .75), pct(du, .9), pct(du, .99), max(du)))
    h = collections.Counter(int(x) // 2 * 2 for x in du)
    print("    2-us bins:  " + "  ".join("%d-%d:%d" % (b, b + 2, h[b]) for b in sorted(h)))
    print("    slow (duration > %.2f x its own card/kind/site clean median): %d of %d = %.1f %%;  duration > 15 us: %.1f %%"
          % (thr, sum(x["slow"] for x in L), nL, 100.0 * sum(x["slow"] for x in L) / nL, 100.0 * sum(1 for v in du if v > 15) / nL))
    # ---- [B]
    print("\n[B] what state the launches ran in (idle gap >= %.0f ms of the card within %.0f ms before the launch = 'ramp'; DMA = overlap with an H2D copy into the card)"
          % (args["idle_ms"], args["window_ms"]))
    print("    %-26s %6s %7s %9s %9s %10s %12s" % ("class", "n", "share", "median us", "slow %", "excess ms", "excess us/tok"))
    classes = [("clean", lambda x: not x["post"] and x["dma"] == 0), ("ramp only", lambda x: x["post"] and x["dma"] == 0),
               ("DMA overlap only", lambda x: not x["post"] and x["dma"] > 0), ("ramp and DMA overlap", lambda x: x["post"] and x["dma"] > 0)]
    tot_exc = sum(x["exc"] for x in L)
    for name, f in classes:
        v = [x for x in L if f(x)]
        if not v:
            continue
        e = sum(x["exc"] for x in v)
        print("    %-26s %6d %6.1f%% %9.1f %8.1f%% %10.2f %12.1f" % (name, len(v), 100.0 * len(v) / nL, med([x["dur"] for x in v]),
              100.0 * sum(x["slow"] for x in v) / len(v), e / 1000.0, e / ntok))
    print("    %-26s %6d %6.1f%% %9.1f %8.1f%% %10.2f %12.1f   (excess = sum of max(0, duration - baseline); the plan's estimate was 250 us/token)"
          % ("all", nL, 100.0, med(du), 100.0 * sum(x["slow"] for x in L) / nL, tot_exc / 1000.0, tot_exc / ntok))
    cl = [x["dur"] for x in clean]
    print("    clean launches: median %.1f  p90 %.1f  p99 %.1f  max %.1f us;  slow-flagged among them: %d of %d (%.2f %%)"
          % (med(cl), pct(cl, .9), pct(cl, .99), max(cl) if cl else float('nan'), sum(1 for x in clean if x["slow"]), len(clean),
             100.0 * sum(1 for x in clean if x["slow"]) / max(1, len(clean))))
    if not args["skip"] and ntok > 8:
        st5 = [x for x in L if x["t"] > 5]
        n5 = ntok - 5
        print("    tokens > 5 only (after the post-prefill swap burst), same baselines, %d launches, %d tokens:" % (len(st5), n5))
        for name, f in classes:
            v = [x for x in st5 if f(x)]
            if v:
                e = sum(x["exc"] for x in v)
                print("      %-26s %6d %6.1f%% slow %5.1f%%  excess %7.1f us/token" % (name, len(v), 100.0 * len(v) / len(st5),
                      100.0 * sum(x["slow"] for x in v) / len(v), e / n5))
        e = sum(x["exc"] for x in st5)
        print("      %-26s %6d  slow %5.1f%%  excess %7.1f us/token" % ("all", len(st5), 100.0 * sum(x["slow"] for x in st5) / len(st5), e / n5))
        stt = set(t for t in tokgap if t > 5 and tokgap[t] >= args["idle_ms"])
        sv = [x for x in st5 if x["t"] in stall_tok_any(tokgap, args["idle_ms"])]
        nv = [x for x in st5 if x["t"] not in stall_tok_any(tokgap, args["idle_ms"])]
        print("      tokens with a host gap >= %.0f ms before them: %d of %d; slow launches there %d of %d (%.1f %%), elsewhere %d of %d (%.2f %%)"
              % (args["idle_ms"], len(stt), n5, sum(x["slow"] for x in sv), len(sv), 100.0 * sum(x["slow"] for x in sv) / max(1, len(sv)),
                 sum(x["slow"] for x in nv), len(nv), 100.0 * sum(x["slow"] for x in nv) / max(1, len(nv))))
    # ---- [C]
    print("\n[C] per card:   n  median  p90  slow%%(all)  n_clean  slow%%(clean)  n_ramp  n_DMA")
    for a in agents:
        v = [x for x in L if x["a"] == a]
        c = [x for x in v if not x["post"] and x["dma"] == 0]
        print("    dev%d (%s, layers %d-%d): %5d %6.1f %6.1f %8.1f%% %7d %8.1f%% %7d %6d" % (cidx[a], a, il0[a], il0[a] + nper[a] // 2 - 1, len(v),
              med([x["dur"] for x in v]), pct([x["dur"] for x in v], .9), 100.0 * sum(x["slow"] for x in v) / len(v), len(c),
              100.0 * sum(x["slow"] for x in c) / max(1, len(c)), sum(x["post"] for x in v), sum(1 for x in v if x["dma"] > 0)))
    # ---- [D]
    print("\n[D] clean launches by site and layer kind:  n  median  p10  p90  p99  max   (a STATIC per-site level, not a second mode)")
    for site in ("attn", "ffn"):
        for kind in ("KDA", "DSA"):
            v = [x["dur"] for x in clean if x["site"] == site and x["kind"] == kind]
            if v:
                print("    %-5s %-4s %5d %7.1f %6.1f %6.1f %6.1f %6.1f" % (site, kind, len(v), med(v), pct(v, .1), pct(v, .9), pct(v, .99), max(v)))
    lay = collections.defaultdict(list)
    for x in clean:
        lay[(x["il"], x["site"])].append(x["dur"])
    lv = {k: med(v) for k, v in lay.items()}
    if lv:
        print("    per (layer, site) clean medians: min %.1f  median %.1f  max %.1f us; spread beyond kind/site: see the next line" % (min(lv.values()), med(list(lv.values())), max(lv.values())))
        print("    clean median by layer (attn/ffn): " + " ".join("%d:%.1f/%.1f" % (il, lv.get((il, "attn"), float("nan")), lv.get((il, "ffn"), float("nan"))) for il in range(0, acc)))
    # ---- [E]
    print("\n[E] ramp curve: launches within %.0f ms after an idle gap of the card, ratio duration/baseline by time since the gap ended" % args["window_ms"])
    pl = [x for x in L if x["post"]]
    if pl:
        bins = [0, 3, 6, 9, 12, 16, 20, 25, 30, 38, 45, 1e9]
        print("    gap length (ms)   " + "  ".join("%3d-%-3d" % (bins[i], bins[i + 1] if bins[i + 1] < 1e8 else 99) for i in range(len(bins) - 1)))
        for lo, hi in ((5, 20), (20, 40), (40, 1e9)):
            row = []
            for i in range(len(bins) - 1):
                v = [x["ratio"] for x in pl if lo <= x["idle_len"] < hi and bins[i] <= x["since"] < bins[i + 1]]
                row.append("%4.2f(%d)" % (med(v), len(v)) if v else "   -   ")
            print("    idle %3d-%-4s ms: " % (lo, "inf" if hi > 1e8 else int(hi)) + "  ".join(row))
        print("    (cell = median ratio (n); a clock / power-state ramp decays from ~2x to 1x with time; a cache or placement effect would not)")
    else:
        print("    none")
    # ---- [F]
    print("\n[F] is it the card or the kernel?  every main-stream kernel of the same card, ratio to its own clean median, in the first %.0f ms after the idle gap vs later in the window" % (args["window_ms"] / 3))
    sigk = lambda k: (k["a"], k["n"], k["gx"], k["gy"], k["wg"])
    ramp_iv = {}
    for a in agents:
        gaps, _ = idle_gaps[a]
        ramp_iv[a] = [(g[1], g[1] + args["window_ms"] * 1e6) for g in gaps]
    def in_ramp(k, frac=1.0):
        for (s, e) in ramp_iv[k["a"]]:
            if s <= k["s"] < s + (e - s) * frac:
                return True
        return False
    bysig = collections.defaultdict(list)
    for k in win:
        if k["st"] == main_stream(k, bystream):
            bysig[sigk(k)].append(k)
    rows = []
    for sg, v in bysig.items():
        c = [(x["e"] - x["s"]) for x in v if not in_ramp(x) and heads[0] < x["s"]]
        r = [(x["e"] - x["s"]) for x in v if in_ramp(x, 1.0 / 3)]
        if len(c) >= 20 and len(r) >= 3:
            rows.append((sg, med(c) / NS, med(r) / NS, len(r), sum(c)))
    rows.sort(key=lambda r: -r[4])
    print("    %-5s %-26s %5s %5s  %9s %9s %6s %5s" % ("card", "kernel (grid x wg)", "gy", "wg", "clean us", "ramp us", "ratio", "n"))
    for sg, c, r, n, _ in rows[:14]:
        print("    dev%-2d %-26s %5d %5d  %9.1f %9.1f %6.2f %5d" % (cidx[sg[0]], sg[1], sg[3], sg[4], c, r, r / c if c else float('nan'), n))
    if rows:
        rs = [r / c for _, c, r, _, _ in rows if c > 0]
        print("    median ratio over these kernel classes: %.2f  (hc fn alone: %.2f)" % (med(rs), med([x["ratio"] for x in pl if x["since"] < args["window_ms"] / 3] or [float('nan')])))
    # ---- [G]
    print("\n[G] clean launches: producer kernel two before it (same stream) and the gap before the launch")
    prod = collections.defaultdict(list)
    for x in clean:
        prod[(x["prev"], x["prev2"], x["site"])].append(x)
    for key, v in sorted(prod.items(), key=lambda kv: -len(kv[1]))[:8]:
        print("    %-16s <- %-16s site %-4s n %5d  median %5.1f  p90 %5.1f  slow %4.1f%%  gap median %5.1f us" % (key[0], key[1], key[2], len(v),
              med([x["dur"] for x in v]), pct([x["dur"] for x in v], .9), 100.0 * sum(x["slow"] for x in v) / len(v), med([x["gap"] for x in v])))
    gb = [(0, 4), (4, 6), (6, 10), (10, 30), (30, 1e9)]
    print("    by gap before the launch (us):  " + "  ".join("%s-%s: n %d slow %.1f%% med %.1f" % (lo, "inf" if hi > 1e8 else hi,
          len([x for x in clean if lo <= x["gap"] < hi]), 100.0 * sum(1 for x in clean if lo <= x["gap"] < hi and x["slow"]) / max(1, len([x for x in clean if lo <= x["gap"] < hi])),
          med([x["dur"] for x in clean if lo <= x["gap"] < hi])) for lo, hi in gb))
    qs = collections.defaultdict(list)
    for x in clean:
        qs[(x["a"], x["q"])].append(x["dur"])
    print("    by hardware queue (clean): " + "  ".join("%s q%d: n %d med %.1f" % (a[-1], q, len(v), med(v)) for (a, q), v in sorted(qs.items())))
    # ---- [H]
    print("\n[H] overlap tests (launches not after an idle gap)")
    nr = [x for x in L if not x["post"]]
    def tst(label, f):
        yes = [x for x in nr if f(x)]
        no = [x for x in nr if not f(x)]
        print("    %-44s with: n %5d slow %5.1f%% median %5.1f   without: n %5d slow %5.1f%% median %5.1f" % (label, len(yes),
              100.0 * sum(x["slow"] for x in yes) / max(1, len(yes)), med([x["dur"] for x in yes]), len(no), 100.0 * sum(x["slow"] for x in no) / max(1, len(no)),
              med([x["dur"] for x in no])))
    tst("H2D copy into the card (adapter swap DMA)", lambda x: x["dma"] > 0)
    tst("kernels of other streams on the card", lambda x: x["oth"] > 0)
    tst("peer copies from/to the card", lambda x: x["p2p"] > 0)
    dm = [x for x in nr if x["dma"] > 0]
    if dm:
        print("    DMA overlap: launch overlapped %.0f %% of its time on median; slow ratio median %.2f" % (100.0 * med([x["dma"] / x["dur"] for x in dm]), med([x["ratio"] for x in dm])))
    # ---- [I]
    print("\n[I] launch vs the gap before it:  mean over groups (us)")
    for name, v in (("clean", clean), ("ramp (first third of the window)", [x for x in L if x["post"] and x["since"] < args["window_ms"] / 3])):
        v = [x for x in v if x["gap"] == x["gap"]]
        if v:
            print("    %-34s n %5d  duration mean %6.1f median %6.1f   gap before mean %6.1f median %6.1f   (gap = previous kernel's end -> this start)"
                  % (name, len(v), st.mean([x["dur"] for x in v]), med([x["dur"] for x in v]), st.mean([x["gap"] for x in v]), med([x["gap"] for x in v])))
    # ---- [J]
    print("\n[J] host gaps at token boundaries (previous head end -> first kernel of dev0), tokens with a gap >= %.0f ms; kernel-record index of the event" % args["idle_ms"])
    ks_start = [k["s"] for k in K]
    ev = []
    for t in sorted(tokgap):
        if tokgap[t] >= args["idle_ms"]:
            i = bisect.bisect_left(cr_start, heads[t - 1])
            ev.append((t, tokgap[t], bisect.bisect_left(ks_start, copyrows[i]["s"])))
    prevn = None
    for t, g, n in ev:
        print("    token %2d: gap %6.1f ms   record index %6d%s" % (t, g, n, "   (+%d since the previous event)" % (n - prevn) if prevn is not None else ""))
        prevn = n
    recs_tok = (len([k for k in K if heads[0] < k["s"] <= heads[-1]])) / float(ntok_all)
    print("    kernel records per token: %.0f;  events: %d of %d tokens" % (recs_tok, len(ev), ntok_all))
    # ---- [K]
    if args["tokens"]:
        print("\n[K] per token:  host gap before | per card: median hc us, slow launches, H2D busy ms | token wall ms")
        for t in range(1, ntok_all + 1):
            line = "    tok %2d  gap %6.1f |" % (t, tokgap.get(t, float("nan")))
            for a in agents:
                v = [x for x in L if x["t"] == t and x["a"] == a]
                if not v:
                    continue
                line += " dev%d med %5.1f slow %2d dma %5.1f |" % (cidx[a], med([x["dur"] for x in v]), sum(x["slow"] for x in v), sum(x["dma"] for x in v))
            line += " wall %6.1f" % ((heads[t] - heads[t - 1]) / 1e6)
            print(line)
    return dict(dir=d, L=L, ntok=ntok, tokgap=tokgap, clean=clean, base=base)


def main_stream(k, bystream):
    best, bn = None, -1
    for (a, s), v in bystream.items():
        if a == k["a"] and len(v) > bn:
            best, bn = s, len(v)
    return best


def compare(r1, r2):
    print("=" * 110)
    print("agreement of the slow flags between\n   %s\n   %s" % (r1["dir"], r2["dir"]))
    c1 = {(x["t"], x["c"], x["il"], x["site"]): x["slow"] for x in r1["L"]}
    c2 = {(x["t"], x["c"], x["il"], x["site"]): x["slow"] for x in r2["L"]}
    keys = set(c1) & set(c2)
    a = sum(1 for k in keys if c1[k] and c2[k])
    b = sum(1 for k in keys if c1[k] and not c2[k])
    c = sum(1 for k in keys if not c1[k] and c2[k])
    print("    same (token, card, layer, site) cell: slow in both %d, only in the first %d, only in the second %d  -> Jaccard %.2f" % (a, b, c, a / max(1, a + b + c)))
    def cells(c):
        return set((k[0], k[1]) for k, v in c.items() if v)
    def tokcard(r):
        out = collections.defaultdict(int)
        for x in r["L"]:
            out[(x["t"], x["c"])] += x["slow"]
        return out
    t1, t2 = tokcard(r1), tokcard(r2)
    s1 = set(k for k, v in t1.items() if v >= 10)
    s2 = set(k for k, v in t2.items() if v >= 10)
    print("    (token, card) pairs with >= 10 slow launches: first %d, second %d, both %d" % (len(s1), len(s2), len(s1 & s2)))
    print("    first only: %s" % sorted(s1 - s2))
    print("    second only: %s" % sorted(s2 - s1))
    # by position alone: layer/site slow fractions correlate?
    def pos(r):
        out = collections.defaultdict(list)
        for x in r["clean"]:
            out[(x["il"], x["site"])].append(x["dur"])
        return {k: med(v) for k, v in out.items()}
    p1, p2 = pos(r1), pos(r2)
    ks = sorted(set(p1) & set(p2))
    if len(ks) > 3:
        xs = [p1[k] for k in ks]
        ys = [p2[k] for k in ks]
        mx, my = st.mean(xs), st.mean(ys)
        sx = sum((x - mx) ** 2 for x in xs) ** .5
        sy = sum((y - my) ** 2 for y in ys) ** .5
        r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")
        print("    clean per-(layer, site) medians of the two runs: correlation %.3f over %d positions (a static per-position level reproduces)" % (r, len(ks)))


def parse(argv):
    args = dict(slow=1.25, idle_ms=5.0, window_ms=30.0, tokens=False, sig=None, skip=0)
    dirs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--slow":
            args["slow"] = float(argv[i + 1]); i += 1
        elif a == "--idle-ms":
            args["idle_ms"] = float(argv[i + 1]); i += 1
        elif a == "--window-ms":
            args["window_ms"] = float(argv[i + 1]); i += 1
        elif a == "--skip":
            args["skip"] = int(argv[i + 1]); i += 1
        elif a == "--sig":
            p = argv[i + 1].split(","); args["sig"] = (p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])); i += 1
        elif a == "--tokens":
            args["tokens"] = True
        elif a.startswith("-"):
            sys.exit("unknown option " + a)
        else:
            dirs.append(a)
        i += 1
    return dirs, args


def main():
    dirs, args = parse(sys.argv[1:])
    if not dirs:
        sys.exit(__doc__)
    res = [analyse(d, args) for d in dirs]
    res = [r for r in res if r]
    if len(res) > 1:
        print("=" * 110)
        print("summary (ratio threshold %.2f, idle >= %.0f ms, window %.0f ms)" % (args["slow"], args["idle_ms"], args["window_ms"]))
        print("    %-34s %6s %8s %9s %10s %10s %12s %12s" % ("trace", "n", "median", "slow %", "slow%clean", "ramp n", "excess us/t", "clean exc/t"))
        for r in res:
            L = r["L"]
            n = len(L)
            ex = sum(x["exc"] for x in L) / r["ntok"]
            cx = sum(x["exc"] for x in r["clean"]) / r["ntok"]
            print("    %-34s %6d %8.1f %8.1f%% %9.2f%% %10d %12.1f %12.1f" % (r["dir"].rstrip("/").split("/")[-1], n, med([x["dur"] for x in L]),
                  100.0 * sum(x["slow"] for x in L) / n, 100.0 * sum(1 for x in r["clean"] if x["slow"]) / max(1, len(r["clean"])),
                  sum(x["post"] for x in L), ex, cx))
        for i in range(len(res) - 1):
            compare(res[i], res[i + 1])


if __name__ == "__main__":
    main()
