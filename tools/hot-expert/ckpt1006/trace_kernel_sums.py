#!/usr/bin/env python3
"""trace_kernel_sums.py DIR [DIR ...] [--tokens N] [--kernels a,b,c] -- per decode token, the summed duration (all three cards) of chosen kernel classes in rocprofv3 kernel traces of
glm_decode_trace_chain.sh; medians over the last N tokens (default 20) that are not hiccup tokens (a token whose wall is > 1.25x the median wall is left out: swap rounds / host stalls, see record
D2). Token boundary = the once-a-token k_glm5_hc_mean on the last card, as rowsplit_trace_paired.py. Prints, per DIR, one line per kernel class and the sum of the chosen classes; the point is to
compare arms of one binary that differ in one flag (D1 --gemv-lds, D3 --gemv-rowsplit-waves): kernel durations repeat to the microsecond between identical runs. python3 stdlib only; run ON THE RIG
(imports decode_trace_report from ~/src/colibri/tools/hot-expert/ckpt1006 or the same dir)."""
import sys, os, statistics as st, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ronald/src/colibri/tools/hot-expert/ckpt1006")
import decode_trace_report as r

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ntok = 20; kern = ["k_gemm_batch", "k_gemv_rowsplit", "k_reduce_splits_gemm_w"]
a = sys.argv[1:]
for i, x in enumerate(a):
    if x == "--tokens": ntok = int(a[i + 1]); args.remove(a[i + 1])
    if x == "--kernels": kern = a[i + 1].split(","); args.remove(a[i + 1])
res = {}
for d in args:
    ks = r.load(d)                                   # rows: (start, end, agent, short_name, ...)
    agents = sorted(set(k[2] for k in ks), key=lambda a_: int(a_.split()[-1]))
    last = [a_ for a_ in agents if a_ != "Agent 0"][-1]
    ends = sorted(k[1] for k in ks if k[2] == last and k[3] == "k_glm5_hc_mean")
    bounds = ends[-(ntok + 1):]
    walls = [(bounds[i + 1] - bounds[i]) / 1e6 for i in range(len(bounds) - 1)]
    medw = st.median(walls)
    good = [i for i, w in enumerate(walls) if w <= 1.25 * medw]
    per = collections.defaultdict(lambda: [0.0] * len(walls)); cnt = collections.defaultdict(lambda: [0] * len(walls))
    for k in ks:
        t = k[1]
        if t <= bounds[0] or t > bounds[-1]: continue
        j = next(i for i in range(len(bounds) - 1) if t <= bounds[i + 1])
        per[k[3]][j] += (k[1] - k[0]) / 1e3; cnt[k[3]][j] += 1        # us
    res[d] = (walls, good, per, cnt)
    print("== %s : %d tokens, %d clean (wall <= 1.25 x median %.1f ms)" % (d.rstrip("/").split("dtrace_")[-1], len(walls), len(good), medw))
    tot = [0.0] * len(walls)
    for name in kern:
        v = per.get(name)
        if v is None: print("   %-28s absent" % name); continue
        for i in range(len(walls)): tot[i] += v[i]
        m = st.median([v[i] for i in good]); c = st.median([cnt[name][i] for i in good])
        print("   %-28s median %8.1f us a token  (%d launches a token)" % (name, m, c))
    print("   %-28s median %8.1f us a token ; token wall median %.2f ms" % ("SUM of the above", st.median([tot[i] for i in good]), st.median([walls[i] for i in good])))
if len(args) >= 2:
    print("== paired differences against the first DIR, per token over the tokens clean in BOTH (sum of the chosen classes, us):")
    base = args[0]
    for d in args[1:]:
        gb, gd = set(res[base][1]), set(res[d][1]); both = sorted(gb & gd)
        if not both: print("   %s: no common clean tokens" % d); continue
        def tsum(x, i): return sum(res[x][2][n][i] for n in kern if n in res[x][2])
        diffs = [tsum(d, i) - tsum(base, i) for i in both]
        print("   %-30s n=%d  mean %+.1f  median %+.1f  sd %.1f  SE %.1f us a token" % (d.rstrip("/").split("dtrace_")[-1], len(both), st.mean(diffs), st.median(diffs), st.stdev(diffs) if len(diffs) > 1 else 0, (st.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else 0))
