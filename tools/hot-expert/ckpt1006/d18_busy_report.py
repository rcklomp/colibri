#!/usr/bin/env python3
"""d18_busy_report.py DIR [--tokens N] -- D18 READ part 2 (DECODE-OPEN-ITEMS-PLAN section 9): per card, how much of one decode token is busy, split by
stream, plus the card's copies (rocprofv3 --memory-copy-trace) and the fetch kernels (k_glm5_stage reads host memory over the card's link).

A depth-1 pipeline overlaps two tokens. Whatever the schedule, a token cannot cost less than the busiest resource's own work for it (the union of that
card's kernels on all streams; or that card's link: copies + stage kernels). With acceptance a, a miss redoes the head half: per accepted token the
busiest resource carries B (1 + (1 - a) h) with h ~ the head share of B. The bound is printed against the 15 % bar (S <= 0.85 W).
Tracing inflates every duration; the ratios (busy / wall) are what is read. Token boundaries as decode_trace_report.py (k_glm5_hc_mean on the last card).
"""
import csv, glob, re, sys, statistics as st, collections

def short(n):
    m = re.search(r"(k_[A-Za-z0-9_]+)", n)
    return m.group(1) if m else n.split("(")[0][:40]

def union(iv):
    u = 0; cs = ce = None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None: u += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None: u += ce - cs
    return u

def clip(iv, a, b):
    return [(max(s, a), min(e, b)) for s, e in iv if e > a and s < b]

def main():
    d = sys.argv[1]
    ntok = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 12
    ks = []
    for f in glob.glob(d + "/**/*kernel_trace.csv", recursive=True):
        for r in csv.DictReader(open(f)):
            try:
                ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], short(r["Kernel_Name"]), r.get("Stream_Id", "?"), r.get("Queue_Id", "?")))
            except (ValueError, KeyError):
                pass
    cps = []
    for f in glob.glob(d + "/**/*memory_copy_trace.csv", recursive=True):
        rows = list(csv.DictReader(open(f)))
        if rows: print("copy columns:", ", ".join(rows[0].keys()))
        for r in rows:
            try:
                cps.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r.get("Src_Agent_Id", "?"), r.get("Dst_Agent_Id", "?"),
                            r.get("Direction", r.get("Operation", "?")), int(r.get("Bytes", r.get("Size", 0)) or 0)))
            except (ValueError, KeyError):
                pass
    ks.sort(); cps.sort()
    agents = sorted({k[2] for k in ks}, key=lambda a: int(a.split()[-1]))
    by = {a: [k for k in ks if k[2] == a] for a in agents}
    last = agents[-1]
    ends = [k[1] for k in by[last] if k[3] == "k_glm5_hc_mean"]
    if len(ends) < 3:
        print("too few tokens:", len(ends)); return
    toks = list(range(max(1, len(ends) - ntok), len(ends)))
    t0, t1 = ends[toks[0] - 1], ends[toks[-1]]
    T = t1 - t0; n = len(toks)
    wall = [(ends[i] - ends[i - 1]) / 1e6 for i in toks]
    print("tokens %d, traced wall per token: median %.2f ms (min %.2f, max %.2f); window %.1f ms" % (n, st.median(wall), min(wall), max(wall), T / 1e6))
    print("copies in the window: %d" % len([c for c in cps if c[1] > t0 and c[0] < t1]))
    worst = 0.0
    for a in agents:
        kv = [(k[0], k[1]) for k in by[a]]
        kv = clip(kv, t0, t1)
        streams = collections.defaultdict(list); names = collections.defaultdict(collections.Counter)
        for k in by[a]:
            if k[1] > t0 and k[0] < t1:
                streams[k[4]].append((max(k[0], t0), min(k[1], t1))); names[k[4]][k[3]] += 1
        cv = clip([(c[0], c[1]) for c in cps if a in (c[2], c[3])], t0, t1)
        cbytes = sum(c[5] for c in cps if a in (c[2], c[3]) and c[1] > t0 and c[0] < t1)
        stage = clip([(k[0], k[1]) for k in by[a] if k[3] == "k_glm5_stage"], t0, t1)
        uk, uc, ul, ua = union(kv), union(cv), union(cv + stage), union(kv + cv)
        print("\n%s: kernels union %.0f %%, copies union %.0f %% (%.1f MB/token), link (copies + stage) %.0f %%, kernels+copies %.0f %%" %
              (a, 100 * uk / T, 100 * uc / T, cbytes / n / 1e6, 100 * ul / T, 100 * ua / T))
        for s, iv in sorted(streams.items(), key=lambda x: -union(x[1])):
            print("   stream %-4s %5.0f %%  %s" % (s, 100 * union(iv) / T, ", ".join("%s x%.0f" % (nm, c / n) for nm, c in names[s].most_common(5))))
        worst = max(worst, uk / T, ul / T)
    print("\nbusiest resource: %.0f %% of the token" % (100 * worst))
    for acc in (0.7, 0.8, 0.9):
        for h in (0.5,):
            S = worst * (1 + (1 - acc) * h)
            print("  a=%.1f: pipelined token >= %.2f W (bar: <= 0.85 W) -> %s" % (acc, S, "possible" if S <= 0.85 else "CANNOT reach the bar"))

if __name__ == "__main__":
    main()
