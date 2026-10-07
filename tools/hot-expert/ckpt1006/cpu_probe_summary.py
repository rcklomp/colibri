#!/usr/bin/env python3
# cpu_probe_summary.py PROBE_TXT T0 T1 -- the probe's per-second lines, split into the IDLE window (before T0) and the DECODE window [T0, T1] (epoch seconds):
# mean busy % of each hardware thread and of each engine thread (grouped by tid), the engine's total CPU use in cores, and how many physical cores are idle (< 10 %).
import sys, collections, statistics as st
f, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = []
for l in open(f):
    d = dict(x.split("=", 1) for x in l.rstrip("\n").replace(" cpu=", "\tcpu=").replace(" eng=", "\teng=").replace(" other=", "\tother=").split("\t") if "=" in x)
    rows.append((float(d["t"]), [float(x) for x in d["cpu"].split()], [x for x in d.get("eng", "").split() if x], d.get("other", "")))
def window(a, b): return [r for r in rows if a <= r[0] < b]
for name, ws in (("IDLE (engine loaded, nothing running)", window(0, t0 - 1)), ("DECODE (requests in flight)", window(t0 + 2, t1 - 1))):
    if not ws: print("%s: no samples" % name); continue
    n = len(ws); ncpu = len(ws[0][1])
    cpu = [st.mean(r[1][i] for r in ws) for i in range(ncpu)]
    print("== %s: %d one-second samples" % (name, n))
    print("   hardware threads busy %%: %s" % " ".join("cpu%d=%.0f" % (i, cpu[i]) for i in range(ncpu)))
    half = ncpu // 2
    pc = [(max(cpu[i], cpu[i + half]), i) for i in range(half)]   # a physical core = thread i and its sibling i+half (the usual numbering on this box)
    print("   physical cores (max of the two threads): %s | cores below 10 %% busy: %d of %d" % (" ".join("%d=%.0f" % (i, v) for v, i in pc), sum(1 for v, _ in pc if v < 10), half))
    acc = collections.defaultdict(list); info = {}
    for r in ws:
        seen = {}
        for e in r[2]:
            tid, comm, busy, psr, aff = e.split(":", 4); seen[tid] = float(busy); info[tid] = (comm, aff)
        for tid in info: acc[tid].append(seen.get(tid, 0.0))
    tot = [sum(float(e.split(":")[2]) for e in r[2]) / 100.0 for r in ws]
    print("   engine total: %.2f cores busy on average (max %.2f)" % (st.mean(tot), max(tot)))
    for tid, v in sorted(acc.items(), key=lambda kv: -st.mean(kv[1]))[:14]:
        print("   engine thread %-8s %-16s busy %5.1f %%  allowed cpus %s" % (tid, info[tid][0], st.mean(v), info[tid][1]))
    oth = collections.Counter()
    for r in ws:
        for o in r[3].split():
            nm, b = o.rsplit(":", 1); oth[nm] += float(b) / n
    print("   other busy processes (mean %% of a core): %s" % (", ".join("%s %.0f" % kv for kv in oth.most_common(5)) or "none"))
