#!/usr/bin/env python3
"""rocprof_summary.py DIR -- from rocprofv3 CSVs: memory copies by direction and size class, and the kernels whose names look like copies."""
import csv, glob, collections, sys
d = sys.argv[1]
def rows(pat):
    out = []
    for f in glob.glob(d + "/**/" + pat, recursive=True):
        out += list(csv.DictReader(open(f)))
    return out
mc = rows("*memory_copy_trace.csv")
print("memory copy records:", len(mc), "columns:", list(mc[0].keys()) if mc else None)
if mc:
    k = mc[0].keys()
    dirc = next((c for c in k if "Direction" in c or "direction" in c), None)
    szc = next((c for c in k if c.lower().startswith("size") or c.lower() == "bytes"), None)
    t0 = next((c for c in k if "Start" in c), None); t1 = next((c for c in k if "End" in c), None)
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in mc:
        sz = int(r[szc]) if szc else 0
        cls = "<64KB" if sz < 65536 else "<1MB" if sz < (1 << 20) else "<16MB" if sz < (16 << 20) else ">=16MB"
        a = agg[(r[dirc], cls)]; a[0] += 1; a[1] += sz; a[2] += (int(r[t1]) - int(r[t0])) / 1e6
    print("%-34s %-8s %8s %12s %12s" % ("direction", "size", "count", "GB", "sum ms"))
    for (dr, cls), (n, b, ms) in sorted(agg.items()):
        print("%-34s %-8s %8d %12.3f %12.1f" % (dr, cls, n, b / 1e9, ms))
kt = rows("*kernel_trace.csv")
print("kernel records:", len(kt))
if kt:
    k = kt[0].keys(); nm = next(c for c in k if "Kernel_Name" in c or "Kernel_name" in c or c.lower() == "kernel_name")
    t0 = next(c for c in k if "Start" in c); t1 = next(c for c in k if "End" in c)
    agg = collections.defaultdict(lambda: [0, 0.0])
    for r in kt:
        a = agg[r[nm][:90]]; a[0] += 1; a[1] += (int(r[t1]) - int(r[t0])) / 1e6
    print("--- kernels whose name suggests a copy / blit / fill:")
    for n, (c, ms) in sorted(agg.items(), key=lambda x: -x[1][1]):
        if any(w in n.lower() for w in ("copy", "blit", "rocclr", "fill", "memcpy")):
            print("%8d  %10.1f ms  %s" % (c, ms, n))
    print("--- top 8 kernels by total time:")
    for n, (c, ms) in sorted(agg.items(), key=lambda x: -x[1][1])[:8]:
        print("%8d  %10.1f ms  %s" % (c, ms, n))
