#!/usr/bin/env python3
"""repro_report.py DIR -- reads DIR/p1_run.log and DIR/p2_run.log of glm_repro_chain.sh (D5: run-to-run reproducibility of GLM decode past 2 051 tokens of depth).
Per ladder step k: the greedy ids of r_k (process 1, reference), x_k (process 1 again) and y_k (process 2), the first index where they differ, and the taps of the LAST position that are not bit-exact
(oracle maxabs != 0) against the dump of r_k, by layer (ascending): the lowest layer with a non-exact tap and which taps there are the root-cause candidates. python3 stdlib only."""
import re, sys, collections

def parse(path):
    cfgs = collections.OrderedDict(); cur = None
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return cfgs
    for ln in lines:
        m = re.match(r"=== gate-plan config (\S+)", ln)
        if m:
            cur = m.group(1); cfgs[cur] = {"ids": None, "taps": [], "summary": None}; continue
        if cur is None:
            continue
        if ln.startswith("greedy_ids:"):
            cfgs[cur]["ids"] = [int(t) for t in ln.split(":", 1)[1].split()]
        m = re.match(r"oracle (\S+) cos=(\S+) maxabs=(\S+)", ln)
        if m and not ln.startswith("oracle summary"):
            cfgs[cur]["taps"].append((m.group(1), float(m.group(3))))
        if ln.startswith("oracle summary"):
            cfgs[cur]["summary"] = ln[:160]
    return cfgs

def first_diff(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))

def bad_by_layer(taps):
    by = collections.defaultdict(list)
    for name, mx in taps:
        if mx != 0.0:
            m = re.match(r"(.*)-(\d+)(\.\d+)?$", name)
            by[int(m.group(2)) if m else -1].append(m.group(1) + (m.group(3) or "") if m else name)
    return by

d = sys.argv[1].rstrip("/")
p1, p2 = parse(d + "/p1_run.log"), parse(d + "/p2_run.log")
ks = sorted({int(n.split("_")[1]) for n in list(p1) + list(p2) if re.match(r"[rxy]_\d+$", n)})
print("== greedy ids: r = process-1 reference, x = process 1 again, y = process 2 (a fresh load). 'same' or the first differing index")
for tag, pre in (("DEEP (2 300 ids)", ""), ("CONTROL (1 500 ids)", "c")):
    rows = [(k, "") for k in ks] if not pre else [(8, "c")]
    for k, _ in rows:
        names = ("c_r", "c_x", "c_y") if pre else ("r_%d" % k, "x_%d" % k, "y_%d" % k)
        r = (p1.get(names[0]) or {}).get("ids"); x = (p1.get(names[1]) or {}).get("ids"); y = (p2.get(names[2]) or {}).get("ids")
        fx, fy = first_diff(r, x), first_diff(r, y)
        print("  %-20s k=%-3d r=%s  x vs r: %s  y vs r: %s" % (tag, k, "n=%d" % len(r) if r else "MISSING", "same" if (x is not None and fx is None) else ("MISSING" if x is None else "differs at %d" % fx),
                                                              "same" if (y is not None and fy is None) else ("MISSING" if y is None else "differs at %d" % fy)))
print()
print("== taps of the last position NOT bit-exact against r (layer ascending; first 6 layers with any)")
for name in list(p1) + list(p2):
    if not re.match(r"(x|y)_\d+$|c_(x|y)$", name):
        continue
    cfg = (p1.get(name) or p2.get(name))
    taps = cfg["taps"]
    by = bad_by_layer(taps)
    tot = sum(len(v) for v in by.values())
    print("  %-6s taps compared %d, not exact %d  | %s" % (name, len(taps), tot, cfg["summary"] or "no oracle summary"))
    for il in sorted(by)[:6]:
        print("      layer %-3s : %s" % (il, " ".join(sorted(by[il]))))
