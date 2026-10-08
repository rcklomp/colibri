#!/usr/bin/env python3
"""d14_report.py RUN.LOG -- reads the log of glm_d14_chain.sh (--gate-plan): per config its greedy ids, the oracle summary (compared / not bit-exact), the greedy_ids line of the oracle, and the lowest layers whose taps are not bit-exact (maxabs != 0). python3 stdlib only."""
import re, sys, collections
cfgs = collections.OrderedDict(); cur = None
for ln in open(sys.argv[1], errors="replace").read().splitlines():
    m = re.match(r"=== gate-plan config (\S+)", ln)
    if m:
        cur = m.group(1); cfgs[cur] = {"ids": "", "bad": [], "n": 0, "summary": "", "gid": "", "rc": "?"}; continue
    if cur is None:
        continue
    c = cfgs[cur]
    if ln.startswith("greedy_ids:"): c["ids"] = ln.split(":", 1)[1].strip()
    m = re.match(r"oracle (\S+) cos=(\S+) maxabs=(\S+)", ln)
    if m and not ln.startswith("oracle summary"):
        c["n"] += 1
        if float(m.group(3)) != 0.0: c["bad"].append((m.group(1), float(m.group(3))))
    if ln.startswith("oracle summary"): c["summary"] = ln[:110]
    if ln.startswith("oracle greedy_ids"): c["gid"] = ln[:60]
    m = re.match(r"gate_plan_result name=(\S+) rc=(\d+)", ln)
    if m and m.group(1) in cfgs: cfgs[m.group(1)]["rc"] = m.group(2)
print("config  rc  taps  not-exact  greedy_ids-oracle  lowest layers with a non-exact tap")
for n, c in cfgs.items():
    by = collections.defaultdict(list)
    for name, mx in c["bad"]:
        mm = re.match(r"(.*)-(\d+)(\.\d+)?$", name)
        by[int(mm.group(2)) if mm else -1].append(mm.group(1) if mm else name)
    low = "; ".join("L%d %s" % (l, ",".join(sorted(set(by[l]))[:4])) for l in sorted(by)[:4])
    print("%-6s %-3s %5d %9d  %-24s %s" % (n, c["rc"], c["n"], len(c["bad"]), c["gid"][:24], low or "-"))
print("ids:", {n: c["ids"][:60] for n, c in cfgs.items()})
