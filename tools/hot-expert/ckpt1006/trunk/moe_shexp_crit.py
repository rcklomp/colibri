#!/usr/bin/env python3
"""moe_shexp_crit.py TRACE_CSV [NTOK=12] [--dump] -- is the shared expert (sh_up/gate/down, k_gemm_batch on the owner, concurrent
with the owner's own k_glm5_stage on a side queue) on the MoE layer's critical path?

Per MoE layer on the owner (window: k_glm5_plan end -> k_ds4_moe_accum start): the owner's MAIN queue (the queue of the
sh GEMVs) busy time and idle time; the owner's own stage end; the last helper k_glm5_scatter end; which of these
three ends last before moe_accum. stdlib only."""
import csv, re, sys, statistics as st, collections

def short(n):
    m = re.search(r"(k_[A-Za-z0-9_]+(?:<[^>]*>)?)", n)
    return m.group(1).replace(" ", "") if m else n.split("(")[0][:40]

def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100.0; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

path = sys.argv[1]; ntok = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 12
ks = []
for r in csv.DictReader(open(path)):
    try: ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], r["Queue_Id"], short(r["Kernel_Name"]), int(r["Grid_Size_Y"])))
    except (ValueError, KeyError): pass
ks.sort()
agents = sorted(set(k[2] for k in ks), key=lambda a: int(a.split()[-1]))
by = {a: [k for k in ks if k[2] == a] for a in agents}
heads = [k for k in by[agents[-1]] if k[4] == "k_glm5_hc_mean"]
ends = [h[1] for h in heads]
toks = list(range(max(1, len(ends) - ntok), len(ends)))
rows = []
dumped = 0
for ti in toks:
    ta, tb = ends[ti - 1], ends[ti]
    win = {a: [k for k in by[a] if k[0] >= ta and k[1] <= tb] for a in agents}
    for a in agents:
        w = win[a]
        for i, k in enumerate(w):
            if k[4] != "k_glm5_plan": continue
            acc = next((x for x in w[i:] if x[4] == "k_ds4_moe_accum"), None)
            if not acc: continue
            mainq = acc[3]
            seg = [x for x in w if x[0] >= k[1] and x[1] <= acc[0]]
            main = [x for x in seg if x[3] == mainq]
            own_stage = [x for x in seg if x[4] == "k_glm5_stage"]
            sh = [x for x in main if x[4].startswith("k_gemm_batch")]
            helpers_end = 0
            for h in agents:
                if h == a: continue
                sc = [x for x in win[h] if x[4] == "k_glm5_scatter" and x[0] >= k[1] and x[1] <= acc[0] + 2000000]
                if sc: helpers_end = max(helpers_end, min(sc, key=lambda x: abs(x[1] - acc[0]))[1])
            busy = sum(x[1] - x[0] for x in main)
            main_end = max((x[1] for x in main), default=k[1])
            stage_end = max((x[1] for x in own_stage), default=0)
            # the main-queue work that does NOT depend on the own stage: kernels that END before the own stage ends
            sh_dur = sum(x[1] - x[0] for x in sh)
            last = max((("main", main_end), ("own_stage", stage_end), ("helpers", helpers_end)), key=lambda x: x[1])
            # main-queue idle between the sh_up start and the moe_accum start
            idle = (acc[0] - (sh[0][0] if sh else k[1])) - busy
            pre = [x for x in main if x[0] < stage_end]
            pre_end = max((x[1] for x in pre), default=k[1])
            gain_cap = min(max(0, pre_end - stage_end), max(0, main_end - helpers_end)) / 1e3
            rows.append(dict(gain_cap=gain_cap, pre_minus_stage=(pre_end - stage_end) / 1e3, sh=sh_dur / 1e3, busy=busy / 1e3, idle=idle / 1e3, last=last[0],
                             main_minus_other=(main_end - max(stage_end, helpers_end)) / 1e3,
                             own_stage=(len(own_stage) > 0), period=(acc[0] - k[1]) / 1e3,
                             stage_overlap_sh=sum(1 for x in sh if any(s[0] < x[1] and s[1] > x[0] for s in own_stage))))
            if "--dump" in sys.argv and dumped < 2 and own_stage:
                dumped += 1
                print("--- layer on %s, plan end %d" % (a, k[1]))
                for x in seg:
                    print("  q%s %-26s %8.1f -> %8.1f  (%6.1f us)" % (x[3], x[4], (x[0] - k[1]) / 1e3, (x[1] - k[1]) / 1e3, (x[1] - x[0]) / 1e3))
                print("  helpers_end %.1f  accum start %.1f" % ((helpers_end - k[1]) / 1e3, (acc[0] - k[1]) / 1e3))
T = len(toks)
print("MoE layers: %d (%.1f/token); with an own stage on the owner: %d" % (len(rows), len(rows) / T, sum(r["own_stage"] for r in rows)))
c = collections.Counter(r["last"] for r in rows)
print("who ends last before moe_accum:", dict(c))
for lab, sel in (("all", lambda r: True), ("own stage present", lambda r: r["own_stage"]), ("no own stage", lambda r: not r["own_stage"])):
    v = [r for r in rows if sel(r)]
    if not v: continue
    print("%-18s n %4d  sh GEMVs med %.1f us (sum %.2f ms/tok)  main busy med %.1f  main idle med %.1f p10 %.1f  "
          "main_end - max(own stage, helpers) med %.1f p90 %.1f  sh overlapping own stage med %.0f of 3" %
          (lab, len(v), st.median([r["sh"] for r in v]), sum(r["sh"] for r in v) / 1e3 / T, st.median([r["busy"] for r in v]),
           st.median([r["idle"] for r in v]), pct([r["idle"] for r in v], 10),
           st.median([r["main_minus_other"] for r in v]), pct([r["main_minus_other"] for r in v], 90),
           st.median([r["stage_overlap_sh"] for r in v])))
crit = [r for r in rows if r["last"] == "main"]
print("layers where the owner's main queue ends last: %d (%.1f/token); by how much (us): med %.1f p90 %.1f; "
      "sum over them of min(margin, sh time) %.2f ms/token" %
      (len(crit), len(crit) / T, st.median([r["main_minus_other"] for r in crit]) if crit else 0,
       pct([r["main_minus_other"] for r in crit], 90) if crit else 0,
       sum(min(r["main_minus_other"], r["sh"]) for r in crit) / 1e3 / T))
print("main-queue work launched before the own stage ends, minus the stage end (us): med %.1f p90 %.1f; layers > 0: %d/%d" %
      (st.median([r["pre_minus_stage"] for r in rows]), pct([r["pre_minus_stage"] for r in rows], 90),
       sum(1 for r in rows if r["pre_minus_stage"] > 0), len(rows)))
print("upper bound of what a faster shared expert can save (min(pre_end - stage_end, main_end - helpers_end, sh time) summed): %.2f ms/token" %
      (sum(min(r["gain_cap"], r["sh"]) for r in rows) / 1e3 / T))
