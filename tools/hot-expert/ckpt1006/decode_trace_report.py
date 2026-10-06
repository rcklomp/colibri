#!/usr/bin/env python3
"""decode_trace_report.py DIR [--tokens N] [--layers] -- read a rocprofv3 kernel trace of GLM DECODE tokens
(glm_decode_trace_chain.sh) and decompose one MoE layer's critical path.

What it assumes (GLM5.md section 9): the cards run layer ranges one after another (agent 1 = layers 0-14, agent 2 = 15-29,
agent 3 = 30-44; the head runs once a token on the last card: k_glm5_hc_mean). For a MoE layer the OWNER runs layer_a, then
k_glm5_router -> k_glm5_plan; every other card (the HELPERS) wakes on the plan, runs k_glm5_stage (a fetch over its own host
link), the expert GEMVs, k_glm5_scatter; the owner's own staging runs on a side stream, then it joins and sums
(k_ds4_moe_accum). Timestamps are rocprofv3's, one clock for all cards. Under tracing every duration is inflated; read the
STRUCTURE (who waits for whom, the gaps), not the absolute milliseconds.

Per layer (medians over the traced tokens): layer_a (previous layer's last kernel -> this router), router+plan, helper wake
latency (plan end -> helper's first stage), the fetch (stage kernel), helper compute (stage end -> scatter end), the join
slack (last helper scatter end -> accum start), the tail (accum -> next layer's first kernel), and per card busy / idle.
"""
import csv, glob, re, sys, statistics as st, collections

def short(n):
    m = re.search(r"(k_[A-Za-z0-9_]+)", n)
    return m.group(1) if m else n.split("(")[0][:40]

def load(d):
    rows = []
    for f in glob.glob(d + "/**/*kernel_trace.csv", recursive=True):
        rows += list(csv.DictReader(open(f)))
    out = []
    for r in rows:
        try:
            out.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], short(r["Kernel_Name"]),
                        r["Queue_Id"], r["Stream_Id"]))
        except (ValueError, KeyError):
            pass
    out.sort()
    return out

def med(x):
    return st.median(x) if x else float("nan")

def us(ns):
    return ns / 1000.0

def main():
    d = sys.argv[1]
    ntok = 8
    per_layer = "--layers" in sys.argv
    if "--tokens" in sys.argv:
        ntok = int(sys.argv[sys.argv.index("--tokens") + 1])
    ks = load(d)
    if not ks:
        print("no kernel records under", d); return
    agents = sorted(set(k[2] for k in ks), key=lambda a: int(a.split()[-1]))
    gpu = [a for a in agents if a != "Agent 0"]
    by = {a: [k for k in ks if k[2] == a] for a in gpu}
    print("agents:", ", ".join("%s: %d kernels" % (a, len(by[a])) for a in gpu))
    # tokens: one k_glm5_hc_mean a token on the last card (the prefill's last chunk has one too)
    last = gpu[-1]
    heads = [k for k in by[last] if k[3] == "k_glm5_hc_mean"]
    print("head kernels (tokens incl. the prompt's last chunk):", len(heads))
    if len(heads) < 3:
        print("too few tokens traced"); return
    # token i spans (end of head i-1, end of head i]; the last ntok tokens
    ends = [h[1] for h in heads]
    toks = list(range(max(1, len(ends) - ntok), len(ends)))
    wall = [us(ends[i] - ends[i - 1]) for i in toks]
    print("traced decode tokens: %d  wall per token (us, traced): median %.0f  min %.0f  max %.0f" % (len(toks), med(wall), min(wall), max(wall)))
    # per card busy fraction in the window
    t0, t1 = ends[toks[0] - 1], ends[toks[-1]]
    print()
    print("per card over the traced window (%.1f ms):" % (us(t1 - t0) / 1000.0))
    for a in gpu:
        ev = [k for k in by[a] if k[0] >= t0 and k[1] <= t1]
        busy = sum(k[1] - k[0] for k in ev)
        # union of intervals
        u = 0; cur_s = cur_e = None
        for s, e, *_ in sorted(ev):
            if cur_e is None or s > cur_e:
                if cur_e is not None: u += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None: u += cur_e - cur_s
        print("  %s: %6d kernels  sum %.1f ms  union busy %.1f ms (%.0f %%)  %.0f kernels/token" %
              (a, len(ev), us(busy) / 1000.0, us(u) / 1000.0, 100.0 * u / (t1 - t0), len(ev) / len(toks)))
    # kernel classes in the window, all cards
    agg = collections.defaultdict(lambda: [0, 0])
    for k in ks:
        if k[0] >= t0 and k[1] <= t1:
            a = agg[k[3]]; a[0] += 1; a[1] += k[1] - k[0]
    print("\nkernel classes in the window (per token: count, summed time over all cards):")
    for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:24]:
        print("  %-28s %7.1f /token  %8.0f us /token  (%5.1f us each)" % (n, c / len(toks), us(t) / len(toks), us(t) / max(c, 1)))
    # layers: routers per owner card within each token
    rec = collections.defaultdict(list)
    lay_rows = []
    for ti in toks:
        ta, tb = ends[ti - 1], ends[ti]
        win = {a: [k for k in by[a] if k[0] >= ta and k[1] <= tb] for a in gpu}
        routers = []
        for a in gpu:
            for k in win[a]:
                if k[3] == "k_glm5_router":
                    routers.append((k[0], a))
        routers.sort()
        for li, (rs, own) in enumerate(routers):
            ow = win[own]
            # owner's router and the plan right after it
            ridx = next(i for i, k in enumerate(ow) if k[0] == rs and k[3] == "k_glm5_router")
            rk = ow[ridx]
            pk = next((k for k in ow[ridx:ridx + 6] if k[3] == "k_glm5_plan"), None)
            if pk is None: continue
            # the next router anywhere = the layer's end bound
            nxt = routers[li + 1][0] if li + 1 < len(routers) else tb
            # previous layer's last kernel on the owner = start of layer_a (the previous kernel before layer_a's first)
            prev_end = max([k[1] for k in ow[:ridx]] or [ta])
            # layer_a: from the end of the owner's previous moe_accum (hc_post follows) to the router start; approximate by the
            # last k_ds4_moe_accum / k_ds4_hc_post on the owner before the router
            acc_prev = [k for k in ow[:ridx] if k[3] in ("k_ds4_hc_post",)]
            la_start = acc_prev[-1][1] if acc_prev else ta
            # a dense layer's hc_post is also in acc_prev; take the last one that is before this router's layer_a (hc_pre)
            helpers = [a for a in gpu if a != own]
            hl = []
            for h in helpers:
                st_k = next((k for k in win[h] if k[0] >= pk[1] - 5000 and k[0] < nxt and k[3] == "k_glm5_stage"), None)
                if st_k is None: continue
                sc_k = next((k for k in win[h] if k[0] >= st_k[1] and k[0] < nxt and k[3] == "k_glm5_scatter"), None)
                hl.append((h, st_k, sc_k))
            acc = next((k for k in ow[ridx:] if k[3] == "k_ds4_moe_accum" and k[0] < nxt), None)
            if acc is None or not hl: continue
            helpers_end = max((sc[1] if sc else st_[1]) for _, st_, sc in hl)
            r = dict(owner=own, layer_idx=li,
                     layer_a=us(rk[0] - la_start),
                     router_plan=us(pk[1] - rk[0]),
                     wake=[us(s[0] - pk[1]) for _, s, _ in hl],
                     fetch=[us(s[1] - s[0]) for _, s, _ in hl],
                     hcomp=[us((sc[1] - s[1])) for _, s, sc in hl if sc],
                     join=us(acc[0] - helpers_end),
                     tail=us(nxt - acc[1]),
                     span=us(nxt - rk[0]))
            lay_rows.append(r)
            rec["layer_a"].append(r["layer_a"]); rec["router_plan"].append(r["router_plan"])
            rec["wake"] += r["wake"]; rec["fetch"] += r["fetch"]; rec["hcomp"] += r["hcomp"]
            rec["join"].append(r["join"]); rec["tail"].append(r["tail"]); rec["span"].append(r["span"])
    print("\nMoE layer decomposition, %d layer-instances (us, medians; traced timings are inflated):" % len(lay_rows))
    rows = [("layer_a (prev hc_post end -> router)", "layer_a"), ("router + plan", "router_plan"),
            ("helper wake (plan end -> helper's stage start)", "wake"), ("fetch (stage kernel, each helper)", "fetch"),
            ("helper compute (stage end -> scatter end)", "hcomp"), ("join slack (last helper end -> moe_accum start)", "join"),
            ("tail (moe_accum end -> next router)", "tail"), ("router -> next router (the layer period)", "span")]
    for lab, key in rows:
        v = rec[key]
        if v:
            print("  %-52s median %7.0f  p10 %7.0f  p90 %7.0f  (n=%d)" % (lab, med(v), sorted(v)[len(v) // 10], sorted(v)[(len(v) * 9) // 10], len(v)))
    if per_layer:
        print("\nper layer-instance:  owner  period  layer_a  wake(h1,h2)  fetch(h1,h2)  hcomp  join  tail")
        for r in lay_rows[: 3 * 45]:
            print("  %-8s %6.0f %7.0f  %s  %s  %s  %5.0f %5.0f" % (r["owner"], r["span"], r["layer_a"],
                  ",".join("%.0f" % x for x in r["wake"]), ",".join("%.0f" % x for x in r["fetch"]),
                  ",".join("%.0f" % x for x in r["hcomp"]), r["join"], r["tail"]))

if __name__ == "__main__":
    main()
