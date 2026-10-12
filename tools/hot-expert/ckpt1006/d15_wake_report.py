#!/usr/bin/env python3
"""d15_wake_report.py TRACE_DIR [--tokens N] -- D15 / D21 (record §L5-D15): when does each card's decode stage kernel START after the owner's plan,
and what would the token's critical fetch be if card 0 (rocprof Agent 1) started on time?
TRACE_DIR holds prof_kernel_trace.csv and prof_memory_copy_trace.csv (glm_decode_trace_chain.sh's out/). Per MoE layer of the last N decode tokens:
the delay plan end -> the card's first stage kernel after it, split by role (owner / helper) and by whether a host-to-device copy (the adapter's
swap round) was in flight to that card at the plan's end; then the critical fetch (plan end -> the last stage end with work) as traced and with every
late (> 60 us) card-0 helper stage moved to plan end + 20 us at its own duration (an UPPER bound: an on-time card 0 shares the pair's link more).
"""
import csv, sys, bisect, statistics as st, collections


def main():
    d = sys.argv[1]
    nt = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 24
    ks = []
    for r in csv.DictReader(open(d + "/prof_kernel_trace.csv")):
        n = r["Kernel_Name"]
        for key in ("k_glm5_plan", "k_glm5_stage", "k_glm5_hc_mean"):
            if key in n and "stage_tab" not in n:
                ks.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], key)); break
    ks.sort()
    mc = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Destination_Agent_Id"])
                for r in csv.DictReader(open(d + "/prof_memory_copy_trace.csv")))
    heads = [k for k in ks if k[3] == "k_glm5_hc_mean" and k[2] == "Agent 3"]
    t0, t1 = heads[-nt - 1][1], heads[-1][1]
    plans = [k for k in ks if k[3] == "k_glm5_plan" and t0 < k[0] < t1]
    cards = ("Agent 1", "Agent 2", "Agent 3")
    stg = {a: sorted(k for k in ks if k[3] == "k_glm5_stage" and k[2] == a) for a in cards}
    starts = {a: [k[0] for k in stg[a]] for a in cards}
    delay = collections.defaultdict(list)
    real = fixed = 0; late = swapl = 0
    for i, p in enumerate(plans):
        nxt = plans[i + 1][0] if i + 1 < len(plans) else t1
        ends, ends_fix, swap = {}, {}, False
        for a in cards:
            j = bisect.bisect_left(starts[a], p[1])
            if j >= len(stg[a]) or stg[a][j][0] > nxt: continue
            s = stg[a][j]; dl = s[0] - p[1]; du = s[1] - s[0]
            busy = any(m[2] == a and m[0] <= p[1] <= m[1] for m in mc)
            delay[(a, "owner" if a == p[2] else "helper", busy)].append(dl / 1e3)
            if du < 50000: continue
            swap |= busy
            ends[a] = s[1]
            mv = a == "Agent 1" and a != p[2] and dl > 60000
            ends_fix[a] = p[1] + 20000 + du if mv else s[1]
            late += mv
        if swap: swapl += 1; continue
        if ends: real += max(ends.values()) - p[1]; fixed += max(ends_fix.values()) - p[1]
    print("%s: %d decode tokens, %d MoE layers" % (d, nt, len(plans)))
    for k in sorted(delay):
        v = sorted(delay[k])
        print("  %s %-6s swap copy in flight=%-5s n=%4d  plan end -> stage start  p10 %5.0f  median %5.0f  p90 %5.0f us" % (
            k[0], k[1], k[2], len(v), v[len(v) // 10], st.median(v), v[9 * len(v) // 10]))
    print("  critical fetch (plan end -> last stage end), swap-round layers (%d) left out: as traced %.2f ms a token; card 0 helper on time "
          "(%d layers moved) %.2f ms: upper bound %.2f ms a token" % (swapl, real / 1e6 / nt, late, fixed / 1e6 / nt, (real - fixed) / 1e6 / nt))


if __name__ == "__main__":
    main()
