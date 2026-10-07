# crit_fetch2.py CSV [ntok=12] -- the critical fetch a decode token, per token, over the LAST ntok tokens (token boundaries = the once-a-token k_glm5_hc_mean on the last card, as
# decode_trace_report.py): per MoE layer the span of the three cards' stage kernels with work (earliest start -> latest end), summed over the token's layers.
import csv, sys, statistics as st, collections
f = sys.argv[1]; ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 12
st_k = []; heads = collections.defaultdict(list)
for r in csv.DictReader(open(f)):
    n = r["Kernel_Name"]; s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
    if "k_glm5_stage" in n and e - s > 20000: st_k.append((s, e))
    elif "k_glm5_hc_mean" in n: heads[r["Agent_Id"]].append(e)
last = max(heads, key=lambda a: len(heads[a]))
ends = sorted(heads[last])
st_k.sort()
per = []
for i in range(len(ends) - ntok, len(ends)):
    lo, hi = ends[i - 1], ends[i]
    ks = [k for k in st_k if lo < k[0] <= hi]
    cl = []; cur = None
    for s, e in ks:
        if cur is None or s > cur[1] + 100000: 
            if cur: cl.append(cur)
            cur = [s, e]
        else: cur[1] = max(cur[1], e)
    if cur: cl.append(cur)
    per.append((sum(c[1] - c[0] for c in cl) / 1000.0, len(cl), (hi - lo) / 1000.0))
print("last %d tokens: critical fetch a token (us) %s" % (ntok, " ".join("%.1f" % p[0] for p in per)))
print("  mean %.1f us  median %.1f  | layers with a fetch a token: mean %.1f | token wall (traced) mean %.1f us" % (
      st.mean(p[0] for p in per), st.median(p[0] for p in per), st.mean(p[1] for p in per), st.mean(p[2] for p in per)))
