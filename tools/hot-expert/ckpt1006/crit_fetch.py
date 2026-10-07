# per MoE layer instance: the critical fetch = the span of the three cards' stage kernels (earliest start -> latest end); sum a token's layers; report mean/median over the decode tail.
import csv, sys, statistics as st
f = sys.argv[1]; ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 36
k = []
for r in csv.DictReader(open(f)):
    if "k_glm5_stage" in r["Kernel_Name"]:
        s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        if e - s > 20000:        # > 20 us: a fetch with work (the empty ones run 1-3 us and are not on the critical path)
            k.append((s, e, r["Agent_Id"]))
k.sort()
tail = k[-ntok * 126:] if len(k) > ntok * 126 else k
cl = []; cur = None
for s, e, a in tail:
    if cur is None or s > cur[1] + 100000:      # a gap of > 100 us starts the next layer
        if cur: cl.append(cur)
        cur = [s, e, 1]
    else:
        cur[1] = max(cur[1], e); cur[2] += 1
cl.append(cur)
spans = [(c[1] - c[0]) / 1000.0 for c in cl]
print("layer clusters in the tail: %d (expect ~%d for %d tokens x 42 layers minus empty layers)" % (len(cl), ntok * 42, ntok))
print("critical fetch a layer: mean %.0f us  median %.0f  p90 %.0f ;  a token (sum over %d tokens): %.1f ms" % (st.mean(spans), st.median(spans), sorted(spans)[int(.9 * len(spans))], ntok, sum(spans) / ntok / 1000.0))
