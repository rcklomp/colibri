import csv, collections, statistics as st, sys
f = sys.argv[1]
rows = []
for r in csv.DictReader(open(f)):
    if "k_glm5_stage" in r["Kernel_Name"]:
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"]))
rows.sort()
# decode part: stage kernels in decode run ~ 3 per MoE layer... take the last 36 tokens' worth = 36 * 126 kernels
tail = rows[-36 * 126:]
by = collections.defaultdict(list)
for s, e, a in tail: by[a].append((e - s) / 1000.0)   # us
print("stage kernels in the decode tail: %d" % len(tail))
for a in sorted(by):
    v = sorted(by[a]); n = len(v)
    p = lambda q: v[min(n - 1, int(q * n))]
    short = [x for x in v if x > 50]    # a fetch with work (p10 was ~1 us: a card with nothing to fetch)
    print("%s n=%d  with-work n=%d  p10 %.0f  p50 %.0f  p90 %.0f  max %.0f us  | mean of with-work %.0f" % (a, n, len(short), p(.1), p(.5), p(.9), v[-1], st.mean(short) if short else 0))
    # histogram of with-work durations in 100-us bins
    h = collections.Counter(int(x // 100) * 100 for x in short)
    print("   hist(us):", " ".join("%d:%d" % (k, h[k]) for k in sorted(h) if h[k] >= max(3, 0.02 * len(short))))
