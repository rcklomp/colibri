#!/usr/bin/env python3
"""rowsplit_trace_paired.py -- index-aligned per-token wall of four rocprof decode traces (glm_rowsplit_trace_seq.sh: rowsplit 0, 1, 1, 0 -> ~/bench/franken/glm5/dtrace_rs{0_1,1_2,1_3,0_4}); run ON THE RIG (it imports decode_trace_report from ~/src/colibri/tools/hot-expert/ckpt1006).
A token counts only if both same-flag pairs agree within 1 ms and none exceeds 100 ms (the 100-320 ms hiccup tokens are placement-swap rounds that land on different tokens in different runs); the other tokens give the paired on-off difference. 2026-10-07: 17 of 24 usable, median -1.66 ms; the 15 cleanest tokens -1.68 ms (SE 0.13)."""
import sys, statistics as st
sys.path.insert(0, "/home/ronald/src/colibri/tools/hot-expert/ckpt1006")
import decode_trace_report as r
def walls(name):
    ks = r.load("/home/ronald/bench/franken/glm5/dtrace_" + name)
    agents = sorted(set(k[2] for k in ks), key=lambda a: int(a.split()[-1]))
    last = [a for a in agents if a != "Agent 0"][-1]
    ends = [k[1] for k in ks if k[2] == last and k[3] == "k_glm5_hc_mean"]
    return [(ends[i] - ends[i - 1]) / 1e6 for i in range(len(ends) - 24, len(ends))]   # ms
ARMS = sys.argv[1:5] if len(sys.argv) >= 5 else ["rs0_1", "rs1_2", "rs1_3", "rs0_4"]      # trace dir suffixes after dtrace_: off, on, on, off (group traces: gr0_1 gr1_2 gr1_3 gr0_4)
o1, n2, n3, o4 = (walls(x) for x in ARMS)
rows = []
for i in range(24):
    ok = abs(o1[i] - o4[i]) < 1.0 and abs(n2[i] - n3[i]) < 1.0 and max(o1[i], o4[i], n2[i], n3[i]) < 100
    off = (o1[i] + o4[i]) / 2; on = (n2[i] + n3[i]) / 2
    rows.append((i, off, on, ok))
sel = [(i, a, b) for i, a, b, ok in rows if ok]
d = [b - a for _, a, b in sel]
print("tokens usable (both same-flag pairs agree within 1 ms, none above 100 ms): %d of 24" % len(sel))
print("off mean %.2f ms  on mean %.2f ms  paired on-off: mean %.2f  median %.2f  sd %.2f  SE %.2f  min %.2f max %.2f" % (
    st.mean(a for _, a, _ in sel), st.mean(b for _, _, b in sel), st.mean(d), st.median(d), st.stdev(d), st.stdev(d) / len(d) ** 0.5, min(d), max(d)))
print("relative: %.1f %% of the token" % (100 * st.mean(d) / st.mean(a for _, a, _ in sel)))
print("per-token on-off:", " ".join("%d:%.2f" % (i, b - a) for i, a, b in sel))
print("excluded tokens:", " ".join("%d(off %.0f/%.0f on %.0f/%.0f)" % (i, o1[i], o4[i], n2[i], n3[i]) for i, _, _, ok in rows if not ok))
