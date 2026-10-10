#!/usr/bin/env python3
"""pf2c_kernel_table.py DIR [--tokens N] [--chunk C] [--top K] [--min-us U] -- per KERNEL device time of a rocprofv3 kernel trace of a GLM PREFILL
(glm_pf2c_trace_chain.sh), per card (agent), in us per prompt token: summed duration / tokens. Kernels are grouped by (name with template arguments, grid X); a rocBLAS (Tensile)
kernel keeps the first 70 characters of its name. Columns: launches a chunk (count / chunks), mean and median duration, us/token, share of that card's kernel time.
The median next to the mean shows first-chunk or MoE-dependent spread. The pipeline overlaps chunks, so sums over the whole run are exact per-kernel totals but not a critical path.
Under tracing every duration is inflated by the serialisation of the traced queues; read WHICH kernels make a class and their relative size. python3 stdlib only (the rig has no numpy)."""
import csv, glob, re, sys, statistics as st, collections

a = sys.argv[1:]
d = a[0]
tokens, chunk, top, minus = 8192, 1024, 28, 1.0
for i, x in enumerate(a):
    if x == "--tokens": tokens = int(a[i + 1])
    if x == "--chunk": chunk = int(a[i + 1])
    if x == "--top": top = int(a[i + 1])
    if x == "--min-us": minus = float(a[i + 1])
nchunks = max(1, tokens // chunk)


def norm(n):
    n = n.replace("fk::(anonymous namespace)::", "").replace("void ", "", 1) if n.startswith("void ") else n.replace("fk::(anonymous namespace)::", "")
    n = n.replace("fk::(anonymous namespace)::", "")
    if n.startswith("Cijk_") or n.startswith("Custom_") or "Tensile" in n:
        return "rocBLAS:" + n[:70]
    m = re.match(r"([A-Za-z0-9_:]+(?:<[^()]*>)?)\(", n)
    return m.group(1) if m else n.split("(")[0][:60]


rows = []
for f in glob.glob(d + "/**/*kernel_trace.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        try:
            rows.append((r["Agent_Id"], norm(r["Kernel_Name"]), int(r["Grid_Size_X"]), (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3,
                         int(r["Start_Timestamp"]), int(r["Workgroup_Size_X"])))
        except (ValueError, KeyError):
            pass
if not rows:
    print("no kernel records under", d); sys.exit(1)
agents = sorted(set(r[0] for r in rows), key=lambda s: int(s.split()[-1]))
print("kernel records: %d, agents %s, tokens %d, chunks %d" % (len(rows), agents, tokens, nchunks))
# the run's kernel-active window per agent: warm-up kernels (model load, table edits) have tiny launch counts and are cut by --min-us
for ag in agents:
    ks = [r for r in rows if r[0] == ag]
    total = sum(r[3] for r in ks)
    g = collections.defaultdict(list)
    for r in ks:
        g[(r[1], r[2], r[5])].append(r[3])
    print("\n=== %s: %d launches, summed kernel time %.1f ms = %.1f us/token (over %d tokens)" % (ag, len(ks), total / 1e3, total / tokens, tokens))
    print("  %-62s %7s %6s %9s %9s %9s %6s" % ("kernel <template> (grid X / workgroup)", "n/chunk", "", "mean us", "median us", "us/token", "share"))
    items = sorted(g.items(), key=lambda kv: -sum(kv[1]))
    shown = 0
    for (n, gx, wg), v in items:
        ut = sum(v) / tokens
        if ut < minus: continue
        shown += 1
        if shown > top: break
        print("  %-62s %7.1f %6s %9.1f %9.1f %9.2f %5.1f%%" % ("%s [%d/%d]" % (n, gx, wg), len(v) / nchunks, "", st.mean(v), st.median(v), ut, 100 * sum(v) / total))
    rest = sum(sum(v) for k, v in items[shown:]) / tokens if shown >= top else 0.0
    if rest: print("  %-62s %7s %6s %9s %9s %9.2f" % ("(smaller rows)", "", "", "", "", rest))
# a class view: the kernel families the profile classes are made of, summed per agent in us/token
FAM = [("hc: rms_norm_mul (24-row mix input) + mix GEMM + split/wsum/post", r"k_ds4_hc_|k_rms_norm_mul"),
       ("trunk GEMM f32 / lds (k_gemm_*)", r"k_gemm_"),
       ("f16 path: dequantise / convert / epilogue", r"k_f16_|k_cvt_f16|f16_epi|k_gemm_f16|silu_mul_f16"),
       ("rocBLAS (Tensile) GEMMs", r"^rocBLAS:"),
       ("KDA conv / step / gate / l2 / gated norm", r"kda_|k_l2_norm|k_gated_rms"),
       ("indexer", r"idx_|k_glm5_layer_norm|k_ds4_topk"),
       ("attention", r"attn_|mla"),
       ("routed experts (moe / stage / scatter / swiglu)", r"k_glm5_moe|k_glm5_stage|k_glm5_scatter|k_glm5_swiglu|k_ds4_moe_accum|k_glm5_plan|k_glm5_router|k_glm5_chunk_plan"),
       ("elementwise (unary / binary / scale)", r"k_unary|k_binary|k_scale")]
print("\n=== kernel families, us/token per card (a kernel counts in the FIRST family whose pattern matches)")
print("  %-66s %s" % ("family", "  ".join("%9s" % ag for ag in agents)))
cls = {}
for ag in agents:
    cls[ag] = collections.defaultdict(float)
    for r in rows:
        if r[0] != ag: continue
        for lab, pat in FAM:
            if re.search(pat, r[1]):
                cls[ag][lab] += r[3] / tokens; break
        else:
            cls[ag]["(other)"] += r[3] / tokens
for lab in [f[0] for f in FAM] + ["(other)"]:
    print("  %-66s %s" % (lab, "  ".join("%9.1f" % cls[ag][lab] for ag in agents)))
