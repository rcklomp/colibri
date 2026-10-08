#!/usr/bin/env python3
"""serve_ab_clock_report.py [DIR] [--all] [--csv OUT.csv] [--min-samples N]   (python3 stdlib only; 3.9+; runs on the Mac and on the rig)

Reads the output of glm_serve_ab_chain.sh with SAB_SAMPLE=1 (default DIR ~/bench/franken/glm5/serve_ab):
    <arm>.res.json   the driver's rows: request tag, decode tok/s, decode_s, t_start / t_end (epoch s)
    <arm>.gpu.tsv    gpu_sampler.py arm: one row per card (dev = PCI address) every 5 s: sclk mclk fclk socclk pcie_lvl power_w temp_hot temp_edge temp_mem busy memb avg_sclk thr
    <arm>.cpu.tsv    gpu_sampler.py arm: one row every 5 s: effective MHz of the busy hardware threads (and of all), utilisation, Tctl
    <arm>.gw.log     the engine log (only used for request end times when an old driver wrote no t_end)
and prints, per arm and per request, the decode tok/s next to the mean (min-max) of the GPU shader clock, memory clock, hotspot and edge temperature,
power of each card and the CPU MHz of the busy threads, taken over the samples inside that request's DECODE window [t_end - decode_s, t_end];
then the Pearson correlation of per-request tok/s with each of those quantities (and with the request's worst value of it).

How to read it: an odd arm is one whose tok/s sits several percent away from the same request in the arms of the same build (column vs-grp). Look at that arm's block:
a card whose sclk mean / min is lower, whose hotspot is higher, whose power sits on the cap (300 W), or CPU MHz lower than in its sibling arms, is the lead. The correlation
tables say whether the lead holds across all requests. `raw` pools every request (the prompt and the build also move tok/s: W1 at depth 9k is slower than W2); `within`
subtracts the mean of each (request, build) group from both tok/s and the quantity, which removes those two effects and keeps only what differs between arms of the
same build. A `*` marks |r| above the two-sided 5 % limit for that many points; 30 tests are run, so one or two stars are expected by chance. ~20 requests with 3-6
samples each is a lead, not a proof. `const` = the quantity did not vary (no information).
"""
import calendar, csv, functools, glob, json, math, os, re, sys, time

DIR_DEFAULT = os.path.expanduser("~/bench/franken/glm5/serve_ab")
# (column, label, unit, worst): `worst` is the extreme that hurts, used for the second correlation row of each quantity
GPU_Q = [("sclk", "sclk", "MHz", "min"), ("mclk", "mclk", "MHz", "min"), ("temp_hot", "hotspot", "C", "max"), ("temp_edge", "edge", "C", "max"), ("power_w", "power", "W", "max")]
GPU_X = [("avg_sclk", "avg_sclk", "MHz", "min"), ("fclk", "fclk", "MHz", "min"), ("socclk", "socclk", "MHz", "min"), ("temp_mem", "memtemp", "C", "max"),
         ("busy", "busy", "%", "min"), ("memb", "membusy", "%", "max"), ("pcie_lvl", "pcie_lvl", "", "min")]
CPU_Q = [("mhz_busy", "cpu busy MHz", "MHz", "min")]
CPU_X = [("mhz_all", "cpu all MHz", "MHz", "min"), ("util_all", "cpu util", "%", "max"), ("tctl_c", "cpu Tctl", "C", "max"), ("n_busy", "busy threads", "", "min")]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}
NAN = float("nan")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def ok(x):
    return x == x and x not in (float("inf"), float("-inf"))


def mean(v):
    return sum(v) / len(v) if v else NAN


def median(v):
    s = sorted(v); n = len(s)
    return NAN if not n else (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]))


def read_tsv(path):
    """-> (meta, rows): meta from '# ... key=value' header lines, rows = list of {column: text}; (None, []) when the file is absent."""
    if not os.path.exists(path):
        return None, []
    meta, cols, rows = {}, None, []
    with open(path, errors="replace") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                for k, v in re.findall(r"(\w+)=(\S+)", ln):
                    meta.setdefault(k, v)
            elif cols is None:
                cols = ln.split("\t")
            elif ln:
                p = ln.split("\t")
                if len(p) == len(cols):
                    rows.append(dict(zip(cols, p)))
    return meta, rows


def engine_ends(path, tz_offset):
    """{req number: epoch end} from the engine log's `YYYY-MM-DD HH:MM:SS [serve-glm5] req=N` lines (fallback when the driver wrote no t_end)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for ln in f:
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[serve-(?:glm5|ds4)\] req=(\d+) ", ln)
            if m:
                out[int(m.group(2))] = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) - tz_offset
    return out


def stats(vals):
    v = [x for x in vals if ok(x)]
    return (mean(v), min(v), max(v)) if v else (NAN, NAN, NAN)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-12 * max(1.0, abs(mx)) ** 2 * n or syy <= 1e-12 * max(1.0, abs(my)) ** 2 * n:
        return "const"
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def rcrit(df):
    if df < 1:
        return 2.0
    t = (T975.get(df) or T975[max(k for k in T975 if k <= df)]) if df <= 120 else 1.960     # between table rows: the next lower df (the larger t)
    return t / math.sqrt(df + t * t)


def fmt_r(r, df):
    if r is None:
        return "   n<4"
    if r == "const":
        return " const"
    return "%+.2f%s" % (r, "*" if abs(r) >= rcrit(df) else " ")


def arm_order(name):
    m = re.match(r"(\d+)_", name)
    return (int(m.group(1)) if m else 10 ** 6, name)


def build_of(name):
    m = re.match(r"\d+_(.*)", name)
    return m.group(1) if m else name


def short(dev):
    return dev[5:] if dev.startswith("0000:") else dev


def load(d, with_all):
    arms, notes = [], []
    gq = GPU_Q + (GPU_X if with_all else []); cq = CPU_Q + (CPU_X if with_all else [])
    for rp in sorted(glob.glob(os.path.join(d, "*.res.json")), key=lambda p: arm_order(os.path.basename(p)[:-9])):
        arm = os.path.basename(rp)[:-9]
        try:
            rows = json.load(open(rp))
        except (OSError, ValueError) as e:
            notes.append("%s: cannot read the result file (%s)" % (arm, e)); continue
        gm, gr = read_tsv(os.path.join(d, arm + ".gpu.tsv")); cm, cr = read_tsv(os.path.join(d, arm + ".cpu.tsv"))
        if gm is None:
            notes.append("%s: no %s.gpu.tsv (arm ran unsampled): tok/s only" % (arm, arm))
        elif not gr:
            notes.append("%s: %s.gpu.tsv has no rows (the sampler died at once; see %s.sampler.log)" % (arm, arm, arm))
        tz = int(fnum((gm or cm or {}).get("tz_offset_s", 0)) or 0)
        ends = None
        for r in gr + cr:
            for k in r:
                if k not in ("arm", "req", "dev", "thr", "mhz_cpus", "util_cpus"):
                    r[k] = fnum(r[k])
        devs = sorted({r["dev"] for r in gr})
        g_by = {}
        for r in gr:
            g_by.setdefault(r["dev"], []).append(r)
        arm_info = {"arm": arm, "build": build_of(arm), "devs": devs, "period": fnum((gm or cm or {}).get("period_s")), "reqs": [],
                    "t_first": min([r["t_epoch"] for r in gr] or [NAN]), "t_last": max([r["t_epoch"] for r in gr] or [NAN]),
                    "hot_first": {dv: (g_by[dv][0]["temp_hot"], max(x["temp_hot"] for x in g_by[dv] if ok(x["temp_hot"])) if any(ok(x["temp_hot"]) for x in g_by[dv]) else NAN) for dv in devs}}
        for row in rows:
            rec = {"arm": arm, "build": arm_info["build"], "tag": row.get("tag", "?"), "tok_s": fnum(row.get("tok_s")), "decode_s": fnum(row.get("decode_s")),
                   "emitted": row.get("emitted", "?"), "prefill_s": fnum(row.get("prefill_s")), "n": 0, "ntag": 0, "q": {}, "window": None, "src": ""}
            t_end = fnum(row.get("t_end")); rec["src"] = "driver"
            if not ok(t_end) and "req" in row:
                if ends is None:
                    ends = engine_ends(os.path.join(d, arm + ".gw.log"), tz)
                t_end = float(ends.get(row["req"], NAN)); rec["src"] = "engine log (1 s)"
            if ok(t_end) and ok(rec["decode_s"]) and gr:
                lo, hi = t_end - rec["decode_s"], t_end; rec["window"] = (lo, hi)
                for dv in devs:
                    w = [r for r in g_by[dv] if lo <= r["t_epoch"] <= hi]
                    if dv == devs[0]:
                        rec["n"] = len(w); rec["ntag"] = sum(1 for r in w if r["req"] == rec["tag"])
                    for col, lab, unit, worst in gq:
                        rec["q"][(dv, col)] = stats([r.get(col, NAN) for r in w])
                    bits = [int(r["thr"], 16) for r in w if re.match(r"^[0-9a-f]+$", r.get("thr", "-"))]
                    rec.setdefault("thr", {})[dv] = functools.reduce(lambda x, y: x | y, bits) if bits else None
                wc = [r for r in cr if lo <= r["t_epoch"] <= hi]
                for col, lab, unit, worst in cq:
                    if col == "mhz_busy":      # mean of the per-sample means; lowest / highest busy thread seen
                        rec["q"][("cpu", col)] = (stats([r["mhz_busy"] for r in wc])[0], stats([r["mhz_busy_min"] for r in wc])[1], stats([r["mhz_busy_max"] for r in wc])[2])
                    else:
                        rec["q"][("cpu", col)] = stats([r.get(col, NAN) for r in wc])
                rec["nbusy"] = mean([r["n_busy"] for r in wc if ok(r["n_busy"]) and r["n_busy"] >= 0])
            arm_info["reqs"].append(rec)
        arms.append(arm_info)
    # tok/s against the same request in the arms of the same build
    grp = {}
    for a in arms:
        for r in a["reqs"]:
            if ok(r["tok_s"]):
                grp.setdefault((r["tag"], r["build"]), []).append(r)
    for g in grp.values():
        md = median([r["tok_s"] for r in g])
        for r in g:
            r["vs_grp"] = 100.0 * (r["tok_s"] / md - 1.0) if len(g) >= 2 and md > 0 else NAN
    return arms, notes, gq, cq


def quantities(arms, gq, cq):
    """[(label, key, stat)] in print order: for each card each GPU quantity (mean and worst), then the CPU ones."""
    devs = sorted({dv for a in arms for dv in a["devs"]})
    out = []
    for dv in devs:
        for col, lab, unit, worst in gq:
            nm = "%s %s%s" % (short(dv), lab, " " + unit if unit else "")
            out.append((nm, (dv, col), "mean")); out.append((nm, (dv, col), worst))
    for col, lab, unit, worst in cq:
        nm = lab if (not unit or unit in lab) else lab + " " + unit
        out.append((nm, ("cpu", col), "mean")); out.append((nm, ("cpu", col), worst))
    return out


def pick(rec, key, stat):
    s = rec["q"].get(key)
    if not s:
        return NAN
    return {"mean": s[0], "min": s[1], "max": s[2]}[stat]


def correlations(arms, qs):
    recs = [r for a in arms for r in a["reqs"] if ok(r["tok_s"])]
    res = []
    for nm, key, stat in qs:
        pts = [(r, pick(r, key, stat)) for r in recs]
        pts = [(r, x) for r, x in pts if ok(x)]
        xs = [x for _, x in pts]; ys = [r["tok_s"] for r, _ in pts]
        raw = pearson(xs, ys) if len(pts) >= 4 else None
        by = {}
        for r, x in pts:
            by.setdefault((r["tag"], r["build"]), []).append((x, r["tok_s"]))
        by = {k: v for k, v in by.items() if len(v) >= 2}
        dx, dy = [], []
        for v in by.values():
            mx, my = mean([a for a, _ in v]), mean([b for _, b in v])
            dx += [a - mx for a, _ in v]; dy += [b - my for _, b in v]
        win = pearson(dx, dy) if len(dx) >= 4 else None
        res.append({"name": nm, "stat": stat, "raw": raw, "n": len(pts), "win": win, "nw": len(dx), "groups": len(by)})
    return res


def f0(x):
    return "%4.0f" % x if ok(x) else "   -"


def trip(s, unit_fmt="%.0f"):
    if not s or not ok(s[0]):
        return "-"
    return (unit_fmt + " (" + unit_fmt + "-" + unit_fmt + ")") % s


def report(arms, notes, gq, cq, min_samples, out):
    w = out.write
    nreq = sum(1 for a in arms for r in a["reqs"] if ok(r["tok_s"]))
    w("=== serve-path A/B clock report: %d arms, %d requests with a decode tok/s (decode window = [t_end - decode_s, t_end]; mean (min-max) over the samples inside)\n" % (len(arms), nreq))
    for n in notes:
        w("NOTE %s\n" % n)
    for a in arms:
        span = "" if not ok(a["t_first"]) else "  sampler %s - %s UTC, period %g s" % (time.strftime("%H:%M:%S", time.gmtime(a["t_first"])), time.strftime("%H:%M:%S", time.gmtime(a["t_last"])), a["period"])
        w("\n--- arm %s  build=%s%s\n" % (a["arm"], a["build"], span))
        if a["devs"]:
            w("    hotspot C at arm start -> max over the arm: " + "   ".join("%s %s -> %s" % (short(dv), f0(a["hot_first"][dv][0]).strip(), f0(a["hot_first"][dv][1]).strip()) for dv in a["devs"]) + "\n")
        w("    %-14s %7s %8s %9s %8s %5s %4s\n" % ("request", "tok/s", "vs-grp", "decode_s", "emitted", "n", "tag"))
        for r in a["reqs"]:
            vs = "%+.1f%%" % r["vs_grp"] if ok(r.get("vs_grp", NAN)) else "-"
            w("    %-14s %7s %8s %9s %8s %5d %4d%s\n" % (r["tag"], "%.2f" % r["tok_s"] if ok(r["tok_s"]) else "-", vs, "%.1f" % r["decode_s"] if ok(r["decode_s"]) else "-",
                                                      r["emitted"], r["n"], r["ntag"], "" if r["src"] == "driver" else "   (times from the %s)" % r["src"]))
            if r["window"] is None:
                if a["devs"]:
                    w("        no sample window (no t_end / decode_s for this request)\n")
                continue
            if r["n"] < min_samples:
                w("        only %d sample%s in the decode window (%.1f s): statistics are not meaningful\n" % (r["n"], "" if r["n"] == 1 else "s", r["window"][1] - r["window"][0]))
                if r["n"] == 0:
                    continue
            nb = r.get("nbusy", NAN)
            w("        cpu     busy-thread MHz %-24s %s\n" % (trip(r["q"].get(("cpu", "mhz_busy"))), "(%.1f threads busy on average)" % nb if ok(nb) else ""))
            for dv in a["devs"]:
                parts = []
                for col, lab, unit, worst in gq:
                    parts.append("%s %s" % (lab, trip(r["q"].get((dv, col)))))
                t = r.get("thr", {}).get(dv)
                w("        %-7s %s   throttle bits %s\n" % (short(dv), "   ".join(parts), "-" if t is None else "0x%x" % t))
    qs = quantities(arms, gq, cq)
    cor = correlations(arms, qs)
    nw = max([c["nw"] for c in cor] or [0]); ng = max([c["groups"] for c in cor] or [0])
    w("\n=== Pearson r of per-request decode tok/s with the sampled quantities of the same request's decode window (n = requests with both values)\n")
    w("    raw    : all requests pooled (prompt and build also move tok/s)\n")
    w("    within : tok/s and the quantity both minus the mean of their (request, build) group; %d groups with >= 2 arms, %d points\n" % (ng, nw))
    w("    stat   : mean over the window, or the worst extreme (min for clocks, max for temperature / power); * = |r| above the two-sided 5 % limit\n")
    w("    %-28s %-5s %8s %4s   %8s %4s\n" % ("quantity", "stat", "r raw", "n", "r within", "n"))
    for c in cor:
        w("    %-28s %-5s %8s %4d   %8s %4d\n" % (c["name"], c["stat"], fmt_r(c["raw"], c["n"] - 2), c["n"], fmt_r(c["win"], c["nw"] - c["groups"] - 1), c["nw"]))
    for lab, k, nk, dfn in (("within", "win", "nw", lambda c: c["nw"] - c["groups"] - 1), ("raw", "raw", "n", lambda c: c["n"] - 2)):
        top = sorted([c for c in cor if isinstance(c[k], float)], key=lambda c: (-round(abs(c[k]), 9), c["name"], c["stat"]))[:5]
        w("    largest |r %s|: %s\n" % (lab, "; ".join("%s %s %+.2f%s" % (c["name"], c["stat"], c[k], "*" if abs(c[k]) >= rcrit(dfn(c)) else "") for c in top) or "none computable"))
    w("    (5 s snapshots, 3-6 per request: a lead to follow with the raw files, not a proof. Raw files: <arm>.gpu.tsv / <arm>.cpu.tsv, one row per card / per sample.)\n")
    return qs


def write_csv(path, arms, qs):
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["arm", "build", "request", "tok_s", "vs_group_pct", "decode_s", "emitted", "n_samples"] + ["%s %s" % (nm, st) for nm, _, st in qs])
        for a in arms:
            for r in a["reqs"]:
                cw.writerow([a["arm"], a["build"], r["tag"], r["tok_s"], r.get("vs_grp", NAN), r["decode_s"], r["emitted"], r["n"]] + ["%.3f" % pick(r, k, st) if ok(pick(r, k, st)) else "" for _, k, st in qs])


def main(argv, out=None):
    out = out or sys.stdout
    d, csvp, allq, mins = DIR_DEFAULT, None, False, 2
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--all":
            allq = True
        elif a == "--csv" and i + 1 < len(argv):
            i += 1; csvp = argv[i]
        elif a == "--min-samples" and i + 1 < len(argv):
            i += 1; mins = int(argv[i])
        elif a in ("-h", "--help") or a.startswith("--"):
            out.write(__doc__); return 2
        else:
            d = a
        i += 1
    arms, notes, gq, cq = load(d, allq)
    if not arms:
        out.write("no *.res.json in %s\n" % d); return 1
    qs = report(arms, notes, gq, cq, mins, out)
    if csvp:
        write_csv(csvp, arms, qs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
