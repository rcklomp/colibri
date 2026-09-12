#!/usr/bin/env python3
"""Compare the EDGE0 #1 probe's runs: greedy text, teacher_forcing, last_logits.

Same oracles and the same parser as g15_compare.py (CLAUDE.md names them for
this engine); a different pair list, because the variable under test is the TIER
CONFIGURATION rather than a numerics knob.

It also prints the mediating variable for each run -- the fraction of routed
expert evaluations that took the UNCLAMPED GPU kernel, from the engine's own
`[PROF] eg=...(experts=N) cpu=...(n=K)` line -- and checks the two
pre-committed bounds:

  * the null control (sA1 vs sA2) must be bit-identical, or the probe is void;
  * the A-vs-B gap must not exceed §G15's extreme (6/42, 8/1232, cos 0.992324 /
    0.981143), because A-vs-B moves a strict subset of the activations across
    the clamp boundary that §G15's EXPERTS_CPU knob moved. Exceeding it means
    the clamp is not the whole mechanism and the result is a contradiction to
    report, not a number to tune.
"""
import sys, os, math, re

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench/edge0_probe_out")

STATS = ("decode ", "[MAP] ", "experts hits ", "vision_tokens ", "greedy")


def parse(tag):
    p = os.path.join(OUT, tag + ".out")
    if not os.path.exists(p):
        return None
    tf, lg, text = None, None, []
    for line in open(p, errors="replace"):
        if line.startswith("teacher_forcing"):
            tf = line.split()[1:]
        elif line.startswith("last_logits"):
            lg = [float(v) for v in line.split()[1:]]
        elif any(line.startswith(s) for s in STATS):
            continue
        else:
            text.append(line)
    return dict(tag=tag, tf=tf, lg=lg, text="".join(text).strip())


def prof(tag):
    """(gpu_experts, cpu_experts, unclamped_fraction, tier_lines) from the .err"""
    p = os.path.join(OUT, tag + ".err")
    if not os.path.exists(p):
        return None
    eg = cpu = None
    tier = []
    for line in open(p, errors="replace"):
        m = re.match(r"\[PROF\] eg=[^(]+\(disp=\d+ experts=(\d+)\) cpu=[^(]+\(n=(\d+)\)", line)
        if m:
            eg, cpu = int(m.group(1)), int(m.group(2))
        if line.startswith("[VK] preload"):
            tier.append(line.strip())
    if eg is None:
        return None
    tot = eg + cpu
    return dict(eg=eg, cpu=cpu, frac=(eg / tot if tot else float("nan")), tier=tier)


def logit_diff(a, b):
    n = min(len(a), len(b))
    dot = sa = sb = 0.0
    mx = 0.0
    for i in range(n):
        dot += a[i] * b[i]; sa += a[i] * a[i]; sb += b[i] * b[i]
        d = abs(a[i] - b[i])
        if d > mx:
            mx = d
    num = sum((a[i] - b[i]) ** 2 for i in range(n))
    cos = dot / math.sqrt(sa * sb) if sa and sb else float("nan")
    rel = math.sqrt(num / sb) if sb else float("nan")
    ia = max(range(n), key=lambda i: a[i])
    ib = max(range(n), key=lambda i: b[i])
    return cos, mx, rel, ia, ib


def tf_stats(a, b):
    if a is None or b is None:
        return None, None
    n = min(len(a), len(b))
    bad = sum(1 for i in range(n) if a[i] != b[i])
    return bad, n


def compare(ref, cand):
    """returns (tf_bad, tf_n, cos) or None"""
    r, c = parse(ref), parse(cand)
    if r is None or c is None:
        print("  %-10s vs %-10s : MISSING" % (cand, ref))
        return None
    print("  %s  vs  %s" % (cand, ref))
    bad, n = tf_stats(r["tf"], c["tf"])
    if bad is None:
        print("     teacher_forcing : n/a")
    elif bad == 0:
        print("     teacher_forcing : identical (%d positions)" % n)
    else:
        first = next(i for i in range(n) if r["tf"][i] != c["tf"][i])
        print("     teacher_forcing : %d of %d differ, first at %d" % (bad, n, first))
    cos = None
    if r["lg"] and c["lg"]:
        cos, mx, rel, ia, ib = logit_diff(c["lg"], r["lg"])
        print("     last_logits     : cos %.9f  relL2 %.4g  maxabs %.4g  argmax %d vs %d %s"
              % (cos, rel, mx, ia, ib, "SAME" if ia == ib else "DIFF"))
    if not r["text"] and not c["text"]:
        print("     greedy text     : n/a (--greedy 0, prefill only)")
    else:
        same = r["text"] == c["text"]
        print("     greedy text     : %s" % ("IDENTICAL" if same else "*** DIFFERS ***"))
        if not same:
            rl, cl = r["text"], c["text"]
            k = 0
            while k < min(len(rl), len(cl)) and rl[k] == cl[k]:
                k += 1
            print("        first divergence at char %d" % k)
            print("        ref : ...%s" % rl[max(0, k - 60):k + 90].replace("\n", " "))
            print("        cand: ...%s" % cl[max(0, k - 60):k + 90].replace("\n", " "))
    return (bad, n, cos)


print("=== what each configuration actually placed, and how much ran UNCLAMPED ===")
print("  %-8s %10s %10s %10s   %s" % ("run", "gpu(uncl)", "cpu(clamp)", "uncl.frac", "tier"))
for tag in ("sA1", "sA2", "sB1", "sB2", "sC", "lA", "lB", "lC", "sA_nc", "sB_nc"):
    p = prof(tag)
    if p is None:
        continue
    print("  %-8s %10d %10d %9.2f%%   %s" % (tag, p["eg"], p["cpu"], 100 * p["frac"],
                                             " | ".join(x.replace("[VK] preload", "") for x in p["tier"])))

PAIRS = [
    ("sA1", "sA2", "NULL CONTROL: same config twice. Must be bit-identical or the probe is void"),
    ("sB1", "sB2", "null control for config B"),
    ("sA1", "sB1", "HEADLINE short: tier 1695/1695 vs dev2/dev3 off, same binary, same weights"),
    ("sA2", "sB2", "HEADLINE short, repeat"),
    ("sA1", "sC",  "midpoint short: half tier"),
    ("lA",  "lB",  "HEADLINE long (1232 positions): 1695/1695 vs dev2/dev3 off"),
    ("lA",  "lC",  "midpoint long: half tier"),
    ("sA_nc", "sB_nc", "FALSIFICATION ARM: EXPERTS_CPU=2 on both -- tier not consulted, "
                       "so a difference here would be a residency effect that is NOT the clamp"),
]
res = {}
for ref, cand, why in PAIRS:
    if parse(ref) is None or parse(cand) is None:
        continue
    print("\n-- %s" % why)
    res[(ref, cand)] = compare(ref, cand)

print("\n=== pre-committed checks ===")
null = res.get(("sA1", "sA2"))
if null:
    ok = null[0] == 0 and null[2] is not None and null[2] > 1 - 1e-12
    print("  null control bit-identical : %s" % ("YES" if ok else "*** NO -- PROBE IS VOID ***"))
for key, label, g15_tf, g15_n, g15_cos in (
        (("sA1", "sB1"), "short A-vs-B", 6, 42, 0.992324),
        (("lA", "lB"), "long A-vs-B", 8, 1232, 0.981143)):
    r = res.get(key)
    if not r or r[0] is None:
        continue
    bad, n, cos = r
    over = bad > g15_tf or (cos is not None and cos < g15_cos)
    print("  %-13s : %d of %d TF, cos %s  | §G15 extreme %d of %d, cos %.6f -> %s"
          % (label, bad, n, ("%.6f" % cos) if cos is not None else "n/a", g15_tf, g15_n, g15_cos,
             "*** EXCEEDS THE BOUND - CONTRADICTION, REPORT IT ***" if over else "within bound"))
fals = res.get(("sA_nc", "sB_nc"))
if fals:
    bad, n, cos = fals
    clean = bad == 0 and (cos is None or cos > 1 - 1e-12)
    print("  falsification arm          : %s"
          % ("bit-identical -- the tier config has NO effect once both paths compute the same "
             "function; the gap is the clamp and nothing else (REJECT edge0 #1)"
             if clean else
             "*** DIFFERS -- a residency effect beyond the clamp exists, edge0 #1 gets a second look ***"))
print()
