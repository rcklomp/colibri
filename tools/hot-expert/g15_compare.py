#!/usr/bin/env python3
"""Compare the G15 probe's runs: greedy text, teacher_forcing, last_logits.

Reads the *.out files g15_probe.sh writes and prints one table. The oracles are
the ones CLAUDE.md names for this engine: greedy output text identical,
teacher_forcing identical, and a cosine / max-abs / argmax diff on the last
token's logits.
"""
import sys, os, math

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench/g15_probe_out")


def parse(tag):
    p = os.path.join(OUT, tag + ".out")
    if not os.path.exists(p):
        return None
    tf, lg, text = None, None, []
    # The engine prints its own timing and mmap counters on stdout after the
    # generated text. Those lines are NOT model output and a run that is a
    # tenth of a second slower must not read as "the text changed" -- an
    # earlier version of this script reported exactly that against a candidate
    # whose logits were bit-identical to the pristine.
    STATS = ("decode ", "[MAP] ", "experts hits ", "vision_tokens ", "greedy")
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


def tf_diff(a, b):
    if a is None or b is None:
        return "n/a"
    if a == b:
        return "identical (%d positions)" % len(a)
    n = min(len(a), len(b))
    bad = sum(1 for i in range(n) if a[i] != b[i])
    first = next(i for i in range(n) if a[i] != b[i])
    return "%d of %d differ, first at %d" % (bad, n, first)


def compare(ref, cand):
    r, c = parse(ref), parse(cand)
    if r is None or c is None:
        print("  %-14s vs %-14s : MISSING" % (cand, ref))
        return
    print("  %s  vs  %s" % (cand, ref))
    print("     teacher_forcing : %s" % tf_diff(r["tf"], c["tf"]))
    if r["lg"] and c["lg"]:
        cos, mx, rel, ia, ib = logit_diff(c["lg"], r["lg"])
        print("     last_logits     : cos %.9f  relL2 %.4g  maxabs %.4g  argmax %d vs %d %s"
              % (cos, rel, mx, ia, ib, "SAME" if ia == ib else "DIFF"))
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


pairs = [
    ("s1_pristine", "s2_cand_off", "STANDING BAR: knobs off must be identical to pristine"),
    ("s1_pristine", "s3b_cpu_noclamp", "control B: placement only, CPU swiglu left UNCLAMPED like the GPU kernel"),
    ("s1_pristine", "s3_cpu_int4", "control: placement AND the clamp (see G13) -- not placement alone"),
    ("s3b_cpu_noclamp", "s3_cpu_int4", "what the clamp alone is worth"),
    ("s3_cpu_int4", "s4_cpu_int3", "THE PROBE: int3 against the same placement"),
    ("s1_pristine", "s4_cpu_int3", "end to end: int3 against what the model actually says"),
    ("l1_pristine", "l3b_cpu_noclamp", "long prompt, control B (unclamped, placement only)"),
    ("l1_pristine", "l3_cpu_int4", "long prompt, control (placement + clamp)"),
    ("l3b_cpu_noclamp", "l4_cpu_int3", "long prompt, int3 against the unclamped control"),
    ("l3_cpu_int4", "l4_cpu_int3", "long prompt, THE PROBE"),
    ("l1_pristine", "l4_cpu_int3", "long prompt, end to end"),
]
for ref, cand, why in pairs:
    if parse(ref) is None or parse(cand) is None:
        continue
    print("\n-- %s" % why)
    compare(ref, cand)
print()
