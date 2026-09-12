#!/usr/bin/env python3
"""qp_compare.py -- the three oracles for roadmap item QP, on qp_probe.sh's output.

For a pair of runs (A = control, B = candidate arm) it reports:

  teacher_forcing  how many of the N teacher-forced argmax predictions differ.
                   This is THE oracle for a format question: §G15's methodological
                   note is that greedy text alone is ambiguous on a chaotic prompt
                   (pure float reassociation already moved GLM's 128-token
                   continuation) while N independent teacher-forced predictions
                   separate 0 from 16 cleanly.
  greedy           the 128-token continuation, identical or not.
  last_logits      cosine, relL2, max-abs and argmax of the last prompt
                   position's full logit row.

QP's kill line, taken from §G14/§G15's rejection class and NOT softened here:
any short-prompt teacher_forcing change at all, or any long-prompt cosine below
0.99, is a rejection. The script prints VERDICT lines that apply it literally,
with no rounding.

Usage:
    qp_compare.py <dir> <ctrlTag> <candTag> [label]
    qp_compare.py <dir> --table            # every pair QP's gate calls for
"""
import math
import os
import struct
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench/qp_probe_out")


def tf_line(tag):
    p = os.path.join(D, f"{tag}.log")
    if not os.path.exists(p):
        return None
    for line in open(p, errors="replace"):
        if line.startswith("teacher_forcing"):
            return [int(x) for x in line.split()[1:]]
    return None


def greedy_text(tag):
    p = os.path.join(D, f"{tag}.log")
    if not os.path.exists(p):
        return None
    got = None
    for line in open(p, errors="replace"):
        if line.startswith("Text      :") or line.startswith("Ids       :"):
            got = line.rstrip("\n")
    return got


def logits(tag, which="tf"):
    p = os.path.join(D, f"{tag}.{which}.f32")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    raw = open(p, "rb").read()
    return struct.unpack(f"<{len(raw)//4}f", raw)


def cmp_logits(a, b):
    if a is None or b is None:
        return "MISSING"
    if len(a) != len(b):
        return f"LENGTH {len(a)} vs {len(b)}"
    dot = na = nb = 0.0
    num = den = mx = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
        d = x - y
        num += d * d; den += y * y
        if abs(d) > mx:
            mx = abs(d)
    cos = dot / math.sqrt(na * nb) if na > 0 and nb > 0 else float("nan")
    rel = math.sqrt(num / den) if den > 0 else float("nan")
    aa = max(range(len(a)), key=lambda i: a[i])
    ab = max(range(len(b)), key=lambda i: b[i])
    return dict(cos=cos, rel=rel, maxabs=mx, argmax_a=aa, argmax_b=ab)


def pair(ctrl, cand, label="", long_prompt=False):
    ta, tb = tf_line(ctrl), tf_line(cand)
    if ta is None or tb is None:
        tf = f"MISSING ({ctrl}={ta is not None}, {cand}={tb is not None})"
        ndiff, n = None, None
    elif len(ta) != len(tb):
        tf = f"LENGTH {len(ta)} vs {len(tb)}"
        ndiff, n = None, None
    else:
        ndiff = sum(1 for x, y in zip(ta, tb) if x != y)
        n = len(ta)
        tf = f"{ndiff} of {n} differ" if ndiff else f"identical {n}/{n}"
    ga, gb = greedy_text(ctrl), greedy_text(cand)
    if ga is None or gb is None:
        gr = "MISSING"
    else:
        gr = "identical" if ga == gb else "DIFFERS"
    # Byte equality first: for the knobs-off row the gate is IDENTICAL, and a
    # cosine of 1.000000000 is not the same claim as "the same bytes".
    pa = os.path.join(D, f"{ctrl}.last.f32")
    pb = os.path.join(D, f"{cand}.last.f32")
    if os.path.exists(pa) and os.path.exists(pb):
        same = open(pa, "rb").read() == open(pb, "rb").read()
        byteeq = f"last-generated logits {'BIT-IDENTICAL' if same else 'differ'} " \
                 f"({os.path.getsize(pa)} bytes)"
    else:
        byteeq = "last-generated logits MISSING"
    lg = cmp_logits(logits(cand, "tf"), logits(ctrl, "tf"))
    lgl = cmp_logits(logits(cand, "last"), logits(ctrl, "last"))
    print(f"\n=== {cand} vs {ctrl}  {label}")
    print(f"  teacher_forcing : {tf}")
    print(f"  greedy 128      : {gr}")
    print(f"  bytes           : {byteeq}")
    for name, r in (("last prompt pos", lg), ("last generated ", lgl)):
        if isinstance(r, str):
            print(f"  {name} : {r}")
        else:
            print(f"  {name} : cos {r['cos']:.9f}  relL2 {r['rel']:.4e}  "
                  f"maxabs {r['maxabs']:.4e}  argmax {r['argmax_a']} vs {r['argmax_b']}"
                  f"{'' if r['argmax_a'] == r['argmax_b'] else '  ARGMAX MOVED'}")
    # the kill line, applied literally
    if ndiff is not None:
        if not long_prompt and ndiff > 0:
            print(f"  VERDICT: REJECTED -- short-prompt teacher_forcing changed "
                  f"({ndiff} of {n}); QP's kill line is ANY change")
        elif long_prompt and isinstance(lg, dict) and lg["cos"] < 0.99:
            print(f"  VERDICT: REJECTED -- long-prompt cosine {lg['cos']:.6f} < 0.99")
        elif not long_prompt:
            print(f"  VERDICT: short-prompt teacher_forcing clean")
        else:
            print(f"  VERDICT: long-prompt cosine {lg['cos']:.6f} >= 0.99, "
                  f"{ndiff} of {n} teacher_forcing changes")
    return ndiff, lg


TABLE = [
    ("R2", "R1", "candidate knobs OFF vs pristine -- must be identical"),
    ("R3", "R1", "placement only: every routed expert on the CPU"),
    ("R4", "R3", "(a) int4-g64 experts, IDENTICALLY PLACED control"),
    ("R4", "R1", "(a) int4-g64 experts end to end"),
    ("R5", "R2", "(b) int8 dense, per-row"),
    ("R6", "R2", "(b) int8 dense, g64"),
    ("R7", "R2", "(c) int8 LM head, per-row"),
    ("R8", "R2", "(c) int8 LM head, g64"),
]

if len(sys.argv) > 2 and sys.argv[2] == "--table":
    for prefix, longp in (("s", False), ("l", True)):
        print(f"\n########## {'LONG' if longp else 'SHORT'} PROMPT ##########")
        for cand, ctrl, label in TABLE:
            if os.path.exists(os.path.join(D, f"{prefix}-{cand}.log")):
                pair(f"{prefix}-{ctrl}", f"{prefix}-{cand}", label, long_prompt=longp)
elif len(sys.argv) > 3:
    pair(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "",
         long_prompt=sys.argv[2].startswith("l-"))
else:
    print(__doc__)
    sys.exit(2)
