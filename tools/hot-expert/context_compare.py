#!/usr/bin/env python3
"""context_compare.py -- pair context_ladder.py jsonl rows across arms and let
gate_lib.sh decide, rather than reading the numbers by hand.

FRANKEN-ENGINE-PLAN-2026-09-15.md, item H1: H2's chain produces four runs in
the order A1, B1, B2, A2 (the GLM arm brackets the pair, MEASURING.md /
CLAUDE.md's "run arms interleaved" rule). This script does not know or care
which engine each label names -- it groups labels into "arms" by stripping the
trailing run number (A1, A2 -> arm A; B1, B2 -> arm B) and pairs rows BY TURN
INDEX, the same turn number in each run because the chain gives every run in
an arm the identical --steps/--followups sequence.

At a turn where exactly two arm groups are present, each with >=2 samples
(A1+A2, B1+B2), it prints:
  * the depth reached, in EACH label's own prompt_tokens AND in characters
    (cursor_chars -- byte-identical across arms by construction, unlike the
    token count, which depends on the tokenizer)
  * D_A x2, D_B x2 (decode tok/s) through gate_ab_verdict
  * for a followup-kind turn, TTFT_inc x2 (the incremental wait at that depth)
    through gate_ab_verdict, separately
gate_ab_verdict is sourced from gate_lib.sh, not reimplemented: it refuses a
verdict from a single unpaired sample and reports NO VERDICT on overlap
(CLAUDE.md "How a change is measured", the +20%/+11% incident).

Reuse ratio (FRANKEN-ENGINE-PLAN \xa72.4, Q2): for any label carrying
cold-sweep rows (kind=cold-sweep, HTTP mode only -- see context_ladder.py),
r = T_lad(d) / T_cold(d) where T_lad(d) is that SAME label's ladder-turn ttft_s
at the nearest depth to the sweep's target and T_cold(d) is the cold-sweep
row's own ttft_s. r <= 0.25: reuse. r >= 0.8: no reuse. Otherwise NO VERDICT --
this is a single-sample ratio per depth, not an A/B, so gate_ab_verdict does
not apply here; the thresholds themselves are the check.

Usage:
  context_compare.py --rows A1=ctxA1.jsonl B1=ctxB1.jsonl B2=ctxB2.jsonl A2=ctxA2.jsonl
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE_LIB = os.path.join(HERE, "gate_lib.sh")

ARM_RE = re.compile(r"^([A-Za-z]+)")


def arm_of(label):
    m = ARM_RE.match(label)
    return m.group(1) if m else label


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def gate_ab_verdict(label, a_vals, b_vals):
    """Runs gate_lib.sh's gate_ab_verdict over ssh-free bash, and returns its
    exit code: 0 separated, 1 NO VERDICT (overlap), 2 REFUSED (< 2 samples)."""
    if not os.path.isfile(GATE_LIB):
        sys.exit(f"REFUSED: {GATE_LIB} not found -- cannot apply gate_ab_verdict's rule "
                 f"by hand (that is exactly the failure mode gate_lib.sh exists to prevent)")
    script = f'set -u; source "{GATE_LIB}"; gate_ab_verdict "$1" "$2" "$3"'
    r = subprocess.run(["bash", "-c", script, "_", label,
                        " ".join(str(x) for x in a_vals), " ".join(str(x) for x in b_vals)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r.returncode


def depth_line(turn, by_label):
    parts = []
    for label, rec in sorted(by_label.items()):
        tok = rec.get("prompt_tokens")
        ch = rec.get("cursor_chars")
        parts.append(f"{label} depth={tok if tok is not None else '?'} tok"
                     f"/{ch if ch is not None else '?'} chars")
    print(f"turn {turn}: " + ", ".join(parts))


def compare_ladder(runs):
    """runs: {label: [rows]}. Pairs rows by turn index across arm groups."""
    groups = {}
    for label in runs:
        groups.setdefault(arm_of(label), []).append(label)

    by_turn = {}
    for label, rows in runs.items():
        for rec in rows:
            t = rec.get("turn")
            if t is None or rec.get("kind") not in ("ladder", "followup"):
                continue
            by_turn.setdefault(t, {})[label] = rec

    overall_rc = 0
    for turn in sorted(by_turn):
        by_label = by_turn[turn]
        depth_line(turn, by_label)

        arm_names = sorted(g for g in groups if any(l in by_label for l in groups[g]))
        if len(arm_names) != 2:
            print(f"  (turn {turn}: {len(arm_names)} arm group(s) present -- "
                  f"gate_ab_verdict needs exactly two, skipping)")
            continue
        ga, gb = arm_names

        def vals(field, kind_filter=None):
            out = {ga: [], gb: []}
            for label, rec in by_label.items():
                if kind_filter and rec.get("kind") != kind_filter:
                    continue
                v = rec.get(field)
                if v is not None:
                    out[arm_of(label)].append(v)
            return out[ga], out[gb]

        da, db = vals("decode_tps")
        if da or db:
            rc = gate_ab_verdict(f"decode@turn{turn} ({ga} vs {gb})", da, db)
            overall_rc = max(overall_rc, rc if rc != 2 else 2)

        ta, tb = vals("ttft_s", kind_filter="followup")
        if ta or tb:
            rc = gate_ab_verdict(f"ttft_inc@turn{turn} ({ga} vs {gb})", ta, tb)
            overall_rc = max(overall_rc, rc if rc != 2 else 2)
    return overall_rc


def reuse_ratio(runs):
    """Q2 (FRANKEN-ENGINE-PLAN \xa72.4): r = T_lad(d)/T_cold(d) per label per
    sweep depth. Not an A/B -- one label's own ladder against its own cold
    sweep -- so this applies the plan's fixed thresholds directly rather than
    gate_ab_verdict, which compares two independent arms."""
    any_sweep = False
    for label, rows in sorted(runs.items()):
        ladder = [r for r in rows if r.get("kind") == "ladder" and r.get("prompt_tokens") is not None]
        sweeps = [r for r in rows if r.get("kind") == "cold-sweep"]
        if not sweeps:
            continue
        any_sweep = True
        print(f"\nreuse ratio, {label}:")
        for sw in sweeps:
            d = sw.get("target_new")
            t_cold = sw.get("ttft_s")
            if not ladder or t_cold is None:
                print(f"  depth {d}: REFUSED -- no ladder rows or no cold ttft_s to compare")
                continue
            nearest = min(ladder, key=lambda r: abs(r["prompt_tokens"] - d))
            t_lad = nearest.get("ttft_s")
            if t_lad is None or t_cold == 0:
                print(f"  depth {d}: REFUSED -- missing ttft_s")
                continue
            r = t_lad / t_cold
            if r <= 0.25:
                verdict = "REUSE"
            elif r >= 0.8:
                verdict = "NO REUSE"
            else:
                verdict = "NO VERDICT"
            print(f"  depth {d:>6}: T_lad={t_lad:.2f}s (turn {nearest.get('turn')}, "
                  f"depth {nearest['prompt_tokens']})  T_cold={t_cold:.2f}s  "
                  f"r={r:.3f}  {verdict}")
    if not any_sweep:
        print("\nreuse ratio: no cold-sweep rows in any input file -- nothing to compute "
              "(--cold-sweep is HTTP mode only, see context_ladder.py)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True, metavar="LABEL=PATH",
                    help="one jsonl file per run, e.g. A1=a1.jsonl B1=b1.jsonl "
                         "B2=b2.jsonl A2=a2.jsonl. Arm group = label's leading "
                         "letters (A1/A2 -> A).")
    args = ap.parse_args()

    runs = {}
    for spec in args.rows:
        if "=" not in spec:
            sys.exit(f"REFUSED: --rows entries are LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        if not os.path.isfile(path):
            sys.exit(f"REFUSED: {path} (label {label}) does not exist")
        rows = load_rows(path)
        if not rows:
            sys.exit(f"REFUSED: {path} (label {label}) is empty -- an empty comparison "
                     f"is not a pass (gate_lib.sh's own rule, applied to the input here)")
        runs[label] = rows

    groups = {}
    for label in runs:
        groups.setdefault(arm_of(label), []).append(label)
    print("arms: " + ", ".join(f"{g}={sorted(ls)}" for g, ls in sorted(groups.items())))

    rc = compare_ladder(runs)
    reuse_ratio(runs)
    return rc


if __name__ == "__main__":
    sys.exit(main())
