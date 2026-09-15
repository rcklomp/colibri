#!/usr/bin/env python3
"""Parse the Q9 step 0 sweep logs (b{0,1}_r{rows}_rep{N}.log, plus
dense_gpu1.log) into the V*(S) table. Run on the rig against ~/bench/q9s0_out
after q9_step0_verify_cost.sh and q9_step0_dense_gpu1.sh. See
tools/hot-expert/Q9-MTP-SPEC-2026-09-15.md "Step 0" for the formula.
"""
import re, sys, os, math, glob

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/bench/q9s0_out")

def parse_log(path):
    text = open(path, errors="replace").read()
    d = {}
    m = re.search(r"\[enc\] prompt tokens: (\d+)", text)
    d["prompt_tokens"] = int(m.group(1)) if m else None

    # sections: split on the [OPTIME] === ... === header lines
    sections = {}
    cur = None
    for line in text.splitlines():
        hm = re.match(r"\[OPTIME\] === (.+?): (\d+) forwards, ([\d.]+) ms/forward measured ===", line)
        if hm:
            cur = hm.group(1)
            sections[cur] = {"forwards": int(hm.group(2)), "step_ms": float(hm.group(3)), "rows": {}}
            continue
        if cur is not None:
            rm = re.match(r"\[OPTIME\]\s+([a-zA-Z0-9_-]+)\s+([\d.]+) ms/token\s+([\d.]+)%", line)
            if rm:
                sections[cur]["rows"][rm.group(1).strip()] = float(rm.group(2))
    d["sections"] = sections

    place = {}
    for pm in re.finditer(r"\[OPTIME\] placement\s+(total|decode):\s+gpu=(\d+)\s+cpu=(\d+)", text):
        place[pm.group(1)] = (int(pm.group(2)), int(pm.group(3)))
    d["placement"] = place
    return d

def dense_ms(section):
    """dn-proj + qsa-proj + lm-head, ms (section['forwards']==1 so 'ms/token'==ms total)."""
    r = section["rows"]
    return r.get("dn-proj", 0.0) + r.get("qsa-proj", 0.0) + r.get("lm-head", 0.0)

def main():
    logs = {}
    for p in glob.glob(os.path.join(OUT, "b*_rep*.log")):
        name = os.path.basename(p)[:-4]
        mo = re.match(r"b(\d)_r(\d+)_rep(\d)", name)
        if not mo: continue
        batch, rows, rep = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
        logs[(batch, rows, rep)] = parse_log(p)

    prompt_tokens = None
    for d in logs.values():
        if d["prompt_tokens"]:
            prompt_tokens = d["prompt_tokens"]; break
    print(f"prompt_tokens (S_total) = {prompt_tokens}")

    dense_gpu1 = None
    dg1_path = os.path.join(OUT, "dense_gpu1.log")
    if os.path.exists(dg1_path):
        dd = parse_log(dg1_path)
        dec = dd["sections"].get("decode only")
        if dec and dec["forwards"] == 1:
            dense_gpu1 = dense_ms(dec)
    print(f"dense_gpu(1) = {dense_gpu1}")
    print()

    rows_values = [1, 2, 4, 8, 16, 32]
    print(f"{'rows':>5} {'rep':>3} {'V_total_ms':>11} {'blocks':>7} {'V_cpu(S)':>10} "
          f"{'dense_cpu(S)':>13} {'cpu_exp':>8} {'gpu_exp':>8}")
    per_rows = {r: [] for r in rows_values}
    for rows in rows_values:
        for rep in (1, 2, 3):
            d = logs.get((1, rows, rep))
            if not d: continue
            sec = d["sections"].get("prefill only")
            if not sec: continue
            V_total = sec["step_ms"]
            blocks = math.ceil(prompt_tokens / rows)
            V_cpu = V_total / blocks
            dcpu = dense_ms(sec) / blocks
            g, c = d["placement"].get("total", (0, 0))
            per_rows[rows].append((V_cpu, dcpu))
            print(f"{rows:>5} {rep:>3} {V_total:>11.3f} {blocks:>7} {V_cpu:>10.4f} "
                  f"{dcpu:>13.4f} {c:>8} {g:>8}")

    print()
    print("leg0 (row-at-a-time control) expert counts, for the exact-match check:")
    for rep in (1, 2, 3):
        d = logs.get((0, 0, rep))
        if not d: continue
        g, c = d["placement"].get("total", (0, 0))
        print(f"  rep{rep}: cpu={c} gpu={g}")
    print("leg1 rows=1 expert counts (must match leg0 to the unit):")
    for rep in (1, 2, 3):
        d = logs.get((1, 1, rep))
        if not d: continue
        g, c = d["placement"].get("total", (0, 0))
        print(f"  rep{rep}: cpu={c} gpu={g}")

    if dense_gpu1 is None:
        print("\nV*(S) NOT COMPUTED: dense_gpu(1) missing (run q9_step0_dense_gpu1.sh)")
        return

    print()
    print(f"{'rows(S)':>7} {'V_cpu(S) median':>16} {'dense_cpu(S) median':>20} {'V*(S)':>10}")
    vstar = {}
    for rows in rows_values:
        vals = per_rows[rows]
        if not vals: continue
        vcpu_med = sorted(v[0] for v in vals)[len(vals)//2]
        dcpu_med = sorted(v[1] for v in vals)[len(vals)//2]
        vs = vcpu_med - dcpu_med + dense_gpu1 * (1 + 0.1 * (rows - 1))
        vstar[rows] = vs
        print(f"{rows:>7} {vcpu_med:>16.4f} {dcpu_med:>20.4f} {vs:>10.4f}")

    if 1 in vstar and 4 in vstar:
        ratio = vstar[4] / vstar[1]
        print()
        print(f"V*(1) = {vstar[1]:.4f} ms")
        print(f"V*(4) = {vstar[4]:.4f} ms")
        print(f"V*(4)/V*(1) = {ratio:.4f}")
        print("GATE: " + ("DEAD (>= 2.4)" if ratio >= 2.4 else "PASS -- proceed to step 1"))

if __name__ == "__main__":
    main()
