#!/bin/bash
# P0 — the executable gate for every item on PREFILL-ROADMAP-2026-09.md.
#
#   prefill_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# An item is DONE when this exits 0 and its printed deltas are in the commit
# body. Nothing else counts. It runs, in order:
#
#  (a) oracle, 600-token prompt: teacher-forcing argmax IDENTICAL at every
#      position, candidate vs pristine (prefill_profile.sh on both -- so the
#      per-bucket table comes for free and shows WHICH bucket moved);
#  (b) oracle, last-position logits on the same prompt: cosine >= 1-1e-4,
#      argmax equal; max|diff| printed (a knob that changes numerics needs
#      this number in its commit body);
#  (c) TTFT on the serve path: ttft_serve.py --engine on both binaries, sizes
#      30/300/1000, twice each; deltas printed per size.
#
# Exit 1 on an oracle failure, 2 on a harness refusal (residency, a running
# engine). The TTFT deltas are never a pass/fail here -- the roadmap states the
# required delta per item; the gate makes the number honest and reproducible.
#
# Budget today (3.3 tok/s prefill): ~40 min. It shrinks as items land.
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT=$HOME/bench/prefill_prompt_600.txt
OUT=$HOME/bench/prefill_gate_$TAG
mkdir -p "$OUT"

if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi

# Every step here spawns an engine that maps 180 GiB; unmapping it takes seconds to a minute,
# and the NEXT step refuses while it lives ("one engine at a time"). Without this wait the
# candidate's oracle run never starts and the gate compares a full file against an empty one
# -- which is exactly how the P8 gate failed on 2026-09-09 (`teacher_forcing: NO DATA
# (pristine 782 words, candidate 0)`). CLAUDE.md states the rule; the shared gate did not
# honour it. Same shape as wait_no_engine() in p7_gate.sh.
wait_no_engine() {
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "an engine is still alive after 240 s (pids $(pgrep -x glm53 | tr "\n" " ")) -- killing it"
  pkill -9 -x glm53; sleep 3
  pgrep -x glm53 >/dev/null && { echo "REFUSED: cannot clear the engine"; return 1; }
  return 0
}
if [ ! -s "$PROMPT" ]; then
  # deterministic ~600-token prompt: the head of the measurement record
  python3 - "$HERE/ROME-3x7900XTX-2026-09-04.md" > "$PROMPT" <<'PY'
import sys; t=open(sys.argv[1]).read(); print("Summarise these notes in one sentence.\n\n"+t[:2100])
PY
fi

echo "=== prefill_gate $TAG: pristine=$PRISTINE candidate=$CAND $(date -Is)"
# PRISTINE_SHADERS=<dir>: the pristine side loads its .spv files from there
# (a shader-only change would otherwise compare a binary against itself,
# both loading c/shaders). Copy c/shaders before rebuilding to make it.
shaders_for() { if [ "$1" = pristine ] && [ -n "${PRISTINE_SHADERS:-}" ]; then echo "$PRISTINE_SHADERS"; else echo "$HERE/../../c/shaders"; fi; }
for side in pristine candidate; do
  bin=$PRISTINE; [ $side = candidate ] && bin=$CAND
  export COLI_VK_SHADERS="$(shaders_for $side)"
  echo "--- (a)(b) oracle run: $side (shaders: $COLI_VK_SHADERS)"
  # the oracle half runs the CPU recurrence (COLI_KDA_GPU=0, the pristine
  # numerics) unless ORACLE_KDA_GPU says otherwise; the TTFT half below runs
  # the serving default (ttft_serve.py's engine_env: COLI_KDA_GPU=2).
  COLI_KDA_GPU=${ORACLE_KDA_GPU:-0} "$HERE/prefill_profile.sh" "$bin" "$PROMPT" "$TAG-$side" | tee "$OUT/profile_$side.txt" | grep -v "^teacher_forcing\|^last_logits"
  grep "^teacher_forcing" "$HOME/bench/prefill_profile_$TAG-$side.log" > "$OUT/tf_$side.txt" 2>/dev/null
  grep "^last_logits" "$HOME/bench/prefill_profile_$TAG-$side.log" > "$OUT/logits_$side.txt" 2>/dev/null
  wait_no_engine || exit 2
done

echo "--- (a) teacher forcing"
# An empty comparison is not a pass. Both oracle runs aborted on residency during the P8 gate
# (2026-09-09) and this printed "IDENTICAL (0 positions)" with TF=0 -- a gate that passes on
# nothing. The oracle needs a plausible number of positions before its verdict means anything.
TF_N=$(wc -w < "$OUT/tf_pristine.txt"); TF_M=$(wc -w < "$OUT/tf_candidate.txt")
if [ "${TF_N:-0}" -lt 100 ] || [ "${TF_M:-0}" -lt 100 ]; then
  echo "teacher_forcing: NO DATA (pristine $TF_N words, candidate $TF_M) -- the oracle run did not produce a comparison"
  TF=1
elif cmp -s "$OUT/tf_pristine.txt" "$OUT/tf_candidate.txt"; then
  echo "teacher_forcing: IDENTICAL ($(wc -w < "$OUT/tf_pristine.txt") positions)"; TF=0
else
  python3 - "$OUT/tf_pristine.txt" "$OUT/tf_candidate.txt" <<'PY'
import sys
a=open(sys.argv[1]).read().split()[1:]; b=open(sys.argv[2]).read().split()[1:]
bad=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
print(f"teacher_forcing: {len(bad)} of {len(a)} positions DIFFER; first at {bad[:10]}")
PY
  TF=1
fi

echo "--- (b) last-position logits"
python3 - "$OUT/logits_pristine.txt" "$OUT/logits_candidate.txt" <<'PY' && LG=0 || LG=1
import sys, math
a=[float(x) for x in open(sys.argv[1]).read().split()[1:]]
b=[float(x) for x in open(sys.argv[2]).read().split()[1:]]
assert len(a)==len(b) and a, "logit vectors differ in length"
dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
cos=dot/(na*nb); mx=max(abs(x-y) for x,y in zip(a,b))
ia=max(range(len(a)),key=a.__getitem__); ib=max(range(len(b)),key=b.__getitem__)
print(f"logits: cosine={cos:.7f} max_abs={mx:.4g} argmax {ia} vs {ib} {'OK' if ia==ib else 'DIFFERS'}")
sys.exit(0 if (cos>=1-1e-4 and ia==ib) else 1)
PY

echo "--- (c) serve-path TTFT (ttft_serve.py --engine), twice per size"
for side in pristine candidate; do
  bin=$PRISTINE; [ $side = candidate ] && bin=$CAND
  export COLI_VK_SHADERS="$(shaders_for $side)"
  wait_no_engine || exit 2
  python3 "$HERE/ttft_serve.py" --engine "$bin" --sizes 30,300,1000 --repeat 2 --warm \
      --tag "$TAG-$side" --json "$OUT/ttft.jsonl" | tee "$OUT/ttft_$side.txt" | grep -v "^\[resid"
  wait_no_engine || exit 2
done
# MIN_SPEEDUP (default 1.0): the candidate's median TTFT speedup at EVERY size
# must reach it and its decode sanity window must stay within 10% of the
# pristine's, or the gate exits 3. p2c (2026-09-06) passed the oracle and was
# 0.65x on the serve path; a chain that only checked the oracle served it.
python3 - "$OUT/ttft.jsonl" "$TAG" "${MIN_SPEEDUP:-1.0}" <<'PY' && SP=0 || SP=3
import json, sys, statistics as st
recs=[json.loads(l) for l in open(sys.argv[1])]
tag=sys.argv[2]; need=float(sys.argv[3]); ok=True
def rows(side): return [r for r in recs if r["tag"]==f"{tag}-{side}" and r["kind"]=="ttft" and r["ttft_s"]]
def fmt(rs): return " / ".join("%.1fs" % r["ttft_s"] for r in rs)
print("%6s %28s %28s %8s %12s" % ("size", "pristine ttft (both runs)", "candidate ttft (both runs)", "speedup", "decode p/c"))
for size in sorted({r["target"] for r in rows("pristine")}):
    p=[r for r in rows("pristine") if r["target"]==size]
    c=[r for r in rows("candidate") if r["target"]==size]
    if p and c:
        sp=st.median(x["ttft_s"] for x in p)/st.median(x["ttft_s"] for x in c)
        dp=st.median(x.get("decode_tps") or 0 for x in p); dc=st.median(x.get("decode_tps") or 0 for x in c)
        print("%6d %28s %28s %7.2fx %5.1f/%-5.1f" % (size, fmt(p), fmt(c), sp, dp, dc))
        if sp < need or (dp > 0 and dc < 0.9 * dp): ok=False
    else: ok=False
print("speedup gate:", "PASS" if ok else "FAIL (need >= %gx at every size and decode within 10%%)" % need)
sys.exit(0 if ok else 1)
PY

echo "=== prefill_gate $TAG: teacher_forcing $([ $TF = 0 ] && echo PASS || echo FAIL), logits $([ $LG = 0 ] && echo PASS || echo FAIL), speedup $([ $SP = 0 ] && echo PASS || echo FAIL); outputs in $OUT"
[ $TF = 0 ] && [ $LG = 0 ] || exit 1
exit $SP
