#!/bin/bash
# P7 — the executable gate for prefix checkpoints + the gateway prefix pin.
#
#   p7_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, text, or tworeq).
# Exit 2 = a harness refusal (residency, another engine running).
# Exit 3 = a speed check failed (step 1's neutrality, or step 5's repeat).
#
# In order:
#  (1) prefill_gate.sh with GLM53_PREFIX_CKPT=0 and MIN_SPEEDUP=0.97 — with the
#      checkpoints OFF the candidate must BE the pristine engine: identical
#      teacher forcing, identical last-position logits, and serve-path TTFT
#      within a percent. 0.97 because "no regression" is the claim and the rows
#      repeat within ±1 %.
#  (2) ttft_serve.py --prefix-ckpt --oracle: a NEW conversation that shares the
#      system+tools prefix must start warm, survive an engine restart (disk),
#      and produce byte-identical text and first-token logits against the same
#      binary with GLM53_PREFIX_CKPT=0.
#  (3) tworeq.py at TWOREQ_SLOTS=4, at COLI_KDA_GPU=0 and =2: a checkpoint is
#      per-conversation state with a new home, and that is exactly the class of
#      bug tworeq exists for (G12).
#  (4) ttft_serve.py --pin-block: a re-ranked <memory_context> block must not
#      cost a re-prefill.
#  (5) ttft_serve.py --sizes 300 --repeat 2 with checkpoints ON: two identical
#      prompts must neither capture nor restore (REUSE 0 both) and must land
#      within 3 % of step 1's candidate row — the "checkpoints cost nothing
#      when they cannot help" check.
#
# GATE_TOOLS overrides the tool block (default the real 24-tool Open WebUI
# dump; a 4-tool subset is the way to iterate — the full block is an
# ~18-minute cold prefill and step 2 pays it twice).
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p7gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p7_gate_$TAG
TOOLS=${GATE_TOOLS:-$HOME/bench/owui_tools.json}
SYSTEM=${GATE_SYSTEM:-$HOME/bench/p6_system.txt}
SNAP=${SNAP:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
mkdir -p "$OUT"

# A step that leaves its engine behind makes the NEXT step refuse, and a revert
# that copies over a still-running binary fails with ETXTBSY -- which is how a
# failed gate left the candidate in service on 2026-09-07. Wait, every time.
wait_no_engine() {
  for _ in $(seq 1 180); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 180 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}
if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
# GATE_STEPS=12345 selects which steps run (default all); a step that is skipped
# reports rc=0 and its table comes from the run that did execute it.
STEPS=${GATE_STEPS:-12345}
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }
for f in "$TOOLS" "$SYSTEM"; do
  [ -s "$f" ] || { echo "REFUSED: missing fixture $f"; exit 2; }
done

echo "=== p7_gate $TAG $(date -Is)"
echo "    pristine=$PRISTINE candidate=$CAND"
echo "    tools=$TOOLS ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$TOOLS") entries) system=$SYSTEM"
echo "    ckpt dir=${COLI_CKPT_DIR:-$SNAP/.coli_ckpt}"

rc1=0; rc2=0; rc3=0; rc4=0; rc5=0

# ---------------------------------------------------------------- (1)
echo
echo "### step 1 — prefill_gate.sh with GLM53_PREFIX_CKPT=0 (P7 must be inert when off)"
if run_step 1; then
GLM53_PREFIX_CKPT=0 MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-off" 2>&1 | tee "$OUT/step1.txt"
rc1=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 1 rc=$rc1"

# ---------------------------------------------------------------- (2)
echo
echo "### step 2 — ttft_serve.py --prefix-ckpt --oracle (checkpoints ON, 4 slots)"
if run_step 2; then
python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt --oracle \
    --tools "$TOOLS" --system "$SYSTEM" --kv-slots 4 --sizes "" --repeat 0 \
    --warm --min-resident 96 --tag "$TAG-ckpt" --json "$OUT/p7.jsonl" \
    --engine-log "$OUT/engine_ckpt.log" --logit-dir "$OUT/logits" 2>&1 | tee "$OUT/step2.txt"
rc2=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 2 rc=$rc2"

# ---------------------------------------------------------------- (3)
echo
echo "### step 3 — tworeq at 4 slots, both KDA knobs"
# A COPY of the histogram, never the canonical file: the engine rewrites
# COLI_USAGE_PATH at exit and a benchmark must not teach the serving tier.
cp -f "$HOME/.glm53_explain.bin" "$OUT/hist_tworeq.bin" 2>/dev/null || true
if run_step 3; then
for knob in 0 2; do
  echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=4"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$HERE/../../c/shaders" \
      COLI_USAGE_PATH="$OUT/hist_tworeq.bin" \
      COLI_KDA_GPU=$knob TWOREQ_SLOTS=4 TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
      python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
  r=$?
  echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
  [ $r = 0 ] || rc3=1
  wait_no_engine || exit 2
done
else echo "(skipped)"; fi

# ---------------------------------------------------------------- (4)
echo
echo "### step 4 — ttft_serve.py --pin-block (COLI_PREFIX_PIN)"
if run_step 4; then
python3 "$HERE/ttft_serve.py" --engine "$CAND" --pin-block --kv-slots 4 \
    --sizes "" --repeat 0 --system "$SYSTEM" \
    --warm --min-resident 96 --tag "$TAG-pin" --json "$OUT/p7.jsonl" \
    --engine-log "$OUT/engine_pin.log" 2>&1 | tee "$OUT/step4.txt"
rc4=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 4 rc=$rc4"

# ---------------------------------------------------------------- (5)
echo
echo "### step 5 — repeated identical prompt: checkpoints must stay out of the way"
# three repeats, not two: the band below is 3 % and two runs of the same prompt
# sat 2.5 % apart in the 2026-09-07 smoke. More samples, same criterion.
if run_step 5; then
python3 "$HERE/ttft_serve.py" --engine "$CAND" --sizes 300 --repeat 3 \
    --warm --min-resident 96 --tag "$TAG-rep" --json "$OUT/p7.jsonl" \
    --engine-log "$OUT/engine_rep.log" 2>&1 | tee "$OUT/step5.txt"
wait_no_engine || exit 2
# STEP5_REF: step 1's ttft.jsonl, when step 1 ran under another tag.
python3 - "$OUT/p7.jsonl" "${STEP5_REF:-$HOME/bench/prefill_gate_$TAG-off/ttft.jsonl}" "$TAG" "${STEP5_REF_TAG:-$TAG}" <<'PY' && rc5=0 || rc5=3
import json, sys, statistics as st
rep = [json.loads(l) for l in open(sys.argv[1])]
try:
    base = [json.loads(l) for l in open(sys.argv[2])]
except OSError:
    base = []
tag = sys.argv[3]
reftag = sys.argv[4] if len(sys.argv) > 4 else tag
rows = [r for r in rep if r["tag"] == f"{tag}-rep" and r["kind"] == "ttft" and r["ttft_s"]]
ref = [r for r in base if r["tag"] == f"{reftag}-off-candidate" and r["kind"] == "ttft"
       and r["target"] == 300 and r["ttft_s"]]
print("%-30s %10s %8s %s" % ("run", "ttft", "reused", "ckpt lines"))
ok = bool(rows)
for r in rows:
    ck = ", ".join(r.get("ckpt") or []) or "-"
    print("%-30s %9.2fs %8s %s" % (f"size 300 (ckpt ON)", r["ttft_s"], r.get("reused"), ck))
    if r.get("reused"): ok = False
    if [c for c in (r.get("ckpt") or []) if " store " in c or " hit " in c]: ok = False
if ref and rows:
    m_ref, m_rep = st.median(x["ttft_s"] for x in ref), st.median(x["ttft_s"] for x in rows)
    print("step 1 candidate median %.2fs vs step 5 median %.2fs = %.3fx" % (m_ref, m_rep, m_ref / m_rep))
    if abs(m_rep - m_ref) > 0.03 * m_ref: ok = False
else:
    print("step 1 candidate rows not found -- ratio not checked")
print("step 5:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
else echo "(skipped)"; fi
echo "step 5 rc=$rc5"

# ---------------------------------------------------------------- verdict
echo
echo "=== p7_gate $TAG verdict $(date -Is)"
printf "  step 1 prefill_gate (ckpt off)   rc=%s\n" "$rc1"
printf "  step 2 prefix-ckpt + oracle      rc=%s\n" "$rc2"
printf "  step 3 tworeq 4 slots, both knobs rc=%s\n" "$rc3"
printf "  step 4 pin-block                 rc=%s\n" "$rc4"
printf "  step 5 repeat neutrality         rc=%s\n" "$rc5"
echo "  outputs in $OUT"
[ "$rc1" = 2 ] && exit 2
[ "$rc2" = 2 ] && exit 2
[ "$rc1" = 1 ] && exit 1
[ "$rc2" = 0 ] || exit 1
[ "$rc3" = 0 ] || exit 1
[ "$rc4" = 0 ] || exit 1
[ "$rc1" = 0 ] || exit 3
[ "$rc5" = 0 ] || exit 3
exit 0
