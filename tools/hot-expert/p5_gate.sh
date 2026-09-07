#!/bin/bash
# P5 — the executable gate for "the sequential remainder".
#
#   p5_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, tworeq).
# Exit 2 = a harness refusal (a running engine, a missing fixture or shader set).
# Exit 3 = a speed check failed.
#
# P5 has three sub-items behind three knobs, and they are gated separately
# because they touch different things:
#
#   P5.1  COLI_MLA_HEADVEC  the MLA attention score pass, CPU, bit-identical
#   P5.2  COLI_KDA_ROWS     the chunk's S recurrence steps in ONE submit,
#                           GPU, only alive at COLI_KDA_GPU != 0
#   P5.3  COLI_ROUTER_LANES the router dot, CPU, bit-identical
#
# Steps:
#
#  (a)  prefill_gate.sh at the DEFAULT oracle knob (COLI_KDA_GPU=0). That knob
#       is the CPU recurrence, so P5.2 is not even reached here: (a) is the
#       oracle for P5.1 and P5.3, and bit-identical is the requirement.
#  (a2) prefill_gate.sh with ORACLE_KDA_GPU=2 — the SERVING knob, where P5.2
#       actually runs. Without this step P5.2 has no oracle at all: the GPU
#       recurrence never executes at COLI_KDA_GPU=0 and (a) would pass a
#       broken batched submit. Bit-identical is still the requirement, because
#       the batched form is the same shader in the same order.
#  (b)  prefill_profile.sh at the serving knob on the 600- and 3 000-token
#       prompts, all knobs off vs all knobs on in the SAME binary (plus the
#       pristine binary as the honest baseline), for the per-bucket table.
#       Run by the chain, not here — it is a diagnostic, not a pass/fail.
#  (c)  tworeq.py at TWOREQ_SLOTS=4, COLI_KDA_GPU=2 and =0: IDENTICAL and no
#       "forcing" line. The batched recurrence writes the same per-slot device
#       state; this is the check that says so across sessions.
#  (d)  ttft_serve.py --prefix-ckpt with the 4-tool fixture: the P7 capture
#       still sees the post-chunk state after the batched submit.
#  (e)  the live rows through the gateway — the chain runs them after serving.
#
# PRISTINE_SHADERS is MANDATORY here and the script refuses without it:
# kda_step.comp gained a push constant, so the pristine binary pushes 20 bytes
# into a 24-byte block if it loads the candidate's .spv and `tok` is garbage.
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p5gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p5_gate_$TAG
TOOLS=${GATE_TOOLS:-$HOME/bench/owui_tools_4.json}
SYSTEM=${GATE_SYSTEM:-$HOME/bench/p6_system.txt}
SLOTS=${P5_SLOTS:-4}
mkdir -p "$OUT" "$OUT/ckpt"

STEPS=${GATE_STEPS:-agcd}
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }

wait_no_engine() {
  for _ in $(seq 1 180); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 180 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}

if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
[ -x "$PRISTINE" ] || { echo "REFUSED: no pristine binary at $PRISTINE"; exit 2; }
[ -x "$CAND" ]     || { echo "REFUSED: no candidate binary at $CAND"; exit 2; }
for f in "$TOOLS" "$SYSTEM"; do [ -s "$f" ] || { echo "REFUSED: missing fixture $f"; exit 2; }; done
: "${PRISTINE_SHADERS:?REFUSED: P5 changes kda_step.comp -- set PRISTINE_SHADERS to the pre-P5 c/shaders copy}"
[ -f "$PRISTINE_SHADERS/kda_step.spv" ] || { echo "REFUSED: no kda_step.spv in $PRISTINE_SHADERS"; exit 2; }
export PRISTINE_SHADERS
SHADERS=$HERE/../../c/shaders
[ -f "$SHADERS/kda_step.spv" ] || { echo "REFUSED: no built shaders at $SHADERS"; exit 2; }
if cmp -s "$PRISTINE_SHADERS/kda_step.spv" "$SHADERS/kda_step.spv"; then
  echo "REFUSED: PRISTINE_SHADERS is the same kda_step.spv as the candidate's -- the snapshot was taken after the build"; exit 2
fi

cp -f "$HOME/.glm53_explain.bin" "$OUT/hist.bin" 2>/dev/null || true
export COLI_CKPT_DIR="$OUT/ckpt"

echo "=== p5_gate $TAG $(date -Is)"
echo "    pristine=$PRISTINE ($(sha256sum "$PRISTINE" | cut -c1-16)) shaders=$PRISTINE_SHADERS"
echo "    candidate=$CAND ($(sha256sum "$CAND" | cut -c1-16)) shaders=$SHADERS"
echo "    steps=$STEPS slots=$SLOTS tools=$TOOLS"

rca=0; rca2=0; rcc=0; rcd=0

# ---------------------------------------------------------------- (a)
echo
echo "### step a — prefill_gate at the oracle knob (COLI_KDA_GPU=0): P5.1 + P5.3, bit-identical"
if run_step a; then
GLM53_PREFIX_CKPT=0 MIN_SPEEDUP=${MIN_SPEEDUP:-1.0} \
  bash "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-cpu" 2>&1 | tee "$OUT/step_a.txt"
rca=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step a rc=$rca"

# ---------------------------------------------------------------- (a2)
echo
echo "### step a2 — prefill_gate at the SERVING knob (ORACLE_KDA_GPU=2): P5.2's only oracle"
if run_step g; then
GLM53_PREFIX_CKPT=0 ORACLE_KDA_GPU=2 MIN_SPEEDUP=${MIN_SPEEDUP:-1.0} \
  bash "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-gpu" 2>&1 | tee "$OUT/step_a2.txt"
rca2=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step a2 rc=$rca2"

# ---------------------------------------------------------------- (c)
echo
echo "### step c — tworeq at $SLOTS slots, both KDA knobs, no forcing line"
if run_step c; then
for knob in 2 0; do
  echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=$SLOTS"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
      COLI_CKPT_DIR="$OUT/ckpt" GLM53_PREFIX_CKPT=0 \
      COLI_KDA_GPU=$knob TWOREQ_SLOTS=$SLOTS TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
      python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
  r=$?
  forced=$(grep -c "forcing COLI_KDA_GPU=0" "$OUT/tworeq_kda$knob.txt")
  echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
  echo "    forcing lines: $forced"
  [ "$r" = 0 ] || rcc=1
  if [ "$knob" = 2 ] && [ "$forced" != 0 ]; then
    echo "    FAIL: the recurrence fell back to the CPU -- this run says nothing about P5.2"; rcc=1
  fi
  wait_no_engine || exit 2
done
else echo "(skipped)"; fi
echo "step c rc=$rcc"

# ---------------------------------------------------------------- (d)
echo
echo "### step d — P7 capture/restore still works after the batched recurrence"
if run_step d; then
rm -f "$OUT/ckpt"/*.bin
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
    COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
    COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
    COLI_CKPT_DIR="$OUT/ckpt" COLI_KDA_GPU=2 GLM53_PREFIX_CKPT=1 GLM53_VERBOSE=1 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt \
      --tools "$TOOLS" --system "$SYSTEM" --kv-slots "$SLOTS" \
      --sizes "" --repeat 0 --warm --min-resident 96 \
      --tag "$TAG-ckpt" --json "$OUT/ckpt.jsonl" \
      --engine-log "$OUT/engine_ckpt.log" 2>&1 | tee "$OUT/step_d.txt"
rcd=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep -E "CKPT (store|hit)" "$OUT/engine_ckpt.log" || echo "(no CKPT line)"
grep -E "^prefix-ckpt:|^VERDICT " "$OUT/step_d.txt" | tail -6
if ! grep -q "^VERDICT prefix-ckpt: PASS" "$OUT/step_d.txt"; then rcd=1; fi
else echo "(skipped)"; fi
echo "step d rc=$rcd"

# ---------------------------------------------------------------- verdict
echo
echo "=== p5_gate $TAG verdict $(date -Is)"
printf "  step a  oracle+TTFT at COLI_KDA_GPU=0 (P5.1, P5.3)   rc=%s\n" "$rca"
printf "  step a2 oracle+TTFT at COLI_KDA_GPU=2 (P5.2)         rc=%s\n" "$rca2"
printf "  step c  tworeq %d slots, both knobs                   rc=%s\n" "$SLOTS" "$rcc"
printf "  step d  prefix checkpoint capture/restore            rc=%s\n" "$rcd"
echo "  outputs in $OUT"
if [ "$rca" = 2 ] || [ "$rca2" = 2 ] || [ "$rcc" = 2 ] || [ "$rcd" = 2 ]; then exit 2; fi
if [ "$rca" = 1 ] || [ "$rca2" = 1 ] || [ "$rcc" = 1 ] || [ "$rcd" = 1 ]; then exit 1; fi
if [ "$rca" != 0 ] || [ "$rca2" != 0 ] || [ "$rcc" != 0 ] || [ "$rcd" != 0 ]; then exit 3; fi
exit 0
