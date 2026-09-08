#!/bin/bash
# CANCEL — the executable gate for "glm53 honours CANCEL while a turn is in
# flight".
#
#   cancel_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle or a named check failed.
# Exit 2 = a harness refusal (a running engine, a missing fixture or shader set).
# Exit 3 = a speed check failed (the poll is not free).
#
# The item this gates was already shipped once, broken: G16 passed a `--greedy
# 64` CLI oracle that never executes serve_one, and the engine still ran a
# cancelled 1 230-token turn to the end (251.3 s, measured 2026-09-06). So every
# step here runs the SERVE path, and the two that decide are the ones a CLI run
# cannot reach: the cancel itself, and the state a cancelled prefill leaves
# behind.
#
# Steps, in the order they RUN: b, c, a, d. The table at the end is in
# alphabetical order, the execution is not, and the difference is deliberate:
# step (a) is a four-hour prefill_gate and steps (b) and (c) are the ones that
# can fail for a reason this change introduced. Finding out at hour four that
# the engine never honoured the cancel is how a session burns a day.
#
#  (a) prefill_gate.sh — the poll must be FREE. The pristine binary has no poll
#      at all, so the candidate's TTFT and decode rows against it ARE the cost
#      of one select(2) per 128-token chunk and per generated token. Bit-
#      identical is required: nothing here touches arithmetic.
#  (b) the cancel, in BOTH phases. Which phase a cancel lands in is a property
#      of the prompt, not of the flag: at 1 236 prompt tokens the prefill is
#      ~170 s, so --cancel 5 is inside it. The decode-phase one does not guess:
#      --cancel-phase decode asks for an answer long enough to have a decode
#      worth cancelling and counts the seconds from the FIRST TOKEN. It also
#      makes the harness check the engine's own account of where it stopped,
#      so a run cannot pass for a phase it did not exercise.
#  (c) resume. A cancelled prefill must leave the slot describing exactly the
#      positions the session reached: the identical prompt resubmitted must
#      REUSE that number, and must produce the same greedy text as a run that
#      was never cancelled. This is the oracle for "the partial state is
#      correct", which no timing can give.
#  (d) the rest of the serve path is untouched: tworeq at 4 slots at both KDA
#      knobs (IDENTICAL, no forcing line) and P7's capture/restore.
#  (e) the live rows through the gateway — the chain runs them after serving.
#
# NO SHADER CHANGES: this is C only, and the script refuses if any .spv differs
# from the pristine set (p5b_gate.sh's rule, same reason).
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-cancelgate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/cancel_gate_$TAG
TOOLS=${GATE_TOOLS:-$HOME/bench/owui_tools_4.json}
SYSTEM=${GATE_SYSTEM:-$HOME/bench/p6_system.txt}
SLOTS=${CANCEL_SLOTS:-4}
mkdir -p "$OUT" "$OUT/ckpt"

STEPS=${GATE_STEPS:-abcd}
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
: "${PRISTINE_SHADERS:?REFUSED: set PRISTINE_SHADERS to the pristine c/shaders copy}"
[ -f "$PRISTINE_SHADERS/kda_step.spv" ] || { echo "REFUSED: no kda_step.spv in $PRISTINE_SHADERS"; exit 2; }
# prefill_profile.sh -- which the pristine side of step (a) runs with
# COLI_VK_SHADERS pointed here -- refuses unless it finds a .comp SOURCE, so a
# snapshot of .spv alone silently turns the pristine half into no half at all.
[ -f "$PRISTINE_SHADERS/qmatmul.comp" ] || { echo "REFUSED: $PRISTINE_SHADERS has no qmatmul.comp; prefill_profile.sh will refuse the pristine side"; exit 2; }
export PRISTINE_SHADERS
SHADERS=$HERE/../../c/shaders
[ -f "$SHADERS/kda_step.spv" ] || { echo "REFUSED: no built shaders at $SHADERS"; exit 2; }
for spv in "$SHADERS"/*.spv; do
  b=$(basename "$spv")
  cmp -s "$spv" "$PRISTINE_SHADERS/$b" || { echo "REFUSED: $b differs from the pristine set -- CANCEL is C only"; exit 2; }
done
echo "    shaders: candidate == pristine on $(ls "$SHADERS"/*.spv | wc -l) .spv files (C only)"

cp -f "$HOME/.glm53_explain.bin" "$OUT/hist.bin" 2>/dev/null || true

# Every gate runs with checkpoints OFF and a private dir (CLAUDE.md): two gates
# nearly passed for the wrong reason because the candidate restored a
# checkpoint the pristine had just written. Step d turns them back on for its
# own run, in the same private dir.
export COLI_CKPT_DIR="$OUT/ckpt"

echo "=== cancel_gate $TAG $(date -Is)"
echo "    pristine=$PRISTINE ($(sha256sum "$PRISTINE" | cut -c1-16)) shaders=$PRISTINE_SHADERS"
echo "    candidate=$CAND ($(sha256sum "$CAND" | cut -c1-16)) shaders=$SHADERS"
echo "    steps=$STEPS slots=$SLOTS tools=$TOOLS"

rca=0; rcb=0; rcc=0; rcd=0

serve_env() {   # the serving knobs, one place
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
      COLI_CKPT_DIR="$OUT/ckpt" COLI_KDA_GPU=2 GLM53_VERBOSE=1 "$@"
}

# ---------------------------------------------------------------- (b)
echo
echo "### step b — the cancel itself, in both phases"
if run_step b; then
# b1: 1 236 prompt tokens, cancelled at 5 s -- inside the prefill.
echo "--- b1 prefill-phase cancel (--sizes 1000 --cancel 5)"
serve_env GLM53_PREFIX_CKPT=0 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --sizes 1000 --repeat 0 \
      --cancel 5 --cancel-phase prefill --kv-slots "$SLOTS" --gen 8 \
      --warm --min-resident 96 --tag "$TAG-c5" --json "$OUT/cancel.jsonl" \
      --engine-log "$OUT/engine_c5.log" 2>&1 | tee "$OUT/step_b1.txt"
rb1=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep -E "^cancel:|^VERDICT " "$OUT/step_b1.txt"
[ "$rb1" = 0 ] || rcb=1

# b2: --cancel-phase decode makes the harness ask for a long answer AND count
# the seconds from the first token, so this is a decode-phase cancel however
# long the prefill takes and however briefly the model chooses to talk. The
# first attempt at this asked for 20 s from SUBMIT on a 27-token prompt whose
# whole turn lasted 12.3 s: the cancel never fired and the step measured
# nothing (2026-09-08 22:14).
echo "--- b2 decode-phase cancel (--sizes 30 --cancel 10 after the first token)"
serve_env GLM53_PREFIX_CKPT=0 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --sizes 30 --repeat 0 \
      --cancel 10 --cancel-phase decode --kv-slots "$SLOTS" --gen 8 \
      --warm --min-resident 96 --tag "$TAG-c20" --json "$OUT/cancel.jsonl" \
      --engine-log "$OUT/engine_c20.log" 2>&1 | tee "$OUT/step_b2.txt"
rb2=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep -E "^cancel:|^VERDICT " "$OUT/step_b2.txt"
[ "$rb2" = 0 ] || rcb=1
grep -h "CANCEL " "$OUT/engine_c5.log" "$OUT/engine_c20.log" || echo "(no CANCEL line -- the engine did not report a phase)"
else echo "(skipped)"; fi
echo "step b rc=$rcb"

# ---------------------------------------------------------------- (c)
echo
echo "### step c — a cancelled prefill resumes, and the resumed state is the same state"
if run_step c; then
serve_env GLM53_PREFIX_CKPT=0 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --sizes 1000 --repeat 0 \
      --cancel-resume 30 --kv-slots "$SLOTS" --gen 24 \
      --warm --min-resident 96 --tag "$TAG-resume" --json "$OUT/cancel.jsonl" \
      --engine-log "$OUT/engine_resume.log" 2>&1 | tee "$OUT/step_c.txt"
rcc=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep -E "^cancel-resume:|^VERDICT " "$OUT/step_c.txt"
grep -hE "CANCEL |REUSE " "$OUT/engine_resume.log" | tail -8
else echo "(skipped)"; fi
echo "step c rc=$rcc"

# The two steps this change can break come first, and if either fails the
# four-hour half is not spent: the gate has already failed.
if [ "$rcb" != 0 ] || [ "$rcc" != 0 ]; then
  echo
  echo "=== cancel_gate $TAG: step b/c FAILED (rcb=$rcb rcc=$rcc) -- skipping the "
  echo "    prefill_gate and tworeq halves, the gate is already lost"
  echo "  outputs in $OUT"
  exit 1
fi

# ---------------------------------------------------------------- (a)
echo
echo "### step a — prefill_gate: bit-identical, and the poll costs nothing"
if run_step a; then
GLM53_PREFIX_CKPT=0 MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  bash "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-poll" 2>&1 | tee "$OUT/step_a.txt"
rca=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step a rc=$rca"

# ---------------------------------------------------------------- (d)
echo
echo "### step d — the rest of the serve path: tworeq at both knobs, P7 capture/restore"
if run_step d; then
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
  [ "$r" = 0 ] || rcd=1
  if [ "$knob" = 2 ] && [ "$forced" != 0 ]; then
    echo "    FAIL: the recurrence fell back to the CPU -- this run is not at the serving knob"; rcd=1
  fi
  wait_no_engine || exit 2
done
echo "--- P7 capture/restore"
rm -f "$OUT/ckpt"/*.bin
serve_env GLM53_PREFIX_CKPT=1 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt \
      --tools "$TOOLS" --system "$SYSTEM" --kv-slots "$SLOTS" \
      --sizes "" --repeat 0 --warm --min-resident 96 \
      --tag "$TAG-ckpt" --json "$OUT/ckpt.jsonl" \
      --engine-log "$OUT/engine_ckpt.log" 2>&1 | tee "$OUT/step_d.txt"
rd=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep -E "CKPT (store|hit)" "$OUT/engine_ckpt.log" || echo "(no CKPT line)"
grep -E "^prefix-ckpt:|^VERDICT " "$OUT/step_d.txt" | tail -6
if [ "$rd" != 0 ] || ! grep -q "^VERDICT prefix-ckpt: PASS" "$OUT/step_d.txt"; then rcd=1; fi
else echo "(skipped)"; fi
echo "step d rc=$rcd"

# ---------------------------------------------------------------- verdict
echo
echo "=== cancel_gate $TAG verdict $(date -Is)"
printf "  step a  bit-identical + the poll is free              rc=%s\n" "$rca"
printf "  step b  cancel confirmed in both phases               rc=%s\n" "$rcb"
printf "  step c  cancelled prefill resumes, same text          rc=%s\n" "$rcc"
printf "  step d  tworeq %d slots + P7 capture/restore           rc=%s\n" "$SLOTS" "$rcd"
echo "  outputs in $OUT"
if [ "$rca" = 2 ]; then exit 2; fi
if [ "$rca" = 1 ] || [ "$rcb" != 0 ] || [ "$rcc" != 0 ] || [ "$rcd" != 0 ]; then exit 1; fi
if [ "$rca" != 0 ]; then exit 3; fi
exit 0
