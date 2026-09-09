#!/bin/bash
# P7b — the executable gate for "capture beyond a restored checkpoint".
#
#   p7b_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, text, tworeq, or one of
#          the three checkpoint cases).
# Exit 2 = a harness refusal (residency, another engine running, a fixture).
# Exit 3 = a speed check failed (step 1's neutrality).
#
# What P7b changes is WHERE the prefill splits, and only that: after a
# checkpoint RESTORE the engine may now plan a further capture, and between a
# plan and a gateway hint it takes the LONGER one. Nothing about the arithmetic
# moves, so step 1 must still show the candidate IS the pristine engine with
# checkpoints off, and step 3's oracle must still find conv 3 identical to the
# same prompt served with GLM53_PREFIX_CKPT=0.
#
# In order:
#  (1) prefill_gate.sh with GLM53_PREFIX_CKPT=0 and MIN_SPEEDUP=0.97 — off, the
#      candidate must BE the pristine engine: identical teacher forcing,
#      identical last-position logits, serve-path TTFT within a percent.
#  (2) tworeq.py at TWOREQ_SLOTS=4, at COLI_KDA_GPU=0 and =2: IDENTICAL, and
#      the "forcing COLI_KDA_GPU=0" guard must stay silent at knob 2.
#  (3) the three checkpoint cases, each in its own engine and its own private
#      checkpoint directory:
#      (3a) ttft_serve.py --prefix-ckpt-partial --oracle — THE NEW CASE. A
#           partial checkpoint must not stop the rest of the stable prefix from
#           ever being captured: conv 2 restores it, prints
#           `CKPT plan ... after-restore`, and stores the longer prefix; conv 3
#           reuses the whole thing. This step FAILS on the pristine binary,
#           which is what makes it a gate and not a printout.
#      (3b) ttft_serve.py --prefix-ckpt --oracle — P7's own case, unchanged.
#      (3c) ttft_serve.py --pin-block — P7's pin, unchanged.
#
# GATE_TOOLS overrides the tool block (default the 4-tool Open WebUI subset:
# the 34-tool block is a ~24-minute cold prefill and step 3a pays it twice).
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p7bgate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p7b_gate_$TAG
TOOLS=${GATE_TOOLS:-$HOME/bench/owui_tools_4.json}
SYSTEM=${GATE_SYSTEM:-$HOME/bench/p6_system.txt}
SNAP=${SNAP:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
mkdir -p "$OUT"

# A step that leaves its engine behind makes the NEXT step refuse, and a revert
# that copies over a still-running binary fails with ETXTBSY (p7, 2026-09-07).
wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 240 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}
if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
STEPS=${GATE_STEPS:-123}
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }
for f in "$TOOLS" "$SYSTEM"; do
  [ -s "$f" ] || { echo "REFUSED: missing fixture $f"; exit 2; }
done

# The serving checkpoint directory is never touched: every step gets its own,
# so no step inherits a prefix another one captured and the gateway's own
# checkpoints survive the gate untouched.
CKPT_ROOT=${COLI_CKPT_DIR:-$OUT/ckpt}
mkdir -p "$CKPT_ROOT"

echo "=== p7b_gate $TAG $(date -Is)"
echo "    pristine=$PRISTINE candidate=$CAND"
echo "    tools=$TOOLS ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$TOOLS") entries) system=$SYSTEM"
echo "    ckpt root=$CKPT_ROOT (per-step subdirectories; the served $SNAP/.coli_ckpt is untouched)"

rc1=0; rc2=0; rc3a=0; rc3b=0; rc3c=0

# ---------------------------------------------------------------- (1)
echo
echo "### step 1 — prefill_gate.sh with GLM53_PREFIX_CKPT=0 (P7b must be inert when off)"
if run_step 1; then
GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$CKPT_ROOT/step1" MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-off" 2>&1 | tee "$OUT/step1.txt"
rc1=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 1 rc=$rc1"

# ---------------------------------------------------------------- (2)
echo
echo "### step 2 — tworeq at 4 slots, both KDA knobs"
# A COPY of the histogram, never the canonical file: the engine rewrites
# COLI_USAGE_PATH at exit and a benchmark must not teach the serving tier.
cp -f "$HOME/.glm53_explain.bin" "$OUT/hist_tworeq.bin" 2>/dev/null || true
if run_step 2; then
for knob in 2 0; do
  echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=4"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$HERE/../../c/shaders" \
      COLI_USAGE_PATH="$OUT/hist_tworeq.bin" \
      COLI_CKPT_DIR="$CKPT_ROOT/step2" \
      COLI_KDA_GPU=$knob TWOREQ_SLOTS=4 TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
      python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
  r=$?
  forced=$(grep -c "forcing COLI_KDA_GPU=0" "$OUT/tworeq_kda$knob.txt")
  echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
  echo "    forcing lines: $forced"
  [ "$r" = 0 ] || rc2=1
  if [ "$knob" = 2 ] && [ "$forced" != 0 ]; then
    echo "    FAIL: the guard fired at knob 2 -- this run measured the CPU recurrence"
    rc2=1
  fi
  wait_no_engine || exit 2
done
else echo "(skipped)"; fi
echo "step 2 rc=$rc2"

# ---------------------------------------------------------------- (3a)
echo
echo "### step 3a — ttft_serve.py --prefix-ckpt-partial --oracle (THE P7b CASE)"
if run_step 3; then
COLI_CKPT_DIR="$CKPT_ROOT/step3a" \
python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt-partial --oracle \
    --tools "$TOOLS" --system "$SYSTEM" --kv-slots 4 --sizes "" --repeat 0 \
    --warm --min-resident 96 --tag "$TAG-partial" --json "$OUT/p7b.jsonl" \
    --engine-log "$OUT/engine_partial.log" --logit-dir "$OUT/logits_partial" 2>&1 | tee "$OUT/step3a.txt"
rc3a=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 3a rc=$rc3a"

# ---------------------------------------------------------------- (3b)
echo
echo "### step 3b — ttft_serve.py --prefix-ckpt --oracle (P7's case must still pass)"
if run_step 3; then
COLI_CKPT_DIR="$CKPT_ROOT/step3b" \
python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt --oracle \
    --tools "$TOOLS" --system "$SYSTEM" --kv-slots 4 --sizes "" --repeat 0 \
    --warm --min-resident 96 --tag "$TAG-ckpt" --json "$OUT/p7b.jsonl" \
    --engine-log "$OUT/engine_ckpt.log" --logit-dir "$OUT/logits_ckpt" 2>&1 | tee "$OUT/step3b.txt"
rc3b=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 3b rc=$rc3b"

# ---------------------------------------------------------------- (3c)
echo
echo "### step 3c — ttft_serve.py --pin-block (P7's pin must still pass)"
if run_step 3; then
COLI_CKPT_DIR="$CKPT_ROOT/step3c" \
python3 "$HERE/ttft_serve.py" --engine "$CAND" --pin-block --kv-slots 4 \
    --sizes "" --repeat 0 --system "$SYSTEM" \
    --warm --min-resident 96 --tag "$TAG-pin" --json "$OUT/p7b.jsonl" \
    --engine-log "$OUT/engine_pin.log" 2>&1 | tee "$OUT/step3c.txt"
rc3c=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 3c rc=$rc3c"

# ---------------------------------------------------------------- verdict
echo
echo "=== p7b_gate $TAG verdict $(date -Is)"
printf "  step 1  prefill_gate (ckpt off)      rc=%s\n" "$rc1"
printf "  step 2  tworeq 4 slots, both knobs   rc=%s\n" "$rc2"
printf "  step 3a prefix-ckpt-partial + oracle rc=%s\n" "$rc3a"
printf "  step 3b prefix-ckpt + oracle         rc=%s\n" "$rc3b"
printf "  step 3c pin-block                    rc=%s\n" "$rc3c"
echo "  outputs in $OUT"
[ "$rc1" = 2 ] && exit 2
[ "$rc3a" = 2 ] && exit 2
[ "$rc1" = 1 ] && exit 1
[ "$rc2" = 0 ] || exit 1
[ "$rc3a" = 0 ] || exit 1
[ "$rc3b" = 0 ] || exit 1
[ "$rc3c" = 0 ] || exit 1
[ "$rc1" = 0 ] || exit 3
exit 0
