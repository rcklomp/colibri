#!/bin/bash
# P9 — the executable gate for "the conversation ledger".
#
#   p9_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, tworeq) or a client
#          behaviour did not turn, or a divergence was not byte-identical.
# Exit 2 = a harness refusal (residency, another engine running, no gateway).
# Exit 3 = a speed check failed (step 1's neutrality).
#
# P9 changes c/openai_server.py and ONE trailing field of the engine's DONE STAT
# line. So the engine must still be bit-identical where it counts (step 1's
# oracle: teacher forcing identical, max_abs=0) and neutral on the serve path
# (MIN_SPEEDUP=0.97) — unlike P8, the binary genuinely differs, so the speed
# verdict is judged, not downgraded.
#
#  (1) prefill_gate.sh, GLM53_PREFIX_CKPT=0, MIN_SPEEDUP=0.97, private
#      COLI_CKPT_DIR, PROFILE_MIN_RESIDENT=97.
#  (2) tworeq.py at TWOREQ_SLOTS=4, COLI_KDA_GPU=2 and =0: IDENTICAL, and no
#      "forcing COLI_KDA_GPU=0" line at knob 2.
#  (3) THE THREE CLIENT BEHAVIOURS, in THREE arms:
#        on   COLI_LEDGER=1                                     (the item)
#        off  COLI_LEDGER=0                     -- today's path, pins included
#        bare COLI_LEDGER=0 COLI_PREFIX_PIN=0 COLI_REPLY_PIN=0
#                                               -- the pre-P7/P8 gateway
#      The spec asks for "PASS with COLI_LEDGER=1 and FAIL with COLI_LEDGER=0".
#      Taken literally that is not measurable for behaviours (a) and (b): at
#      COLI_LEDGER=0 the gateway keeps P7's pin and P8's reply pin, which were
#      built for exactly those two, so the control would pass and prove nothing
#      about either mechanism. The honest control for "the ledger subsumes the
#      class" is the gateway with NO mechanism at all, and the honest control for
#      "the ledger is not a regression" is today's path. Both are run and both
#      tables are reported. Expected, and stated before the run: (a) and (b) pass
#      `on` and `off` and fail `bare`; (c) passes `on` only, because it is the
#      behaviour neither pin covers.
#  (4) DIVERGENCE CORRECTNESS. Regenerate, edit-an-earlier-message and branch:
#      each renders text byte-identical to a COLD gateway's rendering of the same
#      transcript (greedy), and each logs `ledger=reset`. Slow is allowed, wrong
#      is not.
#  (5) THE INVARIANT IN PRODUCTION SHAPE is a Mac-side step (a real browser):
#      tools/hot-expert/p9_ui_multiturn.sh. `GATE_STEPS=5` here audits the
#      gateway log for it: zero MISMATCH and expect_reuse == engine_reuse on
#      every continuation.
#
# GATE_STEPS selects: "12" is the offline half (gateway must be DOWN), "34" the
# live half (gateway must be UP), "5" the log audit. Default "1234".
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p9gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p9_gate_$TAG
LOG=${GLM53_LOG:-$HOME/glm53_server.log}
STEPS=${GATE_STEPS:-1234}
SEEDS=${P9_SEEDS:-3}
AUDIT_N=${P9_AUDIT_N:-12}
mkdir -p "$OUT"
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 240 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}
gw_stop() { "$HOME/bench/p7_stop.sh"; }
# ~/start_glm53.sh EXPORTS COLI_PREFIX_PIN=1 unconditionally, so an arm that has
# to run without P7's pin cannot just put it in the environment -- the serving
# script overwrites it, and the "no mechanism" arm would silently measure P7.
# So the arms run a copy of the owner's own script with that one line (and the
# checkpoint knob, for symmetry) made overridable, generated from the original at
# run time so it cannot drift from what is in service. The FINAL restart uses the
# owner's file verbatim.
ARM_SCRIPT="$OUT/start_glm53_arm.sh"
make_arm_script() {
  sed -e 's/^export COLI_PREFIX_PIN=1$/export COLI_PREFIX_PIN=${COLI_PREFIX_PIN:-1}/' \
      -e 's/^export GLM53_PREFIX_CKPT=1$/export GLM53_PREFIX_CKPT=${GLM53_PREFIX_CKPT:-1}/' \
      "$HOME/start_glm53.sh" > "$ARM_SCRIPT"
  grep -q 'COLI_PREFIX_PIN:-1' "$ARM_SCRIPT" || { echo "the arm script did not take"; return 1; }
  chmod +x "$ARM_SCRIPT"
}
gw_start() {   # gw_start [VAR=VAL ...] — the documented restart, plus this arm's knobs
  local script="$HOME/start_glm53.sh"
  [ $# -gt 0 ] && { make_arm_script || return 1; script="$ARM_SCRIPT"; }
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      -u COLI_LEDGER -u COLI_PREFIX_PIN \
      "$@" SKIP_WARM=1 setsid nohup "$script" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$HOME/.colibri_api_key")" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && { sleep 2; return 0; }
    sleep 5
  done
  echo "gateway did not answer /v1/models within 10 min"; return 1
}

echo "=== p9_gate $TAG $(date -Is) steps=$STEPS seeds=$SEEDS"
echo "    pristine=$PRISTINE candidate=$CAND"
echo "    outputs in $OUT"

rc1=0; rc2=0; rc3=0; rc4=0; rc5=0
ran1=0; ran2=0; ran3=0; ran4=0; ran5=0
verdict_line() {  # verdict_line <label> <ran> <rc>
  if [ "$2" = 1 ]; then printf "  %-36s rc=%s\n" "$1" "$3"
  else printf "  %-36s -- (not run)\n" "$1"; fi
}

# ---------------------------------------------------------------- (1)
if run_step 1; then
  echo
  echo "### step 1 — prefill_gate.sh, checkpoints off (the engine gains one STAT field)"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
  if [ "$(sha256sum "$PRISTINE" | cut -d" " -f1)" = "$(sha256sum "$CAND" | cut -d" " -f1)" ]; then
    echo "    NOTE: pristine and candidate are THE SAME BYTES -- P9 was expected to change"
    echo "          the engine by one STAT field, so this is a build that did not take"
  else
    echo "    pristine and candidate are different binaries (expected: the 8th STAT field)"
  fi
  GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt/step1" MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
    PROFILE_MIN_RESIDENT=${PROFILE_MIN_RESIDENT:-97} \
    "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-off" 2>&1 | tee "$OUT/step1.txt"
  rc1=${PIPESTATUS[0]}; ran1=1
  wait_no_engine || exit 2
  echo "step 1 rc=$rc1"
fi

# ---------------------------------------------------------------- (2)
if run_step 2; then
  echo
  echo "### step 2 — tworeq at 4 slots, both KDA knobs"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running"; exit 2; fi
  ran2=1
  # A COPY of the histogram, never the canonical file: the engine rewrites
  # COLI_USAGE_PATH at exit and a benchmark must not teach the serving tier.
  cp -f "$HOME/.glm53_explain.bin" "$OUT/hist_tworeq.bin" 2>/dev/null || true
  for knob in 2 0; do
    echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=4"
    env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
        COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
        COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
        COLI_VK_SHADERS="$HERE/../../c/shaders" \
        COLI_USAGE_PATH="$OUT/hist_tworeq.bin" \
        COLI_CKPT_DIR="$OUT/ckpt/step2" \
        COLI_KDA_GPU=$knob TWOREQ_SLOTS=4 TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
        python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
    r=$?
    forced=$(grep -c "forcing COLI_KDA_GPU=0" "$OUT/tworeq_kda$knob.txt")
    echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
    echo "    forcing lines: $forced"
    [ "$r" = 0 ] || rc2=1
    if [ "$knob" = 2 ] && [ "$forced" != 0 ]; then
      echo "    FAIL: the guard fired at knob 2 -- this run measured the CPU recurrence"; rc2=1
    fi
    wait_no_engine || exit 2
  done
  echo "step 2 rc=$rc2"
fi

# ---------------------------------------------------------------- (3) live
case_arm() {   # case_arm <arm> <case> -> "<pass> <fail>" on stdout, detail on stderr
  local arm=$1 case=$2 pass=0 fail=0 seed f script
  case "$case" in
    memory)    script="$HERE/p9_memory_case.sh";;
    reasoning) script="$HERE/p8_reasoning_case.sh";;
    strip)     script="$HERE/p9_strip_case.sh";;
    *) echo "unknown case $case" >&2; return 2;;
  esac
  for seed in $(seq 1 "$SEEDS"); do
    f="$OUT/case_${case}_${arm}_$seed.txt"
    echo "--- case=$case arm=$arm seed=$seed $(date -Is)" >&2
    "$script" --seed "$seed" > "$f" 2>&1
    cat "$f" >&2
    if grep -q "verdict=PASS" "$f"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
  done
  echo "$pass $fail"
}

declare -A RES
if run_step 3; then
  echo
  echo "### step 3 — the three client behaviours, three arms"
  pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: the gateway is not running"; exit 2; }
  ran3=1
  for arm in on off bare; do
    case "$arm" in
      on)   echo "--- arm ON:   COLI_LEDGER=1 (the shipped default)";;
      off)  echo "--- arm OFF:  COLI_LEDGER=0 -- today's path, P7 pin + P8 reply pin"
            gw_stop || exit 2; wait_no_engine || exit 2
            gw_start COLI_LEDGER=0 || exit 2;;
      bare) echo "--- arm BARE: COLI_LEDGER=0 COLI_PREFIX_PIN=0 COLI_REPLY_PIN=0 -- no mechanism"
            gw_stop || exit 2; wait_no_engine || exit 2
            gw_start COLI_LEDGER=0 COLI_PREFIX_PIN=0 COLI_REPLY_PIN=0 || exit 2;;
    esac
    for case in memory reasoning strip; do
      read -r p f <<<"$(case_arm "$arm" "$case")"
      RES[$arm-$case]="$p $f"
      echo "  $case arm=$arm: $p/$SEEDS PASS, $f/$SEEDS FAIL"
    done
  done
  echo "--- restarting the gateway on the shipped default (COLI_LEDGER=1)"
  gw_stop || exit 2; wait_no_engine || exit 2; gw_start || exit 2

  echo
  printf '%-12s %-18s %-18s %-18s\n' "behaviour" "ledger ON" "ledger OFF (today)" "no mechanism"
  for case in memory reasoning strip; do
    printf '%-12s %-18s %-18s %-18s\n' "$case" \
      "$(echo "${RES[on-$case]}" | awk '{print $1"/'"$SEEDS"' PASS"}')" \
      "$(echo "${RES[off-$case]}" | awk '{print $1"/'"$SEEDS"' PASS"}')" \
      "$(echo "${RES[bare-$case]}" | awk '{print $1"/'"$SEEDS"' PASS"}')"
  done
  for case in memory reasoning strip; do
    read -r p _f <<<"${RES[on-$case]}"
    [ "$p" = "$SEEDS" ] || { echo "  FAIL: $case does not reuse prompt+gen with the ledger on ($p/$SEEDS)"; rc3=1; }
  done
  # The control has to fail, or the case never reproduced the behaviour and the
  # ON arm proves nothing. The bar is 2 of 3 for the two cases that depend on the
  # model or on a vector search; the synthetic one is deterministic and must fail
  # every time.
  for case in memory reasoning; do
    read -r _p f <<<"${RES[bare-$case]}"
    [ "$f" -ge 2 ] || { echo "  FAIL: $case did not reproduce without a mechanism ($f/$SEEDS failed, need >= 2)"; rc3=1; }
  done
  read -r _p f <<<"${RES[bare-strip]}"
  [ "$f" = "$SEEDS" ] || { echo "  FAIL: the synthetic strip case must fail with no mechanism ($f/$SEEDS)"; rc3=1; }
  read -r _p f <<<"${RES[off-strip]}"
  [ "$f" -ge 2 ] || { echo "  NOTE: today's path already survives the strip case ($f/$SEEDS failed) -- the class claim is weaker than expected"; fi_note=1; }
  echo "step 3 rc=$rc3"
fi

# ---------------------------------------------------------------- (4) live
if run_step 4; then
  echo
  echo "### step 4 — divergence correctness: regenerate, edit, branch"
  pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: the gateway is not running"; exit 2; }
  ran4=1
  DIV="$OUT/divergence"; mkdir -p "$DIV"
  echo "--- warm phase (the ledger holds these conversations)"
  "$HERE/p9_divergence.sh" --phase warm --out "$DIV" 2>&1 | tee "$OUT/step4_warm.txt"
  w=${PIPESTATUS[0]}
  echo "--- restarting the gateway: the cold phase must meet an EMPTY ledger"
  gw_stop || exit 2; wait_no_engine || exit 2; gw_start || exit 2
  echo "--- cold phase"
  "$HERE/p9_divergence.sh" --phase cold --out "$DIV" 2>&1 | tee "$OUT/step4_cold.txt"
  c=${PIPESTATUS[0]}
  echo "--- comparison"
  "$HERE/p9_divergence.sh" --phase compare --out "$DIV" 2>&1 | tee "$OUT/step4_compare.txt"
  cmp_rc=${PIPESTATUS[0]}
  [ "$w" = 0 ] && [ "$c" = 0 ] && [ "$cmp_rc" = 0 ] || rc4=1
  echo "step 4 rc=$rc4 (warm=$w cold=$c compare=$cmp_rc)"
fi

# ---------------------------------------------------------------- (5) audit
if run_step 5; then
  echo
  echo "### step 5 — the invariant in the log: the last $AUDIT_N requests"
  ran5=1
  python3 - "$LOG" "$AUDIT_N" 2>&1 | tee "$OUT/step5.txt" <<'PY'
import re, sys
log, n = sys.argv[1], int(sys.argv[2])
rows = [l.strip() for l in open(log, errors="replace") if l.startswith("[ledger] ")]
rows = rows[-n:] if n else rows
mism = [l for l in rows if "MISMATCH" in l]
broken = [l for l in rows if "ledger=broken" in l]
checked = [l for l in rows if "expect_reuse=" in l and "expect_reuse=-" not in l]
for l in rows:
    print("  " + l[:150])
print(f"lines={len(rows)} checked={len(checked)} MISMATCH={len(mism)} broken={len(broken)}")
if not rows:
    print("NO [ledger] LINES AT ALL -- COLI_REQ_LOG off, or the ledger never ran")
sys.exit(1 if (mism or broken or not rows) else 0)
PY
  rc5=${PIPESTATUS[0]}
  echo "step 5 rc=$rc5"
fi

# ---------------------------------------------------------------- verdict
echo
echo "=== p9_gate $TAG verdict $(date -Is)"
verdict_line "step 1  prefill_gate (ckpt off)" "$ran1" "$rc1"
verdict_line "step 2  tworeq 4 slots, both knobs" "$ran2" "$rc2"
verdict_line "step 3  three behaviours, three arms" "$ran3" "$rc3"
verdict_line "step 4  divergence == cold render" "$ran4" "$rc4"
verdict_line "step 5  no MISMATCH in the log" "$ran5" "$rc5"
[ "$ran3$ran4" = "11" ] || echo "  NOTE: this run did NOT gate the item -- the live half is what P9 is about"
echo "  outputs in $OUT"
[ "$rc1" = 2 ] && exit 2
[ "$rc1" = 1 ] && exit 1
[ "$rc2" = 0 ] || exit 1
[ "$rc3" = 0 ] || exit 1
[ "$rc4" = 0 ] || exit 1
[ "$rc5" = 0 ] || exit 1
[ "$rc1" = 0 ] || exit 3
exit 0
