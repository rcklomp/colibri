#!/bin/bash
# P8 — the executable gate for "the reply pin".
#
#   p8_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (teacher forcing, logits, tworeq, text identity)
#          or the regression case did not turn.
# Exit 2 = a harness refusal (residency, another engine running, no gateway).
# Exit 3 = a speed check failed (step 1's neutrality).
#
# P8 is a GATEWAY change: c/openai_server.py only, no engine source, no shader.
# So the first half must show the engine is exactly what it was, and the second
# half is the whole point — it runs on the SERVED gateway, through Open WebUI's
# own backend, because that is where the bug lives.
#
#  (1) prefill_gate.sh with GLM53_PREFIX_CKPT=0, MIN_SPEEDUP=0.97 and a private
#      COLI_CKPT_DIR: bit-identical and neutral, by construction.
#  (2) tworeq.py at TWOREQ_SLOTS=4, COLI_KDA_GPU=2 and =0: IDENTICAL, and no
#      "forcing COLI_KDA_GPU=0" line at knob 2.
#  (3) THE REGRESSION. p8_reasoning_case.sh three times with COLI_REPLY_PIN=1
#      (all three must PASS) and three times with COLI_REPLY_PIN=0 (the control:
#      it must FAIL, or the case did not reproduce the bug and proves nothing).
#  (4) TEXT IDENTITY. Two fixed first-turn prompts at temperature 0, once in
#      each arm: the bytes the client receives must not depend on the knob.
#      First turns on purpose — the pin cannot fire on a conversation with no
#      prior assistant message, so any difference here would be the pin
#      touching the response path, which it must never do. On a turn where the
#      pin DOES fire the prompt is deliberately different (that IS the item),
#      so an identity claim there would be false.
#
# GATE_STEPS selects: "12" is the offline half (gateway must be DOWN), "34" the
# live half (gateway must be UP; the gate restarts it itself for the OFF arm and
# leaves it back on the default when it is finished). Default "1234".
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p8gate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p8_gate_$TAG
LOG=${GLM53_LOG:-$HOME/glm53_server.log}
STEPS=${GATE_STEPS:-1234}
mkdir -p "$OUT"
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 240 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}
gw_stop() { "$HOME/bench/p7_stop.sh"; }
gw_start() {   # gw_start [VAR=VAL ...] — the documented restart, plus this arm's knobs
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      "$@" SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$HOME/.colibri_api_key")" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && { sleep 2; return 0; }
    sleep 5
  done
  echo "gateway did not answer /v1/models within 10 min"; return 1
}

echo "=== p8_gate $TAG $(date -Is) steps=$STEPS"
echo "    pristine=$PRISTINE candidate=$CAND"
echo "    outputs in $OUT"

rc1=0; rc2=0; rc3=0; rc4=0

# ---------------------------------------------------------------- (1)
if run_step 1; then
  echo
  echo "### step 1 — prefill_gate.sh, checkpoints off (P8 touches no engine code)"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2; fi
  SELF=0
  [ "$(sha256sum "$PRISTINE" | cut -d" " -f1)" = "$(sha256sum "$CAND" | cut -d" " -f1)" ] && SELF=1
  echo "    pristine and candidate are $([ $SELF = 1 ] && echo "THE SAME BYTES" || echo "different binaries")"
  GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="$OUT/ckpt/step1" MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
    "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-off" 2>&1 | tee "$OUT/step1.txt"
  rc1=${PIPESTATUS[0]}
  wait_no_engine || exit 2
  # A file compared with ITSELF cannot be slower than itself: when the two
  # binaries are byte-identical the speed half of prefill_gate measures the
  # harness, not a change, and its verdict is not evidence about this item.
  # Measured here 2026-09-09 on exactly that comparison: 2.84/2.72 s (pristine)
  # against 2.92/2.82 s (candidate) on the 27-token row -> 0.97x -> rc 3, with
  # 1.00x at 390 and 1 236 tokens and `teacher_forcing IDENTICAL (782
  # positions), cosine=1.0000000 max_abs=0`. So the ORACLE half still has to
  # pass (rc 1 stays fatal) and the numbers still go in the commit body; only
  # the speed verdict is downgraded, and only when the bytes are identical.
  # This is not a threshold: a candidate whose binary differs is judged at
  # MIN_SPEEDUP as before.
  if [ "$rc1" = 3 ] && [ "$SELF" = 1 ]; then
    echo "step 1: the speed check FAILED on a self-comparison (identical bytes) -- INFORMATIONAL"
    echo "        the oracle half passed; the table above is the harness's noise floor at 27 tokens"
    rc1=0
  fi
  echo "step 1 rc=$rc1"
fi

# ---------------------------------------------------------------- (2)
if run_step 2; then
  echo
  echo "### step 2 — tworeq at 4 slots, both KDA knobs"
  if pgrep -x glm53 >/dev/null; then echo "REFUSED: a glm53 is running"; exit 2; fi
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

# ---------------------------------------------------------------- (3)+(4), live
# Two fixed first-turn prompts for the identity check. Short, deterministic, and
# with no prior assistant turn, so the pin provably cannot fire on them.
ID_P1='Name the three primary additive colours, comma separated.'
ID_P2='What is the capital of Portugal? Answer in one word.'
identity_run() {   # identity_run <arm>
  local arm=$1 i=0 p
  for p in "$ID_P1" "$ID_P2"; do
    i=$((i + 1))
    python3 - "$p" "$OUT/identity_${arm}_$i.json" <<'PY'
import json, os, sys, urllib.request
key = open(os.path.expanduser("~/.colibri_api_key")).read().strip()
body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": sys.argv[1]}],
        "max_tokens": 96, "temperature": 0, "stream": False}
req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
msg = json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]["message"]
out = {"content": msg.get("content"), "reasoning_content": msg.get("reasoning_content")}
open(sys.argv[2], "w").write(json.dumps(out, ensure_ascii=False, sort_keys=True))
print(f"  identity {sys.argv[2].split('/')[-1]}: "
      f"{len((out['content'] or '').encode())} B content, "
      f"{len((out['reasoning_content'] or '').encode())} B reasoning")
PY
  done
}
case_arm() {   # case_arm <arm> -> the three runs on stderr, "<pass> <fail>" on stdout
  local arm=$1 pass=0 fail=0 seed f
  : > "$OUT/case_$arm.txt"
  for seed in 1 2 3; do
    f="$OUT/case_${arm}_$seed.txt"
    echo "--- reasoning case, arm=$arm seed=$seed $(date -Is)" >&2
    "$HERE/p8_reasoning_case.sh" --seed "$seed" > "$f" 2>&1
    cat "$f" >&2; cat "$f" >> "$OUT/case_$arm.txt"
    if grep -q "verdict=PASS" "$f"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
  done
  echo "$pass $fail"
}

if run_step 3 || run_step 4; then
  echo
  echo "### steps 3 and 4 — the live half, on the SERVED gateway"
  pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: the gateway is not running"; exit 2; }
  echo "--- arm A: COLI_REPLY_PIN default (1)"
  run_step 4 && identity_run on
  ONP=0; ONF=0
  if run_step 3; then read -r ONP ONF <<<"$(case_arm on)"; fi

  echo "--- restarting the gateway with COLI_REPLY_PIN=0 (the control arm)"
  gw_stop || exit 2
  wait_no_engine || exit 2
  gw_start COLI_REPLY_PIN=0 || exit 2
  echo "--- arm B: COLI_REPLY_PIN=0"
  run_step 4 && identity_run off
  OFFP=0; OFFF=0
  if run_step 3; then read -r OFFP OFFF <<<"$(case_arm off)"; fi

  echo "--- restarting the gateway on the shipped default (COLI_REPLY_PIN=1)"
  gw_stop || exit 2
  wait_no_engine || exit 2
  gw_start || exit 2

  if run_step 3; then
    echo
    echo "step 3 — pin ON: $ONP/3 PASS, $ONF/3 FAIL;  pin OFF (control): $OFFP/3 PASS, $OFFF/3 FAIL"
    [ "$ONP" = 3 ] || { echo "  FAIL: the follow-up turn does not reuse prompt+gen with the pin on"; rc3=1; }
    # The control has to fail, or the case never reproduced the bug: a first turn
    # that happens not to reason reuses fine with the pin off too, and then the
    # ON arm proves nothing. Two of three is the bar because whether the model
    # reasons is the model's choice, not the harness's.
    [ "$OFFF" -ge 2 ] || { echo "  FAIL: the control did not re-prefill ($OFFF/3 failed, need >= 2)"; rc3=1; }
    echo "step 3 rc=$rc3"
  fi
  if run_step 4; then
    echo
    for i in 1 2; do
      if cmp -s "$OUT/identity_on_$i.json" "$OUT/identity_off_$i.json"; then
        echo "step 4 prompt $i: IDENTICAL ($(wc -c < "$OUT/identity_on_$i.json") B)"
      else
        echo "step 4 prompt $i: DIFFERS"
        diff <(cat "$OUT/identity_on_$i.json") <(cat "$OUT/identity_off_$i.json") | head -5
        rc4=1
      fi
    done
    echo "step 4 rc=$rc4"
  fi
fi

# ---------------------------------------------------------------- verdict
echo
echo "=== p8_gate $TAG verdict $(date -Is)"
printf "  step 1  prefill_gate (ckpt off)      rc=%s\n" "$rc1"
printf "  step 2  tworeq 4 slots, both knobs   rc=%s\n" "$rc2"
printf "  step 3  reasoning case, on vs off    rc=%s\n" "$rc3"
printf "  step 4  text identity                rc=%s\n" "$rc4"
echo "  outputs in $OUT"
[ "$rc1" = 2 ] && exit 2
[ "$rc1" = 1 ] && exit 1
[ "$rc2" = 0 ] || exit 1
[ "$rc3" = 0 ] || exit 1
[ "$rc4" = 0 ] || exit 1
[ "$rc1" = 0 ] || exit 3
exit 0
