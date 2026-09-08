#!/bin/bash
# CANCEL chain: glm53 honours CANCEL while a turn is in flight.
#
# Structure is p5b_chain.sh's, and for the same reasons:
#   stop gateway -> snapshot the SERVED shaders -> merge p0-sync -> build ->
#   qwen38 C tests + test_serve_poll -> smoke -> cancel_gate.sh (a,b,c,d) ->
#   serve the candidate only on rc 0 -> live rows (WITH --warm) -> the curl
#   disconnect -> verified sha256 revert on any failure -> the gateway
#   restarted on EVERY exit path.
#
# The one addition is the live curl case at the end. Every other check in this
# chain drives the engine through a harness that speaks the protocol properly;
# the bug being fixed was reported by a person closing a browser tab, and that
# path -- an HTTP client that just stops reading -- goes through code (the
# gateway's disconnect detection, then CANCEL, then the engine's poll) that no
# harness exercises end to end. `curl -m 8` is the cheapest honest version of
# it, and the request AFTER it is the measurement: seconds, not minutes.
#
# This chain changes NO shader, so the revert is the binary alone -- but the
# pristine shader set is still verified by sha before the gateway comes back.
set -u
TAG=cancel$(date +%m%d%H%M)
OUT=~/bench/cancel_chain_out; mkdir -p $OUT
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR=$OUT/ckpt; mkdir -p $COLI_CKPT_DIR
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.p5b
PRISTINE_SHA=65d7d70d96e248bc8ac0c04538694c4321c3344b0b08cc7b43eea07ca80bd94d
PSHADERS=~/bench/shaders_p5b
PSHADER_SHA=""          # filled from the SERVED shaders below, before anything is built
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders

start_gateway() {
  unset COLI_CKPT_DIR GLM53_PREFIX_CKPT GLM53_NO_CANCEL_POLL COLI_MLA_POOL
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  sleep 100
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}

revert_and_serve_pristine() {
  echo "--- REVERTING: $PRISTINE back in service"
  ~/bench/p7_stop.sh
  wait_no_engine || echo "  (engine still up; the cp may fail with ETXTBSY)"
  if [ -n "$PREMERGE" ]; then (cd ~/src/colibri && git reset --hard "$PREMERGE"); fi
  cp -f $PSHADERS/*.spv $SHADERS/ || echo "SHADER REVERT COPY FAILED"
  cp $PRISTINE ~/src/colibri/c/glm53 || echo "REVERT COPY FAILED"
  b=$(sha256sum ~/src/colibri/c/glm53 | cut -d" " -f1)
  s=$(sha256sum $SHADERS/kda_step.spv | cut -d" " -f1)
  if [ "$b" != "$PRISTINE_SHA" ]; then echo "REVERT FAILED: served binary is NOT the pristine one ($b)";
  else echo "revert verified (binary): $b"; fi
  if [ "$s" != "$PSHADER_SHA" ]; then echo "REVERT FAILED: kda_step.spv is NOT the pristine one ($s)";
  else echo "revert verified (shader): $s"; fi
  start_gateway
}

on_exit() {
  rc=$?
  if [ "$SERVED" != 1 ]; then
    echo "=== chain exiting rc=$rc without a served candidate"
    if ! pgrep -f "openai_[s]erver.py" >/dev/null; then revert_and_serve_pristine; fi
  fi
  echo "=== cancel chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== cancel chain start $(date -Is) tag=$TAG"
sha256sum $PRISTINE ~/src/colibri/c/glm53
[ "$(sha256sum $PRISTINE | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || { echo "PRISTINE SHA MISMATCH"; exit 2; }
# The pristine shader set is the one IN SERVICE right now, before anything is
# merged or built. Snapshot it here rather than trusting a directory that some
# earlier chain left behind.
# The .comp sources go too, not just the .spv: prefill_profile.sh checks for
# qmatmul.comp before it will run, and a snapshot of .spv alone made the
# pristine side of the smoke refuse (first chain, 2026-09-08 21:52).
mkdir -p $PSHADERS && cp -f $SHADERS/*.spv $SHADERS/*.comp $PSHADERS/ || { echo "SHADER SNAPSHOT FAILED"; exit 2; }
PSHADER_SHA=$(sha256sum $PSHADERS/kda_step.spv | cut -d" " -f1)
echo "pristine shaders snapshotted to $PSHADERS ($(ls $PSHADERS/*.spv | wc -l) files, kda_step $PSHADER_SHA)"

echo "--- stopping the gateway"
~/bench/p7_stop.sh || exit 2
wait_no_engine || exit 2

echo "--- merging p0-sync into hot-expert-tier"
cd ~/src/colibri || exit 2
git checkout hot-expert-tier || exit 2
PREMERGE=$(git rev-parse HEAD); echo "pre-merge HEAD $PREMERGE"
git merge --no-edit p0-sync || { echo "MERGE FAILED"; exit 2; }
git log --oneline -1

echo "--- building glm53 + qwen38 + qwen38-vk"
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|warning: .*cancel|glslc" | head -20
[ -x ~/src/colibri/c/glm53 ] || { echo "BUILD FAILED"; exit 2; }
sha256sum ~/src/colibri/c/glm53 $SHADERS/kda_step.spv
cmp -s $SHADERS/kda_step.spv $PSHADERS/kda_step.spv && echo "shaders unchanged by the build" \
  || { echo "A SHADER CHANGED -- this item must not touch one"; revert_and_serve_pristine; exit 4; }

echo "--- the four qwen38 C tests + test_serve_poll (glm53.c now includes serve_poll.h)"
QT=0
for t in prefix config metrics serve_framing; do
  make -C c tests/test_qwen38_$t VK=1 >/dev/null 2>&1
  ./c/tests/test_qwen38_$t >/dev/null 2>&1; r=$?; echo "  test_qwen38_$t exit=$r"
  [ "$r" = 0 ] || QT=1
done
make -C c tests/test_serve_poll >/dev/null 2>&1
./c/tests/test_serve_poll >/dev/null 2>&1; r=$?; echo "  test_serve_poll exit=$r"
[ "$r" = 0 ] || QT=1
[ "$QT" = 0 ] || { echo "C TESTS FAILED"; revert_and_serve_pristine; exit 4; }

# --------------------------------------------------------------- smoke
echo
echo "### smoke — the 30-token prompt at the serving knob, poll on and poll off"
python3 - > ~/bench/prefill_prompt_30.txt <<'PY'
print("Summarise these notes in one sentence.\n\nThe rig has three GPUs.")
PY
SMOKE_OK=1
for off in 0 1; do
  wait_no_engine || exit 2
  echo "--- smoke GLM53_NO_CANCEL_POLL=$off"
  GLM53_NO_CANCEL_POLL=$off COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh ~/src/colibri/c/glm53 ~/bench/prefill_prompt_30.txt "$TAG-smoke$off" 128 \
    > $OUT/smoke_$off.txt 2>&1
  r=$?
  echo "  rc=$r"; tail -12 $OUT/smoke_$off.txt
  [ "$r" = 0 ] || { echo "  SMOKE FAIL: run rc=$r"; SMOKE_OK=0; }
  grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smoke$off.log > $OUT/tf_$off.txt
done
wait_no_engine || exit 2
echo "--- smoke pristine (its own shaders)"
COLI_VK_SHADERS=$PSHADERS COLI_KDA_GPU=2 \
  bash $HERE/prefill_profile.sh $PRISTINE ~/bench/prefill_prompt_30.txt "$TAG-smokepris" 128 \
  > $OUT/smoke_pris.txt 2>&1
echo "  rc=$?"; tail -12 $OUT/smoke_pris.txt
grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smokepris.log > $OUT/tf_pris.txt
cmp -s $OUT/tf_0.txt $OUT/tf_1.txt   && echo "  poll on vs off (same binary): IDENTICAL" || { echo "  SMOKE FAIL: the knob changes teacher forcing"; SMOKE_OK=0; }
cmp -s $OUT/tf_0.txt $OUT/tf_pris.txt && echo "  candidate vs pristine:       IDENTICAL" || { echo "  SMOKE FAIL: the candidate differs from the pristine binary"; SMOKE_OK=0; }
if [ "$SMOKE_OK" != 1 ]; then
  echo "=== SMOKE FAILED -- not spending hours on a broken binary"
  revert_and_serve_pristine; exit 4
fi

# --------------------------------------------------------------- the gate
echo
echo "### cancel_gate.sh (steps a, b, c, d)"
wait_no_engine || exit 2
PRISTINE_SHADERS=$PSHADERS MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  bash $HERE/cancel_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate.log
GATE=${PIPESTATUS[0]}; echo "CANCEL GATE rc=$GATE"

# --------------------------------------------------------------- verdict
echo
echo "=== CANCEL verdict: gate rc=$GATE"
wait_no_engine || exit 2
if [ "$GATE" != 0 ]; then
  echo "=== THE GATE FAILED -- reverting to $PRISTINE"
  revert_and_serve_pristine
  exit 3
fi

SERVED=1
echo "=== CANCEL IN SERVICE: $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
start_gateway

# --------------------------------------------------------------- (e) live
echo
echo "### (e) live, through the gateway"
python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --sizes 1000 --repeat 1 --warm \
    --cancel 5 --cancel-phase prefill \
    --tag $TAG-live --json $OUT/live.jsonl 2>&1 | tee $OUT/live.txt | tail -25
LIVE=${PIPESTATUS[0]}; echo "live cancel rc=$LIVE"
~/bench/owui_report.sh 3

echo
echo "### (e2) a real client disconnect: curl -m 8 on a 1 000-token completion"
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
MODEL=$(curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "model id: $MODEL"
python3 - "$MODEL" > $OUT/curl_big.json <<'PY'
import json, sys
text = open("/home/ronald/src/colibri/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md").read()
user = ("Read the following notes and answer in one sentence: what machine "
        "are they about?\n\n" + text[:4200])
print(json.dumps({"model": sys.argv[1], "stream": True, "max_tokens": 256,
                  "temperature": 0,
                  "messages": [{"role": "user", "content": user}]}))
PY
python3 - "$MODEL" > $OUT/curl_small.json <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "stream": False, "max_tokens": 8,
                  "temperature": 0,
                  "messages": [{"role": "user", "content": "Say OK."}]}))
PY
echo "--- curl -m 8 (the browser tab closing)"
date -Is
curl -s -m 8 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d @$OUT/curl_big.json http://127.0.0.1:8081/v1/chat/completions > $OUT/curl_big.out
rc=$?
echo "curl exit=$rc (28 = timed out and hung up, which is the point)"
date -Is
echo "--- the short request that follows: its ttft is the measurement"
S=$(date +%s.%N)
curl -s -m 300 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d @$OUT/curl_small.json http://127.0.0.1:8081/v1/chat/completions > $OUT/curl_small.out
rc=$?
E=$(date +%s.%N)
echo "curl exit=$rc"
python3 -c "print(f'short request after the disconnect: {$E - $S:.2f} s')"
head -c 400 $OUT/curl_small.out; echo
echo "--- the gateway's own record"
~/bench/owui_report.sh 4
grep -E "CANCEL [0-9]+ at " ~/glm53_server.log | tail -5
