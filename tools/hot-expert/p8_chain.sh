#!/bin/bash
# P8 chain: the reply pin. Structure is p7b_chain.sh's, for the same reasons:
#   verify the pristine snapshot -> wait for nothing in flight -> stop gateway ->
#   merge p0-sync -> build -> C tests + the gateway's python tests ->
#   p8_gate.sh steps 1,2 (engine offline) -> restart the gateway on the merged
#   tree -> p8_gate.sh steps 3,4 (live, through Open WebUI's own backend) ->
#   accept_live.sh -> verified revert on any failure -> the gateway restarted on
#   EVERY exit path.
#
# The one structural difference: P8 changes c/openai_server.py and NOTHING the
# compiler sees, so "serving the candidate" is restarting the gateway on the
# merged tree, and the binary must come out of the build byte-identical to the
# one already in service. The chain checks that by sha before it trusts
# anything else.
set -u
TAG=p8$(date +%m%d%H%M)
OUT=~/bench/p8_chain_out; mkdir -p $OUT
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.p7b
PRISTINE_SHA=f68d1cacbd4afa86de4b6695f57841367941c3687079049b7629db27fb1c7a21
PSHADERS=~/bench/shaders_p7b
PSHADER_SHA=""
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders
LOG=~/glm53_server.log

start_gateway() {   # the documented restart, with no gate knob leaking into it
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > $LOG 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat ~/.colibri_api_key)" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}

revert_and_serve_pristine() {
  echo "--- REVERTING: $PRISTINE and the pre-merge tree back in service"
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
  echo "revert verified (tree): $(cd ~/src/colibri && git log --oneline -1)"
  start_gateway
}

on_exit() {
  rc=$?
  if [ "$SERVED" != 1 ]; then
    echo "=== chain exiting rc=$rc without a served candidate"
    if ! pgrep -f "openai_[s]erver.py" >/dev/null; then revert_and_serve_pristine; fi
  fi
  echo "=== p8 chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p8 chain start $(date -Is) tag=$TAG"
sha256sum $PRISTINE ~/src/colibri/c/glm53
[ "$(sha256sum $PRISTINE | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || { echo "PRISTINE SHA MISMATCH"; exit 2; }
[ -s $PSHADERS/kda_step.spv ] || { echo "NO PRISTINE SHADER SNAPSHOT in $PSHADERS"; exit 2; }
PSHADER_SHA=$(sha256sum $PSHADERS/kda_step.spv | cut -d" " -f1)
cmp -s $SHADERS/kda_step.spv $PSHADERS/kda_step.spv \
  || { echo "the snapshot does not match the served shaders"; exit 2; }
echo "pristine shaders: $PSHADERS ($(ls $PSHADERS/*.spv | wc -l) files, kda_step $PSHADER_SHA)"

echo "--- the owner's last requests (nothing may be in flight)"
~/bench/owui_report.sh 2
for _ in $(seq 1 360); do
  posts=$(grep -c "POST /v1/chat/completions\|POST /v1/completions" $LOG 2>/dev/null || echo 0)
  reqs=$(grep -c "\[req\] " $LOG 2>/dev/null || echo 0)
  [ "$posts" -le "$reqs" ] && break
  echo "    a request is in flight ($posts posted, $reqs finished) -- waiting"
  sleep 10
done

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
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|glslc" | head -20
[ -x ~/src/colibri/c/glm53 ] || { echo "BUILD FAILED"; exit 2; }
sha256sum ~/src/colibri/c/glm53 $SHADERS/kda_step.spv
cmp -s $SHADERS/kda_step.spv $PSHADERS/kda_step.spv && echo "shaders unchanged by the build" \
  || { echo "A SHADER CHANGED -- this item must not touch one"; revert_and_serve_pristine; exit 4; }
if [ "$(sha256sum ~/src/colibri/c/glm53 | cut -d' ' -f1)" = "$PRISTINE_SHA" ]; then
  echo "binary unchanged by the build (P8 is a gateway-only item, as claimed)"
else
  echo "NOTE: the built binary differs from the pristine one -- step 1 of the gate decides"
fi

echo "--- the four qwen38 C tests + test_serve_poll + the glm53 frame test"
QT=0
for t in prefix config metrics serve_framing; do
  make -C c tests/test_qwen38_$t VK=1 >/dev/null 2>&1
  ./c/tests/test_qwen38_$t >/dev/null 2>&1; r=$?; echo "  test_qwen38_$t exit=$r"
  [ "$r" = 0 ] || QT=1
done
for t in test_serve_poll test_glm53_cancel_frames; do
  make -C c tests/$t >/dev/null 2>&1
  ./c/tests/$t >/dev/null 2>&1; r=$?; echo "  $t exit=$r"
  [ "$r" = 0 ] || QT=1
done
echo "--- the gateway's python tests (the file P8 changes)"
(cd ~/src/colibri/c && python3 -m unittest tests.test_glm53_reply_pin tests.test_openai_server \
   tests.test_anthropic_messages tests.test_chat_thinking tests.test_openai_tools_e2e \
   > $OUT/pytests.txt 2>&1); r=$?
echo "  python gateway tests exit=$r ($(grep -E '^Ran [0-9]+ tests' $OUT/pytests.txt))"
[ "$r" = 0 ] || QT=1
[ "$QT" = 0 ] || { echo "TESTS FAILED (see $OUT/pytests.txt)"; revert_and_serve_pristine; exit 4; }

# ------------------------------------------------- the gate, offline half
echo
echo "### p8_gate.sh steps 1 and 2 (the engine must be exactly what it was)"
wait_no_engine || exit 2
GATE_STEPS=12 MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  bash $HERE/p8_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate12.log
G12=${PIPESTATUS[0]}; echo "P8 GATE (steps 1,2) rc=$G12"
wait_no_engine || exit 2
if [ "$G12" != 0 ]; then
  echo "=== THE OFFLINE HALF FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------------- serve, then the live half
echo
echo "=== serving the merged tree (gateway restart; the binary is unchanged)"
start_gateway
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "GATEWAY DID NOT START"; revert_and_serve_pristine; exit 3; }

echo
echo "### p8_gate.sh steps 3 and 4 (the regression, live, and text identity)"
GATE_STEPS=34 bash $HERE/p8_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate34.log
G34=${PIPESTATUS[0]}; echo "P8 GATE (steps 3,4) rc=$G34"
if [ "$G34" != 0 ]; then
  echo "=== THE LIVE HALF FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------------- acceptance on the user's path
echo
echo "### accept_live.sh — the four checks on Open WebUI's own backend"
pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
bash $HERE/accept_live.sh 2>&1 | tee $OUT/accept_live.log
AL=${PIPESTATUS[0]}; echo "accept_live rc=$AL"
if [ "$AL" != 0 ]; then
  echo "=== accept_live FAILED -- reverting"
  revert_and_serve_pristine
  exit 5
fi

SERVED=1
echo
echo "=== P8 IN SERVICE: tree $(cd ~/src/colibri && git log --oneline -1)"
echo "=== binary $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
