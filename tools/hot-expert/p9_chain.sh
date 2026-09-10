#!/bin/bash
# P9 chain: the conversation ledger. Structure is p8_chain.sh's, exactly, for the
# same reasons:
#   snapshot the pristine -> verify it -> wait for nothing in flight -> stop the
#   gateway -> merge p0-sync -> build -> C tests + the gateway's python tests ->
#   p9_gate.sh steps 1,2 (engine offline) -> restart the gateway on the merged
#   tree -> p9_gate.sh steps 3,4 (live) -> accept_live.sh -> verified revert on
#   any failure -> the gateway restarted on EVERY exit path.
#
# The one structural difference from P8: P9 DOES change the engine -- by one
# trailing field on the DONE STAT line -- so the built binary is expected to
# differ from the one in service, step 1's speed verdict is judged at
# MIN_SPEEDUP rather than downgraded, and the pristine snapshot has to be taken
# BEFORE anything stops, because the binary in service is the one this item is
# measured against.
set -u
TAG=p9$(date +%m%d%H%M)
OUT=~/bench/p9_chain_out; mkdir -p $OUT
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.p8
PSHADERS=~/bench/shaders_p8
PRISTINE_SHA=""
PSHADER_SHA=""
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders
LOG=~/glm53_server.log

start_gateway() {   # the documented restart, with no gate knob leaking into it
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      -u COLI_LEDGER -u COLI_PREFIX_PIN \
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
  echo "=== p9 chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p9 chain start $(date -Is) tag=$TAG"

# --- the pristine snapshot, taken BEFORE anything stops --------------------
# The binary in service right now IS the pristine for this item (P8, gateway-only,
# byte-identical to P7b's f68d1cac...). It has no snapshot of its own yet, so the
# chain takes one and then verifies it, rather than trusting a name.
echo "--- snapshotting the binary in service as $PRISTINE"
cp -p ~/src/colibri/c/glm53 $PRISTINE || { echo "SNAPSHOT FAILED"; exit 2; }
mkdir -p $PSHADERS && cp -p $SHADERS/*.spv $PSHADERS/ || { echo "SHADER SNAPSHOT FAILED"; exit 2; }
cmp -s $PRISTINE ~/src/colibri/c/glm53 || { echo "SNAPSHOT DOES NOT MATCH THE TREE"; exit 2; }
cmp -s $PSHADERS/kda_step.spv $SHADERS/kda_step.spv || { echo "SHADER SNAPSHOT MISMATCH"; exit 2; }
PRISTINE_SHA=$(sha256sum $PRISTINE | cut -d" " -f1)
PSHADER_SHA=$(sha256sum $PSHADERS/kda_step.spv | cut -d" " -f1)
echo "pristine: $PRISTINE $PRISTINE_SHA"
echo "pristine shaders: $PSHADERS ($(ls $PSHADERS/*.spv | wc -l) files, kda_step $PSHADER_SHA)"
echo "tree in service: $(cd ~/src/colibri && git log --oneline -1)"

echo "--- the owner's last requests (nothing may be in flight)"
~/bench/owui_report.sh 2
# `grep -c` prints 0 when it matches nothing AND exits 1 doing it, so a
# `|| echo 0` here appends a SECOND zero and the comparison is never numeric --
# the chain then parks in this loop for an hour (measured 2026-09-09 20:30).
for _ in $(seq 1 360); do
  posts=$(grep -c "POST /v1/chat/completions\|POST /v1/completions" $LOG 2>/dev/null)
  reqs=$(grep -c "\[req\] " $LOG 2>/dev/null)
  [ "${posts:-0}" -le "${reqs:-0}" ] && break
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
  echo "WARNING: the built binary is IDENTICAL to the pristine -- P9 adds one STAT"
  echo "         field to c/glm53.c, so an identical binary means the build did not take"
else
  echo "the built binary differs from the pristine one, as expected (the 8th STAT field)"
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
echo "--- the gateway's python tests (the file P9 changes)"
(cd ~/src/colibri/c && python3 -m unittest tests.test_glm53_ledger tests.test_glm53_reply_pin \
   tests.test_openai_server tests.test_anthropic_messages tests.test_chat_thinking \
   tests.test_openai_tools_e2e > $OUT/pytests.txt 2>&1); r=$?
echo "  python gateway tests exit=$r ($(grep -E '^Ran [0-9]+ tests' $OUT/pytests.txt))"
[ "$r" = 0 ] || QT=1
echo "--- the glm53 renderer against the checkpoint's own chat_template.jinja"
(cd ~/src/colibri/c && python3 tests/test_glm53_chat_template.py \
   --template ~/models/GLM-5.3-Flash-colibri-int4-g64/chat_template.jinja) > $OUT/template.txt 2>&1
r=$?; echo "  chat template exit=$r ($(tail -1 $OUT/template.txt | cut -c1-100))"
[ "$r" = 0 ] || QT=1
[ "$QT" = 0 ] || { echo "TESTS FAILED (see $OUT/)"; revert_and_serve_pristine; exit 4; }

# ------------------------------------------------- the gate, offline half
echo
echo "### p9_gate.sh steps 1 and 2 (the engine gains one STAT field and nothing else)"
wait_no_engine || exit 2
GATE_STEPS=12 MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} PROFILE_MIN_RESIDENT=${PROFILE_MIN_RESIDENT:-97} \
  bash $HERE/p9_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate12.log
G12=${PIPESTATUS[0]}; echo "P9 GATE (steps 1,2) rc=$G12"
wait_no_engine || exit 2
if [ "$G12" != 0 ]; then
  echo "=== THE OFFLINE HALF FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------------- serve, then the live half
echo
echo "=== serving the merged tree"
start_gateway
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "GATEWAY DID NOT START"; revert_and_serve_pristine; exit 3; }

echo
echo "### p9_gate.sh steps 3 and 4 (three behaviours x three arms, then divergence)"
GATE_STEPS=34 P9_SEEDS=${P9_SEEDS:-3} bash $HERE/p9_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate34.log
G34=${PIPESTATUS[0]}; echo "P9 GATE (steps 3,4) rc=$G34"
if [ "$G34" != 0 ]; then
  echo "=== THE LIVE HALF FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------------- acceptance on the user's path
echo
echo "### accept_live.sh — the four checks on Open WebUI's own backend, plus the ledger's alarm"
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
echo "=== P9 IN SERVICE: tree $(cd ~/src/colibri && git log --oneline -1)"
echo "=== binary $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
echo "=== still owed from the Mac: p9_ui_multiturn.sh (step 5), accept_ui.sh, ui/ui_matrix.sh"
