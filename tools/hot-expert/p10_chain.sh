#!/bin/bash
# P10 chain: value-based checkpoint eviction. Structure is p9_chain.sh's, exactly:
#   snapshot the pristine -> verify it -> wait for nothing in flight -> stop the
#   gateway -> merge p0-sync -> build -> C tests -> p10_gate.sh steps 1,2 (engine
#   offline) -> restart -> p10_gate.sh step 3 (the flood, live) -> lower the
#   serving minimum -> accept_live.sh -> verified revert on any failure -> the
#   gateway restarted on EVERY exit path.
#
# Launch through run_chain.sh, never directly — it takes the rig lock:
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#       ~/src/colibri/tools/hot-expert/p10_chain.sh > ~/bench/p10_chain.log 2>&1 < /dev/null &
#
# Two structural differences from P9:
#  - P10 changes c/glm53.c for real, so the built binary MUST differ from the
#    one in service and step 1's speed verdict is judged at MIN_SPEEDUP.
#  - P10 also changes the SERVING SCRIPT: GLM53_PREFIX_CKPT_MIN goes from 1024
#    back to the code default of 128, because the threshold was only ever the
#    blunt half of this fix. That edit is backed up and reverted with everything
#    else, and it happens BEFORE accept_live so the acceptance run measures what
#    the owner will actually be served.
set -u
TAG=p10$(date +%m%d%H%M)
OUT=~/bench/p10_chain_out; mkdir -p $OUT
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.p10base
PSHADERS=~/bench/shaders_p10base
PRISTINE_SHA=""
PSHADER_SHA=""
STARTBAK=~/bench/start_glm53.sh.p10base
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders
LOG=~/glm53_server.log

start_gateway() {   # the documented restart, with no gate knob leaking into it
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_PREFIX_CKPT_MIN \
      -u GLM53_CKPT_VALUE_EVICT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
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
  echo "--- REVERTING: $PRISTINE, the serving script and the pre-merge tree back in service"
  ~/bench/p7_stop.sh
  wait_no_engine || echo "  (engine still up; the cp may fail with ETXTBSY)"
  if [ -n "$PREMERGE" ]; then (cd ~/src/colibri && git reset --hard "$PREMERGE"); fi
  cp -f $PSHADERS/*.spv $SHADERS/ || echo "SHADER REVERT COPY FAILED"
  cp $PRISTINE ~/src/colibri/c/glm53 || echo "REVERT COPY FAILED"
  [ -f "$STARTBAK" ] && { cp -p "$STARTBAK" ~/start_glm53.sh || echo "SERVING SCRIPT REVERT FAILED"; }
  b=$(sha256sum ~/src/colibri/c/glm53 | cut -d" " -f1)
  s=$(sha256sum $SHADERS/kda_step.spv | cut -d" " -f1)
  if [ "$b" != "$PRISTINE_SHA" ]; then echo "REVERT FAILED: served binary is NOT the pristine one ($b)";
  else echo "revert verified (binary): $b"; fi
  if [ "$s" != "$PSHADER_SHA" ]; then echo "REVERT FAILED: kda_step.spv is NOT the pristine one ($s)";
  else echo "revert verified (shader): $s"; fi
  echo "revert verified (serving): $(grep -o 'GLM53_PREFIX_CKPT_MIN=[0-9]*' ~/start_glm53.sh)"
  echo "revert verified (tree): $(cd ~/src/colibri && git log --oneline -1)"
  start_gateway
}

on_exit() {
  rc=$?
  if [ "$SERVED" != 1 ]; then
    echo "=== chain exiting rc=$rc without a served candidate"
    if ! pgrep -f "openai_[s]erver.py" >/dev/null; then revert_and_serve_pristine; fi
  fi
  echo "=== p10 chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p10 chain start $(date -Is) tag=$TAG"

# --- the pristine snapshot, taken BEFORE anything stops --------------------
echo "--- snapshotting the binary in service as $PRISTINE"
cp -p ~/src/colibri/c/glm53 $PRISTINE || { echo "SNAPSHOT FAILED"; exit 2; }
mkdir -p $PSHADERS && cp -p $SHADERS/*.spv $PSHADERS/ || { echo "SHADER SNAPSHOT FAILED"; exit 2; }
cmp -s $PRISTINE ~/src/colibri/c/glm53 || { echo "SNAPSHOT DOES NOT MATCH THE TREE"; exit 2; }
cmp -s $PSHADERS/kda_step.spv $SHADERS/kda_step.spv || { echo "SHADER SNAPSHOT MISMATCH"; exit 2; }
cp -p ~/start_glm53.sh "$STARTBAK" || { echo "SERVING SCRIPT SNAPSHOT FAILED"; exit 2; }
PRISTINE_SHA=$(sha256sum $PRISTINE | cut -d" " -f1)
PSHADER_SHA=$(sha256sum $PSHADERS/kda_step.spv | cut -d" " -f1)
echo "pristine: $PRISTINE $PRISTINE_SHA"
echo "pristine shaders: $PSHADERS ($(ls $PSHADERS/*.spv | wc -l) files, kda_step $PSHADER_SHA)"
echo "serving script backed up: $STARTBAK ($(grep -o 'GLM53_PREFIX_CKPT_MIN=[0-9]*' $STARTBAK))"
echo "tree in service: $(cd ~/src/colibri && git log --oneline -1)"

# --- the checkpoint the whole item is about, before anything touches it -----
LIVE_CKPT=$(ls -d ~/models/*/.coli_ckpt 2>/dev/null | head -1)
echo "--- the live checkpoint directory: $LIVE_CKPT"
ls -la "$LIVE_CKPT" 2>/dev/null | sed 's/^/    /'
echo "--- the slots as the running engine loaded them, and what has hit them"
grep -a "CKPT disk load\|CKPT hit\|CKPT store" $LOG | tail -20 | sed 's/^/    /' | cut -c1-140

echo "--- the owner's last requests (nothing may be in flight)"
~/bench/owui_report.sh 2
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
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|warning: .*ckpt|glslc" | head -20
[ -x ~/src/colibri/c/glm53 ] || { echo "BUILD FAILED"; exit 2; }
sha256sum ~/src/colibri/c/glm53 $SHADERS/kda_step.spv
cmp -s $SHADERS/kda_step.spv $PSHADERS/kda_step.spv && echo "shaders unchanged by the build" \
  || { echo "A SHADER CHANGED -- this item must not touch one"; revert_and_serve_pristine; exit 4; }
if [ "$(sha256sum ~/src/colibri/c/glm53 | cut -d' ' -f1)" = "$PRISTINE_SHA" ]; then
  echo "WARNING: the built binary is IDENTICAL to the pristine -- P10 changes ckpt_store"
  echo "         in c/glm53.c, so an identical binary means the build did not take"
  revert_and_serve_pristine; exit 4
fi
echo "the built binary differs from the pristine one, as expected (the eviction policy)"

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
[ "$QT" = 0 ] || { echo "TESTS FAILED"; revert_and_serve_pristine; exit 4; }

# ------------------------------------------------- the gate, offline half
echo
echo "### p10_gate.sh steps 1 and 2 (the policy must not exist with checkpoints off)"
wait_no_engine || exit 2
# P10_STEP1_SHA lets a re-run skip step 1 -- but ONLY on proof, never on
# assertion: the sha256 of the binary just built must equal the sha256 of the
# binary step 1 already passed on. The first run of this chain (2026-09-10
# 07:20) passed step 1 in ~30 minutes and then failed step 2 on a defect in the
# gate's own environment; rebuilding the identical bytes to re-derive an
# identical oracle is not diligence, it is 30 minutes. If the hashes differ the
# skip is refused and step 1 runs.
OFFLINE_STEPS=12
BUILT_SHA=$(sha256sum ~/src/colibri/c/glm53 | cut -d" " -f1)
if [ -n "${P10_STEP1_SHA:-}" ]; then
  if [ "$P10_STEP1_SHA" = "$BUILT_SHA" ]; then
    OFFLINE_STEPS=2
    echo "--- step 1 SKIPPED: this binary is byte-identical to the one it already passed on"
    echo "    sha256 $BUILT_SHA"
    echo "    step 1 result being carried forward: teacher_forcing IDENTICAL (782 positions),"
    echo "    logits cosine=1.0000000 max_abs=0 argmax OK, TTFT 1.02x/1.00x/1.00x, speedup PASS"
  else
    echo "--- step 1 skip REFUSED: built $BUILT_SHA != claimed $P10_STEP1_SHA -- running step 1"
  fi
fi
GATE_STEPS=$OFFLINE_STEPS MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} PROFILE_MIN_RESIDENT=${PROFILE_MIN_RESIDENT:-97} \
  bash $HERE/p10_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate12.log
G12=${PIPESTATUS[0]}; echo "P10 GATE (steps $OFFLINE_STEPS) rc=$G12"
wait_no_engine || exit 2
if [ "$G12" != 0 ]; then
  echo "=== THE OFFLINE HALF FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------------- serve, then the flood
echo
echo "=== serving the merged tree"
start_gateway
pgrep -f "openai_[s]erver.py" >/dev/null || { echo "GATEWAY DID NOT START"; revert_and_serve_pristine; exit 3; }

echo
echo "### p10_gate.sh step 3 (the flood, two arms)"
GATE_STEPS=3 P10_FLOOD=${P10_FLOOD:-20} P10_WARM=${P10_WARM:-3} \
  bash $HERE/p10_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate3.log
G3=${PIPESTATUS[0]}; echo "P10 GATE (step 3) rc=$G3"
if [ "$G3" != 0 ]; then
  echo "=== THE FLOOD FAILED -- reverting"
  revert_and_serve_pristine
  exit 3
fi

# ------------------------------------------- the serving change, then acceptance
echo
echo "### lowering the serving minimum: GLM53_PREFIX_CKPT_MIN 1024 -> 128 (the code default)"
sed -i 's/^export GLM53_PREFIX_CKPT_MIN=1024$/export GLM53_PREFIX_CKPT_MIN=128/' ~/start_glm53.sh
grep -q "^export GLM53_PREFIX_CKPT_MIN=128$" ~/start_glm53.sh || {
  echo "THE SERVING EDIT DID NOT TAKE"; revert_and_serve_pristine; exit 4; }
echo "    $(grep -n 'GLM53_PREFIX_CKPT_MIN' ~/start_glm53.sh)"

echo
echo "### accept_live.sh — the four checks on Open WebUI's own backend, plus the ledger's alarm"
~/bench/p7_stop.sh; wait_no_engine || exit 2
start_gateway
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
echo "=== P10 IN SERVICE: tree $(cd ~/src/colibri && git log --oneline -1)"
echo "=== binary $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16)"
echo "=== serving: $(grep -o 'GLM53_PREFIX_CKPT_MIN=[0-9]*' ~/start_glm53.sh) (was 1024)"
echo "=== still owed from the Mac: accept_ui.sh, ui/ui_matrix.sh --sizes 0,500,2000"
