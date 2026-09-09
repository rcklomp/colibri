#!/bin/bash
# P7b chain: glm53 captures the rest of the stable prefix AFTER a restore.
#
# Structure is cancel_chain.sh's / p5b_chain.sh's, and for the same reasons:
#   verify the pristine snapshot -> stop gateway -> merge p0-sync -> build ->
#   qwen38 C tests + test_serve_poll + the glm53 frame test -> p7b_gate.sh ->
#   serve the candidate only on rc 0 -> the LIVE check through Open WebUI's own
#   backend -> verified sha256 revert on any failure -> the gateway restarted on
#   EVERY exit path.
#
# The live step is a gate, not a printout (cancel_chain.sh's lesson): what the
# owner needs from P7b is that a NEW chat in his browser starts warm, and only
# a UI-shaped turn can show that. ~/bench/owui_ui_warm.sh sends one from a
# throwaway saved chat (saved chat + session_id + memory on, the exact shape the
# browser sends) and deletes it afterwards. Two runs with DIFFERENT questions:
# with an identical question the second turn could be answered by slot reuse,
# and the claim here is about the shared prefix, not the slot.
#
# This chain changes NO shader, so the revert is the binary alone -- but the
# pristine shader set is still verified by sha before the gateway comes back.
set -u
TAG=p7b$(date +%m%d%H%M)
OUT=~/bench/p7b_chain_out; mkdir -p $OUT
# A private checkpoint directory for the whole gate: the served
# $SNAP/.coli_ckpt (the gateway's own checkpoints, including the partial 2 163
# one this item is about) must survive the gate untouched.
export COLI_CKPT_DIR=$OUT/ckpt; mkdir -p $COLI_CKPT_DIR
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.cancel
PRISTINE_SHA=f41fc5cc5931c248df5098d20f70ec220ad95ca485b580f0e4eb906b4ba28e88
PSHADERS=~/bench/shaders_cancel
PSHADER_SHA=""          # from the SERVED shader snapshot, checked below
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders

start_gateway() {
  unset COLI_CKPT_DIR GLM53_PREFIX_CKPT COLI_MLA_POOL
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
  echo "=== p7b chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p7b chain start $(date -Is) tag=$TAG"
sha256sum $PRISTINE ~/src/colibri/c/glm53
[ "$(sha256sum $PRISTINE | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || { echo "PRISTINE SHA MISMATCH"; exit 2; }
# The pristine shaders were snapshotted from the SERVED tree before this chain
# stopped anything (2026-09-09, 9 .spv). Verify they are still that set.
[ -s $PSHADERS/kda_step.spv ] || { echo "NO PRISTINE SHADER SNAPSHOT in $PSHADERS"; exit 2; }
PSHADER_SHA=$(sha256sum $PSHADERS/kda_step.spv | cut -d" " -f1)
cmp -s $SHADERS/kda_step.spv $PSHADERS/kda_step.spv \
  || { echo "the snapshot does not match the served shaders"; exit 2; }
echo "pristine shaders: $PSHADERS ($(ls $PSHADERS/*.spv | wc -l) files, kda_step $PSHADER_SHA)"

echo "--- the owner's last requests (nothing may be in flight)"
~/bench/owui_report.sh 2

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
[ "$QT" = 0 ] || { echo "C TESTS FAILED"; revert_and_serve_pristine; exit 4; }

# --------------------------------------------------------------- the gate
echo
echo "### p7b_gate.sh (steps 1, 2, 3a, 3b, 3c)"
wait_no_engine || exit 2
PRISTINE_SHADERS=$PSHADERS MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  bash $HERE/p7b_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate.log
GATE=${PIPESTATUS[0]}; echo "P7B GATE rc=$GATE"

echo
echo "=== P7b verdict: gate rc=$GATE"
wait_no_engine || exit 2
if [ "$GATE" != 0 ]; then
  echo "=== THE GATE FAILED -- reverting to $PRISTINE"
  revert_and_serve_pristine
  exit 3
fi

SERVED=1
echo "=== P7b IN SERVICE: $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
start_gateway

# --------------------------------------------------------------- live
echo
echo "### live — a UI-shaped turn through Open WebUI's own backend, twice"
# Run 1 pays whatever the served checkpoints cannot cover and must CAPTURE the
# rest of the prefix (the `CKPT plan ... after-restore` line when a partial
# checkpoint was restored). Run 2 is the owner's actual need: a new chat that
# starts warm. Different questions, so slot reuse cannot be what makes run 2
# fast.
LIVEFAIL=0
for round in 1 2; do
  q="Say OK."; [ $round = 2 ] && q="Reply with the single word ready."
  echo "--- live run $round: $q"
  date -Is
  ~/bench/owui_ui_warm.sh "$q" 2>&1 | tee $OUT/live_$round.txt
  date -Is
  ~/bench/owui_report.sh 1 | tee $OUT/report_$round.txt
done
echo "--- the engine's own checkpoint lines from this pair"
grep -E "CKPT (hit|plan|store)" ~/glm53_server.log | tail -8
if grep -q "after-restore" $OUT/live_1.txt ~/glm53_server.log; then
  echo "live: the engine planned a capture AFTER a restore (the P7b line)"
else
  echo 'live: NOTE -- no after-restore line; run 1 had no partial checkpoint to extend'
fi
# The bound is run 2: a new chat, a prefix it has never seen in this
# conversation, and it must start warm. 4 000 reused tokens and a ttft in tens
# of seconds instead of the 380-713 s measured on 2026-09-09 07:17/07:24.
python3 - $OUT/report_2.txt <<'PY' || LIVEFAIL=1
import re, sys
body = open(sys.argv[1]).read().strip()
line = body.splitlines()[-1] if body else ""
m = re.search(r"prompt\s+(\d+) tok\s+reused\s+(\d+)\s+ttft\s+([\d.]+)s", line)
print("live run 2:", line)
if not m:
    print("live: FAIL -- no [req] line to read"); sys.exit(1)
prompt, reused, ttft = int(m.group(1)), int(m.group(2)), float(m.group(3))
ok = reused >= 4000 and ttft < 60
print(f"live run 2: reused {reused}/{prompt} in {ttft:.2f}s -> {'PASS' if ok else 'FAIL'} "
      f"(bounds: reused >= 4000, ttft < 60 s)")
sys.exit(0 if ok else 1)
PY

if [ "$LIVEFAIL" != 0 ]; then
  echo
  echo "=== THE LIVE STEP FAILED -- the gate passed but a new chat does not start warm."
  echo "    Reverting to $PRISTINE."
  SERVED=0
  revert_and_serve_pristine
  exit 5
fi
echo
echo "=== live step PASSED; $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16) stays in service"
