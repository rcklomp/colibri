#!/bin/bash
# P5b chain: the MLA weighted pool.
#
# Structure is p5_chain.sh's:
#   stop gateway -> merge p0-sync -> build -> qwen38 C tests -> smoke ->
#   per-bucket profiles at 600 and 3 000 tokens for COLI_MLA_POOL 0/1/2 ->
#   p5b_gate.sh (a, a2, c, d) -> serve the candidate only on rc 0 -> live rows
#   (WITH --warm; p5's chain forgot it and was refused at 91.7 % residency) ->
#   verified revert on any failure -> the gateway restarted on EVERY exit path.
#
# P5b changes NO shader, so the revert is the binary alone -- but the pristine
# shader set is still verified by sha before the gateway comes back, and
# p5b_gate.sh refuses if any .spv differs from ~/bench/shaders_devmerge.
set -u
TAG=p5b$(date +%m%d%H%M)
OUT=~/bench/p5b_chain_out; mkdir -p $OUT
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR=$OUT/ckpt; mkdir -p $COLI_CKPT_DIR
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.devmerge
PRISTINE_SHA=5ed096a6b57a12350e804ed6858019ff7be72d52febb464c1276055b65a1677d
PSHADERS=~/bench/shaders_devmerge
PSHADER_SHA=b7d57f05b6f9ee48b61288ea76ffb4bd9fe25386bc8c7f0a1a6f0ff78e546478
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders

start_gateway() {
  unset COLI_CKPT_DIR GLM53_PREFIX_CKPT COLI_MLA_HEADVEC COLI_KDA_ROWS COLI_ROUTER_LANES COLI_MLA_POOL
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  sleep 100
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}

revert_and_serve_pristine() {
  echo "--- REVERTING: ~/bench/glm53.devmerge back in service"
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
  echo "=== p5b chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p5b chain start $(date -Is) tag=$TAG"
sha256sum $PRISTINE ~/src/colibri/c/glm53 $PSHADERS/kda_step.spv
[ "$(sha256sum $PRISTINE | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || { echo "PRISTINE SHA MISMATCH"; exit 2; }
[ "$(sha256sum $PSHADERS/kda_step.spv | cut -d' ' -f1)" = "$PSHADER_SHA" ] || { echo "PRISTINE SHADER SHA MISMATCH"; exit 2; }

echo "--- stopping the gateway"
~/bench/p7_stop.sh || exit 2
wait_no_engine || exit 2

echo "--- merging p0-sync into hot-expert-tier"
cd ~/src/colibri || exit 2
git checkout hot-expert-tier || exit 2
PREMERGE=$(git rev-parse HEAD); echo "pre-merge HEAD $PREMERGE"
git merge --no-edit p0-sync || { echo "MERGE FAILED"; exit 2; }
git log --oneline -1

echo "--- building glm53 + qwen38 + qwen38-vk (only glm53.c changed; the others are the insurance)"
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|glslc" | head -20
[ -x ~/src/colibri/c/glm53 ] || { echo "BUILD FAILED"; exit 2; }
sha256sum ~/src/colibri/c/glm53 $SHADERS/kda_step.spv
echo "--- the four qwen38 C tests"
QT=0
for t in prefix config metrics serve_framing; do
  make -C c tests/test_qwen38_$t VK=1 >/dev/null 2>&1
  ./c/tests/test_qwen38_$t >/dev/null 2>&1; r=$?; echo "  test_qwen38_$t exit=$r"
  [ "$r" = 0 ] || QT=1
done
[ "$QT" = 0 ] || { echo "QWEN38 TESTS FAILED"; revert_and_serve_pristine; exit 4; }

# --------------------------------------------------------------- prompts
python3 - > ~/bench/prefill_prompt_3000.txt <<'PY'
t = open("/home/ronald/src/colibri/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md").read()
print("Summarise these notes in one sentence.\n\n" + t[:10500])
PY
python3 - > ~/bench/prefill_prompt_30.txt <<'PY'
print("Summarise these notes in one sentence.\n\nThe rig has three GPUs.")
PY
wc -c ~/bench/prefill_prompt_600.txt ~/bench/prefill_prompt_3000.txt

# --------------------------------------------------------------- smoke
echo
echo "### smoke — the 30-token prompt at the serving knob, COLI_MLA_POOL 2, 1, 0"
SMOKE_OK=1
for m in 2 1 0; do
  wait_no_engine || exit 2
  echo "--- smoke COLI_MLA_POOL=$m"
  COLI_MLA_POOL=$m COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh ~/src/colibri/c/glm53 ~/bench/prefill_prompt_30.txt "$TAG-smoke$m" 128 \
    > $OUT/smoke_$m.txt 2>&1
  r=$?
  echo "  rc=$r"; tail -20 $OUT/smoke_$m.txt
  [ "$r" = 0 ] || { echo "  SMOKE FAIL: run rc=$r"; SMOKE_OK=0; }
done
if [ "$SMOKE_OK" != 1 ]; then
  echo "=== SMOKE FAILED -- not spending four hours on a broken binary"
  revert_and_serve_pristine; exit 4
fi
wait_no_engine || exit 2
echo "--- smoke pristine (its own shaders), 30-token prompt at COLI_KDA_GPU=2"
COLI_VK_SHADERS=$PSHADERS COLI_KDA_GPU=2 \
  bash $HERE/prefill_profile.sh $PRISTINE ~/bench/prefill_prompt_30.txt "$TAG-smokepris" 128 \
  > $OUT/smoke_pris.txt 2>&1
echo "  rc=$?"; tail -20 $OUT/smoke_pris.txt
grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smokepris.log > $OUT/tf_pris.txt

echo "--- smoke oracle: the three knob settings against each other and against the pristine binary"
SOK=1
for m in 2 1 0; do grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smoke$m.log > $OUT/tf_$m.txt; done
cmp -s $OUT/tf_2.txt $OUT/tf_0.txt && echo "  pool=2 vs pool=0 (same binary): IDENTICAL" || { echo "  SMOKE FAIL: the blocked pool changes teacher forcing"; SOK=0; }
cmp -s $OUT/tf_1.txt $OUT/tf_0.txt && echo "  pool=1 vs pool=0 (same binary): IDENTICAL" || { echo "  SMOKE FAIL: the padded scratch changes teacher forcing"; SOK=0; }
cmp -s $OUT/tf_2.txt $OUT/tf_pris.txt && echo "  pool=2 vs pristine (KDA=2):    IDENTICAL" || { echo "  SMOKE FAIL: the candidate differs from the pristine binary"; SOK=0; }
if [ "$SOK" != 1 ]; then revert_and_serve_pristine; exit 4; fi

# --------------------------------------------------------------- profiles
echo
echo "### (b) per-bucket profiles at the serving knob, COLI_MLA_POOL 0 / 1 / 2"
prof() {   # <tag> <bin> <prompt> <poolmode> <shaders>
  wait_no_engine || exit 2
  echo "--- profile $1 (bin=$(basename $2) prompt=$3 COLI_MLA_POOL=$4)"
  COLI_VK_SHADERS=$5 COLI_MLA_POOL=$4 COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh $2 ~/bench/prefill_prompt_$3.txt "$TAG-$1" 128 \
    2>&1 | tee $OUT/prof_$1.txt | tail -24
}
for r in 1 2;   do prof "600_pristine_r$r" $PRISTINE 600 2 $PSHADERS; done
for r in 1 2 3; do prof "600_m0_r$r" ~/src/colibri/c/glm53 600 0 $SHADERS; done
for r in 1 2 3; do prof "600_m1_r$r" ~/src/colibri/c/glm53 600 1 $SHADERS; done
for r in 1 2 3; do prof "600_m2_r$r" ~/src/colibri/c/glm53 600 2 $SHADERS; done
prof "3000_pristine" $PRISTINE 3000 2 $PSHADERS
for r in 1 2 3; do prof "3000_m0_r$r" ~/src/colibri/c/glm53 3000 0 $SHADERS; done
for r in 1 2;   do prof "3000_m1_r$r" ~/src/colibri/c/glm53 3000 1 $SHADERS; done
for r in 1 2 3; do prof "3000_m2_r$r" ~/src/colibri/c/glm53 3000 2 $SHADERS; done

# --------------------------------------------------------------- the gate
echo
echo "### p5b_gate.sh (steps a, a2, c, d)"
wait_no_engine || exit 2
PRISTINE_SHADERS=$PSHADERS MIN_SPEEDUP=${MIN_SPEEDUP:-1.0} \
  bash $HERE/p5b_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate.log
GATE=${PIPESTATUS[0]}; echo "P5B GATE rc=$GATE"

# --------------------------------------------------------------- verdict
echo
echo "=== P5b verdict: gate rc=$GATE"
wait_no_engine || exit 2
if [ "$GATE" = 0 ]; then
  SERVED=1
  echo "=== P5b IN SERVICE: $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
  start_gateway
  echo "### (e) live rows through the gateway"
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --sizes 300,1000 --repeat 2 --warm \
      --tag $TAG-live --json $OUT/live.jsonl 2>&1 | tee $OUT/live.txt | tail -20
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --multiturn --side-request --warm \
      --system ~/bench/p6_system.txt --sizes 300 --repeat 0 --gen 16 \
      --tag $TAG-mt --json $OUT/live.jsonl 2>&1 | tee $OUT/live_mt.txt | tail -20
  ~/bench/owui_report.sh 6
else
  echo "=== THE GATE FAILED -- reverting to ~/bench/glm53.devmerge"
  revert_and_serve_pristine
  exit 3
fi
