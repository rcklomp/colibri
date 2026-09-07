#!/bin/bash
# P5 chain: the sequential remainder (MLA attention score pass, the KDA
# recurrence's S submits, the router dot).
#
# Structure is p6b_chain.sh's / rp4_chain.sh's:
#   stop gateway -> merge p0-sync -> build (glm53 + both qwen38, the shader
#   too) -> qwen38 C tests -> smoke -> p5_gate.sh (a, a2, c, d) -> the
#   per-bucket profiles at 600 and 3 000 tokens -> serve the candidate only on
#   rc 0 -> live rows -> verified revert on any failure -> the gateway is
#   restarted on EVERY exit path.
#
# The revert has one more job than rp4_chain's: P5 changes kda_step.comp, the
# .spv files are gitignored, so `git reset --hard` does NOT put the pristine
# shader back. ~/bench/shaders_p5base is copied over c/shaders explicitly, and
# the copy is verified by sha before the gateway is restarted -- a pristine
# binary that pushes 20 bytes of push constants into the candidate's 24-byte
# block would read a garbage token index.
set -u
TAG=p5$(date +%m%d%H%M)
OUT=~/bench/p5_chain_out; mkdir -p $OUT
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR=$OUT/ckpt; mkdir -p $COLI_CKPT_DIR
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.rp4
PRISTINE_SHA=325c9c7ac0c0b2cd5a56e40406de6438f288b97cf766b9f6301d5460e198e500
PSHADERS=~/bench/shaders_p5base
PSHADER_SHA=ed9e3de42770781cdf154562b90ee873f67a3df1a8b2b16ab32c1c1cff21e9ce
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders

start_gateway() {
  unset COLI_CKPT_DIR GLM53_PREFIX_CKPT COLI_MLA_HEADVEC COLI_KDA_ROWS COLI_ROUTER_LANES
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  sleep 100
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}

revert_and_serve_pristine() {
  echo "--- REVERTING: ~/bench/glm53.rp4 and the pre-P5 shaders back in service"
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
  echo "=== p5 chain done $(date -Is)"
}
trap on_exit EXIT

echo "=== p5 chain start $(date -Is) tag=$TAG"
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

echo "--- building glm53 + qwen38 + qwen38-vk (backend_vulkan.c and kda_step.comp changed)"
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|glslc" | head -20
[ -x ~/src/colibri/c/glm53 ] || { echo "BUILD FAILED"; exit 2; }
sha256sum ~/src/colibri/c/glm53 ~/src/colibri/c/qwen38 ~/src/colibri/c/qwen38-vk $SHADERS/kda_step.spv
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
echo "### smoke — the 30-token prompt at the serving knob, all knobs on then all off"
SMOKE_OK=1
for cfg in "on:1:1:1" "off:0:0:0"; do
  name=${cfg%%:*}; rest=${cfg#*:}
  hv=${rest%%:*}; rest=${rest#*:}; kr=${rest%%:*}; rl=${rest##*:}
  wait_no_engine || exit 2
  echo "--- smoke $name (HEADVEC=$hv KDA_ROWS=$kr ROUTER_LANES=$rl)"
  COLI_MLA_HEADVEC=$hv COLI_KDA_ROWS=$kr COLI_ROUTER_LANES=$rl COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh ~/src/colibri/c/glm53 ~/bench/prefill_prompt_30.txt "$TAG-smoke$name" 128 \
    > $OUT/smoke_$name.txt 2>&1
  r=$?
  echo "  rc=$r"; tail -22 $OUT/smoke_$name.txt
  grep -m1 "^teacher_forcing" ~/bench/prefill_profile_$TAG-smoke$name.log | cut -c1-120
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
echo "  rc=$?"; tail -22 $OUT/smoke_pris.txt
grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smokepris.log > $OUT/tf_pris.txt

echo "--- smoke oracle: knobs on vs knobs off, same binary, 30-token prompt"
grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smokeon.log  > $OUT/tf_on.txt
grep "^teacher_forcing" ~/bench/prefill_profile_$TAG-smokeoff.log > $OUT/tf_off.txt
SOK=1
cmp -s $OUT/tf_on.txt $OUT/tf_off.txt  && echo "  on vs off (same binary):  IDENTICAL" || { echo "  SMOKE FAIL: the knobs change teacher forcing"; SOK=0; }
cmp -s $OUT/tf_on.txt $OUT/tf_pris.txt && echo "  on vs pristine (KDA=2):   IDENTICAL" || { echo "  SMOKE FAIL: the candidate differs from the pristine binary at the serving knob"; SOK=0; }
if [ "$SOK" != 1 ]; then revert_and_serve_pristine; exit 4; fi

# --------------------------------------------------------------- profiles
# (b): per-bucket, serving knob, cap 512, chunk 128, GLM53_PREFIX_CKPT=0.
# The 600-token prompt carries the sub-item attribution (four configurations);
# the 3 000-token prompt is the one P5 is for and gets three repeats of the two
# headline rows plus the pristine binary.
echo
echo "### (b) per-bucket profiles at the serving knob"
prof() {   # <tag> <bin> <prompt> <headvec> <kdarows> <routerlanes> <shaders>
  wait_no_engine || exit 2
  echo "--- profile $1 (bin=$(basename $2) prompt=$3 hv=$4 kr=$5 rl=$6)"
  COLI_VK_SHADERS=$7 COLI_MLA_HEADVEC=$4 COLI_KDA_ROWS=$5 COLI_ROUTER_LANES=$6 COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh $2 ~/bench/prefill_prompt_$3.txt "$TAG-$1" 128 \
    2>&1 | tee $OUT/prof_$1.txt | tail -24
}
for r in 1 2;   do prof "600_pristine_r$r" $PRISTINE 600 1 1 1 $PSHADERS; done
for r in 1 2 3; do prof "600_off_r$r"      ~/src/colibri/c/glm53 600 0 0 0 $SHADERS; done
prof "600_p1"    ~/src/colibri/c/glm53 600 1 0 0 $SHADERS
prof "600_p12"   ~/src/colibri/c/glm53 600 1 1 0 $SHADERS
for r in 1 2 3; do prof "600_on_r$r"       ~/src/colibri/c/glm53 600 1 1 1 $SHADERS; done
for r in 1 2;   do prof "3000_off_r$r"     ~/src/colibri/c/glm53 3000 0 0 0 $SHADERS; done
for r in 1 2 3; do prof "3000_on_r$r"      ~/src/colibri/c/glm53 3000 1 1 1 $SHADERS; done
prof "3000_pristine" $PRISTINE 3000 1 1 1 $PSHADERS

# --------------------------------------------------------------- the gate
echo
echo "### p5_gate.sh (steps a, a2, c, d)"
wait_no_engine || exit 2
PRISTINE_SHADERS=$PSHADERS MIN_SPEEDUP=${MIN_SPEEDUP:-1.0} \
  bash $HERE/p5_gate.sh $PRISTINE ~/src/colibri/c/glm53 $TAG 2>&1 | tee $OUT/gate.log
GATE=${PIPESTATUS[0]}; echo "P5 GATE rc=$GATE"

# --------------------------------------------------------------- verdict
echo
echo "=== P5 verdict: gate rc=$GATE"
wait_no_engine || exit 2
if [ "$GATE" = 0 ]; then
  SERVED=1
  echo "=== P5 IN SERVICE: $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
  start_gateway
  echo "### (e) live rows through the gateway"
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --sizes 300,1000 --repeat 2 \
      --tag $TAG-live --json $OUT/live.jsonl 2>&1 | tee $OUT/live.txt | tail -20
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --multiturn --side-request \
      --system ~/bench/p6_system.txt --sizes 300 --repeat 0 --gen 16 \
      --tag $TAG-mt --json $OUT/live.jsonl 2>&1 | tee $OUT/live_mt.txt | tail -20
  ~/bench/owui_report.sh 6
else
  echo "=== THE GATE FAILED -- reverting to ~/bench/glm53.rp4 and the pre-P5 shaders"
  revert_and_serve_pristine
  exit 3
fi
