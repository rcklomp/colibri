#!/bin/bash
# Upstream `dev` merge chain (housekeeping item, PREFILL-ROADMAP-2026-09.md).
#
# Structure is p5_chain.sh's / p6b_chain.sh's: stop gateway -> merge p0-sync ->
# build -> tests -> gate -> serve only on rc 0 -> live check -> verified
# sha256 revert on any failure -> the gateway is restarted on EVERY exit path.
#
# What this chain checks that the P-item chains did not:
#   (0) the shard mapping moved from the fork's g_fmap[] to st.h's
#       st_map_shard_range (#1325). That is the one change here that could put
#       model bytes in ANONYMOUS memory, so the [MAP] line, majflt and the
#       MemAvailable floor are compared against the pristine's own run BEFORE
#       anything long runs, and the chain stops if MemAvailable drops.
#   (6) rome_bench.sh's glm53 case, PAIRED (pristine then candidate in the same
#       session): the roadmap's stated gate for this item. Its last recorded
#       rows are from 2026-09-05, before P2..P7, so the honest comparison is
#       the pair, not the table. No passwordless sudo here, so the cold column
#       is NOT a dropped-cache cold row and is labelled as such.
set -u
TAG=dm$(date +%m%d%H%M)
OUT=~/bench/devmerge_out; mkdir -p $OUT
SERVED=0
PREMERGE=""
PRISTINE=~/bench/glm53.p5
PRISTINE_SHA=5dd26f652028ad69cc47a16c74e4c25840a655f7899e679a73937119116b8932
PSHADERS=~/bench/shaders_p5
PSHADER_SHA=b7d57f05b6f9ee48b61288ea76ffb4bd9fe25386bc8c7f0a1a6f0ff78e546478
HERE=~/src/colibri/tools/hot-expert
SHADERS=~/src/colibri/c/shaders
SRC=~/src/colibri
TOOLS=~/bench/owui_tools_4.json
SYSTEM=~/bench/p6_system.txt
MODEL=~/models/GLM-5.3-Flash-colibri-int4-g64
CKPT=$OUT/ckpt; mkdir -p $CKPT

start_gateway() {
  unset COLI_CKPT_DIR GLM53_PREFIX_CKPT COLI_USAGE_PATH
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  sleep 120
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}

revert_and_serve_pristine() {
  echo "--- REVERTING: ~/bench/glm53.p5 and the pre-merge tree back in service"
  ~/bench/p7_stop.sh
  wait_no_engine || echo "  (engine still up; the cp may fail with ETXTBSY)"
  if [ -n "$PREMERGE" ]; then (cd $SRC && git checkout -- . ; git reset --hard "$PREMERGE"); fi
  cp -f $PSHADERS/*.spv $SHADERS/ || echo "SHADER REVERT COPY FAILED"
  cp $PRISTINE $SRC/c/glm53 || echo "REVERT COPY FAILED"
  b=$(sha256sum $SRC/c/glm53 | cut -d" " -f1)
  s=$(sha256sum $SHADERS/kda_step.spv | cut -d" " -f1)
  [ "$b" = "$PRISTINE_SHA" ] && echo "revert verified (binary): $b" || echo "REVERT FAILED: served binary is NOT the pristine one ($b)"
  [ "$s" = "$PSHADER_SHA" ] && echo "revert verified (shader): $s" || echo "REVERT FAILED: kda_step.spv is NOT the pristine one ($s)"
  start_gateway
}

on_exit() {
  rc=$?
  if [ "$SERVED" != 1 ]; then
    echo "=== chain exiting rc=$rc without a served candidate"
    if ! pgrep -f "openai_[s]erver.py" >/dev/null; then revert_and_serve_pristine; fi
  fi
  echo "=== devmerge chain done $(date -Is)"
}
trap on_exit EXIT

# MemAvailable floor while a binary runs a short prompt: the mapping change is
# the only thing here that could make the engine allocate the model instead of
# mapping it, and the number the record wants is the FLOOR, not the value at a
# moment nobody chose.
memfloor() {   # <label> <binary> <shaderdir>  -> $OUT/mem_<label>.txt
  local label=$1 bin=$2 sh=$3
  wait_no_engine || exit 2
  ( while :; do awk '/^MemAvailable/{print $2}' /proc/meminfo; sleep 5; done ) > $OUT/mem_$label.raw &
  local sampler=$!
  COLI_VK_SHADERS=$sh COLI_KDA_GPU=2 \
    bash $HERE/prefill_profile.sh $bin ~/bench/prefill_prompt_30.txt "$TAG-mem$label" 128 \
    > $OUT/memrun_$label.txt 2>&1
  local r=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local floor=$(sort -n $OUT/mem_$label.raw | head -1)
  local elog=$HOME/bench/prefill_profile_$TAG-mem$label.log
  echo "$label rc=$r MemAvailable floor: $((floor/1048576)) GB ($floor kB), samples $(wc -l < $OUT/mem_$label.raw)" | tee $OUT/mem_$label.txt
  grep -m1 "^\[MAP\]" $elog | tee -a $OUT/mem_$label.txt || echo "  (no [MAP] line)" | tee -a $OUT/mem_$label.txt
  grep -m1 "^\[PROF\] mmap" $elog | tee -a $OUT/mem_$label.txt || true
  grep -m1 "esperti: slot da" $elog | tee -a $OUT/mem_$label.txt || true
  grep -m1 "budget esperti" $elog | tee -a $OUT/mem_$label.txt || true
  grep -m1 "^experts hits" $elog | tee -a $OUT/mem_$label.txt || true
  grep -m1 "prefill .* token in" $elog | tee -a $OUT/mem_$label.txt || true
  grep -m1 "^teacher_forcing" $elog | cut -c1-60 | tee -a $OUT/mem_$label.txt || true
  echo "$floor" > $OUT/floor_$label
}

# rome_bench.sh's glm53 case, run on this box against whatever is at c/glm53.
# The remote body is EXTRACTED from the harness itself so the regime is the
# recorded one (8 pinned threads, 1695 caps, a COPY of the histogram,
# residency asserted, the tier lines asserted); only the destination of the
# mechanical row is redirected, so the rig's tree stays clean for the merge.
romebench() {   # <config-name>
  wait_no_engine || exit 2
  awk "/bash -s \\\$REMOTE_ARGV/{f=1;next} /^REMOTE_SCRIPT\$/{f=0} f" $SRC/tools/rome_bench.sh > $OUT/rb_body.sh
  sed -i "s|^RECORD_PATH=.*|RECORD_PATH=\"$OUT/datapoint_rows.md\"|" $OUT/rb_body.sh
  echo "--- rome_bench glm53 $1 ($(sha256sum $SRC/c/glm53 | cut -c1-16))"
  bash $OUT/rb_body.sh glm53 "$1" "" > $OUT/rb_$1.txt 2>&1
  echo "  rc=$? $(grep -c 'Cache eviction requires sudo' $OUT/rb_$1.txt) sudo-warnings"
  grep -E "^RESULT:|^Rotating median|GPU tier confirmed|Appended row" $OUT/rb_$1.txt | sed 's/^/  /'
  tail -1 $OUT/datapoint_rows.md
}

echo "=== devmerge chain start $(date -Is) tag=$TAG"
free -g
cp -f ~/.glm53_explain.bin $OUT/hist.bin || { echo "no histogram: the tier would preload empty"; exit 2; }
sha256sum $PRISTINE $SRC/c/glm53 $PSHADERS/kda_step.spv $SHADERS/kda_step.spv
[ "$(sha256sum $PRISTINE | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || { echo "PRISTINE SHA MISMATCH"; exit 2; }
[ "$(sha256sum $PSHADERS/kda_step.spv | cut -d' ' -f1)" = "$PSHADER_SHA" ] || { echo "PRISTINE SHADER SHA MISMATCH"; exit 2; }

echo "--- stopping the gateway"
~/bench/p7_stop.sh || exit 2
wait_no_engine || exit 2

echo "--- merging p0-sync into hot-expert-tier"
cd $SRC || exit 2
git checkout hot-expert-tier || exit 2
PREMERGE=$(git rev-parse HEAD); echo "pre-merge HEAD $PREMERGE"
git merge --ff-only p0-sync || { echo "MERGE FAILED (not a fast-forward)"; exit 2; }
git log --oneline -1

echo "--- building glm53 + qwen38 + qwen38-vk"
make -C c glm53 qwen38 qwen38-vk VK=1 2>&1 | grep -Ei "error|glslc" | head -20
[ -x $SRC/c/glm53 ] || { echo "BUILD FAILED"; revert_and_serve_pristine; exit 2; }
sha256sum $SRC/c/glm53 $SRC/c/qwen38 $SRC/c/qwen38-vk $SHADERS/kda_step.spv
cp $SRC/c/glm53 $OUT/glm53.cand
echo "--- shaders vs the P5 set (dev touches no shader: these must be identical)"
SHOK=1
for f in $PSHADERS/*.spv; do
  cmp -s "$f" "$SHADERS/$(basename $f)" || { echo "  DIFFERS: $(basename $f)"; SHOK=0; }
done
[ "$SHOK" = 1 ] && echo "  all .spv byte-identical to $PSHADERS" || echo "  WARNING: a shader changed; the gate's PRISTINE_SHADERS matters"

# ------------------------------------------------------------ (1) tests
echo
echo "### step 1 — the four qwen38 C tests, the chat-template pin, the python suite"
QT=0
for t in prefix config metrics serve_framing; do
  make -C c tests/test_qwen38_$t VK=1 >/dev/null 2>&1
  ./c/tests/test_qwen38_$t > $OUT/test_$t.txt 2>&1; r=$?
  echo "  test_qwen38_$t exit=$r  $(tail -1 $OUT/test_$t.txt | cut -c1-80)"
  [ "$r" = 0 ] || QT=1
done
python3 c/tests/test_glm53_chat_template.py --template $MODEL/chat_template.jinja > $OUT/test_template.txt 2>&1
r=$?; echo "  test_glm53_chat_template exit=$r  $(tail -1 $OUT/test_template.txt | cut -c1-100)"
[ "$r" = 0 ] || QT=1
( cd c && timeout 1800 python3 -m unittest discover -s tests -p "test_*.py" ) > $OUT/test_python.txt 2>&1
r=$?; echo "  python unittest exit=$r  $(tail -3 $OUT/test_python.txt | tr '\n' ' ' | cut -c1-100)"
[ "$r" = 0 ] || QT=1
if [ "$QT" != 0 ]; then echo "=== STEP 1 FAILED"; revert_and_serve_pristine; exit 4; fi

# ------------------------------------------------------------ (2) mapping
echo
echo "### step 2 — the mapping change: [MAP], MemAvailable floor, candidate vs pristine"
memfloor pristine $PRISTINE $PSHADERS
memfloor candidate $SRC/c/glm53 $SHADERS
FP=$(cat $OUT/floor_pristine); FC=$(cat $OUT/floor_candidate)
DROP=$(( (FP - FC) / 1048576 ))
echo "MemAvailable floor: pristine $((FP/1048576)) GB, candidate $((FC/1048576)) GB, drop ${DROP} GB"
if [ "$DROP" -gt 5 ]; then
  echo "=== STOP: the merged engine's MemAvailable floor is ${DROP} GB below the pristine's."
  echo "=== That is the mapping change putting model bytes in anonymous memory. Not serving."
  revert_and_serve_pristine; exit 5
fi

# ------------------------------------------------------------ (3) prefill_gate
echo
echo "### step 3 — prefill_gate.sh (oracle + serve-path TTFT), MIN_SPEEDUP=0.97"
wait_no_engine || exit 2
GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=$CKPT COLI_USAGE_PATH=$OUT/hist.bin \
PRISTINE_SHADERS=$PSHADERS MIN_SPEEDUP=0.97 \
  bash $HERE/prefill_gate.sh $PRISTINE $SRC/c/glm53 $TAG 2>&1 | tee $OUT/gate.log
GATE=${PIPESTATUS[0]}; echo "PREFILL GATE rc=$GATE"
wait_no_engine || exit 2

# ------------------------------------------------------------ (4) tworeq
echo
echo "### step 4 — tworeq at 4 slots, both KDA knobs"
cp -f ~/.glm53_explain.bin $OUT/hist_tworeq.bin 2>/dev/null || true
TW=0
for knob in 2 0; do
  wait_no_engine || exit 2
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist_tworeq.bin" \
      COLI_CKPT_DIR="$CKPT" GLM53_PREFIX_CKPT=0 \
      COLI_KDA_GPU=$knob TWOREQ_SLOTS=4 TWOREQ_EXE="$SRC/c/glm53" GLM53_VERBOSE=1 \
      python3 $HERE/tworeq.py > $OUT/tworeq_kda$knob.txt 2>&1
  r=$?
  forced=$(grep -c "forcing COLI_KDA_GPU=0" $OUT/tworeq_kda$knob.txt)
  echo "  tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' $OUT/tworeq_kda$knob.txt || echo 'NO RESULT') (rc=$r, forcing lines: $forced)"
  grep -m1 "KDA slot pool" $OUT/tworeq_kda$knob.txt | sed 's/^/    /' || true
  [ "$r" = 0 ] || TW=1
  [ "$knob" = 2 ] && [ "$forced" != 0 ] && { echo "    FAIL: the pool did not cover 4 slots"; TW=1; }
done
echo "tworeq rc=$TW"

# ------------------------------------------------------------ (5) P7 paths
echo
echo "### step 5 — checkpoints and the pin survive the merge"
wait_no_engine || exit 2
rm -f $CKPT/*.bin
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
    COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
    COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
    COLI_CKPT_DIR="$CKPT" COLI_KDA_GPU=2 GLM53_PREFIX_CKPT=1 GLM53_VERBOSE=1 \
    python3 $HERE/ttft_serve.py --engine $SRC/c/glm53 --prefix-ckpt \
      --tools $TOOLS --system $SYSTEM --kv-slots 4 \
      --sizes "" --repeat 0 --warm --min-resident 96 \
      --tag "$TAG-ckpt" --json $OUT/p7.jsonl \
      --engine-log $OUT/engine_ckpt.log 2>&1 | tee $OUT/step5_ckpt.txt
CK=${PIPESTATUS[0]}
grep -E "CKPT (store|hit|disk)" $OUT/engine_ckpt.log | head -8 | sed 's/^/  /' || true
wait_no_engine || exit 2
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
    COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
    COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
    COLI_CKPT_DIR="$CKPT" COLI_KDA_GPU=2 GLM53_PREFIX_CKPT=1 GLM53_VERBOSE=1 \
    python3 $HERE/ttft_serve.py --engine $SRC/c/glm53 --pin-block --kv-slots 4 \
      --sizes "" --repeat 0 --system $SYSTEM \
      --warm --min-resident 96 --tag "$TAG-pin" --json $OUT/p7.jsonl \
      --engine-log $OUT/engine_pin.log 2>&1 | tee $OUT/step5_pin.txt
PIN=${PIPESTATUS[0]}
wait_no_engine || exit 2
echo "prefix-ckpt rc=$CK  pin-block rc=$PIN"

# ------------------------------------------------------------ (6) rome_bench
echo
echo "### step 6 — rome_bench.sh glm53, paired (pristine binary, then the merged one)"
echo "NOTE: no passwordless sudo on this box, so the page cache is NOT dropped:"
echo "      the 'cold' column below is a warm-cache first request, not the recorded cold row."
wait_no_engine || exit 2
cp -f $PRISTINE $SRC/c/glm53 || { echo "cp pristine failed"; }
romebench "devmerge-pristine-p5"
wait_no_engine || exit 2
cp -f $OUT/glm53.cand $SRC/c/glm53 || { echo "cp candidate failed"; }
romebench "devmerge-candidate"
wait_no_engine || exit 2
echo "--- datapoint rows from this session"
cat $OUT/datapoint_rows.md
git -C $SRC checkout -- tools/hot-expert/ROME-3x7900XTX-2026-09-04.md 2>/dev/null || true
RBP=$(grep -m1 "^Rotating median" $OUT/rb_devmerge-pristine-p5.txt | awk '{print $3}')
RBC=$(grep -m1 "^Rotating median" $OUT/rb_devmerge-candidate.txt | awk '{print $3}')
RB=$(awk -v p="${RBP:-0}" -v c="${RBC:-0}" 'BEGIN{ if (p>0 && c >= 0.97*p) print 0; else print 1 }')
echo "rome_bench rotating median: pristine ${RBP:-?} tok/s, candidate ${RBC:-?} tok/s -> $(awk -v p="${RBP:-0}" -v c="${RBC:-0}" 'BEGIN{ if(p>0) printf "%.3fx", c/p; else print "n/a" }') (need >= 0.97x, rc=$RB)"

# ------------------------------------------------------------ verdict
echo
echo "=== VERDICT: prefill_gate=$GATE tworeq=$TW ckpt=$CK pin=$PIN rome_bench=$RB"
sha256sum $SRC/c/glm53
if [ "$GATE" = 0 ] && [ "$TW" = 0 ] && [ "$CK" = 0 ] && [ "$PIN" = 0 ] && [ "$RB" = 0 ]; then
  SERVED=1
  echo "=== MERGE IN SERVICE: $(sha256sum $SRC/c/glm53 | cut -c1-16), serving script unchanged"
  start_gateway
  echo "### live rows through the gateway"
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --sizes 300,1000 --repeat 2 --warm \
      --tag $TAG-live --json $OUT/live.jsonl 2>&1 | tee $OUT/live.txt | tail -20
  python3 $HERE/ttft_serve.py --url http://127.0.0.1:8081 --multiturn --side-request \
      --system $SYSTEM --sizes 300 --repeat 0 --gen 16 --warm \
      --tag $TAG-mt --json $OUT/live.jsonl 2>&1 | tee $OUT/live_mt.txt | tail -20
  ~/bench/owui_report.sh 5
  echo "--- CKPT / [pin] lines in the server log"
  grep -E "CKPT |\[pin\]" ~/glm53_server.log | tail -10
  free -g
else
  echo "=== THE GATE FAILED — reverting to ~/bench/glm53.p5"
  revert_and_serve_pristine
  exit 3
fi
