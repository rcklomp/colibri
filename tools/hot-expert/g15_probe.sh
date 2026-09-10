#!/bin/bash
# g15_probe.sh -- roadmap item 4h (G15) step 1: does GLM-5.3 survive int3 experts?
#
# The item is explicit that the numerics probe comes FIRST and that speed does
# not matter if accuracy fails, so this script measures no speed. It runs the
# same prompt through four configurations of the same model and diffs their
# output:
#
#   R1 pristine  (c4a3e68)              tier on, normal    -- the reference
#   R2 candidate                        tier on, no knobs  -- must equal R1 byte for byte
#   R3 candidate GLM53_EXPERTS_CPU=1    every routed expert on the CPU, int4
#   R4 candidate GLM53_EXPERTS_CPU=1
#                GLM53_I3_SIM=1         every routed expert on the CPU, int3
#
# R3 exists because R4 changes two things against R1 (placement and precision);
# R3 isolates the placement half, so a difference between R3 and R4 is int3 and
# nothing else. R2 is this project's standing bar: a probe that perturbs the
# default path is not a probe.
#
# Oracles, per CLAUDE.md and §G14: the greedy text over 128 tokens, the
# teacher_forcing line, and last_logits (cosine / max-abs / argmax).
#
# Runs the owner's gateway down for its duration: takes ~/bench/.rig.lock first
# so gateway_watchdog.sh does not race back in, and restores the exact binary
# that was in service on every exit path.
#
# Usage (on the rig):  ~/bench/g15_probe.sh [short|long|all]
set -u

PHASE="${1:-all}"
SRC="$HOME/src/colibri"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
OUT="$HOME/bench/g15_probe_out"
PRISTINE_SHA=c4a3e68
BRANCH=perf/g15-int3-experts

mkdir -p "$OUT"
. "$SRC/tools/hot-expert/rig_lock.sh"

RESTART_GATEWAY=0
SERVED_SAVED=0
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  # Put back the exact binary that was in service, whatever state the build left.
  if [ "$SERVED_SAVED" = 1 ]; then
    cp -f "$HOME/bench/glm53.g15base" "$SRC/c/glm53" 2>/dev/null || true
    ( cd "$SRC" && git checkout -q "$BRANCH" 2>/dev/null ) || true
  fi
  if [ "$RESTART_GATEWAY" = 1 ]; then
    echo "[g15] restarting the owner's gateway"
    SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
    KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
    for _ in $(seq 1 90); do
      sleep 10
      code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
               http://127.0.0.1:8081/v1/models 2>/dev/null) || true
      [ "${code:-}" = 200 ] && break
    done
    echo "[g15] gateway back, /v1/models=${code:-?}"
  fi
  rig_lock_release
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

rig_lock_take g15probe || exit 3

for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[g15] $_eng is running -- refusing"; exit 1; fi
done
if pgrep -x glm53 >/dev/null 2>&1; then
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[g15] stopping the owner's gateway for the duration of this probe"
    pkill -f "openai_[s]erver.py" || true
    sleep 2
    pkill -9 -x glm53 || true
    for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
    RESTART_GATEWAY=1
  else
    echo "[g15] a glm53 is running that is not the gateway -- refusing"; exit 1
  fi
fi

echo "[g15] saving the in-service binary"
cp -f "$SRC/c/glm53" "$HOME/bench/glm53.g15base"
SERVED_SAVED=1

echo "[g15] building pristine ($PRISTINE_SHA)"
( cd "$SRC" && git checkout -q "$PRISTINE_SHA" && make -C c glm53 VK=1 ) > "$OUT/build_pristine.log" 2>&1 \
  || { echo "[g15] pristine build FAILED"; tail -20 "$OUT/build_pristine.log"; exit 1; }
cp -f "$SRC/c/glm53" "$HOME/bench/glm53.g15pristine"

echo "[g15] building candidate ($BRANCH)"
( cd "$SRC" && git checkout -q "$BRANCH" && make -C c glm53 qwen38 qwen38-vk VK=1 ) > "$OUT/build_cand.log" 2>&1 \
  || { echo "[g15] candidate build FAILED"; tail -20 "$OUT/build_cand.log"; exit 1; }
cp -f "$SRC/c/glm53" "$HOME/bench/glm53.g15cand"
cp -f "$HOME/bench/glm53.g15base" "$SRC/c/glm53"     # tree binary back to what serves

echo "[g15] pristine  sha256=$(sha256sum "$HOME/bench/glm53.g15pristine" | cut -c1-16)"
echo "[g15] candidate sha256=$(sha256sum "$HOME/bench/glm53.g15cand" | cut -c1-16)"
echo "[g15] in-service sha256=$(sha256sum "$HOME/bench/glm53.g15base" | cut -c1-16)"

P_SHORT=$(cat "$HOME/bench/prompt_glm.txt")
P_LONG=$(python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*30)")

run() {                      # run <tag> <binary> <greedy> <prompt> [ENV=V ...]
  local tag="$1" bin="$2" greedy="$3" prompt="$4"; shift 4
  echo "[g15] $(date +%H:%M:%S) run $tag  ($*)"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/g15_hist.bin"     # frozen per run: the
  rm -rf /tmp/g15_ckpt; mkdir -p /tmp/g15_ckpt             # engine rewrites it at exit
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH=/tmp/g15_hist.bin
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=/tmp/g15_ckpt
    export COLI_TIMERS=1 GLM53_VERBOSE=1
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$M" --prompt "$prompt" --greedy "$greedy" --logits 512 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[g15] $(date +%H:%M:%S) run $tag rc=$rc  $(wc -c < "$OUT/$tag.out") bytes"
  grep -E '^\[VK\] preload:|^\[VK\] preload dev' "$OUT/$tag.err" | tail -3
  grep -E '^\[OPTIME\] (ffn_moe|total)|^\[PROF\]' "$OUT/$tag.err" | tail -4
  return $rc
}

if [ "$PHASE" = short ] || [ "$PHASE" = all ]; then
  run s1_pristine  "$HOME/bench/glm53.g15pristine" 128 "$P_SHORT"
  run s2_cand_off  "$HOME/bench/glm53.g15cand"     128 "$P_SHORT"
  run s3_cpu_int4  "$HOME/bench/glm53.g15cand"     128 "$P_SHORT" GLM53_EXPERTS_CPU=1
  run s4_cpu_int3  "$HOME/bench/glm53.g15cand"     128 "$P_SHORT" GLM53_EXPERTS_CPU=1 GLM53_I3_SIM=1
fi

if [ "$PHASE" = long ] || [ "$PHASE" = all ]; then
  run l1_pristine  "$HOME/bench/glm53.g15pristine" 0 "$P_LONG"
  run l3_cpu_int4  "$HOME/bench/glm53.g15cand"     0 "$P_LONG" GLM53_EXPERTS_CPU=1
  run l4_cpu_int3  "$HOME/bench/glm53.g15cand"     0 "$P_LONG" GLM53_EXPERTS_CPU=1 GLM53_I3_SIM=1
fi

echo
echo "=== G15 step 1 verdict ==="
python3 "$SRC/tools/hot-expert/g15_compare.py" "$OUT"
