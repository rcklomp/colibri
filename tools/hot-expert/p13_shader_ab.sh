#!/bin/bash
# p13_shader_ab.sh -- P13 follow-up: the ONE systematic difference between
# every X2 reproduction (x2_iso_chain.sh's p_gpu2, x2_regime_chain.sh's qB)
# and this bisect's 12/12 non-reproductions is COLI_VK_SHADERS: X2 always
# pointed it at ~/src/colibri-x2/c/shaders (the perf/x2-kl-oracle clone)
# EVEN WHEN RUNNING THE PRISTINE BINARY; this bisect always used a binary's
# own matching shaders dir. kda_step.spv sizes differ (colibri: 12860 B,
# built 09-10; colibri-x2: 11696 B, built 09-16) even though kda_step.comp
# SOURCE is byte-identical in both trees -- a toolchain/rebuild difference,
# not a code change, but the PRISTINE BINARY has never been run against
# its own tree's current shaders in either X2's tests or its own build
# timestamp, only against whichever c/shaders happened to be pointed at.
#
# X,P,P,X on the pristine binary, COLI_KDA_GPU=2, --greedy 0, packet:
#   X1/X2: COLI_VK_SHADERS=~/src/colibri-x2/c/shaders (X2's exact invocation)
#   P1/P2: COLI_VK_SHADERS=~/src/colibri/c/shaders    (this bisect's own)
# gate_compare X1 vs X2, P1 vs P2, X1 vs P1 -- reports which pair(s) DIFFER.
set -u
TAG=p13sh$(date +%m%d%H%M)
OUT=~/bench/p13_shader_out; mkdir -p "$OUT"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
PRISTINE=~/src/colibri/c/glm53
PRISTINE_SHA=a8e10ecf09edc05c3049c667ffcba05beb73b30415409a40214a4ba0e1f565db
SHADERS_X=~/src/colibri-x2/c/shaders
SHADERS_P=~/src/colibri/c/shaders
HERE=~/src/colibri-p13/tools/hot-expert
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

RESTART_GATEWAY=0

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[p13sh] gateway back, /v1/models=${code:-?}"
}

# glm53 is expected to be running here (the gateway) -- stop_gateway kills
# it properly (wrapper first, then the engine); never treat it as a stray.
wait_no_engine() {
  for _ in $(seq 1 120); do
    pgrep -x glm53 >/dev/null 2>&1 || pgrep -x qwen38 >/dev/null 2>&1 || pgrep -x qwen38-vk >/dev/null 2>&1 || return 0
    sleep 2
  done
  echo "[p13sh] FATAL: an engine is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[p13sh] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine || return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine || true
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "[p13sh] accept_live.sh"
  "$HOME/src/colibri/tools/hot-expert/accept_live.sh"
  echo "=== p13_shader_ab exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== p13_shader_ab $TAG $(date -Is)"
for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[p13sh] $_eng is running -- refusing"; exit 1; fi
done

stop_gateway || exit 1

[ "$(sha256sum "$PRISTINE" | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || {
  echo "[p13sh] REFUSED: $PRISTINE sha does not match the expected pristine ($PRISTINE_SHA)"; exit 2; }
echo "[p13sh] pristine=$(sha256sum "$PRISTINE" | cut -c1-16)"
echo "[p13sh] SHADERS_X kda_step.spv: $(sha256sum "$SHADERS_X/kda_step.spv" | cut -c1-16) ($(stat -c%s "$SHADERS_X/kda_step.spv") bytes)"
echo "[p13sh] SHADERS_P kda_step.spv: $(sha256sum "$SHADERS_P/kda_step.spv" | cut -c1-16) ($(stat -c%s "$SHADERS_P/kda_step.spv") bytes)"

echo "--- warming the model (assume warm; cat is cheap if so)"
cat "$M"/*.safetensors > /dev/null 2>&1 || true

PACKET=$(cat "$HERE/x2_packet_450.txt")

run() {                      # run <tag> <shaders>
  local tag="$1" shaders="$2"
  for _eng in glm53 qwen38 qwen38-vk; do
    pgrep -x "$_eng" >/dev/null 2>&1 && { echo "[p13sh] $_eng already running before $tag -- refusing"; return 9; }
  done
  echo "[p13sh] $(date +%H:%M:%S) run $tag  shaders=$shaders"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/p13sh_hist_$tag.bin"
  rm -rf "/tmp/p13sh_ckpt_$tag"; mkdir -p "/tmp/p13sh_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/p13sh_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/p13sh_ckpt_$tag"
    export GLM53_VERBOSE=1 COLI_KDA_GPU=2
    "$PRISTINE" --model "$M" --prompt "$PACKET" --greedy 0 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[p13sh] $(date +%H:%M:%S) run $tag rc=$rc  $(grep -c ^teacher_forcing "$OUT/$tag.out") tf-line(s)"
  wait_no_engine || true
  return $rc
}

run X1 "$SHADERS_X"
run P1 "$SHADERS_P"
run P2 "$SHADERS_P"
run X2 "$SHADERS_X"

. "$HERE/gate_lib.sh"
echo
echo "=== P13 shader A/B: $TAG ==="
gate_compare "X1 vs X2 (both x2-tree shaders)" "$OUT/X1.out" "$OUT/X2.out" '^teacher_forcing'
gate_compare "P1 vs P2 (both colibri shaders)" "$OUT/P1.out" "$OUT/P2.out" '^teacher_forcing'
gate_compare "X1 vs P1 (cross)"                "$OUT/X1.out" "$OUT/P1.out" '^teacher_forcing'
gate_compare "X2 vs P2 (cross)"                "$OUT/X2.out" "$OUT/P2.out" '^teacher_forcing'

RESTART_GATEWAY=1
exit 0
