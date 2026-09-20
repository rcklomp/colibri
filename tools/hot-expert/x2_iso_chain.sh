#!/bin/bash
# x2_iso_chain.sh -- X2 follow-up: is G12's degenerate teacher_forcing on the
# 450-row packet a tree regression or a perf/x2-kl-oracle regression?
#
# The §X2 record's first pass found COLI_KDA_GPU=2 producing a degenerate,
# repetitive teacher_forcing line (token 154822 over and over) against a sane
# COLI_KDA_GPU=0 on the SAME x2cand binary -- the opposite of the record's own
# G12 finding (TF identical at every length tested, 42 through 1260 positions).
# An 8-expert tier shift cannot produce that (the clamp row, which swaps EVERY
# routed expert between GPU and CPU, costs KL 0.028 -- three orders of
# magnitude less than G12's 12.75), so the KDA-slot-pool diagnosis in the first
# pass does not survive this check and must not stand as recorded.
#
# Three runs, same packet, teacher_forcing only (no GLM53_LOGIT_DUMP_ALL --
# this only needs the printed line, not a KL number):
#   (1) PRISTINE serving binary  ~/src/colibri/c/glm53   COLI_KDA_GPU=2
#   (2) same pristine binary                              COLI_KDA_GPU=0
#   (3) perf/x2-kl-oracle binary ~/bench/glm53.x2cand      COLI_KDA_GPU=0
# gate_compare on the teacher_forcing line decides which of two things broke:
#   (1) != (2)         -> the TREE's CPU KDA path (COLI_KDA_GPU=0) is broken;
#                          record that, with the first divergent position, and
#                          stop -- bisecting it is a different item.
#   (1) == (2) != (3)  -> perf/x2-kl-oracle broke COLI_KDA_GPU=0; find the
#                          cause before recording anything else about G12.
#
# Same shape as x2_chain.sh: stop gateway, wait_no_engine, warm+resident,
# trap restarts the gateway on every exit path, accept_live.sh at the end.
# Budget <= 20 min -- three prefill-only runs on a 564-token packet.
set -u
TAG=x2iso$(date +%m%d%H%M)
OUT=~/bench/x2_iso_out; mkdir -p "$OUT"
SRC_X2=~/src/colibri-x2
HERE="$SRC_X2/tools/hot-expert"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
PRISTINE=~/src/colibri/c/glm53
PRISTINE_SHA=a8e10ecf09edc05c3049c667ffcba05beb73b30415409a40214a4ba0e1f565db
X2CAND=~/bench/glm53.x2cand
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
  echo "[x2iso] gateway back, /v1/models=${code:-?}"
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[x2iso] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[x2iso] FATAL: a glm53 is still alive after 240s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  # Kill glm53 ONLY when the gateway is about to be restarted here. On the normal
  # path the body has already restarted it (RESTART_GATEWAY=0) and that glm53 IS the
  # owner's served engine: killing it left openai_server.py up with a defunct engine,
  # /v1/models=200 and every chat a 500 "engine dispatcher stopped" (2026-09-20).
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    pkill -9 -x glm53 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== x2_iso_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== x2_iso_chain $TAG $(date -Is)"
for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[x2iso] $_eng is running -- refusing"; exit 1; fi
done

stop_gateway || exit 1

[ "$(sha256sum "$PRISTINE" | cut -d' ' -f1)" = "$PRISTINE_SHA" ] || {
  echo "[x2iso] REFUSED: $PRISTINE sha does not match the expected pristine ($PRISTINE_SHA)"; exit 2; }
[ -f "$X2CAND" ] || { echo "[x2iso] REFUSED: $X2CAND missing"; exit 2; }
echo "[x2iso] pristine=$(sha256sum "$PRISTINE" | cut -c1-16)  x2cand=$(sha256sum "$X2CAND" | cut -c1-16)"

echo "--- warming the model (assume warm from the prior x2 run; cat is cheap if so)"
cat "$M"/*.safetensors > /dev/null 2>&1 || true

PACKET=$(cat "$HERE/x2_packet_450.txt")

run() {                      # run <tag> <binary> [ENV=V ...]
  local tag="$1" bin="$2"; shift 2
  echo "[x2iso] $(date +%H:%M:%S) run $tag  bin=$(basename "$bin")  ($*)"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/x2iso_hist_$tag.bin"
  rm -rf "/tmp/x2iso_ckpt_$tag"; mkdir -p "/tmp/x2iso_ckpt_$tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC_X2/c/shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH="/tmp/x2iso_hist_$tag.bin"
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR="/tmp/x2iso_ckpt_$tag"
    export GLM53_VERBOSE=1
    for kv in "$@"; do export "${kv?}"; done
    "$bin" --model "$M" --prompt "$PACKET" --greedy 0 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[x2iso] $(date +%H:%M:%S) run $tag rc=$rc  $(grep -c ^teacher_forcing "$OUT/$tag.out") tf-line(s)"
  grep -E '^\[VK\] preload:|^\[VK\] KDA slot pool' "$OUT/$tag.err"
  return $rc
}

run p_gpu2 "$PRISTINE" COLI_KDA_GPU=2
run p_gpu0 "$PRISTINE" COLI_KDA_GPU=0
run x_gpu0 "$X2CAND"   COLI_KDA_GPU=0

echo
echo "=== X2 isolation: pristine GPU=2 vs GPU=0 vs x2cand GPU=0 ==="
. "$HERE/gate_lib.sh"
gate_compare "pristine GPU=2 vs pristine GPU=0" "$OUT/p_gpu2.out" "$OUT/p_gpu0.out" '^teacher_forcing'
R1=$?
gate_compare "pristine GPU=0 vs x2cand GPU=0"   "$OUT/p_gpu0.out" "$OUT/x_gpu0.out" '^teacher_forcing'
R2=$?
gate_compare "pristine GPU=2 vs x2cand GPU=0"   "$OUT/p_gpu2.out" "$OUT/x_gpu0.out" '^teacher_forcing'
R3=$?

echo
if [ $R1 -ne 0 ]; then
  echo "VERDICT: pristine GPU=2 != pristine GPU=0 -- the TREE's CPU KDA path is broken (not this branch)."
elif [ $R2 -ne 0 ]; then
  echo "VERDICT: pristine GPU=0 == GPU=2, but x2cand GPU=0 differs -- perf/x2-kl-oracle broke the CPU KDA path."
else
  echo "VERDICT: all three agree -- G12's first-pass degenerate output was not reproduced here."
fi

echo
echo "[x2iso] restoring the gateway before accept_live.sh"
start_gateway
RESTART_GATEWAY=0

echo "[x2iso] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ACCEPT_RC=$?
echo "[x2iso] accept_live.sh rc=$ACCEPT_RC"

exit $ACCEPT_RC
