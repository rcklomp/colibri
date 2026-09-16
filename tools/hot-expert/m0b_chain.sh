#!/bin/bash
# m0b_chain.sh -- FRANKEN-M0b: does host-staged (non-P2P) all_reduce run at
# all, and if so what is the small-message latency L against the plan's
# 33 us TP3 break-even bar (FRANKEN-ENGINE-PLAN-2026-09-15.md sec5)?
#
# M0 (record sec FRANKEN-M0) got RCCL to initialise, see all three GPUs, and
# build its ring/tree topology, then fail every run -g3/-g2 at P2P transport
# setup (`hipIpcGetMemHandle failed: invalid argument`, transport/p2p.cc:235)
# before any collective ran -- no latency, no bandwidth, no transport line
# was obtainable. This chain forces RCCL off the P2P transport
# (NCCL_P2P_DISABLE=1) so it falls back to a host-staged path (shared memory
# or sockets); if that still fails, it also disables SHM
# (NCCL_SHM_DISABLE=1) to force pure sockets. Modeled on h0_chain.sh:
# rig-lock-only entry, stop gateway (wait for it to die), assert VRAM free,
# run, restore, restart gateway + accept_live.sh on every exit path.
#
# Launch only via run_chain.sh:
#   setsid nohup ~/src/colibri-m0b/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-m0b/tools/hot-expert/m0b_chain.sh \
#       > ~/bench/m0b_chain.log 2>&1 < /dev/null &
set -u
TAG=m0b_$(date +%m%d%H%M)
OUT=~/bench/m0b_out; mkdir -p "$OUT"
HERE=~/src/colibri-m0b/tools/hot-expert
RCCL_BIN=~/src/rccl-tests/build/all_reduce_perf
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "FATAL: an engine is still alive after 240 s"; return 1
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== m0b_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh ---"
  "$HERE/accept_live.sh" || echo "ACCEPT_LIVE FAILED"
  echo "=== results dir: $OUT"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== m0b_chain $TAG $(date -Is)"

echo "--- step 1: stop gateway"
stop_gateway || exit 1

echo "--- step 1b: pre-checks (engines idle, VRAM free)"
for e in glm53 qwen38 qwen38-vk; do
  pgrep -x "$e" >/dev/null && { echo "FATAL: $e still running after stop_gateway"; exit 1; }
done
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v"
  [ "$v" -lt 1073741824 ] || { echo "FATAL: card$c VRAM $v >= 1 GiB after stopping the gateway"; exit 1; }
done

[ -x "$RCCL_BIN" ] || { echo "FATAL: $RCCL_BIN missing"; exit 1; }

run_variant() {
  local name="$1" gpus="$2"; shift 2
  local out="$OUT/${TAG}_${name}.log"
  echo "--- $name: all_reduce_perf -b 8 -e 128M -f 2 -g $gpus  env: $*"
  env NCCL_DEBUG=INFO LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib "$@" \
    "$RCCL_BIN" -b 8 -e 128M -f 2 -g "$gpus" > "$out" 2>&1
  local rc=$?
  echo "$name rc=$rc -> $out"
  return $rc
}

echo "--- step 2: M0b runs (NCCL_P2P_DISABLE=1, host-staged)"
run_variant g3_p2pdis 3 NCCL_P2P_DISABLE=1
G3_RC=$?
run_variant g2_p2pdis 2 NCCL_P2P_DISABLE=1
G2_RC=$?

if [ "$G3_RC" != 0 ] || grep -qE "Test NCCL failure|Cuda failure" "$OUT/${TAG}_g3_p2pdis.log" 2>/dev/null; then
  echo "--- g3 still failing with P2P disabled alone; retry with SHM also disabled (pure sockets)"
  run_variant g3_p2pdis_shmdis 3 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
fi
if [ "$G2_RC" != 0 ] || grep -qE "Test NCCL failure|Cuda failure" "$OUT/${TAG}_g2_p2pdis.log" 2>/dev/null; then
  echo "--- g2 still failing with P2P disabled alone; retry with SHM also disabled (pure sockets)"
  run_variant g2_p2pdis_shmdis 2 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
fi

echo "--- step 3: final VRAM check"
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v"
  [ "$v" -lt 1073741824 ] || echo "WARNING: card$c VRAM not free before restart ($v bytes)"
done

exit 0
