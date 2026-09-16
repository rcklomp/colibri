#!/bin/bash
# m0c_chain.sh -- FRANKEN-M0c: does RCCL's P2P setup succeed now that
# rccl-tests is rebuilt against the venv ROCm 10.0.0 (FRANKEN-H0d)?
#
# M0 (record sec FRANKEN-M0) got RCCL to initialise, see all three GPUs, and
# build its ring/tree topology, then fail every run -g3/-g2 at P2P transport
# setup (`hipIpcGetMemHandle failed: invalid argument`) before any collective
# ran, under system ROCm 6.2.0. FRANKEN-H0d's rocm_p2p_probe found
# hipIpcGetMemHandle now succeeds on all three devices under the venv ROCm
# 10.0.0 -- this chain is the load-bearing test of whether that clears M0's
# actual collective, not just the probe's lone allocation. RCCL is rebuilt
# (tools/hot-expert/ separately; see FRANKEN-H0d/M0c in the record for the
# venv recipe) into ~/src/rccl-tests/build-rocm10, LD_LIBRARY_PATH pointed at
# the venv root's lib instead of /opt/rocm-6.2.0/lib.
#
# Runs P2P ENABLED first (no NCCL_P2P_DISABLE) for -g3 and -g2; if that still
# fails, falls back to NCCL_P2P_DISABLE=1 for a like-for-like comparison
# against M0b's host-staged numbers (g3: 8 B 30.8 us, 4 KB 34.6 us,
# 64 KB 61.2 us, 7.9 GB/s at 128 MB). Modeled on h0_chain.sh/m0b_chain.sh:
# rig-lock-only entry, stop gateway (wait for it to die), assert VRAM free,
# run, restore, restart gateway + accept_live.sh on every exit path.
#
# Launch only via run_chain.sh:
#   setsid nohup ~/src/colibri-h0d/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-h0d/tools/hot-expert/m0c_chain.sh \
#       > ~/bench/m0c_chain.log 2>&1 < /dev/null &
set -u
TAG=m0c_$(date +%m%d%H%M)
OUT=~/bench/m0c_out; mkdir -p "$OUT"
HERE=~/src/colibri-h0d/tools/hot-expert
RCCL_BIN=~/src/rccl-tests/build-rocm10/all_reduce_perf
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
ROCM_ROOT=$(~/venvs/rocm/bin/rocm-sdk path --root)

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
  echo "=== m0c_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "--- accept_live.sh ---"
  "$HERE/accept_live.sh" || echo "ACCEPT_LIVE FAILED"
  echo "=== results dir: $OUT"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== m0c_chain $TAG $(date -Is) (venv ROCm $ROCM_ROOT)"

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
  env NCCL_DEBUG=INFO LD_LIBRARY_PATH="$ROCM_ROOT/lib" "$@" \
    "$RCCL_BIN" -b 8 -e 128M -f 2 -g "$gpus" > "$out" 2>&1
  local rc=$?
  echo "$name rc=$rc -> $out"
  return $rc
}

echo "--- step 2: M0c runs, P2P ENABLED (no NCCL_P2P_DISABLE), venv ROCm 10"
run_variant g3_p2pen 3
G3_RC=$?
run_variant g2_p2pen 2
G2_RC=$?

if [ "$G3_RC" != 0 ] || grep -qE "Test NCCL failure|Cuda failure" "$OUT/${TAG}_g3_p2pen.log" 2>/dev/null; then
  echo "--- g3 still failing with P2P enabled; retry NCCL_P2P_DISABLE=1 (like-for-like vs M0b)"
  run_variant g3_p2pdis 3 NCCL_P2P_DISABLE=1
fi
if [ "$G2_RC" != 0 ] || grep -qE "Test NCCL failure|Cuda failure" "$OUT/${TAG}_g2_p2pen.log" 2>/dev/null; then
  echo "--- g2 still failing with P2P enabled; retry NCCL_P2P_DISABLE=1 (like-for-like vs M0b)"
  run_variant g2_p2pdis 2 NCCL_P2P_DISABLE=1
fi

echo "--- step 3: final VRAM check"
for c in 0 1 2; do
  v=$(VRAM "$c")
  echo "card$c: vram_used=$v"
  [ "$v" -lt 1073741824 ] || echo "WARNING: card$c VRAM not free before restart ($v bytes)"
done

exit 0
