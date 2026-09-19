#!/bin/bash
# f1_probe_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F1, the
# falsifier probe that has to run BEFORE F1's engine work (record sec
# F1-STEP0, plan rev 20 task B): at batch 1 (one decode token), what does an
# on-demand streamed miss cost end to end through the ENGINE's own ring code
# -- fill, submit, compute, readback -- per window, against the 1.747
# ms/window the CPU path pays today at ~3.12 experts/window (record sec
# F1-STEP0, req 8, 18 669-token decode)? Below that number F1 has something
# to gain; at or above it F1 is dead regardless of how it is wired.
#
# Modelled on f2_ring_probe_chain.sh, with one difference: that chain did NOT
# stop the gateway (its footprint fit the tier's VRAM reserve). This one
# DOES, on request -- stop / build / run / restart on every exit path,
# accept_live.sh at the end -- so the probe's timing has no contention from
# the served glm53 on the same three cards and no VRAM-reserve arithmetic to
# get wrong. Target <= 10 min gateway down (`f1_ring_probe`'s own default
# sweep is 2 splits x 5 k values x 2000 windows, each window a handful of
# milliseconds -- low minutes in practice; see the header comment on
# F1_DECODE_PROBE in f2_ring_probe.c for the full design).
#
# The probe binary is f2_ring_probe (F1's mode lives in that source file,
# gated behind F1_DECODE_PROBE=1 -- it already links the engine's own
# coli_vk_ring_*/coli_vk_expert_group_issue* code and there is no reason to
# fork a second binary for a second mode of the same probe).
#
# Launch ONLY through run_chain.sh (it takes the rig lock; this script does
# not stop the gateway on its own without it):
#   setsid nohup ~/src/colibri-f1s0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f1s0/tools/hot-expert/f1_probe_chain.sh \
#       >> ~/bench/f1_probe.log 2>&1 < /dev/null &
#
# The build (`make -C c glm53 VK=1` in ~/src/colibri-f1s0, so the shaders and
# backend_vulkan.o this probe links against exist) happens BEFORE this is
# launched, not inside it, and needs no lock; this chain builds the PROBE
# binary itself (it is not part of `make glm53`).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)                 # e.g. ~/src/colibri-f1s0
PROBE="$HERE/f1_ring_probe"                     # built here, not checked in
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
SPV="$ROOT/c/shaders/qmatmul.spv"
GLOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
OUT=~/bench/f1_probe_out; mkdir -p "$OUT"
TAG=f1p$(date +%m%d%H%M)

# f1_ring_probe's own CLI positionals (dir, spv, slots, min_seconds, reps,
# where) still have to parse even in F1_DECODE_PROBE mode -- shard_open() and
# coli_vk_init() run before the mode check -- but slots/min_seconds/reps/
# where are IGNORED by the decode-probe path, which takes its own knobs
# (F1_DECODE_KS, F1_DECODE_WINDOWS, F1_DECODE_WHERE) below.
DECODE_KS=${F1_DECODE_KS:-1,2,3,4,6}
DECODE_WINDOWS=${F1_DECODE_WINDOWS:-2000}
DECODE_WHERE=${F1_DECODE_WHERE:-1}              # 1 = host ring, matching F2's shipped placement

echo "=== f1_probe_chain $TAG $(date -Is)"
echo "=== ks=$DECODE_KS windows=$DECODE_WINDOWS where=$DECODE_WHERE"
echo "=== THE GATEWAY IS DOWN FOR THE WHOLE CHAIN"

VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
VRAM_TOTAL() { cat "/sys/class/drm/card$1/device/mem_info_vram_total" 2>/dev/null || echo -1; }

precheck() {   # precheck <label>
  local label=$1
  echo "--- [$label] pre-checks $(date -Is)"
  for e in glm53 qwen38 qwen38-vk; do
    echo "[$label] pgrep -x $e: $(pgrep -x "$e" | wc -l)"
  done
  for c in 0 1 2; do
    echo "[$label] card$c vram_used=$(VRAM "$c") vram_total=$(VRAM_TOTAL "$c")"
  done
}

wait_no_engine() {
  for _ in $(seq 1 120); do
    if ! pgrep -x glm53 >/dev/null && ! pgrep -f "glm53[.]" >/dev/null; then return 0; fi
    sleep 2
  done
  echo "FATAL: a glm53 is still alive after 240 s"; return 1
}

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_MAXT -u COLI_TIMERS \
      -u F1_DECODE_PROBE -u F1_DECODE_KS -u F1_DECODE_WINDOWS -u F1_DECODE_WHERE \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh >> "$GLOG" 2>&1 < /dev/null &
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
  wait_no_engine
}

# --------------------------------------------------------------- main --
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  if pgrep -x glm53 >/dev/null 2>&1 || pgrep -f "glm53[.]" >/dev/null 2>&1; then
    pkill -9 -x glm53 2>/dev/null || true
    wait_no_engine
  fi
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== f1_probe_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results dir: $OUT (tag $TAG)"
  echo "--- accept_live.sh (the request AFTER the chain is part of the measurement)"
  "$HERE/accept_live.sh"
  alive_rc=$?
  echo "--- accept_live.sh exit=$alive_rc"
  [ "$rc" -eq 0 ] && rc=$alive_rc
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

precheck "chain-start"
[ -f "$SPV" ] || { echo "FATAL: $SPV missing -- run make -C c glm53 VK=1 first"; exit 1; }

stop_gateway || exit 1
precheck "after-stop"

echo "--- VRAM preflight (>= 2 GiB free on every card -- the gateway is down, this should be all of it)"
MIN_FREE=$((2 * 1024 * 1024 * 1024))
fail=0
for c in 0 1 2; do
  used=$(VRAM "$c"); total=$(VRAM_TOTAL "$c")
  if [ "$used" -lt 0 ] || [ "$total" -lt 0 ]; then echo "card$c: FATAL unreadable VRAM"; fail=1; continue; fi
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
  [ "$free" -ge "$MIN_FREE" ] || { echo "card$c: FATAL free=$free < $MIN_FREE"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "FATAL: insufficient free VRAM after stopping the gateway, aborting"; exit 1; }

echo "--- build (f2_ring_probe.c, F1's mode included, links the engine's own backend_vulkan.c)"
BUILD_LOG="$OUT/${TAG}_build.log"
gcc -O2 -fopenmp -DCOLI_VULKAN "$HERE/f2_ring_probe.c" "$ROOT/c/backend_vulkan.c" \
    -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
  echo "FATAL: probe build failed rc=$rc"; cat "$BUILD_LOG"; exit 1
fi
echo "build ok -> $PROBE"

echo "--- run (8 threads pinned, as every recorded number on this box)"
PROBE_OUT="$OUT/${TAG}.txt"
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    COLI_VK_SHADERS="$ROOT/c/shaders" \
    F1_DECODE_PROBE=1 F1_DECODE_KS="$DECODE_KS" F1_DECODE_WINDOWS="$DECODE_WINDOWS" \
    F1_DECODE_WHERE="$DECODE_WHERE" \
    "$PROBE" "$GLM_SNAP" "$SPV" 8 1.0 1 "$DECODE_WHERE" > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then echo "FATAL: probe exited non-zero"; cat "$PROBE_OUT"; exit 1; fi

echo "--- ROW lines (the falsifier's own numbers)"
grep '^ROW ' "$PROBE_OUT"
echo "--- INFO / WARN (includes the linear-fit lines: fixed vs marginal ms/window)"
grep -E '^(INFO|WARN) ' "$PROBE_OUT"

precheck "post-probe"
echo "=== f1_probe_chain body done $(date -Is) -- exit trap runs restart/accept_live.sh next"
exit 0
