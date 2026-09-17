#!/bin/bash
# f2_step0_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, step 0:
# F0/F0b (record §VK-STREAM) measured the GPU-side leg of F2's streamed
# prefill (62-63 GB/s on three cards, shader-host-read on the backend's own
# queue family). F0's own single-thread memcpy into that staging buffer
# managed only 15 GB/s -- this step measures the CPU-side fill,
# multi-threaded, in the scattered access pattern F2 will actually have,
# and the fill overlapped with the GPU read in a real ring-buffer pipeline
# (tools/hot-expert/staging_fill_probe.c). Gate: pipeline aggregate on three
# cards >= 60 GB/s or F2's chunk arithmetic is off by 4x (plan §8.3, F2 row).
#
# Modelled directly on vk_stream_chain.sh: this chain does NOT stop the
# owner's gateway (same reasoning -- the gateway is idle between requests,
# this is a measurement window not a live-traffic one; the probe's own
# buffers, up to ~1.8 GiB of HOST memory plus small per-card VRAM
# allocations, are trivial next to the GLM tier's resident experts). It
# still runs through run_chain.sh for the rig lock only -- no gateway
# stop/restart here -- and checks free VRAM itself before starting.
#
# Launch through run_chain.sh, never directly:
#   setsid nohup ~/src/colibri-f2s0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f2s0/tools/hot-expert/f2_step0_chain.sh \
#       >> ~/bench/f2_step0.log 2>&1 < /dev/null &
set -u
TAG=f2s0_$(date +%m%d%H%M)
OUT=~/bench/f2_step0_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/staging_fill_probe"
SHARD_DIR="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"

echo "=== f2_step0_chain $TAG $(date -Is)"
echo "note: the gateway is NOT stopped by this chain (see header) -- it is idle overnight and this is a measurement window, not a live-traffic one."

echo "--- step 0: preflight -- print only, no changes"
echo "probe binary: $PROBE (built below)"
echo "shard dir: $SHARD_DIR"
ls -la "$SHARD_DIR"/*.safetensors | wc -l
echo "pgrep glm53:"; pgrep -x glm53 || echo "(not running)"
echo "pgrep qwen38:"; pgrep -x qwen38 || echo "(not running)"
echo "pgrep qwen38-vk:"; pgrep -x qwen38-vk || echo "(not running)"
nproc
free -g
echo "recent gateway requests, BEFORE (visible if a coincidental request lands mid-run):"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable or no requests yet)"

echo "--- step 1: preflight VRAM check (>= 600 MiB free on every card; print only, no changes)"
VRAM_USED() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
VRAM_TOTAL() { cat "/sys/class/drm/card$1/device/mem_info_vram_total" 2>/dev/null || echo -1; }
MIN_FREE=$((600 * 1024 * 1024))
fail=0
for c in 0 1 2; do
  used=$(VRAM_USED "$c")
  total=$(VRAM_TOTAL "$c")
  if [ "$used" -lt 0 ] || [ "$total" -lt 0 ]; then
    echo "card$c: FATAL could not read vram_used/vram_total"
    fail=1
    continue
  fi
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
  [ "$free" -ge "$MIN_FREE" ] || { echo "card$c: FATAL free=$free < required=$MIN_FREE"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "FATAL: insufficient free VRAM on at least one card, aborting"; exit 1; }

echo "--- step 2: build (VK_ICD_FILENAMES per vk_stream_chain.sh's convention)"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
REDUCE_SPV="$HERE/vk_stream_reduce.spv"
BUILD_LOG="$OUT/${TAG}_build.log"
echo "building $PROBE"
gcc -O2 -pthread "$HERE/staging_fill_probe.c" -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
  echo "FATAL: probe build failed rc=$rc, see $BUILD_LOG"
  cat "$BUILD_LOG"
  exit 1
fi
echo "build ok -> $PROBE"
echo "compiling $REDUCE_SPV (F0b's shader-host-read shader, unmodified, reused by this step's pipeline test)"
glslc --target-env=vulkan1.2 "$HERE/vk_stream_reduce.comp" -o "$REDUCE_SPV" >> "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$REDUCE_SPV" ]; then
  echo "FATAL: glslc failed rc=$rc, see $BUILD_LOG"
  cat "$BUILD_LOG"
  exit 1
fi
echo "shader ok -> $REDUCE_SPV"

echo "--- step 3: run the probe (min 2 s/trial, median of 5 reps per config)"
PROBE_OUT="$OUT/${TAG}.txt"
env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    "$PROBE" 2.0 "$SHARD_DIR" "$REDUCE_SPV" > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then
  echo "FATAL: probe exited non-zero, see $PROBE_OUT"
  cat "$PROBE_OUT"
  exit 1
fi

echo "--- step 4: ROW lines"
grep '^ROW ' "$PROBE_OUT"

echo "--- step 5: INFO / WARN lines (machine facts, residency, device setup)"
grep -E '^(INFO|WARN) ' "$PROBE_OUT"

echo "--- step 6: postflight VRAM check (should be back near preflight levels)"
for c in 0 1 2; do
  used=$(VRAM_USED "$c")
  total=$(VRAM_TOTAL "$c")
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
done

echo "recent gateway requests, AFTER (to spot a coincidental request that landed mid-run):"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable)"

echo "=== f2_step0_chain done tag=$TAG $(date -Is)"
exit 0
