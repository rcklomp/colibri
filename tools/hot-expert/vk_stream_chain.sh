#!/bin/bash
# vk_stream_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F0 (+F0b): does
# Vulkan/RADV host->device streaming match the HIP rate §PCIE-STREAM
# measured (1 card 28.0 GB/s, three cards concurrent 61.4-62.0 GB/s)?
# Builds and runs tools/hot-expert/vk_stream_probe.c and its
# vk_stream_reduce.comp shader (F0b: the transfer-preferred queue family
# F0 originally used is RADV's SDMA engine, too slow to decide the gate on
# alone -- F0b adds hv-coherent-gfxq and shader-host-read on the backend's
# own queue family, backend_vulkan.c's pick).
#
# UNLIKE most chains in this directory, this one does NOT stop the owner's
# gateway, for the same reason pcie_stream_chain.sh doesn't: the gateway is
# idle between requests and this is a measurement window, not a live-traffic
# one. The probe's own buffers are small relative to a card's VRAM (dst
# 256 MiB, staging up to 1 GiB of HOST memory, not VRAM) next to the GLM
# tier's resident experts, so headroom is not in question -- but the chain
# still checks free VRAM itself (>= 600 MiB per card, via
# /sys/class/drm/card*/device/mem_info_vram_{used,total}) before starting
# and refuses otherwise, rather than assuming it. It still runs through
# run_chain.sh (rig lock only -- no gateway stop/restart here; run_chain.sh's
# own end-of-chain /v1/models check is the in-service proof).
#
# Launch through run_chain.sh, never directly:
#   setsid nohup ~/src/colibri-f0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f0/tools/hot-expert/vk_stream_chain.sh \
#       >> ~/bench/vk_stream.log 2>&1 < /dev/null &
set -u
TAG=vkstream_$(date +%m%d%H%M)
OUT=~/bench/vk_stream_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/vk_stream_probe"
SHARD="$HOME/models/GLM-5.3-Flash-colibri-int4-g64/model-00002-of-00062.safetensors"

echo "=== vk_stream_chain $TAG $(date -Is)"
echo "note: the gateway is NOT stopped by this chain (see header)."

echo "--- step 0: preflight -- print only, no changes"
echo "probe binary: $PROBE (built below if missing)"
echo "shard file for the page-cache analogue: $SHARD"
ls -la "$SHARD" 2>&1
echo "pgrep glm53:"; pgrep -x glm53 || echo "(not running)"
nproc
free -g

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

echo "--- step 2: build (VK_ICD_FILENAMES per llama-profile.sh's convention)"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
REDUCE_SPV="$HERE/vk_stream_reduce.spv"
BUILD_LOG="$OUT/${TAG}_build.log"
# Always rebuild (not "build if missing"): F0b changed vk_stream_probe.c and
# added vk_stream_reduce.comp, and a stale executable from F0's run must not
# be reused silently.
echo "building $PROBE"
gcc -O2 -pthread "$HERE/vk_stream_probe.c" -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
  echo "FATAL: probe build failed rc=$rc, see $BUILD_LOG"
  cat "$BUILD_LOG"
  exit 1
fi
echo "build ok -> $PROBE"
echo "compiling $REDUCE_SPV (F0b shader-host-read path)"
glslc --target-env=vulkan1.2 "$HERE/vk_stream_reduce.comp" -o "$REDUCE_SPV" >> "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -f "$REDUCE_SPV" ]; then
  echo "FATAL: glslc failed rc=$rc, see $BUILD_LOG"
  cat "$BUILD_LOG"
  exit 1
fi
echo "shader ok -> $REDUCE_SPV"

echo "--- step 3: run the probe (min 2 s/trial, interleaved 1,2,3,3,2,1)"
PROBE_OUT="$OUT/${TAG}.txt"
env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    "$PROBE" 2.0 "$SHARD" "$REDUCE_SPV" > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then
  echo "FATAL: probe exited non-zero, see $PROBE_OUT"
  cat "$PROBE_OUT"
  exit 1
fi

echo "--- step 4: ROW lines"
grep '^ROW ' "$PROBE_OUT"

echo "--- step 5: INFO breakdown / setup lines (memory types, bus-id mapping, batch size)"
grep '^INFO ' "$PROBE_OUT"

echo "--- step 6: postflight VRAM check (should be back near preflight levels)"
for c in 0 1 2; do
  used=$(VRAM_USED "$c")
  total=$(VRAM_TOTAL "$c")
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
done

echo "=== vk_stream_chain done tag=$TAG $(date -Is)"
exit 0
