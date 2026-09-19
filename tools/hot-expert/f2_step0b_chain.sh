#!/bin/bash
# f2_step0b_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, step 0b:
# does the copy go away entirely? §F2-STEP0's ring pipeline reached 38.6
# GB/s aggregate on three cards, below the plan's 60 GB/s bar, even though
# CPU fill alone reached 59+ GB/s and F0's GPU read alone reached 62-63
# GB/s -- either the ring's own handshake overhead, or DRAM traffic roughly
# tripling (fill read + staging write + PCIe read of the same bytes), and
# both go away if the memcpy-into-staging step is removed. §VK-STREAM found
# VK_EXT_external_memory_host import fails ONLY for file-backed memory;
# this step imports ANONYMOUS host memory (populated once from the shards)
# and reads it in place with F0's own shader-host-read dispatch --
# tools/hot-expert/staging_import_probe.c.
#
# Same chain shape as f2_step0_chain.sh / vk_stream_chain.sh: run only via
# run_chain.sh (rig lock only -- gateway NOT stopped, idle overnight, this
# is a measurement window), VRAM preflight, build, run, ROW/INFO lines.
# Peak extra host RAM at any instant is ONE region (<=96 GiB), never a sum
# of several -- the probe destroys/unmaps each size before allocating the
# next. Bounded to <= 15 min of rig time by the coordinator's instruction;
# the probe itself stops its scale sweep at the first size where mmap or
# import fails on any device.
#
# Launch through run_chain.sh, never directly:
#   setsid nohup ~/src/colibri-f2s0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f2s0/tools/hot-expert/f2_step0b_chain.sh \
#       >> ~/bench/f2_step0b.log 2>&1 < /dev/null &
set -u
TAG=f2s0b_$(date +%m%d%H%M)
OUT=~/bench/f2_step0b_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/staging_import_probe"
SHARD_DIR="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"

echo "=== f2_step0b_chain $TAG $(date -Is)"
echo "note: the gateway is NOT stopped by this chain (see header) -- idle overnight, measurement window not live traffic."

echo "--- step 0: preflight -- print only, no changes"
echo "probe binary: $PROBE (built below)"
echo "shard dir: $SHARD_DIR"
echo "pgrep glm53:"; pgrep -x glm53 || echo "(not running)"
nproc
free -g
echo "MemAvailable:"; grep MemAvailable /proc/meminfo
echo "recent gateway requests, BEFORE:"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable or no requests yet)"

echo "--- step 1: preflight VRAM check (>= 600 MiB free on every card; print only, no changes)"
VRAM_USED() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
VRAM_TOTAL() { cat "/sys/class/drm/card$1/device/mem_info_vram_total" 2>/dev/null || echo -1; }
MIN_FREE=$((600 * 1024 * 1024))
fail=0
for c in 0 1 2; do
  used=$(VRAM_USED "$c"); total=$(VRAM_TOTAL "$c")
  if [ "$used" -lt 0 ] || [ "$total" -lt 0 ]; then echo "card$c: FATAL could not read vram_used/vram_total"; fail=1; continue; fi
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
  [ "$free" -ge "$MIN_FREE" ] || { echo "card$c: FATAL free=$free < required=$MIN_FREE"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "FATAL: insufficient free VRAM on at least one card, aborting"; exit 1; }

echo "--- step 2: build"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
REDUCE_SPV="$HERE/vk_stream_reduce.spv"
BUILD_LOG="$OUT/${TAG}_build.log"
echo "building $PROBE"
gcc -O2 -pthread "$HERE/staging_import_probe.c" -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then echo "FATAL: probe build failed rc=$rc, see $BUILD_LOG"; cat "$BUILD_LOG"; exit 1; fi
echo "build ok -> $PROBE"
if [ ! -f "$REDUCE_SPV" ]; then
  echo "compiling $REDUCE_SPV (F0b's shader-host-read shader, unmodified, reused here)"
  glslc --target-env=vulkan1.2 "$HERE/vk_stream_reduce.comp" -o "$REDUCE_SPV" >> "$BUILD_LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "$REDUCE_SPV" ]; then echo "FATAL: glslc failed rc=$rc, see $BUILD_LOG"; cat "$BUILD_LOG"; exit 1; fi
fi
echo "shader ok -> $REDUCE_SPV"

echo "--- step 3: run the probe (scale sweep 8/32/64/96 GiB x {anon,anon-huge}, stops at first failure)"
PROBE_OUT="$OUT/${TAG}.txt"
env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    "$PROBE" "$SHARD_DIR" "$REDUCE_SPV" > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then echo "FATAL: probe exited non-zero, see $PROBE_OUT"; cat "$PROBE_OUT"; exit 1; fi

echo "--- step 4: ROW lines"
grep '^ROW ' "$PROBE_OUT"

echo "--- step 5: INFO / WARN lines"
grep -E '^(INFO|WARN) ' "$PROBE_OUT"

echo "--- step 6: postflight VRAM check"
for c in 0 1 2; do
  used=$(VRAM_USED "$c"); total=$(VRAM_TOTAL "$c"); free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
done
echo "MemAvailable after:"; grep MemAvailable /proc/meminfo

echo "recent gateway requests, AFTER:"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable)"

echo "=== f2_step0b_chain done tag=$TAG $(date -Is)"
exit 0
