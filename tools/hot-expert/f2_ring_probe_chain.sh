#!/bin/bash
# f2_ring_probe_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, the
# microbenchmark that runs BEFORE the engine is touched (design note
# tools/hot-expert/F2-STREAM-PREFILL-DESIGN-2026-09-19.md §4, assumption (a)):
# what does one wave of streamed experts cost -- fill alone, compute alone,
# and the two in the order the engine will run them?
#
# Like f2_step0_chain.sh this does NOT stop the owner's gateway: it runs
# through run_chain.sh for the RIG LOCK only. The probe's footprint is
# slots x 14.16 MB per card (48 slots = 680 MB) out of the 1.5-2.5 GB the
# 1695-expert count cap leaves free, so it fits in the reserve; the preflight
# below refuses to start if any card has less than 1.2 GiB free. The gateway
# being idle is checked by owui_report.sh before and after, as F2 step 0 did.
#
#   setsid nohup ~/src/colibri-f2/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f2/tools/hot-expert/f2_ring_probe_chain.sh \
#       >> ~/bench/f2_ring_probe.log 2>&1 < /dev/null &
set -u
TAG=f2rp_$(date +%m%d%H%M)
OUT=~/bench/f2_ring_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PROBE="$HERE/f2_ring_probe"
SHARD_DIR=${GLM_SNAP:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
SPV="$ROOT/c/shaders/qmatmul.spv"
SLOTS=${F2_SLOTS:-48}
SECS=${F2_SECS:-1.0}
REPS=${F2_REPS:-5}

echo "=== f2_ring_probe_chain $TAG $(date -Is)"
echo "note: the gateway is NOT stopped by this chain (see header)."
echo "probe=$PROBE shards=$SHARD_DIR spv=$SPV slots=$SLOTS secs=$SECS reps=$REPS"

echo "--- step 0: preflight (print only)"
echo "pgrep glm53:"; pgrep -x glm53 || echo "(not running)"
echo "pgrep qwen38:"; pgrep -x qwen38 || echo "(not running)"
echo "pgrep qwen38-vk:"; pgrep -x qwen38-vk || echo "(not running)"
nproc; free -g
echo "recent gateway requests, BEFORE:"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable)"

echo "--- step 1: VRAM preflight (>= 1.2 GiB free on every card)"
MIN_FREE=$((1200 * 1024 * 1024))
fail=0
for c in 0 1 2; do
  used=$(cat "/sys/class/drm/card$c/device/mem_info_vram_used" 2>/dev/null || echo -1)
  total=$(cat "/sys/class/drm/card$c/device/mem_info_vram_total" 2>/dev/null || echo -1)
  if [ "$used" -lt 0 ] || [ "$total" -lt 0 ]; then echo "card$c: FATAL unreadable VRAM"; fail=1; continue; fi
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
  [ "$free" -ge "$MIN_FREE" ] || { echo "card$c: FATAL free=$free < $MIN_FREE"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "FATAL: insufficient free VRAM, aborting"; exit 1; }

echo "--- step 2: build"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
BUILD_LOG="$OUT/${TAG}_build.log"
gcc -O2 -fopenmp -DCOLI_VULKAN "$HERE/f2_ring_probe.c" "$ROOT/c/backend_vulkan.c" \
    -o "$PROBE" -lvulkan -lm > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
  echo "FATAL: probe build failed rc=$rc"; cat "$BUILD_LOG"; exit 1
fi
echo "build ok -> $PROBE"
[ -f "$SPV" ] || { echo "FATAL: $SPV missing -- run make -C c glm53 VK=1 first"; exit 1; }

echo "--- step 3: run (8 threads pinned, as every recorded number on this box)"
PROBE_OUT="$OUT/${TAG}.txt"
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
    COLI_VK_SHADERS="$ROOT/c/shaders" \
    "$PROBE" "$SHARD_DIR" "$SPV" "$SLOTS" "$SECS" "$REPS" "${F2_WHERE:-2}" > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then echo "FATAL: probe exited non-zero"; cat "$PROBE_OUT"; exit 1; fi

echo "--- step 4: ROW lines"
grep '^ROW ' "$PROBE_OUT"
echo "--- step 5: INFO / WARN"
grep -E '^(INFO|WARN) ' "$PROBE_OUT"

echo "--- step 6: VRAM postflight"
for c in 0 1 2; do
  used=$(cat "/sys/class/drm/card$c/device/mem_info_vram_used" 2>/dev/null)
  total=$(cat "/sys/class/drm/card$c/device/mem_info_vram_total" 2>/dev/null)
  echo "card$c: vram_used=$used vram_total=$total vram_free=$((total - used))"
done
echo "recent gateway requests, AFTER:"
~/bench/owui_report.sh 3 2>&1 || echo "(owui_report.sh unavailable)"
echo "=== f2_ring_probe_chain done tag=$TAG $(date -Is)"
exit 0
