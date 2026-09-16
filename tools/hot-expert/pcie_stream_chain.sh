#!/bin/bash
# pcie_stream_chain.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md §4 item X6: the
# measurement X6's own row derived instead of ran. Builds and runs
# pcie_stream_probe (tools/hot-expert/pcie_stream_probe.cpp) to find the
# aggregate host->device bandwidth when 1, 2 or 3 RX 7900 XTX cards pull
# different 14 MiB blocks (one GLM routed expert) concurrently, pinned and
# pageable host memory, plus a 256 MiB pinned block for the large-transfer
# asymptote, and an 8-thread DRAM-read control for comparison.
#
# UNLIKE every other chain in this directory, this one does NOT stop the
# owner's gateway: the gateway is idle between requests (this is a
# measurement window, not a live-traffic one) and each card carries the
# GLM tier's ~1.4 GiB VRAM headroom above its expert preload, which is
# enough for the probe's small H2D staging buffers. The chain still checks
# free VRAM itself (>= 600 MiB per card) before starting and refuses
# otherwise, rather than assuming the headroom is there. It still runs
# through run_chain.sh (rig lock only -- no gateway stop/restart is needed
# or performed here; run_chain.sh's own end-of-chain /v1/models check is
# the in-service proof).
#
# A 63 GB gpt-oss-120b download is in flight on the NVMe (unrelated to this
# item) and must not share the disk with this measurement, even though the
# probe itself never touches the NVMe after its own binary is loaded into
# the page cache -- the concern is contention noise on repeat counts, not
# correctness, so the chain waits (bounded) for it to clear rather than
# racing it.
#
# Launch through run_chain.sh, never directly:
#   setsid nohup ~/src/colibri-pcie/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-pcie/tools/hot-expert/pcie_stream_chain.sh \
#       >> ~/bench/pcie_stream.log 2>&1 < /dev/null &
set -u
TAG=pcie_$(date +%m%d%H%M)
OUT=~/bench/pcie_out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/pcie_stream_probe"
ROCM_ROOT=$(~/venvs/rocm/bin/rocm-sdk path --root)

VRAM_USED() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }
VRAM_TOTAL() { cat "/sys/class/drm/card$1/device/mem_info_vram_total" 2>/dev/null || echo -1; }

echo "=== pcie_stream_chain $TAG $(date -Is) (venv ROCm $ROCM_ROOT)"
echo "note: the gateway is NOT stopped by this chain (see header)."

echo "--- step 0: wait for the gpt-oss-120b download to clear (bounded, <= 25 min)"
DONE_MARK=~/models/gpt-oss-120b/.download_done
released_reason=""
t0=$(date +%s)
for _ in $(seq 1 50); do
  if [ -f "$DONE_MARK" ]; then
    released_reason="download_done marker present"
    break
  fi
  if ! pgrep -f "hf [d]ownload" >/dev/null; then
    released_reason="no hf download process running"
    break
  fi
  now=$(date +%s)
  if [ $((now - t0)) -ge 1500 ]; then
    released_reason="25 min wait elapsed, proceeding anyway"
    break
  fi
  sleep 30
done
[ -n "$released_reason" ] || released_reason="loop exhausted"
echo "released: $released_reason"

echo "--- step 1: preflight VRAM check (>= 600 MiB free on every card; print only, no changes)"
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

echo "--- step 2: build (no lock needed for a build, but we hold it already via run_chain.sh)"
[ -x "$PROBE" ] || {
  echo "building $PROBE"
  BUILD_LOG="$OUT/${TAG}_build.log"
  CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15 \
    "$ROCM_ROOT/bin/hipcc" --offload-arch=gfx1100 -O2 -fopenmp \
    "$HERE/pcie_stream_probe.cpp" -o "$PROBE" > "$BUILD_LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -x "$PROBE" ]; then
    echo "FATAL: build failed rc=$rc, see $BUILD_LOG"
    cat "$BUILD_LOG"
    exit 1
  fi
  echo "build ok -> $PROBE"
}

echo "--- step 3: run the probe (min 2 s/trial, interleaved 1,2,3,3,2,1)"
PROBE_OUT="$OUT/${TAG}.txt"
env LD_LIBRARY_PATH="$ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    "$PROBE" 2.0 > "$PROBE_OUT" 2>&1
rc=$?
echo "probe rc=$rc -> $PROBE_OUT"
if [ "$rc" -ne 0 ]; then
  echo "FATAL: probe exited non-zero, see $PROBE_OUT"
  cat "$PROBE_OUT"
  exit 1
fi

echo "--- step 4: ROW lines"
grep '^ROW ' "$PROBE_OUT"

echo "--- step 5: postflight VRAM check (should be back near preflight levels)"
for c in 0 1 2; do
  used=$(VRAM_USED "$c")
  total=$(VRAM_TOTAL "$c")
  free=$((total - used))
  echo "card$c: vram_used=$used vram_total=$total vram_free=$free"
done

echo "=== pcie_stream_chain done tag=$TAG $(date -Is)"
exit 0
