#!/bin/bash
# rome_bench.sh: Run one benchmark configuration on the rome box.
# Drops caches, warms one model, verifies residency with fincore, pins 8 threads,
# sets GLM caps (COLI_VK_EXPERTS2/3=1695, .glm53_explain.bin), runs datapoint.py,
# and appends a row to the record.
#
# Usage: rome_bench.sh [engine] [config-name]
# Engines: qwen38-vk (default), qwen38, glm53
# Gate: two consecutive runs of the same config within 3%

set -eu

# Parameters (with defaults)
ENGINE="${1:-qwen38-vk}"
CONFIG_NAME="${2:-baseline}"
REMOTE_HOST="rome"

# Run the benchmark on the remote box via SSH
ssh "$REMOTE_HOST" bash -s "$ENGINE" "$CONFIG_NAME" << 'REMOTE_SCRIPT'
set -eu

COLIBRI_SRC="${HOME}/src/colibri"
RECORD_PATH="${COLIBRI_SRC}/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md"

# Engine parameters passed from local script
ENGINE="$1"
CONFIG_NAME="$2"

# Engine-specific configuration
case "$ENGINE" in
  qwen38)
    # CPU path: Q38_VULKAN deliberately unset.
    ENGINE_BIN="${COLIBRI_SRC}/c/qwen38"
    MODEL_SNAP="${HOME}/models/Qwen3.8-Flash-Next-FP8"
    THREADS=8
    CAP=512
    MAX_NEW=80
    ;;
  qwen38-vk)
    # The GPU expert tier is gated on Q38_VULKAN=1 (qwen38_core.h:989), and
    # dev2/dev3 additionally on COLI_VK_DEV2/3 being SET -- unset means "don't
    # try", not "auto" (:1001). Without these the qwen38-vk binary runs as a
    # plain CPU engine and silently reports CPU numbers.
    ENGINE_BIN="${COLIBRI_SRC}/c/qwen38-vk"
    MODEL_SNAP="${HOME}/models/Qwen3.8-Flash-Next-FP8"
    THREADS=8
    CAP=512
    MAX_NEW=80
    export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    # The shader path defaults to "shaders/qmatmul.spv" RELATIVE TO CWD, and
    # this script runs datapoint.py from the repo root where no such directory
    # exists (they live in c/shaders). Absolute, so CWD stops mattering.
    export COLI_VK_SHADERS="${COLIBRI_SRC}/c/shaders"
    ;;
  glm53)
    # 182 GiB int4 checkpoint. Vulkan is gated on COLI_VULKAN=1 (glm53.c:2118)
    # and dev2/dev3 on COLI_VK_DEV2/3 as above.
    ENGINE_BIN="${COLIBRI_SRC}/c/glm53"
    MODEL_SNAP="${HOME}/models/GLM-5.3-Flash-colibri-int4-g64"
    THREADS=8
    CAP=512
    MAX_NEW=64
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="${COLIBRI_SRC}/c/shaders"   # see qwen38-vk note above
    # COLI_USAGE_PATH is read at startup to fill the tier (glm53.c:2137) AND
    # rewritten by a destructor at exit (:1563) with this run's accumulated
    # counts. Pointing it at ~/.glm53_explain.bin directly would therefore let
    # each measurement mutate the histogram the next one preloads from, so
    # runs would not be comparable. Copy it per run and use the copy, leaving
    # the canonical file untouched.
    GLM_HIST_SRC="${HOME}/.glm53_explain.bin"
    GLM_HIST="/tmp/rome_bench_glm_hist_$$.bin"
    if [ ! -f "$GLM_HIST_SRC" ]; then
      echo "Error: GLM histogram $GLM_HIST_SRC not found; the tier would preload empty" >&2
      exit 1
    fi
    cp "$GLM_HIST_SRC" "$GLM_HIST"
    export COLI_USAGE_PATH="$GLM_HIST"
    trap 'rm -f "$GLM_HIST"' EXIT
    ;;
  *)
    echo "Unknown engine: $ENGINE" >&2
    exit 1
    ;;
esac

echo "[rome_bench] Starting benchmark: engine=$ENGINE config=$CONFIG_NAME"
echo "[rome_bench] ENGINE_BIN=$ENGINE_BIN"
echo "[rome_bench] MODEL_SNAP=$MODEL_SNAP"

# Check for concurrent engines
if pgrep -x qwen38 qwen38-vk glm53 >/dev/null 2>&1; then
  echo "Error: Another engine is already running" >&2
  exit 1
fi

# Drop caches (requires sudo; may prompt for password)
echo "[rome_bench] Dropping page cache..."
sync
if ! echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
  echo "[rome_bench] Warning: Cache eviction requires sudo password. Run in interactive session:"
  echo "              ssh rome sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'"
  echo "[rome_bench] Or configure sudoers for passwordless cache drop"
  echo "[rome_bench] Proceeding with measurement (page cache may be warm)"
fi
sleep 1

# Warm the model: read it sequentially into page cache
if [ -d "$MODEL_SNAP" ]; then
  echo "[rome_bench] Warming model at $MODEL_SNAP..."
  # Find the largest shard and read it to verify warmup works
  find "$MODEL_SNAP" -type f \( -name "*.safetensors" -o -name "*.bin" \) -print0 | while IFS= read -r -d '' file; do
    cat "$file" > /dev/null 2>&1
    echo "[rome_bench] Read $file"
  done

  # Verify residency with fincore
  echo "[rome_bench] Verifying residency with fincore..."
  fincore "$MODEL_SNAP"/*.safetensors 2>/dev/null | tail -1 || echo "[rome_bench] fincore check complete"
else
  echo "Warning: Model snapshot not found at $MODEL_SNAP" >&2
  exit 1
fi

# Set environment for the measurement (8 threads on physical cores)
export OMP_NUM_THREADS=$THREADS
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# GLM-specific caps
export COLI_VK_EXPERTS2=1695
export COLI_VK_EXPERTS3=1695
export COLI_TIMERS=1

# Run datapoint.py in persistent mode
echo "[rome_bench] Running datapoint.py..."
cd "$COLIBRI_SRC"

# Run the benchmark and capture output
OUTPUT=$(python3 c/tools/datapoint.py \
  --engine "$ENGINE_BIN" \
  --snap "$MODEL_SNAP" \
  --mode persistent \
  --cap "$CAP" \
  --max-new "$MAX_NEW" \
  --rotating-runs 4 \
  --warm-runs 1 \
  2>&1) || {
  echo "Error: datapoint.py failed" >&2
  exit 1
}

echo "$OUTPUT"

# Assert the configuration we asked for is the one that actually came up.
# datapoint.py prints a "| GPU |" row only when the engine reports a device,
# so its absence on a Vulkan config means the tier never initialised and the
# numbers are CPU numbers. Failing here is the point: a mislabelled row in the
# record is worse than no row (this fired for real on 2026-09-04, when a
# qwen38-vk run with Q38_VULKAN unset was written up as "3 GPUs").
# Assert on the ENGINE's own preload lines, not on datapoint.py's "| GPU |"
# row: that row comes from the SERVE hwinfo frame, which qwen38 does not send
# even with the tier fully up, so it is a false negative.
tier_fail() {
  echo "Error: $ENGINE ran with NO GPU expert tier -- these would be CPU numbers." >&2
  echo "       $1" >&2
  echo "       Check Q38_VULKAN/COLI_VULKAN, COLI_VK_DEV2/3, COLI_VK_SHADERS," >&2
  echo "       and (glm53) COLI_USAGE_PATH. Refusing to record." >&2
  exit 1
}
case "$ENGINE" in
  qwen38-vk)
    echo "$OUTPUT" | grep -q 'Vulkan tier preloaded' \
      || tier_fail "no '[qwen38] Vulkan tier preloaded' line (qwen38_core.h:1054)."
    n=$(echo "$OUTPUT" | sed -n 's/.*Vulkan tier preloaded \([0-9][0-9]*\) of.*/\1/p' | head -1)
    [ -n "$n" ] && [ "$n" -gt 0 ] || tier_fail "tier preloaded 0 experts."
    echo "[rome_bench] GPU tier confirmed up ($n experts):"
    ;;
  glm53)
    echo "$OUTPUT" | grep -q 'no usage history loaded' \
      && tier_fail "glm53 reported 'tier empty' (glm53.c:1611): COLI_USAGE_PATH did not load."
    echo "$OUTPUT" | grep -q '\[VK\] preload:' \
      || tier_fail "no '[VK] preload:' line (glm53.c:1669)."
    # The recorded GLM configuration is all three devices; a silently missing
    # dev2/dev3 is the documented way this box produces a wrong GLM baseline.
    for d in dev2 dev3; do
      echo "$OUTPUT" | grep -q "\[VK\] $d ready" \
        || tier_fail "$d never came up; this is not the recorded 3-GPU configuration."
    done
    echo "[rome_bench] GPU tier confirmed up (3 devices):"
    ;;
esac
echo "$OUTPUT" | grep -E 'Vulkan tier preloaded|\[VK\] (preload|dev[23] ready)' | sed 's/^/    /'

# Extract rotating median tok/s from the Workload summary table.
# Row shape: | **rotating prompts (primary)** | n | median tok/s | sigma | p95 | hit |
# so field 4 (1-indexed after the leading empty split) is the median.
ROTATING_TOK_S=$(echo "$OUTPUT" | grep '\*\*rotating prompts (primary)\*\*' | awk -F'|' '{print $4}' | xargs 2>/dev/null || echo "")

if [ -z "$ROTATING_TOK_S" ]; then
  echo "Error: Could not extract rotating tok/s from output" >&2
  echo "Debug: Full output was:" >&2
  echo "$OUTPUT" | tail -30 >&2
  exit 1
fi

# Both pulled from the Workload summary / Decode tables:
# "| warm-identical (upper bound) | n | median tok/s | ..." -> field 4
# "| cold request (...) | prompt tok | completion tok | request s | TTFT s | decode tok/s | ..." -> field 7
WARM_TOK_S=$(echo "$OUTPUT" | grep 'warm-identical (upper bound)' | grep -v '^\*\*' | awk -F'|' '{print $4}' | xargs 2>/dev/null || echo "n/a")
COLD_TOK_S=$(echo "$OUTPUT" | grep 'cold request' | awk -F'|' '{print $7}' | xargs 2>/dev/null || echo "n/a")
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo ""
echo "=========================================="
echo "RESULT: $ENGINE config=$CONFIG_NAME"
echo "Rotating median: $ROTATING_TOK_S tok/s"
echo "=========================================="

# Append a mechanical row to the record's datapoint log (create the section on first use).
if [ -f "$RECORD_PATH" ]; then
  if ! grep -q "^## Datapoint log (tools/rome_bench.sh)" "$RECORD_PATH"; then
    {
      echo ""
      echo "## Datapoint log (tools/rome_bench.sh)"
      echo ""
      echo "Mechanical rows appended by the harness itself, one per run. Not curated prose;"
      echo "cross-reference against the narrative sections above for interpretation."
      echo ""
      echo "| timestamp (UTC) | engine | config | cap | max-new | threads | rotating median | warm-identical | cold |"
      echo "|---|---|---|---:|---:|---:|---:|---:|---:|"
    } >> "$RECORD_PATH"
  fi
  echo "| $TS | $ENGINE | $CONFIG_NAME | $CAP | $MAX_NEW | $THREADS | ${ROTATING_TOK_S} tok/s | ${WARM_TOK_S} tok/s | ${COLD_TOK_S} tok/s |" >> "$RECORD_PATH"
  echo "[rome_bench] Appended row to $RECORD_PATH"
else
  echo "[rome_bench] Warning: record not found at $RECORD_PATH, row not appended" >&2
fi
REMOTE_SCRIPT

exit $?
