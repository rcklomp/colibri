#!/usr/bin/env bash
# run_m5.sh — runs the M5 launch/sync-floor binary and records its output.
#
# NOT run by the agent that built this harness: this script starts an
# engine-adjacent process that opens /dev/kfd (GPU 0, and briefly all three
# for the P2P section) the moment it runs. Only the orchestrating session,
# under the rig lock with GLM idle, runs this.
#
# Usage: ./run_m5.sh [path-to-m5_bench-binary]
#   Default binary path: ./m5_bench (next to this script)
#
# Writes:
#   ~/bench/m5/m5_<timestamp>.txt   (key=value lines, stdout of the binary)
#   ~/bench/m5/m5_<timestamp>.json  (same data as JSON, written by the binary)

set -euo pipefail

BIN="${1:-$(dirname "$0")/m5_bench}"
OUT_DIR="$HOME/bench/m5"
TS="$(date -u +%Y%m%d_%H%M%S)"
TXT="$OUT_DIR/m5_${TS}.txt"
JSON="$OUT_DIR/m5_${TS}.json"

mkdir -p "$OUT_DIR"

if [ ! -x "$BIN" ]; then
    echo "error: binary not found or not executable: $BIN" >&2
    exit 1
fi

echo "running $BIN, writing $TXT and $JSON" >&2
"$BIN" --json "$JSON" | tee "$TXT"
rc=${PIPESTATUS[0]}

if [ "$rc" -ne 0 ]; then
    echo "m5_bench exited $rc (CHECK_FAIL if the numerics oracle failed — see $TXT)" >&2
fi

exit "$rc"
