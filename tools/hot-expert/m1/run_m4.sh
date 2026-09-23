#!/usr/bin/env bash
# run_m4.sh — runs the M4 streaming-bandwidth-under-contention binary and
# records its output.
#
# NOT run by the agent that built this harness: this script starts an
# engine-adjacent process that opens /dev/kfd on all three cards the moment
# it runs (device discovery touches every card before any copy starts).
# Only the orchestrating session, under the rig lock with GLM idle (check
# `pgrep -x glm53`, `pgrep -x qwen38`, `pgrep -x qwen38-vk` first — one
# pattern per call — CLAUDE.md's rig-serialisation rule), runs this.
#
# Usage: ./run_m4.sh [path-to-m4_stream-binary] [extra args to the binary]
#   Default binary path: ./m4_stream (next to this script). The first
#   argument is treated as a binary-path override only if it names an
#   executable file; otherwise every argument is forwarded to m4_stream.
#   Extra args (all optional, forwarded to m4_stream): --tokens N (default 6),
#   --warmup N (default 1), --reps N (default 3), --compute-duty D
#   (default 0.5), --compute-max-secs S (default 30).
#
# Writes:
#   ~/bench/m4/m4_<timestamp>.txt   (key=value lines, stdout of the binary)
#   ~/bench/m4/m4_<timestamp>.json  (same data as JSON, written by the binary)
#
# Exact run command this script executes (for the record):
#   ./m4_stream --json ~/bench/m4/m4_<timestamp>.json [extra args]

set -euo pipefail

BIN="$(dirname "$0")/m4_stream"
if [ $# -gt 0 ] && [ -x "$1" ] && [ "${1#-}" = "$1" ]; then
    BIN="$1"
    shift
fi

OUT_DIR="$HOME/bench/m4"
TS="$(date -u +%Y%m%d_%H%M%S)"
TXT="$OUT_DIR/m4_${TS}.txt"
JSON="$OUT_DIR/m4_${TS}.json"

mkdir -p "$OUT_DIR"

if [ ! -x "$BIN" ]; then
    echo "error: binary not found or not executable: $BIN" >&2
    exit 1
fi

echo "running $BIN --json $JSON $*, writing $TXT" >&2
"$BIN" --json "$JSON" "$@" | tee "$TXT"
rc=${PIPESTATUS[0]}

if [ "$rc" -ne 0 ]; then
    echo "m4_stream exited $rc (CHECK_FAIL if the copy-correctness oracle failed — see $TXT)" >&2
fi

exit "$rc"
