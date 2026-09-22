#!/usr/bin/env bash
# run_m3.sh — runs the M3 QSA-attention / Gated-DeltaNet microbenchmark
# (m3_attn) and records its output.
#
# NOT run by the agent that built this harness: the binary opens the GPU
# the moment it runs (hipSetDevice(0), hipMalloc). Only the orchestrating
# session, under the rig lock (tools/hot-expert/rig_lock.sh) with GLM idle,
# runs this — see tools/hot-expert/M3-BRIEF-2026-09-22.md and
# tools/hot-expert/M1-M5-BRIEF-2026-09-22.md's "Build and run discipline".
#
# Run as root in the container, like run_m1.sh/m5_chain: a non-root
# --user cannot open /dev/kfd.
#
# Usage: ./run_m3.sh [path-to-this-dir]
#   Default: the directory this script lives in (m3_attn is expected next
#   to it, from `make m3_attn`).
#
# Writes:
#   ~/bench/m3/m3_<timestamp>.txt   (key=value lines, m3_attn's stdout)
#   ~/bench/m3/m3_<timestamp>.json  (the same key=value pairs, from --json)

set -uo pipefail

DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DOCKER_IMAGE="rocm/dev-ubuntu-24.04:7.14.0-full"
OUT_DIR="$HOME/bench/m3"
TS="$(date -u +%Y%m%d_%H%M%S)"
TXT="$OUT_DIR/m3_${TS}.txt"
JSON="$OUT_DIR/m3_${TS}.json"

mkdir -p "$OUT_DIR"

DOCKER_GPU_FLAGS=(--device /dev/kfd --device /dev/dri --group-add video
                   --security-opt seccomp=unconfined --ipc=host)

if [ ! -x "$DIR/m3_attn" ]; then
    echo "error: binary not found or not executable: $DIR/m3_attn" >&2
    exit 127
fi

echo "running m3_attn ($DIR/m3_attn)..." >&2
docker run --rm \
    "${DOCKER_GPU_FLAGS[@]}" \
    -v /home/ronald:/home/ronald -w "$DIR" \
    -e LD_LIBRARY_PATH="/opt/rocm/lib" \
    "$DOCKER_IMAGE" "./m3_attn" --json "/home/ronald/bench/m3/m3_${TS}_inner.json" \
    > "$TXT" 2>&1
RC=$?

# The container's docker-internal --json path and the host path above are
# the same bind-mounted file (both under /home/ronald); copy it to the
# expected output name for a stable filename regardless of exit path.
if [ -f "$OUT_DIR/m3_${TS}_inner.json" ]; then
    mv "$OUT_DIR/m3_${TS}_inner.json" "$JSON"
fi

echo "m3_attn exit code: ${RC}" >> "$TXT"
echo "wrote $TXT" >&2
[ -f "$JSON" ] && echo "wrote $JSON" >&2

if [ "$RC" -ne 0 ]; then
    echo "m3_attn exited non-zero (1 = CHECK_FAIL, see $TXT)" >&2
    exit 1
fi
exit 0
