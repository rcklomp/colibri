#!/usr/bin/env bash
# run_m1.sh — runs the M1 routed-expert-kernel binaries (candidate 1: ggml/
# llama.cpp, candidate 2: hipFire, candidate 3: native IQ3_S/IQ4_NL) and
# records their output.
#
# NOT run by the agent that built this harness: all three binaries open the
# GPUs the moment they run (ggml_backend_cuda_init(0) / hipMalloc). Only the
# orchestrating session, under the rig lock (tools/hot-expert/rig_lock.sh)
# with GLM idle, runs this — see
# tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, "Build and run discipline".
#
# Candidate 3 here is m1_native: hipFire's kernel shape reading the experts in
# their NATIVE GGUF formats (gate/up IQ3_S, down IQ4_NL), K_TOP=10, no
# requantisation and no padding (design rev 7 section 9.3, L0 step 1). It
# links ggml's CPU libs to quantise the weights and to dequantise them for
# the oracle, so it needs the llama.cpp lib dir on LD_LIBRARY_PATH just as
# m1_ggml does. Colibri's own Vulkan path (qmatmul_gate_up.comp /
# qmatmul_tile.comp via c/backend_vulkan.c) is still NOT built and NOT run
# here.
#
# Note m1_native spends minutes in host-side IQ3_S quantisation before it
# prints anything (512 experts, single-threaded inside ggml) — the same cost
# m1_ggml pays. A silent first few minutes is expected, not a hang.
#
# Usage: ./run_m1.sh [path-to-this-dir]
#   Default: the directory this script lives in (m1_ggml, m1_hipfire and
#   m1_native are expected next to it, from `make all`).
#
# Writes:
#   ~/bench/m1/m1_<timestamp>.txt   (key=value lines, both binaries' stdout,
#                                     each candidate's block separated by a
#                                     `# --- <candidate> ---` marker)
#   ~/bench/m1/m1_<timestamp>.json  (the same key=value pairs, keyed, plus
#                                     an exit_code per candidate)

set -uo pipefail

DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LLAMA_LIB_DIR="/home/ronald/src/llama-glm53/build-hip/bin"
DOCKER_IMAGE="rocm/dev-ubuntu-24.04:7.14.0-full"
OUT_DIR="$HOME/bench/m1"
TS="$(date -u +%Y%m%d_%H%M%S)"
TXT="$OUT_DIR/m1_${TS}.txt"
JSON="$OUT_DIR/m1_${TS}.json"

mkdir -p "$OUT_DIR"

DOCKER_GPU_FLAGS=(--device /dev/kfd --device /dev/dri --group-add video
                   --security-opt seccomp=unconfined --ipc=host)

# Runs one binary in docker with GPU access and prints its stdout. The
# caller captures this with `out=$(run_one ...) ; rc=$?` -- deliberately NOT
# using `set -e` here so a CHECK_FAIL (exit 1) from candidate 1 does not
# skip candidate 2.
run_one() {
    local bin="$1" ld_path="$2"
    if [ ! -x "$DIR/$bin" ]; then
        echo "error: binary not found or not executable: $DIR/$bin" >&2
        return 127
    fi
    docker run --rm \
        "${DOCKER_GPU_FLAGS[@]}" \
        -v /home/ronald:/home/ronald -w "$DIR" \
        -e LD_LIBRARY_PATH="$ld_path" \
        "$DOCKER_IMAGE" "./$bin"
}

: > "$TXT"

echo "running m1_ggml ($DIR/m1_ggml)..." >&2
GGML_OUT="$(run_one "m1_ggml" "/opt/rocm/lib:${LLAMA_LIB_DIR}")"
GGML_RC=$?
{
    echo "# --- m1_ggml (candidate 1: llama.cpp / ggml) ---"
    printf '%s\n' "$GGML_OUT"
    echo "# m1_ggml exit code: ${GGML_RC}"
} >> "$TXT"

echo "running m1_hipfire ($DIR/m1_hipfire)..." >&2
HIPFIRE_OUT="$(run_one "m1_hipfire" "/opt/rocm/lib")"
HIPFIRE_RC=$?
{
    echo "# --- m1_hipfire (candidate 2: hipFire) ---"
    printf '%s\n' "$HIPFIRE_OUT"
    echo "# m1_hipfire exit code: ${HIPFIRE_RC}"
} >> "$TXT"

echo "running m1_native ($DIR/m1_native)..." >&2
NATIVE_OUT="$(run_one "m1_native" "/opt/rocm/lib:${LLAMA_LIB_DIR}")"
NATIVE_RC=$?
{
    echo "# --- m1_native (candidate 3: native IQ3_S/IQ4_NL, hipFire shape) ---"
    printf '%s\n' "$NATIVE_OUT"
    echo "# m1_native exit code: ${NATIVE_RC}"
} >> "$TXT"

echo "wrote $TXT" >&2

# key=value -> a flat JSON object, dependency-free (no python/numpy on this
# rig, see CLAUDE.md's "The rig has no numpy" -- avoid jq/python here too so
# this always runs regardless of what else is installed).
{
    echo "{"
    printf '  "m1_ggml_exit_code": %d,\n' "$GGML_RC"
    printf '  "m1_hipfire_exit_code": %d,\n' "$HIPFIRE_RC"
    printf '  "m1_native_exit_code": %d,\n' "$NATIVE_RC"
    grep -E '^[a-zA-Z_][a-zA-Z0-9_]*=' "$TXT" | while IFS='=' read -r k v; do
        # Quote the value unless it parses as a plain (possibly signed,
        # possibly decimal) number.
        if [[ "$v" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            printf '  "%s": %s,\n' "$k" "$v"
        else
            printf '  "%s": "%s",\n' "$k" "$v"
        fi
    done | sed '$ s/,$//'
    echo "}"
} > "$JSON"

echo "wrote $JSON" >&2

if [ "$GGML_RC" -ne 0 ] || [ "$HIPFIRE_RC" -ne 0 ] || [ "$NATIVE_RC" -ne 0 ]; then
    echo "one or more candidates exited non-zero (1 = CHECK_FAIL, see $TXT)" >&2
    exit 1
fi
exit 0
