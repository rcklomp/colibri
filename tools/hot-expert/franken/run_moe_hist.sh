#!/usr/bin/env bash
# run_moe_hist.sh <model> <prompt-file> <out> — runs moe_hist (M2,
# FRANKEN-ENGINE-DESIGN-2026-09-22.md section 4 row M2) on the GPUs, in the
# same docker image and pattern as tools/hot-expert/m1/run_m1.sh /
# run_m3.sh: --device /dev/kfd /dev/dri, root in the container (a non-root
# --user cannot open /dev/kfd, per run_m3.sh's own comment), --fit on so
# common_fit_params() sizes n_gpu_layers/tensor_split/context to the box's
# free VRAM the way llama-server does.
#
# NOT run by the agent that built this tool, and not run by moe_hist.cpp's
# own CPU-only build/test pass (see tools/hot-expert/franken/README.md's M2
# entry and the report that built this): the moment this script runs,
# moe_hist opens every GPU the fitted n_gpu_layers puts weights on. Only the
# orchestrating session, holding the rig lock
# (tools/hot-expert/rig_lock.sh) with GLM idle (`serve_alt.sh status`, or
# accept that this run competes with the live gateway for VRAM/PCIe and
# account for that in the numbers), runs this — launch it detached
# (`ssh -n -f rome 'setsid nohup ... &'`, CLAUDE.md's rig-driving section)
# and poll the log; a foreground run over ssh dies with the connection.
#
# Usage: ./run_moe_hist.sh <model-first-shard.gguf> <prompt-file> <out-dir>
#   Env overrides (all optional):
#     CTX=32768            context size (must be >= the prompt's token count)
#     BATCH=512            llama_decode chunk size
#     BUDGET_GB=4          per-card VRAM budget for the (d) miss-bytes calc
#                          -- NOT the card's full 24 GiB: this sizes the
#                          hottest-experts-replicated-per-card set design
#                          3.2 describes, alongside the resident placement
#                          the rest of a card's VRAM is already spent on.
#     N_GPU_LAYERS=999     passed through; --fit on may still lower it
#     THREADS=8            CPU threads for the parts that stay on the CPU
#     MOE_HIST_BIN=<path>  default: this script's directory / moe_hist
#
# Writes moe_hist's own outputs directly to <out-dir> (summary.txt,
# layer_<il>.csv per MoE layer) plus this run's stdout/stderr and exit code
# to <out-dir>/run.log.

set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <model-first-shard.gguf> <prompt-file> <out-dir>" >&2
    exit 1
fi

MODEL="$1"
PROMPT_FILE="$2"
OUT_DIR="$3"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_HIST_BIN="${MOE_HIST_BIN:-$DIR/moe_hist}"
DOCKER_IMAGE="rocm/dev-ubuntu-24.04:7.14.0-full"
LLAMA_LIB_DIR="/home/ronald/src/llama-glm53/build-hip/bin"

CTX="${CTX:-32768}"
BATCH="${BATCH:-512}"
BUDGET_GB="${BUDGET_GB:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
THREADS="${THREADS:-8}"

if [ ! -x "$MOE_HIST_BIN" ]; then
    echo "error: binary not found or not executable: $MOE_HIST_BIN (run 'make moe_hist' first)" >&2
    exit 127
fi
if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "error: prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"

DOCKER_GPU_FLAGS=(--device /dev/kfd --device /dev/dri --group-add video
                   --security-opt seccomp=unconfined --ipc=host)

echo "run_moe_hist: model=$MODEL prompt_file=$PROMPT_FILE out=$OUT_DIR" | tee "$RUN_LOG" >&2
echo "run_moe_hist: ctx=$CTX batch=$BATCH budget_gb=$BUDGET_GB n_gpu_layers=$N_GPU_LAYERS threads=$THREADS" | tee -a "$RUN_LOG" >&2

docker run --rm \
    "${DOCKER_GPU_FLAGS[@]}" \
    -v /home/ronald:/home/ronald -w "$DIR" \
    -e LD_LIBRARY_PATH="/opt/rocm/lib:${LLAMA_LIB_DIR}" \
    "$DOCKER_IMAGE" "./$(basename "$MOE_HIST_BIN")" \
        --model "$MODEL" \
        --prompt-file "$PROMPT_FILE" \
        --out "$OUT_DIR" \
        --ctx "$CTX" \
        --batch "$BATCH" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --fit on \
        --budget-gb "$BUDGET_GB" \
        --threads "$THREADS" \
    >> "$RUN_LOG" 2>&1
RC=$?

echo "run_moe_hist: exit code $RC" | tee -a "$RUN_LOG" >&2
echo "run_moe_hist: log at $RUN_LOG, summary at $OUT_DIR/summary.txt" >&2
exit "$RC"
