#!/bin/bash
# glm_gemm_micro_chain.sh -- PF2a: the TRUNK GEMM MICROBENCHMARK on ONE card, no model load (2026-10-09; franken-engine branch pf2-wmma, franken/decode/bench_gemm_wmma.hip).
# Prefill's trunk GEMMs (Q8_0 weights x a 1 024-token chunk) are 0.95 ms/token of 3.30 (record §L5-PF9) and run on the engine's k_gemm_lds, an f32 FMA kernel (no matrix instructions).
# The bench times, on the GLM-5.3-Flash trunk shapes, the engine's OWN path (Backend::gemv_batch at --gemm-lds 1) against rocBLAS f16 (GEMM alone, and with the Q8_0 -> f16 dequantise and
# the activation convert it needs per call) and the WMMA kernels, with each candidate's error against the engine's result; then a projected trunk GEMM time per token over the whole model.
# Plan stop rule (PF2): the candidate must reach 1.5x the engine's TFLOPs on these shapes, else PF2 stops.
# Launch through run_chain.sh (the service is stopped, the reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_gemm_micro_chain.sh > ~/bench/glm_gemm_micro_chain.log 2>&1 < /dev/null &
# Env: GM_BIN (default ~/src/franken-engine/franken/decode/bench_gemm_wmma, `make -C franken/decode bench-gemm`), GM_ARGS (bench arguments: --T N, --iters N, --only SHAPE), GM_TIMEOUT (s, default 900),
# GM_OUT (default ~/bench/franken/glm5/gemm_micro), GM_ENV (extra docker -e options).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${GM_BIN:-$HOME/src/franken-engine/franken/decode/bench_gemm_wmma}
O=${GM_OUT:-$HOME/bench/franken/glm5/gemm_micro}; mkdir -p "$O"; LOG="$O/bench_$(date +%H%M%S).log"
say_end() { echo "=== glm_gemm_micro exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_gemm_micro start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker rm -f gemm_micro >/dev/null 2>&1
( sleep ${GM_TIMEOUT:-900}; docker kill gemm_micro >/dev/null 2>&1 && echo "TIMEOUT: bench killed after ${GM_TIMEOUT:-900} s (a hung kernel?)" ) &
WD=$!
docker run --name gemm_micro --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 ${GM_ENV:-} \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" ${GM_ARGS:-} > "$LOG" 2>&1
rc=$?
kill $WD 2>/dev/null
echo "bench rc=$rc  log=$LOG"
grep -aE "HIP error|Memory access|out of memory|Segmentation|Aborted|rocBLAS error" "$LOG" | head -3 | cut -c1-200
cut -c1-240 "$LOG"
[ "$rc" -ne 0 ] && { echo "--- bench log tail:"; tail -n 8 "$LOG" | cut -c1-200; }
say_end $rc
