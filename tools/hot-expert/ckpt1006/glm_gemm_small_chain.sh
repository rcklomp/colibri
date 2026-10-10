#!/bin/bash
# glm_gemm_small_chain.sh -- PF2c: correctness + speed of the small-row trunk GEMMs of a prefill chunk, per arm (k_gemm_batch / k_gemm_lds / k_gemm_lds_sk), against a float64 host reference
# (franken-engine branch pf2c-minrows, franken/decode/bench_gemm_small.hip, `make bench-gemm-small`). ONE card, no model. Launch through run_chain.sh.
# Env: GS_BIN, GS_ARGS (--iters N), GS_TIMEOUT (900 s), GS_OUT (~/bench/franken/glm5/gemm_small).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${GS_BIN:-$HOME/src/franken-engine/franken/decode/bench_gemm_small}
O=${GS_OUT:-$HOME/bench/franken/glm5/gemm_small}; mkdir -p "$O"; LOG="$O/bench_$(date +%H%M%S).log"
say_end() { echo "=== glm_gemm_small exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_gemm_small start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker rm -f gemm_small >/dev/null 2>&1
( sleep ${GS_TIMEOUT:-900}; docker kill gemm_small >/dev/null 2>&1 && echo "TIMEOUT: bench killed after ${GS_TIMEOUT:-900} s" ) &
WD=$!
docker run --name gemm_small --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" ${GS_ARGS:-} > "$LOG" 2>&1
rc=$?
kill $WD 2>/dev/null
echo "bench rc=$rc  log=$LOG"
grep -aE "HIP error|Memory access|out of memory|Segmentation|Aborted" "$LOG" | head -3 | cut -c1-200
cut -c1-200 "$LOG"
say_end "$rc"
