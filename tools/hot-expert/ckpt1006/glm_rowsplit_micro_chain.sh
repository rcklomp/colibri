#!/bin/bash
# glm_rowsplit_micro_chain.sh -- the --gemv-rowsplit MICROBENCHMARK on ONE card, no model load (2026-10-07; franken-engine branch gemv-rowsplit, GLM5.md section 20).
# Is k_gemv_rowsplit (one workgroup a row, the waves loop over the row's K splits with k_gemm_batch's own per-split code, partials summed in LDS in s order) BIT-IDENTICAL
# to the k_gemm_batch + k_reduce_splits_gemm_w pair it replaces, and how many us does it take per launch for the six split-GEMV signatures a GLM decode token runs 278 times
# (hc_*_fn 24x16384/512 splits, f_a-class 128x4096/128, KDA beta 64x4096/128, indexer proj F32 32x4096/128, kv_a 512x4096/32, router F32 288x4096/57)?
# bench_rowsplit (franken/decode/bench_rowsplit.hip) #includes the real decode_gpu.hip and links main's own backend as the reference: every case prints EXACT or MISMATCH
# (memcmp against main at GLM's shipped reduce settings; the backend dispatch at rowsplit 0/1 and waves auto/4/8/16; the direct timing launches; a T=13 chunk), then the
# timing table (hot / cold / dependent-chain us per launch sequence) and the estimated decode saving. Exit 1 on any mismatch. Allocates ~0.3 GB on card 0 for the cold copies.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <bench log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_rowsplit_micro_chain.sh > ~/bench/glm_rowsplit_micro_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_rowsplit_micro_chain.log ~/bench/franken/glm5/rowsplit_micro/bench.log
# Env: RSM_ENV (extra docker -e options, e.g. "-e AMD_LOG_LEVEL=3"), RSM_ARGS (bench arguments, e.g. --exact-only), RSM_TIMEOUT (s, default 900; the bench container is killed after it), RSM_BIN (default ~/bench/franken_bin/bench_rowsplit, `make -C franken/decode bench-rowsplit`), RSM_OUT (default ~/bench/franken/glm5/rowsplit_micro).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${RSM_BIN:-$HOME/bench/franken_bin/bench_rowsplit}
O=${RSM_OUT:-$HOME/bench/franken/glm5/rowsplit_micro}; rm -rf "$O"; mkdir -p "$O"
say_end() { echo "=== glm_rowsplit_micro exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_rowsplit_micro start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker rm -f rowsplit_micro >/dev/null 2>&1
( sleep ${RSM_TIMEOUT:-900}; docker kill rowsplit_micro >/dev/null 2>&1 && echo "TIMEOUT: bench killed after ${RSM_TIMEOUT:-900} s (a hung kernel?)" ) &
WD=$!
docker run --name rowsplit_micro --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 ${RSM_ENV:-} \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" ${RSM_ARGS:-} > "$O/bench.log" 2>&1
rc=$?
kill $WD 2>/dev/null
echo "bench rc=$rc"
grep -aE "HIP error|Memory access|out of memory|Segmentation|Aborted" "$O/bench.log" | head -3 | cut -c1-200
echo "=== exactness: $(grep -ac ' EXACT ' "$O/bench.log") EXACT lines, $(grep -ac 'MISMATCH' "$O/bench.log") MISMATCH lines"
grep -a 'MISMATCH' "$O/bench.log" | head -20 | cut -c1-200
grep -aE "^[a-z_0-9]+: [0-9]+ matrices" "$O/bench.log" | cut -c1-200
sed -n '/^=== TIMING/,$p' "$O/bench.log" | cut -c1-200
grep -a "^  timing " "$O/bench.log" | cut -c1-200
[ "$rc" -ne 0 ] && { echo "--- bench log tail:"; tail -n 8 "$O/bench.log" | cut -c1-200; }
say_end $rc
