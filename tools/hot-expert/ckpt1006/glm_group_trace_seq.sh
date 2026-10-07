#!/bin/bash
# glm_group_trace_seq.sh -- does --glm-gemv-group shorten the decode token's CRITICAL PATH? (2026-10-08; follows glm_rowsplit_trace_seq.sh: wall-clock medians cannot resolve ~1 ms, the kernel trace can.)
# One rocprofv3 kernel trace (glm_decode_trace_chain.sh: 2 048 prompt tokens, 24 settle, N decode tokens, shipped flags, adaptation on) per arm, in the order 0 1 1 0 of --glm-gemv-group, each its own process:
# the group's own kernels (k_gemm_batch / k_gemv_rowsplit launches a layer, ~150 launches a token fewer), the layer_a / tail segments of decode_trace_report.py and the per-token kernel sums have a noise
# floor far below the wall clock's. Shipped flags in every arm: --fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1.
# Launch: setsid nohup ~/bench/glm_group_trace_seq.sh > ~/bench/glm_group_trace_seq.log 2>&1 < /dev/null &   (it calls run_chain.sh itself: one lock per arm; glm_decode_trace_chain.sh must be in ~/bench)
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_group_trace_seq.log      then:  python3 -I ckpt1006/decode_trace_report.py ~/bench/franken/glm5/dtrace_gr0_1 --tokens 20   (and gr1_2)
# Env: TS_BIN (default ~/bench/franken_bin/franken_decode_glm.group), TS_TOKENS (decode tokens traced, default 24), TS_ARMS (default "0 1 1 0").
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${TS_BIN:-$HOME/bench/franken_bin/franken_decode_glm.group}
N=${TS_TOKENS:-24}
echo "=== glm_group_trace_seq start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) arms=${TS_ARMS:-0 1 1 0}"
i=0
for arm in ${TS_ARMS:-0 1 1 0}; do
  i=$((i+1))
  out=$HOME/bench/franken/glm5/dtrace_gr${arm}_$i
  rig_quiet_wait 600 || { echo "=== glm_group_trace_seq exit rc=3 (rig not quiet before arm $i)"; exit 3; }
  echo "--- arm $i: --glm-gemv-group $arm -> $out $(date -Is)"
  TRACE_BIN="$BIN" TRACE_OUT="$out" TRACE_TOKENS=$N TRACE_EXTRA="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group $arm" \
    "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_decode_trace_chain.sh" > "$HOME/bench/glm_group_trace_arm$i.log" 2>&1
  rc=$?
  echo "--- arm $i done rc=$rc $(date -Is): $(grep -ac . "$out/out"/*/*kernel_trace.csv 2>/dev/null | tail -1)"
  [ $rc -ne 0 ] && { echo "=== glm_group_trace_seq exit rc=$rc (arm $i failed; see ~/bench/glm_group_trace_arm$i.log)"; exit $rc; }
done
echo "=== glm_group_trace_seq exit rc=0 $(date -Is)"
