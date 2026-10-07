#!/bin/bash
# glm_rowsplit_trace_seq.sh -- does --gemv-rowsplit shorten the decode token's CRITICAL PATH? (2026-10-07; follows glm_rowsplit_ab_chain.sh, whose wall-clock medians could not resolve a ~2 ms effect:
# shipped-adaptation block B-A -0.20 ms with a +-2 ms interval, fixed-placement block -1.77 ms with an SE of 1.5; the microbenchmark said -2.28 ms for the 278 split launches.)
# One rocprofv3 kernel trace (glm_decode_trace_chain.sh: 2 048 prompt tokens, 24 settle, N decode tokens, shipped flags, adaptation on) per arm, in the order 0 1 1 0, each its own process:
# the kernel-level quantities (the split GEMVs' own time, the layer_a / tail segments of decode_trace_report.py, per-token kernel sums) have a noise floor far below the wall clock's.
# Launch: setsid nohup ~/bench/glm_rowsplit_trace_seq.sh > ~/bench/glm_rowsplit_trace_seq.log 2>&1 < /dev/null &   (it calls run_chain.sh itself: one lock per arm)
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_rowsplit_trace_seq.log      then:  python3 -I ckpt1006/decode_trace_report.py ~/bench/franken/glm5/dtrace_rs0 --tokens 20   (and rs1)
# Env: TS_BIN (default ~/bench/franken_bin/franken_decode_glm.rowsplit), TS_TOKENS (decode tokens traced, default 24), TS_ARMS (default "0 1 1 0").
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${TS_BIN:-$HOME/bench/franken_bin/franken_decode_glm.rowsplit}
N=${TS_TOKENS:-24}
echo "=== glm_rowsplit_trace_seq start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) arms=${TS_ARMS:-0 1 1 0}"
i=0
for arm in ${TS_ARMS:-0 1 1 0}; do
  i=$((i+1))
  out=$HOME/bench/franken/glm5/dtrace_rs${arm}_$i
  rig_quiet_wait 600 || { echo "=== glm_rowsplit_trace_seq exit rc=3 (rig not quiet before arm $i)"; exit 3; }
  echo "--- arm $i: --gemv-rowsplit $arm -> $out $(date -Is)"
  TRACE_BIN="$BIN" TRACE_OUT="$out" TRACE_TOKENS=$N TRACE_EXTRA="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit $arm" \
    "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_decode_trace_chain.sh" > "$HOME/bench/glm_rowsplit_trace_arm$i.log" 2>&1
  rc=$?
  echo "--- arm $i done rc=$rc $(date -Is): $(grep -ac . "$out/out"/*/*kernel_trace.csv 2>/dev/null | tail -1)"
  [ $rc -ne 0 ] && { echo "=== glm_rowsplit_trace_seq exit rc=$rc (arm $i failed; see ~/bench/glm_rowsplit_trace_arm$i.log)"; exit $rc; }
done
echo "=== glm_rowsplit_trace_seq exit rc=0 $(date -Is)"
