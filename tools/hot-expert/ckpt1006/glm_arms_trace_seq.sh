#!/bin/bash
# glm_arms_trace_seq.sh -- one rocprofv3 decode-token kernel trace per ARM, arms in a given order, each its own process (2026-10-08, plan DECODE-OPEN-ITEMS D1 / D3; generalises
# glm_group_trace_seq.sh, which varies one flag 0 1 1 0). Wall-clock medians cannot resolve ~0.1-1 ms; the kernel trace repeats to the microsecond (record §L5-GLM-ROWSPLIT).
# Each arm = NAME=extra flags for the plan line, e.g.   TS_ARMS="b=|l=--gemv-lds 0|r=--gemv-rowsplit-waves 1|r=--gemv-rowsplit-waves 1|l=--gemv-lds 0|b="   (arms split on '|', name before the first '=')
# The shipped flags are in every arm (TS_SHIPPED): --fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1 (adaptation on, the trace chain's default plan).
# Output dir of arm i: ~/bench/franken/glm5/dtrace_<name>_<i>  (the name+index scheme rowsplit_trace_paired.py wants: `python3 -I rowsplit_trace_paired.py b_1 l_2 l_5 b_6` = base / treatment / treatment / base).
# Launch (on the rig; it calls run_chain.sh itself, one lock per arm; glm_decode_trace_chain.sh must be in ~/bench):
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/bench/glm_arms_trace_seq.sh > ~/bench/glm_arms_trace_seq.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_arms_trace_seq.log      then:  python3 -I ckpt1006/decode_trace_report.py ~/bench/franken/glm5/dtrace_b_1 --tokens 20   (per arm)
# Env: TS_BIN (default ~/bench/franken_bin/franken_decode_glm_d3), TS_TOKENS (decode tokens traced, default 24), TS_ARMS (required), TS_SHIPPED, TS_TAG (appended to every out dir name, default empty).
# An arm that fails (rc != 0, incl. a refusal rc 3) is retried once after the rig has been quiet; a second failure ends the sequence and says so (no sequence waits only for success).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${TS_BIN:-$HOME/bench/franken_bin/franken_decode_glm_d3}
N=${TS_TOKENS:-24}
SHIPPED=${TS_SHIPPED:---fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1}
[ -n "${TS_ARMS:-}" ] || { echo "FATAL: TS_ARMS is empty"; echo "=== glm_arms_trace_seq exit rc=2"; exit 2; }
echo "=== glm_arms_trace_seq start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) arms=$TS_ARMS"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_arms_trace_seq exit rc=2 $(date -Is)"; exit 2; }
i=0
IFS='|' read -r -a ARMS <<< "$TS_ARMS"
for spec in "${ARMS[@]}"; do
  i=$((i+1))
  name=${spec%%=*}; extra=${spec#*=}
  out=$HOME/bench/franken/glm5/dtrace_${name}${TS_TAG:-}_$i
  for try in 1 2; do
    rig_quiet_wait 600 || { echo "=== glm_arms_trace_seq exit rc=3 (rig not quiet before arm $i)"; exit 3; }
    echo "--- arm $i ($name) try $try: [$SHIPPED $extra] -> $out $(date -Is)"
    TRACE_BIN="$BIN" TRACE_OUT="$out" TRACE_TOKENS=$N TRACE_EXTRA="$SHIPPED $extra" \
      "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_decode_trace_chain.sh" > "$HOME/bench/glm_arms_trace_arm$i.log" 2>&1
    rc=$?
    echo "--- arm $i done rc=$rc $(date -Is): $(grep -ac . "$out"/out/*/*kernel_trace.csv 2>/dev/null | tail -1) kernel rows; flags as run: $(grep -a -m1 'gemv_fused_reduce=' "$out/gate_run.log" 2>/dev/null | cut -c1-200)"
    [ $rc -eq 0 ] && break
    [ $try -eq 2 ] && { echo "=== glm_arms_trace_seq exit rc=$rc (arm $i failed twice; see ~/bench/glm_arms_trace_arm$i.log)"; exit $rc; }
    echo "--- arm $i failed rc=$rc, retrying once"; tail -n 5 "$HOME/bench/glm_arms_trace_arm$i.log" | cut -c1-200
  done
done
echo "=== glm_arms_trace_seq exit rc=0 $(date -Is)"
