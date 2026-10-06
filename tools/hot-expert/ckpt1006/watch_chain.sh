#!/bin/bash
# watch_chain.sh <chain-log> [engine-log] -- the watcher every rig chain gets (run on the Mac, as a Monitor command).
#
# Why it exists (2026-10-07): three decode-prefetch gate starts died at setup (VRAM budget twice). The ENGINE's own error
# line (`HIP error hipMalloc ... out of memory`, `--decode-prefetch needs 362 MB ...`) goes to gate_run.log, NOT to the
# chain log, so a watcher on the chain log alone saw only "16 of 16 FAIL" and the cause had to be dug out by hand. This
# one reads BOTH logs, reports every end state (success, failure, refusal, a dead engine, silence) and, on any bad end,
# prints the last lines of both logs so the cause arrives with the event.
#
#   watch_chain.sh ~/bench/glm5_dpf_chain.log ~/bench/franken/glm5/dpf/gate_run.log
#
# The engine log is optional (a chain that has none is watched on its chain log only). Exits when the chain prints
# its `exit rc=` marker (every chain's on_exit does), or after 12 h. Output: one line per interesting event.
RIG=${RIG:-rome}
CHAIN=${1:?chain log on the rig}
ENGINE=${2:-}
SILENT_POLLS=${SILENT_POLLS:-27}              # 27 polls x 45 s = 20 minutes without a new line in either log
PAT='=== gate-plan|=== glm5|=== dpf|=== run_chain|^[a-z0-9_]+ +(none|exact|cos|ref) +(ok|FAIL|PASS)|FAIL|PASS|rror|HIP error|[Aa]bort|core dumped|out of memory|OOM|needs [0-9]|REFUSED|Killed|Traceback|glm5_dpf |glm5_decode_ms_median|decode ms/token|covered_frac|vram_dev.*(caches|steady)|resident=|exit rc|active=0|decode_prefetch='
last_c=0; last_e=0; quiet=0; dead=0; bad=0
while :; do
  out=$(ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 "$RIG" '
    c='"$CHAIN"'; e='"${ENGINE:-/dev/null}"'
    nc=$(wc -l < $c 2>/dev/null || echo 0); ne=$(wc -l < $e 2>/dev/null || echo 0)
    echo "N $nc $ne"
    # a live engine = the GPU process (the binary under the docker image shows as the container "franken_engine" or the gate process)
    if pgrep -f "[g]lm5_gpu_gate.sh|[f]ranken_decode_glm|[f]ranken_dec_glm|[f]ranken_decode_ds4|[f]ranken_decode " >/dev/null; then echo "ALIVE 1"; else echo "ALIVE 0"; fi
    echo "--- chain"; tail -n +'$((last_c+1))' $c 2>/dev/null
    echo "--- engine"; [ "$e" != /dev/null ] && tail -n +'$((last_e+1))' $e 2>/dev/null
    echo "--- tails"; tail -n 8 $c 2>/dev/null | sed "s/^/CHAINTAIL /"; [ "$e" != /dev/null ] && tail -n 8 $e 2>/dev/null | sed "s/^/ENGINETAIL /"
  ' 2>&1) || { echo "ssh failed (retrying)"; sleep 30; continue; }
  nc=$(echo "$out" | sed -n 's/^N \([0-9]*\) .*/\1/p' | head -1); ne=$(echo "$out" | sed -n 's/^N [0-9]* \([0-9]*\)/\1/p' | head -1)
  alive=$(echo "$out" | sed -n 's/^ALIVE //p' | head -1)
  body=$(echo "$out" | sed -n '/^--- chain/,/^--- tails/p' | grep -v '^--- ')
  ev=$(echo "$body" | grep -E "$PAT" | cut -c1-260)
  [ -n "$ev" ] && echo "$ev"
  echo "$ev" | grep -qE "FAIL|rror|[Aa]bort|core dumped|out of memory|OOM|REFUSED|Killed|Traceback|active=0|needs [0-9]" && bad=1
  if [ "${nc:-0}" = "$last_c" ] && [ "${ne:-0}" = "$last_e" ]; then quiet=$((quiet+1)); else quiet=0; fi
  last_c=${nc:-$last_c}; last_e=${ne:-$last_e}
  if [ "$quiet" -ge "$SILENT_POLLS" ]; then echo "SILENT: no new line in either log for $((SILENT_POLLS*45/60)) min (engine alive=$alive)"; quiet=0; fi
  # a dead engine with a chain that has not exited = a silent death (a crashed engine, a refused launch)
  if [ "$alive" = 0 ] && ! echo "$body" | grep -q "exit rc"; then dead=$((dead+1)); else dead=0; fi
  if [ "$dead" -ge 4 ]; then echo "DEAD: no engine/gate process for 3 min and the chain has not exited"; bad=1; dead=0; fi
  if echo "$body" | grep -q "exit rc"; then
    [ "$bad" = 1 ] && { echo "--- last lines of both logs (a bad end):"; echo "$out" | grep -E "^(CHAINTAIL|ENGINETAIL) " | cut -c1-260; }
    echo "WATCH END (bad=$bad)"; exit 0
  fi
  sleep 45
done
