#!/bin/bash
# glm_serve_ab_chain.sh -- the SERVE path, old build against the monotonic-position build (2026-10-07; record section L5-GLM-ADAPT-MONO).
# Why: the acceptance run of the cap build logged exactly ONE adapter window (`adapt_all pos=4573`, the first request) and none after it, through a 349-token answer and the
#  browser chats: the adapter's cadence ran on the runner's sequence position, which jumps back for every new chat, so a chat that stays below the previous one's end took no snapshot.
#  franken-engine branch adapt-mono (5b47ef9) runs the cadence on a monotonic position. The CLI gate cannot show it (its pos never rewinds): this chain drives the real gateway.
# Each arm: stop everything, start the serving build through start_franken_glm.sh (the shipped defaults: G=8, cap 64, half-life 512), wait for :8081, run serve_ab_driver.py (W1 a long
#  document, then four short fresh greedy chats, the last a repeat of the second), stop everything. Arms, in order: OLD (the installed cap build), NEW (adapt-mono), OLD, NEW.
#  Greedy below 2 051 tokens of depth is deterministic, so W2-W5 must emit the SAME text in both builds (the emitted counts must match: placement moves no bit) and only the speed may differ.
# Launch through run_chain.sh (the reservation flag stays in place; the chain never moves it), watch with a tail of $SAB_OUT/*.res.txt and the chain log.
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_serve_ab_chain.sh > ~/bench/glm_serve_ab_chain.log 2>&1 < /dev/null &
# 2026-10-10 (PF2): SAB_OLD_GL / SAB_NEW_GL set FRANKEN_GEMM_LDS for the old / new arm (the same binary can be run in two GEMM modes: SAB_OLD=SAB_NEW=franken_dec_glm.f16 SAB_OLD_GL=1 SAB_NEW_GL=3).
# Env: SAB_OLD / SAB_NEW (binaries), SAB_ARMS (default "old new old new"), SAB_OUT (default ~/bench/franken/glm5/serve_ab), SAB_START / SAB_PGREP (another model's start script and engine process pattern).
# Clock sampling (D6, 2026-10-08; why: 2 of 16 arms moved 5-7 % with no known cause, records L5-GLM-ROWSPLIT / L5-GLM-GEMV-GROUP): SAB_SAMPLE=1 (default; 0 = off, nothing else changes) runs
#  `gpu_sampler.py arm` beside every arm, from the arm's start (stop, wait, load, requests) to the end of the driver: every SAB_SAMPLE_PERIOD seconds (default 5) it appends one row per card to
#  $SAB_OUT/<arm>.gpu.tsv (PCI-address labelled sclk, mclk, hotspot/edge/mem temperature, power, throttle bits) and one row to <arm>.cpu.tsv (effective MHz of the busy threads), read from sysfs only, nice 19,
#  each row tagged with the arm and with the request the driver is running (SAB_MARK). It is stopped by sampler_stop on every path out of an arm and by the EXIT trap, and exits by itself when this shell
#  is gone. At the end serve_ab_clock_report.py prints per request the decode tok/s beside the sampled quantities, and the Pearson r over all requests (also saved as $SAB_OUT/clock_report.txt).
#  SAB_SAMPLER / SAB_REPORT name the two scripts (default ~/bench/gpu_sampler.py, ~/bench/serve_ab_clock_report.py); copy them, serve_ab_driver.py and this chain to ~/bench first.
set -u
. "$HOME/bench/chain_preflight.sh"
HE=$HOME/src/colibri/tools/hot-expert; START=${SAB_START:-$HE/franken/start_franken_glm.sh}; PG=${SAB_PGREP:-franken_dec_[g]lm}   # DeepSeek: SAB_START=$HE/franken/start_franken_ds4.sh SAB_PGREP=franken_decode_[d]s4
OLD=${SAB_OLD:-$HOME/bench/franken_bin/franken_dec_glm.cap}; NEW=${SAB_NEW:-$HOME/bench/franken_bin/franken_dec_glm.adm}
O=${SAB_OUT:-$HOME/bench/franken/glm5/serve_ab}; rm -rf "$O"; mkdir -p "$O"
KEY=$(cat "$HOME/.colibri_api_key")
SAMPLE=${SAB_SAMPLE:-1}; SPERIOD=${SAB_SAMPLE_PERIOD:-5}; SAMPLER=${SAB_SAMPLER:-$HOME/bench/gpu_sampler.py}; REPORT=${SAB_REPORT:-$HOME/bench/serve_ab_clock_report.py}; SAMP=""
sampler_start() {   # $1 = arm tag; never fatal: a sampler that cannot start leaves the arm unsampled
  [ "$SAMPLE" = 1 ] || return 0
  python3 -I "$SAMPLER" arm "$O/$1" "$1" "$SPERIOD" "$O/$1.mark" > "$O/$1.sampler.log" 2>&1 < /dev/null &
  SAMP=$!; sleep 1
  kill -0 "$SAMP" 2>/dev/null || { echo "arm $1: WARNING the clock sampler did not start (see $O/$1.sampler.log); this arm runs unsampled"; wait "$SAMP" 2>/dev/null; SAMP=""; }
  if [ -n "$SAMP" ] && ! grep -q SAB_MARK "$HOME/bench/serve_ab_driver.py" 2>/dev/null; then echo "arm $1: WARNING ~/bench/serve_ab_driver.py is the old one (no request marker, no t_end): the report falls back to the engine log's times"; fi
}
sampler_stop() {    # idempotent; by PID only, never by pattern
  [ -n "$SAMP" ] || return 0
  kill "$SAMP" 2>/dev/null
  for _ in 1 2 3 4 5 6; do kill -0 "$SAMP" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$SAMP" 2>/dev/null && kill -9 "$SAMP" 2>/dev/null
  kill -0 "$SAMP" 2>/dev/null && echo "WARNING: clock sampler pid $SAMP did not exit" || wait "$SAMP" 2>/dev/null
  SAMP=""
}
[ -x "$OLD" ] && [ -x "$NEW" ] || { echo "FATAL: missing binary ($OLD / $NEW)"; echo "=== glm_serve_ab exit rc=2 $(date -Is)"; exit 2; }
[ -e "$HOME/bench/.dev_reserved" ] || { echo "FATAL: the reservation flag is gone: this chain starts an engine and must run under it"; echo "=== glm_serve_ab exit rc=2 $(date -Is)"; exit 2; }
echo "=== glm_serve_ab start $(date -Is) old=$(sha256sum "$OLD" | cut -c1-16) new=$(sha256sum "$NEW" | cut -c1-16)"
trap 'sampler_stop; rig_stop_serving; echo "=== glm_serve_ab chain end $(date -Is): everything stopped"' EXIT
n=0; rcs=0
for arm in ${SAB_ARMS:-old new old new}; do
  n=$((n+1)); tag="${n}_$arm"; bin=$OLD; gl=${SAB_OLD_GL:-}; [ "$arm" = new ] && { bin=$NEW; gl=${SAB_NEW_GL:-}; }
  echo "=== arm $tag bin=$(basename "$bin") $(date -Is)"
  sampler_start "$tag"
  rig_stop_serving; rig_quiet_wait 900 || { echo "arm $tag: rig not quiet"; rcs=1; sampler_stop; continue; }
  log="$O/$tag.gw.log"
  env FRANKEN_DOCKER_BIN=$bin FRANKEN_LOG=$log SKIP_WARM=1 ${gl:+FRANKEN_GEMM_LDS=$gl} setsid nohup "$START" > "$log" 2>&1 < /dev/null &
  up=0
  for i in $(seq 1 240); do
    sleep 5
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "$PG" > /dev/null; then up=1; break; fi
  done
  [ $up = 1 ] || { echo "arm $tag: the engine did not come up"; tail -6 "$log" | cut -c1-200; rcs=1; rig_stop_serving; sampler_stop; continue; }
  echo "arm $tag up after $((i*5)) s"
  mark=""; [ -n "$SAMP" ] && mark="$O/$tag.mark"
  SAB_MARK="$mark" python3 -I "$HOME/bench/serve_ab_driver.py" "$log" "$O/$tag.res.json" 2>&1 | tee "$O/$tag.res.txt"
  sampler_stop
  echo "arm $tag adapter windows in the whole log: $(grep -a -c 'adapt_all pos=' "$log")"
  rig_stop_serving
done
echo "=== SUMMARY (decode tok/s per request; W2-W5 are fresh short chats after the long W1)"
for f in "$O"/*.res.txt; do echo "--- $(basename "$f" .res.txt)"; cut -c1-150 "$f"; done
if [ "$SAMPLE" = 1 ]; then
  echo "=== CLOCK REPORT (per request: decode tok/s beside the GPU / CPU state sampled every ${SPERIOD} s inside its decode window; raw rows: $O/<arm>.gpu.tsv, <arm>.cpu.tsv; saved as $O/clock_report.txt)"
  python3 -I "$REPORT" "$O" 2>&1 | tee "$O/clock_report.txt"
fi
echo "=== glm_serve_ab exit rc=$rcs $(date -Is)"
exit $rcs
