#!/bin/bash
# glm_cores_probe_chain.sh -- how many CPU cores does the SERVING engine leave free during a real decode? (2026-10-07; record section L5-GLM-CPULANE: a CPU lane needs 5-6 free physical cores)
# One serving process (the installed franken_dec_glm through start_franken_glm.sh, shipped defaults), cpu_probe.py samples every hardware thread and every thread of the engine once a second:
# ~12 s idle (engine loaded, nothing running), then serve_curve_driver.py sends N short fresh greedy chats (default 3 x 300 tokens, ~60 s of decode), then the probe ends. The summary
# (cpu_probe_summary.py) prints, for the idle window and the decode window: busy % of each hardware thread and of each physical core, the physical cores below 10 % busy, the engine's threads
# (busy %, allowed CPUs) and the busiest other processes. The engine is stopped at the end. Launch through run_chain.sh (the reservation flag stays in place):
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_cores_probe_chain.sh > ~/bench/glm_cores_probe_chain.log 2>&1 < /dev/null &
# Env: CP_BIN (default the installed binary), CP_N (chats, default 3), CP_OUT (default ~/bench/franken/glm5/cores_probe).
set -u
. "$HOME/bench/chain_preflight.sh"
HE=$HOME/src/colibri/tools/hot-expert; START=$HE/franken/start_franken_glm.sh
BIN=${CP_BIN:-$HOME/bench/franken_bin/franken_dec_glm}; N=${CP_N:-3}
O=${CP_OUT:-$HOME/bench/franken/glm5/cores_probe}; rm -rf "$O" 2>/dev/null; mkdir -p "$O"
KEY=$(cat "$HOME/.colibri_api_key")
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_cores_probe exit rc=2 $(date -Is)"; exit 2; }
[ -e "$HOME/bench/.dev_reserved" ] || { echo "FATAL: the reservation flag is gone"; echo "=== glm_cores_probe exit rc=2 $(date -Is)"; exit 2; }
echo "=== glm_cores_probe start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) chats=$N"
trap 'rig_stop_serving; echo "=== glm_cores_probe chain end $(date -Is): everything stopped"' EXIT
rig_stop_serving; rig_quiet_wait 900 || { echo "rig not quiet"; echo "=== glm_cores_probe exit rc=3 $(date -Is)"; exit 3; }
log="$O/engine.gw.log"
env FRANKEN_DOCKER_BIN=$BIN FRANKEN_LOG=$log SKIP_WARM=1 setsid nohup "$START" > "$log" 2>&1 < /dev/null &
up=0
for i in $(seq 1 240); do
  sleep 5
  if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "franken_dec_[g]lm" > /dev/null; then up=1; break; fi
done
[ $up = 1 ] || { echo "the engine did not come up"; tail -6 "$log" | cut -c1-200; echo "=== glm_cores_probe exit rc=1 $(date -Is)"; exit 1; }
echo "engine up after $((i*5)) s; settling 20 s"; sleep 20
python3 -I "$HOME/bench/cpu_probe.py" $((12 + 25 * N + 14)) "$O/probe.txt" franken_dec_glm &
PROBE=$!
sleep 12
T0=$(date +%s.%N)
python3 -I "$HOME/bench/serve_curve_driver.py" "$log" "$O/chats.json" "$N" 2>&1 | tee "$O/chats.txt"
T1=$(date +%s.%N)
wait $PROBE
echo "=== window: decode from $T0 to $T1"
python3 -I "$HOME/bench/cpu_probe_summary.py" "$O/probe.txt" "$T0" "$T1" 2>&1 | tee "$O/summary.txt"
echo "engine threads as the host sees them (names, once): $(ps -eLo comm 2>/dev/null | sort | uniq -c | sort -rn | head -5 | tr '\n' ';')"
echo "=== glm_cores_probe exit rc=0 $(date -Is)"
