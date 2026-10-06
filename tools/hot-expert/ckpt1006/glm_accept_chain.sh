#!/bin/bash
# glm_accept_chain.sh -- bring the CANDIDATE serving build up behind the real gateway and Open WebUI, hold it while accept_ui.sh (from the Mac; it also runs accept_live.sh over ssh) judges it,
# then stop everything and put the reservation back. Launch ONLY through run_chain.sh (it holds the rig lock, so gateway_watchdog.sh cannot start the Colibri gateway while the flag is aside).
#   BIN (default franken_dec_glm.hgblk) with FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256.
# NOT through run_chain.sh: accept_ui.sh refuses while any process named run_chain.sh is alive (it takes that for a measurement). The chain takes the rig lock itself, the way run_chain.sh does.
# Launch:  setsid nohup ~/bench/glm_accept_chain.sh > ~/bench/glm_accept_chain.log 2>&1 < /dev/null &
# Protocol: the line "=== READY_FOR_UI" in the chain log says the gateway is up and the flag is moved aside; the caller touches ~/bench/.accept_ui_done when it has finished (45 min cap).
set -u
BIN=${BIN:-$HOME/bench/franken_bin/franken_dec_glm}      # the INSTALLED served binary; the shipped defaults of start_franken_glm.sh are NOT overridden below
OUT=$HOME/bench/glm_accept; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert
KEY=$(cat "$HOME/.colibri_api_key")
START=$HE/franken/start_franken_glm.sh
F=$HOME/bench/.dev_reserved; MARK=$HOME/bench/.accept_ui_done
. "$HE/rig_lock.sh"
rig_lock_take glm_accept_chain || exit 3
. "$HOME/bench/chain_preflight.sh"     # rig_quiet_wait / rig_stop_serving: ONE copy of the waits (VRAM, container, compile), see that file
quiet_wait() { rig_quiet_wait 900; }
stop_all() { rig_stop_serving; }
restore() { [ -e "$F.accept-off" ] && mv "$F.accept-off" "$F"; echo "=== reservation flag: $(ls -la "$F" 2>&1 | cut -c1-70)"; }
trap 'restore; stop_all; rig_lock_release; echo "=== accept chain exit $(date -Is): flag restored, gateway and engine stopped, lock released (rig reserved)"' EXIT INT TERM
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
echo "=== glm_accept start $(date -Is) bin=$(basename $BIN) $(sha256sum "$BIN" | cut -c1-16)"
stop_all; quiet_wait || exit 2
log="$OUT/gw_$(date +%H%M%S).log"
env FRANKEN_DOCKER_BIN=$BIN FRANKEN_LOG=$log SKIP_WARM=1 setsid nohup "$START" > "$log" 2>&1 < /dev/null &
up=0
for i in $(seq 1 180); do
  sleep 5
  if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "franken_dec_[g]lm" > /dev/null; then up=1; break; fi
done
[ $up = 1 ] || { echo "FAIL: the candidate did not come up"; tail -8 "$log" | cut -c1-200; exit 1; }
echo "up after $((i*5)) s: $(grep -a '^\[start\]' "$log" | head -1 | cut -c1-200)"
mv "$F" "$F.accept-off" || { echo "no reservation flag to move aside"; exit 2; }
rm -f "$MARK"
# accept_ui.sh's own rig-side step asks `serve_alt.sh status` for the gateway log; this gateway is not started by serve_alt, so that step reads the stale ~/glm53_server.log and its verdict is
# meaningless (first run, 2026-10-06: reused=4548 from the old Colibri log while the engine reused 4096). So accept_live runs HERE with the right log, first: it also captures the tool block,
# so the browser's chat A afterwards is warm. Ignore the rig-side part of accept_ui's output.
echo "=== accept_live (GLM53_LOG=$log) $(date -Is)"
GLM53_LOG=$log "$HE/accept_live.sh" 2>&1 | cut -c1-250
echo "=== accept_live rc=${PIPESTATUS[0]} $(date -Is)"
echo "=== READY_FOR_UI $(date -Is) log=$log"
for i in $(seq 1 240); do [ -e "$MARK" ] && break; sleep 5; done
echo "=== hold ended: marker=$([ -e "$MARK" ] && echo yes || echo TIMEOUT) $(date -Is)"
grep -aE "serve-glm5\] req=|CKPT" "$log" | tail -12 | cut -c1-260
