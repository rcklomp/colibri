#!/bin/bash
# glm_accept_chain.sh -- bring the CANDIDATE serving build up behind the real gateway and Open WebUI, hold it while accept_ui.sh (from the Mac; it also runs accept_live.sh over ssh) judges it,
# then stop everything and put the reservation back. Launch ONLY through run_chain.sh (it holds the rig lock, so gateway_watchdog.sh cannot start the Colibri gateway while the flag is aside).
#   BIN (default franken_dec_glm.hgblk) with FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256.
# Protocol: the line "=== READY_FOR_UI" in the chain log says the gateway is up and the flag is moved aside; the caller touches ~/bench/.accept_ui_done when it has finished (45 min cap).
set -u
BIN=${BIN:-$HOME/bench/franken_bin/franken_dec_glm.hgblk}
OUT=$HOME/bench/glm_accept; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert
KEY=$(cat "$HOME/.colibri_api_key")
START=$HE/franken/start_franken_glm.sh
F=$HOME/bench/.dev_reserved; MARK=$HOME/bench/.accept_ui_done
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
quiet_wait() { for _ in $(seq 1 180); do [ "$(vram_max)" -lt 1024 ] && [ -z "$(docker ps -aq -f name='^franken_engine$' 2>/dev/null)" ] && return 0; sleep 5; done; echo "rig not quiet: vram $(vram_max) MiB"; return 1; }
stop_all() {
  pkill -f "openai_[s]erver.py" 2>/dev/null; sleep 2; pkill -9 -f "openai_[s]erver.py" 2>/dev/null
  pkill -9 -f "franken_dec_[g]lm" 2>/dev/null
  docker stop -t 5 franken_engine >/dev/null 2>&1 || true
  for _ in $(seq 1 120); do [ -z "$(docker ps -aq -f name='^franken_engine$' 2>/dev/null)" ] && break; sleep 5; done
}
restore() { [ -e "$F.accept-off" ] && mv "$F.accept-off" "$F"; echo "=== reservation flag: $(ls -la "$F" 2>&1 | cut -c1-70)"; }
trap 'restore; stop_all; echo "=== accept chain exit $(date -Is): flag restored, gateway and engine stopped (rig reserved)"' EXIT INT TERM
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
echo "=== glm_accept start $(date -Is) bin=$(basename $BIN) $(sha256sum "$BIN" | cut -c1-16)"
stop_all; quiet_wait || exit 2
log="$OUT/gw_$(date +%H%M%S).log"
env FRANKEN_GLM_CHUNK=1024 FRANKEN_SNAP_EVERY=1024 FRANKEN_GLM_STAGE_MB=256 FRANKEN_DOCKER_BIN=$BIN FRANKEN_LOG=$log SKIP_WARM=1 setsid nohup "$START" > "$log" 2>&1 < /dev/null &
up=0
for i in $(seq 1 180); do
  sleep 5
  if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $KEY" -w '%{http_code}' http://127.0.0.1:8081/v1/models)" = 200 ] && pgrep -f "franken_dec_[g]lm" > /dev/null; then up=1; break; fi
done
[ $up = 1 ] || { echo "FAIL: the candidate did not come up"; tail -8 "$log" | cut -c1-200; exit 1; }
echo "up after $((i*5)) s: $(grep -a '^\[start\]' "$log" | head -1 | cut -c1-200)"
mv "$F" "$F.accept-off" || { echo "no reservation flag to move aside"; exit 2; }
rm -f "$MARK"
echo "=== READY_FOR_UI $(date -Is) log=$log"
for i in $(seq 1 540); do [ -e "$MARK" ] && break; sleep 5; done
echo "=== hold ended: marker=$([ -e "$MARK" ] && echo yes || echo TIMEOUT) $(date -Is)"
grep -aE "serve-glm5\] req=|CKPT" "$log" | tail -12 | cut -c1-260
