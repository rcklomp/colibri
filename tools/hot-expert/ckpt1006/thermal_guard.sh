#!/bin/bash
# thermal_guard.sh -- software thermal cap for a long rig job (written 2026-10-10 after cards 1 and 2 sat at 104-109 C junction, limit 110, under the PF2 quality run; the 61-minute soak had already reached 112 C).
# Not a fix: the cause is hardware (fan tach 0 rpm on cards 0 and 2, card 2 at pwm 255 and still 0 rpm). This only keeps the job below the driver limits until the owner has looked at the cooling.
# It FREEZES the engine container (`docker pause`: the GPUs go idle, the model, the KV cache and the open requests stay) when the hottest junction or memory temperature is within a margin of its critical
# value, and thaws it once every card is well below. The quality client waits up to 3 h on a request (quality_eval.py send_chat timeout=10800), so a pause costs time, never a result.
# Reads each card's own limits (temp2_crit junction 110, temp3_crit memory 108), pauses at crit-PAUSE_MARGIN (default 12: 98 / 96 C), thaws at crit-RESUME_MARGIN (default 25: 85 / 83 C), polls every 1 s
# (the first run, 10 C and 2 s, overshot to 104-106 C junction; a pause takes effect in well under a second, the die heats about 1 C a second).
# It leaves when the job it guards is gone (GUARD_WATCH = a pgrep -f pattern, default the PF2 quality chain; empty = never) and ALWAYS thaws the container on the way out (TERM / INT / EXIT).
# Launch (rig): ssh -n -f rome 'GUARD_WATCH="glm_f16_quality_chai[n]" setsid nohup ~/bench/thermal_guard.sh > ~/bench/thermal_guard.log 2>&1 < /dev/null &'
# Env: GUARD_FILTERS (docker ps filters of the containers to freeze, space separated, default "name=franken_engine ancestor=rocm/dev-ubuntu-24.04:7.14.0-full": the serving engine AND the gate / gate-plan
# containers the chains start with `docker run`), PAUSE_MARGIN, RESUME_MARGIN, GUARD_WATCH, GUARD_POLL (1).
set -u
FILTERS=${GUARD_FILTERS:-"name=franken_engine ancestor=rocm/dev-ubuntu-24.04:7.14.0-full"}; PM=${PAUSE_MARGIN:-12}; RM=${RESUME_MARGIN:-25}; W=${GUARD_WATCH-glm_f16_quality_chai[n]}; POLL=${GUARD_POLL:-1}
log() { echo "$(date -u +%FT%TZ) $*"; }
ids() { local f; for f in $FILTERS; do docker ps -q --filter "$f" --filter "status=$1" 2>/dev/null; done | sort -u; }
paused() { [ -n "$(ids paused)" ]; }
thaw() { local i; for i in $(ids paused); do docker unpause "$i" >/dev/null 2>&1 && log "THAW $i"; done; }
trap 'thaw; log "guard exit"; exit 0' TERM INT
trap 'thaw' EXIT
# worst margin over all cards: how many degrees the hottest sensor is below its critical value (junction temp2, memory temp3)
margin() {
  local h t c m=999
  for h in /sys/class/drm/card*/device/hwmon/hwmon*; do
    for s in 2 3; do
      t=$(cat "$h/temp${s}_input" 2>/dev/null) || continue; c=$(cat "$h/temp${s}_crit" 2>/dev/null) || continue
      d=$(( (c - t) / 1000 )); [ "$d" -lt "$m" ] && m=$d
    done
  done
  echo "$m"
}
log "guard start filters=[$FILTERS] pause_at=crit-$PM thaw_at=crit-$RM watch='$W'"
gone=0; n=0
while :; do
  m=$(margin)
  if paused; then
    [ "$m" -ge "$RM" ] && thaw
  elif [ "$m" -le "$PM" ]; then
    for i in $(ids running); do docker pause "$i" >/dev/null 2>&1 && log "PAUSE $i (hottest sensor $m C below its critical value)"; done
  fi
  n=$((n + 1)); [ $((n % 300)) -eq 0 ] && log "alive margin=$m paused=$(paused && echo yes || echo no)"
  if [ -n "$W" ]; then
    if pgrep -f "$W" >/dev/null; then gone=0; else gone=$((gone + 1)); [ "$gone" -ge 5 ] && { log "watched job gone"; break; }; fi
  fi
  sleep "$POLL"
done
thaw
log "guard exit"
