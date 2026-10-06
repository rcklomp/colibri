#!/bin/bash
# gateway_watchdog.sh — bring the owner's gateway back if it is down for the wrong reason.
#
# The gateway is the daily service and every chain restarts it on every exit path — but a
# chain killed mid-flight (a dying ssh session, an OOM, a reboot) leaves it down until someone
# notices. This notices. It is deliberately dumb and lock-aware:
#
#   * a chain holding the rig lock is maintenance -> do nothing, whatever the gateway's state
#   * the reservation flag ~/bench/.dev_reserved exists -> do nothing (no service to keep up)
#   * gateway answering /v1/models -> do nothing
#   * gateway process alive but not answering yet -> wait, it warms for ~90 s after a restart
#   * otherwise -> restore the SERVICE ENGINE (service_lib.sh), and log why
#
# WHICH engine it restores (2026-10-06): the one named in ~/bench/service_engine, default franken-glm -- the Franken
# engine through `serve_alt.sh franken-glm`, with no silent fallback to the old Colibri engine (a failed start leaves
# :8081 empty and says so in this log); `colibri-glm` = the old engine. Details and the "engine alive" rule: service_lib.sh.
# A Franken restore is capped at 3 attempts in 3 hours (a cold start loads 144 GB); then it logs "needs a person" and stops
# trying (re-arm: rm ~/bench/.watchdog_attempts).
#
# Run from cron every 5 minutes. The crontab runs ~/bench/gateway_watchdog.sh, a SYMLINK to this file in the repo checkout
# (resolved below, so service_lib.sh and rig_lock.sh are found next to the real file). It never touches a binary and never
# runs an engine itself; engines are started by serve_alt.sh / start_glm53.sh.
#
# Test hooks (cron sets none of them): WATCHDOG_LOG, WATCHDOG_GRACE (seconds a server may outlive its engine, default 180),
# plus service_lib.sh's SERVICE_* hooks. Regression test: test_watchdog.sh.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); . "$HERE/rig_lock.sh"; . "$HERE/service_lib.sh"
LOG="${WATCHDOG_LOG:-$HOME/bench/gateway_watchdog.log}"
GRACE="${WATCHDOG_GRACE:-180}"
ATTEMPTS="$HOME/bench/.watchdog_attempts"      # one epoch second per Franken restore attempt
say() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" >> "$LOG"; }
svc_say() { say "$@"; }

if rig_lock_maintenance; then exit 0; fi                      # a chain owns the box
if [ -e "$HOME/bench/.dev_reserved" ]; then exit 0; fi   # the owner reserved the rig for development: no service to keep up (2026-09-25)

KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
code=$(service_models_code)

# /v1/models is NOT a liveness check. On 2026-09-16 a chain's exit trap killed the gateway's
# engine child (pkill -9 -x glm53) while openai_server.py stayed up: /v1/models kept answering
# 200 for 28 minutes, every chat request failed with "colibri engine dispatcher stopped", and
# this watchdog saw nothing. The server never respawns its engine, so "server alive, engine
# gone" is a dead gateway whatever the HTTP code says. The grace (180 s) covers a fresh restart
# that has not spawned its engine yet.
srv=$(pgrep -f "openai_[s]erver.py" | head -1)
if [ -n "$srv" ] && ! engine_alive; then
  age=$(ps -o etimes= -p "$srv" 2>/dev/null | tr -d ' ')
  if [ "${age:-0}" -gt "$GRACE" ]; then
    say "gateway server alive (pid $srv, ${age}s) but its engine (glm53 / franken_dec_glm) is GONE (/v1/models=$code) — restarting"
    pkill -f "openai_[s]erver.py"; sleep 3
    code=000
    srv=""
  fi
fi

[ "$code" = 200 ] && exit 0

if [ -n "$srv" ] && pgrep -f "openai_[s]erver.py" >/dev/null; then
  # up but not answering: either warming after a restart, or busy with a long prefill, both
  # of which are normal here. Only complain if it stays that way.
  say "gateway process alive, /v1/models=$code — leaving it alone"
  exit 0
fi

service_engine_select
if [ "$SERVICE_ENGINE_NAME" = franken-glm ]; then
  now=$(date +%s)
  recent=$(awk -v n="$now" '$1 > n - 10800' "$ATTEMPTS" 2>/dev/null | wc -l | tr -d ' ')
  if [ "${recent:-0}" -ge 3 ]; then
    say "Franken restore: ${recent} attempts in the last 3 h and the gateway is still down — NOT trying again, needs a person (last log: $SERVICE_FRANKEN_LOG; re-arm with: rm $ATTEMPTS)"
    exit 1
  fi
  { awk -v n="$now" '$1 > n - 10800' "$ATTEMPTS" 2>/dev/null; echo "$now"; } > "$ATTEMPTS.new" && mv -f "$ATTEMPTS.new" "$ATTEMPTS"
  say "gateway DOWN (/v1/models=$code, no process, no chain holding the lock) — attempt $((recent+1)) of 3 in 3 h"
else
  say "gateway DOWN (/v1/models=$code, no process, no chain holding the lock)"
fi
if service_restore; then rm -f "$ATTEMPTS"; exit 0; fi
exit 1
