#!/bin/bash
# gateway_watchdog.sh — bring the owner's gateway back if it is down for the wrong reason.
#
# The gateway is the daily service and every chain restarts it on every exit path — but a
# chain killed mid-flight (a dying ssh session, an OOM, a reboot) leaves it down until someone
# notices. This notices. It is deliberately dumb and lock-aware:
#
#   * a chain holding the rig lock is maintenance -> do nothing, whatever the gateway's state
#   * gateway answering /v1/models -> do nothing
#   * gateway process alive but not answering yet -> wait, it warms for ~90 s after a restart
#   * otherwise -> restart it exactly the way the start script does, and log why
#
# Run from cron every 5 minutes. It never touches a binary and never runs an engine itself.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/rig_lock.sh"
LOG="${WATCHDOG_LOG:-$HOME/bench/gateway_watchdog.log}"
say() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $*" >> "$LOG"; }

if rig_lock_maintenance; then exit 0; fi                      # a chain owns the box

KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
        http://127.0.0.1:8081/v1/models 2>/dev/null)
[ "$code" = 200 ] && exit 0

if pgrep -f "openai_[s]erver.py" >/dev/null; then
  # up but not answering: either warming after a restart, or busy with a long prefill, both
  # of which are normal here. Only complain if it stays that way.
  say "gateway process alive, /v1/models=$code — leaving it alone"
  exit 0
fi

say "gateway DOWN (/v1/models=$code, no process, no chain holding the lock) — restarting"
pkill -9 -x glm53 2>/dev/null
SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do
  sleep 10
  code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
          http://127.0.0.1:8081/v1/models 2>/dev/null)
  [ "$code" = 200 ] && { say "gateway back after $((i*10))s"; exit 0; }
done
say "gateway did NOT come back within 10 min — needs a person"
exit 1
