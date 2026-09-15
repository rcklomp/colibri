#!/bin/bash
# Clean gateway restart with an ARGUMENT PROOF.
# Two traps this script exists to avoid, both hit on 2026-09-13:
#  1. a # comment between backslash-continued argument lines makes bash join the logical
#     line, treat the # as a comment and SILENTLY DROP every argument after it. `bash -n`
#     passes. So we read /proc/<pid>/cmdline and require each argument by name.
#  2. editing start_glm53.sh while a copy of it is still running: bash re-reads the file at
#     its saved byte offset and executes whatever is now there. So kill the SCRIPT SHELL
#     first, not just the python it spawned.
set -u
. ~/src/colibri/tools/hot-expert/rig_lock.sh
rig_lock_take "gw-restart" || { echo "LOCK BUSY: $(rig_lock_holder)"; exit 3; }
# CLAUDE.md: a script that stops the gateway must restart it on EVERY exit path.
# The abort branches below (engine/gateway still alive) used to exit with the
# service down and leave it to gateway_watchdog.sh, i.e. up to 5 minutes of
# downtime for the owner. Restore it here instead; the watchdog stays the net
# for the case where this script itself is killed.
gw_restore() {
  if ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[gw] restoring the gateway on exit"
    SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
  fi
  rig_lock_release
}
trap gw_restore EXIT
bash -n ~/start_glm53.sh || { echo "SYNTAX ERROR -- not restarting"; exit 1; }

# Drift check: ~/start_glm53.sh is checked into the repo verbatim at
# tools/hot-expert/rig/start_glm53.sh so it HAS history -- on 2026-09-13 nobody could say who
# set --max-tokens 256 or why, because the file had no history anywhere. Report drift; do not
# block on it (the rig copy is what runs, and an emergency edit must still be able to restart).
REPO_COPY=$HOME/src/colibri/tools/hot-expert/rig/start_glm53.sh
if [ -f "$REPO_COPY" ]; then
  if diff -q "$HOME/start_glm53.sh" "$REPO_COPY" >/dev/null; then
    echo "[gw] start_glm53.sh matches the repo copy"
  else
    echo "[gw] WARNING: ~/start_glm53.sh DIFFERS from the repo copy -- commit the change:"
    diff "$REPO_COPY" "$HOME/start_glm53.sh" | head -20
  fi
else
  echo "[gw] note: no repo copy at $REPO_COPY to compare against"
fi

pkill -f "start_[g]lm53.sh"; sleep 1          # the script shell first, so nothing respawns
pkill -f "openai_[s]erver.py"; sleep 3
pkill -9 -x glm53 2>/dev/null
for i in $(seq 1 90); do pgrep -x glm53 >/dev/null || break; sleep 1; done
pgrep -x glm53 >/dev/null && { echo "ENGINE STILL ALIVE after ${i}s -- abort"; exit 1; }
pgrep -f "openai_[s]erver.py" >/dev/null && { echo "GATEWAY STILL ALIVE -- abort"; exit 1; }
echo "[gw] all down after ${i}s; restarting"

SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
K=$(cat ~/.colibri_api_key)
for i in $(seq 1 180); do
  sleep 2
  [ "$(curl -s -o /dev/null -m 5 -w "%{http_code}" -H "Authorization: Bearer $K" http://127.0.0.1:8081/v1/models)" = 200 ] && { echo "[gw] answering after $((i*2))s"; break; }
done
PID=$(pgrep -f "openai_[s]erver.py" | head -1)
[ -n "$PID" ] || { echo "[gw] NO GATEWAY PROCESS -- FAIL"; exit 1; }
CMD=$(tr "\0" " " < /proc/$PID/cmdline)
FAIL=0
for want in "--max-tokens 4096" "--kv-slots 4" "--allowed-host 127.0.0.1" "--allowed-host localhost" "--allowed-host host.docker.internal" "--allowed-host rome.local"; do
  case "$CMD" in *"$want"*) echo "  ok   $want";; *) echo "  MISS $want"; FAIL=1;; esac
done
grep -q "KV_SLOTS=4" ~/glm53_server.log && echo "  ok   engine reports KV_SLOTS=4" || { echo "  MISS engine KV_SLOTS=4"; FAIL=1; }
echo "[gw] script shells now running: $(pgrep -fc "start_[g]lm53.sh")"
[ $FAIL -eq 0 ] && echo "[gw] ARGS VERIFIED" || echo "[gw] ARGUMENT CHECK FAILED"
exit $FAIL
