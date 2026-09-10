#!/bin/bash
# install_watchdog.sh — put the gateway watchdog in the rig's crontab (idempotent).
#
# Run ON THE RIG, and only when no chain is running: the watchdog restarts the gateway when it
# is down and no chain holds the rig lock, so installing it while a pre-lock chain has the
# gateway stopped on purpose would fight that chain. Chains launched through run_chain.sh
# hold the lock and are safe.
#
#   install_watchdog.sh [--every 5] [--remove]
set -u
HERE=$(cd "$(dirname "$0")" && pwd); EVERY=5; REMOVE=0
while [ $# -gt 0 ]; do
  case "$1" in --every) EVERY="$2"; shift 2;; --remove) REMOVE=1; shift;;
               *) echo "unknown argument: $1"; exit 2;; esac
done
LINE="*/$EVERY * * * * $HERE/gateway_watchdog.sh"
if pgrep -f "_[c]hain.sh" >/dev/null; then
  echo "REFUSED: a chain is running ($(pgrep -af '_[c]hain.sh' | head -1)) — install when the rig is idle"
  exit 3
fi
current=$(crontab -l 2>/dev/null | grep -v "gateway_watchdog.sh")
if [ "$REMOVE" = 1 ]; then
  printf '%s\n' "$current" | crontab -
  echo "watchdog removed; crontab now:"; crontab -l | grep -v "^#"
  exit 0
fi
printf '%s\n%s\n' "$current" "$LINE" | grep -v '^$' | crontab -
echo "installed: $LINE"
crontab -l | grep -v "^#"
echo "--- dry run (should do nothing while the gateway answers)"
"$HERE/gateway_watchdog.sh" && echo "watchdog: no action taken (correct while the gateway is up)"
tail -3 "${WATCHDOG_LOG:-$HOME/bench/gateway_watchdog.log}" 2>/dev/null
