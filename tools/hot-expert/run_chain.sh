#!/bin/bash
# run_chain.sh <chain-script> [args…] — the single entry point for anything that stops the
# owner's gateway. Takes the rig lock, so a second chain (a stray launcher, a second session,
# a cron) is refused instead of interleaving; releases it on every exit path, including a
# signal; and reports what the chain left in service.
#
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/p9_chain.sh \
#       > ~/bench/p9_chain.log 2>&1 < /dev/null &
#
# Launch chains this way and nothing else has to remember the rules.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/rig_lock.sh"
CHAIN=${1:?a chain script}; shift || true
[ -x "$CHAIN" ] || { echo "run_chain: $CHAIN is not executable"; exit 2; }
NAME=$(basename "$CHAIN" .sh)

rig_lock_take "$NAME" || exit 3
trap 'rig_lock_release' EXIT INT TERM

echo "=== run_chain $NAME $(date +%Y-%m-%dT%H:%M:%S%z) (lock held by $(rig_lock_holder))"
"$CHAIN" "$@"; rc=$?
echo "=== run_chain $NAME exited rc=$rc $(date +%Y-%m-%dT%H:%M:%S%z)"

# Whatever the chain decided, the owner's service must be up when we let go of the lock.
KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
        http://127.0.0.1:8081/v1/models 2>/dev/null)
if [ "$code" != 200 ]; then
  echo "run_chain: the gateway is NOT answering after the chain (/v1/models=$code) -- restarting"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53 2>/dev/null
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
  for i in $(seq 1 60); do sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
            http://127.0.0.1:8081/v1/models 2>/dev/null)
    [ "$code" = 200 ] && break
  done
fi
echo "run_chain: in service $(sha256sum "$HOME/src/colibri/c/glm53" | cut -c1-16), /v1/models=$code"
exit $rc
