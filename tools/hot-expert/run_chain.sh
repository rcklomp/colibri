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
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); . "$HERE/rig_lock.sh"; . "$HERE/service_lib.sh"
svc_say() { echo "run_chain: $*"; }
CHAIN=${1:?a chain script}; shift || true
[ -x "$CHAIN" ] || { echo "run_chain: $CHAIN is not executable"; exit 2; }
NAME=$(basename "$CHAIN" .sh)

rig_lock_take "$NAME" || exit 3
# The chain runs from a PRIVATE COPY, next to the original (so `dirname $0` still resolves): bash reads a script incrementally, so
# overwriting a running chain (an scp, a cp, an editor) made it resume in the middle of the NEW file -- on 2026-10-07 a chain whose
# engine had just died carried on into garbage (`line 47: ama-glm53/build-hip/bin: No such file`, rc=127) and reported a death in
# setup. With the copy, a chain script can be replaced at any time; RUN_CHAIN_ORIG names the original for a chain that wants it.
SNAP="$(dirname "$CHAIN")/.run_chain.$NAME.$$.sh"
cp -p "$CHAIN" "$SNAP" 2>/dev/null && chmod +x "$SNAP" || SNAP="$CHAIN"
trap '[ "$SNAP" != "$CHAIN" ] && rm -f "$SNAP"; rig_lock_release' EXIT INT TERM
export RUN_CHAIN_ORIG="$CHAIN"

echo "=== run_chain $NAME $(date +%Y-%m-%dT%H:%M:%S%z) (lock held by $(rig_lock_holder))"
"$SNAP" "$@"; rc=$?
echo "=== run_chain $NAME exited rc=$rc $(date +%Y-%m-%dT%H:%M:%S%z)"

# Whatever the chain decided, the owner's service must be up when we let go of the lock -- unless the rig is reserved.
# Which engine "the service" is, and whether it is alive, is service_lib.sh's call (2026-10-06: this block used to test
# `ps -C glm53` only and restart the OLD Colibri engine, which would have killed a healthy Franken service).
KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
code=$(service_models_code)
# A server whose engine child is gone still answers /v1/models=200 while every chat request
# fails (2026-09-16: a chain's exit trap killed the engine by name; 28 min unnoticed). Treat
# "server alive, no live engine" as down.
if [ "$code" = 200 ] && pgrep -f "openai_[s]erver.py" >/dev/null && ! engine_alive; then
  echo "run_chain: /v1/models=200 but the gateway's engine (glm53 / franken_dec_glm) is GONE -- treating as down"
  code=dead-engine
fi
if [ -e "$HOME/bench/.dev_reserved" ]; then
  echo "run_chain: rig reserved for development (~/bench/.dev_reserved): gateway left down"; code=reserved
elif [ "$code" != 200 ]; then
  echo "run_chain: the gateway is NOT answering after the chain (/v1/models=$code) -- restoring the service engine"
  rig_lock_release                     # serve_alt.sh takes the lock itself and refuses while another process holds it
  service_restore && code=200 || code=restore-failed
fi
if [ "$code" = 200 ]; then service_engine_select; echo "run_chain: in service (engine: $SERVICE_ENGINE_NAME, alive: $(engine_alive && echo yes || echo NO)), /v1/models=$code"
else echo "run_chain: NOT in service: $code"; fi
exit $rc
