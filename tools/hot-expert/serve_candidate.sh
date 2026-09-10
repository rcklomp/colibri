#!/bin/bash
# serve_candidate.sh <candidate-binary> <pristine-binary> [pristine-shaders-dir]
#
# The LAST step of every chain: put a gated candidate in service and prove it on the user's
# path, or put the pristine back. Written after P7b (2026-09-09), when a binary that had passed
# every engine gate left the owner with 380-second new chats because nothing had looked at
# the request after the one under test. Runs on the rig.
#
#  1. wait until no request is in flight, stop the gateway, kill the engine, wait for it to die
#  2. install the candidate into ~/src/colibri/c/glm53 (sha256 verified)
#  3. restart the gateway with the serving script as it is (SKIP_WARM), wait for /v1/models
#  4. accept_live.sh -- the four checks on Open WebUI's own backend
#  5. on any failure: pristine binary (+ shaders) back, sha256 verified, restart, /v1/models
# Exit 0 only when the candidate is serving AND accept_live.sh passed.
set -u
CAND=${1:?candidate}; PRISTINE=${2:?pristine}; PSHADERS=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
# The rig lock: if a chain holds it, this IS that chain (run_chain.sh took it and we inherit
# the environment). If nothing holds it, take it, so a hand-run serve can never interleave
# with a chain. Released on every exit path.
. "$HERE/rig_lock.sh"
if ! rig_lock_holder >/dev/null; then rig_lock_take "serve_candidate" || exit 3
  trap 'rig_lock_release' EXIT INT TERM; fi; TREE="$HOME/src/colibri"; BIN="$TREE/c/glm53"; LOG="$HOME/glm53_server.log"
sha() { sha256sum "$1" | cut -c1-16; }
restart_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for i in $(seq 1 120); do
    curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$HOME/.colibri_api_key")" -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null | grep -q 200 && return 0
    sleep 5
  done
  echo "gateway did not answer /v1/models within 10 min"; return 1
}
stop_gateway() {
  # never cut a user's turn: wait until the last POST has its [req] line
  for i in $(seq 1 360); do
    posts=$(grep -c "POST /v1/chat/completions\|POST /v1/completions" "$LOG" 2>/dev/null || echo 0)
    reqs=$(grep -c "\[req\] " "$LOG" 2>/dev/null || echo 0)
    [ "$posts" -le "$reqs" ] && break; sleep 10
  done
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53 2>/dev/null
  for i in $(seq 1 60); do pgrep -x glm53 >/dev/null || break; sleep 2; done
  pgrep -x glm53 >/dev/null && { echo "engine still alive after stop"; return 1; }
  return 0
}
install_bin() {   # install_bin <src> [shaders]
  cp -p "$1" "$BIN.new" && mv -f "$BIN.new" "$BIN" || return 1
  [ "$(sha "$1")" = "$(sha "$BIN")" ] || { echo "sha mismatch after install"; return 1; }
  if [ -n "${2:-}" ] && [ -d "$2" ]; then cp -p "$2"/*.spv "$TREE/c/shaders/" || return 1; fi
  return 0
}
echo "=== serve_candidate $(date -Is) candidate=$(sha "$CAND") pristine=$(sha "$PRISTINE")"
stop_gateway || exit 2
if install_bin "$CAND" && restart_gateway && "$HERE/accept_live.sh"; then
  echo "=== IN SERVICE: candidate $(sha "$BIN"), accept_live PASS $(date -Is)"; exit 0
fi
echo "=== FAILED -- reverting to the pristine $(date -Is)"
stop_gateway
install_bin "$PRISTINE" "$PSHADERS" || echo "REVERT INSTALL FAILED -- check c/glm53 by hand"
restart_gateway
echo "=== pristine back: $(sha "$BIN") (expected $(sha "$PRISTINE")); accept_live on it:"; "$HERE/accept_live.sh" --canary
exit 1
