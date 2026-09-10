#!/bin/bash
# P11 chain: ckpt_disk_touch must prove the file is the capture it thinks it is.
# Structure is p10_chain.sh's, minus the build: the candidate is already built in
# a separate worktree (~/src/colibri-p11), so the served tree and the served
# binary are untouched until the gate has passed.
#
# Launch through run_chain.sh, never directly -- it takes the rig lock:
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#       ~/src/colibri/tools/hot-expert/p11_chain.sh > ~/bench/p11_chain.log 2>&1 < /dev/null &
#
# Steps:
#   1. verify the pristine snapshot is byte-for-byte what is in service
#   2. stop the gateway, wait for the engine to actually die (ETXTBSY trap)
#   3. prefill_gate.sh pristine vs candidate -- BINARY PATHS, nothing installed
#   4. the guard's own regression test: a file that is NOT this capture must be
#      left untouched. This is the test the P10 gate did not have, and the whole
#      reason this item exists.
#   5. install + restart + accept_live.sh; verified revert on any failure
#   6. the gateway is restarted on EVERY exit path
set -u
TAG=p11$(date +%m%d%H%M)
OUT=~/bench/p11_chain_out; mkdir -p "$OUT"
PRISTINE=~/bench/glm53.p11base
CAND=~/src/colibri-p11/c/glm53
HERE=~/src/colibri/tools/hot-expert
LOG=~/glm53_server.log
SERVED=0
INSTALLED=0

start_gateway() {
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_PREFIX_CKPT_MIN \
      -u GLM53_CKPT_VALUE_EVICT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      -u COLI_LEDGER -u COLI_PREFIX_PIN \
      SKIP_WARM=1 setsid nohup ~/start_glm53.sh > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat ~/.colibri_api_key)" \
         -w '%{http_code}' http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  sleep 3
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 120); do pgrep -x glm53 >/dev/null || return 0; sleep 2; done
  echo "FATAL: an engine is still alive after 240 s"; return 1
}

revert() {
  [ "$INSTALLED" = 1 ] || return 0
  cp -f "$PRISTINE" ~/src/colibri/c/glm53
  local got want
  got=$(sha256sum ~/src/colibri/c/glm53 | cut -d" " -f1)
  want=$(sha256sum "$PRISTINE" | cut -d" " -f1)
  [ "$got" = "$want" ] && echo "REVERTED, verified: $(echo "$got" | cut -c1-16)" \
                       || echo "FATAL: revert did NOT verify ($got vs $want)"
  INSTALLED=0
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP          # never run this handler twice
  [ "$SERVED" = 1 ] || revert
  pgrep -f "openai_[s]erver.py" >/dev/null || start_gateway
  echo "=== p11_chain exit rc=$rc served=$SERVED in-service=$(sha256sum ~/src/colibri/c/glm53 | cut -c1-16)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== P11 chain $TAG $(date -Is)"

# --- 1. the pristine must be exactly what is in service, or the gate is meaningless
insvc=$(sha256sum ~/src/colibri/c/glm53 | cut -d" " -f1)
prist=$(sha256sum "$PRISTINE" | cut -d" " -f1)
[ "$insvc" = "$prist" ] || { echo "REFUSED: pristine != in service ($prist vs $insvc)"; exit 2; }
[ -x "$CAND" ] || { echo "REFUSED: candidate $CAND missing"; exit 2; }
cand=$(sha256sum "$CAND" | cut -d" " -f1)
[ "$cand" != "$prist" ] || { echo "REFUSED: candidate is identical to pristine -- nothing was built"; exit 2; }
echo "pristine $(echo "$prist" | cut -c1-16)  candidate $(echo "$cand" | cut -c1-16)"

# --- 2. stop the gateway
stop_gateway || exit 2

# --- 3. the shared oracle. Private ckpt dir + checkpoints OFF, per CLAUDE.md:
#        two gates nearly passed because the candidate restored what the pristine wrote.
export GLM53_PREFIX_CKPT=0
export COLI_CKPT_DIR="$OUT/ckpt_gate"
mkdir -p "$COLI_CKPT_DIR"
"$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG" > "$OUT/gate.log" 2>&1
grc=$?
grep -E "teacher_forcing|cosine|max_abs|TTFT|size " "$OUT/gate.log" | tail -12
[ "$grc" = 0 ] || { echo "GATE FAILED rc=$grc -- see $OUT/gate.log"; exit 1; }
echo "step 3 PASS (bit-identical; this change touches no compute path)"

# --- 4. the guard's own regression test (the test P10's gate did not have).
#        Five cases against the real ckpt_disk_touch; case C fails on the
#        pre-P11 binary and passes after, verified against both builds.
( cd ~/src/colibri-p11 && make -C c tests/test_glm53_ckpt_touch VK=1 ) > "$OUT/touch_build.log" 2>&1
[ -x ~/src/colibri-p11/c/tests/test_glm53_ckpt_touch ] || {
  echo "GUARD TEST did not build -- see $OUT/touch_build.log"; tail -5 "$OUT/touch_build.log"; exit 1; }
~/src/colibri-p11/c/tests/test_glm53_ckpt_touch > "$OUT/touch_case.log" 2>&1
trc=$?
grep -E "^[A-E] ok|check failed|all cases passed" "$OUT/touch_case.log" | head -8
[ "$trc" = 0 ] || { echo "GUARD TEST FAILED rc=$trc -- see $OUT/touch_case.log"; exit 1; }
echo "step 4 PASS (a foreign file is left untouched; the normal path still counts)"

# --- 5. install, restart, accept on the owner's own path
unset GLM53_PREFIX_CKPT COLI_CKPT_DIR
cp -f "$CAND" ~/src/colibri/c/glm53 && INSTALLED=1
echo "installed $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16)"
start_gateway
"$HERE/accept_live.sh" > "$OUT/accept.log" 2>&1
arc=$?
tail -12 "$OUT/accept.log"
[ "$arc" = 0 ] || { echo "accept_live FAILED rc=$arc -- reverting"; exit 1; }

SERVED=1
echo "=== P11 IN SERVICE: $(sha256sum ~/src/colibri/c/glm53 | cut -c1-16), serving script unchanged"
