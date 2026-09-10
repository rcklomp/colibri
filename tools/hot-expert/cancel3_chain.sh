#!/bin/bash
# cancel3_chain.sh — gate the corrected cancel fix (3d80c07 + 07fa39a), gateway python only.
#
# The bug: an abandoned request holds the engine for its whole prefill, ~1 in 3 (measured
# 2026-09-09: 158-255 s). Cause: the CANCEL can reach the engine before it has dequeued that
# SUBMIT, so it cancels nothing, answers ERROR NOT_FOUND, and the request runs on. The first
# attempt at a fix (740a9e2) deferred the CANCEL to the ACCEPT frame and was worse -- 5/5 at
# ~155 s with no CANCEL at all, because ACCEPT only arrives once prefill is done. This pair
# keeps the immediate send and retries on the NOT_FOUND ack, and stops the dispatcher
# stranding a request whose cancel failed.
#
# Gate: ~/bench/check4.sh (colibri-77's harness -- waits for an idle engine, so it measures
# the cancel path and not a backlog) six cycles, every one <= 45 s AND a CANCEL line each
# time; then accept_live.sh in full. Serves on pass, reverts the tree and restarts otherwise.
set -u
TREE=$HOME/src/colibri; LOG=$HOME/glm53_server.log; K=$(cat $HOME/.colibri_api_key)
LAST_GATED=$(cd $TREE && git rev-parse HEAD)
echo "=== cancel3 chain $(date +%Y-%m-%dT%H:%M:%S%z); last gated tree: $(cd $TREE && git log --oneline -1)"
restart() {
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53 2>/dev/null
  for _ in $(seq 1 60); do pgrep -x glm53 >/dev/null || break; sleep 2; done
  SKIP_WARM=1 setsid nohup $HOME/start_glm53.sh > $LOG 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' -H "Authorization: Bearer $K" \
         http://127.0.0.1:8081/v1/models 2>/dev/null)" = 200 ] && return 0
    sleep 5
  done
  echo "gateway did not come back"; return 1
}
revert() {
  echo "--- REVERTING the tree to $LAST_GATED"
  (cd $TREE && git reset -q --hard "$LAST_GATED" && git log --oneline -1)
  restart
}
cd $TREE || exit 2
git merge -q --ff-only p0-sync || { echo "merge refused (p0-sync is not a fast-forward)"; exit 2; }
echo "merged: $(git log --oneline -1)"
grep -q cancel_retries c/openai_server.py || { echo "the corrected fix is not in the tree"; revert; exit 2; }
grep -q "transient = (message ==" c/openai_server.py || { echo "the dispatcher half is missing"; revert; exit 2; }
restart || { revert; exit 2; }

echo "--- check4, six cycles (bound: <= 45 s AND a CANCEL line every time)"
CYCLES=6
out=$($HOME/bench/check4.sh $CYCLES | tr -d "\r"); echo "$out"
# Count the cycles that PARSED, not just the ones that failed: a run that produced no
# readable lines would otherwise report "0 bad" and pass on nothing -- the same shape as the
# empty-oracle pass and the 0-checked ledger alarm this project fixed the same night.
fail=$(echo "$out" | CYCLES=$CYCLES python3 -c '
import os, re, sys
want = int(os.environ["CYCLES"]); seen = bad = 0
for line in sys.stdin:
    m = re.search(r"answered in ([\d.]+)s, engine CANCEL lines \+(\d+)", line)
    if not m: continue
    seen += 1
    if float(m.group(1)) > 45 or int(m.group(2)) < 1: bad += 1
if seen != want:
    print(f"NO DATA: {seen} of {want} cycles parsed", file=sys.stderr)
    print(want)                      # unparseable == failed
else:
    print(bad)')
echo "cycles above the bound or without a CANCEL: $fail (of $CYCLES)"
echo "--- accept_live"
$TREE/tools/hot-expert/accept_live.sh; alrc=$?
echo "accept_live rc=$alrc"
if [ "$fail" = 0 ] && [ $alrc = 0 ]; then
  echo "=== IN SERVICE: $(cd $TREE && git log --oneline -1)"
  (cd $TREE && git rev-parse HEAD > $HOME/bench/served.commit); exit 0
fi
echo "=== GATE FAILED (bad cycles=$fail, accept_live=$alrc)"; revert; exit 1
