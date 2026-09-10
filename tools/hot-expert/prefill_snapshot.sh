#!/bin/bash
# prefill_snapshot.sh -- timestamp the PREFILL track's (PREFILL-ROADMAP-2026-09.md, P-items)
# current performance against the LIVE gateway. Written 2026-09-10 after a re-baseline
# attempt used tools/rome_bench.sh (the DECODE-THROUGHPUT track's own harness, G-items,
# short prompts, no checkpoint/ledger) and produced a number that looked like a regression
# and was not -- it was the wrong tool for this track. See tools/hot-expert/MEASURING.md.
#
# Unlike rome_bench.sh, this script is READ-ONLY against the served gateway: it does not
# stop anything, does not spawn a competing engine, and needs no rig lock -- it is exactly
# what accept_live.sh / accept_ui.sh already do, run in the same spirit as those.
#
# It also does NOT write to any record file. It prints the fresh numbers next to the nearest
# historical numbers of the SAME regime (same tool, same sizes) so a human or an agent can
# eyeball plausibility before anything gets committed -- the step that was skipped on
# 2026-09-10 and let a bad number reach a commit. Appending the result to
# PREFILL-ROADMAP-2026-09.md is a separate, deliberate edit after that review, not automatic.
#
# Usage: prefill_snapshot.sh [tag]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROADMAP="$HERE/PREFILL-ROADMAP-2026-09.md"
TAG="${1:-snapshot-$(date +%Y%m%d-%H%M)}"
K=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
URL=http://127.0.0.1:8081

echo "=== prefill_snapshot $TAG $(date -Is)"

# Read-only against the gateway, but still a TIMING measurement: a chain that
# stops, restarts or floods the gateway underneath it makes these numbers a
# measurement of that chain. So this does not TAKE the lock (it must not stop
# anyone), but it refuses to produce numbers while someone else holds it --
# "one benchmark at a time" applies to the passive side too.
. "$HERE/rig_lock.sh"
if holder=$(rig_lock_holder); then
  echo "REFUSED: the rig lock is held by: $holder"
  echo "         Its chain is driving this gateway; TTFT measured now would be that"
  echo "         chain's numbers, not the served binary's. Wait for it to finish."
  exit 3
fi

pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: gateway not running"; exit 2; }
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -H "Authorization: Bearer $K" "$URL/v1/models")
[ "$code" = 200 ] || { echo "REFUSED: /v1/models=$code"; exit 2; }
echo "gateway up, binary=$(sha256sum "$HOME/src/colibri/c/glm53" 2>/dev/null | cut -c1-16)"

echo
echo "--- TTFT at the track's standard sizes (27,384,1230,3462), against the live gateway"
OUT=$(mktemp -d)
python3 "$HERE/ttft_serve.py" --url "$URL" --sizes 27,384,1230,3462 --repeat 2 --warm \
    --tag "$TAG" --json "$OUT/ttft.jsonl" 2>&1 | tee "$OUT/ttft.txt"

echo
echo "--- accept_live.sh (checkpoint reuse, ledger invariant, CANCEL path)"
"$HERE/accept_live.sh" 2>&1 | tee "$OUT/accept_live.txt"
AL_RC=${PIPESTATUS[0]}

echo
echo "--- nearest historical numbers already in $ROADMAP (for comparison, not authority) ---"
grep -oE '(TTFT|ttft)[^.]*[0-9][0-9.]*[×xs]' "$ROADMAP" | tail -15
echo
echo "--- most recent rev headline ---"
awk '/^\*\*Rev [0-9]+/{print; c++} c&&/^\*\*Rev [0-9]+/&&c>1{exit}' "$ROADMAP" | head -6

echo
echo "=== prefill_snapshot done. accept_live=$([ "$AL_RC" = 0 ] && echo PASS || echo FAIL)."
echo "    Raw output kept at $OUT -- review the TTFT numbers against the historical ones above"
echo "    (same order of magnitude, same direction as the last landed item) before writing"
echo "    anything into $ROADMAP by hand. This script does not append for you."

# Exit with accept_live's verdict, not the echo's. Anything driving this
# non-interactively (an agent, a loop, a chain) reads the exit code, and a
# script whose stated job is to catch a regression must not report success
# when its own acceptance check just failed.
exit "$AL_RC"
