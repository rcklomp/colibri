#!/bin/bash
# doc_currency.sh -- is what CLAUDE.md tells you to read first actually CURRENT?
#
# CLAUDE.md's read-first order calls the roadmaps "current by construction" and
# warns that a stale pointer sends a session chasing something already landed.
# On 2026-09-15 three separate stale pointers were found in one day, each only
# because the owner pushed: the prefill roadmap was four days behind (P12 and a
# live incident missing), its item table had no rows for P11/P12, and Track Q's
# "what is next" still told the reader to run a probe that had been completed.
#
# The check is blunt on purpose: every merge commit that lands an item should be
# mentioned in the docs. If a landed item's name appears nowhere in the roadmaps
# or the record, the docs are behind and this exits 1. No rig, no engine, seconds.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
RM=tools/hot-expert/ROADMAP-2026-09.md
PF=tools/hot-expert/PREFILL-ROADMAP-2026-09.md
RC=tools/hot-expert/ROME-3x7900XTX-2026-09-04.md
SINCE="${1:-2026-09-01}"
fail=0

echo "=== landed items since $SINCE vs the docs ==="
# item ids out of merge subjects: "merge perf/q10-..." / "P11 gated" / "Q5 --" etc.
ids=$(git log --merges --since="$SINCE" --format=%s | grep -oiE '\b(P|Q|G|C|RP)[0-9]+[a-z]?\b' | tr 'a-z' 'A-Z' | sort -u)
for id in $ids; do
  hits=$(grep -oiE "\b$id\b" "$RM" "$PF" "$RC" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$hits" -eq 0 ]; then printf "  MISSING  %-5s landed, appears in NO doc\n" "$id"; fail=1
  else printf "  ok       %-5s (%s mentions)\n" "$id" "$hits"; fi
done

echo
echo "=== pointer freshness ==="
newest=$(git log --merges -1 --format=%ad --date=short)
prev=$(grep -oE 'rev [0-9]+, [0-9]{4}-[0-9]{2}-[0-9]{2}' "$PF" | head -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}')
printf "  newest merge:        %s\n  prefill roadmap rev: %s\n" "$newest" "${prev:-UNKNOWN}"
if [ -n "${prev:-}" ] && [ "$prev" \< "$newest" ]; then
  echo "  STALE: the newest landed work is newer than the roadmap's own rev date"; fail=1
else echo "  ok: roadmap rev is not behind the newest merge"; fi

echo
echo "=== 'next' pointers that name a finished item ==="
for f in "$RM" "$PF"; do
  grep -oiE 'next item is [^.]{0,60}' "$f" 2>/dev/null | while read -r l; do
    nid=$(echo "$l" | grep -oiE '\b(P|Q|G|C|RP)[0-9]+[a-z]?\b' | head -1 | tr 'a-z' 'A-Z')
    [ -z "$nid" ] && continue
    if git log --merges --since="$SINCE" --format=%s | grep -qiE "\b$nid\b"; then
      echo "  SUSPECT  $(basename "$f"): '$l' -- but $nid appears in a merge subject (landed?)"
    fi
  done
done

echo
[ $fail -eq 0 ] && echo "doc_currency: PASS" || echo "doc_currency: FAIL -- the docs are behind the tree"
exit $fail
