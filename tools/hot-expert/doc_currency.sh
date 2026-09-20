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
PL=tools/hot-expert/FRANKEN-ENGINE-PLAN-2026-09-15.md   # the F items (added 2026-09-20)
SINCE="${1:-2026-09-01}"
fail=0

echo "=== landed items since $SINCE vs the docs ==="
# item ids out of merge subjects: "merge perf/q10-..." / "P11 gated" / "Q5 --" etc.
ids=$(git log --merges --since="$SINCE" --format=%s | grep -oiE '\b(P|Q|G|C|RP|F)[0-9]+[a-z]?\b' | tr 'a-z' 'A-Z' | grep -vE '^F16C?$' | sort -u)
for id in $ids; do
  hits=$(grep -oiE "\b$id\b" "$RM" "$PF" "$RC" "$PL" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$hits" -eq 0 ]; then printf "  MISSING  %-5s landed, appears in NO doc\n" "$id"; fail=1
  else printf "  ok       %-5s (%s mentions)\n" "$id" "$hits"; fi
done

echo
echo "=== Franken plan: every landed F item has a row in its item table ==="
# Added 2026-09-20: the plan's rev log was current while its section 8.3 table
# had stopped at F7 -- F8, F9a and F10 landed with no row, and nothing noticed,
# because this script did not know F items existed.
for id in $(echo "$ids" | grep -E '^F[0-9]+[A-Z]?$'); do
  base=$(echo "$id" | sed -E 's/[A-Z]$//')       # F6A is carried by the F6 row
  if grep -qiE "^\| \*\*($id|$base)\b" "$PL"; then printf "  ok       %-5s has a row\n" "$id"
  else printf "  MISSING  %-5s landed, no row in the plan's item table\n" "$id"; fail=1; fi
done
rev_top=$(grep -oE '^> \*\*Rev [0-9]+' "$PL" | head -1 | grep -oE '[0-9]+')
rev_max=$(grep -oE '^> \*\*Rev [0-9]+' "$PL" | grep -oE '[0-9]+' | sort -n | tail -1)
if [ "${rev_top:-0}" != "${rev_max:-0}" ]; then echo "  STALE: the plan's top rev ($rev_top) is not its highest ($rev_max)"; fail=1
else echo "  ok: plan top rev is $rev_top (look this up before writing the next one)"; fi

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
    nid=$(echo "$l" | grep -oiE '\b(P|Q|G|C|RP|F)[0-9]+[a-z]?\b' | head -1 | tr 'a-z' 'A-Z')
    [ -z "$nid" ] && continue
    if git log --merges --since="$SINCE" --format=%s | grep -qiE "\b$nid\b"; then
      echo "  SUSPECT  $(basename "$f"): '$l' -- but $nid appears in a merge subject (landed?)"
    fi
  done
done

echo "=== references that no longer resolve ==="
for f in $(git ls-files 'tools/hot-expert/*.md' CLAUDE.md); do
  grep -oE '\b(tools|c)/[A-Za-z0-9_./-]+\.(sh|py|c|h|md|mjs|comp|spv)\b' "$f" 2>/dev/null | sort -u | while read -r p; do
    [ -e "$p" ] || echo "  DEAD  $f -> $p"
  done
done | sort -u | grep . && fail=1 || echo "  ok: every repo path named in the docs exists"

echo
echo "=== bare filenames named in the docs that exist nowhere ==="
# NOT just prefixed paths: the dead-reference check below used to require a
# tools/ or c/ prefix, so `patch_trace.py` sitting in a markdown table was
# invisible -- and eight such scripts had been deleted while README.md still
# documented them. Rig-side scripts (~/bench) and build artefacts (.spv, and
# generated fixtures) are excluded: they legitimately do not live in the repo.
git ls-files 'tools/hot-expert/*.md' CLAUDE.md | xargs grep -ohE '`[A-Za-z0-9_][A-Za-z0-9_.-]*\.(py|sh|mjs)`' 2>/dev/null \
  | tr -d '`' | sort -u | while read -r n; do
      grep -qE "\b$n\b" <<<"$(git ls-files)" && continue
      case "$n" in *_chain.sh|q7_lib.sh|owui_report.sh) continue;; esac   # rig-side, live in ~/bench
      grep -qE "(~/bench|bench/)$n" $(git ls-files 'tools/hot-expert/*.md' CLAUDE.md) 2>/dev/null && continue
      # a name the docs explicitly record as deleted is documentation, not rot;
      # look at the surrounding paragraph, not the single line -- the marker
      # phrase and the name are usually a couple of lines apart.
      grep -qE "had been deleted|no longer exists|is gone|glob matching nothing" \
        <<<"$(grep -h -B4 -A4 "$n" $(git ls-files 'tools/hot-expert/*.md' CLAUDE.md) 2>/dev/null)" && continue
      echo "  DEAD  $n named in the docs, present nowhere in the repo"
    done | sort -u | grep . && fail=1 || echo "  ok: every bare script name in the docs resolves"

echo
echo "=== untracked files loitering in the working tree ==="
n=$(git status --porcelain | grep -c '^??') || true
if [ "${n:-0}" -gt 0 ]; then
  echo "  $n untracked file(s) -- these keep git status dirty and hide real changes:"
  git status --porcelain | grep '^??' | sed 's/^/    /'
  fail=1
else echo "  ok: none"; fi

echo
echo
[ $fail -eq 0 ] && echo "doc_currency: PASS" || echo "doc_currency: FAIL -- the docs are behind the tree"
exit $fail
