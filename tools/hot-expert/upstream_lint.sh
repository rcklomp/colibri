#!/bin/bash
# upstream_lint.sh -- scan a patch/branch for references that mean nothing
# OUTSIDE this fork, before it is sent to another project.
#
# Written 2026-09-15 after a reviewer on an upstream PR had to ask for it: two
# comments referred to "G10", "G7's router fix" and "G8's compiled
# max_active_levels probe" -- row labels from this fork's roadmap, opaque to a
# reader of that tree. Worse, when asked, the sibling PR was declared clean from
# memory and was not: it carried two more labels, a link to
# tools/hot-expert/README.md (a path that does not exist upstream), "the rome
# rig" and "see the record".
#
#   upstream_lint.sh <base> [head]     e.g. upstream_lint.sh upstream/dev HEAD
#
# Exits 1 if any added line carries a fork-only reference. Checks only ADDED
# lines, so it does not complain about the host project's own history.
set -u
BASE="${1:?usage: upstream_lint.sh <base> [head]}"; HEAD_REF="${2:-HEAD}"
added=$(git diff "$BASE...$HEAD_REF" -- '*.c' '*.h' '*.comp' '*.py' | grep '^+' | grep -v '^+++')
fail=0; leaks=0; stale=0
flag() {  # flag <regex> <what>
  local hits; hits=$(printf '%s\n' "$added" | grep -nE "$1" | head -8)
  if [ -n "$hits" ]; then
    echo "  LEAK: $2"; printf '%s\n' "$hits" | sed 's/^/      /'; leaks=1; fail=1
  fi
}
echo "=== upstream_lint $BASE...$HEAD_REF ==="
flag '\b(G|Q|P|RP|C)[0-9]+[a-z]?\b'          'roadmap row labels (G4, Q10, P7b...) -- say what they describe'
flag 'tools/hot-expert'                       'fork-only paths'
flag 'rome rig|the rome box|\brome\b'         'fork machine names'
flag 'see the record|the record §|record §'   'fork-only documents'
flag 'CLAUDE\.md'                             'fork-only governance file'
flag '~/bench/'                               'fork-only rig paths'
# --- base freshness ------------------------------------------------------
# A branch that falls far behind its base fails the host project's CI with ZERO
# jobs executed, which reads as a broken patch rather than a stale base. That is
# what happened to PR #1321 on 2026-09-15: 43 commits behind, dev had added a
# step to .github/workflows/ci.yml, and the run died before starting. Rebasing
# fixed it with no change to the patch.
behind=$(git rev-list --count "$HEAD_REF..$BASE" 2>/dev/null || echo 0)
echo "=== base freshness ==="
printf '  %s commits behind %s\n' "$behind" "$BASE"
if [ "${behind:-0}" -ge 25 ]; then
  echo "  STALE: rebase before pushing -- this is the shape that fails CI with zero jobs"
  stale=1; fail=1
elif [ "${behind:-0}" -ge 1 ]; then
  echo "  ok, but not current -- rebase if CI behaves oddly"
else
  echo "  ok: current with the base"
fi
echo
echo
if [ $fail -eq 0 ]; then echo "upstream_lint: CLEAN"
else
  # name the right remedy: rewording a comment does not fix a stale base, and
  # rebasing does not fix a leaked fork label. Reporting one as the other is how
  # a tool sends someone to edit code that is fine.
  [ "$leaks" = 1 ] && echo "upstream_lint: FORK REFERENCES LEAKED -- reword before sending"
  [ "$stale" = 1 ] && echo "upstream_lint: BASE IS STALE -- rebase before sending"
fi
exit $fail
