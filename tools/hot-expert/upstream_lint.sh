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
fail=0
flag() {  # flag <regex> <what>
  local hits; hits=$(printf '%s\n' "$added" | grep -nE "$1" | head -8)
  if [ -n "$hits" ]; then
    echo "  LEAK: $2"; printf '%s\n' "$hits" | sed 's/^/      /'; fail=1
  fi
}
echo "=== upstream_lint $BASE...$HEAD_REF ==="
flag '\b(G|Q|P|RP|C)[0-9]+[a-z]?\b'          'roadmap row labels (G4, Q10, P7b...) -- say what they describe'
flag 'tools/hot-expert'                       'fork-only paths'
flag 'rome rig|the rome box|\brome\b'         'fork machine names'
flag 'see the record|the record §|record §'   'fork-only documents'
flag 'CLAUDE\.md'                             'fork-only governance file'
flag '~/bench/'                               'fork-only rig paths'
echo
[ $fail -eq 0 ] && echo "upstream_lint: CLEAN" || echo "upstream_lint: LEAKS FOUND -- reword before sending"
exit $fail
