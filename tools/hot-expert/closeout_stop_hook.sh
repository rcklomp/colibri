#!/bin/bash
# closeout_stop_hook.sh -- the Claude Code STOP hook of the closeout rule (since 2026-10-07; registered in ~/.claude/settings.json, user scope).
# Runs when a turn ends. Reads the hook's JSON on stdin. It blocks the stop (exit 2, stderr goes to the model) ONLY when ALL of these hold:
#   - the session works in ~/Projects, ~/Projects/colibri* or ~/Projects/franken-engine* (any other project: silent exit 0);
#   - it has not already blocked this stop once (`stop_hook_active`: never a loop);
#   - the local state differs from the last PASS (a stamp of both repos' HEAD + working-tree fingerprint: an unchanged, already-verified state costs ~50 ms);
#   - closeout_check.sh --quick FAILs (an unsynced or dirty repo, stale docs, a handoff that does not name the installed binary, a missing memory file);
#   - the rig is IDLE: while a rig job runs the session is legitimately waiting for its watcher, and the failures are expected mid-run.
# A timeout, an unreachable rig or any trouble in this script itself NEVER blocks (exit 0). To switch it off: delete the hook entry in ~/.claude/settings.json.
IN=$(cat)
parsed=$(printf '%s' "$IN" | python3 -I -c 'import sys,json
try:
    d=json.load(sys.stdin); print(1 if d.get("stop_hook_active") else 0); print(d.get("cwd",""))
except Exception:
    print(1); print("")' 2>/dev/null)
active=$(printf '%s\n' "$parsed" | sed -n 1p); cwd=$(printf '%s\n' "$parsed" | sed -n 2p)
[ "$active" = 0 ] || exit 0
case "$cwd" in "$HOME/Projects"|"$HOME/Projects/colibri"*|"$HOME/Projects/franken-engine"*) ;; *) exit 0;; esac
C=$HOME/Projects/colibri; F=$HOME/Projects/franken-engine
[ -d "$C/.git" ] && [ -d "$F/.git" ] || exit 0
fp() { printf '%s %s %s' "$(git -C "$1" rev-parse HEAD 2>/dev/null)" "$(git -C "$1" status --porcelain 2>/dev/null | cksum)" "$(git -C "$1" rev-parse origin/"$2" 2>/dev/null)"; }
STATE="$(fp "$C" hot-expert-tier) | $(fp "$F" main)"
STAMP=$HOME/.cache/closeout.stamp; mkdir -p "$HOME/.cache" 2>/dev/null
[ "$(cat "$STAMP" 2>/dev/null)" = "$STATE" ] && exit 0
out=$(perl -e 'alarm 55; exec @ARGV' bash "$C/tools/hot-expert/closeout_check.sh" --quick 2>&1); rc=$?
if [ $rc = 0 ]; then printf '%s' "$STATE" > "$STAMP"; exit 0; fi
case "$out" in *rig_busy=*) ;; *) exit 0;; esac              # no machine line: the check itself broke or timed out
echo "$out" | grep -q '^rig_busy=1' && exit 0                 # a rig job is running: a legitimate wait
{
  echo "CLOSEOUT CHECK FAILED -- do not end the task like this (rule since 2026-10-07: the owner must never have to ask for this). Fix it, then finish; if something here cannot be fixed, say so plainly in the final message."
  echo "$out" | grep -E '^(FAIL|WARN)' | cut -c1-300
  echo "How: run the task-closeout skill (full check: bash $C/tools/hot-expert/closeout_check.sh --full); commit, push to Gitea, relay to the rig (handoff 06b section 2); GitHub only with the owner's go-ahead."
} >&2
exit 2
