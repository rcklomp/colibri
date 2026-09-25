#!/bin/bash
# preflight.sh <smoke-command...> -- run on the rig BEFORE any long GPU job (2026-09-24).
# Refuses (exit 3) when the rig is busy, then runs the given command as a SHORT smoke run and
# exits with its status. Each long chain calls this with the same binary and flags as the real
# run, cut to a few tokens, so a wrong path, missing library, OOM at load or bad flag costs a
# minute of GPU time instead of an hour. Usage (on the rig):
#   ~/src/colibri/tools/hot-expert/preflight.sh docker run ... ./franken_decode_glm ... --tokens 1 2 3 --time 2
set -u
fail() { echo "PREFLIGHT REFUSED: $*"; exit 3; }
if [ -d ~/bench/.rig.lock ]; then
  owner=$(cat ~/bench/.rig.lock/owner 2>/dev/null); pid=$(echo "$owner" | awk '{print $2}')
  kill -0 "$pid" 2>/dev/null && [ "${PREFLIGHT_OWN_LOCK:-0}" != 1 ] && fail "rig lock held by: $owner"
fi
# exclude this script's own process tree: a caller's command line that merely NAMES a gate
# script matched itself (2026-09-25). Only processes outside our ancestry count as busy.
anc=" $$ "; p=$$; for _ in 1 2 3 4 5 6; do p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " "); [ -z "$p" ] || [ "$p" = 1 ] && break; anc="$anc$p "; done
busy=$(pgrep -f "quality_[e]val.py|franken_quality_[c]hain|_gpu_[g]ate.sh" | while read q; do case "$anc" in *" $q "*) ;; *) ps -o pid=,args= -p "$q";; esac; done | head -3)
[ -n "$busy" ] && fail "a measurement is running: $busy"
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done
[ "${PREFLIGHT_ALLOW_VRAM:-0}" != 1 ] && [ $m -gt 1024 ] && fail "VRAM in use (max ${m} MiB on a card) -- the gateway or another engine is still up"
[ $# -eq 0 ] && { echo "PREFLIGHT OK (no smoke command given)"; exit 0; }
echo "--- preflight smoke: $*" | cut -c1-200
start=$(date +%s)
"$@" > /tmp/preflight_smoke.$$ 2>&1; rc=$?
tail -5 /tmp/preflight_smoke.$$ | cut -c1-200
grep -qiE "HIP error|out of memory|fault|error while loading|No such file|CHECK_FAIL|Traceback" /tmp/preflight_smoke.$$ && rc=${rc:-1} && [ $rc -eq 0 ] && rc=1
rm -f /tmp/preflight_smoke.$$
echo "PREFLIGHT $( [ $rc -eq 0 ] && echo OK || echo FAIL ) rc=$rc in $(( $(date +%s) - start )) s"
exit $rc
