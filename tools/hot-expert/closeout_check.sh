#!/bin/bash
# closeout_check.sh [--quick|--full] -- the "homework" at the end of every finished task (rule since 2026-10-07; the owner had to ask for it after every task).
# Deterministic, read-only (the one write is `git fetch` in --full), no GPU, never pushes anywhere. Checks:
#   repos      colibri (branch hot-expert-tier) and franken-engine (main): working tree clean, nothing unpushed to Gitea, the rig's copy at the same commit (and its tree clean),
#              the public GitHub fork's lag as INFO (it is pushed only with the owner's go-ahead, never by this script)
#   docs       tools/hot-expert/doc_currency.sh (the entry-point docs name the plan's top rev, no dead references, no loitering files);
#              the handoff names the sha of the INSTALLED serving binary
#   rig        idle or busy (lock, engine/gateway/chain processes, the franken_engine container); VRAM empty when idle; the reservation flag present; one worktree on
#              the rig; no sequencer scripts left over from the last 24 h; ~/bench copies of repo scripts that DRIFTED from the repo (the stale-copy trap of 2026-10-07)
#   config     the task-closeout skill installed == the repo copy; the Stop hook registered in ~/.claude/settings.json; every memory file the index names exists
# Exit 0 when no FAIL (WARN and INFO never fail), 1 otherwise. The last stdout line is machine-readable: `rig_busy=0|1`.
# --quick: what the Stop hook runs (no fetch, no drift scan, only WARN/FAIL lines printed); --full: everything, every line printed.
set -u
MODE=full; [ "${1:-}" = "--quick" ] && MODE=quick
C=$HOME/Projects/colibri; F=$HOME/Projects/franken-engine
CB=hot-expert-tier; FB=main
fails=0; warns=0; busy=0
ok()   { [ "$MODE" = full ] && echo "ok    $*"; return 0; }
info() { [ "$MODE" = full ] && echo "info  $*"; return 0; }
warn() { echo "WARN  $*"; warns=$((warns+1)); }
fail() { echo "FAIL  $*"; fails=$((fails+1)); }
with_alarm() { local n=$1; shift; perl -e 'alarm shift; exec @ARGV' "$n" "$@"; }

# ---------------------------------------------------------------- the rig, in ONE round trip (a script on stdin: no pgrep pattern ever sits in a command line)
LIST=""
if [ "$MODE" = full ]; then
  for f in "$C"/tools/hot-expert/ckpt1006/*.sh "$C"/tools/hot-expert/ckpt1006/*.py; do
    [ -f "$f" ] && LIST="$LIST$(basename "$f") $(shasum -a 256 "$f" | cut -c1-16)
"
  done
fi
RIGOUT=$(with_alarm 25 ssh -o ConnectTimeout=5 -o BatchMode=yes rome bash -s 2>/dev/null <<EOF
echo "c_head=\$(git -C ~/src/colibri rev-parse $CB 2>/dev/null)"
echo "c_dirty=\$(git -C ~/src/colibri status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo "f_head=\$(git -C ~/src/franken-engine rev-parse $FB 2>/dev/null)"
echo "f_dirty=\$(git -C ~/src/franken-engine status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo "f_worktrees=\$(git -C ~/src/franken-engine worktree list 2>/dev/null | wc -l | tr -d ' ')"
echo "lock=\$([ -d ~/bench/.rig.lock ] && head -1 ~/bench/.rig.lock/owner 2>/dev/null || echo none)"
echo "flag=\$([ -e ~/bench/.dev_reserved ] && echo present || echo ABSENT)"
echo "procs=\$(pgrep -fa '[f]ranken_dec_glm|[f]ranken_decode_glm|[f]ranken_decode_ds4|[o]penai_server|[r]un_chain.sh|[g]lm_[a-z0-9_]*chain.sh|[r]ocprofv3|[g]lm53' 2>/dev/null | grep -v 'tail ' | wc -l | tr -d ' ')"
echo "containers=\$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')"
m=0; for d in /sys/class/drm/card[0-9]/device; do u=\$(( \$(cat \$d/mem_info_vram_used 2>/dev/null || echo 0) / 1048576 )); [ \$u -gt \$m ] && m=\$u; done; echo "vram_max=\$m"
echo "installed=\$(sha256sum ~/bench/franken_bin/franken_dec_glm 2>/dev/null | cut -c1-16)"
echo "seq_leftovers=\$(find ~/bench -maxdepth 1 -name 'seq*.sh' -mmin -1440 2>/dev/null | wc -l | tr -d ' ')"
while read -r name sum; do
  [ -z "\$name" ] && continue
  [ -f ~/bench/\$name ] || continue
  [ "\$(sha256sum ~/bench/\$name | cut -c1-16)" = "\$sum" ] || echo "drift=\$name"
done <<'LISTEND'
$LIST
LISTEND
EOF
)
rv() { printf '%s\n' "$RIGOUT" | sed -n "s/^$1=//p" | head -1; }

# ---------------------------------------------------------------- repos
repo_check() {   # label dir branch rig_head_key rig_dirty_key
  local label=$1 dir=$2 br=$3 rk=$4 dk=$5
  [ -d "$dir/.git" ] || { fail "$label: no clone at $dir"; return; }
  local cur; cur=$(git -C "$dir" branch --show-current)
  [ "$cur" = "$br" ] || warn "$label: the Mac clone is on '$cur', not '$br' (the checks below use '$br')"
  local dirty; dirty=$(git -C "$dir" status --porcelain | wc -l | tr -d ' ')
  [ "$dirty" = 0 ] && ok "$label: Mac working tree clean" || fail "$label: $dirty uncommitted/untracked path(s) in the Mac clone ($(git -C "$dir" status --porcelain | head -3 | tr '\n' ';' | cut -c1-120))"
  [ "$MODE" = full ] && with_alarm 25 git -C "$dir" fetch -q origin 2>/dev/null
  if git -C "$dir" rev-parse -q --verify "origin/$br" >/dev/null; then
    local ahead behind
    ahead=$(git -C "$dir" rev-list --count "origin/$br..$br"); behind=$(git -C "$dir" rev-list --count "$br..origin/$br")
    [ "$ahead" = 0 ] && ok "$label: nothing unpushed to Gitea" || fail "$label: $ahead commit(s) on '$br' not pushed to Gitea (origin)"
    [ "$behind" = 0 ] || warn "$label: Gitea is $behind commit(s) ahead of the Mac clone"
  else
    warn "$label: no origin/$br ref to compare with"
  fi
  local mine rig; mine=$(git -C "$dir" rev-parse "$br" 2>/dev/null); rig=$(rv "$rk")
  if [ -z "$RIGOUT" ]; then warn "$label: the rig is unreachable, its copy was not compared"
  elif [ "$rig" = "$mine" ]; then ok "$label: the rig copy is at the same commit ($(echo "$mine" | cut -c1-8))"
  else fail "$label: the rig's '$br' is at $(echo "$rig" | cut -c1-8), the Mac's at $(echo "$mine" | cut -c1-8) (relay: handoff 06b section 2)"; fi
  local rd; rd=$(rv "$dk"); [ -z "$rd" ] || [ "$rd" = 0 ] || warn "$label: the rig's working tree has $rd changed path(s)"
}
repo_check "colibri"        "$C" "$CB" c_head c_dirty
repo_check "franken-engine" "$F" "$FB" f_head f_dirty
if [ "$MODE" = full ] && git -C "$C" remote get-url fork >/dev/null 2>&1; then
  fr=$(with_alarm 20 git -C "$C" ls-remote fork "refs/heads/$CB" 2>/dev/null | cut -f1)
  if [ -n "$fr" ]; then lag=$(git -C "$C" rev-list --count "$fr..$CB" 2>/dev/null || echo "?"); info "GitHub fork (PUBLIC) lags $lag commit(s): pushed only with the owner's go-ahead after a secret scan, never by a session on its own"; fi
fi
fw=$(rv f_worktrees); [ -z "$fw" ] || [ "$fw" = 1 ] || warn "the rig's franken-engine has $fw worktrees: per-item build trees are temporary, remove yours (git worktree remove --force)"

# ---------------------------------------------------------------- docs
dc=$(cd "$C" && bash tools/hot-expert/doc_currency.sh 2>&1); dcrc=$?
if [ $dcrc = 0 ]; then ok "doc_currency PASS"; else fail "doc_currency FAIL: $(printf '%s\n' "$dc" | grep -E 'DEAD|STALE|FAIL|MISSING|untracked|names' | head -3 | tr '\n' ';' | cut -c1-240)"; fi
inst=$(rv installed)
if [ -n "$inst" ]; then
  if grep -q "$inst" "$C/tools/hot-expert/HANDOFF-2026-10-06b.md"; then ok "the handoff names the installed serving binary ($inst)"
  else fail "the handoff does not name the installed franken_dec_glm (sha256 $inst): update its 'Installed' bullet and the plan"; fi
fi

# ---------------------------------------------------------------- rig state
lock=$(rv lock); procs=$(rv procs); cont=$(rv containers); vram=$(rv vram_max); flag=$(rv flag)
[ "$lock" != none ] && [ -n "$lock" ] && busy=1
[ "${procs:-0}" != 0 ] && [ -n "$procs" ] && busy=1
case "$cont" in *franken_engine*) busy=1;; esac
if [ -z "$RIGOUT" ]; then :
elif [ $busy = 1 ]; then warn "the rig is BUSY (lock: $lock, engine/chain processes: ${procs:-?}, containers: $cont): a rig job is running -- a watcher must be armed for it (CLAUDE.md: no rig job runs unwatched)"
else
  ok "the rig is idle (no lock, no engine, no chain)"
  [ "${vram:-0}" -gt 1024 ] 2>/dev/null && warn "the rig is idle but a card holds $vram MiB of VRAM"
fi
[ "$flag" = ABSENT ] && warn "the rig's reservation flag ~/bench/.dev_reserved is ABSENT: the watchdog may restart a service nobody asked for"
sl=$(rv seq_leftovers); [ -z "$sl" ] || [ "$sl" = 0 ] || warn "$sl sequencer script(s) ~/bench/seq*.sh from the last 24 h on the rig (recent one-offs: remove yours; older sessions' are listed in the handoff clean-up item)"
drift=$(printf '%s\n' "$RIGOUT" | sed -n 's/^drift=//p' | tr '\n' ' ')
[ -z "$drift" ] || warn "~/bench copies that DIFFER from the repo's ckpt1006/: $drift(a stale copy once pointed a chain at a removed worktree; scp the repo version before launching)"

# ---------------------------------------------------------------- config
SK=$C/tools/hot-expert/skills/task-closeout/SKILL.md; IS=$HOME/.claude/skills/task-closeout/SKILL.md
if [ -f "$SK" ]; then
  if [ -f "$IS" ] && cmp -s "$SK" "$IS"; then ok "the task-closeout skill is installed and equals the repo copy"
  else warn "the installed skill ($IS) is missing or differs from the repo copy ($SK): cp it"; fi
fi
grep -q closeout_stop_hook "$HOME/.claude/settings.json" 2>/dev/null && ok "the Stop hook is registered in ~/.claude/settings.json" || warn "the closeout Stop hook is NOT registered in ~/.claude/settings.json"
for idx in "$HOME"/.claude/projects/-Users-ronald-Projects*/memory/MEMORY.md; do
  [ -f "$idx" ] || continue
  d=$(dirname "$idx"); miss=""
  for f in $(grep -o '](\([^)]*\.md\))' "$idx" | sed 's/](\(.*\))/\1/'); do [ -f "$d/$f" ] || miss="$miss $f"; done
  [ -z "$miss" ] && ok "memory index $(basename "$(dirname "$d")"): every named file exists" || fail "memory index $idx names missing file(s):$miss"
done

[ "$MODE" = full ] && cat <<'REMIND'

For the model (not scriptable): the plan has a Rev entry for what changed (look up the top Rev first; CLAUDE.md and the handoff name it), the record has a section for it
(new sections at the END, numbers + which oracle passed), the handoff's state / open-task / recipe / trap lines are current (installed sha, speeds), the memory note says what a
future session needs and nothing the repo already records, scratch is cleaned (rig worktrees, sequencers, temp scripts), no watcher or rig job is left unreported, and the final
message says plainly what was verified and what was not, what needs the owner's decision (GitHub pushes, hardware), in few words and few numbers.
REMIND
echo "closeout: $fails FAIL, $warns WARN"
echo "rig_busy=$busy"
[ $fails = 0 ]
