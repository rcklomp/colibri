#!/bin/bash
# rig_lock.sh — one lock, two purposes: only one chain may run, and the gateway watchdog must
# not fight a chain that stopped the gateway on purpose.
#
# Written 2026-09-09 after a leftover launcher fired a second chain the moment the owner's
# laptop rejoined the network, while another was mid-gate. Nothing was damaged that time; the
# lock is so there is no next time.
#
#   . rig_lock.sh            # source it, then:
#   rig_lock_take <name>     # exclusive; refuses (exit 3) if another holder is alive
#   rig_lock_release         # idempotent, safe in a trap
#   rig_lock_holder          # prints "<name> <pid> <started>" or nothing
#   rig_lock_maintenance     # 0 while a chain holds the lock (the watchdog reads this)
#
# The lock is a directory (atomic create) holding a pid file. A holder whose pid is gone is
# stale and is cleared automatically — a chain killed by a dying ssh session must not wedge
# the rig until someone notices.
RIG_LOCK_DIR="${RIG_LOCK_DIR:-$HOME/bench/.rig.lock}"

rig_lock_holder() {
  [ -d "$RIG_LOCK_DIR" ] || return 1
  local pid name started
  read -r name pid started < "$RIG_LOCK_DIR/owner" 2>/dev/null || return 1
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$name $pid $started"; return 0; fi
  return 1
}

rig_lock_take() {
  local name="${1:?a name for the lock holder}"
  for _ in $(seq 1 3); do
    if mkdir "$RIG_LOCK_DIR" 2>/dev/null; then
      echo "$name $$ $(date +%Y-%m-%dT%H:%M:%S%z)" > "$RIG_LOCK_DIR/owner"
      RIG_LOCK_HELD=1
      return 0
    fi
    if ! rig_lock_holder >/dev/null; then
      echo "rig_lock: clearing a stale lock ($(cat "$RIG_LOCK_DIR/owner" 2>/dev/null))"
      rm -rf "$RIG_LOCK_DIR"; continue
    fi
    break
  done
  echo "rig_lock: REFUSED, held by $(rig_lock_holder)"
  return 3
}

rig_lock_release() {
  [ "${RIG_LOCK_HELD:-0}" = 1 ] || return 0
  rm -rf "$RIG_LOCK_DIR"; RIG_LOCK_HELD=0
}

# 0 = a chain holds the lock, so the gateway may legitimately be down.
rig_lock_maintenance() { rig_lock_holder >/dev/null; }
