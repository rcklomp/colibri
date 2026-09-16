#!/bin/bash
# franken_full_launch.sh -- arms the full (non-smoke, both arms) H2 run for a
# fixed wall-clock start time, then retries run_chain.sh until it actually
# acquires the rig lock, not one-shot.
#
# run_chain.sh exits 3 when the lock is held (rig_lock.sh); a one-shot
# `sleep $SECS; run_chain.sh ...` launched at 21:00 UTC loses the whole run
# to a single REFUSED if anything else holds the lock at that exact instant.
# This retries every 5 min for up to 2 h past the target start, logs every
# refusal with a timestamp, and gives up with a loud line if the window
# closes without ever acquiring it -- coordinator amendment, 2026-09-16.
#
# Usage: franken_full_launch.sh <target-epoch-seconds> [chain-args...]
#   franken_full_launch.sh $(date -d '21:00 today' +%s)
# Launch itself under setsid nohup so it survives the ssh session:
#   setsid nohup ~/src/colibri-h2/tools/hot-expert/franken_full_launch.sh \
#       <target-epoch> >> ~/bench/franken_chain.log 2>&1 < /dev/null &
# and record $! -- that pid is what "is it armed" is checked against, not
# the eventual franken_chain.sh pid (which does not exist until the lock is
# actually acquired, possibly hours later).
set -u
TARGET=${1:?target start time, epoch seconds (e.g. $(date -d '21:00 today' +%s))}
shift || true
CHAIN_ARGS=("$@")
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_CHAIN="$HERE/run_chain.sh"
CHAIN="$HERE/franken_chain.sh"
GIVE_UP_AFTER=7200   # 2 h past TARGET, coordinator's own bound

echo "=== franken_full_launch armed pid=$$ $(date -Is)"
echo "=== target start: $(date -Is -d "@$TARGET") (epoch $TARGET)"
echo "=== give-up time: $(date -Is -d "@$((TARGET + GIVE_UP_AFTER))") (target + ${GIVE_UP_AFTER}s)"
echo "=== chain args: ${CHAIN_ARGS[*]:-(none -- full run, both arms)}"

now=$(date +%s)
if [ "$now" -lt "$TARGET" ]; then
  wait_s=$((TARGET - now))
  echo "=== sleeping ${wait_s}s until target start"
  sleep "$wait_s"
fi

deadline=$((TARGET + GIVE_UP_AFTER))
attempt=0
rc=3
while [ "$(date +%s)" -lt "$deadline" ]; do
  attempt=$((attempt + 1))
  echo "=== attempt $attempt $(date -Is): launching run_chain.sh franken_chain.sh ${CHAIN_ARGS[*]:-}"
  "$RUN_CHAIN" "$CHAIN" "${CHAIN_ARGS[@]}"
  rc=$?
  echo "=== attempt $attempt $(date -Is): run_chain.sh exit=$rc"
  if [ "$rc" -ne 3 ]; then
    echo "=== franken_full_launch: chain actually ran (rc=$rc, not a lock refusal) -- done"
    exit "$rc"
  fi
  echo "=== attempt $attempt $(date -Is): REFUSED (lock held) -- retrying in 300s"
  sleep 300
done

echo "=== franken_full_launch: GIVE UP -- deadline $(date -Is -d "@$deadline") passed after $attempt attempts, lock never freed. THE FULL H2 RUN DID NOT HAPPEN TONIGHT."
exit 3
