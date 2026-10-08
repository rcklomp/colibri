#!/bin/bash
# glm_pf0_seq.sh -- runs the two PF0 chains one after the other, each under run_chain.sh (rig lock), retrying when the rig is busy (rc 3: lock held / preflight refused).
# Reports every end state on its own log line; a chain that fails for another reason does not stop the next one (separate binaries).
H=$HOME/src/colibri/tools/hot-expert
for c in glm_pf0_main_chain glm_pf0_skip_chain; do
  for try in 1 2 3 4 5 6; do
    "$H/preflight.sh" && "$H/run_chain.sh" "$HOME/bench/$c.sh" > "$HOME/bench/$c.log" 2>&1 < /dev/null
    rc=$?
    [ "$rc" -ne 3 ] && break
    echo "[seq] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
  done
  echo "[seq] $c finished rc=$rc $(date -Is)"
done
echo "[seq] done $(date -Is)"
