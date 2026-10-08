#!/bin/bash
# glm_pf0_seq3.sh -- waits for glm_pf0_seq2.sh, then runs the skeleton-timeline chain under run_chain.sh.
H=$HOME/src/colibri/tools/hot-expert
until grep -q '^\[seq2\] .*finished' "$HOME/bench/glm_pf0_seq2.log" 2>/dev/null; do sleep 30; done
c=glm_pf0_sktl_chain
for try in 1 2 3 4 5 6; do
  "$H/preflight.sh" && "$H/run_chain.sh" "$HOME/bench/$c.sh" > "$HOME/bench/$c.log" 2>&1 < /dev/null
  rc=$?
  [ "$rc" -ne 3 ] && break
  echo "[seq3] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
done
echo "[seq3] $c finished rc=$rc $(date -Is)"
