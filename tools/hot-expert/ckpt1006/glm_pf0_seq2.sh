#!/bin/bash
# glm_pf0_seq2.sh -- waits for glm_pf0_seq.sh to finish, then runs the SDMA variant chain under run_chain.sh (same retry rule on rc 3).
H=$HOME/src/colibri/tools/hot-expert
until grep -q '^\[seq\] done' "$HOME/bench/glm_pf0_seq.log" 2>/dev/null; do sleep 30; done
c=glm_pf0_sdma_chain
for try in 1 2 3 4 5 6; do
  "$H/preflight.sh" && "$H/run_chain.sh" "$HOME/bench/$c.sh" > "$HOME/bench/$c.log" 2>&1 < /dev/null
  rc=$?
  [ "$rc" -ne 3 ] && break
  echo "[seq2] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
done
echo "[seq2] $c finished rc=$rc $(date -Is)"
