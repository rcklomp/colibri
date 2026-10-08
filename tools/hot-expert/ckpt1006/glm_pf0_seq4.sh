#!/bin/bash
# glm_pf0_seq4.sh -- waits for glm_pf0_seq3.sh, then repeats the skip-class study on the RECORD text (a technical document: concentrated routing, 12 % slower than the prose prompt).
H=$HOME/src/colibri/tools/hot-expert
until grep -q '^\[seq3\] .*finished' "$HOME/bench/glm_pf0_seq3.log" 2>/dev/null; do sleep 30; done
c=glm_pf0_skip_chain
for try in 1 2 3 4 5 6; do
  "$H/preflight.sh" && PF0_PROMPT=$HOME/bench/franken/glm5/rec_depth/rec_ids.txt PF0_MASKS="0 31 29 30 1" PF0_ROUTES=0 PF0_OUT=$HOME/bench/franken/glm5/pf0_skip_rec \
     "$H/run_chain.sh" "$HOME/bench/$c.sh" > "$HOME/bench/${c}_rec.log" 2>&1 < /dev/null
  rc=$?
  [ "$rc" -ne 3 ] && break
  echo "[seq4] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
done
echo "[seq4] $c finished rc=$rc $(date -Is)"
