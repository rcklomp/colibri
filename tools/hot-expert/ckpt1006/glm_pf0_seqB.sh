#!/bin/bash
# glm_pf0_seqB.sh -- replaces glm_pf0_seq3/seq4: after the SDMA variant chain (seq2) it runs, in this order, the help-copy chain, the skeleton-timeline chain and the record-text skip-class chain.
H=$HOME/src/colibri/tools/hot-expert
until grep -q '^\[seq2\] .*finished' "$HOME/bench/glm_pf0_seq2.log" 2>/dev/null; do sleep 30; done
run() {   # run <tag> <chain> [VAR=val ...]
  local tag=$1 c=$2; shift 2
  for try in 1 2 3 4 5 6; do
    "$H/preflight.sh" && env "$@" "$H/run_chain.sh" "$HOME/bench/$c.sh" > "$HOME/bench/${tag}.log" 2>&1 < /dev/null
    rc=$?
    [ "$rc" -ne 3 ] && break
    echo "[seqB] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
  done
  echo "[seqB] $tag finished rc=$rc $(date -Is)"
}
run glm_pf0_hc_chain glm_pf0_hc_chain A=1
run glm_pf0_sktl_chain glm_pf0_sktl_chain A=1
run glm_pf0_skip_chain_rec glm_pf0_skip_chain PF0_PROMPT=$HOME/bench/franken/glm5/rec_depth/rec_ids.txt PF0_MASKS="0 31 29 30 1" PF0_ROUTES=0 PF0_OUT=$HOME/bench/franken/glm5/pf0_skip_rec
echo "[seqB] done $(date -Is)"
