#!/bin/bash
# glm_pf9b_seq.sh -- PF9 ERRATUM re-run (2026-10-09). The skip chains of glm_pf9_seq.sh (and PF0's glm_pf0_skip_chain / glm_pf0_sktl_chain) ran a binary whose --debug-skip bit 1 (skip the routed-expert
# kernels) was a NO-OP in the chunk path with FRANKEN_GLM_MOE_G >= 2: the check sat after the `if (cp && mg > 1) { k_glm5_moe_blk ...; return; }` branch of moe_launch. build_pf0dbg.sh now puts it first.
# This sequencer re-runs the skip classes with the fixed binary, with a POSITIVE CONTROL (PF0_CTRL=1: a masked --profile run whose expert classes must read ~0):
#   1. glm_pf0_skip_chain on the prose prompt, --glm-help-copy 1       -> ~/bench/franken/glm5/pf9b_skip
#   2. the same on the technical text (the record)                      -> ~/bench/franken/glm5/pf9b_skip_rec
# Launch:  setsid nohup bash ~/bench/glm_pf9b_seq.sh > ~/bench/glm_pf9b_seq.log 2>&1 < /dev/null &
H=$HOME/src/colibri/tools/hot-expert; B=$HOME/bench
until grep -qE '^done|PATCH FAILED' "$B/build_pf9_dbg2.log" 2>/dev/null; do sleep 15; done
grep -q 'PATCH FAILED' "$B/build_pf9_dbg2.log" && { echo "[seq] the patch failed"; exit 2; }
run() {   # run <tag> <chain> [VAR=val ...]
  local tag=$1 c=$2; shift 2
  for try in 1 2 3 4 5 6; do
    "$H/preflight.sh" && env "$@" "$H/run_chain.sh" "$B/$c.sh" > "$B/${tag}.log" 2>&1 < /dev/null
    rc=$?
    [ "$rc" -ne 3 ] && break
    echo "[seq] $c refused (rc 3), retry $try $(date -Is)"; sleep 300
  done
  echo "[seq] $tag finished rc=$rc $(date -Is)"
}
run glm_pf9b_skip_prose glm_pf0_skip_chain PF0_EXTRA="--glm-help-copy 1" PF0_MASKS="0 31 29 30 1 3" PF0_ROUTES=0 PF0_CTRL=1 PF0_OUT=$B/franken/glm5/pf9b_skip
run glm_pf9b_skip_rec glm_pf0_skip_chain PF0_EXTRA="--glm-help-copy 1" PF0_MASKS="0 31 29 30 1 3" PF0_ROUTES=0 PF0_CTRL=1 PF0_PROMPT=$B/franken/glm5/rec_depth/rec_ids.txt PF0_OUT=$B/franken/glm5/pf9b_skip_rec
echo "[seq] done $(date -Is)"
