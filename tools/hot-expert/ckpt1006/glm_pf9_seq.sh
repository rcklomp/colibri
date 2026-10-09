#!/bin/bash
# glm_pf9_seq.sh -- PF9 (plan Rev 102): the three rig chains one after the other, each under run_chain.sh (rig lock, retried when the lock is refused):
#   1. glm_pf9_main_chain    timeline of a 32-chunk prompt, profile, depth curve        (binary franken_decode_glm_pf9main)
#   2. glm_pf0_skip_chain    skip classes with --glm-help-copy 1 on the prose prompt     (binary franken_decode_glm_pf0dbg, rebuilt from main)
#   3. glm_pf0_skip_chain    the same on the technical text (the record)
# ERRATUM: chains 2 and 3 used a binary whose expert skip (bit 1) did nothing in the chunk path (see glm_pf9b_seq.sh); only chain 1 (no masks) and chain 2's non-expert classes stand. Chain 3 was killed.
# Waits for both builds (build_pf0dbg.sh, then PF0_PLAIN=1) to have finished.  Launch:  setsid nohup bash ~/bench/glm_pf9_seq.sh > ~/bench/glm_pf9_seq.log 2>&1 < /dev/null &
H=$HOME/src/colibri/tools/hot-expert; B=$HOME/bench
until grep -q '^done' "$B/build_pf9_plain.log" 2>/dev/null; do sleep 20; done
[ -x "$B/franken_bin/franken_decode_glm_pf9main" ] && [ -x "$B/franken_bin/franken_decode_glm_pf0dbg" ] || { echo "[seq] a binary is missing"; exit 2; }
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
run glm_pf9_main_chain glm_pf9_main_chain A=1
run glm_pf9_skip_prose glm_pf0_skip_chain PF0_EXTRA="--glm-help-copy 1" PF0_MASKS="0 31 29 30 1 3" PF0_ROUTES=0 PF0_OUT=$B/franken/glm5/pf9_skip
run glm_pf9_skip_rec glm_pf0_skip_chain PF0_EXTRA="--glm-help-copy 1" PF0_MASKS="0 31 29 30 1 3" PF0_ROUTES=0 PF0_PROMPT=$B/franken/glm5/rec_depth/rec_ids.txt PF0_OUT=$B/franken/glm5/pf9_skip_rec
echo "[seq] done $(date -Is)"
