#!/bin/bash
# glm_pf2c_seq.sh -- PF2c (plan Rev 116): the three rig chains that follow the speed A/B, one after the other, each under run_chain.sh (rig lock, retried when refused):
#   1. glm_pf2c_trace_chain   rocprofv3 kernel trace of the candidate (--gemm-lds-min-rows 16 --gemm-lds-splitk 1): what the trunk is made of NOW
#   2. glm_pf2c_kl_chain      next-token KL of the candidate against the bit-exact arm and the shipped f16 arm (24 windows)
#   3. glm_all_gate_chain     the standard IQ4_XS `all` gate on the PF2c build (default arms must stay exact: required before the merge)
# Launch:  setsid nohup bash ~/bench/glm_pf2c_seq.sh > ~/bench/glm_pf2c_seq.log 2>&1 < /dev/null &
H=$HOME/src/colibri/tools/hot-expert; B=$HOME/bench; D=$HOME/src/franken-engine/franken/decode
[ -x "$D/franken_decode_glm_pf2c" ] || { echo "[seq] no $D/franken_decode_glm_pf2c"; exit 2; }
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
run glm_pf2c_trace_sk glm_pf2c_trace_chain PF2C_BIN=$D/franken_decode_glm_pf2c PF2C_EXTRA="--gemm-lds-min-rows 16 --gemm-lds-splitk 1" PF2C_OUT=$B/franken/glm5/pf2c_trace_sk
run glm_pf2c_kl_chain glm_pf2c_kl_chain A=1
run glm_pf2c_all_gate glm_all_gate_chain AG_BIN=franken_decode_glm_pf2c AG_TAG=pf2c
echo "[seq] done $(date -Is)"
