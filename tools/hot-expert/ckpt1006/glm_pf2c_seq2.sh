#!/bin/bash
# glm_pf2c_seq2.sh -- PF2c (plan Rev 116): the merge gates on the FINAL build, one after the other:
#   1. glm_all_gate_chain     the standard IQ4_XS `all` gate (default arms must stay exact), under run_chain.sh
#   2. glm_pf2c_quality_chain MMLU-Pro 70 + needle through the real gateway on the PF2c serving build, started directly (serve_alt.sh takes the rig lock), thermal_guard.sh beside it
# Launch:  setsid nohup bash ~/bench/glm_pf2c_seq2.sh > ~/bench/glm_pf2c_seq2.log 2>&1 < /dev/null &
H=$HOME/src/colibri/tools/hot-expert; B=$HOME/bench; D=$HOME/src/franken-engine/franken/decode
[ -x "$D/franken_decode_glm_pf2c" ] && [ -x "$B/franken_bin/franken_dec_glm.pf2c" ] || { echo "[seq2] missing a PF2c binary"; exit 2; }
for try in 1 2 3 4 5 6; do
  "$H/preflight.sh" && env AG_BIN=franken_decode_glm_pf2c AG_TAG=pf2c "$H/run_chain.sh" "$B/glm_all_gate_chain.sh" > "$B/glm_pf2c_all_gate.log" 2>&1 < /dev/null
  rc=$?; [ "$rc" -ne 3 ] && break
  echo "[seq2] all gate refused (rc 3), retry $try $(date -Is)"; sleep 300
done
echo "[seq2] all gate finished rc=$rc $(date -Is)"
[ "$rc" -ne 0 ] && { echo "[seq2] the all gate did not pass: no quality run"; exit "$rc"; }
setsid nohup "$B/glm_pf2c_quality_chain.sh" > "$B/glm_pf2c_quality_chain.log" 2>&1 < /dev/null &
QP=$!
sleep 120
GUARD_WATCH="glm_pf2c_quality_chai[n]" setsid nohup "$B/thermal_guard.sh" > "$B/thermal_guard.log" 2>&1 < /dev/null &
echo "[seq2] quality chain pid $QP, guard started $(date -Is)"
wait "$QP"; echo "[seq2] quality chain finished rc=$? $(date -Is)"
echo "[seq2] done $(date -Is)"
