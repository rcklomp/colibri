#!/bin/bash
# f3s1_kl.sh -- the KL half of F3 step 1's oracle, run OUTSIDE the rig lock
# on the dumps f3s1_kl_chain.sh left behind (leg 1: int8 container, tier
# OFF vs tier ON over dev2+dev3).
#
#   nohup nice -n 15 ~/src/colibri-f3s1/tools/hot-expert/f3s1_kl.sh \
#       >> ~/bench/f3s1_kl_compare.log 2>&1 &
#
# It needs no engine, no GPU and no gateway downtime: kl_compare.py is pure
# Python over two float32 dumps written by Q36_LOGIT_DUMP_ALL (the same
# "GLKD" container glm53's dumps use -- proved by V1 step 2, unchanged here).
#
# THE BAR (per the F3 step 1 brief, tighter than the generic project bar):
# tier-on vs tier-off over dev2+dev3 is a REASSOCIATION of the SAME int8
# products -- the same row-wise int8 weights and per-row scales applied on
# both sides, summed in a different order (AVX2 lanes on the CPU,
# subgroupAdd across lanes on the GPU), exactly the argument V1 made for its
# own int4/gs64 tier-on vs tier-off pair (mean KL -1.20e-10 / 2.71e-09,
# top-1 100.00% both prompts, record V1). So the bar here is V1's own floor,
# not the generic "strictly better than the mildest recorded defect" bar
# v1_step2a_kl.sh and f3s0_kl.sh use for a genuinely different quantization
# or an unproven kernel change:
#
#   PASS  mean KL < 1e-6  AND  top-1 == 100.00%
#
# For scale (record ROME-3x7900XTX-2026-09-04.md, §X2 and §V1):
#   V1's own tier-on/off floor              mean KL ~1e-10        top-1 100.00%
#   accepted as "same" at depth (F2 clamp)  mean KL 3.5e-05       top-1 99.89%
#   REJECTED as a defect (mildest, clamp)   mean KL 0.0284        top-1 92.73%
set -u
SRC=~/src/colibri-f3s1
HERE="$SRC/tools/hot-expert"
OUT=${1:-~/bench/f3s1_kl_out}
OUT=$(eval echo "$OUT")
. "$HERE/gate_lib.sh"

echo "=== f3s1_kl $(date -Is)  dumps in $OUT"
echo "=== bar: mean KL < 1e-6 and top-1 == 100.00% (V1's own tier-on/off floor, record V1)"
echo

RC=0
kl_leg() {                # kl_leg <label> <ref.dump> <cand.dump>
  local label="$1" ref="$2" cand="$3" out
  out=$(python3 "$HERE/kl_compare.py" "$label" "$ref" "$cand" 2>&1); local krc=$?
  echo "$out"
  if [ $krc -ne 0 ]; then echo "     VERDICT: REFUSED"; RC=1; return; fi
  local mk t1
  mk=$(echo "$out" | sed -nE 's/.*mean KL\(ref\|\|cand\) *: *([0-9.eE+-]+).*/\1/p')
  t1=$(echo "$out" | sed -nE 's/.*top-1 agreement *: *([0-9.]+)%.*/\1/p')
  if python3 -c "import sys; sys.exit(0 if abs(float('$mk'))<1e-6 and float('$t1')>=100.0 else 1)"; then
    echo "     VERDICT: PASS  (mean KL $mk, |.|<1e-6; top-1 $t1% == 100.00%)"
  else
    echo "     VERDICT: FAIL  (mean KL $mk, top-1 $t1% -- against |.|<1e-6 and ==100.00%)"
    RC=1
  fi
  echo
}

kl_leg "F3s1 tier-on vs tier-off, int8, short prompt"  "$OUT/i8_off_sh.dumpall" "$OUT/i8_on_sh.dumpall"
kl_leg "F3s1 tier-on vs tier-off, int8, ladder packet" "$OUT/i8_off_pk.dumpall" "$OUT/i8_on_pk.dumpall"

echo "=== f3s1_kl rc=$RC $(date -Is)"
exit $RC
