#!/bin/bash
# f3s2a_kl.sh -- the KL half of F3 step 2a's oracle, run OUTSIDE the rig lock
# on the dumps f3s2a_kl_chain.sh left behind (leg 2: Q36_VK_TRUNK=0 vs =1,
# int8 container, tier on dev2+dev3 in both arms).
#
#   nohup nice -n 15 ~/src/colibri-f3s2a/tools/hot-expert/f3s2a_kl.sh \
#       >> ~/bench/f3s2a_kl_compare.log 2>&1 &
#
# It needs no engine, no GPU and no gateway downtime: kl_compare.py is pure
# Python over two float32 dumps written by Q36_LOGIT_DUMP_ALL.
#
# THE BAR (per the F3 step 2a brief, the SAME argument F3 step 1 used for its
# own tier-on/off pair, record §F3-STEP1): trunk-on vs trunk-off is a
# REASSOCIATION of the SAME row-wise int8 products qwen36.c's dense-i8
# quantization already computes for the CPU path (per-row scale, y[o] =
# acc*sc[o]) -- AVX2-lane summation vs the GPU's subgroup reduction, not a
# different quantization or a different weight. So this item's bar is F3
# step 1's own floor, tightened from the caller's brief:
#
#   PASS  top-1 >= 99.0%  AND  mean KL < 1e-4
#
# For scale (record ROME-3x7900XTX-2026-09-04.md, §F3-STEP1 and §V1):
#   F3 step 1's own tier-on/off floor           mean KL ~1e-10       top-1 100.00%
#   accepted as "same" at depth (F2 clamp)      mean KL 3.5e-05      top-1 99.89%
#   REJECTED as a defect (mildest, clamp)       mean KL 0.0284       top-1 92.73%
# If the measured numbers land close to the tier's own ~1e-10 floor rather
# than merely under 1e-4, say so plainly -- the tighter number is the more
# informative one to report.
set -u
SRC=~/src/colibri-f3s2a
HERE="$SRC/tools/hot-expert"
OUT=${1:-~/bench/f3s2a_kl_out}
OUT=$(eval echo "$OUT")
. "$HERE/gate_lib.sh"

echo "=== f3s2a_kl $(date -Is)  dumps in $OUT"
echo "=== bar: mean KL < 1e-4 and top-1 >= 99.0% (F3 step 1's own tier-on/off argument)"
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
  if python3 -c "import sys; sys.exit(0 if abs(float('$mk'))<1e-4 and float('$t1')>=99.0 else 1)"; then
    echo "     VERDICT: PASS  (mean KL $mk, |.|<1e-4; top-1 $t1% >= 99.0%)"
  else
    echo "     VERDICT: FAIL  (mean KL $mk, top-1 $t1% -- against |.|<1e-4 and >=99.0%)"
    RC=1
  fi
  echo
}
flip_leg() {               # flip_leg <label> <ref=trunk-off.dump> <cand=trunk-on.dump>
  local label="$1" ref="$2" cand="$3"
  echo "--- flip-margin, $label (ref=trunk off/CPU, cand=trunk on/dev0; how close were"
  echo "    the flipped positions -- CLAUDE.md's F7 rule: a kernel that reorders a"
  echo "    float sum is judged by whether its flips are near-ties, not by a bit-exact"
  echo "    bar)"
  python3 "$HERE/flip_margin.py" "$ref" "$cand" 2>&1 | sed 's/^/    /'
  echo
}

kl_leg   "F3s2a trunk-on vs trunk-off, int8, short prompt"  "$OUT/trunk_off_sh.dumpall" "$OUT/trunk_on_sh.dumpall"
flip_leg "short prompt"                                     "$OUT/trunk_off_sh.dumpall" "$OUT/trunk_on_sh.dumpall"
kl_leg   "F3s2a trunk-on vs trunk-off, int8, ladder packet" "$OUT/trunk_off_pk.dumpall" "$OUT/trunk_on_pk.dumpall"
flip_leg "ladder packet"                                    "$OUT/trunk_off_pk.dumpall" "$OUT/trunk_on_pk.dumpall"

echo "=== f3s2a_kl rc=$RC $(date -Is)"
exit $RC
