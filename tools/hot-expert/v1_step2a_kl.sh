#!/bin/bash
# v1_step2a_kl.sh -- the KL half of V1 step 2a's oracle, run OUTSIDE the rig
# lock on the dumps v1_step2a_chain.sh left behind.
#
#   nohup nice -n 15 ~/src/colibri-v1/tools/hot-expert/v1_step2a_kl.sh \
#       >> ~/bench/v1_step2a_kl.log 2>&1 &
#
# It needs no engine, no GPU and no gateway downtime: kl_compare.py is pure
# Python over two float32 dumps. On the 625-position packet at vocab 248 320
# that is ~15 minutes a pair, which is exactly why it does not run inside the
# chain -- fifteen minutes of the owner's service held down for arithmetic
# that could be done any time.
#
# THE BAR, AND WHERE IT COMES FROM (record §X2's rows are the scale):
#
#   G15, int3 experts      REJECTED   mean KL 0.251   top-1 78.01%
#   G14, int8 kernel       REJECTED   mean KL 0.0296  top-1 93.44%
#   the swiglu clamp       a known, unfixed correctness GAP
#                                     mean KL 0.0284  top-1 92.73%
#
# V1's tier-on path is a REASSOCIATION of the same products: the same packed
# int4 weights (proved bit-exact through the staging XOR by
# tests/test_qwen36_vk_nibble), the same gs64 group scales applied per group
# on both sides, summed in a different order (AVX2 lanes per group on the CPU,
# subgroupAdd across lanes on the GPU) -- plus one real ordering change the
# engine makes on its own: with the tier on, qwen36.c's moe() computes the
# SHARED expert per token inside the routed loop so it overlaps the GPU group,
# instead of once per chunk in qwen_shared_experts_cpu(). So the accumulation
# order into out[] differs as well.
#
# Expected, therefore: KL orders of magnitude below any of the three rows
# above. The bar applied here is the loosest one that is still defensible --
#
#   PASS  mean KL < 0.0284 AND top-1 >= 99.0%
#
# i.e. strictly better than the MILDEST thing this project has recorded as a
# defect. Landing anywhere near 0.028 would NOT be a pass in spirit and the
# report says so: for a pure reassociation that would mean something else is
# different, and the right response is to find it, not to widen the bar.
set -u
SRC=~/src/colibri-v1
HERE="$SRC/tools/hot-expert"
OUT=${1:-~/bench/v1_step2a_out}
OUT=$(eval echo "$OUT")
. "$HERE/gate_lib.sh"

echo "=== v1_step2a_kl $(date -Is)  dumps in $OUT"
echo "=== bar: mean KL < 0.0284 and top-1 >= 99.0%  (X2: G14 rejected at 0.0296/93.44%,"
echo "===      the swiglu clamp gap at 0.0284/92.73%, G15 rejected at 0.251/78.01%)"
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
  if python3 -c "import sys; sys.exit(0 if float('$mk')<0.0284 and float('$t1')>=99.0 else 1)"; then
    echo "     VERDICT: PASS  (mean KL $mk < 0.0284, top-1 $t1% >= 99.0%)"
  else
    echo "     VERDICT: FAIL  (mean KL $mk, top-1 $t1% -- against < 0.0284 and >= 99.0%)"
    RC=1
  fi
  echo
}

# short first: 22 positions, seconds, and it says whether the packet run is
# worth the quarter of an hour.
kl_leg "V1 tier-on vs tier-off, short prompt"  "$OUT/off_sh.dumpall" "$OUT/on_sh.dumpall"
kl_leg "V1 tier-on vs tier-off, ladder packet" "$OUT/off_pk.dumpall" "$OUT/on_pk.dumpall"

echo "=== v1_step2a_kl rc=$RC $(date -Is)"
exit $RC
