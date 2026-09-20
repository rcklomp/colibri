#!/bin/bash
# f3s0_kl.sh -- the KL half of F3 step 0, run OUTSIDE the rig lock on the
# dumps f3s0_kl_chain.sh left behind.
#
#   nohup nice -n 15 ~/src/colibri-f3s0/tools/hot-expert/f3s0_kl.sh \
#       >> ~/bench/f3s0_kl_compare.log 2>&1 &
#
# It needs no engine, no GPU and no gateway downtime: kl_compare.py and
# flip_margin.py are pure Python over two float32 dumps written by
# Q36_LOGIT_DUMP_ALL (the same "GLKD" container glm53's dumps use, so both
# scripts read a qwen36 dump with no change -- proved by V1 step 2).
#
# Direction: ref = int8 (row-wise, higher precision), cand = int4-gs64 (the
# container V1 already serves). This is "what does int4 cost against int8",
# the question F3 step 0 exists to answer -- NOT "which container is more
# correct against some third ground truth" (there is none here; both are
# quantizations of the same bf16 weights, and the bf16 originals are not
# on this box).
#
# THIS IS NOT A PASS/FAIL GATE. Unlike v1_step2a_kl.sh (which had a bar
# because tier-on vs tier-off is supposed to be the SAME model, reassociated),
# int4 and int8 are DIFFERENT quantizations and some divergence is expected
# and correct. The bars below are printed for orientation only, taken from
# this track's own accepted/rejected numbers (record ROME-3x7900XTX-2026-09-04.md
# X2 and V1 sections):
#
#   "the same model, reassociated" floor    mean KL ~1e-10        top-1 100.00%   (V1's own tier-on/off)
#   accepted as "same" at depth             mean KL 3.5e-05       top-1 99.89%    (F2's clamp KL)
#   REJECTED as a defect (mildest)          mean KL 0.0284        top-1 92.73%    (the swiglu clamp gap)
#   REJECTED as a defect (int8 kernel)      mean KL 0.0296        top-1 93.44%    (G14)
#   REJECTED as a defect (int3 experts)     mean KL 0.251         top-1 78.01%    (G15)
#
# Landing near or above the "mildest rejected defect" row means int4 is
# costing real quality here and int8 is worth building. Landing near the
# "accepted as same" row means int4 already captures what int8 would, and
# the 70 GB BF16 download + new int8 tier kernel (V1-STEP0's fmt=1 path,
# unused so far) is not worth it on quality grounds alone.
set -u
SRC=~/src/colibri-f3s0
HERE="$SRC/tools/hot-expert"
OUT=${1:-~/bench/f3s0_kl_out}
OUT=$(eval echo "$OUT")
. "$HERE/gate_lib.sh"

echo "=== f3s0_kl $(date -Is)  dumps in $OUT"
echo "=== orientation only, no PASS/FAIL: see this file's header for the reference rows"
echo

kl_leg() {                # kl_leg <label> <ref=i8.dump> <cand=i4.dump>
  local label="$1" ref="$2" cand="$3"
  python3 "$HERE/kl_compare.py" "$label" "$ref" "$cand"
  echo
}

flip_leg() {               # flip_leg <label> <ref=i8.dump> <cand=i4.dump>
  local label="$1" ref="$2" cand="$3"
  echo "--- flip-margin, $label (ref=int8, cand=int4; how decided was int8 on the"
  echo "    positions where int4's argmax differs?)"
  python3 "$HERE/flip_margin.py" "$ref" "$cand" 2>&1 | sed 's/^/    /'
  echo
}

echo "########## SHORT PROMPT (22-ish positions) ##########"
kl_leg   "int4 vs int8, short prompt"  "$OUT/sh_i8.dumpall"  "$OUT/sh_i4.dumpall"
flip_leg "short prompt"                "$OUT/sh_i8.dumpall"  "$OUT/sh_i4.dumpall"

echo "########## LADDER PACKET (625 positions) ##########"
kl_leg   "int4 vs int8, ladder packet" "$OUT/pk_i8.dumpall"  "$OUT/pk_i4.dumpall"
flip_leg "ladder packet"               "$OUT/pk_i8.dumpall"  "$OUT/pk_i4.dumpall"

echo "=== f3s0_kl done $(date -Is)"
echo "=== next: fold these numbers into F3-STEP0-2026-09-20.md's decision line and"
echo "===       into the plan/record if this becomes a landed step"
