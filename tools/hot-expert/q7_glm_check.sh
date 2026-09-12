#!/bin/bash
# q7_glm_check.sh -- roadmap item Q7-gpu: the OTHER engine, because Q7 touches
# two shared files.
#
# Q7 adds fmt=9 to c/shaders/qmatmul.comp and three fmt=9 cases to
# c/backend_vulkan.c (rowwords, scale_floats, upload_tensor). glm53 links that
# object and loads that shader, so CLAUDE.md's rule applies without exception:
# "A shared-file change must rebuild and re-measure both engines." The spec says
# the same, per step and not per item. Nothing in glm53 asks for fmt=9 and no
# existing branch of either file was edited, so the expected result is IDENTICAL
# output -- which is exactly why it is checked rather than asserted.
#
# Unlike qp_glm_check.sh this builds nothing: both binaries already exist, the
# pristine as ~/bench/glm53.q7base (+ ~/bench/shaders_q7base) and the candidate
# in the Q7 worktree. Each side runs with ITS OWN shaders, because "the fmt=9
# addition is invisible to glm53" is a claim about the .spv as much as the .o.
#
# Usage (on the rig):  q7_glm_check.sh
set -u
. /home/ronald/bench/q7_lib.sh

SRC=~/src/colibri
M=~/models/GLM-5.3-Flash-colibri-int4-g64
OUT=~/bench/q7_glm_out; mkdir -p "$OUT"
PRIS=~/bench/glm53.q7base
PRIS_SH=~/bench/shaders_q7base
CAND="$WT/c/glm53"
CAND_SH="$WT/c/shaders"

trap q7_on_exit EXIT INT TERM HUP
q7_take_lock "q7-glm" 120 || exit 3
pgrep -x qwen38-vk >/dev/null && { log "REFUSED: qwen38-vk running"; exit 1; }
stop_gateway || exit 1
q7_build_engines || { log "REFUSED: build failed"; exit 1; }

[ -x "$PRIS" ] || { log "REFUSED: no pristine glm53 at $PRIS"; exit 1; }
log "pristine  $(sha256sum "$PRIS" | cut -c1-16)  shaders $(sha256sum "$PRIS_SH/qmatmul.spv" | cut -c1-16)"
log "candidate $(sha256sum "$CAND" | cut -c1-16)  shaders $(sha256sum "$CAND_SH/qmatmul.spv" | cut -c1-16)"

python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(Qwen) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/Qwen3.8-Flash-Next-FP8")))
PY
log "warming GLM"
find "$M" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
free -g

P_SHORT=$(cat ~/bench/prompt_glm.txt)

glmrun(){   # glmrun <tag> <binary> <shaders>
  local tag="$1" bin="$2" sh="$3"
  cp -f ~/.glm53_explain.bin /tmp/q7_glm_hist.bin
  rm -rf /tmp/q7_glm_ckpt; mkdir -p /tmp/q7_glm_ckpt
  log "--- glm $tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$sh"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH=/tmp/q7_glm_hist.bin
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=/tmp/q7_glm_ckpt
    export COLI_TIMERS=1 GLM53_VERBOSE=1
    "$bin" --model "$M" --prompt "$P_SHORT" --greedy 128 --logits 512 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  log "  rc=$? bytes=$(wc -c < "$OUT/$tag.out")"
  grep -E '^\[VK\] preload|^\[MAP\]' "$OUT/$tag.err" | tail -3 | sed 's/^/  /'
}

glmrun glm_base "$PRIS" "$PRIS_SH"
glmrun glm_cand "$CAND" "$CAND_SH"

echo
echo "=== glm53 oracle: the Q7 shared-file change must be invisible ==="
for what in teacher_forcing last_logits greedy; do
  # Check the LINE exists before hashing it. Hashing the empty output of a failed
  # grep gives the same digest on both sides and prints "IDENTICAL" for a line
  # neither run produced -- glm53 emits the `greedy` prefix only when it has NO
  # tokenizer, so that is not hypothetical (record §QP got this wrong once and
  # says so). A check that passes on nothing is worse than no check.
  la=$(grep -m1 "^$what" "$OUT/glm_base.out")
  lb=$(grep -m1 "^$what" "$OUT/glm_cand.out")
  if [ -z "$la" ] && [ -z "$lb" ]; then
    echo "  $what: ABSENT in both runs -- this oracle did not run, see the whole-stdout diff below"
    continue
  fi
  [ -n "$la" ] || { echo "  $what: MISSING in base but present in candidate"; continue; }
  [ -n "$lb" ] || { echo "  $what: MISSING in candidate but present in base"; continue; }
  a=$(printf '%s' "$la" | md5sum | cut -c1-16)
  b=$(printf '%s' "$lb" | md5sum | cut -c1-16)
  [ "$a" = "$b" ] && echo "  $what: IDENTICAL ($a)" || echo "  $what: DIFFERS ($a vs $b)"
done
if diff -q <(grep -v "^decode " "$OUT/glm_base.out") <(grep -v "^decode " "$OUT/glm_cand.out") >/dev/null; then
  echo "  STDOUT MINUS THE TIMING LINE: IDENTICAL (this is the greedy-continuation oracle)"
else
  echo "  STDOUT MINUS THE TIMING LINE: DIFFERS"
  diff <(grep -v "^decode " "$OUT/glm_base.out") <(grep -v "^decode " "$OUT/glm_cand.out") | head -10
fi
if cmp -s "$OUT/glm_base.out" "$OUT/glm_cand.out"; then
  echo "  WHOLE STDOUT: BYTE-IDENTICAL ($(stat -c%s "$OUT/glm_base.out") bytes)"
else
  echo "  WHOLE STDOUT: DIFFERS -- $(cmp "$OUT/glm_base.out" "$OUT/glm_cand.out" 2>&1 | head -1)"
fi
echo "  [MAP] / preload / hit counters:"
for t in glm_base glm_cand; do
  printf "    %-9s " $t
  grep -oE "esperti mappabili [0-9]+/[0-9]+|copy=[0-9]+" "$OUT/$t.err" | tr '\n' ' '
  grep -oE "preload: [0-9]+ heat-ranked" "$OUT/$t.err" | tr '\n' ' '
  echo
done
log "glm53 check done; outputs in $OUT"
