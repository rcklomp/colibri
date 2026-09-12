#!/bin/bash
# qp_glm_check.sh -- roadmap item QP: the OTHER engine, because QP touched a
# shared file.
#
# QP(d) needed one new function in c/backend_vulkan.c (coli_vk_ballast_gb) and
# one line in c/backend_vulkan.h. glm53 links that object, so CLAUDE.md's rule
# applies without exception: "A shared-file change must rebuild and re-measure
# both engines." Nothing in glm53 calls the new function and no existing line of
# that file was edited, so the expected result is IDENTICAL output -- which is
# exactly why it has to be checked rather than asserted.
#
# Builds glm53 at the branch point and at the candidate, runs §G1's prompt
# through both in the standard 3-GPU configuration, and compares glm53's own
# oracle: the `teacher_forcing` line, `last_logits`, and the greedy continuation.
# Restores the binary that was in service on every exit path, gateway included.
#
# Usage (on the rig):  tools/hot-expert/qp_glm_check.sh
# The ballast smoke (QP(d) step 0) is the second half: it shows that the knob
# actually moves the tier before six rome_bench.sh invocations are spent on it.
set -u

SRC=~/src/colibri
M=~/models/GLM-5.3-Flash-colibri-int4-g64
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
OUT=~/bench/qp_glm_out; mkdir -p "$OUT"
BRANCH=perf/qp-format-probe
BASE=de2dd3d            # the branch point: the Q2 merge on hot-expert-tier
SERVED=~/bench/glm53.qpbase
GATEWAY_UP=0

log(){ echo "[$(date -Is)] $*"; }
. "$SRC/tools/hot-expert/rig_lock.sh"

wait_no_engine(){ for _ in $(seq 1 240); do pgrep -x "$1" >/dev/null || return 0; sleep 1; done
  echo "REFUSED: $1 still running"; return 1; }

on_exit(){
  rc=$?
  log "=== qp_glm_check exiting rc=$rc"
  pkill -9 -x glm53 2>/dev/null; pkill -9 -x qwen38-vk 2>/dev/null
  pkill -9 -f "bench/glm53\.[q]p" 2>/dev/null
  wait_no_engine glm53
  # Put the tree's glm53 back to the binary that was serving, and the worktree
  # back on the branch, whatever state the two builds left.
  [ -f "$SERVED" ] && cp -f "$SERVED" "$SRC/c/glm53"
  ( cd "$SRC" && git checkout -q "$BRANCH" ) || true
  if [ "${KEEP_DOWN:-0}" = 1 ]; then
    log "KEEP_DOWN=1: leaving the gateway down for the QP(d) campaign"
  else
    log "restarting the gateway"
    SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
    for _ in $(seq 1 90); do
      pgrep -f "openai_[s]erver.py" >/dev/null && pgrep -x glm53 >/dev/null && break
      sleep 10
    done
    log "gateway: server=$(pgrep -f "openai_[s]erver.py" | wc -l) engine=$(pgrep -x glm53 | wc -l)"
  fi
  rig_lock_release
  log "=== qp_glm_check done"
}
trap on_exit EXIT INT TERM HUP

rig_lock_take "qp-glm-check" || { echo "rig busy"; exit 3; }

if pgrep -x glm53 >/dev/null; then
  log "stopping the owner's gateway"
  pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
  wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; exit 1; }
fi
cp -f "$SRC/c/glm53" "$SERVED"
log "in-service glm53 saved: sha256=$(sha256sum "$SERVED" | cut -c1-16)"

# ---- the two builds ----------------------------------------------------------
log "building glm53 at the branch point $BASE"
( cd "$SRC" && git checkout -q "$BASE" && make -C c glm53 VK=1 ) > "$OUT/build_base.log" 2>&1 \
  || { log "base build FAILED"; tail -20 "$OUT/build_base.log"; exit 1; }
cp -f "$SRC/c/glm53" ~/bench/glm53.qpbasebuild
log "building glm53 at $BRANCH"
( cd "$SRC" && git checkout -q "$BRANCH" && make -C c glm53 qwen38 qwen38-vk VK=1 ) > "$OUT/build_cand.log" 2>&1 \
  || { log "candidate build FAILED"; tail -20 "$OUT/build_cand.log"; exit 1; }
cp -f "$SRC/c/glm53" ~/bench/glm53.qpcand
log "base      sha256=$(sha256sum ~/bench/glm53.qpbasebuild | cut -c1-16)"
log "candidate sha256=$(sha256sum ~/bench/glm53.qpcand | cut -c1-16)"
grep -iE 'error' "$OUT/build_cand.log" | head -5

# ---- GLM page cache ----------------------------------------------------------
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

glmrun(){   # glmrun <tag> <binary>
  local tag="$1" bin="$2"
  cp -f ~/.glm53_explain.bin /tmp/qp_glm_hist.bin
  rm -rf /tmp/qp_glm_ckpt; mkdir -p /tmp/qp_glm_ckpt
  log "--- glm $tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
    export COLI_USAGE_PATH=/tmp/qp_glm_hist.bin
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=/tmp/qp_glm_ckpt
    export COLI_TIMERS=1 GLM53_VERBOSE=1
    "$bin" --model "$M" --prompt "$P_SHORT" --greedy 128 --logits 512 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  log "  rc=$? bytes=$(wc -c < "$OUT/$tag.out")"
  grep -E '^\[VK\] preload' "$OUT/$tag.err" | tail -2 | sed 's/^/  /'
}

glmrun glm_base "$HOME/bench/glm53.qpbasebuild"
glmrun glm_cand "$HOME/bench/glm53.qpcand"

echo
echo "=== glm53 oracle: the shared-file change must be invisible ==="
for what in teacher_forcing last_logits greedy; do
  a=$(grep -m1 "^$what" "$OUT/glm_base.out" | md5sum | cut -c1-16)
  b=$(grep -m1 "^$what" "$OUT/glm_cand.out" | md5sum | cut -c1-16)
  [ -n "$a" ] || { echo "  $what: MISSING in base"; continue; }
  [ "$a" = "$b" ] && echo "  $what: IDENTICAL ($a)" || echo "  $what: DIFFERS ($a vs $b)"
done
if cmp -s "$OUT/glm_base.out" "$OUT/glm_cand.out"; then
  echo "  WHOLE STDOUT: BYTE-IDENTICAL ($(stat -c%s "$OUT/glm_base.out") bytes)"
else
  echo "  WHOLE STDOUT: DIFFERS -- $(cmp "$OUT/glm_base.out" "$OUT/glm_cand.out" 2>&1 | head -1)"
fi

# ---- QP(d) step 0: does the ballast actually move the tier? ------------------
# Six rome_bench.sh invocations are ~4 hours. Spend one fresh-process run per
# level first: if `Vulkan tier preloaded N` and the OPTIME placement line do not
# move, the knob is not the independent variable the item thinks it is and the
# campaign would produce three identical rows.
log "=== QP(d) step 0: ballast vs tier count, fresh process, COLI_TIMERS=1"
python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
log "warming Qwen"
find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done

for gb in 0 3.4 5.4 6.8; do
  cp -f "$QSNAP/.coli_usage" /tmp/qp_hist.bin
  tag="ballast_$gb"
  log "--- $tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export SNAP="$QSNAP" COLI_USAGE=/tmp/qp_hist.bin N_NEW=32 NOSTREAM=1 COLI_TIMERS=1
    [ "$gb" != 0 ] && export Q38_VK_BALLAST_GB="$gb"
    "$SRC/c/qwen38-vk" 512 8 /tmp/qp_prompt_short.txt ) > "$OUT/$tag.log" 2>&1
  log "  rc=$?"
  grep -E "QP\(d\) ballast|Vulkan tier preloaded|OPTIME\] placement|^Speed:" "$OUT/$tag.log" | sed 's/^/  /'
done
