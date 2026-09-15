#!/bin/bash
# q9_step0_verify_cost.sh -- Q9 spec step 0 (tools/hot-expert/Q9-MTP-SPEC-2026-09-15.md,
# "Step 0 -- the verify cost curve V(S)"). Two parts:
#
#   1. ORACLE: BASE (pristine, no Q38_PREFILL_ROWS) vs CAND (this branch, no
#      Q38_PREFILL_ROWS -- i.e. default) must be bit-identical: teacher_forcing
#      and last-position logits, on the chunked leg (Q38_PREFILL_BATCH=1), the
#      only leg the knob's code touches.
#   2. SWEEP: CAND only, Q38_PREFILL_BATCH=0 (control, does not depend on the
#      knob) and =1 x Q38_PREFILL_ROWS in {1,2,4,8,16,32}, COLI_TIMERS=1,
#      Q38_DENSE_GPU=7 (so a later decode step, if one runs -- see
#      q9_step0_dense_gpu1.sh, N_NEW=1 alone never calls step() a second time
#      -- measures the real S=1 GPU dense path). 3 repeats, order decorrelated
#      (ascending, descending, ascending) so a systematic drift (thermal,
#      background load) does not alias with S. Reports ms/token by bucket,
#      CPU/GPU expert counts, and the required S=1-reproduces-row-at-a-time
#      check to the unit.
#
# Run q9_step0_dense_gpu1.sh separately afterward for dense_gpu(1) (N_NEW=2,
# so exactly one real decode-phase step() call happens); q9_step0_parse.py
# turns both into the V*(S) table.
set -u
WT=${WT:-/home/ronald/src/colibri-q9s0}
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
BASE=${BASE:-~/bench/qwen38-vk.q9s0base}
CAND=$WT/c/qwen38-vk
SH=$WT/c/shaders
OUT=~/bench/q9s0_out; mkdir -p "$OUT"

. /home/ronald/bench/q7_lib.sh
trap q7_on_exit EXIT INT TERM HUP

q7_take_lock "q9-step0" 120 || exit 3
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
[ -x "$BASE" ] || { log "REFUSED: no $BASE"; exit 1; }
[ -x "$CAND" ] || { log "REFUSED: no $CAND"; exit 1; }

log "base:  $(sha256sum "$BASE" | cut -c1-16)"
log "cand:  $(sha256sum "$CAND" | cut -c1-16)   tree $(git -C "$WT" log --oneline -1)"
log "shaders (cand) $(sha256sum "$SH/qmatmul.spv" | cut -c1-16)"
stop_gateway || exit 1

python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("[evict] GLM ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY

P=/tmp/q9s0_prompt.txt
python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*6)" > "$P"

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_TIMERS=1

warm(){ find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done; }
resid(){
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q9s0_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q9s0_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid $1] shards={n} resident={res/tot*100:.4f}% short={short}')
assert short==0, 'NOT 100% RESIDENT'
"
}

warm
resid boot || { log "REFUSED: Qwen not fully resident"; exit 1; }

# ---------------------------------------------------------------- ORACLE ----
oracle_run(){
  local tag="$1" bin="$2"
  cp -f "$QSNAP/.coli_usage" /tmp/q9s0_oracle_hist.bin
  env SNAP="$QSNAP" COLI_USAGE=/tmp/q9s0_oracle_hist.bin N_NEW=1 NOSTREAM=1 \
      COLI_VK_SHADERS="$SH" \
      Q38_PREFILL_BATCH=1 Q38_TF=1 Q38_TF_DUMP="$OUT/oracle_$tag.f32" \
      "$bin" 512 8 "$P" > "$OUT/oracle_$tag.log" 2>&1
  log "  oracle $tag rc=$? tf_len=$(grep -c '^teacher_forcing' "$OUT/oracle_$tag.log") dump=$(stat -c%s "$OUT/oracle_$tag.f32" 2>/dev/null || echo 0)B"
}
log "=== oracle: default Q38_PREFILL_ROWS (unset) on both binaries, chunked leg ==="
oracle_run base "$BASE"
oracle_run cand "$CAND"
bt=$(grep -c '^teacher_forcing' "$OUT/oracle_base.log"); ct=$(grep -c '^teacher_forcing' "$OUT/oracle_cand.log")
if [ "$bt" -lt 1 ] || [ "$ct" -lt 1 ]; then
  log "ORACLE REFUSED: missing teacher_forcing line (base=$bt cand=$ct)"; exit 1
fi
if diff <(grep '^teacher_forcing' "$OUT/oracle_base.log") <(grep '^teacher_forcing' "$OUT/oracle_cand.log") >/dev/null; then
  log "ORACLE teacher_forcing: IDENTICAL ($bt line(s))"
else
  log "ORACLE teacher_forcing: DIFFERS -- STOP, this must be a no-op at default"; exit 1
fi
if cmp -s "$OUT/oracle_base.f32" "$OUT/oracle_cand.f32"; then
  log "ORACLE logits: BYTE-IDENTICAL"
else
  log "ORACLE logits: DIFFER -- STOP, this must be a no-op at default"; exit 1
fi

# ----------------------------------------------------------------- SWEEP ----
run(){
  local batch="$1" rows="$2" rep="$3"
  local tag="b${batch}_r${rows}_rep${rep}"
  cp -f "$QSNAP/.coli_usage" /tmp/q9s0_hist.bin
  if [ "$batch" = 1 ]; then
    env SNAP="$QSNAP" COLI_USAGE=/tmp/q9s0_hist.bin N_NEW=1 NOSTREAM=1 \
        COLI_VK_SHADERS="$SH" Q38_DENSE_GPU=7 \
        Q38_PREFILL_BATCH=1 Q38_PREFILL_ROWS="$rows" \
        "$CAND" 512 8 "$P" > "$OUT/$tag.log" 2>&1
  else
    env SNAP="$QSNAP" COLI_USAGE=/tmp/q9s0_hist.bin N_NEW=1 NOSTREAM=1 \
        COLI_VK_SHADERS="$SH" Q38_DENSE_GPU=7 \
        Q38_PREFILL_BATCH=0 \
        "$CAND" 512 8 "$P" > "$OUT/$tag.log" 2>&1
  fi
  local rc=$?
  log "  $tag rc=$rc"
  [ $rc -eq 0 ] || { tail -20 "$OUT/$tag.log"; exit 1; }
}

log "=== sweep: 3 repeats, order decorrelated ==="
rows_asc="1 2 4 8 16 32"
rows_desc="32 16 8 4 2 1"
for rep in 1 2 3; do
  log "--- repeat $rep: leg0 control ---"
  run 0 0 "$rep"
  order="$rows_asc"; [ "$rep" = 2 ] && order="$rows_desc"
  for rows in $order; do
    log "--- repeat $rep: leg1 rows=$rows ---"
    run 1 "$rows" "$rep"
  done
done

log "q9 step0 sweep done; logs in $OUT"
