#!/bin/bash
# Supplementary to q9_step0_verify_cost.sh: measure dense_gpu(1), the single
# real decode-phase S=1 GPU-dense forward (dn-proj+qsa-proj+lm-head, mask 7).
# N_NEW=1 never calls step() a second time (qwen38.c generate(): the first
# token is read off the prefill's own last-position logits and the loop
# breaks before any decode step()), so the main sweep's "decode only" OPTIME
# bank never had forwards>0 -- found while inspecting the first completed
# sweep log, not assumed. N_NEW=2 forces exactly one decode step() call.
set -u
WT=${WT:-/home/ronald/src/colibri-q9s0}
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
CAND=$WT/c/qwen38-vk
SH=$WT/c/shaders
OUT=~/bench/q9s0_out; mkdir -p "$OUT"

. /home/ronald/bench/q7_lib.sh
trap q7_on_exit EXIT INT TERM HUP

q7_take_lock "q9-step0-dense-gpu1" 60 || exit 3
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
stop_gateway || exit 1

python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("[evict] GLM ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY

P=/tmp/q9s0_prompt.txt
[ -f "$P" ] || python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*6)" > "$P"

find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/q9s0dg_fincore.txt
python3 -c "
PAGE=4096; tot=res=0; short=0
for line in open('/tmp/q9s0dg_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid] resident={res/tot*100:.4f}% short={short}')
assert short==0, 'NOT 100% RESIDENT'
"

cp -f "$QSNAP/.coli_usage" /tmp/q9s0dg_hist.bin
env SNAP="$QSNAP" COLI_USAGE=/tmp/q9s0dg_hist.bin N_NEW=2 NOSTREAM=1 \
    Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto COLI_VK_SHADERS="$SH" \
    Q38_DENSE_GPU=7 Q38_PREFILL_BATCH=1 COLI_TIMERS=1 \
    OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    "$CAND" 512 8 "$P" > "$OUT/dense_gpu1.log" 2>&1
rc=$?
log "dense_gpu1 probe rc=$rc"
grep -A2 '=== decode only' "$OUT/dense_gpu1.log" | head -5
log "done"
