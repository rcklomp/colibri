#!/bin/bash
# The standing per-op profiling campaign for glm53 on rome (record §RP2).
#
# Edit the run list at the bottom; everything above is the procedure that makes
# the numbers comparable to §G3 / §RP1 / §RP2. Parse the output with
# profile_parse.py in this directory.
#
# Part 2's guard caught the model at 91.7% resident BEFORE its first run, with
# 51 of 62 shards short -- after part 1 had verified 100% at ITS start. Each
# fresh glm53 process transiently allocates enough anon memory (expert cache,
# 136 MB KDA state, KV) to push the kernel into reclaim, and the page cache it
# drops does not come back when the process exits. 59 GiB shows free afterwards,
# so this is invisible unless you look at the model's own pages.
#
# Therefore: RE-WARM AND ASSERT BEFORE EVERY RUN, and record residency after
# each one to quantify the drop. Asserting once at the top of a campaign -- what
# part 1 did, and what RP1 did in its own way -- is not enough.
set -u
cd "$HOME/src/colibri/c"
M=$HOME/models/GLM-5.3-Flash-colibri-int4-g64

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$HOME/src/colibri/c/shaders"
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
export COLI_TIMERS=1
export COLI_USAGE_PATH=/tmp/rp2_hist.bin      # same frozen copy throughout
P=$(cat "$HOME/bench/prompt_glm.txt")

resid() {   # $1 = label, $2 = "assert" | "report"
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/rp2_fincore.txt
  python3 -c "
import sys
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/rp2_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
pct=res/tot*100
print(f'[resid $1] shards={n} resident={pct:.4f}% short={short}')
if '$2'=='assert':
    assert n==62 and short==0, 'NOT 100% RESIDENT'
"
}

warm() { find "$M" -type f -name "*.safetensors" -print0 \
         | while IFS= read -r -d '' f; do cat "$f" > /dev/null; done; }

run() {   # $1 = tag, $2 = knob
  warm
  resid "$1-pre" assert || { echo "ABORT before $1"; exit 1; }
  echo "### ===== $1 (COLI_KDA_GPU=$2) ====="
  COLI_KDA_GPU=$2 ./glm53 --model "$M" --prompt "$P" --greedy 128 2>&1 \
    | grep -E "^\[OPTIME\]|^decode |^\[PROF\] eg"
  resid "$1-post" report
}

# alternating, and starting with ON so the order is counterbalanced against
# part 1 (which started with OFF and whose first run was its slowest)
run rp2c-on-1  2
run rp2c-off-1 0
run rp2c-on-2  2
run rp2c-off-2 0
run rp2c-off-3 0
run rp2c-on-3  2
echo "### done"
