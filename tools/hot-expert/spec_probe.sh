#!/bin/bash
# SPEC-PROBE: how does a K-token verify pass scale on this box?
#
# Every technique in the AngelSpec family (DFlash/DFlash2/DFly/DFlare/DSpark/MTP)
# shares one operation: the target verifies a BLOCK of K tokens in a single
# forward. The drafter only changes how many of those K are accepted. So the
# ceiling on this hardware is set by how the verify pass scales with K, which is
# measurable today with no drafter at all.
#
# forward_prefill already chunks by GLM53_PREFILL_CHUNK, and a chunk IS a
# verify-shaped span: K tokens through all 45 layers against existing context.
# Because the prompt is FIXED, the per-token routing is identical at every chunk
# size -- so any change in [PROF] cpu n= is purely the distinct-expert union
# effect, and any change in time is purely batching. Controlled by construction.
#
# The number that decides everything: on rome ~21% of expert activations miss
# the GPU tier and stream from system RAM (38.3 ms/token, 2nd largest bucket).
# A datacenter box has every expert in HBM and does not pay this at all, which
# is why published speedups do not transfer here.
set -u
cd "$HOME/src/colibri/c"
M=$HOME/models/GLM-5.3-Flash-colibri-int4-g64

export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$HOME/src/colibri/c/shaders"
export COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695
export COLI_TIMERS=1
cp "$HOME/.glm53_explain.bin" /tmp/spec_hist.bin
export COLI_USAGE_PATH=/tmp/spec_hist.bin

# ~250-token prompt: the RP2 prompt repeated, so it is deterministic and shares
# this campaign's provenance. Repetition is fine -- we are measuring routing
# breadth and batching, not output quality.
P=$(python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*8)")

resid() {
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/spec_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/spec_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid $1] shards={n} resident={res/tot*100:.4f}% short={short}')
assert n==62 and short==0, 'NOT 100% RESIDENT'
"
}
warm() { find "$M" -type f -name "*.safetensors" -print0 \
         | while IFS= read -r -d '' f; do cat "$f" > /dev/null; done; }

for K in 1 2 4 8 16; do
  warm
  resid "K=$K-pre" || { echo "ABORT before K=$K"; exit 1; }
  echo "### ===== CHUNK=$K ====="
  start=$(date +%s.%N)
  GLM53_PREFILL_CHUNK=$K ./glm53 --model "$M" --prompt "$P" --greedy 0 2>&1 \
    | grep -E "^\[PROF\]|^\[OPTIME\]|prompt|token" | head -12
  end=$(date +%s.%N)
  echo "### wall(load+prefill)=$(echo "$end - $start" | bc)s"
done
echo "### done"
