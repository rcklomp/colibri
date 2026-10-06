#!/bin/bash
# glm_ring_ab_chain.sh -- GLM prefill sensitivity to the staging ring size (--glm-stage-mb; auto = min(1536, free VRAM - 1024) = ~1170/1256/832 MB).
# One GPU process (franken_decode_glm of the pf-embed-kernel worktree, kernel upload on, default environment), 8 192 prose tokens,
# 262 144 cells, chunk 512, staged, swaps held, --gemm-lds 1; palindrome auto 800 400 400 800 auto. If prefill slows as the ring shrinks,
# ring depth (and so the VRAM spent on it) is a lever; if not, the DMA/dependency structure is. Launch through run_chain.sh.
set -u
D=${EMB_WT:-/home/ronald/src/franken-engine-pfemb}/franken/decode
O=$HOME/bench/franken/glm5/ring_ab; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17
PR=/home/ronald/bench/franken/glm5/prose8400.txt
BASE="--tokens-file $PR --ctx 262144 --chunk 512 --time-prefill 8192 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
{ echo "ring_auto $BASE"; echo "ring_800 $BASE --glm-stage-mb 800"; echo "ring_400 $BASE --glm-stage-mb 400"
  echo "ring_400b $BASE --glm-stage-mb 400"; echo "ring_800b $BASE --glm-stage-mb 800"; echo "ring_autob $BASE"; } > "$O/plan.txt"
echo "=== ring_ab start $(date -Is) worktree $(git -C ${D%/franken/decode} log --oneline -1 | cut -c1-70)"; cat "$O/plan.txt" | cut -c1-60
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald \
  -v /home/ronald:/home/ronald -w $D rocm/dev-ubuntu-24.04:7.14.0-full \
  ./franken_decode_glm --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"
grep -aE "^glm5_stage dev|gate_plan_summary" "$O/gate_run.log" | cut -c1-200 | head -8
echo "=== prefill ms/token per config"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-10s %s ms/token\n", c, a[2] }' "$O/gate_run.log"
echo "=== ring_ab end $(date -Is)"
