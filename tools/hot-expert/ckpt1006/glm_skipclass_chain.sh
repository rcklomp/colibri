#!/bin/bash
# glm_skipclass_chain.sh -- DEBUG (timing only), follow-up to glm_skipbisect_chain.sh: ONE compute class at a time beside the staged DMA.
# mask 29 = only the trunk GEMMs run (experts, KDA, attention, indexer skipped); mask 30 = only the routed-expert kernels run (trunk GEMMs, KDA,
# attention, indexer skipped). The bisect showed the skeleton (31) at 47.6 GB/s and the trunk GEMMs worth ~2.45 ms a token, the experts ~2.3;
# the timeline of each class alone says whether a class costs its time on the critical path (links idle in gaps) or by slowing the links (links
# busy at a lower rate). Same binary (build_dbgskip.sh). Smoke arms of both masks at 1 024 tokens first, then A,B,B,A, then a --timeline run each.
# Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_dbgskip
O=$HOME/bench/franken/glm5/skipclass; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
BASE="--tokens-file $PR --ctx 262144 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
FULL="$BASE --time-prefill 8192"
{ echo "smoke29 $BASE --time-prefill 1024 --debug-skip 29"; echo "smoke30 $BASE --time-prefill 1024 --debug-skip 30"
  echo "m29_a $FULL --debug-skip 29"; echo "m30_a $FULL --debug-skip 30"; echo "m30_b $FULL --debug-skip 30"; echo "m29_b $FULL --debug-skip 29"
  echo "tl29 $FULL --debug-skip 29 --timeline $O/timeline_29.csv --timeline-skip 2"
  echo "tl30 $FULL --debug-skip 30 --timeline $O/timeline_30.csv --timeline-skip 2"; } > "$O/plan.txt"
echo "=== skipclass start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"; cut -c1-20 "$O/plan.txt"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald \
  -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
  "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"
grep -aE "gate_plan_summary" "$O/gate_run.log" | head -2
echo "=== prefill ms/token per config (dma-equiv = 180.7 MB a token / ms: the staged bytes of the 8k prompt at the rate that would just carry them)"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-9s %8.3f ms/token  dma-equiv %5.1f GB/s\n", c, a[2], 180.7 / a[2] }' "$O/gate_run.log"
for t in 29 30; do
  f="$O/timeline_$t.csv"
  if [ -f "$f" ]; then
    echo "=== timeline, mask $t"
    python3 -I $HOME/src/franken-engine/franken/decode/glm5_timeline.py "$f" --tokens 512 > "$O/timeline_${t}_report.txt" 2>&1
    sed -n '/=== steady state/,/stream floor/p' "$O/timeline_${t}_report.txt" | cut -c1-200 | head -22
  fi
done
echo "=== skipclass end $(date -Is)"
