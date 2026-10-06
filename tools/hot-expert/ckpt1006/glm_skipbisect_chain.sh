#!/bin/bash
# glm_skipbisect_chain.sh -- DEBUG (timing only): GLM prefill with the heavy kernels removed class by class (--debug-skip MASK of the
# build_dbgskip.sh binary; 1 experts, 2 trunk GEMMs, 4 KDA, 8 attention, 16 indexer (implies 8); staging copies, events, norms, hyper-
# connection, router and plan always run; output is garbage). Handoff 2026-10-06 §5.1(a): record §M7-HOSTSRC left the staged DMA at
# 21.5 GB/s with the experts running and 27 with them skipped, against the probe replay's 57 and M4's 52. If the skeleton (31) reaches ~57
# the rest of the engine's compute slows the links and the masks say which class; if it does not, the structure of the copy issue is the cause.
# ONE GPU process. Two smoke arms of the riskiest masks at 1 024 tokens come first (a fault costs the load, not the bisect), then the ladder
# normal / 1 / 5 / 25 / 3 / 7 / 31 and its mirror (A,B,B,A), then two --timeline runs (31 and 3) for GB/s while a copy is in flight.
# Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_dbgskip
O=$HOME/bench/franken/glm5/skipbisect; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
BASE="--tokens-file $PR --ctx 262144 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
FULL="$BASE --time-prefill 8192"
{ echo "smoke31 $BASE --time-prefill 1024 --debug-skip 31"; echo "smoke7 $BASE --time-prefill 1024 --debug-skip 7"
  for m in 0 1 5 25 3 7 31; do echo "m${m}_a $FULL --debug-skip $m"; done
  for m in 31 7 3 25 5 1 0; do echo "m${m}_b $FULL --debug-skip $m"; done
  echo "tl31 $FULL --debug-skip 31 --timeline $O/timeline_31.csv --timeline-skip 2"
  echo "tl3 $FULL --debug-skip 3 --timeline $O/timeline_3.csv --timeline-skip 2"; } > "$O/plan.txt"
echo "=== skipbisect start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"; cut -c1-20 "$O/plan.txt"
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
for t in 31 3; do
  f="$O/timeline_$t.csv"
  if [ -f "$f" ]; then
    echo "=== timeline, mask $t"
    python3 -I $HOME/src/franken-engine/franken/decode/glm5_timeline.py "$f" --tokens 512 > "$O/timeline_${t}_report.txt" 2>&1
    sed -n '/=== steady state/,/stream floor/p' "$O/timeline_${t}_report.txt" | cut -c1-200 | head -22
  fi
done
echo "=== skipbisect end $(date -Is)"
