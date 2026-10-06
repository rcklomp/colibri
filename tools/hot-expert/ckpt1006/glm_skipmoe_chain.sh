#!/bin/bash
# glm_skipmoe_chain.sh -- DEBUG (timing only): GLM prefill with the routed-expert GEMM launches SKIPPED (--debug-skip-moe 1; staging copies,
# events, trunk, adaptation unchanged; output is garbage) against the normal run, in ONE GPU process (build_dbgmoe.sh binary), palindrome
# normal / skip / skip / normal, then one --timeline run of the skipped path. Does the in-engine staged DMA approach M4's 52 GB/s when the
# expert kernels are not running? (record §M7-HOSTSRC). Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_dbgmoe
O=$HOME/bench/franken/glm5/skipmoe; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
BASE="--tokens-file $PR --ctx 262144 --chunk 512 --time-prefill 8192 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
{ echo "n_a $BASE --debug-skip-moe 0"; echo "s_a $BASE --debug-skip-moe 1"; echo "s_b $BASE --debug-skip-moe 1"; echo "n_b $BASE --debug-skip-moe 0"
  echo "s_tl $BASE --debug-skip-moe 1 --timeline $O/timeline.csv --timeline-skip 2"; } > "$O/plan.txt"
echo "=== skipmoe start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald \
  -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
  "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"
grep -aE "gate_plan_summary" "$O/gate_run.log" | head -2
echo "=== prefill ms/token per config"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-8s %s ms/token\n", c, a[2] }' "$O/gate_run.log"
if [ -f "$O/timeline.csv" ]; then
  python3 -I $HOME/src/franken-engine-pfemb/franken/decode/glm5_timeline.py "$O/timeline.csv" --tokens 512 > "$O/timeline_report.txt" 2>&1
  sed -n '/=== steady state/,/stream floor/p' "$O/timeline_report.txt" | cut -c1-200 | head -22
fi
echo "=== skipmoe end $(date -Is)"
