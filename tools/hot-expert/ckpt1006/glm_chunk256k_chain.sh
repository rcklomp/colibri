#!/bin/bash
# glm_chunk256k_chain.sh -- does a 1024-row chunk fit the 262 144-cell (256k) context the service needs, and is it still faster there? (record §L5-GLM-CHUNK: the 400 MB ring OOMed on dev 2.)
# Attempt 1: ring 256 MB, expert budget 17 GB (as served). If the log shows an OOM, attempt 2: the same with --expert-gb 16 (a GB less resident experts a card = ~3 GB more staged a chunk, a few %
# of the chunk's DMA, against half the bytes a token at 1024). Each attempt is its own process: smoke arm of chunk 1024 first, then 8 192-token arms chunk 512 (the served baseline: auto ring)
# against chunk 1024, A,B,B,A. Launch through run_chain.sh.
set -u
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_chunk1024}
O=$HOME/bench/franken/glm5/chunk256k; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; PR=/home/ronald/bench/franken/glm5/prose8400.txt
B="--tokens-file $PR --ctx 262144 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
for att in "r256:17:256" "r256eg16:16:256"; do
  IFS=: read tag EG RING <<< "$att"; D=$O/$tag; mkdir -p "$D"
  { echo "smoke1024 $B --chunk 1024 --glm-stage-mb $RING --time-prefill 2048"
    echo "c512auto_a $B --chunk 512 --time-prefill 8192"; echo "c1024_a $B --chunk 1024 --glm-stage-mb $RING --time-prefill 8192"
    echo "c1024_b $B --chunk 1024 --glm-stage-mb $RING --time-prefill 8192"; echo "c512auto_b $B --chunk 512 --time-prefill 8192"; } > "$D/plan.txt"
  echo "=== attempt $tag (expert-gb $EG, ring $RING MB) start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D/plan.txt" > "$D/gate_run.log" 2>&1
  echo "gpu rc=$?"
  grep -aE "gate_plan_summary|glm5_stage dev|vram_dev[0-9]_used.*caches|HIP error|Memory access|out of memory" "$D/gate_run.log" | cut -c1-175 | head -12
  awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-12s %8.3f ms/token = %6.1f tok/s\n", c, a[2], 1000 / a[2] }' "$D/gate_run.log"
  grep -aq "gate_plan_summary configs=5 ok=5" "$D/gate_run.log" && { echo "attempt $tag complete"; break; }
  grep -aqE "out of memory" "$D/gate_run.log" || { echo "attempt $tag failed for a reason other than OOM; not retrying"; break; }
  sleep 20
done
echo "=== chunk256k end $(date -Is)"
