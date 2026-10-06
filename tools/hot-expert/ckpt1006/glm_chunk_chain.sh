#!/bin/bash
# glm_chunk_chain.sh -- GLM prefill with a 1024-row chunk against 512 (franken_decode_glm_chunk1024 of build_chunk1024.sh). Every chunk copies all the staged experts
# (92.5 GB), so DMA bytes per token halve at 1024. Same ring (--glm-stage-mb 400, which §M7-HOSTSRC measured flat) in both arms so the VRAM is equal. One process,
# a 1024-token smoke arm of chunk 1024 first (an OOM or a fault costs the load, not the comparison), then c512 / c1024 / c1024 / c512 at 8 192 tokens.
# Prefill numbers only (the prose has no oracle here); bit-identity across chunk sizes is checked by a separate chain. Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_chunk1024
CTX=${CTX:-65536}      # 262 144 cells cost 3.3 GB of cache a card: chunk 1024 + the ring did not fit (first run: hipMalloc of the ring failed on dev 2); 65 536 isolates the chunk-size effect
O=$HOME/bench/franken/glm5/chunk2; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
B="--tokens-file $PR --ctx $CTX --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 400"
{ echo "smoke1024 $B --chunk 1024 --time-prefill 2048"
  echo "c512_a $B --chunk 512 --time-prefill 8192"; echo "c1024_a $B --chunk 1024 --time-prefill 8192"
  echo "c1024_b $B --chunk 1024 --time-prefill 8192"; echo "c512_b $B --chunk 512 --time-prefill 8192"; } > "$O/plan.txt"
echo "=== chunk start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"; cut -c1-30 "$O/plan.txt"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|vram_dev[0-9]_used|glm5_stage dev|FATAL|Memory access|hipError|out of memory" "$O/gate_run.log" | cut -c1-170 | head -14
echo "=== prefill ms/token per config"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-10s %8.3f ms/token  = %6.1f tok/s\n", c, a[2], 1000 / a[2] }' "$O/gate_run.log"
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-150
echo "=== chunk end $(date -Is)"
