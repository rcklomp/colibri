#!/bin/bash
# glm_chunk_diag_chain.sh -- is the chunk-1024 full-model divergence (1 652 of 1 788 taps not exact, first at layer 4) a real rows >= 512 bug or an artifact of growing the per-card chunk buffers in a process that
# had already run a 512-row runner? Here chunk 1024 runs FIRST in a fresh process against the saved chunk-512 reference (ref512_keep, dumped earlier), then chunk 512 as the sanity arm (must be exact), then
# chunk 1024 again. Same prompt/options as glm_chunk_exact_chain.sh. Launch through run_chain.sh.
set -u
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_chunk1024}
O=$HOME/bench/franken/glm5/chunk_diag; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; KEEP=$B/ref512_keep; P2200=$B/prose2200.txt
F="--tokens-file $P2200 --ctx 4096 --greedy 8 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 400"
[ -d "$KEEP" ] || { echo "FATAL: no reference $KEEP"; exit 2; }
{ echo "c1024first $F --chunk 1024 --oracle $KEEP"; echo "c512sanity $F --chunk 512 --oracle $KEEP"; echo "c1024again $F --chunk 1024 --oracle $KEEP"; } > "$O/plan.txt"
echo "=== chunk_diag start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; grep -aE "gate_plan_summary|Memory access|HIP error|out of memory" "$O/gate_run.log" | head -3 | cut -c1-170
echo "=== bit-identity per config"; $HOME/bench/oracle_verdict.sh "$O/gate_run.log"
echo "=== chunk_diag end $(date -Is)"
