#!/bin/bash
# glm_rocprof_chain.sh -- DEBUG (timing meaningless under tracing): are the staged GLM copies executed by the SDMA engine or as blit KERNELS on the CUs?
# Web research (2026-10-06) found no report of the symptom of §M7-SKIPCLASS, but the runtime's own flags say hipMemcpyAsync uses SDMA above
# GPU_FORCE_BLIT_COPY_SIZE (16 KB; the staged copies are 11.67 MB) and a blit kernel below it, and P2P below ROC_P2P_SDMA_SIZE (1 MB) by blit
# (the helper write-backs). If the 11.67 MB copies showed up as kernels they would be competing with the trunk GEMMs for CUs -- that would be the
# cause of the in-flight rate loss. One process, rocprofv3 --kernel-trace --memory-copy-trace, normal prefill (mask 0), 1 536 tokens (3 chunks).
# Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_dbgskip
O=$HOME/bench/franken/glm5/rocprof; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
echo "prof_m0 --tokens-file $PR --ctx 262144 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --time-prefill 1536 --debug-skip 0" > "$O/plan.txt"
echo "=== rocprof start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full /opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d "$O/out" -o prof -- \
  "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"
grep -aE "gate_plan_summary|^prefill_tokens=" "$O/gate_run.log" | cut -c1-160 | head -3
find "$O/out" -name "*.csv" | head; echo "=== analysis"
python3 -I $HOME/bench/rocprof_summary.py "$O/out"
echo "=== rocprof end $(date -Is)"
