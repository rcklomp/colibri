#!/bin/bash
# glm_stageprod_chain.sh -- staged against IN-PLACE prefill at the PRODUCTION sizes (record §L5-GLM-STAGECROSS: in place beat staging at every chunk size 16-256 rows; staging was only 4 % faster
# when it was introduced, §L5-GLM-STAGE, with the old kernels). 8 192 tokens, 262 144 cells, LDS GEMM, adaptation held; chunk 1024 [staged, in place, in place, staged] then chunk 512 [staged, in place]
# (descending sizes; the staged config first so the 256 MB ring is allocated as in every earlier run). Same kernel on the same bytes: bit-identical, so only the speed is in question.
# Binary franken_decode_glm_hgblk (the shipped code). Launch through run_chain.sh.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_hgblk}
O=$HOME/bench/franken/glm5/stageprod; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
rig_quiet_wait 1800 || exit 3
B="--tokens-file $PR --ctx 262144 --time-prefill 8192 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 256"
{ echo "s1024_a $B --chunk 1024 --glm-prefill-stage 1"; echo "u1024_a $B --chunk 1024 --glm-prefill-stage 0"
  echo "u1024_b $B --chunk 1024 --glm-prefill-stage 0"; echo "s1024_b $B --chunk 1024 --glm-prefill-stage 1"
  echo "s512_a $B --chunk 512 --glm-prefill-stage 1";   echo "u512_a $B --chunk 512 --glm-prefill-stage 0"; } > "$O/plan.txt"
echo "=== stageprod start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; grep -aE "HIP error|Memory access|out of memory|gate_plan_summary" "$O/gate_run.log" | head -3 | cut -c1-170
grep -aE "vram_dev[0-9]_used.*caches" "$O/gate_run.log" | tail -3 | cut -c1-120
echo "=== prefill ms/token (s = staged, u = in place)"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-8s %8.3f ms/token = %6.1f tok/s\n", c, a[2], 1000 / a[2] }' "$O/gate_run.log"
echo "=== stageprod end $(date -Is)"
