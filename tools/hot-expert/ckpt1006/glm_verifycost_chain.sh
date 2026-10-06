#!/bin/bash
# glm_verifycost_chain.sh -- what does a speculative-decoding VERIFY step cost on this rig? GLM-5.3-Flash's GGUF carries the NextN/MTP head (blk.45.nextn.*; llama.cpp PR 25980 shows 0.83/0.65/0.49
# cumulative acceptance at draft positions 1-3 on a VRAM-resident model). Our decode is bound by streaming MISSED experts; verifying k+1 rows needs the UNION of their experts, which grows nearly linearly
# if the token-to-token overlap is only 13-25 % (record §L4-LOOKAHEAD-STEP0) -- unless bigger batched fetches run at a much better link rate than a one-token fetch (21 GB/s). Measure it: the
# prefill chunk path WITHOUT staging reads only the experts the chunk's rows pick, in place over each card's link (= a decode-style miss path for T rows). ms/token at chunk 1, 2, 3, 4, 8 over the same
# 256 tokens (adaptation held, LDS GEMM on) gives the cost per ROW; the break-even for MTP with ~2.5 tokens a step is a step cost under ~2.5x the chunk-1 step.
# ONE process, chunk sizes DESCENDING (a config of a smaller chunk after a larger one does not regrow the buffers; the chunk-size transition artifact of §L5-GLM-CHUNK needs no oracle here),
# each size twice. Binary: franken_decode_glm_hgblk (the shipped code). Launch through run_chain.sh.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_hgblk}
O=$HOME/bench/franken/glm5/verifycost; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
rig_quiet_wait 1800 || exit 3
B="--tokens-file $PR --ctx 262144 --time-prefill 256 --glm-prefill-stage 0 --adapt-prefill 0 --gemm-lds 1"
{ for c in 8 4 3 2 1; do echo "c${c}_a $B --chunk $c"; echo "c${c}_b $B --chunk $c"; done; } > "$O/plan.txt"
echo "=== verifycost start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"; cut -c1-12 "$O/plan.txt" | tr "\n" " "; echo
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; grep -aE "HIP error|Memory access|out of memory|gate_plan_summary" "$O/gate_run.log" | head -3 | cut -c1-170
echo "=== ms per token (a row of a chunk) by chunk size; step cost = chunk x ms/token; ratio to the chunk-1 step"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); r[c] = a[2]; print c, a[2] }' "$O/gate_run.log" | sort -k1,1 | \
  awk '{ n=substr($1,2,length($1)-3)+0; ms=$2; printf "%-6s chunk %-2d  %8.3f ms/token  = step %8.1f ms\n", $1, n, ms, n*ms }'
echo "=== verifycost end $(date -Is)"
