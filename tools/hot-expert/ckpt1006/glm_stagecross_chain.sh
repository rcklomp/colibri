#!/bin/bash
# glm_stagecross_chain.sh -- where does STAGING a chunk's missed experts stop paying? The served prefill stages ALL the missed experts of a layer for every chunk of more than one row (92.5 GB a chunk,
# routing-independent): ~2 s of DMA whatever the chunk holds. The in-place path (--glm-prefill-stage 0) reads only the experts the rows pick: 34 ms for 1 row, 40-90 ms a row at 2-8 rows
# (record §L5-GLM-VERIFYCOST). Per chunk size 256, 128, 64, 32, 16: the same 512 prose tokens unstaged then staged, ms/token and the chunk step. The crossover is the stage-above-this-many-rows
# threshold. Same code, same bytes: bit-identical either way. One process, sizes descending. Launch through run_chain.sh.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_hgblk}
O=$HOME/bench/franken/glm5/stagecross; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
rig_quiet_wait 1800 || exit 3
B="--tokens-file $PR --ctx 65536 --time-prefill 512 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 256"
{ for c in 256 128 64 32 16; do echo "u${c} $B --chunk $c --glm-prefill-stage 0"; echo "s${c} $B --chunk $c --glm-prefill-stage 1"; done; } > "$O/plan.txt"
echo "=== stagecross start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; grep -aE "HIP error|Memory access|out of memory|gate_plan_summary" "$O/gate_run.log" | head -3 | cut -c1-170
echo "=== chunk rows: unstaged vs staged (ms per token; step = rows x ms/token)"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); ms[c] = a[2] }
  END { n = split("256 128 64 32 16", S, " "); for (i = 1; i <= n; i++) { r = S[i]; u = ms["u" r]; s = ms["s" r];
        printf "rows %-4d  unstaged %8.2f ms/token (step %8.1f ms)   staged %8.2f ms/token (step %8.1f ms)   %s\n", r, u, r*u, s, r*s, (u < s ? "UNSTAGED wins" : "staged wins") } }' "$O/gate_run.log"
echo "=== stagecross end $(date -Is)"
