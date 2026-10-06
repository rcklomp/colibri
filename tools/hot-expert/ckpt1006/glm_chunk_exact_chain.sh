#!/bin/bash
# glm_chunk_exact_chain.sh -- is a 1024-row chunk BIT-IDENTICAL to the 512 path? (franken_decode_glm_chunk1024; the engine's design: a chunk row is exactly the decode step at its
# position, so the chunk size must not move a bit.) One process:
#   long1024s  layers 0-3, the 2 200 prose tokens (three 1024-row chunks: 1024 + 1024 + 152; the top-512 indexer regime past 2 048) against the EXISTING gpu5 reference g2200
#   ref512     the FULL model, the same 2 200 tokens, chunk 512, staged, LDS GEMM, + 8 greedy decode tokens: dumps its taps (no comparison)
#   c1024x     the same run at chunk 1024, compared tap by tap with ref512 -- every tap must read maxabs=0 (the oracle's own PASS only means cos >= 0.999)
# --glm-stage-mb 400 keeps the VRAM of the 1024-row scratch. Launch through run_chain.sh.
set -u
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_chunk1024}     # BIN=...glm.integ ONLY_REF=1: the same full-model 512 run on the STOCK engine (does the fault need the chunk patch?)
O=$HOME/bench/franken/glm5/chunk_exact2${TAGX:-}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; REF5=$B/gpu5; P2200=$B/prose2200.txt
F="--tokens-file $P2200 --ctx 4096 --greedy 8 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 400"
# ref512 FIRST: the first run of the first run (the 2 200-token full model with a decode tail, first attempt faulted as the 2nd config after a chunk-1024 runner in the same process)
{ echo "ref512 $F --chunk 512 --dump $O/ref512"
  if [ "${ONLY_REF:-0}" != 1 ]; then
    echo "c1024x $F --chunk 1024 --oracle $O/ref512"
    echo "long1024s --layers 0-3 --no-head --tokens-file $P2200 --ctx 4096 --chunk 1024 --glm-prefill-stage 1 --glm-stage-mb 400 --oracle $REF5/g2200"
  fi; } > "$O/plan.txt"
echo "=== chunk_exact start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"
grep -aE "gate_plan_summary|FATAL|Memory access|hipError|out of memory|at most" "$O/gate_run.log" | cut -c1-170 | head -6
echo "=== bit-identity per config (a tap is exact when maxabs=0)"
awk '/^=== gate-plan config/ { c = $4 } /^oracle [^ ]+ cos=/ { n[c]++; split($4, a, "="); if (a[2] != "0") { bad[c]++; if (bad[c] <= 5) print "  NOT EXACT in " c ": " $0 } } /^oracle summary/ { s[c] = $0 } /^gate_plan_result/ { r[c] = $0 }
  END { for (c in s) printf "%-10s taps=%d not_exact=%d | %s\n", c, n[c], bad[c] + 0, s[c] }' "$O/gate_run.log" | cut -c1-230
grep -aE "^gate_plan_result" "$O/gate_run.log" | cut -c1-100
echo "=== chunk_exact end $(date -Is)"
