#!/bin/bash
# glm_comb_chain.sh -- the combined build (assignment-blocked experts + 1024-row chunk): exactness, then timing.
# P1 (FRANKEN_GLM_MOE_G=4): FIRST config ex1024 = the full model, 2 200 tokens + 8 greedy, chunk 1024, against the saved stock chunk-512 reference ref512_keep (must read maxabs=0 on every tap;
#    it is the first config on purpose: a second config of a different chunk size in the same process diverges, record §L5-GLM-CHUNK), then timing t_a, t_b (8 192 tokens, 262 144 cells, chunk 1024,
#    ring 256 MB, staged, LDS GEMM). P2 (G=8): timing. P3 (G=4): timing again -- the two G=4 processes bracket. Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_comb
O=$HOME/bench/franken/glm5/comb; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; KEEP=$B/ref512_keep; P2200=$B/prose2200.txt; PR=$B/prose8400.txt
[ -x "$BIN" ] && [ -d "$KEEP" ] || { echo "FATAL: missing binary or reference"; exit 2; }
F="--tokens-file $P2200 --ctx 4096 --greedy 8 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 400"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 256"
echo "=== comb start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
run() {   # tag G plan-file
  local tag=$1 g=$2 plan=$3 D=$O/$1; mkdir -p "$D"; cp "$plan" "$D/plan.txt"
  m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
  [ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; return 3; }
  echo "=== $tag (G=$g) start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=$g \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D/plan.txt" > "$D/gate_run.log" 2>&1
  echo "gpu rc=$?"
  $HOME/bench/oracle_verdict.sh "$D/gate_run.log"
  awk -v t=$tag '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-8s %-6s %8.3f ms/token = %6.1f tok/s\n", t, c, a[2], 1000 / a[2] }' "$D/gate_run.log"
  grep -aE "HIP error|Memory access|out of memory" "$D/gate_run.log" | head -1 | cut -c1-160
  grep -aE "vram_dev[0-9]_used.*caches" "$D/gate_run.log" | tail -3 | cut -c1-120
}
{ echo "ex1024 $F --chunk 1024 --oracle $KEEP"; echo "t_a $TP"; echo "t_b $TP"; } > /tmp/comb_p1.$$; run p1_g4 4 /tmp/comb_p1.$$
{ echo "t_a $TP"; echo "t_b $TP"; } > /tmp/comb_p2.$$; run p2_g8 8 /tmp/comb_p2.$$; run p3_g4 4 /tmp/comb_p2.$$
rm -f /tmp/comb_p1.$$ /tmp/comb_p2.$$; echo "=== comb end $(date -Is)"
