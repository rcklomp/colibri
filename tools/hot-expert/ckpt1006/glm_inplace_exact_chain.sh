#!/bin/bash
# glm_inplace_exact_chain.sh -- is the IN-PLACE prefill path (--glm-prefill-stage 0) bit-identical to the staged one? (record §L5-GLM-STAGECROSS: it is faster at every chunk size; same kernel, same bytes.)
#   P1  the FULL model, 2 200 prose tokens + 8 greedy, chunk 1024, stage 0, first config of a fresh process, against the saved stock chunk-512 STAGED dump ref512_keep
#   P2  the standard small-chunk exact configs with stage 0 (these are the follow-up-turn sizes that will take this path): c32u, c512u (136 tokens + 8 greedy against gpu5/g136_eager) and long512u
#       (layers 0-3, 2 200 tokens against gpu5/g2200); adaptation held, as served
# Every tap must read maxabs=0. Binary franken_decode_glm_hgblk (the shipped engine code). Launch through run_chain.sh.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${BIN:-$HOME/bench/franken_bin/franken_decode_glm_hgblk}
O=$HOME/bench/franken/glm5/inplace_exact; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; KEEP=$B/ref512_keep; REF5=$B/gpu5; P2200=$B/prose2200.txt; T136=$B/t136.txt
[ -d "$KEEP" ] && [ -d "$REF5/g136_eager" ] && [ -d "$REF5/g2200" ] || { echo "FATAL: references missing"; exit 2; }
run() {   # tag plan-file
  local D=$O/$1; mkdir -p "$D"; cp "$2" "$D/plan.txt"
  rig_quiet_wait 1800 || return 3
  echo "=== $1 start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D/plan.txt" > "$D/gate_run.log" 2>&1
  echo "gpu rc=$?"; grep -aE "HIP error|Memory access|out of memory|gate_plan_summary" "$D/gate_run.log" | head -2 | cut -c1-170
  $HOME/bench/oracle_verdict.sh "$D/gate_run.log"
}
echo "=== inplace_exact start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
echo "ex1024u --tokens-file $P2200 --ctx 4096 --greedy 8 --adapt-prefill 0 --gemm-lds 1 --chunk 1024 --glm-prefill-stage 0 --oracle $KEEP" > /tmp/ie_p1.$$; run p1_full /tmp/ie_p1.$$
{ echo "c32u --tokens-file $T136 --ctx 4096 --greedy 8 --chunk 32 --glm-prefill-stage 0 --adapt-prefill 0 --oracle $REF5/g136_eager"
  echo "c512u --tokens-file $T136 --ctx 4096 --greedy 8 --chunk 512 --glm-prefill-stage 0 --adapt-prefill 0 --oracle $REF5/g136_eager"
  echo "long512u --layers 0-3 --no-head --tokens-file $P2200 --ctx 4096 --chunk 512 --glm-prefill-stage 0 --oracle $REF5/g2200"; } > /tmp/ie_p2.$$; run p2_small /tmp/ie_p2.$$
rm -f /tmp/ie_p1.$$ /tmp/ie_p2.$$; echo "=== inplace_exact end $(date -Is)"
