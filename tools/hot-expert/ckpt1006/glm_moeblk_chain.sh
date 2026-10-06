#!/bin/bash
# glm_moeblk_chain.sh -- gate and time the register-blocked routed-expert kernel (franken-engine branch moe-regblock f6c69e5, binary franken_decode_glm_moeblk 396d0cff496dfdbf).
# 1. the standard `emb` gate (glm5_emb_chain.sh with EMB_WT at the worktree): the four exact configs must be bit-identical against the gpu5 references; STOP here if the gate fails.
# 2. timing of the pf2 config (8 192 tokens, 262 144 cells, chunk 512, staged, LDS GEMM, ring auto), two configs a process, one process a group size:
#    FRANKEN_GLM_MOE_G = 4 (the new default), 1 (the old kernel), 8, 4 again -- the two G=4 processes bracket the others. Launch through run_chain.sh.
set -u
WT=$HOME/src/franken-engine-moe; D=$WT/franken/decode
BIN=$HOME/bench/franken_bin/franken_decode_glm_moeblk
O=$HOME/bench/franken/glm5/moeblk; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
cp -p "$BIN" "$D/franken_decode_glm"                        # the gate expects this name inside the worktree (same bytes)
echo "=== moeblk start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) worktree $(git -C $WT log --oneline -1)"
EMB_WT=$WT EMB_OUT=$O/emb_gate $HOME/bench/glm5_emb_chain.sh > "$O/emb_gate.log" 2>&1; rc=$?
echo "emb gate rc=$rc"; grep -aE "GPU ORACLE|configs=|exact|verdict|PASS|FAIL" "$O/emb_gate.log" | grep -av "^oracle " | cut -c1-170 | tail -14
[ -f "$O/emb_gate/gate_run.log" ] && { echo "--- exactness"; $HOME/bench/oracle_verdict.sh "$O/emb_gate/gate_run.log"; }
[ $rc -eq 0 ] || { echo "GATE FAILED (rc=$rc): the kernel is not shippable, timing skipped"; exit $rc; }
TP="--tokens-file $PR --ctx 262144 --chunk 512 --time-prefill 8192 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1"
for g in 4 1 8 4; do
  D2=$O/g${g}_$(date +%H%M%S); mkdir -p "$D2"
  { echo "t_a $TP"; echo "t_b $TP"; } > "$D2/plan.txt"
  echo "=== G=$g start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=$g \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D2/plan.txt" > "$D2/gate_run.log" 2>&1
  echo "gpu rc=$?"
  awk -v g=$g '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "G=%s %s %8.3f ms/token = %6.1f tok/s\n", g, c, a[2], 1000 / a[2] }' "$D2/gate_run.log"
  grep -aE "HIP error|Memory access|out of memory" "$D2/gate_run.log" | head -1 | cut -c1-160
done
echo "=== moeblk end $(date -Is)"
