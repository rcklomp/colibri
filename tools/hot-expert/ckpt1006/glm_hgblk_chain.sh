#!/bin/bash
# glm_hgblk_chain.sh -- gate and time the blocked head_gemv (franken-engine branch head-gemv-blk f39b449 on top of chunk1024 = moe_blk + the 1024-row chunk; binary franken_decode_glm_hgblk).
#   P1  exactness: the FULL model, 2 200 tokens + 8 greedy, chunk 1024, FIRST config of a fresh process against the saved stock chunk-512 dump ref512_keep (every tap must read maxabs=0)
#   P2  the standard `emb` gate (glm5_emb_chain.sh, EMB_WT at the worktree): the four exact configs must be bit-identical.  STOP after P1/P2 if either is not exact.
#   P3..P6  timing (8 192 tokens, 262 144 cells, chunk 1024, ring 256 MB, two configs a process), one process a setting: new default (FRANKEN_GLM_HG_G=4, _R=2), the OLD kernel (_G=1), new default
#   again, then G=8 R=4. The two default processes bracket the old one. Launch through run_chain.sh.
set -u
WT=$HOME/src/franken-engine-hg; D=$WT/franken/decode
BIN=$HOME/bench/franken_bin/franken_decode_glm_hgblk
O=$HOME/bench/franken/glm5/hgblk${TIMING_ONLY:+_t}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; KEEP=$B/ref512_keep; P2200=$B/prose2200.txt; PR=$B/prose8400.txt
[ -x "$BIN" ] && [ -d "$KEEP" ] || { echo "FATAL: missing binary or reference"; exit 2; }
echo "=== hgblk start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) $(git -C $WT log --oneline -1 | cut -c1-90)"
F="--tokens-file $P2200 --ctx 4096 --greedy 8 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 400"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --glm-stage-mb 256"
run() {   # tag "ENV=.. ENV=.." plan-file
  local tag=$1 envs=$2 plan=$3 Dd=$O/$1; mkdir -p "$Dd"; cp "$plan" "$Dd/plan.txt"
  # the previous process's 144 GB unpin takes minutes (D state): WAIT for the cards (and the container) instead of refusing -- the first run lost three timing processes to this
  for _ in $(seq 1 180); do
    m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
    [ $m -le 1024 ] && [ -z "$(docker ps -aq 2>/dev/null | head -c1)" -o -z "$(docker ps -aq -f ancestor=rocm/dev-ubuntu-24.04:7.14.0-full 2>/dev/null)" ] && break; sleep 5
  done
  [ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB after 15 min"; return 3; }
  local ev=(); for kv in $envs; do ev+=(-e "$kv"); done
  echo "=== $tag [${envs:-defaults}] start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 "${ev[@]}" \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$Dd/plan.txt" > "$Dd/gate_run.log" 2>&1
  echo "gpu rc=$?"
  $HOME/bench/oracle_verdict.sh "$Dd/gate_run.log"
  awk -v t=$tag '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-12s %-6s %8.3f ms/token = %6.1f tok/s\n", t, c, a[2], 1000 / a[2] }' "$Dd/gate_run.log"
  grep -aE "HIP error|Memory access|out of memory" "$Dd/gate_run.log" | head -1 | cut -c1-160
}
exact_ok() { ! grep -aE "not_exact=[1-9]" "$O/$1/gate_run.log" > /dev/null 2>&1 && $HOME/bench/oracle_verdict.sh "$O/$1/gate_run.log" | grep -q "not_exact=0"; }
if [ "${TIMING_ONLY:-0}" != 1 ]; then
echo "ex1024 $F --chunk 1024 --oracle $KEEP" > /tmp/hg_p1.$$; run p1_exact "" /tmp/hg_p1.$$
$HOME/bench/oracle_verdict.sh "$O/p1_exact/gate_run.log" | grep -q "not_exact=0" || { echo "P1 NOT EXACT: stopping"; rm -f /tmp/hg_p1.$$; exit 1; }
cp -p "$BIN" "$D/franken_decode_glm"
EMB_WT=$WT EMB_OUT=$O/emb_gate $HOME/bench/glm5_emb_chain.sh > "$O/emb_gate.log" 2>&1; rc=$?
echo "emb gate rc=$rc"; [ -f "$O/emb_gate/gate_run.log" ] && $HOME/bench/oracle_verdict.sh "$O/emb_gate/gate_run.log"
[ $rc -eq 0 ] || { echo "EMB GATE FAILED (rc=$rc): timing skipped"; rm -f /tmp/hg_p1.$$; exit $rc; }
fi
{ echo "t_a $TP"; echo "t_b $TP"; } > /tmp/hg_t.$$
run p3_new   ""                        /tmp/hg_t.$$
run p4_old   "FRANKEN_GLM_HG_G=1"      /tmp/hg_t.$$
run p5_new   ""                        /tmp/hg_t.$$
run p6_g8r4  "FRANKEN_GLM_HG_G=8 FRANKEN_GLM_HG_R=4" /tmp/hg_t.$$
rm -f /tmp/hg_p1.$$ /tmp/hg_t.$$; echo "=== hgblk end $(date -Is)"
