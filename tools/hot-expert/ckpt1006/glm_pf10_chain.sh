#!/bin/bash
# glm_pf10_chain.sh -- PF10 (ACTION-LIST row 5f, prefill plan sections 0b / 0c): does a different LAYER SPLIT balance the three cards in prefill? On the installed model (the IQ3_XXS-expert hybrid), the
# shipped flags, `--expert-gb 17`, chunk 1 024.
# Basis (record §L5-PF9 point 4, IQ4_XS): per chunk the main streams are 2.64 / 2.90 / 3.37 s (78 / 86 / 100 %) on layers 0-14 / 15-29 / 30-44; card 2 holds the head and the lone link and binds;
# equal shares would be -12 % at most (3.37 -> 2.97 s). Layers 0-2 are dense (card 0 has 12 MoE layers of 15, the other two 15), every fourth layer is a DSA layer (about twice a KDA layer).
# The engine's `--split L1,L2` starts card 1 at layer L1 and card 2 at layer L2 (glm5_model.cpp: `il >= split_[i]`); it is a PROCESS option, so every split is its own process (a model load each, ~3-6 min).
# Per split, ONE gate-plan process, three configs (all measurements, no check): w0 warm-up (prose 4 096, discarded), p (prose 8 192 tokens + 32 decode tokens at depth, settle 64), r (technical 8 192).
# Splits (cards 0 / 1 / 2 hold): 15,30 = 15/15/15 (the shipped), 16,31 = 16/15/14, 17,32 = 17/15/13, 17,31 = 17/14/14, 16,32 = 16/16/13, then 15,30 AGAIN last (drift of the day: the splits are separate processes).
# A split whose process dies (VRAM: a layer moved is 0.2-0.3 GB of trunk + KV on its new card) is reported and the chain goes on. Stop rule (plan 0c): the best split must beat 15,30 by > 0.3 ms/token of
# prefill on BOTH workloads, else PF10 stops. The winner then gets an A,B,B,A against the shipped split in a second chain run (PF10_SPLITS="15,30 W 15,30 W").
# Launch (rig idle, the service stopped, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf10_chain.sh > ~/bench/glm_pf10_chain.log 2>&1 < /dev/null &
# Env: PF10_D (decode dir with the binaries, default the rig's franken-engine main tree), PF10_BIN (franken_decode_glm_pf14), PF10_MODEL, PF10_EG (17), PF10_SPLITS (the list above), PF10_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF10_D:-$HOME/src/franken-engine/franken/decode}; BIN=$D/${PF10_BIN:-franken_decode_glm_pf14}; VERDICT=$D/glm5_gate_verdict.sh
O=${PF10_OUT:-$HOME/bench/franken/glm5/pf10}; mkdir -p "$O"
M=${PF10_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
EG=${PF10_EG:-17}; P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
SPLITS=${PF10_SPLITS:-"15,30 16,31 17,32 17,31 16,32 15,30"}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1"
say_end() { echo "=== glm_pf10 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$VERDICT" "$M" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf10 start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) model=$(basename "$M") expert-gb=$EG splits=[$SPLITS]"
rig_quiet_wait 1800 || say_end 3
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
dk() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
         -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full "$@"; }
i=0
for S in $SPLITS; do
  i=$((i + 1)); T="s${i}_${S/,/_}"
  { echo "w0 --tokens-file $PR $BASE --time-prefill 4096"
    echo "p  --tokens-file $PR $BASE --time-prefill 8192 --time 32 --time-settle 64"
    echo "r  --tokens-file $REC $BASE --time-prefill 8192"; } > "$O/plan_$T.txt"
  printf 'w0 none\np none\nr none\n' > "$O/checks_$T.txt"
  echo "=== split $S ($T) $(date +%T)"
  python3 -I "$HOME/bench/gpu_sampler.py" run "$O/samples_$T.tsv" "$O/gate_$T.log" 0.25 & SAMP=$!
  dk "$BIN" --model "$M" --placement "$P" --expert-gb "$EG" --split "$S" --gate-plan "$O/plan_$T.txt" > "$O/gate_$T.log" 2>&1
  rc=$?; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  echo "[$T] engine rc=$rc"
  grep -aE "^placement dev=|vram_dev[0-9]_steady|HIP error|Memory access|out of memory|FATAL" "$O/gate_$T.log" | cut -c1-200 | head -8
  [ "$rc" -ne 0 ] && { echo "--- engine log tail ($T)"; tail -n 8 "$O/gate_$T.log" | cut -c1-200; }
done
echo "=== prefill ms/token (8 192 tokens, first chunk included) and decode median at depth ~8 290, per split (separate processes, order as run)"
printf '%-12s %-14s %-10s %-10s %-10s %s\n' split "prose pf" "tech pf" "decode" "" "min steady free MB (c0/c1/c2)"
i=0
for S in $SPLITS; do
  i=$((i + 1)); T="s${i}_${S/,/_}"; L="$O/gate_$T.log"
  pp=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "p" { split($5, a, "="); print a[2]; exit }' "$L")
  pr=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "r" { split($5, a, "="); print a[2]; exit }' "$L")
  dd=$(awk '/^=== gate-plan config/ { c = $4 } /^glm5_decode_ms_median=/ && c == "p" { split($1, a, "="); print a[2]; exit }' "$L")
  fr=$(grep -a "vram_dev[0-9]_steady" "$L" | sed 's/.*free_mb=\([0-9]*\).*/\1/' | tail -3 | tr '\n' '/')
  printf '%-12s %-14s %-10s %-10s %-10s %s\n' "$S" "${pp:-FAIL}" "${pr:-FAIL}" "${dd:-FAIL}" "" "${fr:-NA}"
done
say_end 0
