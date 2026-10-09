#!/bin/bash
# glm_pf2_chain.sh -- PF2 (ACTION-LIST row 5g): the f16 / rocBLAS trunk GEMM (`--gemm-lds 3`) against the shipped `--gemm-lds 1`, on the installed hybrid, shipped flags, `--expert-gb 17`, chunk 1 024.
# Per arm ONE gate-plan process, configs (all measurements, no check): w0 warm-up (prose 4 096, discarded; it also loads rocBLAS's kernels), p (prose 8 192 + 32 decode tokens at depth, settle 64),
# r (technical 8 192), and, for every arm, q = prose 4 096 with --profile (the f16 arm's range statistics and per-class times; its timing is NOT used).
# Arms are processes (a model load each, ~1 min from a warm cache), order A B B A by default: PF2_ARMS="1 3 3 1" (the --gemm-lds value of each). Driver: PF2_D (decode dir with the binary), PF2_BIN
# (franken_decode_glm_f16), PF2_MODEL (the hybrid), PF2_EG (17), PF2_OUT.
# Launch (rig idle, the service stopped, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf2_chain.sh > ~/bench/glm_pf2_chain.log 2>&1 < /dev/null &
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF2_D:-$HOME/src/franken-engine/franken/decode}; BIN=$D/${PF2_BIN:-franken_decode_glm_f16}
O=${PF2_OUT:-$HOME/bench/franken/glm5/pf2}; mkdir -p "$O"
M=${PF2_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
EG=${PF2_EG:-17}; P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
ARMS=${PF2_ARMS:-"1 3 3 1"}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --glm-help-copy 1"
say_end() { echo "=== glm_pf2 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$M" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf2 start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) model=$(basename "$M") expert-gb=$EG arms=[$ARMS]"
rig_quiet_wait 1800 || say_end 3
dk() { local G=$1; shift; docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
         -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=$G -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
         -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
         -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full "$@"; }
i=0
for G in $ARMS; do
  i=$((i + 1)); T="a${i}_g${G}"
  { echo "w0 --tokens-file $PR $BASE --gemm-lds $G --time-prefill 4096"
    echo "p  --tokens-file $PR $BASE --gemm-lds $G --time-prefill 8192 --time 32 --time-settle 64"
    echo "r  --tokens-file $REC $BASE --gemm-lds $G --time-prefill 8192"
    echo "q  --tokens-file $PR $BASE --gemm-lds $G --time-prefill 4096 --profile"; } > "$O/plan_$T.txt"
  printf 'w0 none\np none\nr none\nq none\n' > "$O/checks_$T.txt"
  echo "=== arm $i: --gemm-lds $G ($T) $(date +%T)"
  dk "$G" "$BIN" --model "$M" --placement "$P" --expert-gb "$EG" --gate-plan "$O/plan_$T.txt" > "$O/gate_$T.log" 2>&1
  rc=$?
  echo "[$T] engine rc=$rc"
  grep -aE "vram_dev[0-9]_steady|HIP error|Memory access|out of memory|FATAL|gemm-lds 3:|rocBLAS|rocblas" "$O/gate_$T.log" | cut -c1-200 | sort -u | head -8
  [ "$rc" -ne 0 ] && { echo "--- engine log tail ($T)"; tail -n 8 "$O/gate_$T.log" | cut -c1-200; }
done
echo "=== prefill ms/token (8 192 tokens, first chunk included) and decode median at depth ~8 290, per arm (separate processes, order as run)"
printf '%-8s %-12s %-12s %-10s %s\n' "mode" "prose pf" "tech pf" "decode" "f16 stats (profile run)"
i=0
for G in $ARMS; do
  i=$((i + 1)); T="a${i}_g${G}"; L="$O/gate_$T.log"
  pp=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "p" { split($5, a, "="); print a[2]; exit }' "$L")
  pr=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "r" { split($5, a, "="); print a[2]; exit }' "$L")
  dd=$(awk '/^=== gate-plan config/ { c = $4 } /^glm5_decode_ms_median=/ && c == "p" { split($1, a, "="); print a[2]; exit }' "$L")
  st=$(grep -a "prof_gemm_f16_calls" "$L" | sort -u | tr '\n' ';' | cut -c1-200)
  printf '%-8s %-12s %-12s %-10s %s\n' "$G" "${pp:-FAIL}" "${pr:-FAIL}" "${dd:-FAIL}" "$st"
done
say_end 0
