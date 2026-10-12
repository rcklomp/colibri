#!/bin/bash
# glm_pf15_chain.sh -- PF15 (ACTION-LIST row 5i, PREFILL plan section 0j, record §L5-PF15-READ): the DSA attention of a prefill chunk on k_glm5_attn_wmma (f16 WMMA, a workgroup a row,
# no partials) against the shipped k_glm5_attn_part + combine. The shipped config (hybrid, all shipped flags, --expert-gb 16.85, chunk 1 024) with ONLY --glm-attn-wmma changing, one binary.
# Per arm ONE gate-plan process, configs: w0 warm-up (prose 4 096, discarded), p (prose 8 192), r (technical 8 192), q (prose 8 192 --profile: the class times, glm5_mla_attention).
# Arms are processes, order A B B A by default: PF15_ARMS="0 64 64 0" (the --glm-attn-wmma value of each: 0 = never, N = launches of >= N rows).
# Step 0: bench_attn_wmma (kernel vs float64, one card) when PF15_BENCH=1 (default).
# Env: PF15_D (decode dir of the branch build), PF15_BIN (franken_decode_glm_pf15), PF15_MODEL (the hybrid), PF15_EG (16.85), PF15_OUT.
# Launch (rig idle, service OFF, reservation flag in place): setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf15_chain.sh > ~/bench/glm_pf15_chain.log 2>&1 < /dev/null &
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF15_D:-$HOME/src/franken-engine-pf15/franken/decode}; BIN=$D/${PF15_BIN:-franken_decode_glm_pf15}
O=${PF15_OUT:-$HOME/bench/franken/glm5/pf15}; mkdir -p "$O"
M=${PF15_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
EG=${PF15_EG:-16.85}; P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
ARMS=${PF15_ARMS:-"0 64 64 0"}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --fetch-assign optimal --gemv-fused-reduce 0 --glm-gemv-group 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1"
say_end() { echo "=== glm_pf15 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$M" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf15 start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) model=$(basename "$M") expert-gb=$EG arms=[$ARMS]"
rig_quiet_wait 1800 || say_end 3
dk() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 "$@"; }
ENV="-e FRANKEN_GLM_MOE_G=8 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald"
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
if [ "${PF15_BENCH:-1}" = 1 ]; then
  echo "=== step 0: bench_attn_wmma (card 0) $(date +%T)"
  dk -e HIP_VISIBLE_DEVICES=0 $ENV -w "$D" $IMG ./bench_attn_wmma > "$O/bench.log" 2>&1; echo "bench rc=$?"; cat "$O/bench.log"
fi
i=0
for W in $ARMS; do
  i=$((i + 1)); T="a${i}_w${W}"
  { echo "w0 --tokens-file $PR $BASE --glm-attn-wmma $W --time-prefill 4096"
    echo "p  --tokens-file $PR $BASE --glm-attn-wmma $W --time-prefill 8192"
    echo "r  --tokens-file $REC $BASE --glm-attn-wmma $W --time-prefill 8192"
    echo "q  --tokens-file $PR $BASE --glm-attn-wmma $W --time-prefill 8192 --profile"; } > "$O/plan_$T.txt"
  echo "=== arm $i: --glm-attn-wmma $W ($T) $(date +%T)"
  dk $ENV -w /home/ronald/bench $IMG "$BIN" --model "$M" --placement "$P" --expert-gb "$EG" --gate-plan "$O/plan_$T.txt" > "$O/gate_$T.log" 2>&1
  rc=$?
  echo "[$T] engine rc=$rc"
  grep -aE "vram_dev[0-9]_steady|HIP error|Memory access|out of memory|FATAL|glm_attn_wmma" "$O/gate_$T.log" | cut -c1-200 | sort -u | head -8
  [ "$rc" -ne 0 ] && { echo "--- engine log tail ($T)"; tail -n 8 "$O/gate_$T.log" | cut -c1-200; }
done
echo "=== prefill ms/token (8 192 tokens, first chunk included), per arm (separate processes, order as run)"
printf '%-6s %-12s %-12s %s\n' "wmma" "prose pf" "tech pf" "glm5_mla_attention us/token (profile, prose 8 192), per card"
i=0
for W in $ARMS; do
  i=$((i + 1)); T="a${i}_w${W}"; L="$O/gate_$T.log"
  pp=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "p" { split($5, a, "="); print a[2]; exit }' "$L")
  pr=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "r" { split($5, a, "="); print a[2]; exit }' "$L")
  at=$(awk '/^=== gate-plan config/ { c = $4 } c == "q" && /^prof_glm5_mla_attention_us=/ { split($1, a, "="); k = k " " a[2] } END { print k }' "$L")
  printf '%-6s %-12s %-12s %s\n' "$W" "${pp:-FAIL}" "${pr:-FAIL}" "${at:-?}"
done
say_end 0
