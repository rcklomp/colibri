#!/bin/bash
# glm_pf2c_chain.sh -- PF2c (ACTION-LIST row 5h): the small-row trunk GEMMs (24-row hyper-connection mix, KDA beta / f_a / g_a: 64-128 rows) take the split-K k_gemm_batch at a
# 1 024-token chunk, 128 x 128 workgroups per row group (record §L5-PF2c, rocprofv3 trace); --gemm-lds-min-rows 16 sends them to k_gemm_lds. The shipped config (hybrid, --gemm-lds 3,
# --expert-gb 16.85, chunk 1 024) with ONLY that knob changing, one binary. Per arm ONE gate-plan process, configs: w0 warm-up (prose 4 096, discarded), p (prose 8 192), r (technical
# 8 192), q (prose 4 096 --profile: the class times). Arms are processes, order A B B A by default: PF2C_ARMS="256 16 16 256" (the --gemm-lds-min-rows value of each; "R:S" adds
# --gemm-lds-splitk S, e.g. "256:0 16:0 16:1 16:1 16:0 256:0").
# Env: PF2C_D (decode dir), PF2C_BIN (franken_decode_glm_pf2c), PF2C_MODEL (the hybrid), PF2C_EG (16.85), PF2C_OUT.
# Launch (rig idle, service OFF, reservation flag in place): setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf2c_chain.sh > ~/bench/glm_pf2c_chain.log 2>&1 < /dev/null &
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF2C_D:-$HOME/src/franken-engine/franken/decode}; BIN=$D/${PF2C_BIN:-franken_decode_glm_pf2c}
O=${PF2C_OUT:-$HOME/bench/franken/glm5/pf2c}; mkdir -p "$O"
M=${PF2C_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
EG=${PF2C_EG:-16.85}; P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
ARMS=${PF2C_ARMS:-"256 16 16 256"}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --glm-help-copy 1 --gemm-lds 3"
say_end() { echo "=== glm_pf2c exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$M" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf2c start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) model=$(basename "$M") expert-gb=$EG arms=[$ARMS]"
rig_quiet_wait 1800 || say_end 3
dk() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
         -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
         -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
         -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full "$@"; }
i=0
for A in $ARMS; do
  R=${A%%:*}; S=0; case "$A" in *:*) S=${A##*:};; esac
  i=$((i + 1)); T="a${i}_r${R}_s${S}"
  { echo "w0 --tokens-file $PR $BASE --gemm-lds-min-rows $R --gemm-lds-splitk $S --time-prefill 4096"
    echo "p  --tokens-file $PR $BASE --gemm-lds-min-rows $R --gemm-lds-splitk $S --time-prefill 8192"
    echo "r  --tokens-file $REC $BASE --gemm-lds-min-rows $R --gemm-lds-splitk $S --time-prefill 8192"
    echo "q  --tokens-file $PR $BASE --gemm-lds-min-rows $R --gemm-lds-splitk $S --time-prefill 4096 --profile"; } > "$O/plan_$T.txt"
  printf 'w0 none\np none\nr none\nq none\n' > "$O/checks_$T.txt"
  echo "=== arm $i: --gemm-lds-min-rows $R --gemm-lds-splitk $S ($T) $(date +%T)"
  dk "$BIN" --model "$M" --placement "$P" --expert-gb "$EG" --gate-plan "$O/plan_$T.txt" > "$O/gate_$T.log" 2>&1
  rc=$?
  echo "[$T] engine rc=$rc"
  grep -aE "vram_dev[0-9]_steady|HIP error|Memory access|out of memory|FATAL|gemm_lds_min_rows" "$O/gate_$T.log" | cut -c1-200 | sort -u | head -8
  [ "$rc" -ne 0 ] && { echo "--- engine log tail ($T)"; tail -n 8 "$O/gate_$T.log" | cut -c1-200; }
done
echo "=== prefill ms/token (8 192 tokens, first chunk included), per arm (separate processes, order as run)"
printf '%-10s %-12s %-12s %s\n' "minrows:sk" "prose pf" "tech pf" "profile classes (us/token, cards 0/1/2): trunk_kda trunk_hc norm trunk_ffn trunk_idx"
i=0
for A in $ARMS; do
  R=${A%%:*}; S=0; case "$A" in *:*) S=${A##*:};; esac
  i=$((i + 1)); T="a${i}_r${R}_s${S}"; L="$O/gate_$T.log"
  pp=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "p" { split($5, a, "="); print a[2]; exit }' "$L")
  pr=$(awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ && c == "r" { split($5, a, "="); print a[2]; exit }' "$L")
  cl=$(awk '/^=== gate-plan config/ { c = $4 } c == "q" && /^prof_glm5_trunk_kda_us=/ { split($1, a, "="); k = k " " a[2] } c == "q" && /^prof_glm5_trunk_hc_us=/ { split($1, a, "="); h = h " " a[2] } c == "q" && /^prof_norm_us=/ { split($1, a, "="); n = n " " a[2] }
        c == "q" && /^prof_glm5_trunk_ffn_us=/ { split($1, a, "="); f = f " " a[2] } c == "q" && /^prof_glm5_trunk_indexer_us=/ { split($1, a, "="); x = x " " a[2] }
        END { printf "kda[%s ] hc[%s ] norm[%s ] ffn[%s ] idx[%s ]", k, h, n, f, x }' "$L")
  printf '%-10s %-12s %-12s %s\n' "$R:$S" "${pp:-FAIL}" "${pr:-FAIL}" "$cl"
done
say_end 0
