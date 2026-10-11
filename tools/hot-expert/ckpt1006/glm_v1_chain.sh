#!/bin/bash
# glm_v1_chain.sh -- V1 (ACTION-LIST row 3, DECODE-OPEN-ITEMS-PLAN section 8): an external reference past 2 051 tokens for the FULL model.
# Why: the deep references (`gpu5/g2200`, `ref512_keep`) are the engine's own GPU dumps (`g2200` = layers 0-3 only); nothing tied the GPU past the indexer's top-k
#  threshold (2 048 keys) to the CPU arm, which computes every row separately with its own kernels. Here the CPU arm runs the served hybrid on 2 300 prose tokens at
#  chunk 1 (+ 8 greedy) and writes (a) the last row's taps (--dump) and (b) its routed expert set at EVERY position and layer (--routing-out, franken-engine branch
#  v1-routing-out, merged to main); every GPU configuration is then held to both: taps by cosine (CPU and GPU kernels differ, so not bit-exact: the 6-token gate is 1 786 of 1 788
#  not bit-exact at cos 1), expert sets per position (`routing exact_sets`, the count past 2 048, the first position that differs), greedy ids. The exact-path
#  chunks (--gemm-lds 0) are also held bit for bit to the GPU chunk-1 dump; the shipped chunks (--gemm-lds 3, f16 trunk GEMM, PF2) are not bit-exact by design.
# Stages: 1 = CPU arm speed on 128 tokens (stop if 2 300 would take more than V1_MAX_H hours); 2 = the CPU reference; 3 = the GPU configurations, ONE process each
#  (each comparison is the first config of a fresh process): c1 (shipped, chunk 1, dumps g_c1), c1024x / c512x (exact path vs g_c1 and vs the CPU sets),
#  c1024 / c512 (shipped vs the CPU), c1024big (shipped, the served 262 144-cell window); 4 = the follow-up of stage 3 (the first routing flip, the GPU's own sets).
# Gate (plan section 8): a table (config, taps, not exact, min cos, sets) in the record; a non-exact tap where exactness is expected is a bug until explained.
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_v1_chain.sh > ~/bench/glm_v1_chain.log 2>&1 < /dev/null &
# Env: V1_GPU, V1_CPU (binaries), V1_OUT, V1_ONLY (1,2,3; 4 on demand), V1_N (2300), V1_THREADS (8, the physical cores, as every CPU reference), V1_MAX_H (14).
set -u
. "$HOME/bench/chain_preflight.sh"
D=$HOME/src/franken-engine/franken/decode   # build first (main has --routing-out since 0753b05): make gpu GPU_BIN=franken_decode_glm_v1; make cpu CPU_BIN=franken_decode_glm_v1_cpu
GPU=${V1_GPU:-$D/franken_decode_glm_v1}; CPU=${V1_CPU:-$D/franken_decode_glm_v1_cpu}
O=${V1_OUT:-$HOME/bench/franken/glm5/v1}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
N=${V1_N:-2300}; TH=${V1_THREADS:-8}; MAXH=${V1_MAX_H:-14}
ONLY=${V1_ONLY:-1,2,3}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
say_end() { echo "=== glm_v1 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$GPU" "$CPU" "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_v1 start $(date -Is) gpu=$(sha256sum "$GPU" | cut -c1-16) cpu=$(sha256sum "$CPU" | cut -c1-16) stages=$ONLY n=$N"
rig_quiet_wait 1800 || say_end 3
REF=$O/cpu$N
# the oracle lines -> one summary: taps, not bit-exact, min cos and its tap, index sets not fully contained
summ() {
  awk '/^oracle [^ ]+ cos=/{n++; c=substr($3,5)+0; m=substr($4,8)+0; if (m!=0) ne++; if (!s || c<mc) {mc=c; wk=$2; s=1}}
       /^oracle [^ ]+ contained=/{ix++; if (substr($3,11)+0<1) ixb++} /^oracle [^ ]+ exact=/{ix++; if (substr($3,7)+0!=1) ixb++}
       END{printf "taps=%d not_bitexact=%d min_cos=%.6f worst=%s index_sets=%d index_not_full=%d\n", n, ne, mc, wk, ix, ixb}' "$1"
  grep -aE "^greedy_ids|^routing exact_sets|^routing deep_|ORACLE|^oracle summary|^prompt_tokens=" "$1" | cut -c1-200
}
rc_all=0
if has 1; then
  echo "=== 1. CPU arm speed, 128 tokens, chunk 1, $TH threads $(date +%T)"
  "$CPU" --model "$M" --tokens-file "$PR" --max-tokens 128 --cpu --ctx 4096 --threads "$TH" --chunk 1 > "$O/cpu_speed.log" 2>&1
  rc=$?; echo "cpu rc=$rc"; grep -aE "^prompt_tokens=" "$O/cpu_speed.log"
  [ "$rc" -ne 0 ] && { tail -n 10 "$O/cpu_speed.log" | cut -c1-200; say_end "$rc"; }
  ms=$(grep -aoE "prompt_ms=[0-9.]+" "$O/cpu_speed.log" | cut -d= -f2)
  proj_h=$(awk -v ms="$ms" -v n="$N" 'BEGIN{printf "%.1f", ms / 128 * (n + 8) / 3600000}')
  echo "cpu_ms_per_token=$(awk -v ms="$ms" 'BEGIN{printf "%.0f", ms/128}') projected_hours_for_$N=$proj_h (limit $MAXH)"
  awk -v p="$proj_h" -v l="$MAXH" 'BEGIN{exit !(p > l)}' && { echo "STOP: the CPU reference would take ${proj_h} h"; say_end 4; }
fi
if has 2; then
  echo "=== 2. CPU reference: $N tokens + 8 greedy, chunk 1 $(date +%T)"
  "$CPU" --model "$M" --tokens-file "$PR" --max-tokens "$N" --cpu --ctx 4096 --threads "$TH" --chunk 1 --greedy 8 --dump "$REF" --routing-out "$REF" > "$REF.log" 2>&1
  rc=$?; echo "cpu rc=$rc $(date +%T)"; grep -aE "^prompt_tokens=|^greedy_ids|^dumped|^routing_out" "$REF.log" | cut -c1-200
  [ "$rc" -ne 0 ] && { tail -n 10 "$REF.log" | cut -c1-200; say_end "$rc"; }
fi
if has 3; then
  [ -f "$REF/index.txt" ] && [ -f "$REF/moe_ids.txt" ] || { echo "FATAL: no CPU reference in $REF"; say_end 2; }
  # the shipped decode/prefill flags (the launcher's FRANKEN_* switches as CLI flags; the MoE group size is read from the environment)
  SHIP="--tokens-file $PR --max-tokens $N --greedy 8 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --fetch-assign optimal \
 --gemv-fused-reduce 0 --glm-gemv-group 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1"
  gpu() {   # name, then the configuration's own options (after the shipped ones: a later option wins)
    local n=$1; shift
    echo "$n $SHIP $*" > "$O/plan_$n.txt"
    echo "=== 3. $n $(date +%T): $*"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
      "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan_$n.txt" > "$O/$n.log" 2>&1
    local rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && rc_all=$rc
    summ "$O/$n.log"
  }
  gpu c1       --ctx 4096 --chunk 1 --oracle "$REF" --routing "$REF" --dump "$O/g_c1"
  gpu c1024x   --ctx 4096 --chunk 1024 --gemm-lds 0 --oracle "$O/g_c1" --routing "$REF"
  gpu c512x    --ctx 4096 --chunk 512 --gemm-lds 0 --oracle "$O/g_c1" --routing "$REF"
  gpu c1024    --ctx 4096 --chunk 1024 --oracle "$REF" --routing "$REF"
  gpu c512     --ctx 4096 --chunk 512 --oracle "$REF" --routing "$REF"
  gpu c1024big --ctx 262144 --chunk 1024 --oracle "$REF" --routing "$REF"
  echo "CPU greedy: $(grep -a "^greedy_ids" "$REF.log")"
fi
if has 4; then
  # stage 4 (added after stage 3: the GPU's expert sets matched the CPU's at 75 % of (position, layer) pairs, first difference at position 40 layer 42,
  # agreement falling smoothly with layer depth, layers 0-16 cos 1 at position 2 299): is that rounding noise compounded by discrete routing flips, or a bug?
  # (a) the first flip: both arms dumped at position 40 (a 41-token prompt), the router's biased scores of layer 42 -> the 8th/9th gap, offline;
  # (b) the GPU's own per-position sets (chunk 1, and the shipped chunk 1024 whose f16 trunk GEMM is a known-benign perturbation, PF2) for agreement curves by position.
  F=${V1_FLIP:-41}
  echo "=== 4a. CPU arm, $F tokens $(date +%T)"
  "$CPU" --model "$M" --tokens-file "$PR" --max-tokens "$F" --cpu --ctx 4096 --threads "$TH" --chunk 1 --dump "$O/cpu$F" > "$O/cpu$F.log" 2>&1
  echo "cpu rc=$?"; grep -aE "^prompt_tokens=|^dumped" "$O/cpu$F.log"
  SHIP="--tokens-file $PR --max-tokens $N --greedy 8 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --fetch-assign optimal \
 --gemv-fused-reduce 0 --glm-gemv-group 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1"
  gpu4() {
    local n=$1; shift
    echo "$n $SHIP $*" > "$O/plan_$n.txt"
    echo "=== 4. $n $(date +%T): $*"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
      "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan_$n.txt" > "$O/$n.log" 2>&1
    local rc=$?; echo "engine rc=$rc"
    grep -aE "^prompt_tokens=|^dumped|^routing_out|^greedy_ids|^routing exact_sets|^routing deep_" "$O/$n.log" | cut -c1-200
  }
  gpu4 flip    --ctx 4096 --chunk 1 --max-tokens "$F" --greedy 0 --oracle "$O/cpu$F" --dump "$O/g$F"
  gpu4 r_c1    --ctx 4096 --chunk 1 --routing "$REF" --routing-out "$O/r_c1"
  gpu4 r_c1024 --ctx 4096 --chunk 1024 --routing "$REF" --routing-out "$O/r_c1024"
fi
say_end "$rc_all"
