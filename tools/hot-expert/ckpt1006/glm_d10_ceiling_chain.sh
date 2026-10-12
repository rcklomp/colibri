#!/bin/bash
# glm_d10_ceiling_chain.sh -- G2 / D10 sizing on the UNTRACED token (ACTION-LIST row 7, DECODE-OPEN-ITEMS-PLAN section 10; record §L5-D21 said: size a fetch
# lever on the engine's own token time, not on the traced fetch sum). The probe build `--d10-drop K` (franken-engine branch d10-ceiling-probe, NOT for
# serving: changes answers) drops the last K missed slabs of every decode layer -- not fetched, not computed: a CPU lane that costs nothing. The token's
# change at K = 1 is the CEILING of D10 (which takes about one straggler slab a layer, and is not free).
# One process, the §L5-D21 plan and flags, arms K = 0,1,2,1,0 as gate-plan lines (A,B,C,B,A), each with its own prompt/settle.
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d10_ceiling_chain.sh > ~/bench/glm_d10_ceiling_chain.log 2>&1 < /dev/null &
# Env: D10C_OUT, D10C_TOKENS (32), D10C_ARMS ("0 1 2 1 0").
set -u
. "$HOME/bench/chain_preflight.sh"
GPU=$HOME/bench/franken_bin/franken_decode_glm_d10p
O=${D10C_OUT:-$HOME/bench/franken/glm5/d10c}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
N=${D10C_TOKENS:-32}
ARMS=${D10C_ARMS:-0 1 2 1 0}
say_end() { echo "=== glm_d10c exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d10c start $(date -Is) arms=[$ARMS] N=$N"
rig_quiet_wait 1800 || say_end 3
[ -x "$GPU" ] || { echo "FATAL: no $GPU"; say_end 2; }
for f in "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "gpu=$(sha256sum "$GPU" | cut -c1-16)"
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMM_LDS_MIN_ROWS=16 -e FRANKEN_GEMM_LDS_SPLITK=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal \
 -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
DKO="--rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench"
IMG=rocm/dev-ubuntu-24.04:7.14.0-full; DK="docker run $DKO $IMG"
DEC="--fetch-assign optimal --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1"
: > "$O/plan.txt"
for k in $ARMS; do
  echo "t --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 $DEC --time-prefill 2048 --time $N --time-settle 48 --d10-drop $k" >> "$O/plan.txt"
done
echo "=== untraced timed arms $(date +%T)"
$DK "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/untraced.log" 2>&1
rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && tail -n 15 "$O/untraced.log" | cut -c1-200
grep -aE "glm5_decode_ms_median|glm5_token_ms |\[d10-drop\]" "$O/untraced.log" | cut -c1-250
say_end $rc
