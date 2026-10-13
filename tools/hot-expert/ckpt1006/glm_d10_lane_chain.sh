#!/bin/bash
# The d10l binaries were deleted after the install (2026-10-13); to rerun, build main's lane CLI/serve with ckpt1006/build_d10adapt.sh and pass the path (edit GPU=).
# glm_d10_lane_chain.sh -- D10 (ACTION-LIST row 8): the CPU lane, first GPU runs. One process: a check arm (--cpu-lane-check 1: CPU vs GPU row rel-L2) then lane off,on,on,off at the §L5-D21 plan and shipped flags.
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d10_lane_chain.sh > ~/bench/glm_d10_lane_chain.log 2>&1 < /dev/null &
# Env: D10L_OUT, D10L_TOKENS (32), D10L_ARMS ("check 0 6 6 0": check = --cpu-lane 6 --cpu-lane-check 1, N = --cpu-lane N).
set -u
. "$HOME/bench/chain_preflight.sh"
GPU=$HOME/bench/franken_bin/franken_decode_glm_d10l
O=${D10L_OUT:-$HOME/bench/franken/glm5/d10l}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
N=${D10L_TOKENS:-32}
ARMS=${D10L_ARMS:-check 0 6 6 0}
say_end() { echo "=== glm_d10l exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d10l start $(date -Is) arms=[$ARMS] N=$N"
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
  case $k in check) L="--cpu-lane 6 --cpu-lane-check 1";; *) L="--cpu-lane $k";; esac
  echo "t --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 $DEC --time-prefill 2048 --time $N --time-settle 48 $L" >> "$O/plan.txt"
done
echo "=== untraced timed arms $(date +%T)"
$DK "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/untraced.log" 2>&1
rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && tail -n 15 "$O/untraced.log" | cut -c1-200
grep -aE "glm5_decode_ms_median|glm5_token_ms |\[cpu-lane|REFUSED|FATAL" "$O/untraced.log" | cut -c1-250
say_end $rc
