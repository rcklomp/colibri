#!/bin/bash
# glm_d18_chain.sh -- D18 READ part 2 (ACTION-LIST row 5j, DECODE-OPEN-ITEMS-PLAN section 9): how busy is each card, and each card's host link, within one
# decode token of the SHIPPED build? A depth-1 pipeline (Project Maya) overlaps two tokens; no schedule can make a token cheaper than the busiest resource's
# work for it, so if one card's kernels (all streams together) or its copies already fill most of a token, the pipeline cannot reach the 15 % bar.
# Stages: 1 = build the CLI from the rig's franken-engine main (franken_decode_glm_d18, not installed);
#         2 = one untraced timed run (W at the shipped flags, the same plan as the trace);
#         3 = rocprofv3 kernel + memory-copy trace of N decode tokens after a settle (timings under tracing are inflated; ratios are what counts).
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d18_chain.sh > ~/bench/glm_d18_chain.log 2>&1 < /dev/null &
# Report: python3 ckpt1006/d18_busy_report.py ~/bench/franken/glm5/d18/trace
# Env: D18_OUT, D18_ONLY (1,2,3), D18_TOKENS (16).
set -u
. "$HOME/bench/chain_preflight.sh"
D=$HOME/src/franken-engine/franken/decode
GPU=$D/franken_decode_glm_d18
O=${D18_OUT:-$HOME/bench/franken/glm5/d18}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
N=${D18_TOKENS:-16}
ONLY=${D18_ONLY:-1,2,3}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
say_end() { echo "=== glm_d18 exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d18 start $(date -Is) stages=$ONLY engine=$(git -C "$HOME/src/franken-engine" log --oneline -1)"
rig_quiet_wait 1800 || say_end 3
if has 1; then
  echo "=== 1. build $(date +%T)"
  ( cd "$D" && nice -n 19 make -j8 gpu GPU_BIN=franken_decode_glm_d18 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -5 )
fi
[ -x "$GPU" ] || { echo "FATAL: no $GPU"; say_end 2; }
for f in "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "gpu=$(sha256sum "$GPU" | cut -c1-16)"
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMM_LDS_MIN_ROWS=16 -e FRANKEN_GEMM_LDS_SPLITK=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal \
 -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
DKO="--rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench"
IMG=rocm/dev-ubuntu-24.04:7.14.0-full; DK="docker run $DKO $IMG"
# the decode settings the launcher passes as FRANKEN_* env reach only the SERVER; the CLI takes them as flags (an env here is ignored: the first run of this
# chain printed gemv_rowsplit=0 fetch_assign=pattern)
DEC="--fetch-assign optimal --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1"
# 2 048 prompt tokens (two chunks: the scratch is at its steady size), 48 settle tokens (the adapter has run three swap rounds), N timed
echo "t --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 $DEC --time-prefill 2048 --time $N --time-settle 48" > "$O/plan.txt"
rc_all=0
if has 2; then
  echo "=== 2. untraced timed run $(date +%T)"
  $DK "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/untraced.log" 2>&1
  rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && { rc_all=$rc; tail -n 10 "$O/untraced.log" | cut -c1-200; }
  grep -aE "glm5_decode_ms_median|glm5_token_ms |free|VRAM|vram" "$O/untraced.log" | cut -c1-250 | head -12
fi
if has 3; then
  echo "=== 3. traced run $(date +%T)"
  rm -rf "$O/trace"; mkdir -p "$O/trace"
  $DK /opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d "$O/trace/out" -o prof -- \
    "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/trace/run.log" 2>&1
  rc=$?; echo "traced rc=$rc"; [ "$rc" -ne 0 ] && { rc_all=$rc; tail -n 10 "$O/trace/run.log" | cut -c1-200; }
  grep -aE "glm5_decode_ms_median|glm5_token_ms " "$O/trace/run.log" | cut -c1-250
  find "$O/trace/out" -name "*.csv" | head; du -sh "$O/trace/out"
fi
# Stage 4 (two decode processes at once, for the stretch factor) was REMOVED after it hung the rig on 2026-10-12 14:01 UTC: each engine process holds a
# ~110 GB host mirror of the model (host_mirror= in its log) and the rig has 247 GB, so two processes exhaust host RAM. Never run two engine processes at once.
say_end $rc_all
