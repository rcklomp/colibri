#!/bin/bash
# glm_d16_chain.sh -- D16 (ACTION-LIST row 2, DECODE-OPEN-ITEMS-PLAN section 8): the per-token `ms` vector of decode, in the timed loop and in the serving loop.
# Why: `--time` and `[serve-glm5]` printed only a median (the timed loop) and a request mean (the server), which hide a stall that hits under half the tokens; the kernel traces
#  (record §L5-GLM-D2) saw 9-12 tokens a run with a 24-45 ms host gap between the argmax read-back and the next token's first kernel, and D6's disturbed serve-A/B arms had cards
#  that waited. franken-engine branch d16-token-ms prints, per decode token, the wall and the host phases of the step (adapter tick, embedding rows, issue, the argmax download =
#  the wait for the cards) and `out` = the loop's time outside step() (sampling, the wire): `glm5_token_ms ...` (summary), `glm5_token_ms_vec=...`, `glm5_token_gap ...` (every
#  token more than 5 ms over the median). The server prints the summary per request, the vector with FRANKEN_GLM_TOKEN_MS=1 (the wrapper allowlist carries it).
# Stages: 1 = three separate CLI processes, the shipped flags, 8 192 prose tokens of prefill, 64 untimed settle tokens, 200 timed tokens (plan gate: the vector for >= 3 runs);
#         2 = the serve path through the real gateway: glm_serve_ab_chain.sh with both arms on the D16 build and the vector on (W1 a long document, then four short chats).
# Gate (plan section 8): if no token in 200 is more than 5 ms over the median, D17 is dropped; otherwise the gap tokens' phases say where the host spends it.
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d16_chain.sh > ~/bench/glm_d16_chain.log 2>&1 < /dev/null &
# Env: D16_GPU (CLI binary), D16_SERVE (serving binary), D16_OUT, D16_ONLY (1,2), D16_TIME (200), D16_RUNS (3).
set -u
. "$HOME/bench/chain_preflight.sh"
D=$HOME/src/franken-engine/franken/decode
GPU=${D16_GPU:-$D/franken_decode_glm_d16}; SRV=${D16_SERVE:-$HOME/bench/franken_bin/franken_dec_glm.d16}
O=${D16_OUT:-$HOME/bench/franken/glm5/d16}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
N=${D16_TIME:-200}; RUNS=${D16_RUNS:-3}
ONLY=${D16_ONLY:-1,2}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
say_end() { echo "=== glm_d16 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$GPU" "$SRV" "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_d16 start $(date -Is) gpu=$(sha256sum "$GPU" | cut -c1-16) serve=$(sha256sum "$SRV" | cut -c1-16) stages=$ONLY"
rig_quiet_wait 1800 || say_end 3
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMM_LDS_MIN_ROWS=16 -e FRANKEN_GEMM_LDS_SPLITK=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal \
 -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
rc_all=0
if has 1; then
  echo "t --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 --time-prefill 8192 --time $N --time-settle 64" > "$O/plan.txt"
  for r in $(seq 1 "$RUNS"); do
    echo "=== 1. timed loop, process $r $(date +%T)"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
      "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/cli_$r.log" 2>&1
    rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && { rc_all=$rc; tail -n 10 "$O/cli_$r.log" | cut -c1-200; continue; }
    grep -aE "^prefill_tokens=|glm5_time_settle|glm5_decode_ms_median|glm5_token_ms |glm5_token_gap" "$O/cli_$r.log" | cut -c1-330
  done
fi
if has 2; then
  echo "=== 2. serve path $(date +%T)"
  SAB_OLD=$SRV SAB_NEW=$SRV SAB_ARMS="new new" SAB_NEW_ENV="FRANKEN_GLM_TOKEN_MS=1" SAB_OUT=$O/serve bash "$HOME/bench/glm_serve_ab_chain.sh"
  rc=$?; echo "serve chain rc=$rc"; [ "$rc" -ne 0 ] && rc_all=$rc
  for f in "$O"/serve/*.log; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f")"; grep -aE "token_ms |token_gap" "$f" | sed 's/^.*\[serve-glm5\] //' | cut -c1-330 | head -40
  done
fi
if has 3; then
  # stage 3 (added after stage 1 found the gaps at every 16th token = the adapter's swap round, its copies a burst on the links the next token's fetch uses):
  # the existing knob first -- --adapt-every N (smaller, more frequent bursts, the same per-token byte budget), one process per value, the MEAN token ms is the measure
  # (the median hides the bursts); hit rate and swap MB from the adapt_total lines.
  for ev in ${D16_EVERY:-16 4 8 16}; do
    sed "s/--time-settle 64/--time-settle 64 --adapt-every $ev/" "$O/plan.txt" > "$O/plan_ev$ev.txt"
    echo "=== 3. adapt-every $ev $(date +%T)"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
      "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan_ev$ev.txt" > "$O/ev${ev}_$(date +%H%M%S).log" 2>&1
    rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && rc_all=$rc
    f=$(ls -t "$O"/ev${ev}_*.log | head -1)
    grep -aE "glm5_token_ms |glm5_decode_ms_median|adapt_total" "$f" | cut -c1-260
  done
fi
if has 4; then
  # stage 4 (stage 3 found that the swap copies cost what their bytes cost on the links, burst or spread): the byte budget itself, --adapt-mb-per-token X
  # (shipped 64) against the miss rate it buys; one process per value, cadence 16, the mean token ms is the measure.
  for mb in ${D16_MB:-64 16 32 128 64}; do
    sed "s/--time-settle 64/--time-settle 64 --adapt-mb-per-token $mb/" "$O/plan.txt" > "$O/plan_mb$mb.txt"
    echo "=== 4. adapt-mb-per-token $mb $(date +%T)"
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
      "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan_mb$mb.txt" > "$O/mb${mb}_$(date +%H%M%S).log" 2>&1
    rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && rc_all=$rc
    f=$(ls -t "$O"/mb${mb}_*.log | head -1)
    grep -aE "glm5_token_ms " "$f" | cut -c1-200
  done
fi
say_end "$rc_all"
