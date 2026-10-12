#!/bin/bash
# glm_d18_probe_chain.sh -- D18 part 3 (DECODE-OPEN-ITEMS-PLAN section 9): the stretch factor s, measured. franken-engine branch d18-probe,
# `--probe-two-seq N` (GLM5.md section 30): two runners on ONE loaded model, each with its own backends / streams / scratch / rings, issued
# from two host threads; windows A alone, B alone, A + B at once, A alone again; s = A's median beside B / A's median alone.
# ONE engine process at a time (two processes exhaust host RAM, record §L5-D18-PART2): the two runs below are sequential.
# Run 1 as is; run 2 with GPU_MAX_HW_QUEUES=8 (ten streams a card may share four hardware queues by default, which would inflate s;
# 16 HUNG in the concurrent window on 2026-10-12, hence the watchdog).
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d18_probe_chain.sh > ~/bench/glm_d18_probe_chain.log 2>&1 < /dev/null &
# Env: PROBE_BIN, PROBE_OUT, PROBE_N (64), PROBE_EGB (15.5), PROBE_RUNS ("default hwq8"; hwqN = GPU_MAX_HW_QUEUES=N), PROBE_LIMIT_S (420, a run).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PROBE_BIN:-$HOME/src/franken-engine-d18/franken/decode/franken_decode_glm_d18p}
O=${PROBE_OUT:-$HOME/bench/franken/glm5/d18probe}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; N=${PROBE_N:-64}; EGB=${PROBE_EGB:-15.5}
say_end() { echo "=== glm_d18_probe exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] || { echo "FATAL: no $BIN"; say_end 2; }
echo "=== glm_d18_probe start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) N=$N expert_gb=$EGB"
rig_quiet_wait 1800 || say_end 3
# 4 096 prose tokens (the file holds token ids): A continues the first 2 048, B the second
head -c 1000000 "$HOME/bench/franken/glm5/prose8400.txt" | tr -s ' \n' '\n' | grep -v '^$' | head -4096 | tr '\n' ' ' > "$O/prompt4096.txt"
echo "t --tokens-file $O/prompt4096.txt --ctx 8192 --chunk 256 --adapt 0 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 \
--fetch-assign optimal --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1 --time-settle 32 --probe-two-seq $N" > "$O/plan.txt"
rc_all=0
for run in ${PROBE_RUNS:-default hwq8}; do
  EXTRA=""; case "$run" in hwq*) EXTRA="-e GPU_MAX_HW_QUEUES=${run#hwq}";; esac
  # a watchdog: the 2026-10-12 run with 16 hardware queues hung in the concurrent window (two threads spinning, the GPUs silent)
  docker rm -f d18probe >/dev/null 2>&1
  ( sleep "${PROBE_LIMIT_S:-420}"; docker kill d18probe >/dev/null 2>&1 && echo "WATCHDOG: run $run killed after ${PROBE_LIMIT_S:-420} s" ) &
  WD=$!
  echo "=== run $run $(date +%T)"
  docker run --rm --name d18probe --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $EXTRA \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model "$M" --placement "$P" --expert-gb "$EGB" --gate-plan "$O/plan.txt" > "$O/$run.log" 2>&1
  rc=$?; kill $WD 2>/dev/null; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && rc_all=$rc
  grep -aE "probe2|after caches|both runners|rror|abort" "$O/$run.log" | cut -c1-300 | head -24
done
say_end $rc_all
