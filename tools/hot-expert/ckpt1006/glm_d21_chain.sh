#!/bin/bash
# glm_d21_chain.sh -- D21 (CLOSED 2026-10-12, record §L5-D21: no gain; the baseline trace it reads was removed with the item) (DECODE-OPEN-ITEMS-PLAN section 10, record §L5-D15): is card 0's decode helper on time again? Card 0 started its stage kernel ~350 us after the owner's
# plan in every 2026-10-12 trace (on time, ~20 us, through 2026-10-08); suspect the OverDrive reboot of 2026-10-11. Run AFTER `fan_od.sh disable` + reboot.
# 1. builds the CLI from the rig's franken-engine main (franken_decode_glm_d21; not installed); 2. one rocprofv3 trace of the served hybrid at the shipped decode flags
# (2 048 prompt tokens, 48 settle, 24 traced; the D15 arms' exact flags); 3. d15_wake_report.py on it and on the kept baseline ~/bench/franken/glm5/d21_baseline_trace.
# Launch (rig idle, service OFF): setsid nohup ~/src/colibri/tools/hot-expert/ckpt1006/glm_d21_chain.sh > ~/bench/glm_d21_chain.log 2>&1 < /dev/null &
set -u
. "$HOME/bench/chain_preflight.sh"
D=$HOME/src/franken-engine/franken/decode; BIN=franken_decode_glm_d21; H=$HOME/src/colibri/tools/hot-expert/ckpt1006
O=$HOME/bench/franken/glm5/d21_trace
say_end() { echo "=== glm_d21 exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d21 start $(date -Is) engine=$(git -C "$HOME/src/franken-engine" log --oneline -1 | cut -c1-80) ppfeaturemask=$(cat /sys/module/amdgpu/parameters/ppfeaturemask)"
( cd "$D" && nice -n 19 make -j8 gpu GPU_BIN=$BIN 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -3 )
[ -x "$D/$BIN" ] || { echo "FATAL: no $D/$BIN"; say_end 2; }
rig_quiet_wait 600 || say_end 3
TRACE_BIN="$D/$BIN" TRACE_OUT="$O" TRACE_TOKENS=24 TRACE_SETTLE=48 TRACE_EG=16.85 \
TRACE_MODEL=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf \
TRACE_EXTRA="--adapt-prefill-cap 64 --adapt-halflife 512 --fetch-assign optimal --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1" \
  "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_decode_trace_chain.sh"; rc=$?
[ $rc -ne 0 ] && say_end $rc
docker run --rm -v /home/ronald/bench/franken/glm5:/w rocm/dev-ubuntu-24.04:7.14.0-full chown -R 1000:1000 /w/d21_trace >/dev/null 2>&1
for t in "$HOME/bench/franken/glm5/d21_baseline_trace/out" "$O/out"; do
  echo "--- $t"; python3 -I "$H/d15_wake_report.py" "$t" --tokens 24 | grep -E "helper swap copy in flight=False|upper bound"
done
grep -a -o "glm5_decode_ms_median=[0-9.]*" "$O/gate_run.log"
rm -f "$D/$BIN"
say_end 0
