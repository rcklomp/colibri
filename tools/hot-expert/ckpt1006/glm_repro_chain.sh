#!/bin/bash
# glm_repro_chain.sh -- D5 of DECODE-OPEN-ITEMS-PLAN-2026-10-08: decode past 2 051 tokens of depth is not run-to-run reproducible (handoff 06b section 4 item 7). WHERE does it first differ?
# The oracle compares the taps of the LAST position a config processed (decode_oracle.h), and every config prints `greedy_ids:`. Prompt = the first 2 300 token ids of prose8400.txt (> 2 051), chunk 512, shipped
# decode flags with adaptation OFF (no placement swaps as a variable). A ladder of greedy lengths k (the last position = decode step k): for each k a reference config with --dump, then
#   PROCESS 1 (one load): r_k (--dump $O/d_k), x_k (--oracle $O/d_k)   = the same process twice
#   PROCESS 2 (a fresh load): y_k (--oracle $O/d_k from process 1)      = a different process
# plus a CONTROL at depth 1 500 (< 2 051, known exact): c_r / c_x / c_y with greedy 8.
# Report: ckpt1006/repro_report.py <O> (greedy_ids equality per k and the first non-exact taps by layer). The dump dirs are ROOT-owned (docker): remove them through a container, never sudo.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_repro_chain.sh > ~/bench/glm_repro_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_repro_chain.log ~/bench/franken/glm5/repro/p1_run.log
# Env: RP_BIN (default ~/bench/franken_bin/franken_decode_glm_d3), RP_OUT (default ~/bench/franken/glm5/repro), RP_KS (default "1 4 8 12 16 24"), RP_DEPTH (default 2300), RP_EXTRA (more flags for every config).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${RP_BIN:-$HOME/bench/franken_bin/franken_decode_glm_d3}
O=${RP_OUT:-$HOME/bench/franken/glm5/repro}
say_end() { echo "=== glm_repro exit rc=$1 $(date -Is)"; exit "$1"; }
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
KS=${RP_KS:-1 4 8 12 16 24}; DEPTH=${RP_DEPTH:-2300}
FLAGS="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1 --adapt 0 ${RP_EXTRA:-}"
echo "=== glm_repro start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) depth=$DEPTH ks=[$KS] flags=[$FLAGS]"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
[ -f "$B/prose8400.txt" ] || { echo "FATAL: prose8400.txt missing"; say_end 2; }
if [ -d "$O" ]; then docker run --rm -v "$(dirname "$O")":/w rocm/dev-ubuntu-24.04:7.14.0-full rm -rf "/w/$(basename "$O")" >/dev/null 2>&1; fi
mkdir -p "$O"
tr ' ' '\n' < "$B/prose8400.txt" | grep -v '^$' | head -$DEPTH | tr '\n' ' ' > "$O/prompt_deep.txt"
tr ' ' '\n' < "$B/prose8400.txt" | grep -v '^$' | head -1500 | tr '\n' ' ' > "$O/prompt_ctl.txt"
echo "prompt ids: deep=$(wc -w < "$O/prompt_deep.txt") control=$(wc -w < "$O/prompt_ctl.txt")"
DEEP="--tokens-file $O/prompt_deep.txt --ctx 4096 --chunk 512"; CTL="--tokens-file $O/prompt_ctl.txt --ctx 4096 --chunk 512"
: > "$O/p1_plan.txt"; : > "$O/p2_plan.txt"
for k in $KS; do
  echo "r_$k $DEEP --greedy $k $FLAGS --dump $O/d_$k" >> "$O/p1_plan.txt"
  echo "x_$k $DEEP --greedy $k $FLAGS --oracle $O/d_$k" >> "$O/p1_plan.txt"
  echo "y_$k $DEEP --greedy $k $FLAGS --oracle $O/d_$k" >> "$O/p2_plan.txt"
done
echo "c_r $CTL --greedy 8 $FLAGS --dump $O/d_ctl" >> "$O/p1_plan.txt"
echo "c_x $CTL --greedy 8 $FLAGS --oracle $O/d_ctl" >> "$O/p1_plan.txt"
echo "c_y $CTL --greedy 8 $FLAGS --oracle $O/d_ctl" >> "$O/p2_plan.txt"
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
run_plan() {   # name plan log
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$2" > "$3" 2>&1
}
run_plan p1 "$O/p1_plan.txt" "$O/p1_run.log"; rc1=$?
echo "process 1 rc=$rc1 $(date -Is)"; grep -aE "HIP error|Memory access|out of memory" "$O/p1_run.log" | head -2 | cut -c1-170
[ $rc1 -ne 0 ] && { echo "process 1 failed; not starting process 2"; tail -n 5 "$O/p1_run.log" | cut -c1-200; say_end $rc1; }
rig_quiet_wait 600 || { echo "FATAL: rig not quiet before process 2"; say_end 3; }
run_plan p2 "$O/p2_plan.txt" "$O/p2_run.log"; rc2=$?
echo "process 2 rc=$rc2 $(date -Is)"; grep -aE "HIP error|Memory access|out of memory" "$O/p2_run.log" | head -2 | cut -c1-170
python3 -I "$HOME/bench/repro_report.py" "$O" 2>&1 | cut -c1-240
say_end $((rc1 + rc2))
