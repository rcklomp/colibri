#!/bin/bash
# glm_q6k_stall_chain.sh -- D4 of DECODE-OPEN-ITEMS-PLAN-2026-10-08: which launch of the Q6_K split case (bench_rowsplit `q6k_fallback`: 64 rows x K 4096, nsplit 16) hangs/faults the card?
# bench_q6k_stall (franken/decode/bench_q6k_stall.hip) runs ONE variant per process (gemm / gemm_nostage / reduce / pair / api_this / api_main) with an in-process watchdog and prints a RESULT line;
# this chain runs a list of variants, one docker container each (docker-kill backstop), counts new amdgpu `sq_intr` lines in the kernel log after each (type 2 = a wave memory violation),
# and runs a known-good control (gemm 64x4096 nsplit 1, the unsplit Q6_K head's shape) after every variant that did not end OK so a wedged card is noticed at once.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <out log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_q6k_stall_chain.sh > ~/bench/glm_q6k_stall_chain.log 2>&1 < /dev/null &
# Env: QS_BIN (default ~/bench/franken_bin/bench_q6k_stall), QS_OUT (default ~/bench/franken/glm5/q6k_stall), QS_LIST (override the variant list: lines 'VARIANT rows K nsplit', separated by ';'),
#      QS_ENV (extra docker -e options, e.g. "-e AMD_SERIALIZE_KERNEL=3 -e HIP_LAUNCH_BLOCKING=1"), QS_WD (per-variant watchdog seconds, default 20).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${QS_BIN:-$HOME/bench/franken_bin/bench_q6k_stall}
O=${QS_OUT:-$HOME/bench/franken/glm5/q6k_stall}; mkdir -p "$O"; : > "$O/results.txt"
WD=${QS_WD:-20}
say_end() { echo "=== glm_q6k_stall exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_q6k_stall start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) env=${QS_ENV:-}"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
DEFAULT_LIST="gemm 64 4096 1;pair 64 4096 1;gemm 64 4096 2;gemm 64 4096 4;gemm 64 4096 8;gemm 64 4096 16;gemm_nostage 64 4096 16;reduce 64 4096 16;pair 64 4096 16;gemm 64 8192 16;gemm 1024 4096 16;gemm 64 4096 32;api_this 64 4096 0;api_main 64 4096 0"
LIST=${QS_LIST:-$DEFAULT_LIST}
nq() { journalctl -k --no-pager --since "$1" 2>/dev/null | grep -c 'sq_intr' ; }
IFS=';' read -r -a VARS <<< "$LIST"
bad=0; i=0
for v in "${VARS[@]}"; do
  i=$((i+1)); set -- $v; var=$1; rows=$2; K=$3; ns=$4
  t0=$(date '+%Y-%m-%d %H:%M:%S'); name="q6kstall_$i"
  docker rm -f "$name" >/dev/null 2>&1
  ( sleep $((WD + 25)); docker kill "$name" >/dev/null 2>&1 && echo "BACKSTOP: container $name killed" ) &
  BK=$!
  docker run --name "$name" --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 ${QS_ENV:-} \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" "$var" "$rows" "$K" "$ns" "$WD" > "$O/v$i.log" 2>&1
  rc=$?
  kill $BK 2>/dev/null; sleep 2
  sq=$(nq "$t0")
  res=$(grep -a '^RESULT' "$O/v$i.log" | tail -1 | cut -c1-200)
  [ -n "$res" ] || res="RESULT $var (no result line; rc=$rc) $(tail -n 2 "$O/v$i.log" | tr '\n' ' ' | cut -c1-160)"
  echo "[$i] $var rows=$rows K=$K nsplit=$ns rc=$rc sq_intr=$sq : $res" | tee -a "$O/results.txt"
  case "$res" in *" OK"*) ;; *)
    bad=$((bad+1))
    # control: the unsplit Q6_K shape the head runs in production must still work, else the card is wedged
    docker rm -f q6kstall_ctl >/dev/null 2>&1
    docker run --name q6kstall_ctl --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 \
      -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
      rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" gemm 64 4096 1 20 > "$O/ctl$i.log" 2>&1
    cres=$(grep -a '^RESULT' "$O/ctl$i.log" | tail -1 | cut -c1-120)
    echo "    control after [$i]: ${cres:-no result}" | tee -a "$O/results.txt"
    case "$cres" in *" OK"*) ;; *) echo "FATAL: the card does not run the control after [$i]; stopping"; say_end 4;; esac ;;
  esac
done
echo "=== summary: $i variants, $bad not OK; kernel-log sq_intr lines are counted per variant above"
say_end 0
