#!/bin/bash
# f8_bench_run.sh -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F8 step 0.
# Runs ONLY tools/hot-expert/f8_cpu_expert_bench.c, a standalone CPU
# microbenchmark that links quant.h directly -- no engine, no Vulkan, no
# gateway downtime. The gateway stays UP and is not touched beyond reading
# its log to prove no request landed mid-pass.
#
# Launch from the Mac session with (this script runs ON THE RIG):
#   ssh -n -f rome 'cd ~/src/colibri-f8 && setsid nohup \
#       tools/hot-expert/f8_bench_run.sh >> ~/bench/f8_bench.log 2>&1 < /dev/null &'
# then poll ~/bench/f8_bench.log / ~/bench/f8_pass{1,2}.txt with bounded ssh
# loops (sleep >= 60s between polls). Total wall budget <= 15 min.
#
# Safety: no sudo, no cache drop, no gateway stop/restart. Refuses to run
# under 20 GiB MemAvailable. Runs the whole bench TWICE; if the served
# gateway's own request count moved during a pass, that pass is
# CONTAMINATED (someone was chatting with the model on the same 8 cores)
# and is retried, up to 3 tries.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BENCH_SRC="$HERE/f8_cpu_expert_bench.c"
BIN="/tmp/f8_cpu_expert_bench"
LOG_SERVER="$HOME/glm53_server.log"
OUT_DIR="$HOME/bench"
N_WINDOWS="${F8_N_WINDOWS:-3000}"
POOL_EXPERTS="${F8_POOL_EXPERTS:-350}"

mkdir -p "$OUT_DIR"
. "$HERE/rig_lock.sh"

rig_lock_take f8_bench || { echo "FATAL: rig lock held, aborting"; exit 3; }
trap 'rig_lock_release' EXIT

echo "=== f8_bench_run.sh starting $(date -u +%FT%TZ) ==="
echo "ROOT=$ROOT N_WINDOWS=$N_WINDOWS POOL_EXPERTS=$POOL_EXPERTS"

avail_gib=$(awk '/MemAvailable/{printf "%d", $2/1024/1024}' /proc/meminfo)
echo "MemAvailable=${avail_gib}GiB"
if [ "$avail_gib" -lt 20 ]; then
  echo "FATAL: MemAvailable ${avail_gib}GiB < 20GiB floor, refusing"
  exit 1
fi

echo "--- build ---"
BUILD_LOG="$OUT_DIR/f8_build.log"
gcc -O3 -march=native -fopenmp -pthread -Wall -Wextra -Wno-unused-parameter \
    -Wno-unused-function "$BENCH_SRC" -o "$BIN" -lm -lpthread > "$BUILD_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ] || [ ! -x "$BIN" ]; then
  echo "FATAL: build failed rc=$rc"; cat "$BUILD_LOG"; exit 1
fi
echo "build ok -> $BIN"

req_count() {
  [ -f "$LOG_SERVER" ] && grep -c '\[req\]' "$LOG_SERVER" 2>/dev/null || echo 0
}

run_pass() {
  local pass_num="$1" out_file="$2" tries=0
  while [ "$tries" -lt 3 ]; do
    tries=$((tries + 1))
    local before after
    before=$(req_count)
    echo "--- pass $pass_num try $tries: req_count_before=$before ---"
    env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
        "$BIN" "$N_WINDOWS" "$POOL_EXPERTS" "$((42 + pass_num))" > "$out_file" 2>&1
    local bench_rc=$?
    after=$(req_count)
    echo "pass $pass_num try $tries: req_count_after=$after bench_rc=$bench_rc"
    if [ "$bench_rc" -ne 0 ]; then
      echo "FATAL: pass $pass_num try $tries bench exited $bench_rc"; cat "$out_file"; return 1
    fi
    if [ "$before" != "$after" ]; then
      echo "CONTAMINATED: pass $pass_num try $tries, req_count moved $before -> $after, retrying"
      mv "$out_file" "${out_file}.contaminated_try${tries}"
      continue
    fi
    echo "pass $pass_num try $tries: clean (req_count unchanged at $before)"
    return 0
  done
  echo "FATAL: pass $pass_num contaminated on all 3 tries"
  return 1
}

echo "--- pass 1 ---"
run_pass 1 "$OUT_DIR/f8_pass1.txt" || exit 1
echo "--- pass 2 ---"
run_pass 2 "$OUT_DIR/f8_pass2.txt" || exit 1

echo "=== f8_bench_run.sh done $(date -u +%FT%TZ) ==="
echo "outputs: $OUT_DIR/f8_pass1.txt $OUT_DIR/f8_pass2.txt"
exit 0
