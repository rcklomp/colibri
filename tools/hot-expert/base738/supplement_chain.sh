#!/bin/bash
# baseline_chain.sh -- Franken baseline on kernel 7.0.0-38 (RIG-ENV-2026-10-05 says nothing was
# measured after 7.0.0-34 -> 7.0.0-38; every earlier number predates it).
# Engine = franken-engine tip b5cf4e0 built as franken_decode_base738 (no GPU needed to build);
# nothing else in the stack changed. Launch ONLY through run_chain.sh (rig lock):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/base738/baseline_chain.sh \
#       > ~/bench/base738/chain.log 2>&1 < /dev/null &
# Tier 1: warm cache -> smoke -> Qwen3.8 decode (short depth, 32k, 262k) x2 fresh processes -> M4 x2.
# No sudo, no cache drop, no DPM change (cards stay at `auto`, as in every earlier row).
set -u
OUT=$HOME/bench/base738; mkdir -p "$OUT"
SRC=$HOME/src/franken-engine
BIN=$SRC/franken/decode/franken_decode_base738
M4=$SRC/m1/m4_stream
GDIR=$HOME/models/Qwen3.8-Flash-Next/UD-IQ4_XS
GGUF=$GDIR/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
HE=$HOME/src/colibri/tools/hot-expert

D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 45); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; }
vm() { awk '/^(pgfault|pgmajfault|pgscan_direct|pgsteal_direct|numa_hint_faults|thp_fault_alloc) /{printf "%s=%s ",$1,$2}' /proc/vmstat; echo; }

env_snapshot() {
  echo "--- env $1 $(date -Is)"
  echo "kernel=$(uname -r)  cmdline=$(cat /proc/cmdline)"
  echo "engine_src=$(git -C "$SRC" log -1 --format='%h %s' | cut -c1-90) dirty=$(git -C "$SRC" status --short | wc -l)"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -c1-16)  m4_sha256=$(sha256sum "$M4" | cut -c1-16)"
  echo "image=$(docker image inspect $IMG --format '{{.Id}}' | cut -c1-19)"
  for d in 0000:83:00.0 0000:86:00.0 0000:48:00.0; do
    p=/sys/bus/pci/devices/$d
    echo "gpu $d dpm=$(cat $p/power_dpm_force_performance_level) link=$(cat $p/current_link_speed)/x$(cat $p/current_link_width) vram_used=$(( $(cat $p/mem_info_vram_used)/1048576 ))MiB"
  done
  echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled) numa_balancing=$(cat /proc/sys/kernel/numa_balancing) swappiness=$(cat /proc/sys/vm/swappiness) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  free -m | sed -n 2p
  echo "vmstat $(vm)"
}

echo "=== baseline_chain start $(date -Is)"
env_snapshot start | tee "$OUT/env_start_supp.txt"
[ "$(uname -r)" = 7.0.0-38-generic ] || { echo "FATAL: not on 7.0.0-38-generic"; exit 1; }
[ "$(vram_max)" -lt 1024 ] || { echo "FATAL: VRAM in use"; exit 1; }

# 1. warm the page cache: a cold run pays 100k+ major faults and is not a baseline (CLAUDE.md).
#    The read is itself a data point: sequential NVMe read rate on this kernel, cold.
echo "=== warm $(date +%T)"
if [ "${SKIP_WARM:-0}" = 1 ]; then echo "warm skipped (already resident, see fincore below)"; else
sz=$(du -cb "$GDIR"/*.gguf | tail -1 | cut -f1); v0=$(vm); t0=$(date +%s.%N)
cat "$GDIR"/*.gguf > /dev/null; t1=$(date +%s.%N)
awk -v s="$sz" -v a="$t0" -v b="$t1" 'BEGIN{printf "cold_read bytes=%d seconds=%.1f GBps=%.3f\n", s, b-a, s/(b-a)/1e9}'
echo "vmstat_before $v0"; echo "vmstat_after  $(vm)"
fi
command -v fincore >/dev/null && fincore "$GDIR"/*.gguf | tee "$OUT/fincore.txt"

# 2. preflight smoke: same binary, 1 token, short context
echo "=== preflight smoke $(date +%T)"
PREFLIGHT_OWN_LOCK=1 "$HE/preflight.sh" bash -c '"$@"; rc=$?; [ $rc -eq 4 ] && rc=0; exit $rc' _ docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
  -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" \
  "$BIN" --model "$GGUF" --tokens 1 --devices 3 --layers 0-47 --ctx 4096 --time 2 || { echo "FATAL: smoke failed"; exit 2; }

KEYS='ms_median|prefill_ms|prefill_tokens|issue_ms|HIP error|fault|rc='
run() {  # run <label> <args...>
  local label=$1; shift
  vram_wait
  echo "=== $label $(date +%T)"
  local v0 v1; v0=$(vm)
  D "$BIN" --model "$GGUF" --devices 3 --layers 0-47 --threads 8 "$@" > "$OUT/$label.log"; rc=$?
  v1=$(vm)
  grep -E "$KEYS" "$OUT/$label.log" | cut -c1-200
  echo "rc=$rc"; echo "vmstat_before $v0"; echo "vmstat_after  $v1"
}

# supplement: the 32k prefill row (record shape: --ctx 32768 --time-prefill 32768, no --time) and 262k depth with the graph on
run prefill32k_a    --tokens 1 --ctx 32768 --chunk 256 --gemm-lds 1 --time-prefill 32768
run prefill32k_b    --tokens 1 --ctx 32768 --chunk 256 --gemm-lds 1 --time-prefill 32768
run depth256k_hg1   --tokens 1 --ctx 262144 --chunk 256 --gemm-lds 1 --hip-graph 1 --time-prefill 262000 --time 32

env_snapshot end | tee "$OUT/env_end_supp.txt"
echo "=== baseline_chain done $(date -Is)"
exit 0
