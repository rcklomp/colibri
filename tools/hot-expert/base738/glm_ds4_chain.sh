#!/bin/bash
# glm_ds4_chain.sh -- Franken baseline on kernel 7.0.0-38, part 2: DeepSeek-V4-Flash and
# GLM-5.3-Flash (part 1, Qwen3.8 + M4: baseline_chain.sh / supplement_chain.sh; record
# §BASELINE-7.0.0-38). Same binary as part 1: franken-engine b5cf4e0 built as
# franken_decode_base738 -- franken_decode dispatches on the GGUF's architecture, so one binary
# runs all three models. Launch ONLY through run_chain.sh (rig lock):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/base738/glm_ds4_chain.sh \
#       > ~/bench/base738/chain_glmds4.log 2>&1 < /dev/null &
# DeepSeek: the depth probe and prefill commands of ds4_gpu_gate.sh (s5-depth, s5-prefill), the
# gate's own stale paths replaced. GLM: glm5_gpu_gate.sh pf2 (one GPU process, 15 configs) as
# glm5e_chain.sh ran it, output in a NEW directory so the gpu5 reference dumps are only read.
# No sudo, no cache drop, no DPM change. Models are read from a cold-ish cache; DeepSeek's shards
# are read once first (timed), GLM is read by the engine's own load (load_s is a datum).
set -u
OUT=$HOME/bench/base738/glmds4; mkdir -p "$OUT"
SRC=$HOME/src/franken-engine
BIN=$SRC/franken/decode/franken_decode_base738
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
HE=$HOME/src/colibri/tools/hot-expert
DSDIR=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M
DSM=$DSDIR/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix
MM=$HOME/bench/franken/ds4/adapt/mmlu_8k_ids.txt
T6="0 671 6102 294 8760 344"                 # BOS + "The capital of France is"

D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; }
vm() { awk '/^(pgfault|pgmajfault|pgscan_direct|pgsteal_direct|numa_hint_faults|thp_fault_alloc) /{printf "%s=%s ",$1,$2}' /proc/vmstat; echo; }

env_snapshot() {
  echo "--- env $1 $(date -Is)"
  echo "kernel=$(uname -r)"
  echo "engine_src=$(git -C "$SRC" log -1 --format='%h %s' | cut -c1-90) dirty=$(git -C "$SRC" status --short | wc -l)"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -c1-16) image=$(docker image inspect $IMG --format '{{.Id}}' | cut -c1-19)"
  for d in 0000:83:00.0 0000:86:00.0 0000:48:00.0; do
    p=/sys/bus/pci/devices/$d
    echo "gpu $d dpm=$(cat $p/power_dpm_force_performance_level) link=$(cat $p/current_link_speed)/x$(cat $p/current_link_width) vram_used=$(( $(cat $p/mem_info_vram_used)/1048576 ))MiB"
  done
  echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled) numa_balancing=$(cat /proc/sys/kernel/numa_balancing) swappiness=$(cat /proc/sys/vm/swappiness) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  free -m | sed -n 2p
  echo "vmstat $(vm)"
}

echo "=== glm_ds4_chain start $(date -Is)"
env_snapshot start | tee "$OUT/env_start.txt"
[ "$(uname -r)" = 7.0.0-38-generic ] || { echo "FATAL: not on 7.0.0-38-generic"; exit 1; }
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; exit 1; }
[ "$(vram_max)" -lt 1024 ] || { echo "FATAL: VRAM in use"; exit 1; }
for f in "$MM" "$MIX/layer_3.csv" "$HOME/bench/franken/glm5/gpu5/g136_eager/index.txt" "$HOME/bench/franken/glm5/prose8400.txt"; do
  [ -e "$f" ] || { echo "FATAL: input missing: $f"; exit 1; }
done

# ======================= DeepSeek-V4-Flash =======================
echo "=== DS4 warm $(date +%T)"
sz=$(du -cb "$DSDIR"/*.gguf | tail -1 | cut -f1); v0=$(vm); t0=$(date +%s.%N)
cat "$DSDIR"/*.gguf > /dev/null; t1=$(date +%s.%N)
awk -v s="$sz" -v a="$t0" -v b="$t1" 'BEGIN{printf "ds4_read bytes=%d seconds=%.1f GBps=%.3f\n", s, b-a, s/(b-a)/1e9}'
echo "vmstat_before $v0"; echo "vmstat_after  $(vm)"
fincore "$DSDIR"/*.gguf | tee "$OUT/fincore_ds4.txt"

echo "=== DS4 preflight smoke $(date +%T)"
PREFLIGHT_OWN_LOCK=1 "$HE/preflight.sh" bash -c '"$@"; rc=$?; [ $rc -eq 4 ] && rc=0; exit $rc' _ \
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
  -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" \
  "$BIN" --model "$DSM" --tokens $T6 --ctx 512 --placement "$MIX" --expert-gb 20 --greedy 4 \
  || { echo "FATAL: DS4 smoke failed"; exit 2; }

DKEYS='^ds4_probe|ds4_decode_ms|^prefill_tokens|load_s=|hip_graph_dev|greedy_ids|HIP error|fault|rc='
runds() {  # runds <label> <args...>
  local label=$1; shift
  vram_wait
  echo "=== ds4 $label $(date +%T)"
  local v0 v1; v0=$(vm)
  D "$BIN" --model "$DSM" --placement "$MIX" --expert-gb 20 "$@" > "$OUT/ds4_$label.log"; rc=$?
  v1=$(vm)
  grep -E "$DKEYS" "$OUT/ds4_$label.log" | cut -c1-220
  echo "rc=$rc"; echo "vmstat_before $v0"; echo "vmstat_after  $v1"
}
DP="--tokens-file $MM --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32"
TP="--tokens-file $MM --ctx 262144 --chunk 256"
runds depth_g1_a   $DP --hip-graph 1
runds depth_g1_b   $DP --hip-graph 1
runds depth_g0     $DP --hip-graph 0
runds prefill8k_a  $TP --time-prefill 8192  --time 32 --hip-graph 1
runds prefill8k_b  $TP --time-prefill 8192  --time 32 --hip-graph 1
runds prefill32k   $TP --time-prefill 32768 --time 32 --hip-graph 1
runds prefill8k_lds_a $TP --time-prefill 8192 --gemm-lds 1
runds prefill8k_lds_b $TP --time-prefill 8192 --gemm-lds 1

# ======================= GLM-5.3-Flash =======================
# pf2 = the plan behind §L5-GLM-LDS: smoke, 4 exact configs against the gpu5 references, one cos
# config, then timing at 8 192 prompt tokens / 262 144 cells / chunk 512 / staged / swaps held,
# A,B,B,A twice (snapshot DMA on/off, --gemm-lds 0/1) with 32 timed decode tokens each.
vram_wait
echo "=== GLM preflight $(date +%T)"
PREFLIGHT_OWN_LOCK=1 "$HE/preflight.sh" || { echo "FATAL: preflight refused"; exit 2; }
mkdir -p "$OUT/glm_d" "$OUT/glm_gate"
ln -sf "$BIN" "$OUT/glm_d/franken_decode_glm"
echo "=== GLM pf2 gate $(date +%T)"
v0=$(vm)
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1 -e GLM5_GPU_OK=1 -e GLM5_D="$OUT/glm_d" -e GLM5_GATE_OUT="$OUT/glm_gate" \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald \
  -v /home/ronald:/home/ronald -w "$SRC/franken/decode" "$IMG" bash ./glm5_gpu_gate.sh pf2 2>&1 | tee "$OUT/glm_gate.txt"
rc=${PIPESTATUS[0]}; v1=$(vm)
echo "=== glm gate rc=$rc"; echo "vmstat_before $v0"; echo "vmstat_after  $v1"
grep -E "load_s=|gate_plan_summary" "$OUT/glm_gate/gate_run.log" | cut -c1-260

vram_wait
env_snapshot end | tee "$OUT/env_end.txt"
echo "=== glm_ds4_chain done $(date -Is)"
exit 0
