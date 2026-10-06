#!/bin/bash
# ds4_adapt_profile_chain.sh -- where do the +44 ms/token of DeepSeek's adaptation go?
# (ds4_adapt_decode_chain.sh, 2026-10-06: --adapt 0 47.2 ms, --adapt 1 91.1-92.7 ms at depth 8 192 and 64,
#  the no-swap control (--adapt-mb-per-token 0) just as slow.) Tip binary (franken-engine b5cf4e0), the
#  deepseek_mix placement, 262 144 cells, 16 greedy + 32 timed decode tokens, one engine run per arm:
#   P0 --adapt 0 --profile            P1 --adapt 1 (served values) --profile    (per-class device time a card)
#   E0 --adapt 0 --hip-graph 0        E1 --adapt 1 --hip-graph 0                (eager: is graph replay involved?)
#   N1 --adapt 1 --adapt-every 100000000  (counters on, the tick never snapshots: counters vs tick)
#   G0 --adapt 0 --hip-graph 1        G1 --adapt 1 --hip-graph 1                 (graph, no profiler: the plain pair)
# Launch through run_chain.sh.
set -u
OUT=$HOME/bench/ds4gate/adapt_profile; mkdir -p "$OUT"
BIN=$HOME/bench/franken_bin/franken_decode_ds4.tip
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
M=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix
T="0 671 6102 294 8760 344"
D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald/bench:/home/ronald/bench -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; return 1; }
BASE="--model $M --tokens $T --placement $MIX --expert-gb 20 --ctx 262144 --greedy 16 --time 32"
SERVED="--adapt 1 --adapt-every 16 --adapt-mb-per-token 64 --adapt-halflife 2048"
KEYS='ds4_decode_ms|^--- device|prof_ds4|prof_norm|prof_elem|prof_router|prof_boundary|prof_copy|prof_argmax|prof_gap_idle|prof_total|prof_busy|prof_launches|prof_host_syncs|ds4_miss|adapt_total|hip_graph_dev|HIP error'
run() {  # run <label> <args...>
  local label=$1; shift
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  echo "=== $label $* $(date +%T)"
  D "$BIN" $BASE "$@" > "$OUT/$label.log"; rc=$?
  grep -aE "$KEYS" "$OUT/$label.log" | cut -c1-210 | head -60
  echo "rc=$rc"
}
echo "=== ds4_adapt_profile start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary"; exit 2; }
run G0 --adapt 0 --hip-graph 1
run G1 $SERVED --hip-graph 1
run E0 --adapt 0 --hip-graph 0
run E1 $SERVED --hip-graph 0
run N1 --adapt 1 --adapt-every 100000000 --adapt-mb-per-token 0 --hip-graph 1
run P0 --adapt 0 --profile
run P1 $SERVED --profile
echo "=== summary: ds4_decode_ms per arm"
for a in G0 G1 E0 E1 N1 P0 P1; do printf '%-3s ' $a; grep -a "ds4_decode_ms" "$OUT/$a.log" | head -1 | cut -c1-160; done
echo "=== ds4_adapt_profile end $(date -Is)"
