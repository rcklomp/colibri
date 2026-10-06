#!/bin/bash
# ds4_adapt_bisect_chain.sh -- bisect "DeepSeek adaptation + graph replay = +40 ms/token" (record §DS4-ADAPT-DECODE).
# Known (ds4_adapt_profile_chain, tip b5cf4e0, deepseek_mix, 262144 cells, 16 greedy + 32 timed): graph replay, adapt off 51.3 ms;
# adapt on 88.8 ms; adapt on with the tick never firing and no swaps (N1) 92.7 ms; adapt on EAGER 48.8 ms; graphs captured once and
# replayed 52 times in every graph arm (973/932/940 nodes). So: not capture, not ticks, not swaps. Candidates:
#   (i)  the router's counter atomics (adapt only) -> arm S: binary .nostats, FRANKEN_NO_ADAPT_STATS=1 (adapt on, router gets a null pointer)
#   (ii) the host mirror of ALL experts (adapt only; missed experts read from a ~38 GB mapped region) with the staged miss path
#        -> arm M: --miss-stage 0 (in-place host reads), against its own adapt-off control arm C
# Arms: R = adapt 0 (reference) | N = adapt 1 N1-style | S = N + null stats | M = N + --miss-stage 0 | C = adapt 0 + --miss-stage 0
# Order R N S M C C M S N R. Launch through run_chain.sh.
set -u
OUT=$HOME/bench/ds4gate/adapt_bisect; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert; . "$HE/gate_lib.sh"
TIP=$HOME/bench/franken_bin/franken_decode_ds4.tip
NOSTATS=$HOME/bench/franken_bin/franken_decode_ds4.nostats
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
M=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix
T="0 671 6102 294 8760 344"
D() { local envs=(); [ -n "${DENV:-}" ] && envs=(-e "$DENV"); docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 "${envs[@]}" \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; return 1; }
BASE="--model $M --tokens $T --placement $MIX --expert-gb 20 --ctx 262144 --greedy 16 --time 32 --hip-graph 1"
NARGS="--adapt 1 --adapt-every 100000000 --adapt-mb-per-token 0"
declare -A SAMP; FAILS=0
arm() {   # arm <name>
  local a=$1 bin=$TIP args="" DENV=""
  case $a in
    R) args="--adapt 0";;
    N) args="$NARGS";;
    S) args="$NARGS"; bin=$NOSTATS; DENV="FRANKEN_NO_ADAPT_STATS=1";;
    M) args="$NARGS --miss-stage 0";;
    C) args="--adapt 0 --miss-stage 0";;
  esac
  [ -x "$bin" ] || { echo "FATAL: $bin missing"; exit 2; }
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  local tag="${a}_$(date +%H%M%S)"
  echo "=== arm $a $args ${DENV:+[$DENV]} $(date +%T)"
  DENV=$DENV D "$bin" $BASE $args > "$OUT/$tag.log"; local rc=$?
  local ms; ms=$(grep -a "^ds4_decode_ms_median" "$OUT/$tag.log" | sed -n 's/.*median=\([0-9.]*\).*/\1/p')
  echo "  decode_ms_median=$ms rc=$rc"; grep -aE "^hip_graph_dev0" "$OUT/$tag.log" | cut -c1-150
  [ $rc -eq 0 ] || FAILS=$((FAILS+1)); SAMP[$a]="${SAMP[$a]:-}$ms "
}
echo "=== ds4_adapt_bisect start $(date -Is) tip=$(sha256sum $TIP | cut -c1-16) nostats=$(sha256sum $NOSTATS 2>/dev/null | cut -c1-16)"
for a in R N S M C C M S N R; do arm $a; done
echo "=== verdicts (decode ms/token, graph replay): R adapt off | N adapt on | S adapt on, null stats | M adapt on, miss-stage 0 | C adapt off, miss-stage 0"
gate_ab_verdict "R vs N  (adapt effect)"                "${SAMP[R]}" "${SAMP[N]}"
gate_ab_verdict "N vs S  (router counters)"             "${SAMP[N]}" "${SAMP[S]}"
gate_ab_verdict "R vs S  (adapt minus counters)"        "${SAMP[R]}" "${SAMP[S]}"
gate_ab_verdict "C vs M  (adapt effect, in-place misses)" "${SAMP[C]}" "${SAMP[M]}"
gate_ab_verdict "N vs M  (staged vs in-place, adapt on)" "${SAMP[N]}" "${SAMP[M]}"
echo "=== ds4_adapt_bisect end $(date -Is) failures=$FAILS"
exit $(( FAILS > 0 ))
