#!/bin/bash
# ds4_adapt_decode_chain.sh -- does DeepSeek's adaptive placement halve decode after a long prefill?
# Found 2026-10-06 in ds4_shared_gate_chain.sh (record §DS4-ADAPT-DECODE): with --adapt 1 the depth probe
# (64 / 8 192 / 64, hip-graph 1, deepseek_mix placement) read 89-93 ms/token at depth 8 192 AND back at
# depth 64, on both the pre-change and the tip binary, against 46.5-47 ms in §BASELINE-7.0.0-38-DS4-GLM,
# whose probe had adaptation off. The served launcher runs adaptation on (FRANKEN_ADAPT=1, every 16,
# 64 MB a token a card, half-life 2 048: ds4_serve.cpp).
# Arms (tip binary franken_decode_ds4.tip, b5cf4e0), one engine run each, probe depths 64 / 8192 / 64:
#   X0  --adapt 0                                   (no counters, no swaps)
#   X1  --adapt 1 at the served values              (counters + swaps)
#   X2  --adapt 1 --adapt-mb-per-token 0            (counters and snapshots run, NOTHING moves)
# Order X0 X1 X2 X2 X1 X0 (palindrome: every pair of arms is interleaved). Launch through run_chain.sh.
set -u
OUT=$HOME/bench/ds4gate/adapt_decode; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert
. "$HE/gate_lib.sh"
BIN=$HOME/bench/franken_bin/franken_decode_ds4.tip
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
M=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix
MM=$HOME/bench/franken/ds4/adapt/mmlu_8k_ids.txt
D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; return 1; }
BASE="--model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32 --hip-graph 1"
declare -A ARG=( [X0]="--adapt 0"
                 [X1]="--adapt 1 --adapt-every 16 --adapt-mb-per-token 64 --adapt-halflife 2048"
                 [X2]="--adapt 1 --adapt-every 16 --adapt-mb-per-token 0 --adapt-halflife 2048" )
declare -A S64 S8K
FAILS=0
echo "=== ds4_adapt_decode start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; exit 2; }
for arm in X0 X1 X2 X2 X1 X0; do
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  tag="${arm}_$(date +%H%M%S)"
  echo "=== arm $arm ${ARG[$arm]} $(date +%T)"
  D "$BIN" $BASE ${ARG[$arm]} > "$OUT/$tag.log"; rc=$?
  grep -aE "^ds4_probe|^adapt_total|^adapt_all" "$OUT/$tag.log" | tail -6 | cut -c1-200
  echo "rc=$rc"; [ $rc -eq 0 ] || FAILS=$((FAILS+1))
  S64[$arm]="${S64[$arm]:-}$(grep -a '^ds4_probe depth=64 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')"
  S8K[$arm]="${S8K[$arm]:-}$(grep -a '^ds4_probe depth=8192 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')"
done
echo "=== verdicts (decode ms/token): X0 adapt off | X1 adapt on, served values | X2 adapt on, no swaps"
gate_ab_verdict "depth 8192: X0 vs X1" "${S8K[X0]}" "${S8K[X1]}"
gate_ab_verdict "depth 8192: X2 vs X1" "${S8K[X2]}" "${S8K[X1]}"
gate_ab_verdict "depth 8192: X0 vs X2" "${S8K[X0]}" "${S8K[X2]}"
gate_ab_verdict "depth 64:   X0 vs X1" "${S64[X0]}" "${S64[X1]}"
gate_ab_verdict "depth 64:   X2 vs X1" "${S64[X2]}" "${S64[X1]}"
echo "=== ds4_adapt_decode end $(date -Is) failures=$FAILS"
exit $(( FAILS > 0 ))
