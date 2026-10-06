#!/bin/bash
# hwq_ab_chain.sh -- GPU_MAX_HW_QUEUES=16 against the default (4), interleaved A,B,B,A, on two DECODE paths
# (record §L5-GLM-TIMELINE: -7 % on GLM prefill in two un-interleaved chains, unreplicated).
#   Q: Qwen3.8 decode, short depth, 262144 cells, graph replay  (the baseline_chain.sh decode_short_* command; tip b5cf4e0)
#   D: DeepSeek served configuration: eager, adaptation at the served values, depth probe 64 (32 tokens)
# A = default environment, B = -e GPU_MAX_HW_QUEUES=16. Launch through run_chain.sh.
set -u
OUT=$HOME/bench/hwq_ab; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert; . "$HE/gate_lib.sh"
QBIN=$HOME/src/franken-engine/franken/decode/franken_decode_base738
DBIN=$HOME/bench/franken_bin/franken_decode_ds4.tip
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
QDIR=$HOME/models/Qwen3.8-Flash-Next/UD-IQ4_XS; QM=$QDIR/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
DDIR=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M; DM=$DDIR/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix; MM=$HOME/bench/franken/ds4/adapt/mmlu_8k_ids.txt
D() { local e=(); [ -n "${HWQ:-}" ] && e=(-e "GPU_MAX_HW_QUEUES=$HWQ"); docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 "${e[@]}" \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; return 1; }
declare -A Q DS; FAILS=0
echo "=== hwq_ab start $(date -Is)"
echo "=== warm $(date +%T)"; cat "$QDIR"/*.gguf "$DDIR"/*.gguf > /dev/null
for arm in A B B A; do
  HWQ=""; [ $arm = B ] && HWQ=16
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  tag="$(date +%H%M%S)"; echo "=== arm $arm (HWQ=${HWQ:-default}) Q $(date +%T)"
  HWQ=$HWQ D "$QBIN" --model "$QM" --devices 3 --layers 0-47 --threads 8 --tokens 1 --ctx 262144 --hip-graph 1 --time 32 > "$OUT/Q_${arm}_$tag.log"; rc=$?
  v=$(grep -a "layers0_47_ms_median" "$OUT/Q_${arm}_$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p'); echo "  Q decode ms=$v rc=$rc"; [ $rc -eq 0 ] || FAILS=$((FAILS+1)); Q[$arm]="${Q[$arm]:-}$v "
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  echo "=== arm $arm (HWQ=${HWQ:-default}) D $(date +%T)"
  HWQ=$HWQ D "$DBIN" --model "$DM" --tokens-file "$MM" --placement "$MIX" --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64 --probe-n 32 --hip-graph 0 \
      --adapt 1 --adapt-every 16 --adapt-mb-per-token 64 --adapt-halflife 2048 > "$OUT/D_${arm}_$tag.log"; rc=$?
  v=$(grep -a "^ds4_probe depth=64 " "$OUT/D_${arm}_$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | head -1); echo "  D decode ms=$v rc=$rc"; [ $rc -eq 0 ] || FAILS=$((FAILS+1)); DS[$arm]="${DS[$arm]:-}$v "
done
echo "=== verdicts (decode ms/token; A = default queues, B = GPU_MAX_HW_QUEUES=16)"
gate_ab_verdict "Qwen3.8 decode, 262k cells, graph" "${Q[A]}" "${Q[B]}"
gate_ab_verdict "DeepSeek decode, served config, depth 64" "${DS[A]}" "${DS[B]}"
echo "=== hwq_ab end $(date -Is) failures=$FAILS"; exit $(( FAILS > 0 ))
