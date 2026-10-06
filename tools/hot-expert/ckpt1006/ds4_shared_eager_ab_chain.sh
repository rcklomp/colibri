#!/bin/bash
# ds4_shared_eager_ab_chain.sh -- the SERVED configuration of the DeepSeek gate for the shared ds4_gpu.inc change (473c5cc).
# ds4_shared_gate_chain.sh timed A (5f409c2) vs B (tip) with --hip-graph 1 + adaptation, the regime record §DS4-ADAPT-GRAPH found
# pathological (+40 ms); the service runs EAGER (FRANKEN_HIP_GRAPH default 0). Here: adaptation at the served values, graph off,
# depth probe 64 / 8192 / 64, A,B,B,A. Launch through run_chain.sh.
set -u
OUT=$HOME/bench/ds4gate/shared_eager; mkdir -p "$OUT"
HE=$HOME/src/colibri/tools/hot-expert; . "$HE/gate_lib.sh"
A_BIN=$HOME/bench/franken_bin/franken_decode_ds4.pre473; B_BIN=$HOME/bench/franken_bin/franken_decode_ds4.tip
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
M=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
MIX=$HOME/bench/m2/deepseek_mix; MM=$HOME/bench/franken/ds4/adapt/mmlu_8k_ids.txt
D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; return 1; }
DP="--model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32 --hip-graph 0 --adapt 1 --adapt-every 16 --adapt-mb-per-token 64 --adapt-halflife 2048"
declare -A S64 S8K; FAILS=0
echo "=== ds4_shared_eager_ab start $(date -Is) A=$(sha256sum $A_BIN | cut -c1-16) B=$(sha256sum $B_BIN | cut -c1-16)"
for arm in A B B A; do
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  bin=$A_BIN; [ $arm = B ] && bin=$B_BIN
  tag="${arm}_$(date +%H%M%S)"; echo "=== arm $arm $(date +%T)"
  D "$bin" $DP > "$OUT/$tag.log"; rc=$?
  grep -aE "^ds4_probe" "$OUT/$tag.log" | cut -c1-170; echo "rc=$rc"; [ $rc -eq 0 ] || FAILS=$((FAILS+1))
  S64[$arm]="${S64[$arm]:-}$(grep -a '^ds4_probe depth=64 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')"
  S8K[$arm]="${S8K[$arm]:-}$(grep -a '^ds4_probe depth=8192 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')"
done
echo "=== verdicts (decode ms/token, EAGER, adaptation on; A = pre473, B = tip)"
gate_ab_verdict "decode at depth 64"   "${S64[A]}" "${S64[B]}"
gate_ab_verdict "decode at depth 8192" "${S8K[A]}" "${S8K[B]}"
echo "=== ds4_shared_eager_ab end $(date -Is) failures=$FAILS"; exit $(( FAILS > 0 ))
