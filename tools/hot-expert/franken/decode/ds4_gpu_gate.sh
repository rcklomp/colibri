#!/bin/bash
# tools/hot-expert/franken/decode/ds4_gpu_gate.sh -- L5 step 2's GPU gate for
# DeepSeek-V4-Flash (DEEPSEEK4.md section 9). NOT run by the agent that wrote
# it. The caller holds the rig lock, the gateway is STOPPED and the cards are
# empty (CLAUDE.md "One benchmark at a time"): the GPU run places ~66 GB on
# the cards, pins ~24 GB of host RAM and reads the whole 85 GB file; the CPU
# reference reads ~15-40 GB of it, which is why it belongs in the same window.
#
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [cpu|gpu|time|all]
set -u
[ "${DS4_GPU_OK:-0}" = 1 ] || { echo "refusing: set DS4_GPU_OK=1 (rig lock held, gateway stopped)"; exit 2; }
D=/home/ronald/src/colibri-m1/tools/hot-expert/franken/decode
M=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
P=/home/ronald/bench/m2/deepseek_b60           # M2 histogram, layer_<il>.csv
O=/home/ronald/bench/franken/ds4
T="0 671 6102 294 8760 344"                     # BOS + "The capital of France is"
GPU=$D/franken_decode_ds4                       # make franken_decode_ds4 GPU_BIN=franken_decode_ds4
what=${1:-all}
mkdir -p $O
if [ $what = cpu ] || [ $what = all ]; then
  echo "=== (1) CPU reference: layers 0-42 + head, 6 tokens, greedy 16, dump"
  $D/franken_decode_cpu --model $M --tokens $T --cpu --ctx 512 --threads 8 --greedy 16 \
    --dump $O/cpu_l042 > $O/cpu_l042.log 2>&1; echo "rc=$?"; grep -E "greedy_ids|dumped" $O/cpu_l042.log
fi
if [ $what = gpu ] || [ $what = all ]; then
  echo "=== (2) GPU vs the CPU dump, first run under --sync-debug"
  $GPU --model $M --tokens $T --ctx 512 --placement $P --expert-gb 20 --greedy 16 \
    --oracle $O/cpu_l042 --sync-debug > $O/gpu_l042_sync.log 2>&1; echo "rc=$?"
  grep -E "placement|vram_dev|greedy_ids|oracle summary|ORACLE|FAIL|SYNC-DEBUG" $O/gpu_l042_sync.log | head -40
  echo "=== (2b) the same without --sync-debug"
  $GPU --model $M --tokens $T --ctx 512 --placement $P --expert-gb 20 --greedy 16 \
    --oracle $O/cpu_l042 > $O/gpu_l042.log 2>&1; echo "rc=$?"
  grep -E "greedy_ids|oracle summary|ORACLE" $O/gpu_l042.log
fi
if [ $what = time ] || [ $what = all ]; then
  echo "=== (3) 256k allocated: greedy 16 then 32 timed decode tokens"
  $GPU --model $M --tokens $T --ctx 262144 --placement $P --expert-gb 20 --greedy 16 --time 32 \
    > $O/gpu_time_256k.log 2>&1; echo "rc=$?"
  grep -E "vram_dev|ds4_cache|greedy_ids|ds4_decode_ms" $O/gpu_time_256k.log
fi
echo "=== DONE"
