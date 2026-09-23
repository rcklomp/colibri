#!/bin/bash
# tools/hot-expert/franken/decode/ds4_gpu_gate.sh -- L5 step 2's GPU gate for
# DeepSeek-V4-Flash (DEEPSEEK4.md section 9). NOT run by the agent that wrote
# it. The caller holds the rig lock, the gateway is STOPPED and the cards are
# empty (CLAUDE.md "One benchmark at a time"): the GPU run places ~66 GB on
# the cards, pins ~24 GB of host RAM and reads the whole 85 GB file; the CPU
# reference reads ~15-40 GB of it, which is why it belongs in the same window.
#
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [cpu|gpu|time|all]      (L5 step 2)
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [ident|graph|prof|speed|step3]   (L5 step 3, section 10)
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
# ---- L5 step 3 (DEEPSEEK4.md section 10) ----------------------------------
G="--model $M --tokens $T --placement $P --expert-gb 20"
if [ $what = ident ] || [ $what = step3 ]; then
  echo "=== (4) bit-identity: step 2's paths (in-place byte loads, misses read over PCIe) dump,"
  echo "        then the new paths (LDS-staged loads, VRAM staging ring) against it: every tap cos=1 maxabs=0"
  $GPU $G --ctx 512 --greedy 16 --staged-loads 0 --miss-stage 0 --dump $O/gpu_step2paths > $O/gpu_step2paths.log 2>&1; echo "rc=$?"
  $GPU $G --ctx 512 --greedy 16 --oracle $O/gpu_step2paths > $O/gpu_step3_vs_step2.log 2>&1; echo "rc=$?"
  grep -E "oracle summary|ORACLE|greedy_ids" $O/gpu_step3_vs_step2.log
  echo "not_bitexact=$(grep '^oracle .* cos=' $O/gpu_step3_vs_step2.log | grep -v 'cos=1.000000 maxabs=0 ' | wc -l)"
  echo "=== (4b) and against the CPU arm, as step 2's gate"
  $GPU $G --ctx 512 --greedy 16 --oracle $O/cpu_l042 > $O/gpu_step3_vs_cpu.log 2>&1; echo "rc=$?"
  grep -E "oracle summary|ORACLE|greedy_ids" $O/gpu_step3_vs_cpu.log
fi
if [ $what = graph ] || [ $what = step3 ]; then
  echo "=== (5) --hip-graph 1: prompt and greedy tokens replayed, the last prompt token eager (taps);"
  echo "        against the eager dump of (4): every tap cos=1 maxabs=0, greedy_ids exact"
  $GPU $G --ctx 512 --greedy 16 --hip-graph 1 --oracle $O/gpu_step2paths > $O/gpu_graph_vs_eager.log 2>&1; echo "rc=$?"
  grep -E "oracle summary|ORACLE|greedy_ids" $O/gpu_graph_vs_eager.log
  echo "not_bitexact=$(grep '^oracle .* cos=' $O/gpu_graph_vs_eager.log | grep -v 'cos=1.000000 maxabs=0 ' | wc -l)"
fi
if [ $what = prof ] || [ $what = step3 ]; then
  echo "=== (6) --profile, 256k allocated, 32 tokens: per-class device time a card, miss MB/token"
  $GPU $G --ctx 262144 --greedy 16 --time 32 --profile > $O/gpu_prof_stage.log 2>&1; echo "rc=$?"
  grep -E "^--- device|prof_ds4|prof_norm|prof_elem|prof_router|prof_boundary|prof_copy|prof_argmax|prof_total|prof_launches|prof_host_syncs|ds4_miss|ds4_decode" $O/gpu_prof_stage.log
  echo "=== (6b) the same with step 2's miss path, the host-mapped slots in their own class"
  $GPU $G --ctx 262144 --greedy 16 --time 32 --profile --miss-stage 0 > $O/gpu_prof_inplace.log 2>&1; echo "rc=$?"
  grep -E "^--- device|prof_ds4_expert|prof_ds4_miss|prof_total|ds4_miss|ds4_decode" $O/gpu_prof_inplace.log
fi
if [ $what = speed ] || [ $what = step3 ]; then
  echo "=== (7) timing, 256k allocated, A,B,B,A: eager / graph / graph / eager"
  for g in 0 1 1 0; do
    $GPU $G --ctx 262144 --greedy 16 --time 32 --hip-graph $g > $O/gpu_speed_g$g.log 2>&1
    echo "hip_graph=$g rc=$? $(grep -E 'ds4_decode_ms' $O/gpu_speed_g$g.log)"
  done
  grep -h "hip_graph_dev" $O/gpu_speed_g1.log
fi
echo "=== DONE"
