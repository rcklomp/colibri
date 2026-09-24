#!/bin/bash
# tools/hot-expert/franken/decode/ds4_gpu_gate.sh -- L5 step 2's GPU gate for
# DeepSeek-V4-Flash (DEEPSEEK4.md section 9). NOT run by the agent that wrote
# it. The caller holds the rig lock, the gateway is STOPPED and the cards are
# empty (CLAUDE.md "One benchmark at a time"): the GPU run places ~66 GB on
# the cards, pins ~24 GB of host RAM and reads the whole 85 GB file; the CPU
# reference reads ~15-40 GB of it, which is why it belongs in the same window.
#
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [cpu|gpu|time|all]      (L5 step 2)
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [ident|graph|prof|speed|hits|step3]   (L5 step 3, sections 10-11)
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [adapt-ident|adapt-traj|adapt]      (L5 step 4, section 12)
set -u
[ "${DS4_GPU_OK:-0}" = 1 ] || { echo "refusing: set DS4_GPU_OK=1 (rig lock held, gateway stopped)"; exit 2; }
D=/home/ronald/src/colibri-m1/tools/hot-expert/franken/decode
M=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
P=${DS4_PLACEMENT:-/home/ronald/bench/m2/deepseek_b60}           # M2 histogram, layer_<il>.csv
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
  grep -E "^--- device|prof_ds4|prof_norm|prof_elem|prof_router|prof_boundary|prof_copy|prof_argmax|prof_gap_idle|prof_total|prof_busy|prof_launches|prof_host_syncs|ds4_miss|ds4_decode" $O/gpu_prof_stage.log
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
if [ $what = hits ] || [ $what = step3 ]; then
  echo "=== (8) per layer: resident experts, the hit fraction measured on THIS text, the M2 coverage"
  $GPU $G --ctx 512 --greedy 16 --time 32 --hit-report --hist-out $O/hist_gate > $O/gpu_hits.log 2>&1; echo "rc=$?"
  grep -E "^hit|hist-out" $O/gpu_hits.log
fi
# ---- L5 step 4: adaptive placement (DEEPSEEK4.md section 12) --------------
A=/home/ronald/bench/franken/ds4/adapt          # mmlu_8k_ids.txt: BOS + the first 8 191 tokens of the
                                                # MMLU-Pro half of ~/bench/m2/mixed_corpus.txt (lines 1-2454)
AD="--adapt 1 --adapt-verify 1"
if [ $what = adapt-ident ] || [ $what = adapt ]; then
  echo "=== (9) oracle pair: --adapt 0 --dump, then --adapt 1 --oracle it; swaps every round, every tap cos=1 maxabs=0"
  $GPU $G --ctx 512 --greedy 16 --adapt 0 --dump $O/adapt_off > $O/adapt_off.log 2>&1; echo "rc=$?"
  $GPU $G --ctx 512 --greedy 16 $AD --adapt-every 1 --adapt-mb-per-token 512 --oracle $O/adapt_off > $O/adapt_on.log 2>&1; echo "rc=$?"
  grep -E "^adapt_all|^adapt_total|^adapt_verify|oracle summary|ORACLE|greedy_ids" $O/adapt_on.log
  echo "not_bitexact=$(grep '^oracle .* cos=' $O/adapt_on.log | grep -v 'cos=1.000000 maxabs=0 ' | wc -l) taps=$(grep -c '^oracle .* cos=' $O/adapt_on.log)"
  echo "=== (9b) the same with --hip-graph 1 (the swaps between replays)"
  $GPU $G --ctx 512 --greedy 16 $AD --adapt-every 1 --adapt-mb-per-token 512 --hip-graph 1 --oracle $O/adapt_off > $O/adapt_on_graph.log 2>&1; echo "rc=$?"
  grep -E "^adapt_total|^adapt_verify|oracle summary|ORACLE|greedy_ids" $O/adapt_on_graph.log
  echo "not_bitexact=$(grep '^oracle .* cos=' $O/adapt_on_graph.log | grep -v 'cos=1.000000 maxabs=0 ' | wc -l)"
  echo "=== (9c) 64 prose tokens (the hot set moves most): off --dump, on --oracle"
  P64="--model $M --tokens-file $A/mmlu_8k_ids.txt --max-tokens 64 --placement $P --expert-gb 20 --ctx 512 --greedy 16"
  $GPU $P64 --adapt 0 --dump $O/adapt_off64 > $O/adapt_off64.log 2>&1; echo "rc=$?"
  $GPU $P64 $AD --adapt-every 1 --adapt-mb-per-token 512 --oracle $O/adapt_off64 > $O/adapt_on64.log 2>&1; echo "rc=$?"
  grep -E "^adapt_total|^adapt_verify|oracle summary|ORACLE|greedy_ids" $O/adapt_on64.log
  echo "not_bitexact=$(grep '^oracle .* cos=' $O/adapt_on64.log | grep -v 'cos=1.000000 maxabs=0 ' | wc -l)"
fi
if [ $what = adapt-traj ] || [ $what = adapt ]; then
  # From the technical-record placement ($P), 8 192 prose tokens then 256 greedy
  # and 32 timed decode tokens, 262 144 cells, graph replay. The control is the
  # same run with a zero swap budget: the counters run, nothing moves.
  TR="--model $M --tokens-file $A/mmlu_8k_ids.txt --placement $P --expert-gb 20 --ctx 262144 --greedy 256 --time 32 --hip-graph 1 --adapt-every 64"
  echo "=== (10a) control: --adapt-mb-per-token 0 (misses per 64 tokens, no swaps)"
  $GPU $TR --adapt 1 --adapt-mb-per-token 0 > $O/adapt_traj_ctl.log 2>&1; echo "rc=$?"
  grep -E "^adapt_all" $O/adapt_traj_ctl.log | awk 'NR%8==1'
  grep -E "^adapt_total|ds4_decode_ms" $O/adapt_traj_ctl.log
  echo "=== (10b) adaptation: 64 MB a token a card, half-life 2 048 tokens"
  $GPU $TR $AD --adapt-mb-per-token 64 --adapt-halflife 2048 > $O/adapt_traj.log 2>&1; echo "rc=$?"
  grep -E "^adapt_all" $O/adapt_traj.log | awk 'NR%8==1'
  grep -E "^adapt_all" $O/adapt_traj.log | tail -4
  grep -E "^adapt_total|^adapt_verify|ds4_decode_ms|greedy_ids" $O/adapt_traj.log | cut -c1-200
fi
echo "=== DONE"
