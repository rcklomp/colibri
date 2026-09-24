#!/bin/bash
# tools/hot-expert/franken/decode/glm5_gpu_gate.sh -- L5 GLM step 4's GPU gate for
# GLM-5.3-Flash (GLM5.md section 9). NOT run by the agent that wrote it. The
# caller holds the rig lock, the gateway is STOPPED (GLM-5.3 is the served model
# and this run needs its whole file), the cards are empty (CLAUDE.md "One
# benchmark at a time"). A GPU process PINS the whole expert mirror (143.7 GB of
# host RAM) and uploads ~51 GB of experts + 8.8 GB of trunk: every GPU step
# below re-reads the 149 GB file to build it.
#
# Run inside the ROCm image with the cards (the docker form, GLM5.md 9.6):
#   GLM5_GPU_OK=1 glm5_gpu_gate.sh [cpu|gpu6|ident|long|time|probe|all]
set -u
[ "${GLM5_GPU_OK:-0}" = 1 ] || { echo "refusing: set GLM5_GPU_OK=1 (rig lock held, gateway stopped)"; exit 2; }
D=/home/ronald/src/colibri-m1/tools/hot-expert/franken/decode
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=${GLM5_PLACEMENT:-/home/ronald/bench/m2/glm}   # M2 histogram (technical record): the STARTING set; --adapt 1 learns
EG=${GLM5_EXPERT_GB:-17}                        # GB of experts a card (GLM5.md 5: ~51 GB resident)
O=/home/ronald/bench/franken/glm5/gpu
B=/home/ronald/bench/franken/glm5
T6="154822 785 6722 315 9621 374"               # [gMASK] "The capital of France is"
CPU=$D/franken_decode_glm_cpu                   # make cpu CPU_BIN=franken_decode_glm_cpu
GPU=$D/franken_decode_glm                       # make gpu GPU_BIN=franken_decode_glm
what=${1:-all}
mkdir -p $O
nb() { echo "not_bitexact=$(grep '^oracle .* cos=' $1 | grep -v 'cos=1.000000 maxabs=0 ' | wc -l) taps=$(grep -c '^oracle .* cos=' $1) $(grep -oE 'GLM5 (CPU|GPU) ORACLE [A-Z]+' $1)"; }
mc() { echo "min_cos=$(grep '^oracle .* cos=' $1 | sed 's/.* cos=\([-0-9.]*\).*/\1/' | sort -g | head -1) taps=$(grep -c '^oracle .* cos=' $1) $(grep -oE 'GLM5 (CPU|GPU) ORACLE [A-Z]+' $1)"; }
G="--model $M --placement $P --expert-gb $EG"
if [ $what = cpu ] || [ $what = all ]; then
  echo "=== (1) CPU references (no GPU): the whole model + head, 6 tokens, greedy 16, ctx 4096 (scoring on)"
  $CPU --model $M --tokens $T6 --cpu --ctx 4096 --threads 8 --greedy 16 --dump $O/cpu6 > $O/cpu6.log 2>&1
  echo "rc=$?"; grep -E "greedy_ids|dumped" $O/cpu6.log
  if [ ! -f $O/cpu2200/index.txt ]; then
    echo "=== (1b) layers 0-3, 2 200 prose tokens (the top-512 selecting), ~60 min on the CPU arm"
    $CPU --model $M --tokens-file $B/prose2200.txt --cpu --layers 0-3 --no-head --ctx 4096 --threads 8 \
      --dump $O/cpu2200 > $O/cpu2200.log 2>&1; echo "rc=$?"; grep -E "dumped" $O/cpu2200.log
  fi
fi
if [ $what = gpu6 ] || [ $what = all ]; then
  echo "=== (2) GPU vs the CPU dump, 6 tokens + greedy 16: first under --sync-debug (the HIP multi-GPU rule), then without"
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --oracle $O/cpu6 --sync-debug > $O/gpu6_sync.log 2>&1; echo "rc=$? $(mc $O/gpu6_sync.log)"
  grep -E "placement dev|vram_dev|host_mirror|greedy_ids|oracle summary|FAIL|SYNC-DEBUG|error" $O/gpu6_sync.log | head -40
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --oracle $O/cpu6 > $O/gpu6.log 2>&1; echo "rc=$? $(mc $O/gpu6.log)"
  grep -E "greedy_ids|oracle summary|adapt_total" $O/gpu6.log
fi
if [ $what = ident ] || [ $what = all ]; then
  echo "=== (3) bit-identity on the GPU: the owner-card path (--three-link 0 --adapt 0) dumps; the three-link"
  echo "        split, adaptive placement with a swap every token, and both, each --oracle it: every tap cos=1 maxabs=0"
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --three-link 0 --adapt 0 --dump $O/g6_base > $O/g6_base.log 2>&1; echo "base rc=$?"
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --three-link 1 --adapt 0 --oracle $O/g6_base > $O/g6_tl.log 2>&1
  echo "three-link rc=$? $(nb $O/g6_tl.log)"
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --three-link 0 --adapt 1 --adapt-every 1 --adapt-mb-per-token 512 \
    --adapt-verify 1 --oracle $O/g6_base > $O/g6_ad.log 2>&1
  echo "adapt rc=$? $(nb $O/g6_ad.log)"; grep -E "^adapt_total|^adapt_verify" $O/g6_ad.log
  $GPU $G --tokens $T6 --ctx 4096 --greedy 16 --three-link 1 --adapt 1 --adapt-every 1 --adapt-mb-per-token 512 \
    --adapt-verify 1 --oracle $O/g6_base > $O/g6_tl_ad.log 2>&1
  echo "three-link+adapt rc=$? $(nb $O/g6_tl_ad.log)"; grep -E "^adapt_verify" $O/g6_tl_ad.log
fi
if [ $what = long ] || [ $what = all ]; then
  echo "=== (4) past 2 048 tokens: layers 0-3 on the GPU vs the CPU arm, 2 200 prose tokens (the pool top-512 at work)"
  C22=$O/cpu2200; [ -f $C22/index.txt ] || C22=$B/mine_oracle2200     # the step-2 CPU dump (same taps)
  $GPU $G --tokens-file $B/prose2200.txt --layers 0-3 --no-head --ctx 4096 --oracle $C22 > $O/gpu2200.log 2>&1
  echo "rc=$? $(mc $O/gpu2200.log)"; grep -E "indexer_top_k|indexer_pool_score|l_last-3" $O/gpu2200.log
fi
if [ $what = time ] || [ $what = all ]; then
  echo "=== (5) 262 144 cells allocated: greedy 16, then 32 timed decode tokens; three-link 1 / 0 / 0 / 1 (A,B,B,A)"
  for tl in 1 0 0 1; do
    $GPU $G --tokens $T6 --ctx 262144 --greedy 16 --time 32 --three-link $tl > $O/time_tl$tl.$$.log 2>&1
    echo "three-link=$tl rc=$? $(grep -E 'glm5_decode_ms' $O/time_tl$tl.$$.log)"
  done
  grep -E "vram_dev|glm5_cache|greedy_ids|adapt_total" $O/time_tl1.$$.log | head -12
  echo "=== (5b) the class table (--profile; profiled tokens pay one event sync each)"
  $GPU $G --tokens $T6 --ctx 262144 --greedy 16 --time 32 --profile > $O/time_prof.log 2>&1; echo "rc=$?"
  grep -E "glm5_decode_ms|^---|glm5_|prof_|trunk|expert|router|boundary" $O/time_prof.log | head -80
fi
if [ $what = probe ] || [ $what = all ]; then
  echo "=== (6) depth probe at 262 144 cells: decode 32 tokens at depth 64 and at 8 192 (prefill token by token)"
  $GPU $G --tokens-file $B/prose8400.txt --ctx 262144 --probe-at 64,8192 --probe-n 32 > $O/probe.log 2>&1
  echo "rc=$?"; grep -E "glm5_probe|adapt_all|adapt_total" $O/probe.log | tail -20
fi
echo "=== GLM5 GPU GATE DONE ($what)"
