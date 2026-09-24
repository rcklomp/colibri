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
#   DS4_GPU_OK=1 ds4_gpu_gate.sh [s5-ident|s5-chunk|s5-long|s5-depth|s5-prefill|step5]   (L5 step 5, section 13)
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
# ---- L5 step 5: the depth cost and the batched prefill (DEEPSEEK4.md section 13)
PF=/home/ronald/bench/franken/ds4/pf                  # base_gpu: bc946e2's binary; c1_long: the CPU dump
MIX=/home/ronald/bench/m2/deepseek_mix                # step 3b's mixed-corpus placement
MM=$A/mmlu_8k_ids.txt
T136="$T"; for i in $(seq 26); do T136="$T136 671 6102 294 8760 344"; done
nb() { echo "not_bitexact=$(grep '^oracle .* cos=' $1 | grep -v 'cos=1.000000 maxabs=0 ' | wc -l) taps=$(grep -c '^oracle .* cos=' $1)"; }
if [ $what = s5-ident ] || [ $what = step5 ]; then
  echo "=== (11) decode unchanged: bc946e2's binary --dump, this one --oracle (eager, then graph): every tap cos=1 maxabs=0"
  $PF/base_gpu $G --ctx 512 --greedy 16 --dump $O/s5_base > $O/s5_base.log 2>&1; echo "rc=$?"
  $GPU $G --ctx 512 --greedy 16 --oracle $O/s5_base > $O/s5_new.log 2>&1; echo "rc=$? $(nb $O/s5_new.log)"
  grep -E "oracle summary|ORACLE|greedy_ids" $O/s5_new.log
  $GPU $G --ctx 512 --greedy 16 --hip-graph 1 --oracle $O/s5_base > $O/s5_new_graph.log 2>&1; echo "rc=$? $(nb $O/s5_new_graph.log)"
fi
if [ $what = s5-chunk ] || [ $what = step5 ]; then
  echo "=== (12) the chunk gate: --chunk 1 --dump, then --chunk C --oracle (6 tokens: one chunk; 136: several, pipelined)"
  $GPU $G --ctx 512 --greedy 16 --chunk 1 --dump $O/s5_c1 > $O/s5_c1.log 2>&1; echo "rc=$?"
  $GPU $G --ctx 512 --greedy 16 --chunk 6 --oracle $O/s5_c1 > $O/s5_c6.log 2>&1; echo "chunk=6 rc=$? $(nb $O/s5_c6.log) $(grep -c greedy_ids $O/s5_c6.log)"
  $GPU $G --ctx 512 --greedy 16 --chunk 6 --ds4-expert-gather 0 --oracle $O/s5_c1 > $O/s5_c6g0.log 2>&1; echo "chunk=6 gather=0 rc=$? $(nb $O/s5_c6g0.log)"
  G136="--model $M --tokens $T136 --placement $P --expert-gb 20"
  $GPU $G136 --ctx 512 --greedy 16 --chunk 1 --dump $O/s5_c1_136 > $O/s5_c1_136.log 2>&1; echo "rc=$?"
  for c in 7 32 128; do
    $GPU $G136 --ctx 512 --greedy 16 --chunk $c --oracle $O/s5_c1_136 > $O/s5_c${c}_136.log 2>&1; echo "136 chunk=$c rc=$? $(nb $O/s5_c${c}_136.log)"
  done
  $GPU $G136 --ctx 512 --greedy 16 --chunk 32 --prefill-pipeline 0 --oracle $O/s5_c1_136 > $O/s5_c32np_136.log 2>&1; echo "136 chunk=32 pipeline=0 rc=$? $(nb $O/s5_c32np_136.log)"
  grep -h "greedy_ids" $O/s5_c1_136.log $O/s5_c128_136.log
fi
if [ $what = s5-long ] || [ $what = step5 ]; then
  echo "=== (12b) past 2 048 tokens (the top-512 selects, the new top-k and indexer kernels):"
  echo "        GPU layers 0-3 against the CPU arm's dump of the same 2 200 prose tokens (cos >= 0.999),"
  echo "        then the whole model, --chunk 256 against --chunk 1 (bit for bit) and graph against eager"
  L4="--model $M --tokens-file $MM --max-tokens 2200 --layers 0-3 --no-head --ctx 4096 --placement $MIX --expert-gb 20"
  $GPU $L4 --chunk 256 --oracle $PF/c1_long > $O/s5_long_l03.log 2>&1; echo "rc=$?"
  grep -E "oracle summary|ORACLE|lid_top_k|attn_csa" $O/s5_long_l03.log | head -8
  LF="--model $M --tokens-file $MM --max-tokens 2200 --placement $MIX --expert-gb 20 --ctx 4096 --greedy 16"
  $GPU $LF --chunk 1 --dump $O/s5_long_c1 > $O/s5_long_c1.log 2>&1; echo "rc=$?"
  $GPU $LF --chunk 256 --oracle $O/s5_long_c1 > $O/s5_long_c256.log 2>&1; echo "chunk=256 rc=$? $(nb $O/s5_long_c256.log)"
  $GPU $LF --chunk 256 --hip-graph 1 --oracle $O/s5_long_c1 > $O/s5_long_c256g.log 2>&1; echo "chunk=256 graph rc=$? $(nb $O/s5_long_c256g.log)"
  grep -h "greedy_ids" $O/s5_long_c1.log $O/s5_long_c256.log $O/s5_long_c256g.log
fi
if [ $what = s5-depth ] || [ $what = step5 ]; then
  echo "=== (13) the depth probe: decode 32 tokens at depth 64, 8 192, then 64 again (same process: depth"
  echo "        against run duration), 262 144 cells, the mixed placement, the prompt prefilled in chunks of 256"
  DP="--model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32"
  $GPU $DP --profile > $O/s5_depth_prof.log 2>&1; echo "(13a) eager --profile rc=$?"
  grep -E "^ds4_probe|^--- probe|prof_ds4_(indexer|topk|attention|compressor|rope|trunk|expert|miss)|prof_total|prof_busy|probe_miss" $O/s5_depth_prof.log
  for g in 0 1; do
    $GPU $DP --hip-graph $g > $O/s5_depth_g$g.log 2>&1; echo "(13b) hip_graph=$g rc=$?"; grep -E "^ds4_probe|hip_graph_dev" $O/s5_depth_g$g.log
  done
  $GPU $DP --hip-graph 1 --hip-graph-bucket 262144 > $O/s5_depth_g1big.log 2>&1; echo "(13c) one graph class (no updates) rc=$?"; grep -E "^ds4_probe|hip_graph_dev" $O/s5_depth_g1big.log
  echo "=== (13d) the ADAPT trajectory's configuration (technical placement, --adapt 1, no swaps, graph)"
  $GPU --model $M --tokens-file $MM --placement $P --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32 \
    --hip-graph 1 --adapt 1 --adapt-every 64 --adapt-mb-per-token 0 > $O/s5_depth_adapt.log 2>&1; echo "rc=$?"
  grep -E "^ds4_probe|^adapt_total" $O/s5_depth_adapt.log  echo "=== (13e) the prompt TOKEN BY TOKEN (--chunk 1), as the step-4 trajectory fed it: if (13a-c) show no"
  echo "        growth and this does, the cost is the long run, not the depth (~8-13 min)"
  $GPU --model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 1 --probe-at 64,8192,64 \
    --probe-n 32 --hip-graph 1 > $O/s5_depth_c1.log 2>&1; echo "rc=$?"; grep -E "^ds4_probe" $O/s5_depth_c1.log
fi
if [ $what = s5-prefill ] || [ $what = step5 ]; then
  echo "=== (14) batched prefill, 262 144 cells, chunks of 256, the prompt = the MMLU prose (cycled past 8 192),"
  echo "        then 32 decode tokens at that depth (graph replay)"
  for n in 8192 32768; do
    TP="--model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 256"
    $GPU $TP --time-prefill $n --time 32 --hip-graph 1 > $O/s5_prefill_$n.log 2>&1; echo "n=$n rc=$?"
    grep -E "^prefill_tokens|ds4_decode_ms" $O/s5_prefill_$n.log
  done
  echo "=== (14b) --profile of the 8 192 prefill (per prompt token, chunks 2..): where a chunk's time goes"
  $GPU $TP --time-prefill 8192 --profile > $O/s5_prefill_prof.log 2>&1; echo "rc=$?"
  grep -E "^prefill_tokens|^--- prefill|prof_ds4|prof_total|prof_busy|prefill_miss" $O/s5_prefill_prof.log
  echo "=== (14c) knobs, 8 192 tokens: --chunk 512; --gemm-lds 1 (a summation-order change); --ds4-expert-gather 0"
  $GPU $TP --time-prefill 8192 --chunk 512 > $O/s5_prefill_c512.log 2>&1; echo "chunk=512 rc=$? $(grep -E '^prefill_tokens' $O/s5_prefill_c512.log)"
  $GPU $TP --time-prefill 8192 --gemm-lds 1 > $O/s5_prefill_lds.log 2>&1; echo "gemm_lds=1 rc=$? $(grep -E '^prefill_tokens' $O/s5_prefill_lds.log)"
  $GPU $TP --time-prefill 8192 --ds4-expert-gather 0 > $O/s5_prefill_g0.log 2>&1; echo "gather=0 rc=$? $(grep -E '^prefill_tokens' $O/s5_prefill_g0.log)"
fi
echo "=== DONE"
