#!/bin/bash
# glm_hyb_chain.sh -- PF14 (ACTION-LIST row 5e2, plan section 0g): the IQ2_S routed-expert kernel on the IQ3_XXS-expert HYBRID GGUF (gguf_hybrid.py: our IQ4_XS metadata + trunk, Unsloth
# UD-IQ3_XXS experts), and what it is worth, in one chain:
#   1. the CPU arm of the engine on the hybrid (6 tokens + greedy 16): the reference of this file (no other reference applies: its numbers differ from the IQ4_XS ones);
#   2. ONE GPU process on the hybrid: the GPU against that CPU dump (cos), then the eager chunk-1 owner-card run as this file's own reference and every schedule held to it bit for bit
#      (three-link, graph replay, chunks 6 / 7 / 32 / 136, adaptation) -- the IQ2_S kernel in both the decode path (k_glm5_moe, stage) and the chunk path (k_glm5_moe_blk) -- then the
#      timing arms (prefill ms/token on 8 192 prose tokens and on technical text, decode ms at depth ~8 290), `--expert-gb 17`, shipped flags;
#   3. ONE GPU process on the served IQ4_XS with the same timing arms (the paired comparison; two processes, not interleaved: say so when quoting);
#   4. ONE GPU process on the hybrid at `--expert-gb 13`: the same number of resident experts as 17 GB of IQ4_XS (4 289) takes ~13 GB of the smaller experts, so ~4 GB a card is free:
#      the steady VRAM lines are the budget for PF4 (chunk 2 048), and the timing arms say what decode does with the same resident count and 23 % smaller missed bytes.
# Launch (rig lock): setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_hyb_chain.sh > ~/bench/glm_hyb_chain.log 2>&1 < /dev/null &
# Watch with ckpt1006/watch_chain.sh ~/bench/glm_hyb_chain.log ~/bench/franken/glm5/hyb/gate_hyb.log. Env: HYB_D (decode dir with the pf14 binaries), HYB_OUT, HYB_ONLY (comma list of stages).
set -u
. "$HOME/bench/chain_preflight.sh"
D=${HYB_D:-$HOME/src/franken-engine/franken/decode}
GPU=$D/franken_decode_glm_pf14; CPU=$D/franken_decode_glm_cpu; VERDICT=$D/glm5_gate_verdict.sh
O=${HYB_OUT:-$HOME/bench/franken/glm5/hyb}; mkdir -p "$O"
HYB=${HYB_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}   # HYB_MODEL: another hybrid (the UD-Q3_K_XL-expert one needs the IQ3_XXS kernel: build ff6348a or later)
IXS=$HOME/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
T6="154822 785 6722 315 9621 374"
S6="--tokens $T6 --ctx 4096 --greedy 16"; C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1"
TPD="--tokens-file $PR $BASE --time-prefill 8192 --time 32 --time-settle 64"      # prefill 8 192 prose tokens, then decode at depth ~8 290
TPR="--tokens-file $REC $BASE --time-prefill 8192"                                 # technical text
ONLY=${HYB_ONLY:-1,2,3,4}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
say_end() { echo "=== glm_hyb exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$GPU" "$CPU" "$VERDICT" "$HYB" "$IXS" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_hyb start $(date -Is) gpu=$(sha256sum "$GPU" | cut -c1-16) hybrid=$(stat -c %s "$HYB") bytes stages=$ONLY"
rig_quiet_wait 1800 || say_end 3
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
dk() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
         -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full "$@"; }
rc_all=0
# ---- 1. the CPU reference of the hybrid ----
if has 1; then
  echo "=== 1. CPU reference $(date +%T)"
  rm -rf "$O/cpu6"
  timeout 10800 docker run --rm --security-opt seccomp=unconfined --ulimit memlock=-1 -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
     "$CPU" --model "$HYB" --tokens $T6 --cpu --ctx 4096 --threads 8 --greedy 16 --dump "$O/cpu6" > "$O/cpu6.log" 2>&1
  rc=$?; echo "cpu rc=$rc"; grep -aE "greedy_ids|dumped" "$O/cpu6.log" | cut -c1-200
  [ "$rc" -eq 0 ] && [ -f "$O/cpu6/index.txt" ] || { echo "--- cpu log tail"; tail -n 12 "$O/cpu6.log" | cut -c1-200; say_end 4; }
fi
# ---- 2. the GPU gate + timing on the hybrid ----
gate() {   # gate <name> <model> <expert-gb> <plan file> <checks file>
  local name=$1 model=$2 eg=$3 plan=$4 chk=$5
  : > "$O/gate_$name.log"
  python3 -I "$HOME/bench/gpu_sampler.py" run "$O/samples_$name.tsv" "$O/gate_$name.log" 0.25 & SAMP=$!
  dk "$GPU" --model "$model" --placement "$P" --expert-gb "$eg" --gate-plan "$plan" > "$O/gate_$name.log" 2>&1
  local rc=$?; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  echo "[$name] engine rc=$rc"
  grep -aE "vram_dev[0-9]_(steady|used)|HIP error|Memory access|out of memory|FATAL" "$O/gate_$name.log" | sort | uniq -c | head -10 | cut -c1-160
  [ -n "$chk" ] && "$VERDICT" "$O/gate_$name.log" "$chk" $rc 2>&1 | cut -c1-200 | tee "$O/verdict_$name.txt"
  [ "$rc" -ne 0 ] && { echo "--- engine log tail ($name)"; tail -n 14 "$O/gate_$name.log" | cut -c1-200; }
  return $rc
}
mk() { local f=$1; shift; : > "$O/plan_$f.txt"; : > "$O/checks_$f.txt"; }
cfg() { local f=$1 n=$2 c=$3; shift 3; echo "$n $*" >> "$O/plan_$f.txt"; echo "$n $c" >> "$O/checks_$f.txt"; }
if has 2; then
  mk hyb
  cfg hyb hy_sync6  cos   $S6 --oracle "$O/cpu6" --sync-debug
  cfg hyb hy_gpu6   cos   $S6 --oracle "$O/cpu6"
  cfg hyb hy_eager6 ref   $S6 --three-link 0 --adapt 0 --chunk 1 --hip-graph 0 --dump "$O/g6_eager"
  cfg hyb hy_tl6    exact $S6 --three-link 1 --adapt 0 --oracle "$O/g6_eager"
  cfg hyb hy_graph6 exact $S6 --three-link 0 --adapt 0 --hip-graph 1 --oracle "$O/g6_eager"
  cfg hyb hy_chunk6 exact $S6 --chunk 6 --oracle "$O/g6_eager"
  cfg hyb hy_e136   ref   $C136 --three-link 0 --adapt 0 --chunk 1 --hip-graph 0 --dump "$O/g136_eager"
  cfg hyb hy_c7     exact $C136 --chunk 7 --oracle "$O/g136_eager"
  cfg hyb hy_c32    exact $C136 --chunk 32 --oracle "$O/g136_eager"
  cfg hyb hy_c136np exact $C136 --chunk 136 --prefill-pipeline 0 --oracle "$O/g136_eager"
  cfg hyb hy_c32ad  exact $C136 --chunk 32 $AD --oracle "$O/g136_eager"
  cfg hyb hy_c32g   exact $C136 --chunk 32 --hip-graph 1 --oracle "$O/g136_eager"
  cfg hyb hy_t_a    none  $TPD
  cfg hyb hy_r_a    none  $TPR
  cfg hyb hy_t_b    none  $TPD
  cfg hyb hy_r_b    none  $TPR
  echo "=== 2. GPU gate + timing on the hybrid, expert-gb 17 $(date +%T)"
  gate hyb "$HYB" 17 "$O/plan_hyb.txt" "$O/checks_hyb.txt" || rc_all=5
fi
# ---- 3. the served IQ4_XS, same timing arms ----
if has 3; then
  mk ixs
  cfg ixs ix_t_a none $TPD; cfg ixs ix_r_a none $TPR; cfg ixs ix_t_b none $TPD; cfg ixs ix_r_b none $TPR
  echo "=== 3. IQ4_XS timing, expert-gb 17 $(date +%T)"
  gate ixs "$IXS" 17 "$O/plan_ixs.txt" "" || rc_all=6
fi
# ---- 4. the hybrid at the expert budget that frees ~4 GB a card ----
if has 4; then
  mk h13
  cfg h13 h13_t_a none $TPD; cfg h13 h13_r_a none $TPR; cfg h13 h13_t_b none $TPD; cfg h13 h13_r_b none $TPR
  echo "=== 4. hybrid timing, expert-gb 13 $(date +%T)"
  gate h13 "$HYB" 13 "$O/plan_h13.txt" "" || rc_all=7
fi
# ---- report ----
echo "=== steady VRAM (MB free a card, after the first full-size chunk)"
for n in hyb ixs h13; do [ -f "$O/gate_$n.log" ] && printf "  %-4s " $n && grep -aE "^vram_dev[0-9]_steady" "$O/gate_$n.log" | head -3 | sed 's/vram_dev\([0-9]\)_steady free_mb=\([0-9]*\).*/dev\1 \2/' | tr '\n' ' ' && echo; done
LC_ALL=C awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); p[c] = a[2] } /^glm5_decode_ms_median=/ { split($1, a, "="); d[c] = a[2] }
  END { printf "=== ms/token, whole prompt (prefill) and decode median at depth ~8 290 -- two processes, not interleaved\n"
        split("hy ix h13", G, " "); split("hybrid-17 iq4xs-17 hybrid-13", N, " ")
        for (i = 1; i <= 3; i++) { g = G[i]; t = (g == "h13" ? "h13_t_" : g "_t_"); r = (g == "h13" ? "h13_r_" : g "_r_")
          tp = (p[t "a"] + p[t "b"]) / 2; rp = (p[r "a"] + p[r "b"]) / 2; dd = (d[t "a"] + d[t "b"]) / 2
          printf "  %-11s prefill prose 8k %.3f   technical 8k %.3f   decode %.2f   (arms: %s %s | %s %s | dec %s %s)\n", N[i], tp, rp, dd, p[t "a"], p[t "b"], p[r "a"], p[r "b"], d[t "a"], d[t "b"] } }' "$O"/gate_*.log
say_end $rc_all
