#!/bin/bash
# glm_pf4_chain.sh -- PF4 (ACTION-LIST row 5e, reopened by PF14): prefill chunks of 2 048 rows on the IQ3_XXS-expert hybrid GGUF, with the VRAM the smaller experts free.
# One GPU process (franken-engine pf14-iq2s `franken_decode_glm_pf14`, hybrid, `--expert-gb 13`: the same 4 289 resident experts as 17 GB of IQ4_XS, ~4 GB a card free):
#   1. exactness: the whole model on 2 200 prose ids (the pool top-512 regime): chunk 1024 is this file's reference (dump); chunk 2 048 (pipelined, not pipelined, with adaptation) and
#      chunk 1 500 must match it bit for bit (every tap cos = 1 maxabs 0, greedy ids exact): the tiled chunk-plan kernel and every buffer that grew with the chunk;
#   2. timing, A,B,B,A in ONE process (a warm-up arm first: the first arm of a process reads the host mirror cold): chunk 1024 against 2 048 on prose 8 192, technical 8 192 and technical
#      32 768 tokens, each with 32 decode tokens at the depth after a 64-token settle;
#   3. the steady VRAM lines of every arm: what chunk 2 048 leaves free.
# Launch: setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf4_chain.sh > ~/bench/glm_pf4_chain.log 2>&1 < /dev/null &
# Env: PF4_D (decode dir of the pf14 build), PF4_OUT, PF4_EG (default 13), PF4_MODEL.
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF4_D:-$HOME/src/franken-engine-pf14/franken/decode}
GPU=$D/franken_decode_glm_pf14; VERDICT=$D/glm5_gate_verdict.sh
O=${PF4_OUT:-$HOME/bench/franken/glm5/pf4}; rm -rf "$O"; mkdir -p "$O"
M=${PF4_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
EG=${PF4_EG:-13}; P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
X="--tokens-file $B/prose2200.txt --ctx 4096 --greedy 4"
BASE="--ctx 262144 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1"
say_end() { echo "=== glm_pf4 exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$GPU" "$VERDICT" "$M" "$PR" "$REC" "$B/prose2200.txt"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg x_ref    ref   $X --chunk 1024 --dump "$O/g2200_c1024"
cfg x_c2048  exact $X --chunk 2048 --oracle "$O/g2200_c1024"
cfg x_c2048n exact $X --chunk 2048 --prefill-pipeline 0 --oracle "$O/g2200_c1024"
cfg x_c1500  exact $X --chunk 1500 --oracle "$O/g2200_c1024"
cfg x_c2048a exact $X --chunk 2048 $AD --oracle "$O/g2200_c1024"
cfg w0       none  --tokens-file $PR $BASE --chunk 2048 --time-prefill 4096
for f in p:$PR:8192 r:$REC:8192 l:$REC:32768; do
  k=${f%%:*}; rest=${f#*:}; file=${rest%%:*}; n=${rest#*:}
  for a in c1024_a c2048_a c2048_b c1024_b; do
    ch=${a%%_*}; ch=${ch#c}
    cfg ${k}_$a none --tokens-file $file $BASE --chunk $ch --time-prefill $n --time 32 --time-settle 64
  done
done
echo "=== glm_pf4 start $(date -Is) bin=$(sha256sum "$GPU" | cut -c1-16) model=$(basename "$M") expert-gb=$EG"; cut -c1-120 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I "$HOME/bench/gpu_sampler.py" run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
  -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$GPU" --model "$M" --placement "$P" --expert-gb "$EG" --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-210 | tee "$O/verdict.txt"
[ "$rc" -ne 0 ] && { echo "--- engine log tail:"; tail -n 16 "$O/gate_run.log" | cut -c1-210; }
echo "=== prefill ms/token (whole prompt, first chunk included) and decode median, chunk 1024 against 2048 (A,B,B,A in one process)"
LC_ALL=C awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); p[c] = a[2] } /^glm5_decode_ms_median=/ { split($1, a, "="); d[c] = a[2] }
  END { split("p r l", K, " "); split("prose8192 technical8192 technical32768", N, " ")
        for (i = 1; i <= 3; i++) { k = K[i]
          if (p[k "_c1024_a"] == "" || p[k "_c1024_b"] == "" || p[k "_c2048_a"] == "" || p[k "_c2048_b"] == "") { printf "--- %-15s an arm is missing\n", N[i]; continue }
          a = (p[k "_c1024_a"] + p[k "_c1024_b"]) / 2; b = (p[k "_c2048_a"] + p[k "_c2048_b"]) / 2
          da = (d[k "_c1024_a"] + d[k "_c1024_b"]) / 2; db = (d[k "_c2048_a"] + d[k "_c2048_b"]) / 2
          printf "--- %-15s chunk 1024 %.4f (%s %s)   chunk 2048 %.4f (%s %s)   2048/1024 %.4f (%+.1f %%, %+.3f ms/token)   decode %.2f / %.2f\n", N[i], a, p[k "_c1024_a"], p[k "_c1024_b"], b, p[k "_c2048_a"], p[k "_c2048_b"], b / a, 100 * (b / a - 1), b - a, da, db } }' "$O/gate_run.log"
echo "=== steady VRAM free (MB) a card, after each arm"
grep -aE "^config|^p_c|^r_c|^l_c|^x_c|^w0" "$O/verdict.txt" | cut -c1-120 | head -24
say_end "$rc"
