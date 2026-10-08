#!/bin/bash
# glm_pf0_main_chain.sh -- PF0 (b) the device timeline + the profile and (c) the DEPTH CURVE of the SHIPPED prefill (plan Rev 102, PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md section 3).
# Binary: franken-engine main (code a7bf81a, the installed .d14) built plain as franken_decode_glm_pf0main. Config = the served one: chunk 1 024, in place (--glm-prefill-stage 0),
# --gemm-lds 1, FRANKEN_GLM_MOE_G=8 (an env, not a flag: the docker run passes it), --prefill-pipeline 1 (default), 262 144 cells, swaps held (--adapt-prefill 0), experts 17 GB a card
# from the generic M2 placement (the served placement is the one the earlier chats adapted: not reproducible here, say so in the record).
# ONE process, one load, in this order:
#   w0        8 192 prose tokens, discarded (first-arm effects: cold cards boost higher, the first chunk pays the one-time buffer growths)
#   smoke     2 048 tokens, --timeline of both chunks: a fault in the timeline path in place costs a minute, not the pair
#   tl_a/t/b  8 192 prose tokens: off / --timeline chunks 2-7 (-> timeline.csv) / off   (the timeline's own cost = tl_t against mean(tl_a, tl_b))
#   pf        8 192 prose tokens --profile (per-class device time per token for chunks 2.., fetch MB and link rate of the in-place reads per card)
#   rec*      the DEPTH CURVE: the first 2 048 / 8 192 / 32 768 / 65 536 tokens of the measurement record (the needle test's own corpus, tokenised by llama-tokenize with [gMASK]<sop> first,
#             ~/bench/franken/glm5/rec_depth/rec_ids.txt), ms/token of the whole prompt, A,B,C,D,D,C,B,A. The slope between two points is the marginal ms/token at that depth.
# VRAM budget (CLAUDE.md "Before a build that allocates device memory"): nothing is allocated beyond what the shipped config allocates (this is the shipped config at its largest shapes: 262 144
# cells, chunk 1 024); --timeline adds host memory and 32 768 hipEvents a card (event pool, untimed at attach), --profile a few events a layer.
# Launch (on the rig):  ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf0_main_chain.sh > ~/bench/glm_pf0_main_chain.log 2>&1 < /dev/null &
# Watch from the Mac:   tools/hot-expert/ckpt1006/watch_chain.sh ~/bench/glm_pf0_main_chain.log ~/bench/franken/glm5/pf0_main/gate_run.log
# Env: PF0_BIN, PF0_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF0_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf0main}
O=${PF0_OUT:-$HOME/bench/franken/glm5/pf0_main}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
TLPY=${PF0_TLPY:-$HOME/src/franken-engine/franken/decode/glm5_timeline.py}
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0"
say_end() { echo "=== glm_pf0_main exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
[ "$(wc -w < "$REC")" -ge 66000 ] || { echo "FATAL: $REC has fewer than 66 000 ids"; say_end 2; }
{ echo "w0       --tokens-file $PR $SHIP --time-prefill 8192"
  echo "smoke    --tokens-file $PR $SHIP --time-prefill 2048 --timeline $O/timeline_smoke.csv --timeline-skip 0 --timeline-chunks 2"
  echo "tl_a     --tokens-file $PR $SHIP --time-prefill 8192"
  echo "tl_t     --tokens-file $PR $SHIP --time-prefill 8192 --timeline $O/timeline.csv --timeline-skip 2 --timeline-chunks 6"
  echo "tl_b     --tokens-file $PR $SHIP --time-prefill 8192"
  echo "pf       --tokens-file $PR $SHIP --time-prefill 8192 --profile"
  for n in 2048 8192 32768 65536; do echo "rec${n}_a --tokens-file $REC $SHIP --time-prefill $n"; done
  for n in 65536 32768 8192 2048; do echo "rec${n}_b --tokens-file $REC $SHIP --time-prefill $n"; done
} > "$O/plan.txt"
echo "=== glm_pf0_main start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"; cut -c1-60 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|glm5_timeline " "$O/gate_run.log" | head -8 | cut -c1-200
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-140
[ "$rc" -ne 0 ] && { echo "--- engine log tail:"; tail -n 12 "$O/gate_run.log" | cut -c1-200; }
echo "=== prefill ms/token per config (whole prompt, first chunk included)"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($1, nn, "="); split($5, a, "="); printf "%-10s %6d tokens %9.4f ms/token = %6.1f tok/s\n", c, nn[2], a[2], 1000 / a[2]; ms[c] = a[2] }
  END { printf "--- timeline cost: tl_t %.4f vs mean(tl_a, tl_b) %.4f\n", ms["tl_t"], (ms["tl_a"] + ms["tl_b"]) / 2
        split("2048 8192 32768 65536", D, " "); prev = ""
        for (i = 1; i <= 4; i++) { n = D[i]; va = ms["rec" n "_a"]; vb = ms["rec" n "_b"]; if (va == "" || vb == "") continue; m = (va + vb) / 2; tot = m * n
          printf "--- depth %6d: a %.4f b %.4f mean %.4f ms/token (total %.1f ms)", n, va, vb, m, tot
          if (prev != "") printf "   marginal %.4f ms/token from %d to %d", (tot - ptot) / (n - prev), prev, n
          printf "\n"; prev = n; ptot = tot } }' "$O/gate_run.log"
echo "=== profile (chunks 2.., per prompt token)"
sed -n '/^prefill (per prompt token/,/^glm5_side_dev2/p' "$O/gate_run.log" | cut -c1-240 | head -60
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-160
if [ -s "$O/timeline.csv" ]; then
  echo "=== timeline report"
  python3 -I "$TLPY" "$O/timeline.csv" --tokens 1024 --layers --gantt 4 > "$O/timeline_report.txt" 2>&1
  echo "glm5_timeline.py rc=$? -> $O/timeline_report.txt"
  sed -n '/^=== steady state/,/^  gantt/p' "$O/timeline_report.txt" | cut -c1-220 | head -70
fi
say_end "$rc"
