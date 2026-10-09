#!/bin/bash
# glm_pf9_main_chain.sh -- PF9 (plan Rev 102, PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md section 0b): the device timeline, the profile and the DEPTH CURVE of the prefill WITH --glm-help-copy 1 (the shipped default since
# the .hc install). glm_pf0_main_chain.sh did this for the stalled schedule (8-chunk timeline, period still falling at chunk 7, so the 3.60 ms/token was mostly pipeline fill); this one asks what the
# STEADY STATE looks like: a 32-chunk prompt, the timeline of chunks 16-21, per-card busy shares, the period, and the ms/token curve out to 64 k.
# Binary: franken_decode_glm_pf9main = franken-engine main (df401f7, engine code f2cbfa3) built UNPATCHED (PF0_PLAIN=1 PF0_NAME=franken_decode_glm_pf9main build_pf0dbg.sh). Config = the served one:
# chunk 1 024, in place (--glm-prefill-stage 0), --gemm-lds 1, FRANKEN_GLM_MOE_G=8, --prefill-pipeline 1 (default), 262 144 cells, swaps held (--adapt-prefill 0), experts 17 GB a card from the generic M2
# placement (the served placement is not reproducible here), --glm-help-copy 1 on EVERY config.
# ONE process, one load, in this order:
#   w0      8 192 prose tokens, discarded
#   pf      8 192 prose tokens --profile (per-class device time per token, chunks 2.., all three cards)
#   tl_a/t/b  32 768 tokens of the record text: off / --timeline chunks 16-21 (-> timeline32.csv) / off  (the timeline's own cost = tl_t against mean(tl_a, tl_b))
#   pf32    32 768 tokens of the record text --profile (31 chunks of steady state)
#   rec*    the DEPTH CURVE: the first 2 048 / 8 192 / 32 768 / 65 536 tokens of the record (~/bench/franken/glm5/rec_depth/rec_ids.txt), A,B,C,D,D,C,B,A. The slope between two points is the marginal
#           ms/token at that depth; the 2 k point carries the pipeline fill, the 64 k point the least of it.
# VRAM budget: nothing is allocated beyond what the shipped config allocates (262 144 cells, chunk 1 024); --timeline adds host memory and 32 768 hipEvents a card, --profile a few events a layer.
# Launch (on the rig):  ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf9_main_chain.sh > ~/bench/glm_pf9_main_chain.log 2>&1 < /dev/null &
# Watch from the Mac:   tools/hot-expert/ckpt1006/watch_chain.sh ~/bench/glm_pf9_main_chain.log ~/bench/franken/glm5/pf9_main/gate_run.log
# Env: PF9_BIN, PF9_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF9_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf9main}
O=${PF9_OUT:-$HOME/bench/franken/glm5/pf9_main}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
TLPY=${PF9_TLPY:-$HOME/src/franken-engine/franken/decode/glm5_timeline.py}; UNION=${PF9_UNION:-$HOME/src/colibri/tools/hot-expert/ckpt1006/tl_union.py}
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --glm-help-copy 1"
say_end() { echo "=== glm_pf9_main exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
[ "$(wc -w < "$REC")" -ge 66000 ] || { echo "FATAL: $REC has fewer than 66 000 ids"; say_end 2; }
{ echo "w0       --tokens-file $PR $SHIP --time-prefill 8192"
  echo "pf       --tokens-file $PR $SHIP --time-prefill 8192 --profile"
  echo "tl_a     --tokens-file $REC $SHIP --time-prefill 32768"
  echo "tl_t     --tokens-file $REC $SHIP --time-prefill 32768 --timeline $O/timeline32.csv --timeline-skip 16 --timeline-chunks 6"
  echo "tl_b     --tokens-file $REC $SHIP --time-prefill 32768"
  echo "pf32     --tokens-file $REC $SHIP --time-prefill 32768 --profile"
  for n in 2048 8192 32768 65536; do echo "rec${n}_a --tokens-file $REC $SHIP --time-prefill $n"; done
  for n in 65536 32768 8192 2048; do echo "rec${n}_b --tokens-file $REC $SHIP --time-prefill $n"; done
} > "$O/plan.txt"
echo "=== glm_pf9_main start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"; cut -c1-70 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|glm5_timeline |help_copy" "$O/gate_run.log" | sort | uniq -c | head -8 | cut -c1-200
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
echo "=== profile (chunks 2.., per prompt token; the non-zero lines)"
awk '/^=== gate-plan config/ { c = $4; on = 0 } /^--- prefill \(per prompt token/ { on = 1; print "[" c "] " $0; next }
  on && /^prof_/ && /=0\.00$/ { next } on && /^(prof_|glm5_side_dev)/ { print "[" c "] " substr($0, 1, 230) } /^adapt_total/ { on = 0 }' "$O/gate_run.log" | head -150
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-160
if [ -s "$O/timeline32.csv" ]; then
  echo "=== timeline union (chunks 16-21 of 32)"
  python3 -I "$UNION" "$O/timeline32.csv" --label "help-copy 1, 32 chunks" 2>&1 | cut -c1-230 | tee "$O/timeline_union.txt"
  python3 -I "$TLPY" "$O/timeline32.csv" --tokens 1024 --layers --gantt 4 > "$O/timeline_report.txt" 2>&1
  echo "glm5_timeline.py rc=$? -> $O/timeline_report.txt"
  sed -n '/^=== steady state/,/^  gantt/p' "$O/timeline_report.txt" | cut -c1-220 | head -70
fi
say_end "$rc"
