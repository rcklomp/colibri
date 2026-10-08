#!/bin/bash
# glm_pf0_skip_chain.sh -- PF0 (a) the skip-class attribution and (d) the in-place routing dump of the SHIPPED prefill (plan Rev 102, PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md section 3).
# Binary: franken_decode_glm_pf0dbg = franken-engine main (code a7bf81a) + build_pf0dbg.sh (--debug-skip MASK, TIMING ONLY: garbage output; --debug-route FILE, analysis only: a stream sync per layer).
# Config = the served one (see glm_pf0_main_chain.sh): chunk 1 024, in place, --gemm-lds 1, FRANKEN_GLM_MOE_G=8, 262 144 cells, 8 192 prose tokens, swaps held, generic M2 placement.
# MASK bits: 1 routed-expert kernels, 2 trunk GEMMs/GEMVs, 4 KDA, 8 MLA attention, 16 indexer (implies 8). The skeleton (31) is every launch that is not one of those: norms, hyper-connection, router,
# plan, boundary copies, embedding, the waits. In place the expert kernels READ the experts over the links, so mask bit 1 removes compute and link traffic together (the staged DMA of the
# old attribution, §M7-SKIPCLASS, was separate and always ran).
#   masks:  0 everything | 31 skeleton | 29 skeleton + trunk GEMMs | 30 skeleton + experts | 27 skeleton + KDA | 15 skeleton + indexer | 7 skeleton + indexer + attention |
#           1 everything but the experts | 3 everything but experts and trunk GEMMs (= KDA + indexer + attention)
#   Solo cost of a class = mask(skeleton + class) - mask 31; marginal cost = mask 0 - mask(everything but the class). Classes are sub-additive (§M7-SKIPCLASS), so both are reported.
# Order: w0 (discarded) then the palindrome over the nine masks A..I,I..A (each 8 192 tokens), then three --debug-route configs (the analysis input of PF0 (d) and PF5).
# Launch (on the rig):  ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf0_skip_chain.sh > ~/bench/glm_pf0_skip_chain.log 2>&1 < /dev/null &
# Watch from the Mac:   tools/hot-expert/ckpt1006/watch_chain.sh ~/bench/glm_pf0_skip_chain.log ~/bench/franken/glm5/pf0_skip/gate_run.log
# Env: PF0_BIN, PF0_OUT, PF0_PROMPT (token-ID file of the timing arms, default prose8400), PF0_MASKS (default "0 31 29 30 27 15 7 1 3"), PF0_ROUTES (0 = no --debug-route configs).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF0_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf0dbg}
O=${PF0_OUT:-$HOME/bench/franken/glm5/pf0_skip}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=${PF0_PROMPT:-$B/prose8400.txt}; REC=$B/rec_depth/rec_ids.txt
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0"
MASKS=${PF0_MASKS:-0 31 29 30 27 15 7 1 3}
say_end() { echo "=== glm_pf0_skip exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
{ echo "w0 --tokens-file $PR $SHIP --time-prefill 8192"
  for m in $MASKS; do echo "m${m}_a --tokens-file $PR $SHIP --time-prefill 8192 --debug-skip $m"; done
  for m in $(echo $MASKS | tr ' ' '\n' | tac | tr '\n' ' '); do echo "m${m}_b --tokens-file $PR $SHIP --time-prefill 8192 --debug-skip $m"; done
  if [ "${PF0_ROUTES:-1}" = 1 ]; then
  echo "route_prose   --tokens-file $PR  $SHIP --time-prefill 8192  --debug-route $O/route_prose"
  echo "route_rec8k   --tokens-file $REC $SHIP --time-prefill 8192  --debug-route $O/route_rec8k"
  echo "route_rec32k  --tokens-file $REC $SHIP --time-prefill 32768 --debug-route $O/route_rec32k"
  fi
} > "$O/plan.txt"
echo "=== glm_pf0_skip start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"; cut -c1-70 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|debug_skip=|debug_route=" "$O/gate_run.log" | sort | uniq -c | head -16 | cut -c1-200
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-140
[ "$rc" -ne 0 ] && { echo "--- engine log tail:"; tail -n 12 "$O/gate_run.log" | cut -c1-200; }
echo "=== prefill ms/token per mask (8 192 tokens, whole prompt): a = first pass, b = reverse pass"
awk -v maskl="$MASKS" '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, v, "="); ms[c] = v[2]; printf "%-14s %9.4f ms/token\n", c, v[2] }
  END { nk = split(maskl, K, " ")
        for (i = 1; i <= nk; i++) { k = K[i]; if (ms["m" k "_a"] == "" || ms["m" k "_b"] == "") continue; mm[k] = (ms["m" k "_a"] + ms["m" k "_b"]) / 2
          printf "mask %-2s mean %8.4f   a-b spread %7.4f\n", k, mm[k], ms["m" k "_a"] - ms["m" k "_b"] }
        if (mm[0] != "" && mm[31] != "") {
          printf "--- skeleton (31) %.3f of %.3f ms/token = %.0f %%\n", mm[31], mm[0], 100 * mm[31] / mm[0]
          if (mm[29] != "") printf "--- solo trunk GEMMs   (29 - 31) %.3f\n", mm[29] - mm[31]
          if (mm[30] != "") printf "--- solo experts       (30 - 31) %.3f\n", mm[30] - mm[31]
          if (mm[27] != "") printf "--- solo KDA           (27 - 31) %.3f\n", mm[27] - mm[31]
          if (mm[15] != "") printf "--- solo indexer       (15 - 31) %.3f\n", mm[15] - mm[31]
          if (mm[7]  != "") printf "--- solo indexer+attn  ( 7 - 31) %.3f   attention alone (7 - 15) %.3f\n", mm[7] - mm[31], (mm[15] != "" ? mm[7] - mm[15] : 0)
          if (mm[1]  != "") printf "--- marginal experts   ( 0 -  1) %.3f   (everything else = mask 1 = %.3f)\n", mm[0] - mm[1], mm[1]
          if (mm[3]  != "" && mm[1] != "") printf "--- marginal trunk GEMMs ( 1 -  3) %.3f   (KDA + indexer + attention = mask 3 = %.3f; mask 3 minus skeleton = %.3f)\n", mm[1] - mm[3], mm[3], mm[3] - mm[31]
        } }' "$O/gate_run.log"
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-160
echo "=== route files"; ls -la "$O"/route_* 2>&1 | cut -c1-120
say_end "$rc"
