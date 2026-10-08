#!/bin/bash
# glm_pf0_hc_chain.sh -- PF0 follow-up (plan Rev 102): --glm-help-copy 1 (franken-engine branch pf-helpcopy, binary franken_decode_glm_pfhc): a chunk's helper packet (the plan ids and the x rows) copied by a
# kernel on the helper's stream instead of two hipMemcpyPeerAsync calls. The in-place timeline of the shipped prefill (glm_pf0_main_chain.sh) shows the first MoE layer's packet of every chunk completing only
# after the PREVIOUS chunk's last-layer packet (3-11 ms later), 3.6 s / 1.6 s after it was issued: the runtime completes peer copies into a card in issue order, whatever the stream. HSA_ENABLE_PEER_SDMA=0 changed
# nothing (glm_pf0_sdma_chain.sh: period 5.629 s against 5.617 s), so the order is not the SDMA engine's. This chain asks: does the kernel copy let the chunks overlap?
# ONE process, shipped config (chunk 1 024, in place, G=8, --gemm-lds 1, 262 144 cells), prose8400 first 8 192 tokens, then the record text (technical document):
#   w0 discarded; hc0_a, hc1_a, hc1_t (--timeline chunks 2-7), hc1_b, hc0_b   (0 = hipMemcpyPeerAsync as shipped, 1 = the kernel copy);   r0_a, r1_a, r1_b, r0_b on the record text.
# Then tl_union.py on the timeline. Exactness is NOT judged here (the data copied is identical; the model gate follows if the speed is real).
# Launch through run_chain.sh.   Env: PF0_BIN, PF0_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF0_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pfhc}
O=${PF0_OUT:-$HOME/bench/franken/glm5/pf0_hc}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
UNION=${PF0_UNION:-$HOME/src/colibri/tools/hot-expert/ckpt1006/tl_union.py}
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --time-prefill 8192"
say_end() { echo "=== glm_pf0_hc exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
{ echo "w0     --tokens-file $PR $SHIP --glm-help-copy 0"
  echo "hc0_a  --tokens-file $PR $SHIP --glm-help-copy 0"
  echo "hc1_a  --tokens-file $PR $SHIP --glm-help-copy 1"
  echo "hc1_t  --tokens-file $PR $SHIP --glm-help-copy 1 --timeline $O/timeline.csv --timeline-skip 2 --timeline-chunks 6"
  echo "hc1_b  --tokens-file $PR $SHIP --glm-help-copy 1"
  echo "hc0_b  --tokens-file $PR $SHIP --glm-help-copy 0"
  echo "r0_a   --tokens-file $REC $SHIP --glm-help-copy 0"
  echo "r1_a   --tokens-file $REC $SHIP --glm-help-copy 1"
  echo "r1_b   --tokens-file $REC $SHIP --glm-help-copy 1"
  echo "r0_b   --tokens-file $REC $SHIP --glm-help-copy 0"; } > "$O/plan.txt"
echo "=== glm_pf0_hc start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
rig_quiet_wait 1800 || say_end 3
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && tail -n 10 "$O/gate_run.log" | cut -c1-200
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory" "$O/gate_run.log" | head -3 | cut -c1-200
echo "=== prefill ms/token (help-copy 0 = hipMemcpyPeerAsync, 1 = kernel copy)"
LC_ALL=C awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); ms[c] = a[2]; printf "%-6s %8.4f ms/token = %6.1f tok/s\n", c, a[2], 1000 / a[2] }
  END { if (ms["hc0_a"] != "" && ms["hc0_b"] != "" && ms["hc1_a"] != "" && ms["hc1_b"] != "") { m0 = (ms["hc0_a"] + ms["hc0_b"]) / 2; m1 = (ms["hc1_a"] + ms["hc1_b"]) / 2
          printf "--- prose8400: help-copy 0 %.4f  1 %.4f  ratio %.4f (%.1f %%)\n", m0, m1, m1 / m0, 100 * (m1 / m0 - 1) }
        if (ms["r0_a"] != "" && ms["r0_b"] != "" && ms["r1_a"] != "" && ms["r1_b"] != "") { m0 = (ms["r0_a"] + ms["r0_b"]) / 2; m1 = (ms["r1_a"] + ms["r1_b"]) / 2
          printf "--- record text: help-copy 0 %.4f  1 %.4f  ratio %.4f (%.1f %%)\n", m0, m1, m1 / m0, 100 * (m1 / m0 - 1) } }' "$O/gate_run.log"
[ -s "$O/timeline.csv" ] && python3 -I "$UNION" "$O/timeline.csv" --label "help-copy 1" 2>&1 | cut -c1-230
say_end "$rc"
