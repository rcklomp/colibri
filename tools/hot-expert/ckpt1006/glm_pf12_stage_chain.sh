#!/bin/bash
# glm_pf12_stage_chain.sh -- PF12 (plan Rev 103, PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md section 0c): staged DMA prefill against in place, WITH the ordering fault fixed.
# Why: record §L5-PF9 says the routed experts are 2.2 of 3.6 ms/token (60 %) and link-bound by ~1.4 of it, and that each card runs its trunk work and then its expert phase in ONE in-order stream, so the two add up.
# `--glm-prefill-stage 1` (GLM5.md 12.4) copies layer L+1's missed experts by DMA, off the CUs, while layer L computes -- the mechanism PF3 asks for, already built. It lost to in place by 4 % at 1 024 rows
# (§L5-GLM-STAGECROSS: 5.77 vs 6.01) in the schedule PF0 then found stalled. This chain repeats the comparison with `--glm-help-copy 1` on every config.
# VRAM (CLAUDE.md "Before a build that allocates device memory"): the ring is `--glm-stage-mb` (auto: min(1536, free - 1024) MB) a card; at the served 17 GB of experts the cards have 0.55-0.92 GB free
# (PF9 log, vram_dev*_used_gb 24.83-25.21 of 25.75), so BOTH arms run with `--expert-gb 15` (+2 GB free -> a 1 536 MB ring). The staged smoke is the FIRST staged config and runs at the largest shapes
# (262 144 cells, chunk 1 024) so the ring and the lazily grown GEMV scratch are allocated before any timing arm; the in-place arms then run with the ring present but unused (same VRAM state as the staged ones).
# ONE process, binary franken_decode_glm_pf9main (franken-engine main df401f7, unpatched), in this order:
#   w0 (in place, discarded) | smoke_st (staged, 2 048 tokens, discarded) | prose 8 192 tokens: ip_a st_a st_b ip_b | technical text 8 192: ip_a st_a st_b ip_b | technical text 32 768 (steady state): ip_a st_a st_b ip_b | st_t (staged, timeline chunks 2-7)
# Stop rule (plan): staged < 5 % faster than in place at the same expert size on BOTH prompts -> PF3 takes route B (a second stream per bank).
# Launch (on the rig):  ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf12_stage_chain.sh > ~/bench/glm_pf12_stage_chain.log 2>&1 < /dev/null &
# Env: PF12_BIN, PF12_OUT, PF12_EG (default 15).
# RESULT (2026-10-09, record §L5-PF12): staged was +38 / +59 / +68 % SLOWER than in place (prose 8 k / technical 8 k / technical 32 k) -> stop rule met, PF3 takes route B. A cold start of the engine at --expert-gb 15 took ~12 min (paging); run_chain's quiet-wait does not cover that.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF12_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf9main}
O=${PF12_OUT:-$HOME/bench/franken/glm5/pf12_stage}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=${PF12_EG:-15}; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
UNION=${PF12_UNION:-$HOME/src/colibri/tools/hot-expert/ckpt1006/tl_union.py}; TLPY=${PF12_TLPY:-$HOME/src/franken-engine/franken/decode/glm5_timeline.py}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1"
IP="$BASE --glm-prefill-stage 0"; ST="$BASE --glm-prefill-stage 1"
say_end() { echo "=== glm_pf12_stage exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
{ echo "w0        --tokens-file $PR  $IP --time-prefill 8192"
  echo "smoke_st  --tokens-file $PR  $ST --time-prefill 2048"
  for f in p:$PR:8192 r:$REC:8192 l:$REC:32768; do
    k=${f%%:*}; rest=${f#*:}; file=${rest%%:*}; n=${rest#*:}
    echo "${k}_ip_a --tokens-file $file $IP --time-prefill $n"
    echo "${k}_st_a --tokens-file $file $ST --time-prefill $n"
    echo "${k}_st_b --tokens-file $file $ST --time-prefill $n"
    echo "${k}_ip_b --tokens-file $file $IP --time-prefill $n"
  done
  echo "st_t      --tokens-file $PR  $ST --time-prefill 8192 --timeline $O/timeline_st.csv --timeline-skip 2 --timeline-chunks 6"
} > "$O/plan.txt"
echo "=== glm_pf12_stage start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) expert-gb=$EG"; cut -c1-90 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|glm5_stage dev=" "$O/gate_run.log" | sort | uniq -c | head -8 | cut -c1-200
grep -aE "vram_dev[0-9]_(used|steady)" "$O/gate_run.log" | head -6 | cut -c1-140
[ "$rc" -ne 0 ] && { echo "--- engine log tail:"; tail -n 14 "$O/gate_run.log" | cut -c1-200; }
echo "=== prefill ms/token per config (whole prompt, first chunk included; ip = in place, st = staged DMA; expert-gb $EG)"
LC_ALL=C awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); ms[c] = a[2]; printf "%-10s %8.4f ms/token = %6.1f tok/s\n", c, a[2], 1000 / a[2] }
  END { nok = 0
        split("p r l", K, " "); split("prose8192 technical8192 technical32768", N, " ")
        for (i = 1; i <= 3; i++) { k = K[i]
          if (ms[k "_ip_a"] == "" || ms[k "_ip_b"] == "" || ms[k "_st_a"] == "" || ms[k "_st_b"] == "") { printf "--- %s: an arm is missing\n", N[i]; continue }
          ip = (ms[k "_ip_a"] + ms[k "_ip_b"]) / 2; st = (ms[k "_st_a"] + ms[k "_st_b"]) / 2
          printf "--- %-16s in place %.4f   staged %.4f   staged/in place %.4f (%+.1f %%)\n", N[i], ip, st, st / ip, 100 * (st / ip - 1)
          if (k != "l" && st / ip <= 0.95) nok++ }
        printf "--- stop rule: staged >= 5 %% faster on BOTH 8 192-token prompts? %s\n", (nok == 2 ? "YES -> continue (PF3 route A)" : "NO -> PF3 route B") }' "$O/gate_run.log"
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-160 | grep -E "config|st_t|p_st_a|p_ip_a|l_st_a|l_ip_a"
if [ -s "$O/timeline_st.csv" ]; then
  echo "=== timeline of the staged arm"
  python3 -I "$UNION" "$O/timeline_st.csv" --label "staged, help-copy 1" 2>&1 | cut -c1-230
  python3 -I "$TLPY" "$O/timeline_st.csv" --tokens 1024 --layers > "$O/timeline_report_st.txt" 2>&1; echo "glm5_timeline.py rc=$?"
fi
say_end "$rc"
