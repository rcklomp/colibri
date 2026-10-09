#!/bin/bash
# glm_pf3_probe_chain.sh -- PF3 route B, step 1 (ACTION-LIST row 5d, plan Rev 104 section 0d; builder: build_pf3probe.sh): does the link-bound expert phase of a card run beside its trunk work?
# ONE process, binary franken_decode_glm_pf3probe (franken-engine main df401f7 + the DEBUG-ONLY `--pf3-probe`), the SERVED shape: --expert-gb 17 (the probe allocates two streams and three events, no device
# memory), --ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1, FRANKEN_GLM_MOE_G=8. Each config prefills 8 192 tokens (so the router, the resident set and the KV state are realistic),
# then the probe runs per card a KDA and a DSA MoE layer (the layer's attention + router + plan against the NEXT layer's own expert share, a frozen plan), 14 conditions x forward/reverse.
# Configs: probe_rec (technical text), probe_prose. Lines `pf3 card= il= dsa= cond= run= trunk_ms= expert_ms=` (ms a rep) in gate_run.log; pf3_probe_report.py turns them into the overlap table.
# TIMING ONLY: the probe re-feeds the hidden state to one layer; it is not an oracle and no output is compared.
# Launch (on the rig):  ~/src/colibri/tools/hot-expert/preflight.sh <cmd> && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf3_probe_chain.sh > ~/bench/glm_pf3_probe_chain.log 2>&1 < /dev/null &
# Env: PF3_BIN, PF3_OUT, PF3_EG (default 17), PF3_REPS (default 3).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF3_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf3probe}
O=${PF3_OUT:-$HOME/bench/franken/glm5/pf3_probe}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=${PF3_EG:-17}; REPS=${PF3_REPS:-3}; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
REPORT=${PF3_REPORT:-$HOME/src/colibri/tools/hot-expert/ckpt1006/pf3_probe_report.py}
BASE="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-help-copy 1"
say_end() { echo "=== glm_pf3_probe exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] && [ -f "$REC" ] || { echo "FATAL: missing $BIN or $PR or $REC"; say_end 2; }
{ echo "probe_rec   --tokens-file $REC  $BASE --time-prefill 8192 --pf3-probe $REPS"
  echo "probe_prose --tokens-file $PR   $BASE --time-prefill 8192 --pf3-probe $REPS"
} > "$O/plan.txt"
echo "=== glm_pf3_probe start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) expert-gb=$EG reps=$REPS"; cut -c1-100 "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|pf3_probe (T=|dev=)|stream create failed|probe stopped" "$O/gate_run.log" | sort | uniq -c | head -16 | cut -c1-220
grep -aE "vram_dev[0-9]_(used|steady)" "$O/gate_run.log" | head -6 | cut -c1-140
[ "$rc" -ne 0 ] && { echo "--- engine log tail:"; tail -n 14 "$O/gate_run.log" | cut -c1-200; }
echo "=== pf3 probe report"
python3 -I "$REPORT" "$O/gate_run.log" 2>&1 | cut -c1-200
echo "=== sampler"; python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log" | cut -c1-160 | grep -E "config|probe" | head -6
say_end "$rc"
