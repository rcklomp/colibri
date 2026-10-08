#!/bin/bash
# glm_pf0_sktl_chain.sh -- PF0 follow-up (plan Rev 102): the skip-class run (glm_pf0_skip_chain.sh) found the SKELETON (--debug-skip 31: every launch that is not an expert kernel, trunk GEMM,
# KDA, attention or indexer) at 2.90-2.98 ms/token = 52 % of the 5.53 ms/token, the routed-expert kernels worth ~0 on the critical path (mask 1 = 5.62, mask 30 = 2.93-2.99), the trunk GEMMs
# +2.0, attention +0.46, KDA / indexer ~0. A skeleton that costs 2.97 s a chunk with nothing computing is a latency / serialisation cost, not work. This chain takes the DEVICE TIMELINE of
# the skeleton and of the two neighbours that separate it: one process, binary franken_decode_glm_pf0dbg (timing only, garbage output), shipped config, 8 192 prose tokens:
#   w0     discarded warm-up (mask 0)
#   s31    mask 31 (skeleton) with --timeline chunks 2-7
#   s29    mask 29 (skeleton + trunk GEMMs) with --timeline
#   s1     mask 1  (everything but the experts) with --timeline
#   s0     mask 0  (everything) with --timeline (the control: same binary as the three above)
# then tl_union.py on each. What to read: the per-card main-stream busy share, the period against the chain latency, the longest help_packet and what it ended behind, the host (h_embed block).
# Launch through run_chain.sh.  Env: PF0_BIN, PF0_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF0_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf0dbg}
O=${PF0_OUT:-$HOME/bench/franken/glm5/pf0_sktl}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
UNION=${PF0_UNION:-$HOME/src/colibri/tools/hot-expert/ckpt1006/tl_union.py}; TLPY=$HOME/src/franken-engine/franken/decode/glm5_timeline.py
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --tokens-file $PR --time-prefill 8192"
say_end() { echo "=== glm_pf0_sktl exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] || { echo "FATAL: missing $BIN or $PR"; say_end 2; }
{ echo "w0  $SHIP"
  for m in 31 29 1 0; do echo "s$m $SHIP --debug-skip $m --timeline $O/timeline_$m.csv --timeline-skip 2 --timeline-chunks 6"; done; } > "$O/plan.txt"
echo "=== glm_pf0_sktl start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
rig_quiet_wait 1800 || say_end 3
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?; echo "engine rc=$rc"; [ "$rc" -ne 0 ] && tail -n 10 "$O/gate_run.log" | cut -c1-200
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-5s %8.4f ms/token\n", c, a[2] }' "$O/gate_run.log"
for m in 31 29 1 0; do
  [ -s "$O/timeline_$m.csv" ] || continue
  python3 -I "$UNION" "$O/timeline_$m.csv" --label "mask $m" 2>&1 | cut -c1-230
  python3 -I "$TLPY" "$O/timeline_$m.csv" --tokens 1024 --layers > "$O/report_$m.txt" 2>&1
  grep -a "host: step()\|host calls" "$O/report_$m.txt" | tail -4 | cut -c1-260
done
say_end "$rc"
