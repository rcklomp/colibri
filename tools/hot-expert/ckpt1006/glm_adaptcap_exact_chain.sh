#!/bin/bash
# glm_adaptcap_exact_chain.sh -- is the placement policy REALLY placement-only? --adapt-prefill-cap (franken-engine branch adapt-cap) with heavy swapping, against the default policy's output. (2026-10-07)
# A placement change must move no bit (glm5_adapt.h: the owner sums the eight rows in slot order, whatever card runs a slot). This gate makes the swaps as violent as it can and compares:
#   ac_ref      the reference: 1 500 prose tokens (chunk 512) + 64 greedy decode tokens (< 2 051 deep: decode past it is not run-to-run reproducible), default adapt policy, --dump
#   ac_x_cap1   cap 64, half-life 32, a swap round EVERY token with a 512 MB/token budget, adapt-verify, --oracle ac_ref: every tap must read maxabs=0
#   ac_x_cap2   cap 256, half-life 128, every 4 tokens, adapt-verify, --oracle ac_ref
#   ac_x_nocap  cap 0 (the old behaviour), half-life 32, every token, 512 MB/token, adapt-verify, --oracle ac_ref  (the control: the swaps alone, no cap)
#   ac_c1       chunk 1 + 8 greedy tokens of the 136-token prompt with the cap policy on, against the saved gpu5/g136_eager reference (the pre-adaptation-change engine's output)
# ONE process, one load, the shipped flags where they matter (in place, adaptation held in prefill). The binary is the cap build (franken_decode_glm_adcap).
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> $AC_OUT/gate_run.log:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_adaptcap_exact_chain.sh > ~/bench/glm_adaptcap_exact_chain.log 2>&1 < /dev/null &
# Env: AC_BIN (default the adapt-cap worktree's franken_decode_glm_adcap), AC_OUT (default ~/bench/franken/glm5/adaptcap_exact).
set -u
BIN=${AC_BIN:-$HOME/src/franken-engine-adcap/franken/decode/franken_decode_glm_adcap}
VERDICT=${AC_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${AC_OUT:-$HOME/bench/franken/glm5/adaptcap_exact}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64 --adapt-prefill 0"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8 --chunk 1 --adapt-prefill 0"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg ac_ref     ref   $D1500 --dump $O/s64_ref
cfg ac_x_cap1  exact $D1500 --adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-halflife 32 --adapt-prefill-cap 64 --adapt-verify 1 --oracle $O/s64_ref
cfg ac_x_cap2  exact $D1500 --adapt 1 --adapt-every 4 --adapt-mb-per-token 64 --adapt-halflife 128 --adapt-prefill-cap 256 --adapt-verify 1 --oracle $O/s64_ref
cfg ac_x_nocap exact $D1500 --adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-halflife 32 --adapt-prefill-cap 0 --adapt-verify 1 --oracle $O/s64_ref
cfg ac_c1      exact $C136 --adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-halflife 32 --adapt-prefill-cap 64 --adapt-verify 1 --oracle $B/gpu5/g136_eager
. "$HOME/bench/chain_preflight.sh"
echo "=== glm_adaptcap_exact start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_adaptcap_exact exit rc=2 $(date -Is)"; exit 2; }
[ -d "$B/gpu5/g136_eager" ] || { echo "FATAL: reference gpu5/g136_eager missing"; echo "=== glm_adaptcap_exact exit rc=2 $(date -Is)"; exit 2; }
rig_quiet_wait 1800 || { echo "=== glm_adaptcap_exact exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
$HOME/bench/oracle_verdict.sh "$O/gate_run.log" | tee -a "$O/verdict.txt"
grep -aE "^adapt_total|^adapt_verify" "$O/gate_run.log" | tail -9 | cut -c1-170
echo "=== glm_adaptcap_exact exit rc=$rc $(date -Is)"
exit $rc
