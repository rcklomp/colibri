#!/bin/bash
# glm_q8f_chain.sh -- `--gemv-q8fast 1|2` (the specialised T=1 unsplit all-Q8_0 plain-x decode GEMV k_gemv_q8, franken-engine branch d3-waves, GLM5.md section 22): EXACTNESS ONLY, no timing (the paired kernel trace measures the speed).
# k_gemv_q8 is a copy of k_gemm_batch's Q8_0 loop (same loads, same order of operations; bench_group crosses mode 1/2 with group, lds, rowsplit against main's solo gemv), this is the model gate:
# ONE process, one load, shipped flags (`--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1`) unless an arm says otherwise:
#   chunk 1 + 8 greedy tokens of the 136-token prompt against the pre-change reference (gpu5/g136_eager), q8fast 0 / 1 / 2;
#   64 greedy tokens at depth 1 500 (< 2 051) against IN-PROCESS q8fast-0 dumps, for q8fast 1 and 2, with group 0, with a swap every token (adapt-verify), with --gemv-lds 0,
#   the same at --ctx 2048 (the indexer does not score) against its own dump; a staged chunk-512 line LAST (--glm-prefill-stage 1 allocates the staging ring once and keeps it).
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_q8f_chain.sh > ~/bench/glm_q8f_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_q8f_chain.log ~/bench/franken/glm5/q8f/gate_run.log
# Env: QF_BIN (default ~/bench/franken_bin/franken_decode_glm_q8f), QF_OUT (default ~/bench/franken/glm5/q8f), QF_VERDICT (default the checkout's glm5_gate_verdict.sh).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${QF_BIN:-$HOME/bench/franken_bin/franken_decode_glm_q8f}
VERDICT=${QF_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${QF_OUT:-$HOME/bench/franken/glm5/q8f}
if [ -d "$O" ]; then docker run --rm -v "$(dirname "$O")":/w rocm/dev-ubuntu-24.04:7.14.0-full rm -rf "/w/$(basename "$O")" >/dev/null 2>&1; fi; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
DNS="--tokens-file $O/prose1500.txt --ctx 2048 --chunk 512 --greedy 64"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
NEW="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1 --gemv-rowsplit-waves 1"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg qf_c1q0    exact $C136 --chunk 1 $NEW --gemv-q8fast 0 --oracle $B/gpu5/g136_eager   # the new build with the knob off, against the pre-change engine
cfg qf_c1q1    exact $C136 --chunk 1 $NEW --gemv-q8fast 1 --oracle $B/gpu5/g136_eager
cfg qf_c1q2    exact $C136 --chunk 1 $NEW --gemv-q8fast 2 --oracle $B/gpu5/g136_eager
cfg qf_ref     ref   $D1500 $NEW --gemv-q8fast 0 --dump $O/s64_ref
cfg qf_x1      exact $D1500 $NEW --gemv-q8fast 1 --oracle $O/s64_ref
cfg qf_x2      exact $D1500 $NEW --gemv-q8fast 2 --oracle $O/s64_ref
cfg qf_x2_g0   exact $D1500 --fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 0 --gemv-q8fast 2 --oracle $O/s64_ref
cfg qf_x1_ad   exact $D1500 $NEW --gemv-q8fast 1 $AD --oracle $O/s64_ref
cfg qf_x1_lds0 exact $D1500 $NEW --gemv-q8fast 1 --gemv-lds 0 --oracle $O/s64_ref
cfg qf_ref_ns  ref   $DNS $NEW --gemv-q8fast 0 --dump $O/s64_ns_ref
cfg qf_x_ns    exact $DNS $NEW --gemv-q8fast 2 --oracle $O/s64_ns_ref
# LAST: --glm-prefill-stage 1 allocates the staging ring once and keeps it for the process; the configs above must not carry it
cfg qf_c512s   exact $C136 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 $NEW --gemv-q8fast 1 --oracle $B/gpu5/g136_eager
echo "=== glm_q8f start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_q8f exit rc=2 $(date -Is)"; exit 2; }
[ -d "$B/gpu5/g136_eager" ] && [ -f "$B/prose2200.txt" ] && [ -f "$B/t136.txt" ] || { echo "FATAL: references or prompts missing"; echo "=== glm_q8f exit rc=2 $(date -Is)"; exit 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; echo "=== glm_q8f exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "HIP error|Memory access|out of memory" "$O/gate_run.log" | head -2 | cut -c1-170
echo "=== oracle_verdict (taps with maxabs != 0 per config; must be 0 everywhere)"
$HOME/bench/oracle_verdict.sh "$O/gate_run.log" 2>&1 | cut -c1-200 | tee "$O/oracle_verdict.txt"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
echo "=== the q8fast flag as each configuration ran it (expected: 0 in qf_c1q0 / qf_ref / qf_ref_ns, 1 in qf_c1q1 / qf_x1* / qf_x1_lds0 / qf_c512s, 2 in qf_c1q2 / qf_x2 / qf_x2_g0 / qf_x_ns)"
awk '/^=== gate-plan config/ {c=$4} /^gemv_fused_reduce=/ {n=$0; sub(/.*gemv_q8fast=/, "", n); printf "  %-12s gemv_q8fast=%s\n", c, n}' "$O/gate_run.log"
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-120
echo "=== glm_q8f exit rc=$rc $(date -Is)"
exit $rc
