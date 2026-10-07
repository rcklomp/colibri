#!/bin/bash
# glm_group_chain.sh -- `--glm-gemv-group 1` (franken-engine branch gemv-group, GLM5.md section 21): a decode token's trunk GEMVs that read the same activation as ONE launch each
# (KDA {wq,wk,wv} and {f_a,beta,g_a}; the indexer's {idx_k,idx_gate[,idx_proj]}), every member at the K-split count it has alone -- EXACTNESS ONLY, no timing (the kernel trace and the
# serve-path A/B measure the speed). ONE process, one load, shipped flags (`--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1`) unless an arm says otherwise:
#   chunk 1 + 8 greedy tokens of the 136-token prompt against the pre-change reference (gpu5/g136_eager), group 1 (and group 0: the refactored graph / backend alone);
#   64 greedy tokens at depth 1 500 (< 2 051: decode past it is not reproducible run to run) against IN-PROCESS flag-off dumps (--glm-gemv-group 0), for group 1 with rowsplit 1
#   (shipped), with rowsplit 0 (against a rowsplit-0 flag-off dump), with a swap every token (adapt-verify), with --gemv-lds 0 (an unsplit G1 unstaged);
#   the same at --ctx 2048 (the indexer does not score: G3 is the two-member group, idx_proj stays alone) against its own flag-off dump;
#   a staged chunk-512 line LAST (--glm-prefill-stage 1 allocates the staging ring once and keeps it for the process).
# Run glm_group_micro_chain.sh first (seconds, one card): it must print 0 MISMATCH lines.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_group_chain.sh > ~/bench/glm_group_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_group_chain.log ~/bench/franken/glm5/group/gate_run.log
# Env: GR_BIN (default ~/bench/franken_bin/franken_decode_glm.group, `make -C franken/decode gpu GPU_BIN=franken_decode_glm_group`), GR_OUT (default ~/bench/franken/glm5/group),
# GR_VERDICT (default the checkout's glm5_gate_verdict.sh).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${GR_BIN:-$HOME/bench/franken_bin/franken_decode_glm.group}
VERDICT=${GR_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${GR_OUT:-$HOME/bench/franken/glm5/group}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
DNS="--tokens-file $O/prose1500.txt --ctx 2048 --chunk 512 --greedy 64"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
NEW="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1"
NEW0="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 0"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg gr_c1off   exact $C136 --chunk 1 $NEW --glm-gemv-group 0 --oracle $B/gpu5/g136_eager   # the refactored graph / backend alone (flag off) against the pre-change engine
cfg gr_c1      exact $C136 --chunk 1 $NEW --glm-gemv-group 1 --oracle $B/gpu5/g136_eager
cfg gr_ref     ref   $D1500 $NEW --glm-gemv-group 0 --dump $O/s64_ref
cfg gr_ref_rs0 ref   $D1500 $NEW0 --glm-gemv-group 0 --dump $O/s64_ref_rs0
cfg gr_x       exact $D1500 $NEW --glm-gemv-group 1 --oracle $O/s64_ref
cfg gr_x_rs0   exact $D1500 $NEW0 --glm-gemv-group 1 --oracle $O/s64_ref_rs0
cfg gr_x_ad    exact $D1500 $NEW --glm-gemv-group 1 $AD --oracle $O/s64_ref
cfg gr_x_lds0  exact $D1500 $NEW --glm-gemv-group 1 --gemv-lds 0 --oracle $O/s64_ref
cfg gr_ref_ns  ref   $DNS $NEW --glm-gemv-group 0 --dump $O/s64_ns_ref
cfg gr_x_ns    exact $DNS $NEW --glm-gemv-group 1 --oracle $O/s64_ns_ref
# LAST: --glm-prefill-stage 1 allocates the staging ring once and keeps it for the process; the configs above must not carry it
cfg gr_c512s   exact $C136 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 $NEW --glm-gemv-group 1 --oracle $B/gpu5/g136_eager
echo "=== glm_group start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_group exit rc=2 $(date -Is)"; exit 2; }
[ -d "$B/gpu5/g136_eager" ] && [ -f "$B/prose2200.txt" ] && [ -f "$B/t136.txt" ] || { echo "FATAL: references or prompts missing"; echo "=== glm_group exit rc=2 $(date -Is)"; exit 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; echo "=== glm_group exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "HIP error|Memory access|out of memory" "$O/gate_run.log" | head -2 | cut -c1-170
echo "=== oracle_verdict (taps with maxabs != 0 per config; must be 0 everywhere)"
$HOME/bench/oracle_verdict.sh "$O/gate_run.log" 2>&1 | cut -c1-200 | tee "$O/oracle_verdict.txt"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
echo "=== the group flag as each configuration ran it (gemv_group=1 expected in every gr_x*, gr_c1 and gr_c512s config; the echo line follows the config header)"
awk '/^=== gate-plan config/ {c=$4} /^gemv_fused_reduce=/ {n=$0; sub(/.*gemv_group=/, "", n); printf "  %-12s gemv_group=%s\n", c, n}' "$O/gate_run.log"
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-120
echo "=== glm_group exit rc=$rc $(date -Is)"
exit $rc
