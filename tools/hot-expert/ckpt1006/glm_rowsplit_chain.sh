#!/bin/bash
# glm_rowsplit_chain.sh -- `--gemv-rowsplit 1` (franken-engine branch gemv-rowsplit, GLM5.md section 20): a decode token's 278 K-split trunk GEMVs as ONE launch each,
# one workgroup a row, the waves looping over the row's splits with k_gemm_batch's own per-split code and the partials summed in LDS in s order -- is it BIT-EXACT and is
# it faster? (2026-10-07; record §L5-GLM-TRUNK-GEMV: the split GEMVs are workgroup-count bound, ~19 us each with the reduce launch and two gaps, 5.4 ms a token; expected
# -2.0 to -3.5 ms.) ONE process, one load, shipped flags (`--fetch-assign optimal --gemv-wave-reduce 1`):
#   exactness against the pre-change reference (chunk 1 + 8 greedy tokens, gpu5/g136_eager) with rowsplit 1; 64 greedy tokens at depth 1 500 (< 2 051: decode past it is not
#   reproducible run to run) against an in-process dump made with rowsplit 0 (= main), for rowsplit 1 at waves auto / 16 / 4, rowsplit 1 with a swap every token, and
#   `--gemv-lds 0|1` (newly plumbed for GLM; the backend default 1 stays the default) with rowsplit 0 and 1; a staged chunk-512 line LAST;
#   decode timing A,B,C,D,D,C,B,A -- A rowsplit 0 (= main) / B rowsplit 1 (waves auto) / C rowsplit 1 waves 16 / D rowsplit 1 with --gemv-lds 0.
# Run glm_rowsplit_micro_chain.sh first (seconds, one card): it must print 0 MISMATCH lines.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_rowsplit_chain.sh > ~/bench/glm_rowsplit_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_rowsplit_chain.log ~/bench/franken/glm5/rowsplit/gate_run.log
# Env: RS_BIN (default the worktree's franken_decode_glm_rowsplit, `make -C franken/decode gpu GPU_BIN=franken_decode_glm_rowsplit`), RS_OUT (default ~/bench/franken/glm5/rowsplit).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${RS_BIN:-$HOME/src/franken-engine-rowsplit/franken/decode/franken_decode_glm_rowsplit}
VERDICT=${RS_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${RS_OUT:-$HOME/bench/franken/glm5/rowsplit}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --time 32 --time-settle 64 --adapt-prefill 0"
NEW="--fetch-assign optimal --gemv-wave-reduce 1"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg rs_c1      exact $C136 --chunk 1 $NEW --gemv-rowsplit 1 --oracle $B/gpu5/g136_eager
cfg rs_c1off   exact $C136 --chunk 1 $NEW --gemv-rowsplit 0 --oracle $B/gpu5/g136_eager   # the refactored k_gemm_batch alone (flag off) against the pre-change engine
cfg rs_ref     ref   $D1500 $NEW --gemv-rowsplit 0 --dump $O/s64_ref
cfg rs_x_auto  exact $D1500 $NEW --gemv-rowsplit 1 --oracle $O/s64_ref
cfg rs_x_w16   exact $D1500 $NEW --gemv-rowsplit 1 --gemv-rowsplit-waves 16 --oracle $O/s64_ref
cfg rs_x_w4    exact $D1500 $NEW --gemv-rowsplit 1 --gemv-rowsplit-waves 4 --oracle $O/s64_ref
cfg rs_x_ad    exact $D1500 $NEW --gemv-rowsplit 1 $AD --oracle $O/s64_ref
cfg rs_x_lds0  exact $D1500 $NEW --gemv-rowsplit 0 --gemv-lds 0 --oracle $O/s64_ref
cfg rs_x_rlds0 exact $D1500 $NEW --gemv-rowsplit 1 --gemv-lds 0 --oracle $O/s64_ref
cfg rs_x_rlds1 exact $D1500 $NEW --gemv-rowsplit 1 --gemv-lds 1 --oracle $O/s64_ref
cfg rs_t_a1    none  $TP $NEW --gemv-rowsplit 0
cfg rs_t_b1    none  $TP $NEW --gemv-rowsplit 1
cfg rs_t_c1    none  $TP $NEW --gemv-rowsplit 1 --gemv-rowsplit-waves 16
cfg rs_t_d1    none  $TP $NEW --gemv-rowsplit 1 --gemv-lds 0
cfg rs_t_d2    none  $TP $NEW --gemv-rowsplit 1 --gemv-lds 0
cfg rs_t_c2    none  $TP $NEW --gemv-rowsplit 1 --gemv-rowsplit-waves 16
cfg rs_t_b2    none  $TP $NEW --gemv-rowsplit 1
cfg rs_t_a2    none  $TP $NEW --gemv-rowsplit 0
# LAST: --glm-prefill-stage 1 allocates the staging ring once and keeps it for the process; the 262 144-cell, chunk-1024 timing arms above must not carry it
# (it OOMed the first timing config on 2026-10-07 when this line sat second in glm_stack_chain.sh)
cfg rs_c512s   exact $C136 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 $NEW --gemv-rowsplit 1 --oracle $B/gpu5/g136_eager
echo "=== glm_rowsplit start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_rowsplit exit rc=2 $(date -Is)"; exit 2; }
[ -d "$B/gpu5/g136_eager" ] && [ -f "$PR" ] && [ -f "$B/t136.txt" ] || { echo "FATAL: references or prompts missing"; echo "=== glm_rowsplit exit rc=2 $(date -Is)"; exit 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; echo "=== glm_rowsplit exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "HIP error|Memory access|out of memory" "$O/gate_run.log" | head -2 | cut -c1-170
echo "=== oracle_verdict (taps with maxabs != 0 per config; must be 0 everywhere)"
$HOME/bench/oracle_verdict.sh "$O/gate_run.log" 2>&1 | cut -c1-200 | tee "$O/oracle_verdict.txt"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-120
echo "=== decode ms/token (median of 32 at depth ~8 290): A rowsplit 0 / B rowsplit 1 auto / C rowsplit 1 waves 16 / D rowsplit 1 gemv-lds 0 -- A,B,C,D,D,C,B,A"
awk '/^=== gate-plan config/ {c=$4} /^glm5_decode_ms_median=/ && c ~ /^rs_t_/ {split($1,a,"="); printf "  %-8s %s\n", c, a[2]; d[c]=a[2]}
     END { n=0; split("a1 b1 c1 d1 d2 c2 b2 a2", k, " "); for (i=1;i<=8;i++) if (("rs_t_" k[i]) in d) n++
           if (n==8) { A=(d["rs_t_a1"]+d["rs_t_a2"])/2; Bm=(d["rs_t_b1"]+d["rs_t_b2"])/2; Cm=(d["rs_t_c1"]+d["rs_t_c2"])/2; Dm=(d["rs_t_d1"]+d["rs_t_d2"])/2
             printf "  means: A %.3f  B %.3f (B/A %.3f, %+.2f ms)  C %.3f (C/A %.3f)  D %.3f (D/A %.3f)\n", A, Bm, Bm/A, Bm-A, Cm, Cm/A, Dm, Dm/A } else print "  incomplete: a timing config did not finish" }' "$O/gate_run.log"
echo "=== glm_rowsplit exit rc=$rc $(date -Is)"
exit $rc
