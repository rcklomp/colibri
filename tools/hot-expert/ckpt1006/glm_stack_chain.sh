#!/bin/bash
# glm_stack_chain.sh -- the decode levers stacked on franken-engine main 5a4fa03 (separate split-GEMV reduce): are `--fetch-assign optimal` (the best split of a
# layer's missed slabs over the three cards by miss count) and `--gemv-wave-reduce 1` (the separate reduce one wave a row, lanes load the partials in parallel,
# lane 0 adds them in s order) each BIT-EXACT and each faster? (2026-10-07; record §L5-GLM-GEMV-REDUCE, §L5-GLM-DECODE-TRACE.) ONE process, one load, shipped flags:
#   exactness against the pre-change reference (chunk 1 + 8 greedy tokens, gpu5/g136_eager) with BOTH on, a staged-prefill line with fetch-assign, and 64 greedy tokens at depth 1 500
#   (< 2 051: decode past it is not reproducible run to run) against an in-process dump made with BOTH OFF (= main), for each knob alone, both, and both with a swap every token;
#   decode timing A,B,C,D,D,C,B,A -- A main (pattern, wave 0) / B wave-reduce / C fetch-assign / D both.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_stack_chain.sh > ~/bench/glm_stack_chain.log 2>&1 < /dev/null &
# Env: ST_BIN (default the wave-reduce worktree's franken_decode_glm_wr), ST_OUT (default ~/bench/franken/glm5/stack).
set -u
BIN=${ST_BIN:-$HOME/src/franken-engine-wr/franken/decode/franken_decode_glm_wr}
VERDICT=${ST_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${ST_OUT:-$HOME/bench/franken/glm5/stack}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --time 32 --time-settle 64 --adapt-prefill 0"
OLD="--fetch-assign pattern --gemv-wave-reduce 0"; NEW="--fetch-assign optimal --gemv-wave-reduce 1"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg st_c1      exact $C136 --chunk 1 $NEW --oracle $B/gpu5/g136_eager
cfg st_ref     ref   $D1500 $OLD --dump $O/s64_ref
cfg st_x_wave  exact $D1500 --fetch-assign pattern --gemv-wave-reduce 1 --oracle $O/s64_ref
cfg st_x_fa    exact $D1500 --fetch-assign optimal --gemv-wave-reduce 0 --oracle $O/s64_ref
cfg st_x_both  exact $D1500 $NEW --oracle $O/s64_ref
cfg st_x_bothad exact $D1500 $NEW $AD --oracle $O/s64_ref
cfg st_t_a1    none  $TP $OLD
cfg st_t_b1    none  $TP --fetch-assign pattern --gemv-wave-reduce 1
cfg st_t_c1    none  $TP --fetch-assign optimal --gemv-wave-reduce 0
cfg st_t_d1    none  $TP $NEW
cfg st_t_d2    none  $TP $NEW
cfg st_t_c2    none  $TP --fetch-assign optimal --gemv-wave-reduce 0
cfg st_t_b2    none  $TP --fetch-assign pattern --gemv-wave-reduce 1
cfg st_t_a2    none  $TP $OLD
# LAST: --glm-prefill-stage 1 allocates the staging ring once and keeps it for the process; the 262 144-cell, chunk-1024 timing arms above must not carry it
# (it OOMed the first timing config on 2026-10-07 when this line sat second)
cfg st_c512s   exact $C136 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 $NEW --oracle $B/gpu5/g136_eager
echo "=== glm_stack start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_stack exit rc=2 $(date -Is)"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; echo "=== glm_stack exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
echo "=== decode ms/token (median of 32 at depth ~8 290): A main / B wave-reduce / C fetch-assign / D both -- A,B,C,D,D,C,B,A"
awk '/^=== gate-plan config/ {c=$4} /^glm5_decode_ms_median=/ && c ~ /^st_t_/ {split($1,a,"="); printf "  %-8s %s\n", c, a[2]; d[c]=a[2]}
     END { n=0; split("a1 b1 c1 d1 d2 c2 b2 a2", k, " "); for (i=1;i<=8;i++) if (("st_t_" k[i]) in d) n++
           if (n==8) { A=(d["st_t_a1"]+d["st_t_a2"])/2; Bm=(d["st_t_b1"]+d["st_t_b2"])/2; Cm=(d["st_t_c1"]+d["st_t_c2"])/2; Dm=(d["st_t_d1"]+d["st_t_d2"])/2
             printf "  means: A %.3f  B %.3f (B/A %.3f)  C %.3f (C/A %.3f)  D %.3f (D/A %.3f)\n", A, Bm, Bm/A, Cm, Cm/A, Dm, Dm/A } else print "  incomplete: a timing config did not finish" }' "$O/gate_run.log"
echo "=== glm_stack exit rc=$rc $(date -Is)"
exit $rc
