#!/bin/bash
# glm_fr_chain.sh -- is a SEPARATE reduce launch faster than the fused in-kernel reduce for the high-split decode GEMVs, bit-exactly? (2026-10-07)
# Why: the decode trace (record §L5-GLM-DECODE-TRACE) shows the trunk GEMVs with many K splits take ~10 us + 0.1 us x nsplit
# (hc_*_fn: 512 splits, 62 us for 0.4 MB, 90 launches a token = 5.6 ms; ffn_gate_inp: 57 splits, 22 us): the fused reduce's one counter
# per output row, incremented by every split workgroup, serialises. franken-engine main `5a4fa03` wires `--gemv-fused-reduce 0|1`
# and `--gemv-fused-max-split N` (fuse only up to N splits) into the GLM runner, per --gate-plan line. ONE process, one load:
#   exactness (every tap bit-exact against the pre-change reference, chunk 1 + 8 greedy tokens; and 64 greedy tokens at depth 1 500 against an
#   in-process default dump) for the cap-64 and the all-separate variants, then decode timing A,B,C,C,B,A:
#   A default (all fused) / B --gemv-fused-max-split 64 / C --gemv-fused-reduce 0 (all separate).
# NOTE (2026-10-07): since franken-engine 5a4fa03 the GLM default IS the separate reduce, so on a current binary arm A ("default") equals arm C; the experiment's
# historical A arm was `--gemv-fused-reduce 1` (add it to the A lines to repeat it). The chain is kept as the record of how the result was measured.
# Launch through run_chain.sh and watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#     ~/bench/glm_fr_chain.sh > ~/bench/glm_fr_chain.log 2>&1 < /dev/null &
# Env: FR_BIN (default franken-engine main's franken_decode_glm, `make -C franken/decode gpu`), FR_OUT (default ~/bench/franken/glm5/fr), FR_CAP (default 64).
set -u
BIN=${FR_BIN:-$HOME/src/franken-engine/franken/decode/franken_decode_glm}
VERDICT=${FR_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${FR_OUT:-$HOME/bench/franken/glm5/fr}; rm -rf "$O"; mkdir -p "$O"
CAP=${FR_CAP:-64}
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --time 32 --time-settle 64 --adapt-prefill 0"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
cfg fr_c1_cap  exact $C136 --chunk 1 --gemv-fused-max-split $CAP --oracle $B/gpu5/g136_eager
cfg fr_c1_sep  exact $C136 --chunk 1 --gemv-fused-reduce 0 --oracle $B/gpu5/g136_eager
cfg fr_ref     ref   $D1500 --dump $O/s64_ref
cfg fr_x_cap   exact $D1500 --gemv-fused-max-split $CAP --oracle $O/s64_ref
cfg fr_x_sep   exact $D1500 --gemv-fused-reduce 0 --oracle $O/s64_ref
cfg fr_t_a1    none  $TP
cfg fr_t_b1    none  $TP --gemv-fused-max-split $CAP
cfg fr_t_c1    none  $TP --gemv-fused-reduce 0
cfg fr_t_c2    none  $TP --gemv-fused-reduce 0
cfg fr_t_b2    none  $TP --gemv-fused-max-split $CAP
cfg fr_t_a2    none  $TP
echo "=== glm_fr start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) cap=$CAP"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_fr exit rc=2 $(date -Is)"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; echo "=== glm_fr exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
# the verdict names an engine death and its cause (glm5_gate_verdict.sh LOG CHECKS ENGINE_RC); the timing table follows
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
echo "=== decode ms/token (median of 32 at depth ~8 290): A default, B cap $CAP, C all separate -- A,B,C,C,B,A"
awk '/^=== gate-plan config/ {c=$4} /^glm5_decode_ms_median=/ && c ~ /^fr_t_/ {split($1,a,"="); printf "  %-8s %s\n", c, a[2]; d[c]=a[2]}
     END { if (("fr_t_a1" in d) && ("fr_t_a2" in d) && ("fr_t_b1" in d) && ("fr_t_b2" in d) && ("fr_t_c1" in d) && ("fr_t_c2" in d)) {
             A=(d["fr_t_a1"]+d["fr_t_a2"])/2; Bm=(d["fr_t_b1"]+d["fr_t_b2"])/2; Cm=(d["fr_t_c1"]+d["fr_t_c2"])/2
             printf "  means: A %.3f  B %.3f (B/A %.3f)  C %.3f (C/A %.3f)\n", A, Bm, Bm/A, Cm, Cm/A } else print "  incomplete: a timing config did not finish" }' "$O/gate_run.log"
echo "=== glm_fr exit rc=$rc $(date -Is)"
exit $rc
