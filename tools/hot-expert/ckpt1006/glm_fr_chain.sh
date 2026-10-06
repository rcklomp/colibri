#!/bin/bash
# glm_fr_chain.sh -- does `--gemv-fused-reduce 0` speed up GLM DECODE, bit-exactly? (2026-10-07)
# Why: the decode trace (record §L5-GLM-DECODE-TRACE) shows the trunk GEMVs with many K splits take ~10 us + 0.1 us x nsplit
# (hc_*_fn: 512 splits, 62 us, 90 launches a token = 5.6 ms; ffn_gate_inp: 57 splits, 22 us; ...): one counter per output
# row incremented by every split workgroup serialises at the L2. decode_gpu.hip says the fused in-kernel reduce and the
# separate k_reduce_splits kernel "cannot move a bit"; the knob exists (`--gemv-fused-reduce 0|1`, global option). This runs
# the SAME config in three processes, fused (default) / separate / fused (A,B,A: the knob is process-wide), each with
# the chunk-1 exactness line against the pre-change reference and the decode timing (two consecutive timing configs a process).
# Launch through run_chain.sh, watch with watch_chain.sh <chain log> <engine log of the last process>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#     ~/bench/glm_fr_chain.sh > ~/bench/glm_fr_chain.log 2>&1 < /dev/null &
# Env: FR_BIN (default the dpf worktree's franken_decode_glm_dpf, --decode-prefetch default 0 = the main path),
#      FR_OUT (default ~/bench/franken/glm5/fr), FR_ARMS (default "1 0 1": the --gemv-fused-reduce value of each process).
set -u
BIN=${FR_BIN:-$HOME/src/franken-engine-dpf/franken/decode/franken_decode_glm_dpf}
O=${FR_OUT:-$HOME/bench/franken/glm5/fr}; rm -rf "$O"; mkdir -p "$O"
ARMS=${FR_ARMS:-"1 0 1"}
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --time 32 --time-settle 64 --adapt-prefill 0"
cat > "$O/plan.txt" <<EOF
fr_c1 --tokens-file $B/t136.txt --ctx 4096 --greedy 8 --chunk 1 --oracle $B/gpu5/g136_eager
fr_t1 $TP
fr_t2 $TP
EOF
echo "=== glm_fr start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) arms=[$ARMS]"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
rc=0; i=0
for FR in $ARMS; do
  i=$((i+1)); L="$O/arm${i}_fused${FR}.log"
  echo "--- arm $i: --gemv-fused-reduce $FR  $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gemv-fused-reduce $FR --gate-plan "$O/plan.txt" > "$L" 2>&1
  r=$?; [ $r -ne 0 ] && rc=$r
  echo "arm $i rc=$r: $(grep -aE '^glm5_decode_ms_median=' "$L" | sed 's/ tok_s.*//' | tr '\n' ' ')"
  echo "arm $i taps not bit-exact (fr_c1): $(awk '/^=== gate-plan config/ {c=$4} c=="fr_c1" && /^oracle .*maxabs=/ { if ($0 !~ /maxabs=0 /) n++; t++ } END {print n+0 " of " t+0}' "$L")"
  grep -aE "HIP error|rror:|abort" "$L" | head -3 | cut -c1-200
done
echo "=== glm_fr exit rc=$rc $(date -Is)"
exit $rc
