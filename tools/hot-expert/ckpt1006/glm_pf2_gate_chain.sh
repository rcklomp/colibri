#!/bin/bash
# glm_pf2_gate_chain.sh -- PF2 (ACTION-LIST row 5g): how far does the f16 / rocBLAS trunk GEMM (`--gemm-lds 3`) move the engine's numbers? ONE GPU process on the hybrid, the engine's
# own eager chunk-1 run as the reference (two prompts: 136 tokens = the gate's t136, and the first 2 048 of the prose file), then:
#   --gemm-lds 0 at chunk 1 024     exact  (the default arm is untouched by PF2: bit for bit)
#   --gemm-lds 1 at chunk 136/1 024 cos    (the shipped arm: the control for what a reassociated GEMM costs)
#   --gemm-lds 3 at chunk 136/1 024 cos    (the f16 arm; T >= 128 so it engages) -- every tap cos >= 0.999 and the greedy ids held to the reference
#   --gemm-lds 3 at chunk 32        cos    (T < 128: must fall back to mode 1 and match mode 1's numbers)
# The last config prints --profile's f16 statistics (largest |x| / |result| the f16 arm saw, and how many values the f16 range could not hold: must be 0).
# Launch (rig idle, service stopped): setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf2_gate_chain.sh > ~/bench/glm_pf2_gate_chain.log 2>&1 < /dev/null &
# Env: PF2G_D (decode dir), PF2G_BIN (franken_decode_glm_f16), PF2G_MODEL (the hybrid), PF2G_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF2G_D:-$HOME/src/franken-engine/franken/decode}; BIN=$D/${PF2G_BIN:-franken_decode_glm_f16}; VERDICT=$D/glm5_gate_verdict.sh
O=${PF2G_OUT:-$HOME/bench/franken/glm5/pf2_gate}; mkdir -p "$O"
M=${PF2G_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt
say_end() { echo "=== glm_pf2_gate exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$VERDICT" "$M" "$PR" "$B/t136.txt"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf2_gate start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
[ -s "$B/prose2048.txt" ] || cut -d' ' -f1-2048 "$PR" > "$B/prose2048.txt"
rig_quiet_wait 1800 || say_end 3
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"; P2K="--tokens-file $B/prose2048.txt --ctx 4096 --greedy 8"
PL=$O/plan.txt; CK=$O/checks.txt; : > "$PL"; : > "$CK"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$PL"; echo "$n $c" >> "$CK"; }
cfg f2_e136     ref   $C136 --three-link 0 --adapt 0 --chunk 1 --hip-graph 0 --dump $O/g136_eager
cfg f2_e2k      ref   $P2K  --three-link 0 --adapt 0 --chunk 1 --hip-graph 0 --dump $O/g2k_eager
cfg f2_k1024g0  exact $P2K  --chunk 1024 --gemm-lds 0 --oracle $O/g2k_eager
cfg f2_k136g1   cos   $C136 --chunk 136  --gemm-lds 1 --oracle $O/g136_eager
cfg f2_k136g3   cos   $C136 --chunk 136  --gemm-lds 3 --oracle $O/g136_eager
cfg f2_k1024g1  cos   $P2K  --chunk 1024 --gemm-lds 1 --oracle $O/g2k_eager
cfg f2_k1024g3  cos   $P2K  --chunk 1024 --gemm-lds 3 --oracle $O/g2k_eager
cfg f2_k32g3    cos   $C136 --chunk 32   --gemm-lds 3 --oracle $O/g136_eager
cfg f2_prof     none  $P2K  --chunk 1024 --gemm-lds 3 --profile
LOG=$O/gate.log
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
  -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
  "$BIN" --model "$M" --placement "$P" --expert-gb 17 --gate-plan "$PL" > "$LOG" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "HIP error|Memory access|out of memory|FATAL|gemm-lds 3:|rocBLAS|rocblas" "$LOG" | sort -u | head -5 | cut -c1-200
"$VERDICT" "$LOG" "$CK" $rc 2>&1 | cut -c1-220 | tee "$O/verdict.txt"
grep -a "prof_gemm_f16_calls" "$LOG" | sort -u | cut -c1-200
[ "$rc" -ne 0 ] && { echo "--- engine log tail"; tail -n 14 "$LOG" | cut -c1-200; }
# the reference dumps are root-owned (written by the engine container): remove them through docker unless PF2G_KEEP=1
[ "${PF2G_KEEP:-0}" = 1 ] || docker run --rm -v /home/ronald:/home/ronald rocm/dev-ubuntu-24.04:7.14.0-full bash -c "rm -rf $O/g136_eager $O/g2k_eager"
say_end $rc
