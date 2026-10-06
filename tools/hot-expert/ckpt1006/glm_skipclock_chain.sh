#!/bin/bash
# glm_skipclock_chain.sh -- DEBUG (timing only), follow-up to glm_skipclass_chain.sh: WHY does compute running beside the staged DMA slow the copies
# themselves (per link while a copy is in flight: skeleton 13.2/13.3/22.7 GB/s, only the trunk GEMMs 9.6/9.7/16.5, only the experts 10.4/10.9/17.9,
# everything ~8/8/13)? Passive sampler (gpu_sampler.py, 4 Hz, read-only amdgpu sysfs) beside the same gate-plan: each card's shader / memory / fabric /
# SoC clock, PCIe DPM level, board power and busy %, per config. Arms (mask of build_dbgskip.sh): 31 skeleton, 29 trunk GEMMs only, 30 experts only,
# 0 normal, as an A,B,B,A palindrome, 8 192 tokens each. If a loaded card drops its SoC / fabric clock, downshifts PCIe or sits on the power cap, that
# is the in-flight rate loss; if every clock and the link state are flat across the arms, the cause is not a clock and the m7 probe needs consumer
# kernels shaped like the trunk GEMM (handoff 2026-10-06 §5.1(b)). Launch through run_chain.sh.
set -u
BIN=$HOME/bench/franken_bin/franken_decode_glm_dbgskip
TAG=${1:-base}; O=$HOME/bench/franken/glm5/skipclock_$TAG; mkdir -p "$O"      # TAG: the power setting under test (clock_test.sh passes peak / compute)
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
FULL="--tokens-file $PR --ctx 262144 --chunk 512 --glm-prefill-stage 1 --adapt-prefill 0 --gemm-lds 1 --time-prefill 8192"
{ for m in 31 29 30 0; do echo "m${m}_a $FULL --debug-skip $m"; done; for m in 0 30 29 31; do echo "m${m}_b $FULL --debug-skip $m"; done; } > "$O/plan.txt"
for d in /sys/class/drm/card[0-9]/device; do echo "power state $(basename $(readlink -f $d)): level=$(cat $d/power_dpm_force_performance_level) profile=$(grep -E "^ *[0-9]+ .*\*" $d/pp_power_profile_mode | head -1 | tr -s " ")"; done
echo "=== skipclock[$TAG] start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"; cut -c1-20 "$O/plan.txt"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
: > "$O/gate_run.log"
python3 -I $HOME/bench/gpu_sampler.py run "$O/samples.tsv" "$O/gate_run.log" 0.25 & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
echo "gpu rc=$?"; kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
grep -aE "gate_plan_summary" "$O/gate_run.log" | head -2
echo "=== prefill ms/token per config"
awk '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-9s %8.3f ms/token  dma-equiv %5.1f GB/s\n", c, a[2], 180.7 / a[2] }' "$O/gate_run.log"
echo "=== sampler: per config and card, means (10 % trimmed each end). card2 = 48:00.0 = the lone link; card1 = 83:00.0 = engine dev0"
python3 -I $HOME/bench/gpu_sampler.py summary "$O/samples.tsv" "$O/gate_run.log"
echo "=== skipclock[$TAG] end $(date -Is)"
