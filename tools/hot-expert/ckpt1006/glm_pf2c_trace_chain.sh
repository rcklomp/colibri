#!/bin/bash
# glm_pf2c_trace_chain.sh -- PF2c step 1 (ACTION-LIST row 5h, PREFILL plan section 0h): what is left in the trunk after PF2, per KERNEL.
# The `--profile` classes of the f16 build (record §L5-PF2 item 3) say trunk KDA ~180, hyper-connection (hc_pre's 24-row mix GEMM) ~115, norm ~60-70, indexer ~30-40, FFN ~25 us/token; the
# device timeline (`--timeline`) has only classes (layer_a / moe_b / ...), no kernel names, so the per-kernel read is a rocprofv3 kernel trace: one process, the INSTALLED binary
# (franken_dec_glm = .f16) and model (the IQ3_XXS hybrid), the shipped env of start_franken_glm.sh, one 8 192-token prose prefill (8 chunks of 1 024), --gemm-lds 3.
# DEBUG: durations under tracing are inflated for the whole chunk, the per-kernel durations (and which kernel / grid a class is made of) are what it shows.
# VRAM: the shipped config, nothing extra (rocprofv3 is host-side buffers). Launch (rig idle, service OFF, reservation flag in place) through run_chain.sh:
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf2c_trace_chain.sh > ~/bench/glm_pf2c_trace_chain.log 2>&1 < /dev/null &
# Env: PF2C_BIN, PF2C_MODEL, PF2C_OUT, PF2C_TOKENS (8192), PF2C_TABLE (the analysis script), PF2C_EXTRA (more flags for the plan line, e.g. "--gemm-lds-min-rows 16").
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF2C_BIN:-$HOME/bench/franken_bin/franken_dec_glm}
O=${PF2C_OUT:-$HOME/bench/franken/glm5/pf2c_trace}; rm -rf "$O"; mkdir -p "$O"
M=${PF2C_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt
N=${PF2C_TOKENS:-8192}
TABLE=${PF2C_TABLE:-$HOME/bench/pf2c_kernel_table.py}
say_end() { echo "=== glm_pf2c_trace exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$M" "$PR" "$TABLE"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf2c_trace start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) model=$(basename "$M") tokens=$N"
echo "t --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --glm-help-copy 1 --gemm-lds 3 ${PF2C_EXTRA:-} --time-prefill $N" > "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
  -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
  /opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d "$O/out" -o prof -- \
  "$BIN" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "^prefill_tokens=|gate_plan_summary|vram_dev[0-9]_steady|HIP error|Memory access|out of memory|FATAL|gemm-lds 3:" "$O/gate_run.log" | cut -c1-200 | sort -u | head -10
[ "$rc" -ne 0 ] && { echo "--- engine log tail"; tail -n 10 "$O/gate_run.log" | cut -c1-200; }
find "$O/out" -name "*.csv" | head
echo "=== per-kernel table"
python3 -I "$TABLE" "$O/out" --tokens "$N" --chunk 1024 2>&1 | tee "$O/table.txt" | cut -c1-230
say_end "$rc"
