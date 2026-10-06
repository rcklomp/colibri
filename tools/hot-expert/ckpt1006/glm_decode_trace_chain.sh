#!/bin/bash
# glm_decode_trace_chain.sh -- where does one GLM DECODE token's time go? (2026-10-07; DEBUG: timings under tracing are
# inflated, the STRUCTURE -- kernel order, gaps, cross-card waits -- is what it shows.)
# Why: the decode-prefetch work (record §L4-LOOKAHEAD-STEP1) found ~0.8 ms per layer of FIXED cost from a handful of extra
# stream/event operations and tiny kernels, i.e. the layer is latency-bound, not only fetch-bound. rocprofv3 --kernel-trace
# gives every kernel's start/end on a common clock for all three cards, so the layer's critical path can be read without
# new engine code. Shipped flags (chunk 1024, expert-gb 17, adaptation on), no prefetch.
# Launch (on the rig) through run_chain.sh, watched with ckpt1006/watch_chain.sh:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh \
#     ~/bench/glm_decode_trace_chain.sh > ~/bench/glm_decode_trace_chain.log 2>&1 < /dev/null &
# Env: TRACE_BIN (default the dpf worktree's franken_decode_glm_dpf, which at --decode-prefetch 0 is the main path),
#      TRACE_OUT (default ~/bench/franken/glm5/dtrace), TRACE_TOKENS (decode tokens traced after the settle, default 12).
set -u
BIN=${TRACE_BIN:-$HOME/src/franken-engine-dpf/franken/decode/franken_decode_glm_dpf}
O=${TRACE_OUT:-$HOME/bench/franken/glm5/dtrace}; rm -rf "$O"; mkdir -p "$O"
N=${TRACE_TOKENS:-12}
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; PR=/home/ronald/bench/franken/glm5/prose8400.txt
# 2 048 prompt tokens (two 1 024-row chunks, so the lazily grown scratch is at its steady size), 24 settle + N timed decode tokens
echo "dtrace --tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 2048 --time $N --time-settle 24 --adapt-prefill 0" > "$O/plan.txt"
echo "=== glm_decode_trace start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; exit 2; }
m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); [ $u -gt $m ] && m=$u; done
[ $m -gt 1024 ] && { echo "REFUSED: a card holds $m MiB"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full /opt/rocm/bin/rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d "$O/out" -o prof -- \
  "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "gpu rc=$rc"
grep -aE "glm5_decode_ms_median|HIP error|rror|abort" "$O/gate_run.log" | cut -c1-200 | head -5
find "$O/out" -name "*.csv" | head
echo "=== glm_decode_trace exit rc=$rc $(date -Is)"
exit $rc
