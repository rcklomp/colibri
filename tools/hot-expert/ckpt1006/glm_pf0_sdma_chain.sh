#!/bin/bash
# glm_pf0_sdma_chain.sh -- PF0 follow-up (plan Rev 102): the shipped prefill's chunks barely overlap. The in-place timeline (glm_pf0_main_chain.sh) shows, in every recorded chunk, the helper
# packet (two hipMemcpyPeerAsync: the plan ids + the chunk's x rows) of the FIRST MoE layer (card 0, layer 3) taking 3.6 s (card 1) and 1.6 s (card 2), ending within 3-11 ms of the
# previous chunk's LAST-layer packet (layer 44, owner card 2) and of card 2's boundary copy: copies INTO a card complete in issue order and a copy that waits for an event of another card
# (the owner's plan, released ~4 s later) blocks every later copy into that card, including those of the next chunk, which the host issued 12 s earlier. Hypothesis: one in-order copy
# queue per destination card (the SDMA engine) -> head-of-line blocking. Test: the same shipped config, timeline on, with ONE environment variable changed per process:
#   peer0  HSA_ENABLE_PEER_SDMA=0   device-to-device copies by blit kernels on the stream's own queue instead of the SDMA engine (H2D / D2H unchanged)
#   sdma0  HSA_ENABLE_SDMA=0        every copy by blit kernels
#   q16    GPU_MAX_HW_QUEUES=16     more hardware queues a device (the §L5-GLM-TIMELINE H1b knob; -8 % prefill in the STAGED config, +8 % decode)
# Each process: w0 (discarded), tl_a (off), tl_t (--timeline chunks 2-7), tl_b (off); then tl_union.py on its timeline. The baseline of the same chain is glm_pf0_main_chain.sh's tl_* arms
# (5.40-5.51 ms/token, period 5.617 s, 1.59 chunks in flight). Numerics: only the copy engine changes, bit-identical by construction; an adopted variant still goes through the model gate.
# Launch through run_chain.sh; watch with watch_chain.sh <chain log> $PF0_OUT/gate_run_all.log.   Env: PF0_BIN, PF0_OUT, PF0_VARIANTS ("peer0 sdma0 q16").
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${PF0_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pf0main}
O=${PF0_OUT:-$HOME/bench/franken/glm5/pf0_sdma}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
UNION=${PF0_UNION:-$HOME/src/colibri/tools/hot-expert/ckpt1006/tl_union.py}
SHIP="--ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0"
say_end() { echo "=== glm_pf0_sdma exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$PR" ] || { echo "FATAL: missing $BIN or $PR"; say_end 2; }
echo "=== glm_pf0_sdma start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
: > "$O/gate_run_all.log"; worst=0
for v in ${PF0_VARIANTS:-peer0 sdma0 q16}; do
  case $v in peer0) ENVV="HSA_ENABLE_PEER_SDMA=0";; sdma0) ENVV="HSA_ENABLE_SDMA=0";; q16) ENVV="GPU_MAX_HW_QUEUES=16";; *) echo "unknown variant $v"; continue;; esac
  D=$O/$v; mkdir -p "$D"
  { echo "w0   --tokens-file $PR $SHIP --time-prefill 8192"
    echo "tl_a --tokens-file $PR $SHIP --time-prefill 8192"
    echo "tl_t --tokens-file $PR $SHIP --time-prefill 8192 --timeline $D/timeline.csv --timeline-skip 2 --timeline-chunks 6"
    echo "tl_b --tokens-file $PR $SHIP --time-prefill 8192"; } > "$D/plan.txt"
  rig_quiet_wait 1800 || { echo "rig not quiet before $v"; worst=3; continue; }
  echo "=== variant $v ($ENVV) start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 -e "$ENVV" \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D/plan.txt" > "$D/gate_run.log" 2>&1
  rc=$?; cat "$D/gate_run.log" >> "$O/gate_run_all.log"
  echo "variant $v engine rc=$rc"; [ "$rc" -ne 0 ] && { worst=$rc; tail -n 8 "$D/gate_run.log" | cut -c1-200; }
  awk -v v="$v" '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-6s %-5s %8.4f ms/token = %6.1f tok/s\n", v, c, a[2], 1000 / a[2] }' "$D/gate_run.log"
  [ -s "$D/timeline.csv" ] && python3 -I "$UNION" "$D/timeline.csv" --label "$v ($ENVV)" 2>&1 | cut -c1-220
done
say_end "$worst"
