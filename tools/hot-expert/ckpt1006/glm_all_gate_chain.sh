#!/bin/bash
# glm_all_gate_chain.sh -- the standard GLM-5.3 GPU gate (`glm5_gpu_gate.sh all`, GLM5.md sections 9.6 / 11.6 / 12.6) on a BUILD, as a chain: ONE GPU process on the served IQ4_XS, every
# schedule held to the CPU / eager references, then the verdict table. Required before a franken-engine branch is merged to main (a branch that touches a kernel every chunk or token runs).
# PF14 (2026-10-09): run on `franken_decode_glm_pf14` (branch pf14-iq2s: IQ2_S / IQ3_XXS expert types, chunks up to 2 048, the chunk-plan kernel streaming ids through LDS tiles).
# The output goes to its OWN directory (default ~/bench/franken/glm5/gpu5_$AG_TAG), so the existing gpu5 references of the installed build are not overwritten. Needs the CPU references
# ~/bench/franken/glm5/gpu/{cpu6,cpu2200} (they exist; they do not depend on the build).
# Launch: setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_all_gate_chain.sh > ~/bench/glm_all_gate_chain.log 2>&1 < /dev/null &
# Env: AG_D (decode dir holding the binaries; default the pf14 worktree), AG_BIN (default franken_decode_glm_pf14), AG_TAG (default pf14), AG_OUT.
set -u
. "$HOME/bench/chain_preflight.sh"
D=${AG_D:-$HOME/src/franken-engine-pf14/franken/decode}; BIN=${AG_BIN:-franken_decode_glm_pf14}; TAG=${AG_TAG:-pf14}
O=${AG_OUT:-$HOME/bench/franken/glm5/gpu5_$TAG}
say_end() { echo "=== glm_all_gate exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$D/$BIN" "$D/franken_decode_glm_cpu" "$D/glm5_gpu_gate.sh" "$D/glm5_gate_verdict.sh" "$HOME/bench/franken/glm5/gpu/cpu6/index.txt" "$HOME/bench/franken/glm5/gpu/cpu2200/index.txt"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }
done
echo "=== glm_all_gate start $(date -Is) bin=$BIN sha=$(sha256sum "$D/$BIN" | cut -c1-16) out=$O"
rig_quiet_wait 1800 || say_end 3
mkdir -p "$O"
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e GLM5_GPU_OK=1 -e GLM5_D="$D" -e GLM5_GPU_BIN="$BIN" -e GLM5_GATE_OUT="$O" \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full bash "$D/glm5_gpu_gate.sh" all > "$O/gate.txt" 2>&1
rc=$?
echo "gate rc=$rc (0 PASS, 1 FAIL, 3 the engine died)"; tail -n 40 "$O/gate.txt" | cut -c1-210
say_end "$rc"
