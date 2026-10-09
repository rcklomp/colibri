#!/bin/bash
# build_pf14.sh -- builds the PF14 binaries from the rig's franken-engine MAIN tree (PF14 / PF4 were merged to main on 2026-10-09; there is no worktree any more):
#   franken_decode_glm_pf14   the GPU/CLI binary the gate chains run (glm_all_gate_chain.sh, glm_hyb_chain.sh, glm_pf4_chain.sh default to this directory)
#   franken_decode_glm_cpu    the CPU reference arm
#   franken_dec_glm_pf14      the serving binary, copied to ~/bench/franken_bin/franken_dec_glm.pf14 (NOT installed: the served binary stays franken_dec_glm = .hc)
# Do not run it while a chain uses these binaries. Usage: ~/bench/build_pf14.sh
set -u
SRC=$HOME/src/franken-engine; OUT=$HOME/bench/franken_bin
cd "$SRC/franken/decode" || exit 1
git log --oneline -1
for t in "gpu GPU_BIN=franken_decode_glm_pf14" "cpu CPU_BIN=franken_decode_glm_cpu" "glm-serve-gpu GLM_GPU_BIN=franken_dec_glm_pf14"; do
  echo "== make $t"; nice -n 19 make -j4 $t 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -6
done
ls -la franken_decode_glm_pf14 franken_decode_glm_cpu franken_dec_glm_pf14 2>&1 | cut -c1-120
[ -x franken_dec_glm_pf14 ] && cp -p franken_dec_glm_pf14 "$OUT/franken_dec_glm.pf14" && sha256sum "$OUT/franken_dec_glm.pf14" | cut -c1-16
echo done
