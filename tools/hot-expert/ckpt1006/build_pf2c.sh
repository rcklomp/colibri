#!/bin/bash
# build_pf2c.sh -- builds the PF2c binaries (--gemm-lds-min-rows, ACTION-LIST row 5h) from the rig's franken-engine tree (the branch that is checked out; set PF2C_NAME for the suffix):
#   franken_decode_glm_$NAME   the GPU/CLI binary the gate chains run
#   franken_dec_glm_$NAME      the serving binary, copied to ~/bench/franken_bin/franken_dec_glm.$NAME (NOT installed)
# Do not run it while a chain uses these binaries or the GPUs (a build beside a timing arm costs it 2-3 %). Usage: PF2C_NAME=pf2c ~/bench/build_pf2c.sh
set -u
N=${PF2C_NAME:-pf2c}; SRC=$HOME/src/franken-engine; OUT=$HOME/bench/franken_bin
cd "$SRC/franken/decode" || exit 1
git log --oneline -1; git status -sb | head -3
for t in "gpu GPU_BIN=franken_decode_glm_$N" "glm-serve-gpu GLM_GPU_BIN=franken_dec_glm_$N"; do
  echo "== make $t $(date +%T)"; nice -n 19 make -j8 $t 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
done
ls -la franken_decode_glm_$N franken_dec_glm_$N 2>&1 | cut -c1-120
[ -x franken_dec_glm_$N ] && cp -p franken_dec_glm_$N "$OUT/franken_dec_glm.$N" && sha256sum "$OUT/franken_dec_glm.$N" | cut -c1-16
echo "done $(date +%T)"
