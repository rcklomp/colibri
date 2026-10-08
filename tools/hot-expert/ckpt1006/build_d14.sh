#!/bin/bash
# build_d14.sh -- the D14 fix (franken-engine branch d14-attn-grid, pushed to the rig as refs/heads/d14-push): attention grid from the widest row of a launch. Builds the SERVING binary
# franken_dec_glm_d14 (it is also the CLI/gate binary: a non---serve command line goes to glm5_main) and copies it to ~/bench/franken_bin/franken_dec_glm.d14. Removes its worktree.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-d14; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" d14-push || exit 1
cd "$WT/franken/decode" || exit 1
git log --oneline -1
nice -n 19 make -j4 glm-serve-gpu GLM_GPU_BIN=franken_dec_glm_d14 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_dec_glm_d14 ] && cp -p franken_dec_glm_d14 "$OUT/franken_dec_glm.d14" && sha256sum "$OUT/franken_dec_glm.d14" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
