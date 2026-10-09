#!/bin/bash
# build_pf14.sh -- PF14 (ACTION-LIST row 5e2, plan section 0g): franken-engine branch pf14-iq2s (pushed to the rig as refs/heads/pf14-push): the IQ2_S routed-expert gate/up kernel for the
# GLM engine (Unsloth UD-IQ3_XXS experts). Builds in a PERSISTENT worktree ~/src/franken-engine-pf14 (the gate script runs binaries from its decode dir: GLM5_D=...): the GPU/CLI binary
# franken_decode_glm_pf14, the CPU reference franken_decode_glm_cpu, and the serving binary franken_dec_glm_pf14 (copied to ~/bench/franken_bin/franken_dec_glm.pf14, NOT installed: the served
# binary stays franken_dec_glm = .hc). Remove the worktree when PF14 is closed: git -C ~/src/franken-engine worktree remove --force ~/src/franken-engine-pf14
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-pf14; OUT=$HOME/bench/franken_bin
if [ -d "$WT" ]; then git -C "$WT" checkout -q --detach pf14-push || exit 1; else git -C "$SRC" worktree add --detach "$WT" pf14-push || exit 1; fi
cd "$WT/franken/decode" || exit 1
git log --oneline -1
for t in "gpu GPU_BIN=franken_decode_glm_pf14" "cpu CPU_BIN=franken_decode_glm_cpu" "glm-serve-gpu GLM_GPU_BIN=franken_dec_glm_pf14"; do
  echo "== make $t"; nice -n 19 make -j4 $t 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -6
done
ls -la franken_decode_glm_pf14 franken_decode_glm_cpu franken_dec_glm_pf14 2>&1 | cut -c1-120
[ -x franken_dec_glm_pf14 ] && cp -p franken_dec_glm_pf14 "$OUT/franken_dec_glm.pf14" && sha256sum "$OUT/franken_dec_glm.pf14" | cut -c1-16
echo done
