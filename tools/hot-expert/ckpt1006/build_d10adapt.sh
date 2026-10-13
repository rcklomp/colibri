#!/bin/bash
# build_d10adapt.sh -- D10 LANE ADAPTATION (franken-engine branch `d10-lane-adapt` = main 0436ded + one commit; GLM5.md section 33.9): the CPU lane's 4-lane table follows the lane's
# MEASURED slab time (--cpu-lane-adapt 0|1, default 1 with the lane on; --cpu-lane-adapt-halflife SLOTS, 64; serve FRANKEN_GLM_CPU_LANE_ADAPT, _ADAPT_HALFLIFE). NOT merged, nothing has run on a GPU.
# Builds BOTH binaries in a throw-away worktree of the rig's franken-engine (like build_d10probe.sh):
#   franken_decode_glm_d10a   make gpu           GPU_BIN=...   (the CLI / gate-plan binary)
#   franken_dec_glm_d10a      make glm-serve-gpu   GLM_GPU_BIN=... (the serving binary)
# and copies them next to the installed ones in ~/bench/franken_bin/ (nothing else there is touched), prints their sha256 prefixes, removes the worktree.
# COMPILE ONLY: starts no engine, uses no GPU (hipcc offline codegen in docker without --device). nice 19, make -j2.
# REFUSES to run while the serve A/B chain runs (any CPU load on the rig biases the lane's arms): D10A_FORCE=1 overrides.
# Reach the rig first (Mac, in ~/Projects/franken-engine-d10adapt or ~/Projects/franken-engine):
#   git push ssh://rome/home/ronald/src/franken-engine d10-lane-adapt:refs/heads/d10-lane-adapt
# Then on the rig:  ~/bench/build_d10adapt.sh      (D10A_BRANCH / D10A_CLI / D10A_SERVE override the branch and the binary names)
# Use: CLI  franken_decode_glm_d10a ... --fetch-assign optimal --cpu-lane 6 [--cpu-lane-adapt 1] [--cpu-lane-adapt-halflife 64]   (no --hip-graph: adaptation is off under graphs)
#      serve  FRANKEN_GLM_FETCH_ASSIGN=optimal FRANKEN_GLM_CPU_LANE=6 [FRANKEN_GLM_CPU_LANE_ADAPT=1] [FRANKEN_GLM_CPU_LANE_ADAPT_HALFLIFE=64] franken_dec_glm_d10a
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-d10a; OUT=$HOME/bench/franken_bin
BR=${D10A_BRANCH:-d10-lane-adapt}; CLI=${D10A_CLI:-franken_decode_glm_d10a}; SRV=${D10A_SERVE:-franken_dec_glm_d10a}
if [ "${D10A_FORCE:-0}" != 1 ] && pgrep -f "glm_[s]erve_ab_chain" >/dev/null; then echo "the serve A/B chain is running: not compiling (D10A_FORCE=1 overrides)"; exit 2; fi
git -C "$SRC" rev-parse --verify -q "refs/heads/$BR" >/dev/null || { echo "no branch $BR in $SRC (push it from the Mac first)"; exit 1; }
[ -e "$WT" ] && { echo "$WT exists (a build in progress, or a left-over: git -C $SRC worktree remove --force $WT)"; exit 1; }
git -C "$SRC" worktree add --detach "$WT" "$BR" || exit 1
cd "$WT/franken/decode" || exit 1
git log --oneline -1 | cut -c1-100
nice -n 19 make -j2 gpu GPU_BIN=$CLI 2>&1 | grep -vE "^docker run|^\s*$" | tail -25
rc1=${PIPESTATUS[0]}
nice -n 19 make -j2 glm-serve-gpu GLM_GPU_BIN=$SRV 2>&1 | grep -vE "^docker run|^\s*$" | tail -25
rc2=${PIPESTATUS[0]}
mkdir -p "$OUT"
for B in "$CLI" "$SRV"; do
  if [ -x "$B" ]; then cp -p "$B" "$OUT/" && sha256sum "$OUT/$B" | cut -c1-16 && ls -la "$OUT/$B"
  else echo "BUILD FAILED: no $B (make rc=$rc1/$rc2)"; fi
done
git -C "$WT" rev-parse --short HEAD
git -C "$SRC" worktree remove --force "$WT"; echo done
