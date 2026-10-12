#!/bin/bash
# build_d10probe.sh -- D10 CEILING PROBE (G2, DECODE-OPEN-ITEMS-PLAN D10; record: the --d10-drop section): a DEBUG-ONLY build of franken-engine branch `d10-ceiling-probe`
# (main 791ad20 + one commit 1a54da3; NOT merged, never installed; ARCHIVED 2026-10-12 as franken-engine tag `archive/d10-ceiling-probe`, record §L5-G2-CEILING): the CLI binary franken_decode_glm_d10p with `--d10-drop K`, a per-config run option (a flag of the
# `--gate-plan` config line or of the command line, like --fetch-assign). For a DECODE token (T == 1) the LAST K (0..8) missed slots of every MoE layer are DROPPED: no card fetches them, no
# card computes them (plan card -1), the owner zeroes their down rows (their weighted contribution is exactly 0, no renormalisation), and the kept missed slots are assigned as the engine
# would assign n' = n - dropped misses (pattern index / --fetch-assign optimal table row n'). It simulates a free CPU lane taking K slabs a layer off the GPUs: the CEILING of what a CPU lane
# could save. The OUTPUT IS NOT THE MODEL'S (timing only). Chunks (T > 1, the prompt) are untouched. K = 0 = main's plan (bit-identical output).
# POSITIVE CONTROL (printed once after the --time windows; also with `--d10-drop 0`, where dropped/token must be 0 and fetched == missed):
#   [d10-drop] K=1 tokens=N dropped/token=X fetched/token=Y missed/token=Z layers/token=W (device tally ...)
# missed/token is the count BEFORE the drop: compare it between the K=0 and K>0 runs -- the dropped model walks a different token path, so its miss rate can differ (a confound).
# The serve path does not read this option (its own env parser, glm5_serve.cpp); only the CLI / gate-plan binary has it.
# Builds in a throw-away worktree of the rig's franken-engine (like build_pf0dbg.sh), copies the binary next to the installed ones as ~/bench/franken_bin/franken_decode_glm_d10p (the installed
# franken_dec_glm / .pf15 are not touched), prints its sha256 prefix, removes the worktree. COMPILE ONLY: starts no engine, uses no GPU (hipcc offline codegen in docker without --device).
# Reach the rig first (Mac, in ~/Projects/franken-engine):  git push ssh://rome/home/ronald/src/franken-engine archive/d10-ceiling-probe^{commit}:refs/heads/d10-ceiling-probe
# Then on the rig:  ~/bench/build_d10probe.sh      (D10P_BRANCH / D10P_NAME override the branch and the binary name; do not run while a chain compiles from the same tree)
# The timed run with the drop: glm_d10_ceiling_chain.sh (record §L5-G2-CEILING).
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-d10p; OUT=$HOME/bench/franken_bin
BR=${D10P_BRANCH:-d10-ceiling-probe}; NAME=${D10P_NAME:-franken_decode_glm_d10p}
git -C "$SRC" rev-parse --verify -q "refs/heads/$BR" >/dev/null || { echo "no branch $BR in $SRC (push it from the Mac first)"; exit 1; }
git -C "$SRC" worktree add --detach "$WT" "$BR" || exit 1
cd "$WT/franken/decode" || exit 1
git log --oneline -1 | cut -c1-100
nice -n 19 make -j4 gpu GPU_BIN=$NAME 2>&1 | grep -vE "^docker run|^\s*$" | tail -25
rc=${PIPESTATUS[0]}
if [ -x "$NAME" ]; then
  mkdir -p "$OUT" && cp -p "$NAME" "$OUT/" && sha256sum "$OUT/$NAME" | cut -c1-16 && ls -la "$OUT/$NAME"
else
  echo "BUILD FAILED: no $NAME (make rc=$rc)"
fi
git -C "$WT" rev-parse --short HEAD
git -C "$SRC" worktree remove --force "$WT"; echo done
