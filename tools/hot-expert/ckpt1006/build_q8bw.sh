#!/bin/bash
# build_q8bw.sh -- D7: worktree ~/src/franken-engine-d7 of franken-engine branch d7-push (pushed from the Mac's d7-q8bw), builds bench_q8bw (docker + hipcc, offline codegen
# for gfx1100) and copies it to ~/bench/franken_bin/bench_q8bw. Builds only; starts no GPU program, takes no rig lock.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-d7; OUT=$HOME/bench/franken_bin
echo "=== build_q8bw start $(date -Is)"
if [ ! -d "$WT" ]; then git -C "$SRC" worktree add --detach "$WT" d7-push || exit 1
else git -C "$WT" checkout -q --detach d7-push && git -C "$WT" reset -q --hard d7-push || exit 1; fi
git -C "$WT" log --oneline -1
cd "$WT/franken/decode" || exit 1
rm -f bench_q8bw bench_q8bw.o
nice -n 19 make -j4 bench-q8bw > "$HOME/bench/build_q8bw.make.log" 2>&1; rc=$?
echo "make rc=$rc"
grep -nE "warning|error|Error" "$HOME/bench/build_q8bw.make.log" | head -60 | cut -c1-260
grep -vE "^docker run|^\s+\.\./|^\s*$" "$HOME/bench/build_q8bw.make.log" | tail -15 | cut -c1-260
if [ -x bench_q8bw ]; then cp -p bench_q8bw "$OUT/bench_q8bw" && echo "bench_q8bw $(sha256sum "$OUT/bench_q8bw" | cut -c1-16)"; fi
echo "=== build_q8bw exit rc=$rc $(date -Is)"
