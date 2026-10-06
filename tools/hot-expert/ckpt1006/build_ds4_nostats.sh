#!/bin/bash
# build_ds4_nostats.sh -- tip (b5cf4e0) + a ONE-LINE debug switch: FRANKEN_NO_ADAPT_STATS=1 passes a null counter pointer to the DS4 router
# (so adaptation is on -- mirror, tables -- but the router's atomicAdd counters never run). Build only; temporary worktree removed after.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-ds4dbg; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" b5cf4e0 || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
p='ds4_graph.cpp'; s=open(p).read()
old='adapt_ ? adapt_->stats(il) : nullptr, T);'
new='(adapt_ && !std::getenv("FRANKEN_NO_ADAPT_STATS")) ? adapt_->stats(il) : nullptr, T);'
assert s.count(old)==1; open(p,'w').write(s.replace(old,new))
PY
grep -n "cstdlib" ds4_graph.cpp | head -1 || sed -i '1,/#include/ s/#include/#include <cstdlib>\n#include/' ds4_graph.cpp
nice -n 19 make -j4 ds4-gpu DS4_GPU_BIN=franken_decode_ds4.nostats 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -5
[ -x franken_decode_ds4.nostats ] && cp -p franken_decode_ds4.nostats "$OUT/" && sha256sum "$OUT/franken_decode_ds4.nostats" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
