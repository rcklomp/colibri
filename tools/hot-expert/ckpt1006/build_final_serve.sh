#!/bin/bash
# build_final_serve.sh -- branch glm-prefill-final = head-gemv-blk (blocked head_gemv + blocked experts + 1024-row chunk) + the snapshot-spacing fix in glm5_serve.cpp, and its SERVING binary
# franken_dec_glm.final. The fix: snap_points() dropped a snapshot point closer than chunk_/2 to the previous one; at chunk 1024 that is 512, so the turn-boundary snapshot at the end of the shared
# tool block (~4548) lost to the multiple of FRANKEN_SNAP_EVERY at 4096 and a new chat reused 4096 of 4575 tokens (accept_live check 2: reused >= prompt-256, first token 4.04 s against 1.7-1.8).
# The gap stays at the old absolute 256 (min(chunk_/2, 256): identical for chunk <= 512), so the worst reuse shortfall stays under 256 tokens.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-final; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add -b glm-prefill-final "$WT" head-gemv-blk || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
p = 'glm5_serve.cpp'; s = open(p).read()
a = "            if (p - last < chunk_ / 2 || p <= from || p >= n) continue;"
b = ("            // the minimum gap between snapshot points is 256 tokens whatever the chunk: chunk_ / 2 is 512 at a 1024-row chunk, and a turn-boundary snapshot\n"
     "            // within 512 of a multiple of snap_every_ (the end of a 4.5k tool block next to 4096) was dropped, so a new chat reused 4096 of 4575 tokens\n"
     "            if (p - last < std::min(chunk_ / 2, 256) || p <= from || p >= n) continue;")
assert s.count(a) == 1, s.count(a); open(p, 'w').write(s.replace(a, b))
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; git -C "$SRC" worktree remove --force "$WT"; exit 1; }
git add glm5_serve.cpp && git commit -q -F - <<'MSG'
glm5-serve: keep the snapshot-point gap at 256 tokens at a 1024-row chunk

snap_points() dropped a point closer than chunk_ / 2 to the previous one: 256 at chunk 512, 512 at
chunk 1024. At 1024 the turn-boundary snapshot at the end of the shared ~4.5k-token tool block (~4548)
was dropped next to the multiple of FRANKEN_SNAP_EVERY at 4096, so a new chat reused 4096 of 4575 tokens
and its first token took 4.04 s against 1.7-1.8 s (accept_live check 2: reused >= prompt - 256).
The gap is now min(chunk_ / 2, 256): unchanged for chunk <= 512, and a boundary within 256 of a multiple
is the only one dropped, so the reuse shortfall stays under 256 tokens.

Found by accept_live on the combined build (record section L5-GLM-CHUNK); no engine arithmetic touched.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A8atmSKSpLq5mCFAYtqbVc
MSG
git log --oneline -3
nice -n 19 make -j4 glm-serve-gpu GLM_GPU_BIN=franken_dec_glm_final 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_dec_glm_final ] && cp -p franken_dec_glm_final "$OUT/franken_dec_glm.final" && sha256sum "$OUT/franken_dec_glm.final" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
