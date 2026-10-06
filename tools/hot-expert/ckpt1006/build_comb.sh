#!/bin/bash
# build_comb.sh -- EXPERIMENT build: branch moe-regblock (the assignment-blocked routed-expert kernel, f6c69e5, bit-exact in the emb gate) + the chunk cap raised from 512 to 1024
# (the four changes of build_chunk1024.sh). Binary franken_decode_glm_comb. Not merged.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-comb; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" moe-regblock || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
def sub(p, pairs):
    s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, 'w').write(s)
sub('glm5_gpu.inc', [
 ("constexpr int GLM5_CHUNK_MAX_ASSIGN = 512 * G5::N_EXPERT_USED;", "constexpr int GLM5_CHUNK_MAX_ASSIGN = 1024 * G5::N_EXPERT_USED;"),
 ('            if (t_max > 512) die("moe_reserve: a chunk is at most 512 rows");', '            if (t_max > 1024) die("moe_reserve: a chunk is at most 1024 rows");'),
])
sub('glm5_graph.cpp', [
 ("    Tm_ = std::max(1, std::min(cfg_.max_chunk, 512));", "    Tm_ = std::max(1, std::min(cfg_.max_chunk, 1024));"),
 ('    if (r.chunk > 512) { why = "--chunk is at most 512"; return false; }', '    if (r.chunk > 1024) { why = "--chunk is at most 1024"; return false; }'),
])
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; git -C "$SRC" worktree remove --force "$WT"; exit 1; }
nice -n 19 make -j4 gpu GPU_BIN=franken_decode_glm_comb 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_decode_glm_comb ] && cp -p franken_decode_glm_comb "$OUT/" && sha256sum "$OUT/franken_decode_glm_comb" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
