#!/bin/bash
# build_chunk1024.sh -- EXPERIMENT build of franken-engine `main` with the prefill chunk cap raised from 512 to 1024 rows (binary franken_decode_glm_chunk1024).
# Why: every chunk copies ALL the staged experts (92.5 GB, routing-independent: record §L5-GLM-TIMELINE), so DMA bytes per token scale as 1/chunk; the engine
# capped chunks at 512 in four places (glm5_graph.cpp Tm_ and the CLI check, glm5_gpu.inc moe_reserve and the chunk-plan kernel's LDS array
# `sid[512 * 8]`; 1024 rows = 32 KB of LDS, under the 64 KB limit; 2048 would need the ids read from global memory instead). Chunk 1024 doubles the per-chunk
# scratch (~350 MB a card for the helper contexts): run it with a smaller ring (--glm-stage-mb 400; the ring size was flat 400 MB - 1.2 GB, §M7-HOSTSRC).
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-chunk; OUT=$HOME/bench/franken_bin
git -C "$SRC" worktree add --detach "$WT" main || exit 1
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
nice -n 19 make -j4 gpu GPU_BIN=franken_decode_glm_chunk1024 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -8
[ -x franken_decode_glm_chunk1024 ] && cp -p franken_decode_glm_chunk1024 "$OUT/" && sha256sum "$OUT/franken_decode_glm_chunk1024" | cut -c1-16
git -C "$SRC" worktree remove --force "$WT"; echo done
