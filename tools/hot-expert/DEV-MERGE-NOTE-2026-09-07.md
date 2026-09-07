# Upstream `dev` merge — conflict surface, dry run 2026-09-07

`git merge-tree --write-tree hot-expert-tier upstream/dev` on the Mac
checkout (fork 129 ahead / 93 behind `JustVugg/colibri` `dev` as of
2026-09-07 07:00). Three files conflict, not one; everything else
auto-merges (`.gitignore`, `c/Makefile`, `c/openai_server.py`,
`docs/ENVIRONMENT.md`, …).

| file | hunks | what it is | resolution |
|---|---|---|---|
| `c/glm53.c` | 3 (at ~1647, ~1955, ~2029 of the merged file) | the fork's engine-local expert mapping (`Slot.mapped`, `g_fmap`/`GLM53_MAXFD`, `g_map_active` + `piece_mappable`, the `g_map_copy` counter) vs `dev`'s `st_map_shard_range` path (#1325) | **take `dev`'s** in all three (the roadmap's known conflict; #1324 was closed as superseded on 2026-09-07). Drop `expert_map_init` and the `[MAP] … file mappati` line; keep any fork-only counters only if something still reads them |
| `c/qwen38_core.h` | 1 (~1345) | the same mapping in the Qwen engine: fork's `q38_mapped_range` vs `dev`'s `st_map_shard_range` (behind `COLI_MAP_EXPERTS=1`) | **take `dev`'s**; then all four qwen38 C tests must pass (`tests/test_qwen38_prefix` is the one that used to SIGFPE) |
| `c/family_registry.py` | 1 (~950) | the fork's P6 comment block above `max_kv_slots` (ours 5 lines, theirs none) | **keep ours** (comment only; the value 16 is not in conflict) |

After resolving: `make -C c qwen38 qwen38-vk glm53 VK=1`, the four qwen38
tests, `prefill_gate.sh <served> <merged>` (expect neutral, `MIN_SPEEDUP=0.97`)
and `rome_bench.sh` reproducing its last rows — the gate the roadmap already
states. Serve only on rc 0 through a chain script.
