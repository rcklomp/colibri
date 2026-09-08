# Upstream `dev` merge — conflict surface, dry run 2026-09-07, done 2026-09-08

## What the dry run said (2026-09-07)

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

## What actually conflicted (2026-09-08, `upstream/dev` = 1ccee43)

`dev` moved 32 commits in the day between the dry run and the merge, and the
fork gained P6b, RP4 and P5. The merge base is `12a5c46`; the fork was **149
ahead / 125 behind**. The same three files conflict — the dry run's file list
held — but with **eight hunks, not five**, and three of them are not about
mapping at all:

| file | hunks | what it is | how it was resolved |
|---|---|---|---|
| `c/family_registry.py` | 1 | as predicted: the fork's P6 comment above `max_kv_slots` | ours, text refreshed (P6b lifted the clause about the forced CPU recurrence). The value 16 was never in conflict |
| `c/glm53.c` | 1 (Slot + the mapping table) | as predicted | `dev`'s primitive, fork's semantics — see below |
| `c/glm53.c` | 1 (`expert_read`) | as predicted | `dev`'s primitive, fork's semantics — see below |
| `c/glm53.c` | 2 (`run_layers`, the two hyper-connection sites) | **new**: the fork's parallel `coli_hc_pre`/`coli_hc_post` (G10, P2.3) and its `optime` timers vs `dev`'s new per-site wall clock `m->t_attn`/`m->t_ffn` (822a5c0, the dashboard's Profile tab) | **union**. Neither existed on the other side; the fork's structure is kept and `dev`'s `t_phase`/`now_s()` accounting is added inside it |
| `c/glm53.c` | 1 (`forward_span`, the head) | **new**: the fork's P2.3 last-row-only head + `optime` vs `dev`'s `m->t_head`/`m->forwards` | **union**, same reason |
| `c/qwen38_core.h` | 1 (`q38_load_native_fp8_ranges`) | as predicted | `dev`'s primitive, fork's prefault and counters |
| `c/qwen38_core.h` | 1 (before `q38_expert_get`) | **new**, adjacency only: the fork's `Q38_VK_TIER` block vs `dev`'s `q38_ehit_mark` | **union**; the two are unrelated |

Everything else auto-merged, including `c/openai_server.py`, `c/Makefile`,
`c/st.h`, `c/resource_plan.py`, `c/coli`, `docs/`.

## Where the resolution deviates from "take dev's", and why

`dev`'s hunk is the right *primitive* (`st_map_shard_range` in `st.h`, one
implementation for every engine — that is the whole point of #1325) and the
wrong *policy* for this rig. Taken literally it would have:

1. **turned the mapping off.** Upstream ships it opt-in (`COLI_MAP_EXPERTS=1`).
   Both of these engines were measured ON it. `st.h` therefore gained
   `COLI_MAP_EXPERTS_DEFAULT` (0 unless an engine defines it); `glm53.c` and
   `qwen38.c` define it to 1 before including `st.h`. The environment still
   wins both ways, so `COLI_MAP_EXPERTS=0` is the A/B and `GLM53_NO_MMAP` /
   `Q38_NO_MMAP` keep working. This is the direction upstream is going anyway:
   #1350's commit message calls itself "preparazione al flip del default di
   `COLI_MAP_EXPERTS` (#1325)".
2. **dropped the LRU sizing.** `expert_cache_init` reads `g_map_all` — "every
   expert of this checkpoint is servable from the mapping, so a slot costs no
   memory of ours" — and sizes the cache at one slot per expert. Nothing in
   `st.h` can compute that, so `expert_map_init` stays, reimplemented over
   `st_map_shard_range`. Measured on the rig: **288 slots per layer over 42
   sparse layers, 12 096/12 096 experts mappable, `copy=0`** on both binaries.
3. **raced.** `st.h` maps lazily on first use and its per-fd table is written
   without synchronisation; `glm53`'s `expert_read` runs inside an OpenMP
   region. `expert_map_init` forces every shard's mapping on one thread at
   load, which is also where `GLM53_POPULATE_LOAD` hooks.
4. **dropped both prefaults.** `glm53`'s is behind `GLM53_MMAP_POPULATE`,
   default OFF, and §G1/§G1b of the record are that knob's measurements;
   `qwen38`'s `q38_populate_range` is default ON and worth 4–6× on its CPU
   expert path. Both call sites are restored, and the bind counters they feed
   are byte-identical across the merge (`gpu=1645 cpu=5801 contig=0 split=5801`).

Dropped as the note asked: `g_fmap[]`/`g_fmaplen`/`GLM53_MAXFD`,
`q38_shard_map/base/len/tried`, `q38_shard_mapped`, `q38_mapped_range`,
`q38_unmap_shards` (no callers), `Slot.mapped` (written, never read). Kept
because something reads them: `g_map_serve`/`g_map_copy` and `g_n_bind_*`
(`[PROF]` and the streaming test), `q38_map_serve`/`q38_map_copy`
(`[Q38PROF]`). The `[MAP]` line stays, now reading
`[MAP] 59 file mappati (st_map_shard_range), esperti mappabili 12096/12096` —
it is the line the gate reads to see that the mapping is up, and the one thing
that would have shown a silent fall back to `pread` into anonymous memory.

## Result

Gate: §"Upstream dev merge" in `ROME-3x7900XTX-2026-09-04.md`.
