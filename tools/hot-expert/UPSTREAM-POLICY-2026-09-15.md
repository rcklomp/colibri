# Upstream policy: diverge by default, cherry-pick by exception (decided 2026-09-15)

> The question put to this session was "converge or diverge": the fork is 231
> commits behind `upstream/dev` and 356 ahead, a whole-`dev` merge is blocked
> on #1519, and the last merge (2026-09-08, `DEV-MERGE-NOTE-2026-09-07.md`)
> would, taken literally, have shipped four traps. This note records the
> decision, the evidence it rests on, the one claim in the hand-off that
> turned out to be false, and the condition under which the decision is
> revisited.

## Decision

**No more whole-`dev` merges. `upstream/dev` is a source of individually
cherry-picked fixes, each through the normal gate, each naming its upstream
sha in the commit body. Convergence flows the other way: our items go
upstream as PRs against `upstream/dev` through `upstream_lint.sh`, as #1521
and #1524 already did.**

Re-open a full re-port — not a merge — only when **both** hold: #1519 is
resolved upstream, *and* upstream lands something this box needs that cannot
be taken as a cherry-pick (the concrete candidate is `expert_ffn.h`, upstream's
single routed-expert kernel, once it reaches `glm53`/`qwen38` and someone has
measured it faster here). Then it is a planned project: our landed items
re-applied one at a time onto a fresh `upstream/dev` checkout, each re-gated,
sized in days, with its own note. It is not a `git merge`.

## The evidence

**Churn since the merge base `1ccee43` on the files that matter here.**

| file | upstream | ours |
|---|---:|---:|
| `c/qwen38_core.h` | +230 / −8 | **+1 775 / −105** |
| `c/glm53.c` | +774 / −162 | **+3 373 / −115** |
| `c/backend_vulkan.c` | +13 | **+1 568 / −47** |
| `c/openai_server.py` | +288 / −11 | **+1 200 / −50** |
| `c/st.h` | **+237 / −15** | +14 / −1 |
| `c/quant.h` | +35 | +265 |
| `c/sparse_index.h` | 0 | +152 |
| `c/Makefile` | +160 / −45 | +16 / −2 |
| `c/shaders/` | +537 (8 files) | our tiled shaders |

Every file where upstream moved a lot is a file where we moved five to a
hundred times more, on the same functions: upstream's qwen38 commits are a
CUDA expert tier and int8 dense residents on the GPU (`a5d177b`, `19ab934`,
`1b4d19a`) — their version of our Q7, on a backend this box does not have —
and a timer-bank split (`ca93285`) we did ourselves in §Q-PROFILE; upstream's
glm53 commits are a Metal MoE path, mirror-drive expert reads, English
diagnostics (75 lines of renamed strings, **including the `[MAP] … file
mappati` line every gate here greps**), a CANCEL fix (#1332) for a bug this
fork fixed its own way on 2026-09-09, and routing telemetry.

**What upstream has that this box would actually use**, read commit by commit
over the 231:

| sha | what | verdict |
|---|---|---|
| `5ea97e8` | `glm53` serve: an over-long prompt returns `CONTEXT_EXCEEDED` with the numbers instead of `BAD_REQUEST` | **cherry-pick** — 5 lines, our gateway already maps the error name (6 references in `openai_server.py`), our engine never emits it |
| `fc6e7bb` | publish the lazy HITS table under a lock (a parallel first touch segfaulted qwen36's CUDA warmstart) | take if `ehit_mark` is ever reached from a parallel region here; today it is called from the serve loop — **park, harmless** |
| `941d5fe` | `coli_omp_tune_threads` in the four engines that never sized their OpenMP team | **not needed** — `start_glm53.sh` line 16 already pins `OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`, and so do `coli` and `datapoint.py` |
| `d799354` | stop ids from `config.json` as well as `generation_config.json` | **not needed** — GLM-5.3's two files carry the same `eos_token_id` list |
| `88082cc`, `57e5115` | `st.h`: missing core tensor diagnosed through the index; duplicate names resolved | nice-to-have diagnostics; **park** |
| `772f961` | Vulkan: reserve the tier's RAM on integrated GPUs | discrete cards here; **no** |
| `8e09479` | CANCEL honoured mid-turn (#1332) | **no** — ours landed 2026-09-09 (prefill roadmap "Closed dependency: CANCEL") and is gated |
| everything else | CUDA / HIP / Metal / Windows / qwen36 / deepseek-v41 / mirror drives | **no** — not this box |

One cherry-pick of five lines is the whole of what 231 commits offer this
rig today. Against that, a merge costs: conflicts in the four most-changed
files; the string rename that breaks every `[MAP]` grep in the gates; two
double-fixes (CANCEL, the timer banks) to unpick; the `st.h` policy fight
again (`DEV-MERGE-NOTE` items 1–4: mapping default, LRU sizing, the race,
both prefaults); and then `devmerge_chain.sh`'s six steps of rig time. And
#1519 is still open, so the merged `st.h` would still race — our tree avoids
it only because `expert_map_init` maps every shard on one thread at load,
which is exactly the kind of local policy a merge overwrites.

**The hand-off's "partial merge" claim is false, and it matters because it
was the strongest argument for converging.** It said the 2026-09-08 merge took
upstream's `test_qwen38_native_weights.c` without the engine code it
exercises (`qt_ready`, `qt_note`, `q38_tier_note`, a `q38_bind_fp8_slot`
site from `a5d177b`), citing the test passing on `upstream/dev` and failing
at line 635 here. Checked on 2026-09-15 by building the test at three
commits on the Mac (arm64, scalar, so line 587 is out of the way):

| tree | default | `COLI_MAP_EXPERTS=0` | `COLI_MAP_EXPERTS=1` |
|---|---|---|---|
| `1ccee43` (the merge base — no `a5d177b` either) | PASS | — | **FAIL 635** |
| `upstream/dev` | PASS | — | **FAIL 635** |
| ours | FAIL 635 | PASS | FAIL 635 |

`a5d177b` is not an ancestor of the merge base, so nothing from it could have
been "partially" taken; the test file is byte-identical across all three
trees; and the merge base passes without those symbols. Line 635 asserts that
a fast native-FP8 expert lives in a slot-owned slab. Our fork serves experts
from the shard mapping by default (`COLI_MAP_EXPERTS_DEFAULT 1`, `d907d78`,
2026-09-03; the merge kept it deliberately, `DEV-MERGE-NOTE` item 1), so the
slab is released by design and the pointer is NULL — bytes, read counters
and gate/up/down adjacency all correct. Upstream's own test fails the same
way with the mapping on, and will fail for them the day #1350 flips the
default. The test is fixed in this tree to accept either shape (an
upstream-worthy change); no engine code was wrong. **The fork's divergence
here was a policy we chose and measured, not a hunk we dropped.**

## How to take a cherry-pick

1. `git cherry-pick -x <sha>` onto a `perf/` or `fix/` branch; `-x` puts the
   upstream sha in the body.
2. Build both engines; the item's own oracle (bit-identical expected for a
   diagnostics change) and `accept_live.sh` if it touches serving.
3. Land through the normal chain; one row in the record if a number moved,
   one line in the roadmap's housekeeping if not.
4. `doc_currency.sh` before the commit.

## How to send something up

`tools/hot-expert/upstream_lint.sh upstream/dev <branch>` before opening and
after every edit — the CLAUDE.md section "Sending anything upstream" applies
unchanged. The base-freshness check it runs is why: #1321's CI died with zero
jobs at 43 commits behind, and that reads as a broken patch in the
notification mail.
