# Roadmap: GLM-5.3 and Qwen3.8 on the rome box (written 2026-09-04)

Two tracks, planned so that most of the work runs on Opus 5 or smaller
models, with Fable reserved for the few decisions that need it. Every item
carries the measured evidence it rests on (from `ROME-3x7900XTX-2026-09-04.md`)
and a pass/fail gate, so a smaller model can execute it without judging
whether it worked: the gate judges.

## Model tiers, and what makes an item safe for a smaller one

| tier | use for | why it is safe |
|---|---|---|
| **Haiku 4.5** | running the campaign script, collecting numbers into the record, applying a fully specified patch, doc edits | the task is mechanical and the oracle (greedy text + logits diff, `datapoint.py` tables) decides |
| **Sonnet 5** | ports of an existing, measured pattern from one engine to the other; knobs; harness scripts; tests | the design already exists in the repo; the model copies and measures |
| **Opus 5** | new kernels and shaders with a numerics oracle; per-op profiling and its interpretation; anything where a measurement can contradict the plan | needs judgement, but the gates are still numeric |
| **Fable 5.1** | the two design documents (dense-off-DRAM, MTP), the choice of KDA approach after the GLM profile, and arbitration when a measurement contradicts the record | the expensive part is deciding, not typing |

Rule for every item, whatever the tier: measure before, measure after, same
prompt, same regime (`tools/datapoint.py`, physical-core threads, page cache
state verified with `fincore`), greedy text identical or the logit diff
explained, and the numbers go in the commit body. The record's three traps
(thread count, per-device caps, foreign-branch binaries) are in the harness so
nobody has to remember them.

## Cross-cutting first (week 1, Sonnet / Haiku)

| id | status | item | tier | gate |
|---|---|---|---|---|
| C0 | **DONE** 09-04 | `tools/rome_bench.sh`: drops caches, warms **one** model, verifies residency, pins 8 threads, sets the GLM caps, runs `datapoint.py`, appends a row. Asserts the GPU tier is really up and **refuses to record** otherwise — six config defects found writing it, incl. a mislabelled row it retro-corrected (record §C0) | Sonnet | 0.24% between runs, vs 3%: **met** |
| C1 | **DONE 09-06 — MERGED.** The gate is met and the merge is done. `tests/test_qwen38_prefix` SIGFPEd on every tree since `2d3cf7e`: `ensure_kv` computes `m->max_t / c->idx_ratio` and the ratio is 0 for any Model built directly rather than parsed from a config — SERVE paths and the test's own fabricated Model. Real checkpoints cannot reach it (`Q38_NEED` validates `idx_ratio>0` at load). Guarded to one block when the ratio is absent; for every config with `idx_ratio>0` it computes exactly what it always did. **exit 136 → exit 0**, all four qwen38 C tests pass, all three engines build clean, and `perf/rome-cpu-path` (67 commits) is merged into `hot-expert-tier` as `2c311a0` — verified building and passing from a clean checkout on the rig *before* the push | record §C1 bisected it 3/3 | unblocks the merge | done | Opus | test exit 0: **met**; merge: **done** |
| C2 | **DONE** 09-06, **re-taken 09-11** | Both engines now have the same four numbers. The 09-06 table is record §C2; it is superseded by **§Q-REBASE** (4.04 / 4.09 rotating, 5.23 / 5.25 warm, 3.45 / 3.37 cold), which also closes the "provisional cold column" flag §C2 left open | Haiku | table filled: **met** |

## Track G: GLM-5.3 — 2.60–2.75 tok/s rotating (was 1.65 at G0), 8 threads, 3 GPUs

Fresh-process decode is **156.77 ms/token knob-off and 134.90 knob-on**, down
from G3's 373.4 (**2.38× / 2.77×**) — re-profiled three times (**§RP1**,
corrected by **§RP1-CORRECTION**, then **§RP2** and **§RP3**). Two figures
exist because `COLI_KDA_GPU=2` (G12) is not bit-identical and ships off.

**Read §G14 and item 4f before planning anything.** RP3 called the two leading
buckets blocked and recommended buying VRAM; that was wrong, and item 4f
records why. The CPU int4 expert bucket (38.6 ms/token, and the leader knob-on
at 28.6%) is **not** blocked — it is **~57% memory and ~43% arithmetic in situ**,
and the arithmetic half has now been attacked to exhaustion (G14: everything
faster than +10% changes the model's output). **The memory half was attacked too, and it
lost: G15 (item 4h) simulated `fmt=5` int3 experts and GLM-5.3 does not survive
them — 16 of 1232 teacher-forced predictions change, cosine 0.878, item dead at
its own first gate (record §G15).** Its by-product is now the open item on this
engine: the routed-expert GPU kernel omits GLM-5.3's swiglu clamp, and that is
worth 8 of 1232 teacher-forced predictions and the long-prompt argmax by itself.
KDA remains solved-but-opt-in via G12.

**Status 2026-09-06 — TRACK G IS FINALISED AND MERGED.** `perf/rome-cpu-path`
(67 commits) is merged into `hot-expert-tier` as `2c311a0`, verified building
and passing from a clean checkout on the rig before the push. C0–C2 and
G0–G14 done, three re-profiles (RP1, RP2, RP3) done, SPEC-PROBE answered no.
**C1's SIGFPE is fixed and G6 has landed**, so the two long-standing blockers
are closed. G15 (int3 experts, item 4h) was executed on 2026-09-11 and is
**dead**: its numerics probe ran first, as the item required, and killed it
before any converter or kernel was built (record §G15).

**Update 2026-09-11 — G5 DONE, TRACK G FULLY CLOSED.** G5 was parked on a
product question ("what context length does the deployment see") that this
rig cannot answer, not on a correctness or feasibility blocker — the owner's
request to execute it specifically settled that prioritization question, so
it ran. Cached the pooled DSA-indexer block keys (record §G5): bit-identical
`teacher_forcing`/`last_logits` at 781 and 1509 tokens, cache on and off, and
the indexer's per-context-token constant drops 2.4–2.5× (1.97→0.78 µs at
ctx≈786, 2.07→0.87 µs at ctx≈1514), a real cut to the track's one genuinely
O(context²)-total component. Ships on by default (`GLM53_NO_INDEX_CACHE=1` is
the A/B). **Track G is now fully closed: G0–G15 all resolved** — landed
(G0–G14 except opt-in items), shipped opt-in (G12), correctly killed (G15),
or done and on by default (G5) — **plus the swiglu-clamp gap G15's control
run uncovered, which is the only open item this engine's decode path has
left.** The track is clear for phase Q.
RP3 found nothing to reorder and concluded the track was out of measurable
items; **that conclusion was wrong and is corrected in 4f**. Two profiles exist
because `COLI_KDA_GPU=2` ships off: **156.77 ms/token knob-off, 134.90
knob-on** (both before G14's ~2.5%, which is opt-in). The old header number (3.17
tok/s "fresh-process") is superseded twice over — G0 established the
persistent baseline this track is actually gated on (1.65–1.66), and
G2+G4+G7+G8+G9+G10 moved it to 2.60–2.75. G3 replaced the whole ordering
rationale with a measured profile; the items below G4 come from it, not
from the original plan.

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| G0 | **DONE** 09-04 — the yardstick everything since is measured against (record §G0): rotating **1.66 / 1.65**, warm-identical 2.06 / 2.07, cold 1.23 / 1.25, cold TTFT 17.9 / 17.5 s. Not comparable to the old 3.17 fresh-process figure; different regime | GLM had only fresh-process numbers | — | done | Haiku | four numbers in the record: **met** (0.61% between runs) |
| G1 | **DONE** 09-04 — ported, and it **regressed** (record §G1): rotating 1.65 → 1.48, cold 1.23 → 0.98, reproduced twice, bit-identical either way. Shipped **off by default** (`GLM53_MMAP_POPULATE`), the opposite of Qwen's default, because shipping it on would ship the regression. Whole-mapping-at-load (`GLM53_POPULATE_LOAD`) also off: 122 s of extra load for no measurable gain. Root cause in G1b | Qwen: fault stalls made its CPU expert path 4–6× slower than its own kernel | — | done | Sonnet | teacher_forcing identical **and** measured vs G0: **met** (the answer was "no") |
| G1b | **Done** — root-caused G1's regression (record §G1b): it was prefaulting the 32% of binds that the GPU then serves from VRAM; `mmap_lock` contention ruled out (1 thread ≈ 8 threads per call). Fixed, still net-negative on this box (7.5 thread-s of `madvise` to save 0.76), kept off by default because every fault here is minor — on a RAM-constrained box the sign flips | G1 | — | done | Opus | bit-identical; measured |
| G2 | **DONE** 09-04 (record §G2) — they **were** sequential: three submit-and-wait calls back to back, so dev2 could not start until dev0's fence signalled. Replaced with round-based issue-all/take-all reusing the `_issue`/`_take` pair already in `backend_vulkan.c`. Rotating 1.65–1.66 → **1.71 / 1.69**. Byte-for-byte identical output. Ships unconditionally, no knob. **The CPU-share-in-between half was deliberately deferred** — that is G9 | Qwen: 2.37 vs 0.63 ms per layer | +3–4%, not Qwen's 73%: GLM's per-device groups are 1–2 experts, so the three submits were already short | done | Sonnet | teacher_forcing identical; tok/s vs G0: **met** |
| G3 | **Done 2026-09-04** — per-op profile, record §G3: `[OPTIME]` timers (the `[ATTN]` timer never existed in the tree) + `perf` flat. Decode token 373 ms fresh-process: MoE 199 (CPU experts 115 at 27% of DRAM bandwidth, router 41 single-thread scalar, GPU groups 22, shared 20), KDA 78, MLA 55, mHC 35. **69% of the token runs on one core** (60.8% of cycles are libgomp spin). | supersedes the Sep 2 numbers: KDA is 21% of the token, not 37% | — | done | Opus | table in the record: **met** |
| **gate** | **Done** — Fable read G3: the roadmap's KDA description was wrong on three counts (inner loops already AVX2-vectorized; the "L=512 scalar loop" is MLA's; not bandwidth-bound at 15× its floor). Chosen: **CPU** — parallelise `coli_kda_step`, `expf(alog)` hoisted; **no shader**. The CPU-vs-shader call held on execution; the *sizing* did not (see G4: the decomposition missed 1.05 ms/call of GPU submits, and the win came from the conv loop, not the heads). | G3 §KDA decomposition: 0.9 ms DRAM + ~1 ms transcendentals + ~0.4 ms memmove = the measured 2.3 | — | done | Fable | `G4-KDA-SPEC-2026-09-04.md`: **written** |
| G4 | **Done 2026-09-04** (record §G4): three bit-identical commits took `[OPTIME] kda` 2.301 → 1.598 ms/call (78 → 54 ms/token) and the rotating median 1.69–1.71 → **1.84/1.83**. The ≤0.6 ms/call gate was **not** met and is unreachable this way: projections + `ko` are 1.05 ms/call of GPU submits, which G3 never attributed. Head parallelism was worth 2%; the conv-channel loop was worth 2.7×. Follow-up split out as G12. | G3 §KDA; procedure and stop condition in §"Executing G4" below, which is what caught the spec's error | — | done | Opus | rotating median: **met**; per-call gate: **not met, reason recorded** |
| G5 | **DONE 09-11 (record §G5)** — cached the pooled DSA-indexer block keys in a new `coli_sparse_index_select_range_cached` (`sparse_index.h`), ported the *idea* from Qwen `2d3cf7e` (a pool's mix depends only on its own members, never the query) into GLM's different pooling function (softmax mixture, not RoPE'd average), guarded the divisor the same way `df8ddc5` guarded Qwen's `ensure_kv` for C1. `GLayerState.pool_cache`/`pool_cache_count`, zeroed for free by `session_open`'s `calloc`, explicitly re-zeroed at the two places a session's content is overwritten from outside `mla_layer` (`ckpt_restore`, segment restore). `GLM53_NO_INDEX_CACHE=1` is the A/B, default on | the only O(context²)-total component; Qwen's fix cut its index phase 66% at 1.6k tokens | `index/ctx` constant **1.97→0.78 µs (2.52×) at ctx≈786, 2.07→0.87 µs (2.39×) at ctx≈1514** — a constant-factor cut, not a complexity-class change (the per-query score loop stays O(pools)/token, exactly as Qwen's own commit says it must); **10.3 ms/token at ctx≈786, 20.1 ms/token at ctx≈1514** | 1 day | Sonnet | **met**: `teacher_forcing`/`last_logits` bit-identical at 781 and 1509 tokens, cache ON and OFF; `[OPTIME] mla split` shows the constant drop directly; short-context (`rome_bench.sh`, ctx≈106) rotating median unchanged within noise (3.08/3.02 pristine vs 3.10/3.08 candidate) |
| G6 | **DONE 09-06.** The dev2/dev3 preload now stops on the VRAM budget as well as the count cap. dev0's loop always did this; dev2/dev3 stopped only on the cap or an allocation failure, and their allocations are HOST_VISIBLE and spill to host RAM over ReBAR — so they never fail at the VRAM limit, which is how a large cap put 91 GB "in VRAM". `coli_vk_mem_budget2/3` already existed and were simply never called. **Reserve is 1.0 GB on the expert-only devices, not dev0's 3.0**: trying 3.0 first fires at the standard 1695 cap and stops the tier at 1600, which would silently change routing and invalidate every recorded number — a safety guard must stop a runaway, not re-tune the tier (`COLI_VK_TIER_RESERVE_GB` overrides). Note an *unset* cap skips dev2/dev3 entirely rather than filling them, so this row's original gate text described a path that cannot happen | an unlimited cap put 91 GB in host RAM | safety, not speed | done | Sonnet | cap 1695: guard does not fire, 1695/1695, GLM output identical to pre-G6 — **met**; cap 2200: stops at 1752 on "25.0 of 25.7 GB used, 1.0 reserve", MemAvailable 242 GB, no spill — **met** |
| G7 | **DONE** 09-05 (record §G7) — parallelised the 288-row router score, inner (expert) loop not outer (token) loop: `score` has no per-token dimension, so parallelising over tokens would race. `[OPTIME] moe split: router` 0.998 → **0.137 ms/call** (7.3×). Rotating 1.84/1.83 → **2.01/2.05**. Bit-identical both prompts | G3: 41 ms/token, 0.98 ms/call, single thread, scalar reduction GCC will not vectorize; 1.2 GMAC/s | −36 ms/token; got exactly that | done | Sonnet | bit-identical: **met**; tok/s vs G4: **met** (+9–12%) |
| G8 | **DONE** 09-05 (record §G8) — parallelised both per-head loops. Checked nested-OMP oversubscription empirically before writing anything (a compiled probe: `max_active_levels=1` on this box, so nesting collapses safely). Per-thread `score`/`pooled` scratch, same class of fix as G4's KDA. `[OPTIME] mla` 5.013 → **2.066 ms/call** (2.43×). Rotating 2.01/2.05 → **2.19/2.17**. Bit-identical both prompts | G3: 55 ms/token at 151 tokens of context, 5.0 ms/call, single thread, **O(context)** (4.1 ms at 87 tokens) | −32.4 ms/token at ctx=88; correction in the record re: context-shape | done | Sonnet | bit-identical: **met**; tok/s vs G7: **met** (+6–9%) |
| G9 | **DONE** 09-05 (record §G9) — CPU-only experts now saved into a deferred list and run once in the issue/take gap (`can_defer = g_vk_ready && n_union <= block`; always true for decode). `[PROF]` eg/cpu went from serial (2.637s+13.612s=16.249s) to nested (eg=11.749s ⊇ cpu=11.193s) — the ~2.6s of pure GPU wait is almost entirely hidden inside CPU compute that was already larger than it. Rotating 2.19/2.17 → **2.33/2.35**. Bit-identical both prompts | G3: GPU groups 22 ms/token of pure wait; CPU experts 115 ms run *before* issue today | −22 ms/token; beat it (fresh-process +13.3%, gate +7–8%) | done | Sonnet | bit-identical: **met**; tok/s vs G8: **met** (+7–8%) |
| G10 | **DONE** 09-05 (record §G10) — parallelised the ~400k-MAC mix and the destination×column combine over independent rows/columns, each one's own reduction untouched (G7's pattern). **The malloc removal did not survive contact**: dropping `coli_hc_pre`'s two per-call `malloc`s needed `hc`'s range knowable at compile time, which measurably changed GCC's rounding in the small hc-bounded reductions (~1.9e-4 on `last_logits`, `teacher_forcing` unaffected) — bisected across five different compiler-flag mitigations, none restored bit-identity; **kept the mallocs, shipped the parallelism**. `[OPTIME] hc+norm` 0.393 → **0.097 ms/site** (4.05×). Rotating 2.33/2.35 → **2.60/2.75**. Bit-identical both prompts. Shared header (DeepSeek V4): rebuilt clean, not re-measured (no rig checkpoint) | G3: 35 ms/token, 0.39 ms/site × 90 sites, single thread | −28 ms/token; beat it (−26.6 ms/token measured, +11–17% gate) | done | Sonnet | bit-identical: **met** (mallocs kept); tok/s vs G9: **met** (+11–17%) |
| G11 | **PART 1 DONE** 09-05 (record §G11) — expert path fused: one OMP region with three worksharing constructs instead of three regions with a serial swiglu between, bit-identical, **−2.25 ms/token on `ffn_moe` (−3.2%)**, ~140 fewer OMP regions/token. **Part 2 (the kernel) is NOT worth doing as scoped** — see §RP1-CORRECTION. The bucket is **40.3 ms/token, not 74.8 and not 115**. Microbenchmarked before touching the engine (`g11_bench.c`, `g11_path.c`, both sized to defeat the 128 MB L3): the isolated kernel does 21.95 GB/s at 8 threads; a bit-trick nibble→float decode is *exactly* bit-identical but only 1.03×; a second accumulator breaks the FMA dependency chain for **1.30×** but is not bit-identical; four accumulators is worse than two. **Once the path is fused the faster kernel is worth only ~1.06×** — so nothing numerics-changing shipped | §RP1-CORRECTION: **40.3 ms/token, and no longer the largest bucket — KDA is, at 53.3** | part 1 delivered **−2.25 ms/token**; part 2 judged **not worth 2–3 days** for ~1.06× on a 40 ms bucket behind an env knob | part 1 done | Opus | bit-identical both prompts: **met**; `[PROF] cpu` −4.0% across 4 paired runs (fresh-process tok/s cannot resolve 1%: ±10% GPU-submit jitter) |
| G14 | **DONE 09-06 (record §G14).** Four alternative int4 expert kernels, each measured *in the engine* against a pristine binary. Only the float one (bit-trick decode + 4 accumulators) preserved the output: **1.095×** on the CPU expert path, greedy text identical over 128 tokens, `teacher_forcing` exact over 1232 positions, `last_logits` relL2 3.0e-6 short. Ships as `GLM53_I4_FAST`, off by default (not bit-identical; §G10 set that bar). The three quantised-activation variants (int8 1.32×, int16 1.075×, mixed) **all changed the greedy text** and were removed — removal itself worth 4% on the survivor, 64 KB of dead stack arrays in a function called ~70×/token. **Two findings worth more than the kernel:** a probe can size throughput on synthetic data but *not accuracy* (bench said relL2 3.9e-3, engine said 0.144); and isolated-to-in-engine attenuation is 0.55–0.65×, so in situ this path is **~57% memory / 43% arithmetic**, not the ALU-bound thing §G3's isolated figure suggests | §G3's "27% of bandwidth"; §RP3's wrongly-blocked bucket (item 4f) | arithmetic half now exhausted; **memory half is item 4h** | 3 commits | Opus | knob OFF bit-identical: **met**; knob ON greedy identity: **met**; serving gate: below floor, not run |
| G13 | **DONE 09-06 (record §G13).** Fused the shared expert's `mv(gate)`+`mv(up)` into one GPU submit via `coli_vk_matmul_pair` (already production code in `kimi_k3.c`). The routed-expert fused kernel was tried first and rejected: `qmatmul_gate_up.comp` computes `silu(gate)*up` with **no clamp**, and GLM-5.3's `swiglu_limit=10.0` is regularly exceeded — a pre-existing gap in the routed-expert GPU path, never caught before because no earlier change diffed a clamped and an unclamped computation of the same op; out of scope to fix here. `swiglu_clamped` and `down` stay unchanged. **3 submits/call → 2**, bit-identical (`--logits` exact on both G4 prompts, greedy text identical for 128 tokens). `[OPTIME] shared` 0.373 → **0.291 ms/call (−22%)**, ×42 = **−3.4 ms/token**, reproduced 3× within 1% | RP2: shared expert 14.2 ms/token in both knob positions, three round trips + a host round trip, called "round-trip-dominated" since §G3 | **−3.4 ms/token, delivered** | 1 commit | Sonnet | bit-identical: **met**; serving gate: **inconclusive at ~2.1% of the token (same class as G11 part 1), reported as such** |
| G12 | **DONE 09-05 (record §G12 stages 1, 2a, 2b, 2c).** The KDA recurrence, its gating and its output norm run on dev0, and a layer's projections + recurrence + `ko` record into **one submit instead of two** (`COLI_KDA_GPU=2`). **`[OPTIME] kda` 1.48–1.57 → 0.93 ms/call — the < 1.0 gate is met**, repeating to three decimals; **−18.7 to −21.9 ms/token**, inside the spec's own −15 to −22 band; fresh-process +13.5%; **serving gate +11.7%** rotating (2.90 vs 2.595, paired alternating) with warm-identical agreeing at +11.1%. **The gate also found a shipping bug the numerics oracles could not**: `coli_vk_kda_init` kept the previous conversation's recurrence across sessions, so request N+1 of a warm engine continued request N — visible only as an *inverted* warm-identical column, because every oracle here runs one request per process (record §Stage 2c; `tools/hot-expert/tworeq.py` is now the oracle for that class). Greedy text identical through 128 decode tokens and 0 `teacher_forcing` mismatches across 1260 prefill positions. **Stays off by default**: the logit cosine is 0.99992 at 1260 positions and the cause is irreducible — GLSL `exp()` vs libm `expf()`, ~1.1M calls/token; matching the CPU's norm-reduction order exactly was tested and changed nothing (2.6e-7) while costing 3.6% | §RP1-CORRECTION: KDA was the largest bucket at 53.3 ms/token, 35.0 of it GPU submits | **−18.7 to −21.9 ms/token, delivered** (opt-in) | 3 commits | Opus | `kda` < 1.0 ms/call: **met at 0.93**; greedy identity: **met** |
| G15 | **DEAD 09-11 — the probe killed it (record §G15).** Item 4h's numerics probe ran first, as the item requires, and GLM-5.3 does **not** survive int3 experts. `fmt=5` was simulated on the int4 weights on disk (`matmul_i4_sim3`, `GLM53_I3_SIM=1`) with every routed expert forced onto the CPU (`GLM53_EXPERTS_CPU=1`) so the ~79% the GPU tier serves could not mask it. Against the identically-placed int4 control: **13 of 42** short-prompt `teacher_forcing` predictions differ and **16 of 1232** long-prompt ones, `last_logits` cos **0.9726 / 0.8778**, greedy text differs, final argmax changes on the long prompt. §G12 shipped opt-in at cos 0.99992 with identical text and §G14 rejected int8 activations at cos 0.98964 — int3 is an order of magnitude past the rejection line. The transform was proved to BE int3 first (`rome_i3sim.c`: vs the tree's own `pack_int3_g64`+`matmul_i3`, relL2 1.7e-7) and the probe's double-quantisation pessimism priced at **1.08x**, so neither can explain it. **No converter, no shader, no fmt=5 kernel, and no speed measured** — the item says speed does not matter if accuracy fails. **The probe's by-product is the bigger result**: the routed-expert GPU kernel omits GLM-5.3's swiglu clamp (§G13's finding, never measured), and turning it on changes **6 of 42 / 8 of 1232** teacher-forced predictions and the long-prompt argmax — so the engine's output depends on which experts are tier-resident. Not in scope here; recorded as the next item | §G14: the bucket is bytes, not ALU | −13 to −16 ms/token, **not collected: the gate failed first** | probe only | Opus | knobs off bit-identical to pristine: **met**; greedy identity + `teacher_forcing`: **FAILED, item dead** |

Target for the track, rewritten from the profile: G3 measured 373 ms per
decode token fresh-process (585 ms rotating, G2's 1.71 tok/s). G4 and
G7–G10 are ~−195 ms of it with no new kernel and no numerics change; G11 is
the one kernel and the largest bucket. Order by ms-per-day: **G7 (hours),
G9, G4, G8, G10, then G11.** Expected roughly 1.8–2× on the rotating median
from G4+G7–G10 alone; the fresh-process split is the evidence, the rotating
median through `rome_bench.sh` is the gate for each. The KDA shader is off
the list until these are done (spec, last section).

### Executing G4 (read this, then the spec; nothing here needs re-deriving)

**Facts, verified 2026-09-04.** `c/delta_attention.h` is included by
`c/glm53.c` **only** — `kimi_k3.c` has its own `kda_forward`, `qwen36.c`
none — so despite its docstring it is not shared in practice; nothing else
compiles it and there is no sibling oracle to run. `glm53.c`'s only calls are
`coli_kda_step` (in `kda_layer`) and `coli_kda_scratch_floats` (session
open, `s->kda_scratch = malloc(...)`). `kda_state`/`kda_window` are
`calloc`'d once per session per KDA layer and never reset mid-session; they
are also exported as raw byte spans for segment migration
(`ColiSegmentStateSpan` table, grep `st->kda_window,`) — **which is why the
conv window keeps its `[channel][kernel]` layout and there is no ring
index: a position counter would change that span's meaning.** There is no
690-token document on the rig; construct the long prompt (below). The rig
has no `kimi_k3`/`qwen36` checkpoint and no torch, so "rebuild clean" is the
whole of what can be checked for them.

**Three commits, in this order, each built, oracled and timed before the
next.** Run everything with the G3 environment (record §G3): 3 GPUs,
`COLI_USAGE_PATH` pointing at a *copy* of `~/.glm53_explain.bin`,
`OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close`, cap 512.

1. **Hoist the constant.** In `kda_layer` (`glm53.c`, grep `expf(l->alog[h])`)
   the decay loop computes `expf(l->alog[h])` inside the `d` loop: 8,192
   calls per token for 64 distinct values. Compute it once per `h` in the
   outer loop into a local and use that. `alog` is a weight; same `expf` of
   the same input is the same bits. No struct change.
2. **Heads in parallel.** In `coli_kda_step` (`delta_attention.h`), put
   `#pragma omp parallel for schedule(static)` (guarded `#ifdef _OPENMP`) on
   the `for (int head ...)` loop. Every buffer that loop touches is disjoint
   per head **except `memory`**, which is one `v_dim` buffer shared by all
   heads — a shared `memory` under parallelism corrupts every head's
   read-back while still producing plausible tokens (`README.md` explains
   why that is the failure the `teacher_forcing` line exists for). Give each
   thread its own: `memory = scratch + 3*width + omp_get_thread_num()*v_dim`,
   and grow `coli_kda_scratch_floats()` from `mixed + v_dim` to
   `mixed + T*v_dim` with `T = omp_get_max_threads()` under `_OPENMP`, else
   1. `glm53`'s allocation follows automatically. Do not add `num_threads()`
   caps (measured and rejected on Qwen, record §"tried and rejected"). The
   arithmetic and its order inside a head do not change.
3. **Conv channels in parallel.** Same pragma on the
   `for (int channel = 0; channel < 3 * width; channel++)` loop above it;
   each channel's `window` slice and `mixed[channel]` are disjoint. The
   per-channel `memmove` stays (it is 12 bytes; spread eight ways it stops
   mattering) — this is what replaces the ring index.

**Oracle, per commit.** Build the pristine binary from the previous commit
into `/tmp` exactly as record §G2 does (`git show <sha>:c/glm53.c`, same
gcc line, `-I.`), run both with `--greedy 0 --logits 512` on the same
histogram copy, and `diff` the outputs: identical or the commit does not
land. Two prompts: the standard one from §G1, and a long one built as
`python3 -c "print((open('/home/ronald/bench/prompt_glm.txt').read().strip()+' ')*30)"`
(~700 tokens: many tokens through the recurrence and the conv window).

**Measure, per commit.** `COLI_TIMERS=1 ./glm53 --model ... --prompt "$P"
--greedy 128 512` and read `[OPTIME] kda ... ms/call`; put before/after in
the commit body. Expected: step 1 alone ~2.30 → ~2.15; step 2 to ~0.5;
step 3 to ~0.35–0.45. Gate ≤ 0.6.

**Gate, once.** `./tools/rome_bench.sh glm53 g4-run1` and `g4-run2`;
rotating median against G2's 1.69–1.71 in the record; both numbers in the
final commit body and one row in the record. Rebuild `kimi_k3` and `qwen36`
too and say in the body that they compiled (they cannot be exercised here).

**Stop conditions.** If any `diff` is non-empty, stop and report the first
divergent position — do not loosen the oracle to a tolerance, this change
has no legitimate numerics delta. If step 2 lands and `[OPTIME] kda` is not
below 0.8 ms/call, stop and report before step 3; that would mean the
decomposition in §G3 is wrong somewhere and the profile needs another look
(Opus), not more parallelism.

## Before Phase Q: the prefill track (added 2026-09-06, late)

The first interactive use of GLM-5.3 (Open WebUI, 2026-09-06) showed that
every gate on this roadmap measured decode on ~40-token prompts and none
measured time-to-first-token on a real prompt. That work has its own roadmap,
gate and record sections: **`PREFILL-ROADMAP-2026-09.md`** (rev 5) — P0
harness/oracle/gate landed, P1 answered (prefix reuse works on a real turn
2), P2 landed (bit-identical, 1.05–1.10× TTFT on the serve path), P4
(S-tiled shaders) next. Its gate is `prefill_gate.sh`; a prefill item is
done when that script exits 0. Phase Q below stands, but interactive use of
GLM-5.3 is gated on the prefill track, not on anything here.

## Phase Q: start here (rewritten 2026-09-11 from the Q-PROFILE; arbitrated the same day, record §Q-ARB)

**Baseline, re-taken 2026-09-11 (record §Q-REBASE).** `qwen38-vk`, cap 512,
max-new 80, 8 threads, tier asserted at **14 673 of 21 858** (dev0 24.4/25.7 GB),
binary `24a075ee4116326e`: **4.04 / 4.09 rotating, 5.23 / 5.25 warm-identical,
3.45 / 3.37 cold**, evicting, i.e. comparable to every historical row here.
The 09-04 / 09-06 / 09-09 rows are **superseded and historical only**. The
"provisional cold column" flag on the old §C2 baseline is **closed**: 3.45 /
3.37 reproduces 09-09's 3.38 / 3.44 and 09-04's 3.47.

**Track G's two weeks of shared-file kernel work are measurably neutral here** —
every column lands inside the harness's own noise — and §Q-REBASE says why in
code: the P4 tile pipelines require `S>1 && fmt∈{1,4}` and Qwen's experts are
`fmt=8` at `S=1`, `quant.h`'s changes are purely additive with `matmul_fp8`
untouched, P6b's KDA VRAM pool is never allocated by this engine, and RP4's
timestamps are off by default.

**The profile now exists: record §Q-PROFILE.** Decode token **191.2 ms**
fresh-process, three repeats agreeing to 0.5% on every bucket, accounting
residual 0.04 ms, and a `perf` flat profile agreeing with the timers to 0.7
percentage points. `tools/hot-expert/q_profile_run.sh` reproduces it. **Every Q
item below is re-derived from it; the six items that used to be here were
ordered by guesswork and two of them turned out to be measuring nothing.**

The three facts that reorganise the track:

1. **55% of the decode token (105.8 ms) is a BF16 weight stream running at
   82–88% of the machine's DRAM ceiling.** 6.81 GB/token through
   `dense-matmul` at **75.7 GB/s** plus 1.27 GB of LM head at **80.4 GB/s**.
   The ceiling is ~92 GB/s, not the 70.9 the profile text quotes: the same
   campaign's microbenchmark reaches **87.9–92.7 GB/s at 8 threads** with four
   accumulators, and the box does 91.6 at 16 threads. Q4 is that last
   10–14%; after it, **no CPU-side change touches these 106 ms** — only a
   different device (Q7's GPU arm) or fewer bytes (Q7's int8 arm) can.
2. **Only 36.3% of the token is single-threaded** (31.75% libgomp spin × 8/7),
   against **69%** for GLM-5.3 at §G3. The cheap parallelism wins that took
   track G from 1.65 to 2.75 tok/s are mostly already taken on this engine;
   what is left of them is ~26 ms/token, and it is in two places (§Q1, §Q2).
3. **The rotating median is a 247 ms token and the profile is a 191 ms
   token.** 4.04 tok/s is 247.5 ms; warm-identical 5.23 tok/s is 191.2 ms —
   the fresh-process profile's number to the decimal. The **56 ms/token**
   between them is expert-miss service on prompt rotation (slot-cache fills
   and shard reads; 15–35 ms of it survives in §Q-REBASE's no-evict rows), and
   it appears in no row of the profile table. **Q1–Q7 act on the 191 ms; only
   Q8 acts on the 56.** Every rotating estimate below is diluted by it — a
   −46 ms item is +32% on warm-identical and +23% on the rotating median.

**What transfers from track G, at no cost:**

| lesson | where it came from |
|---|---|
| Assert 100% page residency **before every run**, not once per campaign. **Qwen's constant is 94.3699%** (GLM's is 91.6347%), reproducible to four decimals | §RP2, §RP3, §Q-PROFILE |
| A probe can size **throughput** on synthetic data but **never accuracy** — the distribution is what carries the answer | §G14 |
| The serving gate cannot resolve under ~3–4%; `[OPTIME]` timers hold to ~1% (here: 0.5%). Say "inconclusive" rather than rounding | §G11 part 1, §G13, §Q-PROFILE |
| An oracle that runs **one request per process** cannot see session-lifetime state. `tworeq.py` is the check | §G12 stage 2c |
| A bucket is only "blocked" if what was *declined* is the same kind of thing as what is being *proposed* | item 4f, §G14 |
| Isolated kernel speedups attenuate ~0.55–0.65× in-engine when the path streams cold weights — **and ~0.95× when it streams resident dense tensors** (§Q-PROFILE: LM head 15.011 isolated vs 15.8 in-engine). Use the figure for the path in hand | §G14, §Q-PROFILE |
| Put a **sub-timer in first**: §G3's KDA decomposition missed 1.05 ms/call and only a sub-timer found it. §Q-PROFILE's within-bucket splits are arithmetic from shapes, so this is mandatory for Q1, Q2 and Q3 — **and the gate is re-derived from what it measures, before the candidate is built** (Q1's sub-timer halved its bucket and showed its ≤ 60 leg unreachable; the leg stayed and the row went to arbitration for nothing) | §G4, §Q1 |
| **A format is answered by a simulation on the weights already on disk, before any converter, shader or kernel exists** — and speed is not measured if accuracy fails | §G15 |
| Freeze the routing histogram per campaign or runs mutate what the next one preloads. **qwen38 reads `COLI_USAGE`, not `COLI_USAGE_PATH`**, and `rome_bench.sh` does **not** freeze it for qwen38 — every Q row so far mutated the next one's preload | §C0, §RP1, §Q-PROFILE |

**What does NOT transfer — now checked, not assumed:**

- **Scale — the old entry here was wrong and the numbers were already in the
  record.** It said Qwen's tier holds 14 673 experts against GLM's 4 686 and
  therefore "VRAM-bound is a GLM finding, not a Qwen one". **Qwen's tier is
  equally VRAM-saturated:** 14 673 × 4.92 MB ≈ 72 GB, which is all three cards
  (3 × 25.7 less the 1.5 GB per-device reserve `q38vk_expert_ensure` keeps),
  dev0 reports 24.4 of 25.7 GB, and the preload stops with **7 185 candidates
  left unplaced**. GLM's 4 686 × 14.16 MB + 3.78 GB dense ≈ 70 of 72 GB. The
  counts differ because a Qwen FP8 expert is **2.9× smaller** than a GLM int4
  one, not because Qwen has room. Both engines are VRAM-bound; Qwen just places
  **80.80%** of its routed activations against GLM's ~79%. **One thing that
  does follow from the larger count, and matters for Q7: the margin is cold.**
  The tier's coldest resident expert serves at most the resident mean
  (80.8% / 14 673 = 0.0055% of activations) and at least the unplaced mean
  (19.2% / 7 185 = 0.0027%), so a GB of the coldest experts is worth under
  1 ms/token of CPU-expert time — which is what makes the split in §Q-ARB a
  one-sided decision rather than a trade.
- **The clamp gap — checked, and it does not exist here.**
  `qmatmul_gate_up.comp` computes `silu(gate)*up` unclamped; Qwen's CPU expert
  path computes `q38_silu(eg)*eu`, also unclamped; and Qwen's `config.json` has
  **no `swiglu_limit` key at all** (GLM's is 10.0). The two paths compute the
  same function, so §G15's by-product — a model whose output depends on which
  experts happen to be tier-resident — **has no Qwen analogue** beyond ordinary
  FP accumulation order, which the 09-04 rows already measured at max-abs
  7.6e-6, cosine 1.0, same argmax. This is a correction, not an over-correction:
  it rests on the config file and on both code paths, not on an inference from
  GLM.
- **§SPEC-PROBE's answer — still does not transfer, and now has an
  instrument.** Its arithmetic rests on the share of expert activations that
  stream from RAM; Qwen's is 19.2%, not GLM's 21%, so it is not obviously
  different *today* — but §Q8 takes it to the never-routed tail (the ~1 600 to
  2 700 experts with no history at all), which fires SPEC-PROBE's own stated
  reversal condition in substance. The chunk probe is Q9's first step and the
  `[OPTIME] placement` counters added by §Q-PROFILE are what it reads.

## Track Q: Qwen3.8 — 4.04 / 4.09 rotating, 5.23 / 5.25 warm, 3.45 / 3.37 cold (record §Q-REBASE), 191.2 ms/token fresh-process (§Q-PROFILE)

**Rewritten 2026-09-11 against the profile and the microbenchmark, and
arbitrated the same day (§Q-ARB below, record §Q-ARB).** The previous six rows
are superseded: they were written before anything had measured this engine, the
roadmap said so, and the evidence contradicted three of them outright. The items
are **renumbered**, because four of them changed identity; the old→new map is:

| old | new | what happened |
|---|---|---|
| Q0 | — | **DEAD, both halves.** The tier is VRAM-limited, not history-limited |
| Q1 (fuse gr read/write) | **Q2**, re-scoped | the fusion is not where the time is; the *serial scalar* half is |
| Q2 (FP8 residual) | — | **DEAD.** The pass it targets is 0.18 ms/token, 0.1% |
| Q3 (dense off DRAM) | **Q7** | still real, still Fable — and now decided: two arms, one probe picks (§Q-ARB) |
| Q4 (MTP) | **Q9** | unchanged in substance, now gated behind Q8 |
| Q5 (4-acc BF16, "S>1 only, TTFT not decode") | **Q4** | **mis-scoped: it is 1.10–1.14× at S=1 too**, measured — and its rank now depends on QP |
| Q6 (int8/int4 experts) | **Q8** | re-sized (its number is the rotating/warm gap, not the CPU bucket), and it is the item that unlocks Q9 |
| — | **Q1, Q3, Q5** | new, straight out of the profile: ports of measured track-G patterns |
| — | **Q6** | the LM head — **folded into Q7** as its first tensor (see the row) |
| — | **QP** | new: the format-and-eviction probe day that decides the whole back half — **DONE 2026-09-12, and it deleted Q8 and Q7-cpu outright** |

Every item's "bucket" column is a **measured** `[OPTIME]` figure from
§Q-PROFILE. Where the item targets part of a bucket, the split is arithmetic
from the tensor shapes and is marked *(est.)* — **step 0 of any such item is a
sub-timer that replaces the estimate with a measurement**, which is what §G4
had to learn the hard way. **Added at Q1's arbitration (2026-09-12, record
§Q1): step 0's output includes the gate, re-derived from the measurement and
written into the record before the candidate is built — the roadmap's
absolute threshold is provisional until then.** Q1's ≤ 60 was derived from an
"~15.9 (est.)" that step 0 measured at 7.98, so the threshold sat below the
bucket's floor before a pragma was written; re-deriving a threshold from a
measurement taken before any candidate number exists is not tuning, and
keeping one the measurement has already shown unreachable is a formality
that sends the row to arbitration for nothing. The re-derived gate is stated
as a fraction of the step-0 bucket (Q1, Q3: at least half), not as a new
absolute. Q2's "~9.9 (est.)" is the first row this binds.

| id | item | bucket (measured) | evidence | expected | effort | tier | gate |
|---|---|---:|---|---|---|---|---|
| **Q3** | **DONE 2026-09-11, MERGED 2026-09-12 by Fable arbitration** (record §Q3 `### Arbitration`; was `perf/q3-shared-expert-into-gap`). The reorder works exactly as described and is byte-identical on the logits at 32 and 128 tokens; it is the *size* that missed its own estimate. `vk-take` **12.62 → 8.41 ms/token**, `moe` **55.90 → 51.75, −4.14**, the token **190.47 → 186.62 ms (−3.85)**, fresh-process +1.5%, serving warm-identical +0.9%, rotating 4.045 → 4.115 mean (inside the harness's own spread). Measured cause, in the numbers and not tuned away: only **54% of the shared expert is absorbed**, because the gap's length varies per layer and the layers with a large CPU share had already spent theirs. **The gate's two numeric legs as written (`vk-take` ≤ 8.0, `moe` ≥ −5.0) were not met — 8.41 / −4.14 — and the reason is recorded:** they were derived from the −5 to −7 estimate, which assumed the whole 7.2 ms is absorbed, i.e. that every layer's wait exceeds its own shared-expert cost; the mean wait does (0.263 vs 0.150 ms/layer), the distribution does not, and the quantity a reorder can claim is `Σ_l min(shared_l, wait_l)` = 3.98 ms. A 5.0 threshold sat above that ceiling before the first run — G4's case (per-call gate unreachable by the mechanism, merged with the reason recorded). **Corrected gate, as it should have read: bit-identical; `moe` down ≥ half the shared bucket (≥ 3.6 ms/token); `Δvk-take` = `Δmoe` within the repeat spread — met at −4.14 / −4.21.** Two sizing corrections carried downstream: the idle left in the gap is **8.4 ms/token (`vk-take` after), not 3.4** — 3.4 is the shared-expert work that *spilled out* of the gap in the layers whose wait was already spent, and the two live in different layers; and that 8.4 is unevenly distributed, so added CPU-side work (Q5's slack, §Q-ARB's cold-expert eviction) should be priced as absorbed at ~54%, not 100%. Q7-gpu (more GPU work) spends that idle in the slack layers and lengthens the GPU side in the CPU-long ones; Q8 (shorter GPU side) shrinks it; neither may count the 8.4 as free in a layer that is already CPU-long. Original row follows. **The shared expert into the GPU gap.** In `q38_moe_decode` the order was router → shared expert (CPU, 8 threads) → issue the routed groups → CPU share in the gap → take. The shared expert is pure CPU work sitting **before** the issue, so none of it overlaps the GPU. Move the issue above it. §G9 on a different engine, and the highest ms-per-day on the board | vk-take **13.0 ms/token (6.8%) of pure idle wait**; shared expert **7.2 ms/token** on the wrong side of the issue | §G9 hid GLM's 22 ms/token of GPU wait inside CPU work that was already larger; here the CPU work exists and is simply ordered wrong. The gain is bounded by the smaller of the two figures, 7.2 | **−5 to −7 ms/token** (measured **−3.85**; the estimate assumed full absorption) | half a day | Sonnet (Fable for the gate call) | **bit-identical** (`ys` accumulation order unchanged): **met**; `[OPTIME] vk-take` ≤ 8 / `moe` ≥ −5 as written: **not met (8.41 / −4.14), reason recorded**; corrected gate `moe` ≥ −3.6 with `Δvk-take` = `Δmoe`: **met — merged** |
| **Q1** | **DONE 2026-09-12, MERGED 2026-09-12 by Fable arbitration** (record §Q1 `### Arbitration`; was `perf/q1-deltanet-serial-tail`). **Corrected gate, as it should have read once step 0 had measured the bucket: bit-identical; the three loops the item names (`dn-conv` + `dn-qknorm` + `dn-gnorm`) down by at least half their step-0 figure (≤ 4.0 ms/token), equivalently `[OPTIME] deltanet` ≤ 66.3 (the 62.33 the item cannot touch, plus 4.0); untouched sub-buckets flat; the token down by the bucket's delta within spread — met at 1.56 / 63.87 / `dn-recur` −0.10 / −6.02 against −6.35.** This is **not** §Q3's finding, and the two are filed apart on purpose: Q3's target was measured right and its *mechanism's yield model* was wrong (a reorder claims `Σ min`, 54%); Q1's mechanism delivered above its evidence (5× where §G4/§G8 said 2.5×) and its *target* was mis-measured (an arithmetic subtraction had swept the already-parallel recurrence into the serial column). The procedural fix — step 0 re-derives the gate before the candidate is built — is now the rule in the "(est.)" paragraph above this table, and binds Q2. The `dn-recur` finding below is logged as **Q10**, with its sizing corrected there (the state fits L2; the loop is more likely latency-bound than bandwidth-bound). The rotating pristine at 4.19 against §Q-REBASE's 4.04/4.09 is the disk-service row moving on unchanged code (warm-identical agrees with §Q3's to 0.3%); §Q-REBASE's rows are pre-Q3 and are no longer any item's rotating pristine — each campaign carries its own. Executing agent's row follows. **The gate's `[OPTIME] deltanet ≤ 60` leg is not met and is unreachable by this item's mechanism** (record §Q1). The change works and is bit-identical: `deltanet` **70.22 → 63.87 (−6.35)**, the token **185.86 → 179.84 (−6.02)**, last-token logits byte-identical (993 280 bytes) at 32 tokens with timers off and on and at 128 tokens, rotating **4.19 → 4.27 (+1.9%)** and warm-identical **5.315 → 5.495 (+3.4%, = −6.2 ms/token, agreeing with the fresh-process figure)**. **What missed, and why it was always going to:** `[OPTIME] deltanet` is **63.87**, and the parts of `deltanet` this item does not touch measure `dn-proj` **52.84** + `dn-recur` **9.19** + `dn-rest` 0.30 = **62.33 ms/token**, so `deltanet` cannot reach 60 by parallelising these three loops even if they cost nothing. Both the ≤ 60 threshold and the −8 to −13 expectation came from §Q-PROFILE's **estimate** of ~15.9 ms/token of serial scalar, and **step 0 — which this gate itself mandates — measured that bucket at 7.98**: `dn-conv` 5.97, `dn-qknorm` 0.39, `dn-gnorm` 1.62, while **`dn-recur` 9.29 of the "estimate" is the 48-head recurrence, which already had a `parallel for`**, and `dn-proj` is 52.26 rather than the 55.0 the bandwidth arithmetic assumed. Of the 7.98 that existed, the change took **80%** (7.98 → 1.56: conv 5.9×, gnorm 4.3×, qknorm 2.3×). That is §G4's case and §Q3's case a third time — a per-op threshold set from an arithmetic estimate and found unreachable by the measurement the item was required to take first — and it is **reported, not tuned: no threshold here was edited to fit the numbers.** Three things it leaves behind for the rest of the track: (i) **`deltanet`'s floor is ~62 ms/token** until the dense projections leave the CPU (`dn-proj` 4.16 GB at 78.7 GB/s — Q7's bucket); (ii) **`dn-recur` is at the MEMORY ceiling, not the thread ceiling** — 680 MB/token of recurrent state in four traversals = 74 GB/s against this box's 70.9 — and **folding the `state *= alpha` decay into the `k·state` traversal removes a third of that traffic (~−2.5 ms/token) bit-identically**, which is a **new item, not Q1's scope**; (iii) in-engine attenuation on a serial scalar loop is **inverse** (7.98 in-engine vs 3.29 isolated, because the lone loop runs beside seven spinning libgomp threads), so **§Q2 should beat its isolated ratio** — and §Q2's bucket is the same arithmetic-subtraction estimate, so its step 0 may re-size it in either direction. Rejected and deleted, with the measurement that rejected it: the **literal four-`parallel for` port** (1.755/1.584 ms-token isolated against the fused region's 1.645/1.372, and 144 fork/joins per token instead of 36 at 5.3 µs each). Original row follows. **DeltaNet's serial tail.** Parallelise the `CD=10 240`-channel causal conv (`for d < CD`: one silu and a 3-tap shift each, 36 layers = **368 640 serial iterations/token**), the per-head q/k normalisation, and the per-head output RMSNorm. Only the 48-head recurrence has a `parallel for` today. This is §G4 step 3 + §G8 on a different engine. **Check every buffer the parallelised loop touches for sharing before adding the pragma** — §G4's `memory` was one shared `v_dim` buffer and it corrupts silently (plausible tokens, wrong `teacher_forcing`) | deltanet **70.5 ms/token (36.9%)**, of which **~15.9 (est.)** is serial scalar and ~55.0 is its own bandwidth-bound projections | §G4 step 3 got **2.7×** on GLM's identical conv-channel loop; §G8 got 2.43× on GLM's per-head loops. Channel slices and head slices are disjoint; no reduction changes order | **−8 to −13 ms/token** (−4 to −7% of the token) | 1–2 days | Sonnet | sub-timer first (conv / norms / projections split); **bit-identical** greedy text + last-token logits; `[OPTIME] deltanet` ≤ 60 ms/token; rotating median vs 4.04 / 4.09 — **as written: bit-identical met; ≤ 60 not met (63.87) and unreachable (floor 62.33); corrected gate (touched loops ≤ 4.0, `deltanet` ≤ 66.3): met at 1.56 / 63.87 — merged** |
| **Q2** | **DONE 2026-09-12, MERGED 2026-09-12** (record §Q2 and §Q2 step 0; was `perf/q2-gated-residual-serial-tail`, merge commit on `hot-expert-tier`). **The first row the step-0 rule above bound, and it worked as intended: the gate was re-derived from the measurement before the candidate was built (commit `c11ec82`, two commits before the first candidate number) and every leg of the corrected gate was then met, so this row did NOT go to arbitration.** Measured: `gr-read` **26.86 → 21.79 (−5.08)**, `gr-mix` **6.56 → 1.43 (4.6×)**, the token **179.49 → 174.18 (−5.31)**, last-token logits byte-identical (993 280 bytes) at 32 tokens with timers off and on and at 128 tokens, fresh-process 4.37 → 4.51 tok/s (+3.1%), TTFT 4.10 → 3.95 s, rotating **4.28 → 4.41 (+3.0%, both candidate runs above both pristine runs)** and warm-identical **5.52 → 5.655 (+2.4% = −4.32 ms/token)**. **Step 0 re-sized the bucket from the roadmap's "~9.9 (est.)" to 7.62** (`gr-rms` 1.06 + `gr-mix` 6.56) — a different error from Q1's: the two matmuls measure **18.2**, not the 16.9 the bandwidth arithmetic assumed, and 0.90 of the remainder is serial scalar work this item does not name (`gr-lowsilu` 0.24, `gr-rest` 0.66 — and §G10 forbids touching the allocations that make up `gr-rest`). **`gr-read ≤ 20` is NOT MET (21.79) and was shown unreachable BEFORE the candidate existed**: 19.11 ms/token of the bucket is untouched work and the two named loops cannot go below 0.27 + 0.82 at four and eight threads, so the floor is 20.20 with zero fork/join overhead. Corrected gate, stated at step 0 and met: **bit-identical; `gr-rms` + `gr-mix` ≤ 3.81 (half the step-0 bucket) — met at 2.49; equivalently `gr-read` ≤ 22.9 — met at 21.79; untouched sub-buckets flat — met (max +0.14); the token down by the bucket's delta within spread — met at −5.31 against −5.08.** The expected delta was re-derived at step 0 as **−5.3 to −5.7 with a −6.5 mechanism ceiling**, i.e. the roadmap's −5 to −8 at its low end only; measured −5.31. **Rejected and deleted, with the measurement that rejected it: the rms half — the loop this row's own text names first.** Parallelising the four `q38_rms0` calls over `b` takes `gr-rms` 1.07 → 0.82 and pushes **`gr-apply` 0.18 → 0.53** (reproducible to ±0.01 over three repeats in two campaigns, isolated with a throwaway `Q38_GR_RMS_PAR` knob): the four threads pull all 40 KB of `hyper` into four private L2s and the serial `q38_gr_apply` then writes all of it from one, paying cross-CCX invalidation on this 4-CCX part. Net on the token **174.65 vs 174.76 — zero within the spread**, so it does not ship. **Three things for the rest of the track:** (i) **`gr-read`'s floor is ~19.1**, of which 17.8 is the two 10240×320 matmuls (1.28 GB/token) — Q7's bucket, and the CPU-side mix a GPU arm must absorb with that pair is now 1.43 rather than 6.56, which makes Q7's fused-pair variant **more** attractive; (ii) **§Q1's "§Q2 should beat its own isolated ratio" did NOT hold** — on this op the isolated serial figures are *higher* than the engine's (1.52 vs 1.06, 7.42 vs 6.56), so inverse attenuation is a property of Q1's loops and not a rule; (iii) a new general gate requirement, now in Q10's and Q11's rows: **an item that parallelises a loop must report the buckets that neighbour the buffers it touches**, because `gr-read` alone would have scored the rejected arm at −5.34 and called it the winner. The follow-up is logged as **Q11**. Original row follows. **Read Q1's row first (2026-09-12, record §Q1): this row's "~9.9 (est.)" is the same arithmetic subtraction that Q1's "~15.9 (est.)" was, and Q1's mandatory step-0 sub-timer found that one to be 7.98 — half of it, because the subtraction had swept an already-parallel loop into the serial column. Take the sub-timer FIRST and re-derive the −5 to −8 and the ≤ 20 from it; do not build against these numbers.** Two corrections from Q1 that run the other way: in-engine attenuation on a serial scalar loop is **inverse** (7.98 in-engine vs 3.29 isolated — a lone scalar loop runs beside seven spinning libgomp threads, §RP1), so an isolated serial-vs-parallel ratio understates the win; and the right shape is **one `omp parallel` region with several `omp for`s**, not one `parallel for` per loop (measured: 5.3 µs per region, and Q1's four-pragma arm was slower in both cache regimes). Original row follows. **The gated residual's serial tail.** Parallelise `q38_gr_read`'s four `q38_rms0` calls (a `double` accumulator over H=2560, one per hyper-column `b` — parallelise **over `b`**, four ways, so each reduction keeps its order) and its `C=4 × H=2560` sigmoid mix (parallel over `d`; the inner `for b<C` sum stays serial per `d`) — together **≈1M `expf`/token, single-threaded, at 97 sites**. This is §G10's mHC item in a different engine | gr-read **26.7 ms/token (14.0%)**, of which **~9.9 (est.)** is serial scalar and ~16.9 is its two 10240×320 matmuls | §G10 got **4.05×** on GLM's structurally identical `coli_hc_pre`. `perf` puts `step.constprop.0` + `q38_gr_read.constprop.0` + `__expf_fma` at 1.85% of *cycles* against ~5% of *wall* — the signature of serial code | **−5 to −8 ms/token** (the rms half is only four-way parallel; do not promise 8×) | 1 day | Sonnet | sub-timer first: **done, and it re-sized the bucket to 7.62 and re-derived the gate before the candidate (record §Q2 step 0)**; **bit-identical** — **met** (byte-identical at 32 with timers off and on and at 128, plus the harness oracle); `[OPTIME] gr-read` ≤ 20 ms/token **as written: NOT MET (21.79) and unreachable by this mechanism (floor 20.20)**; corrected gate (`gr-rms` + `gr-mix` ≤ 3.81, `gr-read` ≤ 22.9, untouched flat, token ≥ the bucket's delta): **met at 2.49 / 21.79 / +0.14 max / −5.31 — merged, no arbitration needed** |
| **QP** | **DONE 2026-09-12, MERGED 2026-09-12** (record §QP; was `perf/qp-format-probe`). **Every gate leg met, and three of the four answers are NOs — which is what the item was for.** Knobs off: last-token logits **BIT-IDENTICAL** to pristine (993 280 bytes by `cmp`) and greedy text identical, on the 30-token prompt and the 1200-token one. Transform oracle first, per §G15: the int4 sim's levels differ from the tree's own `quantize_i4_grouped` + `matmul_i4_grouped` in **0 of 1 638 400** (output relL2 **0.000e+00** — the same bits, not reassociation-close) and from the **converter's own** `quant_int4_grouped(gs=64)` in **0 of 655 360**; the int8 packer differs from quant.h's own `quantize_rows(bits=8)` in **0 of 5 242 880** and from the converter's `quant_int8(8)` in **0 of 655 360**; the level map is exhaustive over 256 fp8 bytes × 256 group-absmax bytes × 4 scales with 0 disagreements. **(a) int4-g64 experts: REJECTED** — against an identically-placed control that is reassociation-clean (cos 1.000000000, relL2 4.9e−7, greedy identical, so Qwen has **no** §G15-style clamp divergence between its CPU and GPU expert paths): **3 of 30 short-prompt `teacher_forcing` changes, 319 of 1200 long-prompt, cos 0.926429**, greedy diverges. **Q8 dead.** **(b) int8 dense: REJECTED both groupings** — per-row 1 of 30 / 272 of 1200 / **cos 0.980191**, g64 2 of 30 / 235 of 1200 / **cos 0.982097**, both below §G14's own rejected 0.98964, and this is the weights-only f32-activation best case. **Q7-cpu dead, Q7-gpu's int8 `fmt` dead, Q4 first.** **(c) int8 LM head: fails the gate as written on BOTH arms, by one leg each, and is NOT rounded** — per-row 0 of 30 short TF and cos 0.999970 but the 128-token greedy diverges; g64 cos 0.999987 (tighter than §G12's 0.99992, which shipped opt-in) with identical greedy text but 1 of 30 short TF, decided on n=30 where the n=1200 leg passes both. **Handed to Fable with both branches priced; a session does not get to soften a kill line.** **(d) the eviction curve: measured**, and it confirmed §Q-ARB's ledger to 0.2% (13 586 experts placed at 5.4 GB against a predicted 13 570, 1 087 evicted against ~1 100). Warm-identical **+1.77 / +2.91 / +4.06 ms/token** at 3.4 / 5.4 / 6.8 GB (0.60 per GB, ~0.27 per evicted expert, monotonic, ≤0.97 ms within a level); rotating **+7.9 / +11.8 / +10.2** but its own N=0 pair spans 5.12 ms and it is non-monotonic, so **the warm-identical row is the one that resolves this** and the +8-to-+22 bound was 3–5× pessimistic on it. **§Q-ARB's gated-residual condition is MET** (6.8 vs 5.4 = +1.16 ms warm / −1.68 rotating, criterion 10 ms), so Q7-gpu's set is the whole 6.8 GB dense stream at a measured 4.06 ms/token of VRAM cost. **One new item out of it, Q12:** the same per-tensor int8 perturbation is fine on ONE tensor (the head, logit relL2 7.7e−3, ~2× the weight relL2 of 3.9e−3) and fatal across **48 layers** (0.198, a ~50× amplification that is entirely **depth**) — so "int8 the dense set" and "int8 a dense tensor" are different questions and QP answered only the first. **Shared file:** QP(d) needed `coli_vk_ballast_gb` in `c/backend_vulkan.c`, so glm53 was rebuilt and re-checked — `teacher_forcing`, `last_logits`, the 128-token continuation and the mmap/hit counters all **identical**; only its decode timing line differs (6.115 vs 6.034 tok/s, one fresh-process run each), and the reason glm53 was not re-baselined in the serving regime is stated in the record rather than left to be discovered. **Two of my own oracles were wrong before any verdict was believed, and both are recorded:** the greedy comparison was reading only the first line of a multi-line continuation (it called two divergent 128-token outputs identical), and the glm53 check hashed an absent line and would have reported IDENTICAL for nothing. Original row follows. **The probe day — three formats and one eviction, simulated before anything is built.** (a) **Experts fp8→int4 g64** (Q8's life or death), simulated in the CPU expert path on the FP8 weights already on disk with **every routed expert forced onto the CPU** so the tier cannot mask it — `GLM53_I3_SIM`/`GLM53_EXPERTS_CPU` are the pattern; Qwen has neither knob and needs both (`Q38_I4_SIM`, `Q38_EXPERTS_CPU`). Unlike §G15 the double quantisation here is **not** a pessimism: the converter's own source is FP8, so the sim *is* the converter. (b) **Dense BF16→int8** (Q7's format arm), per-row and g64, simulated on the resident BF16 tensors at load. (c) **The LM head →int8**, same sim, reported separately — the head may survive where the projections do not, or vice versa. (d) **The eviction probe**: a `Q38_VK_BALLAST_GB` knob that allocates N GB of idle VRAM on dev0 **before** the preload (the engine's own `q38vk_expert_ensure` then places 1.5 GB-reserve-first, hottest-first, and the coldest experts fall off dev3 on their own — no placement code), run at N = 3.4, 5.4 and 6.8 through `rome_bench.sh`: **that is Q7-gpu's VRAM cost as a rotating-median number, measured with zero engine code** | — | §G15: GLM-5.3 did not survive int3 (16 of 1232 TF changes, cos 0.878); §G14: int8 *activations* changed the text at cos 0.990; §G12 shipped opt-in at cos 0.99992 with identical text. Whether Qwen's experts take int4 and its dense weights take int8 has **never been asked**, and every item after Q5 depends on the answer | four answers; the sims cost nothing in tok/s and are not timed | Opus 1–2 days for (a)–(c) (one harness, three sims); Sonnet half a day for the two knobs and the ballast; Haiku/`rome-bench` for the ballast rows | Opus (+ Sonnet knobs) | knobs off **bit-identical** to pristine; each sim against the **identically-placed** control (`EXPERTS_CPU` both sides): `teacher_forcing` over the short prompt and ≥1k positions, greedy identity over 128 tokens, `last_logits` cosine — **kill line: any short-prompt TF change or long-prompt cos < 0.99 (§G14/§G15's rejection class)**; the ballast rows: three rotating medians and the placement % in the record |
| **Q5** | **Fuse the CPU expert's three matmuls into one OpenMP region.** `q38_moe_decode`'s CPU share calls `q38_weight_matmul` three times per expert — **3 fork/joins × 92 CPU experts = 276 OMP regions per token**. This is §G11 part 1, which was bit-identical on GLM. **Re-scoped from the draft: it is worth ~0 on the fresh-process token today**, because the CPU share runs *inside* the GPU gap and the GPU is the longer side (13.0 ms of idle remained after it before Q3; **8.4 ms after it**, measured as `vk-take` — record §Q3 — but unevenly distributed across layers: Q3's uniform 0.15 ms/layer of added CPU work was absorbed at only 54%, which makes this row's "worth ~0 on the token" firmer, not weaker); making the hidden side faster moves the counter, not the token. Its value is (i) slack in the gap, which Q7-gpu's eviction spends, and (ii) the rotating regime, where more experts miss the tier and the CPU side can be the longer one | cpu-experts **15.4 ms/token (8.1%)**, hidden | `rome_cpubench.c`: the same ten experts cost **1.602 ms/token-layer through the engine's loop and 1.387 fused — 1.16×**. Isolated and in-engine already agree to 4% (0.16 vs 0.167 ms/expert), so this is not an attenuation gamble | **−2.1 ms/token on the counter; 0 to −2 on the token** | half a day | Sonnet | **bit-identical** (§G11 part 1 was); `[OPTIME] cpu-experts` ≤ 13.5 ms/token; **report `step` and `vk-take` beside it and say plainly if the token did not move** — the counter is the gate, the token is the honest footnote |
| **Q4** | **DONE 2026-09-12, MERGED 2026-09-12** (record §Q4; was `perf/q4-bf16-four-accumulators`). **Every leg of the gate met, and the expected delta landed near the middle of its range.** One engine file, `c/qwen38_core.h` — `glm53.c` does not include it, so glm53 was NOT rebuilt or re-measured (unlike §QP, which touched a shared file). Knob `Q38_BF16_ACC4`, default **0**. **Knob OFF: BIT-IDENTICAL** to pristine — last-token logits byte-equal (993 280 B by `cmp`) and greedy text identical on the 30-token prompt AND the 1 200-position one, `teacher_forcing` 30/30 and **1200/1200**. **Knob ON: greedy text IDENTICAL over 128 tokens after BOTH prompts**, short-prompt `teacher_forcing` clean 30/30, last-token cosine **1.000000000** (relL2 2.18e−6) short and **0.999998812** (relL2 1.54e−3) long, argmax unmoved in both — **but 3 of 1200 long-prompt `teacher_forcing` predictions differ, and that is stated here rather than buried**: it is reassociation accumulating through 1 200 positions of KV cache (§G14 measured 36× more of it, relL2 0.055, and shipped), it is not run-to-run noise (the knob-off pair is byte-identical across two binaries), and the transform oracle measures the candidate kernel as **1.29–1.94× CLOSER to exact long-double arithmetic than the pristine one**, so at those three positions the pristine and a strictly more accurate computation disagree. **Transform oracle first** (`tools/hot-expert/rome_q4acc.c`, §G15's rule): over exact-arithmetic data the two kernels are **bit-identical on 1 332 dots across 36 inner dimensions** (the engine's own plus multiples of 8-not-32, of neither, I<32, I<8) and both equal the exact sum, **0 mismatches**, plus 2 048 more at S=32 and a negative control that fires. **Isolated** (`rome_cpubench.c`, 8 threads, pools > L3 and > Infinity Cache; the gated residual's 10240×320 and 320×10240 added for this item): **1.06–1.17× at S=1** on every dense shape including the short-I pair, **3.07–3.24× at S=32**. **In-engine:** `[OPTIME] dense-matmul` **89.06 → 80.56 (−8.50)**, `lm-head` **15.45 → 13.93 (−1.53)**, bucket **−10.02**, `step` **175.60 → 165.80 (−9.79)**, every untouched sub-bucket flat (dn-conv +0.02, gr-mix −0.00, cpu-experts +0.09, vk-issue +0.02) and `vk-take` **+0.30** in the expected direction; **all nine profile runs share the identical expert hit rate (49.5 %, hit=16364 miss=16699)**, so the timing pair is controlled in §G14's sense. **Attenuation is ~1.00×, not §Q-PROFILE's 0.95× and nothing like §G14's 0.55–0.65×.** **Serving A/B** (`rome_bench.sh`, interleaved, two repeats): rotating **4.41 / 4.38 → 4.54 / 4.55** (+3.4 %, both candidate runs above both pristine), warm-identical **5.63 / 5.62 → 5.96 / 5.96** = **−9.99 ms/token**, agreeing with the counter to 0.3 ms. **Free extra:** the 30-token prefill forward **4 318 → 2 949 ms (1.46×)**, which is the old Q5 prefill claim measured rather than extrapolated. **ONE THING LEFT FOR FABLE, NOT DECIDED HERE: whether the default should flip.** The knob ships off, so the −10 ms/token is not in the default binary; the case for turning it on is stronger than anything this record has shipped opt-in, and the case against is the 3 of 1 200 — a decision about the engine's default output, not one the session that wrote the kernel should take. Original row follows. **RANK DECIDED 2026-09-12: QP(b) died, so this row's own conditional fires — "if int8 dies, Q4 is the only CPU-side gain left on the 106 ms and goes first". It is now the first item on the track** (record §QP (b)). It also inherits the whole 105.8 ms/token rather than sharing it: Q7-cpu is dead, so nothing else is going to take these bytes on the CPU, and the "not additive with Q7" caveat now means only "not additive with Q7-**gpu**, which moves the same bytes to dev0". Original row follows. **Four accumulators in the BF16 dense GEMV at S=1** (was Q5, which said "S>1 only, TTFT not decode" — **measurement says otherwise**). Both kernels need it: `q38_matmul_bf16` and the inlined loop in `q38_dense_matmul_multi`, 30.49% and 24.91% of user cycles. Breaks the single FMA dependency chain; **changes the summation order**, so it ships behind a knob with the logit diff recorded. **Its rank is decided by QP(b):** if int8 dense survives, Q7-cpu replaces this kernel on the tensors that matter and Q4 becomes what it was originally — a prefill item (1.8× at S=32) — and moves behind Q7; if int8 dies, Q4 is the only CPU-side gain left on the 106 ms and goes first | `dense-matmul` **90.0** + lm-head **15.8** = **105.8 ms/token (55%)** | `rome_cpubench.c`, 8 threads, pools > L3, this engine's exact shapes: **79.6→90.8, 81.8→90.2, 80.7→87.9, 84.7→92.7 GB/s — 1.10–1.14× at S=1**. In-engine attenuation on *this* path is ~0.95×, **not** §G14's 0.55–0.65× | **−8 to −12 ms/token** on today's kernel; **not additive with Q7** — both act on the same bytes | 1 day | Opus | knob **off** bit-identical to pristine; knob **on**: greedy text identical over 128 tokens and the last-token logit cosine in the commit body; `[OPTIME] dense-matmul` and `lm-head`; rotating median |
| **Q6** | **ARBITRATED 2026-09-12 (Fable, record §QP `### Q6 arbitration`): REJECTED on both arms — the head goes to dev0 as BF16 (1.27 GB) as step 3 of `Q7-DENSE-GPU-SPEC-2026-09-12.md`, and no more data is requested.** The decision rests on the size of the prize, not only on the letter of the gate: the int8 head's only surviving value is 0.64 GB of VRAM (0.38 ms/token at §QP (d)'s 0.60/GB) plus ~1.6 ms of dev0 streaming — **~2 ms/token, ~1 %**, below what the serving harness can resolve — because the CPU-side −9 ms is dominated outright by the GPU BF16 head (~3–6 ms against 15.8, no numerics change) and is therefore not a live option whatever QP(c) says. Against that, g64's 1-of-30 short-prompt change is **not** a sample-size artefact: the n = 1200 leg measured **3 changes (0.25 %)** on the same arm, so the format has a real, nonzero prediction-change rate and the short leg merely sampled it; per-row's 8 of 1200 and divergent greedy text are worse. And CLAUDE.md would ship it opt-in regardless — a 1 % knob nobody turns on is dead code. Option 3 (a larger n) is declined because no sample size changes the value side. **Two rulings for later rows:** (i) the kill line stands as written — one short-prompt TF change with a nonzero long-prompt rate is a measured rate; one change with **0 of ≥ 1000** on the long leg is the near-tie case and goes to arbitration with the top-2 margin at that position, which is the number that decides; (ii) this row's "if the head fails int8 it has no GPU form worth building" is withdrawn — the BF16 `fmt` is being built for the dense set anyway, so the head rides on it at zero kernel cost. Executing agent's row follows. **QP(c) ANSWERED 2026-09-12 and the answer is "no int8 form clears the gate as written — and that call belongs to Fable, not to a session" (record §QP (c)).** This row's own condition was "its only formats are the ones QP(c) answers; if the head fails int8 it has no GPU form worth building". Measured, against an identically-configured control: **int8 per-row — 0 of 30 short `teacher_forcing` changes and long cos 0.999970, but the 128-token greedy continuation DIVERGES; int8 g64 — cos 0.999987 (tighter than §G12's 0.99992, which this track shipped opt-in) and the greedy continuation IDENTICAL, but 1 of 30 short predictions changes.** Each arm fails exactly one leg, neither is rounded, and the short-prompt leg that rejected g64 is deciding on **n = 30** where the long-prompt leg (n = 1200, 3 changes = 0.25 %) passes both arms. **So: on the gate as written the head has no int8 form, it goes to dev0 as BF16 at 1.27 GB rather than 0.64, and the CPU-side −9 ms/token below is unavailable. If Fable relaxes the short-prompt leg for the g64 arm only, both come back.** Priced both ways in §QP; the decision is explicitly not taken here. Original row follows. **The LM head — folded into Q7.** It is one tensor of one shape through one kernel, which makes it the cleanest first A/B for whichever arm Q7 takes: the first int8 tensor on the CPU (1.27 → 0.64 GB at 93.6 GB/s: **−9 ms/token**) or the first dense tensor on dev0 (int8 1.58 ms at 403 GB/s in the 09-04 matrix: **−13 ms/token**, 0.64 GB of VRAM). It is not a separate item because its only formats are the ones QP(c) answers; if the head fails int8 it has no GPU form worth building | lm-head **15.8 ms/token (8.2%)**, 1.27 GB at **80.4 GB/s** | 09-04 placement matrix; today's isolated re-run 15.0 ms CPU at 8 threads | inside Q7's figure | inside Q7 | Opus | inside Q7's gates; QP(c) first |
| **Q7** | **STEP 0 DONE AND PASSED, STEP 1 BUILT AND AWAITING ARBITRATION, STEP 4 REFUSED ON ITS OWN STEP-0 NUMBER, STEPS 2–3 NOT STARTED (2026-09-12, Opus; record §Q7; branch `perf/q7-gpu-dense-stream`, NOT merged).** **Step 0 passed every leg.** `fmt=9` BF16 in `qmatmul.comp`: layout **bit-exact at 7 of 7 shapes** against the engine kernel on order-independent integer data (including an odd-I/odd-O case); the dequant **exhaustively** checked over all 65 536 bf16 codes × both lane parities, where the only disagreements are the **508 denormals** RADV flushes to zero — and `Q38_DENSE_GPU_SCAN=1` then found **0 of 3 977 707 520 bf16 values in the real checkpoint are denormal**, so that class is unreachable and the item's numerics reduce to summation order exactly as the spec claims. Against an **f64 truth** the GPU is **0.21–0.65× the CPU kernel's own error**, i.e. 1.5–4.8× *more* accurate than what it replaces. (Two of the spec's own step-0 bars were wrong and are corrected in the record: "relL2 ≤ 1e−6" was stated against a *scalar f32* reference less accurate than the GPU, and the exhaustive leg first flagged **negative zero**, which any sum starting at +0.0 produces on the CPU too.) **Step-0 timing, medians of 10, pools > Infinity Cache, two repeats:** `dn-proj` **19.64 / 19.83** (gate ≤ 26.4 → clears, step 1 built), `qsa-proj` **6.83 / 7.44**, `lm-head` **2.74 / 2.85** (gate ≤ 7.9 → clears by 2.8×), **`gr-pair` 38.12 / 21.19 against a ≤ 9.2 gate**. **Step 4 is therefore NOT BUILT and needs no arbitration — the spec pre-authorised the refusal** ("×97 must be ≤ 9.2 or the step is not built and the reason is the measurement"): the one-submit chained form costs **21.2–38.1 ms/token against the pair's 18.45 on the CPU**, i.e. slower than the CPU arm before its 1.28 GB of VRAM is paid for, because three tiny dispatches with a barrier between them run at **33.6–60.4 GB/s** against the head's 465. `struct PC` was never changed and glm53's push-constant range is untouched. **Consequence: Q12 can no longer move anything** — the gated-residual pair was its only remaining lever. **Step 1 (DeltaNet, `Q38_DENSE_GPU=1`): the oracle is clean and the prize is real, but two numeric legs of the gate are missed.** Oracle — knob off **BIT-IDENTICAL** to pristine (993 280 bytes) with `teacher_forcing` identical 30/30 and 1200/1200 on both prompts; knob on **greedy identical, `teacher_forcing` identical 30/30 AND 1200/1200, last-token relL2 1.9388e−06 short / 4.1945e−07 long against a ≤ 1e−5 bar, cosine 1.000000000, argmax unchanged** — the same class §QP's R3-vs-R1 measured at 1.9e−6 / 4.9e−7, agreeing to the digit. `[OPTIME]` decode-only, three fresh-process repeats, medians: **the token 171.269 → 139.740, −31.53 ms (18.4 %)**, `dn-proj` 52.859 → **26.979**, `vk-dense` 0 → 26.955, `dense-matmul` 89.744 → 65.344, the four neighbouring `dn-*` buckets **flat**, and the VRAM ledger **exact** (printed 4.17 GB / 180 tensors against the spec's 4.16; tier preloaded **13 827** against a predicted ~13 830, error **3**; placement 91.25 → 89.47 %). The token fell **more** than its bucket because `moe` also fell 6.95 (`vk-take` −6.01, `cpu-experts` +1.36): a smaller tier means a smaller GPU expert group and less idle wait — in *this* regime the eviction is a net gain, which is not a contradiction of §QP(d)'s +2.5 warm-identical price but a different regime, and the serving A/B that would price it was **not run**. **The gate: `dn-proj` ≤ 26.40 → 26.979, MISSED by 2.2 %; and `dn-proj` within 1.25× of step 0 (≤ 24.55) → 26.979 = 1.37×, MISSED.** **The cause is measured and it is NOT the kernel:** `bench_q7_dutycycle` replays the identical two submits with a busy-wait of the engine's own measured gap between them (0.26 ms of CPU recurrence between A and B) and nothing else changed — back-to-back **18.61**, with the engine's gaps **27.46–34.67**, against the engine's 26.98. dev0 idles in sub-millisecond bursts 72 times a token and never reaches the clocks a back-to-back loop holds it at. **So the spec's own step-0 remedy ("the fix is in the kernel — wider loads, `uvec4` per lane") would not close this gap**, and 26.98 falls outside the spec's 200–400 GB/s bracket (14.8–25.2), which is precisely the escalation condition the spec names: *"Fable is not asked again unless a step misses its numeric leg by a margin the step-0 bracket did not contain."* **ARBITRATION REQUESTED on step 1, with the question stated: does a −31.53 ms/token step with a clean oracle, an exact VRAM ledger and a 2.2 % miss on a bucket threshold ship, given the miss is a GPU duty-cycle property that no kernel edit addresses?** Steps 2 (QSA, gate `qsa-proj` ≤ **7.90** — now a MEASURED bucket of 15.803, not §Q-PROFILE's ~16.4 estimate — step-0 predicts 6.83–7.44) and 3 (LM head, step-0 predicts 2.74–2.85 against ≤ 7.9) are **not started**, because an ungated step is not proceeded past. **The other engine was re-checked at this step** (two shared files): glm53 `teacher_forcing`, `last_logits`, the greedy continuation, `[MAP] 12096/12096 copy=0` and the 1296/1695/1695 preload all **IDENTICAL**, the only stdout difference being the wall-clock line. Previous row text follows. | **SPEC WRITTEN 2026-09-12 (Fable): `tools/hot-expert/Q7-DENSE-GPU-SPEC-2026-09-12.md` — build from it, Opus, 3–5 days; nothing in it needs re-deriving.** Its four calls, each with the arithmetic in the spec: **(i) submit plan — two submits per DeltaNet/QSA layer** (the CPU recurrence/attention sits between them; one-per-layer needs a G12-class kernel for ~2.9 ms/token and is out of scope), the head one submit, and **the gated-residual pair as ONE submit per site or not at all**: `up` chained behind `down` through `coli_vk_matmul_multi`'s existing `src` mechanism with `silu(x/C)` applied while the shader stages its 320-float input (a push-constant flag, no extra dispatch). The pair does **not** fold into the neighbouring layer's submit, and Q2's 6.56 → 1.43 does not change that: the mix is CPU work between two GPU stages whatever its size, and putting it on-device costs three elementwise dispatches per site that §G12 measured at ~60–100 µs each inside an open command buffer — 291/token = 17–29 ms, more than the pair's whole 18.45 ms. Two submits per site nets ~−1 ms (not worth 1.28 GB); one submit nets −5 to −8; that is the only form built. **(ii) Q12 does not gate this**: one int8 tensor already fails QP's greedy leg, the depth amplification is ~√n, so any subset worth ≥ 1 GB is inside the failed class; and the only tensor set where a surviving int8 CPU arm would beat GPU BF16 on the token is the pair, which is step 4 and last — Q12 may run in the gap before it. **(iii) the router stays**, for a third reason beyond the two here: a reassociation on the router flips *routing* near ties, and the routing decision is the one place on this engine where order-only numerics are not harmless. **Numerics bar: NOT §G12's 0.99992** (that was GLSL `exp` in a recurrent state); this item reassociates exact bf16×f32 products with no transcendental and no device state in steps 1–3, the class §QP's R3-vs-R1 measured at relL2 4.9e−7 / 1.9e−6 with identical text — so the bar is **greedy identical, `teacher_forcing` identical over 30 and 1200 positions, last-token relL2 ≤ 1e−5**, any TF change reported with its margin and arbitrated, never rounded. **Gates re-derived per the step-0 rule**: `dn-proj` ≤ 26.4 (half of 52.8; the old "`dense-matmul` under 55" implied ≤ 17.8, the 400 GB/s end of the bracket with nothing to spare, and is replaced), `qsa-proj` ≤ half its new sub-timer, `lm-head` ≤ 7.9, the pair's four buckets ≤ 9.2 — each also within 1.25× of step 0's measured per-submit prediction. **Expected: −42 to −59 ms/token net of the measured VRAM price for steps 1–3 (174 → ~116–132 ms), −5 to −8 more for step 4.** Order: Q4 (running) → Q7 steps 0–3 → step 4 → Q9. Previous row text follows. **ARM DECIDED 2026-09-12 — it is Q7-gpu at BF16, over the WHOLE 6.8 GB dense set, and Q7-cpu is dead** (record §QP). QP(b) killed both int8 forms: the CPU arm's weight format fails the gate (cos 0.980191 per-row / 0.982097 g64, below §G14's own rejected 0.98964) in the weights-only, f32-activation, same-summation-order best case, so the `maddubs` kernel this row is built around — which needs int8 activations as well — is worse than what already failed; and the `int8 fmt` variant of the GPU arm dies with it. **So the −50 to −60 ms/token Q7-cpu line, the largest single number this track had, is withdrawn**, and what is left is the BF16 `fmt` in `qmatmul.comp`, whose numerics are summation order only. QP(d) then priced the VRAM: **+1.77 / +2.91 / +4.06 ms/token of warm-identical time at 3.4 / 5.4 / 6.8 GB of dense reservation** (0.60 ms/token per GB; ~0.26–0.29 ms per expert pushed off the tier; 707 / 1 087 / 1 413 evicted), against this row's −35 to −50 for DeltaNet + QSA — **under 10 % of the gain**. **And §Q-ARB's condition for the gated-residual pair is MET** (6.8 GB is +1.16 ms/token against 5.4 on warm-identical, −1.68 on the rotating median — the criterion was 10 ms), so the pair is IN: the tensor set is DeltaNet + QSA + the gated-residual pair, 6.8 GB, and §Q2 has already reduced the CPU-side sigmoid mix a fused pair must absorb from 6.56 to 1.43 ms/token. **Two things the spec no longer has to decide and one it now does:** the arm is decided and the VRAM ledger is measured rather than bounded; what it does have to decide is whether **Q12** (which subset of the 676 dense tensors DOES take int8 — §QP's depth finding) runs first, since a surviving subset would revive a CPU arm for exactly those tensors and change what is worth moving to dev0. The head goes to dev0 as **BF16 at 1.27 GB** unless Fable relaxes QP(c) (Q6's row). Original row follows. **The dense stream off the CPU's BF16 floor** (was Q3; absorbs Q6). **Decided in §Q-ARB, two arms, QP(b) picks:** **Q7-cpu** — int8 weights for the dense projections *quantised at load* (no second checkpoint), through the `maddubs` kernel already in `rome_cpubench.c`; **2.35× per call, no VRAM, 1–2 days**, and it composes with the GPU arm rather than replacing it. **Q7-gpu** — the dense projections on **dev0** (the only device with a dense dispatch path; dev2/dev3 are expert-tier only) as a BF16 `fmt` in `qmatmul.comp` (summation-order numerics only) or int8 if QP(b) allows; **dense VRAM is allocated before the preload and the tier fills what is left** — the per-GB rule in §Q-ARB. What moves, in order of ms per submit: DeltaNet projections (4.16 GB, ~55 ms, 2 submits/layer), QSA projections (1.24 GB, ~16 ms, 2 submits/layer), the gated-residual pair (1.28 GB, ~17 ms) **only if** its two dependent GEMVs and the sigmoid mix record into the neighbouring layer's submit — 97 sites × 2 round trips would eat two thirds of it; the router (0.13 GB, 1.7 ms) **never**: 48 submits cost more than it does | `dense-matmul` **90.0 ms/token (47.0%)** at **75.7 GB/s** | Q7-cpu: INT8 `maddubs` **0.280 ms vs BF16 0.658 at 2560×10240**, 8 threads, pools > L3 — half the bytes at a higher rate. Q7-gpu: 09-04 matrix, GPU int8 GEMV 132–403 GB/s effective, submit+fence **53–66 µs**, 4 tensors per submit 0.247 vs 0.378 ms. **VRAM is the constraint on the GPU arm, not throughput** — and §Q-ARB shows it is a cheap one | Q7-cpu **−50 to −60 ms/token** (dense −45 to −50 at ~2.2× in-engine, head −8 to −9; no VRAM). Q7-gpu **−35 to −50** for DeltaNet+QSA at BF16 against today's CPU figure (the range is the submit plan: one submit per layer or two), **−10 more** for a fused gated-residual pair, ~5–8 less against a post-Q4 figure — **but only −14 to −20 after Q7-cpu** (int8 GPU at 425 GB/s batched vs int8 CPU at 93.6: 2.7 GB is ~7–10 ms on dev0 against ~29 on the CPU), which may not buy 3–5 days | design 1 session; Q7-cpu 1–2 days; Q7-gpu 3–5 days | **Fable** (the spec: tensor contract, per-layer submit plan, VRAM ledger — the *decision* is already taken in §Q-ARB) then Opus | Q7-cpu: knob off bit-identical; knob on: QP(b)'s numbers reproduced in-engine, `[OPTIME] dense-matmul` ≤ 45 ms/token, rotating median. Q7-gpu: knob off bit-identical; BF16 `fmt`: greedy identity over 128 tokens and last-token logits within 1e-5; `[OPTIME] dense-matmul` per moved set (sub-timer per set, DeltaNet first — **stop and re-profile if DeltaNet alone does not bring it under 55**); `tworeq.py`; **the VRAM ledger per device and the resulting tier count and placement % in the record** |
| **Q8** | **DEAD 2026-09-12 — QP(a) failed, and this row's own text says what that means** (record §QP). int4-g64 on the routed experts, simulated bit-for-bit against the tree's `quantize_i4_grouped` + `matmul_i4_grouped` *and* against the converter's own `quant_int4_grouped(gs=64)`, with every routed expert forced onto the CPU and an identically-placed control that is reassociation-clean (cos 1.000000000, relL2 4.9e−7): **3 of 30 short-prompt `teacher_forcing` predictions change, 319 of 1200 on the long prompt, `last_logits` cos 0.926429, greedy text diverges.** Both kill lines fire at both prompt lengths. **No converter, no shader, no `fmt=4` tier path and no second checkpoint were written**, per this row's own gate and §G15's precedent — the 64–68 GB of expert shards, the ~65 GB ledger and the −56 ms rotating target are all withdrawn. Three qualifications that belong to the verdict: **f32 scales were used**, which is the favourable half of this row's own fp16/f32 choice, so the fp16 variant is worse and the verdict holds *a fortiori*; the probe is **pessimistic nowhere** (fp8 is the converter's own source, so there is no §G15-style double quantisation to discount); and what died is **`fmt=4` specifically** — symmetric absmax int4 over 64 inputs — **not "int4 for this model"**: g32, a zero-point form, and mixed precision (`down_proj` left at fp8) are untested and out of QP's scope, and `Q38_I4_SIM` plus `rome_q4sim.c` are in the tree so anyone reviving the idea re-probes instead of re-litigating. A hypothesis for *why*, offered as such: a Qwen routed expert is **640 wide** against GLM's 4096 and `down_proj` contracts over only 640 inputs, so far fewer products are averaged per output and quantisation noise is suppressed much less. **Downstream: Q9's gate does not change** — SPEC-PROBE's reversal condition was "after Q8 the miss share is the never-routed tail", and there is no Q8, so the ~1.2× cap stands as Q9's null hypothesis. Original row follows. **int4 experts — and what they unlock** (was Q6). A Qwen expert is 4.92 MB in FP8 and **2.61–2.76 MB in int4 g64 with its scales** (fp16 or f32 scales; the draft's "2.46 MB, all in ~60 GB" counted nibbles only), so all 24 576 are **64–68 GB** and the 21 858 with history are **57–60 GB** — which is why the dense set is reserved *first* and the experts fill the remaining ~65 GB (§Q-ARB): with fp16 scales every expert fits, with f32 scales every expert that has ever been routed does. **QP(a) is its gate and runs first; if it fails the item is dead and no converter, shader or kernel is written** (§G15). What it removes is not the 15.4 ms CPU-expert bucket, which is hidden inside the GPU wait: it is the **56 ms/token between the rotating median and warm-identical** (miss service), plus ~10–15 ms fresh-process from the staging loop (`q38vk_expert_ensure` + `memcpy`, 7.6 ms) and a shorter int4 GPU gap (09-04 matrix: int4 12.6 vs fp8-emul 22 ms/token per device) | cpu-experts **15.4** (hidden); vk-issue staging **~7.6**; tier **14 673 of 21 858, 7 185 unplaced**; placement **80.80%**; **rotating − warm = 56 ms/token** | The GPU int4 path is production for GLM (`fmt=4`, `vk_tile_ok4`), so is the CPU kernel (`matmul_i4_grouped`), so is the preload; **new is the converter and a second set of expert shards on disk (~64–68 GB — check the NVMe's free space before writing it)**. §G14: a probe sizes throughput on synthetic data but **never** accuracy — QP(a) runs on the real activations | **−10 to −15 ms/token fresh-process (arithmetic) plus up to −56 on the rotating median — the second number is what it is for**; and it fires §SPEC-PROBE's reversal for Q9 | after QP(a): 3+ days, a second checkpoint on disk | Opus (build; the ledger and order are in §Q-ARB) | QP(a) passed; converter round-trips the sim bit-for-bit (`rome_i3sim.c` is the pattern); placement ≥ 21 858 with the dense set reserved; `expert-read` and the staging share → ~0; **rotating median converging on warm-identical** is the headline gate; `tworeq.py` |
| **Q9** | **MTP speculative decoding** (was Q4). **⚠ RE-GATE BEFORE STARTING. The "GATED BEHIND Q8" clause is now moot: Q8 is dead (QP(a), record §QP), so SPEC-PROBE's reversal condition — "after Q8 the miss share is the never-routed tail" — never fires, and §SPEC-PROBE's ~1.2× cap stands as this row's null hypothesis rather than something Q8 was going to lift.** That makes the chunk probe (step 1) the whole decision, and it makes this row *cheaper to refuse* than it was: it is half a day to find out. Original row follows. **⚠ RE-GATE BEFORE STARTING and GATED BEHIND Q8.** Load the MTP module, draft up to 4 tokens per step reusing the QSA top-k indices, verify in one forward, accept the matching prefix (tech report Table 4: 4.06 mean accepted) | — | §SPEC-PROBE capped this family at ~1.2× optimistic **for GLM**, because the experts that miss the tier are cold ones adjacent tokens never share. Qwen's miss share is **19.2%**, close enough to GLM's 21% that the same cap is the null hypothesis — **but after Q8 the miss share is the never-routed tail and SPEC-PROBE's own reversal condition fires** | largest multiplier on the board, **after Q8 and not before** | chunk probe half a day; spec 1 session + 5–8 days | **Fable** (spec) then Opus + Sonnet | **step 1 is the chunk probe on qwen38** (`Q38_PREFILL_BATCH` × the `[OPTIME] placement` counters): report the CPU-expert count at block size 1→16. Then accepted text == greedy text for 1k tokens; accepted length ≥ 3.5; rotating tok/s |
| **Q10** | **The DeltaNet recurrence's state traffic** (new, logged at Q1's arbitration 2026-09-12 from Q1's step-0 sub-timers, record §Q1 "What this changes" and `### Arbitration`; not sized by a measurement yet). `dn-recur` is 9.19 ms/token and already 48-way parallel; per head and token it makes four traversals of a 128×128 f32 state (64 KB): decay `state *= α` (read+write), `k·state` (read), the rank-1 update (read+write), `q·state` (read). Two foldings, both order-preserving per element: **(a)** drop the decay pass — compute `fl(s·α)` on the fly inside `k·state` and again inside the rank-1 update as its addend (`state = fl(s·α) + k·δ`; the addend of an FMA is never contracted, so the value is the same float both times): 6 → 4 traffic units; **(b)** fold `q·state` into the rank-1 pass with `d` outer and `value` inner: 4 → 3, every `current[value]` still accumulates in ascending `d`. **Sizing, corrected at arbitration:** §Q1's "680 MB/token at 74 GB/s, the memory ceiling" assumes all four traversals hit DRAM, but a head's state is 64 KB and this CPU has **512 KB of L2 per core**, so traversals two to four are L2-served and DRAM sees ~226 MB/token (~25 GB/s). The likelier cost is the two **`d`-inner strided reductions** (`previous`, `current`: stride 512 B, 128 dependent adds per output, unvectorised across `value`), which (b)'s loop order also removes | dn-recur **9.19 ms/token** (§Q1), 4.8% of the token | Q1's sub-timers; the traversal arithmetic above; `lscpu` | **−2.5 ms/token** on the bandwidth reading; **more on the latency reading** — step 0 says which, and the gate is derived from it, not from this cell | step 0 half a day; the change half a day | Sonnet (Opus if the vector form of (b) needs checking against contraction) | **step 0 first**: the recurrence added to `rome_dnbench.c` with its oracle extended to the state, timed L2-hot against cold, and `perf stat` cycles against bytes — the gate is then written as a fraction of the 9.19 (per the "(est.)" rule) before the change is built; **bit-identical** (the microbenchmark's `memcmp` on the state over 36 layers × 64 tokens, then greedy text + last-token logits — contraction must match in both forms, and the oracle decides, not the argument); `[OPTIME] dn-recur` and `deltanet` reported, untouched sub-buckets flat; serving A/B with its own pristine rows |
| **Q11** | **The gated residual's write-back, and the 0.35 ms/token §Q2 could not keep** (new, logged at Q2 2026-09-12, record §Q2 "the rms half, rejected" — and unlike most new rows its measurement is already taken). `q38_gr_apply` is a serial scalar loop at 96 sites per token: `for b<C for d<H: hyper[s*W+b*H+d] += inject[s*C+b]*block[s*H+d]`, 10 240 fused multiply-adds per site, **no reduction at all**, so `collapse(2)` over (b,d) or over (s,b) is bit-identical by construction — the easiest pragma left on this engine. Its own bucket is small (0.15–0.18 ms/token) but it is the **precondition for re-applying §Q2's rejected rms pragma**: §Q2 measured that parallelising `q38_gr_read`'s four `q38_rms0` calls wins 0.33 in `gr-read` and loses 0.35 in `gr-apply`, because four threads pull all 40 KB of `hyper` into four private L2s and this loop then writes all of it from one thread, paying cross-CCX invalidation on this 4-CCX EPYC. With this loop parallel the invalidation is paid eight ways and the rms half becomes free. **Do both halves in one item and gate them together**, or the second half will look like a regression again | gr-apply **0.15–0.18 ms/token** (96 sites) **plus** the 0.33 §Q2 measured and left on the table | §Q2's three-arm in-engine table (pristine / rms+mix / mix only: `gr-apply` 0.18 / 0.53 / 0.17, `gr-rms` 1.07 / 0.82 / 1.07, step 179.29 / 174.65 / 174.76), medians of three repeats, reproducible to ±0.01 in two campaigns | **−0.4 to −0.5 ms/token** (0.15 from this loop + 0.33 from the rms half, less one fork/join per site) — small, and it is the cheapest remaining serial scalar loop on the engine | half a day | Sonnet | **bit-identical** (no reduction anywhere in either loop, so this is by construction; greedy text + last-token logits anyway); `[OPTIME] gr-apply` ≤ 0.10 **and** `gr-rms` ≤ 0.85 **in the same binary**; **report `gr-apply` and `gr-read` together and the token beside them** — §Q2's lesson is that either loop alone moves time between these two buckets; serving A/B with its own pristine rows |
| **Q12** | **RE-SEQUENCED 2026-09-12 (Opus, from Q7 step 0's measurement; record §Q7): its one remaining lever is GONE, so this item is now worth nothing on the token and should be closed unless someone wants the boundary for its own sake.** The Q7 spec's sequencing argument ended "even a survivor beats the GPU BF16 arm on the token only for the gated-residual pair, which is Q7's last step". **Q7 step 0 measured that pair on the GPU at 21.2–38.1 ms/token against its 18.45 on the CPU** — the GPU arm is slower than the CPU arm there, so the pair stays on the CPU whatever Q12 finds, and Q7 step 4 was refused. A surviving int8 subset could still make the pair's CPU arm faster (~7.9 vs 18.45), which is the ONLY thing left for this item — but that is now a plain CPU-precision question with no placement consequence, and it has to clear QP's unsoftened kill line to claim it. Previous row text follows. | **SEQUENCED 2026-09-12 (Fable, Q7 spec §"Q12 sequencing"): does NOT gate Q7-gpu.** One int8 tensor already fails QP's greedy leg (the head, §QP (c)); the depth amplification is a random walk (676 tensors = 26× one tensor ≈ √676), so a subset worth ≥ 1 GB (≥ 36 tensors, ~6×) sits inside the failed class; and even a survivor beats the GPU BF16 arm on the token only for the gated-residual pair (CPU int8 ~7.9 vs GPU ~9–12 + 1.2 ms), which is Q7's last step. Run it in the gap before Q7 step 4 if rig time allows; it can move the pair's placement and nothing else. Original row follows. **How much of the dense set DOES take int8? — the subset question QP(b) did not ask** (new, logged at QP 2026-09-12, record §QP (b) "why depth, not per-tensor precision"). QP(b) quantised all **676** BF16 layer tensors at once and the model failed; the same measurement also showed **why**, and the why is the item: the int8 weight perturbation is **relL2 3.9e−3**, one int8 tensor (the head, QP(c)) moves the logits by **7.7e−3** — about 2×, no depth — and the identical per-tensor perturbation applied across **48 layers** moves them by **0.198**, a **~50× amplification that is entirely depth**. So "int8 the dense set" and "int8 a dense tensor" are different questions and QP answered only the first. Find the largest subset of the 676 that clears QP's gate, biggest-bucket-first: `dn-proj` is ~52.8 ms/token of the 90.0 in `dense-matmul` and the gated-residual pair is 17.8, so even a partial subset is worth real time on the CPU — and if a subset *does* clear it, Q7-cpu comes back from the dead for exactly those tensors | `dense-matmul` **90.0** (of which `dn-proj` 52.8, the GR pair 17.8), lm-head **15.8** | §QP's depth arithmetic above; the whole apparatus already exists — `Q38_I8_DENSE`, `Q38_I8_ROUTER`, `Q38_TF`, `qp_probe.sh`, `qp_compare.py`, and a transform oracle that is bit-identical to `quantize_rows` | unknown by construction — **this is a probe, and "no subset survives" is a complete answer** | **step 0 is a tensor-group filter knob (half a day, Sonnet); the probe itself is one qp_probe.sh sweep per subset (~25 min of rig time each)** | Sonnet for the knob, Opus for the verdict | the filter knob **off bit-identical**; then QP's own gate, unchanged and unsoftened, per subset: short-prompt `teacher_forcing` unchanged, long-prompt (≥1000) cos ≥ 0.99, greedy identity over 128 tokens; **report every subset tried including the ones that failed** — the boundary is the result, not the first subset that passes |

### Deleted, with the measurement that deleted them

- **Q0 (populate-at-load + longer routing history) — DEAD, both halves.**
  Half 1: `Q38_MMAP_POPULATE` is **already on by default** on this engine, and
  §G1/§G1b measured the whole-mapping variant on GLM at **122 s of extra load
  for no measurable gain**, with the first-touch cost relocated rather than
  removed — the record's own Q row already said the rotating number did not
  move. Half 2: **measured false.** Seven processes on 2026-09-11 all printed
  `preloaded 14673 of 21858 ... (dev0 24.4/25.7 GB)`; the histogram already
  offers 21 858 candidates and the preload stops because the cards are full.
  **More history cannot add an expert to a full card.** Its purpose — more
  experts resident — is inherited by **Q8**, which is the only way to get them.
- **Old Q2 (FP8 widened residual stream) — DEAD.** Its premise was
  "−10 to −15 ms/token of memory traffic". Measured: the widened residual is
  4×2560 floats = **40 KB per token**, and `gr-apply`, the pass that writes it,
  is **0.18 ms/token — 0.1% of the token**. `gr-read`'s cost is its 10240×320
  weight matrices (§Q2 above), not the residual. There is no 10–15 ms there.
  If it is ever revived it is a long-context/prefill item, not a decode one.
- **A pooled-key cache for the indexer (the Qwen twin of §G5) — NOT AN ITEM.**
  Qwen already has it (`IK_pooled`, commit `2d3cf7e` — the commit §G5 ported
  *from*), and the indexer measures **0.15 ms/token at ctx≈70 and 0.28 at
  ctx≈180**. It is not worth a day at any context this box will run.
- **Q6 as a standalone item — folded into Q7** (row above). One tensor with
  the same two formats as the rest of the dense set does not need its own
  design, and sizing it "after Q4, against Q4's number" double-counted with
  Q7 in the other direction.

### §Q-ARB — the VRAM split between the dense stream and the expert tier, decided

The draft said Q7 and Q8 "compete for the same 72 GB from opposite directions
and must be designed against each other", and left that for this session.
Read with the numbers, they do not compete on equal terms, and the right
answer is a rule plus a probe rather than a negotiated split. (The old
roadmap's version of this mistake was different from the one the draft
describes: it paired dense-off-DRAM with **MTP**, which needs verify
bandwidth rather than VRAM, and left the item that actually shares the
VRAM — int4 experts — as "(later)", unpaired.)

**1. The per-GB rule: dense weight is worth ≥ 10× more per GB of VRAM than
the marginal expert, so the dense set is reserved first and the tier fills
the rest.** A GB of dense BF16 costs **13.2 ms/token** on the CPU (6.81 GB in
90.0 ms) and ~2–3 ms on the GPU (bandwidth plus its share of submits): about
**−10 ms/token per GB moved**. A GB at the cold end of the tier is 203 FP8
experts, each serving ≤ 0.0055% of activations (the resident mean; the true
figure for the coldest is nearer the unplaced mean of 0.0027%), i.e. ≤ 1.1%
of activations, ≤ 5.4 more CPU experts per token, ≤ **0.9 ms/token** of
CPU-expert time — and that time runs inside a GPU gap that, after Q3, still
has 8.4 ms/token of idle in it (`vk-take` after Q3, record §Q3; this paragraph
originally estimated 5.8 from the full 7.2 being absorbed, and only 54% was —
12.6 − 3.98 absorbed = 8.4 left, and it is distributed so that a uniform
0.15 ms/layer of added CPU work was absorbed at 54%: price the eviction's CPU
time at that rate, not at 100%), while each expert moved off the GPU also
shortens the GPU side by 0.073 ms (28.4 ms of gap for 388 experts). The
margin is cold because the tier is large; this is the one consequence of the
14 673 count that the old "not VRAM-bound" entry was reaching for.

**2. The ledger, both cases.** Budget ≈ 72.2 GB as measured (3 × 25.7 less the
1.5 GB per-device reserve and scratch). Dense set: DeltaNet 4.16 + QSA 1.24 +
gated residual 1.28 = 6.68 GB BF16 (3.34 int8); LM head 1.27 BF16 / 0.64 int8;
the router stays on the CPU.

| case | dense on dev0 | left for experts | experts placed | evicted vs today |
|---|---:|---:|---|---|
| **A. Q8 alive** (int4 g64) | 6.68 + 0.64 = **7.3 GB** | **64.9 GB** | 24 850 at 2.61 MB (fp16 scales) — **all 24 576**; 23 500 at 2.76 MB (f32) — **all 21 858 with history** | — (the tier grows by ~8 800 experts) |
| **B. Q8 dead** (FP8), DeltaNet + QSA only | **5.4 GB** | 66.8 GB | 13 570 | ~1 100 coldest (7.5%) |
| **B. Q8 dead**, whole dense set | **6.7 GB** | 65.5 GB | 13 300 | ~1 380 coldest (9.4%) |
| ~~**B, int8 fmt** if QP(b) allows~~ | ~~3.3 GB~~ | ~~68.9 GB~~ | ~~14 000~~ | ~~670 (4.6%)~~ **— STRUCK OUT: QP(b) does not allow it (record §QP)** |

**RESOLVED 2026-09-12 (record §QP): case A is impossible (Q8 is dead) and the
int8 row is struck out (QP(b) is dead), so the live rows are B at 5.4 GB and B at
6.8 GB — and QP(d) measured both.** The ledger arithmetic above was right to
within 0.2%: it predicted 13 570 experts placed at 5.4 GB and ~1 100 evicted, and
the engine places **13 586** and evicts **1 087**; it predicted 13 300 at 6.7 GB
and the engine places **13 260** at 6.8. The activation-share bound of 2.9–7.6%
lands at **2.90 percentage points** of placement over the whole 6.8 GB.

**What QP(d) replaces the bound with.** Case B's eviction costs, *measured*:
**+1.77 / +2.91 / +4.06 ms/token of warm-identical time at 3.4 / 5.4 / 6.8 GB**,
a straight line at **0.60 ms/token per GB**, equivalently **~0.26–0.29 ms/token
per expert pushed off the tier** (707 / 1 087 / 1 413 evicted, +6.7 / +11.1 /
+13.9 CPU-served experts per token). On the rotating median the same three levels
read **+7.9 / +11.8 / +10.2 ms/token**, i.e. the low half of the "+8 to +22"
bound — but that row's own N=0 pair spans 5.12 ms/token and it comes out
non-monotonic, so **the warm-identical row is the one that resolves this and the
bound was 3–5× pessimistic on it**. Against −35 to −50 from the move, the whole
dense set on dev0 costs **under 10% of its own gain**. The original bound
follows, for the record: by the bounds in rule 1, **2.9–7.6% of activations →
14–36 more CPU experts/token → 2.3–6.1 ms of CPU-expert time, ~0–2 ms of it
visible on the fresh-process token** (the rest hides in the gap), and on the
rotating median at most +8 to +22 ms/token if the 56 ms miss surplus scales
linearly with the miss share — against −35 to −50 from the move. Net positive in
every cell.

**3. Sequencing: the probe collapses the dependency, so the items are decided
in order, not co-designed.** What the draft called co-design is, with the
ledger in hand, one unknown — does Qwen's expert take int4 — and one
unknown does not need a joint document; it needs to be answered before the
Q7 spec is written, and QP(a) answers it in a day. The Fable session then
writes the Q7 spec against **one** ledger row, not two, and Q8's build order
in the same sitting. Build order after that: **Q7-cpu first if int8 dense
survived** (1–2 days, no VRAM, the largest single number on the table),
**then Q8 if alive** (it de-contends the VRAM and it is the only item that
acts on the rotating median's 56 ms), **then Q9**; Q7-gpu comes after those, and only if a
re-profile after Q7-cpu still shows ≥ 15 ms/token in the dense stream —
by the matrix it is worth −14 to −20 there, and 3–5 days of Opus for that is
a decision to take on a measured figure, not now. If int8 dense died: Q4 →
Q8 (if alive) → Q7-gpu at BF16 with QP(d)'s number as its VRAM cost → Q9.
**That last sentence is the one that applies (QP, 2026-09-12): int8 dense died
and Q8 is dead, so the order is Q4 → Q7-gpu at BF16 over the whole 6.8 GB dense
set → Q9, with QP(d)'s measured 4.06 ms/token as the VRAM cost. The "Q7-cpu
first" branch and the "only if a re-profile still shows ≥ 15 ms/token" condition
on Q7-gpu both belonged to the int8 branch and are void — Q7-gpu is no longer
"after Q7-cpu", it is the only dense arm there is.**
**Q4 LANDED 2026-09-12 (record §Q4), so the next item in that order is Q7-gpu,
whose spec is already written (`Q7-DENSE-GPU-SPEC-2026-09-12.md`) — start at its
step 0.** Q4's number is −9.79 ms/token on the token and −8.50 on
`dense-matmul`, **behind `Q38_BF16_ACC4=1`, which is not the default**, so
Q7-gpu's step-0 buckets must be taken with the knob in the SAME state on both
sides of every A/B, exactly as that spec already says.

| QP(b) int8 dense | QP(a) int4 experts | Q7 arm and order |
|---|---|---|
| survives | alive | Q7-cpu (int8, all dense + head) → Q8 → Q9; Q7-gpu (int8 `fmt`, 3.3 GB, uncontended) **only if a re-profile after Q7-cpu still shows ≥ 15 ms/token in it**; Q4 as a prefill item |
| survives | dead | Q7-cpu → Q7-gpu on the same re-profile condition (int8 `fmt`, 3.3 GB, evicts ~670) → Q4 as a prefill item |
| dies | alive | Q4 → Q8 → Q7-gpu (BF16 `fmt`, 6.7 GB, uncontended) → Q9 |
| **dies** | **dead** | **← THIS IS THE ROW THAT FIRED (QP, 2026-09-12).** Q4 → Q7-gpu (BF16 `fmt`, DeltaNet + QSA first, 5.4 GB, evicts 1 087 measured) **→ and the gated-residual pair IS included: its condition was "only if QP(d) at 6.8 GB is within 10 ms of 5.4", and 6.8 GB measures +1.16 ms/token of warm-identical time against 5.4 (−1.68 on the rotating median, i.e. faster, inside that row's noise) — within 10 ms by an order of magnitude.** So Q7-gpu's tensor set is the whole 6.8 GB dense stream, at a measured VRAM cost of **4.06 ms/token** against an expected −35 to −50. Q9 is not gated behind anything any more (no Q8), and **Q12** — which subset of the dense set does take int8 — is the one thing that could still put a CPU arm back on the table, so Fable decides whether it runs before Q7-gpu is specced. **DECIDED 2026-09-12: it does not; the spec is written (`Q7-DENSE-GPU-SPEC-2026-09-12.md`) and Q12 can only ever move the gated-residual pair, which is its last step.** |

**4. What each spec must contain before Opus starts.**

*Q7-gpu — WRITTEN 2026-09-12: `tools/hot-expert/Q7-DENSE-GPU-SPEC-2026-09-12.md`; the Q7 row above carries its decisions. The list below is what it was asked to contain and is kept for the audit; one item in it is amended there: the stop condition "`dense-matmul` under 55 with DeltaNet alone" is re-derived to `dn-proj ≤ 26.4` (half the step-0 bucket), because 55 implied ≤ 17.8, the optimistic end of the bracket.* The original ask: the **tensor contract** — which tensors, in which
`fmt`, on dev0, with the row layout `qmatmul.comp` expects for BF16 (a new
`fmt`; the shader has int8/int4/int3/fp8 today), how they are uploaded, and
whether the host copy is kept (the knob-off path needs it; the knob-on path
should not hold 6.7 GB twice);
the **per-layer submit plan** — DeltaNet: {qkv, z, b, a} in one submit, the
recurrence on the CPU (a G12-style on-device recurrence is explicitly out of
scope for v1), `out` in a second; QSA: {q, k/v, idx_qk} then `o`; the
activation round trip per submit (host-visible `x`, cached read-back of `y`,
the sizes); the **VRAM ledger per device** with the tier count and placement
% it predicts, checked against what the preload prints; the expected ms per
moved set from the 09-04 matrix; the oracle (knob off bit-identical; BF16
`fmt` greedy identity over 128 tokens and last-token logits within 1e-5;
`tworeq.py`, because per-conversation activations now cross a device
boundary); and the stop condition (DeltaNet lands first, alone; if
`dense-matmul` is not under 55 ms/token with it, stop and re-profile before
QSA). *Q7-cpu* needs no spec beyond QP(b)'s numbers: the kernel exists, the
quantisation is at load, the gate is in the row.

*Q8 (the build spec, written only if QP(a) passed):* the converter (fp8 →
int4 g64, the sim's exact transform, with a `rome_i3sim.c`-style
bit-for-bit check against the tree's own `pack_int4`/`matmul_i4_grouped`);
the second set of expert shards and where they live on the NVMe; the
`fmt=4` tier path (production for GLM — port, do not write); the preload
unchanged except that the dense reservation precedes it; the CPU path for
whatever still misses; and the gates in the row, of which **rotating
converging on warm-identical** is the one that says the item did what it is
for.

### Order, and why

**Q3 → Q1 → Q2** first: **−18 to −28 ms/token** between them, **every one
bit-identical**, every one a port of a track-G pattern already measured on the
other engine (G9, G4 step 3 / G8, G10). Q3 leads because it is half a day for
−5 to −7 — the best ms-per-day on the board, the same reason §G3 put G7
("hours") ahead of larger items — and because it is a pure reorder. Sonnet,
about three days.

**Amended 2026-09-11 by the measurement (record §Q3):** Q3 was built and came
in at **−3.85 ms/token**, bit-identical, below its own −5 to −7 and below both
numeric legs of its gate. The three-item total is therefore **−17 to −27**, and
Q3's ms-per-day is still the best on the board — the item was correctly ranked
and incorrectly sized. Q1 and Q2 are unaffected: neither depends on the GPU gap.

**Amended again 2026-09-12 by Q1's measurement (record §Q1):** Q1 came in at
**−6.02 ms/token on the token and −6.35 on `deltanet`**, bit-identical, below
its own −8 to −13 — **because its bucket was half what the estimate said.** Its
mandatory step-0 sub-timer measured the three loops it names at **7.98 ms/token,
not ~15.9**: 58% of §Q-PROFILE's "serial scalar" difference (70.5 − 55.0) is the
**48-head recurrence, the one loop that already had a `parallel for`**, and
`dn-proj` measured 52.26 against the 55.0 the bandwidth arithmetic assumed. Of
the 7.98 actually available, Q1 took **80%** (7.98 → 1.56). So the three-item
total becomes **−14.9 to −17.9** with Q2 still at its estimate, and **Q2's own
estimate is the identical arithmetic-subtraction shape** (`gr-read` 26.7 minus
16.9 of matmul = ~9.9 "serial scalar") — **expect its step 0 to re-size it too,
and do not commit to the −5 to −8 until it has.** Two corrections that run the
other way and belong to whoever sizes Q2: in-engine **attenuation on a serial
scalar loop is INVERSE** (Q1's three loops cost 7.98 ms/token in the engine
against 3.29 isolated, because a lone scalar loop runs beside seven spinning
libgomp threads — §RP1's load-bearing spin), so an isolated serial-vs-parallel
ratio **understates** the win; and the fused-region form is the right shape
(one fork/join per layer rather than one per loop, measured at 5.3 µs/region).
The one downstream consequence is **Q5**, which now has even less room to turn
its counter into a token (see the §Q3 note). **Q3 merged 2026-09-12 by
arbitration** (record §Q3): the gate's numeric legs assumed the shared expert
would be absorbed in full and were unreachable by a reorder; corrected to half
the shared bucket, which the measurement meets at 54%. **Q1 merged 2026-09-12
by arbitration** (record §Q1): a different finding — the mechanism out-delivered
its evidence and the *target* had been over-counted by the estimate; corrected
to half the step-0 bucket, met at 80%. Two bit-identical items, −9.9 ms/token
between them on the fresh-process token (190.47 → 179.84 across the two
campaigns), and one new row out of Q1's sub-timers, **Q10** (the recurrence's
state traffic), placed after Q2 and before QP because it is the same kind of
work as Q1 and Q2 and its step 0 is a day at most.

**Amended a third time 2026-09-12 by Q2's measurement (record §Q2), and this is
the row where the procedure paid for itself.** Q2 came in at **−5.31 ms/token on
the token and −5.08 on `gr-read`**, bit-identical — **inside** its −5 to −8
expectation, and **it did not go to arbitration**, because step 0 re-derived the
gate from the measurement before the candidate was built (the bucket is 7.62,
not ~9.9; `gr-read ≤ 20` has a floor of 20.20 and is unreachable; the corrected
thresholds were committed two commits before the first candidate number) and
every corrected leg was then met. **The three-item total is −15.2 ms/token
measured** (190.47 → 174.18 across three campaigns is −16.3, of which ~1 is
campaign offset), against the −18 to −28 this section opened with. Q2 also
**rejected half of its own item on measurement**: parallelising the four
`q38_rms0` calls moves 0.35 ms/token into `q38_gr_apply` (cross-CCX
invalidation of `hyper`) for the 0.33 it gains, so that pragma was deleted and
logged as **Q11** together with the `q38_gr_apply` parallelisation that makes it
free. **Q11 is half a day and −0.4 to −0.5**; it is placed with Q10, after QP
rather than before it, because both are now small next to what QP decides.

**QP: DONE 2026-09-12 (record §QP), and it did what it was placed here to do —
it deleted more work than it created.** The four answers, in the order the rest
of this section depends on them:

- **(a) int4-g64 experts: DEAD.** 3 of 30 short-prompt `teacher_forcing`
  changes, 319 of 1200 long-prompt, cos 0.926429. **Q8 is dead**, and with it
  its converter, its second 64–68 GB checkpoint, its `fmt=4` tier port, and its
  −56 ms rotating target. Q9's SPEC-PROBE reversal never fires.
- **(b) int8 dense: DEAD, both groupings** (per-row cos 0.980191, g64 0.982097 —
  both below §G14's own rejected 0.98964), **and this was the weights-only,
  f32-activation, same-summation-order best case**, so the `maddubs` kernel
  Q7-cpu was built around is worse than what failed. **Q7-cpu is dead and
  Q7-gpu's `int8 fmt` with it. Q7 is a BF16 arm or nothing, and Q4 goes first.**
- **(c) int8 LM head: fails the gate as written on both arms, marginally**, and
  is the one thing QP hands to Fable as a decision rather than an answer
  (§QP (c) prices both branches).
- **(d) the eviction curve: measured**, and it validated §Q-ARB's ledger
  arithmetic to within 0.2% (predicted 13 570 experts placed at 5.4 GB of dense
  reservation, measured 13 586).

**What QP cost and what it saved: one day, against the 3+ days of Q8's build,
the 1–2 of Q7-cpu, and a Fable spec session that would have had to choose
between two ledgers.** §G15's precedent held exactly. Everything below this
paragraph is written against these answers.

Then **Q5** (half a day, Sonnet), placed here because its counter tells the
design session how much CPU-side slack the gap has, and re-scoped to say that
the token may not move.

Then the **Fable session**, whose agenda QP has shortened to three items: the
**Q7-gpu BF16 spec** (there is no arm to choose between any more — §Q-ARB's
decision table resolved to its last row), the **QP(c) call** on whether the int8
head's g64 arm is allowed past a short-prompt leg that rejected it on 1 of 30
while the 1200-position leg passed it, and whether **Q12** (which subset of the
676 dense tensors does take int8 — the depth finding in §QP (b)) is worth a
Sonnet knob plus a few probe sweeps before Q7-gpu is written, since a surviving
subset would revive Q7-cpu for exactly those tensors. Q8's build spec is not on
the agenda: there is no Q8. **Q9** is no longer gated behind it — its chunk
probe stands alone and is half a day to refuse.

**The numerics bar, stated once so the back half does not ship for nothing.**
After Q1/Q2/Q3/Q5, **every remaining item changes numerics** — Q4 and a BF16
`fmt` by summation order, Q7-cpu/int8 `fmt` and Q8 by weight format — and the
standing rule ships each behind a knob, off by default. A knob that is off
delivers nothing to the gateway. The bar this track proposes, for the owner to
confirm or veto once: **default-on** at §G12's evidence class — `teacher_forcing`
identical over ≥ 1k positions, greedy identity over 128 tokens, last-token
cosine ≥ 0.9999; **opt-in** down to §G14's rejection line (cosine ≥ 0.99, greedy
text identical); **dead** below it. Summation-order changes should land in the
first class; QP says which of the formats do — **and the answer is none of
them.** Measured 2026-09-12 against this exact bar: int4-g64 experts cos
**0.9264** and int8 dense cos **0.9802 / 0.9821** are all **below the dead
line**; the int8 LM head is cos **0.999970 / 0.999987**, i.e. **inside the
default-on class on the cosine leg**, and it is the greedy/short-TF legs that
reject it — which is precisely why §QP (c) hands it to the owner-and-Fable bar
above rather than deciding it. **The only remaining numerics change on the
track is therefore summation order** (Q4, and a BF16 `fmt` in Q7-gpu), which is
the class this bar already puts default-on, so the "knob that is off delivers
nothing to the gateway" problem has largely dissolved: there is no weight
format left to ship behind one.

**Target for the track, per column, because the columns are different tokens.**
In ms of the 191-ms fresh-process token: Q3+Q1+Q2+Q5 take it to **~163–173
with no numerics change** (**~165–175** with Q3's measured −3.85 in place of its
estimated −5 to −7, record §Q3; **~172–176** with Q1's measured −6.02 in place
of its estimated −8 to −13 as well, record §Q1; **~174** with Q2's measured −5.31 in
place of its estimated −5 to −8 as well, record §Q2 — all three bit-identical
items are now measured and the no-numerics floor of this track is
**~174 ms/token fresh-process, 4.41 tok/s rotating, 5.66 warm-identical**, with
Q5 (0 to −2, counter only) and Q11 (−0.4 to −0.5) the only bit-identical rows
left); then **the BF16 branch, which QP(b) made the only branch** (Q4) to
**~150–165** — **measured 2026-09-12: −9.79 ms/token on the `step` counter, i.e.
~165 ms/token fresh-process, 4.545 tok/s rotating, 5.96 warm-identical, at the
TOP of that range and only with `Q38_BF16_ACC4=1`, which is not the default
(record §Q4)**; then Q7-gpu at BF16 to **~110–135**. Converted to the **rotating
median, cumulative against today's 247 ms**: the bit-identical four
**+9–13%**; then **+13–19%** (Q4); then **+31–50%** with Q7-gpu.
Warm-identical sees the same savings over 191 ms, so about a third more in
percentage terms.

**The int8 branch's column is struck out, not deferred** (QP(b), 2026-09-12):
the "~103–123 then ~85–110, +40–55% then +50–78%" cells were Q7-cpu's, and
Q7-cpu is dead. **This roughly halves the ceiling this track can claim** — from
+50–78% to +31–50% on the rotating median — and the honest statement of the new
ceiling is the point of writing it down rather than quietly dropping the row.
**The 56 ms of miss service is now untouched by everything on the track**, not
just "until Q8": Q8 was the only item that acted on it and there is no Q8. The
two columns (rotating and warm-identical) therefore stay 56 ms apart, and
closing them would need a new item nobody has written — the first place to look
is not a weight format but **Q12**, and after that the miss *service* path
rather than the miss *rate*. The draft's "+25 to +30% from Q1–Q6" straddled the
two columns and two branches; stated per column with one branch left it is
honest, and the rotating figure is the one the owner sees.

## Sequencing across both tracks

**State on 2026-09-05.** Done: C0, G0, G1, G1b, G2, G3, the Fable gate, G4,
G7, G8, G9, G10. Blocked: C1 (needs the `2d3cf7e` divisor guarded first);
G12 (needs a GPU kernel to be real, not a plumbing fix — skipped, not
escalated, same trigger as the original KDA-shader rejection). Not started:
C2, G5, G6, all of track Q; G11 part 1 done, part 2 closed; **G12 reopened with
a spec (second Fable gate)**. GLM rotating median went
1.65–1.66 (G0) → 1.71/1.69 (G2) → 1.84/1.83 (G4) → 2.01/2.05 (G7) →
2.19/2.17 (G8) → 2.33/2.35 (G9) → **2.60/2.75** (G10); fresh-process
decode 373.4 → **162.1 ms/token** (§RP1-CORRECTION; RP1's own 199.4 was
1.23× too high on a 93%-warm page cache). The original week-1/week-2 plan
below is retired: G3's profile replaced its rationale, and the four items
it added (G7–G10) are each cheaper than anything that was on it. **RP1
then re-sized what is left** — G11's expected saving and G5's per-context
cost both changed, and 23.5 ms/token of the measured gain turned out to be
a warmer routing histogram rather than any landed item.

**Next, in order.** Positions below are from measured evidence, not ranking
by guess; where a position is a judgment call rather than a number, it says so.

> **Read §RP1-CORRECTION before using this list.** As of 09-05 the largest
> remaining bucket is **KDA (53.3 ms/token, 32.9%)**, of which **35.0 is GPU
> submits**, not the CPU experts (40.3). That fired the revisit condition in
> `G4-KDA-SPEC-2026-09-04.md`, and **the second Fable gate has ruled: yes** —
> `G12-KDA-GPU-SPEC-2026-09-05.md`, record §"Fable gate 2". **G12 is the next
> item, Opus executes it, sized at −15 to −22 ms/token.**

1. ~~**G7 — router.**~~ **DONE 09-05** (record §G7). 0.998 → 0.137 ms/call,
   rotating 1.84/1.83 → **2.01/2.05**. Bit-identical.
2. ~~**G8 — MLA heads.**~~ **DONE 09-05** (record §G8). `[OPTIME] mla`
   5.013 → 2.066 ms/call (2.43×), rotating 2.01/2.05 → **2.19/2.17**.
   Bit-identical. Record carries a correction: the saving's absolute size
   tracks context up to the indexer's 2051-token cap, it is not flat
   everywhere — only the relative 3.08× speedup is context-independent.
3. ~~**G9 — CPU/GPU overlap.**~~ **DONE 09-05** (record §G9). `[PROF] eg`
   went from serial with `cpu` (16.249s combined) to nested inside it
   (eg=11.749s ⊇ cpu=11.193s), rotating 2.19/2.17 → **2.33/2.35**.
   Bit-identical. Beat the −22 ms/token estimate (fresh-process +13.3%).
4. ~~**G12 — the KDA recurrence on the GPU.**~~ **DONE 09-05** (record §G12
   stages 1, 2a, 2b, 2c). `[OPTIME] kda` **1.48–1.57 → 0.93 ms/call**, gate
   met; serving gate **+11.7%** (2.90 vs 2.595 rotating). Ships behind
   `COLI_KDA_GPU=2`, off by default: cosine 0.99992 at 1260 positions, cause
   irreducible (GLSL `exp()`). The gate run exposed a session-state bug the
   single-request oracles were structurally unable to see — see §Stage 2c, and
   run `tools/hot-expert/tworeq.py` for anything that relocates per-conversation
   state.

   *Original entry, kept for the reasoning that selected it:* reopened at the
   second Fable gate on the first gate's own revisit condition, which
   §RP1-CORRECTION made true (KDA 53.3 ms/token, largest; items 1–6 closed).
   Spec: `G12-KDA-GPU-SPEC-2026-09-05.md`. One recurrence shader is new;
   the chained submit is `coli_vk_attn_qprep`, ported. Behind
   `COLI_KDA_GPU`, off by default, not bit-identical — the oracle is a
   1000-step shader drift test plus long-horizon greedy identity, and the
   go/no-go is `[OPTIME] kda` 1.577 → < 1.0 ms/call. **−15 to −22
   ms/token**, 2–4 days, Opus. The earlier blocked analysis (record §G12)
   was right that it needs a kernel; what changed is that the kernel is now
   worth it and mostly already exists as a pattern.

   **The re-profile happened — see RP2, record §RP2.** −21.9 ms/token
   confirmed from outside G12's own campaign.
4b. ~~**RP2 — re-profile after G11 and G12.**~~ **DONE 09-05** (record §RP2).
   Measured in **both knob positions**, because `COLI_KDA_GPU=2` ships off:
   **knob-off 159.08 ms/token** (KDA 53.35, 33.5%, largest) and **knob-on
   137.19** (CPU int4 experts 38.28, 27.9%, largest). Confirmed G12's −21.9
   and G11 part 1's −1.99 from outside their own campaigns; routing
   bit-identical to RP1-CORRECTION, so RP1's histogram confound is gone.
   Changed the standing procedure: **assert 100% page residency before every
   run, not once per campaign** — every `glm53` process drops the model to
   91.6347% and it does not come back. `tools/hot-expert/profile_run.sh`.

4c. ~~**G13 — the shared expert's three submits.**~~ **DONE 09-06** (record
   §G13). The routed experts' own fused kernel turned out to be the wrong
   template: `qmatmul_gate_up.comp` computes `silu(gate)*up` with **no
   clamp**, and GLM-5.3's `swiglu_limit=10.0` is regularly exceeded in
   practice — wiring the shared expert through it failed `--logits`
   outright. **That gap is pre-existing in the routed-expert GPU path since
   G0-era and was never caught, because no earlier change ever diffed a
   clamped and an unclamped computation of the same op** — recorded as a
   finding, out of scope to fix here. Shipped instead: `coli_vk_matmul_pair`
   (already production code in `kimi_k3.c`), fusing only gate+up into one
   submit and leaving `swiglu_clamped` + `down` untouched. **3 submits/call
   → 2**, bit-identical (`--logits` exact on both G4 prompts, greedy text
   identical for 128 tokens). `[OPTIME] shared` **0.373 → 0.291 ms/call
   (−22%), × 42 = −3.4 ms/token**, reproduced 3× within 1%. Serving gate
   **inconclusive** at ~2.1% of the token — same situation as G11 part 1,
   and reported as such rather than rounded to a headline number the gate
   cannot actually support.

4d. ~~**SPEC-PROBE — is the AngelSpec family (DFlash/DFlash2/DFly/DFlare/
   DSpark/MTP) worth it here?**~~ **ANSWERED 09-05, no** (record §SPEC-PROBE).
   They all share one operation — verify K tokens in one forward — so the
   ceiling was measurable with **no drafter at all**, via
   `GLM53_PREFILL_CHUNK`. **The CPU expert count is 35 252 at every block size
   from 1 to 16, exactly**: the experts that miss the GPU tier are the cold
   ones, and adjacent tokens never share a cold expert. GPU expert calls over
   the same tokens dedup −49%, which is the same fact from the other side.
   Verify-of-K costs 0.912·K singles, so it needs **>91% acceptance to break
   even**; published GLM-5.3 DFlash2 acceptance is 3.85/8 = 48% → **0.53×**.
   Optimistic ceiling with a purpose-built verify: ~1.23×. **Declined — the
   drafter was never the binding constraint.**

   **Reversal condition:** this is a conclusion about the machine, not the
   technique. It rests on ~21% of expert activations streaming from system RAM.
   If the tier ever holds the whole model the linear term vanishes and this
   family becomes strongly attractive. Revisit on a VRAM capacity change.

4e. ~~**RP3 — re-profile after G13.**~~ **DONE 09-06** (record §RP3).
   **156.77 ms/token knob-off, 134.90 knob-on.** Confirmed G13 from outside
   its own campaign (shared expert −3.43 / −3.19 against its claimed −3.4),
   routing bit-identical for the third campaign running, residency landing on
   91.6347% to four decimals again on a changed binary. **The ranking is
   unchanged, position for position, in both knob positions** — G13 shrank #4
   without reordering anything. Two thirds of G13's win reached the total; the
   ~1.0 ms/token shortfall sits mostly in a CPU-expert bucket that rose ~1% in
   both configs with non-overlapping within-campaign ranges — **flagged, not
   explained**, and to be re-checked at RP4 against a different change rather
   than theorised about now.

4f. ~~**The track has run out of items the gate can measure.**~~
   **WRONG, and corrected on 09-06 — read this before trusting anything above
   it.** RP3 concluded both leading buckets were blocked and recommended a VRAM
   capacity change, i.e. buying hardware to escape the one problem this engine
   exists to solve. Colibri's premise is a large MoE in RAM with only the hot
   experts in VRAM; "the CPU expert path is VRAM-bound" is the **problem
   statement**, not a terminus.

   The error was concrete and checkable: RP3 said that bucket had been
   "examined and declined twice". It had not. §G11 part 2 declined three
   *float-domain* variants of the existing algorithm and §SPEC-PROBE declined
   speculative decoding, which is not that path at all. **Nobody had tried a
   different algorithm** — while §G3 had been saying since the first profile
   that `matmul_i4_grouped` is "ALU/decode-bound at 27% of bandwidth".
   The lesson is narrower than "re-profile more often": **a bucket is only
   blocked if what was declined is the same kind of thing as what is being
   proposed.**

4g. ~~**G14 — a different algorithm for the CPU expert kernel.**~~ **DONE 09-06**
   (record §G14). Four kernels built, each measured *in the engine* against a
   pristine binary. Only one preserved the output:

   | mode | isolated | in engine | greedy text | verdict |
   |---|---:|---:|---|---|
   | int8, `maddubs` | 1.7–2.3× | 1.32× | **differs** | rejected |
   | int16, `madd_epi16` | — | 1.075× | **differs** | rejected |
   | int8 gate/up, float down | — | — | **differs** | rejected |
   | **float, bit-trick + 4 acc** | 1.26–1.65× | **1.095×** | **identical** | **shipped**, `GLM53_I4_FAST` |

   GLM-5.3 does not tolerate quantised activations: one outlier in a group of
   64 — the thing `swiglu_limit=10.0` exists to clamp — crushes the other 63.
   The three rejected kernels were then **removed**, which was itself worth 4%
   on the survivor (64 KB of dead stack arrays in a function called ~70× per
   token). §G11 was right to decline this kernel and G14-PROBE's −16 ms/token
   projection was wrong: it extrapolated an isolated microbenchmark.

   **Two findings worth more than the kernel:**
   - **A probe can size throughput on synthetic data; it cannot size accuracy.**
     The microbenchmark reported relL2 3.9e−3 on uniform random data; the engine
     showed 0.144. Instruction mix and memory traffic survive synthetic data;
     the activation distribution does not.
   - **In situ this path is much closer to memory-bound than §G3's isolated
     figure implies.** Isolated-to-in-engine attenuation is 0.55–0.65×
     consistently, because the engine streams **985 MB/token of cold expert
     weights** out of mmap where the bench cycled 406 MB with cache hits.
     Solving `time = mem + alu` from the two measured points puts the bucket at
     roughly **43% arithmetic, 57% memory**.

4h. ~~**G15 — int3 experts.**~~ **DEAD 2026-09-11 — the probe killed it**
   (record §G15). The premise was right and the arithmetic behind it is
   unchanged: `fmt=5` takes an expert slot 13.5 → 10.5 MiB, which is 22% fewer
   bytes streamed per CPU expert (−4.8 ms/token) and ~28% more experts resident
   in the same 72 GB (−13 to −16 ms/token combined). **None of that was
   collected, because the gate this item put first says not to.**

   The probe was built exactly as scoped — simulate int3 on the int4 weights
   already on disk, no converter, no shader, no `fmt=5` kernel — and run with
   every routed expert forced onto the CPU so the ~79% the GPU tier serves from
   unmodified int4 could not mask it:

   | pair | what changes | short TF (42) | long TF (1232) | `last_logits` cos |
   |---|---|---|---|---|
   | knobs off vs pristine | nothing | identical | — | 1.000000000, relL2 0 |
   | `EXPERTS_CPU=2` vs pristine | placement only | identical | **identical** | 0.999999996 / 0.999024 |
   | `EXPERTS_CPU=1` vs `=2` | the swiglu clamp | 6 differ | 8 differ | 0.992324 / 0.981143 |
   | **`I3_SIM=1` vs `EXPERTS_CPU=1`** | **int3 alone** | **13 differ** | **16 differ** | **0.972634 / 0.877795** |

   §G12 shipped opt-in at cosine 0.99992 with **identical** greedy text; §G14
   rejected int8 activations at 0.98964. int3 is an order of magnitude past the
   rejection line, the greedy text differs at both prompt lengths, and the
   long-prompt argmax changes. **3-bit absmax over 64 weights is not enough
   resolution for this model's experts, and no kernel can fix a format.**

   Two things were proved before the verdict was accepted, so that it is a
   verdict about int3 and not about the probe. The simulation **is** int3:
   `tools/hot-expert/rome_i3sim.c` checks it against the tree's own
   `pack_int3_g64` + `matmul_i3` at relL2 1.7e−7, exhaustively over the
   nibble→level map, and the probe's one structural deviation — it double-
   quantises where a converter would go fp8→int3 once — is priced at **1.08×**
   more RMS error, nowhere near enough to matter. Knobs off is bit-identical to
   the pristine binary, whose sha256 matched the binary in service.

   **The by-product is worth more than the verdict.** §G13 found that the
   routed-expert GPU kernel computes `silu(gate)*up` with **no clamp** while the
   CPU path applies `swiglu_limit = 10.0`, and left it as a finding. This probe
   had to measure it, because the control run misbehaved until it did: placement
   alone is `teacher_forcing`-identical over 42 **and 1232** positions, and the
   clamp alone changes **6 of 42 and 8 of 1232** teacher-forced predictions plus
   the long-prompt argmax. **GLM-5.3's output today therefore depends on which
   experts happen to be tier-resident.** That is a bigger effect than several
   landed wins on this track. It is out of scope for this item and is left as
   the next real one; `GLM53_EXPERTS_CPU=1/2` is the instrument that measures it
   and stays in the tree for that reason.

   Opus, one day, spent as budgeted. **Do not re-open this without a different
   format** — finer groups, or int3 only for the cold tail of experts, both of
   which give back most of the 22%.

5. ~~**G10 — mHC.**~~ **DONE 09-05** (record §G10). `[OPTIME] hc+norm`
   0.393 → **0.097 ms/site** (4.05×), rotating 2.33/2.35 → **2.60/2.75**.
   Bit-identical, but only after reverting the malloc-removal half of the
   item — see record §G10 for the five compiler-flag mitigations tried and
   why none worked; the mallocs stay, the parallelism shipped.
6. **G5 — pooled-key cache.** Now placed on evidence rather than deferred:
   the indexer never flattens, and RP1 re-measured it at **1.958 µs per
   context token per call** (was fitted at 2.09) = 0.0215 ms/ctx-token
   across 11 layers: **2.3 ms/token at ctx 106**, 44 at 2k, 176 at 8k, 706
   at 32k. It overtakes the *entire* CPU-expert bucket at **ctx ≈ 3.5k**,
   and MLA's own attention core — the comparison §G8 was actually
   ranking — at **ctx ≈ 10k**, not the 3.5k §G8 stated (that figure used an
   idealised 8× rather than G8's measured 3.08×; RP1 corrects it, and it
   moves G5 *further out*). **Its real priority is a product question** —
   what context length does the deployment see? At 2k it is worth little;
   at 32k it dwarfs everything else in this table.
7. ~~**G11 — the int4 expert kernel and its path.**~~ **PART 1 DONE 09-05**
   (record §G11); **part 2 closed as not worth it.** The path fusion shipped
   bit-identical for −2.25 ms/token. The kernel half was microbenchmarked
   rather than assumed, and the answer was no: 1.30× on the isolated kernel
   from breaking the FMA chain, but only ~1.06× once the path is fused, and
   not bit-identical — so it would ship behind an env knob for ~1% of the
   token. **§RP1-CORRECTION is the thing to read here**: RP1's CPU-expert
   bucket was 1.86× too big (93%-warm page cache), the real bucket is
   40.3 ms/token, and **KDA at 53.3 is now the largest**.
8. **G6 — preload VRAM budget.** Not a speed item and not rankable here. Do it
   whenever the preload path is next touched, or immediately if anyone might
   run with an unset cap — it is the guard against the incident that put 91 GB
   "in VRAM" and evicted the page cache.
9. **C1** whenever someone guards the `2d3cf7e` divisor. Blocks nothing else.
10. **Track Q** has its baseline (§Q-REBASE), its profile (§Q-PROFILE) and its
   VRAM arbitration (§Q-ARB) as of 2026-09-11; its items are re-derived from
   them rather than guessed. Order: **Q3 → Q1 → Q2** (three Sonnet-days,
   bit-identical, the whole of the cheap parallelism left on that engine).
   **Q3 is merged (2026-09-12, arbitration in record §Q3): −3.85 ms/token,
   bit-identical; its gate's numeric legs had assumed full absorption of the
   shared expert and were corrected to half the bucket, met at 54%. The gap
   keeps 8.4 ms/token of idle, unevenly distributed.**
   **Q1 is merged (2026-09-12, arbitration in record §Q1): bit-identical,
   `deltanet` 70.22 → 63.87 and the token 185.86 → 179.84 (−6.02). Its
   `deltanet ≤ 60` leg was unreachable — 62.33 ms/token of `deltanet` is the
   projections plus the already-parallel recurrence, and step 0 measured the
   bucket the item names at 7.98 ms/token rather than the estimated ~15.9 —
   and the gate was corrected to half that bucket, met at 80%. Not §Q3's
   lesson: there the mechanism's yield was over-modelled, here the target
   was over-counted.**
   **Q2 is merged (2026-09-12, record §Q2 — NO arbitration, and that is the
   point): bit-identical, `gr-read` 26.86 → 21.79 (−5.08) and the token
   179.49 → 174.18 (−5.31), rotating 4.28 → 4.41, warm-identical 5.52 → 5.655.
   Its step 0 measured the named bucket at 7.62 rather than the estimated ~9.9,
   showed `gr-read ≤ 20` unreachable (floor 20.20) and wrote the corrected gate
   into the record BEFORE the candidate was built; every corrected leg was then
   met. It also rejected half of its own item on measurement — parallelising the
   four `q38_rms0` calls moves 0.35 ms/token into `q38_gr_apply` through
   cross-CCX invalidation of `hyper` for the 0.33 it gains — and logged that
   half as Q11 with the `q38_gr_apply` pragma that makes it free.** **All three
   bit-identical Track-Q items are now landed; the no-numerics floor of this
   engine is ~174 ms/token fresh-process.** Then **Q10** (the recurrence's
   state traffic, out of Q1's sub-timers; step 0 decides whether it is a
   bandwidth or a latency item) and **Q11** (half a day, −0.4 to −0.5). Then
   **QP — DONE 2026-09-12 (record §QP), and it answered all four: int4-g64
   experts DEAD (cos 0.9264, 319 of 1200 teacher-forced predictions changed) so
   **Q8 is dead**; int8 dense DEAD in both groupings (cos 0.980/0.982, below
   §G14's own rejected 0.98964) so **Q7-cpu is dead, Q7 is a BF16 arm, and Q4
   goes first**; the int8 LM head fails the gate as written on both arms by one
   leg each and is handed to Fable rather than rounded; the ballast curve
   measured and §Q-ARB's ledger arithmetic confirmed to 0.2%. Knobs off are
   bit-identical to pristine. One new item out of it: **Q12**, which subset of
   the 676 dense tensors does take int8 — the same per-tensor perturbation is
   fine on one tensor (logit relL2 7.7e−3) and fatal across 48 layers (0.198),
   so the failure is depth, not precision.** Then Q5, then the Fable session
   (Q7-gpu's BF16 spec, the QP(c) call, whether Q12 runs first), then the build
   order in §Q-ARB point 3 — whose decision table has resolved to its last row.

### Keeping the profile honest: a re-measurement cadence, not a one-off

G3 was expensive because it *built* the instrumentation from nothing —
`[OPTIME]` timers, a `perf` methodology, cross-validation to 3 ms in 47 s.
That cost is sunk: the timers are permanent, gated on `COLI_TIMERS=1`, free
in the serving regime. Reading them again is not building them again, so
"re-derive the order after three or four land" is now a standing procedure
with two speeds, not a one-line reminder:

- **Cheap, after every landed item (Haiku):** the *full* `[OPTIME]`
  breakdown, not just the before/after of the one thing that changed. This
  catches drift in buckets nobody touched and confirms the running ms
  figures in this document are still real numbers, not stale interpolation.
  Evidence this matters, from this branch: **G7 landed at exactly its
  estimate** (−36 predicted, −36.2 measured) but **G8 missed by 28%** (−45
  predicted, −32.4 measured), which by itself moved the road's projected
  end-state from ~134 to ~147 ms/token. One of two landed items already
  needed this; assume the next ones will too.
- **Expensive, at milestones (Opus — profiling and its interpretation, same
  tier as G3 itself):** a fresh `perf record` flat profile, repeated in
  full. The **before-G11** one is **DONE 09-05 — record §RP1**, and it
  earned its cost: it found that the road's own arithmetic was right to
  within a rounding error (373.4 − the five items' measured deltas = 222.9;
  measured 199.4) but that **23.5 ms/token of the gain was the routing
  histogram, not code** — the same-sized expert tier now serves 79% of
  activations from GPU where G3 saw 54%. It also **re-sized G11 downward**
  (its bucket is 74.8 ms/token, not 115, so "−60 to −80" was never
  reachable), found that **G3's barrier-spin model stopped working**
  (predicted 36%, measured 56.35%) and that **the spin is load-bearing**
  (`OMP_WAIT_POLICY=passive` costs 7.5%), and that **only 57% of the
  core-time in the CPU-expert window is in the kernel**. None of that was
  visible from the cheap `[OPTIME]` reads. **That next one is also done —
  §RP2, after G11 and G12**, and it hardened the procedure a second time
  (below).
  **RP1 then had to be corrected by exactly the repeat-measurement this
  cadence exists to force** (§RP1-CORRECTION): its CPU-expert bucket was
  1.86× too high because `fincore` was spot-checked on the tail shards
  instead of asserted across all 62 — and that one bucket was the whole
  basis for the ordering, so correcting it inverted the ranking. **Procedure
  hardened:** assert 100% residency on *every* shard; stop netdata first
  (it drives `dockerd` to poll continuously — and note `ps`'s `%CPU` is a
  lifetime average, so read `top` for the instantaneous figure); take the
  A/B paired and twice. The internal `[OPTIME]`/`[PROF]` timers hold to
  0.3% where fresh-process tok/s scatters ±10% on GPU-submit jitter, so for
  a CPU-side change the timers are the instrument and the wall-clock is not.
  **RP2 then hardened it again, by tripping over the same class of error one
  level up** (§RP2). RP1-CORRECTION's rule was "assert 100% across all 62
  shards"; RP2 obeyed it, asserted once at the top of the campaign, and was
  caught minutes later by its own guard finding the model at **91.71%
  resident, 51 of 62 shards short**. Every `glm53` process drops residency
  from 100.0000% to **91.6347%** — reproducibly, to four decimals — because
  it transiently allocates enough anon memory to force reclaim, and the
  ~15 GiB never comes back; `free` then shows 59 GiB free, so nothing looks
  wrong unless you look at the model's own pages. **RP1 checked the wrong
  files; RP2 checked at the wrong time.** Standing rule now: **re-warm and
  assert before EVERY run**, report residency after each, and use
  `tools/hot-expert/profile_run.sh` rather than re-deriving it.
  **And profile in both knob positions** whenever a shipped knob changes a
  bucket: RP2's two profiles put a different bucket at the top.

Every figure in this document comes from a machine where 69% of the decode
token was single-threaded when G3 was taken. Fix several of those buckets
and the remaining numbers shift — a plan built on stale figures is exactly
what G3 had to tear up in the first place, and letting this road quietly
become the same kind of stale plan would be the identical mistake one level
down. **RP1 is the proof that this is not hypothetical:** two of the
roadmap's own numbers (G11's expected saving, G5's per-context cost) were
stale, and one of its headline gains turned out to be partly unearned.

> **Long context was measured on 2026-09-05** — see the record's long-context
> section. Short version: the indexer really is O(context²) per generation
> (2.09 µs per context token, never flattens), but the attention core is
> larger below ~27k context and is *bounded* at 2051 tokens of work, so **G8
> beats G5 at every context length anyone is likely to run here**, and G5's
> turn arrives after G8 lands. One thing is still derived rather than
> measured: that `attn` flattens above 2k. It follows from a constant bound in
> the code; a ~4k run would confirm it, and if it does not flatten, G5 moves
> ahead of G8.

The rig serialises benchmarks, so parallel sessions must share `C0`'s script
and never benchmark at the same time; code work can overlap freely.

## What is deliberately not on the list

- AVX-512 anything (Zen 2), FP8 hardware paths (RDNA3), Tailscale.
- GPU prefill for Qwen: measured in commit `21999ff` as not helping
  prefill-bound long context; CPU AVX2 is throughput-competitive at 32 rows.
- Per-kernel thread caps and extra FMA chains in `matmul_fp8`: measured,
  rejected, code removed.
- Chasing the repeated-prompt upper bound: it is already at the tier's
  hit-rate ceiling; every gain now shows up on rotating prompts and the first
  request, so that is what the gates use.
