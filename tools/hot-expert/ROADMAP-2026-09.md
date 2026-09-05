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
| C1 | **BLOCKED** 09-04 | Merge `perf/rome-cpu-path` into `hot-expert-tier`. **Gate failed** (record §C1): `test_qwen38_prefix` passes on `hot-expert-tier` and SIGFPEs on the branch — bisected to `2d3cf7e`, unguarded `m->max_t / c->idx_ratio` at `qwen38_core.h:2417`, deterministic 3/3. Also `qwen38-tiny-check` cannot run here at all (needs torch, absent). Branch is now ~19 commits, not five. **Unblock:** guard the divisor, then re-run | Opus (review) | not met; do not merge until it is |
| C2 | not started | Add `qwen38` to the record's steady-state table for GLM as well (G0 below) so both engines have the same four numbers | Haiku | table filled |

## Track G: GLM-5.3 — 2.60–2.75 tok/s rotating (was 1.65 at G0), 8 threads, 3 GPUs

Fresh-process decode is **162.1 ms/token, down from G3's 373.4 (2.30×)** —
re-profiled 09-05 (**§RP1**) and then **corrected** (**§RP1-CORRECTION**),
because RP1's own CPU-expert bucket was inflated 1.86× by a page cache that
was only 93% warm. Read **§RP1-CORRECTION first**: it inverts the ranking.
**KDA (53.3 ms/token, 32.9%) is now the largest bucket, not the CPU experts
(40.3, 24.8%)**, and 35.0 ms/token of KDA is GPU submits.

Status 2026-09-05: G0–G4, G7, G8, G9, G10 and the before-G11 re-profile
(RP1) done; G5/G6/G11 open, G12 **done** (reopened after Fable gate 2 and landed — record §G12). The old header number (3.17
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
| G5 | Cache the pooled DSA-indexer block keys in `sparse_index.h`. **Port the idea from Qwen `2d3cf7e`, not the code** — that is the commit C1 bisected a SIGFPE to (unguarded `m->max_t / c->idx_ratio`); a naive mirror carries the bug into GLM, whose `index_kpool`/`index_topk` can likewise be 0 in a synthetic config. Guard the divisor. Confirmed applicable 09-05: `sparse_index.h:90` re-pools **the entire prefix on every call**, and `mla_layer` is called once per decode token — O(context) per token, O(context²) per generation, exactly Qwen's shape | the only O(context²) component; Qwen's fix cut its index phase 66% at 1.6k tokens | RP1 re-measured it at **1.958 µs per context token per call** (11 calls/token = 0.0215 ms/ctx-token): **2.3 ms/token at ctx 106 (1.1%)**, 44 at 2k, 176 at 8k, 706 at 32k. Overtakes the *entire* CPU-expert bucket at **ctx ≈ 3.5k** | 1 day | Sonnet | teacher_forcing identical at 690 and 1642 tokens; the dsa-index timer now exists (`[OPTIME] mla split`, added for the long-context section) |
| G6 | Make the dev2/dev3 preload loops stop on the VRAM budget, not only on a count cap | an unlimited cap put 91 GB "in VRAM" and evicted the page cache | safety, not speed | half day | Sonnet | `COLI_VK_EXPERTS2` unset fills to budget − reserve and no further |
| G7 | **DONE** 09-05 (record §G7) — parallelised the 288-row router score, inner (expert) loop not outer (token) loop: `score` has no per-token dimension, so parallelising over tokens would race. `[OPTIME] moe split: router` 0.998 → **0.137 ms/call** (7.3×). Rotating 1.84/1.83 → **2.01/2.05**. Bit-identical both prompts | G3: 41 ms/token, 0.98 ms/call, single thread, scalar reduction GCC will not vectorize; 1.2 GMAC/s | −36 ms/token; got exactly that | done | Sonnet | bit-identical: **met**; tok/s vs G4: **met** (+9–12%) |
| G8 | **DONE** 09-05 (record §G8) — parallelised both per-head loops. Checked nested-OMP oversubscription empirically before writing anything (a compiled probe: `max_active_levels=1` on this box, so nesting collapses safely). Per-thread `score`/`pooled` scratch, same class of fix as G4's KDA. `[OPTIME] mla` 5.013 → **2.066 ms/call** (2.43×). Rotating 2.01/2.05 → **2.19/2.17**. Bit-identical both prompts | G3: 55 ms/token at 151 tokens of context, 5.0 ms/call, single thread, **O(context)** (4.1 ms at 87 tokens) | −32.4 ms/token at ctx=88; correction in the record re: context-shape | done | Sonnet | bit-identical: **met**; tok/s vs G7: **met** (+6–9%) |
| G9 | **DONE** 09-05 (record §G9) — CPU-only experts now saved into a deferred list and run once in the issue/take gap (`can_defer = g_vk_ready && n_union <= block`; always true for decode). `[PROF]` eg/cpu went from serial (2.637s+13.612s=16.249s) to nested (eg=11.749s ⊇ cpu=11.193s) — the ~2.6s of pure GPU wait is almost entirely hidden inside CPU compute that was already larger than it. Rotating 2.19/2.17 → **2.33/2.35**. Bit-identical both prompts | G3: GPU groups 22 ms/token of pure wait; CPU experts 115 ms run *before* issue today | −22 ms/token; beat it (fresh-process +13.3%, gate +7–8%) | done | Sonnet | bit-identical: **met**; tok/s vs G8: **met** (+7–8%) |
| G10 | **DONE** 09-05 (record §G10) — parallelised the ~400k-MAC mix and the destination×column combine over independent rows/columns, each one's own reduction untouched (G7's pattern). **The malloc removal did not survive contact**: dropping `coli_hc_pre`'s two per-call `malloc`s needed `hc`'s range knowable at compile time, which measurably changed GCC's rounding in the small hc-bounded reductions (~1.9e-4 on `last_logits`, `teacher_forcing` unaffected) — bisected across five different compiler-flag mitigations, none restored bit-identity; **kept the mallocs, shipped the parallelism**. `[OPTIME] hc+norm` 0.393 → **0.097 ms/site** (4.05×). Rotating 2.33/2.35 → **2.60/2.75**. Bit-identical both prompts. Shared header (DeepSeek V4): rebuilt clean, not re-measured (no rig checkpoint) | G3: 35 ms/token, 0.39 ms/site × 90 sites, single thread | −28 ms/token; beat it (−26.6 ms/token measured, +11–17% gate) | done | Sonnet | bit-identical: **met** (mallocs kept); tok/s vs G9: **met** (+11–17%) |
| G11 | **PART 1 DONE** 09-05 (record §G11) — expert path fused: one OMP region with three worksharing constructs instead of three regions with a serial swiglu between, bit-identical, **−2.25 ms/token on `ffn_moe` (−3.2%)**, ~140 fewer OMP regions/token. **Part 2 (the kernel) is NOT worth doing as scoped** — see §RP1-CORRECTION. The bucket is **40.3 ms/token, not 74.8 and not 115**. Microbenchmarked before touching the engine (`g11_bench.c`, `g11_path.c`, both sized to defeat the 128 MB L3): the isolated kernel does 21.95 GB/s at 8 threads; a bit-trick nibble→float decode is *exactly* bit-identical but only 1.03×; a second accumulator breaks the FMA dependency chain for **1.30×** but is not bit-identical; four accumulators is worse than two. **Once the path is fused the faster kernel is worth only ~1.06×** — so nothing numerics-changing shipped | §RP1-CORRECTION: **40.3 ms/token, and no longer the largest bucket — KDA is, at 53.3** | part 1 delivered **−2.25 ms/token**; part 2 judged **not worth 2–3 days** for ~1.06× on a 40 ms bucket behind an env knob | part 1 done | Opus | bit-identical both prompts: **met**; `[PROF] cpu` −4.0% across 4 paired runs (fresh-process tok/s cannot resolve 1%: ±10% GPU-submit jitter) |
| G12 | **DONE 09-05 (record §G12 stages 1, 2a, 2b, 2c).** The KDA recurrence, its gating and its output norm run on dev0, and a layer's projections + recurrence + `ko` record into **one submit instead of two** (`COLI_KDA_GPU=2`). **`[OPTIME] kda` 1.48–1.57 → 0.93 ms/call — the < 1.0 gate is met**, repeating to three decimals; **−18.7 to −21.9 ms/token**, inside the spec's own −15 to −22 band; fresh-process +13.5%; **serving gate +11.7%** rotating (2.90 vs 2.595, paired alternating) with warm-identical agreeing at +11.1%. **The gate also found a shipping bug the numerics oracles could not**: `coli_vk_kda_init` kept the previous conversation's recurrence across sessions, so request N+1 of a warm engine continued request N — visible only as an *inverted* warm-identical column, because every oracle here runs one request per process (record §Stage 2c; `tools/hot-expert/tworeq.py` is now the oracle for that class). Greedy text identical through 128 decode tokens and 0 `teacher_forcing` mismatches across 1260 prefill positions. **Stays off by default**: the logit cosine is 0.99992 at 1260 positions and the cause is irreducible — GLSL `exp()` vs libm `expf()`, ~1.1M calls/token; matching the CPU's norm-reduction order exactly was tested and changed nothing (2.6e-7) while costing 3.6% | §RP1-CORRECTION: KDA was the largest bucket at 53.3 ms/token, 35.0 of it GPU submits | **−18.7 to −21.9 ms/token, delivered** (opt-in) | 3 commits | Opus | `kda` < 1.0 ms/call: **met at 0.93**; greedy identity: **met** |

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

## Track Q: Qwen3.8 (today 5.35 repeated / 3.69 rotating / 3.59 cold, 3 GPUs, 8 threads)

Two families: placement items that finish the branch's work, and the tech
report's mechanisms that Colibri does not use yet.

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| Q0 | Populate the whole mapping at load (knob, default on when the checkpoint fits RAM) and capture a longer routing history so the preload fills VRAM (14,673 of ~14,700 slots; histogram had 21,858 hot experts) | rotating-prompt CPU number did not move with prefault-at-bind: first-touch cost relocates, it does not vanish | rotating ≈ cold-request gap closes; ~+10% rotating | 1 day | Sonnet | rotating median vs today's 3.69 |
| Q1 | Fuse the gated-residual read and write into one pass each with the group RMSNorm folded in (tech report §2.2, "traversed once per block in each direction") | `q38_gr_read` makes several passes over the 4×2560 widened stream plus three small matmuls, ×2 per layer ×48 | −10 to −20 ms/token | 1–2 days | Opus | logits within 1e-5 of pristine, greedy text identical |
| Q2 | FP8 storage for the widened residual stream (tech report §2.2: "halves the bytes moved, almost no loss in quality") | the residual is FP32 today; the report says the gates bound its range | −10 to −15 ms/token of memory traffic; **changes numerics** | 1–2 days | Opus | env-gated; report the logit cosine and the tiny-check result; ship off by default unless within the near-tie tolerance |
| Q3 | Dense set off DRAM: a BF16 `fmt` in `qmatmul.comp` (numerics-preserving) and a per-layer hybrid where the dense projections run on the GPU and DeltaNet/QSA stay on the CPU | resident BF16 114–137 ms/token at the CPU's 100 GB/s floor; GPU int8 GEMV 132–400 GB/s effective; submit+fence 55 µs; 4 tensors per submit measured | −70 to −90 ms/token → ~5 tok/s rotating | design 1 session + 3–5 days | **Fable** (design: what moves, submit plan per layer, VRAM split with the expert tier) then Opus (implement) | logits within 1e-5 (BF16 fmt) ; rotating median; VRAM accounting in the record |
| Q4 | MTP speculative decoding: load the MTP module, draft up to 4 tokens per step reusing the QSA top-k indices, verify in one forward, accept the matching prefix (tech report Table 4: 4.06 mean accepted) | decode is bandwidth-bound: the dense 8.5 GB is read once per verify step, experts ~4×; ≈ 2.4× tokens per second at equal bytes | the largest multiplier; measured only by accepted tokens per second on rotating prompts, never on a repeated prompt | spec 1 session + 5–8 days | **Fable** (spec: tensor contract from the checkpoint, verify semantics for DeltaNet state rollback and QSA index reuse, acceptance metric) then Opus (draft head + verify path) and Sonnet (harness: accepted-length histogram, identity of the accepted text against plain greedy) | accepted text == greedy text for 1k tokens; accepted length ≥ 3.5 on the datapoint prompts; rotating tok/s |
| Q5 | 4-accumulator BF16 matmul for prefill rows (S>1 only) | 9.0 → 5.1 ms at S=32 in isolation | TTFT, not decode | half day | Sonnet | greedy text identical; TTFT on the 690-token document |
| Q6 | (later) int8/int4 expert conversion for VRAM density | fp8-emul 1.7× slower than int8 on the GPU; halves VRAM per expert | ~−9 ms/token plus a larger resident tier | 3+ days, second checkpoint on disk | Opus | numerics vs tiny-check; hit rate on rotating |

Order: Q0 and Q5 first (cheap, Sonnet), Q1 next (Opus, bit-close), then the
Fable design session for Q3 and Q4 together (they compete for VRAM and for
the per-layer submit budget, so they must be designed against each other),
then Q3 implementation, then Q4. Q2 slots anywhere after Q1 but ships off
by default until its numerics are recorded.

Target for the track: ~5 tok/s rotating from Q0–Q3, then ×2 or better from
Q4 measured as accepted tokens per second.

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

   **Next item is G5/G6, or a re-profile:** KDA was the largest bucket at 53.3
   ms/token and G12 took ~19–22 of that out, so the ranking that ordered this
   list is stale again. RP1's own lesson was that the ordering, not the item,
   is what a re-profile buys.
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
10. **Track Q** is untouched. Q0 and Q5 are a day between them.

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
  visible from the cheap `[OPTIME]` reads. One more is scheduled: **after
  G11 lands**, before ordering G5 against whatever is left.
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
