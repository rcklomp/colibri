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

| id | item | tier | gate |
|---|---|---|---|
| C0 | `tools/rome_bench.sh`: one script that drops caches, warms **one** model, verifies residency, pins 8 threads, sets the GLM caps (`COLI_VK_EXPERTS2/3=1695`, `.glm53_explain.bin`), runs `datapoint.py`, and appends a row to the record | Sonnet | two consecutive runs of the same config within 3% |
| C1 | Merge `perf/rome-cpu-path` into `hot-expert-tier` (five commits, all measured) | Opus (review) | `qwen38` tiny-check and the four C unit tests that pass today still pass; `glm53` rebuilds |
| C2 | Add `qwen38` to the record's steady-state table for GLM as well (G0 below) so both engines have the same four numbers | Haiku | table filled |

## Track G: GLM-5.3 (today 3.17 tok/s fresh-process, 8 threads, 3 GPUs)

Ordered by expected gain per unit of effort. G1 and G2 are ports of what
already landed for Qwen; G3 is the fork in the road; G4 is the real project.

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| G0 | Persistent-engine baseline (`datapoint.py --engine glm53`): cold, warm-identical, rotating | GLM has only fresh-process numbers | none; it is the yardstick | half day | Haiku | four numbers in the record |
| G1 | Prefault mmap'd experts at bind, and a populate-whole-mapping-at-load knob, ported from `qwen38_core.h` (`q38_populate_range`) into `glm53.c`'s `expert_read` mmap path | Qwen: fault stalls made the CPU expert path 4–6× slower than its kernel; GLM's CPU share is ~1/3 of its experts | first request and rotating prompts; unknown magnitude, measure | 1 day | Sonnet | teacher_forcing identical; rotating median and cold TTFT vs G0 |
| G2 | Check whether `glm53` dispatches dev0/dev2/dev3 expert groups sequentially; if so, issue-all/take-all with the CPU share in between, as in `q38_moe_decode` | Qwen: 2.37 vs 0.63 ms per layer | 0 if already concurrent; up to −50 ms/token if not | half day to 1 day | Sonnet | teacher_forcing identical; tok/s vs G0 |
| G3 | Per-op profile of GLM decode on this box: `perf` flat + the `[ATTN]`/`[PROF]` timers, split KDA / MLA / expert group / CPU experts / dense | Sep 2 profile: KDA 3.25 ms × 2074 calls ≈ 110 ms of a 300 ms token; MLA 64 ms | none; it decides G4 | half day | Opus | a table like the Qwen one in the record |
| **gate** | Fable reads G3 and picks the KDA approach: (a) AVX2 vectorize the scalar inner loop (per token × head × position over L=512), (b) move KDA's recurrence to the GPU via a new shader, or (c) both, split by layer | the audit called KDA "deepest bottleneck, big effort" | — | one session | Fable | a one-page spec with the numerics oracle named |
| G4 | Implement the chosen KDA path | G3 | the largest GLM item; if KDA halves, ~−55 ms/token → ~3.8–4.0 tok/s | 2–4 days | Opus | teacher_forcing within the documented near-tie tolerance; per-op timer for KDA before/after |
| G5 | Cache the pooled DSA-indexer block keys in `sparse_index.h` (mirror of Qwen commit `2d3cf7e`) | the only O(context²) component; Qwen's fix cut its index phase 66% at 1.6k tokens | nothing at short prompts; matters at 8k+ | 1 day | Sonnet | teacher_forcing identical at 690 and 1642 tokens; qsa/dsa-index timer |
| G6 | Make the dev2/dev3 preload loops stop on the VRAM budget, not only on a count cap | an unlimited cap put 91 GB "in VRAM" and evicted the page cache | safety, not speed | half day | Sonnet | `COLI_VK_EXPERTS2` unset fills to budget − reserve and no further |

Target for the track: from 3.2 to about 4 tok/s on rotating prompts, with
G4 carrying most of it. If G3 shows KDA is already near its bandwidth floor,
stop after G5 and say so.

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

- **Week 1 (Sonnet/Haiku, can run as parallel sessions on the rig one at a
  time):** C0, C2, G0, G1, G2, Q0, Q5. Each is a day or less and each has a
  numeric gate.
- **Week 2 (Opus):** C1 merge; G3 profile; Q1; G5; G6.
- **Fable session (one):** read G3, pick the KDA approach; write the Q3 and
  Q4 designs against each other. Output: three one-page specs with oracles.
- **Weeks 3–5 (Opus, two parallel lines because the code paths are
  disjoint):** G4 on `glm53.c`; Q3 then Q4 on `qwen38_core.h` +
  `backend_vulkan.c` + `qmatmul.comp`.
- **Fable review (one short session):** only if a measurement contradicts a
  spec, or before merging Q4.

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
