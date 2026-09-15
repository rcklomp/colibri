# Handoff — edge0 techniques to evaluate for Colibri

**Status:** evaluation assignment — not a roadmap item, not a "land this" order.
**Goal:** assess a set of techniques from an external MoE-inference framework, implement the best candidate(s) on **one** model, measure them, and decide keep / reject / defer with evidence.
**Model:** pick ONE to test on first — your call (guidance in "Which model" below).

## Read before you plan anything

1. `tools/hot-expert/ROME-3x7900XTX-2026-09-04.md` — the record of what was tried and rejected. **A feature missing from this tree does NOT mean it is novel.** Several techniques below are *known to already exist or have already been rejected in this repo* — the two biggest ones are called out explicitly under #1 and #2 so you do not re-derive them. For any remaining concept, grep the record + roadmaps + `git log --oneline --all` (`routing prediction`, `expert prefetch`, `prerouter`, `PILOT`, `COUPLE`, `hot pin`, `evict`, `LRU`, `LoRA`, `adapter`, `staged slots`, `double buffer`) before assuming it is new.
2. Everything else is the normal discipline (CLAUDE.md): one item → one `perf/...` branch off `hot-expert-tier`; oracle + measured delta in the commit body; `MEASURING.md` decides which harness measures which track; gate before "done". Nothing here overrides that.

## What edge0 is

[edge0](https://github.com/Edge0-AI/edge0) is an open-source streaming-MoE inference framework (MLX / Apple Silicon). Its recipe is **SSD expert offload + a trained "prerouter" that predicts routing one token ahead + Recover-LoRA** (int4 base + LoRA distilled from the FP teacher). We do not want its MLX/SSD implementation — we want the algorithmic ideas, evaluated against Colibri's Vulkan / RAM-tier streaming.

**Framing caveat that matters before anything else:** edge0's +59% decode comes from *SSD* offload, where expert-load latency is the dominant stall. Colibri streams experts from RAM tiers (dev2/dev3 preload), where load latency is far lower. **Do not expect +59% to transfer.** This is not a hypothesis — see #1: it was already measured on this rig and the gain did not materialize.

## The techniques, in priority order

### 1. Prerouter (routing prediction) — ALREADY EXPLORED AND REJECTED on throughput

**This is `PILOT` — the router-lookahead prefetch this repo already has.** `PILOT=1` (default off), `PILOT_REAL`, `PILOT_TWO`, `PILOT_K`, `WIDE`, and the coupling-table variant `COUPLE`/`COUPLE_D` are exactly edge0's "predict next-layer routing ahead of time" idea. `CHANGELOG.md` records it as 71.6% predictive. And it was measured and **rejected**: `docs/experiments/inference-paper-test-matrix-2026-07-28.md`, row **M3** — *"`PREFETCH=1` lowered replay throughput 2.5%; a cross-domain coupling table predicted almost exclusively resident experts, and bounded non-resident/real-load variants did not reduce misses or felt wait."*

That rejection is the framing caveat made concrete: on RAM-resident tiers the predicted experts were already resident, so there was no stall to hide. edge0's gain is an SSD-latency artifact. **Do not re-open the throughput angle.**

**The one genuinely-new angle worth evaluating — correctness.** The M3 rejection predates the 2026-09-11 finding (CLAUDE.md, §G15): *"GLM-5.3's output depends on which experts happen to be tier-resident."* A predictor that pins the experts *actually needed* next would reduce that residency-dependence — a correctness fix, not a speed fix. That framing has never been tested, because the correctness gap was not known when M3 ran. If you take this item, that is the question: **does a per-token prediction + pin (or a coupling table, or `PILOT`'s machinery repurposed) make routing *consistent* across tier configurations?** Measure with the `teacher_forcing` oracle + the residency-dependence A/B from §G15, not with tok/s.

**Effort/risk.** If the correctness angle doesn't pan out quickly, this is a "defer" — the throughput version is settled (rejected), don't spend a branch re-measuring it.

### 2. pin_bonus — ALREADY EXISTS

edge0's "predicted experts get a retention bonus in the hot cache" is already shipped as **`PILOT_EVICT_GUARD`** (`docs/ENVIRONMENT.md`: "keep pilot-prefetched experts from being evicted before they are used"). And the more general question — *which* expert to evict — is already superseded by **P10 value-based eviction** (`P10-VALUE-EVICTION-SPEC-2026-09.md`: evict by worth, not recency; "LRU is the wrong question here"). **Skip.** Nothing here to add.

### 3. Staged fixed-slot double-buffering — likely overlaps existing slot machinery

**What it is.** edge0's decode workhorse: routed-expert indices map to a *slot table that never leaves the GPU* (zero host sync per layer per step), fixed slots + one overflow zero-slot, slot tables and the stacked graph cached per expert set, an "incremental stack" that avoids rebuilding graph nodes.

**Why it likely doesn't apply.** Colibri already has a slot-based expert mapping (`st_map_shard_range`, "288 slots per layer over 42" in `DEV-MERGE-NOTE-2026-09-07.md`) and the `EXPERT_STORE_AUTO` / `HOT_ROWS16` machinery. Before spending time, confirm from the record + §G/§Q whether the GPU-side expert selection already avoids per-layer host round-trips. If it does, this is already-explored — stop there.

### 4. Recover-LoRA (quality recovery) — conditional, likely skip

**What it is.** Freeze the int4 base, train LoRA adapters by distillation from the FP teacher, keep adapters unmerged. edge0 recovers most of the 4-bit loss (~3.9 pts avg on their eval).

**Why it's likely not for Colibri.** It is a training/distillation pipeline, not an inference change, and it only pays off if there is a *measured* int4 quality gap. Colibri's quant-quality landscape is already well-mapped (§G15 int3 DEAD, §QP int4 probe, `I4S` IDOT numerics note). No prior LoRA attempt was found — but absent a named quality problem, this is not worth a branch. Treat as "defer unless someone names a concrete quality gap."

### 5. MoESpec / RouterKind unification — refactor only, skip

**What it is.** edge0 collapses all routing into two families (`softmax-topk` vs `sigmoid-group`) behind one `MoESpec`, with bit-identical routing math pinned by tests.

**Why it's likely not for Colibri.** You have per-model `.c` files; unifying them is refactor value with no speed delta. Only worth it if you see concrete duplication pain while doing #1. Do not spend a branch on this alone.

## Which model to test on

Your call. Guidance, not a directive: the only live candidate here (#1's correctness angle) targets **glm53** — it is the engine with the tier-residency correctness gap, the `teacher_forcing` oracle, and the `~/.glm53_explain.bin` histogram. qwen38 has no such gap noted. Pick one, test on it, report; extend only if the first result clearly warrants it.

## What to report / the decision bar

For each technique, end with a short verdict:

- **already-explored** — where, and what the record concluded (for #1 this is PILOT/M3; for #2 this is `PILOT_EVICT_GUARD`/P10). Is there a materially different angle? If not, stop there.
- **measured delta** — on the chosen model, correct harness per `MEASURING.md`, oracle passed, headline repeated twice, numbers in the commit body. For #1's correctness angle, report the residency-dependence A/B (routing consistency across tier configs), not tok/s.
- **verdict** — KEEP (merge with gate per repo rule), REJECT (with the number that killed it), or DEFER (with the precise reason).

A "not worth it" verdict with a measured number is a successful outcome — this is an evaluation, not a "land everything" order.

## Effort / ROI summary

| # | Technique | Status | First move |
|---|---|---|---|
| 1 | Prerouter | throughput **rejected** (PILOT/M3); correctness angle untested | read §G15 + M3, then test correctness framing on glm53 only |
| 2 | pin_bonus | **already exists** (`PILOT_EVICT_GUARD`, P10) | skip |
| 3 | Staged slots | likely overlaps slot mapping | confirm overlap, else skip |
| 4 | Recover-LoRA | not tried, conditional | skip unless a quality gap is named |
| 5 | MoESpec | refactor only | skip |

**Net: the only candidate worth a branch is #1 re-framed as a correctness fix on glm53. Everything else is already done, already rejected, or not worth the effort.**
