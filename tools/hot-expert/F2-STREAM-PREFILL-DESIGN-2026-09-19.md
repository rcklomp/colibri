# F2 design: chunked batched prefill with expert streaming (GLM-5.3, 2026-09-19)

Item: `FRANKEN-ENGINE-PLAN-2026-09-15.md` §8.3 row **F2**, gate as re-based by
rev 16:

> **≥ 1.6× on the 18k ladder-turn TTFT (≤ 737 s against 1 179 s) and ≥ 2.0× on
> turn 1**, A,B,B,A, oracle unchanged; the llama.cpp 108 s row is still
> reported beside it. Otherwise the item is rejected.

Everything below is either **measured** (a record section, a live rig read, or
a number computed here from the rig's own usage histogram), or **projected**
with the derivation shown. No engine was run to write this note.

---

## 0. Geometry and residency, measured

`~/models/GLM-5.3-Flash-colibri-int4-g64/config.json` (`text_config`), read on
the rig 2026-09-19:

| | |
|---|---:|
| `hidden_size` (D) | 4 096 |
| `moe_intermediate_size` (I) | 2 048 |
| `n_routed_experts` per layer (E) | 288 |
| `num_experts_per_tok` (topk) | 8 |
| `num_hidden_layers` / `first_k_dense_replace` | 45 / 3 → **42 MoE layers** |
| `swiglu_limit` | 10.0 |

One expert = gate[2048,4096] + up[2048,4096] + down[4096,2048], int4 group-64.
Weights `(I+1)/2` B/row, scales 4 B per 64 → 3 × 4.72 MB = **14.16 MB**, which
is the engine's own `slot da 14.2 MB` line. For fmt=4 with these shapes
`rowWords*4 == cpu_rb` exactly (`c/backend_vulkan.c:298` `rowwords`,
`:895` `upload_tensor`), so a ring fill is six flat `memcpy`s, not a row loop.

Tier as served (`~/glm53_server.log`, 06:55:07 today): dev0 1 248, dev2 1 695,
dev3 1 695 = **4 638 of 12 096 resident (38.3 %)**. Replaying
`vk_preload_tier`'s heat rank (`c/glm53.c:2711-2800`) against the live
histogram `~/.glm53_explain.bin` (12 960 entries, 7 754 non-zero) reproduces
exactly those three counts and gives the per-layer split:

* non-resident per layer: **mean 177.6, min 65 (layer 3), max 220 (layer 18)**;
  7 458 non-resident experts = **105.6 GB** if every one of them is touched.
* usage-weighted resident share **0.8934** → the non-resident set takes
  **10.66 % of routed calls**, not the 21 % §8.1 still quotes (that predates
  the 1 248/1 695/1 695 tier).

**The tier caps do not change in this item.** Every recorded number is at
`COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695`; the ring is carved from VRAM
the 1 695 count cap already leaves unused (§1).

---

## 1. Where the ring's VRAM comes from

Live read, 2026-09-19 07:14 UTC, gateway up, from
`/sys/class/drm/card*/device/mem_info_vram_{total,used}` (no engine run):

| DRM card | total | used | free |
|---|---:|---:|---:|
| card0 | 25.753 GB | 24.205 GB | **1.548 GB** |
| card1 | 25.753 GB | 23.203 GB | **2.550 GB** |
| card2 | 25.753 GB | 24.208 GB | **1.545 GB** |

card1 is Vulkan **dev0**: 1 248 × 14.16 MB = 17.67 GB of experts + 3.78 GB
dense (`[VK] denso residente: 551 matrici (~3.78 GB)`) + 0.624 GB KDA pool
(`[VK] KDA slot pool: 4 x 156 MB`) + the KV mirror ≈ 23.2 GB. card0/card2 are
dev2/dev3: 1 695 × 14.16 MB = 24.00 GB + ~0.2 GB of pools. This inference is
confirmed at probe time by `coli_vk_mem_budget/2/3`, which the probe prints.

These 1.5–2.5 GB are **not** the G6 reserve being violated. `COLI_VK_TIER_RESERVE_GB`
is 1.0 on dev2/dev3 and the preload stops on *whichever comes first*, count or
budget (`c/glm53.c:2756-2772`); at cap 1 695 the **count** stops it — CLAUDE.md
records that the budget would not fire until 1 752. So ~0.8 GB per expert card
sits above the reserve and ~1.0 GB is the reserve itself, and taking ring
memory out of it does not evict a single tier expert and does not change
routing. What it *does* reduce is the headroom the reserve exists to protect,
so the ring is sized conservatively and is allocated **after** the preload.

**Ring size: 48 slots per card, default, knob `COLI_PREFILL_RING_SLOTS`.**
48 × 14.16 MB = **680 MB per card, 2.04 GB across the three.** That leaves
865 MB free on dev2/dev3 and 1.87 GB on dev0 for the expert-group scratches,
which grow with the chunk (`eg_x/eg_h/eg_y`, `c/backend_vulkan.c:1243-1245`:
`total*D*4 + total*I*4 + total*D*4` = 40 KB per packed row; at chunk 512 a
device holding a third of 4 096 row-expert pairs needs ~55 MB, worst case
167 MB if it held all of them). The ring allocates once at first streamed
prefill and is never resized; `coli_vk_ring_init` returns 0 on any device
whose budget cannot take it and that device is then simply not used for
streaming (its share goes to the others) — a partial ring is a valid config,
not an error.

48 is chosen so that the common layer needs **one wave on each card**:
177.6 non-resident experts per layer spread over three cards by link share
(§2) is 86/46/46, and the worst layer (220) is 106/57/57. One wave needs
`ceil(106/48) = 3` on dev2 in the worst layer and 2 in the mean layer — see
§2 for the balanced-wave scheme that keeps every card busy in every wave.
It is also under the backend's hard `count <= 64` per submit
(`c/backend_vulkan.c:1221`).

---

## 2. The per-chunk schedule

Per MoE layer, per prefill chunk of S rows, `ffn_moe` already computes the
distinct-expert union in first-appearance order (`c/glm53.c:3318-3330`) and
already splits it into *resident* (dispatched to the device that holds it,
`c/glm53.c:3420-3452`) and *non-resident* (deferred to the CPU int4 path,
`c/glm53.c:3453-3461` / `ffn_moe_run_deferred_cpu`, `c/glm53.c:3086`). F2
changes only the second branch.

### 2.1 How many non-resident experts a chunk actually needs

Two models bracket it, both computed from the live histogram:

| | S=128 | S=512 | S=2048 |
|---|---:|---:|---:|
| distinct non-res / layer, **histogram-weighted** (`1-(1-p_e)^{8S}`) | 55.0 | 73.5 | 74.2 |
| distinct non-res / layer, **uniform-within-cold** (upper bound) | 81.5 | 163 | 178 |
| bytes / token, histogram model | 255.7 MB | **85.4 MB** | 21.5 MB |
| bytes / token, upper bound | 379 MB | **189.5 MB** | 51.7 MB |

The histogram model saturates at 74.2 because only 3 116 of the 7 458
non-resident experts have *any* recorded use; the upper bound assumes the
5 206 zero-history experts are chosen as often as the cold ones that do have
history. The truth is between, and **the spread is 2.2× on the item's headline
input.** It is not modelled further: the engine gets a counter under
`COLI_TIMERS=1` that reports the real distinct-non-resident count per
layer-chunk, and the oracle run measures it. Every projection below is given
for both ends.

The important structural fact is that the count **saturates**: the bytes a
chunk streams are nearly independent of S above ~512, so the per-token
streaming cost falls as 1/S. That is why F2 is a *chunk* item and not only a
*placement* item.

### 2.2 Card assignment

§F2-STEP0 measured, on three cards sharing one 8-thread fill pool: dev2
18.67 GB/s (its upstream link is its own), dev0 and dev3 9.96/9.99 GB/s each
(they share a link) — aggregate 38.64 GB/s. Assignment is by those weights,
**0.483 / 0.259 / 0.259** (dev2 / dev0 / dev3), so all three links finish
together. Experts are handed out in union order, largest-share card first.

### 2.3 Waves

```
nw = ceil(n_nonres / (ring_slots * n_active_cards))        # waves this layer
for w in 0 .. nw-1:
    take this wave's share of the remaining experts, split 0.483/0.259/0.259
    fill:   one OpenMP parallel-for over the FLAT (dev, slot) task list of
            all three cards  -> the 8 physical cores drive all three links at
            once, which is the configuration §F2-STEP0 measured at 38.6 GB/s
    issue:  coli_vk_expert_group_issue / _issue2 / _issue3, one per card
    take:   _take / _take2 / _take3, accumulate into out[]
```

`nw` is 1 for a mean layer at S=512 with ring 48 (73.5 ≤ 144) on the histogram
model and 2 on the upper bound (163 > 144); the worst layer (220) needs 2.
A layer whose non-resident set exceeds `ring*3` therefore simply pays a second
pass over the *same* ring slots, which is correct because a slot is refilled
only after the submit that read it has been joined.

**Overlap.** The one overlap taken in v1 is free and already has a hook: the
resident groups are issued first, and `ffn_moe`'s existing G9 gap between
issue and take (`c/glm53.c:3531-3545`, today used to run the deferred CPU
experts) fills ring wave 0 while the resident dispatches are in flight. No
double-buffering *inside* the ring: that would halve the usable slots to buy
an overlap worth ~2 % (compute is projected at 0.08–0.25 ms/token against
2.2–4.9 ms/token of streaming, §4). If the microbenchmark says compute is not
that small, the ring splits into two halves and this paragraph is wrong —
which is why the microbenchmark runs before the engine is touched.

---

## 3. The chunk size, and why it is its own A/B

`forward_prefill` reads `GLM53_PREFILL_CHUNK`, default **128**
(`c/glm53.c:4493-4495`). The plan wants 512.

**Chunk size alone is NOT bit-identical.** The reason is exact, not
statistical: `union_ids` is built in *first-appearance order over the whole
chunk* (`c/glm53.c:3322-3330`), so for a token `t` the order in which its 8
routed contributions are added into `out[t]` depends on which tokens precede
`t` **within its chunk** — and that set changes when the chunk boundary moves.
Both accumulation loops inherit that order: the CPU one
(`c/glm53.c:3104-3109`) and the GPU one (`c/glm53.c:3564-3588`, which walks
dev0's rows, then dev1's, then dev2's, in union order within each). Float
addition is not associative, so every token that is not in the first chunk
gets a reassociated 9-term sum (shared expert + 8 routed). Tokens 0..127 are
bit-identical; the rest are not.

The magnitude is one reassociation of nine same-sign-ish terms, i.e. ~1e-7
relative — far below §G15's clamp effect — but it is not zero and the argmax
can flip on a near-tie, so it gets its own oracle line and its own row, and
**the default stays 128.**

Consequence for the item: `COLI_PREFILL_STREAM=1` raises the *effective
default* chunk to 512 (still overridable by `GLM53_PREFILL_CHUNK`), and
`COLI_PREFILL_STREAM=0` changes nothing at all. One knob, one measured delta,
knob-off bit-identical. The chunk-only A/B
(`GLM53_PREFILL_CHUNK=512`, stream off) is reported separately so Fable can
see how much of the gate is chunk and how much is streaming.

---

## 4. Projected cost, against the gate

Baseline, record §RP-F6a, the 18 439-token ladder turn on the served binary
(`15462dc2093b8ad0`): **136.4 ms per new token**, of which `ffn_moe` 69.8
(51 %), `mla.attn` 28.4, `mla.index` 13.4, `kda` 13.2, `hc+norm` 4.8,
`mla.proj` 3.7, `ffn_dense` 3.1. TTFT for the turn 1 179.4 s over 8 853 new
tokens. At depth 0 (turn 1, ctx 1 287): 106 ms/token, moe 61 %, TTFT 136.5 s.

**Streaming cost at S=512**, from §F2-STEP0's measured 38.637 GB/s aggregate:

| | histogram model | upper bound |
|---|---:|---:|
| bytes/token | 85.4 MB | 189.5 MB |
| ms/token streamed | **2.21** | **4.90** |

(The record's 6.22 ms/token used the plan's 123 GB/chunk, i.e. "every
non-resident expert, every chunk". Both models here are at or below it.)

**GPU compute on the streamed experts.** Per layer-chunk at S=512 the streamed
set is 73–163 experts producing 0.1066 × 512 × 8 = 437 row-expert pairs
(2.7–6.0 rows per expert). The kernel is weight-bandwidth-bound: it reads
73–163 × 14.16 MB = 1.03–2.31 GB of VRAM per layer-chunk, split over three
cards at ~900 GB/s each → 0.38–0.86 ms per layer-chunk at peak, 42 layers →
16–36 ms per chunk = **0.031–0.070 ms/token at peak, 0.08–0.25 ms/token at a
realistic 30–40 % of peak.** This is the number the microbenchmark
(`tools/hot-expert/f2_ring_probe.c`) measures rather than assumes; it is the
single assumption that, if wrong by 50×, kills the item.

**Fixed per-wave overhead**: 1–2 waves per layer-chunk, each three
`issue`/`take` pairs; at ~1 ms of fence/submit per wave that is 42–84 ms per
512-row chunk = **0.08–0.16 ms/token**.

**Projected 18k token** (everything but `ffn_moe` unchanged; `ffn_moe` becomes
stream + compute + wave overhead + the resident groups and shared expert it
already contains, which the `moe split: router= shared=` line will size):

| | ms/token | × vs 136.4 |
|---|---:|---:|
| upper-bound stream, realistic compute (4.90 + 0.25 + 0.16 + 5 residual) | **77.1** | 1.77× |
| histogram stream, realistic compute (2.21 + 0.25 + 0.16 + 5 residual) | **74.2** | 1.84× |
| gate | ≤ 85.25 | 1.60× |

"5 residual" is the part of today's 69.8 that is *not* the non-resident CPU
path — the resident-tier dispatch, the shared expert, the router, and the host
accumulate — held at its present value. The margin against the gate is
**8–11 ms/token**, i.e. the item passes only if that residual is under ~13
ms/token. §RP-F6a's follow-up rows put `ffn_moe` at 2.9–3.0 ms/call in a
decode-shaped window where the non-resident path is also active, so a 5
ms/token residual in a 512-row chunk is plausible but **unproven**, and it is
the second thing the oracle run measures (`moe split` + `[PROF] eg=/cpu=`).

**Turn 1** (106 ms/token, moe 61 %): 106 − 64.7 + (2.2…4.9 + 0.3 + 3) =
**43.8–46.5 ms/token → 2.28–2.42×**, against the 2.0× gate. Turn 1 has 1 287
tokens = 3 chunks of 512, so the saturation argument is weaker there (S=512 is
where the distinct-count curve has just flattened); the upper-bound column is
the one to believe.

**Verdict of this note: the gate can be met, but not with room to spare, and
it is met only if (a) the streamed-expert GPU compute is ≤ ~1 ms/token and
(b) the non-CPU residual of `ffn_moe` is ≤ ~13 ms/token.** Both are measured
before the engine is touched (a: the microbenchmark; b: a `COLI_TIMERS=1`
read of the served binary). If either fails, F2 is reported as not meeting its
gate and this note says so rather than being tuned.

---

## 5. The oracle

The gap: `c/shaders/qmatmul_gate_up.comp:141` computes `silu(g)*u` with no
bound; the CPU path applies `swiglu_limit = 10.0` (`coli_v4_swiglu`,
`c/deepseek_v4.c:1713`). Moving a non-resident expert from CPU to GPU
therefore changes its numerics *by design*, and §G13/§G15 already quantified
that gap in isolation (6 of 42 and 8 of 1 232 `teacher_forcing` predictions).

**Decision: do not add the clamp to the streamed path's shader.** Three
reasons, in order of weight:

1. A streamed expert must match the **resident GPU** path, not the CPU path.
   Adding a clamp to streamed experts only would make an expert's numerics
   depend on whether it happened to land in the ring or in the tier — a
   *second* instance of the §G13 defect ("GLM-5.3's output depends on which
   experts are tier-resident"), not a fix for it.
2. `qmatmul_gate_up.comp` and its tiled sibling are shared with `qwen38-vk`
   (`coli_vk_gate_up`). A seventh push-constant field forces a re-measure of
   the other engine for zero numerics benefit here.
3. The clamp gap is a named open correctness item on this engine (CLAUDE.md,
   §G13). F2 must not half-fix it in one path and leave the other.

**The reference that isolates placement already exists and costs nothing.**
`GLM53_EXPERTS_CPU=2` is "every routed expert on the CPU, *unclamped*"
(`c/glm53.c:3299` `rlimit = experts_cpu_on() == 2 ? INFINITY : swiglu_limit`),
built for exactly this attribution in §G15. So:

| # | comparison | expectation | what it proves |
|---|---|---|---|
| **O1** | `COLI_PREFILL_STREAM=0` (default) vs `~/bench/glm53.f2base` (the served binary, `15462dc2093b8ad0`) | **bit-identical**: `teacher_forcing` identical, last-token logits identical | the knob is inert when off |
| **O2** | `COLI_PREFILL_STREAM=1` vs `COLI_PREFILL_STREAM=0`, same binary | differs; report cosine / max-abs / argmax-agreement | the shipping delta (placement **+** clamp, on 10.7 % of calls) |
| **O3** | `COLI_PREFILL_STREAM=1` vs `GLM53_EXPERTS_CPU=2`, same binary | agreement to float tolerance | **placement only, no clamp on either side** — the streamed kernel is correct |
| **O4** | `GLM53_PREFILL_CHUNK=512` (stream off) vs default 128 | differs by reassociation only (§3); report cosine / max-abs / argmax | the chunk change, separately |

O1 and O4 run on a short prompt and on a ≥ 4 000-token prompt. O2/O3 run on
both. `GLM53_LOGIT_DUMP_ALL` exists in the served binary, so the pristine copy
`~/bench/glm53.f2base` can produce the every-position dump O1 needs. All four
go through `gate_compare` from `gate_lib.sh`, which refuses an empty
comparison.

---

## 6. What is built

* `c/backend_vulkan.{c,h}`: a ring API, shared by the three device contexts
  and deliberately independent of `glm53.c` so F1 can reuse it at batch 1.
  `coli_vk_ring_init(dev, slots, fmt, D, I, gs)`,
  `coli_vk_ring_slots(dev)`, `coli_vk_ring_tensors(dev, i, &g, &u, &d)`,
  `coli_vk_ring_fill(dev, i, gw,gs, uw,us, dw,ds)` (thread-safe for distinct
  `(dev, i)`; a flat `memcpy` when `rowWords*4 == cpu_rb`, which is the case
  for these shapes), `coli_vk_ring_free()`, `coli_vk_ring_bytes(dev)`.
  Nothing in the existing paths calls any of it, so `qwen38-vk` is unchanged
  by construction (no shader change, no push-constant change, no change to
  any function it calls).
* `c/glm53.c`: `COLI_PREFILL_STREAM=1` (off by default) routes the
  non-resident branch of `ffn_moe` through the ring **only when
  `tokens > 1`** (prefill; decode is untouched and is F1's item), plus a
  `COLI_TIMERS=1`-only counter of distinct non-resident experts per
  layer-chunk.
* `tools/hot-expert/f2_ring_probe.c`: the microbenchmark, §4's assumption (a).
* `tools/hot-expert/f2_gate_chain.sh`: oracle phase, then the A,B,B,A ladder
  (`F2_REUSE_ORACLE=1` to run phase 2 alone; `F2_LADDER_STEPS` for a short
  first rung to ~4.5k before the multi-hour 18k one).

## 7. What is deliberately not built

* No double-buffered ring (§2.3) until compute is measured.
* No clamp in the shader (§5).
* No change to `COLI_VK_EXPERTS2/3`, to `COLI_VK_TIER_RESERVE_GB`, or to the
  default `GLM53_PREFILL_CHUNK` (§0, §3).
* No decode path change (F1).
