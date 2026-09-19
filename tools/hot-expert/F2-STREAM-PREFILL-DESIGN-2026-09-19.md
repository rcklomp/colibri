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

## 1. Where the ring lives (amended 2026-09-19 after the microbenchmark)

**The ring is in HOST memory and costs no VRAM at all.** This section was
written the other way round — a 48-slot ring per card carved out of the VRAM
the 1 695 cap leaves unused — and the microbenchmark refuted it before a line
of engine code was written. The measurement is in §4b; the short version is
that a CPU `memcpy` into ReBAR write-combined VRAM runs at 11.9–13.1 GB/s per
card and 16.3 GB/s on three, while the same `memcpy` into cached DRAM runs at
48.0–58.4 GB/s per card and 58.6 GB/s on three — and the GPU reading that DRAM
over PCIe costs 52 GB/s, which more than pays the difference back. End to end
one wave of 144 experts (2.04 GB) takes **74.3 ms with the ring in host RAM
against 102.6 ms with it in VRAM**.

So the VRAM accounting below is no longer a constraint on the item. It is kept
because it is the evidence for why the VRAM ring was tried at all (it fits,
comfortably), and because it is the thing a future reader will want when F1
asks the same question at batch 1.

### 1.1 The VRAM the ring would have used, and did not

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

The probe confirmed a 48-slot VRAM ring fits: 0.683 GB per card, leaving
0.86 GB free on dev2/dev3 and 1.87 GB on dev0, with the gateway up and the
tier untouched. It is simply slower than putting the same ring in DRAM.

### 1.2 What the ring actually costs

**Two banks of `COLI_PREFILL_RING_SLOTS` (default 32) slots per card, in host
RAM**: 64 slots × 14.16 MB = 0.91 GB per card, **2.72 GB across the three**,
out of 241 GB MemAvailable with a 183 GB model in the page cache. VRAM cost:
zero, so the tier, the G6 reserve and the 1 695 caps are untouched by
construction, not by arithmetic. The only real cost is page cache: 2.72 GB of
the ~58 GB of slack above the model.

Two banks, not one, because the fill and the compute now use **different
resources** — CPU stores into DRAM against GPU DMA reads out of DRAM — so wave
w+1's fill can run while wave w's dispatch is in flight. With a VRAM ring both
legs were the same PCIe write path and overlapping bought nothing; that is why
the first draft of §2.3 said not to bother, and why it is now wrong.

32 slots per bank is chosen against the measured non-resident counts (§2.1):
the mean layer at chunk 512 needs 73.5–163 distinct non-resident experts,
which over three cards by link share is 25–54 each — one wave in the
histogram case, two in the upper-bound case, so the double buffering is
exercised either way. It is also under the backend's hard `count <= 64` per
submit (`c/backend_vulkan.c:1221`). `coli_vk_ring_init` returns however many
slots it got; a device with none is simply not used for streaming and its
share goes to the others.

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

**Overlap (rewritten after §4b).** Two overlaps, both taken:

1. Wave 0's fill runs in the gap G9 already opened between issuing the
   resident groups and joining them (`c/glm53.c`, the `first_round` hook that
   today runs the deferred CPU experts). A ring fill is a `memcpy` into mapped
   host memory and touches no Vulkan object, so it is safe with three groups
   in flight.
2. Wave w+1's fill runs while wave w is in flight, alternating banks. With the
   ring in host RAM this is worth real time, not 2 %: the fill is 58.6 GB/s of
   CPU stores to DRAM and the dispatch is 52 GB/s of GPU reads from DRAM, and
   serially they measured 27.4 GB/s. The ceiling of the overlap is DRAM
   bandwidth (91.6 GB/s read, §PCIE-STREAM) shared between the two, so the
   gain is real but under 2×; the projection below uses the **serial** 27.4
   GB/s and treats anything better as margin.

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

~~The magnitude is one reassociation of nine same-sign-ish terms, i.e. ~1e-7
relative — far below §G15's clamp effect — but it is not zero and the argmax
can flip on a near-tie.~~

**That sentence was wrong and the oracle said so (record §F2b, O4) — though not
in the way the first reading of O2/O4 suggested; see §F2b's O5 row for the
correction.** Per token
the error *is* one reassociation of nine terms, but it feeds the next layer and
the next position, and it compounds: at 1 064 tokens the chunk change is
invisible (`teacher_forcing` identical, max_abs 1.9e-3), and at **6 327 tokens
it moves 205 of 6 327 argmaxes (3.24 %), max_abs 7.24, cosine 0.995** — *more*
than the swiglu-clamp gap that streaming itself brings. The depth a
reassociation is checked at is load-bearing; a shallow check here would have
reported "free". It gets its own oracle line and its own row, and **the default
stays 128.**

Consequence for the item: `COLI_PREFILL_STREAM=1` raises the *effective
default* chunk to 512 (still overridable by `GLM53_PREFILL_CHUNK`), and
`COLI_PREFILL_STREAM=0` changes nothing at all. One knob, one measured delta,
knob-off bit-identical. The chunk-only A/B
(`GLM53_PREFILL_CHUNK=512`, stream off) is reported separately so Fable can
see how much of the gate is chunk and how much is streaming.

---

## 4a. The microbenchmark (measured 2026-09-19, the number the item lives on)

`tools/hot-expert/f2_ring_probe.c` through `f2_ring_probe_chain.sh`
(`run_chain.sh`, rig lock only, gateway up and idle before and after, 8 threads
pinned, tag `f2rp_09190737`, output `~/bench/f2_ring_out/f2rp_09190737.txt`).
It links `c/backend_vulkan.c` and calls `coli_vk_ring_*` and
`coli_vk_expert_group_issue/2/3` — the code the engine runs, not a model of it.
48 slots per card, three cards, 144 experts = 2.04 GB per wave, real GLM shapes,
source bytes read from the served model's shards at fixed-seed random offsets.
GB = 1e9.

| | ring in ReBAR VRAM | ring in host RAM |
|---|---:|---:|
| fill, dev0 alone | 11.938 | 47.969 |
| fill, dev2 alone | 12.699 | 58.396 |
| fill, dev3 alone | 13.116 | 58.434 |
| **fill, three cards** | **16.260** | **58.565** |
| compute only, 1 row/expert (ms) | 2.513 | 38.937 |
| compute only, 3 rows/expert (ms) | 2.961 | 39.428 |
| compute only, 6 rows/expert (ms) | 4.864 | 39.973 |
| compute only, 12 rows/expert (ms) | 10.206 | 51.944 |
| **wave = fill + compute, 3 rows (ms)** | **103.718** | **74.687** |
| wave, effective GB/s | 19.65 | **27.29** |

Three things this settles.

1. **A CPU store into write-combined VRAM is not a GPU DMA read over PCIe.**
   §PCIE-STREAM's 61–62 GB/s and F0b's 59.9–63.3 GB/s are the GPU pulling; the
   CPU pushing over the same link manages 16.3 GB/s on three cards. The
   design's §1 was built on the wrong one of those and is corrected above.
2. **The host ring wins end to end by 1.39×** (74.7 ms vs 103.7 ms per wave)
   *and* costs no VRAM. Its compute leg is 39 ms — the shader reading 2.04 GB
   of host memory over PCIe at 52.4 GB/s — which is 16× the VRAM-resident
   compute but is paid back twice over by the fill.
3. **Compute is flat in rows up to 6 and only then starts to grow** (38.9 →
   40.0 → 51.9 ms at 1 → 6 → 12 rows). The schedule gives a streamed expert
   2.7–6.0 rows in a 512-row chunk, i.e. exactly the flat part: the streamed
   experts are weight-bandwidth-bound, not arithmetic-bound, and adding rows
   to a chunk is close to free for them. That is the second reason the chunk
   wants to be large.

The serial wave rate **27.4 GB/s** is what §4b projects with. The engine
double-buffers (§2.3), whose ceiling is the two legs' own rates (58.6 and
52.4) capped by DRAM bandwidth (91.6 GB/s read, §PCIE-STREAM); anything the
overlap buys is margin, not assumed.

---

## 4b. Projected cost, against the gate

Baseline, record §RP-F6a, the 18 439-token ladder turn on the served binary
(`15462dc2093b8ad0`): **136.4 ms per new token**, of which `ffn_moe` 69.8
(51 %), `mla.attn` 28.4, `mla.index` 13.4, `kda` 13.2, `hc+norm` 4.8,
`mla.proj` 3.7, `ffn_dense` 3.1. TTFT for that turn 1 179.4 s over 8 853 new
tokens. At depth 0 (turn 1, ctx 1 287): 106 ms/token, moe 61 %, TTFT 136.5 s.

**Streamed cost at S=512**, bytes from §2.1, rate from §4a (27.4 GB/s serial,
fill and compute both inside it):

| | histogram model | upper bound |
|---|---:|---:|
| bytes/token, 18k turn | 85.4 MB | 189.5 MB |
| **ms/token streamed** | **3.12** | **6.92** |
| bytes/token, turn 1 (1 287 tokens = 3 chunks) | 101.9 MB | 226 MB |
| **ms/token streamed, turn 1** | **3.72** | **8.25** |

**Projected 18k token** — everything but `ffn_moe` unchanged; `ffn_moe` becomes
the streamed cost plus the part of today's 69.8 that is *not* the non-resident
CPU path (the resident-tier dispatch, the shared expert, the router and the
host accumulate), carried at an assumed 5 ms/token:

| | ms/token | × vs 136.4 |
|---|---:|---:|
| upper-bound bytes (6.92 + 5 residual) | **78.5** | **1.74×** |
| histogram bytes (3.12 + 5 residual) | **74.7** | **1.83×** |
| gate | ≤ 85.25 | 1.60× |

**Projected turn 1** (106 − 64.7 + streamed + 3 residual):

| | ms/token | × vs 106 |
|---|---:|---:|
| upper-bound bytes | **52.6** | **2.02×** |
| histogram bytes | **48.0** | **2.21×** |
| gate | ≤ 53.0 | 2.00× |

**Verdict of this note.** The 1.6× bar at 18k has 7–11 ms/token of margin and
should be met. **The 2.0× bar on turn 1 does not: at the pessimistic end of
the non-resident-count bracket the projection is 2.02×, which is inside the
noise of a single ladder arm.** That is stated here, before the run, so that a
turn-1 miss is read as the projection having said so rather than as a surprise.
Both numbers depend on two quantities this note could only bracket or assume:

* **the distinct non-resident count per layer-chunk** (73.5 vs 163, a 2.2×
  spread) — now counted by the engine under `COLI_TIMERS=1` and reported on
  the `[STREAM]` line in both arms;
* **the non-CPU residual of `ffn_moe`** (assumed 5 ms/token at 18k, 3 at
  turn 1) — read from the `moe split: router= shared=` line and `[PROF] eg=`.

If the ladder contradicts the projection, the contradiction goes in the record
and the item stops there. It is not tuned until it agrees.

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
