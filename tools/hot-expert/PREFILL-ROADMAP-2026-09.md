# Prefill / TTFT roadmap — GLM-5.3 on rome (opened 2026-09-06, rev 13, 2026-09-07 20:10)

A separate track, because it has a different goal, a different gate, and a
different bottleneck from everything in `ROADMAP-2026-09.md`. That roadmap
optimised **decode throughput** (tok/s on short prompts). This one is about
**time-to-first-token on real prompts** — the number a person actually waits on
in an interactive UI. Nothing in the decode roadmap moves it.

**Rev 13 (2026-09-07, 20:10): RP4 answered, and it closes the expert shader.**
`COLI_VK_TIMESTAMPS` (off by default, bit-identical on and off, TTFT 1.00× with
it on) puts `VK_QUERY_TYPE_TIMESTAMP` queries inside the expert group's command
buffer on all three devices and, via `VK_EXT_calibrated_timestamps`, splits the
CPU-side `eg` wait into queue latency, GPU busy per phase and fence tail. The
answer, at 781 and 3 462 tokens with three repeats each: **of `eg`'s 58.1
ms/token, 53.6 is the CPU experts running concurrently inside the same timer
(92 %), GPU busy on the busiest device is 1.97, and the time the GPU actually
makes the engine wait is 0.068 ms/token — 0.12 %.** P4b and P4c did not fail to
move the expert group; the bucket was never GPU work, and both of §P4c's
remaining suspects (per-dispatch fixed cost, occupancy at 8 output rows) are
refuted by their own numbers. The per-expert table does find two real kernel
facts — a one-row expert costs 2.7–3.4× a two-row one (the per-row shader's LDS
bank conflict, `vk_tile_ok4` requires `S > 1`), and the R = 8 tile is not flat
in rows because it re-reads 67.1 MB of activations per row against a 14.16 MB
weight stream — but both are capped by `gpu_late`: **≤ 0.07 ms of a 120.6 ms
prefill token, and ≤ 1.96 ms of a 167.5 ms decode token (1.2 %)**. Neither was
built. The larger residual is host-side: `eg − cpu` is 4.5 ms/token in prefill
and ~6.0 in decode for 126 issue/take pairs per token, which is `VK_PROF`'s
question and §G13's shape of fix. **The prefill levers are unchanged and are
not the GPU:** CPU experts 53.6 ms/token at 781 tokens, MLA attention 69.6
ms/token at 3 462 (40 % of the token) — that is P5. Next: P5, then the
upstream-`dev` merge.

**Rev 12 (2026-09-07, 17:00): P6b in service — the CPU recurrence P6 forced is
gone.** One KDA state+window set per KV slot on dev0, allocated in `model_load`
**before** the expert preload (after it, the 624 MB would come out of dev0's
3.0 GB reserve); `coli_vk_kda_*` and `GSession` carry the slot; `slots_init`
forces `COLI_KDA_GPU=0` only when the pool is short, loudly. `p6b_gate.sh`
exits 0 on all six steps: `tworeq` at 4 slots IDENTICAL at `COLI_KDA_GPU=2`
**with zero `forcing` lines** (the check that makes the oracle mean anything)
and at `=0`; bit-identical against the P7 binary at one slot (782 teacher-forced
positions, last-logit `max_abs=0`, TTFT 1.00/0.98/0.98×); decode at four slots
**5.702 tok/s vs 5.728 at one slot on the GPU (0.996×) and 4.990–5.089 with the
CPU recurrence (+13.2 %)**; the live side-request case through the gateway
reuses **1 339/1 355 in 3.08 s** with no `forcing` line; and the device→host
state sync moved off write-combined memory — **370 MB checkpoint captured with
a 69 ms sync (156 MB of KDA spans, ~2 262 MB/s) where the old path was
~14 MB/s** — which is what makes a P7 checkpoint possible at the GPU knob at
all (conv 2 with the 34-tool block: 5.15 s, t1/t2 = 279×). Price: the preload
drops **1 296 → 1 248 heat-ranked experts** (48, 3.7 % of dev0's tier), hit
rate unchanged. Serving since 16:50 at `--kv-slots 4` with `COLI_KDA_GPU=2`.
Next: RP4, P5, then the upstream-`dev` merge. (RP4 landed the same day — see
rev 13.)

**Rev 11 (2026-09-07, 12:30): P7 in service.** Prefix checkpoints in the
engine (the `deepseek_v4.c` `v4_ckpt_*` pattern over the segment adapter's own
span list, persisted under `<SNAP>/.coli_ckpt`) plus the gateway pin. `p7_gate.sh`
exits 0 on all five steps: with `GLM53_PREFIX_CKPT=0` the candidate is
bit-identical to the pristine engine (teacher forcing identical at 782
positions, last-logit `max_abs=0`, serve-path TTFT 0.98–0.99×); with
checkpoints on, a **new** conversation carrying the 34-tool Open WebUI block
goes **1 452 s → 5.30 s** (6 328 of 6 346 tokens restored) and **9.72 s** after
the engine is killed and respawned; the restored answer is byte-identical over
56 greedy tokens with first-token logits `cosine=1.0000000, max_abs=2.146e-05`;
`tworeq` at 4 slots is identical at both KDA knobs; a re-ranked
`<memory_context>` block costs 17 prefilled tokens instead of 1 005
(126.82 s → 3.00 s). Through the live gateway, restarted with the gate's
checkpoints on disk and nothing in memory, turn 1 of a conversation the process
has never seen answers in **8.93 s**. Serving since 12:25 with
`GLM53_PREFIX_CKPT=1 COLI_PREFIX_PIN=1`. Next: P6b, RP4, P5, then the
upstream-`dev` merge. (P6b landed the same day — see rev 12.)

**Rev 10 (2026-09-07, 06:40): P7 scoped by measurement; spec written.** The
first thing P7 did was the two-turn chat with memory context on, through
Open WebUI's own backend: with the owner's single memory turn 2 reused
162/177 tokens (2.1 s); with four memories the block's ranking changed
between turns, the system message with it, and turn 2 reused **0/195**
(21.6 s) on a different slot (§P7 step 0 in the record). So P7 is a
checkpoint **and** a gateway-side prefix pin; the design is
`P7-PREFIX-CKPT-SPEC-2026-09.md` (engine checkpoints ported from
`deepseek_v4.c`, disk-persistent; `COLI_PREFIX_PIN` on the gateway; the
executable gate `p7_gate.sh`). Build: Opus, on `perf/p7-prefix-ckpt`.

**Rev 9 (2026-09-07, 01:30): P6 in service.** The gateway's 1-slot cap for
glm53 was unjustified and is now 16; the engine forces the CPU recurrence
for more than one slot; identity holds across slots; and through the live
gateway a follow-up turn with a title request in between reuses 1 339/1 355
tokens: **171 s → 3.1 s**. P4c landed at 1.03×. Day total on a fresh 384-token
prompt: 65.6 → 44.9 s (1.46×); on a follow-up turn: minutes → seconds.
Next: P6b (per-slot device state, recovers ~10 % decode), P7 (checkpoint the
system+tools prefix so a *new* conversation starts warm), RP4 (GPU
timestamps on the expert group), P5.

**Rev 8 (2026-09-07, night): P6 step 1 measured, P4b declined, P4c gating.**
The single KV slot is what loses Open WebUI its prefix: a title request
between two turns → turn 2 re-prefills (191.6 s, REUSE 0); with
`--kv-slots 4` and the gateway's own routing → 3.2 s (REUSE 1 339/1 355).
Serving switched to 4 slots + `COLI_KDA_GPU=0` (the device holds one KDA
state); P6b (per-slot device state) recovers the ~10 % decode. P4b (LDS
staging) measured 0.91× and was refused by the gate; P4c (vec4 loads) is
gating.

**Rev 7 (2026-09-06, 21:20): P4 landed — bit-identical (no knob), 1.13× at
384 tokens, in service. Day total: 1.44× at 384 tokens, 1.39× at 1 230,
three items, every one bit-identical and gated on the serve path.** The
remaining ms/token at the CPU-recurrence knob: MoE 66 (CPU experts bound),
MLA 34 (attention 27), KDA 23 (recurrence 13). P5 next, then P6/P7 for Open
WebUI; the per-bucket split at the *serving* knob is being taken now.

**Rev 6 (2026-09-06, 20:48): P3 landed — 1.23× at 384 tokens, 1.19× at
1 230, bit-identical, in service.** The CPU half of the MoE bucket was the
binding one (99 → 68 ms/token in the profile). P4 (tiled shaders) is
implemented and gating now. Cumulative since the morning's pristine: 1.28× at
384 tokens.

**Rev 5 (2026-09-06, late): P2 landed — honestly 1.05–1.10× on the serve
path, not the 1.5× the spec projected.** Bit-identical, decode unchanged, in
service. The projection was made from a profile at `COLI_KDA_GPU=0`; at the
serving knob the pristine's fused KDA chain was already cheap and the S-row
GPU matmul re-reads weights per row, so the batching only removed submit
overhead. The structure is right; the gain belongs to P4 (S-tiled shaders,
dense and expert), which is now the top item. Rule: **time profiles at the
serving knob; use `COLI_KDA_GPU=0` for identity only.**

**Rev 4 (2026-09-06, night): P2 implemented and gated twice.** The oracle
is bit-identical and the profile shows 1.18×, but the first serve-path gate
caught two regressions the profile could not see — a decode tax from OpenMP
`if`-clauses (5.4 → 2.1 tok/s) and a 0.65× TTFT from reading the KDA state
back from device memory per chunk — and the chain auto-served the second one
because the gate treated TTFT as informational. Both fixed; the gate now exits
non-zero below `MIN_SPEEDUP` or on a decode drop; p2d pending (§P2 in the
record). Lesson written into P0: *a gate that can pass a regression is not a
gate.*

**Rev 3 (2026-09-06, evening): P0 and P1 are done** — the serve-path
baseline is measured (§P0/P1 in the record), prefix reuse turns out to *work*
on a real turn 2 (408 of 424 tokens reused, 2.7 s instead of 65 s), and the
CANCEL gate is executable and fails as predicted. P6/P7 are re-scoped from
"build prefix caching" to "make the reuse that exists survive Open WebUI".

**Rev 2 (2026-09-06, a few hours after rev 1):** rev 1's diagnosis — "prefill
is streaming-bound, batching cannot help, prefix caching is the only lever" —
was **wrong**, and it was wrong because it was asserted from one sentence in
the record instead of from the profile that already existed. The corrected
diagnosis is below; it changes the item list and the ordering. Rev 1's items
survive as P1 (reuse diagnosis) and P6/P7 (prefix caching); the real ceiling
is materially higher than rev 1 said.

## Why this exists (read before planning anything)

On 2026-09-06 GLM-5.3 was served to Open WebUI for the first time. It was
effectively unusable, and the reason was never a bug in any landed item — it
was a **scope gap the whole project shared**: every optimisation and every gate
measured decode on ~40-token prompts. Prefill of a real prompt was never
measured, never optimised, never tested. The first interactive use exposed it.

**Measured, clean, 2026-09-06** (direct CLI, `COLI_TIMERS`, engine hard-killed
first so no orphan contaminated it):

| prompt | wall (incl. ~77 s one-time load) | isolated prefill |
|---|---:|---|
| ~4 tokens | 76.8 s | — |
| ~602 tokens | 259.6 s | **598 tokens in 183 s → ~3.3 tok/s** |

Prefill runs at **decode speed**. Consequences, all measured or derived:
- a plain short chat prompt (~30–100 tokens): **10–30 s** to first token
- a prompt carrying Open WebUI's 34 builtin tools (~6 000 tokens):
  **~30 minutes** to first token, every turn

The 34 tools are ~5 600 tokens of JSON schema attached to *every* message. That
is what turned "hi" into a 30-minute request.

## The diagnosis (rev 2) — prefill is decode in a loop

SPEC-PROBE (`ROME-3x7900XTX-2026-09-04.md`) prefilled the same 330 tokens at
chunk sizes 1→16 with the full `[OPTIME]` split. Nobody read it as a prefill
profile. Read that way, in ms per prompt token:

| bucket | K=1 | K=16 | K1/K16 | batches on any normal engine? | why it does not here |
|---|---:|---:|---:|---|---|
| **total (layers)** | 183.6 | 169.8 | 1.08× | | |
| cpu experts | 56.1 | 54.9 | 1.02× | yes, per expert across the rows that chose it | `mlp3_cpu` is called **once per token at S=1** (`glm53.c` ffn_layer, "per ogni token che lo ha scelto") even though the expert's weights were already read once per chunk |
| eg — GPU expert group | 65.6 | 62.5 | 1.05× | yes | the group *does* gather rows per expert (`vrows[]`), but `qmatmul_gate_up.comp` dispatches `(O/8, rows, 1)`: **every row re-reads the whole expert from VRAM** — S rows cost S× one row |
| shared expert | 14.3 | 15.2 | 0.94× | yes, dense | `mlp3_shared` per token, S=1 |
| router | 5.7 | 5.5 | 1.05× | yes | per token, one OpenMP team per token |
| kda.proj | 22.1 | 19.1 | 1.16× | yes, 8 plain matmuls | `kda_layer` loops `for t < tokens`, one fused GPU submit **per token** (`coli_vk_kda_layer(row)` is 1-row by signature) |
| kda.ko | 13.4 | 13.7 | 0.97× | yes, plain matmul | same loop, `KMV` per token |
| kda.step | 13.2 | 11.6 | 1.14× | **no** — sequential recurrence | (legitimate) |
| mla.proj | 6.9 | 5.8 | 1.18× | yes | `mla_layer` loops per token, `vk_batch_mv` at M=1 |
| mla.attn | 17.1 | 17.1 | 1.00× | mostly (sparse indexer selects per token) | scalar C per (token, head) |
| hc+norm | 9.0 | 8.9 | 1.01× | yes | per-token loops |

Decomposition at K=1: **cpu-expert stream 31 %, sequential recurrence 7 %,
everything else 62 % — and that 62 % (114 ms/token) is ordinary matmul and
attention work that batches on every other engine.** Perfect batching at K=16
would take it to ~7 ms/token; it stayed at 103.

The serve path runs at the default `GLM53_PREFILL_CHUNK=128` (not set in
`~/start_glm53.sh`; SPEC-PROBE never measured above 16). The chunk only
dedupes the *disk/RAM read* of expert weights — real (GPU expert-calls fell
75 628 → 47 921 from K=1 to K=16 in the `[PROF]` line) but irrelevant, because
**every stage costs per row what it costs per token in decode**, and the rows
are the same at every chunk size.

Three things rev 1 got wrong, stated so they are not repeated:
1. "GEMM batching buys almost nothing here" — it buys nothing *because the code
   does not batch*. The GPU dense matmul already takes `S` rows
   (`coli_vk_matmul(..., S, I, O, gs)`); `mv()` hard-codes 1. The CPU int4
   kernels already take `S`. `quant.h:1317` already carries a 1×4 row-tiled
   int4 kernel written for "the prefill union delivers nr=2..16 rows per
   expert" — on the Qwen engine. None of it is wired into glm53's prefill.
2. "Cold experts do not dedup across tokens" rested on `[PROF] cpu n=35252`,
   which counts (token, expert) *pairs* — constant by construction. It proved
   nothing. The `eg experts=` counter does show dedup.
3. "Faster kernels cannot fix TTFT" — the kernels are not the problem; the
   call shape is.

**What the corrected ceiling looks like** (derived, to be measured by P0):
batching the dense stages (P2) removes most of ~70 ms/token; row-batching
the CPU expert compute (P3) and S-tiling the GPU expert shader (P4) attack
the 56 + 65 (overlapped) MoE ms/token; the recurrence (13) and the sparse
attention (17) remain. A floor around 60–80 ms/token is plausible → **2.5–3×
on every prefill, including the first one of a new prefix**, on top of which
prefix reuse (P6/P7) makes repeated prefixes ~free. Rev 1 offered only the
latter.

## The prerequisite nobody built (this comes first, always)

**P0 — a prefill oracle, a serve-path TTFT harness, and an executable gate.**
The root cause of 2026-09-06 is that the engine was validated only as a
decode-throughput benchmark, never as an interactive server. The gates in
`ROADMAP-2026-09.md` were prose; "incomplete gates" cost this project days.
Here a gate is a **script that exits non-zero**, and an item is done when the
script passes, with its output pasted in the commit body. P0 delivers three
instruments in `tools/hot-expert/`:

1. `prefill_profile.sh` — the **diagnostic**: teacher-forcing CLI run
   (`--ids`, no `--greedy`, so the whole run *is* prefill) with
   `COLI_TIMERS=1` at the serve chunk size, on a fixed 600-token and a fixed
   3 000-token id list, printing the table above in ms/token. Every P item is
   measured against its row.
2. `ttft_serve.py` — the **gate instrument**: drives the real serve protocol
   through `openai_server.Engine` (the class the gateway uses — the pattern
   `tworeq.py` already proves works), measures first-token latency separated
   from decode at 30 / 300 / 3 000 / 6 000 prompt tokens, and runs the
   multi-turn check (submit A, then A+B: was the prefix reused?). Residency is
   asserted before every run and the engine is verified idle (`pgrep`) before
   and after. Refuses to run otherwise.
3. `prefill_gate.sh <pristine-binary> <candidate-binary>` — the **gate**
   (exit 1 on an oracle failure, **exit 3 when the candidate's median TTFT
   speedup is below `MIN_SPEEDUP` (default 1.0) at any size or its decode
   sanity window drops more than 10 %** — added after p2c passed the oracle
   at 0.65× and a chain served it):
   (a) teacher-forcing argmax **identical at every position** over the
   600-token list and (b) last-position logits cosine ≥ 1−1e-4 / argmax equal
   over the 3 000-token list, both vs pristine; (c) `ttft_serve.py` on both
   binaries, twice each, printing the deltas. Exit 0 only if (a)(b) pass; the
   deltas are the item's number.

A prefill item validated on the `--greedy` CLI path is not validated (the G16
CANCEL fix passed a `--greedy 64` oracle that never executed the serve loop it
changed, and shipped broken). Half a day to a day. Opus spec, Haiku/Sonnet
build.

## The items

Effort and effect are honest, not optimistic. Ordering is by
(expected effect × confidence) / effort, with the instrument first and the
cheapest possibly-decisive diagnosis second.

**Order after rev 7:** P5 (the recurrence's S per-token submits into one
command buffer per layer per chunk; the sparse attention, 27 ms/token and
growing with context; a vectorised router behind a knob), then P6/P7 for Open
WebUI — the 24.5-minute tool prompt is now ~17 minutes and only reuse turns
it into seconds. Every projection is checked against a profile taken at the
serving knob (rev 5's rule).

| id | item | mechanism | expected effect on TTFT | effort | tier | gate (executable) |
|---|---|---|---|---|---|---|
| **P0** ✅ | Oracle + serve-path TTFT harness + gate script | `ttft_serve.py`, `prefill_profile.sh`, `prefill_gate.sh` (commits `65f0c8a`…`3777078`) | **baseline (pristine): 21 tok 3.1 s · 384 tok 66.7/65.6 s (171 ms/tok) · 1 230 tok 241.5/241.2 s (196 ms/tok)**, reproducible within 1 %; long rows on the P2 binary: **3 562 tok 845 s · "hi" + 34 tools (5 439 tok) 1 470 s = 24.5 min** | done | — | met: ±1 % at three sizes; pristine-vs-pristine gate run in the record |
| **P1** ✅ | Why prefix reuse "does not fire" | measured with the engine's own `REUSE` line: a real turn 2 `[A, reply, B]` reused 408/424 tokens, **TTFT 2.74 s vs 65.45 s**; the rev-1 "same prompt twice" test was the *regenerate* case, which cannot reuse by design (the KDA state cannot rewind past prompt + reply) | reuse already works for a well-formed conversation; what Open WebUI loses it to is the single slot (side requests evict the session) and per-turn prefix changes | done | — | met; the Open WebUI-specific cause is P6's first step, with `GLM53_VERBOSE=1` now in `~/start_glm53.sh` so the server log carries REUSE lines |
| **P2** ✅ | Batch the dense stages (`P2-BATCH-DENSE-SPEC-2026-09.md`; `b9c193e`…`0b75c42`; **landed 2026-09-06 18:58**: oracle bit-identical, serve-path TTFT 1.10× / 1.05× / 1.06× at 21 / 384 / 1 230 tokens, decode unchanged — §P2 in the record) | in `kda_layer` / `mla_layer` / `ffn_layer`: gather the chunk's rows and call the existing S-row kernels once per weight per chunk (`coli_vk_matmul` S>1 for kq/kk/kv/kfa/kb/kga/kfb/kgb/ko, qa/kva/iwk/ikpg/iwp, qb/iwq, shared gate/up/down; router as one matmul); the KDA recurrence stays a per-token loop between the batched projections and the batched ko. The GPU shader is per-row independent (`s = WorkGroupID.y`), so rows are bit-identical to today | ~70 ms/token → ~10; **~1.5×** on prefill of any prefix | 2–4 days | Opus | `prefill_gate.sh` (a) bit-identical argmax; TTFT delta at 600 and 3 000 tokens ≥ 1.3×, both runs |
| **P3** ✅ | Row-batched CPU expert compute (`c695a48`; **landed 2026-09-06 20:48**: bit-identical, serve-path TTFT 1.05× / 1.23× / 1.19× at 21 / 384 / 1 230 tokens vs P2, decode unchanged; profile ffn_moe 99 → 68 ms/token — §P3 in the record) | `coli_i4_rows4` decodes a weight row once per four activation rows; `cpu_expert_rows` runs each CPU expert once on the chunk rows that chose it | done | — | — | met |
| **P4** ✅ | S-tiled GPU matmul shaders — experts **and** dense (`f0197d0`; **landed 2026-09-06 21:20**: teacher-forcing identical and logits max-abs 0 — bit-identical, no knob; serve-path TTFT 1.11× / 1.13× / 1.11× at 21 / 384 / 1 230 vs P3; kda 31.7 → 22.7 ms/token in the profile — §P4 in the record) | `qmatmul_gate_up.comp` / down / `qmatmul.comp`: one weight read per output row applied to all S rows (register tile over rows), instead of `(O/8, rows, 1)` re-reading the matrix per row. P2 showed the dense path needs it too: batched kda.proj+ko still cost 18 ms/token for ~2 of arithmetic | eg per token drops with the chunk's dedup (×~1.6 at K=16, ×~3–4 at K=128) — the MoE bucket's floor moves | 2–4 days | Opus (shader) | `prefill_gate.sh` (b) logits within tolerance — the reduction order changes, so this ships behind a knob; TTFT delta |
| **P5** | The sequential remainder | kda.step 13 ms/token: the S per-token `kda_step` submits of one chunk recorded into one command buffer (one round trip per layer per chunk instead of S); mla.attn 27 ms/token at 500 ctx and growing: vectorise the per-(token, head) dot/softmax or run it on `attention_absorb.comp`; router 5 ms/token: vectorised dot behind a knob (order changes) | ~30 → ~10–15 ms/token | 2–3 days | Opus | `prefill_gate.sh`; TTFT delta |
| **P6** ✅ | Make reuse survive Open WebUI (`cb9c11e`; **in service 2026-09-06 23:30**: `--kv-slots 4`, `COLI_KDA_GPU=0`; `tworeq` identical across slots at both knobs; live-gateway side-request case turn 2 **171 s → 3.1 s**, REUSE 1 339/1 355 — §P6 in the record). Confirmed on the owner's own Open WebUI on 2026-09-07 with tools and side tasks off: follow-up turn REUSE 32/41, ttft 5.3 s, 180 tokens at 4.4 tok/s. Stability *with* memory context and the tool block on is P7's question | | | | | met |
| **P6b** ✅ | Per-slot KDA device state — spec `P6B-KDA-SLOT-STATE-SPEC-2026-09.md`, built on `perf/p6b-kda-slots`, **in service 2026-09-07 16:50** (binary 53ccbb42…, `--kv-slots 4`, `COLI_KDA_GPU=2`) | `G.kda_slot[layer][slot]` state/window on dev0 (34 × 4.58 MB = 156 MB per slot), `coli_vk_kda_pool_init` allocates every slot's set BEFORE `vk_preload_tier` (after it, the 624 MB would eat dev0's 3.0 GB reserve); `coli_vk_kda_init/step/layer/sync/upload` and `GSession` take the slot; `slots_init` forces the CPU recurrence only when the pool is short. `coli_vk_kda_sync` goes through a `vkCmdCopyBuffer` into HOST_CACHED staging instead of a memcpy from write-combined memory | measured: decode at 4 slots **5.039 → 5.702 tok/s (+13.2 %)**, within 0.996× of the 1-slot GPU number; bit-identical at one slot (`max_abs=0`); live gateway turn 2 REUSE **1 339/1 355 in 3.08 s** with no `forcing` line; the state sync **~14 MB/s → ~2 262 MB/s** (370 MB checkpoint, 69 ms), which is what lets P7 capture at the GPU knob; price 1 296 → **1 248** preloaded experts (48, 3.7 %) | done | Opus (spec: Fable) | `p6b_gate.sh` exits 0 on all six steps (§P6b in the record) |
| **P7** ✅ | Checkpoint the stable prefix (tools + base system) **and pin the per-turn context block** — spec `P7-PREFIX-CKPT-SPEC-2026-09.md`, built on `perf/p7-prefix-ckpt`, **in service 2026-09-07 12:25** (binary 19e28e72…, `GLM53_PREFIX_CKPT=1 COLI_PREFIX_PIN=1`, `--kv-slots 8`, `COLI_KDA_GPU=0`) | engine: the session's span list (11 DSA layers' latent/ikeys/igates rows + 34 KDA states/windows; **370 MB at 6 328 tokens**) copied at the prefix boundary the gateway sends in the 8th SUBMIT field (`src=hint`, verified against the prompt's own ids) or at the LCP of successive fresh prompts, persisted under `<SNAP>/.coli_ckpt`. Gateway: `COLI_PREFIX_PIN=1` keeps the `<memory_context>` block byte-identical across a conversation's turns | measured: new conversation with the 34-tool block **1 452 s → 5.30 s** (engine) / **8.93 s** (live gateway, restored from disk after a restart); re-ranked memory block **126.82 s → 3.00 s**, 17 tokens prefilled instead of 1 005; off (`GLM53_PREFIX_CKPT=0`) the engine is bit-identical to the pristine one | done | Opus (spec: Fable) | `p7_gate.sh` exits 0 on all five steps (§P7 in the record); live gateway `--prefix-ckpt` and `--pin-block` PASS with `CKPT hit` and `[pin] … hit` in the server log |
| P9 (parked) | Pin mapped expert slots (`compat_mlock` at fill, unlock at evict) | upstream's datapoint on PR #1324 (2026-09-06): below ~1:1 model:RAM the kernel page LRU and the engine slot LRU disagree and a hit re-faults inside the matmul; rome sits near 1:1 once the engine's anon memory is counted, but shows ~0 major faults per request at 99 % residency — a robustness item (deterministic residency), not speed | none measured here; `ttft_serve.py` now prints majflt per request — build this only if that number stops being ~0 | 1 day | Sonnet | majflt per request stays 0 across a full gate with no `--warm` |
| — | Housekeeping (not this track): merge upstream `dev` | #1325 (portable `st.h` mapping) and #1350 (RSS counts owned memory — the fix for the `eabeb9a` thrash CLAUDE.md warns about) are in upstream `dev` since 2026-09-05; this fork is 103 ahead / 64 behind since 2026-08-31. **Known conflict:** the fork's `glm53.c` still carries the #1324-style engine-local expert mapping (`expert_map_init`, `[MAP] … file mappati`); #1324 was closed as superseded on 2026-09-07 and the merge must replace that path with `dev`'s `st_map_shard_range` one, then re-run `prefill_gate.sh` and `rome_bench.sh`. #1321 (cache budget) is rebased on `dev` and open | — | 1–2 days | Opus | all four qwen38 tests pass, `prefill_gate.sh` and `rome_bench.sh` reproduce their last rows |
| **P4b** ✗ / **P4c** ✅ | Tile shader variants: LDS staging measured 0.91× and was refused by the gate; vec4 x loads passed at 1.03× (neutral profile) and are in service. **RP4 settled why neither moved the expert group: the 58 ms/token bucket is 92 % CPU experts overlapping inside the same timer, GPU busy on the busiest device is 1.97 ms/token, and the GPU makes the engine wait 0.068 ms/token.** There was nothing there to move; a shader that ran in zero time would have saved 0.06 % of the token. P4b's premise (get the activations out of global memory) was right — the tile re-reads 67.1 MB of activations per row, 4.7× its whole weight stream — and its implementation wrong; P4c's premise (too many loads, not too many bytes) was wrong. **No further expert-shader work on this track** | | | | Opus | `prefill_gate.sh` |
| **RP4** ✅ | GPU-side timestamps on the expert group (`COLI_VK_TIMESTAMPS`, off by default; `perf/rp4-vk-timestamps`, **in service 2026-09-07 20:00**, binary `325c9c7a…`, serving script unchanged) | per-device `VK_QUERY_TYPE_TIMESTAMP` pool written at the top of the expert group's command buffer, between the gate+up and down phases and after the down phase (`=1`), plus per expert (`=2`, serializing and honest about it); `VK_EXT_calibrated_timestamps` (asked for only when the knob is set) maps the device clock onto `CLOCK_MONOTONIC`, which splits the CPU-side `eg` wait into queue latency / GPU busy per phase / fence tail, and yields `gpu_late` — the only part of the wait the GPU owns. `prefill_profile.sh` prints it as a row group under `eg` | measured at 781 and 3 462 tokens, three repeats each: GPU busy **1.97 / 0.93 / 0.89** ms/token on dev0/dev2/dev3, `gpu_late` **0.068** ms/token (0.12 % of `eg`), `queue_lat` 0.07–0.25; per-expert 232–235 µs (dev0, 9.3 rows) vs 84–86 µs (dev2/dev3, ~5 rows); a **one-row** expert costs 2.7–3.4× a two-row one because `vk_tile_ok4` requires `S > 1` and the per-row shader hits an LDS bank conflict. Prefill wall identical off / `=1` / `=2` | done | Opus | `rp4_chain.sh`: `prefill_gate.sh` OFF **rc 0** (bit-identical, TTFT 0.99×/1.00×/1.00×), `tworeq` 4 slots at `COLI_KDA_GPU=2` IDENTICAL with 0 forcing lines, `prefill_gate.sh` ON **rc 0** (bit-identical, TTFT 1.00× at every size) — §RP4 in the record |
| P8 | Capacity (cross-ref) | int3 experts (`ROADMAP-2026-09.md` 4h / G15) | partial; tracked on the main roadmap | weeks | Opus | its own gate |

Rev 1's "P4 — 4-accumulator BF16 prefill kernel (Q5)" is dropped from this
track: it is a Qwen-engine BF16 kernel, and the finding above is that the call
shape, not the kernel, is the problem.

## The honest ceiling, stated up front

- P2–P5 speed **every** prefill, including the first of a new prefix, by an
  estimated 2.5–3× (to be measured). A 6 000-token tool prompt goes from ~30 min
  to ~10 min: better, still not interactive. **P6/P7 remain necessary for
  Open WebUI with tools**; P2–P5 make everything P6 cannot cache tolerable.
- A 32k-token document summarised cold is ~2.7 h today and ~1 h after P2–P5.
  Only capacity (P8 / more VRAM) changes that, and only partly.

## The alternative this track must not bury

Recorded so a future session weighs it honestly: **this box may not be the
right home for low-latency interactive use with large prompts.** Colibri's
strength is running a model too big to fit in VRAM at all. A model that
*fits* gives interactive tools that respond in seconds today, with none of
P0–P7. If the requirement is "tools that work," that is the faster answer; if
it is specifically "GLM-5.3 with tools on this hardware," this track is the
way.

## Discipline this track inherits from 2026-09-06 (non-negotiable)

1. **Validate on the path the user uses.** A prefill/serve item proven on the
   `--greedy` CLI path is not proven. `ttft_serve.py` or nothing.
2. **Measure the user's request, not a proxy.** Traffic from inside the Open
   WebUI container shares its source IP with the app — read the app's own
   record (its DB, its logs).
3. **Assert residency before every measurement.** A cold cache alone turned a
   2.8-second "hi" into 12 minutes. `ttft_serve.py` prints residency before
   every request and refuses below 96 % — 100 % is unreachable with an engine
   up (a fresh process evicts ~3 %; the gateway's sits at 91.6 %), so a gate
   compares both binaries at the *same* printed number, not at a fiction.
4. **One in-flight request while diagnosing.** An orphaned generation (a
   timed-out probe with the CANCEL bug unfixed) blocks the queue and poisons
   every later number. Kill hard, confirm `pgrep -x glm53` empty, then measure.
5. **No claim of "it works" without an end-to-end pass on the real path**,
   shown to the user, reproducible.
6. **Read the profile before asserting the bottleneck.** Rev 1 of this file is
   the counter-example.

## Open dependency: CANCEL (still unfixed)

`glm53` does not honour CANCEL — **measured 2026-09-06 with the new gate**: a
1 230-token prompt cancelled at 5 s held the engine for 251 s (the whole
prefill and the generation). Prefill work makes this worse: a 17-minute tool
prompt a user gives up on holds the engine for 17 minutes. The G16 attempt
failed for lack of a serve-path oracle; `ttft_serve.py --cancel 5` is that
oracle now (PASS = engine confirms within seconds and the next request is
served). Fix it as the next engine change after P2, or before P2 if a user is
actively hitting it: it is also what makes every measurement session safe.
