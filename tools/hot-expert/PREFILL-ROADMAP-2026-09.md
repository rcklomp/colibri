# Prefill / TTFT roadmap — GLM-5.3 on rome (opened 2026-09-06, rev 2 same day)

A separate track, because it has a different goal, a different gate, and a
different bottleneck from everything in `ROADMAP-2026-09.md`. That roadmap
optimised **decode throughput** (tok/s on short prompts). This one is about
**time-to-first-token on real prompts** — the number a person actually waits on
in an interactive UI. Nothing in the decode roadmap moves it.

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
3. `prefill_gate.sh <pristine-binary> <candidate-binary>` — the **gate**:
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

| id | item | mechanism | expected effect on TTFT | effort | tier | gate (executable) |
|---|---|---|---|---|---|---|
| **P0** | Oracle + serve-path TTFT harness + gate script | see above | none directly — every item below is measured with it | ½–1 day | Opus spec, Sonnet build | `ttft_serve.py` reproduces its own numbers within ±10 % on two consecutive runs at all four sizes; `prefill_gate.sh` passes pristine-vs-pristine |
| **P1** | Diagnose why prefix reuse does not fire | read the serve path: `conversation_cache_slot`, `slot_remember`, how a request is matched to a slot and what invalidates it; the 806-token prompt sent twice took 184 s then 179 s | unknown: a bug costs a day and could make turn 2+ of a conversation seconds; an absent feature becomes P6 | ≤ 1 day | Opus | a written account in the record naming the exact condition that fails; if it is a bug, the fix passes `prefill_gate.sh` and the multi-turn check shows turn-2 TTFT ≪ turn-1 |
| **P2** | Batch the dense stages | in `kda_layer` / `mla_layer` / `ffn_layer`: gather the chunk's rows and call the existing S-row kernels once per weight per chunk (`coli_vk_matmul` S>1 for kq/kk/kv/kfa/kb/kga/kfb/kgb/ko, qa/kva/iwk/ikpg/iwp, qb/iwq, shared gate/up/down; router as one matmul); the KDA recurrence stays a per-token loop between the batched projections and the batched ko. The GPU shader is per-row independent (`s = WorkGroupID.y`), so rows are bit-identical to today | ~70 ms/token → ~10; **~1.5×** on prefill of any prefix | 2–4 days | Opus | `prefill_gate.sh` (a) bit-identical argmax; TTFT delta at 600 and 3 000 tokens ≥ 1.3×, both runs |
| **P3** | Row-batched CPU expert compute | per expert, run `mlp3_cpu` once with S = the rows that chose it (already gathered for the GPU path); port `quant.h`'s 1×4 row-tile (K2) from the Qwen engine | cpu-expert 56 → ~25–35 ms/token | 1–2 days | Sonnet (port) | `prefill_gate.sh`; TTFT delta on top of P2 |
| **P4** | S-tiled GPU expert shader | `qmatmul_gate_up.comp` / down: one weight read per output row applied to all S rows of that expert (register tile over rows), instead of `(O/8, rows, 1)` re-reading the expert per row | eg per token drops with the chunk's dedup (×~1.6 at K=16, ×~3–4 at K=128) — the MoE bucket's floor moves | 2–4 days | Opus (shader) | `prefill_gate.sh` (b) logits within tolerance — the reduction order changes, so this ships behind a knob; TTFT delta |
| **P5** | The sequential remainder | kda.step 13 ms/token: GPU step shader per token in one command buffer per chunk vs CPU; mla.attn 17 ms/token: vectorise the per-(token, head) dot/softmax or run it on `attention_absorb.comp` | ~30 → ~10–15 ms/token | 2–3 days | Opus | `prefill_gate.sh`; TTFT delta |
| **P6** | Prefix caching that works, KV **and** KDA state | at a turn boundary, reuse the cached forward state for the longest matching prefix; snapshot/restore MLA KV **and** KDA recurrent state (reuse G12's `kda_sync`/`kda_upload`) | turn 2+ of a conversation: prefill only the new tokens — **minutes → seconds** for a tool prompt | **1–3 weeks**, high risk (G12's cross-session leak and the CANCEL wedge were both this class of bug) | Opus, Fable for the state-boundary design | multi-turn check: turn-2 TTFT within a few × its *new-token* count; greedy text bit-identical to no-cache over 512 tokens on the serve path |
| **P7** | Warm the stable prefix at startup | prefill the system+tool block once when the server learns it, so turn 1 of a new conversation reuses it | first turn of every conversation → seconds after a one-time warm | +2–4 days on P6 | Opus | a fresh conversation's first turn reuses the pre-warmed prefix; TTFT independent of tool-block size |
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
   2.8-second "hi" into 12 minutes. `ttft_serve.py` refuses to run below 99 %.
4. **One in-flight request while diagnosing.** An orphaned generation (a
   timed-out probe with the CANCEL bug unfixed) blocks the queue and poisons
   every later number. Kill hard, confirm `pgrep -x glm53` empty, then measure.
5. **No claim of "it works" without an end-to-end pass on the real path**,
   shown to the user, reproducible.
6. **Read the profile before asserting the bottleneck.** Rev 1 of this file is
   the counter-example.

## Open dependency: CANCEL (still unfixed)

`glm53` does not honour CANCEL between generated tokens — an aborted request
runs to `max_tokens` holding the single engine slot (revert `73e770c`). Prefill
work makes this worse: a 30-minute prefill a user gives up on holds the engine
for the full 30 minutes. **P0's harness is what a correct CANCEL fix needs** —
the G16 attempt failed for lack of a serve-path oracle. Fix CANCEL as part of,
or immediately after, P0, with a `ttft_serve.py --cancel` case as its gate.
