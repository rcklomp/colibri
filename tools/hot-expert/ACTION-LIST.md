# ACTION LIST -- the order of work (living document)

Last updated: **2026-10-08 night** (plan Rev 101; V2 added during the closeout read-through). Row 1 (PF0) is `doing`; nothing else has been started. The detail of every item (basis, gain, gate, stop rule) is in its plan; this file only holds the **order**, the **status** and the **pointer to the result**:
`DECODE-OPEN-ITEMS-PLAN-2026-10-08.md` (D items, V1; section 8 is the second round), `PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md` (PF items), `HANDOFF-2026-10-06b.md` (state and recipes).

## How this list is kept (rules)

1. **Status** is one of `todo`, `doing`, `gated` (waits for a decision, the gate is in the row), `blocked` (say what blocks it), `done`, `dropped`, `parked`.
2. **One `doing` row that needs the rig at a time** (CLAUDE.md "one benchmark at a time"). A rig job that runs detached keeps its row `doing` until its watcher has reported.
3. **Update it as part of the work, not after:** `doing` when a chain is launched; `done` or `dropped` when the result is in the record (fill **Result** with the record section and one number or verdict, then add a line to the Done log); when a gate is decided, rewrite the `gated` rows it releases into the queue (give them an order number) or drop them with the reason. The task-closeout skill reads this file (governance step).
4. **Never delete a row.** A finished or dropped row stays in its table with its status and Result; the Done log (section 5) is append-only, newest first.
5. **A new item gets a row in a plan first** (the ID must resolve there: `doc_currency.sh` checks it), then a row here.
6. The owner's decisions (section 4) are not worked around: a row that needs one is `gated` on it.

## 1. Queue (do in this order)

| # | ID | Action | Status | Needs | Plan | Gate / stop rule | Result |
|---|---|---|---|---|---|---|---|
| 1 | PF0 | Re-attribute the SHIPPED prefill (no attribution exists for the served build, only for the old staged chunk-512 one): (a) skip-class masks at the shipped config, (b) device timeline chunks 3-7 (per card and link busy share, waits at card boundaries, bank reuse), (c) the depth curve at ~2 k / 8 k / 32 k (/ 64 k), (d) in-place hit rate by layer. A Sonnet agent rebases and builds the skip-class binary; the orchestrator runs the chains | doing (2026-10-08 night: binaries `franken_decode_glm_pf0main` / `_pf0dbg` built from main f072093, chains `ckpt1006/glm_pf0_main_chain.sh` (timeline, profile, depth curve) then `glm_pf0_skip_chain.sh` (skip classes, routing dump) launched through `glm_pf0_seq.sh`; in place the timeline has no DMA records, so the link rate comes from `--profile`) | rig (GPU, exclusive), ~1 day | PREFILL section 3 (PF0) | ends with a table and a ranked list in the record; PF1-PF6 start only if PF0 shows their target above the stop-rule size (PREFILL section 5) | |
| 2 | D16 | The per-token `ms` vector of the timed loop (`--time`) and of the serving loop (`[serve-glm5]` prints only the request median); Sonnet, ~1 hour, plus one ~10-minute chain. Same session as PF0 (same kind of timing chain) | todo | rig, ~10 min | DECODE section 8 (D16) | vector printed for >= 3 runs; if no gap above 5 ms appears in 200 tokens, D17 is dropped | |
| 3 | V1 | The external reference past 2 051 tokens: dump the engine's CPU arm for the FULL model at a 2 300-token prompt (measure its speed on 128 tokens first; run detached, overnight, no other measurement beside it) and compare GPU chunk 1 / 512 / 1 024 against it. Sonnet writes the chain | todo | rig CPU + RAM (exclusive), ~0.5 day, mostly waiting | DECODE section 8 (V1) | a table (config, taps, not exact) in the record; a non-exact tap is a bug until explained | |
| 4 | V2 | The needle (8 needles, 30 k / 60 k / 120 k / 200 k) on the installed `.d14` build: the 8/8 result of 2026-10-06 predates the D14 prefill fix. `QE_PARTS=needle ARMS=franken-glm ... franken_quality_chain.sh` (handoff §5 'Long context'), ~1 h, detached with a watcher; the rig is under the gateway rules of that chain, so run it in its own session slot | todo | rig (GPU, exclusive; starts the engine through `serve_alt.sh`), ~1 h | DECODE section 8 (V2) | 8 of 8 as before; any miss the 2026-10-06 build got is a bug until explained | |
| 5 | G1 | **Decision after PF0:** rank PF1 / PF2 (PF2a first) / PF3 / PF5 by PF0's numbers and replace this row with the chosen items, in order, in this table | gated (on #1) | none | PREFILL sections 3-5 | PF0's table exists | |
| 6 | D15 | Where do the other two thirds of the fetch model's gain go: predicted against realised bytes and stage-kernel time per layer and card from the existing traces, then one paired trace with the pattern and the optimal table. After G1 because its ceiling is ~2 ms of ~60 | todo (after #5) | offline ~1 day, then one trace chain | DECODE section 8 (D15) | the realised / predicted ratio per card for the two tables explained within 20 %, or 'not explained, excluded: ...' | |
| 7 | G2 | **Decision:** build the CPU lane (D10, 1.3-2.7 ms, 2-4 %, multi-day, quality-gated) only if D15 leaves the fetch at no more than the model's headroom AND the owner wants the next 2-4 % (owner decision O5) | gated (on #6 and O5) | none | DECODE section 8, D10 row | | |

## 2. Gated work (start only when its gate says so; the gate is in the row)

| # | ID | Action | Status | Needs | Plan | Gate / stop rule | Result |
|---|---|---|---|---|---|---|---|
| - | D17 | Find what the host does between the argmax read-back and the next token's first launch and remove it | gated (on D16) | rig | DECODE section 8 (D17) | only if D16 shows real gaps; the stall share of the ~30 % disturbed serve-A/B arms is the prize | |
| - | PF1 | Pipeline depth across the three cards (more than two chunks in flight; the boundary residual has two banks today) | gated (on PF0) | rig, 2-4 days | PREFILL section 3 (PF1) | PF0 shows idle cards and NOT binding links; stop if the cards are busy >= 85 % or the links bind. Bubble 2 / (2 + M): pays on long prompts | |
| - | PF2 | K-quant trunk GEMM: PF2a first (a microbenchmark of dequantise-to-bf16 + rocBLAS / hipBLASLt on the K-quant shapes against `k_gemm_batch<8,1>`, no new kernel), then a WMMA tile kernel | gated (on PF0) | rig, 3-5 days | PREFILL section 3 (PF2) | trunk GEMMs still >= 1.5 ms/token at the shipped config; stop if < 1.5x `k_gemm_lds` TFLOPs on the K-quant shapes or the quality gate drops outside the previous binary's Wilson interval or loses a needle | |
| - | PF3 | Compute beside the copies: link idle share and in-flight GB/s under compute at the shipped (in place) config, then the mitigations of PREFILL section 6 | gated (on PF0) | rig, 1-3 days | PREFILL section 3 (PF3) | PF0 (b) shows idle or slowed links; stop if predicted < 0.3 ms/token | |
| - | PF4 | Chunk 2 048 / a layer-major MoE window | gated (on PF0) | rig, 2-3 days, VRAM first | PREFILL section 3 (PF4) | the 1198 / chunk term still exists in place and VRAM allows; stop if < 0.3 ms/token | |
| - | PF5 | A resident set for prefill (hit rate by layer against what a long prompt wants) | gated (on PF0) | rig, half a day | PREFILL section 3 (PF5) | PF0 (d) shows a hit-rate gap | |
| - | PF6 | Long-prompt scaling (indexer O(n^2), KDA scans) | gated (on PF0) | rig, 1-3 days | PREFILL section 3 (PF6) | PF0 (c): ms/token grows by more than ~15 % from 8 k to 64 k | |

## 3. Parked (do not start without a new reason or the owner's word)

| # | ID | Action | Status | Needs | Plan | Gate / stop rule | Result |
|---|---|---|---|---|---|---|---|
| - | D9 | Tensor-level split of a missed expert over two cards | parked | rig, 2-4 days | DECODE section 3 (D9) | modeled 0.6-0.9 ms after the realisation factor 0.33, below the 1.5 ms bar; re-size after D15 | record §L5-GLM-D9A |
| - | D8 | Four waves per row for KDA o / DSA o | parked | rig, 1 day | DECODE section 3 (D8) | <= ~0.4 ms, below the 0.3 ms stop rule once the exact-order constraint is counted | record §L5-GLM-D7 |
| - | D13 | Q6_K copy of the q8fast idea for the lm_head | parked | rig, hours | DECODE section 0 (D13) | ~0.1 ms | |
| - | D11 | Persist the placement average across restarts | parked | rig | DECODE section 3 (D11) | evidence against (the first request after a cold start decodes as fast as the later ones) | |
| - | D12 | Validate the DeepSeek adapter position fix (Gitea branch `adapt-ds4`) | parked | rig | DECODE section 3 (D12) | only if DeepSeek is served | |
| - | PF7 | Staging / DMA tricks back on | parked | rig | PREFILL section 3 (PF7) | in place beats staging at every chunk size; revisit only if PF3 changes what the links can do | |
| - | PF8 | CPU-assisted prefill | parked | rig | PREFILL section 3 (PF8) | needs the same CPU kernels as D10; only after G2 | |

## 4. Owner decisions (nothing in section 1 waits for them except where a row says so)

| # | Decision | Needed for |
|---|---|---|
| O1 | **GitHub:** the public fork lags (4 commits on 2026-10-08 night: the D14 notes, the worktree clean-up, the two new plans). Each push needs the go-ahead and the secret scan | nothing; housekeeping |
| O2 | **When GLM goes back into service** (the rig is reserved; recipe handoff §3; the first real unattended restore is unobserved, so do it watched) | nothing in the queue; the service is OFF |
| O3 | **Is prefill worth weeks?** It shows only in cold chats (28.4 s for the tool prompt) and big documents (~2 minutes for 25 k tokens, computed); warm chats are untouched (PREFILL section 0) | PF1-PF6 beyond PF0 |
| O4 | **The quality bar for a non-bit-exact prefill kernel**: proposed 'inside the Wilson interval of the previous binary on MMLU-Pro 70 and no needle regression' | PF2 |
| O5 | **Whether the next 2-4 % of decode is worth a multi-day build (D10)** | G2 |

## 5. Done log (append-only, newest first)

- 2026-10-08 night -- **plans for the next round written** (plan Rev 101): decode plan section 8, the new prefill plan, this list. Record: none (planning only).
- 2026-10-08 night -- **the two loose worktrees resolved** (the Mac's X1 checkout, the rig's `colibri-m1-ds4`; both directories removed, the X1 branch and the M1 bundle kept): decode plan section 6 item 3.
- 2026-10-08 night -- **D14 solved, fixed, gated, installed** (`franken_dec_glm` = `.d14` sha `f365fc7c014696ca`; rollbacks `.fin`, `.gr`, `.rs`, `.adm`): the prefill attention grid was sized from the last row of a chunk; record §L5-GLM-D14, GLM5.md 22.6, plan Rev 100. D5 explained by it (erratum in §L5-GLM-D5).
- 2026-10-08 evening -- **first round of the decode plan** (plan Rev 99): D7c `--gemv-q8fast` and D3 waves rule installed (`.fin`, ~-1.6 ms a token, bit-exact), D4 Q6_K split fault fixed, D1 / D2 closed, D6 sampler done, D9 / D8 parked after D9a: records §L5-GLM-D1D3 ... -D7.
