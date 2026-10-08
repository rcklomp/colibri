# PLAN 2026-10-08 (night) -- the open PREFILL items of GLM-5.3-Flash on `rome` (plan Rev 101)

Companion of `DECODE-OPEN-ITEMS-PLAN-2026-10-08.md`; the master plan is `FRANKEN-ENGINE-PLAN-2026-09-15.md`, the state is `HANDOFF-2026-10-06b.md`. This file owns **prefill only**. It is separate from the decode plan for one reason: **the gate is different.** A decode change must be bit-exact (maxabs 0 on 1 788 taps) and is judged by the paired kernel trace; a prefill change that reorders a sum (a new GEMM tile kernel) cannot be bit-exact and goes through the **quality gate** (`franken_quality_chain.sh`: the 70-question MMLU-Pro sample and the needle set, record §L0-QUALITY), and the instrument is the prefill device timeline (`--timeline`), not the decode trace. The served `--gemm-lds 1` is the precedent: prefill 16.9 -> 9.3 ms/token, "a last-bits order change, judged by the quality run" (`start_franken_glm.sh`).

Status: **nothing in this plan has been started.** Every number below is quoted with its source and its conditions; "estimate" means no run supports it. Written before measuring anything: item PF0 exists because the last attribution of prefill time was taken at a configuration that is no longer served.

## 0. Why prefill matters to the owner (and how much)

- Served prefill: **5.77 -> 5.62 ms/token (~178 tok/s) at 1 024-row chunks, in place (no staging), `FRANKEN_GLM_MOE_G=8`** (record §L5-GLM-G8: 5.769 -> 5.621 at 262 144 cells; handoff §1). Decode is ~60 ms a token; one prefill token costs a tenth of that, so prefill speed is invisible in a chat that hits the prefix cache.
- It shows in three places: (1) **a cold new chat** after a restart or a changed system prompt: 28.4 s for the 4 573-token tool prompt (`accept_live`, record §L5-GLM-STAGECROSS acceptance paragraph); the warm case is 0.84-0.86 s because 4 548 of 4 576 tokens are reused; (2) **a long document**: 5.6 ms x 25 000 tokens is ~140 s (computed, depth effects not included: whether ms/token grows with depth at the shipped config is not known, PF0); (3) the first message of a follow-up that pastes new material.
- So the prize is "minutes to seconds on big cold prompts", not tokens-per-second in chat. No target number has been set by the owner; items carry stop rules (section 5).

## 1. What is known about where prefill time goes (and how old it is)

- **Attribution, taken at the OLD staged config** (record §M7-SKIPCLASS: 8 192 prose tokens, chunk 512, staged experts, `--gemm-lds 1`, A,B,B,A, skip masks): everything 9.02 ms/token; trunk GEMMs worth **2.45** (from the experts-skipped state), experts 2.33, attention + indexer 0.49, KDA 0.11; the skeleton (staging DMA, norms, hyper-connection, router, boundary copies, no compute) 3.74. Classes are sub-additive (solo costs sum to 6.2 against 5.29 together). **That config is no longer served** (staging is off since 2026-10-06, chunk 1 024, blocked expert kernel, G=8), so these shares are a hypothesis for the shipped build, not a measurement of it.
- **Links:** with no compute beside them the staged DMA reached 47.6 GB/s aggregate (91 % of the 52 GB/s M4 probe); compute running beside the copies leaves the links idle (busy share 98 % -> 70-84 %) **and** slows each copy in flight (-21..-27 % per link) (§M7-SKIPCLASS). Memory/fabric clocks are not the cause (§M7-CLOCKTEST).
- **Bytes per token fall as 1/chunk:** fit P(chunk) = 7.26 + 1198/chunk ms/token (staged, §L5-GLM-CHUNK); chunk 1 024 vs 512 measured -12 %. Chunk 2 048 would be worth ~0.6 ms/token by that fit but needs ~1.2 GB/card of helper scratch restructured and the chunk-plan kernel's LDS array moved to global memory (handoff §4 item 5); the cards are ~99 % full at steady state (free MiB 732 / 818 / 458, CLAUDE.md "Before a build that allocates device memory").
- **Per-card utilisation is ~50 % in prefill** (handoff §4 item 6: "not investigated"). The engine already issues chunks without waiting (`--prefill-pipeline`, served default 1) and hands the residual between cards through **two banks** (`Scratch::hout[0|1]`, `boundary_recv(..., bank)`): the pipeline is at most two chunks deep by construction. With three cards and layers split 15 / 15 / 15 a steady-state pipeline needs three chunks in flight to keep every card busy. This is the reading to test, not a result.
- **Trunk GEMMs:** `k_gemm_lds` 1.62 + `k_gemm_batch<8,1>` 0.84 ms/token (old config); **every K-quant matrix is refused by the LDS path** and runs the generic kernel (handoff §4 item 3).
- **Placement in prefill:** held during prefill (`FRANKEN_ADAPT_PREFILL=0`, a held span counts as at most 64 tokens in the average); placement raised prefill 5.77 -> 5.62.
- **Correctness note (D14, 2026-10-08):** until tonight three rows in four of every chunk past 2 048 tokens lost their 1-3 newest keys in the DSA layers. Timing is unchanged by the fix (31.18 s for 2 300 tokens at chunk 512, old and new); any prefill claim from now on is checked against a chunk-1 reference at depth 2 300 (section 2).

## 2. How every item is measured

1. **Speed instrument:** A,B,B,A palindromes in one process, 8 192 prose tokens, 262 144 cells, shipped flags (chunk 1 024, in place, `--gemm-lds 1`, G=8, `--prefill-pipeline 1`), `--time-prefill`; the prefill device timeline (`--timeline`, chunks 3-7, `glm5_timeline.py`) for busy shares per card and link; the skip-class build (`ckpt1006/build_dbgskip.sh`, masks 1 experts, 2 trunk GEMMs, 4 KDA, 8 MLA attention, 16 indexer) for attribution; `gpu_sampler.py` beside every chain. One odd arm is not a result; name any arm left out.
2. **Exactness where nothing reorders** (scheduling, pipelining, buffers): the model gate (`glm_q8f_chain.sh` configuration set, maxabs 0) **plus the D14 gate**: a chunked prefill of a 2 300-token prompt against a chunk-1 prefill (`ckpt1006/glm_d14_chain.sh`, 1 788 of 1 788 taps). Never two chunked runs with the same last chunk: they share any fault and agree (this hid D14).
3. **Quality gate where a sum is reordered** (GEMM tile kernel, WMMA, int8 activations): `franken_quality_chain.sh` on the served engine against the previous binary (MMLU-Pro 70 + needle, same reasoning budget) AND a layer-wise error report against the bit-exact build (cos and maxabs per tap on the model-gate prompts, so a regression has a place, not only a score). The change is labelled "quality-gated, not bit-exact" from the first commit and in the record.
4. **VRAM first:** any item that allocates device memory starts from the shipped config's log (steady free MiB per card) and writes the budget before the build; optional buffers degrade, they do not abort.
5. Landing and acceptance as in the decode plan: franken-engine via the Mac clone, `glm_accept_chain.sh` + `run_ui2.sh`, the task-closeout skill; GitHub only on the owner's go-ahead after the secret scan. Tiers: Sonnet for harness and ports of an existing pattern, Opus for a real kernel; subagents never start a GPU program.

## 3. The items

Effort = working time of one session including the chains. "Gain" is ms/token off ~5.6 unless stated.

### Phase 0 - attribution of the SHIPPED prefill (decides everything below)

| ID | Item | Gain | Basis | Effort | Tier |
|---|---|---|---|---|---|
| PF0 | re-attribute the shipped prefill. (a) skip-class masks at the shipped config (rebase `build_dbgskip.sh` onto main; masks 31, 29, 30, 1, 3, A,B,B,A, 8 192 tokens); (b) the device timeline at chunks 3-7: per card busy share, per link busy share and in-flight GB/s, the waits at the card boundaries (`TL_UPSTREAM_WAIT`), the bank reuse; (c) the **depth curve**: ms/token at ~2 k, 8 k, 32 k (and 64 k if the cells allow) so attention / indexer growth is seen; (d) the in-place hit rate of the experts a chunk reads (`k_glm5_fetch_count`) by layer | 0 ms; **the numbers PF1-PF6 are sized with** | the only attribution is from the old staged config (section 1) | 1 day: two or three chains, one Sonnet build | Sonnet builds, me |

PF0 ends with a table in the record and a ranked list. **Decision rule:** each of PF1-PF6 is started only if PF0 shows its target above its stop-rule size (section 5).

### Phase 1 - candidates, each gated by PF0

| ID | Item | Gain (label) | Basis | Effort | Tier |
|---|---|---|---|---|---|
| PF1 | **pipeline depth across the three cards**: more than two chunks in flight (more banks for the boundary residual and for per-chunk scratch; or a per-layer "wavefront" schedule), so every card works while another waits | up to the idle share PF0 finds (~50 % idle per card at best; much less if the links, not the cards, are the limit) -- **estimate** | the two-bank design; ~50 % utilisation (handoff §4 item 6); the carried state (KDA conv/ssm, DSA rings) is per layer and per card, so layer l of chunk k + 1 only needs layer l of chunk k: no arithmetic changes | 2-4 days; scratch per bank is Tm x 40 KB x 2 plus helper buffers: VRAM budget first | Opus |
| PF2 | **trunk GEMMs for K-quants**: a WMMA (gfx1100 `v_wmma_f32_16x16x16_*`) or LDS-tiled kernel that accepts the K-quant matrices the LDS path refuses, and/or a faster Q8_0 path (dequantise-to-f16 + WMMA, or int8 WMMA with per-block activation quantisation) | trunk GEMMs are 2.45 ms/token at the old config (~27 % of it); a 1.5-2x faster GEMM would be **-0.8..-1.2 ms (14-21 %) -- estimate** | outside prior art in section 6; the LDS kernel's precedent (16.9 -> 9.3) | 3-5 days; microbenchmark first (TFLOPs vs `k_gemm_lds`) | Opus |
| PF3 | **compute beside the copies**: in the shipped (in place) path the experts are read by the kernel over the link, not copied by DMA; PF0 (b) says whether the links are still idle 16-30 % of the time and slowed in flight. Mitigations from section 6 (queue counts, `HSA_ENABLE_SDMA`, hipHostRegister / pinned flags, kernel-side read granularity and prefetch distance, launching the expert kernel earlier than the trunk of the same layer) | link-bound share of the 5.6 -- **unknown until PF0** | §M7-SKIPCLASS (staged config); the in-place read path was not profiled | 1-3 days | Opus |
| PF4 | **chunk 2 048**: restructure helper scratch and move the chunk-plan LDS array to global memory | ~0.6 (fit of the staged config; in place **unverified**) | handoff §4 item 5; §L5-GLM-CHUNK | 2-3 days; blocked by VRAM unless PF1/PF3 free room | Opus |
| PF5 | **a resident set for prefill**: the hit rate of the in-place reads by layer (PF0 d) against what a long prompt would want held; a prefill-specific placement or cap instead of the decode-learned set | unknown | placement alone moved prefill 5.77 -> 5.62 | measurement first: half a day | me |
| PF6 | **long-prompt scaling** (only if PF0 (c) shows ms/token grows by more than ~15 % from 8 k to 64 k): the indexer scores every pool for every row (O(n^2) over a prompt), KDA scans; chunkwise tricks from section 6 | depends on the curve | closed so far: 'the long-context check' only tested retrieval to 200 k, not time | 1-3 days | Opus |

### Phase 2 - parked

| ID | Item | Why |
|---|---|---|
| PF7 | staging back on / DMA tricks | in place beats staging at every chunk size (§L5-GLM-STAGECROSS: -4..-7 % at 512-1 024 rows); revisit only if PF3 changes what the links can do |
| PF8 | CPU-assisted prefill (the EPYC computes part of the experts) | decode's CPU lane (D10) needs the same quantised CPU kernels; look at it only after D10's gate |

## 4. Order and expected total

1. **PF0 first, always** (1 day). It is cheap and it either confirms the 50 %-idle / trunk-GEMM story or kills it.
2. Then, by PF0's ranking: PF1 if the cards idle and the links do not; PF2 if the trunk GEMMs are still >= 1.5 ms/token; PF3 if the links idle or slow under compute; PF5 as a measurement beside either.
3. PF4 and PF6 only if their targets appear in PF0. PF7/PF8 parked.
- **Realistic total, if PF0 confirms the old shares:** PF1 and PF2 together could take 1-2 ms off 5.6 (20-35 %) -- an **estimate** with the per-card idle share and the GEMM speed-up as its two unknowns; if PF0 says the links are the limit, the answer is "little", and the plan ends at PF3/PF5 with a record entry.

## 5. Stop rules

- PF1 stops if PF0 shows the cards busy >= 85 % or the links the binding resource (a pipeline deeper than the links can feed gains nothing).
- PF2 stops if the microbenchmark does not reach 1.5x `k_gemm_lds`'s TFLOPs on the K-quant shapes, or the quality gate shows a drop outside the Wilson interval of the previous binary on MMLU-Pro or any needle miss that the old binary got right.
- PF3, PF4: stop if the predicted gain is < 0.3 ms/token after PF0.
- Any item whose exactness gate shows a non-exact tap in a change that claims "no bit moves" is fixed or dropped, never shipped behind a quality claim; a deliberate order change goes through section 2 rule 3 from the start.
- A speed claim needs the timeline or A,B,B,A palindrome AND the serve-path number (`accept_live` cold new chat, a 25 k-token document through the gateway). One odd arm is not a result.

## 6. Outside prior art (collected 2026-10-08 night; unverified unless stated)

[A research pass over llama.cpp / ik_llama.cpp / vLLM / SGLang / KTransformers / FLA / ROCm threads was running when this file was committed; its findings are added in the next commit.]

## 7. Owner's decisions

1. Whether prefill is worth the next weeks at all (section 0: it shows in cold chats and big documents, not in tokens-per-second in chat).
2. The quality bar for a non-bit-exact prefill kernel: "inside the Wilson interval of the previous binary on MMLU-Pro 70 and no needle regression" is proposed; the owner may want stricter.
3. GitHub pushes as always (go-ahead plus secret scan).
