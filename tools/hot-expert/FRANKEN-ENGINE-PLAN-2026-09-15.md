# Franken-engine plan: a VRAM-resident model class on rome, and what to do about it (Fable, 2026-09-15)

> Written at the Fable tier, from the Mac, while the rig lock was held by a
> benchmark with ~7.5 h to run. Nothing here was run on the rig. Every number
> is one of three kinds and is labelled: **measured** (tonight's context ladder
> on this rig, or a row in the record / roadmaps / CLAUDE.md), **published**
> (the three reference projects' own figures, on their hardware, not ours),
> or **projected** (derived here, with the derivation shown). A projection is
> a thing to be beaten or refuted by the items below, never a result.
>
> The question the owner asked is "should the compute backend move to HIP, and
> should this box serve a 27–35B VRAM-resident model instead of dragging 180 GB
> through host RAM". This plan's answer, before any measurement: **those are two
> questions, the second one is the real one, and the first one is mostly
> already answered by the profile.** Sections 1–5 say why and what to measure.
>
> **Rev 102 (2026-10-09 CEST) -- PF0 DONE, AND IT FOUND THE BIGGEST PREFILL LEVER LEFT: THE SHIPPED PREFILL'S CHUNKS BARELY OVERLAPPED. One flag takes prefill from 5.61 to 3.60 ms/token (-36 %) on the prose prompt and from 6.22 to 3.62 (-42 %) on a technical document, bit-exact, installed.**
> Record §L5-PF0, engine notes `GLM5.md` section 23, `PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md` section 0b. **Finding.** The in-place device timeline (it works in place; the link rate comes from `--profile`) shows 1.59 chunks in flight and a period of 5.617 s = the latency of ONE chunk: the first MoE layer's helper packet (two `hipMemcpyPeerAsync`) of every chunk takes 3.6 s because the runtime completes peer copies into a card in ISSUE order, whatever the stream, and it queues behind the previous chunk's last-layer packet, which waits for that layer's plan (5 of 5 chunks, to within 1-11 ms). Not the SDMA engine and not the queue count (`HSA_ENABLE_PEER_SDMA=0`, `HSA_ENABLE_SDMA=0`, `GPU_MAX_HW_QUEUES=16` change nothing). **Fix:** `--glm-help-copy 1` (launcher default since today): the packet as a copy kernel on the helper's own stream; same process, A,B,B,A: 5.615 -> 3.601 (prose), 6.216 -> 3.624 (technical text); 2.88 chunks in flight; the busiest card is now card 2 (main stream 99.7 %). **Exactness.** `--pipeline-taps 1` (new) lets an oracle compare a PIPELINED prefill (no earlier gate ever ran overlapping chunks, because `--time-prefill 0` flushes every chunk); reference = a serialised run of the same flags (`--gemm-lds 1` is a last-bits order change against T = 1, so a chunk-1 reference can never be bit-equal: the first version of the gate read cos 0.96-0.99 for that reason alone); 7 configs x 1 788 taps, not_exact = 0. accept_live PASS, browser first token 1.05 s, engine prefill of the 4.6k-token cold tool prompt 19.15 s against 27.35 s, **needle 8/8 at 30k / 60k / 120k / 200k** (115 / 224 / 442 / 735 s against 191 / 415 / 821 / 1 366 s; also action V2: the first needle run after the D14 fix). Installed `franken_dec_glm.hc` `b875b844`, rollback `.d14`. **The rest of PF0.** Skip classes (ms/token of 5.60): skeleton 2.94 (the schedule, not work), trunk GEMMs +2.05, attention +0.45, KDA +0.10, indexer +0.04, routed experts +0.02 (they cost nothing on the critical path); a 1 024-row chunk uses 264-288 of 288 experts a layer, hit rate 35-38 %, the hindsight-best prefill resident set saves <= 2.3 % of the slab reads; links balanced and ~38 % busy; depth flat (6.16 -> 6.20 ms/token, 2 k to 64 k). **Plan consequences.** PF1 delivered; PF3, PF4, PF5, PF6 dropped (no target in place / stop rule met); PF2 gated on the new PF9 (re-attribute the new schedule); new PF10 (balance the cards) and PF11 (the skeleton's unexplained ~80 ms a layer on the owner). **Not verified:** exactness beyond a 6 144-token prompt except through the needle answers; the sustained speed of a long prompt with the fix; the mechanism beyond the 5/5 coincidences and the three negative controls.
>
> **Rev 101 (2026-10-08 CEST, night) -- THE NEXT ROUND IS PLANNED, NOTHING IN IT IS EXECUTED: the decode plan gets a second round (section 8) and prefill gets its own plan, because the two have different gates.**
> `DECODE-OPEN-ITEMS-PLAN-2026-10-08.md` section 8: **V1** an external reference past 2 051 tokens (the engine's CPU arm, full model, 2 300-token prompt, against GPU chunk 1 / 512 / 1 024: the old gates compared the engine with itself and the D14 fault hid in two chunked runs that shared a last chunk), **D16** the per-token `ms` vector (are the 24-45 ms host gaps real?), **D15** where the fetch model's other two thirds go (up to ~2 ms), then the D10 CPU lane (1.3-2.7 ms) only if D15 and the owner say so; D9, D8, D13, D11, D12 stay parked; realistic best case 2-4 ms of ~60. New `PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md` (items PF0-PF8): prefill is 5.6-5.8 ms/token (~178 tok/s) and shows in cold chats (28.4 s for the 4.6k-token tool prompt) and big documents, not in chat speed; a prefill change that reorders a sum (a K-quant WMMA / tile GEMM) is judged by the quality gate, not by bit-identity, and by the device timeline, not the decode trace. **PF0 first**: the only attribution of prefill time (trunk GEMMs 2.45, experts 2.33, attention + indexer 0.49 of 9.0 ms/token) is from the old staged chunk-512 config, so the shipped build is un-attributed; the per-card utilisation (~50 %) is not investigated and the pipeline is two banks deep by construction. Candidates gated by PF0: deeper pipeline across the three cards, a K-quant GEMM, compute-beside-copy interference, chunk 2 048 (VRAM-blocked), a prefill resident set, long-prompt scaling; realistic best case, only if PF0 confirms the old shares, 1-2 ms of 5.6 (an estimate). Outside prior art (section 6 of the prefill plan, a research pass; unverified unless said): llama.cpp's layer-split pipeline parallelism and the CUDA-only TurboPrefill (1.6-3x at 16 K, author-reported) support a deeper pipeline across cards, whose bubble 2 / (2 + M) pays on long prompts only; stew675's bf16-WMMA dequantising GEMM measured +11-14 % whole-model on a 7900 XTX over int8 MMQ; 300-400 tok/s is called realistic with host-resident experts (anecdotal), against ~178 here. The order of work, with a status per item, is kept in `ACTION-LIST.md` (queue: PF0, D16, V1, V2 = the needle re-run on `.d14`, decision G1, D15, decision G2; checked by `doc_currency.sh`). Installed binary unchanged (`.d14` `f365fc7c014696ca`). The service is still OFF.
>
> **Rev 100 (2026-10-08 CEST, night) -- D14 SOLVED: the non-reproducible decode past 2 051 tokens was a QUALITY BUG in the prefill attention (a grid sized from the last row of a chunk), not a race; fixed, gated bit-exact, INSTALLED.**
> Installed `~/bench/franken_bin/franken_dec_glm` = `franken_dec_glm.d14`, sha256 `f365fc7c014696ca`, franken-engine code `a7bf81a` (main `f072093`, GLM5.md 22.6); rollbacks `.fin` `ca2bc21c19b451c0`, `.gr` `1f95c34b2d8bbf06`, `.rs` `c21f9a663b28c135`, `.adm` `c97b0eb293f4d181`. `mla_attn` launched the attention chunks for the LAST row of a prefill chunk; in the sparse regime a row has 2 048 + (p + 1) % 4 keys, so a chunk ending on a pool boundary (every chunk of 256 / 512 / 1 024) gave its rows with a 1-3 key tail 64 chunks instead of 65: chunk 64 was never written and the combine folded in stale buffer content (three rows in four past 2 048 tokens lost their newest 1-3 keys in all 11 DSA layers; the result also depended on what earlier work had left in the buffers). Found by reading what the oracle's taps are (the last PROMPT row, not a decode step -- the D5 reading was wrong, erratum in the record) and confirmed by an independent read-only Opus audit; the prediction that a 2 301-id prompt is reproducible held (8 of 8 configurations exact). Size of the error before the fix: logits of the last prompt row cos 0.9895 (maxabs 1.56) against the chunk-1 reference at a 2 300-token prompt; nothing below 2 048 tokens. Gates: chunk 512 / 256 / 333 / 512 against a chunk-1 prefill: old 1 655 of 1 788 taps not exact each time, fixed 1 788 of 1 788 exact; eager vs `--hip-graph 1` decode exact (old: ids differ at the 5th token); model gate `GLM5 GPU GATE PASS` (12 configurations, maxabs 0); prefill time unchanged (31.18 s for 2 300 tokens, old and new); `accept_live` PASS (warm new chat 0.84 s), browser first token 1.06 / 1.31 s. NOT verified: agreement with an external (llama.cpp) reference past 2 051 tokens (the reference was the engine's own T = 1 path), the effect on long-chat answer quality beyond the logits error, a serve-path A/B of the fix. Record `§L5-GLM-D14` (new at the end; `§L5-GLM-D5` carries a pointer erratum), tooling `glm_d14_chain.sh`, `d14_report.py`, `build_d14.sh`. D14 closed; open: D15, D13, the per-token `ms` vector. The service is still OFF.
>
> **Rev 99 (2026-10-08 CEST, evening) -- THE DECODE OPEN-ITEMS PLAN EXECUTED (Phases 0-1 and what they opened): a specialised Q8_0 decode GEMV (`--gemv-q8fast`, bit-exact, layer period 1 403 -> 1 373 us, paired token wall -1.49 ms) and a rows-based waves rule are INSTALLED; the Q6_K split fault is fixed; the decode non-reproducibility past 2 051 tokens is localised and is NOT what the plan suspected.**
> Installed `~/bench/franken_bin/franken_dec_glm` = `franken_dec_glm.fin`, sha256 `ca2bc21c19b451c0`, franken-engine code `f0fda34` (main `96ff255`, GLM5.md section 22); rollbacks `.gr` `1f95c34b2d8bbf06`, `.rs` `c21f9a663b28c135`, `.adm` `c97b0eb293f4d181`; launcher defaults `FRANKEN_GEMV_Q8FAST=2`, `FRANKEN_GEMV_ROWSPLIT_WAVES=1` (the wrapper forwards the first; an older binary ignores both). Accepted: model gate `GLM5 GPU GATE PASS` (12 configurations x 1 788 taps, maxabs 0), `accept_live` PASS, browser first token 1.06 / 1.05 s. Outcomes per item (record sections `§L5-GLM-D1D3`, `-D2`, `-D9A`, `-D4`, `-D5`, `-D7`, `-D6`, all new at the end of the record; the table is section 0 of `DECODE-OPEN-ITEMS-PLAN-2026-10-08.md`): **D7** the bandwidth gate was the surprise: the exact-order 16-byte-load kernel (D7b) is SLOWER than a plain copy of the engine's own Q8_0 loop, and that copy (new item D7c) is 17-21 % faster per launch than the generic `k_gemm_batch`; in the paired kernel trace it takes the kernel sum -2.00 ms and the token wall -1.49 ms (SE 0.08, 17 of 17 tokens, -2.3 %); D8 parked; **D3** the rows rule -0.12 ms; **D1** closed (the knob is global and fails its gate); **D2** closed (the 22 % slow launches were the old kernel); **D4** fixed (a flat load below the LDS allocation in the Q6_K staged-x path, an address-only fix, verified on the card); **D9a** D9 parked (modeled +1.9..2.6 ms, realization 0.33 -> 0.6..0.9 ms < 1.5); **D6** the serve-A/B sampler exists and was used; **D5** decode past 2 051 tokens is not reproducible even inside one process: a timing-dependent perturbation of layer 4's carried KDA conv window from decode step 2-3 on, not removed by serialising launches (new item D14). **Serve-path A/B, two runs (14 arms):** ten arms are undisturbed and give +2.7 % (+2.5..+2.8 % on every request type, <= 0.06 tok/s spread inside a build); four arms were disturbed (three slow, one cold-start fast, both builds hit, cause unidentified) and are named and left out; with all 14 arms the mean difference is +0.6 %. The install rests on the exactness gate, the paired trace and the acceptance run; the A/B agrees where the box was undisturbed. New items D13 (Q6_K head, ~0.1 ms), D14 (root cause of D5, correctness), D15 (where the fetch model's other two thirds go). The service is still OFF.
>
> **Rev 98 (2026-10-08 CEST) -- A PLAN FOR THE OPEN DECODE ITEMS: `DECODE-OPEN-ITEMS-PLAN-2026-10-08.md` (twelve items D1-D12 in four phases, with gains labelled measured / estimate / unknown, gates, tiers and stop rules).**
> Phase 0 (cheap, first): D1 `--gemv-lds` speed for GLM, D2 the 22 % slow `hc_*_fn` launches, D3 a rows-based waves rule, D4 the Q6_K nsplit-16 stall in main's reference backend (correctness insurance), D5 decode non-reproducibility past 2 051 tokens (verification coverage), D6 the odd serve-A/B arms. Phase 1 (decision gates): D7 a 16-byte-load bandwidth microbenchmark, D9a re-sizing the tensor-level expert split with the corrected 62-64 GB/s bound. Phase 2 (only if a gate passes): D9 the split (1-2 ms), D7b + D8 wide loads and the KDA o / DSA o tail (<= ~1 ms together). Phase 3 (parked): D10 the CPU lane (1.3-2.7 ms), D11 persisting the placement average (probably ~0: a cold start decodes as fast as later requests), D12 the DeepSeek adapter check. If everything in Phases 0-2 lands: about 2-4 ms (3-6 %) more (estimate). The hardware option stays closed while all seven slots are occupied. No new measurement in this revision.
>
> **Rev 97 (2026-10-08 01:10 CEST) -- THE GROUPED GEMV (candidate B) LANDED AND INSTALLED: BIT-EXACT, 162 FEWER LAUNCHES A TOKEN, -38 US A MoE LAYER (~ -1.6 MS A TOKEN) IN THE PAIRED KERNEL TRACE, +2.5 % DECODE ON THE REAL GATEWAY (record §L5-GLM-GEMV-GROUP).**
> `--glm-gemv-group` (franken-engine main `68e31cd`, kernel code `1d12b52`; engine default off, launcher default on): the KDA q|k|v, KDA f_a|beta|g_a and the indexer's key|gate|weights GEMVs, which read one activation, as ONE launch each. The rule that keeps it exact: every member keeps the split count it has ALONE, so a group merges only when all solo counts are equal and the launch is forced to that count (the concatenation's own formula would give 52 where f_a, beta and g_a each split 128 ways); anything else runs the solo launches. Exact: 572 microbenchmark configs EXACT, 8 gate configs x 1 788 taps maxabs 0 (rowsplit on and off, swap every token, gemv-lds 0, a non-scoring context), device code byte-identical to main's. Faster: layer period 1 439 -> 1 401 us (-38 us), per-token wall -1.64 ms (SE 0.15), serve path +2.1..+3.0 % per request in two orders. Installed `franken_dec_glm` = `.gr` 1f95c34b (rollbacks `.rs` c21f9a66, `.adm` c97b0eb2). With rowsplit: ~-3.3 ms (-5 %) of the decode token in the trace from the trunk-GEMV work. Lessons: (1) the serve-path A/B still shows an odd arm now and then (two of sixteen arms so far moved 5-7 % for no visible reason): run both orders and quote the pairs that agree; (2) the trace out-measured the microbenchmark (-1.6 vs -1.0 ms): dependent launches in the real graph cost more than the benchmark's chain. Remaining in the trunk-GEMV list: E (the tail of KDA o / DSA o: 1.33 residency rounds), G (`--gemv-lds` for GLM: plumbed, bit-exact in the gate, speed untested), wider weight loads, a rows-based waves rule (~0.13 ms), the unexplained 22 % slow hc-fn launches (~0.25 ms).
>
> **Rev 96 (2026-10-07 22:30 CEST) -- THE ROW-SPLIT GEMV (A+) LANDED AND INSTALLED: BIT-EXACT, -1.7 MS A TOKEN IN THE PAIRED KERNEL TRACE, +2.8 % DECODE ON THE REAL GATEWAY (record §L5-GLM-ROWSPLIT).**
> `--gemv-rowsplit` (franken-engine main `fa1c233`, kernel `e10f196`, engine default off, launcher default on): one launch per split GEMV, one workgroup a row, the waves loop over the row's splits with `k_gemm_batch`'s own per-split code, partials summed in LDS in split order. Exact: 319 microbenchmark cases, 18 gate configs x 1 788 taps maxabs 0, identical adapter statistics on the gateway; no new VRAM. Faster: MoE layer period 1 477 -> 1 436 us (-41 us a layer), paired per-token wall -1.68 ms (SE 0.13) on 15 clean tokens, serve path +2.4..+3.2 % per request in two orders of a four-arm A/B; the in-process wall-clock medians could not resolve it (+-2 ms intervals) -- lesson: a ~2 ms effect needs the kernel trace or the serve-path A/B (same-build arms agree to 0.4 % there), not 32-48-token medians. Installed `franken_dec_glm` = c21f9a66 (rollback `.adm` c97b0eb2). Next in the trunk-GEMV list: B (merge same-input GEMVs, nsplit forced; -0.3..-0.6 ms KDA q+k+v, ~-0.4 ms f_a+beta+g_a), E (tail of KDA o / DSA o), G (`--gemv-lds` for GLM: now plumbed, untested), wider weight loads, a rows-based waves rule (~0.13 ms). Open: the synthetic Q6_K nsplit-16 case stalls the card in main's own reference backend (production never runs it).
>
> **Rev 95 (2026-10-07 19:00 CEST) -- THE TRUNK GEMVs: 278 SMALL K-SPLIT LAUNCHES COST 5.4 MS A TOKEN TO READ 0.33 GB (WORKGROUP-COUNT BOUND); BUILDING THE BIT-EXACT ONE-LAUNCH ROW-SPLIT KERNEL (A+); 2.2 MS OF THE "18.9 MS" IS HIDDEN SHARED-EXPERT WORK (record §L5-GLM-TRUNK-GEMV).**
> Offline analysis of three existing decode traces (no GPU) plus outside sources: the big trunk matrices run at 590-720 GB/s (74-90 % of 800) and have little to give; the 278 split-K launches (hc fn, f_a/g_a/indexer k+gate, beta, indexer proj, kv_a, router) obey t = 4.0 us + 3.62 us per 1000 workgroups, each followed by a 4.8 us reduce launch and two ~3.3 us gaps (5.4 ms a token in total, on the critical path); the shared expert (4.35 ms) is 79-100 % overlapped with the fetch and worth 0.02 ms. Decision: build A+ (one workgroup a row, the waves loop over the row's splits with the SAME per-split code, partials in LDS, one lane sums them in s order: bit-identical, no second launch), expected -2.0 to -3.5 ms a token; branch `gemv-rowsplit`, flag `--gemv-rowsplit` default off until gated; a standalone bit-exactness + timing microbenchmark over the real shapes first. Queued behind it: B (merge same-input GEMVs, nsplit forced), G (`--gemv-lds` for GLM), E (tail of the two long matrices), wider weight loads. The outside research says the 800 GB/s bound is conservative (952 GB/s measured with 16-byte non-temporal loads) and that our Q8_0 loop issues byte loads. Nothing measured on the new kernel yet.
>
> **Rev 94 (2026-10-07 18:20 CEST) -- THE CPU LANE IS FEASIBLE ON ALL THREE COUNTS AND WORTH 2-4 %: THE NUMERICS COST 0.54 % OF A LAYER'S MoE OUTPUT FOR ONE SLOT OF EIGHT (record §L5-GLM-CPULANE-QUALITY); RECOMMENDATION: PARK IT, BUILD THE TRUNK GEMVs FIRST.**
> The three questions that could kill it are answered: sync 20-40 us a layer (rev 92), 6 of 8 physical cores free (rev 93), numerics: int8 activations on one expert slot change the layer's MoE output by 0.54 % on average (max 2.6 %), a third of what llama.cpp
> injects in every layer (672 real expert slots, two prompts; harness reproduces the GPU taps to 2e-7). The lane is a quality-GATED change (off by default, quality chain on, logit diff in the commit). Gain: modeled 4-5.4 ms a token, realistically 1.3-2.7 ms
> (2-4 %), a multi-day engine build (plan -> host, pinned worker pool with spin barriers on cores 2-7, engine threads pinned to 0-1, ggml kernels linked, slot-ordered accumulation, quality gate); a faster IQ3_S AVX2 kernel (2.6 GB/s a core) could double it.
> **Parked, fully specified in the records; the trunk GEMVs (<= ~6 ms, bit-exact if the lane order is kept) rank above it.**
>
> **Rev 93 (2026-10-07 18:00 CEST) -- THE CPU LANE'S CORE BUDGET IS MEASURED: THE SERVING ENGINE USES TWO CORES IN DECODE (TWO BUSY-WAITING THREADS) AND NONE IDLE; 6 OF 8 PHYSICAL CORES ARE FREE (record §L5-GLM-CPULANE-CORES).**
> A 6-worker lane (0.57 ms an expert) fits on cores 2-7 with the engine's two threads pinned to cores 0-1 (they float today). Of the three questions that could kill the lane, two are now answered in its favour (sync cost: rev 92; cores: this rev);
> left: the numerics (a quality-gated int8-activation lane; an offline experiment on real expert weights and real activations is being written) and the IQ3_S kernel speed. The modeled gain stays 4-5 ms, realistically 1.3-2.7 ms (2-4 %).
>
> **Rev 92 (2026-10-07 17:30 CEST) -- THE CPU LANE'S SYNC COST IS MEASURED: ~20 us A LAYER IDLE, 32-41 us UNDER THE FULL STREAM (record §L5-GLM-CPULANE-SYNC); THE MODEL'S 60-120 us WAS 2-4x TOO PESSIMISTIC.**
> Round trip GPU kernel -> pinned host flag -> spinning host thread -> pinned result -> GPU wait kernel: 3.7 us bare, 13 us with 16 KB each way; the full layer pattern adds 16-22 us (spin wait), 20-26 us (`hipStreamWaitValue32`), under the three-card
> stream 32 / 41 us, and the GPU streams keep 99.4-100.1 % of their rates. The spin kernel is the choice. Modeled saving with d = 35 us: 4.0 ms (5 cores), 5.4 ms (6 cores), 7.5 ms (8); at the realisation seen for the whole-slab table (1/3 to 1/2) 1.3-2.7 ms
> = **2-4 % of a decode token** for a multi-day engine build. Still open: the numerics (a quality-gated int8-activation lane vs the GPU's float), the cores free in decode, a faster IQ3_S AVX2 kernel (the lever that could double it). **Recommendation:** measure the cores and the quality
> effect (cheap) before any engine work; do not start the lane build for 2-4 % alone.
>
> **Rev 91 (2026-10-07 14:50 CEST) -- THE CPU LANE IS MEASURED: A REAL EXPERT IN 0.57-0.68 ms ON 5-6 CORES WHILE THREE GPUs STREAM, WITH ~3-4 % LOSS ON EITHER SIDE; MODELED 3-5 ms A TOKEN, THREE OPEN QUESTIONS (record §L5-GLM-CPULANE).**
> Standalone benchmark `tools/hot-expert/cpulane/` (ggml AVX2 kernels, the real IQ3_S + IQ4_XS expert): one expert split over 5 / 6 / 8 cores 0.68 / 0.57 / 0.44 ms (17 / 20 / 26 GB/s, linear, COMPUTE-bound: the weak link is the IQ3_S
> kernel at 2.6 GB/s a core); with the CPU working while the three cards stream (aggregate 61.9 GB/s alone, §L5-GLM-FETCH-FLOOR-ERRATUM confirmed) the GPU side loses 1.5 / 2.8 / 4.1 % at 2 / 4 / 6 workers and the CPU 0-3 %.
> The FreeToken worry (DDR contention) does not bite with the real mix; it does with the memory-hungrier IQ4_XS kernel above ~35 GB/s. Modeled saving 3.7-5.0 ms at d = 60 us, honest range 1.5-5 ms (2-8 %). **Open, each can kill it:**
> (1) numerics (ggml quantises activations to int8, the GPU kernels use float: a quality-gated change, or slower float AVX2 kernels), (2) how many cores the engine's own host threads leave free in decode, (3) the GPU-host-GPU sync cost per layer
> (a microbenchmark answers it in an hour; the lookahead work found 0.8 ms a layer of fixed stream/event cost). Next: the sync microbenchmark, then the IQ3_S kernel, then the numerics decision.
>
> **Rev 90 (2026-10-07 13:45 CEST) -- `--fetch-assign optimal` IS A REAL -0.97 ms A TOKEN (-3.0 % OF THE CRITICAL FETCH) AND IS NOW THE DEFAULT; THE MODEL HAD PREDICTED A THIRD MORE (record §L5-GLM-FETCH-FLOOR-ERRATUM addendum).**
> Three rocprofv3 decode traces of identical flags, per-token critical fetch (the span of the three cards' stage kernels per layer, summed): pattern 32.64 and 32.63 ms (noise floor 0.01 ms), optimal 31.68 ms, fewer in 10 of
> 12 tokens; the per-card kernel durations follow the link model (417 us a slab on the lone card; 417 / 678 us on the pair when its partner is idle / busy). The decode-timing A/B of rev 87 could not resolve a 1.4 % effect.
> Bit-exact, `FRANKEN_GLM_FETCH_ASSIGN=optimal` in the start script (the wrapper's allowlist forwards it). A tensor-level split would be worth ~2 ms if a third of the model carries over: low priority.
>
> **Rev 89 (2026-10-07 13:30 CEST) -- CORRECTION OF REV 88 POINT (1): THE FETCH IS NOT "AT THE LINK CEILING"; THE FINER-SPLIT LEVER IS OPEN AGAIN, CEILING ~5.6 ms A TOKEN (record §L5-GLM-FETCH-FLOOR-ERRATUM, §M4-TOPOLOGY).**
> Rev 88 divided the fetch bytes by 52 GB/s, which is §M4's EQUAL-split figure (total bytes over the slowest card's time), and concluded the shipped slab table was 0.2 ms from a perfect split. With the loads
> balanced the three links run at once at 28 + 18 + 18 = 62-64 GB/s (the engine prints 62.4): the critical fetch (1 686 MB in 35.2 ms = 47.9 GB/s) is at 77 % of that, the best whole-slab split models at 32.6 ms and a
> perfect continuous split at 27.0 ms (5.6 ms below the table; `ckpt1006/lane_model.py`). The unexplained part: `--fetch-assign optimal` (modeled -2.6 ms) showed nothing measurable, so expect a fraction of 5.6 ms, and
> find out why before building a tensor-level split (gate+up | down on two cards: bit-exact, one more ~30 us hand-off per cut expert). Points (2)-(4) of rev 88 (the placement cap, the adapter position fix, G=8) and the
> installed build stand; the CPU-lane arithmetic never used the aggregate and stands. **Topology (§M4-TOPOLOGY):** all three cards run PCIe 4.0 x16; 83:00.0 and 86:00.0 hang off the SAME root complex (`0000:80`) and
> together reach 36 GB/s, 48:00.0 is alone on `0000:40`; root complex `0000:c0` carries no device. **A hardware option for the owner:** if a ROMED8-2T x16 slot hangs off `0000:c0`, moving the 86:00.0 card there
> would let all three run at 28 GB/s (84 GB/s, +31 %): the balanced ideal fetch 27.0 -> ~20 ms a token. Untested; the board manual's block diagram says whether such a slot exists, `m4_chain.sh` measures it.
>
> **Rev 88 (2026-10-07 11:55 CEST) -- THE PLACEMENT POLICY WAS THE DECODE LEVER: ADAPTATION WAS DEAD AFTER THE FIRST LONG REQUEST OF A SERVING PROCESS (fixed, serve path +15 %);
> A PREFILL CAP FOR THE PLACEMENT AVERAGE (-4 to -24 % in the gates); G=8 SHIPPED (-2.6 % prefill); THE FETCH-BALANCE LEVER IS CLOSED BY ARITHMETIC (records §L5-GLM-G8,
> §L5-GLM-FETCH-FLOOR, §L5-GLM-ADAPT, §L5-GLM-ADAPT-MONO, §L5-GLM-DECODE-TRACE2).** (1) **Fetch is link-bound:** the critical fetch (35.2 ms a token) moves 1 686 MB at 48 GB/s = 92 % of M4's
> 52 GB/s for all three cards; the "8 ms of imbalance" of rev 87 divided by the sum of single-link rates (62.4), which M4 refutes; at 52 GB/s a perfect split is 0.2 ms better than the shipped
> slab table (offline model, `ckpt1006/lane_model.py`). Only FEWER MISSED BYTES help (a CPU lane that avoids the links is the other structural option: 3-5 ms if an AVX2 expert kernel reaches
> ~20 GB/s, unmeasured). (2) **The placement average is the prompt:** a held prefill enters it as one snapshot of the whole span, so after 8 192 prompt tokens decode cannot move the placement
> (hit 0.61 -> 0.56 over the decode). `--adapt-prefill-cap N` (a span counts as at most N tokens; shipped 64 with half-life 512): chat prompt 53.8 -> 51.2-51.5 ms, 8 192-token prompt 64.8 ->
> 49.5-56.7 ms, missed MB a token -40..-66 %, bit-exact (1 788 taps x 4 with a swap round every token). (3) **The serving bug:** `Adapter::tick` ran its cadence on the runner's SEQUENCE position, which
> jumps back for every new chat; a chat shorter than the previous one's end took no snapshot at all (the acceptance log of the cap build: ONE adapter window, none after). A monotonic position fixes it;
> serve-path A/B (real gateway, greedy, old build against new, two arms each): four short chats after a long request 13.5 -> 15.6 tok/s mean, the repeated prompt +19 %, windows 0 -> 25 a chat, identical
> text; persistent 14-chat sessions (a new topic every chat): old 14.5, fixed 16.1-16.5 tok/s, and with half-life 2048 only 14.7-15.5 (the half-life matters as much as the fix; 128 and 512 are alike, shipped 512); the CLI gates are unchanged (their position never rewinds). The DeepSeek adapter (`ds4_adapt.cpp`) has the same condition: not tested, not fixed. (4) **G=8:** `FRANKEN_GLM_MOE_G=8` bit-exact at chunk
> 1024 / 512 / 32 in place, prefill 5.769 -> 5.621 ms/token; the rig's docker wrapper dropped knobs not on its env allowlist (now fixed). franken-engine main `aa7ff56` (engine code `5b47ef9`, notes GLM5.md
> section 19). **Both serving builds passed acceptance (accept_live PASS + browser first token 1.06 / 1.31 s) and the last is INSTALLED (~11:50 CEST): `~/bench/franken_bin/franken_dec_glm` = `.adm`, sha256
> `c97b0eb293f4d181`; rollbacks `.cap` `7751760b419de4bc`, `.wr` `11acdb9135664f19`. The service is still off (rig reserved).** Open decode levers, by size: placement follow-ups (the DeepSeek port, persisting the
> average across restarts, watching `adapt_all` counts in the real service), the trunk GEMVs (18.9 ms a token at ~440 GB/s: <= ~6 ms realistic, quality gate if the split changes), the CPU lane.
>
> **Rev 87 (2026-10-07 03:20 CEST) -- DECODE -8.2 % MORE: THE SEPARATE REDUCE ONE WAVE A ROW (`k_reduce_splits_gemm_w`), BIT-EXACT; FETCH-ASSIGN
> BUILT, EXACT, NO MEASURABLE GAIN (record §L5-GLM-GEMV-REDUCE addendum).** A second trace showed the new hot spot of rev 86's change: the thread-per-row reduce
> (278 launches a token, ~20 us each, 5.5 ms; up to 512 dependent loads a row). One wave a row, the lanes loading the partials in parallel into LDS, lane 0 summing
> them in the same order: A 68.23 -> B 62.62 ms/token in one process (A,B,C,D,D,C,B,A); both landed changes together ~-10.5 % against the installed build
> (14.3 -> 16.0 tok/s). franken-engine main `24b2826`. `--fetch-assign optimal` (an optimal split of a layer's missed slabs over the three cards by miss count,
> simulated -8.6 % of the critical fetch) is exact but shows nothing on top of B (D/B 0.997): off by default. Process: `run_chain.sh` now runs a private copy of the chain
> script (an scp over a running chain made bash read the new file mid-run), the staged-prefill gate line goes last. **The build carrying both changes passed acceptance and is INSTALLED
> (2026-10-07 03:35 CEST): `~/bench/franken_bin/franken_dec_glm` = `franken_dec_glm.wr`, sha256 `11acdb9135664f19`, franken-engine `24b2826`; `accept_live` PASS (warm new chat
> 0.89 s, 4 548 of 4 576 tokens reused; follow-up turn 1.2 s; a request behind an abandoned one 1.4 s; ledger 0 mismatches), `accept_ui` browser first token 1.31 s on both
> chats; rollback `franken_dec_glm.final` (sha 90f5e845306f79e0). The service is still off (rig reserved).** Open decode levers, by size: the
> trunk GEMVs and the remaining reduce (the 24-row matrices ~10 us each), the critical fetch (~35 ms of ~63: a finer split of the missed experts over the links up to ~8 ms).
>
> **Rev 86 (2026-10-07 02:20 CEST) -- DECODE -4.7 % FOR FREE: THE SPLIT-GEMV REDUCE AS A SEPARATE LAUNCH (record §L5-GLM-GEMV-REDUCE),
> BIT-EXACT, A FLAG; GLM DEFAULT NOW.** The decode trace (§L5-GLM-DECODE-TRACE) showed the K-split trunk GEMVs costing ~10 us + 0.1 us x nsplit
> (the 512-split hc_*_fn GEMVs: 62 us for 0.4 MB, 5.6 ms a token): the fused reduce's one atomic counter per output row serialises. The other path
> (`k_reduce_splits_gemm`, same partials, same order) was reachable only from the Qwen CLI; franken-engine `5a4fa03` wires `--gemv-fused-reduce 0|1`
> and `--gemv-fused-max-split N` into the GLM runner and makes 0 the GLM default (CLI, gate, serve env `FRANKEN_GEMV_FUSED_REDUCE`). Exact (1 788 / 1 788
> taps, chunk 1 + 8 tokens, and 64 tokens at depth 1 500); decode 69.95 -> 66.66 ms/token (A,B,C,C,B,A, one process). **Not yet in service: the installed
> binary is unchanged; acceptance (`glm_accept_chain.sh`, then `run_ui2.sh`) on a build carrying it is the gate.** Also measured: the steady-state free
> VRAM at the shipped flags is 732 / 818 / 458 MiB a card (use it for budgets). Open decode levers, by size: the trunk GEMVs further (~10 ms, a reorder needs the
> quality gate), an optimal per-layer assignment of the missed slabs to the three cards (built, exact, no measurable gain: rev 87), a
> finer split of the missed experts over the links (up to ~8 ms).
>
> **Rev 85 (2026-10-07 01:30 CEST) -- DECODE LOOKAHEAD PREFETCH IS BUILT, BIT-EXACT AND A NET LOSS: CLOSED (records
> §L4-LOOKAHEAD-OUTSIDE, §L4-LOOKAHEAD-STEP1).** Outside sources first (llama.cpp's prefetch is prefill-only; a decode prefetch
> that syncs the host cost 8-13 % elsewhere), then a device-driven design: predict layer l+1's top-k experts on a side stream, copy the
> missed ones into VRAM rings, redirect the real plan. Exact in every comparable line (chunk-1 1 788 / 1 788 in dry and use mode, a
> 64-token decode identical across off / dry / use / use + swaps), **slower in time: 70.8 ms/token off, 95-101 ms on** at k = 2, 4, 8 --
> ~31 ms of FIXED machinery (0.8 ms a layer from a few extra streams, events and P2P hops, far above the §M5 floors) plus 3-14 us per MB
> moved against 9-27 us saved per MB covered; best case without the fixed cost <= 8 %. The fixed cost is NOT synchronisation latency (the
> decode trace, record §L5-GLM-DECODE-TRACE: helper wake 20 us, join 21 us, 65 us of a 1.6 ms layer); the added work disturbs the critical path on the
> cards (HW-queue sharing / host-read kernels beside the trunk kernels; `GPU_MAX_HW_QUEUES=16` run under way). **The trace of one decode token
> (`ckpt1006/glm_decode_trace_chain.sh`, rocprofv3): a layer is half fetch (critical ~35 ms of 76, ~8 ms of it imbalance), half trunk (~36 ms, the
> trunk GEMVs at ~40 % of VRAM bandwidth: ~15 ms headroom).** Found on the way: **decode past 2 051 tokens of
> context is not run-to-run reproducible** (two mode-0 runs diverge at the 10th generated token; below 2 051 identical), so exactness gates
> run below it. Process: three gate starts died at setup on VRAM budget (rule in CLAUDE.md, two-log watcher `ckpt1006/watch_chain.sh`,
> a gate verdict that names an engine death -- franken-engine main `89303c5`). The `decode-prefetch` branch is kept, not merged.
>
> **Rev 84 (2026-10-06 22:00 CEST) -- THE ROOT CLOCK TEST RAN (record §M7-CLOCKTEST): CLOCKS ARE NOT
> THE CAUSE OF THE SLOW COPIES; NO SOFTWARE KNOB; THE DEFAULT POWER PROFILE STAYS.** The owner ran
> `ckpt1006/clock_test.sh` once (sudo): with `profile_peak` and with the COMPUTE profile the memory /
> fabric / SoC clocks sat at their maxima (1 249 / 2 301 / 1 500 MHz) in every arm, and the prefill was
> not faster (normal arm 9.15 default, 9.90 peak = +8 %, 9.45 compute = +3 %: the lower shader clock);
> the skeleton and the loaded arms did not move. Rev 81's hypothesis (clocks fall 11-18 % under compute)
> is refuted: a bystander. Open question narrowed to arbitration inside the GPU between kernels and
> the copy engines, which no setting reaches. Under `profile_peak` one card touched 110.1 C junction
> (its critical limit); the cards are restored to `auto`. Open: G = 8; the trunk GEMMs; decode by
> lookahead prefetch; the service is still off (reserved).**
>
> **Rev 83 (2026-10-06 20:00 CEST) -- PREFILL STAGING IS OFF IN SERVICE: IN PLACE IS FASTER AT EVERY
> CHUNK SIZE AND BIT-IDENTICAL; NEW CHATS AND FOLLOW-UPS START ~1 S SOONER (records §L5-GLM-VERIFYCOST,
> §L5-GLM-STAGECROSS, §L5-GLM-SHIPPED).** (1) A probe of what a speculative-decoding verify step costs
> (GLM's GGUF does carry the NextN/MTP head, `blk.45.nextn.*`; llama.cpp merged support) showed the
> existing multi-row path at 5.4-6.9x the cost of a 1-row step for at most ~2.5 tokens: MTP does not pay
> while decode is bound by missed experts; dropped. (2) The same probe exposed that the served prefill
> stages ALL missed experts (~92 GB, ~2 s) for every chunk of more than one row. In place (only the
> experts the rows pick) is faster at every size: 16-256 rows -23..-77 %, 1024 rows 5.77 vs 6.01 ms/token
> (A,B,B,A), 512 rows 6.98 vs 7.51; bit-exact (full model + decode at chunk 1024: 1 788 of 1 788 taps; small
> chunks 1 788 / 1 788 / 124). (3) `start_franken_glm.sh` `FRANKEN_GLM_PREFILL_STAGE` default 0. Acceptance
> on the installed binary with the shipped defaults: `accept_live` PASS (warm new chat 0.86 s against 1.86;
> follow-up turn 1.1 s against 2.2-2.4) and `accept_ui` PASS (1.05 s / 1.05 s to the first token on
> screen). The needle test of the installed build: 8/8 at 30k / 60k / 120k / 200k tokens, 35-38 % faster.
> (4) An Open WebUI stale model list ("Model not found") made the first acceptance run fail for a reason
> outside the engine; the chain now refreshes it. Open: G = 8; the trunk GEMMs (2.45 ms a token, need a
> quality-gated kernel); lookahead prefetch for decode (ceiling +27 %); the owner's sudo root test
> (`ckpt1006/clock_test.sh`). The service is still off (reserved). HANDOFF-2026-10-06 §9a.**
>
> **Rev 82 (2026-10-06 18:00 CEST) -- THE GLM PREFILL SPEED-UP PASSED THE ACCEPTANCE GATE, IS
> INSTALLED AS THE SERVED BINARY AND ITS BRANCHES ARE PUSHED (record §L5-GLM-SHIPPED).**
> `accept_live` (own log) found one defect in the first candidate: with a 1024-row chunk the
> snapshot-point gap `chunk_ / 2` became 512 and a new chat reused 4 096 of 4 575 tokens (first
> token 4.04 s against 1.84 s); fixed in franken-engine `glm-prefill-final` a1362d2 (gap
> `min(chunk_ / 2, 256)`). The final candidate `franken_dec_glm.final` 90f5e845306f79e0: `accept_live`
> PASS (chat B warm 4 548 of 4 575 reused, 1.86 s) and `accept_ui` PASS in a real browser (chat A
> 2.31 s, chat B 2.56 s). Prefill at the served 262 144 cells: 9.53 -> 6.02 ms/token (-36.8 %,
> 105 -> 166 tok/s), through the gateway on cold ~8.3k prompts 9.87 -> 7.19 (-27 %), bit-exact.
> Pushed as branches `moe-regblock`, `chunk1024`, `head-gemv-blk`, `glm-prefill-final` and
> fast-forwarded into franken-engine `main` (8256d53..a1362d2, the rig's repo too);
> `~/bench/franken_bin/franken_dec_glm` = the final build (`.ckpt` is the
> rollback), `start_franken_glm.sh` defaults chunk 1024 / snapshots 1024 / ring 256 MB. **The service
> is still off** (reserved). The installed build also passes the needle test, 8/8 at 30k / 60k / 120k / 200k tokens, 35-38 % faster than the old build (record end of §L5-GLM-SHIPPED). Open: G = 8 (+1.9 %) is not
> gated for exactness; the trunk GEMMs (2.45 ms a token) need a quality-gated tile kernel; the
> clock-pinning root test needs the owner's sudo (`ckpt1006/clock_test.sh`). HANDOFF-2026-10-06 §9a.**
>
> **Rev 81 (2026-10-06 11:30 CEST) -- GLM PREFILL 9.5 -> 6.4 MS/TOKEN (-33 %, 105 -> 156
> TOK/S) WITH NO CHANGE OF A BIT: A BLOCKED EXPERT KERNEL AND A 1024-ROW CHUNK; THE
> "DMA-RATE" STALL WAS COMPUTE RUNNING BESIDE THE COPIES (records §M7-SKIPCLASS,
> §L5-GLM-CHUNK, §L5-GLM-MOEBLK).** (1) Rev 80's open factor ~2 answered with a timing-only
> `--debug-skip MASK` build (`ckpt1006/build_dbgskip.sh`): with no compute beside them the
> three links move 47.6 GB/s (91 % of M4's 52), so the engine's copy-issue structure, events,
> router/plan and boundary P2P are cleared; each compute class alone cuts the DMA (trunk GEMMs
> only 29.9 GB/s, experts only 27.8) by leaving the links idle AND slowing every copy in flight
> (-21..-27 % a link). Critical-path cost a token: trunk GEMMs 2.45 ms, experts 2.33, attention +
> indexer 0.49, KDA 0.11. A clock/power sampler: memory, fabric and SoC clocks fall 11-18 % under
> compute, PCIe stays gen4 x16, power stays under the 300 W cap, temperature is not it --
> correlation only, cause NOT established (the clock-pinning test needs root: `ckpt1006/
> clock_test.sh`, one `sudo bash ~/bench/clock_test.sh start`, restores the settings itself).
> (2) The owner asked for outside sources first: Reddit (public RSS), X (a patched twitter-cli),
> GitHub/GitLab issues. No report of the symptom; but they showed that a bigger micro-batch
> amortises expert streaming over PCIe (380 -> 2 091 tok/s on a 3090), and the engine capped
> chunks at 512 in four places. (3) **Chunk 1024:** -10.4 % at the served 262 144-cell context
> (ring 256 MB; 400 MB OOMs dev 2), bit-exact in a fresh process (1 788 of 1 788 taps against the
> stock chunk-512 dump); `--gate-plan` configs of DIFFERENT chunk sizes in one process diverge
> (state carried across configs; a harness artifact, not root-caused). (4) **`k_glm5_moe_blk`**
> (an Opus agent; branch `moe-regblock` f6c69e5): each weight row decoded once for G = 4
> assignments, same per-lane accumulation order; bit-exact in the standard `emb` gate (four exact
> configs), -16 % at chunk 512; with chunk 1024: **6.41 ms/token (G=4), 6.29 (G=8, not gated)**
> against 9.53 stock. (5) Landed ON THE RIG ONLY as local branch `chunk1024` (42d7ddc) of
> `~/src/franken-engine`; not pushed, `main` untouched; the serving path (FRANKEN_GLM_CHUNK=1024,
> FRANKEN_SNAP_EVERY=1024, FRANKEN_GLM_STAGE_MB=256) was A/B-tested through the real gateway
> (`glm_serve_chunk_chain.sh`): cold ~8.3k prompts 9.87 -> 7.19 ms/token, -27.1 % on means (-20.8 %
> conservative), about 1.8 GB VRAM free a card in service; a blocked `head_gemv` (0.5 ms a token,
> branch `head-gemv-blk` f39b449) is being gated.
> Next: HANDOFF-2026-10-06 §9.**
>
> **Rev 80 (2026-10-06 07:00 CEST) -- THE HANDOFF'S FIVE ITEMS RUN. GLM PREFIX
> CHECKPOINTS SERVE END TO END; DEEPSEEK'S SHARED CHANGE GATED AND A TRAP FOUND;
> THE GLM PREFILL STALL FOUND, HALF-FIXED AND THE HOST SOURCE EXONERATED
> (records §L5-GLM-CKPT-E2E, §DS4-SHARED-GATE, §DS4-ADAPT-GRAPH, §M6-SMALLGEMV,
> §L5-GLM-TIMELINE, §L5-GLM-EMBED, §M7-HOSTSRC, §L4-LOOKAHEAD-STEP0).** (1) GLM
> checkpoints: `accept_live` PASS and `accept_ui` PASS on the checkpoint build (a
> new chat over the 4.5k-token tool block reuses 4 548 of 4 575 tokens, first
> token 1.72 s; cold 41 s); it is now the served `franken_dec_glm`. Faults found
> in the harness, none in the engine: `accept_ui` had been testing llama-swap's
> `glm53-flash` (Open WebUI has two connections since 2026-10-05 and no default
> model; the probe now selects its model and verifies which one answered),
> conformance phase 3 sent Qwen's markers, the docker wrapper dropped the
> snapshot knobs, `serve_alt glm` left the gateway up. (2) DeepSeek's shared
> `ds4_gpu.inc` change: bit-exact in five configurations, decode unchanged in the
> served configuration (47.87 vs 47.96 ms). **New: adaptation + the staged miss
> path + graph replay together run 81 % slower (51 -> 92 ms); any two are fine;
> the service runs eager and is unaffected; rev 71's "wrong placement" reading of
> the 89-102 ms is corrected.** (3) GLM prefill: a device timeline (`--timeline`)
> found card 0's 8.4 MB embedding upload waiting ~3.0 s of every ~4.4 s chunk
> behind the staging ring's DMAs; the host is not late. Moving it to a kernel is
> bit-exact and worth -3 %; 16 hardware queues -8 % more at +8 % GLM decode (not
> for the service); with no expert kernels prefill would be -25 % (the upper
> bound for a better expert chunk kernel); the staged DMA reaches 21 GB/s
> aggregate (27 without the expert kernels) against M4's 52, and **the host
> source, its span, flags, copy size, queue count, ring size and generic compute
> contention are all exonerated (a replay of the engine's own copy lists
> reaches 57 GB/s)**: a factor of ~2 in what the engine does beside the copies
> is the open GLM question. (4) Card 0's small GEMVs (handoff item 5) are NOT the
> card: a microbenchmark reads 145-147 GB/s on all three. (5) Layer lookahead
> STEP0: one layer ahead predicts 57 % (prose) to 70 % (chat) of GLM's experts and
> covers 55-60 % of the missed bytes at ~1.15x the traffic (previous token: 24 % /
> 13 %); positive, modest, not adopted. MTP for Qwen3.8 is blocked at STEP0: the
> served GGUF has no `mtp` tensors (the FP8 checkpoint does). The two upstream
> PRs and `step0_gate.sh` were not touched. Next: the trunk-skip and consumer-
> shaped DMA probes, a routed-expert chunk kernel, the served-path snapshot D2H
> candidate, DeepSeek's embedding upload, lookahead STEP1; see HANDOFF-2026-10-06.**
>
> **Rev 79 (2026-09-26 04:20 CEST) -- GLM ON THE FRANKEN ENGINE: QUALITY =
> LLAMA.CPP'S, ~4x FASTER (record §L5-GLM-QUALITY, §L5-GLM-CKPT).** MMLU-Pro
> 70: greedy 58/70 (82.9 %), sampled as the model card says 60/70 (85.7 %),
> p = 0.63; needle 8/8 at 30k-200k. The misses are the model's: llama.cpp,
> greedy, on the 12 Franken misses gets 1 right (a near-tie); the
> budget-exhausted answers are overthinking at xhigh (all-distinct tails),
> in both engines. Prefix checkpoints bit-identical (2/2). Setup faults
> fixed on the way: a leaked FRANKEN_BIN, a 4 096 max-tokens cap. Next:
> GLM with checkpoints end to end through the gateway and Open WebUI
> (accept_ui), then the pipeline between cards (dev0 busy 30 % in prefill),
> the DeepSeek gate for the shared snapshot change.**
>
> **Rev 78 (2026-09-25 09:20 CEST) -- GLM PREFILL 1.8x WITH THE LDS GEMM
> (record §L5-GLM-LDS); GLM BEHIND THE GATEWAY BUILT. The rig is reserved
> for development (owner, 2026-09-25): nothing serves on 8081, chains leave
> the gateway down (`~/bench/.dev_reserved`).** Chunk 512 staged prefill
> 16.9 -> 9.3 ms/token with `--gemm-lds 1` (A,B,B,A), decode 67-70 ms at 8k
> unchanged; bit-identical paths all exact; the LDS kernel moves one
> near-tied expert at the last layer (greedy exact) -- judged by the quality
> run, as on Qwen3.8. The counter read-back fix measured nothing. GLM serve
> port (franken-engine `glm-serve` b9bd25d, colibri `franken-glm` arm,
> live-prefix reuse only, no checkpoints yet) is under its first serve test.
> Next: serve test -> GLM quality (Franken with LDS vs llama.cpp) -> the
> pipeline between cards (dev0 busy 30 %) -> checkpoints for prefix reuse;
> the DeepSeek gate for the shared snapshot change.**
>
> **Rev 77 (2026-09-25 08:00 CEST) -- GLM STAGED PREFILL LANDED, +4 %; THE
> LEVER IS CHUNK SIZE (record §L5-GLM-STAGE).** Missed experts staged into a
> VRAM ring by DMA plus swaps held during prefill: 34 of 34 gate configs
> pass, all staged paths bit-identical; prefill 17.64/17.73 -> 16.98/17.00
> ms/token (A,B,B,A), decode unchanged. Merged to franken-engine main
> (`5f409c2`, knobs default off). The profile moved the wait from missed
> experts to idle time: at T=256 a chunk uses nearly every expert, so it
> fetches all non-resident experts (92.7 GB a chunk) whatever the routing;
> per-token traffic falls as 1/chunk. Chunk 512 measured: -10 % (16.8 vs 18.6 ms), bit-identical; the adaptation read-back (2-4.4 ms a token, queued behind the staging DMA) and the trunk are now the limit.
> Then GLM behind the gateway and its quality run.**
>
> **Rev 76 (2026-09-25 07:00 CEST; DeepSeek browser acceptance PASS on an idle engine, first token 1.3 s on screen -- DeepSeek done for L5) -- GLM STEP 5 GATED: BIT-IDENTICAL, PREFILL
> MISS-BOUND (record §L5-GLM-STEP5).** One GPU load ran all 23 gate
> configurations: batched prefill, graph replay, three-link and adaptation
> are bit-identical to the eager reference (the one "failure" was the gate
> script checking a dump-only config for taps; fixed in franken-engine
> `a08fff1`). Prefill is 17.5-18.1 ms a token against DeepSeek's 5.6: dev0
> waits 7.7 ms a token on missed experts and the other cards wait on dev0,
> while adaptation swaps GBs a chunk over the same links. Decode timing moved
> inside each A,B,B,A (placement still learning), so no graph verdict yet.
> Next, in order: (1) prefill with swaps held and each chunk's distinct
> misses fetched once over all three links; (2) decode timing on a settled
> placement; (3) GLM behind the gateway, accept_live/accept_ui, quality run
> against llama.cpp's GLM.**
>
> **Rev 75 (2026-09-25 04:30 CEST) -- THE DEEPSEEK LOOPS ARE NEAR-TIES, NOT A BUG
> (record §L5-DS4-SERVE).** Teacher-forced along llama.cpp's greedy sequence,
> Franken differs at 2.3 % of steps, all near-ties in both engines
> (activation rounding, as on Qwen3.8). Greedy loops are chance; serving
> samples normally. DeepSeek is done for L5 apart from browser acceptance on
> an idle engine. Next: GLM batched prefill + graph + profiler fix + load-once
> gates, then GLM behind the gateway and its quality run; the engine's
> working copy moves to frankenstack/franken-engine.**
>
> **Rev 74 (2026-09-25) -- DEEPSEEK QUALITY: NOT DISTINGUISHABLE, BUT A DEFECT
> FOUND (record §L5-DS4-SERVE).** 75.7 % vs llama.cpp's 82.9 %, p = 0.18;
> 6 of llama.cpp's 7 wins are Franken answers running to the 16k budget
> without stopping (11 truncations vs 4; equal lengths otherwise). First
> DeepSeek item: reproduce mmlu:570 and read how it fails to stop.
> Franken engine repo created: frankenstack/franken-engine (107 commits,
> pushed by claude-bot).**
>
> **Rev 73 (2026-09-24 20:10 CEST) -- GLM-5.3 ON THE GPUS (record §L5-GLM-GPU).**
> Bit-identical to its CPU arm on every path (three-link, adaptive, both);
> 13.6-14.9 tok/s at a 256k window against Colibri's ~5; three-link split
> ~37 %; no depth cost. Next for GLM: batched prefill, graph replay, the
> profiler fix, load-once gates, then serving behind the gateway + quality.
> Process: the rig sat idle 5 h (a refused step, no watcher) -- rules added
> (CLAUDE.md: watchers for every end state, preflight, retry on rc 3).
> Franken engine extracted into its own repository (107 commits of history),
> push pending the Gitea bot account.**
>
> **Rev 72 (2026-09-24 08:15 CEST) -- DEEPSEEK SERVED, GLM ON THE CPU ARM
> (record §L5-DS4-SERVE, §L5-GLM-STEP1).** DeepSeek on the Franken engine
> behind the gateway: accept_live PASS, new chat 0.9 s, cold 4.6k block 21 s;
> quality run in progress. GLM-5.3 layers 0-5 match llama.cpp on the CPU arm
> incl. the real top-512 selection; KV 3.3 GB at 256k. Next: GLM GPU kernels
> (KDA, absorbed MLA, 8-of-288 experts, three-link stream). Two environment
> items on the Mac need the owner: keychain (Gitea push) and Local Network
> permission (accept_ui's browser).**
>
> **Rev 71 (2026-09-24 06:15 CEST) -- DEEPSEEK: NO DEPTH COST, BATCHED PREFILL
> (record §L5-DS4-STEP5).** Decode 47.2 / 47.7 / 48.1 ms at depth 64 / 8k /
> 33k = 21 tok/s; rev 70's "depth cost" was the wrong placement's misses
> (104 ms at every depth) -- corrected. Prefill 5.6 ms/token (4.0 with the LDS
> GEMM), every chunk gate bit-identical incl. the first real top-512 on the
> GPU. Next: DeepSeek behind the gateway + quality (as L0 step 4 for Qwen);
> then GLM-5.3 on the streamed path (L5 GLM).**
>
> **Rev 70 (2026-09-24 03:45 CEST) -- ADAPTIVE PLACEMENT WORKS (record
> §L5-DS4-ADAPT).** Bit-identical with swaps; over 8.4k prose tokens from a
> wrong placement: misses 474 -> 101 MB/token, hit 93-96 %. New finding:
> DeepSeek's decode grows from 50 ms (depth 53) to 89-102 ms (depth 8.5k) --
> its attention/indexer depth cost, to profile next with the trunk GEMV on
> short rows; DeepSeek batched prefill still missing.**
>
> **Rev 69 (2026-09-24 02:45 CEST) -- DEEPSEEK 19.8 TOK/S AT 256K (record
> §L5-DS4-STEP3b).** Bit-identical throughout; placement learned on mixed
> text cut misses 742 -> 276 MB a token; graph replay: 50.4 ms a token. The
> hot-expert set is domain-dependent (measured), so the next item is
> ADAPTIVE placement (online counts, background swaps: design L2), then the
> trunk GEMV on short rows (both models). Profiler artifacts of step 3
> corrected in the record.**
>
> **Rev 68 (2026-09-24) -- DEEPSEEK-V4 ON THE GPUS (record §L5-DS4-STEP2).**
> Correct (220 taps >= 0.9999, greedy identical), 11.0 tok/s at a 256k
> allocation vs llama.cpp's 8; ~6-9x off its bytes -- profile and fix next
> (expert GEMV loads, miss path off the critical path, graph replay).**
>
> **Rev 67 (2026-09-24) -- L5 STEP 1: DEEPSEEK-V4 ON THE ENGINE'S CPU ARM
> (record §L5-DS4-STEP1).** Graph model-dispatched (Qwen3.8 bit-identical);
> deepseek4 layers 0-3 match llama.cpp (173/173 taps, cos >= 0.99956,
> expert selections exact on 6 tokens); six new bit-exact weight decoders;
> KV 1.8 GB at 256k. Next: DeepSeek GPU kernels + miss path, then GLM on the
> streamed path.**
>
> **Rev 66 (2026-09-23 23:40 CEST) -- M4 MEASURED (record §M4). The whole
> measurement program M0-M5 is done.** Three cards streaming at once: 52 GB/s
> (28 a card alone, 36 for the shared-link pair together); PCIe-aware
> assignment gives the lone card ~44 % of the stream. GLM-5.3 on the
> streamed path: ~19 ms a token of stream at ~1 GB missed -> ~25-30 tok/s
> realistic. In build: L5 step 1 (DeepSeek-V4 on the engine, CPU oracle
> first). Next: DeepSeek on the GPU; then GLM on the streamed path
> (L0b/L1/L2).**
>
> **Rev 65 (2026-09-23 23:15 CEST) -- M2 MEASURED (record §M2, design rev 13).**
> DeepSeek-V4-Flash misses 60 MB a token with 60 GB of experts resident: a
> near-resident model, next after Qwen3.8 on the engine. GLM-5.3 misses
> ~1.0-1.4 GB a token: streaming-bound, a ~30-45 tok/s ceiling at the
> 61 GB/s aggregate against 5 today. Placement by histogram is the lever;
> temporal lookahead is weak. The prefill "regression" of rev 64 was the PLE
> page cache in the benchmark path, not code (A,B,B,A). Next: M4 (three
> cards streaming at once), then L5 DeepSeek (resident route), then GLM on
> the streamed path (L0b/L1/L2).**
>
> **Rev 64 (2026-09-23 -- DECODE >= 40 TOK/S AT 256K (record §L0-PERF-2).**
> HIP-graph replay of each card's token, bit-identical (greedy exact): decode
> 24.4 ms = 40.9 tok/s at 262k depth (25.8 before), 21.6-22.2 ms short. The
> design's decode target is met. Open: a prefill slowdown on the same binary
> (430 vs 354 s for 262k) to bisect before serving the graph; card 0's slow
> small GEMVs are not host issue. Also open, the rest of the ladder: M2
> (histograms, tool built, DeepSeek/GLM runs queued), M4, L0b/L1, L2-L4,
> L5 (DeepSeek and GLM on the engine).**
>
> **Rev 63 (2026-09-23 12:15 CEST, 10:15 UTC) -- FIRST PERF PASS (record
> §L0-PERF-1).** Prefill 1.63 -> 1.40 ms/token; cold 262k prompt 467 -> 354 s
> (5.9 min; llama.cpp 17; target ~5). Down gather bit-identical and on (the
> divergence was fp-contract chosen per kernel; ISA-proven), QSA rows
> batched (0.19 -> 0.04 ms). Decode: four ISA-exact GEMV changes cut
> launches 1 198 -> 1 004 and won nothing (23.0 / 25.8 ms): the small
> matrices are floor-bound at ~7 µs a launch. Next for decode: measure a HIP
> graph of the real token body (M5 said 7 % on empty kernels; dependent
> real kernels may differ), and card 0's slower small-GEMV set. Reply cap
> for the Franken gateway raised to 16 384.**
>
> **Rev 62 (2026-09-23 10:00 CEST, 08:00 UTC) -- L0 QUALITY MEASURED, GATE MET
> (record §L0-QUALITY).** Franken engine as served vs llama.cpp's Qwen3.8,
> same 70 MMLU-Pro items, reasoning xhigh, 16k budget: 77.1 % vs 80.0 %,
> McNemar p = 0.69, NOT DISTINGUISHABLE (the Franken arm carried the
> gateway's 4 096-token reply cap: 10 truncations vs 7); needles 8/8 at
> 30k-200k on both, the engine 1.2-1.9x faster. FRANKEN_GEMM_LDS=1 stays the
> serving default. **L0 is complete: design §9.5's four steps and both
> gates (>= 40 tok/s at 256k: 38.4, 4 % short with named owners; quality
> within llama.cpp's spread: yes).** GLM restored, accept_live PASS. Next:
> the named performance items (decode's small-matrix GEMVs and launch
> floor; prefill's experts, per-row QSA, the down gather) and the reply
> cap; then L0b/L5 (DeepSeek, GLM streamed: M2, M4).**
>
> **Rev 61 (2026-09-23 03:20 CEST, 01:20 UTC) -- L0 STEP 4 GATE MET: accept_live
> PASS, accept_ui PASS on the Franken engine (record §L0-STEP4).** Browser
> first token 0.80 s, warm new chat reused 4 839/4 883 at 0.45 s, follow-up
> 0.4 s; check 3's tolerance made template-aware (Qwen's markers cost 38
> tokens a turn; the engine reused every token it held -- my renderer
> diagnosis was wrong and the agent disproved it with the bytes). GLM
> restored. What remains of L0: the quality number at the served settings
> (decides FRANKEN_GEMM_LDS); then the perf items and the streamed models.**
>
> **Rev 60 (2026-09-23 03:05 CEST, 01:05 UTC) -- THIRD SERVE: FIRST TOKEN ON
> SCREEN 0.8 s (record §L0-STEP4).** Async turn-boundary checkpoints,
> resident PLE table, LDS GEMM served: Open WebUI new chat first token 0.81 s
> in a real browser (GLM 3.6-3.8), warm new chat reused 4 839 of 4 883 tokens
> at 0.45 s ttft, cold 4.9k prompt 7.9 s; 4 of 5 accept_live checks pass, the
> fifth (follow-up reuse 161 vs >= 171) is the gateway's qwen38 renderer
> dropping the empty think block on re-render -- fix with the build. GLM
> restored. Next: renderer fix -> accept_live/accept_ui PASS -> quality at
> served settings -> decide the serving default -> perf items -> streamed
> models.**
>
> **Rev 59 (2026-09-23 01:45 CEST, 23:45 UTC 09-22) -- STEP 4 FIRST SERVE (record
> §L0-STEP4).** The Franken engine served real chats behind the owner's
> gateway and Open WebUI: first token on screen 6.8 s on the 4.9k tool
> block, 22-27 tok/s; 4 of 5 accept_live checks pass. Prefix reuse is the
> open item (reused=0 across chats, empty on short follow-ups, and a reported
> reuse still re-prefilled everything) plus launcher env (ledger) and
> thinking-off for the UI -- all with the build. The engine runs inside the
> ROCm 7.14 image (host ROCm 6.2 lacks libllama's libhipblas). GLM restored.
> Second serve (00:05 UTC): the reuse fixes landed but per-chunk
> checkpoints serialised the pipeline (prefill 42 s for 4.9k) and cold PLE
> rows cost 3.8 s -- both known design items (async checkpoints, pinned PLE
> table §9.1), with the build. Next session: run them, then
> accept_live/accept_ui PASS, quality.**
>
> **Rev 58 (2026-09-22 22:05 CEST, 20:05 UTC) -- THE 256K NUMBER (record
> §L0-256K, design rev 11).** Franken engine, Qwen3.8-Flash-Next, three cards,
> 262 144 window: cold 262 000-token prompt 467 s (7.8 min, 1.78 ms/token);
> **decode at 256k depth 26.1 ms/token = 38.4 tok/s** (llama.cpp on this
> rig: 17 min and 6 tok/s; design target >= 40 and ~5 min). Depth 6 -> 262k
> costs 2.8 ms/token, as §M3 predicted. Remaining for L0: the gateway
> (step 4), accept_live/accept_ui, the quality number at the served
> settings; then the named decode/prefill optimisations; then DeepSeek/GLM
> (streamed, L0b/L5).**
>
> **Rev 57 (2026-09-22 21:50 CEST, 19:50 UTC) -- PREFILL 7.7 -> 3.0 -> 1.63 ms
> A TOKEN (record §L0-PREFILL-2).** Chunk pipelining across the cards (94-100 %
> busy, bit-identical, default on) and an LDS-tiled trunk GEMM behind
> --gemm-lds 1 (last-bits order change, measured). 8 192 tokens in 13.4 s;
> prefill is at llama.cpp's rate, 1.6x the <= 1 ms target. RDNA3's int8 dot
> IS reachable (`__builtin_amdgcn_sudot4` -> v_dot4_i32_iu8) but that GEMM
> variant has a numerics bug and no speed gain; off. Running now: the
> 262 000-token prefill and decode at that depth -- the design's headline.**
>
> **Rev 56 (2026-09-22 21:00 CEST, 19:00 UTC) -- PREFILL BISECTED AND PROFILED
> (record §L0-PREFILL).** The chunked run's divergence is entirely the
> expert row-gather's summation order (with it off: bit-identical on the
> GPU); decode restored to 23.3 ms with a TILE=1 GEMM specialisation. Prefill
> 7.6 ms/token = three cards in sequence (2.5 ms each) + a first-cut GEMM
> 6x off its bytes + batched experts far off theirs. Next, in order:
> pipeline chunks across the cards (~3x, legitimate for prefill), an
> LDS-tiled int8 GEMM (dot4 if the toolchain exposes it on gfx1100), the
> gather's order and efficiency. Then the depth ladder and step 4.**
>
> **Rev 55 (2026-09-22 20:30 CEST, 18:30 UTC) -- BATCHED PREFILL EXISTS (record
> §L0-PREFILL).** T-wide graph, GDN over T, causal QSA over T (proof in
> decode/PREFILL.md), expert row-gather batching. CPU arm: chunked ==
> token-by-token bit for bit (1 592 taps, uneven splits too). GPU: identical
> except QSA layers 23-39's attention outputs (cos 0.999995+) -- being made
> bit-exact. First rate: 8 192 tokens in 61.3 s = 7.48 ms/token (llama.cpp
> 0.8-2.0; target <= 1) -- unprofiled first cut. The unified GEMM kernel
> regressed decode 22.7 -> 27.0 ms; a T=1 specialisation restores it. Also
> fixed today: hipFree per token in the top-k (8k depth 23.2 ms/token, +0.5
> over depth 6). Next: the two fixes, prefill profile, then the depth ladder
> to 256k and step 4 (gateway).**
>
> **Rev 54 (2026-09-22 19:10 CEST, 17:10 UTC) -- L0 STEP 3: THE WHOLE MODEL ON
> THREE CARDS (record §L0-STEP3, design rev 10).** Qwen3.8-Flash-Next, 256k
> allocated (23.1/22.5/23.4 GB used, 2.4-3.3 GB free per card), P2P
> boundaries, Q6_K head, device argmax. Greedy: 12 of 16 tokens identical to
> llama.cpp's CPU reference ("It sounds like you might be having a
> frustrating moment! 😤 ..."), routing overlap 0.92. **22.9 ms a token =
> 43.7 tok/s at short context** -- llama.cpp is 34-40 ms there and 66 ms
> with a 256k window. Four placement/addressing bugs found and fixed from
> the failure text (device-wide current device, indexer cache 8x, Q6_K row
> stride in elements, events on the wrong device). Not measured yet: the
> depth ladder to 256k (next), quality; known cost: the first card is
> CPU-issue-bound (agent removing per-launch host cost now).**
>
> **Rev 53 (2026-09-22 18:30 CEST, 16:30 UTC) -- STEP 2b: 11.7 -> 7.6 ms for 16
> layers on one card (record §L0-STEP2, commit c0633eb).** Batched GEMVs
> over shared activations, epilogue fusions, no top-k when the context fits
> the budget: 931 -> 389 launches, GPU still = CPU graph (534/534). Big
> projections at 434-497 GB/s (55-60 % of the bound); the remaining trunk
> gap is the small matrices (hc, shared expert, router: 128-245 GB/s,
> latency-bound) and the ~2 ms launch floor of 389 launches. Extrapolated:
> ~23 ms a token over three cards (43 tok/s) at short context vs llama.cpp's
> 34-40 -- already past the design's >= 40 target, projected. Next: split-K
> for every matrix under ~10 MB, further epilogue fusion; then step 3
> (three cards + lm_head, greedy text oracle).**
>
> **Rev 52 (2026-09-22 17:50 CEST, 15:50 UTC) -- L0 STEP 2 RUNS (record
> §L0-STEP2, design rev 9).** The one-card decode loop over layers 0-15
> exists (`tools/hot-expert/franken/decode/`, 35 kernels, one graph with a
> CPU and a GPU backend); the GPU equals the CPU graph (534/534 taps cos
> 1.0); against llama.cpp's 1 256-tap CPU dump it matches to the floor
> through layer 6 and diverges beyond by router near-ties that llama.cpp's
> own activation quantisation flips (measured with a --quant-act arm and a
> jitter arm). The oracle bar for L0 is therefore the design's: routing set
> per layer + greedy text on real prompts. Timing: 11.7 ms for 16 layers,
> 0 host syncs, 931 launches -- launch-bound small kernels (~4 ms) and a
> trunk GEMV at 2.7x its bytes; experts at §M1's number. Two harness facts
> corrected by the port: the indexer has ONE key head (M3's depth term was
> 4x too high) and M3's GDN step was not the model's op. Step 2b
> commissioned: fusion to <= 15 launches a layer, trunk GEMV at the bytes.**
>
> **Rev 51 (2026-09-22 14:45 CEST, 12:45 UTC) -- L0 STEP 1 (record §L0-STEP1,
> §M1 candidate 3; design rev 8).** The loader exists
> (`tools/hot-expert/franken/`): 64.9 GB of Qwen3.8 on three cards by layer
> range in 12.8 s, 22.2 / 20.9 / 21.8 GB per card, the 28.8 GB n-gram table
> in host RAM (rule: record §PLE-GATHER, CLAUDE.md), PLE hash ported and
> cross-checked. The expert format is decided without requantisation: a
> native IQ3_S/IQ4_NL GEMV in hipFire's shape runs 60.5 µs a 10-expert step
> (385 GB/s, cos 1.0) = hipFire's own step time with 19 % fewer bytes. Next:
> step 2 -- one card, layers 0-15, decode loop with GDN + QSA + MoE, oracle
> per layer against llama.cpp's dumped activations.**
>
> **Rev 50 (2026-09-22 12:30 CEST, 10:30 UTC) -- M3 CLOSED, GO/NO-GO PASSED
> (record §M3 v2, design rev 7).** Opus-tier kernels: wave-per-block indexer
> scan (775 GB/s streamed from VRAM at 256k), single-launch radix-select
> top-k (28 µs), flash-style sparse attention (38 µs). A QSA layer is 60 /
> 92 / 142 µs at 32k / 128k / 256k with 12 rotating cache copies defeating
> the 96 MB Infinity Cache (the single-copy run read 2.3 TB/s and was not
> recorded as the result); 1.7 ms a token for 12 layers + 0.57 ms GDN =
> 2.3 ms of attention at 256k against a 10-12 ms go/no-go. The first cut's
> 2 ms top-k was a host bisection with a memcpy per round. Every term of the
> resident token is now measured on this hardware: ~10-12 ms a token, 80-100
> tok/s at 256k projected (design §6). M0, M1, M3, M5 done; **L0 may start**
> (design §7: Qwen3.8 resident on three cards, layer-range pipeline,
> hipFire-class trunk and expert kernels with K_TOP=10 and a group-128/64
> format, M3's attention kernels, Colibri's gateway). M2/M4 are for the
> streamed models.**
>
> **Rev 49 (2026-09-22 12:00 CEST, 10:00 UTC) -- M1 MEASURED, M3 FIRST CUT
> (record §M1, §M3; design rev 6).** M1 on Qwen3.8's real expert shapes:
> hipFire's MoE GEMV kernels 452 GB/s at batch 1 and 826 (= the VRAM bound)
> at batch 32; llama.cpp's served mul_mat_id path 174 / 271. Design §2 row 3
> decided for hipFire's kernel shape, with two ports (runtime K_TOP=10, a
> group-128/64 format for the expert down's K=640). The first ggml run's
> oracle was broken (allocator reused the inputs of a reused graph); fixed,
> the oracle now refuses zero outputs, the clobbered timing is discarded.
> M3: the GGUF DOES carry compress_ratios=4 (a build agent said not; verified
> from the header), so llama.cpp's depth numbers are QSA numbers. With plain
> first-cut kernels a QSA layer is 2.6-3.2 ms (top-k 2.05 ms, sparse
> attention 0.48 ms, indexer scan 0.03-0.28 ms -- the only depth-dependent
> part), GDN 0.67 ms a token for 36 layers. The bytes say ~0.15 ms a layer.
> M3 is NOT closed and the go/no-go is not decided: two kernels (radix
> top-k, coalesced sparse attention) are commissioned at the Opus tier and
> M3 re-runs on them.**
>
> **Rev 48 (2026-09-22 10:45 CEST, 08:45 UTC) -- M5 MEASURED (record §M5, design
> rev 5): the launch/sync floor on gfx1100 under HIP is 3 µs async, 24 µs
> blocking, 31 µs cross-stream, ~30 µs per P2P boundary; a synthetic 48-layer
> token of 240 plain launches on one card runs at 794 GB/s, the VRAM bandwidth
> bound (800 measured) -- hipGraph saves 7 %. §1(b)/(c) of the design are
> confirmed on this hardware. Projection, labelled: Qwen3.8 fully resident is
> ~125 tok/s bandwidth-bound before attention, against llama.cpp's 24-29;
> llama.cpp's gap is its 22 host-mediated graph splits, CPU-side ops and (to
> be measured, M1) its expert kernels at N=640, not the launch floor. M1
> harness (llama.cpp's mul_mat_id path and hipFire's kernels on the real
> shapes) in build; M3 (KV bytes: 1 088 + 408 MiB per card at 256k already
> read from M0d's log = ~18 KB a token; attention time per layer) next.**
>
> **Rev 47 (2026-09-22 09:30 CEST, 07:30 UTC) -- M0 MEASURED (record §F11-M0,
> design rev 3).** Qwen3.8-Flash-Next UD-IQ4_XS under llama.cpp fit, a 32k
> against a 256k window: `llama-fit-params` places the SAME tensors at both
> (61.2 GiB on the cards, 27.5 GiB = 30 % on the host), so the arms differ
> only in the reservation. Decode 24-29 tok/s at 1.5-19.5k with the 32k
> window against 14-17 with the 256k one -- **the 256k reservation alone costs
> llama.cpp 1.6-1.9x decode and 2.1-2.4x prefill at every depth**, before the
> separate depth slope of §F11-DEPTH (15 -> 6 tok/s). The design's reference
> is therefore ~25 tok/s at 32k / 15 at 256k on this engine, and the new
> engine's attention must cost only the used cells (design §2 row 2, §3.4).
> **Corrected within the hour (design rev 4, record §F11-M0 point 2):** the
> GGUF header shows the 30 % on the host is `per_layer_token_embd` (28.8 GB,
> a CPU gather table), NOT experts -- every routed expert (IQ3_S gate/up,
> IQ4_NL down; the file's "IQ4_XS" names one layer) is resident on the
> cards. The "miss-cost calibration" first written into this rev is
> withdrawn; llama.cpp's 34-40 ms a token is the FULLY RESIDENT case, 4-5x
> off the ~7.5 ms bandwidth bound for ~6.2 GB of VRAM reads a token. M0b
> done: a step (24 tok/s at 32k-128k, 15 at 256k); M0c/M0d: not eviction
> -- llama.cpp's fit moves five layers of expert gate/up (3.2 GB) to the
> CPU at 256k, ~0.5 ms per CPU expert evaluation; a lean engine keeps the
> model resident at 256k. For
> Qwen3.8 the engine is a resident engine (L0 on three cards, no stream);
> streaming stays for DeepSeek/GLM (L0b, L5). GLM back in service, `accept_live` PASS 01:27 UTC after the
> chain exited. Quality run 2 (`f11_quality2_chain.sh`) is not resumed; its
> partial results stay under `~/bench/f11_quality/`. Next: M1 and M3 (read
> the hot loops first), per design §4.
>
> **Rev 46 (2026-09-22 02:40 CEST, 00:40 UTC) -- THE DESIGN EXISTS:
> `FRANKEN-ENGINE-DESIGN-2026-09-22.md` rev 1.** The owner's addendum: this
> is an architect's analysis and design, and inventing what the other
> engines cannot do is in scope. The design: a placement-and-streaming
> engine, layer-range pipeline over three cards (no RCCL, no per-op host
> round trips, KV on the cards), hipFire's HIP kernels for the resident
> trunk and experts (to be measured head to head against llama.cpp's and
> Colibri's, M1), llama.cpp's per-architecture graphs as the reference for
> what each model computes, Colibri's ring/streaming, histogram placement,
> batched prefill and the whole gateway (checkpoints, ledger). Invented:
> router-lookahead prefetch, hottest-expert replication per card,
> PCIe-link-aware layer-range assignment, 256k checkpoints. Measurement
> program M0-M5, each closing a design choice; M0 in flight. Projected
> targets, labelled: >= 40 tok/s and a ~5 min cold prompt at 256k.
>
> **Rev 45 (2026-09-22 01:30 CEST, 23:30 UTC 09-21) -- THE OWNER'S DEFINITION
> OF THE FRANKEN-ENGINE, VERBATIM IN SUBSTANCE, BINDING ON EVERY REV BELOW
> AND ABOVE:** the engine is a NEW engine, created from the parts of the
> other engines (Colibri, llama.cpp, hipFire, hipEngine, ...) that make it
> run optimally on THIS AMD hardware -- three RX 7900 XTX, 247 GiB RAM, one
> 8-core EPYC -- for the models on his list (Qwen3.8, DeepSeek V4.x, GLM-5.3
> and successors) at a 256k window. Serving an existing engine, or one
> engine per model, is NOT the goal; llama.cpp is not built for this rig
> (his words), which is why hipFire and hipEngine were investigated. Revs
> 38-44's framing ("which engine serves which model") was the orchestrator's
> error, twice, and is withdrawn. What stands from them is data: llama.cpp's
> HIP path on Qwen3.8-Flash-Next IQ4_XS gives 15 tok/s at 19k and 6 at 257k
> with ~a quarter of the experts in host RAM; hipFire's HIP kernels give
> 80-130 tok/s on one card for a resident 35B; Colibri's Vulkan tier gives
> 4-5 tok/s on the same class with 60-70 % of the model in RAM but owns the
> placement, checkpoint and ledger logic Open WebUI needs. **Method from
> here: per component of a decode/prefill token -- expert kernel + quant
> format, attention + KV at 256k, expert placement and host->device
> streaming, multi-card dispatch, prefill batching -- measure which existing
> implementation is fastest on gfx1100, then assemble.** The queued fit runs
> (32k vs 256k window) are the first such component number: the cost of
> RAM-resident experts under llama.cpp's HIP MoE path. The quality run of
> rev 43/44 is stopped; quant choice is a workload parameter, not the work.
>
> **Rev 44 (2026-09-22 00:40 CEST, 22:40 UTC 09-21) -- THE FIRST QUALITY
> RUN IS NOT A MODEL COMPARISON; THE OWNER SAID SO AND HE IS RIGHT.** It
> compared GLM on Colibri with reasoning effectively off (median 80 answer
> tokens) against Qwen3.8 and DeepSeek on llama.cpp at their templates'
> default reasoning (382-488), under a 3 500-token budget that cut off 14 and
> 11 of their 70 answers, on a sample that cannot resolve 10 points. Raw
> scores 80.0 / 75.7 / 72.9 %, McNemar p 0.23-0.77, needles 8/8 at every
> depth for all three (the needle test discriminates nothing here). Kept as
> a negative result; the rev-43 decision rule is NOT applied to it. Second
> run, `f11_quality2_chain.sh`: all three on llama.cpp (`serve_alt.sh
> glm-llama`), each at its template default, 210 questions, 16 000-token
> budget, no needles; ends with the Colibri gateway serving. A retry of the
> first run's cut-off questions at 16k showed DeepSeek IQ2_M spending ~30 min
> per hard question at 8 tok/s -- a usability fact on its own.
>
> **Rev 43 (2026-09-21 16:10 CEST, 14:10 UTC) -- THE OWNER WILL NOT JUDGE
> QUALITY BY HAND AND SHOULD NOT HAVE BEEN ASKED TO: IT IS MEASURED.** He is
> at work; "F11 waits on the owner's verdict" (rev 42) handed him the job.
> Replaced by `quality_eval.py` + `f11_quality_chain.sh`: the same frozen 70
> MMLU-Pro questions (5 per category, sha256 `0e697795...`, zero-shot CoT,
> temperature 0, 3 500-token budget) and an 8-needle retrieval test at
> 30k/60k (all) and 120k/200k (the two llama-server models) on Qwen3.8
> IQ4_XS, DeepSeek-V4-Flash IQ2_M and the served GLM-5.3 int4; paired McNemar,
> Wilson intervals, and an explicit NOT DISTINGUISHABLE verdict -- 70 items
> cannot resolve less than ~10 points, so the run can show a quantization is
> BROKEN or clearly worse, not rank near-equals. Pilot on Qwen3.8: first 6
> items correct, 185-508 tokens, no truncation. Running since 14:0x UTC,
> estimate ~6 h; the chain ends with GLM serving and `accept_live` passed.
> Tool calling on Qwen3.8 is verified at the API and through Open WebUI
> (record §F11-TOOLS). **The decision rule, fixed before the numbers are in:**
> a model whose MMLU-Pro score is not distinguishably below GLM-5.3's and
> whose needles are >= 7/8 at every depth it supports replaces GLM as the
> served default, fastest first; otherwise GLM stays and the finding is
> recorded.
>
> **Rev 42 (2026-09-21 14:15 CEST, 12:15 UTC) -- THE SWAP EXISTS AND
> QWEN3.8-FLASH-NEXT IS IN SERVICE FOR THE OWNER'S JUDGEMENT (record
> §F11-SERVE).** `tools/hot-expert/serve_alt.sh qwen38|deepseek|glm|status`.
> Both directions ran: in the browser a new chat's first token 9.6 s cold and
> 0.8 s warm on Qwen3.8; back to GLM with `accept_live` PASS. **Open WebUI
> still shows the model as `glm-5.3-flash` while Qwen answers.** To go back:
> `ssh rome '~/src/colibri/tools/hot-expert/serve_alt.sh glm'`. **F11 now
> waits on the owner's verdict on the answers.** Untested: a tool-calling turn
> through Open WebUI on Qwen3.8; DeepSeek through the swap; two chats at once.
>
> **Rev 41 (2026-09-21 13:50 CEST, 11:50 UTC) -- THE 256k RUNG IS MEASURED
> (record §F11-DEPTH): Qwen3.8-Flash-Next UD-IQ4_XS, llama.cpp HIP, fit, one
> slot, reached 257 018 tokens in both arms.** Decode 14.6-15.3 tok/s to 39k,
> 11.9-12.5 at 84k, 8.4-8.8 at 168k, **6.1-6.5 at 257k**; a cold 257k prompt
> 16.8-17.4 min; a follow-up turn at 257k 1.5 s. The owner's context target is
> met by a model on his list, on hardware he owns, with software already on the
> disk -- at a quantization whose answers nobody has judged yet. **Next: (1)
> the one-command swap so he can judge Qwen3.8 IQ4_XS in Open WebUI (Sonnet);
> (2) only if he keeps it: what the Colibri gateway has that llama-server
> lacks for his daily use -- prefix checkpoints that survive a restart, the
> conversation ledger, the tool-block capture -- measured against
> llama-server's own prompt cache and slot save/restore BEFORE porting
> anything; (3) decode at depth is the number to attack (15 -> 6 tok/s from
> 39k to 257k): profile first.** Where the orchestrator's arithmetic failed
> today (the tokenizer ratio, twice) is in the record section; both times the
> chain's own refusal or the first rung caught it.
>
> **Rev 40 (2026-09-21 12:15 CEST, 10:15 UTC) -- F11 STEP 0 MEASURED (record
> §F11-STEP0): with a 256k window allocated and the cards filled by
> llama.cpp's own fit, Qwen3.8-Flash-Next UD-IQ4_XS decodes 14.4-15.2 tok/s at
> 19k and answers a cold 19k prompt in 35 s; DeepSeek-V4-Flash UD-IQ2_M 8.0
> tok/s and 97 s. Neither slows between 1k and 19k.** Served GLM-5.3: 5.04
> tok/s and 356 s at 18k, 65k window. Colibri's own `qwen38-vk` on the FP8
> Qwen3.8-Flash-Next: 4.04-4.09 tok/s. Gateway restored, `accept_live` PASS
> twice, the second after the chain had fully exited. **What it took:** the
> owner said the rig is a development machine by day, so it ran at once; a
> first run was stopped after one load because `--n-cpu-moe` with an even
> layer split left a card at 5.8 GB of 24 -- the engine's own fit fills all
> three and is what F11 uses from here. **What this does NOT say:** whether
> either quantization answers well enough (the owner's judgement), and speed
> past 21k (nothing on this box has been measured there). **Next, in this
> order:** (1) a swap script, not written yet: one command that swaps the gateway's model for
> Q or D under llama-server on the port Open WebUI already uses, and back, so
> the owner can judge answers -- Sonnet; (2) the depth ladder to 64k / 128k /
> 256k on Q first: at the measured ~0.9 ms per prefill token a cold 256k
> prompt is minutes, not hours, IF the rate holds, which is exactly what the
> rung measures. Nothing is built in Colibri before both are in.
>
> **Rev 39 (2026-09-21 12:00 CEST, 10:00 UTC) -- THE OWNER'S CONTEXT TARGET:
> AT LEAST 256k TOKENS, "the size where models become usable for real work".
> Nothing run.** This is a requirement on every candidate from here on, and it
> is far from what is served: `~/start_glm53.sh` sets `GLM53_MAXT=65536` per
> slot (4 slots), and no ladder on this box has gone past 18.4k. What it
> changes now: **F11 step 0 searches and measures its expert placement with a
> 262144-token KV cache allocated** (`f11_ladder_chain.sh`, `F11_CTX`), not
> 32768 -- the VRAM a 256k window takes comes out of the experts that can be
> resident, so a 32k placement would have measured a configuration the owner
> will not use. The chain prints llama.cpp's KV/compute buffer sizes at load;
> the only figure in hand is an upper bound that prices every layer as dense
> attention (~13 GB for Qwen3.8-Flash-Next, ~11.5 GB for DeepSeek-V4-Flash at
> q8_0), and both models are sparse/compressed-attention, so the real number
> is unknown until tonight. **What step 0 does NOT answer:** speed AT depth.
> The ladder stops at ~18k; prefill time and decode tok/s at 64k/128k/256k
> are a second measurement (hours per rung at GLM-5.3's current prefill
> rate), to be specified from tonight's rates, not projected from them.
>
> **Rev 38 (2026-09-21 10:30 CEST, 08:30 UTC) -- THE OWNER CLOSED F3 AND SET
> THE TRACK'S TERMS. Nothing built, no engine run.** His words, condensed:
> (1) **Qwen3.6 is too far behind; step 2b is not started and F3 is CLOSED.**
> Only the recent Chinese models are acceptable as the box's daily model --
> Qwen3.8, DeepSeek V4.x, GLM-5.3 and their successors. (2) **He does not care
> which inference engine runs.** "Franken-engine" means: borrow whatever works
> from the other engines and combine it into the application that fits this
> rig best. Colibri is one ingredient, not the goal. (3) **Plain 4-bit is
> generally too low quality, but the "smart" quantizations (Unsloth UD-*,
> IQ-*, mixed per-tensor) stay in consideration** for their speed/quality
> balance. What stays from F3: the finding that int4-gs64 costs KL 0.0316 on
> Qwen3.6, the int8 tier path in `qwen36_tier_vk.c`, and the measured proof
> that piecemeal trunk offload is worthless here. The `~/models/qwen36_*`
> containers (57 GB) are deletable.
> **What follows from (1)-(3), and the regime map of 2026-09-16:** on this box
> speed is decided by how much of the model is VRAM-resident (gpt-oss-120b:
> 75 tok/s resident, 28 half in RAM, 17.6 all in RAM at 18k) and both GLM-5.3
> deployments (Colibri int4-g64 184 GB, llama.cpp UD-IQ4_XS 149 GB) are ~3-5
> tok/s at 18k because 60-70 % of the model is in RAM. **Two files of
> acceptable models that would be ~70-80 % resident have been on the rig's
> disk since 2026-09-01 and were never measured:**
> `~/models/Qwen3.8-Flash-Next/UD-IQ4_XS` (93.7 GB, arch `qwen4exp`, 48
> blocks, 512 experts top-10) and
> `~/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M` (90.9 GB, arch
> `deepseek4`). **Corrected the same day:** the source of `~/src/llama.cpp`
> (`15586e2`) names both architectures, but its Vulkan BUILD (2026-08-07) is
> older than `qwen4exp` and its `libllama.so` does not contain the string;
> the HIP build `~/src/llama-glm53/build-hip` (the one the 09-16 GF arms ran,
> in docker) has both (`strings libllama.so.0.3.0`, no GPU touched). F11's
> chain, `f11_ladder_chain.sh`, therefore runs both models on HIP.
> **New item F11, step 0 = a MEASUREMENT, no projection is offered:** both
> files under llama.cpp on the 09-16 context ladder (turns 1-7, to 18k), as
> many layers' experts on the three cards as load, the rest `--n-cpu-moe`;
> arms interleaved Q,D,D,Q; one chain through `run_chain.sh` derived from
> `glm53flash_ladder_chain.sh` (whose exit trap restarts the gateway and never
> kills an engine it did not start -- read 2026-09-21). Sonnet writes the
> chain; the orchestrator reads its trap, launches and polls it, and sends a
> real chat after it exits. It yields decode tok/s and TTFT per depth for each
> model next to GLM-5.3's served numbers (5.04 tok/s and 356 s at 18k). Quality
> is NOT measured by it: whether IQ2_M DeepSeek or IQ4_XS Qwen3.8 answers well
> enough is the owner's judgement in Open WebUI, and the chain's second
> deliverable is a one-command swap (a script not written yet) so he can
> make it. Only after both: decide what, if anything, is built (hot-expert
> placement from Colibri's histogram into llama.cpp's tensor override is the
> obvious Franken candidate; it is a hypothesis until step 0 says residency is
> what limits these two files too).
>
> **Rev 37 (2026-09-20 17:55 CEST, 15:55 UTC) -- F3 STEP 2a: GATE NOT MET
> (+2.0 % against +20 %), and rev 36's own projection is refuted by it.**
> Record §F3-STEP2A: `Q36_VK_TRUNK=1` puts lm_head and the 30 fused DeltaNet
> projections on dev0; numerics at the floor (KL -1.4e-10, 625/625, knob off
> bit-identical to step 1); decode 21.50 -> 22.37 tok/s, +2.0 % conservative.
> lm_head 6.3 -> 2.6 ms, DeltaNet proj 10.8 -> 9.4 ms: a blocking GPU matmul
> costs ~0.3 ms per call here, which is what the CPU already took per layer.
> Rev 36 said "~27 tok/s by arithmetic"; the arithmetic assumed the call cost
> would fall with the work, and it does not. **Piecemeal trunk offload is
> closed: no further Sonnet item of this shape.** Merged behind its knob, off.
> **What is left of F3 is one item, step 2b: the decode token recorded once on
> the GPU -- DeltaNet conv + recurrence, attention with KV on dev0, router,
> top-k, the expert dispatch to dev2/dev3 and lm_head -- with the hidden state
> never returning to the host between layers.** That is the design hipFire
> uses for 88-124 tok/s on one of these cards with these weights. It is a new
> decode path, not a port: Opus, a design note first (which ops have shaders
> today: `kda_*`/`rmsnorm`/`qmatmul*`/`mla_attn_*` for GLM; what DeltaNet's
> gated delta rule, the conv ring and top-k need), then weeks of kernel work,
> plus slots and prefix checkpoints before it can replace GLM as the daily
> model. **It is the owner's spend decision and is NOT started.** Until then
> F3 stands at: Qwen3.6-35B-A3B int8, 100 % resident on two cards, 21.5-22.4
> tok/s, quality cost of int4 removed.
>
> **Rev 36 (2026-09-20 17:15 CEST, 15:15 UTC) -- F3 STEP 1 DONE (Sonnet, one
> afternoon): the int8 container is 100 % resident over dev2+dev3 and costs no
> speed; the profile says what step 2 has to be.** Record §F3-STEP1: 10 240 /
> 10 240 experts, 16.17 GB per card, CPU misses 0; tier on/off mean KL
> -1.7e-10, top-1 100 % over 625 positions; with the int4 container on dev3
> alone the new binary's logits are bit-identical to V1's. A,B,B,A against V1
> (int4, one card): decode 21.54 -> 21.52 tok/s (NO VERDICT, did not move),
> TTFT -2.2 % conservative. **So the quality step 0 asked for is free.** The
> token, 47.0 ms: DeltaNet 19.1 (its projections 10.8), MoE 13.6 (tier issue +
> take 9.9, shared expert 2.6 on the CPU), attention 8.0, lm_head 6.3 -- 71 %
> is CPU trunk. **What that means for step 2, stated before anyone builds:**
> porting the CUDA tier's trunk placement (`qt_dnproj_*`, `qt_lmhead_*`,
> Sonnet) moves 17 ms of CPU work to dev0 but pays a submit + fence + copy per
> call (0.25 ms per layer measured on the tier above): by arithmetic 47 ->
> ~36-38 ms, **~27 tok/s, not hipFire's 88-124.** That class needs the whole
> decode token recorded on the GPU with the hidden state never returning to the
> host -- DeltaNet recurrence, attention + KV, router and top-k on dev0 -- which
> is a new engine path, Opus, weeks: rev 35's "the trunk, not the submit
> model" was half right; the trunk is where the time is, and the per-op round
> trip is why moving it piecemeal will not reach the resident class. **Step 2a
> (Sonnet, next): the trunk-placement port, gate >= +20 % decode A,B,B,A,
> tier on/off KL at the floor.** **Step 2b (Opus, the owner's call on spend):
> the GPU-resident decode token.** Serving Qwen3.6 behind `coli`'s model
> switch also needs slots and prefix checkpoints, which `qwen36` does not have
> (record §V1); not opened.
>
> **Rev 35 (2026-09-20 16:25 CEST, 14:25 UTC by the rig's clock; revs 32-34 are stamped about two hours ahead of it) -- F3 STEP 0 DONE: int4 costs
> quality, int8 is built, and F3's steps are re-cut around what step 0 found.**
> Measured (record §F3-STEP0, chain `f3s0_kl_chain.sh`, gateway down ~6 min,
> `accept_live` PASS): int4-gs64 against row-wise int8 on the CPU engine, same
> binary, containers differing only in the routed experts -- **mean KL 0.0316,
> top-1 92.96 % over 625 positions** (short prompt 0.0353 / 90.91 %). This
> track called 0.0233 / 91.95 % a defect (the swiglu clamp) and 3.5e-05 /
> 99.89 % "same". Half the flips are near-ties, which is what 4-bit noise is;
> it is still a cost on every token, and rev 34's rule decides: **F3 serves
> row-wise int8** (`~/models/qwen36_i8_row`, 35 GiB, converted today). int8
> against BF16 was not measured. **Two corrections to rev 33/34, both from
> `F3-STEP0-2026-09-20.md`:** (1) int8 needs NO new kernel -- the
> `qmatmul*.comp` shaders already carry `fmt 1 = int8` row-wise and the engine
> already labels the container `tier fmt=1`; it needs two expert cards (32.34
> GB against a 24.75 GB budget each), and the multi-device pattern exists,
> tested, in the CUDA tier (`c/qwen36_tier.c`: `home(eid)`, per-device
> budgets, `tests/test_qwen36_tier_multidev.c`). (2) "The per-op submit model
> separates 21.6 tok/s from hipFire's 88-124" was never measured and V1's own
> record says otherwise: tier submits are ~2 % of V1's decode token, the
> whole trunk (attention, DeltaNet, shared expert, lm_head, 4.90 GB F16) runs
> on the CPU and dev0 is idle. **F3's steps, re-cut: step 1 (Sonnet) -- the
> Vulkan tier over dev2+dev3 with the int8 container, a port; gate = 100 %
> residency, tier on/off KL at V1's floor, A,B,B,A decode and TTFT against
> V1's int4 one-card arm. Step 2 step 0 (Sonnet, `COLI_TIMERS=1` on step 1's
> B arm, no extra rig time) -- where the other 98 % of the token goes, per
> bucket. Step 2 (tier decided by that profile) -- the trunk on dev0, largest
> bucket first.** KV is not a constraint: 40 KiB per token, 19.6 GB free on
> dev0 after the trunk, more than the model's 262k context at four slots.
>
> **Rev 34 (2026-09-20 16:45 CEST, 14:45 UTC) -- the owner delegated the model
> choice to Fable; decided: F3 = Qwen3.6-35B-A3B, starting from the int4-gs64
> container that is already on the rig.** Reasons, all from the record: it is
> the only candidate whose engine and Vulkan tier already exist and are gated
> (V1); a sparse 3B-active model is not bandwidth-bound on these cards (88-124
> tok/s measured on one of them by hipFire) while a dense 27B is (~17 tok/s at
> BF16, ~30 at 8-bit, by arithmetic; not measured, and not worth an hour of
> gateway downtime to confirm); int4 leaves ~50 GB of the 72 GB for KV and
> slots. **8-bit is not the starting point:** it needs a new int8 tier kernel
> and a 70 GB BF16 download, so it is built only if step 0's KL of int4
> against int8 on the CPU engine shows int4 costs quality. The dense
> Qwen3.8-27B stays what it is today, a llama.cpp model, not a Colibri item.
> Do not re-open this choice with the owner; he asked for it to be made. [2026-09-22: the owner re-opened and closed it himself, rev 38.]
>
> **Rev 33 (2026-09-20 16:15 CEST, 14:15 UTC) -- THE OWNER NAMED F3's MODEL:
> the sparse 35B at 8-bit. F3 is unblocked; nothing built yet.** Checked the
> same hour, no rig time: (1) **there is no Qwen3.8 35B** -- Qwen's 3.8 family
> on Hugging Face is 27B dense, Flash-Next (the 173 GiB model `qwen38` serves)
> and 2.4T-A95B; the sparse 35B is **Qwen3.6-35B-A3B**, official FP8 repo
> `Qwen/Qwen3.6-35B-A3B-FP8` = **37.5 GB** of safetensors, which is the size
> the owner quoted. (2) Colibri already has its engine (`c/qwen36.c`) and V1's
> Vulkan tier (`c/qwen36_tier_vk.c`, merged; 10 240 int4-gs64 experts resident
> on ONE card in 18.12 GB, 21.6 tok/s), and the int4 container is on the rig
> (`~/models/qwen36_i4_gs64`, 22 GB). (3) **The engine does not read FP8**: its
> 8-bit form is row-wise int8 from `convert_qwen36.py --ebits 8` (from the
> BF16 repo), and the Vulkan tier packs int4 only -- so "8-bit resident" needs
> an int8 tier kernel, and the item's first question is whether 8-bit buys
> quality over the int4 that already runs (a KL measurement, Sonnet). (4) The
> owner's other candidate, Qwen3.8-27B dense, is 55.6 GB at BF16 (30.9 at
> FP8), bandwidth-bound near 17 tok/s across three cards, and Colibri has no
> dense GPU engine: not pursued here; a Q5 GGUF of it already runs under
> llama.cpp on this rig. **F3's step list: step 0 (Sonnet) int4 vs int8 KL on
> the CPU engine + the three-card tier budget with KV on dev0; step 1 (Opus)
> the three-card tier and, if step 0 says so, the int8 kernel; step 2 (Opus)
> the per-op submit model, which is what separates 21.6 tok/s from hipFire's
> 88-124 on the same weights and card.**
>
> **Rev 32 (2026-09-20 15:30 CEST, 13:30 UTC) -- governance audit before a session restart;
> no rig time, no engine change.** The rev log above was current but §8.3's
> table had stopped at F7: it now has rows for F8, F9a and F10, F5 reads
> DEFERRED, F6 reads F6a DONE, F3 reads NEXT, F7's "awaiting a Fable decision"
> is closed, and §8.4 carries the order as executed. `doc_currency.sh` now
> tracks F items and checks this table, because that is how the gap went
> unnoticed for a day. `CLAUDE.md` had never named this plan in its read-first
> list although it has been the only active track since 2026-09-16; it does
> now. The narrative of F0 → F10 (why each decision was taken, what was
> refuted, the working method) is `tools/hot-expert/HANDOFF-2026-09-20.md`.
> One observation from reading the served profile for that file, NOT an item
> yet: in decode at 18.6k KDA `proj` is 2.51 s of a 13.2 s turn (19 %), 0.911
> ms per KDA layer for eight batch-1 GPU matmuls at ~0.4 ms of submit+fence
> each -- G13's `coli_vk_matmul_pair` pattern (one submit) is the obvious
> Sonnet-sized probe; its step 0 is counting submits per KDA layer.
>
> **Rev 31 (2026-09-20 15:15 CEST, 13:15 UTC) -- F10 step 0 read by Fable:
> KDA in prefill is a GPU dispatch-latency problem, and the item is parked for
> the owner.** Record §F10-STEP0 corrected Fable's own brief: in service
> (`COLI_KDA_GPU=2`) the eight projections AND the recurrence run on the GPU;
> `coli_vk_kda_step_rows` records S sequential 64-workgroup dispatches, one per
> token, each fenced by a memory barrier -- 0.111 ms per row per layer, 33.9 s
> of the 357 s 18k turn, latency-bound, not compute-bound. The candidate fix is
> to move the token loop INTO the shader (one dispatch per layer-chunk, each
> head's invocation walking its S tokens in order: the per-head arithmetic and
> order are unchanged, so it can be held to a bit-identical oracle): upper
> bound ~30 s of 357 s (8 %) on long prompts, nothing for decode. That is a
> shader change (Opus by this plan's tiers, ~200-300k tokens by F7's bill) for
> a single-digit gain; **not started -- the owner decides whether it is worth
> the tokens.** `proj` (40.8 s, eight GPU matmuls per layer-chunk) was not
> examined further. Everything measured and cheap on this model has now been
> taken: since 2026-09-18 the 18k turn went 1 174.8 -> 356.4 s and decode at
> 18k 4.30 -> 5.04 tok/s. **F3 (a <= 60 GB VRAM-resident model) is the next
> item of any size.**
>
> **Rev 30 (2026-09-20, Sonnet) -- F10 step 0 done: reading + a CPU
> microbenchmark, no engine touched.** Record §F10-STEP0. `kmv_rows` (the
> eight KDA projections, `proj` in the split) runs on the GPU in the served
> engine (P4c tiled `coli_vk_matmul`, weight reads amortised over 8 rows and
> over the chunk), and the record's own chunk-128-vs-512 data already shows
> it getting cheaper, not doubling, at the larger chunk -- **it is not the
> item's lever.** The CPU recurrence kernel (`coli_kda_step`,
> `delta_attention.h`) is real but memory-bound (~0.5 FLOPs/byte,
> `f10_kda_bench.c` measured 0.30 ms/token at 8 threads on one layer's 4 MiB
> state, 18.2/38.5/56/14.8 GB/s across the single/multilayer x 1/8-thread
> cases), and its only known cliff (136 MiB state vs 128 MiB L3) is already
> removed by the serving knob and not bit-identically fixable otherwise --
> the microbenchmark reproduced that cliff's bimodality on its own, without
> touching the engine. **The actual 2.05x S=512-vs-128 doubling (record
> §F2f, ~37.7 s of KDA's 93.3 s) is entirely in the GPU dispatch path
> (`coli_vk_kda_step_rows`: S sequential 64-workgroup dispatches serialised
> by full memory barriers in one command buffer) -- "observed, not
> diagnosed" per the record, and this item was scoped to not touch Vulkan.**
> Next real step on this bucket is a GPU-side diagnostic chain
> (per-dispatch/barrier timestamps, reusing the existing `VK_PROF`/
> `VK_EXT_calibrated_timestamps` pattern) to learn whether the cost is
> barrier drain, dispatch launch overhead, or GPU power-state cycling --
> **Opus-tier** (profiling and its interpretation, on a new instrumentation
> path), before any fix is designed. Not started.
>
> **Rev 29 (2026-09-20 14:20 CEST, 12:20 UTC) -- F9 re-scoped to F9a, F9a IS
> GATED PASS AND IN SERVICE; the GPU indexer is not needed.** Record §F9a and
> §F9a-VERDICT. Before spending Opus on a GPU indexer, Fable read the score
> pass: a scalar float dot GCC cannot vectorise, the same disease P5.1 cured in
> the attention core. F9a (Sonnet) puts the heads in the SIMD lanes,
> bit-identical (teacher_forcing, text, the full 5.6 GB logit dump and 33 495
> index rows all identical to the served binary): **the 18 439-token turn 455.0
> -> 357.7 s (1.27x), decode at 18k 4.65 -> 5.04 tok/s (+8.3 %), follow-up
> TTFT -20 %**; `index` in decode 2.26 -> 0.46 ms per call. Served `189fd945`
> since 12:13 UTC. **F9 (the GPU version) is CLOSED without being built.**
> Since 2026-09-18 the 18k turn has gone 1 174.8 -> 356.4 s (3.30x) and decode
> at 18k 4.30 -> 5.04 tok/s. What is left on this model, by size: `ffn_moe` in
> prefill (streaming-bound), KDA (chunk 512 made it cost ~2x per token -- a
> Sonnet-sized look at `coli_vk_kda_step_rows` at S = 512 vs 128 is **F10**,
> not started), the CPU misses in decode (~60 ms per token; a kernel redesign,
> Opus, not opened). **F3 (a <= 60 GB model, VRAM-resident) remains the only
> order-of-magnitude lever and waits on the owner naming the model.**
>
> **Rev 28 (2026-09-20 11:00 CEST, 09:00 UTC) -- F8 IS GATED PASS AND IN
> SERVICE.** Record §F8. One OpenMP team per MoE layer (bit-identical in the
> engine, dump-verified) plus the group-vector int4 row kernel in the batch-1
> CPU expert path: **decode +6.4 to +9.6 % conservative, SEPARATED at all
> seven turns to 19.3k (4.270 -> 4.585 tok/s at the deepest)**, TTFT
> unchanged, greedy text identical; `cpu_in` per window 1.868 -> 1.489 ms.
> Served `93f0f681` since 08:54 UTC, `accept_live` and `accept_ui` PASS. The
> CPU misses are still ~60 ms of the decode token and still the largest
> decode bucket; what is left in that kernel is a redesign (several output
> rows per nibble decode), an Opus item, not opened. **Open and NOT started,
> both the owner's call: F9 (the indexer's prefill score pass on the GPU, 25 %
> of the 18k turn, Opus for the shader) and F3 (a <= 60 GB model, fully
> VRAM-resident -- the only order-of-magnitude lever left).**
>
> **Rev 27 (2026-09-20, Sonnet) -- F8 engine step BUILT, NOT GATED.**
> `coli_i4_row_gv` (record §F8-STEP0) is in `c/quant.h`; `GLM53_I4_FAST=2`
> wires it into `mlp3_cpu` and, as a new case not present in step 0, into
> `mlp3_cpu_rows`' per-row tail, with `mlp3_cpu_rows_ok()` relaxed so `=2`
> (unlike `=1`) does not fall back off the rows path. `GLM53_MOE_ONE_TEAM=1`
> fuses a decode window's deferred CPU experts into one OpenMP team
> (falls back to today's per-expert loop, unfused, if any expert in the
> window is not rows-eligible). Both knobs default off, unchanged
> branches when unset. Branch `perf/f8-cpu-expert-gv`; a new C test
> (`tests/test_glm53_f8_moe_fuse`) checks the fusion is bit-identical to
> knob-off and that `=2` differs from and stays within a tight relL2 of the
> default kernel. `tools/hot-expert/f8_gate_chain.sh` (oracle + A,B,B,A
> decode ladder, pristine `~/bench/glm53.f7`) is written but not launched —
> **no in-engine number yet.** Next: run it.
>
> **Rev 26 (2026-09-20 09:15 CEST, 07:15 UTC) -- F8 step 0 ANSWERED (Sonnet,
> record §F8-STEP0).** Three arbitration rounds on the same CPU
> microbenchmark, no engine change, no gateway downtime. Round 1's near-
> linear 1->8 thread scaling looked bandwidth-bound; round 2 tested and
> REFUTED the alternative "FMA-latency-chain" hypothesis directly (§G14's
> own `coli_i4_row_f4`, 4 independent accumulators, is statistically flat
> against the fork/join fix in the engine's own multi-row-per-thread
> structure, and an 8-accumulator extension is measurably worse); round 3
> found the real mechanism -- with gs=64 every group pays a 64-deep SCALAR
> reduction chain (`hsum256` + `fmaf` into the row total) across a 4096-wide
> row, on top of the vector work. A group-vector combine that stays in
> vector form until one `hsum` at the row's end (`coli_i4_row_gv`, same
> bit-trick decode as `f4`, reassociation-only, rel_l2 ~3.3e-07, the same
> order as `f4`'s own ~1e-7) plus fusing the window's three thread-team
> spawns into one (bit-identical) together project **~30% off the 73 ms
> CPU-expert decode bucket, ~21.6 ms/token, upper bound** (§G14's own
> isolated-to-in-engine attenuation, 1.26-1.65x -> 1.095x, means the
> delivered number will be smaller; not yet measured in the engine). F5's
> precondition ("a materially cheaper CPU miss") is PROJECTED but not
> DELIVERED, so **F5 stays deferred** until the change lands and is measured
> in situ. **Order now: wire `coli_i4_row_gv` + the fork/join fusion into
> `ffn_moe_run_deferred_cpu` behind `GLM53_I4_FAST`, gate it like F7
> (`teacher_forcing`+KL, A,B,B,A ladder, a jitter arm) -> re-open F5 if the
> in-engine number holds up; F9 (indexer on the GPU) in parallel, Opus for
> the shader; F3 when a model is named.**
>
> **Rev 25 (2026-09-20 08:00 CEST, 06:00 UTC) -- the profile after F7, F5
> deferred, F8 started, F9 opened.** No rig time: the ladder's own B arm
> (`f709200252_B1`, `[OPTIME req=6 ctx=18439]`, 482.8 s of layer time) is the
> re-profile. **The 18k turn is now: `ffn_moe` 140.6 s (29 %), the DSA indexer
> 119.1 s (25 %), KDA 93.7 s (19 %; proj 41.3 + step 33.8), hc+norm 48.5 s
> (10 %), `mla.proj` 27.7, the attention core 26.5 (5 %; it was 35 %), dense
> 25.1.** Decisions (Fable): (1) **F5 (MTP) is deferred, not run**: its own
> precondition was "after F2/F1 change the verify block's cost"; F1 was
> rejected and the probe shows a k = 2-6 row block is slower streamed than on
> the CPU, so the verify block costs what it cost when Q9 killed MTP here.
> Re-open only if F8 makes a CPU miss materially cheaper. (2) **F8 step 0
> starts now** (Sonnet, a CPU microbenchmark, no gateway downtime): where the
> 0.56 ms of a batch-1 CPU expert goes -- nine OpenMP fork/joins per window,
> the int4 unpack, or memory bandwidth -- because those 73 ms per token are the
> largest decode bucket and decode is what every reply waits for. (3) **F9 is
> opened: the indexer's score pass in prefill, batched on the GPU the way F7
> did the attention core** (25 % of the 18k turn, the largest prefill bucket
> that is still a CPU loop; design first, after F8 step 0 reports, Opus only
> for the shader). Order: **F8 step 0 -> F8 or F9 by its answer -> the other;
> F3 when a model is named.**
>
> **Rev 24 (2026-09-20 07:30 CEST, 05:30 UTC) -- F7 IS GATED PASS AND IN
> SERVICE.** Record §F7-VERDICT. Numerics arbitrated by a jitter arm: a random
> fp32-rounding-sized perturbation of the CPU attention core moves the model as
> far as the GPU kernel does (KL 0.00263 vs 0.00279, top-1 98.17 vs 98.34 %,
> same first position) -- the kernel is sound, and for reordered attention
> kernels the bar is now "no further than the eps = 6e-7 jitter arm". Full 18k
> ladder A,B,B,A: **turn 1 68.2 -> 50.5 s (1.35x), the 18 439-token turn 683.3
> -> 454.6 s (1.50x, bar 1.25)**, decode at full depth unchanged (one follow-up
> row -3.4 %, recorded). Served `e87ae939` since 05:24 UTC, `accept_live` and
> `accept_ui` PASS. F2 + F7 together: the 18k turn 1 174.8 -> 454.6 s (2.58x).
> **Order now: re-profile the 18k prefill token (one timed arm) -> F5; F8 (the
> batch-1 CPU expert path, the largest decode bucket) when the rig is idle; F3
> when a model is named.**
>
> **Rev 23 (2026-09-20 02:00 CEST, 00:00 UTC) -- F7's numerics: the defect hunt
> found NO DEFECT, and rev 22's explanation of the gap was WRONG.** Record §F7
> round 2. Fable's float64 arm (`GLM53_MLA_ATTN_REF64=1`) refuted rev 22's "the
> CPU is the less accurate side": the CPU fp32 path is within **1.7e-05 mean KL
> / 100 % top-1** of float64 on the shallow packet, against the GPU's 0.00279 /
> 98.34 %. Every defect hypothesis was then eliminated **by measurement**: the
> four attention shaders replayed on REAL engine chunks against a float64 core,
> all 512 rows x 64 heads x 11 DSA layers, dense AND sparse, come out at
> GPU/CPU-vs-float64 ratio **0.80-1.27**; the batched o-projection is
> **bit-identical** to the per-row one (mean KL 0, maxabs 0) and never falls back
> to the CPU kernel; the sub-batch loop is **bit-identical** at sb 512 vs 128
> (mean KL 0); tier, streaming, packets and run-to-run noise all identical.
> **The decisive measurement is the in-situ dump** (the knob-ON run writing its
> own `context` for the same chunk, diffed row by row with
> `tools/hot-expert/f7_ctxdiff.py` -- a replay of the CPU run's inputs cannot see
> this): over eleven DSA layers the **median** per-row error goes 2.9e-07 ->
> 6.2e-06, never leaving fp32 level, while the p90/p99/max explode
> (4.1e-07 -> 6.9e-02). The divergence lives entirely in a growing minority of
> near-degenerate softmax rows -- the same population Fable's flip-margin
> analysis found. **So there is nothing to fix, and the proposed shallow bar
> (`ref64||gpu` top-1 >= 99.9 %) is unmeetable by construction**: `ref64||off`
> changes only PRECISION (same summation order, more bits) and so preserves
> near-tie orderings almost perfectly, while any reordered kernel perturbs them
> randomly at the same RMS error. The depth-relative arm is the meaningful one
> and F7 is close: R2/R1 at depth is **1.7x on mean KL and -1.4 pt on top-1**
> against an allowance of 1.2x and -0.3 pt. **Speed, current code: deep 678.8 ->
> 479.0 s = 1.417x, shallow 206.5 -> 131.7 s = 1.57x**; `GLM53_MLA_ATTN_SB=512`
> is bit-identical and slightly cheaper (66 submits instead of 264) but has no
> gate behind it. **Decision needed (Fable): accept F7 against a depth-relative
> bar and say what it is, or require a bit-identical kernel -- which means the
> CPU's summation order on the GPU, i.e. no tiling, i.e. no F7.** Order
> otherwise unchanged: F5; F8 when the rig is idle; F3 when a model is named.
>
> **Rev 22 (2026-09-19 22:15 CEST, 20:15 UTC) -- F7 is BUILT and MEASURED: the
> speed is there and the NUMERICS BAR IS NOT MET. Blocked for arbitration, not
> tuned.** Record §F7, design note
> `tools/hot-expert/F7-MLA-ATTN-GPU-DESIGN-2026-09-19.md`, branch
> `perf/f7-mla-attn-gpu` (`b538764`), knob `GLM53_MLA_ATTN_GPU=1` off by default.
> The whole `_tm2` window -- score, softmax, weighted pool, `kvb_v` value rows --
> is four dispatches on dev0, and the o-projection behind it is one submit per
> chunk instead of 512. **Microbenchmark: one 512-row layer-chunk 1 290 → 74 ms
> (17x)** at the engine's own shapes through the engine's own entry point, with a
> scalar CPU-reference diff of 1.2e-09 on values of 2.3e-03. **Engine: a
> 9 115-token prefill 678.8 → 479.0 s = 1.417x, a 3 013-token one 1.594x**, one
> binary, knob the only difference -- the design projected 1.43x and got it.
> **O1 passed: knob-off is bit-identical to the served `5c01246c`, deep and
> shallow.** But **X2's KL bar is NOT met: mean KL 0.00279 (shallow) and 0.00538
> (deep) are inside the < 0.0284 bar with 5-10x margin, while top-1 agreement is
> 98.34 % and 97.14 % against a >= 99.0 % bar, and it worsens with depth.** The
> record argues this is score-pass reassociation in which the CPU is the LESS
> accurate side (`glm_lane_dots` rounds each product and adds 512 of them in
> sequence, ~5e-04 relative; the GPU's K-tiled fma is ~10x tighter), so making
> the kernel more accurate moves it further from the reference, not closer; the
> probe rules out a layout/dequant/gather defect and the first disagreement is at
> the same position (1420) in both packets. **VRAM answer, for the record: 140 MB
> of shared dev0 scratch at 18k and 236 MB at the 65 536 cap, nothing per layer
> and nothing per slot, expert tier caps untouched** -- the decode path's
> persistent per-layer KV mirror was rejected at 1.47 GB per sequence against
> dev0's 1 447 MiB free. The A,B,B,A ladder was deliberately NOT run: the item is
> blocked and the two CLI arms are the better speed evidence. **Decision needed
> (Fable): accept ~3 % top-1 movement against the reference's own fp32 noise for
> 1.42x, or open a higher-precision score pass on BOTH sides (which changes the
> served binary's numerics and is a different item). Order otherwise unchanged:
> F5; F8 when the rig is idle; F3 when a model is named.**
>
> **Rev 21 (2026-09-19 20:30 CEST, 18:30 UTC) -- F1 is REJECTED by its own
> falsifier probe.** Record §F1-PROBE: a streamed decode miss at batch 1 costs
> 2.70 ms per window at k = 3 (median; fit 1.50 + 0.38*k on three cards, 0.92 +
> 0.70*k on dev2 alone) against 1.747 ms on the CPU path -- slower at every k,
> because one row still pulls the whole 14.16 MB expert across PCIe. The 54.6
> ms/token of CPU misses stays the largest decode bucket; **F8** is opened for
> it (profile the batch-1 int4 CPU expert path and its threading across the
> ~3 misses of a window: 25 GB/s of weight reads is far under the box's memory
> bandwidth; Sonnet, not started). **Order now: F7 design -> F7 -> F5; F8 when
> the rig is idle; F3 when a model is named.**
>
> **Rev 20 (2026-09-19 20:00 CEST, 18:00 UTC) — F1 step 0 ANSWERED: the CPU
> misses ARE decode's critical path, ceiling −54 ms/token (+29 % at 18k); F1
> proceeds to a latency probe before any engine work.** Record §F1-STEP0.
> Chain tag `f1s009191720`, rc=0, `teacher_forcing` **IDENTICAL** against the
> served pristine `5c01246cdea888dc`, `accept_live` PASS, candidate `e8f8ca32`
> **timers only — never put in service.** Three new accumulators inside the
> existing G9 overlap gap (no computation changed) show, at both
> near-decode-only follow-up turns (req 7/8, ctx 18.6k): `take ≈ 0` (0.029–
> 0.032 ms/window) while `cpu_in` (1.75–1.91 ms/window) is far above
> `take_cpu0`, the GPU-alone reference (0.48–0.51 ms/window) — the GPU groups
> are always finished well before the deferred CPU experts, so **the
> CPU-computed non-resident experts, not the GPU fence wait, are the critical
> path in `ffn_moe`.** Ceiling with the CPU path free: (`cpu_in` + `take` −
> `take_cpu0`) × 42 layers = **54.6 ms/token, +29 % decode at 18k** —
> confirms the plan's "≤ 45 ms/token" guess for F1 in kind, revises it upward
> in size. **Not free**: F1's real gain depends on the batch-1 streamed-miss
> latency (fill+submit+compute+readback for ~3.1 experts/window), which is
> unmeasured — below ~1.2 ms/window the gain is still ~25–30 ms/token
> (~+12 %); at or above today's ~1.75 ms/window CPU cost F1 has nothing to
> gain. `cpu_in` does not depend on depth (1.60–2.14 ms/window from ctx 1.3k
> to 18.7k) but is a much larger SHARE of the forward at shallow depth, where
> the other buckets are cheaper. The turn-4 decode row from §F2-LADDER
> (record, restated) remains **not explained by the logs**: a ~1.4 s
> decode-only signal is invisible inside a per-request table that sums an
> entire prefill-dominated window. **Order now: the falsifier probe
> (`f1_ring_probe`/`f1_probe_chain.sh`, batch-1 decode-shaped ring latency,
> k = 1, 2, 3, 4, 6 experts/window) → F1 (if the probe clears) or F7
> (if it does not) → F5; F3 when a model is named.**
>
> **Rev 19 (2026-09-19 18:15 CEST, 16:15 UTC) — F2 IS GATED PASS on the full
> 18k ladder. Both bars met; the order moves on.** Record §F2-LADDER. Chain
> tag `f209191332`, rc=0, `accept_live` PASS, one binary `5c01246cdea888dc`,
> A = clamp on / stream off / chunk 128, B = clamp on / stream on / chunk 512
> (the clamp in BOTH arms, so the A/B measures the streaming and not the
> clamp; A1 reproduces §RP-F6a's unclamped A arm to within 0.4 %).
> **Turn 1 136.8 → 67.6 s = 2.03× (bar 2.0, MET). The 18 439-token turn
> 1 174.8 → 672.6 s = 1.75× (bar 1.6, MET)**, both SEPARATED by
> `gate_ab_verdict` (−50.5 % and −42.6 % conservative). Cold prefill to 18 439
> tokens 37.5 → 21.1 min. Decode at full depth 4.30 → 4.33 tok/s (+0.7 %
> conservative, separated); follow-up TTFT −14.8 to −16.6 %. **One decode row
> goes the wrong way: turn 4 (ctx 9 459) separates AGAINST B by 4.5 %**, the
> only place a prefill-only change slowed decode, and F1 should start there.
> X2's KL bar was met with the clamp on before the run (C2 mean KL 3.48e-05,
> top-1 99.89 %) and knob-off is bit-identical to the served pristine.
> **The prefill token at 18k is now 79.2 ms, and the shape has inverted:
> `ffn_moe` 69.5 → 16.0 ms/token (51 % → 20 %), and the largest bucket is the
> MLA attention core at 27.6 ms/token (34.9 %), the second the DSA indexer at
> 13.4 (17.0 %).** So **F7 is re-based and is the next prefill item** — its own
> 28.4 → ≤ 8 ms/token target would take the 18k token to ~59.4, another 1.33×
> — and the indexer is back to second place without having got slower (F6a
> left it at 13.4 ms/token and it is still 13.4; only the denominator shrank).
> The projection check: rev 18 projected 80.0 ms/token and 1.71×, measured
> 79.16 and 1.75×, and the two component errors nearly cancelled (`ffn_moe`
> came in above the projection, `kda` below it). Beside it and unchanged:
> llama.cpp does this turn in 108 s; F2 closes the gap from 10.9× to 6.2×, it
> does not close it. **Order now: F1 step 0 (the eg/cpu overlap split) → F1 →
> F7 → F5; F3 when a model is named.** Putting the candidate in service is a
> separate step and is not recorded here.
>
> **Rev 18 (2026-09-19 15:45 CEST, 13:45 UTC) — F2 failed X2's KL bar, the
> cause was the swiglu clamp, the clamp is fixed behind a knob, and F2 now
> passes at 99.89 %.** Record §F2g–§F2k. (1) **The missing gate line.** X2's KL
> bar (mean KL < 0.0284 AND top-1 >= 99.0 %) is part of the §8.3 F2 row and
> rev 17 did not apply it. On the same dumps: streaming **fails at top-1
> 94.97–95.05 %** while the mean KL passes with room. (2) **Step 2a, the
> clamp.** `qmatmul_gate_up.comp` and its tiled twin now carry GLM-5.3's
> clamped SwiGLU, transcribed from `swiglu_clamped` (one-sided on gate,
> two-sided on up), behind `GLM53_VK_SWIGLU_CLAMP=1` and applying to resident
> and streamed experts alike. Off is a fact, not a promise: the clamped kernels
> are a second SPIR-V module on a second pipeline layout, built only when a
> positive limit is set, and the unclamped `.spv` files are **byte-identical**
> to the served tree's — so **qwen38-vk is literally unchanged** and owes no
> oracle. The kernel was proved model-free in one minute (huge limit reproduces
> the unclamped kernel exactly; limit 10 does not) before any gateway window
> was spent. Cost: +0.3 %. (3) **With the clamp on, F2's KL gate passes:
> mean KL 3.48e-05, top-1 99.89 % (6 320/6 327)** — 354× better than the
> unclamped line. The clamp itself is worth 91.95 % → 98.84 % top-1 against a
> clamped all-CPU reference, i.e. it cuts the §G13 placement-dependence from
> 8.05 % of argmaxes to 1.16 %; **the SERVED engine fails that bar at 91.95 %
> today.** Measured against the served binary, clamp-alone, clamp+stream@128
> and clamp+stream@512 are the same number (top-1 93.49 / 93.54 / 93.52 %):
> **once the clamp is on, F2 is numerically free, and what ships is the fix.**
> (4) **Step 2c**: streaming no longer moves the prefill chunk. Within one
> chain the chunk is worth 1.12× (509.6 → 453.7 s), not rev 17's 3 % — that
> figure compared two chains and this session measured 6.1 % run-to-run spread
> on the identical quantity. **Recommendation: serve at chunk 512**, because
> turn 1's 2.0× bar was met by 1 % and chunk 128 is ~12 % slower, and because
> the numerics reason for 128 evaporated with the clamp. (5) **Why the chunk
> changes numerics at all**: the indexer is ruled out by reading (`last <= q`
> is per-query and implies `complete[p]`) and the tile threshold by measurement
> (tiled and per-row kernels agree bit for bit on the same row); what is left
> is the union-order accumulation into `out[]`. The MLA absorb's reduction and
> the down projection's tile threshold were not examined. (6) **Still open: the
> 18k ladder**, now to be run with the clamp on in BOTH arms.
>
> **Rev 17 (2026-09-19 13:00 CEST, 11:00 UTC) — F2 is implemented, its oracle
> passes, and its short ladder meets BOTH gate bars; the 18k rung is queued and
> is the only thing left.** Record §F2. (1) **The design was refuted by its own
> microbenchmark before a line of engine code was written**: a CPU `memcpy` into
> ReBAR write-combined VRAM reaches 16.3 GB/s on three cards, against 58.6 GB/s
> into cached DRAM, so the ring lives in HOST memory and the shader reads it
> over PCIe — 74.7 ms per 144-expert wave against 102.6, and **zero VRAM**, so
> the tier, the G6 reserve and the 1 695 caps are untouched by construction.
> A CPU store over PCIe is not a GPU DMA read over PCIe; §PCIE-STREAM's 61 GB/s
> was the second of those. (2) **Oracle**: knob-off is bit-identical to the
> served binary at 2 061 and 6 327 tokens (3.92 GB of every-position logits,
> `cmp`-clean); knob-on against the UNCLAMPED all-CPU reference
> (`GLM53_EXPERTS_CPU=2`) is `teacher_forcing` IDENTICAL with cosine 0.99999998
> and argmax agreeing on 2 061/2 061 positions — **the streamed kernel is
> right**, and separately at an empty tier, i.e. 100 % of experts streamed, it
> is identical too. (3) **Short ladder A,B,B,A, one binary**: turn 1
> 136.6 → 67.5 s = **2.02×** (bar 2.0×, MET — and turn 1 here IS the gate's own
> turn 1, ctx 1 287, reproducing §RP-F6a's 136.5 s to 0.4 %), deepest turn of
> the short rung (5 112) 281.3 → 156.7 s = **1.80×**, decode not dropped,
> follow-up TTFT −17 %, `accept_live` PASS. **The 18k rung is NOT run** (a
> multi-hour outage) and is handed back ready to launch; the arithmetic from the
> measured buckets projects 136.4 → 80.0 ms/token = **1.71×**, TTFT ≈ 690 s
> against the ≤ 737 s the gate allows — a 6 % margin, and a projection.
> (4) **Two things the run found that the plan did not expect.** The chunk
> change that rides in on the knob (128 → 512) is worth only **3 %** of the
> speed at 6.3k (466.6 s at chunk 128 against 451.8 at 512, both streamed),
> because chunk 512 saves 60.2 s in the MoE and gives 46.7 s of it back in the
> KDA and the attention core: **`coli_vk_kda_step_rows` (P5.2) costs 2.05× per
> token at S=512 than at S=128**; fixing that would take F2 from 1.62× to ≈1.77×
> at 6.3k and is a small, well-defined item. And `mla.attn` is 5.9 % worse at
> chunk 512, which F7 should know before it assumes bigger batches are better.
> The chunk costs **nothing** numerically on top of the streaming, which took a
> third oracle arm to establish and reversed the first reading: streaming at the
> DEFAULT chunk moves 5.02 % of the argmaxes and streaming at 512 moves 4.95 %,
> against the chunk's own 3.24 % — comparable nudges to the same near-ties, not
> perturbations that add. So chunk 512 stays, for a different reason than the
> design gave it.
> (5) The streamed path inherits the §G13 swiglu-clamp gap and gives it more of
> the model's calls; F2 deliberately did not clamp only the streamed shader
> (that would make an expert's numerics depend on which side of the ring it
> landed — a second §G13, not a fix). **The clamp is now the item that gates
> making F2 the default.** Order unchanged otherwise: F2's 18k rung → F1 step 0
> → F1 → F7 → F5.
>
> **Rev 16 (2026-09-19 09:15 CEST, 07:15 UTC) — F6a is in service, the
> re-profile is done, and F2's gate is re-based because its old one cannot be
> met by any expert-side item.** (1) **F6a**: gated PASS and serving since
> 2026-09-19 01:07 UTC (`15462dc2093b8ad0`; record §F6a, prefill roadmap rev
> 33). (2) **Re-profile** (record §RP-F6a; F4's ladder on the served binary,
> `COLI_TIMERS=1`, 45 min, `accept_live` PASS): decode at 18k **2.27 → 4.33
> tok/s** (design projected 3.9, ceiling 4.42), the 18k ladder turn's TTFT
> **2 607 → 1 179 s**, cold TTFT at 18.4k 37.8 min, index 24.9 → 2.1–2.2
> ms/call in decode. Before and after are different runs two days apart; the
> paired verdict is §F6a's. (3) **The 18k prefill token is now 136.4 ms:
> experts 69.8 (51 %), attention core 28.4 (21 %), indexer 13.4, KDA 13.2,
> hc+norm 4.8, proj 3.7, dense 3.1.** F2 with streaming at its measured 6.22
> ms/token and free GPU compute reaches 72.8 ms/token: **ceiling 1.87× at
> 18k, 2.4× at depth 0.** The §8.3 gate "≥ 10× on the 18k turn or rejected"
> dates from the reading rev 15 withdrew (113.9 ms/token of experts); 10× is
> 13.6 ms/token, under half of what attention alone costs. **Decision
> (Fable): F2 stays next — it is still the largest bucket at every depth and
> F1 shares its ring buffer — with the gate re-based to ≥ 1.6× on the 18k
> ladder-turn TTFT (≤ 737 s against 1 179 s) and ≥ 2.0× on turn 1, A,B,B,A,
> oracle unchanged; the llama.cpp 108 s row is still reported beside it. The
> order-of-magnitude prefill promise of §8.5 regime A is withdrawn: on this
> profile regime A's long-document wait halves, it does not fall 10×.** A new
> item **F7 — batched MLA attention core in prefill** (28.4 ms/token, 39 % of
> the token after F2; design first, Opus) is opened behind F2; the two
> together bound 18k prefill near 52 ms/token (≈ 2.6×). (4) Decode at 18k is
> now moe 47–49 %, attention 16 %, KDA 16 %, indexer 9 %. F1's target is the
> biggest decode bucket, but the `[PROF]` eg/cpu times overlap and its "≤ 45
> ms/token" is neither confirmed nor refuted; F1's first step is that split.
> Order now: **F2 (re-based gate) → F1 step 0 (eg/cpu overlap) → F1 → F7 →
> F5; F3 when a model is named.** Since regime A no longer promises 10×,
> the owner's §8.5 decision (name a ≤ 60 GB model for F3) carries more weight
> than it did at rev 13.
>
> **Rev 15 (2026-09-18 01:45 CEST, 2026-09-17 23:45 UTC) — the night's
> results, one incident, and the order changed.** (1) **F6 design** (Opus,
> `F6-INDEXER-DESIGN-2026-09-18.md`, merged): the DSA indexer is not a
> numerics problem — its score pass is a single-threaded scalar loop and its
> top-512 is a `wanted × pools` greedy scan (`c/sparse_index.h` ~275–298,
> verified), so **F6a = OpenMP over the pools + an exact heap top-k, bit-
> identical, a Sonnet item**, projected 25.0 → 3.3 ms per MLA call at 18.6k
> and decode 2.21 → 3.9 tok/s (ceiling 4.42). Corrections the design forced:
> §X3 step 0's 26 % lower bound is withdrawn (the follow-up prompt was 20
> tokens, so the indexer is 54 % of the 18k decode token, firmly); the
> attention core is flat with depth (the 2 051 width cap holds); coarser
> pooling is impossible (the pool compressor is a trained `[4][128]` tensor);
> X2's 564-token KL packet sits below the dense threshold and cannot gate
> any indexer change; and **F2's "113.9 ms/token" was the whole prefill
> token, not the expert part** — at 18k a prefill row is ≈ 168 ms indexer +
> 67 ms experts, so F6a is the first prefill lever too and F2's arithmetic
> is re-based after it. **F6a is implemented** (`perf/f6a-index-parallel`,
> bf6afd0, unit test 4 000 randomised trials, `GLM53_INDEX_SCALAR=1` restores
> the old path); its gate chain (oracle on a ≥ 4k-token prompt with logits
> bit-identical, then A,B,B,A to 9.5k with `COLI_TIMERS=1`) is queued on the
> rig and has not run — see (3). (2) **F2 step 0** (record §F2-STEP0/0b): the
> CPU fill alone reaches 59 GB/s at 8 threads, but fill overlapped with the
> GPU read sustains only **38.6 GB/s on three cards** (DRAM traffic triples or
> the handshake dominates — undecided, the number stands): streamed experts
> cost **6.22 ms/token**, not 3.9, still 18× under today's CPU path. The
> no-copy alternative (anonymous RAM imported into Vulkan, read in place)
> imports fine at 96 GiB but its bandwidth collapses with region size (35 →
> 10.7 GB/s from 8 to 64 GiB): rejected; F2 is built on the copy pipeline.
> (3) **Incident:** the 96 GiB import probe, killed at its budget, left one
> thread in uninterruptible sleep in amdgpu's userptr teardown
> (`mmu_interval_read_begin`), ~96 GiB pinned (MemAvailable 228 → 139 GiB),
> holding the rig lock through its parent chain. The gateway serves (its
> page cache is partly evicted; the first requests after this pay faults).
> **Owner action: reboot the rig**; the watchdog restarts the gateway, and
> the queued F6a chain launches itself if the lock frees before 08:00 UTC.
> Order now: F6a gate → re-profile (`COLI_TIMERS=1` A arm at 18k, 75 min) →
> F2 re-based on that profile → F1 → F5; F3 when a model is named.
>
> **Rev 14 (2026-09-17 22:45 CEST, 20:45 UTC) — F0 PASS, F4 answered; §8.3
> amended.** F0 (record §VK-STREAM): on RADV's dedicated transfer family
> (SDMA) three cards stream 27 GB/s — the falsifier — but on the backend's own
> queue family (the one `backend_vulkan.c` picks) `vkCmdCopyBuffer` reaches
> 59.9–61.1 GB/s and a compute shader reading host-visible staging directly,
> with no device copy at all, 62.2–63.3 GB/s: HIP's 61.4–62.0 matched, gate
> PASS, F2 stays Vulkan. Two facts F2 must design around: amdgpu refuses to
> import file-backed (page-cache) memory (`VK_EXT_external_memory_host`
> EACCES), so streamed experts pass through a host-visible staging buffer;
> and the probe's single-thread memcpy into that staging ran at 15 GB/s —
> **the staging fill, not PCIe, is F2's first bottleneck**, so F2 opens with a
> step 0 that measures a multi-threaded fill (8 threads read DRAM at 91.6
> GB/s; a parallel memcpy or `pread` into staging must reach ≥ 60 GB/s or the
> chunk arithmetic in F2 is off by 4×). F4 (record §X3 step 0, verdict): the
> DSA indexer is the depth term — 275 ms of a 509 ms forward at 18.6k in the
> decode-shaped windows, 26–54 % after the small-prefill caveat, slope
> estimate 38 % — so the ≥ 30 % test passes and **X3 enters §8.3 as F6, the
> decode item for GLM-5.3-Flash**, with its ceiling stated: 3.3 tok/s at 18k
> if the indexer holds its shallow cost, 4.3 if it vanished; the MoE bucket
> (125 ms/forward, of which CPU misses ≤ 45) and the KDA layers (42 ms) are
> what remains after it. Order now: F2 step 0 (Sonnet, 1 day) → F2 (Opus)
> in parallel with F6's design (Opus: which selection, what X2 bar, dense-
> identical below 2 052 tokens by construction) → F1 → F5; F3 when a model
> is named. Engine-mode `[OPTIME]` per request is now a standing tool
> (`COLI_TIMERS=1` on any ladder arm).
>
> **Rev 13 (2026-09-17, 02:00 CEST) — the Franken-engine design, §8.** Written
> from the measured regime map only (§FRANKEN-H2, §PCIE-STREAM, §GPTOSS-3CARD,
> §GLM53FLASH-LADDER, §X3 step 0, §G11, §RP1). Owner's stated goal: a model
> larger than one card over the three GPUs with experts or KV in RAM, at usable
> speed at realistic depth; GLM-class as the daily driver; gpt-oss-120b was the
> test vehicle only. §8 says what to build, in what order, with the projected
> delta and the falsifier for each item, and states plainly which goals the
> numbers do not support on this box.
>
> **Rev 12 (2026-09-17 01:30 CEST, 2026-09-16 23:30 UTC) — the same-model
> pair is measured (record §GLM53FLASH-LADDER).** GLM-5.3-Flash under the
> owner's own llama.cpp deployment (UD-IQ4_XS, 18 of the layers on the three
> cards, the rest and every expert in RAM, ROCm 7.14 container) beside
> Colibri's GLM-5.3 (int4-gs64, attention on dev0, 1 695-expert tiers on
> dev2/dev3, the rest in RAM), two runs each on the H2 ladder, A not
> interleaved with GF (same day, separate chains): decode at 1.3k 4.79 vs
> 2.85 tok/s (Colibri ahead, −40 %); at 18k 2.27 vs 2.80 (llama.cpp ahead,
> +23 %); ladder-turn TTFT at 18k 2 561 s vs 108 s (llama.cpp 24× faster
> prefill); cold 19k prefill 213 s on llama.cpp (Colibri: not run, projected
> hours); follow-up TTFT at 18.4k 6.9 s vs 7.1/5.6 s. **Both offload designs
> decode under 3 tok/s at 18k on this model; they differ by 24× on prefill.**
> With §PCIE-STREAM (61 GB/s into three cards) and §GPTOSS-3CARD (75 tok/s
> resident; 28 with half the experts in RAM; 17.6 with all of them), the
> regime map for the Franken-engine on this box is now measured end to end:
> decode at depth is bytes-from-RAM per token divided by the path's rate, and
> only residency or a faster path moves it; prefill is where Colibri's
> current design loses an order of magnitude and llama.cpp's batched
> CPU-expert path does not. What to build from these numbers is the next
> revision's subject, written with the owner, not by a probe.
>
> **Rev 11 (2026-09-17 00:45 CEST, 2026-09-16 22:45 UTC) — the two probes are
> measured; the reuse rule is withdrawn.** Record §PCIE-STREAM: host→GPU one
> card 28.0 GB/s; dev0+dev3 share an upstream link (35.9 GB/s aggregate, 18
> each); three cards concurrently 61.4–62.0 GB/s; CPU DRAM read 91.6 GB/s.
> X6's one-card 25 GB/s derivation is replaced: the three-card stream is 2.8×
> the CPU int4 path's byte rate. Record §GPTOSS-3CARD (gpt-oss-120b MXFP4,
> 63 GB, llama.cpp, the test vehicle — not a daily-driver candidate, per the
> owner): fully resident across the three cards, decode 75 tok/s at 18k on
> Vulkan and 76 on HIP (decode equal within 1 %, HIP prefill 1.7× faster,
> ladder TTFT 4.3 s vs 7.5 s at 18k); experts of 18/36 layers on the CPU:
> 28 tok/s, TTFT 66 s at 18k; all experts on the CPU: 17.6 tok/s, TTFT 127 s.
> Beside Colibri's GLM-5.3 at the same turn (different model): 2.27 tok/s,
> TTFT 2 561 s. **§2.4 Q2's r = T_lad/T_cold rule is withdrawn:** T_lad at
> "16k" is a ladder step that prefills ~9k new tokens, so a reusing server
> reads r ≈ 0.5, which is what every arm produced; the follow-up TTFT at
> depth is the reuse number, and by it llama.cpp (0.3–0.7 s at 18.5k on the
> resident arms) and hipFire (0.44 s at 85k, §FRANKEN-H2) both reuse. **In
> flight:** the owner's own llama.cpp GLM-5.3-Flash deployment (UD-IQ4_XS,
> 18 GPU layers, ROCm 7.14 container) on the same ladder, two runs, beside
> Colibri's A1/A2 — the same-model answer to "under 2 tok/s at realistic
> depth" (record §GLM53FLASH-LADDER when done). Parked: H2b, H4, D-3, L1.
>
> **Rev 10 (2026-09-16 evening) — H2 answered, H2c closed, and the owner's
> redirect.** H2 ran in full (record §FRANKEN-H2): hipFire on Qwen3.6-35B-A3B
> decodes 27× (turn 1) to 39× (18k) faster than GLM-5.3 on this box, with MTP
> on; Q2 (reuse) NO VERDICT at every depth (r 0.45–0.69); Q3 measured W_A(18.6k)
> = 6.87 s against the 17 s projection; Q4 ≈ 39×. H2c: hipEngine v0.5.0 cannot
> serve the 35B Q4_K_M on a 24 GB card (OOM sizing its KV pool after the weights
> take 21.2 GiB; its 35B rows are W7900-48 GB), closed. X3 step 0: the A-arm
> `[OPTIME]` table does not exist and cannot in engine mode (destructor vs
> SIGTERM), UNDECIDED, 1 h 45 m window or the stdin-close fix. D-3 is written
> (`perf/franken-d3`), not built or run. H4's script (`perf/franken-h4`) is not
> usable as written. H2b is written (`perf/franken-h2b`), not run.
> **The owner then said the plan measures the wrong thing:** the goal is a model
> larger than one card, spread over the three GPUs with experts or KV offloaded
> to RAM, at usable speed at realistic depth; every engine he has tried does one
> of those well and the others not (hence "Franken"). Two probes replace the
> single-card engine comparison, both in flight tonight: **PCIE-STREAM** (aggregate
> host→GPU bandwidth with three cards pulling concurrently vs one, pinned and
> pageable, plus CPU DRAM read — the measured replacement for X6's one-card
> arithmetic) and **GPTOSS-3CARD** (gpt-oss-120b MXFP4, 63.4 GB, 36 layers,
> 128 experts top-4, fully resident across the three cards on llama.cpp Vulkan
> and, if the container runs, HIP; then the same model with 18/36 and 36/36
> layers' experts on the CPU — the offload cost curve on a GPU engine, on the H2
> ladder so the rows sit beside GLM's 2.27 tok/s at 18k). The rig already carries
> the owner's own llama.cpp deployments (GLM-5.3-Flash UD-IQ4_XS at ngl 18 with
> CPU offload, Qwen3.8-Flash-Next 125B) in a ROCm 7.14 container; those are the
> baselines the redirect is about, and they have not been measured on this ladder
> either. H2b, H4, D-3, L1 are parked until those two results are in.
>
> **Rev 2 (2026-09-16).** Rev 1's §0 said no pair of engines could run the same
> model in the same placement on both backends. That was reasoned from what is
> on the rig's NVMe, and the owner rightly rejected it: the model set is not what
> is downloaded. Qwen3.6-35B-A3B runs on hipFire (eight registry SKUs, verified
> at its README: `qwen3.6:35b-a3b`, `-mq2`, `-mq3p`, `-mq4p`, `-mfp4`, `-mq4r`,
> `-mq5`, `-mq6`; MQ4R is its validated RDNA3 route) and on this fork
> (`c/qwen36.c`, `c/tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B`, the
> published int4-gs64 container). What stands between them is **one code gap on
> this fork** — `qwen36`'s VRAM tier is CUDA-only — and that gap is a port of a
> pattern this tree already has twice. So the same-weights GPU pair does not
> exist *today* and does exist *after item V1*; §0, §2.6 and §3 are rewritten
> around that, H2 stays first because it needs no code, and a second claimed
> gap (the converter's "int4 path is WIP") is recorded in §1 as stale.

> **Rev 9 (2026-09-16, day 1, night). P13 is closed as a harness bug, and Rev 4's
> "four gates compared garbage with garbage" is WITHDRAWN.** The bisect found no
> culprit (12 runs IDENTICAL, record §P13); the invocation diff found the cause:
> every X2 run pointed `COLI_VK_SHADERS` at its scratch clone, where an earlier
> detour build at `be95eb5` had left a stale `kda_step.spv` whose `conv_channel`
> takes two parameters while the current engine (since P5.2, `75ec4bb`) passes
> three — a host/shader ABI mismatch that only the GPU-KDA path exercises.
> Interleaved X,P,P,X on the pristine binary is deterministic: stale shaders →
> degenerate, own shaders → coherent. `start_glm53.sh` and every gate use a
> binary's own shaders; nothing served or measured was affected. Standing rule
> from it: gitignored build artefacts survive `git checkout` — a chain that
> builds another commit must do so in its own tree -- a `git worktree`, removed when the item lands, NOT a clone (corrected 2026-09-20: this line said "clone", agents took it literally and left 37 of them on the rig; `p13_chain.sh` happens to use one),
> and `COLI_VK_SHADERS` always names the shaders built with the binary. The
> record's §X2 G12 row is therefore an artefact, not a G12 finding.

> **Rev 8 (2026-09-16, day 1, evening). H2 is armed: the full A,B,B,A + C chain
> starts 2026-09-16 21:00 UTC** (`franken_full_launch.sh`, pid 2384291 on the
> rig, retries a held lock every 5 min until 23:00 UTC; the gateway is down for
> the chain, ~4 h; `accept_live.sh` closes it). The B arm's own smoke passed
> (`fk09160857`): hipFire on dev3, thinking OFF through its documented
> `chat_template_kwargs.enable_thinking=false` — the fix for the "open think
> span" error, not a workaround — TTFT 0.39 / 0.52 / 0.18 s at 409 / 833 / 875
> tokens, decode 88 / 100 / 124 tok/s, cold sweep 1 024 → 0.66 s, 2 048 →
> 1.06 s; a fresh process (B2) reproduced within 3 %. **One documented
> asymmetry:** GLM-5.3's template has no thinking-off form (its
> `enable_thinking=false` is "reasoning effort low"), so the A arm still thinks
> and the B arm does not; both arms count the first delta of any kind as TTFT.
> A real chain defect found by the smoke: VRAM after hipFire exits drains over
> ~15 s, so the free-VRAM assert now retries for 60 s instead of failing the
> arm. After the chain: H3 (table, Haiku), H4/X4 (audit page, Haiku), then
> H2b (same weights, same card: `qwen36-vk` at 21.6 tok/s vs hipFire; Haiku
> runs) and D-3 (Opus). M0c (record §FRANKEN-M0c): under the venv ROCm 10,
> RCCL runs over device P2P; 4 KB all-reduce 25.1–25.7 µs, under the 33 µs
> bar — TP3's precondition is met, its case is thin (3.4 vs 4 ms at perfect
> scaling), M1 stands.

> **Rev 7 (2026-09-16, day 1, V1). V1 is BUILT and GATED — steps 1 and 2 both pass
> (record §V1, branch `perf/v1-qwen36-vk-tier`), so the same-weights GPU pair
> §0 says H2b needs now exists.** `make -C c qwen36-vk VK=1` is a Vulkan expert
> tier for `qwen36` behind the existing `qt_*` contract: all **10 240** experts
> (40 × 256, int4-gs64) resident on dev3 in **18.12 GB**, filled in **6.9 s**,
> VRAM hit rate **100 %**, `Q36_VULKAN=1` to turn it on and bit-identical to
> the CPU engine with it off. Per-position KL against the tier-off arm is
> **−1.2e-10 mean / 100 % top-1 / cosine 1.000000000** over the 625-position
> packet — seven to nine orders below the bar §4's X2 rows set. In arm C's own
> harness, A,B,B,A: decode **14.58 → 21.64 tok/s (+45.6 % conservative)**, TTFT
> **33.45 → 10.67 s (−67.8 % conservative)**, with the tier-off arms
> reproducing arm C's floor to within 1–2 %. Two corrections this produced:
> **(a)** V1-STEP0 §(c)/(d)/(e) had the int4 nibble convention backwards — both
> backends take offset binary at their upload API and `qwen36`'s container is
> two's-complement, so the `stage()` XOR is KEPT, not dropped (proved
> exhaustively, `c/tests/test_qwen36_vk_nibble.c`; the document is corrected in
> place). **(b)** `dev_alloc_footprint`'s cudaMalloc granularity curve does NOT
> transfer to RADV — payload accounting is within **0.8 %** of what the driver
> reports, against CUDA's 22–28 %. §3's branch 3 is now measurable rather than
> projected, and what §4's X-items should target on this engine is the
> remaining ~46 ms/token, which is still entirely CPU: V1 moved the routed
> experts only, and the trunk (DeltaNet, attention, dense, LM head, shared
> expert) is untouched. **H2b is unblocked.**
> **Rev 6 (2026-09-16, day 1, afternoon). H0 PASSES; Q1 says proceed; the
> blocker was the box's ROCm, and the fix is user-local.** A ROCm 10.0.0
> installed as AMD's TheRock pip wheels in `~/venvs/rocm` (no sudo, nothing
> outside `$HOME`, the system 6.2.0 untouched; record §FRANKEN-H0d for the
> exact index and packages) runs hipFire v0.3.1 rebuilt against it end to end
> on dev3: throwaway ttft 248.5 ms / **163.8 tok/s**, the ladder question
> 238.7 ms / 147.5 tok/s, both streamed repeats `finish_reason=stop` at
> 248.3 / 248.4 ms — **Q1: D_B(≈0) = 163.8 ≥ 150, proceed** (0.65 × the
> published 253.3; the DPM level was recorded). Identical ttft on the repeat is
> the first evidence for Q2 = no reuse; the cold sweep decides. One trap for
> the chain: the model is a reasoning model and hipFire errors on "open think
> span at end of generation" when `max_tokens` lands inside `<think>`; the H2
> chain handles it symmetrically and documents the choice. The same venv makes
> `hipIpcGetMemHandle` succeed on all three cards (it failed per device under
> 6.2), so M0's device-P2P number is being measured as M0c. **M0b** (host-staged,
> `NCCL_P2P_DISABLE=1`): 8 B 31 µs, 4 KB 34.5 µs, 64 KB 61 µs; the 35B's hidden
> state is 2048 × fp16 = 4 KB, so 80 × 34.5 = 2.8 ms + 1.33 > 4 ms — TP3 decode
> loses on that path even with perfect scaling. **Also in the box's favour:**
> Colibri's engines link Vulkan only and ROCm is not on the loader path, so
> none of this can touch the daily driver. H2 runs tonight (21:00 UTC), the
> full A,B,B,A + C.

> **Rev 5 (2026-09-16, day 1). H0 is closed as O3 on this box as it stands;
> M0 has its answer; the H-track waits on one owner decision.** After
> `rocm-device-libs` was installed, two more mismatches surfaced and were
> lifted user-locally (record §FRANKEN-H0: this ROCm 6.2.0 is a 24.04 build on
> Ubuntu 26.04 — `CPLUS_INCLUDE_PATH` to the GCC 15 headers, and a
> `libxml2.so.2` alias for `ld.lld`). hipFire then compiles and caches its
> kernels but every request fails inside its own forward ("bench_decode
> forward failed", no HIP error exposed), unchanged across `--kv-backend
> vmm/contiguous` and prewarm on/off. hipFire documents "ROCm 6 or newer" for
> this card and took its own 7900 XTX numbers on 6.4.3; 6.2.0 is simply
> untested by them. **Owner decision:** Ubuntu 26.04's own archive ships ROCm
> 7.1 (`apt install rocm`, built for this release, so neither mismatch above
> applies); with it, `h0_chain.sh` reruns unchanged and, on a pass, the full
> H2 chain launches that night. Without it the track ends at O3 and Vulkan
> stays, as §3 says. **M0 (record §FRANKEN-M0):** rccl-tests builds and RCCL
> sees all three cards, but P2P setup fails (`hipIpcGetMemHandle: invalid
> argument`) on both `-g 3` and `-g 2` — no collective runs; TP/EP over RCCL
> is off the table here (§5's first outcome); the kernel line lacks
> `iommu=pt`, an owner-side thing to test, not a session's. hipFire's MTP and
> Redline knobs are recorded for X5 (`HIPFIRE_MTP_MODE`, `HIPFIRE_MTP_K`,
> `HIPFIRE_REPLAY_BACKEND`).

> **Rev 4 (2026-09-16, execution night 1, closing).** Two results that change
> §3 and §4, both measured (record §X1, §X2):
> 1. **X1 is rejected.** Retained command buffers are bit-identical on both
>    engines and worth **2.8 µs of a 25 µs submit** (`VK_PROF`: desc+record
>    2.5–3.1 µs, submit 25–27 µs, wait 331–348 µs per call; ceiling 97 × 2.8 µs
>    = −0.27 ms/token on qwen38-vk, ≈ −0.12 on glm53, against the projected
>    −3 to −8). A,B,B,A: NO VERDICT on every rotating/cold pair; warm-identical
>    qwen38-vk +0.2 % conservative. The branch `perf/x1-retained-cmdbuf` stays
>    unmerged; the record keeps the numbers.
> 2. **The Branch 3 ceiling (≤ 28 tok/s) is withdrawn.** Its derivation took
>    Q7's 0.30 ms per-submit *gap* for driver overhead; the profile says the
>    gap is the engine's own CPU work between submits, and the driver's share
>    is ~25 µs per submit — ~2.8 ms/token for ~120 submits. The lever on a
>    resident path is fewer submits, not cheaper recording. **V1 step 2 is not
>    gated on X1.** X1's own falsifier fired at 1.00×.
> 3. **X2 landed** (`GLM53_LOGIT_DUMP_ALL`, `kl_compare.py`, `gate_kl`;
>    G14/G15/clamp reproduce the record's order: KL 0.030 / 0.251 / 0.028) and
>    found something bigger than its gate: on the pristine serving binary the
>    **CLI/`--prompt` path under `COLI_KDA_GPU=2` generates garbage on the
>    564-token packet** (teacher_forcing dominated by token 154822; decoded text
>    "# 3.1.1.1…"), while the same binary at `=0`, and the served 4-slot path at
>    `=2`, are coherent; an 11-token prompt is fine. Gates whose teacher_forcing
>    oracle came from a CLI run at `=2` (p5, p5b, cancel, devmerge's memfloor)
>    compared garbage with garbage; `prefill_gate.sh` ran its oracle at `=0`
>    and is sound. Bisect is the next item on GLM (prefill roadmap rev 28), not
>    part of this plan. X1's glm53 CLI oracle is void for the same reason; its
>    serving-path evidence is the 4-slot `tworeq` text identity.
> Cost note: the night used one Opus agent (X1), four Sonnet, one Haiku, and
> hit the session limit once; the H-track waits on the owner's package install.

> **Rev 3 (2026-09-16, execution night 1).** Three corrections from running
> the plan, all measured on the rig, none changing the decision tree:
> 1. **H0 and M0 are blocked, not failed.** hipFire (upstream `warpfront/hipfire`
>    v0.3.1) builds, pulls the MQ4R SKU and serves its API on this box, but it
>    JIT-compiles its kernels at request time through ROCm's clang, and this
>    ROCm 6.2.0 install has no `rocm-device-libs` package; rccl-tests fails on
>    the identical error. The triton-bundled bitcode on the box lacks the
>    `oclc_*` control set and clang refuses it. Owner action:
>    `sudo apt install rocm-device-libs` (candidate 1.0.0.60200-66~24.04),
>    then `h0_chain.sh` reruns unchanged. Record: §FRANKEN-H0, §FRANKEN-M0.
> 2. **The H2 validity rule "any A row with majflt > 0 invalidates the arm"
>    was wrong** and is withdrawn: `c790851` had already measured that a
>    loaded `glm53` (~89 GB anon) beside the 183 GiB model on a 247 GiB box
>    caps residency near 92 % and turn 1 always faults (reference ladder
>    `ctx09152003`: majflt 21 663 / 6 611 / 4 521 / 4 176 on turns 1–4 — that
>    IS the serving regime). `franken_chain.sh` now invalidates an arm only on
>    a residency floor breach (< 90 %), prints majflt per row, and flags a
>    turn `MAJFLT-HIGH` above 2× the reference; H3 annotates such cells rather
>    than dropping them.
> 3. **Arm C has its first number** (smoke, `fk09152356`, CPU-only `qwen36`
>    on the gs64 container, all experts resident, 8 threads): 512-token turn
>    TTFT 32.85 s, decode **14.84 tok/s**; follow-up 20.67 s / 14.38 tok/s.
>    That is the floor branch 3 (V1) must beat, and it is already ~4–5× GLM's
>    decode at the same depth on the CPU alone.
> Also found and fixed in passing: every chain ending in `accept_live.sh` was
> truncating its own log (`tee /dev/stderr` under `nohup > log 2>&1`;
> `5062ff4`). H1 landed (`perf/franken-h1`); V1 step 0 landed
> (`tools/hot-expert/V1-STEP0-2026-09-16.md`: the CUDA tier takes gs64 group
> scales, no new shader needed, one nibble-encoding difference to drop).

## 0. Decision in one paragraph

**[2026-09-22: superseded.** The Franken-engine is a NEW engine assembled from the best-measuring parts of Colibri, llama.cpp, hipFire and hipEngine for this rig (plan rev 45); the design and the live program (M0-M5, L0-L5) are `tools/hot-expert/FRANKEN-ENGINE-DESIGN-2026-09-22.md`. F3/Qwen3.6 is closed (rev 38). What follows is kept as history.**] The plan body below (sections 0-8) is the analysis of 2026-09-15 and frames the question as "Colibri vs hipFire, which engine serves which model"; that framing is withdrawn.

The first head-to-head (§2, H2) cannot separate "HIP beats Vulkan" from "a
3B-active model in 960 GB/s VRAM beats a 180 GB model whose misses come from
host DRAM", because **today** no engine here runs the same model in the same
placement on both backends: Colibri's Qwen3.6-35B-A3B engine has a VRAM tier
for CUDA only, so on this box it is a CPU engine. H2 is therefore not designed
to answer the backend question. It answers the **product** question — what the
owner's turn costs at depth in each world, and whether the moat (P7/P9 prefix
reuse) survives — with the one same-weights control possible today (§2.3), and
it needs no code. **The same-weights GPU pair exists after one port, V1 (§2.6):
a Vulkan expert tier for `qwen36`**, the pattern `glm53` and `qwen38-vk` already
use. V1 is worth building on its own account — it is the "second, VRAM-resident
lane" inside Colibri's serving stack, with P7, P9 and the KV slots intact — and
once it exists, H2b (§2.6) is the engine-vs-engine number on the same weights in
the same placement. Even then the backend question proper is answered by the
same-op microbenchmark (§3, D-3), because an engine comparison still folds in
schedulers, formats and submit models; and the profile already says the
backend's ceiling on the daily driver: **the GLM-5.3 token is CPU/DRAM-bound, not
GPU-kernel-bound** (G3: 69 % of the token on one core; §RP1-CORRECTION: CPU int4
experts 40.3 ms/token at ~57 % memory-bound; GPU expert groups were 22 ms of a
373 ms token at G3 and have been overlapped with CPU work since G9). A faster
GPU kernel moves the GLM token by single-digit percent. **No result of the
head-to-head justifies rewriting Colibri's backend.** What it can justify is a
second, VRAM-resident lane on this box, and a short, ranked list of grafts
(§4) — the first of which (retained command buffers) helps every Vulkan
engine here today and is what makes any future Colibri-native resident path
viable at all.

Expected outcome, stated so a surprise reads as one: hipFire runs, decodes
the 35B at ≥ 10× GLM-5.3 at every depth (arithmetic, not a close race), and
the decisive unknown is whether it reuses prefixes across turns. If it does
not, its per-turn wait at 18k is a full re-prefill and Colibri's incremental
turn may still win the number the owner actually waits for.

## 1. What is established (do not re-derive)

| fact | value | kind | source |
|---|---|---|---|
| GLM-5.3 prefill rate vs depth on this rig | ms/token ≈ 113.9 + 0.01258·d (5 points, 643..13 628, fit within ~1.5 %) | measured | tonight's ladder |
| cold prefill of N tokens, integrated | 113.9·N + 0.00629·N² ms → 2k **260 s**, 4k **572 s**, 8k **1 355 s**, 16k **3 555 s**, 18 055 **4 107 s (68 min)**, 64Ki **34 480 s (9.6 h)** | projected from the fit | derivation: ∫₀ᴺ(113.9+0.01258·d)dd |
| GLM-5.3 decode vs depth | 3.71 @1 287, 4.06 @2 559, 3.59 @4 920, 2.99 @9 171, **2.23 @18 055** tok/s (1.82× fall from peak) | measured | tonight's ladder |
| GLM-5.3 serving numbers, decode track | 3.03 / 2.98 rotating, 3.44 / 3.45 warm-identical, 1.86 / 1.90 cold | measured | PREFILL-ROADMAP rev 26 |
| GLM-5.3 fresh-process token | 134.90 ms knob-on (`COLI_KDA_GPU=2`), 156.77 knob-off | measured | ROADMAP Track G header |
| GLM-5.3 new chat / follow-up through the gateway | ~20 s (P7 restores the ~4 000-token tool block) / ~2 s; a changed tool set pays ~10 min cold | measured | CLAUDE.md |
| GLM-5.3 routed expert: size and tier share | 171 GB resident / 12 096 mappable ≈ **14.1 MB per expert**; tier serves ~79 % of routed calls, ~21 % from the CPU | measured | briefing, CLAUDE.md `[MAP]` line, G15 note |
| CPU int4 expert kernel, isolated, 8 threads | 21.95 GB/s | measured | §G11 |
| qwen38-vk serving numbers | warm-identical 7.23, rotating 5.3–5.4 | measured | Q7 row |
| qwen38-vk submit overhead | `vk-issue` 13.8 + `vk-take` 4.8 ms/token; per-submit gaps 0.30 / 1.50 ms; "a 2.7 ms submit ramps its own clocks" | measured | Q9 spec table, Q7 row |
| `backend_vulkan.c` re-records its command buffers on every submit | `vkResetCommandBuffer`/`vkBeginCommandBuffer` at 1007, 1113, 1177, 1268 (56 reset/begin/submit sites) | read | `c/backend_vulkan.c` |
| Colibri's Qwen3.6-35B-A3B engine | `c/qwen36.c`: 40 layers (10 × [3 DeltaNet + 1 attention]), 256 experts top-8 + 1 shared; dense int8 in RAM, experts LRU-cached in RAM; **CPU-only by default, the VRAM tier is CUDA-only** (`qwen36_tier.h` is compiled under `COLI_CUDA` only; no Vulkan symbol in the file; the tier API is ~20 `qt_*` entry points, 28 call sites in `qwen36.c`); int4-gs64 container ~20 GB, ~30 GB RAM | read | `docs/qwen36.md`, `c/qwen36_tier.h`, `c/Makefile` |
| The engine DOES read int4 containers; the converter's "int4 path is WIP" is stale | `qwen36.c` detects packed int4 by on-disk size (lines 1445–1529), unpacks to int8 for the CPU path and **keeps the packed nibbles (`g4/u4/d4`) for a GPU tier**; `qt_init` takes `expert_is_int4`; `docs/qwen36.md` recommends the int4-gs64 container over per-row int4 (GLM's #455 think-loops). So a Vulkan tier holds ~20 GB of packed int4 experts — one card — not ~35 GB of int8. **Open for V1 step 0:** whether the CUDA tier's kernels take gs64 group scales or only per-row `gs/us/ds`; the answer decides which container V1 serves | read | `c/tools/convert_qwen36.py` line 113 vs `c/qwen36.c`, `c/qwen36_tier.h` |
| Also on the rig's disk, and unusable by Colibri | `Qwen3.8-27B-UD-Q5_K_M.gguf` (19.8 GB), a UD-IQ4_XS GGUF of Qwen3.8-Flash-Next, `DeepSeek-V4-Flash-0731-UD-IQ2_M` (85 GB, over the 72 GB of aggregate VRAM). Colibri maps safetensors (`st_map_shard_range`), not GGUF; none of these is a candidate on either side | reported by the coordinator, formats checked against `c/st.h` | — |
| That engine's only numbers | 9.2–11.3 tok/s with a CUDA tier on 8 GB cards; **0.35 tok/s CPU-only before the tier** — on a Threadripper 3945WX, not this box | measured elsewhere | `docs/qwen36-cuda-tier.md` |
| Vulkan vs HIP on the same card, Colibri's own | on an RX 9070 (RDNA4) the Vulkan backend is faster than Colibri's ROCm/HIP backend | measured elsewhere | `docs/vulkan.md` |
| Vulkan vs HIP on the same card, hipEngine's own | p4096: hipEngine 290.6 / 18.69 vs Vulkan (halo box) 420.95 / 24.55 vs llama.cpp HIP 395.02 / 19.63 prefill / decode; the gap attributed to "dataflow submission efficiency" | published | briefing |
| hipFire single-XTX figure | 253.3 tok/s TG128 (empty context); its own multi-turn 191 average, **160 at ~18k**; MQ4R 35B-A3B 18.7 GB, ~22 GB VRAM | published | briefing |
| MTP on host-resident models here | Q9 step 0: V*(4)/V*(1) = 2.668 ≥ 2.4, kill fired; hipEngine 0.955× AR | measured / published | Q9 row, briefing |
| ROCm on the box | 6.2.0 at `/opt/rocm-6.2.0`, not on PATH, never used; `librccl.so` present; rccl-tests not built; all three GPUs PCIe 4.0 x16 CPU-direct, AtomicOpsCap 32/64+, ReqEn+, Routing+ on the bridges | verified tonight | briefing |
| Colibri's tiers fill every card | G6: cap 2200 stops at 1752 experts on "25.0 of 25.7 GB used, 1.0 reserve"; serving runs 1695/1695 on dev2/dev3 and 1248 + KDA pool + dense on dev0 | measured | G6 row, CLAUDE.md |
| dev0's identity | `0000:83:00.0` = `/sys/class/drm/card1`; card numbers are scrambled | measured | record §Q13 |
| run-to-run spread on this box | 3–5 %; an uninterleaved pair read +20 % / +11 % on true +3.9 % / 0 | measured | Q9 spec, `gate_lib.sh` |

Two things the table makes unavoidable:

1. **H2, as it can be run today, is a model-class comparison.** GLM-5.3-Flash
   int4-g64 is 183 GB; hipFire has no GLM. Qwen3.8 (Colibri's other engine) is
   173 GiB of FP8 with 512 routed experts per layer — an MQ4R of it is far
   larger than one card and larger than all three (72 GB), so hipFire cannot
   serve the model class this box serves today at all. The only same-weights
   run available *today* is Colibri's `qwen36` on the CPU, which measures an
   8-core CPU against a 7900 XTX, not Vulkan against HIP. **The same-weights
   GPU pair is one port away (V1, §2.6), not unavailable** — Rev 1 got that
   wrong.
2. **A backend port cannot be paid for by the GLM profile.** Even a 2× GPU
   expert kernel is worth ≤ ~10 ms of a 135 ms token (≤ 7 %), and G14 already
   showed the CPU half of that bucket is exhausted without changing numerics.

## 2. The head-to-head (items H0–H4 today; V1 and H2b after the port)

Two phases. **H2** runs on what exists — no engine code, ~4 h of rig time —
and answers the product question. **H2b** runs after V1 and is the
engine-vs-engine number on the same weights in the same placement. H2 is not
made to wait for V1: its decisive unknown (does hipFire reuse prefixes, Q2)
is independent of anything Colibri builds, and its GLM arm is the daily
driver's own curve.

### 2.1 The pair, and what is honestly not comparable

**Arm A — Colibri, the serving configuration.** `glm53` from the binary in
service, GLM-5.3-Flash-colibri-int4-g64, the gateway's env exactly as
`ttft_serve.engine_env` sets it (8 pinned threads, three devices,
`COLI_VK_EXPERTS2/3=1695`, `COLI_KDA_GPU=2`, a *copy* of the histogram),
`--kv-slots 4`, **`GLM53_PREFIX_CKPT=0` with a private `COLI_CKPT_DIR`** (as
every gate on this track; a restored checkpoint would report a prefill that
never happened), `GLM53_MAXT=32768`. Driven in engine mode by
`context_ladder.py`, the harness that produced tonight's numbers.

**Arm B — hipFire, one card.** Qwen3.6-35B-A3B MQ4R (18.7 GB) on **one** gfx1100
— the card that is Colibri's **dev3** (an expert-only device), so that if a
two-lane configuration is ever built GLM keeps dev0 (dense + KDA) and dev2.
H0 records the PCI-BDF → HIP-index map. Driven over HTTP by the same ladder.

**Arm C — control, same weights, Colibri CPU-only.** `qwen36` on the
int4-gs64 container of the same Qwen3.6-35B-A3B, all experts RAM-resident
(`cache/layer` = 256), 8 pinned threads, served by `coli serve` on a side
port and driven by the same HTTP driver as arm B. **This is not an engine
comparison** — it is (i) the floor any Colibri-native resident path (§3, branch
3) has to beat, (ii) the only same-weights sanity check on hipFire's output,
and (iii) what the owner would have on this model if hipFire fails H0. Bounded
to `--sizes 512,2048`, one repeat, 30 min wall, run last.

| held constant | how |
|---|---|
| the user text | `context_ladder.py`'s deterministic corpus slices from `ROME-3x7900XTX-2026-09-04.md` (624 KB; a 64Ki ladder needs ~236 KB), same `--steps`, same question suffix, byte-identical on every arm |
| the turn structure | one growing conversation; the reply fed back verbatim (engine mode: with P8's pin; HTTP mode: as received) |
| generation | `--gen 128` on every arm (32 is ~0.2 s at 160 tok/s and would measure the SSE chunking, not the engine); temperature 0 |
| the clock and the formula | one harness; decode = (n−1)/(t_done − t_first) on every side; TTFT = first content delta |
| the box's state | GPU DPM level read from sysfs and recorded before every arm, never changed; no other engine (`pgrep -x glm53`, `pgrep -x qwen38`, `pgrep -x qwen38-vk`, `pgrep -f "hipf[i]re"` all empty before each arm); VRAM asserted free before every arm (sysfs `mem_info_vram_used` < 1 GB on all three cards) |
| residency | arm A: fincore ≥ 90 % on the GLM shards before every turn and `majflt` per turn recorded (must stay 0); arm B/C: fincore on their own model files, same floor |
| order | **A, B, B, A**, then C — the GLM arm brackets the pair; each B arm is a fresh server process so B2 cannot inherit B1's KV or prefix cache |

| not comparable, said plainly | why it stays in the report anyway |
|---|---|
| the model (GLM-5.3-Flash vs Qwen3.6-35B-A3B) | it is the decision: the owner is choosing a model class, and no throughput number substitutes for reading the answers (§2.5) |
| the quantisation (int4-g64 / MQ4R / int4-gs64: three formats, none bit-comparable) | arm C vs arm B agreement is reported as a sanity, not an oracle |
| the placement (host + 3 tiers vs one resident card) | this *is* the class difference |
| the backend (RADV Vulkan vs ROCm HIP) | confounded with every row above; **the head-to-head says nothing about it**; D-3 in §3 does |
| depth in tokens (two tokenizers) | depth is reported in each engine's own tokens and in characters (identical by construction); comparison at matched *turn index*, with the token counts printed beside it |

### 2.2 Items

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| **H0** | **hipFire on this box at all.** Under the rig lock with the gateway stopped (a chain, not by hand — the watchdog): check `/dev/kfd` access for the user (`id -nG` includes `render`/`video`; if not, that is an owner action, `usermod`, and the item stops there and says so); `ROCM_PATH=/opt/rocm-6.2.0 rocminfo` lists three gfx1100; build hipFire (Rust toolchain user-local via rustup; `hipcc` from `/opt/rocm-6.2.0/bin`) — **read its README for its minimum ROCm first; if it needs > 6.2, stop and report: installing a second ROCm is a system change the owner decides, not this item**; download the Qwen3.6-35B-A3B MQ4R; serve it on dev3's HIP index on a side port; one request through its HTTP API. Record: the API shape (OpenAI-compatible `/v1/chat/completions` with SSE, or Ollama-style `/api/chat` NDJSON — the driver needs to know), whether the response carries `usage.prompt_tokens` / prompt-eval timings, whether it exposes **any prefix/KV cache across requests** and **any speculative/MTP knob** (both decide items below), its CPU thread setting, and the PCI-BDF → HIP-index map | ROCm 6.2 unused so far; RADV has been the only path | runs, or fails on a named cause | ½ day + a short lock window | Sonnet (+ owner if a group membership is missing) | a coherent 64-token greedy reply on the ladder's question, `usage` or an equivalent token count reported, VRAM released after exit (sysfs `mem_info_vram_used` back under 1 GB on that card) — **or a named blocker** |
| **H1** | **Extend `context_ladder.py` with `--url`** (reuse `ttft_serve.HttpDriver`, add its `--api-key`, `--model-id`, `--server-log`, `--tools` arguments), plus: residency over the files under `--snap` whatever their extension; the P8 reply pin only in engine mode (an unknown message field may be rejected by another server); in HTTP mode the REUSE-based abort is replaced by a loud `REUSE UNVERIFIED (http)` note and `reused=None` in the row — **the cold sweep decides reuse, not a guess**; `--cold-sweep 2048,4096,8192,16384 --sweep-offset-chars 300000` (fresh single-message prompts of those sizes from a disjoint corpus region, each its own conversation, HTTP mode only; **refused in engine mode with the arithmetic** — 2k+4k+8k+16k on GLM is 260+572+1355+3555 s = 96 min per pass); `side`/`arm` fields in every JSON row; decode from `usage.completion_tokens` when the stream reports it, else delta count, and the row says which. If H0 found an Ollama-style API, a second small driver class with the same `run()` contract. **Plus `context_compare`** (new, Python, under `tools/hot-expert/`): reads the jsonl of the four arms, pairs rows by turn index, prints per depth D_A×2, D_B×2, TTFT_inc×2 each, applies `gate_ab_verdict`'s rule (overlap → NO VERDICT; conservative worst-against-best when separated; refuse < 2 samples per arm) by sourcing `gate_lib.sh`, and prints the reuse ratio r (§2.4) per sweep depth | `HttpDriver` already reads SSE, `usage`, and the gateway log; the ladder hardcodes `EngineDriver` | a driver that is symmetric across arms | 1 day | Sonnet | (i) engine mode reproduces tonight's ladder turn 1–2 within the box's spread (374 tokens 45.4 s / 121.5 ms/token; turn 2 REUSE 381/731 at 110.8 ms/token — `eb4dd5b`), (ii) `--url http://127.0.0.1:8081 --steps 256,256 --gen 16` against the **live** gateway (read-only, no stop, prefill-snapshot style) shows the REUSE lines from `~/glm53_server.log` covering the previous prompt on turn 2 — the HTTP driver is proven on the engine whose reuse signal exists before it meets one whose does not |
| **H2** | **The chain, `franken_chain.sh`**, launched only through `run_chain.sh` (lock; refused while the current benchmark holds it): stop gateway → `wait_no_engine` → warm GLM shards, assert ≥ 90 % → **A1** ladder `--steps 1024,1024,2048,4096,8192 --gen 128 --followups 2` (cumulative user text 1/2/4/8/16 Ki; the engine's own count lands near tonight's 18 055 point) → stop engine, assert VRAM free → **B1** start hipFire, throwaway request, ladder with the same steps **plus** `--steps …,8192,8192,8192,8192,8192,8192` to 64Ki (cheap for it; the 64Ki rows are B-only and labelled so; Colibri's 64Ki is the 9.6 h projection, not run), then the cold sweep → stop hipFire, assert VRAM free → **B2** identical, fresh process → assert VRAM free, re-warm GLM, assert ≥ 90 % → **A2** identical to A1 → **C** (`coli serve` on `qwen36`, `--sizes 512,2048`, 30 min cap) → re-warm GLM → restart gateway on every exit path → `accept_live.sh` (the request *after* the measurement is part of the measurement) | every rule in CLAUDE.md "How a change is measured"; MEASURING.md's factor-of-two on cache state | rig ≈ 70 min (A1) + ~20 (B1) + ~20 (B2) + 70 (A2) + ≤ 30 (C) + ~20 warm/restart ≈ **4 h**; ~~schedule at night, tell the owner the gateway is down for it~~ [2026-09-22: run at once] | Sonnet writes, Haiku runs | the chain exits 0 with four jsonl files whose A rows all carry `majflt=0`, every arm's pre-checks logged (DPM level, VRAM free, residency), `accept_live.sh` PASS at the end; **any A row with majflt > 0 or residency < 90 % invalidates that arm and the chain says so instead of averaging it** |
| **H3** | **The table.** `context_compare` output, plus D_B at empty context from the throwaway (the transfer check, §2.4 Q1), plus arm C's rows, plus the reuse ratio, into the record as §FRANKEN-H2 with the raw jsonl paths | — | one table | ½ day | Haiku | every cell either a measured number with its two samples or `NO VERDICT` / `REFUSED` from `gate_lib.sh`; no cell computed by hand |
| **H4** | **The owner's read (§2.5)**, not a gate: the same 12 prompts through both lanes, tabulated side by side, `finish_reason` and tool-call outcome per row | ninfer's 225-response audit is the pattern: throughput is not usable output | a page the owner reads | ½ day | Haiku | all 12 rows present for both lanes with `finish_reason=stop` counts; no scoring |

### 2.3 Why the ladder and not TG128, and why not a 64Ki cold prefill

TG128 at empty context is where hipFire's 253.3 lives and where nobody's
conversation lives: Open WebUI's tool block alone is ~4 000 tokens, and tonight's
curve loses 1.82× between 2.5k and 18k on GLM (hipFire's own multi-turn figures
lose 1.58× from 253 to 160). The ladder gives decode and incremental TTFT at
every depth in one pass, on the shape Open WebUI sends, and the same corpus on
every arm. A cold 64Ki prefill on GLM is 9.6 h projected and would add nothing:
the rate fit already covers 643–13 628 within 1.5 % and the ladder's 16Ki step
lands on the 18k point measured tonight; the B arm runs to 64Ki because it can.

### 2.4 The four numbers the chain must produce, and their thresholds

Let D_X(d) be decode tok/s at depth d, W_X(d) the incremental TTFT of the
follow-up turn at full depth (the two `--followups`), T_B^cold(d) hipFire's
cold-sweep TTFT, and T_B^lad(d) its ladder TTFT at the same depth.

- **Q1 — do hipFire's numbers transfer to this box?** D_B(≈0) from the
  throwaway. `≥ 150` (0.6 × the published 253.3): proceed. `80–150`: read the
  DPM level recorded (§Q13 found dev0 idling low and `high` changing numbers);
  do not change it mid-chain; a second chain with the level pinned is a
  separate, labelled run. `< 80`: **stop; something on this box is wrong**
  (driver, power, card) and no decision is taken on that number.
- **Q2 — does hipFire reuse prefixes?** r = T_B^lad(16k) / T_B^cold(16k).
  `r ≤ 0.25`: reuse. `r ≥ 0.8`: no reuse — every turn re-prefills.
  In between: NO VERDICT from the ratio; the server's own counters (H0)
  decide or the cell says unknown.
- **Q3 — the owner's wait at depth.** W_B(18k) vs W_A(18k). Projected
  W_A(18k) for a ~50-token follow-up = 50 × (113.9 + 0.01258 × 18 055) ms
  = 50 × 341 ms ≈ **17 s** (the chain measures it; the projection is for the
  reader). Without reuse W_B(18k) = T_B^cold(18k).
- **Q4 — decode at depth.** D_B(18k) vs D_A(18k) = 2.23 measured. The
  published 160 would be 72×. Anything ≥ 10× is the same decision.

### 2.6 The port and the same-weights pair (V1, H2b)

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| **V1 — DONE 2026-09-16, GATE PASS (record §V1, `perf/v1-qwen36-vk-tier`)** | **A Vulkan expert tier for `qwen36`.** Port the tier `qwen38-vk` already has (heat-ranked preload from a histogram, per-device budget with a reserve, `COLI_VK_DEV2/3`, expert-group issue/take through `backend_vulkan.c`) behind `qwen36_tier.h`'s existing `qt_*` contract, so the engine's 28 call sites do not move; packed int4 experts (`g4/u4/d4`) in VRAM, ~20 GB on one card. **Step 0 (Sonnet, no rig):** read the CUDA tier's kernels for whether they take gs64 group scales; read `qwen38_core.h`'s `q38vk_*` path for the piece to copy; write the step list. **Step 1 (Opus):** tier off = the CPU engine, bit-identical. **Step 2 (Opus):** tier on. The model must first be fetched and converted (the published int4-gs64 container, or `convert_qwen36.py --gs 64` from the BF16 ~70 GB) — NVMe and page-cache work that waits for the lock, like everything else. **Sequenced after X1**, because Q7's measured gaps put a ≤ 28 tok/s ceiling on a 40-layer per-op-submit path (§3, branch 3), and a same-weights comparison taken under that ceiling would measure Colibri's submit model, not its backend | the pattern exists twice in the tree; `docs/qwen36-cuda-tier.md` measured the CUDA version at 9.2–11.3 tok/s on 8 GB cards, 83 % of experts resident on two cards — here all 256 × 40 fit on one | a Colibri-served VRAM-resident 35B with P7/P9/slots intact; tok/s **unknown** until X1 has a number (the ceiling above is the prior) | 1–3 weeks | Sonnet (step 0), Opus (1–2) | tier off: `teacher_forcing` and logits bit-identical to the CPU path; tier on: X2's KL bar against the CPU path, `tworeq.py` at 4 slots IDENTICAL, `[MAP]`/tier line asserts the residency it claims, decode A,B,B,A vs tier-off through `gate_ab_verdict` |
| **H2b** | **Same weights, same placement, two engines.** A′ = Colibri `qwen36` + V1 on int4-gs64, one card (dev3); B = hipFire on the same card, `-mq4r` **and** `-mq5` (the quant nearest a gs64 int4 by bits per weight; report both rows, do not average them). Same chain shape as H2 (A′,B,B,A′; fresh processes; VRAM and DPM asserted; ladder to 18k with `--gen 128`, cold sweep on both sides — affordable now on both). Colibri's arm keeps its REUSE-line contract; hipFire's keeps the sweep ratio | V1 | the first number that compares engines rather than model classes | ~2 h rig | Haiku runs | `context_compare` per depth through `gate_ab_verdict`; **plus** greedy-text agreement between A′ and B on the first 64 tokens of each turn reported as a count (a sanity on two different quantisations, not an oracle) |

Still not comparable in H2b, said plainly: the quantisation (int4-gs64 vs MQ4R
/ MQ5 — different formats, different error), and everything above the kernels
(scheduler, submit model, KV layout). H2b answers "which engine serves this
model better on this card"; only D-3 answers "which backend runs this op
faster". Both are asked; neither is asked of the other's number.

## 3. Decision tree (after H3), with what falsifies each branch

```
H0 fails on a named blocker ──────────────────────────────► O3: stop. Record the cause.
                                                             Vulkan stays. Nothing else changes.
H0 passes
 └─ Q1 < 80 ────────────────────────────────────────────────► O4: diagnose before deciding (driver /
 └─ Q1 ≥ 150 (or 80–150 with the DPM level explained)        DPM / card). No branch is taken on it.
     ├─ Q2 = no reuse AND W_B(18k) > W_A(18k) ───────────────► O2: the moat is real. hipFire wins decode
     │                                                          and loses the wait at depth. Do NOT
     │                                                          replace the lane. Go to branch 3 or wait
     │                                                          for hipFire to add prefix caching; re-run
     │                                                          H2 when it does.
     ├─ Q2 = reuse (or W_B(18k) ≤ W_A(18k)/2) AND Q4 ≥ 10× ──► O1: the fast lane is real. Two follow-ups:
     │                                                          L1 (two-lane cost on GLM) and the owner's
     │                                                          read (H4). Then branch 1 or 2, owner's call.
     └─ anything else (e.g. Q4 between 3× and 10×, W_B ≈ W_A) ► NO VERDICT on the product question.
                                                                Report the table. Fable arbitrates only
                                                                if the owner wants a call on partial data.
```

**Branch 1 — two lanes.** hipFire serves the 35B on dev3 as a second OpenAI
connection in Open WebUI (no gateway change; the daily driver on 8081 is
untouched); GLM-5.3 runs on **two** cards. Cost to GLM, **projected**: dev3
holds 1 695 of the 4 638 tier-resident experts (37 % of the tier); if hits
scaled with tier size the CPU share would go from ~21 % to ~21 + 0.37 × 79 ≈
50 % of routed calls, the CPU expert bucket from ~40 to ~95 ms/token, the
token from ~135 to ~190 ms — **about −29 % decode**. Heat ranking makes the
coldest third of the tier carry fewer hits than its share, so the bracket is
**−15 to −30 %**, and dev0's KDA pool and dense stream are untouched. Item
**L1** measures it: `tools/rome_bench.sh glm53 <name> COLI_VK_EXPERTS3=` with a
config name that says dev3 is skipped (an unset cap skips the device — CLAUDE.md;
verify the tier line says dev3 absent, or the row is invalid), interleaved
A,B,B,A against the three-card config, **and** `prefill_snapshot.sh` for the
TTFT side (both tracks, because the question is what the owner loses on GLM).
Haiku. Falsifier of the branch: L1 > −30 % or the owner rejects the 35B's
answers in H4.

**Branch 2 — replace.** The owner decides after H4 that the 35B is his daily
model; GLM-5.3 becomes on-demand. Then hipFire + Open WebUI is the stack and
the Colibri moat matters exactly as much as Q2 says: with reuse, P7/P9 are
redundant for that lane; without it, branch 2 is O2 and is not taken. This
branch is the owner's, not a measurement's; the plan only makes its cost
visible. Falsifier: O2, or H4.

**Branch 3 — the Colibri-native VRAM-resident lane: that is V1 (§2.6).** The
35B runs on Colibri with P7/P9/slots/gateway intact. Its ceiling before a line
is written, **projected from Q7's measured gaps**: 40 layers × ≥ 3 submits per
layer × ≥ 0.30 ms per gap ≥ **36 ms/token of submit gaps alone → ≤ 28 tok/s**
before any compute, against hipFire's published 160+. That is why graft X1
(retained command buffers) is a *precondition* of V1, not an optimisation of
it, and why V1's step 2 is not started before X1 has a measured number (V1's
step 0 needs no rig and can start now). Branch 3 is taken — V1 becomes the
serving lane rather than a measurement vehicle — when H2b shows
D_A′(18k) ≥ 0.6 × D_B(18k) and W_A′(18k) ≤ W_B(18k); below that, hipFire is
the better engine for this class on this card and V1 remains what makes H2b
and D-3 honest. Falsifier of the whole branch: X1 lands below 2× on the submit
buckets, in which case the ceiling stands and the lane is dead on arithmetic
before V1's step 2 is built.

> **Measured 2026-09-16 (record §X1): the falsifier fires, at 1.00×, and the
> ceiling's derivation needs correcting rather than the lane killing.** The
> 0.30 ms gap is not overhead a retained buffer can remove — §Q7 measured it as
> the engine's own CPU work between submits. Of that gap, the descriptor writes
> and the command re-record are **2.5–3.1 µs** (in-engine `VK_PROF` split),
> i.e. 0.2–0.9 %; the driver's `vkQueueSubmit` is **22–27 µs** and the round trip
> 331–348 µs. So the per-submit floor for a 40-layer, 3-submit-per-layer resident
> path is **120 × ~25 µs ≈ 2.8 ms/token of driver submit alone**, plus whatever
> CPU work sits in the gaps — and **X1 is not the lever that moves it. Fewer
> submits are** (the P2 / G12 / Q3 / Q7 pattern this fork already uses).
> V1's step 2 is therefore **not blocked on X1** and gets no relief from it.

**D-3 — the backend question, answered on its own terms.** One microbenchmark,
same card, same shape: hipFire's gfx11 MMQ int4 kernel vs `qmatmul_tile.spv`
at GLM's routed-expert GEMV shape (M = 1, and the tiled S-row case), each
warmed, each ≥ 5 repeats, interleaved. `tools/hot-expert/rome_vkbench.c` is the
Vulkan side's existing harness. **Migrating any Colibri op to HIP is justified
only if the HIP kernel is ≥ 1.5× on that op AND the op is ≥ 20 % of the token
it lives in.** On GLM the second condition is false by the profile (§0), so
the outcome can at most name a candidate for the *resident* path of branch 3.
Opus, 1 day, one short lock window. Falsifier of "the backend is not the
lever": ≥ 2× on the op — then X-HIP (a knob-gated HIP expert kernel) enters §4
at the bottom, priced by the op's share.

## 4. The graft list, by value per unit of risk

| rank | id | graft | source | what it replaces or adds in Colibri | expected | risk | tier | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | **X1 — DONE 2026-09-16, REJECTED on measurement (record §X1)** | **Retained command buffers for the per-token-invariant stream** — record once, re-submit per token; keep the MoE expert groups dynamic (their set changes every token). Built behind `COLI_VK_RETAIN_CB`, measured, **not merged** — `perf/x1-retained-cmdbuf` stays as the negative result and carries the `VK_PROF` per-submit phase split that killed it | hipFire "Redline" (record the kernel graph, retain invariant command state); CUDA-Graph analogue. Vulkan supports it natively: a command buffer recorded without `ONE_TIME_SUBMIT` is re-submittable | `backend_vulkan.c` resets and re-records on every submit (lines 1007–1009, 1113–1115, 1177–1179, 1268–1270). Targets: Q7's dense stream on qwen38-vk (`dn-proj`, `qsa-proj`, `lm-head`), G12's KDA path on glm53 | **projected**: a fraction of `vk-issue` 13.8 + `vk-take` 4.8 ms/token on qwen38-vk — the CPU-side re-record and validation, not the GPU wait; call it −3 to −8 ms/token (2–6 % of 138) and be pleased to be wrong upward; on glm53 smaller (its GPU stream is thinner). The larger value is strategic: it is the floor-remover for branch 3 | medium: shared file, both engines rebuild and re-measure; descriptor/buffer addresses must be stable across tokens | Opus | bit-identical (same kernels, same order) on `teacher_forcing`, `last_logits`, `tworeq.py` at 4 slots; `rome_bench.sh qwen38-vk` and `glm53` A,B,B,A through `gate_ab_verdict`; `[OPTIME] vk-issue` before/after in the commit body |
| 2 | **X2** | **KL oracle**: per-position logits over a fixed packet (450 rows, the hipEngine size), mean/max KL and top-1 agreement, as a third oracle beside greedy text and last-token cosine | hipEngine's gate methodology (it rejected a 5 % prefill win on this bar) | adds to `ttft_serve.compare_logits` / the `teacher_forcing` path a per-position dump (`GLM53_LOGIT_DUMP` today dumps the last token); a script that prints mean/max KL and top-1 % | a single scale on which G12 (cos 0.99992, text identical), G14's int8 (cos 0.98964, text changed) and G15 (cos 0.878) order themselves; the swiglu-clamp item (§G15's by-product) gets a number instead of "8 of 1232" | low | Sonnet | reproduces those three cases in the same order; refuses (via `gate_compare`) when either dump is empty |
| 3 | **X3 — step 0 RUN 2026-09-16, UNDECIDED and PENDING a ≈1 h 45 m window (record §X3 step 0): "no extra rig time" is WITHDRAWN — the A-arm engine log carries no `[OPTIME]` line and no engine-mode log ever can (the table is a `destructor`, the harness SIGTERMs the engine), so the split must come from a two-depth CLI `--greedy` run. X3 is NOT dead: the measured token grows +205 ms/token from 2.5k to 18.4k and the record's own post-G5 indexer slope predicts ~144 of it, but that fit was never taken above the 2 051 cap.** | **QSA block-sparse attention at depth for GLM's MLA** (pool 4 → 1 key, select 512 blocks; dense-identical below 2 052 tokens) — **profile first, then decide** | hipEngine QSA | GLM already has a DSA indexer with cached pooled keys (G5); this is a stricter selection at long context. Step 0: one 18k turn with `[OPTIME] mla split` on, from the A1 engine log — which bucket grows from 2.5k to 18k? | if attention/indexer is ≥ 30 % of the 18k token, a knob-gated selection with X2's bar; if the growth is KV read bandwidth, **dead at step 0** | medium (numerics change; ships off by default, per house rule) | Opus | bit-identical below 2 052 tokens by construction (checked, not assumed); KL within X2's bar at 18k; D_A(18k) A,B,B,A |
| 4 | **X4** | **The audit table** — N fixed prompts, `finish_reason`, tool-call outcome, length, per lane, kept as a page | ninfer's 225-response audit; this fork's own 256-cap incident (every reply truncated for a week and no gate saw it) | extends `accept_live.sh` check 5's idea into a table the owner reads; doubles as H4 | product visibility, not speed | very low | Haiku | all rows present, `finish_reason=stop` count reported, no scoring |
| 5 | **X5** | **MTP re-test, but only in a VRAM-resident regime** | ninfer 59–61 % acceptance on resident models; Q9's kill here (2.668) and hipEngine's 0.955× are host-resident results — the verify block's cost is linear in S when the routed experts come from DRAM per row, and sub-linear when the weights are read once per block from VRAM; that is the discriminator, and it does not transfer either way | nothing in Colibri today. If H0 found a speculative/MTP knob in hipFire: A/B it inside hipFire on the ladder (free). If branch 3 is built: Q9's spec applies to *that* engine, from step 0 | unknown; the point is that Q9's number is not evidence about this regime | low (measurement only) | Sonnet | interleaved knob on/off on the B ladder; `gate_ab_verdict`; text identical or divergences explained as near-ties |
| 6 | **X6** | **Bounded double-buffered pinned row-gather** (stream cold experts into the GPU instead of computing them on the CPU) | hipEngine's PLE streaming | would replace the CPU int4 path for tier misses on GLM | **projected, and it loses at batch 1**: one 14.1 MB expert over PCIe 4.0 x16 at a practical ~25 GB/s (of 32 nominal; not measured here) = 0.56 ms + a launch, vs 14.1 MB from DRAM at the measured 21.95 GB/s = 0.64 ms. Equal within the uncertainty, and the miss set changes every token so nothing amortises. Worth revisiting only for prefill chunks, where Q9 step 1 measured 29 % expert dedup across 32 rows on Qwen | medium | — | not scheduled; the derivation is the record |
| 7 | **X7** | Native artifact format / no load-time conversion | ninfer `.ninfer`, hipFire MQ4R | Colibri already has its own containers and maps shards; load time here is page cache, not conversion | none measurable | — | — | not scheduled |
| 8 | **X8** | Exact-batch decode / C1–C8 batching | ninfer | throughput under concurrency; this box has one user and the gateway serialises | none for the owner's latency | — | — | not scheduled unless the usage changes |

## 5. Multi-GPU: three gfx1100, and why "3" is the wrong question for speed

**M0 — the cheap proof (do it; it is information, not a decision).** Under the
lock with the gateway stopped (RCCL needs VRAM the tiers currently fill):
build rccl-tests against `/opt/rocm-6.2.0` (`make MPI=0 HIP_HOME=/opt/rocm-6.2.0
RCCL_HOME=/opt/rocm-6.2.0`), run `all_reduce_perf -b 8 -e 128M -f 2 -g 3` and
`-g 2`. Record: whether it runs at all; the small-message (8 B – 64 KB) latency
in µs; the large-message bus bandwidth in GB/s; and `NCCL_DEBUG=INFO`'s
transport line (P2P over PCIe vs host-staged). Sonnet, ½ day, a 15-minute lock
window. Outcomes:

- **does not run** ("hostcall not supported", atomics, or KFD access): the
  tonight's-lspci reading was necessary but not sufficient; TP/EP via RCCL is
  off the table on this box; only pipeline placement (no collectives) or one
  card. That is a finding worth the half day.
- **runs**: the small-message latency L is the number that matters for decode
  TP, and here is the arithmetic it is measured against. Tensor-parallel decode
  of the 35B places one all-reduce after attention and one after the MLP per
  layer: 40 layers × 2 = **80 collectives per token**. At 4 ms/token on one card
  (the published 250 tok/s) and compute split perfectly three ways, TP3 breaks
  even only if 4/3 + 80·L < 4 ms, i.e. **L < 33 µs** (projected; the proof step
  supplies L). Over PCIe without a GPU-to-GPU fabric that bar is the whole
  question, and the plan does not assume the answer.
- either way, **3 is an awkward TP degree** (head and expert counts rarely
  divide by 3; hipFire's documented TP/EP runs are 4–5 × gfx1201, never
  gfx1100), and **EP at batch 1 is what Colibri already does** across its three
  devices from the host — same latency structure, no new information.

**M1 — the honest use of the third card.** For a model that fits on one card,
the other two are *capacity*, not speed: a second lane (branch 1), or a larger
resident model across cards by pipeline placement (a model-class change again,
with its own H2). **TP3 for decode of a one-card model is not pursued** unless
M0 returns L well under 33 µs, and even then only as a measured item with the
ladder as its gate.

**M2 — not scheduled:** any Colibri multi-device change on the basis of M0.
Colibri's three-device expert groups are already measured (G2: round-based
issue-all/take-all; G9: CPU work in the gap) and nothing in M0 speaks to them.

## 6. Risks, and what this plan will not do

Risks, each with the check that catches it:

- **ROCm user-space on a RADV box.** HIP talks to `amdgpu` through `/dev/kfd`;
  Mesa through `/dev/dri`. They coexist, but `/dev/kfd` needs group membership
  the user may not have, and hipFire may want a newer ROCm than 6.2. Both are
  H0's first checks and both are owner decisions if they fail. **No system
  package is installed and no group is changed by a session.**
- **VRAM not released after a hipFire exit** would make the A2 arm's preload
  spill and the arm invalid. The chain asserts `mem_info_vram_used` < 1 GB on
  every card before every arm and refuses to proceed otherwise.
- **Page cache.** GLM's 182 GiB plus the MQ4R's 18.7 GB fit in 247 GiB; arm C's
  ~30 GB RSS may evict GLM pages, which is why C runs last and the chain
  re-warms GLM before the restart; the owner's first chat after the chain
  must not pay 100k major faults.
- **Clocks.** §Q13 showed the DPM level moves numbers; the chain records it
  before every arm and changes nothing. A run with a different level is a
  different, labelled run.
- **Two tokenizers.** Depth is matched by turn index and characters; token
  counts are printed beside every cell. A reader who compares tok/s across
  arms without the token counts is comparing two units.
- **Streaming granularity.** A server that batches deltas would misreport
  TTFT and decode; `usage.completion_tokens` is preferred and the row says
  which source it used. If neither exists the cell is REFUSED, not estimated.
- ~~The daily driver is down for ~4 h. Once, at night, announced.~~ [2026-09-22: the rig is a development machine; measurements run at once, nothing is scheduled for the night or announced to the owner.]
  restarts it on every exit path and `accept_live.sh` proves it.
- **The current benchmark.** Nothing here runs until it releases the lock;
  `run_chain.sh` refuses otherwise.

This plan will **not**:

- run anything on the rig, stop the gateway, or take the lock before the
  running benchmark finishes;
- write a HIP kernel into Colibri, or open a backend port, on the strength of
  the head-to-head — only D-3 can put a HIP op on the list, and only for the
  resident path;
- run a 64Ki cold prefill on GLM (9.6 h projected for a number the fit
  already gives), or TP3 for decode;
- declare a quality verdict on the 35B — H4 puts the answers side by side and
  the owner reads them;
- merge `upstream/dev`, or change `~/start_glm53.sh`, the tier caps, the DPM
  level, or the histogram;
- treat any of hipFire's published numbers as this rig's until Q1 has been
  measured here.

## 7. Order of execution and effort

| step | what | tier | rig time | wall |
|---|---|---|---|---|
| H1 | ladder `--url`, cold sweep, `context_compare`; gate against tonight's ladder and the live gateway | Sonnet | 0 (live gateway, read-only) | 1 day |
| H0 | hipFire on the box; API/prefix-cache/MTP facts; device map | Sonnet (+ owner) | one short lock window | ½ day |
| M0 | rccl-tests, `-g 3` and `-g 2` | Sonnet | 15 min under lock | ½ day |
| H2 | the chain, A B B A + C, at once (2026-09-22: never at night) | Haiku runs | ~4 h | — |
| H3 | the table into the record | Haiku | 0 | ½ day |
| H4 / X4 | the audit page, both lanes | Haiku | ~1 h under lock (the B lane) | ½ day |
| D-3 | same-op microbench | Opus | 30 min under lock | 1 day |
| X1 | retained command buffers | Opus | ~3 h (both engines, A,B,B,A) | 3–5 days |
| X2 | KL oracle | Sonnet | ~1 h | 1 day |
| V1 step 0 | gs64-vs-per-row scales in the CUDA tier; the `q38vk_*` piece to copy; step list — **can start now, no rig** | Sonnet | 0 | 1 day |
| model fetch + convert | int4-gs64 container or BF16 → `--gs 64`; after the lock frees, never under a running ladder | Haiku | NVMe/page cache, ~1 h | — |
| V1 steps 1–2 | **DONE 2026-09-16, GATE PASS** — the Vulkan tier on dev3 (record §V1) | Opus | ~35 min of lock across three chains | 1 day |
| H2b | same weights, same card, A′,B,B,A′ with `-mq4r` and `-mq5` | Haiku runs | ~2 h | — |
| L1 | GLM on two cards, only under O1 | Haiku | ~2 h | — |
| X3 step 0 | which bucket grows at 18k — **DONE 2026-09-16, UNDECIDED: "from A1's log; no extra rig time" is withdrawn, the log has no `[OPTIME]` line and engine mode cannot produce one (record §X3 step 0). Re-run as the two-depth CLI chain in that section.** | Opus | **≈1 h 45 m, overnight** | ½ day |

Fable's part ends here unless H3 lands in the NO VERDICT region and the owner
wants a call on partial data, or D-3 returns ≥ 2× and the backend question
reopens for the resident path.


## 8. Rev 13 — the Franken-engine on rome: what the measurements say to build

### 8.1 The regime map, measured (all on this box, 2026-09-16)

| regime | engine / model | decode @18k | ladder TTFT @18k | follow-up @18.5k | source |
|---|---|---|---|---|---|
| fully resident, 3 cards | llama.cpp, gpt-oss-120b MXFP4 (63 GB, 5 B active) | 75–76 tok/s | 4.3 s (HIP) / 7.5 s (VK) | 0.3–0.7 s | §GPTOSS-3CARD |
| half the experts in RAM | same | 28 tok/s | 66 s | 3.6 s | §GPTOSS-3CARD |
| all experts in RAM | same | 17.6 tok/s | 127 s | 1.0–6.3 s | §GPTOSS-3CARD |
| GLM-5.3-Flash, llama.cpp offload (18 GPU layers) | IQ4_XS, ROCm 7.14 container | 2.80 tok/s | 108 s | 5.6–7.1 s | §GLM53FLASH-LADDER |
| GLM-5.3-Flash, Colibri (tiers 79 % of calls, rest CPU) | int4-gs64 | 2.27 tok/s | 2 561 s | 6.9 s | §FRANKEN-H2 |
| host→GPU stream, 3 cards concurrent | HIP probe | — | — | — | 61.4–62.0 GB/s (1 card 28.0; dev0+dev3 share a link) |
| CPU int4 expert path | Colibri, 8 threads | — | — | — | 21.95 GB/s (§G11) |

Two things follow before any design. **Decode at depth is bytes-from-RAM per
token divided by the path's rate, plus whatever grows with context.** And on
GLM-5.3-Flash the expert misses are *not* the growing part: the CPU path
streams ~985 MB/token (§G11 note) at 21.95 GB/s ≈ 45 ms, inside a token that
is 235 ms at 2.7k and 441 ms at 18.4k (§X3 step 0). Removing the miss cost
entirely bounds decode at ≈ 1000/(441 − 45) ≈ **2.5 tok/s at 18k**. The 205
ms/token that appears between 2.7k and 18k is unattributed (X3 step 0 could
not run: the `[OPTIME]` table dies with the engine's SIGTERM). **So for the
daily model the decode lever is the depth term, not the expert path**, and
the prefill lever is the expert path — 2 561 s vs llama.cpp's 108 s on the
same model is the order-of-magnitude gap, and it is a prefill gap.

### 8.2 What the numbers do and do not support

- **Supported:** GLM-5.3-Flash prefill from 2 561 s to the ~100 s class at
  18k by chunked, batched prefill with non-resident experts streamed once per
  chunk into the three cards (F2 below). Projected, derivation in F2.
- **Supported:** a fully resident GLM-class model ≤ ~60 GB int4 across the
  three cards at the 75 tok/s class with sub-second follow-ups — the measured
  gpt-oss regime, on Colibri's own three-card tier (V1's code, one card today).
  This is the only measured route to "usable speed at depth" and it requires
  a model that fits 72 GB with KV. The owner picks the model (assumption held
  here: GLM-5.3-Flash stays the daily driver, so this is regime B, optional).
- **Not supported:** GLM-5.3-Flash (182 GiB int4) decoding above ~2.5 tok/s
  at 18k by any placement change alone. 3 390 of 12 096 experts are resident
  (79 % of calls); the other 21 % cost 45 ms; the remaining 396 ms/token are
  attention, indexer, dense, submit. Until F4 attributes the 205 ms growth,
  no decode projection for this model is honest, and none is made.
- **Not supported:** tensor parallelism across the three cards as a speed
  lever. Decode on a resident model is already 75 tok/s with layer split and
  zero cross-card traffic per token; TP adds a 25 µs hop per layer (§M0c) for
  a bandwidth split a 7900 XTX does not need at batch 1. KV offload to RAM:
  the KV of a 18k context is not the bottleneck on either engine (follow-ups
  at 18.5k are 0.3–7 s on all rows); it becomes one only at depths this box
  does not reach in a session.

### 8.3 Items

| id | item | expected (projected, derivation shown) | gate | tier / effort |
|---|---|---|---|---|
| **F0 — DONE 2026-09-17, PASS (rev 14)** | **Vulkan host→device stream rate.** §PCIE-STREAM measured HIP. Colibri is Vulkan/RADV; measure `vkCmdCopyBuffer` from host-visible memory and from a mapped page-cache buffer, 14 MiB blocks, 1/2/3 cards concurrent, same probe shape (`tools/hot-expert/rome_vkbench.c` has the device setup) | ≥ 50 GB/s aggregate on three cards if RADV's upload path matches HIP's; **falsifier:** < 40 GB/s → F2 streams through HIP (the container) or a HIP upload helper, and D-3 leaves the parked list | ROW lines as §PCIE-STREAM; under the lock, gateway up | Sonnet, 1 day |
| **F2 — DONE 2026-09-19, GATED PASS (rev 19).** Turn 1 136.8 → 67.6 s = **2.03×** (bar 2.0); 18 439-token turn 1 174.8 → 672.6 s = **1.75×** (bar 1.6); both SEPARATED. Cold 18k prefill 37.5 → 21.1 min. Decode 4.30 → 4.33 tok/s at full depth. X2 KL with the step-2a clamp: mean KL 3.48e-05, top-1 99.89 %. Knob-off bit-identical to the served pristine. Record §F2, §F2-LADDER | **Chunked batched prefill with expert streaming.** Prefill in chunks of S rows (512 default, knob); per MoE layer, group the chunk's rows by expert (Q9 step 1's grouping exists); resident experts run on their tier as today; each **non-resident** expert is copied once per chunk into a per-card ring buffer (dev2 first — its link is its own; dev0/dev3 share ~33 GB/s) and run through `qmatmul_gate_up_tile.spv` / `qmatmul_tile.spv` on that card; the CPU int4 path stays as `COLI_PREFILL_STREAM=0`. | non-resident set ≈ 8 706 experts × 14.16 MB = 123 GB per chunk; at 61 GB/s = 2.0 s per 512 rows ≈ **3.9 ms/token** for the streamed experts, vs 113.9 ms/token today (fit, §1); with GPU tile compute and the attention/indexer prefill unchanged, 18k prefill projected **≈ 70–130 s** (vs 2 561 s measured, 108 s on llama.cpp). Chunk 2 048 rows: 9 chunks, ~1.1 s/1k tokens of streaming | `prefill_gate.sh` (oracle: `teacher_forcing` identical, last-token logits vs pristine), X2's KL bar, then the H2 ladder A,B,B,A vs today's A rows; ~~≥ 10× on the 18k ladder-turn TTFT~~ **re-based rev 16 (record §RP-F6a: ceiling 1.87×): ≥ 1.6× on the 18k ladder-turn TTFT and ≥ 2.0× on turn 1** or the item is rejected; the row beside llama.cpp's 108 s is reported either way — **turn 1 MET at 2.02× and the short rung's deepest turn (5 112) at 1.80×, record §F2d; the 18k rung is the only bar left and projects 1.71×** | Opus, 2–3 weeks; after F0 |
| **F4 — DONE 2026-09-17, ANSWERED (rev 14): the DSA indexer** | **Attribute the 205 ms/token depth growth on GLM.** The zero-cost route: close the engine's stdin in `ttft_serve.EngineDriver.close()` (not in `openai_server.Engine.close()`) so `optime_print` runs; one A-arm ladder with `COLI_TIMERS=1` then carries the table. Fallback: the 1 h 45 m CLI chain in record §X3 step 0 | which bucket grows: attention core, DSA indexer, KV read, or dense/submit. **This decides the decode item:** ≥ 30 % attention+indexer → X3 (stricter selection, knob-gated, X2 bar); KV-read-bound → a KV layout/placement item; neither → the base token is the target (submit overhead, dense stream) | the `[OPTIME]` table at 2.7k and 18.4k in the record, both arms | Sonnet (fix + one arm, ~2 h rig) then Opus (interpretation) |
| **F1 — REJECTED 2026-09-19 (rev 21, record §F1-PROBE: a streamed batch-1 window costs 2.70 ms against 1.747 ms on the CPU). Step 0 ANSWERED (rev 20).** eg/cpu overlap split, record §F1-STEP0: `take ≈ 0` while `cpu_in ≫ take_cpu0` at 18.6k decode → CPU misses ARE the critical path, ceiling **54.6 ms/token = +29 % decode at 18k** with the CPU path free (was "≤ 45 ms/token"). Not free: the batch-1 streamed-miss latency is unmeasured — below ~1.2 ms/window the realistic gain is ~25–30 ms/token (~+12 %); at ≥ ~1.75 ms/window (today's CPU cost) F1 has nothing to gain. Falsifier probe next (`f1_ring_probe`), before any engine work | **Streamed decode misses** (the ~11 % non-resident share at the served 1695/1695 tier): same ring buffer as F2, at batch 1, prefetch next layer's misses while this layer computes; CPU path becomes the fallback knob. Frees the 8 cores for dense/indexer work | ≤ 45 ms/token on GLM → **revised, record §F1-STEP0: ceiling 54.6 ms/token (+29 % at 18k) if the miss is free; the falsifier probe decides whether it is** | X2's KL bar, text identical, A,B,B,A on the ladder; verdict through `gate_ab_verdict`; **gated on the falsifier probe clearing first** | Opus, 1–2 weeks |
| **F3 — CLOSED BY THE OWNER 2026-09-21 (rev 38): Qwen3.6 is not an acceptable model; step 2b not started. Before that: STEPS 0-2a DONE 2026-09-20 (rev 37). Model Qwen3.6-35B-A3B. Step 0 (§F3-STEP0): int4 costs mean KL 0.0316 / top-1 92.96 % vs row-wise int8 -> int8. Step 1 (§F3-STEP1): int8 100 % resident over dev2+dev3, KL at the floor, 21.5 tok/s = int4's speed. Step 2a (§F3-STEP2A): lm_head + DeltaNet proj on dev0, +2.0 % vs a +20 % gate, NOT MET -- a blocking GPU call costs ~0.3 ms, piecemeal offload closed. Step 2b = the GPU-resident decode token (Opus, design note first, weeks); not started** | **Three-card residency for any model that fits** (regime B): extend V1's Vulkan tier from one card to three (heat-ranked, per-device budget, `COLI_VK_DEV0/2/3`), attention/KV on dev0, tiers on all three, so a ≤ 60 GB int4 GLM-class model is 100 % resident with P7/P9 checkpoints and slots intact | the measured resident regime: 75 tok/s class at 18k (gpt-oss, layer-split; Colibri's number **unknown** until run — the submit-model ceiling of §3 branch 3 was withdrawn but not replaced by a measurement) | tier line asserts 100 % residency; A,B,B,A vs llama.cpp resident on the same GGUF-equivalent weights; the greedy-text agreement count of H2b | Opus, 1–2 weeks; **only when the owner names the model** |
| **F5 — DEFERRED 2026-09-20, NOT RUN (rev 25/26).** Its precondition was a cheaper verify block; F1 was rejected (a k = 2–6 row block is slower streamed than on the CPU) so the block costs what it cost when Q9 killed MTP. F8 then made a CPU miss 20 % cheaper (`cpu_in` 1.868 → 1.489 ms per window); nobody has re-priced the verify block against that, and that re-pricing (a Sonnet-sized arithmetic step from §F8 and §F1-PROBE, no rig) is the only way this re-opens | **MTP in the streamed regime** (X5): hipFire measured 131 tok/s at 1.4k *with* MTP on the resident 35B (§FRANKEN-H2 throwaway `mtp: true`); Q9 killed it here in the CPU-miss regime. Re-test only after F2/F1 change the verify block's cost | unknown; the discriminator is whether the verify block reads streamed experts once per block | interleaved knob on/off on the ladder; text identical | Sonnet, after F1 |
| **F6 — F6a DONE 2026-09-19, GATED PASS, IN SERVICE (rev 15/16, record §F6a, `15462dc2`): OpenMP over the pools + exact heap top-512, bit-identical; decode at 18k 2.27 → 4.33 tok/s, the 18k turn 2 607 → 1 179 s. F9a (below) then took the score pass itself. No F6b is open: coarser pooling is impossible (trained `[4][128]` compressor)** | **The indexer at depth (X3, promoted by F4).** A cheaper DSA selection at depth for GLM's MLA layers: profile says 25 ms per MLA call at 18.6k vs 5.8 at 1.3k (`index/ctx` 4.46 µs per context token per call over the pooled-key cache of G5). Design first (Opus): what the indexer reads per call today, whether a coarser pool or a two-level select cuts the bytes, dense-identical below 2 052 tokens by construction, knob-gated, X2's KL bar | ceiling stated in the record: holding the indexer at its 1.3k cost → ≈ 3.3 tok/s at 18k (from 2.27); the item is worth ≤ +45 % decode at depth and nothing at shallow depth | `[OPTIME req=]` index ms/call at 18.6k before/after; teacher_forcing identical below 2 052; X2 KL above; A,B,B,A ladder through `gate_ab_verdict` | Opus, design 2 days, implementation 1–2 weeks; after F2 step 0 |
| **F7 — DONE 2026-09-20, GATED PASS, IN SERVICE (rev 24, record §F7-VERDICT: 18k turn 1.50x, turn 1 1.35x)** | **Batched MLA attention core in prefill.** Design note `tools/hot-expert/F7-MLA-ATTN-GPU-DESIGN-2026-09-19.md`; branch `perf/f7-mla-attn-gpu`, knob `GLM53_MLA_ATTN_GPU=1`, four new shaders on dev0, latents copied per layer-chunk into 140 MB of shared dev0 scratch (expert tier untouched) | **MEASURED**: core 1 290 -> 74 ms per 512-row layer-chunk (17x); 9 115-token prefill 678.8 -> 479.0 s = **1.417x**, 3 013-token 206.5 -> 131.7 s = **1.57x**. Knob-off bit-identical to the served pristine | O1 PASSED. Numerics: **no defect** -- in-situ per-layer median error stays at fp32 level (2.9e-07 -> 6.2e-06 over 11 layers), o-projection and sub-batching both bit-identical, replay vs float64 ratio 0.80-1.27. The shallow top-1 bar is unmeetable by any reordered kernel; at depth R2/R1 = 1.7x KL / -1.4 pt. Ladder not run | Opus; decided: numerics accepted on the jitter arm (rev 23), gate bar 1.25× met at 1.50× (rev 24) |
| **F8 — DONE 2026-09-20, GATED PASS, IN SERVICE (rev 26–28, record §F8-STEP0, §F8; `93f0f681`)** | **The batch-1 CPU expert path in decode.** `GLM53_MOE_ONE_TEAM=1` (one OpenMP team per MoE layer, bit-identical) + `GLM53_I4_FAST=2` (`coli_i4_row_gv`: the int4 row kernel keeps its group sums in vector form, one `hsum` per row; reassociation only, relL2 ~3e-7). Step 0 refuted two cheaper hypotheses first (bandwidth-bound; FMA latency chain, worth 10 %) | projected ≤ 30 % off the 73 ms CPU-expert bucket; delivered `cpu_in` 1.868 → 1.489 ms per window | **decode +6.4 to +9.6 %, SEPARATED at all seven turns to 19.3k (4.270 → 4.585 tok/s at the deepest)**, TTFT unchanged, greedy text identical. Known gap: the chain's KL report is vacuous for a decode-only change (streamed prefill runs no CPU expert) | Sonnet throughout (~3 rounds) |
| **F9a — DONE 2026-09-20, GATED PASS, IN SERVICE (rev 29, record §F9a, §F9a-VERDICT; `189fd945`). F9 (the GPU indexer) CLOSED WITHOUT BEING BUILT** | **The DSA indexer's score pass with the heads in the SIMD lanes** (P5.1's pattern in `c/sparse_index.h`; each dot keeps its summation order, so bit-identical; on by default, `GLM53_INDEX_LANES=0` restores the scalar dot) | the indexer was 25 % of the 18k turn after F7 | **18 439-token turn 455.0 → 357.7 s (1.27×, bar 1.10), decode at 18k +8.3 %, follow-up TTFT −20 %**; `teacher_forcing`, text, the full logit dump and 33 495 index rows identical to the served binary | Sonnet; Fable read the loop first, which is what saved the Opus shader |
| **F10 — STEP 0 DONE 2026-09-20, THE FIX IS PARKED FOR THE OWNER (rev 30/31, record §F10-STEP0)** | **KDA in prefill.** In service the projections and the recurrence both run on the GPU; `coli_vk_kda_step_rows` records S sequential 64-workgroup dispatches per layer-chunk, each behind a memory barrier: 0.111 ms per row-layer, 33.9 s of the 357 s turn, latency-bound. Candidate fix: the token loop inside the shader (one dispatch per layer-chunk; per-head order unchanged, so a bit-identical oracle applies). `proj` (40.8 s) not examined | ≤ ~30 s of 357 s (8 %) on long prompts, nothing for decode | not built; would gate like F9a (bit-identical dump + A,B,B,A ladder, bar 1.05×) | Opus, ~200–300k tokens by F7's bill; **the owner says "do F10" or it stays parked** |
| **F11 — STEP 0 DONE 2026-09-21 (rev 40, record §F11-STEP0): Qwen3.8-Flash-Next UD-IQ4_XS 14.4-15.2 tok/s at 19k, cold 19k prompt 35 s; DeepSeek-V4-Flash UD-IQ2_M 8.0 tok/s, 97 s; both with a 256k window, cards filled by llama.cpp's fit. DEPTH DONE (rev 41, §F11-DEPTH): Qwen3.8 at 257k decodes 6.1-6.5 tok/s, cold 257k prompt 17 min, follow-up 1.5 s. Next: the owner's quality swap** | **The acceptable models at a smart quantization, mostly VRAM-resident, on whatever engine runs them.** Step 0: `Qwen3.8-Flash-Next` UD-IQ4_XS (93.7 GB) and `DeepSeek-V4-Flash-0731` UD-IQ2_M (90.9 GB) under the rig's llama.cpp on the 09-16 context ladder, experts on the cards as far as they load, arms Q,D,D,Q | Sonnet writes the chain; the orchestrator launches and polls | a measurement, no gate: decode tok/s and TTFT per depth next to GLM-5.3's served 5.04 tok/s / 356 s at 18k; then the owner judges answer quality through a one-command swap | nothing is built before both are in |
| parked | H2b, H4, D-3 (branches exist), L1 | — | — | D-3 re-enters only under F0's falsifier |

### 8.4 Order, and what each step must show before the next

**As executed (rev 32, 2026-09-20) — the paragraph below is the rev 13 plan, kept so the change is visible.**
F0 → F4 → F6 design → F6a → re-profile → F2 (gate re-based to 1.6×/2.0×, plus the swiglu-clamp knob its KL bar forced) → F1 step 0 → F1 probe (REJECTED) → F7 → re-profile → F5 deferred → F8 step 0 → F8 → F9a (F9 closed unbuilt) → F10 step 0 (fix parked). **What is open: F3 (owner names a model), F10's shader (owner says go), F5's re-pricing (cheap, optional). Nothing else is queued, and a new session should not invent an item: re-profile first (`COLI_TIMERS=1` on a ladder B arm costs no extra rig time) and open one from the largest bucket.**


F0 (1 day) → F2 (the prefill item; the owner's long-document wait) in
parallel with F4 (2 h of rig, the decode question) → the decode item F4
names → F1 → F5. ~~F3 runs whenever the owner names a ≤ 60 GB model~~ (closed 2026-09-21); it was
the shortest path to the 75 tok/s regime and does not wait for anything
above. Every item ships behind a knob, off by default, with its oracle and
its A,B,B,A row in the commit body; every projection above is a thing to be
beaten or refuted by that row.

### 8.5 The one decision that is the owner's

**[2026-09-22: this section is superseded. The owner decided: the engine is built, not chosen (rev 45); the models are Qwen3.8, DeepSeek V4.x, GLM-5.3 (rev 38); the target is 256k (rev 39). The A-or-B regime choice and "F3 the moment a model is named" no longer exist. See `FRANKEN-ENGINE-DESIGN-2026-09-22.md` §7.]**

Regime A keeps GLM-5.3-Flash: prefill falls by an order of magnitude (F2),
decode at depth is whatever F4 finds, and stays under ~3 tok/s until then.
Regime B takes a GLM-class model that fits 72 GB: 75 tok/s class at depth,
measured, on the same engine after F3. Both can coexist behind `coli`'s
model switch; the plan builds A first because it is the daily model, and
F3 the moment a model is named.
