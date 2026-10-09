# Plan for the open GLM decode items (2026-10-08) -- executed the same evening, outcomes in section 0

Written after the trunk-GEMV work (record §L5-GLM-TRUNK-GEMV, §L5-GLM-ROWSPLIT, §L5-GLM-GEMV-GROUP; decision log `FRANKEN-ENGINE-PLAN-2026-09-15.md` Rev 95-98).
This file says what is still open, what each item is worth, how it is gated, and in which order it should be done. It does not repeat the evidence: every number below points to a record section.
Labels: **measured** = a record row exists; **estimate** = derived here from measured numbers, with the derivation; **unknown** = no basis yet.

## 0. Status after the first execution (2026-10-08, evening; plan Rev 99 in `FRANKEN-ENGINE-PLAN-2026-09-15.md`; D14 added at night, Rev 100; **what is left and in what order: section 8, plan Rev 101**)

The plan was executed in one session (Phases 0-1 and the part of Phase 2 that the gates opened). Outcomes, each with its record section (`ROME-3x7900XTX-2026-09-04.md`, new sections at its end) and the engine notes (franken-engine `GLM5.md` section 22):

| ID | Outcome | Gain (labelled) |
|---|---|---|
| D1 `--gemv-lds` | **closed, not enabled.** Slower in isolation (+1..7 %/launch), faster in the trace (-0.18 ms wall, SE 0.10) with a layer-period change of 4 us (gate: 10 us, 1 %); the flag is global (it also stages the prefill GEMMs). The same idea carried into a kernel with no dead generality is D7c below. | 0 (measured) |
| D2 slow `hc_*_fn` launches | **closed, nothing to fix.** The 22 % population belonged to the old `k_gemm_batch` hc fn and vanished with the row-split kernel (0.10 % of clean launches now); the rest is a clock ramp after host stalls that looks like a tracing artefact. | ~0.01 ms (measured) |
| D3 rows-based waves rule | **done**: `--gemv-rowsplit-waves 1`, bit-exact (model gate), launcher default. | -0.12 ms wall (SE 0.06), -0.18 ms kernel (measured, paired trace) |
| D4 Q6_K split stall | **done, fixed and verified.** Cause: a flat load below the LDS allocation in the staged-x path of the Q6_K branch (memory violation); an address-only fix; the unsplit head never hit it. | 0 ms (correctness) |
| D5 decode past 2 051 deep | **explained by D14** (the first reading -- a decode-step race in layer 4's conv window -- was wrong: the taps are the last PROMPT row). Serialising launches and `--ctx` changed nothing because there was no race and no layout dependence. | 0 ms (verification) |
| D14 root cause of D5 | **SOLVED, fixed, gated, INSTALLED** (plan Rev 100, record §L5-GLM-D14, GLM5.md 22.6): the prefill attention grid was sized from the last row of a chunk; three rows in four past 2 048 tokens lost their 1-3 newest keys and read stale partial buffers. Chunk 512 / 256 / 333 against a chunk-1 prefill: old 1 655 of 1 788 taps not exact, fixed all exact; eager vs HIP-graph decode exact; model gate PASS (12 configs); prefill time unchanged. The logits error before the fix: cos 0.9895 at a 2 300-token prompt. | 0 ms; **a correctness fix for every prompt past 2 048 tokens** |
| D6 odd serve-A/B arms | **done**: clock / temperature / power / CPU-frequency sampling beside every arm of `glm_serve_ab_chain.sh` (`SAB_SAMPLE`, default 1) and a per-request report with the correlations. | 0 ms (measurement quality) |
| D7 bandwidth gate | **GO, but not for the reason asked.** The 16-byte exact-order kernel (D7b) is slower than a plain copy of the engine's own loop, so D7b is dropped; the copy itself (`--gemv-q8fast`, **new item D7c**) is 17-21 % faster per launch than the generic `k_gemm_batch` and bit-exact. | **-1.49 ms wall (SE 0.08, -2.3 %), layer period 1 403 -> 1 373 us (measured, paired trace)** |
| D8 KDA o / DSA o tail | **parked** (<= ~0.4 ms above the copy, below the 0.3 ms stop rule once the exact-order constraint is counted). | -- |
| D9a / D9 tensor-level split | **D9 parked**: modeled +1.9..2.6 ms, but the model overstated the shipped whole-slab table threefold (realization 0.33), so +0.6..0.9 ms < 1.5 ms. | -- |
| D10 / D11 / D12 | unchanged, parked for the owner's word. | -- |

**New items.** **D13** (optional, ~0.1 ms): the same recipe for the Q6_K lm_head (one launch, 722 us a token). **D14** (correctness): CLOSED the same night -- see its row above and record §L5-GLM-D14. **D15** (sizing): where do the other two thirds of the fetch model's modeled gain go (the model said -2.9 ms for `--fetch-assign optimal`, the trace -0.97 ms)? Every fetch-side lever is sized with that model, so finding the gap comes before building any of them.

**Where decode stands after this session** (the section-1 numbers are the starting point): **installed `~/bench/franken_bin/franken_dec_glm` = `.d14` sha `f365fc7c014696ca`** (the `.fin` build below PLUS the D14 attention-grid fix; franken-engine code `a7bf81a`, main `f072093`; rollbacks `.fin` `ca2bc21c19b451c0`, `.gr` `1f95c34b2d8bbf06`, `.rs`, `.adm`); the specialised GEMV and the waves rule take ~1.6 ms off a ~63.8 ms traced token (-1.49 ms and -0.12 ms, two separately measured effects, paired kernel trace). Accepted: `accept_live` PASS (warm new chat 0.86 s, follow-up 0.9 s, behind an abandoned request 0.8 s), browser first token 1.06 / 1.05 s. **Serve-path A/B, two runs (14 arms):** 10 arms are undisturbed and cluster tightly (<= 0.06 tok/s inside a build on W2-W5); the new binary is +2.5..+2.8 % (mean +2.7 %) on every request type; 4 arms were disturbed (3 slow, 1 cold-start fast, both builds hit) and are named and left out; with all 14 the mean difference is +0.6 % (record §L5-GLM-D7, §L5-GLM-D6). The service is still OFF behind the reservation flag.

## 1. Where decode stands (the starting point of the first round; the current state is in section 0 and section 8)

- Installed `~/bench/franken_bin/franken_dec_glm` = `.gr` sha `1f95c34b2d8bbf06` (rollbacks `.rs` c21f9a66, `.adm` c97b0eb2); franken-engine main `68e31cd`; service OFF behind `~/bench/.dev_reserved`.
- Decode token: ~61 ms at depth ~8.3k with the row-split build (measured, shipped-adaptation arms of the speed A/B); the group build takes ~1.6 ms more off (trace; **estimate** ~59-60 ms, ~16.8 tok/s, not re-measured in the gate); served short chats 16.3-18.1 tok/s (measured, group build). The trace (rocprofv3, inflated) shows a MoE layer period of ~1.40 ms after the trunk-GEMV work (was ~1.48).
- Where the time is (record §L5-GLM-DECODE-TRACE, trace-based): the critical expert fetch (the slowest of the three cards' stage kernels) is ~35 ms of a ~62 ms token and runs at ~77 % of the 62-64 GB/s the three PCIe links deliver together; the rest is the trunk, now ~3.3 ms (5 %) shorter than on 2026-10-07 morning. GEMV kernel time a token: 20.0 -> 18.0 ms.
- **No target number has been set by the owner.** The items are ranked by expected gain per effort and carry stop rules (section 5), so the work ends when the remaining candidates are not worth their cost.

## 2. How every item is measured (rules from the last two days; do not re-derive)

1. Anything that claims "no bit moves" passes the exactness gates first: `glm_group_micro_chain.sh`-style microbenchmark against main's own backend where a kernel changes, then a model gate with `oracle_verdict.sh` (maxabs 0 on every tap; chunk 1 + 8 greedy against `gpu5/g136_eager`, 64 greedy at depth 1 500 against an in-process flag-off dump, with the flag on/off crossed with rowsplit, adaptation every token, `--gemv-lds`, a non-scoring context). Gates may now also run past 2 051 tokens of depth (D14 fixed the prefill attention; compare a CHUNKED prefill with a chunk-1 reference, `ckpt1006/glm_d14_chain.sh`, never two chunked runs with the same last chunk: both carried the same fault and agreed).
2. Speed is judged by the **paired kernel trace** (four runs of one binary, flag order 0 1 1 0; `glm_group_trace_seq.sh` is the template, `decode_trace_report.py` for the layer period, `rowsplit_trace_paired.py` for index-aligned per-token wall) and then by the **serve-path A/B in both orders** (`glm_serve_ab_chain.sh`: old new new old, then new old old new; quote the pairs that agree, name any arm left out). Decode medians of 32-48 tokens cannot resolve an effect below ~3 % (identical-flag arms differ by 1-5 ms; a warm-up arm goes first in any in-process palindrome).
3. Before a build that allocates device memory: the VRAM budget per card from the shipped config's log (free MiB at steady state 732 / 818 / 458), written first (CLAUDE.md "Before a build that allocates device memory").
4. Landing: franken-engine through the Mac clone (ff-merge, push as claude-bot, rig ff); acceptance of the installed binary (`glm_accept_chain.sh` + `run_ui2.sh`); then the task-closeout skill. GitHub only on the owner's go-ahead after the secret scan.
5. Tiers (CLAUDE.md): Sonnet for harness, knobs, ports of an existing pattern; Opus for a real kernel; subagents build and write chains, they never run a GPU program; the orchestrating session runs the chains and watches them.

## 3. The items

Effort = working time of one session including the chains. "Gain" is milliseconds off a ~60 ms token.

### Phase 0 - cheap, mostly offline or one chain each (do first, in this order)

| ID | Item | Gain | Basis | Effort | Tier |
|---|---|---|---|---|---|
| D1 | `--gemv-lds 0\|1` for GLM: speed of staging the activation in LDS | -0.5..+0.5 (sign **unknown**) | plumbed and bit-exact in both gates (`gr_x_lds0`, `rs_x_lds0`); staging applies only to launches whose slice fits 4 096 floats (the unsplit K 4096 matrices: q/k/v, wq_a, the dense and shared-expert up/gate); K 8192/16384 matrices never stage and the row-split kernel never stages | 1 hour + one 25-min trace sequence | Sonnet |
| D2 | the 22 % of `hc_*_fn` launches that take ~21 us instead of 9.4 (90 launches a token) | <= 0.25 | **measured** excess (0.22 x 90 x ~11.6 us); cause unknown, not layer position, not overlap with other kernels | 2-3 hours offline on the existing traces, no GPU | me |
| D3 | rows-based waves rule for `k_gemv_rowsplit` (<= 64 rows 16 waves, 65-200 rows 8, more 4) | ~0.13 | **measured** microbenchmark: f_a class 11.2 -> 10.1 us at 8 waves (x90), router 14.5 -> 13.8 us at 4 (x42); waves move no bit (the microbenchmark covers 4/8/16 on every case) | 1 hour, ride on D1's trace sequence as a third flag value | Sonnet |
| D4 | the synthetic Q6_K nsplit-16 case that stalls the card **in main's own reference backend** (`bench_rowsplit --only-case q6k_fallback`, 4 of 4 runs) | 0 ms; **correctness insurance** | a future model with a small split Q6_K trunk matrix would reach this path in production (today only the unsplit head is Q6_K) | 2-3 hours: bisect launch by launch (k_gemm_batch alone, then the reduce), `AMD_SERIALIZE_KERNEL`, ISA of the Q6_K branch | me / Opus if a kernel bug |
| D5 | decode past 2 051 tokens of depth is not run-to-run reproducible (handoff §4 item 7) | 0 ms; **verification coverage** | every "bit-exact" claim so far is below that depth; candidates: ties in the pooled top-512 selection, float atomics | 3-4 hours: the same prompt twice at depth ~2 300 with `--dump`, `--oracle` to find the FIRST non-exact tap/layer | me |
| D6 | the odd serve-A/B arms (2 of 16 arms moved 5-7 %, cause unknown) | 0 ms; **measurement quality** | records §L5-GLM-ROWSPLIT, §L5-GLM-GEMV-GROUP | 1 hour: sample GPU clocks/temperature and CPU frequency every 5 s into the A/B chain's output, look for a correlation the next time an arm is odd | Sonnet |

D1 gate: layer period (trace) and serve A/B both better by the noise floor (>= 10 us a layer, >= 1 %); otherwise keep the default (1). D3 and D1 share one trace sequence (flag values 0 / 1 / 1 / 0 on each knob in separate sequences, or three-valued if the sequencer takes a flag string).
D4/D5/D6 have no speed gate: they end with a finding in the record.

### Phase 1 - decision gates (cheap measurements that decide whether a build is worth starting)

| ID | Gate | Decides | Effort |
|---|---|---|---|
| D7 | read-only bandwidth microbenchmark of the Q8_0 access pattern (34-byte blocks, one wave a row) at the current byte-per-lane loads against a 16-byte-per-lane variant, on 35-71 MB matrices (the sizes of KDA q/k/v, KDA o, DSA o) | whether D7b and D8 are worth building: **go** if the wider variant is >= 8 % faster at 35 MB | 3-4 hours (Sonnet writes it, one micro chain) |
| D9a | re-run the offline fetch model (`ckpt1006/lane_model.py`, `crit_fetch.py`, `stage_hist.py`) with the corrected balanced bound (62-64 GB/s), the measured miss distribution after the placement policy (hit ~0.66, ~1 300-1 600 MB missed a token) and the cost of a cut expert (~30 us hand-off, bit-exact `gate+up \| down` on two cards) | whether the tensor-level split (D9) can reach >= 1.5 ms | 2-3 hours offline |

### Phase 2 - builds, each only if its gate passed

| ID | Item | Gain | Basis | Effort | Tier |
|---|---|---|---|---|---|
| D7b | Q8_0 trunk GEMV with 16-byte weight loads, per-lane accumulation order preserved (block g into acc[g % 4], then `(a0+a1)+(a2+a3)`, then the wave sum) | <= ~0.8, low confidence | **estimate**: ~10 ms of big-matrix GEMV time at 83-85 % of the 800 GB/s bound; a 7-point rise to ~92 % is 0.8 ms; outside measurements: 16-byte non-temporal loads reach 952 GB/s in a microbenchmark, llama.cpp Q8_0 reaches 81 % of peak (research file `ckpt1006/trunk/gemv_research_2026-10-07.md`) | 1-2 days (a rewrite of the hot loop; blocks are 4-byte aligned only for even index, never 16-byte aligned, so the loads go through LDS or lane shuffles) | Opus |
| D8 | the tail of KDA o and DSA o (4 096 rows, K 8 192 / 16 384, unsplit): 4 096 waves against 3 072 resident = 1.33 rounds; run each row's four accumulator chains on four waves and combine in the same order | 0.2-0.35 | **measured** excess over the bandwidth line: 7.4 us x 34 + 8.1 us x 11 = 0.34 ms is the most it can give | 1 day; share the rewrite with D7b (both change the lane/wave mapping) | Opus |
| D9 | tensor-level split of a missed expert's slab over two cards (`gate+up` on one, `down` on the other), bit-exact (each row's dot product stays on one card, the owner still sums the slots in order) | 1-2 (plan Rev 90 rated it "~2 ms if a third of the model carries over"); the model's perfect-split headroom is 5.6 ms of which `--fetch-assign optimal` captured 0.97 ms (measured) | **estimate**, to be re-sized by D9a | 2-4 days (stage kernel, plan table, one more hand-off per cut expert) | Opus |

Gates: D7b and D8 as section 2 (microbenchmark bit-exact, model gate maxabs 0, trace, serve A/B). D9: the same, plus the VRAM budget (a cut expert changes which card holds which bytes) and a check that the cross-card synchronisation (~65 us a layer) does not grow.

### Phase 3 - parked until a reason appears

| ID | Item | Gain | Why parked |
|---|---|---|---|
| D10 | CPU lane for a layer's straggler slab (record §L5-GLM-CPULANE*, sources `tools/hot-expert/cpulane/`) | 1.3-2.7 realistic (2-4 %) | multi-day build; all three killers answered favourably (sync 20-40 us a layer, 6 of 8 cores free, int8 activations change a layer's MoE output by 0.54 %); remaining weak link: the IQ3_S AVX2 kernel (2.6 GB/s a core); a quality-gated change (`franken_quality_chain.sh`). Start only if D1-D9 leave the owner wanting more |
| D11 | persist the placement average across restarts | probably ~0 | **evidence against**: in every serve A/B the first request after a cold start (W1) decodes as fast as the later ones (15.4-16.3 vs 15.5-16.3 tok/s), so a generic start costs little. Check the first adapter windows of one cold start; build only if they show a warm-up penalty |
| D12 | validate the DeepSeek adapter position fix (Gitea tag `archive/adapt-ds4` = franken-engine `589e855`, untested; to run it: `git checkout -b adapt-ds4 archive/adapt-ds4`) | only if DeepSeek is served | recipe in handoff §4 (two fresh `franken_decode_ds4` builds, `ds4_gpu_gate.sh`, then `glm_serve_ab_chain.sh` with `SAB_START` / `SAB_PGREP`) |

## 4. Order and expected total

1. Phase 0 (one to two days of session time, one GPU chain each for D1/D3, offline for D2/D4/D5, a small edit for D6): realistic gain 0.2-0.4 ms plus three findings that make later work safer.
2. Phase 1 (D7, D9a; half a day): produces the go/no-go numbers.
3. Phase 2 in the order D9 (if D9a >= 1.5 ms), then D7b + D8 together (if D7 >= 8 %).
4. Phase 3 only on the owner's word.
- **If everything in Phases 0-2 lands: about 2-4 ms (3-6 %) more** (estimate; the sum of the ranges above, D9 and D7b carrying most of the uncertainty). That would bring a token to roughly 56-58 ms (~17-18 tok/s).
- Nothing in this plan changes the 62-64 GB/s the three PCIe links deliver; only hardware or fewer missed bytes do (section 6).

## 5. Stop rules

- Trunk work (D1, D3, D7b, D8) stops when the best remaining candidate is predicted below 0.3 ms or its gate fails; the big matrices are already at 83-90 % of the bandwidth bound.
- D9 is parked if D9a predicts < 1.5 ms; D7b/D8 are parked if D7 shows < 8 %.
- An item whose exactness gate shows one non-exact tap is not shipped behind a quality claim: it is fixed or dropped (the project's bar is maxabs 0; a deliberate order change goes through the quality gate instead, and is labelled so from the start).
- A speed claim needs the trace AND the serve-path A/B in both orders. One odd arm is not a result.

## 6. Owner's decisions (nothing here blocks Phases 0-2)

1. **When GLM goes back into service** (the rig is reserved; recipe handoff §3: start the Franken engine with the flag still in place, then remove the flag, then judge with `accept_live.sh` / `accept_ui.sh`).
2. **Hardware.** The fetch ceiling comes from the topology: one card alone gets 28 GB/s, the two cards behind root complex `0000:80` get 36 GB/s together, root complex `0000:c0` has no device (record §M4-TOPOLOGY); a slot there for one card could give ~84 GB/s in total, by far the largest single lever (the fetch is ~35 ms of the token). The owner has said all 7 slots are covered by the three cards and cannot change, so this is **closed unless the owner reopens it** (it would need risers or an enclosure).
3. **Two worktrees that were not from this work -- RESOLVED 2026-10-08 night, both working directories removed (nothing else touched).** The Mac's `.claude/worktrees/agent-*` checkout of `perf/x1-retained-cmdbuf` (16 Sep, an agent's X1 experiment, REJECTED on measurement: plan Rev for X1, record §X1; the branch stays on Gitea and the rig, the numbers are in the record) and the rig's detached `~/src/colibri-m1-ds4` (24 Sep, an early DeepSeek serve version redone as `ds4_serve.cpp`; its commit `492af45` was on no branch, and is kept in `~/bench/leftover-m1-ds4.unmerged.bundle`, verified complete; the "two modified files" this line used to name were two untracked build outputs, `franken_decode_cpu` and `franken_decode_ds4`).
4. **GitHub:** each push of `hot-expert-tier` to the public fork needs the go-ahead and the secret scan.

## 7. Not in this plan (open, but not part of the decode items)

Prefill (handoff §4 items 3, 5, 6: the trunk GEMMs at 2.45 ms/token need the quality gate; chunk 2048; per-card utilisation ~50 %), the latent mixed-chunk `--gate-plan` bug (item 8), the Qwen / DeepSeek chunk and MTP items (item 9). **Closed by measurement, do not redo:** lookahead prefetch, MTP speculative decoding, the fused-counter split reduce (slower than the separate wave reduce), `--hip-graph` (no gain), a second stream for the small GEMV chain, the DMA-rate / fabric-clock theory.

## 8. Second round (plan Rev 101, 2026-10-08 night): what is left, in what order

**State.** Installed `franken_dec_glm` = `.d14` sha `f365fc7c014696ca` (decode ~60 ms a token, ~16.7-18.6 tok/s served; the trunk GEMVs are done, D7c took the big Q8_0 matrices to 752-768 GB/s of an ~870 ceiling). The critical expert fetch is ~35 ms of a token and runs at ~77 % of the 62-64 GB/s the three links deliver together. Nothing in this section changes the links; only fewer or better-split missed bytes do. A separate plan, `PREFILL-OPEN-ITEMS-PLAN-2026-10-08.md`, owns prefill (different gate: the quality gate, not bit-identity).

**V2 was added during the closeout read-through of the same night:** the handoff said the needle result needed no re-run because 'only decode code changed since'; the D14 fix changed the prefill, so that sentence was wrong.

**Why these items and not the others.** The two reasons are (1) every later measurement stands on the exactness gates, and D14 showed that a gate can be blind (the 2 200-token chunk-1024-vs-512 gates compared two chunked runs with the same last chunk, so both carried the same fault and agreed), and (2) the fetch side is 58 % of a token and its model is off by a factor of three, so any fetch lever sized with that model is unsized.

| ID | Item | Gain | Basis | Effort | Tier |
|---|---|---|---|---|---|
| V1 | **an external reference past 2 051 tokens.** The references `gpu5/g136_eager` and `g2200` are the engine's own dumps (`g2200` covers layers 0-3 only, 124 taps); after D14 the fixed GPU prefill equals the engine's own T = 1 path, and nothing ties either to the CPU arm (which computes every row separately) or to llama.cpp at that depth. Dump the CPU arm for the FULL model at a 2 300-token prompt (chunk 1; measure its speed on 128 tokens first, it may take hours: run it detached) and compare GPU chunk 1 / 512 / 1 024 against it | 0 ms; **verification** | the arms were bit-identical at every earlier gate (`long512u`, 124 taps maxabs 0); expectation: maxabs 0 after D14, and a non-zero tap names the next bug | half a day, mostly waiting | Sonnet writes the chain, me |
| V2 | **the needle on the installed `.d14` build.** The 'needle 8/8 to 200 k' result was measured on the 2026-10-06 build, before the D14 fix; the fix changes the prefill of every prompt past 2 048 tokens (three rows in four stop reading stale buffers), so 'long-context retrieval is intact' is unconfirmed for the build that is installed. Re-run the needle part of the quality chain on `.d14` (`QE_PARTS=needle ARMS=franken-glm`, handoff §5 'Long context', ~1 h for 30 k-200 k; it starts the engine through `serve_alt.sh`, so it runs under the same rig rules) | 0 ms; **verification** | the old result passed with the buggy prefill; a correct prefill should pass too, but this is the claim the owner relies on for long documents | 1 hour of chain, detached | me |
| D16 | the per-token `ms` vector of the timed loop (`--time`) and of the serving loop (`[serve-glm5]` prints only the request median): settles whether the 24-45 ms gaps between the argmax read-back and the next token (D2, D6) are real, and how often | 0 ms; a finding that sizes D17 | the serve A/B's disturbed arms (about 3 of 14, 5-15 % low, cards drawing less power: they waited) are the only unexplained loss left on the box | 1 hour + one 10-minute chain | Sonnet |
| D15 | **where do the other two thirds of the fetch model's gain go** (model -2.9 ms for `--fetch-assign optimal`, trace -0.97 ms): per layer and card, predicted against realised bytes and stage-kernel time, from the existing traces (`dtrace_*`, `crit_fetch.py`, `lane_model.py`), then one paired trace with the pattern and the optimal table. Candidates: slab granularity, the two cards behind one root complex slowing each other, the per-layer join waiting on the slowest card, placement hits moving the target | up to ~2 ms if the gap is recoverable; also re-sizes D9 and D10 | model vs trace; a factor 0.33 realisation was already measured | 1 day offline + one trace chain | me |
| D17 | *(only if D16 shows real host stalls)* find what the host does between the argmax read-back and the next token's first launch (read-back wait, scheduler, logging, the adapter, gateway round trip) and remove it | the stall share of the 30 % disturbed arms: 0 to 5-15 % on those arms | none yet | unknown | me |

**Order.** V1, V2 and D16 first (both small, both findings, neither moves speed); D15 next (offline first). Then the decision gate: **build D10 (the CPU lane, 1.3-2.7 ms, 2-4 %) only if D15 leaves the fetch at no more than the model's headroom and the owner still wants the next 2-4 %.** D9, D8, D13 stay parked under their stop rules (0.6-0.9 ms, <= 0.4 ms, ~0.1 ms). D11 and D12 stay parked.

**Gates and stop rules.** V1 ends with a table (config, taps, not exact) in the record; a non-exact tap is a bug until explained. V2 ends with the 8 needles (depth, found / missed, seconds) in the record; any miss that the 2026-10-06 build got is a bug until explained. D16's gate: the vector is printed for >= 3 runs; if no gap above 5 ms appears in 200 tokens D17 is dropped. D15's gate: the realised/predicted ratio per card for the two tables is explained within 20 %, or the item ends with 'not explained, here is what was excluded'. The section-5 stop rules stand; a speed claim needs the paired trace and the serve A/B in both orders.

**Expected total.** About 0 ms from V1 and D16, 0-2 ms from D15, 1.3-2.7 ms from D10 if built: **2-4 ms (3-6 %) at best, roughly 56-58 ms a token (~17-18 tok/s)** (estimate; the same sum as section 4, minus what the first round already took).
