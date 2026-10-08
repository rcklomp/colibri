# Plan for the open GLM decode items (2026-10-08) -- executed the same evening, outcomes in section 0

Written after the trunk-GEMV work (record §L5-GLM-TRUNK-GEMV, §L5-GLM-ROWSPLIT, §L5-GLM-GEMV-GROUP; decision log `FRANKEN-ENGINE-PLAN-2026-09-15.md` Rev 95-98).
This file says what is still open, what each item is worth, how it is gated, and in which order it should be done. It does not repeat the evidence: every number below points to a record section.
Labels: **measured** = a record row exists; **estimate** = derived here from measured numbers, with the derivation; **unknown** = no basis yet.

## 0. Status after the first execution (2026-10-08, evening; plan Rev 99 in `FRANKEN-ENGINE-PLAN-2026-09-15.md`)

The plan was executed in one session (Phases 0-1 and the part of Phase 2 that the gates opened). Outcomes, each with its record section (`ROME-3x7900XTX-2026-09-04.md`, new sections at its end) and the engine notes (franken-engine `GLM5.md` section 22):

| ID | Outcome | Gain (labelled) |
|---|---|---|
| D1 `--gemv-lds` | **closed, not enabled.** Slower in isolation (+1..7 %/launch), faster in the trace (-0.18 ms wall, SE 0.10) with a layer-period change of 4 us (gate: 10 us, 1 %); the flag is global (it also stages the prefill GEMMs). The same idea carried into a kernel with no dead generality is D7c below. | 0 (measured) |
| D2 slow `hc_*_fn` launches | **closed, nothing to fix.** The 22 % population belonged to the old `k_gemm_batch` hc fn and vanished with the row-split kernel (0.10 % of clean launches now); the rest is a clock ramp after host stalls that looks like a tracing artefact. | ~0.01 ms (measured) |
| D3 rows-based waves rule | **done**: `--gemv-rowsplit-waves 1`, bit-exact (model gate), launcher default. | -0.12 ms wall (SE 0.06), -0.18 ms kernel (measured, paired trace) |
| D4 Q6_K split stall | **done, fixed and verified.** Cause: a flat load below the LDS allocation in the staged-x path of the Q6_K branch (memory violation); an address-only fix; the unsplit head never hit it. | 0 ms (correctness) |
| D5 decode past 2 051 deep | **finding, new item D14.** Not the top-k selection or an atomic: a timing-dependent perturbation of layer 4's carried KDA conv window from decode step 2-3 on; not removed by serialising kernels and copies, identical at `--ctx` 2560 / 4096 / 8192 (so not an allocation-layout effect). | 0 ms (verification) |
| D6 odd serve-A/B arms | **done**: clock / temperature / power / CPU-frequency sampling beside every arm of `glm_serve_ab_chain.sh` (`SAB_SAMPLE`, default 1) and a per-request report with the correlations. | 0 ms (measurement quality) |
| D7 bandwidth gate | **GO, but not for the reason asked.** The 16-byte exact-order kernel (D7b) is slower than a plain copy of the engine's own loop, so D7b is dropped; the copy itself (`--gemv-q8fast`, **new item D7c**) is 17-21 % faster per launch than the generic `k_gemm_batch` and bit-exact. | **-1.49 ms wall (SE 0.08, -2.3 %), layer period 1 403 -> 1 373 us (measured, paired trace)** |
| D8 KDA o / DSA o tail | **parked** (<= ~0.4 ms above the copy, below the 0.3 ms stop rule once the exact-order constraint is counted). | -- |
| D9a / D9 tensor-level split | **D9 parked**: modeled +1.9..2.6 ms, but the model overstated the shipped whole-slab table threefold (realization 0.33), so +0.6..0.9 ms < 1.5 ms. | -- |
| D10 / D11 / D12 | unchanged, parked for the owner's word. | -- |

**New items.** **D13** (optional, ~0.1 ms): the same recipe for the Q6_K lm_head (one launch, 722 us a token). **D14** (correctness, before any further speed work at depth > 2 051): find the root cause of the carried-state perturbation of D5 -- next experiments are listed in record §L5-GLM-D5 (the `--ctx` variation is done and changed nothing; a read-back of layer 4's conv window after every step, a poison-fill of the scratch buffers, a dense-attention bisect). **D15** (sizing): where do the other two thirds of the fetch model's modeled gain go (the model said -2.9 ms for `--fetch-assign optimal`, the trace -0.97 ms)? Every fetch-side lever is sized with that model, so finding the gap comes before building any of them.

**Where decode stands after this session** (the section-1 numbers are the starting point): **installed `~/bench/franken_bin/franken_dec_glm` = `.fin` sha `ca2bc21c19b451c0`** (franken-engine code `f0fda34`, main `96ff255`; rollbacks `.gr` `1f95c34b2d8bbf06`, `.rs`, `.adm`); the specialised GEMV and the waves rule take ~1.6 ms off a ~63.8 ms traced token (-1.49 ms and -0.12 ms, two separately measured effects, paired kernel trace). Accepted: `accept_live` PASS (warm new chat 0.86 s, follow-up 0.9 s, behind an abandoned request 0.8 s), browser first token 1.06 / 1.05 s. **The serve-path A/B is supportive, not clean:** six of eight arms agree within groups and give +2.8 % on every request, two arms are odd in opposite directions (a cold first arm +4..7 %, an arm whose cards waited -5..-14 %) and with all eight the mean difference is +0.1 % (record §L5-GLM-D7, §L5-GLM-D6); a second A/B with a discarded warm-up arm is addended there. The service is still OFF behind the reservation flag.

## 1. Where decode stands

- Installed `~/bench/franken_bin/franken_dec_glm` = `.gr` sha `1f95c34b2d8bbf06` (rollbacks `.rs` c21f9a66, `.adm` c97b0eb2); franken-engine main `68e31cd`; service OFF behind `~/bench/.dev_reserved`.
- Decode token: ~61 ms at depth ~8.3k with the row-split build (measured, shipped-adaptation arms of the speed A/B); the group build takes ~1.6 ms more off (trace; **estimate** ~59-60 ms, ~16.8 tok/s, not re-measured in the gate); served short chats 16.3-18.1 tok/s (measured, group build). The trace (rocprofv3, inflated) shows a MoE layer period of ~1.40 ms after the trunk-GEMV work (was ~1.48).
- Where the time is (record §L5-GLM-DECODE-TRACE, trace-based): the critical expert fetch (the slowest of the three cards' stage kernels) is ~35 ms of a ~62 ms token and runs at ~77 % of the 62-64 GB/s the three PCIe links deliver together; the rest is the trunk, now ~3.3 ms (5 %) shorter than on 2026-10-07 morning. GEMV kernel time a token: 20.0 -> 18.0 ms.
- **No target number has been set by the owner.** The items are ranked by expected gain per effort and carry stop rules (section 5), so the work ends when the remaining candidates are not worth their cost.

## 2. How every item is measured (rules from the last two days; do not re-derive)

1. Anything that claims "no bit moves" passes the exactness gates first: `glm_group_micro_chain.sh`-style microbenchmark against main's own backend where a kernel changes, then a model gate with `oracle_verdict.sh` (maxabs 0 on every tap; chunk 1 + 8 greedy against `gpu5/g136_eager`, 64 greedy at depth 1 500 against an in-process flag-off dump, with the flag on/off crossed with rowsplit, adaptation every token, `--gemv-lds`, a non-scoring context). Gates run below 2 051 tokens of depth (item D5 is about that limit).
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
| D12 | validate the DeepSeek adapter position fix (branch `adapt-ds4`, franken-engine `589e855`, Gitea only, untested) | only if DeepSeek is served | recipe in handoff §4 (two fresh `franken_decode_ds4` builds, `ds4_gpu_gate.sh`, then `glm_serve_ab_chain.sh` with `SAB_START` / `SAB_PGREP`) |

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
3. **Two worktrees that are not from this work:** the Mac's `perf/x1-retained-cmdbuf` (its commit is not in `hot-expert-tier`) and the rig's detached `~/src/colibri-m1-ds4` (two modified files): keep or delete?
4. **GitHub:** each push of `hot-expert-tier` to the public fork needs the go-ahead and the secret scan.

## 7. Not in this plan (open, but not part of the decode items)

Prefill (handoff §4 items 3, 5, 6: the trunk GEMMs at 2.45 ms/token need the quality gate; chunk 2048; per-card utilisation ~50 %), the latent mixed-chunk `--gate-plan` bug (item 8), the Qwen / DeepSeek chunk and MTP items (item 9). **Closed by measurement, do not redo:** lookahead prefetch, MTP speculative decoding, the fused-counter split reduce (slower than the separate wave reduce), `--hip-graph` (no gain), a second stream for the small GEMV chain, the DMA-rate / fabric-clock theory.
