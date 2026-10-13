# Plan: a quality benchmark for rome (2026-10-13; QB items)

> **Status (2026-10-13): planned, not started.** The owner's order: when the Franken engine has run out of speed levers, the project gets a benchmark that (a) compares Franken settings and model quantisations against each other, and (b) compares the model with other LLMs from every supplier on the standard capability benchmarks (MMLU-Pro, GPQA, AIME, LiveCodeBench, SWE-bench, IFEval, BFCL, Arena-style preference). Items QB0-QB9 below; the order of work is ACTION-LIST.md (rows QB*). Facts about benchmarks, harnesses and vendor protocols come from a web read of 2026-10-13 (sources at the end); recheck versions before running.

## 0. Why the current test is not enough

The 70-question MMLU-Pro sample + needle (`franken_quality_chain.sh`, used for every quality gate since 2026-09-23) did its job: it catches a broken build. It cannot do either thing the owner now asks for:

1. **It cannot see differences under ~10 points.** 70 items: the 95 % interval of a 78 % score is about [68, 87]. 55, 56, 58 and 59 / 70 (the runs of `.pf15`, `.d10`, IQ4_XS, `.pf2c`) are the same number.
2. **Most of its misses are the answer budget, not the model.** Every run on record lost 6-11 of 70 items to the 16 000-token cap (`.d10`: 9 of its 14 misses; Qwen3.8 through llama.cpp: 7 of 14). The served cap is the same (`FRANKEN_MAX_TOKENS=16384`), so hard chat questions are cut off the same way.
3. **It does not use the vendor protocol, so it cannot be put beside published numbers.** It decodes greedily at `reasoning_effort=xhigh` with a 16 k cap. Z.ai's protocol for GLM-5.x is temperature 1.0, top_p 0.95, `reasoning_effort=max`, output up to 128-164 k tokens (S1-S3, S6).

## 1. Constraints that shape everything

- **Throughput is the binding limit.** One request at a time (`--kv-slots 1`), ~22-25 tok/s, i.e. **~80 k output tokens an hour**. A reasoning model at `effort=max` writes thousands of tokens a question (MMLU-Pro at `xhigh`: mean 3 400, median 970, max 16 000 = the cap). Artificial Analysis spent 181 M output tokens on this model's index run (S9): ~95 days here. Every suite below is sized against this; QB1 is what changes it.
- **Comparability with vendors needs the same benchmark version, item set, prompt, sampling, output cap, number of samples and (for judged benchmarks) the same judge model.** Z.ai publishes no MMLU-Pro / GPQA / AIME / LCB / IFEval / BFCL numbers for GLM-5.3-Flash (its card shows agentic benchmarks only, S1); the nearest vendor numbers are GLM-5.2's (S6). The reliable anchors are the boards that run every vendor's models under ONE protocol: Artificial Analysis (standalone GPQA-D, IFBench, HLE), Epoch AI (GPQA-D, Mock AIME; harness and prompts published), BFCL (its own runner), Scale's SWE-bench Pro (locked protocol) (S10, S17, S36, S33).
- **Judged benchmarks need a frontier judge.** HLE, Arena-Hard-Auto, AlpacaEval and MT-Bench are scored by a GPT / Gemini judge; a local judge makes the number incomparable. That means an API key, money, and sending the model's answers (not private data) to a vendor: owner decision O-QB1.
- **Statistics** (S53; binomial arithmetic in the research notes): to separate two of OUR configurations by 3 points, run them **paired on the same items** (McNemar); with ~10 % of items flipping between configs that needs ~900 items, with 5 % ~450 (95 % confidence, 80 % power). A vendor-comparable single number on a small benchmark (GPQA-D 198, AIME 30) has an irreducible item-sampling error of ±6 / ±14 points (95 %, at a score near 78 %), whatever the number of repeats; say so next to every such number.
- **Hardware:** x86_64, 247 GiB RAM, one NVMe; SWE-bench needs Docker images (Verified ~30 GiB optimised, Pro V2 more) and runs its tests on the CPU while the GPUs decode: the CPU lane (D10) shares those cores (`FRANKEN_GLM_CPU_LANE=0` for agentic runs, or pin the containers away from cores 3-7 and 0).

## 2. Two suites

### Suite C: "Compare" (our configurations against each other)

Purpose: settings, kernels, quantisations, the CPU lane, a new model build. Paired, same items, same seeds, same protocol in both arms, so the question is only "did this change move quality".

| Layer | What | Size | Cost | Detects |
|---|---|---|---|---|
| C0 | Offline numerics: KL divergence / top-1 agreement of the final logits against a reference on a fixed text (the PF14 method), the CPU-arm cosine (V1 method) | 20 k tokens | minutes | numerics changes; cannot rank task quality (routing near-ties, record §L5-PF2c item 7) |
| C1 | IFEval (541) + GPQA-D (198) x 1 + an MMLU-Pro stratified subset (~600 items, frozen, chosen once from the 12 032 with a fixed seed, all 14 categories) | ~1 340 items | ~2.5 days per arm at 1 slot (~5 M tokens; a paired comparison runs two arms; to measure in QB0); about a day per arm at QB1's 1.5x floor | ~4-5 points paired |
| C2 | C1 + LiveCodeBench v6 subset (frozen ~200) + the needle set (30 k-200 k) | | +~1.5 days per arm | adds code and long context |

Protocol for C: the vendor sampling (temperature 1.0, top_p 0.95, effort max) with a FIXED seed per item, and a generous cap (64 k, see QB2), so truncation stops being the dominant error. Both arms in the same rig session, interleaved by item block (A B B A), so drift cancels; each switch is an engine restart (~6.5 min warm; never two engines at once), so the blocks are large enough that restarts stay under ~10 % of wall time. Verdict: McNemar on the paired items + the per-benchmark difference with its interval; a change ships when it is not worse beyond the pre-registered margin (O4's role, now with power).

### Suite V: "Vendor-comparable" (us against other LLMs)

Purpose: one number per benchmark that can stand next to a published number for Claude / GPT / Gemini / DeepSeek / Qwen / GLM. Each benchmark is run in the protocol of the board whose numbers we compare with, and the report names that board, its version and its settings.

| Benchmark | Version / size | Compare with | Protocol to copy | Est. tokens (to measure in QB0) | Judge? |
|---|---|---|---|---|---|
| GPQA Diamond | 198 | Epoch AI hub (16 runs), AA standalone, vendor cards (GLM-5.2 91.2) | Epoch's Inspect task (0-shot CoT, strict `Answer: X`), model's default sampling; 8 repeats | ~200 x 8 x ~10 k = 16 M | no |
| AIME 2026 (+2025) | 30 (+30) | vendor cards (GLM-5.2 99.2), MathArena (4 runs), Epoch Mock AIME | integer answer, math-verify; avg@8 | ~60 x 8 x ~15 k = 7 M | no |
| MMLU-Pro (fixed 2026-01-18 version) | 12 032 (or the frozen 600 of C1 for a partial) | NVIDIA / vendor cards, Vals.ai | TIGER-Lab 5-shot CoT or 0-shot as the anchor did; 1 run | 12 032 x ~3.4 k = 41 M (partial: 2 M) | no |
| LiveCodeBench v6 | 1 055, or the vendor window (2025-02..05) | Qwen / NVIDIA cards (window stated) | lighteval `lcb:codegeneration` or NeMo recipe; avg@4-8 | ~300 x 8 x ~12 k = 29 M (window) | no |
| IFEval + IFBench | 541 + 300 | vendor cards; AA IFBench (5 repeats, loose) | strict + loose; IFBench 5 repeats | ~2 M | no |
| BFCL v4 | ~5 000 entries | the BFCL leaderboard (all vendors, same runner) | `bfcl generate --skip-server-setup` against our endpoint; web-search category needs a search key | ~5 M (short answers) | no (web search: an API) |
| HLE text-only | 2 158 | AA (1 run), vendor cards | the vendor judge (GPT-5.x) | ~2 158 x ~15 k = 32 M | **yes** (O-QB1) |
| SWE-bench Pro V2 (or its HARD 51) | 642 / 51 | Scale's leaderboard, vendor cards (GLM-5.2 62.1 on v1) | Harbor, offline agent phase, 50-min budget a task, mini-SWE-agent or OpenHands | agentic: tens of steps x thousands of tokens a task; HARD 51 first | no |
| Arena-style preference | Arena-Hard-Auto v2 (500 + 250) | LMArena's ordering (proxy only) | the official config's judge (Gemini-2.5 / GPT-4.1) | ~750 x ~5 k = 4 M | **yes** (O-QB1) |

Not planned: BBH, GSM8K, MATH-500, HumanEval / MBPP (saturated: every frontier model is near the ceiling, they no longer separate models; one evalplus pass can be added as a smoke if wanted); SWE-bench Verified (OpenAI stopped reporting it, contaminated; Pro replaced it, S32); MT-Bench (saturated); Chatbot Arena itself (needs human votes, cannot run locally).

At 80 k tokens an hour Suite V as tabled is ~135 M tokens = **~70 days**. With QB1 (concurrent requests) and the partials marked above it becomes a few weeks of rig time, run in pieces; the priority order is GPQA-D, AIME, IFEval/IFBench, LCB window, BFCL, MMLU-Pro partial, then the judged and agentic ones after O-QB1.

## 3. Items

| ID | Item | Gate / stop rule | Effort | Tier |
|---|---|---|---|---|
| QB0 | **Pilot and measure.** On the current build: 20 GPQA-D + 20 AIME + 50 IFEval + 50 MMLU-Pro at the vendor protocol (temp 1.0, top_p 0.95, effort max, cap 64 k), record tokens / item, wall / item, truncation rate, the repeat-to-repeat flip rate q (rerun the same items with another seed). Turns every estimate in section 2 into a measured number; sizes C1 (items needed at the measured q). | done when the tokens-per-item and q table exists in the record | 1 day rig | Sonnet harness, orchestrator runs |
| QB1 | **Throughput: concurrent requests.** The engine serves one slot; a decode token is bound by expert fetches that several requests would SHARE (they hit the same layers; a missed expert is fetched once for all). Measure what N = 2 / 4 / 8 concurrent decode streams give in aggregate tok/s (engine work: KV slots > 1 and a batched decode step; the GEMVs become small GEMMs). This is the only lever that makes Suite V weeks instead of months, and it is a serving win for parallel users too. | stop if the aggregate gain at N = 4 is < 1.5x; a multi-day engine item | multi-day | Opus (kernels), Sonnet (serving) |
| QB2 | **Answer budget and sampling for the served model.** Rerun the 70-item sample (and QB0's MMLU part) at caps 16 k / 32 k / 64 k and greedy vs the vendor sampling; inspect the truncated answers (repetition loop or real reasoning). If the score rises at no cost to normal chats, raise `FRANKEN_MAX_TOKENS` and set the vendor sampling as the gateway default. The cheapest quality gain on the table. | done when the table (cap x sampling: score, truncations) is in the record; the served default (cap, sampling) changes only after O-QB3, and only if truncations fall and short-chat latency is unchanged | 1-2 days rig | Sonnet |
| QB3 | **Harness.** One runner on the Mac (or the rig) against the gateway's OpenAI-compatible endpoint: Inspect (`inspect_evals`: gpqa, aime2025/2026, mmlu_pro, ifeval, bfcl, swe_bench) or lighteval (`lcb:codegeneration`, ifbench), the NeMo Evaluator recipe as the reference for settings. Reasoning separated from the answer (`reasoning_content`, or strip `</think>`), sampling and `reasoning_effort` passed through, 1-hour request timeout, resumable (one result line per item, like `franken_quality_chain.sh`), results in `~/bench/qbench/<suite>/<build>/`. Check that the gateway passes temperature / top_p / seed / reasoning_effort and a > 16 k cap per request. | a 20-item smoke of every benchmark returns parsed answers | 2-3 days | Sonnet |
| QB4 | **Suite C** frozen: the item lists (seeded subsets) committed, the chain `qbench_compare_chain.sh A B` (interleaved blocks, both arms in one session), the verdict script (McNemar + per-benchmark intervals). First use: IQ4_XS vs the IQ3_XXS hybrid, and the D10 lane on vs off -- the two comparisons the owner asked about on 2026-10-13. | the IQ4_XS-vs-hybrid comparison reported with intervals | 2 days + rig time | Sonnet |
| QB5 | **Reference arm.** The same suites through llama.cpp on the same GGUF (engine-independent, as `qwen38_llama` was), on the C1 subset only: separates "the engine" from "the quant". | C1 numbers of both engines inside each other's interval, or a finding | rig time | Haiku runs it |
| QB6 | **Suite V, non-judged** (GPQA-D x8, AIME 2025 + 2026 x8, IFEval + IFBench, LCB window, BFCL v4 without web search, MMLU-Pro partial) in the protocols of section 2, plus the comparison table: our numbers beside the published ones (board, version, settings, date, interval). | the table exists and every row names its anchor | weeks of rig (fewer after QB1) | Haiku runs, Sonnet reports |
| QB7 | **Suite V, judged** (HLE text-only, Arena-Hard-Auto v2) | only after O-QB1 | rig + API budget | Sonnet |
| QB8 | **Agentic coding** (SWE-bench Pro V2 HARD 51 first, then the full 642 if wanted): Docker images on the rig, an agent scaffold (mini-SWE-agent; OpenHands as Z.ai uses), the locked 50-minute budget, CPU lane off or containers pinned away from the lane cores | after QB6; O-QB2 for the disk and the days | days | Sonnet |
| QB9 | **Upkeep.** Benchmarks age (AIME yearly, LCB windows, SWE-bench Pro versions, MMLU-Pro fixes) and leaderboards change protocol (AA index versions). Before each Suite V run: recheck versions, anchors and settings; never compare across versions. | each run's report states versions | an hour a run | Sonnet |

Order: QB0 and QB2 now-ish (cheap, and QB2 may be a real quality gain for the owner today); QB3; QB4 (Suite C, answers the IQ4_XS question with power); QB1 when the speed track is done (it is itself a throughput lever); then QB5 and QB6 (no owner decision); QB7 after O-QB1; QB8 after O-QB2 and QB6.

## 4. Owner decisions

- **O-QB1:** a paid API judge for HLE and Arena-Hard-Auto (a GPT-5.x / Gemini key; cost a few tens of USD a run; the model's answers to public benchmark questions are sent to that vendor). Without it, HLE and the Arena proxy are left out of Suite V (or run with a local judge and marked "not comparable").
- **O-QB2:** disk and rig days for SWE-bench Pro (tens to ~200 GB of Docker images; days of rig time even after QB1).
- **O-QB3:** whether the vendor sampling (temperature 1.0, top_p 0.95, effort max) and a larger answer cap become the served defaults (QB2's outcome), since they change what the owner sees in chat.

## 5. Rules for every QB run

- The protocol is written down BEFORE the run (benchmark version, item list, prompt, sampling, cap, repeats, judge) and printed in the result header; a number without its protocol is not reported.
- Our-vs-our comparisons are paired on the same items and seeds, in one session, interleaved; our-vs-vendor numbers carry the anchor's name, version and date, and the item-sampling interval.
- Every run is resumable and watched (CLAUDE.md "no rig job runs unwatched"); the gateway is under the reservation rules; `accept_live` after a run that touched the service.
- Results: one record section per run at the end of `ROME-3x7900XTX-2026-09-04.md`; the result files under `~/bench/qbench/`.

## Sources (web read 2026-10-13; recheck before use)

S1 https://huggingface.co/zai-org/GLM-5.3-Flash -- S2 https://docs.z.ai/guides/llm/glm-5.3-flash -- S3 https://unsloth.ai/docs/models/glm-5.3-flash -- S6 https://huggingface.co/zai-org/GLM-5.2 -- S9 https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-gemini-3-7-flash -- S10 https://artificialanalysis.ai/methodology/intelligence-benchmarking -- S17 https://epoch.ai/benchmarks -- S20 https://github.com/EleutherAI/lm-evaluation-harness -- S24 https://github.com/huggingface/lighteval -- S30 NVIDIA NeMo Evaluator reproducibility recipe (https://github.com/NVIDIA-NeMo/Evaluator) -- S32 https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/ -- S33 https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro -- S34 https://github.com/scaleapi/SWE-bench_Pro-os -- S36 https://gorilla.cs.berkeley.edu/leaderboard.html -- S38 https://github.com/ShishirPatil/gorilla -- S45 https://github.com/lmarena/arena-hard-auto -- S51 https://github.com/UKGovernmentBEIS/inspect_evals -- S53 https://arxiv.org/abs/2411.00640 (error bars for evals).
