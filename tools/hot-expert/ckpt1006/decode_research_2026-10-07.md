# Decode-lever research: host-resident-expert MoE decode on 3x 7900 XTX, no P2P, EPYC 7F32 (outside sources only)

Date 2026-10-07. Research only; no machine touched. Labels: [M] = measured by the source on stated hardware, [C] = claimed / abstract-level / relayed by a secondary summary, [I] = my inference for this box.
Tools that failed: X reader (python3 -I x_read.py: missing module browser_cookie3 -> not used); Reddit RSS (429 rate limits, mostly noise; only two useful pointers); reddit.com itself blocked. arXiv full texts of FreeToken/Fiddler/HybriMoE read through WebFetch summaries (so numbers are "as relayed").

Box facts used for judgements: expert S ~ 10-20 MB (GLM-5.x int4 experts are ~20 MB per colibri#537; GLM-5.3-Flash is smaller, so use 10-20 MB); at ~25 GB/s one expert = 0.4-0.8 ms per link; 42 layers.

---------------------------------------------------------------------------------------------------

## 0. Headline conclusions (ranked by actionability)

1. FIRST measure, standalone, the concurrent behaviour of the three links and the CPU: (a) 3 simultaneous pinned H2D streams of 10-20 MB chunks, one per card; (b) the same plus an 8-thread IQ4_XS GEMV on the host. Every source that got a CPU/GPU split right (FreeToken, colibri dual-lane) measured the *concurrent pair*, not independent bandwidths. If (a) does not scale to ~3x25 GB/s the limiter is not the links (IO die / root complex / NUMA placement / IOMMU / pinned-buffer placement), and a CPU share would compete for the same DRAM.
2. CPU-executes-a-share is real but its payoff on this box is probably small [I]: FreeToken's rule q* ~ m*BP/BH says the CPU share is 1-BP/BH; with three links BP_agg (up to ~75-85 GB/s) is >= the 20-40 GB/s DRAM headroom you quote, so q* -> m (CPU share ~0). The one place a CPU lane can still help is as a 4th balancing lane that absorbs the straggler expert of the slowest card (see 2.4).
3. Zero-copy GPU kernel reads of host memory are a measured dead end on the only platform that measured it (2.8 GB/s effective vs 56 GB/s bulk DMA). Keep DMA.
4. Cheap hygiene with measured wins elsewhere: register/pin the whole host store and DMA straight from it (no CPU staging memcpy); one contiguous block per expert (gate|up|down) so one copy per expert; use hipMemcpyBatchAsync only for many small copies (per-submission SDMA cost 9-13 us).
5. Continuous/finer splitting across links: NO published measurement found. The only structural argument is granularity arithmetic (section 1) [I].
6. Resident-set policy still has a measured lever: LFU with half-life ~16 beat LRU; swap hysteresis/dwell turned 0.84x into 1.21x in one llama.cpp PR; Belady bound says ~34% of misses avoidable offline. Per-layer non-uniform slot allocation does NOT hold out.
7. ROCm: single 7900 XTX measured 28.4 GB/s H2D / 29.0 D2H / 48.2 bidirectional, so ~25 GB/s practical is ~88% of what the link gives. No published SDMA=0 or 3-card-concurrency numbers for RDNA3 exist that I could find.

---------------------------------------------------------------------------------------------------

## 1. Splitting the per-layer missed-expert fetch across links

Found: no source measuring fine-grained / tensor-level / continuous split of host-resident experts across several PCIe links on consumer cards. Literature either does expert-parallel on NVLink boxes (EPLB, DeepSeek style) or single-GPU offload. Honest negative.

Related measured/claimed items:
- ThunderEP (arXiv 2609.40093, https://arxiv.org/abs/2609.40093) [C, abstract only]: expert-parallel comms for PCIe consumer GPUs where all inter-GPU traffic goes through CPU memory; uses DMA engines instead of SMs, removes relay hops; 2.00x dispatch, 1.53x combine over NCCL, up to 1.66x end-to-end on 2x RTX 4090/5090. Applies in spirit (no P2P, host-staged), but it is about activations, not weight fetch. Abstract gives no copy-engine counts.
- ddvnguyen/llama.cpp#130 (https://github.com/ddvnguyen/llama.cpp/issues/130) [M]: RTX 3060 on PCIe 4.0 x4 (6.10 GB/s achievable) next to a second card: "displacement coefficient 1.00 (the fabric is serial)"; owner rejected "PCIe root-port move" because it moves bandwidth rather than adding it. Lesson for you: verify that the three cards' DMA actually overlap (different root ports/IOD quadrants); their rig did not.
- Granularity arithmetic [I]: whole-expert assignment gives per-layer fetch makespan = ceil(m/3)*S/BP. One extra expert on one card costs 0.4-0.8 ms for that layer, i.e. up to 17-34 ms/token if it happens on every layer. A row-slice split (each card takes 1/3 of an expert's intermediate neurons: gate/up rows + matching down columns) makes the split continuous; the price is a partial-sum reduction of hidden-size vectors (16-24 KB) via host (no P2P) = one extra sync hop per layer. Break-even depends on your per-hop latency; not measured anywhere. Compare against simply giving the straggler expert to the CPU lane (2.4).
- Small-copy overhead (rocm-systems PR 11614, https://github.com/ROCm/rocm-systems/pull/11614) [M, MI355X]: per SDMA submission 9-13 us independent of size; fused batch of 512 x 4 KB copies 9.00 -> 0.51 us/op H2D; 64 KB: 10.88 -> 1.44; 1 MB: 28.11 -> 18.46; saving ~9-10 us per operand. For 10-20 MB copies that is ~1-2% of the copy, so splitting each expert into 3 tensor copies x ~3 cards costs little; splitting into hundreds of slices would matter, then use hipMemcpyBatchAsync (1D already fused in ROCR; 3D rect fusion is an open PR).
- FloE (arXiv 2025, relayed in the Strata digest, https://github.com/xenodeve/Strata-xeno/issues/186#issuecomment-5991404072) [C]: naive expert transfers reach only a fraction of PCIe bandwidth because they span non-contiguous blocks. Action: gate|up|down of one expert in one host block (FreeToken's FTW format does exactly this: 4096-aligned expert banks, flat id l*E+e).
- CUDA-only caveat from FreeToken (relayed, https://github.com/justinchuby/onnx-genai/issues/1759) [C]: cudaMemcpyBatchAsync on registered host memory silently degrades to synchronous copies if a batch mixes in sub-256 KB entries. Unknown for HIP; check hipMemcpyBatchAsync behaviour before trusting it.

---------------------------------------------------------------------------------------------------

## 2. Running a share of missed experts on the CPU

### 2.1 Fiddler (https://arxiv.org/abs/2402.07033, https://arxiv.org/html/2402.07033v3)
[M by paper] HW: Quadro RTX 6000 24 GB + Xeon Gold 6126 48 cores, PCIe 3.0 x16 (32 GB/s); and RTX 6000 Ada + Xeon 8480+ 112 cores, PCIe 4.0 x16. Microbenchmark: weight copy CPU->GPU is "2-5x longer than the actual computation time"; CPU expert latency grows linearly with batch, GPU flat. Rule: run on GPU iff cpu_lat(s) > gpu_lat(s) + transfer_lat. Result 1.26x over llama.cpp single batch (AVX512_BF16 kernel, many cores). Applies: rule yes, numbers no (48-112 cores, AVX512-BF16, bf16 weights, Mixtral-size experts).

### 2.2 FreeToken (arXiv 2608.16157, https://arxiv.org/abs/2608.16157, code https://github.com/FlashML-org/FreeToken) -- the "FreeToken" you asked about exists
[C, via WebFetch of arXiv html and the onnx-genai study]
- Policy q* ~ m*BP/BH: of m missed experts per layer, q* are fetched over PCIe into the GPU cache, the remaining m-q* run on the CPU straight from the pinned host bank; partial sums added (exact, no model change). Derivation: DMA and CPU GEMV share the host memory bus, so CPU gets residual BH-BP.
- Reported platform table (GB/s): RTX 5090 server BP 52.7 / BH 77.3 (Xeon Gold 6459C, 32 thr); RTX 4090 BP 25.1 / BH 63.2 (Xeon 8358P, 32 thr); RTX 4060 laptop BP 11.8 / BH 47.5 (i9-13900H). Desktop 5090 row BP 49.0 / BH 53.8 -> q*->m, i.e. CPU share ~9%.
- Reported results: Qwen3.6-35B-A3B decode 77-83 tok/s on RTX 5090; "1.8-2.3x and 1.5-1.9x over strongest baseline"; 4060 laptop 35B NVFP4 39.3 tok/s (secondary source). 5090 hit rate 84% (16% misses; 39% on DeepSeek-V4-Flash). Note the 2.1x on the 5090 desktop is credited to LRU+pipelined prefetch, NOT to q*.
- Implementation details that transfer (from the onnx-genai study of the code): measure the concurrent pair (pcie_ov/(pcie_ov+cpu_ov)) instead of two independent bandwidths; ratio stored as Q16 fixed point so GPU and CPU agree bit-exactly; always fetch at least one expert (else cache never warms); issue the CPU branch FIRST, then the GPU path; CPU workers pinned to physical cores (SMT siblings add no bandwidth); global LRU across layers; CPU branch captured in the CUDA graph via host-function nodes.
- Judgement: transfers as a method; the benefit scales with (BH-BP)/BP, which on this box is near zero or negative unless the three-link aggregate is much lower than 75 GB/s.

### 2.3 Measured dual-lane experiment (best direct evidence of the CPU-vs-DMA tradeoff)
colibri#537 follow-up (https://github.com/JustVugg/colibri/issues/537, first comment) [M]. HW: 6x RTX 5090, dual Xeon Silver 4510, 251 GB DDR5, GLM-5.2 int4 (~20 MB experts).
- Microbench, DDR pull: CPU alone 70 GB/s; PCIe5 DMA alone 56; concurrent 50:50 -> CPU 70 + DMA 37 = 108 GB/s; 75:25 -> 77 + 42 = 119 (ceiling ~120).
- End to end (baseline CPU-only 8.25 tok/s): pageable cudaMemcpy 4.57 (staging memcpy burns DDR twice); zero-copy kernel reads 1.71; double-buffered pinned DMA 8.16 (parity); persistent lane worker 8.01 ("cancellation": 13% of rows offloaded, CPU leg fell 71 -> 62 GB/s, gain and loss cancel to the millisecond); cudaHostAlloc arenas 6.48 (loses NUMA interleave, CPU leg 38 GB/s).
- Their crossover: offload fraction must exceed ~0.35-0.45 to win; they could only reach 13% (register budget). Wall 2: cudaHostRegister inflated anon RSS ~0.57x (unexplained). Wall 3: mbind interleave is a no-op on driver-owned pinned memory.
- Different regime from yours (their CPU is the main compute lane and the GPUs are the helper); the symmetric lesson is that a helper lane that costs the main lane DRAM bandwidth cancels out. On your box the DRAM is shared by the PCIe fetch itself.

### 2.4 CPU expert GEMV throughput on AVX2 (what is and is not known)
- Strata (https://github.com/xenodeve/Strata-xeno/issues/187 body) [M]: Intel hybrid CPU, AVX2 maddubs i-quant rows (IQ2_XS/IQ2_S/IQ2_XXS/IQ1_M), no AVX-512: ~6.7 GB/s of weights per core, 18.9 GB/s aggregate with 13 workers; "weights run as fast out of RAM as out of cache" => compute-bound on one core, not memory-bound; AVX-VNNI for i-quant rows bit-exact but NULL (0.98-1.03x); software prefetch 0-3%; SIMD mode generic/AVX2/VNNI +-4%; CPU experts = 48% of a 41.9 ms decode round (20.1 ms); pool workers only 58% busy, ~25% of pool time is waking; hard pin vs unpinned thread: 43.8 vs ~27 GB/s.
- ik_llama.cpp [C, community]: row-interleaved repacking took IQ2_XS prompt processing 47.46 -> 149.02 tok/s (3.14x) on identical Zen4 CPU (prompt processing, not TG). Same discussion (https://github.com/ikawrakow/ik_llama.cpp/discussions/164): on AVX2 token generation shows much smaller speedups than PP (Q4_K 8B, 1.15x TG vs 1.78x PP) i.e. TG is mostly memory-bound at low thread counts. NOTE: the summarizer quoted "2 threads: IQ4_XS 13.58 t/s on 4.13 GiB" for a Ryzen 7950X; I did not verify it (implies >50 GB/s on 2 threads, treat as unverified).
- KTransformers (https://github.com/kvcache-ai/ktransformers/releases/tag/v0.5.3, docs AVX2-Tutorial.md) [C]: AVX2-only MoE kernels since v0.5.3 but ONLY for BF16, FP8, GPTQ-INT4 (not GGUF IQ4_XS), recommends threads = physical cores; tutorial gives no tok/s; says memory bandwidth is the bottleneck on AVX2 systems. No Zen2 numbers anywhere. Its Expert Deferral (SOSP25, https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) gives +1.45x for <=0.5% accuracy drop but is NOT bit-exact (defers some experts' contribution), CPU kernel is AMX.
- HybriMoE (arXiv 2504.05897, https://arxiv.org/html/2504.05897) [M by paper]: RTX A6000 + Xeon Gold 5220R restricted to 10 cores; rules: GPU executes cached experts, higher-load first; CPU executes uncached experts, lower-load first, dynamic intra-layer balancing; MRS (minus-recent-score) cache policy S = a*TopP(s)+(1-a)*S with p = 2x activated experts; decode 1.70x over kTransformers (decode 0.21 s -> 0.11 s ablation). Paper does not publish per-expert CPU-vs-PCIe latency.
- No paper reports GB/s per core for 3-4 bit MoE GEMV on Zen2. My estimate [I]: IQ4_XS decode on AVX2 is cheaper than the IQ2 family (nibble + pshufb LUT), so maybe 4-8 GB/s/core => 8 cores ~30-50 GB/s ideal, realistically 20-35 with host threads for 3 GPUs also needing cores. That is 0.4-1.0 ms for a 10-20 MB expert on ALL 8 cores, i.e. roughly equal to one card's fetch time for the same expert. So a CPU lane can absorb about ONE expert per layer if (a) cores are free and (b) DRAM is not saturated by the three fetches.

### 2.5 Decision rule distilled for this box
Per layer, with t_card(i) = estimated finish time on card i and t_cpu = n_cpu*S/BH_conc: assign the misses greedily (largest remaining time first) to whichever of {card0, card1, card2, CPU} finishes earliest, with the CPU lane capped at floor(q) experts where q is calibrated from the CONCURRENT measurement (fetch always >=1 per layer if you want to keep warming the cache). The rule beats q* here because with three links the bound is the straggler, not the aggregate. [I]

---------------------------------------------------------------------------------------------------

## 3. Other levers (not in the closed list)

3.1 Zero-copy GPU reads of pinned host memory -- measured NEGATIVE.
- colibri#537 v2 [M, CUDA, PCIe5]: demand-fetched kernel reads over PCIe ~2.8 GB/s effective vs 56 GB/s bulk DMA ("20x slower than the link"), end-to-end 1.71 tok/s vs 8.25 baseline.
- FreeToken study notes zero-copy works on Linux H200 (no aperture limits, bit-exact to 6.8 GB/step) and "hybrid wins ~8x" over a Windows paging path, but that compares against a 7.8 tok/s OS fallback, not against DMA, and on WDDM it measured ~5.9 GB/s. Zero-copy occupies the same PCIe link as DMA and buys no residency.
- HIP docs: zero-copy memory is coherent => not cached by the GPU by default; "good when accesses are infrequent (perhaps once)". An expert is read exactly once per token, which is the favorable case in principle, but nobody measured gfx1100 with a deeply pipelined (many waves, 16-byte loads) kernel. Possible 1-hour experiment, expectation low [I].

3.2 Remove the CPU staging memcpy (register the host store and DMA from it).
- exllamav3 PR 341 (https://github.com/turboderp-org/exllamav3/pull/341) [M]: RTX 4090 + Ryzen 9 7900X: removing the stager memcpy (11-18 GB/s) and DMAing from a cudaHostRegister'd shared arena: prefill 895 -> 1944 tok/s (Win), decode 24.2 -> 33.6 tok/s (Win), 25.8 -> 28.9 (WSL2). "Every streamed byte costs one DRAM read instead of memcpy plus DMA."
- llama.cpp#25859 comments (https://github.com/ggml-org/llama.cpp/issues/25859): page-locking mmap'd expert tensors: +21% prefill (RTX 3060, DDR4, PCIe4); decode 46.8 -> 50.2 tok/s (+7%) with whole-tensor merged cudaHostRegister (RTX 4090, 2.9 GB VRAM, Qwen3.6-35B-A3B IQ2_M). Single-expert size there ~1-1.55 MB: copy ~60 us (=> ~25 GB/s) vs kernel <10 us, so double-buffering H2D on a 2nd stream gave NO decode gain (5.0 vs 5.5 tok/s). With your 10-20 MB experts copy >> kernel, same conclusion: nothing to overlap inside one card's own stage except with the trunk (already closed).
- Applies: only if your fetch path still goes through a staging buffer or pageable memory. You said experts are pinned, so check you are DMAing from the pinned store itself.
- Cautions [M, CUDA]: registration inflated RSS ~0.57x; pinned alloc breaks mbind interleave. On a single-socket EPYC with NPS1 neither should bite; with NPS2/NPS4 place pinned buffers on the NUMA node nearest each card's root port.

3.3 Resident-set / caching policy (relevant because 10-20% of experts are non-resident).
- ddvnguyen/llama.cpp#130 [M] (RTX 3060 x4 PCIe, 512 experts, top-10, 48 layers, per-layer private slots, miss 47%): T[ms/token] = 28.3 + 0.2842*misses between configs; 10.8 tok/s. Plain LRU (simulated) 10.30 vs shipped LFU half-life 16 10.81; half-life 256/2048 6% worse; Belady/MIN at same capacity 14.22 (34% fewer misses, offline, so a bound); compulsory floor 21.1; non-uniform per-layer slot allocation +0.29% and FAILS split-half hold-out; cache holds only ~5 steps of working set; adjacent-step overlap 2.48 of 10; reuse distance p50=4, p90=35 steps; lookahead staging usefulness 1.91% (consistent with your closed result); lossy host-side tier (re-quantised copy decompressed on device) is their only open item; "a prefetch is still a PCIe fetch: only victim choice removes fetches".
- llama.cpp PR 29887 (https://github.com/ggml-org/llama.cpp/pull/29887, qvac-fabric port) [M]: LRU GPU cache of host experts, cache = ~10% of expert bytes recommended; EPYC 7742 16 cores, PCIe4 x16, Qwen3.8-Flash-Next Q4_0 (65 GB experts): RTX 4090 25.0 -> 39.4 tok/s (hit 72%), 5090 30.8 -> 54.5 (77%); full cache variants 40.7 / 67.8 (hit 77/89%).
- llama.cpp PR 26563 (https://github.com/ggml-org/llama.cpp/pull/26563, closed for rework) [M, single author]: heat-map hot-expert cache with hysteresis + dwell on slot swaps; Qwen3.5-122B IQ2_M: 4.72 stock -> 3.97 (0.84x) -> 5.71 (1.21x) with dwell=16, i.e. swap thrash costs more than the hit-rate gain without hysteresis. Relevant if your resident set is ever updated online.
- FreeToken: global LRU across layers (vs per-layer private slots). sglang PR 42557 (https://github.com/sgl-project/sglang/pull/42557): K-slot LRU per layer, results vs tuned llama.cpp: close when most of the model fits; wins when host link fast; with several concurrent requests and small K/E paging falls behind (11-12 vs 41 tok/s).
- Hot-expert replication across cards [I, no source]: replication cannot reduce misses (a miss is by definition on no card) but, if your dispatcher can choose which card computes a resident hot expert, replication in 2 cards lets you put the fetched (missed) expert on the card with the least resident work, trimming the slowest-card stage. Needs your routing statistics; unmeasured elsewhere.

3.4 Not bit-exact options (listed so you can rule them out): KTransformers Expert Deferral (+1.45x, <=0.5% accuracy), BuddyMoE-style expert substitution (arXiv 2511.10054, not read), lossy requantised host tier (ddvnguyen open item).

3.5 Prior-art caveat, from the Strata digest (https://github.com/xenodeve/Strata-xeno/issues/186#issuecomment-5991404072): "a miss costs 4-8x a hit" and large prefetch speedups in the literature assume fast interconnects; for 512-expert top-10 models "MoE-SpeQ reports per-layer routing entropy ~ theoretical max, no consistently hot experts" -> consistent with your closed lookahead result.

---------------------------------------------------------------------------------------------------

## 4. ROCm / RDNA3 host->device specifics

- [M] ROCm issue #2253 (https://github.com/RadeonOpenCompute/ROCm/issues/2253), Radeon RX 7900 XTX, rocm-bandwidth-test: H2D 28.41 GB/s, D2H 29.01 GB/s, bidirectional 48.2 GB/s (run was reporting page faults on an old kernel/SMU mismatch, numbers still show link-level capability). => your ~25 GB/s practical is ~88% of 28.4.
- [C] HSA_ENABLE_SDMA (default 1) "enables DMA engines in all copy directions"; 0 => copies use blit (shader) kernels which occupy CUs but not SDMA queues (https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html). HSA_ENABLE_PEER_SDMA (default 1) is D2D only (irrelevant, no P2P). No published RDNA3 measurement of SDMA=0 vs 1 H2D throughput; worth a 10-minute A/B but blit copies will compete with compute on the same card [I]. rocm-systems PR 11614 states results are identical with HSA_ENABLE_SDMA=0, so it is a supported mode.
- [C] GPU_MAX_HW_QUEUES (default 4) is about compute HW queues, not SDMA; AMD_SERIALIZE_COPY (default 0) serializes copies if set; PAL_PREPINNED_MEMORY_SIZE default 64 KB is the pageable-copy staging size; HIP_HOST_COHERENT default 0 (https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html). None documented as raising H2D throughput.
- [M, weak] r/ROCm 2026-09-15 "ROCm/RCCL multi-GPU crash on 4x RX 7900 XTX ... later amdgpu SMU/MES/SDMA reset hang" (https://www.reddit.com/r/ROCm/comments/1wh0dpz/): a risk flag that heavy concurrent multi-card SDMA can hit reset hangs; no copy-throughput data.
- Windows-only: rocm-systems#12535 gfx1100 first compute submission after SDMA H2D into recently freed memory never signals fence. Not relevant on Linux.
- Not found: number of SDMA engines on Navi31 (commonly 2), whether each card's SDMA ring limits one copy to < link rate, any 3-card concurrent hipMemcpyAsync data, any ReBAR effect on SDMA. ReBAR concerns CPU access to VRAM through the BAR (relevant to kernel-based or CPU-side writes to VRAM), not SDMA H2D, [I].
- Zen2 topology note [I]: PCIe DMA reads go DRAM -> IO die -> PCIe root and do not traverse the CCD GMI links, so with only 8 cores (2 per CCD) the three cards' DMA should not be limited by core count; the CPU GEMV and the DMA only meet at the memory controllers. If a simultaneous 3-card H2D test falls below ~3x25, suspect the IO-die quadrant/NPS mapping of each card versus where its pinned buffer lives, IOMMU, or the max-payload/relaxed-ordering setting before blaming SDMA.

---------------------------------------------------------------------------------------------------

## 5. Suggested experiments (all standalone microbenchmarks, ordered by cost/benefit)

E1 (30 min): 3 simultaneous pinned hipMemcpyAsync H2D, one per card, 16 MB chunks, 1000 reps; report per-card and aggregate GB/s; then with buffers on the NUMA node of each card vs the other node; then HSA_ENABLE_SDMA=0.
E2 (1 h): E1 plus 8-thread IQ4_XS GEMV on host (llama.cpp ggml kernels are fine) reading a DIFFERENT pinned region; record GEMV GB/s and each card's H2D drop. This is FreeToken's concurrent pair and decides q*.
E3 (1 h): one card, many layouts: one 3-tensor expert as 3 copies vs 1 contiguous copy vs hipMemcpyBatchAsync of 3; confirm no per-copy penalty at 10-20 MB.
E4 (offline, free): replay your logged per-layer miss sets through a scheduler that (a) assigns whole experts greedily to the least-loaded of 3 cards, (b) adds a CPU lane of 0/1/2 experts per layer at the E2 measured speed, (c) allows row-slice splitting with a fixed 30-50 us reduction cost. This gives the upper bound of every lever in sections 1-2 without touching the engine.
E5 (optional): zero-copy kernel read of pinned memory on gfx1100 with a deep-pipelined GEMV; expectation <5 GB/s, only to close the question on RDNA.

## 6. Source index
- Fiddler https://arxiv.org/abs/2402.07033 ; HybriMoE https://arxiv.org/html/2504.05897 ; FreeToken https://arxiv.org/abs/2608.16157 and https://github.com/FlashML-org/FreeToken ; FreeToken study https://github.com/justinchuby/onnx-genai/issues/1759
- ThunderEP https://arxiv.org/abs/2609.40093 ; KTransformers v0.5.3 https://github.com/kvcache-ai/ktransformers/releases/tag/v0.5.3
- colibri#537 https://github.com/JustVugg/colibri/issues/537 ; ddvnguyen#130 https://github.com/ddvnguyen/llama.cpp/issues/130
- llama.cpp PRs/issues https://github.com/ggml-org/llama.cpp/pull/29887 , /pull/26563 , /issues/25859 ; sglang https://github.com/sgl-project/sglang/pull/42557 ; exllamav3 https://github.com/turboderp-org/exllamav3/pull/341
- Strata digests https://github.com/xenodeve/Strata-xeno/issues/186 and /issues/187
- ROCm: https://github.com/ROCm/rocm-systems/pull/11614 , https://github.com/RadeonOpenCompute/ROCm/issues/2253 , ROCR and HIP env-var docs linked above
- Not read, may be worth a look: r/LocalLLaMA "[Paper] Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices" (https://www.reddit.com/r/LocalLLaMA/comments/1v0vp9k/), r/ROCm "dual 7900xtx tensor parallel on limited PCIE x4" (https://www.reddit.com/r/ROCm/comments/1w14qal/), BuddyMoE arXiv 2511.10054, vLLM RFC https://github.com/vllm-project/vllm/issues/38256
