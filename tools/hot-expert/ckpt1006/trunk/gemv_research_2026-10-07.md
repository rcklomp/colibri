# Batch-1 GEMV on RDNA3 (gfx1100 / 7900 XTX): outside-source research

Date of research: 2026-10-07. Method: GitHub API (rcklomp login), raw GitHub files, Reddit public RSS, WebFetch/WebSearch. No GPU, no ssh, no repo edits.
Labels: [M] = measured by the source with a reproducible harness or full table; [C] = claimed / headline only / derived by me from their numbers; [D] = my own derivation (stated).
"Bit-exact?" line = does this change the per-output-element summation order of OUR kernels.

Local copies of the heavy sources (for grep) are in `scratchpad/dl/`: `exl3rocm_README.md`, `hip_roofline.md`, `rocm6409_body.md`, `mmvq.cu`, `AMDGPUUsage.rst`, `peakbw.cc`.

## 0. Ranked, actionable takeaways for this project

1. The per-launch HIP floor on a 7900 XTX is about 2.8-3.3 us (serial dependent chain), not 10 us. Of our ~10 us per small GEMV, about 3 us is submission/dispatch, and the other ~7 us is wave ramp-up, tail, and the drain before the next dependent kernel. Only fewer launches, in-kernel barriers, or a cheaper submission path attack those. [M] (sources 2.1, 2.2, 2.3). Budget view [D]: 660 launches x 3 us = ~2 ms of the 18.9 ms is pure submission; the remainder of the ~6.6 ms fitted launch cost is ramp/tail/drain.
2. Single-launch split-K with a "last block to arrive does the reduction" (atomic counter per output tile) is implemented and shipped on gfx1100 in exllamav3-rocm. It removes the second reduce launch. Bit-exact with our reduce kernel IF the last block sums the partials in the same fixed order our reduce kernel does (not arrival order). [M, 3.1]. This is the single best fit for our "~10 us + 0.1 us x nsplit" cost model: it deletes the second launch and its ~3 us gap and ramp.
3. Multi-matrix launches (one launch, blockIdx.z = matrix index) for projections sharing an input vector were measured on gfx1100: q/k/v bundle 26-28 % faster, whole-round only -3.3 % because it removed ~320 of ~1085 kernels. Never changes summation order. [M, 2.4]. For us: batch the 24-row hyper-connection fn matrices that share the 16 384-long input into one launch (the 90 launches x ~10 us = ~0.9 ms).
4. Grid sizing: make the grid at most one residency wave (exl3: 6 blocks per WGP = 288 blocks, LDS-limited). A partial second wave "roughly doubles the kernel tail"; 3-10 % per matmul in isolation. [M, 3.1]. Bit-exact trick: change output-row tiling (rows per block), NOT nsplit, because nsplit defines the summation order.
5. Compiler hazard: plain loads get folded to load-at-use by InstCombine and serialize the stream; exl3 uses `raw_buffer_load` + `sched_barrier` to pin a 4-deep register prefetch ring. Check our ISA for `s_waitcnt vmcnt(0)` immediately after each load. [M, 3.1]
6. Peak read ceiling is real: 952 GB/s (99 % of 960) with 16-byte non-temporal loads, 1536 blocks x 256 threads, 4 loads in flight per thread. An EXL3-like strided pattern (1 KB per block per K-slice, 139 KB stride, 272 blocks) reaches 856 GB/s (90 %). `torch.sum` reaches 750, device copy 701. [M, 3.1] So 800 GB/s "achievable" is conservative; 850+ is reachable for a clean streamer.
7. Nothing found that makes the 96 MB Infinity Cache help weights re-read once per token with GBs of other traffic in between (see 4.4). Do not plan on it.
8. HIP graph replay buys little on gfx1100 for dependent chains: 2.82 us/node (graph) vs 3.13 us (stream loop). [M, 2.2] exl3 measured +0.5 % for eager vs graph and ~8 us GPU idle per graph launch. [M, 2.3] A retained-PM4 submission path (Redline) is 2.3x faster than direct HIP at median on a 7900 XTX in microbenchmarks, +8 % decode on one engine, but caused GPU VM faults and device resets for its integrator. [M/C, 2.5] High risk; flag only.

## 1. Q1: how the best open batch-1 quantised GEMV kernels reach the memory bound

### 1.1 llama.cpp HIP `mmvq` on RDNA3 (source code, master 2026-10)
Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmvq.cu and .../common.cuh [read directly].
- RDNA3_0 table (`MMVQ_PARAMETERS_RDNA3_0`): for ncols_dst==1 `nwarps=8` for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL; `nwarps=2` for Q6_K; `nwarps=1` for everything else (Q4_K, Q5_K, IQ4_XS, Q2_K, ...). Comment in source: "stricter whitelist than RDNA4. Q2_K / Q5_K / IQ4_XS regress in full quant sweeps".
- `rows_per_cuda_block = 1` for all RDNA tables (only GENERIC/GCN/TURING/GB10 get 2 rows for ncols 2..8). Block = nwarps x 32 threads; nwarps-1 partial sums are combined through `__shared__ tmp_shared[nwarps-1][ncols][rows][warp_size]` then a warp shuffle reduce. Deterministic but the tree depends on nwarps (changing nwarps changes order).
- dot product: `__builtin_amdgcn_sdot4` (signed) / `__builtin_amdgcn_sudot4` (unsigned x signed) (common.cuh ~line 714). Activations are first quantised to Q8_1 in a separate launch.
- PR #19478 (RDNA4 table; nwarps=8, rows_per_block=1): Llama 2 7B Q4_0, R9700 gfx1201, tg128 95.05 -> 104.82 t/s (+10.3 %), "bit-exact match with master" greedy. [M] https://github.com/ggml-org/llama.cpp/pull/19478
- PR #20831 (clamp nwarps for narrow matrices, MoE expert-sized 512-2048 cols): on W7900 gfx1100, ROCm 7.1, llama-2-7b tg512: Q4_0 98.49 -> 98.72, Q5_0 85.76 -> 86.47, Q8_0 66.79 -> 68.02, Q6_K 73.71 -> 76.33 t/s. [M] https://github.com/ggml-org/llama.cpp/pull/20831  (my arithmetic [D]: 98.5 t/s x ~3.8 GB = ~375 GB/s = ~43 % of W7900's 864 GB/s; Q8_0 68 t/s x ~7.2 GB = ~490 GB/s = ~56 %.) Takeaway: nwarps=8 narrow-matrix waste (idle warps still pay __syncthreads and the shared reduce) is exactly the "small matrix floor" effect.
- hipEngine analysis: with `ncols_x=512` only 32 of 256 threads enter the useful inner loop in llama.cpp's Q4_K MMVQ (224/256 idle). https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md section 9.6 [C]. Their rule: match workgroup size to the smallest K in the dispatch set (64 threads for K=512, 128 for K=2048).
- Applies here: the lesson is shape-specific launch configs (nwarps/rows per block) and avoiding dead lanes; llama.cpp's reduction tree and Q8_1 activation quantisation are NOT our order. Not bit-exact with ours; use as a design reference only.

### 1.2 Measured achieved bandwidth on 7900 XTX with llama.cpp (whole-model)
- Issue #28863 (2x 7900 XTX, gfx1100, stock build b10794-10809, ROCm 10.0, llama-bench -r 5, per-card `amd-smi` UMC cross-checked): 7B-class Q8_0 (8.1 GB) on ONE card = 95.9 t/s = 776.4 GB/s = 80.9 % of 960 (card 1: 770.7, 80.3 %); Qwen3.8-27B Q4_K_XL (17.545 GB) = 36.0 t/s = 632.5 GB/s = 65.9 % (card 1: 623.5). Tensor-split across both cards: 43.0 % / 54.8 % per card. [M] https://github.com/ggml-org/llama.cpp/issues/28863
  This is the only >= 80 % llama.cpp datapoint on 7900 XTX that I found: dense Q8_0 (nwarps=8 path, simple dp4a) on big matrices. K-quants (nwarps=1 on RDNA3) sit at ~66 %.
- Issue #20934 (7900 XTX, tg128, Llama-7B Q4_0, same flags): Vulkan RADV 167-177 t/s vs ROCm 129-144 t/s. [M, issue text] https://github.com/ggml-org/llama.cpp/issues/20934 [D]: 3.8 GB x 172 t/s = ~650 GB/s (68 %) Vulkan vs ~520 GB/s (54 %) ROCm. Maintainer notes: Vulkan runs wave64 on RDNA, HIP cannot (wave32 only); wave64 build experiment on W7800 gave +0.2 % to +16 % tg128 across 10 models (ravel7524 comment) but it is an unsupported flag. [C]
- Reddit (Qwen3.8 27B Q4_K_M, 7900 XTX, Vulkan q8 KV, llama-bench): tg512 36.57 t/s. https://www.reddit.com/r/LocalLLaMA/comments/1w7fkhf/qwen38_27b_on_rx_7900_xtx_ollama_rocm_vs_llamacpp/ [M by poster]. [D] ~16.8 GB x 36.6 = ~615 GB/s (64 %).
- Applies here: confirms that a good batch-1 dequant-dot kernel on big matrices reaches ~80 %; kernels with heavier dequant (K-quants, IQ) stay at 55-66 %. Our big trunk matrices at 500-700 GB/s are in that normal range.

### 1.3 exllamav3 HIP port to RDNA3 (best-documented gfx1100 kernel I found)
Source: https://github.com/wtogami/exllamav3-rocm (README.md, "Memory bandwidth utilization" and "Megakernel" sections; downloaded to dl/exl3rocm_README.md). All on RX 7900 XTX, gfx1100, ROCm 7.2.4 and 10.0.
- 4-bit EXL3 tensors: all 401 target matmuls, 1 row: 17.5 ms per pass = 663 GB/s (70 % of 960); 2 rows 628; 4 rows 551; 8 rows 446 (VALU-bound by decode). Per-shape: "~700-770 GB/s at 1 row". [M]
- Microbenchmarks (`rocm_tests/micro/peakbw.cc`, `stridebw.cc`): peak read 952 GB/s (99 %) with 16-byte non-temporal loads, 1536 blocks x 256 threads, 4 in flight; strided EXL3-like pattern 856 GB/s (90 %). [M]
- Kernel design: one wave per 16x16 weight tile column, K split across blocks, split-K reduction by last arriving block (atomic counter), prefetch ring of 4 K-slices (8 was 1-4 % slower), `raw_buffer_load` + `sched_barrier`, CU mode (-mcumode) +6 %, `amdgpu_waves_per_eu` no effect. [M]
- Tried and dropped (no gain, measured): persistent/work-queue matmul scheduling ("no gain once clocks are warm; block-time spread is contention, not imbalance"); LDS staging of weights; loader-wave warp specialisation (1 row 57 -> 79 us, worse); WMMA for M=1/8 rows; larger/smaller x chunks. [M]
- Applies here: yes as a design reference. The persistent/work-queue negative result is directly relevant; their fixed-per-block split is "K-split across blocks", the same family as ours.

### 1.4 Other open engines
- hipEngine (shisa-ai, W7900 + 7900 XTX): docs/ROOFLINE.md. End-to-end W4 decode only 17-34 % of peak because of dispatch count (~894 per token, median inter-kernel gap 3.76 us, mean 32.9 us, 70.6 % of dispatches under 10 us), dequant VALU cost and VGPR/occupancy; "dp4a without coalescing is slower"; "LDS staging regressed or neutral in all tested cases". [M, W7900] https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md
  Occupancy rule from that doc [C]: <=96 VGPR = max waves, <=192 = 8 waves "adequate", >256 starves memory controller (BW falls to 30-40 %). Matches #28863's citation. Scratch>0 on a hot path is a bug.
- hipfire (warpfront, Rust, 7900 XTX): headline "MQ4R 253 tok/s decode, Q8 KV" only; no GB/s or kernel detail. [C] https://github.com/warpfront/hipfire
- qingming-gfx1100-gemv (FP32 GEMV on 7900 XTX; our trunk has F32 matrices): "461/461 wins vs rocBLAS SGEMV, median 1.5149x; 435/435 vs hipBLASLt, median 1.8652x"; shape-driven kernels (compact for small N, packed/multi-row for narrow, split-K, GRID_SPLIT for long reductions with small M, persistent-register kernels); uses "compensated FP32 summation" in the stitching phase (so NOT order-compatible with plain FP32). [C, repo README] https://github.com/uulong950/qingming-gfx1100-gemv ; Reddit https://www.reddit.com/r/ROCm/comments/1w37ov9/gfx1100_handtuned_fp32_gemv_on_rx_7900_xtx_461461/ . Useful as a shape-class taxonomy and a benchmark to compare our F32 GEMVs against rocBLAS; summation differs.
- kernel-anvil (Reddit r/ROCm 2026-03-30): per-shape profiling of llama.cpp MMVQ nwarps and rows_per_block on a 7900 XTX, "1.2x-2.1x per shape" on individual Qwen3-8B Q4_K_M kernels; the 2.25x headline was on a turbo3 KV path, poster later conceded stock is ~26 t/s. [C] https://www.reddit.com/r/ROCm/comments/1s7jgvl/kernelanvil_2x_decode_speedup_on_7900_xtx_by/ . Supports "tune per shape".
- llama.cpp PR #26301 (mmvdq: dequant-to-float matvec for Q4_K/Q5_K/Q6_K, no Q8_1 activation quantisation): gfx1151 (RDNA3.5) only; mean +3.09 % across 38 models; open. [M, RDNA3.5] https://github.com/ggml-org/llama.cpp/pull/26301 . Float dot, so order differs from dp4a.
- ik_llama.cpp: CUDA template instances for `*_r4` types exist (e.g. `ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_k_r4.cu`), but I found NO GPU decode bandwidth data for R4/R8 row-interleaved layouts; they were designed for CPU SIMD. Not applicable to our HIP decode GEMV without a measured case. [not found]
- Vulkan `mul_mat_vec*`: no per-shape GB/s on 7900 XTX found beyond #20934 above. hipEngine's ROCm#6409 data show Vulkan (RADV, wave64) is faster on tiny-dispatch submission and packed-dot loops by 1.05-1.13x on gfx1100 but HIP wins production-shaped Q4/Q6/Q8_0 quantize+dot (Vulkan/HIP ratio 0.39x-0.97x i.e. HIP faster). [M] https://github.com/ROCm/ROCm/issues/6409
- hipBLASLt / rocBLAS gemv: found only the FP32 comparison above (qingming) and AMD's MI355X split-K blog (3.2). No measured rocBLAS gemv bandwidth fractions on gfx1100 found. [not found]
- MLC/TVM/Triton-AMD gemv on RDNA3: nothing found. [not found]

## 2. Q2: small/narrow GEMVs, launch floors, merging, persistent kernels, graphs

### 2.1 Floor of a HIP launch on gfx1100 [M]
ROCm/ROCm#6409 comment by Kaden-Schutt (reproducer `aql_dispatch_floor.cpp`, median of 200 replays, N=512, microseconds per dispatch, tiny atomicAdd kernel): gfx1100 7900 XTX: stream-loop 3.134, graph replay 2.815 (ROCm 7.14 and 10.0 identical). gfx1201 2.559 / 2.144; per-launch-sync 18.5 us (gfx1201). [M] https://github.com/ROCm/ROCm/issues/6409
- Graph shape matters: on ROCm 7.14 with default 4 HW queues, "independent" nodes were 1.03 us/node on gfx1100 (2.7x cheaper than chain 2.80), but 6.8 us at GPU_MAX_HW_QUEUES=8; ROCm 10.0 converts all shapes to chain cost (~2.78-2.80). [M]
- lhl (hipEngine) measurement on W7900, 941-node graph: HIP 3.865 us/node vs Vulkan command buffer 0.876 (4.4x); 1 dispatch alone 14.98 us HIP vs 1.48 us Vulkan (event timing includes launch overhead). [M] same issue.
- Applies here: our 10 us per launch is ~3x the floor. Gaps between dependent kernels: hipEngine median 3.76 us; exl3 ~690 gaps of ~3-4.5 us per round = ~3.4 ms of 34.7 ms (10 %). [M] That is the same ~10 % share you would expect for ~660 launches x 3 us = ~2 ms / 18.9 ms.

### 2.2 HIP graphs
- exllamav3-rocm: "HIP graphs save little on ROCm: hipGraphLaunch spends CPU time per node like eager launches, and each graph launch adds ~8 us of GPU idle. Replacing the per-module graphs with eager launches is ~0.5 % faster. Runtime knobs (HIP_FORCE_DEV_KERNARG, HSA_ENABLE_INTERRUPT, GPU_MAX_HW_QUEUES) do not change the ~3.3 us per-kernel dispatch cost." ROCm 10.0: 2.94 us, unchanged. [M]
- hipEngine: graph replay +2.6-3.9 % on 4K decode (PARO W4); multi-step graph capture regressed. [M, W7900]
- Bit-exact? yes (submission only). Gain on our workload likely small (our launches are already issued from a captured/serial stream? unknown), expect <= 3-5 %.

### 2.3 Persistent / in-kernel barrier / cooperative launch
- exl3-rocm: grid barrier inside a persistent kernel costs ~0.4-2.5 us against ~3-6 us for a kernel boundary; `hipLaunchCooperativeKernel`/`grid.sync` adds 10-20 us PER LAUNCH on ROCm and ate the gain; they use a normal launch with an atomic epoch barrier (grid far below one residency wave, so no deadlock). Their megakernel (`gdn_core_mk`, four kernels merged into one) saved ~0.45 ms per ~36 ms round; kernels per round 1085 -> 718. [M] https://github.com/wtogami/exllamav3-rocm
- Kog AI (MI300X, not RDNA): 4.5 us kernel launch and cleanup + ~0.5 us HBM restart per kernel boundary; their tuned grid sync 0.80-0.93 us vs naive 7.59-7.88 us; grid sync is ~35 % of token time in their single-kernel engine; 3,000+ tok/s batch 1 on a 2B FP16 model; uses 256 of 304 CUs. [C, blog] https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- Net: persistent kernels pay only if the barrier is custom (atomic epoch counter) and the grid is <= one residency wave; cooperative-launch API is a trap on ROCm. No evidence of a persistent GEMV beating per-matrix launches on gfx1100 for plain matmul (exl3 "no gain").
- Programmatic dependent launch equivalent on AMD: none found. [not found]
- `hipExtLaunchKernel` / `hipExtModuleLaunchKernel` cooperative-flag measurements: none found. [not found]

### 2.4 Merging GEMVs that share an input
- exl3-rocm `exl3_rdna3_mgemm` (blockIdx.z per matrix): MLP gate+up, DeltaNet qkv+z, attention q/k/v (14 slices) as one input-transform launch plus one matmul launch. "3.3 % less time per speculative round than separate matmuls (~320 fewer kernels per round; q/k/v alone is 26-28 % faster at 5-8 rows)". [M, gfx1100] Also fused prologues (silu*up before down projection, bit-identical) and rms_norm tail writing the next matmul's input transform. [M]
- exl3 open ideas list: residual RMSNorm as prologue of next matmul estimated ~0.3-0.5 ms (-128 kernels) per ~36 ms round. [C]
- llama.cpp: gate/up+GLU fusion exists for small batches (MMVQ/MMVF) and PR #28702 extends it to MMQ (prefill, GB10 CUDA +2.8 to +14 %); no 7900 XTX batch-1 number. https://github.com/ggml-org/llama.cpp/pull/28702 [M, NVIDIA]
- Bit-exact? Merging launches does not change per-element order if each matrix's tiling/nsplit is kept. Fusing norm into GEMV prologue is exact only if the norm math is replicated identically (same reduction tree) and recomputed per block (extra VALU) or published once.

### 2.5 Cheaper submission (retained command buffers)
- Redline retained submission (Kaden-Schutt, ROCr/HSA, same HSACO as HIP): RX 7900 XTX median 2.34x faster than direct HIP, 1.43x vs Vulkan over 240 rows; serialized dispatch-grid family 6.88x vs HIP at median. [M] lhl on W7900: "+8.13 % decode" in hipEngine, but "intermittent address-zero GPU VM faults ... full-device resets and VRAM loss" on W7900 and RX 7900 XTX; not resolved. [M/C] https://github.com/ROCm/ROCm/issues/6409 and https://github.com/shisa-ai/hipEngine/blob/redline-integration-spike/docs/REDLINE.md
- Bit-exact: yes (same code objects); risk: stability. Not recommended without isolation.

## 3. Q3: split-K versus alternatives, reduce styles, determinism

### 3.1 Single-launch last-arriver reduction (shipped on gfx1100)
exllamav3-rocm README: "k-split across blocks, split-K reduction by the last block to arrive (atomic counter) ... A single graph-patchable launch pair."; epilogue "per-tile counter: the last block to finish a unit does the dependent work (no deadlock risk, no co-residency assumption)". Kernel boundary 3-6 us vs 0.4-2.5 us for in-kernel sync. [M]
- Determinism: the README claims bit-identical results for kernel refactors (test_rdna3_gemm.py 0 failures vs old build) but does not say the split-K partial sum order is fixed; whether it is deterministic depends on whether the last block sums slices 0..nsplit-1 in index order (what you want) or accumulates in arrival order. [D]
- Memory-model requirements on gfx11 (LLVM AMDGPUUsage, GFX10-GFX11 table): agent-scope load-visible = `global_load_b128 ... glc` on gfx11 (table at "AMDGPU Load-Visible Implementation"); partial stores must be released (`__threadfence()`/atomic release) before the counter increment; the last block must read partials with agent-scope acquire/glc loads (L0 is per-CU, not coherent). https://llvm.org/docs/AMDGPUUsage.html (raw rst read locally). [C, from doc]
- Bit-exact? YES if the last block performs exactly the reduce kernel's order. Saves the second launch (3 us submission + ramp + tail), costs a serial tail inside one block (nsplit partial reads per output; for 24 rows and nsplit ~16-64 this is small).

### 3.2 Atomic float adds into the output
- AMD's MI355X LDS-pipelined split-K (ROCm blog 2026-06-29): single-launch split-K with partition 0 initialising C, others spin on a signal then `atomic add` into global C, semaphore counts arrivals; "1.64x average latency improvement" on 32 decode shapes K=7168, 1.79x at M<=8 (best 2.37x), 1.49x on 48 BF16 shapes; determinism "not explicitly addressed". [M/C, MI355X, gfx950, NOT RDNA3] https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html
- rocBLAS: `rocblas_atomics_mode` default `rocblas_atomics_allowed`; "not allowing atomic operations can generally improve determinism and repeatability of results at a cost to performance". [C] https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/enumerations.html
- Bit-exact? NO. Float atomics add in arrival order (order-changing; non-deterministic run-to-run). Exclude unless explicitly marked order-changing.

### 3.3 In-workgroup (LDS) and wave-level reduce
- llama.cpp MMVQ: waves each stride over K, partial sums via LDS then warp shuffle (see 1.1). Deterministic, fixed tree per (nwarps, vdr, warp size). hipEngine: LDS reduction adds barrier stalls between productive work; "Wave32/no-LDS (subgroup-style reduction) gave +3-10 % microbench, 0 % end-to-end" [M, W7900].
- Bit-exact? Only if the new reduction tree is identical to the old: any change in lanes-per-row or waves-per-row changes order. Treat as order-changing.

### 3.4 Dead-lane / narrow-K lesson
- 24 rows x 16 384 K: a 256-thread block per row wastes nothing on K but launches only 24 blocks (24/96 CUs busy) unless K is split; this is the exact case where split-K or "rows x K-slices" grids are needed. exl3 sizes the grid to 288 blocks (6/WGP). hipEngine: "Grid occupancy alone was not sufficient; within-block work distribution was the limiter" (attention split-K lesson). [M]
- Safe-for-bit-exact knobs: threads per block per split, rows per block, load width, unroll/prefetch depth, if the per-thread accumulation order inside a split is preserved. Not safe: nsplit, lanes-per-row.

## 4. Q4: RDNA3 memory-system facts

### 4.1 Load width and prefetch depth
- Best demonstrated: 16-byte (`global_load_b128`) non-temporal loads, UNR=4 independent loads in flight per thread, 1536 blocks of 256 threads: 952 GB/s. The source sweeps grid 192..6144 and UNR 1/4/8 and NT on/off but does not print the table in the README. [M headline] https://github.com/wtogami/exllamav3-rocm (rocm_tests/micro/peakbw.cc)
- Little's-law check [D]: ~960 GB/s x ~0.4-0.5 us loaded latency ~ 400-500 KB in flight across the chip, i.e. ~4-5 KB per CU; 96 CUs x 4 waves x 32 lanes x 16 B x 4 loads = 786 KB, enough. hipEngine measured latency in cycles: L2 131, MALL 612, VRAM ~1045 [M]; so at ~2.5 GHz VRAM ~0.42 us idle latency.
- Per-wave occupancy: exl3 says `amdgpu_waves_per_eu` 8-16 no effect; ring depth 8 slower than 4. [M]

### 4.2 Cache-policy bits
- LLVM AMDGPUUsage (GFX10-GFX11): a `__builtin_nontemporal_load` on gfx11 becomes `global_load ... slc=1 dlc=1` ("If GFX10, omit dlc=1"); gfx11 agent/system-scope coherent loads use `glc`. The doc says nothing about MALL allocation policy per instruction; it states "On GFX10.3 and GFX11 a memory attached last level (MALL) cache exists ... All agents access GPU memory through the MALL cache." [verified in AMDGPUUsage.rst lines ~13679, 13750-13760, 2306]
- Mesa/ACO notes (unofficial): GLC/SLC/DLC carry non-temporal hints; "RDNA's L2 handles hits from streaming accesses by leaving the line in cache but not updating LRU bits". [C] https://fuchsia.googlesource.com/third_party/mesa/+/098259aa18f70ac3e1baf5b5d7aed073b70114db/src/amd/compiler/README-ISA.md
- No A/B (NT vs plain) number for GEMV on gfx1100 found; the 952 GB/s test used NT, and torch.sum (plain loads) reached 750 GB/s, but those differ in more than the NT bit. [not isolated]

### 4.3 Row alignment / partition camping / layouts
- Only data point: strided access with 1 KB per block per K-slice at 139 KB stride, 272 blocks: 856 GB/s (90 %) vs contiguous peak 952 (99 %). The README names `chunk` sweeps 1024/2048/4096/8192 B x splits 1/2/4 in `stridebw.cc` but prints only the headline. [M]
- hipEngine: 128-byte cache lines; "well-written streamer 75-85 % of peak, scattered/low-occupancy 40-60 %". [C]
- ik_llama.cpp `_R4/_R8`: CPU-side layout; no GPU bandwidth evidence. llama.cpp GGML Q8_0/Q6_K "repack" is CPU-only (x86/ARM). [not found for GPU]
- Channel / partition camping on Navi 31: nothing measured found. [not found]

### 4.4 Infinity Cache (96 MB MALL) as a between-token weight store
- Bandwidths [M] (hipEngine ROOFLINE, W7900/Navi 31): L2 2.88 TB/s, MALL 2.30 TB/s, VRAM ~960 GB/s; latency cycles 131 / 612 / ~1045. Chips and Cheese: IC read bandwidth +1.8x over RDNA2, scalar-side IC latency 161 ns; no VRAM GB/s number. [C] https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
- Residency: nothing measured about decode weights staying in MALL between tokens. With 8.3 GB of trunk weights (plus experts) touched per token, reuse distance >> 96 MB, so an LRU-like memory-side cache cannot retain a given matrix across tokens. hipEngine makes the same argument ("weights stream from VRAM on every token"); it notes small projections (256-512 KB) may hit L2 on repeats within a token. [D/C]
- The only cache-resident-weights paper I hit (arXiv 2606.25353) is about multi-socket CPU last-level caches, not Infinity Cache. [N/A]

## 5. Q5: >= 80 % of peak on RDNA3 token generation
- Whole-model: only the dense 7B-class Q8_0 on llama.cpp (80.9 %, #28863). [M]
- Kernel-level: exl3 peak-read microbench 99 %; EXL3 4-bit strided 90 % (856 GB/s); plain-decode matmuls 70 %; weights-streamed-no-decode variant of the gate/up kernel ~830 GB/s. [M]
- How: simple dequant (Q8_0 dp4a), 16-byte loads, deep independent-load queue, grid <= one residency wave, K split across blocks, few VGPRs. Reported obstacles below 80 %: heavy decode VALU (K-quants, trellis), dead lanes on narrow K, VGPR > 128, dispatch gaps (~10 % of token time).

## 6. What I could NOT find
- Any measurement of the launch+ramp+tail cost of a single small (0.4-8 MB) GEMV on gfx1100 isolated from dispatch cost; the ~7 us above the dispatch floor is my subtraction, not a measured number. A one-kernel test (empty-ish kernel with 24 / 96 / 192 / 1536 blocks timed back to back with dependency) would settle it.
- NT vs plain load A/B for streaming GEMV on gfx1100; the effect of `s_prefetch` hints (RDNA3 has no s_prefetch; it is gfx12).
- Channel/partition camping or row-start alignment measurements on Navi 31.
- ik_llama.cpp row-interleaved layouts on GPU; Vulkan `mul_mat_vec` per-shape GB/s on 7900 XTX; MLC/TVM/Triton-AMD gemv on RDNA3; hipBLASLt gemv numbers on gfx1100.
- PDL equivalent on AMD; cooperative-launch flag measurements beyond exl3's "+10-20 us".
- Reddit posts with per-matrix GB/s on a 7900 XTX: RSS gave only titles/first paragraphs (threads read: Qwen3.8 27B Ollama vs Vulkan; kernel-anvil).

## 7. Source index
- https://github.com/ggml-org/llama.cpp/issues/28863 (7900 XTX UMC per-card bandwidth)
- https://github.com/ggml-org/llama.cpp/pull/19478 , /pull/20831 , /pull/26301 , /pull/28702 , /issues/20934
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/mmvq.cu
- https://github.com/wtogami/exllamav3-rocm
- https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md
- https://github.com/ROCm/ROCm/issues/6409
- https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
- https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html
- https://llvm.org/docs/AMDGPUUsage.html (GFX10-GFX11 memory model)
- https://github.com/uulong950/qingming-gfx1100-gemv
- https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture
