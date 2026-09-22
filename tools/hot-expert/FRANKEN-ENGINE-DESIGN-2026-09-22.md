# Franken-engine: architecture for three RX 7900 XTX (design, 2026-09-22, rev 1)

The owner's brief (plan rev 45): a NEW engine, assembled from the parts of
Colibri, llama.cpp, hipFire and hipEngine that measure best on this rig, and
inventing what none of them has, to serve Qwen3.8-Flash-Next, DeepSeek-V4-Flash,
GLM-5.3-Flash and their successors at a 256k window. This file is the design.
Every number is labelled: **measured** (record §, date) or **projected**
(derivation shown; a thing to refute, never a result). Where a choice is
open, the measurement that closes it is named.

## 1. The facts the design must obey

| fact | value | source |
|---|---|---|
| VRAM | 3 × 24 GiB = 72 GiB; two cards share one upstream PCIe link | record hardware § |
| host→device streaming | 61 GB/s with three cards, 28 with one; Vulkan copies reach HIP's rate on the right queue family | §PCIE-STREAM, §VK-STREAM (F0) |
| CPU expert path | ~985 MB/token at 22 GB/s ≈ 45 ms/token for GLM's misses; 8 cores | §G11 |
| a synchronous GPU call | 0.25–0.4 ms whatever the matrix (Vulkan/RADV) | §F3-STEP2A, HANDOFF §5 |
| RCCL tensor parallelism | loses on arithmetic: 25 µs × ~80 hops per token | plan §1 (measured 09-16) |
| resident MoE on three cards, llama.cpp | 75 tok/s at 18k (gpt-oss-120b); 17.6 with all experts in RAM | §GPTOSS-3CARD |
| one card, resident 35B, hipFire kernels | 131 → 79 tok/s down the ladder, 164 short | plan §1 |
| llama.cpp HIP, Qwen3.8-Flash-Next IQ4_XS, ~¼ of experts in RAM, 256k window | 15 tok/s to 39k, 6.1–6.5 at 257k; cold 257k prompt 17 min; 1.0–2.0 ms per prefill token | §F11-STEP0, §F11-DEPTH |
| Colibri, GLM-5.3 int4 with 60–70 % of the model in RAM | 5.04 tok/s at 18k; 18k prompt 356 s | served, §F9a |
| the models | none fits 72 GiB at an acceptable quant: GLM 149–184 GB, Qwen3.8 94 GB (IQ4) / 173 (FP8), DeepSeek 91 GB (IQ2) | disk |

Consequences. (a) **The engine is a placement-and-streaming engine**: the
model never fully fits, so the token's cost is (misses × bytes ÷ stream
rate) + (resident work ÷ kernel efficiency) + (calls × sync cost) +
attention(depth). (b) **Per-op host round trips are ruled out**: at 0.3 ms a
call, a 48-layer token with ten calls per layer spends 150 ms in sync alone,
which is where Colibri's 5 tok/s comes from and why hipFire's whole-token
GPU residency reaches 100+. (c) **Tensor parallelism is ruled out**; the
three cards split the model by LAYER RANGE (pipeline), and the only per-token
inter-card traffic is one hidden vector per boundary (2 560–5 120 floats,
~10–20 µs). (d) **Attention at 256k must live on the cards** in each layer
range's own VRAM: KV for these hybrid/compressed-attention models is small
per token (fit placed Qwen3.8's 256k KV beside 60+ GB of weights), but any
design that pages KV through the host loses at depth.

## 2. The token, and which engine's part does each stage

Decode, one token, three cards, pipeline by layer range; the host is not in
the loop between layers.

| stage | part taken from | why (measured) | what is new |
|---|---|---|---|
| dense trunk (norms, projections, gated attention / GDN recurrence, lm_head) | **hipFire**: HIP kernels tuned for gfx1100, whole-token residency, one command stream per card | 80–130 tok/s on one card is the only resident-trunk number on this hardware; Colibri's per-op Vulkan path is 20× slower on the same class (§F3-STEP2A) | ported to a layer-range pipeline across three cards; the hidden vector crosses cards by P2P copy, never through the host |
| attention + KV at 256k | **llama.cpp**'s per-architecture graphs as the reference for what each model computes (qwen4exp, deepseek4, glm-dsa); kernels from hipFire where they exist; **Colibri's F7 batched-attention and F6a/F9a indexer** as the source for GLM's DSA top-k | llama.cpp is the only code that runs all three architectures correctly today; its speed is not the reason to take it | GLM's DSA indexer on the GPU (F9's closed item, reopened here because in a resident trunk it is no longer a 0.3 ms round trip) |
| routed experts, resident | **hipFire**'s MoE GEMV kernels and quant format (mq4r) OR llama.cpp's IQ4_XS/Q4_K kernels — **to be measured head to head on gfx1100** (§4, M1) | hipFire's one-card number vs llama.cpp's 15 tok/s at similar residency is the strongest hint, not a measurement of the kernels alone | one expert format per model chosen by measured kernel speed × measured quality (the quality harness stays for this, as a workload parameter) |
| routed experts, non-resident | **Colibri**: host-RAM ring + async streaming (F2), placement from usage histograms with a VRAM budget (G6), mmap'd shard mapping (`st.h`) | 2.03× prefill on the ring; the tier logic is the one thing no other engine has | **router lookahead** (§3.1), **cross-card replication of the hottest experts** (§3.2), CPU as the last-resort device only (22 GB/s vs 61 over PCIe) |
| prefill | **Colibri F2/F7**: chunked, batched, experts streamed per layer, attention batched on the GPU | measured 1.75–2.03× on the 18k turn | run on all three cards at once by layer range, with the stream fed from the ring at the 61 GB/s aggregate; target ≤ 1 ms/token → a cold 256k prompt in ~4–5 min (projected: 61 GB/s ÷ ~2 GB of experts per chunk-layer wave, refuted or confirmed by M4) |
| gateway | **Colibri** `openai_server.py`: KV slots, prefix checkpoints on disk, the conversation ledger, deterministic tool-call ids, pinned memory block | the only serving layer that survives Open WebUI's re-sent transcripts (P7–P12); 2–4 s new chats on a 4.4k tool block | checkpoints sized for 256k conversations (KV per token per model measured in M3) |

## 3. What has to be invented

3.1 **Router lookahead prefetch.** A miss costs one expert's bytes over PCIe
(~2.5 MB at int4 for Qwen3.8: 16 MB × 48 layers ÷ 512... measured per model in
M1) plus latency; at 61 GB/s a 20 MB miss wave is ~0.3 ms, the same as a sync
call. Hiding it needs the NEXT layer's expert ids before the next layer runs.
Candidate: run layer l+1's router on layer l's output before layer l's expert
sum is complete (the router input is the post-attention normed state, which
exists before the MoE sum) — exact for the layer's own attention, approximate
across the residual; measure the hit rate of the prediction (M2). Fallback
candidate: MTP draft (F5) as a two-token lookahead.

3.2 **Hottest experts replicated on every card.** In a layer-range pipeline
each expert lives on one card; the histogram's top few hundred experts per
layer-range are cheap to duplicate (a few GB of 72) and every card serving a
prefill chunk then has them local. Measured need: the histogram's
concentration (`~/.glm53_explain.bin` exists for GLM; build the same for the
other two in M2).

3.3 **Layer-range assignment that respects the shared PCIe link.** The two
cards on one upstream link must not both stream misses at full rate: assign
them the layer ranges with the lowest miss rate (from the histogram), the
lone card the highest. Measured input: per-layer miss bytes from M2.

3.4 **256k prefix checkpoints.** At 256k a conversation's KV is hundreds of
MB to a few GB (M3 measures it per model); disk write at NVMe rate (723 MB/s
measured, memory `decode-slowdown`) is seconds — acceptable once per
conversation; the ledger logic is unchanged.

## 4. The measurement program (each closes a design choice; none is a gate on a build)

| id | question | how | closes |
|---|---|---|---|
| M0 (running) | cost of RAM-resident experts under llama.cpp's HIP MoE path: 32k vs 256k window on Qwen3.8 | `f11_ladder_chain.sh` with `F11_CTX`, `--fit-print` | the miss-cost term in §1(a) on real kernels |
| M1 | expert kernel + format head to head on gfx1100: hipFire mq4r vs llama.cpp IQ4_XS/Q4_K vs Colibri int4-g64, same expert shapes (Qwen3.8: 512×48, hidden 2560), batch 1 and batch 32 rows | one microbenchmark binary per candidate, weights in VRAM, 1 000 iterations, median; no engine, no gateway stop | §2 row 3 |
| M2 | expert usage histograms and per-layer miss bytes for Qwen3.8 and DeepSeek at 256k-scale prompts; hit rate of router lookahead | Colibri's histogram tooling ported to read llama.cpp's router output (a debug hook, CPU) | §3.1, §3.2, §3.3 |
| M3 | KV bytes per token per model at 256k; attention time per layer at 32k/128k/256k, candidate kernels | from llama.cpp's own graph on one card, timers | §2 row 2, §3.4 |
| M4 | streaming prefill on three cards: Colibri's F2 ring driven from three queues at once | `vk_stream_probe`-style, real expert sizes | the ≤ 1 ms/token projection |
| M5 | pipeline boundary cost: P2P copy of one hidden vector card→card, and one layer-range's whole-token command stream on hipFire's kernels | microbenchmark | §1(b), §1(c) |

Order: M0 (in flight), M1 and M3 in parallel (no gateway stop for either),
M2, then M5, then M4. Tier: M1/M5 are kernel work (Opus writes the
benchmarks from the two codebases; the orchestrator reads the hot loops
first); M2/M3/M4 are Sonnet ports of existing tooling.

## 5. What this design refuses

- No RCCL / tensor parallelism (measured to lose).
- No per-op host round trips in decode (0.3 ms each).
- No KV paging through the host at depth.
- No plain 4-bit experts without a measured quality number for that model
  (Qwen3.6: KL 0.032; GLM int3: 16 of 1 232 predictions).
- No "serve engine X for model Y": the engines are quarries.

## 6. Targets (projected, to be refuted by the program)

Decode at 256k for Qwen3.8-Flash-Next with ~¾ of the experts resident: the
resident work at hipFire-class efficiency is ~10 ms/token (from 80–130 tok/s
on a comparable active size), the miss wave ~1–3 ms if lookahead hides it,
attention at 256k unknown (M3). If M3 is ≤ 10 ms, the target is **≥ 40 tok/s
at 256k** against llama.cpp's 6 today. Prefill ≤ 1 ms/token → **a cold 256k
prompt in ~5 min** against 17. Both are arithmetic until M1/M3/M4 say
otherwise, and the plan's own rule applies: step 0 is a measurement.
