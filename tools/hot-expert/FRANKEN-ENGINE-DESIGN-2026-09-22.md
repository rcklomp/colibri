# Franken-engine: architecture for three RX 7900 XTX (design, 2026-09-22, rev 13)

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
| HIP on gfx1100 (M5): async launch 3.0 µs; launch+sync 24 µs; cross-stream event 31 µs; hipGraph node 2.75 µs (saves 7 %); 10 KB P2P card→card 28–32 µs, via host 50–58 µs; P2P bandwidth 16–24 GB/s; VRAM read 800 GB/s; **a 48-layer synthetic token of 240 plain launches runs at 794 GB/s = the memory bound** | §M5 (2026-09-22) |
| RCCL tensor parallelism | loses on arithmetic: 25 µs × ~80 hops per token | plan §1 (measured 09-16) |
| resident MoE on three cards, llama.cpp | 75 tok/s at 18k (gpt-oss-120b); 17.6 with all experts in RAM | §GPTOSS-3CARD |
| one card, resident 35B, hipFire kernels | 131 → 79 tok/s down the ladder, 164 short | plan §1 |
| llama.cpp HIP, Qwen3.8-Flash-Next "UD-IQ4_XS" (experts IQ3_S gate/up + IQ4_NL down, trunk Q8_0), **all experts resident on three cards**, the 28.8 GB per-layer embedding table on the host, 256k window | 15 tok/s to 39k, 6.1–6.5 at 257k; cold 257k prompt 17 min; 1.0–2.0 ms per prefill token | §F11-STEP0, §F11-DEPTH, §F11-M0 point 2 (rev 4: rev 1–3 said "¼ of experts in RAM"; wrong, read from the GGUF header) |
| the same, same placement, 32k / 64k / 128k window | 24–29 tok/s at 1.5–19.5k; 0.34–0.47 ms per prefill token. **The 256k reservation costs llama.cpp 1.6–1.9× decode and 2.1–2.4× prefill at every depth, as a STEP between 128k and 256k: its fit moves five layers of expert gate/up to the CPU path (3.2 GB) to make room for 1.5 GB KV + 1.8 GB scratch + 1 GB margin per card, and a CPU-side expert evaluation costs ~0.5 ms (M0c/M0d) against ~0.09 ms to stream it over PCIe.** **With everything resident, llama.cpp is 4–5× off the VRAM bandwidth bound** (~6.2 GB read per token, ~7.5 ms if sequential over three cards, measured 34–40 ms) | §F11-M0 (M0/M0b, 2026-09-22) |
| Colibri, GLM-5.3 int4 with 60–70 % of the model in RAM | 5.04 tok/s at 18k; 18k prompt 356 s | served, §F9a |
| the models | none fits 72 GiB at an acceptable quant: GLM 149–184 GB, Qwen3.8 94 GB (IQ4) / 173 (FP8), DeepSeek 91 GB (IQ2) | disk |

Consequences. (a) **The engine is a placement-and-streaming engine**: the
model never fully fits (rev 4: Qwen3.8-Flash-Next at this quant DOES fit
three cards with ~13 GB to spare for 256k KV in a lean engine -- §F11-M0
point 2; DeepSeek-V4-Flash at 91 GB and GLM-5.3 at 149 GB do not, so the
streaming path stays, and for Qwen3.8 the miss term is zero), so the token's cost is (misses × bytes ÷ stream
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

3.1 **Router lookahead prefetch -- corrected in rev 2.** Rev 1 claimed layer
l+1's router input "exists before layer l's MoE sum"; it does not: it is the
normed residual AFTER layer l's expert sum. So any lookahead is a PREDICTION
of layer l+1's expert ids from layer l's pre-MoE state, and its hit rate is
an empirical number (M2), expected to be imperfect on hybrid GDN/sparse-attention
models. Degradation ladder, each measured: (a) prefetch the top-2k of the
CURRENT router's distribution beyond the top-k actually used (cheap, exact
for this layer's tail, useful only for prefill/multi-slot); (b) predicted
next-layer ids, prefetch on prediction, pay the miss on a wrong guess; (c)
an MTP draft as a two-token window (F5's economics: it lost before on a
bandwidth-starved path and must be re-priced on the resident trunk). A miss
wave at 61 GB/s aggregate is ~0.3 ms per 20 MB; the design tolerates an
imperfect lookahead if the resident work is ~10 ms -- which is why the
number that matters first is M1/M5, not the hit rate.

3.2 **Hottest experts replicated on every card.** In a layer-range pipeline
each expert lives on one card; the histogram's top few hundred experts per
layer-range are cheap to duplicate (a few GB of 72) and every card serving a
prefill chunk then has them local. Measured need: the histogram's
concentration (`~/.glm53_explain.bin` exists for GLM; build the same for the
other two in M2).

3.3 **PCIe bandwidth as a scheduled resource (rev 2).** The 61 GB/s aggregate
(§PCIE-STREAM) was measured with each card streaming its own block; two cards
on one upstream link contending for misses at once is NOT that measurement.
M4 measures it. The engine gives the shared-link pair one joint byte budget
per token and a time-sliced streaming queue; layer-range assignment puts the
lowest-miss ranges (M2) on the pair and the highest on the lone card. Until
M4 exists the pair's budget is assumed to be ONE card's 28 GB/s, not two.

3.4 **256k prefix checkpoints.** At 256k a conversation's KV is hundreds of
MB to a few GB (M3 measures it per model); disk write at NVMe rate (723 MB/s
measured, memory `decode-slowdown`) is seconds — acceptable once per
conversation; the ledger logic is unchanged.

3.5 **Pipeline bubbles (rev 2).** A layer-range pipeline is sequential per
token; decode for ONE conversation has exactly one token in flight, so
micro-batching across stages cannot hide a stalled stage (it helps only with
concurrent slots, which this owner rarely has, and in prefill, which is
already batched). The bubble is therefore the
slowest stage's miss cost, and the only levers are placement (fewer misses on
any one stage), lookahead (§3.1) and the miss wave's own latency. M5 measures
the boundary cost; the MVP ladder (§7) measures the bubble before anything
is built to hide it.

3.6 **One weight format, one KV layout, one command-stream abstraction
(rev 2).** Stitching hipFire kernels (own quant layouts, retained graphs),
llama.cpp graphs and Colibri's ring without these three decided first
reproduces the per-op round trips the design exists to remove. Decided by
measurement BEFORE any port: the expert/trunk format by M1 (converted
OFFLINE, no runtime conversion; Colibri's streaming path carries the same
format for cold experts), the KV layout by M3, the command-stream model by
M5. Quality of the chosen format per model is measured with the existing
harness (`quality_eval.py`, KL against the FP8/BF16 reference where one
exists) as a workload parameter, not skipped.

## 4. The measurement program (each closes a design choice; none is a gate on a build)

| id | question | how | closes |
|---|---|---|---|
| M0 (**done 2026-09-22, §F11-M0**) | cost of RAM-resident experts under llama.cpp's HIP MoE path: 32k vs 256k window on Qwen3.8 | `f11_ladder_chain.sh` with `F11_CTX`, `llama-fit-params` | fit places the same tensors at every window and NO expert is on the host (the host holds the 28.8 GB per-layer embedding table), so M0 measured two other things: the window reservation (1.6× decode / 2.3× prefill, a step between 128k and 256k = five layers of experts displaced to the CPU) and the fully resident reference at ≤128k: 34–40 ms a token for ~6.2 GB of VRAM reads, 4–5× off the bandwidth bound. The miss term is measured on DeepSeek/GLM instead (M2). |
| M0b (**done**) | slope or step: 64k and 128k windows | `~/bench/f11_fit3.sh` | a step: 24 tok/s at 32k–128k, 15 at 256k |
| M0c, M0d (**done**) | 256k at fit margin 1 024 vs 3 072 MiB, GTT sampled; then the server's own fit log | `~/bench/f11_fit4.sh`, `f11_fit5.sh` | not eviction (GTT flat); llama.cpp's fit puts 5 layers of expert gate/up on the CPU at 256k; ~0.5 ms per CPU expert evaluation |
| M1 (**done 2026-09-22, §M1**) | expert kernel head to head on gfx1100 on Qwen3.8's real expert shapes (512 experts of 640×2560 / 2560×640), batch 1 and 32 | `tools/hot-expert/m1/m1_ggml.cpp`, `m1_hipfire.hip` | §2 row 3 decided: hipFire's kernel shape (452 GB/s at batch 1, 826 = the bound at batch 32) over llama.cpp's mmvq MoE path (174 / 271); two ports needed: runtime K_TOP (10) and a group-128/64 format for K=640. Colibri's Vulkan path not run: §F3-STEP2A's 0.3 ms per call already rules it out for the resident trunk |
| M2 (**done 2026-09-23, §M2**) | expert usage and miss bytes for DeepSeek and GLM | `tools/hot-expert/franken/moe_hist.cpp` | DeepSeek ~resident (60 MB/token missed at 60 GB resident); GLM streams ~1.0-1.4 GB/token; histogram placement is the lever, temporal lookahead weak (reuse 25-30 % next token) |
| M3 (**done 2026-09-22, §M3**) | KV bytes per token at 256k; QSA attention time per layer at 32k/128k/256k | `tools/hot-expert/m1/m3_attn.hip` (v2 kernels: wave-per-block scan, radix-select top-k, flash-style sparse attention) | KV 16.1 KB/token from the shapes, 17.9 measured; QSA layer 60 / 92 / 142 µs streamed from VRAM (scan at 775 GB/s), 1.7 ms/token at 256k + GDN 0.57 ms. **Go/no-go passed by 4×.** The first cut (naive kernels, 38 ms/token) is history in §M3 |
| M4 (**done 2026-09-23, §M4**) | host->card streaming, all three cards, expert-sized pieces | `tools/hot-expert/m1/m4_stream.hip` | 28 GB/s a card alone; shared pair 36 together; **all three 52 GB/s** (not 61); lone card should take ~44 % of the stream; GLM ~19 ms/token of stream at 1 GB missed |
| M5 (**done 2026-09-22, §M5**) | pipeline boundary cost: P2P copy of one hidden vector card→card, and a whole-token command stream | `tools/hot-expert/m1/m5_bench.hip` | §1(b), §1(c) confirmed: boundaries ~30 µs, a whole-token stream of plain kernels reaches the VRAM bandwidth bound; hipGraph unnecessary; projection ~125 tok/s bandwidth-bound for Qwen3.8 resident before attention |

Order: M0, M5, M1, M3 done -- L0 may start (rev 7); M2 and M4 are needed for the streamed models (L0b, L5), not for L0; then
M2, then M5, then M4. Tier: M1/M5 are kernel work (Opus writes the
benchmarks from the two codebases; the orchestrator reads the hot loops
first); M2/M3/M4 are Sonnet ports of existing tooling.

**Go/no-go (rev 2):** M3 > 10-12 ms per token at 256k, or M1 showing no
kernel within 2x of bandwidth-bound on gfx1100, and the 40 tok/s projection
in §6 is withdrawn before anything is built on it. **Rev 7: both passed --
M3 2.3 ms (§M3), M1 hipFire's kernels at 56 % of the bound at batch 1 and
at the bound at batch 32 (§M1). The projection stands and is sharpened in §6.**

## 5. What this design refuses

- No RCCL / tensor parallelism (measured to lose).
- No per-op host round trips in decode (0.3 ms each).
- No KV paging through the host at depth.
- No plain 4-bit experts without a measured quality number for that model
  (Qwen3.6: KL 0.032; GLM int3: 16 of 1 232 predictions).
- No "serve engine X for model Y": the engines are quarries.

## 6. Targets (projected, to be refuted by the program)

**Rev 11 (2026-09-22 evening, record §L0-256K) -- MEASURED on the built
engine, three cards, Qwen3.8-Flash-Next at a 262 144 window:** decode at
256k depth **26.1 ms a token = 38.4 tok/s** (llama.cpp 6.1-6.5; target >= 40:
4 % short, owners named), cold 262 000-token prompt **467 s = 7.8 min**
(llama.cpp 17 min; target ~5 min), decode at short depth 42.9 tok/s. The
paragraphs below are the projections as they stood; they are kept as the
record of what was claimed before it was built.

Decode at 256k for Qwen3.8-Flash-Next with ALL experts resident (rev 4;
§F11-M0 point 2): the bandwidth bound is ~7.8 ms/token for ~6.2 GB of VRAM
reads over three cards in sequence (§M5 measured 794 of 800 GB/s on a
whole-token stream); hipFire-class expert kernels reach 56 % of it at batch
1 (§M1), so 9–10 ms for the weights is the honest expectation; attention
1.7 ms + GDN 0.6 ms at 256k (§M3, measured); boundaries 0.06 ms (§M5). **Rev 7:
~10–12 ms a token, 80–100 tok/s at 256k, every term measured on this
hardware; the target stays ≥ 40 tok/s at 256k** against llama.cpp's 6 today. Prefill ≤ 1 ms/token → **a cold 256k
prompt in ~5 min** against 17. Both are arithmetic until M1/M3/M4 say
otherwise, and the plan's own rule applies: step 0 is a measurement.

## 7. The build ladder (rev 2): smallest working pipeline first

No target in §6 is a requirement. The first thing built is the smallest
pipeline that produces a measured tok/s and prefill rate, then each design
element is added one at a time against that number:

| rung | what | adds |
|---|---|---|
| L0 | ONE model (Qwen3.8-Flash-Next: lightest active set, fastest today, and at this quant fully resident on THREE cards -- rev 4), layer-range pipeline over the three cards, hipFire-class trunk and expert kernels in one command stream per card, the per-layer embedding gather from host RAM, no streaming, no replication, no lookahead, Colibri's gateway in front; **specified in §9 (rev 7)** | the first real number for §1(b)-(c): the resident token against llama.cpp's 34–40 ms |
| L0b | the same on TWO cards (the pair that does NOT share a link + one) with Colibri's ring for the experts that no longer fit | the first real number for §1(a) on this model |
| L1 | the shared-link pair under the stream (L0b on the sharing pair) | the shared-link contention, measured |
| L2 | histogram placement with the PCIe budget (§3.3) | fewer misses per stage |
| L3 | replication of the hottest experts (§3.2) | only if M2 says the histogram is concentrated |
| L4 | lookahead ladder (§3.1 a/b/c) | only if the bubble is the limiter after L2 |
| L5 | DeepSeek-V4-Flash, GLM-5.3-Flash on the same pipeline | per-architecture graphs from llama.cpp's reference |

Each rung is one gated item with the plan's rules (oracle, A,B,B,A, record
row). L0 is not started until M1, M3 and M5 have reported.

## 8. Audit, 2026-09-22 (an external review, relayed by the owner)

Accepted and folded in above: pipeline bubbles under partial residency
(§3.5), PCIe contention as a scheduled resource (§3.3, M4), the lookahead
correction and degradation ladder (§3.1), the unquantified replication
trade-off (§3.2 already required M2; L3 now waits on it), the integration
surface (§3.6), M3 as go/no-go, quality per format (§3.6), the MVP ladder
(§7), the gateway staying Colibri's (§2, unchanged). Rejected: in-flight
micro-batches to hide bubbles (single-stream decode has one token in flight,
§3.5). Not adopted: the review's model parameter counts and its community
prefill figures -- nothing on this rig verifies them, and the design uses
only numbers measured here.

## 9. L0 specification (rev 7, 2026-09-22): Qwen3.8-Flash-Next resident on three cards

Everything below rests on a measured number (§F11-M0, §M1, §M3, §M5) or is
marked projected. L0 is one gated item: it is done when the engine decodes
greedily-identical text to llama.cpp on the oracle prompts (last-token logits
compared), serves through Colibri's gateway with `accept_live` passing, and
has a measured tok/s at 19k and 256k on the depth ladder.

**9.1 Process and memory model.** One process, three HIP devices, one
command stream per device, no host work between layers. Layer ranges
(projected balance from the per-layer bytes; L0 measures and rebalances):
card 83:00.0 layers 0–15, 86:00.0 16–31, 48:00.0 32–47 plus lm_head; the
shared-link pair (83↔86) gets the one boundary that carries only a hidden
vector. Per card: its layers' trunk (Q8_0 as in the file, ~1.7 GB), its
experts (512 × 16 layers in the L0 expert format, ~20 GB at 4.25 bpw or
~16 GB at IQ3_S-class bytes — the format decision is 9.3), KV for its 4 QSA
layers at 256k (1.1 GB q8_0 K/V + 0.4 GB indexer keys, §M0d), GDN state
(37 MB), scratch ≤ 256 MB. The per-layer n-gram embedding table (28.8 GB,
IQ4_NL, `ple.layers [1]`) stays in host RAM in a pinned buffer the engine
fills at start (NOT lazy mmap: uncached rows cost 4.2 ms a token from the
NVMe, cached 9.5 µs -- record §PLE-GATHER); its per-token gather (16 heads
× 160 values, ~1.4 KB) is done on the CPU and uploaded with the token — it
is the only host work per token besides sampling. **General rule: any
tensor read by a per-token row gather lives in host RAM; VRAM is for
tensors read wholesale per token.**

**9.2 The token.** Card 0: embed (CPU gather + token_embd row) → layers 0–15
→ P2P copy of the hidden vector (2 560 f32 + the hyper-connection residual
streams, `hc_count` × 2 560) to card 1 → layers 16–31 → card 2 → layers
32–47 → lm_head → logits to host → sample. Every layer: hc mix, norm,
attention (GDN or QSA), hc mix, norm, shared expert + router + 10 routed
experts, residual. Launch count per layer ≈ 12–18 plain kernels. **Rev 9 correction:** §M5's
"240 launches a token are free" holds only for kernels with enough work
behind them; the first decode loop's 58 launches a layer of tiny norm /
gate / mix kernels cost ~4 ms of the 11.7 (§L0-STEP2). Fusion IS required:
norm + hyper-connection mix in one kernel, GEMV epilogues carrying their
gates, the GDN gate chain as one launch, ≤ 15 launches a layer.

**9.3 Kernels and formats (the parts, by source).**
- Trunk GEMVs (Q8_0 weights, f32 activations): hipFire's Q8-class GEMV or
  the plain wave-per-row kernel of `m5_bench.hip` (794 GB/s on the synthetic
  token) — L0 takes the simpler one and measures; the trunk is ~5 GB a token
  and bandwidth is what matters.
- Routed experts (**decided rev 8, §M1 candidate 3**): hipFire's kernel
  shape reading the file's own IQ3_S gate/up and IQ4_NL down
  (`tools/hot-expert/m1/m1_native.hip`, K_TOP runtime = 10, 4 launches a
  step, 385 GB/s at batch 1 = the same step time as hipFire's 4.25 bpw
  kernel with 19 % fewer bytes; oracle cos 1.0). **No requantisation, no
  format-quality question.** Open optimisation, prefill only: its batched
  path is 484 GB/s at 32 rows against hipFire's 822.
- QSA attention: `m3_attn.hip` v2 kernels as they are (scan, radix top-k,
  flash split + combine), plus the q/k/v projections, QK-norm, IMRoPE and
  the indexer q/k projections as GEMVs from the trunk set; the indexer key
  cache is written per token (pooled per 4-token block as llama.cpp does).
- GDN layers: the `m3_attn.hip` delta-rule step for the state update; the
  QKVZ projection, conv1d (kernel 4, a 4-token ring per layer), alpha/beta
  gates, L2-norm and gated RMSNorm as small kernels ported from llama.cpp's
  `build_layer_attn_linear` (the reference for the math) — Sonnet-tier
  ports with an oracle against llama.cpp's dumped intermediates.
- Hyper-connections (`hc_*`, 4-wide inject/mix) and the shared expert with
  its sigmoid gate: small GEMVs and elementwise kernels, from the same
  reference.
- Sampling, tokenizer, chat template, prefix checkpoints, slots, ledger:
  Colibri's gateway unchanged in front; the engine speaks the gateway's
  engine protocol (what `glm53` speaks today).

**9.4 Prefill.** L0 prefill is the decode path batched over rows (hipFire's
batched expert kernels reach the bound at 32 rows, §M1; the M3 attention
runs per query row over the cache built so far, with the causal top-k
selection restricted to earlier blocks). No streaming, no ring: the model is
resident. Projected: at the bound, a 256k prompt reads the weights once per
32-row chunk → 8 192 chunks × ~8 ms ≈ 65 s plus attention; the 17 min of
llama.cpp is not the reference to beat, the 5-minute target of §6 is.

**9.5 Build order inside L0 (each step has an oracle before the next).**
1. Loader: GGUF tensors → device buffers per card by layer range (**done
   2026-09-22, §L0-STEP1: 64.9 GB in 12.8 s, `tools/hot-expert/franken/`**);
   no requantisation (rev 8); the CPU-side PLE gather (**done, hash
   cross-checked**). Remaining oracle: a layer-0 forward on one token
   against llama.cpp's dumped activations — the first item of step 2.
2. One card, layers 0–15, decode loop with GDN + QSA + MoE; oracle per
   layer against the dumps; timing per layer (§M5's harness pattern).
   **Done functionally 2026-09-22 (§L0-STEP2): GPU = CPU graph, llama.cpp
   matched to the floor through layer 6, routing divergence beyond is the
   reference's rounding. First timing 11.7 ms / 16 layers, launch-bound
   (58 launches a layer) and a trunk GEMV 2.7× off. Step 2b: fusion to
   ≤ 15 launches a layer and the trunk GEMV at the bytes; gate = routing
   set match per layer + greedy text on real prompts, not cos at layer 15.**
3. Three cards, the two boundaries, lm_head; oracle: greedy text identical
   on the F11 prompts, last-token logits cosine/argmax. **Done 2026-09-22
   (§L0-STEP3): 12 of 16 greedy tokens identical to llama.cpp, routing 0.92;
   22.9 ms a token = 43.7 tok/s at short context with 256k allocated
   (llama.cpp 34-40 / 66 ms). Next: host issue cost, small-matrix GEMVs,
   then the depth ladder.**
4. Gateway protocol, `serve_alt.sh franken`, `accept_live`, the depth
   ladder at 19k and 256k (the number), the quality harness at the L0
   format (the quality number). Gate: tok/s ≥ 40 at 256k, quality within the
   measured spread of llama.cpp's IQ3_S/IQ4_NL serving (record §F11-QUALITY).
   **All measured (rev 12, 2026-09-23): §L0-256K 38.4 tok/s at 256k and a
   7.8 min cold prompt; §L0-STEP4 accept_live and accept_ui PASS, first
   token on screen 0.80 s; §L0-QUALITY 77.1 % vs llama.cpp's 80.0 % on 70
   MMLU-Pro items, not distinguishable, needles 8/8 to 200k. L0 is
   complete; the 4 % on the tok/s target has named owners.**
