# GLM-5.3-Flash on the Franken engine — L5 GLM steps 1-3 (2026-09-24)

The model, the path DeepSeek-V4 took (DEEPSEEK4.md, the template), and what
steps 1-3 built and proved. **Everything below that says "measured" was
measured on the CPU arm or by the compiler; nothing here is a GPU
measurement.** Section 5 (placement) and section 8 (kernels) are plans and
projections.

Model: `~/models/GLM-5.3-Flash/UD-IQ4_XS/*.gguf` (5 shards, 149 GB, arch
`glm5next`, 1 412 tensors). Reference: `~/src/llama-glm53` @ `39931761a`,
`src/models/glm5next.cpp` (the graph), `src/models/deepseek4.cpp` (the
hyper-connection builders it inherits), `src/models/delta-net-base.cpp`
(the KDA recurrence), `src/llama-kv-cache-kpool.cpp` (the pooled indexer's
masks), `src/llama-graph.cpp` (build_attn_sparse, build_ffn,
build_moe_ffn), and the CPU ops in `ggml/src/ggml-cpu/ops.cpp`
(gated_delta_net, ssm_conv, norm, dsv4_hc_*).

## 1. The header

| key | value | used as |
|---|---|---|
| block_count / nextn_predict_layers | 46 / 1 | 45 text layers (`N_LAYER`); block 45 is the NextN (MTP) block, not run |
| embedding_length / vocab | 4096 / 154 880 | |
| attention.head_count | 64 | KDA heads and MLA query heads |
| attention.head_count_kv | **per-layer array**: 1 on layers 3, 7, ..., 43 and 45, else 0 | 0 = **KDA** layer (34 of 45), 1 = **DSA** layer (11 of 45, plus NextN) |
| leading_dense_block_count / feed_forward_length | 3 / 12 288 | layers 0-2 have a dense FFN; 3-44 are MoE |
| kda.head_dim / ssm.conv_kernel / kda.gate_lower_bound | 128 / 4 / -5 | KDA: q/k/v 64 x 128, a 4-tap causal conv, g = -5·sigmoid(...) |
| attention.q_lora_rank / kv_lora_rank | 1536 / 512 | MLA: wq_a → norm → wq_b; the latent K = V row is 512 |
| key_length_mla / value_length_mla | 256 / 256; **rope.dimension_count 0** | nope-only MLA: no rope anywhere in the text tower |
| attention.indexer.head_count / key_length / top_k / kpool | 32 / 128 / 2048 / 4 | the pooled lightning indexer: pools of 4 cells, top-512 POOLS |
| hyper_connection.count / sinkhorn_iterations / epsilon | 4 / 20 / 1e-6 | DeepSeek-V4's four-stream mix, unchanged |
| expert_count / used / ff; shared | 288 / 8 / 2048; 1 shared, 2048 wide | |
| expert_gating_func / weights_norm / weights_scale | 2 = sigmoid / true / 2.5 | selection by sigmoid + exp_probs_b; weights unbiased, normalised, x2.5 (routed only) |
| swiglu_clamp_exp / _shexp | 10 on every layer | also the dense FFN of layers 0-2 (build_ffn applies it for this arch) |
| layer_norm_rms_epsilon / layer_norm_epsilon | 1e-5 / 1e-6 | RMS norms (hc flat norm included) / the indexer's LayerNorm |

`glm5_shapes.h` compiles these in; `glm5_model.cpp check_hparams()` reads
every one back (and the head_count_kv array against `is_dsa()`) and refuses
a file that disagrees.

## 2. The token, op by op

Embed: `token_embd` (Q8_0) row → **copied** into the four hyper-connection
streams (`hc_init`). Per layer, twice (attention module, FFN module),
**hc_pre** exactly as DeepSeek-V4 (rms_norm of the flat 16 384 → `hc_*_fn`
(Q8_0 here, F32 in DeepSeek) [16384→24] → pre/post/comb with 20 Sinkhorn
passes → `x = Σ pre[h]·H[h]`), and after the module **hc_post**. Then
`attn_norm` / `ffn_norm` (RMS, eps 1e-5).

**KDA layer (34 of 45)**, on `cur = attn_norm(x)` (build_kda_layer):
`q, k, v = wq·cur, wk·cur, wv·cur` (8192 each) → one causal 4-tap conv over
the concatenated 24 576 channels with the previous 3 inputs → SiLU →
l2-norm q and k per 128-head (eps 1e-6) → per-channel gate
`g = -5·sigmoid(-(ssm_a[h]·(f_b·f_a·cur + dt_bias)))` (ssm_a holds
-exp(A_log)) → `beta = sigmoid(ssm_beta·cur)` [64] → the **KDA recurrence**
per head (state 128 x 128 f32): `S[i][:] *= exp(g[i])` (per KEY channel,
the difference from Qwen3.8's GDN), `d = (v - Sᵀk)·beta`, `S += k dᵀ`,
`o = Sᵀq / √128` → RMS norm per head with `ssm_norm` × **sigmoid**(g_b·g_a·cur)
→ `attn_output` [8192→4096]. f, g, beta read the layer input, not the
convolved q/k/v.

**DSA layer (11 of 45)**, build_dsa_layer + build_indexer:
- `qr = rms_norm(wq_a·cur)·q_a_norm` [1536].
- The indexer's STORE (always): `ik = LayerNorm(indexer.attn_k·cur)·w + b`
  [128], `gate = indexer_compressor_gate·cur` [128], both **f16 per cell**.
  When a pool of 4 cells completes (`(p+1) % 4 == 0`): per dim,
  `probs = softmax_j(gate_j + ape[j])`, `pooled = Σ_j key_j·probs_j`, f16.
- The indexer's SCORING (only when n_ctx > 2 051): `iq = indexer.attn_q_b·qr`
  [32 x 128] (no rope), `w = indexer.proj·cur / √(128·32)` (F32 weights),
  `score_b = Σ_h relu(iq_h · pooled_b)·w_h` over the (p+1)/4 complete pools,
  **top-512 pools** (a pool-level top-k: a cell cut would split pools).
- `q = wq_b·qr` [64 x 256]; `kv = rms_norm(wkv_a_mqa·cur)·kv_a_norm` [512],
  cached f16; **absorbed**: `q_abs[h] = wk_b[h]·q[h]` [64 x 512].
- Attention (MQA, no sinks, scale 1/√256): every head against the shared
  latent rows of the attended cells; the cells are **every cell ≤ p while
  the visible pools number ≤ 512**, otherwise the 2 048 cells of the
  selected pools plus the TAIL cells `(p+1)/4·4 .. p` (at most 3; ≤ 2 051
  keys). → `o[h] = wv_b[h]·att[h]` [256] (the latent decompressed) →
  `attn_output` [16384→4096].

**FFN**: layers 0-2 `down(swiglu_clamp(gate·x, up·x, 10))` (12 288 wide);
layers 3-44 `router = sigmoid(ffn_gate_inp·x)` (F32 [4096→288]), top-8 of
`probs + exp_probs_b`, weights = probs[ids] / Σ × 2.5, per expert
`down(swiglu_clamp(gate·x, up·x))` weighted and summed, plus the clamped,
**unscaled** shared expert.

Head: the **unweighted mean** of the four streams (`hc_mean`, not DeepSeek's
learned hc_head) → `output_norm` → `output` (Q6_K [4096→154 880]).

**Skipped**: the NextN block 45 (its own speculative-decoding item, as
DeepSeek's MTP); llama.cpp's rollback snapshot planes (a serving concern).

## 3. KV at 256k

| state | per layer | 256k (262 144 cells) |
|---|---|---|
| DSA latent K = V, f16, 11 layers | 512 x 2 B = 1 024 B a token | 2.95 GB |
| DSA pooled indexer key, f16, 11 layers | 128 x 2 B per 4 tokens = 64 B a token | 0.18 GB |
| DSA per-cell indexer key + gate, f16 | a ring of 4 + C - 1 cells (C = prefill chunk) | ~0 |
| KDA state, f32, 34 layers | 64 x 128 x 128 x 4 B = 4.19 MB, fixed | 0.143 GB |
| KDA conv window, f32, 34 layers | 3 x 24 576 x 4 B = 295 KB, fixed | 0.010 GB |
| **total** | **11 968 B a token + 0.153 GB fixed** | **3.29 GB** (0.86 / 1.14 / 1.14 GB a card by the section-5 split, plus ~0.05 of KDA state each) |

llama.cpp keeps a full indexer row a cell (key, gate and pooled head: 768
B) beside the 1 024-byte latent, 19.7 KB a token, 5.2 GB at 256k; this
engine keeps the per-cell key and gate only until their pool closes. The
cost of that choice is a serving one: a prefix checkpoint must carry the
open pool's ring (up to 3 cells) as recurrent state tagged with its
position, exactly like the KDA state and DeepSeek's compressor rings.

Per token at 256k, a DSA layer scans 65 536 pooled keys (16.8 MB, 268 M
MAC: 32 heads x 128), 11 layers 185 MB and 2.95 G MAC a token, and attends
≤ 2 051 latent rows (2.1 MB). A KDA layer reads and writes its 4.19 MB
state: 285 MB a token over 34 layers, depth-independent.

## 4. Tensor formats in this UD-IQ4_XS file (text tower, layers 0-44)

| tensor | format (layers) | host decoder | GPU kernel |
|---|---|---|---|
| ffn_gate_exps, ffn_up_exps | **IQ3_S** (41 layers), **IQ4_XS** (layer 11) | exists (m1_native_decode.h / decode_quant.h) | lane decoders exist; the table-driven expert kernel (`k_ds4_moe`) needs these instantiations |
| ffn_down_exps | **IQ4_XS** (39 layers), **Q6_K** (layers 11, 12, 44) | exists | as above |
| all KDA / DSA / dense / shared-expert / indexer projections, hc_*_fn | Q8_0 | exists | exists (L0 GEMV and GEMM) |
| attn_k_b [256,512,64], attn_v_b [512,256,64] | Q8_0, per-head slices | exists | L0 GEMV per head works; a batched per-head kernel wanted (64 launches otherwise) |
| ffn_gate_inp, indexer.proj, ssm_a, ssm_dt, ssm_conv1d_*, norms, ape, exp_probs_b | F32 | exists | exists (F32 GEMV) |
| output.weight | Q6_K | exists | exists |
| token_embd | Q8_0, **host** row gather (design 9.1) | ggml `to_float` | — |
| (NextN block 45 only) | Q3_K gate/up, Q4_K down | Q3_K **missing** | not needed until MTP |

The expert slab is 11.67 MB (layer 11: 15.79, layers 12 and 44: 14.09);
143.7 GB of experts, 8.29 GB of trunk, 0.52 GB head, 0.67 GB embedding.
**No new decoder is needed for the text tower.** The four formats it uses
were checked bit-exact against ggml on this file anyway (section 7a).

## 5. Placement: trunk + KV per card, ~51 GB of experts resident, the rest streamed

Layer ranges 0-14 / 15-29 / 30-44 (the head on the last card). From
`franken_decode --model <glm> --cpu --plan-only --placement ~/bench/m2/glm
--expert-gb 17 --ctx 262144` (no model byte read):

| card | layers | trunk | KV + KDA at 256k | expert room | **resident (17 GB)** | in-sample misses a token |
|---|---|---:|---:|---:|---:|---:|
| 0 | 0-14 (12 MoE, 3 DSA) | 3.03 GB | 0.86 + 0.05 GB | 20.03 GB | 1 396 experts | 409 MB |
| 1 | 15-29 (15 MoE, 4 DSA) | 2.63 GB | 1.14 + 0.05 GB | 20.15 GB | 1 457 | 353 MB |
| 2 | 30-44 (15 MoE, 4 DSA) + head | 3.15 GB | 1.14 + 0.05 GB | 19.63 GB | 1 436 | 359 MB |
| **all** | | 8.81 GB | 3.29 GB | 59.8 GB | **4 289 of 12 096, 51 GB** | **1 121 MB** |

"Expert room" is 24 GiB less trunk, positional state and 1.8 GB kept back
(1.0 HIP/allocator reserve, 0.5 prefill-chunk scratch, 0.3 miss staging
ring: two layers x 8 slabs). At 15 / 19 GB a card the plan misses 1 298 /
961 MB a token (in-sample). The ranking is DeepSeek's (greedy by histogram
count over a card's layers), so VRAM goes to the skewed layers: layers 3-6
are nearly flat (normalised entropy 0.987, §M2) and card 0 misses most.

**The histogram is the technical record (M2's corpus)**. DeepSeek showed the
hot set does not transfer to prose (§L5-DS4-STEP3b: 36 → 742 MB a token);
the same is expected here, so the GPU runs start from a mixed-corpus
histogram (`--hist-out`, as DeepSeek did) and the adaptive placement of
§L5-DS4-ADAPT (design L2) is the steady state, not an option.

**The miss path.** ~1.1 GB a token is too much for DeepSeek step 2's
host-mapped in-place reads: those measured ~16 GB/s on the scattered block
reads of a GEMV (§L5-DS4-STEP3), 70 ms a token here. The base is DeepSeek
step 3's: after the router, a side stream DMAs each missed slab
(gate|up|down) into a VRAM staging ring while the shared expert and the
resident experts run, then the missed experts run from the ring. Two
regimes, projected from §M4 (`--plan-only` prints both):

- **owner card** (the DeepSeek base): each card streams its own layers'
  misses while it runs. The cards take turns within a token, so one ~28 GB/s
  link works at a time: **~40 ms a token of stream** at 1.12 GB, overlapped
  only with the shared + resident experts of the same layer (~0.1-0.2 ms of
  a ~0.9 ms/layer stream).
- **three links** (the PCIe-aware split, design §3.3): every layer's missed
  experts are split over the three cards' links at once — 27.5 / 27.5 / 45 %,
  i.e. 17.2 + 17.2 + 28 GB/s, the lone card taking the largest share — and
  **computed where they land**: x (16 KB) goes out to the helper cards by
  P2P, their weighted partial sums (16 KB) come back and are added in slot
  order. **~18 ms a token of stream**. The lone card is `48:00.0`, HIP index
  2 in the record's bus map (the same card that holds the head); to be
  re-verified from `hipDeviceGetPCIBusId` on the first GPU run.

Everything else a token reads is the trunk (8.8 GB with the head, one card
at a time) and the resident experts (~2.8 GB of the 3.92 GB routed): ~12 GB
at the L0 GEMVs' ~500 GB/s is ~24 ms. **Projected: ~60-65 ms a token with the
owner-card stream, ~40-45 ms with the three-link split** (to be refuted by
the GPU run), against llama.cpp's and Colibri's ~5 tok/s (§M2).

## 6. What steps 1-2 built (`tools/hot-expert/franken/decode/`)

| file | what |
|---|---|
| `glm5_shapes.h` | the constants of section 1, namespace `fk::glm5`, `is_dsa(il)` |
| `glm5_model.{h,cpp}` | every tensor by the FILE's name per layer kind, placed per layer range; `attn_k_b`/`attn_v_b` as 64 per-head Mat slices; the expert table (DeepSeek's `ExpertTable`); the header check; `--plan-only` placement with the VRAM and miss projection |
| `glm5_ops.h`, `glm5_cpu.cpp` | the GLM-specific ops as an interface (KDA conv / gate / recurrence, LayerNorm, f16 stores, indexer pool / scores / top-k, absorbed MLA attention with the tail, sigmoid top-8 router, top-8 expert GEMVs, hc_mean) and their CPU implementation |
| `glm5_graph.{h,cpp}` | `Glm5Runner` — the math, written against `Backend` + `Ds4Ops` (what GLM shares with DeepSeek: the HC mix, GEMV, swiglu_clamp, moe_accum) + `Glm5Ops`, taps under glm5next.cpp's cb() names — and `glm5_main`, the CLI |
| `franken_decode.cpp` | dispatch: `general.architecture == glm5next` → `glm5_main` |
| `../oracle_dump.cpp` | the GLM tap set (chosen from the arch), `--ctx N` (the indexer scores only when n_ctx > 2 051) |
| `ds4_quant_emul.cpp` | `--glm-model`: Q8_0 / IQ3_S / IQ4_XS / Q6_K on random blocks and real GLM rows |
| `glm5_oracle.sh` | the oracle run: dump (bounded, page-cache watchdog) then compare |

Scope limits of step 2: T = 1 (the prompt fed token by token — exact, since
KDA is a recurrence and DSA attends causally); CPU only (the runner refuses
a GPU backend); the NextN block not run.

## 7. The oracles (CPU; no GPU touched; 2026-09-24)

**(a) Decoders, bit-exact.** `franken_ds4_quant_emul --glm-model <shard>`:
one-hot extraction of every weight against ggml's own `to_float`, float-bit
equality, 512 random blocks per format and 4 real rows of `token_embd`
(Q8_0), `blk.3.attn_q_b` (Q8_0), `blk.3.ffn_gate_exps` e17 (IQ3_S),
`blk.11.ffn_up_exps` e200 (IQ4_XS), `blk.3.ffn_down_exps` e17 (IQ4_XS),
`blk.12.ffn_down_exps` e5 (Q6_K), `output.weight` (Q6_K): **0 mismatches**,
row dots vs float64 ≤ 2.4e-9 relative; DeepSeek's six sets unchanged;
`emul verdict=PASS`.

**(b) Qwen3.8 and DeepSeek-V4 unchanged** (the commit's own clean worktree):
- Qwen3.8, 6 tokens, layers 0-47, `--oracle` the pre-L5 dump: `compared=1604
  missing=0 incomparable=60`, **0 taps not bit-exact**, 12 indexer_top_k
  contained (8.9 GB of the Qwen3.8 file read).
- DeepSeek-V4 layers 0-3 against the pre-change binary's dumps: 6 and 136
  tokens, **172/172 float taps cos=1 maxabs=0** both; against llama.cpp:
  min cos 0.999575 / 0.999558, routing 24/24 / 533/544, as recorded.

**(c) GLM-5.3 against llama.cpp** (`franken_oracle_dump --stop-after-layer N
--no-extra-bufts --ctx 4096` under `no_populate.so`, CPU; `franken_decode
--cpu --layers 0-N --no-head --ctx 4096`; `~/bench/franken/glm5/`):

| prompt | layers | taps | min cos | routing (exact sets) | with `--quant-act` | GLM file read |
|---|---|---|---|---|---|---|
| 6 tokens `154822 785 6722 315 9621 374` ([gMASK] "The capital of France is") | 0-3 | 125/125 (1 incomparable) | **0.999380** (ffn_swiglu_limited-1) | 5/6 | 0.999587, 6/6 | 1.6 GB |
| 6 tokens | 0-5 (3 KDA+MoE layers) | 205/205 | **0.999380** | 16/18 | | +1.4 GB |
| 136 tokens ([gMASK]<sop> + the sentence x 26 + 4): 34 pools scored, the last position closes a pool | 0-3 | 125/125 | 0.998651 (ffn_swiglu_limited-1), 0.998908 (ffn_out-1); every other tap ≥ 0.99942 | 127/136 | **0.999540**, 126/136, PASS | +0.4 GB |
| 2 200 prose tokens (MMLU-Pro text): 550 pools, **the top-512 selecting** for the last 152 positions | 0-3 | 125/125 | **0.999525** (ffn_swiglu_limited-1) | 1 981/2 200 (overlap 0.988) | 0.999735, 1 998/2 200, PASS | +2.5 GB |

The DSA taps (6 tokens): dsa_q_a_norm 0.999980, dsa_q_b 0.999985,
dsa_kv_a_norm 0.999985, dsa_q_absorbed 0.999978, kqv_out 0.999966, dsa_out
0.999966, indexer_q 0.999989, indexer_weights 0.999999, indexer_gate
0.999995; at 136 tokens `indexer_pool_score` 0.999975 over 34 pools and
`indexer_top_k` contained 1.0 (136 of 136 cells). The KDA taps: kda_qkv /
conv / q_norm / k_norm / gate ≥ 0.999985, kda_scan_out ≥ 0.999993,
kda_normed ≥ 0.999983, kda_out ≥ 0.999969. `indexer_k` is INCOMPARABLE by
the oracle's name rule (Qwen's `indexer_k` is a padded-cache column); the
same key feeds the pool scores, which are compared.

**The floor is the reference's rounding, not the port.** The lowest taps are
layers 0-1's 12 288-wide dense SwiGLU and its output, where llama.cpp's CPU
matmul quantises the activation to Q8_0 before every Q8_0 dot and this arm
does not; `--quant-act` (the Qwen3.8 arm that does the same) lifts the
136-token floor from 0.99865 to 0.99954 and makes the 6-token routing exact.
The routed-expert taps (IQ3_S / IQ4_XS, whose ggml dots take Q8_K
activations, not emulated) are the next-lowest, ~0.9996. The 9-10 of 136
positions with a different expert SET at layer 3 differ by one expert of
eight; that they are near-ties at the cut is the likely reading, not a
verified one (DeepSeek's step-1 wording, and the same status).

**(d) The sparse path, past 2 048 tokens.** 2 200 tokens of the MMLU-Pro
text DeepSeek used (`~/bench/franken/ds4/adapt/mmlu_prose.txt`, tokenised by
llama-tokenize in the ROCm image with no device, `[gMASK]<sop>` first):
at the last position 550 pools are visible and 512 are selected, so the
attention reads 2 048 pooled cells plus the tail, not the whole prefix.
`indexer_pool_score` cos **0.999934** over the 550 pools (0.999973 with
`--quant-act`), `indexer_top_k` **contained 1.0, 2 048 of 2 048 cells** —
the same 512 pools as llama.cpp — then kqv_out 0.999980, dsa_out 0.999978,
l_last-3 0.999977. The whole run: min cos 0.999525, PASS (0.999735 with
`--quant-act`). Routing at layer 3: 1 981 / 2 200 positions with the
identical set of 8, per-expert overlap 0.988 — more near-ties than
DeepSeek's top-6 of 256 (98.65 %), not individually examined. The CPU arm
took ~60 min for the 2 200 tokens (T = 1, scalar lane decoders).

**Slot order**: as DeepSeek's CLI — the per-slot taps are aligned to the
reference's slot order only when the two routers picked the same set at the
last prompt position (`slot_order` line); unequal sets are left to fail.

## 8. GPU kernels needed (step 4), and what exists

Reused as is (bit-identical paths already gated on the GPU for DeepSeek or
Qwen3.8): the L0 GEMV/GEMM for Q8_0 / Q6_K / F32 (trunk, shared expert,
dense FFN, head, per-head wk_b / wv_b slices); `k_ds4_hc_split` / `_wsum` /
`_post` / `_hc_init` (HC = 4, 4096 wide: identical shapes);
`k_ds4_swiglu_clamp`; `k_ds4_moe_accum` (n_used is a parameter); the staging
copy `k_ds4_stage` and the side-stream join; the `--hip-graph` machinery
(position-semantic ops); the L0 backend's `gated_rms_norm`, `l2_norm`,
`rms_norm_mul`, `argmax`, boundary P2P.

New or generalised (Opus tier where marked):
1. **Expert GEMVs through the table for IQ3_S, IQ4_XS, Q6_K**, top-8 of
   288: `k_ds4_moe` is templated on the format and hard-wires DeepSeek's 6
   and 256 (and the router, staging slots, miss counters the same) —
   template them on (n_used, n_expert) and add the three instantiations
   (the lane decoders exist and are bit-exact). The expert-major chunk path
   (`k_ds4_moe_sort`) likewise.
2. **KDA** (Opus): conv + SiLU + l2-norm over 24 576 channels with the
   3-row window (the GDN conv kernel is the pattern); the per-channel gate
   (elementwise); the **recurrence with a per-KEY-channel decay** — Qwen3.8's
   `gdn_step` kernel decays by one scalar a head, KDA by a 128-vector; the
   state is 4.19 MB a layer, read and written every token (285 MB a token
   over 34 layers).
3. **The pooled indexer**: LayerNorm with bias (128 wide), f16 key|gate
   store into the ring, the 4-cell softmax pool (elementwise over 128), the
   pool scan (`k_ds4_lid_scores` shape: 32 heads x 128 instead of 64 x 128,
   no FWHT, pooled keys in f16; compute-bound at 256k like DeepSeek's), the
   top-512 of up to 65 536 pools (`k_ds4_topk` with k = 512, ascending
   output) and the cell expansion with the tail.
4. **Absorbed MLA attention** (Opus): 64 heads x 512 latent against ≤ 2 051
   f16 latent rows, no sinks — `k_ds4_attn_part` / `_combine` are this
   shape (512-wide MQA with a split over key chunks) given a cell list
   instead of raw window + compressed rows; plus a batched per-head GEMV for
   wk_b / wv_b (64 small Q8_0 matrices).
5. **Router**: sigmoid, biased top-8 of 288, normalise, x2.5 (`k_ds4_router`
   is sqrt-softplus top-6 of 256 with hash layers).
6. **hc_mean** (trivial) and the head on the last card.
7. **The three-link miss path** (section 5): per layer, the missed experts
   split across cards, helper-card expert GEMVs on P2P'd x, partial sums
   back. This is new engine plumbing, not a kernel, and the measured
   difference between ~40 and ~18 ms of stream a token decides whether it
   is worth building before adaptive placement.

**Oracles owed on the GPU**: the GPU arm against this CPU arm's `--dump`
(layers 0-3 at 6 / 136 / 2 200 tokens, the last one with the top-512
selecting), then the whole model's greedy text against llama.cpp (needs a
whole-model read: 8.8 GB trunk + the routed experts — only with GLM not
serving).
