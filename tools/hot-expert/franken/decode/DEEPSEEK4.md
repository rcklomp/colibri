# DeepSeek-V4-Flash on the Franken engine — L5 steps 1-3c (2026-09-23/24)

The model, the path Qwen3.8 took (design rev 13 §9; record §L0-STEP1..3), and
what steps 1 and 2 built and proved. **Everything below that says "measured"
was measured on the CPU arm or by the compiler; nothing here is a GPU
measurement.** Step 2 (GPU kernels, placement, the gate commands) is §9;
step 3 (the speed work toward ~15 ms a token) is §10.

Model: `~/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/*.gguf` (3 shards,
85 GB, arch `deepseek4`, 1 328 tensors). Reference: `~/src/llama-glm53`
@ `39931761a`, `src/models/deepseek4.cpp`, `src/llama-kv-cache-dsv4.cpp`,
`src/llama-graph.cpp` (build_moe_ffn, build_ffn, the dsv4 inputs), and the
CPU ops in `ggml/src/ggml-cpu/ops.cpp` (dsv4_hc_*, lightning_indexer, fwht,
swiglu_clamp, rope).

## 1. The header

| key | value | used as |
|---|---|---|
| block_count | 43 (no `nextn_predict_layers`: **no MTP block in this file**) | `N_LAYER` |
| embedding_length / vocab | 4096 / 129 280 | |
| attention.head_count / head_count_kv | 64 / 1 — MQA, and K **is** V (one 512-wide head) | |
| key_length = value_length | 512; rope.dimension_count 64 on the **last** 64 dims (`rope_set_offset(448)`), rope type NORM (adjacent pairs) | |
| attention.q_lora_rank | 1024 | wq_a → norm → wq_b |
| attention.output_group_count / output_lora_rank | 8 / 1024 | grouped wo_a (8 × [4096→1024]) → wo_b [8192→4096] |
| attention.sliding_window | 128 | raw K window |
| attention.compress_ratios | `[0,0, 4,128, 4,128, …, 4,128, 4]` (46 entries; the 3 past 43 are unused) | per-layer attention variant, §2 |
| attention.compress_rope_freq_base | 160 000 | rope of every ratio≠0 layer, the indexer and every compressed row |
| rope.freq_base, yarn factor / orig ctx / beta | 10 000; 16 / 65 536 / 32, 1 | freq_scale 1/16, ext_factor 1 on ratio≠0 layers; ratio-0 layers unscaled |
| attention.indexer.head_count / key_length / top_k | 64 / 128 / 512 (blocks of 4 tokens = 2 048 tokens) | CSA layers |
| hyper_connection.count / sinkhorn_iterations / epsilon | 4 / 20 / 1e-6 | 4 residual streams, Sinkhorn-normalised 4×4 mix |
| expert_count / used / ff | 256 / 6 / 2048; expert_shared_count 1 (2048 wide) | |
| expert_gating_func | 4 = sqrt(softplus) | probs = √softplus(logit) |
| expert_weights_norm / scale | true / 1.5 | w = p / max(Σp, 6.1e-5) × 1.5 |
| hash_layer_count | 3 | layers 0-2 route by **token id** (`ffn_gate_tid2eid`, I32 [6, 129 280]) |
| swiglu_clamp_exp / _shexp | 10 on every layer | `ggml_swiglu_clamp`: g = min(g, 10), u = clamp(u, ±10), h = silu(g)·u |
| layer_norm_rms_epsilon | 1e-6 | |

`ds4_shapes.h` compiles these in; `ds4_model.cpp check_hparams()` reads every
one back off the header and refuses a file that disagrees.

## 2. The token, op by op

Embed: `token_embd` (Q5_K) row → repeated into the 4 hyper-connection streams
`H[4][4096]` (`hc_init`). Per layer (43), twice a layer, **hc_pre**:
`rms_norm(H flat 16384)` → `hc_*_fn` (F32 [16384→24]) → split: `pre[4] =
σ(m·s0+b)+ε`, `post[4] = 2σ(m·s1+b)`, `comb[4×4]` = softmax per source then 20
Sinkhorn row/column normalisations → `x = Σ_h pre[h]·H[h]`. After the block,
**hc_post**: `H'[d] = post[d]·y + Σ_s comb[d,s]·H[s]`.

Attention (`build_attention_impl`) on `cur = rms_norm(x)·attn_norm`:
`qr = norm(wq_a·cur)`; `q = rope_tail(rms_norm_per_head(wq_b·qr))` [64×512];
`kv = rope_tail(norm(wkv·cur))` [512]; the raw window keeps kv (f16) for the
last 128 positions. Then by layer:

| layers | ratio | keys attended by the token at pos p | extra state |
|---|---|---|---|
| 0, 1 | 0 (raw) | raw window (≤128) | — |
| 2, 4, …, 42 (21) | 4 (**CSA**) | raw window + the **indexer's top-512** of the (p+1)/4 compressed rows | compressor ring 8×1024 kv/score f32 (`wkv`,`wgate` [4096→1024] + `ape[p%4]`); at (p+1)%4==0 block b = the **overlapped** softmax-pool (previous block's 4 tokens through the first 512 of their rows, this block's 4 through the second 512; block 0's previous half is zero/−∞) → norm → rope at 4b → f16 row b. The **lightning indexer** does the same at 128 wide (its own ring 8×256), then a 128-point Walsh-Hadamard rotation (the lid cache's `k_rot`), f16. Query side: `iq = rope(idx_q_b·qr)` [64×128] → FWHT; `w = idx_proj·cur / √(128·64)`; score_b = Σ_h relu(iq_h·k_b)·w_h; top-512 |
| 3, 5, …, 41 (20) | 128 (**HCA**) | raw window + **every** one of the (p+1)/128 compressed rows | ring 128×512 kv/score; at (p+1)%128==0 a plain 128-token softmax-pool → norm → rope at 128b → f16 |

Attention itself: 64 query heads against the shared K=V rows, scale 1/√512,
one learned **sink** logit per head in the softmax denominator
(`attn_sinks`), output `Σ p·K`. Then **de-rope** (inverse rotation at p) on
each head's last 64 dims, **grouped wo_a** (head group g = heads 8g..8g+7,
4096 → 1024 each), **wo_b** [8192→4096].

FFN on `cur = rms_norm(x)·ffn_norm`: router `gate_inp` (BF16 [4096→256]);
probs = √softplus; ids = `tid2eid[token]` on layers 0-2, else top-6 of
probs + `exp_probs_b`; w = probs[ids] normalised × 1.5; per expert
`down(swiglu_clamp(gate·x, up·x))` weighted and summed in slot order; plus the
clamped shared expert; `ffn_out = moe + shexp`.

Head: `hc_head` (the same flat norm, `output_hc_fn` [16384→4], σ+ε, weighted
stream sum) → `output_norm` → `output` (Q4_K [4096→129 280]).

**Skipped:** MTP (absent from this file; llama.cpp's `graph_mtp` exists and
would be a separate speculative-decoding item), the reference's rollback
snapshot planes (`n_rs_seq`) — a serving concern (L0 step 4's checkpoints),
not the math.

## 3. KV at 256k

f16, as the reference's default cache type; the compressor rings are f32.

| cache | per layer | 256k window |
|---|---|---|
| raw window, all 43 layers | 128 × 512 × 2 B = 128 KB, fixed | 5.5 MB |
| CSA compressed K, 21 layers | 512 × 2 B per 4 tokens = 256 B/token | 1.41 GB |
| lid keys, 21 layers | 128 × 2 B per 4 tokens = 64 B/token | 0.35 GB |
| HCA compressed K, 20 layers | 512 × 2 B per 128 tokens = 8 B/token | 0.04 GB |
| compressor rings | CSA 64 KB + lid 16 KB, HCA 512 KB | 12 MB |
| **total** | **6 880 B a token** | **1.80 GB (0.60 GB a card)** |

Per token at 256k: every CSA layer scans 65 536 lid keys (16.8 MB a layer,
352 MB a token over 21 layers — against Qwen3.8's 805 MB, 12 × 67 MB, §M3)
and gathers ≤512 CSA rows (0.5 MB); every HCA layer reads its 2 048 rows
(2 MB, 41 MB a token). The scan is the depth-dependent term, as for Qwen3.8,
but with 64 heads per key instead of 4 it is 8 192 MACs per 256-byte key
(11.3 GMAC a token), so its kernel must be checked against compute, not only
bytes.

## 4. Tensor formats in this UD-IQ2_M file

| tensor | format (layers) | GEMV decoder on the host | GPU kernel |
|---|---|---|---|
| ffn_gate_exps, ffn_up_exps | **IQ2_XXS** (42 layers), **IQ2_S** (layer 26) | `ds4_quant.h` (new) | **missing** |
| ffn_down_exps | **IQ3_XXS** (41 layers), **MXFP4** (layers 26, 42) | `ds4_quant.h` (new) | **missing** |
| attn_q_a | **Q5_K** (42 layers), Q6_K (layer 26) | new / existing | Q5_K **missing** |
| ffn_gate_shexp, ffn_up_shexp | **Q5_K** (42 layers), Q6_K (layer 26) | new / existing | Q5_K **missing** |
| ffn_down_shexp | Q6_K (42 layers), Q8_0 (layer 26) | existing | existing (Q6_K is lm_head's kernel today) |
| attn_q_b, attn_kv, attn_output_a/b, compressor kv/gate, indexer q_b, indexer compressor kv/gate | Q8_0 | existing | existing |
| ffn_gate_inp | BF16 | existing | existing |
| hc_*_fn, indexer.proj, norms, ape, sinks, biases | F32 | existing | existing |
| output.weight | **Q4_K** | `ds4_quant.h` (new) | **missing** |
| token_embd | Q5_K, **host** row gather | ggml `to_float` | — (host) |
| ffn_gate_tid2eid (0-2) | I32, **host** row gather | — | — (host) |

The expert slab is 7.54 MB (layer 26: 9.83, layer 42: 8.78); 83.87 GB of
experts, 6.38 GB of trunk, 0.30 GB head, 0.36 GB embedding.
So the GPU side needs **six new decoders**: IQ2_XXS, IQ3_XXS, IQ2_S, MXFP4 for
the experts, Q5_K and Q4_K for the trunk and head. All six exist in lane form
in `ds4_quant.h` (the `decode_quant.h` convention: lane `tid` owns weights
8·tid..8·tid+7 of a 256-block, MXFP4 one weight a lane), bit-exact to ggml
(§7) and written to compile under hipcc (`FK_QUAL`, tables as parameters,
`clang fp contract(off)` where the reference does not fuse).

## 5. Placement: 60 GB of experts + trunk + 256k KV on three cards

Layer ranges 0-14 / 15-28 / 29-42 (the head on the last card):

| card | trunk | KV at 256k | scratch + reserve | expert room | **experts resident (plan)** | of its layers' experts |
|---|---|---|---|---|---|---|
| 0 (layers 0-14) | 2.21 GB | 0.60 GB | 0.5 + 1.0 GB | 21.5 GB | **20 GB** | 20 / 28.9 GB = 69 % |
| 1 (15-28) | 2.09 GB | 0.60 GB | 0.5 + 1.0 GB | 21.6 GB | **20 GB** | 20 / 27.6 GB = 72 % |
| 2 (29-42 + head) | 2.38 GB | 0.60 GB | 0.5 + 1.0 GB | 21.3 GB | **20 GB** | 20 / 27.3 GB = 73 % |

(24 GiB = 25.77 GB a card; "reserve" is the HIP context and allocator slack
the Qwen3.8 run left free; scratch covers a 512-token prefill chunk's MoE
intermediates, ~126 MB, and attention.) **Which** experts: the §M2 usage
histogram per layer (hot set from 32k tokens), which with 60 GB resident
leaves **60 MB a token** of misses (7 884 of 11 008 slabs resident).

The other ~23.9 GB of experts live in **host RAM, pinned** (with token_embd
and tid2eid: design §9.1's gather-table rule). **Miss path for step 2 — a
device-side pointer table, no host round trip:** per layer, a 256-entry table
of expert base addresses; a resident expert points into VRAM, a missing one
at the device-mapped view of its pinned host copy
(`hipHostGetDevicePointer`). The router's ids stay on the card and the same
expert GEMV reads either address; a miss is a zero-copy PCIe read (7.5 MB at
~20-25 GB/s ≈ 0.3-0.4 ms; 60 MB a token ≈ 2.5-3 ms spread over three cards,
projected). Colibri's F2 ring is the upgrade if zero-copy reads measure
badly; the table is also what L2's re-placement edits. **Free lookahead:**
layers 0-2 route by token id, so their misses are known when the token is
embedded and can be prefetched before layer 0 runs.

## 6. What step 1 built (`tools/hot-expert/franken/decode/`)

| file | what |
|---|---|
| `ds4_shapes.h` | the constants of §1, namespace `fk::ds4` |
| `ds4_quant.h` | lane decoders: Q4_K, Q5_K, IQ2_XXS, IQ2_S, IQ3_XXS, MXFP4 (included by `decode_quant.h`; `FkQuantType` gains values 7-12, appended) |
| `ds4_quant_emul.cpp` | `make ds4-emul [DS4_MODEL=<shard>]`: the bit-exactness check (§7a) |
| `ds4_model.{h,cpp}` | every tensor by the FILE's name (`attn_kv_a_norm`, `attn_output_a/b`, `attn_compressor_*`, `indexer.proj`, `indexer_compressor_*`, `exp_probs_b.bias`), placed per layer range; header check |
| `ds4_ops.h`, `ds4_cpu.cpp` | the DeepSeek-specific ops (hc split/Sinkhorn, weighted sum, hc_post, rope-tail with yarn, FWHT, compressor pools, indexer scores, top-k, sink attention, router, swiglu clamp, MoE accumulate) as an interface; the CPU implementation |
| `ds4_graph.{h,cpp}` | `Ds4Runner` — the math, written against `Backend` + `Ds4Ops`, taps under the reference's cb() names — and `ds4_main`, the CLI |
| `franken_decode.cpp` | **architecture dispatch**: `general.architecture == deepseek4` → `ds4_main`; everything else is the qwen4exp path, untouched |
| `../oracle_dump.cpp` | DeepSeek tap names (chosen from the model's arch), `--stop-after-layer N` (the eval callback stops the graph after `l_last-N`), `--no-extra-bufts` (no repacking reads) |
| `../no_populate.c` | LD_PRELOAD shim: libllama maps with MAP_POPULATE / WILLNEED; this keeps the mapping lazy so a dump reads only what it computes |
| `ds4_oracle.sh` | the oracle run: dump (bounded, with a page-cache watchdog) then compare |
| `ds4_gpu.inc`, `ds4_gpu_gate.sh` | step 2: the GPU ops and kernels, the GPU gate commands (§9) |

Scope limits of step 1: T = 1 (the prompt is fed token by token — correct,
since every DS4 attention path is causal per token, but no batched prefill);
CPU only (the runner refuses a GPU backend).

## 7. The oracles (CPU; no GPU touched; 2026-09-23)

**(a) Decoders, bit-exact.** `franken_ds4_quant_emul --model <shard>`: one-hot
extraction of every weight against ggml's own `to_float`, float-bit equality:
512 random blocks per format (131 072 weights; 16 384 for MXFP4) and 4 real
rows of a real tensor per format (`output.weight`, `blk.0.attn_q_a`,
`blk.3.ffn_gate_exps` e17, `blk.26.ffn_up_exps` e200, `blk.3.ffn_down_exps`
e17, `blk.42.ffn_down_exps` e5): **0 mismatches in all 12 sets**; row dots vs
float64 ≤ 3.7e-9 relative. `emul verdict=PASS`.

**(b) Qwen3.8 unchanged.** Pre-change binary (branch tip `cbf0430`, a
temporary worktree) `--chunk 1 --dump`, post-change binary `--oracle` it, 6
tokens, layers 0-47: `compared=1604 missing=0 incomparable=60 refused=0
STEP2 PASS`, **1 592 float taps cos=1.000000 maxabs=0**, 12 indexer_top_k
contained=1.0. (Read 10.2 GB of the Qwen3.8 file once.)

**(c) DeepSeek-V4, layers 0-3 against llama.cpp** (`franken_oracle_dump
--stop-after-layer 3 --no-extra-bufts` under `no_populate.so`, CPU;
`franken_decode_cpu --cpu --layers 0-3 --no-head`):

| prompt | taps | min cos (all 172 float taps) | l_last-0..3 | routing | reads |
|---|---|---|---|---|---|
| 6 tokens `0 671 6102 294 8760 344` (BOS "The capital of France is") | 173/173, 0 missing | **0.999575** (ffn_moe_weighted-2) | 0.999913 / .999906 / .999907 / .999900 | **24/24 sets exact** | 1.61 GB |
| 136 tokens (BOS + that sentence ×27): HCA block 0 compressed and attended, 34 CSA blocks through the indexer, the raw window wrapped | 173/173 | **0.999558** | 0.999921 / .999915 / .999914 / .999905 | 533/544 sets exact (layers 0-2: 100 %, layer 3: 98.65 % overlap) | +0.17 GB |

Distribution (6 / 136 tokens): cos ≥ 0.99999 on 60 / 62 taps, [0.9999,
0.99999) 77 / 65, [0.9995, 0.9999) 35 / 45, none below. The attention taps:
attn_raw 0.999973-0.999996, attn_csa_lid 0.999976-7, attn_hca 0.999965-78,
lid_score_masked 0.999989 over 34 blocks, lid_top_k contained = 1. The
lowest taps are the routed-expert GEMVs (IQ2_XXS/IQ3_XXS), where llama.cpp's
CPU path quantises the activation to Q8_K before the dot and this arm does
not — the same reference-rounding story as Qwen3.8's `--quant-act`
(README), **not yet demonstrated for these types** (no Q8_K arm here).

**Slot order.** On the 136-token prompt layer 3 picked the same six experts
as llama.cpp with two slots swapped: experts 143 and 118 score 9.6574 /
9.6534 in the reference and 9.6580 / 9.6579 here. `build_moe_ffn` sums the
slots, so the order is not part of the math; the CLI aligns the per-slot taps
to the reference's order **only when the sets are equal** and prints a
`slot_order` line; unequal sets are left to fail. The 11 layer-3 positions
with a different set (one expert of six) are not taps and were not
individually examined; that they are near-ties at the cut is the likely
reading, not a verified one.

## 8. Next steps (as of step 1; step 2's are at the end of §9)

1. **GPU kernels** (Opus tier where noted): the six decoders as expert and
   trunk GEMVs (the lane forms exist; the IQ2_XXS/IQ3_XXS grid tables go to
   `__constant__`); the DS4 ops of `ds4_ops.h` behind a `Ds4GpuOps` — sink
   MQA attention over raw-window + selected rows (Opus), the indexer scan
   (64 heads × 128, f16 keys, ReLU-weighted; M3's scan kernel is the pattern)
   plus M3's radix top-k, the compressor pools, hc split/Sinkhorn fused with
   the mix GEMV's epilogue, rope-tail/de-rope, FWHT. Gate: the GPU arm
   against this CPU arm's `--dump`, then against llama.cpp.
2. **The miss path** (§5): pinned host copy of the non-resident experts, the
   per-layer pointer table, placement from `~/bench/m2/deepseek*/`.
3. **Batched prefill** (design §9.4 for this graph): the compressor and the
   indexer are causal per token, so a chunk is T tokens of the same ops.
4. **Oracles still owed:** the head and greedy text (needs a whole-model
   read: ~6.4 GB trunk + experts — to be scheduled against the page-cache
   budget or run while GLM is not serving); a prompt past 2 048 tokens so the
   indexer's top-512 actually selects; an activation-quantisation arm to
   attribute the 0.9996 floor.

## 9. Step 2 (2026-09-24): the GPU side, built and not run

**What was built.** `ds4_gpu.inc` holds 25 new kernels and `Ds4GpuOps`, the
GPU implementation of `ds4_ops.h`. It is `#include`d at the end of
`decode_gpu.hip`'s anonymous namespace, after `GpuBackend`, so every launch
goes through the backend's own `mark()` (device selection per the HIP
multi-GPU rule, launch count, `--profile`, `--sync-debug`). `decode_gpu.hip`
itself gains three include lines, one `friend class Ds4GpuOps;` and the
factory. `ds4_graph.cpp` is now per device: one scratch set and one
`Ds4Ops` per card, layer ranges 0-14 / 15-28 / 29-42, the head on card 2, and
only H (64 KB) crossing a boundary (`boundary_recv`). The expert ids, the
hash-routing ids, the top-k output and the greedy id stay on the device. The
host uploads the embedding row and the three hash layers' 18 ids with the
token (both are gather tables held in host RAM on a GPU run), and reads back
one int.

The ops' interface gained `gemv` (the six new formats go to `k_ds4_gemv`;
Q8_0 / Q6_K / BF16 / F32 go to the L0 engine's own tuned GEMV, unchanged),
`moe_gate_up` / `moe_down` over an **`ExpertTable`** (per layer, 3 × 256
expert addresses in device memory), and `alloc_host_mapped`.

**The qwen4exp GPU path, unchanged by construction and by check.** Both the
pre-L5 tree (`cbf0430`) and this tree were compiled to device assembly
(`hipcc --cuda-device-only -S`). All **65 of 65** pre-existing kernels are
instruction-for-instruction identical (comments and per-TU branch-label
numbering normalised), and their `.amdhsa` descriptors match too, so their
VGPR, LDS and scratch are the same. The CPU gate still passes: 1604/1604, all
1592 float taps cos=1.000000 maxabs=0.

**The new kernels** (`make resources`, gfx1100, `-O3`; none spills):

| kernel | VGPR | SGPR | LDS B | waves/SIMD | what |
|---|---:|---:|---:|---:|---|
| `k_ds4_gemv<Q4_K>` | 35 | 18 | 0 | 16 | lm_head |
| `k_ds4_gemv<Q5_K>` | 52 | 18 | 0 | 16 | q_a, shared-expert gate/up |
| `k_ds4_gemv<IQ2_XXS / IQ2_S / IQ3_XXS / MXFP4>` | 40 / 38 / 38 / 15 | 18 | 0 | 16 | (trunk use: none; kept for completeness) |
| `k_ds4_moe<IQ2_XXS / IQ2_S / IQ3_XXS / MXFP4>` | 40 / 38 / 38 / 15 | 18 | 0 | 16 | routed experts through the table |
| `k_ds4_lid_scores` | 48 | 18 | 33 024 | 6 | the 64-head indexer scan |
| `k_ds4_topk` | 11 | 31 | 1 044 | 16 | radix top-512 |
| `k_ds4_attn_part` / `_combine` | 43 / 11 | 25 / 21 | 32 896 / 0 | 6 / 16 | sink MQA over raw window + selected rows |
| `k_ds4_router` | 40 | 24 | 3 096 | 16 | sqrt-softplus, bias or hash, top-6, weights |
| `k_ds4_hc_split` / `_wsum` / `_post` / `_head_pre` | 50 / 9 / 15 / 7 | ≤54 | 0 | 16 | the four-stream mix, Sinkhorn |
| `k_ds4_rope_tail` / `k_ds4_fwht` / `k_ds4_to_f16` | 25 / 6 / 5 | 18 | 0 / 512 / 0 | 16 | partial RoPE (+inverse), Hadamard, f16 store |
| `k_ds4_comp_pool` / `_swiglu_clamp` / `_moe_accum` | 13 / 11 / 9 | ≤33 | 0 | 16 | compressor pools, MoE tail |

How they are built:
- **GEMV.** One wave per row; each lane runs the `ds4_quant.h` lane decoder
  (the bit-exact ones of §7a) on every block. First cut: the block bytes are
  byte loads spread over the lanes, not a tuned layout.
- **Indexer scan.** Checked against compute as well as bandwidth: at 256k a
  CSA layer scans 65 536 keys, 16.8 MB (21 µs at 800 GB/s) but 537 M MACs
  (~27 µs at a ~20 TFMA/s dual-issue peak, ~55 µs single-issue). It is
  **compute-bound**; 21 layers ≈ 0.6-1.2 ms a token at 256k, projected. The
  first two shapes, one thread per key with 64 or 16 head sums in registers,
  took 256 / 246 VGPR and spilled; the kept shape is four lanes per key with
  a shuffle reduce, 48 VGPR, and reads the keys fully coalesced.
- **Top-k.** Runs only past 512 visible blocks (> 2 048 tokens); below that
  the selection is the identity, as in ggml. Its algorithm was checked on the
  host against a sort: 300 random arrays incl. heavy ties, 0 bad.
- **Attention.** A flash-style split over 32-key chunks held in LDS (rows
  padded to 257 u32 against bank conflicts), 16 heads a workgroup, and a
  combine that puts the sink in the denominator.
- **Numerics.** Every kernel is a transcription of its CPU op. Summation
  order differs (float against the CPU arm's double), and cos/sin are the
  device's; theta, the yarn corr dims and the mscale are computed on the
  host with the CPU arm's libm.

**Placement** (`--placement ~/bench/m2/deepseek_b60 --expert-gb 20`, printed
by `--plan-only` without reading a model byte):

| card | layers | experts resident | VRAM | pinned host | expected misses (in-sample) |
|---|---|---:|---:|---:|---:|
| 0 | 0-14 | 2 653 | 19.99 GB | 8.95 GB | 18.8 MB/token |
| 1 | 15-28 | 2 597 | 20.00 GB | 7.60 GB | 8.7 MB/token |
| 2 | 29-42 (+ head) | 2 623 | 19.99 GB | 7.34 GB | 8.3 MB/token |

The ranking is greedy by histogram count per card. Keeping a slab saves
count × slab bytes of misses for slab bytes of VRAM, so count alone is the
ratio. **The CSVs hold the whole 32 760-token counts, not the first half**
(`moe_hist.cpp write_layer_csv`); the first-half ranking M2 used for its miss
figure is not on disk. So the 35.8 MB a token above is an IN-SAMPLE
estimate, and M2's out-of-sample 60 MB (uniform 185 experts a layer) is the
honest upper reference. Card 0 carries half the misses: layers 0-2 route by
token hash and are the flattest (cov25 0.44-0.46).

VRAM a card at `--ctx 262144`, projected: trunk 2.1-2.4 GB + experts 20.0 GB
+ positional caches 0.60 GB (7 CSA layers × 84 MB + 6-7 HCA × 2 MB + raw
windows) + attention partials ≤ 9 MB + the L0 backend's scratch ≈ 22.8-23.1
GB of 25.77 GB (24 GiB).

**What a miss costs (estimate, not measured).** A miss is a kernel reading a
pinned host page over PCIe (zero-copy), on the critical path: the router
decides, then the expert GEMV reads. There is no overlap in this cut. At
10-20 GB/s effective per link for these scattered 66-98-byte block reads
(below the ~25 GB/s DMA rate: small requests), the in-sample 36 MB a token
costs **~1.8-3.6 ms a token**, and M2's 60 MB costs 3-6 ms. Two of the cards
share one upstream link (§1), so their misses serialise on it. Card 0's 18.8
MB is the largest share, and the cheapest to hide: its three hash layers'
experts are known from the token id before layer 0 runs, so a prefetch into
a VRAM ring can start at embed time (not built).

**The gate** (`ds4_gpu_gate.sh`, refuses without `DS4_GPU_OK=1`; the caller
holds the rig lock with the gateway stopped):

```
cd ~/src/colibri-m1/tools/hot-expert/franken/decode
make franken_decode_ds4 GPU_BIN=franken_decode_ds4      # leaves the served franken_decode alone
M=~/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
T="0 671 6102 294 8760 344"; P=~/bench/m2/deepseek_b60; O=~/bench/franken/ds4
# (1) CPU reference, layers 0-42 + head, 6 tokens, greedy 16 (reads ~15-40 GB of the file)
./franken_decode_cpu --model $M --tokens $T --cpu --ctx 512 --threads 8 --greedy 16 --dump $O/cpu_l042
# (2) GPU against it, first under --sync-debug, then without: every tap + greedy_ids exact
./franken_decode_ds4 --model $M --tokens $T --ctx 512 --placement $P --expert-gb 20 --greedy 16 --oracle $O/cpu_l042 --sync-debug
./franken_decode_ds4 --model $M --tokens $T --ctx 512 --placement $P --expert-gb 20 --greedy 16 --oracle $O/cpu_l042
# (3) 256k allocated: greedy 16, then 32 timed tokens (ds4_decode_ms_median=, tok_s=)
./franken_decode_ds4 --model $M --tokens $T --ctx 262144 --placement $P --expert-gb 20 --greedy 16 --time 32
```

(3) times at **short depth** (6 + 16 positions) with the 256k caches
allocated. The top-512 selection and the full-depth indexer scan need a
prompt past 2 048 tokens: the next gate for them is
`--layers 0-3 --no-head` on a ~2 100-token prompt, CPU `--dump` against GPU
`--oracle` (the CPU side then reads ≤ 8 GB).

**Not claimed:** any GPU number (correctness, VRAM, time); the miss cost
(estimated above); the first-half histogram ranking (not on disk).

**Next** (after the gate):
1. The expert GEMV's block loads: coalesced 16-byte loads per lane instead of
   byte loads. This is the kernel the resident experts' ~2 GB/token rides on.
2. The hash layers' miss prefetch.
3. Batched prefill for this graph.
4. Placement from the first-half counts once `moe_hist` writes them.

## 10. Step 3 (2026-09-24): toward ~15 ms a token (built, not run on a GPU)

**Where step 2 stood (coordinator's run of `5102cdb`, §L5-DS4-STEP2).** The
oracle pair was correct: 214 / 220 taps, all cos ≥ 0.9999, greedy identical.
Speed was **91.1 ms a token** at `--ctx 262144 --expert-gb 20`, depth 53.

**What a token has to read at best.** The whole trunk, 6.38 GB, every token,
plus the six routed experts of 43 layers, ~1.9 GB. That is 8.3 GB: 10.4 ms at
800 GB/s, or ~17 ms at the ~500 GB/s the L0 GEMVs reach. **~15 ms is
therefore the bandwidth floor, not a comfortable target.** It needs the
GEMVs near the bound AND the launch floor and the misses off the critical
path.

**3a (`c8afc6f`): profile, miss path, wide loads.**
- **`--profile` on the DeepSeek path.** It reports per-card device time in
  classes appended to the backend's profiler (printed only when non-zero, so
  a qwen4exp profile is unchanged):
  - trunk GEMVs by kernel: `ds4_trunk_l0_kernel` (Q8_0 / Q6_K / BF16 / F32)
    and `ds4_trunk_kquant` (Q5_K / Q4_K);
  - experts: `ds4_expert_gate_up` / `_down`, and `ds4_expert_host_mapped`
    (the in-place missed slots, `--miss-stage 0`);
  - `ds4_miss_wait`;
  - attention: `ds4_indexer_scan`, `ds4_topk`, `ds4_attention`,
    `ds4_rope_pool_store`;
  - `ds4_hc_mix` (plus `norm`, `elem_hc`), `ds4_moe_tail`, `router`,
    `boundary_p2p`, `argmax`.

  Per card it also prints the total, launches and host syncs a token. The
  miss bytes are counted **on the device** from the routed ids
  (`ds4_miss_mb_per_token_dev<i>`). Only the `--time` tokens are profiled.
- **The miss path, off the critical path** (`--miss-stage 1`, default).
  - After the router, a side stream copies each missed expert (gate|up|down)
    into a per-card VRAM staging ring with coalesced 16-byte loads
    (`k_ds4_stage`); a resident expert is only pointed at.
  - The shared expert now runs between the router and the routed experts
    (independent work, the same reorder on the CPU arm) and overlaps the
    copy; the routed GEMVs then join on an event (`ds4_miss_wait`).
  - Layers 0-2 route by token id, so their copies start at embed time.
  - Five slots a card: ~226-295 MB of VRAM.
- **Wide loads.** Each wave copies its row into LDS with 16-byte loads and
  runs the same decoder on the copy (bit-identical by construction;
  `--staged-loads 0` keeps the byte loads for the check). This matters most
  for a row read over PCIe, where each byte load was its own transaction.
- **Sinkhorn on 4 threads**, each owning a row or column as the single thread
  summed it.

**3b (`1fbb781`): HIP-graph replay (`--hip-graph 1`, off by default).**
- The Qwen runner's mechanism: each card's share of a token is captured once
  per position class (`--hip-graph-bucket`, 1024) and replayed; the
  embedding, the hash ids, the boundaries and the one-int download stay
  outside the graphs.
- To make one graph right for every position of its class, every
  position-dependent op got a position-semantic signature (`ds4_ops.h`) and
  reads the position from a device int the graph advances. Those are:
  raw-window slot, ring slots, ape row, compressor block end and index,
  visible blocks, top-k (identity at ≤ 512 blocks), attention extents.
- Grids are sized for the class's last position, and the kernels bound
  themselves. In a graph the compress chain and the top-k are always
  issued, and the ops decide on the device.
- `--all-ops 1` makes the eager path issue that same op sequence. **On the
  CPU arm, 136 tokens, `--all-ops 1` against the eager `--dump`: 172/172
  float taps cos=1.000000 maxabs=0, lid_top_k contained**, so the graph's
  op sequence is the eager result.
- `hc_init` became an op, because `Backend::copy` on a card is a NULL-stream
  memcpy that no capture may contain.
- `--sync-debug` forces the graph off.

**Checks (all CPU or compile-time):**
- The 65 pre-L5 GPU kernels are instruction- and descriptor-identical to
  `cbf0430`.
- The Qwen3.8 CPU gate passes: 1604/1604, all 1592 float taps bit-identical.
- DeepSeek CPU oracles unchanged: 6 tokens 173/173, min cos 0.999575, routing
  24/24; 136 tokens 173/173, min cos 0.999558, routing 533/544.
- No kernel spills. New kernels (VGPR / LDS B / waves):
  - `k_ds4_gemv` / `k_ds4_moe`: Q5_K 54 / 22 528 / 10, Q4_K 35 / 18 432 /
    14, IQ2_XXS 41 / 8 448 / 16, IQ2_S 40 / 10 496 / 16, IQ3_XXS 38 / 12 544 /
    16, MXFP4 16 / 17 408 / 14;
  - `k_ds4_stage` 8 / 0 / 16; `k_ds4_hc_split` 25 / 64 / 16;
    `k_ds4_store_f16` 5; `k_ds4_ring_put` 4; `k_ds4_add_row` 5;
    `k_ds4_hc_init` ≤ 10.

**The GPU commands** (`ds4_gpu_gate.sh step3`, `DS4_GPU_OK=1`, rig lock held,
gateway stopped; `$O/cpu_l042` is step 2's CPU dump):

```
cd ~/src/colibri-m1/tools/hot-expert/franken/decode
make franken_decode_ds4 GPU_BIN=franken_decode_ds4
DS4_GPU_OK=1 ./ds4_gpu_gate.sh step3
#  (4) --staged-loads 0 --miss-stage 0 --dump  vs  the defaults --oracle: every tap cos=1 maxabs=0
#      and the defaults vs the CPU dump (step 2's gate)
#  (5) --hip-graph 1 --oracle the eager dump: every tap cos=1 maxabs=0, greedy_ids exact
#  (6) --ctx 262144 --greedy 16 --time 32 --profile   (and --miss-stage 0): the class table a card
#  (7) --ctx 262144 --greedy 16 --time 32, --hip-graph 0/1/1/0 (A,B,B,A)
```

(4) proves the new loads and the staging ring bit-identical to the
in-place paths *of this build*. To tie it to `5102cdb` itself, add that
binary's `--dump` (the kernels' arithmetic per row is unchanged by
construction; the Sinkhorn's 4 threads each keep the single thread's order).

**What the profile should decide next** (projected, to be refuted):
- If `ds4_trunk_l0_kernel` dominates at < 500 GB/s, the lever is the L0
  GEMV itself (shared with Qwen3.8, its own gated item).
- If `prof_launches_per_token` × ~3 µs is a large share, batch the GEMVs
  that share an input (the attention block's wkv + compressor + indexer
  projections; wo_a's 8 groups as one launch). That is a summation-order
  change (the split-K count follows the batch), so it goes behind a knob.
- If `ds4_miss_wait` is still large, the staging ring should start earlier:
  a router lookahead, design §3.1.

## 11. Step 3c (2026-09-24): what the step-3 profile really said

**The coordinator's step-3 gate** (`~/bench/franken/ds4/step3.txt`):
- Bit-identity passed everywhere: (4) 2 036/2 036 taps, (5) graph against
  eager bit-identical, greedy exact, CPU oracle PASS.
- Decode: 82.2 ms profiled; (7) printed 71.7 / 69.7 / 69.5 / 71.4 ms (eager /
  graph / graph / eager, A,B,B,A), with 904-943 graph nodes a card. The graph
  is worth 3 %.

The profile's class table was **not** trustworthy. Three bugs of the DeepSeek
CLI's profile flow, none in the engine's work:

1. **The event pool overflowed.** Profiling was on from the first token, but
   only the `--time` tokens closed their interval (`prof_end_token`). The
   prompt and greedy tokens' ~700 events a card piled up past the 8 192-event
   pool, and whatever class was recorded last absorbed the rest of the
   warm-up. That was then divided over the 32 timed tokens: the ~18 ms
   "outlier" in one class on every card (card 0's indexer scan, card 1's
   trunk / rope, card 2's norm), and card totals of 128-136 ms against an 82 ms
   token. **Items 3 and 5 were this.** Fix: every token closes its interval,
   and the counts are reset before the timed tokens.
2. **Idle and upstream waits were charged to real classes.** A card's last
   interval ran to the end-of-token flush: card 0's `hc_mix` 70 ms and card
   1's 39 ms were the idle time while the later cards worked. A downstream
   card's wait for its upstream sat inside `boundary_p2p` (41 / 70 ms). Fix:
   `gap_idle` is marked at the end of each card's share (so
   `prof_busy_us = total - gap_idle` is its own work), and
   `ds4_upstream_wait` is its own class before the boundary copy.
3. **Launches and host syncs were cumulative.** The backend counts per token
   from `timer_start()`, which only the qwen4exp runner called: the 31-33 k
   launches and 1 330-1 400 syncs "a token" were totals since process start.
   **Item 2 was this; the mix is one kernel a call with no host sync.**
   Fix: `timer_start()` on every card at each token start. The graph node
   counts give the real launch count: ~920 a card, ~2 760 a token.

**What was real** (the in-place `--miss-stage 0` run, minus the pool artifact):
- **Trunk:** `ds4_trunk_l0_kernel` ~10.5 ms a card for ~1.9 GB (~180 GB/s;
  `attn_q_b`'s 1 088-byte Q8_0 rows are the short-run case the L0 GEMV is
  known to be slow on, README step 1). `ds4_trunk_kquant` 1.1-1.9 ms.
- **Experts:** resident gate/up + down ~2.2-3.2 ms a card.
- **Missed experts over PCIe:** 18.5 / 15.9 / 12.2 ms for 298 / 255 / 189 MB,
  i.e. ~16 GB/s — **~46 ms a token, the largest term**. In the staged run
  `ds4_miss_wait` was only 0.4 ms, but the copy's ~700 workgroups filled the
  CUs, so the shared expert running beside it absorbed the time, and the
  token barely moved (82 vs 84 ms).

**Item 1, the misses: the ranking is right; the histogram's corpus is wrong
for this text.** `--hit-report` prints, per layer, the resident count, the
hit fraction measured on the run's own routing, and the M2 coverage of the
same resident set. On the CPU arm, same resident sets (`--devices 1 --layers
0-4 --expert-gb 5`):

| text | layer 3 (116 resident) | layer 4 (89 resident) | layers 0-2 |
|---|---|---|---|
| M2's corpus (first 256 tokens of the record) | hit **0.827** vs M2 0.836 | **0.803** vs 0.840 | 0.748-0.770 vs 0.761-0.774 |
| the gate's text (BOS "The capital of France is" + its 16 greedy tokens) | hit **0.606** vs 0.836 | **0.364** vs 0.840 (random: 0.348) | 0.780-0.811 |

On M2's own corpus measured and predicted agree within 1-4 %, so the ids map
correctly and the hottest experts are kept. The same llama.cpp routing
(oracle136, layer 3) puts the chosen experts at mean rank 128-139 of 256 in
M2's counts: uniform. **The hot set of the rig's technical record (markdown
tables, numbers, code) does not transfer to English prose.** The 36 MB
in-sample estimate and M2's 60 MB were both measured in that domain; on the
gate's text 741 MB a token is what the placement actually costs. The first
cards' layers 0-2 are fully resident because their flat token-hash
histograms outrank the skewed layers' cold tails, which is also why card 0
misses most.

**Fix, in the placement input, not the engine:** a histogram of the text the
engine will serve. `--hist-out DIR` writes one in M2's format from any run's
own routing (the routing does not depend on the placement, so any placement
serves the run). Commands in the next step's gate:
`--tokens <a few thousand ids of chat / prose / code> --hist-out
~/bench/m2/ds4_chat`, then `--placement ~/bench/m2/ds4_chat`. Or
`run_moe_hist.sh` with a representative `PROMPT_FILE` (batched prefill,
faster; out-of-sample like M2).

**Engine changes (`79b94a1`), all bit-identical per slot:**
- **Resident first:** a staged layer's resident experts run straight from the
  table while the side stream copies the misses; then the join; then the
  missed slots from their staged copies.
- **Bounded copy:** the staging copy is grid-stride over a bounded grid
  (`--stage-wgs`, 64 workgroups a layer) instead of ~700, leaving the card to
  the main stream.
- **Profile fixes** as above.
- **CPU checks:** DeepSeek oracles unchanged (173/173 twice, routing 24/24
  and 533/544); Qwen3.8 gate 1604/1604 bit-identical; 65/65 pre-L5 kernels
  ISA-identical; `k_ds4_stage` 8 VGPR.

**Rerun:** `DS4_GPU_OK=1 ./ds4_gpu_gate.sh step3` (it now also runs (8),
`--hit-report --hist-out` on the gate's text). **Projected** (to be refuted):
- with a matching histogram, the misses drop toward M2's 60 MB (~4 ms);
- the token then sits at ~30-45 ms. The profiled per-card numbers
  (themselves inflated by the profile's event syncs) leave the L0 trunk GEMV
  at ~180 GB/s, ~31 ms over three cards, as the largest term — **that GEMV,
  not DeepSeek code, is the road to ~15 ms**.

## 12. Step 4 (2026-09-24): adaptive placement, design L2 (built, not run on a GPU)

**Why.** Record §L5-DS4-STEP3b: the hot-expert set depends on the kind of
text. The technical record's histogram misses 742 MB a token on prose; a
mixed-corpus histogram 276 MB. No fixed histogram fits every conversation, so
the engine now learns the hot set from its own routing while it runs and
swaps experts in the background. `--adapt 1` (off by default; needs
`--placement` for the starting set). The qwen4exp path is untouched.

**What was built** (`ds4_adapt.{h,cpp}`, plus hooks in `ds4_ops.h`,
`ds4_gpu.inc`, `ds4_cpu.cpp`, `ds4_model.cpp`, `ds4_graph.cpp`):

1. **Counters in the router.** `k_ds4_router` gets the layer's counter block
   (`ADAPT_STRIDE` = 260 uint32: per expert how often it was chosen, the
   choices that missed, all choices). Thread 0 of the single workgroup
   increments them after it has written the outputs. Nothing the router
   computes reads them. No host sync; the counters sit in the graph like any
   other kernel argument.
2. **The average, on the host.** Every `--adapt-every` tokens (default 16) a
   token boundary queues a device-to-host copy of each card's counters on its
   main stream, then an event. A later boundary polls the event
   (`hipEventQuery`, never a wait) and folds the delta into a per-expert
   decayed average: `ema = ema * 0.5^(tokens / halflife) + delta`, with
   `--adapt-halflife` defaulting to 2 048 tokens.
3. **Re-placement, per card and slab size.** A slot only takes an expert of
   its own byte layout: layers 26 and 42 carry IQ2_S / MXFP4 and have other
   sizes. Within a class, the layers of a card share their slots, so the
   resident count can move between layers. The hottest non-resident experts
   are paired with the coldest resident ones while
   `ema_in > ema_out * (1 + hyst) + margin` (`--adapt-hyst 0.25`,
   `--adapt-margin 1.0`, against thrashing). Pairs are taken by gain until the
   round's budget is spent: `--adapt-mb-per-token` (default 64 MB, per card)
   × the tokens since the last round. At 50 ms a token that is 1.3 GB/s a
   card, against M4's 28 GB/s for one card and 36 GB/s for the shared pair.
   At most one batch per card is in flight.
4. **The swap:**
   - EVICT: a table-edit kernel on the main stream. The outgoing expert's
     entry now points at its host-mirror copy, with miss bytes = slab.
   - COPY: `hipMemcpyAsync` host to VRAM of the incoming expert's gate, up
     and down into the freed slot. It runs on a stream created at the
     device's lowest priority and is fenced behind an event recorded on the
     main stream after the evictions.
   - INSTALL: once the copy event reports complete at a later boundary, the
     main stream waits on it (free by then) and a second table edit points
     the incoming expert at its slot, with miss = 0.
5. **The host mirror.** With `--adapt 1` every expert, resident or not, has a
   pinned device-mapped host copy (all 256 a layer, slab e at offset
   e × slab). An evicted expert is served from it at once, and it is the
   source of every swap copy. **Cost: ~84 GB pinned host RAM, against ~24 GB
   without adaptation**, so a GPU run with `--adapt 1` needs the gateway
   stopped, as every DeepSeek GPU run already does.

**The ordering: why no kernel reads a half-copied slot, and why every tap
stays bit-identical.**

- *Readers of the table and of a slot:*
  - on the card's main stream: `k_ds4_router` (miss bytes), `k_ds4_moe` (the
    table entries and the bytes behind them), `k_ds4_miss_count`;
  - on the staging side stream: `k_ds4_stage`, which reads the table and the
    miss bytes and publishes a resident expert's slot address for the
    `k_ds4_moe` launches of the same layer and token.
- *Every side-stream reader is joined within its own token.* Each
  `moe_stage(slot)` is followed, in the same token, by that layer's
  `moe_gate_up(slot)`, and that makes the main stream wait on `ev_done[slot]`
  before the layer's expert GEMVs. The token-id layers staged at embed time
  are joined by their own layers in the same token. So when the main stream
  reaches the end of token t's work, every reader of token t on either
  stream has finished.
- *Every table edit sits at a token boundary on the main stream.*
  `Adapter::tick(pos)` runs at the top of `Ds4Runner::step`, before the
  embedding upload, the hash-layer staging and every card's segment (eager
  or a graph launch, both on the same stream). An EVICT edit therefore
  follows every kernel of tokens ≤ t in stream order, and so, by the join
  above, every side-stream reader too. It precedes every kernel of tokens
  ≥ t+1, and every side-stream stage kernel of those tokens, because they
  fork from events recorded on the main stream after it. **No kernel runs
  while an entry changes, and no kernel sees a mixed pair of address and
  miss bytes.**
- *The copy into slot S starts only after S is unreferenced.*
  `adapt_copy_begin` records `ev_fence` on the main stream after the EVICT
  edits, and the copy stream waits on it. Before the fence, the last entry
  pointing at S (the outgoing expert's) has been rewritten, and every kernel
  that could have read S through the old entry or a staged pointer has
  finished. Between EVICT and INSTALL no entry names S. The staging pointers
  `sp` are rewritten by every stage kernel before its join, so no stale `sp`
  entry is ever read. **Nothing reads S while it is written.**
- *INSTALL comes only after the copy has landed, for the device too.* The
  host sees `ev_copied` complete. It then queues `hipStreamWaitEvent(main,
  ev_copied)` and only after that the INSTALL edit. The edit and every later
  reader depend on the copy through a device-side edge, not just through the
  host's observation. Only one batch is in flight per card, and a new round
  is planned only after the previous one was installed, so a slot is never
  the target of two copies.
- *The mirror never changes after load*, so a host-mapped read (the in-place
  path, or the staging copy) sees the same bytes at any time.
- *Bit-identity:* an expert's bytes are identical in its VRAM slot, its
  mirror copy and a staged copy. `k_ds4_moe` runs the same decoder with the
  same `staged` flag, which depends on the row size only, over the same
  bytes. The only thing a swap changes is which launch a slot lands in
  (resident first, then staged), and step 3c already made that bit-identical.
  So the routing and every tap of `--adapt 1` equal `--adapt 0`'s, whatever
  swaps happened.

**Instrumentation.**
- Per card, per round: `adapt dev= pos= tokens= hit= miss_mb_per_token=
  swaps= swap_mb= resident= inflight=`. The hit fraction and miss bytes are
  counted by the router against the table as it stood; the swaps are those
  installed since the last line.
- All cards together: `adapt_all pos= ...`.
- At the end: `adapt_total`.
- `--adapt-verify 1` checks, after the run, that each card's table equals the
  host shadow, and it compares every resident expert's VRAM slot byte for
  byte with its mirror copy on the card (`k_ds4_compare`). It then prints
  `adapt_verify ... PASS|FAIL`, and a FAIL fails the run.
- `--tokens-file F [--max-tokens N]` reads a prompt of ids from a file.

**New kernels** (`make resources`, gfx1100, no spills):
- `k_ds4_table_edit`: 15 VGPR, 0 LDS, 16 waves/SIMD, one wave a launch.
- `k_ds4_compare`: 16 VGPR, 0 LDS, 16 waves/SIMD, `--adapt-verify` only.
- `k_ds4_router` is still 40 VGPR / 3 096 B LDS.

**Checks (CPU or compile-time; no GPU touched):**
- **The 65 pre-L5 GPU kernels are ISA- and descriptor-identical to
  `cbf0430`** (65/65).
- DeepSeek CPU oracle against llama.cpp, 6 tokens, layers 0-3: 173/173 taps,
  min cos 0.999575, routing 24/24 (unchanged).
- Qwen3.8 CPU gate: 1604 compared, 0 not bit-exact.
- **The adaptation itself, on the CPU arm with the real routing**
  (`~/bench/franken/ds4/adapt/cpu_check.sh`). On the CPU arm, `--adapt 1`
  really copies the resident set into a separate block, so a wrong slot
  address reads other bytes. Setup: layers 0-3, 48 prose tokens, a 0.5 GB
  "card" (66 experts), `--adapt-every 2 --adapt-mb-per-token 64`.
  - `--adapt 0 --dump` against `--adapt 1 --oracle`: **173/173 taps
    compared, every float tap cos=1 maxabs=0**.
  - 74 swaps (0.56 GB), `adapt_verify ... bad_entries=0 bad_bytes=0 PASS`.
  - Hit fraction 0.082 without adaptation, 0.168 with.

**The GPU gate** (`DS4_GPU_OK=1`, rig lock held, gateway stopped; the docker
form of `~/bench/ds4s3_chain.sh`):

```
cd ~/src/colibri-m1/tools/hot-expert/franken/decode
make franken_decode_ds4 GPU_BIN=franken_decode_ds4         # outside docker: the target calls docker itself
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --ipc=host -e DS4_GPU_OK=1 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
  -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/src/colibri-m1/tools/hot-expert/franken/decode \
  rocm/dev-ubuntu-24.04:7.14.0-full bash ./ds4_gpu_gate.sh adapt 2>&1 | tee ~/bench/franken/ds4/step4.txt
```

- **(9) the oracle pair.** The gate's 6-token text, greedy 16, `--adapt 0
  --dump` against `--adapt 1 --adapt-every 1 --adapt-mb-per-token 512
  --adapt-verify 1 --oracle`. Expected: 2 036/2 036 taps compared,
  `not_bitexact=0`, greedy identical, verify PASS, swaps > 0.
  - (9b) the same with `--hip-graph 1`: swaps between graph replays.
  - (9c) 64 prose tokens, where the hot set moves most.
- **(10) the trajectory.** From the technical-record placement
  (`~/bench/m2/deepseek_b60`, 20 GB a card), 8 192 prose tokens:
  - the prompt is BOS plus the first 8 191 tokens of the MMLU-Pro half of
    `~/bench/m2/mixed_corpus.txt`, in `~/bench/franken/ds4/adapt/mmlu_8k_ids.txt`,
    tokenized by llama-tokenize in the ROCm image with no device;
  - then 256 greedy tokens and 32 timed decode tokens, `--ctx 262144
    --hip-graph 1`, `--adapt-every 64`;
  - (10a) the control, `--adapt-mb-per-token 0`: the counters run, nothing
    moves;
  - (10b) adaptation at 64 MB a token a card, half-life 2 048.
  - The `adapt_all` lines are the miss-MB-per-token trajectory. **Expected,
    to be refuted:** (10a) stays near 740, and (10b) falls toward the
    mixed-corpus 276 MB or below, within a few hundred tokens. Swaps needed
    are on the order of 1 000 experts (7.5 GB) a card, i.e. ~120 tokens at
    64 MB a token. Decode ms then drops toward step 3b's 50 ms or below.

**Not claimed:** any GPU number. That includes whether the swap copy slows
the token while it runs: the copy stream is low priority and SDMA-driven,
and the copies share PCIe with the staging ring's misses.
