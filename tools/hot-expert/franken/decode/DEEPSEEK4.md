# DeepSeek-V4-Flash on the Franken engine — L5 steps 1-2 (2026-09-23/24)

The model, the path Qwen3.8 took (design rev 13 §9; record §L0-STEP1..3), and
what steps 1 and 2 built and proved. **Everything below that says "measured"
was measured on the CPU arm or by the compiler; nothing here is a GPU
measurement.** Step 2 (GPU kernels, placement, the gate commands) is §9.

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
