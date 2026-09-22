# `franken_decode` — L0 step 2: one card, layers 0–15, a decode loop with a per-layer oracle

Deliverable B of `tools/hot-expert/L0-STEP2-BRIEF-2026-09-22.md`
(design rev 8 §9.5 step 2). Built 2026-09-22 on branch `m5-harness`.

The math is ported op for op from `~/src/llama-glm53/src/models/qwen4exp.cpp`
@ `39931761a`, plus `src/models/delta-net-base.cpp` and
`ggml/src/ggml-cpu/ops.cpp` for the two ops whose graph form and fused form
differ (`ggml_gated_delta_net`, `ggml_rope_multi`).

## The two binaries

| binary | what it is |
|---|---|
| `franken_decode_cpu` | the same graph on the host, in plain C++. Links **no HIP at all** (`make ldd-check` proves it), so it cannot touch a card even by accident. |
| `franken_decode` | the same graph on device 0 through the HIP kernels. Uploads ~22 GB; **run it only under the rig lock with the gateway stopped.** |

```
franken_decode[_cpu] --model <any shard.gguf> --tokens <id> [<id> ...]
                     [--layers 0-15] [--oracle DIR] [--ctx N] [--threads N]
                     [--time N] [--min-cos X] [--quant-act] [--verbose]
                     [--dump DIR] [--jitter X]
```

`make` builds both and runs nothing. `make resources` prints the per-kernel
VGPR/SGPR/LDS/spill report (compile-only, no GPU).

## Why `--cpu` is not optional

The agent that wrote this could not run a GPU. `decode_graph.cpp` holds the
math and never touches a float: it calls `Backend` (`decode_backend.h`), which
`decode_cpu.cpp` and `decode_gpu.hip` implement. Both are checked against the
**same** oracle, so a disagreement localises itself — `--cpu` failing is a
port bug in the graph, `--cpu` passing and the GPU run failing is a kernel bug.

## What the file actually contains (and three ways it bites)

Read off the GGUF header, not assumed:

- **A UD quant is not uniform.** `ffn_{gate,up}_exps` are IQ3_S on every layer
  of 0–15 **except layer 2**, which is IQ4_XS; `ffn_down_exps` is IQ4_NL
  except on **layers 2 and 4**, which are Q8_0. Four expert kernels, not two.
- **The indexer has ONE key head.** `blk.N.indexer.k_proj` is `[2560, 128]` and
  `q_proj` is `[2560, 512]` — four *query* heads against one key. The
  reference rectifies each query head's dot **before** summing
  (`qwen4exp.cpp:576-591`). `m3_attn.hip`'s scan kernel assumed four key heads
  and no relu, so it is a cost model for this pass, not a port of it — and the
  real pass reads a **quarter** of the pooled-key bytes §M3 charged the depth
  term.
- **`m3_attn.hip`'s GDN step is not the model's.** It predicts from the
  *undecayed* state; `ops.cpp:11012` decays first. It is also transposed:
  ggml stores the state with the KEY axis contiguous (`s_out[j*S_v + i]`).

## Reading the oracle dump

Deliverable A writes `<dir>/<key>.f32` with `<dir>/index.txt` lines
`<key> <il> <ne0> <ne1> <ne2> <type>`. The payload is the last slice along
the slowest **non-unit** axis, which is *usually* the token axis and three
times is not:

- `hc_norm-3` is the **attention** module's value; `hc_norm-3.2` is the FFN
  module's (`build_hc_mix` runs twice a layer).
- `hc_combine-3` is the **post-attention** residual; `l_last-3` is the
  post-FFN one — `ggml_set_name` overwrites, so the second `hc_combine` never
  reaches the dump under that name. (Same reason `kqv_out` and `v_conv` are
  absent: the dump itself reported them "NOT FOUND in this build's graph".)
- `state_predelta-0` has ne2 = 48 **heads**, so the file is head 47's
  128×128 state, not a token.

Two points cannot be compared at all at a different cache depth, and the
comparator says so instead of failing or skipping silently:

- `indexer_k_pooled` / `indexer_k` — the file is the last *block's* column,
  and the last block is llama.cpp's **padded** `n_kv/ratio` (block 63 for a
  6-token prompt in a 256-cell cache). The selection these keys drive **is**
  checked: `indexer_score` compares every block this engine has.
- `state_predelta` — llama.cpp evaluated the prompt as **one ubatch**, so the
  state it read is the pre-batch (zero) state, while a decode loop reads the
  state after five tokens. The recurrence is checked by `attn_output`, which
  is its output for the same token.

## `--quant-act`: the arm that settles the numerics

llama.cpp's CPU `mul_mat` does **not** dot f32 activations against quantised
weights: it converts the activation to the weight type's `vec_dot_type` first
(Q8_0 for a Q8_0 weight, BF16 for a BF16 one). `--quant-act` makes that
rounding ours. It is a diagnostic arm, never the engine — this engine is the
more accurate side and stays that way.

It is the reason the residual gap is *explained* rather than *tolerated*:
under it, layer 0's whole body matches llama.cpp **exactly**.

## The routing coin-flip

`build_moe_ffn` selects 10 of 512 experts by a softmax top-k, which is
**discontinuous** in its input: at layer 1 the tenth expert's probability is
0.0606 against the eleventh's 0.0600. Measured directly: the plain and
`--quant-act` arms, which differ only by activation rounding, pick the *same*
ten experts at layer 0 and a *different* tenth at layer 1.

It is **not** ulp-chaotic, which is worth stating because the obvious guess
is wrong and the jitter arm refuted it: a 6e-7 perturbation of every block
input flips nothing at all, at any of the 16 layers. It takes a perturbation
of about 4e-4 — the scale of the reference's own activation quantisation, not
of fp32 rounding — before the first flip appears (layer 7).

So a cosine bar on `ffn_moe_out` tests the routing, not the arithmetic — the
same lesson as §F7's jitter arm (CLAUDE.md). `--verbose` prints the chosen
ids and weights so a flip is visible instead of showing up as an unexplained
divergence.

## What is NOT here

No `lm_head` (Q6_K, 521 MB), no final hyper-connection mixer, no sampling, no
gateway, no prefix checkpoints, no three-card boundaries. Those are step 3
and step 4.

## What `--cpu` measured, 2026-09-22

Ids `[248044, 785, 10945, 315, 1495, 374]` (the dump's own), six tokens fed
one at a time, `--ctx 32`, against `~/bench/franken/oracle`. Logs in
`~/bench/franken/cpu_l0*.log`, `jitter*.log`.

**Layer 0 is EXACT.** With `--quant-act` every tap of layer 0 up to and
including the Gated DeltaNet output reads `cos=1.000000 maxabs=0`:
`hc_norm-0`, `hc_gate-0`, `hc_mixed-0`, `hc_inject-0`,
`linear_attn_qkv_mixed-0`, `z-0`, `alpha-0`, `beta-0`, `a_softplus-0`,
`beta_sigmoid-0`, `conv_output_silu-0`, `q_conv-0`, `k_conv-0`,
`attn_output-0`, `shared_expert_gate_sigmoid-0`. The rest of the layer is at
the floor: `linear_attn_out-0` 0.999996, `ffn_shexp-0` 0.999995,
`ffn_moe_out-0` 0.999965, `hc_combine-0` 0.999995, `l_last-0` 0.999993.
Without the arm (the engine's real numerics, f32 activations) the same points
are 0.99988–0.99999 — that difference is llama.cpp's activation quantisation
and **this engine is the more accurate side**.

**Layer 1 (the PLE layer)** `ple_embd` 1.000000, `ple_gated_value-1`
0.999998, `hc_*-1` ≥ 0.99997, `attn_output-1` 0.999995, `l_last-1` 0.999657.

**Layer 3 (QSA)** `indexer_q-3` 0.999905, `indexer_score-3` 0.999999,
`indexer_top_k-3` containment 1.000000 (6 of 6 cells, against a 256-cell
reference selection), `attn_pregate-3` 0.999744, `attn_gated-3` 0.999570,
`attn_output-3` 0.998914, `l_last-3` 0.999810. Its input is already
`hc_mixed-3.2` = 0.999438, so these are upper bounds on layer 3's own error.

**Depth: the MoE router, not the arithmetic.** `l_last` falls smoothly
0.999959 → 0.989552 over layers 0→15 while `ffn_moe_out` collapses from
0.9999 (layers 0–6) to 0.90–0.97 (layers 7–15). `--verbose` shows why: the
plain and `--quant-act` arms, differing only by activation rounding, pick the
*same* ten experts at layer 0 and a *different* tenth at layer 1 — the
contested pair's router probabilities are 0.0606 and 0.0600.

**The jitter arm sizes it** (CLAUDE.md §F7 discipline; `--dump` makes one run
the oracle of another, so this needs no llama.cpp):

| perturbation of every block input | first routing flip | `l_last-15` |
|---|---|---|
| 6e-7 (fp32 rounding) | none — `ffn_moe_out` 1.000000 at every layer | **1.000000** |
| 4e-4 | layer 7 | 0.999287 |
| 4e-3 (q8_0 activation scale) | layer 7 | 0.999271 |
| llama.cpp, plain arm | ~layer 7 | **0.989552** |

So the architecture is **not** ulp-chaotic, and one routing flip costs
`l_last` about 7e-4. The gap to llama.cpp at layer 15 is an order of
magnitude larger than a single-point perturbation explains, because its
activation quantisation happens at *every* matmul rather than once per block.

**STEP2 is FAIL at the brief's bar, and the bar is not reachable as written**:
`cos ≥ 0.999` on `ffn_moe_out-15` asks a top-10-of-512 selection to agree with
a reference whose own selection is decided by its activation quantisation.
The clean fix is one line in deliverable A: **dump the selected expert ids
per layer** (`ffn_moe_topk_ids`). With those, routing can be compared as a
set and, if needed, forced identical — and the arithmetic bar becomes
meaningful at every depth instead of only at layers 0–6.
