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
                     [--dump DIR] [--jitter X] [--profile]
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

**Layer 0 matches to the printed precision.** With `--quant-act` every tap of
layer 0 up to and including the Gated DeltaNet output reads `cos=1.000000`:
`hc_norm-0`, `hc_gate-0`, `hc_mixed-0`, `hc_inject-0`,
`linear_attn_qkv_mixed-0`, `z-0`, `alpha-0`, `beta-0`, `a_softplus-0`,
`beta_sigmoid-0`, `conv_output_silu-0`, `q_conv-0`, `k_conv-0`,
`attn_output-0`, `shared_expert_gate_sigmoid-0`. Only `hc_norm-0` is
bit-identical (`maxabs=0`) — it is the one tap with no matmul in it; the
others carry a `maxabs` of 1e-5 to 1e-2 against vectors whose elements are
O(1)–O(10), i.e. the last-ulp disagreement of two f32 reductions.
The rest of the layer is at the floor: `linear_attn_out-0` 0.999996,
`ffn_shexp-0` 0.999995, `ffn_moe_out-0` 0.999965, `hc_combine-0` 0.999995,
`l_last-0` 0.999993.
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

## The 12.9 ms token, and what was done about it (2026-09-22, second cut)

The first GPU run measured `layers0_15_ms_median=12.9059` (n=32) — about 5x
the bytes (~1.8 GB of weights for these 16 layers is ~2.2 ms at the 800 GB/s
of §M5). Three things were wrong, and `--profile` now measures which of them
mattered.

**1. Host-blocking calls inside the layer body.** Three are gone:

- the QSA per-block bias was built on the host and uploaded **once per QSA
  layer** — a blocking `hipMemcpy` in the middle of the loop. It is a closed
  form of the position, so `k_qsa_expand` computes it.
- the PLE row upload happened inside `layer_ple`. It is now done by `step()`
  before the body, with everything else the token needs.
- `--verbose` downloaded the routed ids **per layer**, which is 16 device
  syncs a token — and `--verbose` was on in the run that measured 12.9 ms.
  The ids are now logged device-to-device and read back after the body.

Per-token uploads also go through a pinned staging ring and `hipMemcpyAsync`,
so they are stream-ordered rather than blocking. `prof_host_syncs_per_token`
must read **0**; anything else is a bug.

**2. Launch geometry: most of the trunk could not fill the card.** One wave
per output row is right for the two wide projections and starves a 96-CU /
192-SIMD card on everything else:

| tensor | rows | K | bytes | old WGs | waves/SIMD | new WGs |
|---|---:|---:|---:|---:|---:|---:|
| `attn_qkv` | 10240 | 2560 | 27.9M | 10240 | 53 | 1280 × 8 waves |
| `ssm_out` | 2560 | 6144 | 16.7M | 2560 | 13 | 2240 |
| `hc_*_down` (×4/layer) | 320 | 10240 | 3.5M | **320** | **1.7** | 2080 |
| `ffn_gate_inp` | 512 | 2560 | 5.2M | **512** | **2.7** | 2048 |
| `ssm_alpha`/`beta` (×2) | 48 | 2560 | 0.5M | **48** | **0.25** | 480 |
| `hc_*_inject` (×4/layer) | 4 | 10240 | 0.16M | **4** | **0.02** | 320 |
| `ffn_gate_inp_shexp` | 1 | 2560 | 0.01M | **1** | **0.005** | 80 |

A workgroup is now 256 threads = 8 waves, one row each, and K is split across
`nsplit` workgroups chosen so `row_groups × nsplit` reaches 2048 — but only
when `row_groups` is under 512, so the two already-full GEMVs are untouched.
Split partials are summed by `k_reduce_splits` in a **fixed order**, so the
result stays reproducible (an `atomicAdd` would not, and the oracle depends
on it). The Q8_0 inner loop keeps decode_quant.h's lane mapping — a wave
reads one block's 32 contiguous `qs` bytes, the widest request ggml's
34-byte AoS stride allows without an unaligned dword load — but unrolls four
blocks instead of two, so four requests are in flight per wave.

The reductions (`rms_norm`, `l2_norm`, `gated_rms_norm`, `ple_gate`) have a
grid fixed by the model (HC=4 groups for the hc modules), so the only lever
is width: the block size is now chosen by the host from `ne0` up to 1024,
with dynamic LDS.

**3. Launch count.** A dependent chain of trivial kernels pays a kernel
latency per link whatever its size, and §M5's "240 launches a token are free"
was measured on five substantial kernels a layer, not on the mostly-trivial
ones a layer here (`prof_launches_per_token` reports the exact figure; it was
never counted before this flag existed). Two fusions that
keep every oracle tap are in: `scale`+`silu` in `build_hc_mix` (32 launches a
token) and the whole GDN gate chain — `sigmoid(beta)`, `softplus(alpha+dt)`,
`*ssm_a`, `exp` — as one launch over 48 elements instead of five (48 a
token). `prof_launches_per_token` reports the rest; if `prof_elem_hc_us` is
still large against its bytes, the remaining elementwise chains
(`sigmoid`+`hc_collapse`, the shared-expert gate and add, the QSA output
gate) are the next candidates and are the same pattern.

**None of this has been run.** The CPU path is unchanged and was re-checked
against its own pre-change dump: 538 of 538 points `cos=1.000000`, STEP2
PASS — so the GPU-side rework cannot have moved the math, and
`--dump`/`--oracle` will say whether it moved the kernels.

## Step 2b (2026-09-22): the launch count, and per-projection bandwidth

The profiled run (`be26445`, box idle) gave `layers0_15_ms_median=11.7251`,
`prof_host_syncs_per_token=0` — the sync work of the first cut was right —
and `prof_launches_per_token=931`, i.e. **58 launches a layer**, with
`trunk_gemv 5890 µs` and `elem_hc 2004 µs` of an 11.7 ms token.

### Why ≤ 15 launches a layer is not reachable, and what is

Seventeen of those 58 were projections, one kernel each however small:
`hc_down`×2, `hc_inject`×2, `hc_up`×2, `ssm_qkv`, `ssm_gate`, `ssm_beta`,
`ssm_alpha`, `ssm_out`, `ffn_gate_inp`, `sh_up`, `sh_gate`, `sh_down`,
`sh_gate_inp`, expert gate_up, expert down. That is already above 15 before a
single elementwise op. The only thing that collapses them is that most
**share an activation vector**, so they can share a kernel:

| batch | matrices | rows | launches |
|---|---|---:|---:|
| `xn` → hc | `hc_down` + `hc_inject` | 324 | 2 → 1 (+1 reduce) |
| `mixed` → GDN | `ssm_qkv` + `ssm_gate` + `ssm_beta` + `ssm_alpha` | 16480 | 4 → 1 |
| `mixed` → QSA | `wq` + `wk` + `wv` + `idx_q` + `idx_k` | 13952 | 5 → 1 |
| `mixed` → FFN | `ffn_gate_inp` + `sh_up` + `sh_gate` + `sh_gate_inp` | 1793 | 4 → 1 |
| `ple_emb` → PLE | `ple_key` + `ple_value` | 12800 | 2 → 1 |

`k_gemv_batch` runs one kernel over the concatenated row space, with a
per-matrix format (the QSA batch mixes Q8_0 and BF16), output pointer and
epilogue. **Predicted** (not measured — `prof_launches_per_token` is the
measurement): **23 a GDN layer, 26 a QSA layer, ~389 a token**, against 931.

The rest of the reduction is fusions that all keep every oracle tap:

- the scale+silu and the sigmoid became GEMV **epilogues** (`GE_SCALE_SILU`,
  `GE_SIGMOID`), riding on the split reduce a 320-row matrix needs anyway;
- `ssm_qkv` writes **straight into the conv ring slot** and the PLE norm into
  its own ring slot, so neither conv needs a copy in front of it;
- the GDN conv and the whole gate chain are one launch (`k_gdn_conv_gate`),
  the gate's 48 elements riding on threads past the channel count;
- `q_conv` and `k_conv` are adjacent, so one `l2_norm` of 32 groups does both;
- `hc_combine` also produces the **next** hc_mix's norm (`k_hc_combine_norm`)
  — both reduce over the same 2 560 elements — including across the layer
  boundary, except before the PLE layer, which rewrites the residual;
- the shared expert's down-projection forms its own `silu(gate)*up`
  activation inside the GEMV (`GX_SILU_MUL`);
- `k_moe_finish` does the weighted expert sum, the shared-expert gate and the
  add in one launch, writing `ffn_moe_out` and `ffn_shexp_gated`; the sigmoid
  lives here rather than in the GEMV epilogue so `shared_expert_gate` stays a
  tap too;
- `k_qsa_qk_post` does the `[q|gate]` split, the three QK-norms, the three
  IMRoPEs and the output gate's sigmoid — nine launches with grids as small as
  four workgroups of one wave — in one kernel of 30 workgroups;
- the attention combine applies the gate and writes both `kqv_out` and
  `attn_gated`;
- **the top-k is not launched at all** when the budget covers the cache. At
  ctx 512 against a 2 051-cell budget the reference's own `ggml_top_k` is the
  identity, so `k_qsa_expand` writes the selection. That was a 1 024-thread
  workgroup doing four passes of LDS histograms and scans over ~70 scores,
  every QSA layer, and it is the largest single piece of the 292 µs a QSA
  layer cost.

### Per-projection bandwidth, and two knobs instead of two guesses

`--profile` now reports, per batch group,
`prof_gemv_<group>_us`, `_mb` and `_gbs` — the weight bytes those launches
actually read over their device time — for `hc_down_inject`, `hc_up`,
`gdn_qkv_gate_beta_alpha`, `ssm_out`, `ffn_router_shexp`, `shexp_down`,
`qsa_q_k_v_indexer`, `attn_output` and `ple`. That is the number to hold
against §M5's 800 GB/s, not the lumped 5 890 µs.

Two things the brief asked about cannot be settled without running, so both
are **runtime knobs** rather than assumptions:

- `--gemv-lds 0|1` stages the activation slice in LDS instead of reading it
  from L1. The kernel is weight-bandwidth-bound and x is small enough to sit
  in L1, so this may well be neutral; sweep it.
- `--gemv-min-rows N` (default 1024) is the row count below which K is split.
  **Total waves == total rows**, so wider workgroups never add parallelism —
  only splitting K does, at the price of a reduce launch. Sweep it against
  `prof_gemv_*_gbs`.

On 16-byte loads: two consecutive Q8_0 blocks are 68 bytes and `qs` sits at
offset 34b+2, so a block's payload is 4-byte aligned only for even b and
**never** 16-byte aligned. A `dwordx4` from such an address is not something
this file will do on the strength of a guess. The widest safe request stays
the 32 contiguous bytes a wave pulls per block; what is tunable is how many
are in flight, so the unroll depth is the compile-time `GEMV_UNROLL`
(default 4).

**Correctness:** the CPU graph after this rewrite is **bit-identical** to
before it — 538 of 538 taps `cos=1.000000` against the pre-rewrite dump,
`missing=0` — and the llama.cpp comparison is unchanged (`l_last-0`
0.999993, `l_last-3` 0.999407 under `--quant-act`). 35 kernels, still zero
scratch and zero spills, occupancy 16 except `attn_flash_split` at 15;
`k_gemv_batch` is 46 VGPR. Not run: no GPU was touched.
