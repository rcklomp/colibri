# qerr: what does ggml's int8-activation expert chain change, on real GLM-5.3-Flash data?

Question: the GPU engine computes a routed expert with FLOAT activations (x and the SwiGLU output h stay float, weights
are IQ3_S / IQ4_XS / Q6_K). A CPU "fourth lane" built on ggml's kernels would compute some expert slots with llama.cpp's
INT8 chain instead. How much does the expert output, and the layer's MoE output, change when 1 of 8 slots (or all 8)
uses the int8 chain? CPU only, no GPU program is started, nothing is written outside this directory.

## Method (per dump D, MoE layer L = 3..44, slot s = 0..7)

Inputs, all read only: the 5 GGUF shards (`gguf_init_from_file(no_alloc)`, shards mmap'd, expert e of a tensor
`[K, N, 288]` is the contiguous slab at `data_offset + tensor_offset + e*N*row_bytes`), and the GPU engine's float32
taps of the last prompt token (`ffn_norm-L`, `ffn_moe_probs_biased-L`, `ffn_moe_weights{,_norm,_scaled}-L`,
`ffn_moe_{gate,up,swiglu_limited,down,weighted}-L`, `ffn_moe_out-L`, `ffn_shexp-L`, `ffn_out-L`).

1. **Candidates / slot identity.** The 8 candidates are the top-8 indices of `ffn_moe_probs_biased-L` (ties: lower index
   first). For every candidate the FLOAT reference output is computed (step 2) and each slot s is assigned the candidate
   whose output has the best cosine with the GPU's `ffn_moe_down-L` slot s. That best cosine is the harness validation:
   `1-cos > 1e-4` (cos < 0.9999) or two slots on one candidate = FATAL (loud banner, run stops, rc 3); `1-cos > 1e-5`
   = WARN. The result also states whether slot order == descending router score.
2. **FLOAT reference ("exact").** gate/up/down rows are dequantised with `ggml_get_type_traits(type)->to_float`, the
   dot products with the float x / h accumulate in double. gate = Wg x, up = Wu x, h = swiglu(gate, up), out = Wd h.
   SwiGLU is the engine's own (franken `glm5_cpu.cpp` `rows_for`): `g = min(gate, 10)`, `u = clamp(up, -10, 10)`,
   `h = g / (1 + exp(-g)) * u`; checked against the `ffn_moe_swiglu_limited` tap (computed in float from the GPU's
   gate/up taps; the check is reported as `swiglu_chk`, and the number of elements beyond the clamp is counted, since
   a clamp that never triggers cannot be distinguished by data).
3. **INT8 chain (ggml).** `quantize_row_q8_K(x)`, `ggml_vec_dot_<type>_q8_K` (the x86 AVX2 kernels of
   `~/src/llama-glm53` @ 39931761a, nrc = 1, one call per weight row) for gate and up, h = swiglu_f(gate_i, up_i) in float
   (`expf`), `quantize_row_q8_K(h)`, `ggml_vec_dot_<type>_q8_K` for down. Two extra "ablation" outputs separate the two
   quantisations: `xq` = only x quantised (int8 gate/up, h float, float down), `hq` = only h quantised (float gate/up,
   int8 down on Q8_K(h_float)).
4. **Metrics.** Expert level: relative L2 `||int8 - float|| / ||float||` and `1 - cos` (computed as
   `||a/|a| - b/|b||^2 / 2`, no cancellation) of the expert output, of gate, of up, and rel L2 of h. Layer level:
   Y = sum_s w_s out_s with the router weight tap that reproduces `ffn_moe_weighted` (picked per layer among
   `weights`, `weights_norm`, `weights_scaled`; it is `weights_scaled`). Baseline is the float reference Y_f.
   `dY` = `||Y_variant - Y_f|| / ||Y_f||` for: slot k alone on int8 (k = 0..7), the slot with the smallest and the
   largest `|w_s| * ||out_s||`, and all 8 on int8 (llama.cpp-like). `dFFN` is the same change relative to
   `||Y_f + shexp||` (the shared expert output stays the GPU tap; `ffn_out = Y + shexp` is verified, not assumed).
5. **Summary.** Distribution (n, mean, median, p90, p99, max) overall, per dump, per type combination; per-layer table;
   worst 5 expert samples (dump, layer, slot, expert id), worst 5 single-slot and all-8 layer samples.

Per-sample lines: `EXP ...` (one expert slot) and `LAY ...` (one layer) carry every number, `TYPES`/`COMBO` list the
quant types per layer (gate/up/down: IQ3_S/IQ3_S/IQ4_XS in 39 layers, IQ3_S/IQ3_S/Q6_K in 12 and 44,
IQ4_XS/IQ4_XS/Q6_K in 11; shapes verified from the file: K=4096, N_FF_EXP=2048, 288 experts).

## Harness validation (what the self-checks print, all in the SUMMARY)

float reference vs GPU `ffn_moe_down` (rel L2), vs GPU gate and up; slot-id `1-cos`; kernel self-check
`vec_dot vs deq(w).deq(Q8_K act)` (the ggml kernel result against dequantised weight row . dequantised Q8_K activation,
i.e. the kernels do what they should, so the int8 error is purely the activation quantisation); `swiglu_chk`;
`Y(GPU down taps)` and `Y(float ref)` vs `ffn_moe_out`; `ffn_out` vs `ffn_moe_out + shexp`; which weight tap won.

## Small test seen so far (NOT the full result; 3 layers of g136 plus 2 of ref512, 1 thread, nice'd)

Layers 7, 11, 12 of `g136_eager` and 7, 8 of `ref512_keep`: slot order == descending `probs_biased` score in every
(dump, layer) so far (candidate rank = slot index); slot-id `1-cos` ~1.8e-14; float reference vs GPU down rel L2
1.5e-7..1.9e-7 (gate/up ~1.1e-7); kernel self-check ~2e-7; `swiglu_chk` ~2e-8 with 0 clamped elements;
`weights_scaled` reproduces `ffn_moe_weighted` (2.5e-8); `ffn_out = ffn_moe_out + ffn_shexp` to 2.8e-8.
First numbers (g136 layers 7, 11, 12 only, do not quote): expert out rel L2 1.4%..3.0%, one-slot dY 0.4%..1.5%,
all-8 dY 1.6%..2.2%. Run the full experiment for the real distributions.

## Build and run

    cd ~/src/cpulane/qerr && make           # builds ./qerr (host gcc/g++; ggml objects from ~/src/llama-glm53, read only)
    ~/src/cpulane/qerr/run_qerr.sh          # both dumps x layers 3..44 x 8 slots, 2 threads, stdout, ends with "=== qerr exit rc=N"

Environment: `QERR_THREADS` (default 2), `QERR_PREFIX` (e.g. `nice -n 19 taskset -c 14,15`). Extra arguments go to
`./qerr`: `--dump LABEL=DIR` (repeatable; default g136 = `~/bench/franken/glm5/gpu5/g136_eager`, ref512 =
`~/bench/franken/glm5/ref512_keep`), `--layers A-B`, `--threads N`, `--model-dir DIR`, `--keep-going` (continue after a
validation failure), `--list-types`. rc: 0 ok, 2 setup/IO error, 3 harness validation failed.

Build: same flags as `../Makefile` (`-O3 -mavx2 -mfma -mf16c -mtune=znver2`, `GGML_AVX2/FMA/F16C`) but host gcc 15
instead of docker's gcc 13 (the cpulane part A/B objects are untouched). Linked objects: `ggml.c` (type traits),
`ggml-quants.c`, `ggml-cpu/arch/x86/quants.c`, `gguf.cpp`, `ggml-threading.cpp` and `qerr_stubs.c` (the three lookup
tables of `ggml-cpu.c` plus four never-called ggml-backend stubs).
Cost: about 1.5 s per (dump, layer) on one warm thread (8 experts, ~12 MB each, both chains); cold shards add disk reads.

## Assumptions and open risks

- One token per dump (the last prompt token of a 136- and a 2200-token prompt): 84 (dump, layer) samples, 672 expert
  slots. Distributions over layers are real, but there are only two activation vectors per layer; token-to-token
  variation is not covered.
- The int8 error here is activation quantisation only (weights identical). llama.cpp-vs-engine differences elsewhere
  (router, attention) are not modelled: the baseline is the engine's own float chain and routing, only the expert
  chain changes.
- The "all 8" case replaces all 8 slot outputs by the int8 chain but keeps router weights, routing and the shared
  expert unchanged; errors of different slots are not independent of each other (same x, same Q8_K(x)), the program
  measures the actual summed change.
- Q8_K(x) is the same for gate and up (as llama.cpp does for a fused gate/up input); h is quantised once per expert.
- The float reference is double-accumulated; the GPU kernel rounds differently (1.8e-7 apart), which is two orders of
  magnitude below the effect measured.
- Slot identification relies on the top-8 of `ffn_moe_probs_biased` being the expert set; if that fails the run stops.
