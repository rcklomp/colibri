# P4 — S-tiled Vulkan matmul shaders, dense and expert (spec, 2026-09-06)

Roadmap: `PREFILL-ROADMAP-2026-09.md` item P4, now the top item. Tier: Opus
implements (shader + backend dispatch); this spec is Fable's. Gate:
`prefill_gate.sh` exit 0 — oracle at `COLI_KDA_GPU=0` exercises the dense
path through `mv()`/`mv_rows_s()` and the expert path through
`coli_vk_expert_group`, so a shader change is fully covered by (a)(b); the
TTFT half at the serving knob plus the decode column is what decides.

## The fact being fixed

Both compute shaders (`qmatmul.comp`, `qmatmul_gate_up.comp`) are matvec
kernels dispatched once per input row: grid `(ceil(O/nsg), S, 1)` with
`s = gl_WorkGroupID.y`, `x[s,:]` staged in shared memory, one subgroup per
output row `o`, lanes striding the packed weight words of row `o`, one
`subgroupAdd`. Weight traffic is therefore **S × |W|** — every row re-reads
the whole matrix from VRAM. Measured consequences (record §PREFILL-PROFILE,
§P2):

- expert group `eg` time is proportional to rows, not experts
  (75 628 → 47 921 expert-calls from K=1 to K=16, eg time −3 %);
- P2's batched KDA projections still cost 18 ms/token for ~2 ms of
  arithmetic (8 + 1 dispatches per layer per chunk of 112 rows, each reading
  its matrix 112 times: ~1.8 GB per matrix at 960 GB/s ≈ 2 ms *per matrix*).

At S=1 (decode) the shader is right and must not change.

## Design — a register tile over rows, same reduction per row

Keep everything about the current kernels (subgroup per output row, lane
stride over packed words, per-format decode, `subgroupAdd`), and give each
lane **R accumulators** instead of one:

```
tile   = R rows of x  (R = 8; a specialisation constant, tuned 4/8/16)
grid   = (ceil(O / nsg), ceil(S / R), 1);  t = gl_WorkGroupID.y;  s0 = t * R
per lane, per packed word wi of output row o:
    pk = w[rowBase + wi]                      // loaded ONCE
    for r in 0..R-1:  if (s0 + r < S)  acc[r] += dot(x[s0 + r, i0..i0+7], decode(pk))
per row r:  sum[r] = subgroupAdd(acc[r]);  lane 0 writes y[s0 + r, o] (with the
            per-row / per-group scale exactly as the current shader applies it)
```

- **x access.** Staging R rows in shared costs R × I × 4 bytes (128 KB at
  R=8, I=4096) — over the 64 KB LDS budget. Do not stage: read
  `x[(s0 + r) * I + i]` from global memory. The whole x tile is a few hundred
  KB shared by every workgroup and lives in L2; the weight stream is the
  traffic that matters. (If profiling shows x reads dominating at R=8, stage
  R=4 rows with I ≤ 4096 as a second variant.)
- **Bit-identity.** For a given (row, o), the lane's word sequence and the
  per-word arithmetic (`a += x * w`, then `sum += a * scale[g]` for grouped
  formats) are the same expressions in the same order as today, only
  interleaved across rows in independent accumulators; the `subgroupAdd`
  tree is the same. Expect teacher-forcing identical. If the compiler
  contracts FMAs differently in the tiled loop, the result differs in the
  last bit: then ship the tiled pipeline behind `COLI_VK_TILE=1` (default
  on only if the oracle is identical) and put the cosine in the commit body.
- **Two pipelines, not one.** Compile the tiled variant as separate SPIR-V
  (`qmatmul_tile.spv`, `qmatmul_gate_up_tile.spv`; Makefile `VK_SPV` +
  `derive_sibling`). The backend picks the tiled pipeline when `S > 1`
  (dense: `coli_vk_matmul`, `coli_vk_matmul_pair`; experts: per expert `c`
  in `eg_prepare_submit` when `rows[c] > 1`, on all three devices `G/G2/G3`)
  and the existing pipeline when `S == 1`. Decode never touches the new
  code — the gate's decode column must read within 1 %.
- **Formats.** Implement fmt 4 (grouped int4, everything on rome) and fmt 1
  (int8) first; fmt 7/8 follow the same pattern and can be added when they
  exist on this box. **fmt 5 (int3-g64) is moot for this engine**: G15
  (`ROADMAP-2026-09.md` 4h, record §G15) simulated it on GLM-5.3's own int4
  weights before building any shader and the model does not survive it
  (`teacher_forcing` diverges, cosine 0.973/0.878) — do not add fmt 5 to the
  tiled pipeline on GLM-5.3's account; it would only be worth doing again for
  a different model or a finer-grained int3 variant. The tiled pipeline
  must refuse (return 0 → caller falls back to the per-row pipeline) for a
  format it does not implement.
- **Push constants.** The existing `struct PC {fmt, S, I, O, rowWords, gs}`
  carries S already; the tiled shader derives its tile index from
  `gl_WorkGroupID.y` and bounds-checks `s0 + r < S`. No new host fields.
- **Descriptor sets.** Unchanged: the tiled shaders bind the same buffers
  (x, w, scale, y; gate_up adds up-weights/scales and h) so `G.dset`,
  `G.eg_gu[]`, `G.eg_dn[]` and `dsl`/`dsl_gu` are reused.

## Expected effect (to be measured — every projection in this track so far
## was optimistic)

Weight traffic per matrix per chunk: `ceil(S/R) × |W|` instead of `S × |W|`.

| path | today | after (R=8) | where it shows |
|---|---:|---:|---|
| dense proj+ko (KDA, 34 layers) | ~18 ms/token | ~3 | kda bucket 31 → ~16 ms/token |
| dense MLA proj, shared expert | ~9 | ~2 | |
| expert group `eg` (overlapped with cpu experts) | 88 ms/token at chunk 128 | rows per expert 2–16 → one tile each: ~15–25 | ffn_moe stays bounded by **cpu experts (84)** until P3 lands — P3 and P4 together take the bucket from 100 to ~40 |

Honest expectation for P4 alone on the serve path: **~1.15–1.25×** (the MoE
bucket is still pinned by the CPU half). P3 + P4: **~1.6–1.9×** on top of P2.
State the serve-path number, not the profile number, in the commit.

## Procedure (no exceptions)

1. Stop the gateway; `pgrep -x glm53` empty. Pristine for this item is the
   P2 binary `32ad1f7b…` — copy `c/glm53` to `~/bench/glm53.p2` first (the
   `~/bench/glm53.pristine` copy is the pre-P2 build and stays for history).
2. `tools/hot-expert/prefill_gate.sh ~/bench/glm53.p2 ~/src/colibri/c/glm53 p4`
   — teacher-forcing identical, logits, TTFT at 30/300/1000 twice, decode
   within 10 % (make it 1 % for this item: the S==1 path is supposed to be
   untouched), `MIN_SPEEDUP=1.1`.
3. `tworeq.py` identical across 3 requests.
4. `prefill_profile.sh` at **`COLI_KDA_GPU=2`** (the serving knob) on the
   600-token prompt, before and after — the per-bucket split in the record.
5. Restart the gateway; one request from Open WebUI completes.

## Non-goals

- The CPU expert rows (P3): independent, and needed for the bucket to move.
- The recurrence and the sparse attention (P5).
- Any change to the S==1 shaders or to decode.
