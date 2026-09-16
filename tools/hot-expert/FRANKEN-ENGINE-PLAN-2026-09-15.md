# Franken-engine plan: a VRAM-resident model class on rome, and what to do about it (Fable, 2026-09-15)

> Written at the Fable tier, from the Mac, while the rig lock was held by a
> benchmark with ~7.5 h to run. Nothing here was run on the rig. Every number
> is one of three kinds and is labelled: **measured** (tonight's context ladder
> on this rig, or a row in the record / roadmaps / CLAUDE.md), **published**
> (the three reference projects' own figures, on their hardware, not ours),
> or **projected** (derived here, with the derivation shown). A projection is
> a thing to be beaten or refuted by the items below, never a result.
>
> The question the owner asked is "should the compute backend move to HIP, and
> should this box serve a 27–35B VRAM-resident model instead of dragging 180 GB
> through host RAM". This plan's answer, before any measurement: **those are two
> questions, the second one is the real one, and the first one is mostly
> already answered by the profile.** Sections 1–5 say why and what to measure.
>
> **Rev 2 (2026-09-16).** Rev 1's §0 said no pair of engines could run the same
> model in the same placement on both backends. That was reasoned from what is
> on the rig's NVMe, and the owner rightly rejected it: the model set is not what
> is downloaded. Qwen3.6-35B-A3B runs on hipFire (eight registry SKUs, verified
> at its README: `qwen3.6:35b-a3b`, `-mq2`, `-mq3p`, `-mq4p`, `-mfp4`, `-mq4r`,
> `-mq5`, `-mq6`; MQ4R is its validated RDNA3 route) and on this fork
> (`c/qwen36.c`, `c/tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B`, the
> published int4-gs64 container). What stands between them is **one code gap on
> this fork** — `qwen36`'s VRAM tier is CUDA-only — and that gap is a port of a
> pattern this tree already has twice. So the same-weights GPU pair does not
> exist *today* and does exist *after item V1*; §0, §2.6 and §3 are rewritten
> around that, H2 stays first because it needs no code, and a second claimed
> gap (the converter's "int4 path is WIP") is recorded in §1 as stale.

> **Rev 6 (2026-09-16, day 1). V1 is BUILT and GATED — steps 1 and 2 both pass
> (record §V1, branch `perf/v1-qwen36-vk-tier`), so the same-weights GPU pair
> §0 says H2b needs now exists.** `make -C c qwen36-vk VK=1` is a Vulkan expert
> tier for `qwen36` behind the existing `qt_*` contract: all **10 240** experts
> (40 × 256, int4-gs64) resident on dev3 in **18.12 GB**, filled in **6.9 s**,
> VRAM hit rate **100 %**, `Q36_VULKAN=1` to turn it on and bit-identical to
> the CPU engine with it off. Per-position KL against the tier-off arm is
> **−1.2e-10 mean / 100 % top-1 / cosine 1.000000000** over the 625-position
> packet — seven to nine orders below the bar §4's X2 rows set. In arm C's own
> harness, A,B,B,A: decode **14.58 → 21.64 tok/s (+45.6 % conservative)**, TTFT
> **33.45 → 10.67 s (−67.8 % conservative)**, with the tier-off arms
> reproducing arm C's floor to within 1–2 %. Two corrections this produced:
> **(a)** V1-STEP0 §(c)/(d)/(e) had the int4 nibble convention backwards — both
> backends take offset binary at their upload API and `qwen36`'s container is
> two's-complement, so the `stage()` XOR is KEPT, not dropped (proved
> exhaustively, `c/tests/test_qwen36_vk_nibble.c`; the document is corrected in
> place). **(b)** `dev_alloc_footprint`'s cudaMalloc granularity curve does NOT
> transfer to RADV — payload accounting is within **0.8 %** of what the driver
> reports, against CUDA's 22–28 %. §3's branch 3 is now measurable rather than
> projected, and what §4's X-items should target on this engine is the
> remaining ~46 ms/token, which is still entirely CPU: V1 moved the routed
> experts only, and the trunk (DeltaNet, attention, dense, LM head, shared
> expert) is untouched. **H2b is unblocked.**

> **Rev 5 (2026-09-16, day 1). H0 is closed as O3 on this box as it stands;
> M0 has its answer; the H-track waits on one owner decision.** After
> `rocm-device-libs` was installed, two more mismatches surfaced and were
> lifted user-locally (record §FRANKEN-H0: this ROCm 6.2.0 is a 24.04 build on
> Ubuntu 26.04 — `CPLUS_INCLUDE_PATH` to the GCC 15 headers, and a
> `libxml2.so.2` alias for `ld.lld`). hipFire then compiles and caches its
> kernels but every request fails inside its own forward ("bench_decode
> forward failed", no HIP error exposed), unchanged across `--kv-backend
> vmm/contiguous` and prewarm on/off. hipFire documents "ROCm 6 or newer" for
> this card and took its own 7900 XTX numbers on 6.4.3; 6.2.0 is simply
> untested by them. **Owner decision:** Ubuntu 26.04's own archive ships ROCm
> 7.1 (`apt install rocm`, built for this release, so neither mismatch above
> applies); with it, `h0_chain.sh` reruns unchanged and, on a pass, the full
> H2 chain launches that night. Without it the track ends at O3 and Vulkan
> stays, as §3 says. **M0 (record §FRANKEN-M0):** rccl-tests builds and RCCL
> sees all three cards, but P2P setup fails (`hipIpcGetMemHandle: invalid
> argument`) on both `-g 3` and `-g 2` — no collective runs; TP/EP over RCCL
> is off the table here (§5's first outcome); the kernel line lacks
> `iommu=pt`, an owner-side thing to test, not a session's. hipFire's MTP and
> Redline knobs are recorded for X5 (`HIPFIRE_MTP_MODE`, `HIPFIRE_MTP_K`,
> `HIPFIRE_REPLAY_BACKEND`).

> **Rev 4 (2026-09-16, execution night 1, closing).** Two results that change
> §3 and §4, both measured (record §X1, §X2):
> 1. **X1 is rejected.** Retained command buffers are bit-identical on both
>    engines and worth **2.8 µs of a 25 µs submit** (`VK_PROF`: desc+record
>    2.5–3.1 µs, submit 25–27 µs, wait 331–348 µs per call; ceiling 97 × 2.8 µs
>    = −0.27 ms/token on qwen38-vk, ≈ −0.12 on glm53, against the projected
>    −3 to −8). A,B,B,A: NO VERDICT on every rotating/cold pair; warm-identical
>    qwen38-vk +0.2 % conservative. The branch `perf/x1-retained-cmdbuf` stays
>    unmerged; the record keeps the numbers.
> 2. **The Branch 3 ceiling (≤ 28 tok/s) is withdrawn.** Its derivation took
>    Q7's 0.30 ms per-submit *gap* for driver overhead; the profile says the
>    gap is the engine's own CPU work between submits, and the driver's share
>    is ~25 µs per submit — ~2.8 ms/token for ~120 submits. The lever on a
>    resident path is fewer submits, not cheaper recording. **V1 step 2 is not
>    gated on X1.** X1's own falsifier fired at 1.00×.
> 3. **X2 landed** (`GLM53_LOGIT_DUMP_ALL`, `kl_compare.py`, `gate_kl`;
>    G14/G15/clamp reproduce the record's order: KL 0.030 / 0.251 / 0.028) and
>    found something bigger than its gate: on the pristine serving binary the
>    **CLI/`--prompt` path under `COLI_KDA_GPU=2` generates garbage on the
>    564-token packet** (teacher_forcing dominated by token 154822; decoded text
>    "# 3.1.1.1…"), while the same binary at `=0`, and the served 4-slot path at
>    `=2`, are coherent; an 11-token prompt is fine. Gates whose teacher_forcing
>    oracle came from a CLI run at `=2` (p5, p5b, cancel, devmerge's memfloor)
>    compared garbage with garbage; `prefill_gate.sh` ran its oracle at `=0`
>    and is sound. Bisect is the next item on GLM (prefill roadmap rev 28), not
>    part of this plan. X1's glm53 CLI oracle is void for the same reason; its
>    serving-path evidence is the 4-slot `tworeq` text identity.
> Cost note: the night used one Opus agent (X1), four Sonnet, one Haiku, and
> hit the session limit once; the H-track waits on the owner's package install.

> **Rev 3 (2026-09-16, execution night 1).** Three corrections from running
> the plan, all measured on the rig, none changing the decision tree:
> 1. **H0 and M0 are blocked, not failed.** hipFire (upstream `warpfront/hipfire`
>    v0.3.1) builds, pulls the MQ4R SKU and serves its API on this box, but it
>    JIT-compiles its kernels at request time through ROCm's clang, and this
>    ROCm 6.2.0 install has no `rocm-device-libs` package; rccl-tests fails on
>    the identical error. The triton-bundled bitcode on the box lacks the
>    `oclc_*` control set and clang refuses it. Owner action:
>    `sudo apt install rocm-device-libs` (candidate 1.0.0.60200-66~24.04),
>    then `h0_chain.sh` reruns unchanged. Record: §FRANKEN-H0, §FRANKEN-M0.
> 2. **The H2 validity rule "any A row with majflt > 0 invalidates the arm"
>    was wrong** and is withdrawn: `c790851` had already measured that a
>    loaded `glm53` (~89 GB anon) beside the 183 GiB model on a 247 GiB box
>    caps residency near 92 % and turn 1 always faults (reference ladder
>    `ctx09152003`: majflt 21 663 / 6 611 / 4 521 / 4 176 on turns 1–4 — that
>    IS the serving regime). `franken_chain.sh` now invalidates an arm only on
>    a residency floor breach (< 90 %), prints majflt per row, and flags a
>    turn `MAJFLT-HIGH` above 2× the reference; H3 annotates such cells rather
>    than dropping them.
> 3. **Arm C has its first number** (smoke, `fk09152356`, CPU-only `qwen36`
>    on the gs64 container, all experts resident, 8 threads): 512-token turn
>    TTFT 32.85 s, decode **14.84 tok/s**; follow-up 20.67 s / 14.38 tok/s.
>    That is the floor branch 3 (V1) must beat, and it is already ~4–5× GLM's
>    decode at the same depth on the CPU alone.
> Also found and fixed in passing: every chain ending in `accept_live.sh` was
> truncating its own log (`tee /dev/stderr` under `nohup > log 2>&1`;
> `5062ff4`). H1 landed (`perf/franken-h1`); V1 step 0 landed
> (`tools/hot-expert/V1-STEP0-2026-09-16.md`: the CUDA tier takes gs64 group
> scales, no new shader needed, one nibble-encoding difference to drop).

## 0. Decision in one paragraph

The first head-to-head (§2, H2) cannot separate "HIP beats Vulkan" from "a
3B-active model in 960 GB/s VRAM beats a 180 GB model whose misses come from
host DRAM", because **today** no engine here runs the same model in the same
placement on both backends: Colibri's Qwen3.6-35B-A3B engine has a VRAM tier
for CUDA only, so on this box it is a CPU engine. H2 is therefore not designed
to answer the backend question. It answers the **product** question — what the
owner's turn costs at depth in each world, and whether the moat (P7/P9 prefix
reuse) survives — with the one same-weights control possible today (§2.3), and
it needs no code. **The same-weights GPU pair exists after one port, V1 (§2.6):
a Vulkan expert tier for `qwen36`**, the pattern `glm53` and `qwen38-vk` already
use. V1 is worth building on its own account — it is the "second, VRAM-resident
lane" inside Colibri's serving stack, with P7, P9 and the KV slots intact — and
once it exists, H2b (§2.6) is the engine-vs-engine number on the same weights in
the same placement. Even then the backend question proper is answered by the
same-op microbenchmark (§3, D-3), because an engine comparison still folds in
schedulers, formats and submit models; and the profile already says the
backend's ceiling on the daily driver: **the GLM-5.3 token is CPU/DRAM-bound, not
GPU-kernel-bound** (G3: 69 % of the token on one core; §RP1-CORRECTION: CPU int4
experts 40.3 ms/token at ~57 % memory-bound; GPU expert groups were 22 ms of a
373 ms token at G3 and have been overlapped with CPU work since G9). A faster
GPU kernel moves the GLM token by single-digit percent. **No result of the
head-to-head justifies rewriting Colibri's backend.** What it can justify is a
second, VRAM-resident lane on this box, and a short, ranked list of grafts
(§4) — the first of which (retained command buffers) helps every Vulkan
engine here today and is what makes any future Colibri-native resident path
viable at all.

Expected outcome, stated so a surprise reads as one: hipFire runs, decodes
the 35B at ≥ 10× GLM-5.3 at every depth (arithmetic, not a close race), and
the decisive unknown is whether it reuses prefixes across turns. If it does
not, its per-turn wait at 18k is a full re-prefill and Colibri's incremental
turn may still win the number the owner actually waits for.

## 1. What is established (do not re-derive)

| fact | value | kind | source |
|---|---|---|---|
| GLM-5.3 prefill rate vs depth on this rig | ms/token ≈ 113.9 + 0.01258·d (5 points, 643..13 628, fit within ~1.5 %) | measured | tonight's ladder |
| cold prefill of N tokens, integrated | 113.9·N + 0.00629·N² ms → 2k **260 s**, 4k **572 s**, 8k **1 355 s**, 16k **3 555 s**, 18 055 **4 107 s (68 min)**, 64Ki **34 480 s (9.6 h)** | projected from the fit | derivation: ∫₀ᴺ(113.9+0.01258·d)dd |
| GLM-5.3 decode vs depth | 3.71 @1 287, 4.06 @2 559, 3.59 @4 920, 2.99 @9 171, **2.23 @18 055** tok/s (1.82× fall from peak) | measured | tonight's ladder |
| GLM-5.3 serving numbers, decode track | 3.03 / 2.98 rotating, 3.44 / 3.45 warm-identical, 1.86 / 1.90 cold | measured | PREFILL-ROADMAP rev 26 |
| GLM-5.3 fresh-process token | 134.90 ms knob-on (`COLI_KDA_GPU=2`), 156.77 knob-off | measured | ROADMAP Track G header |
| GLM-5.3 new chat / follow-up through the gateway | ~20 s (P7 restores the ~4 000-token tool block) / ~2 s; a changed tool set pays ~10 min cold | measured | CLAUDE.md |
| GLM-5.3 routed expert: size and tier share | 171 GB resident / 12 096 mappable ≈ **14.1 MB per expert**; tier serves ~79 % of routed calls, ~21 % from the CPU | measured | briefing, CLAUDE.md `[MAP]` line, G15 note |
| CPU int4 expert kernel, isolated, 8 threads | 21.95 GB/s | measured | §G11 |
| qwen38-vk serving numbers | warm-identical 7.23, rotating 5.3–5.4 | measured | Q7 row |
| qwen38-vk submit overhead | `vk-issue` 13.8 + `vk-take` 4.8 ms/token; per-submit gaps 0.30 / 1.50 ms; "a 2.7 ms submit ramps its own clocks" | measured | Q9 spec table, Q7 row |
| `backend_vulkan.c` re-records its command buffers on every submit | `vkResetCommandBuffer`/`vkBeginCommandBuffer` at 1007, 1113, 1177, 1268 (56 reset/begin/submit sites) | read | `c/backend_vulkan.c` |
| Colibri's Qwen3.6-35B-A3B engine | `c/qwen36.c`: 40 layers (10 × [3 DeltaNet + 1 attention]), 256 experts top-8 + 1 shared; dense int8 in RAM, experts LRU-cached in RAM; **CPU-only by default, the VRAM tier is CUDA-only** (`qwen36_tier.h` is compiled under `COLI_CUDA` only; no Vulkan symbol in the file; the tier API is ~20 `qt_*` entry points, 28 call sites in `qwen36.c`); int4-gs64 container ~20 GB, ~30 GB RAM | read | `docs/qwen36.md`, `c/qwen36_tier.h`, `c/Makefile` |
| The engine DOES read int4 containers; the converter's "int4 path is WIP" is stale | `qwen36.c` detects packed int4 by on-disk size (lines 1445–1529), unpacks to int8 for the CPU path and **keeps the packed nibbles (`g4/u4/d4`) for a GPU tier**; `qt_init` takes `expert_is_int4`; `docs/qwen36.md` recommends the int4-gs64 container over per-row int4 (GLM's #455 think-loops). So a Vulkan tier holds ~20 GB of packed int4 experts — one card — not ~35 GB of int8. **Open for V1 step 0:** whether the CUDA tier's kernels take gs64 group scales or only per-row `gs/us/ds`; the answer decides which container V1 serves | read | `c/tools/convert_qwen36.py` line 113 vs `c/qwen36.c`, `c/qwen36_tier.h` |
| Also on the rig's disk, and unusable by Colibri | `Qwen3.8-27B-UD-Q5_K_M.gguf` (19.8 GB), a UD-IQ4_XS GGUF of Qwen3.8-Flash-Next, `DeepSeek-V4-Flash-0731-UD-IQ2_M` (85 GB, over the 72 GB of aggregate VRAM). Colibri maps safetensors (`st_map_shard_range`), not GGUF; none of these is a candidate on either side | reported by the coordinator, formats checked against `c/st.h` | — |
| That engine's only numbers | 9.2–11.3 tok/s with a CUDA tier on 8 GB cards; **0.35 tok/s CPU-only before the tier** — on a Threadripper 3945WX, not this box | measured elsewhere | `docs/qwen36-cuda-tier.md` |
| Vulkan vs HIP on the same card, Colibri's own | on an RX 9070 (RDNA4) the Vulkan backend is faster than Colibri's ROCm/HIP backend | measured elsewhere | `docs/vulkan.md` |
| Vulkan vs HIP on the same card, hipEngine's own | p4096: hipEngine 290.6 / 18.69 vs Vulkan (halo box) 420.95 / 24.55 vs llama.cpp HIP 395.02 / 19.63 prefill / decode; the gap attributed to "dataflow submission efficiency" | published | briefing |
| hipFire single-XTX figure | 253.3 tok/s TG128 (empty context); its own multi-turn 191 average, **160 at ~18k**; MQ4R 35B-A3B 18.7 GB, ~22 GB VRAM | published | briefing |
| MTP on host-resident models here | Q9 step 0: V*(4)/V*(1) = 2.668 ≥ 2.4, kill fired; hipEngine 0.955× AR | measured / published | Q9 row, briefing |
| ROCm on the box | 6.2.0 at `/opt/rocm-6.2.0`, not on PATH, never used; `librccl.so` present; rccl-tests not built; all three GPUs PCIe 4.0 x16 CPU-direct, AtomicOpsCap 32/64+, ReqEn+, Routing+ on the bridges | verified tonight | briefing |
| Colibri's tiers fill every card | G6: cap 2200 stops at 1752 experts on "25.0 of 25.7 GB used, 1.0 reserve"; serving runs 1695/1695 on dev2/dev3 and 1248 + KDA pool + dense on dev0 | measured | G6 row, CLAUDE.md |
| dev0's identity | `0000:83:00.0` = `/sys/class/drm/card1`; card numbers are scrambled | measured | record §Q13 |
| run-to-run spread on this box | 3–5 %; an uninterleaved pair read +20 % / +11 % on true +3.9 % / 0 | measured | Q9 spec, `gate_lib.sh` |

Two things the table makes unavoidable:

1. **H2, as it can be run today, is a model-class comparison.** GLM-5.3-Flash
   int4-g64 is 183 GB; hipFire has no GLM. Qwen3.8 (Colibri's other engine) is
   173 GiB of FP8 with 512 routed experts per layer — an MQ4R of it is far
   larger than one card and larger than all three (72 GB), so hipFire cannot
   serve the model class this box serves today at all. The only same-weights
   run available *today* is Colibri's `qwen36` on the CPU, which measures an
   8-core CPU against a 7900 XTX, not Vulkan against HIP. **The same-weights
   GPU pair is one port away (V1, §2.6), not unavailable** — Rev 1 got that
   wrong.
2. **A backend port cannot be paid for by the GLM profile.** Even a 2× GPU
   expert kernel is worth ≤ ~10 ms of a 135 ms token (≤ 7 %), and G14 already
   showed the CPU half of that bucket is exhausted without changing numerics.

## 2. The head-to-head (items H0–H4 today; V1 and H2b after the port)

Two phases. **H2** runs on what exists — no engine code, ~4 h of rig time —
and answers the product question. **H2b** runs after V1 and is the
engine-vs-engine number on the same weights in the same placement. H2 is not
made to wait for V1: its decisive unknown (does hipFire reuse prefixes, Q2)
is independent of anything Colibri builds, and its GLM arm is the daily
driver's own curve.

### 2.1 The pair, and what is honestly not comparable

**Arm A — Colibri, the serving configuration.** `glm53` from the binary in
service, GLM-5.3-Flash-colibri-int4-g64, the gateway's env exactly as
`ttft_serve.engine_env` sets it (8 pinned threads, three devices,
`COLI_VK_EXPERTS2/3=1695`, `COLI_KDA_GPU=2`, a *copy* of the histogram),
`--kv-slots 4`, **`GLM53_PREFIX_CKPT=0` with a private `COLI_CKPT_DIR`** (as
every gate on this track; a restored checkpoint would report a prefill that
never happened), `GLM53_MAXT=32768`. Driven in engine mode by
`context_ladder.py`, the harness that produced tonight's numbers.

**Arm B — hipFire, one card.** Qwen3.6-35B-A3B MQ4R (18.7 GB) on **one** gfx1100
— the card that is Colibri's **dev3** (an expert-only device), so that if a
two-lane configuration is ever built GLM keeps dev0 (dense + KDA) and dev2.
H0 records the PCI-BDF → HIP-index map. Driven over HTTP by the same ladder.

**Arm C — control, same weights, Colibri CPU-only.** `qwen36` on the
int4-gs64 container of the same Qwen3.6-35B-A3B, all experts RAM-resident
(`cache/layer` = 256), 8 pinned threads, served by `coli serve` on a side
port and driven by the same HTTP driver as arm B. **This is not an engine
comparison** — it is (i) the floor any Colibri-native resident path (§3, branch
3) has to beat, (ii) the only same-weights sanity check on hipFire's output,
and (iii) what the owner would have on this model if hipFire fails H0. Bounded
to `--sizes 512,2048`, one repeat, 30 min wall, run last.

| held constant | how |
|---|---|
| the user text | `context_ladder.py`'s deterministic corpus slices from `ROME-3x7900XTX-2026-09-04.md` (624 KB; a 64Ki ladder needs ~236 KB), same `--steps`, same question suffix, byte-identical on every arm |
| the turn structure | one growing conversation; the reply fed back verbatim (engine mode: with P8's pin; HTTP mode: as received) |
| generation | `--gen 128` on every arm (32 is ~0.2 s at 160 tok/s and would measure the SSE chunking, not the engine); temperature 0 |
| the clock and the formula | one harness; decode = (n−1)/(t_done − t_first) on every side; TTFT = first content delta |
| the box's state | GPU DPM level read from sysfs and recorded before every arm, never changed; no other engine (`pgrep -x glm53`, `pgrep -x qwen38`, `pgrep -x qwen38-vk`, `pgrep -f "hipf[i]re"` all empty before each arm); VRAM asserted free before every arm (sysfs `mem_info_vram_used` < 1 GB on all three cards) |
| residency | arm A: fincore ≥ 90 % on the GLM shards before every turn and `majflt` per turn recorded (must stay 0); arm B/C: fincore on their own model files, same floor |
| order | **A, B, B, A**, then C — the GLM arm brackets the pair; each B arm is a fresh server process so B2 cannot inherit B1's KV or prefix cache |

| not comparable, said plainly | why it stays in the report anyway |
|---|---|
| the model (GLM-5.3-Flash vs Qwen3.6-35B-A3B) | it is the decision: the owner is choosing a model class, and no throughput number substitutes for reading the answers (§2.5) |
| the quantisation (int4-g64 / MQ4R / int4-gs64: three formats, none bit-comparable) | arm C vs arm B agreement is reported as a sanity, not an oracle |
| the placement (host + 3 tiers vs one resident card) | this *is* the class difference |
| the backend (RADV Vulkan vs ROCm HIP) | confounded with every row above; **the head-to-head says nothing about it**; D-3 in §3 does |
| depth in tokens (two tokenizers) | depth is reported in each engine's own tokens and in characters (identical by construction); comparison at matched *turn index*, with the token counts printed beside it |

### 2.2 Items

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| **H0** | **hipFire on this box at all.** Under the rig lock with the gateway stopped (a chain, not by hand — the watchdog): check `/dev/kfd` access for the user (`id -nG` includes `render`/`video`; if not, that is an owner action, `usermod`, and the item stops there and says so); `ROCM_PATH=/opt/rocm-6.2.0 rocminfo` lists three gfx1100; build hipFire (Rust toolchain user-local via rustup; `hipcc` from `/opt/rocm-6.2.0/bin`) — **read its README for its minimum ROCm first; if it needs > 6.2, stop and report: installing a second ROCm is a system change the owner decides, not this item**; download the Qwen3.6-35B-A3B MQ4R; serve it on dev3's HIP index on a side port; one request through its HTTP API. Record: the API shape (OpenAI-compatible `/v1/chat/completions` with SSE, or Ollama-style `/api/chat` NDJSON — the driver needs to know), whether the response carries `usage.prompt_tokens` / prompt-eval timings, whether it exposes **any prefix/KV cache across requests** and **any speculative/MTP knob** (both decide items below), its CPU thread setting, and the PCI-BDF → HIP-index map | ROCm 6.2 unused so far; RADV has been the only path | runs, or fails on a named cause | ½ day + a short lock window | Sonnet (+ owner if a group membership is missing) | a coherent 64-token greedy reply on the ladder's question, `usage` or an equivalent token count reported, VRAM released after exit (sysfs `mem_info_vram_used` back under 1 GB on that card) — **or a named blocker** |
| **H1** | **Extend `context_ladder.py` with `--url`** (reuse `ttft_serve.HttpDriver`, add its `--api-key`, `--model-id`, `--server-log`, `--tools` arguments), plus: residency over the files under `--snap` whatever their extension; the P8 reply pin only in engine mode (an unknown message field may be rejected by another server); in HTTP mode the REUSE-based abort is replaced by a loud `REUSE UNVERIFIED (http)` note and `reused=None` in the row — **the cold sweep decides reuse, not a guess**; `--cold-sweep 2048,4096,8192,16384 --sweep-offset-chars 300000` (fresh single-message prompts of those sizes from a disjoint corpus region, each its own conversation, HTTP mode only; **refused in engine mode with the arithmetic** — 2k+4k+8k+16k on GLM is 260+572+1355+3555 s = 96 min per pass); `side`/`arm` fields in every JSON row; decode from `usage.completion_tokens` when the stream reports it, else delta count, and the row says which. If H0 found an Ollama-style API, a second small driver class with the same `run()` contract. **Plus `context_compare`** (new, Python, under `tools/hot-expert/`): reads the jsonl of the four arms, pairs rows by turn index, prints per depth D_A×2, D_B×2, TTFT_inc×2 each, applies `gate_ab_verdict`'s rule (overlap → NO VERDICT; conservative worst-against-best when separated; refuse < 2 samples per arm) by sourcing `gate_lib.sh`, and prints the reuse ratio r (§2.4) per sweep depth | `HttpDriver` already reads SSE, `usage`, and the gateway log; the ladder hardcodes `EngineDriver` | a driver that is symmetric across arms | 1 day | Sonnet | (i) engine mode reproduces tonight's ladder turn 1–2 within the box's spread (374 tokens 45.4 s / 121.5 ms/token; turn 2 REUSE 381/731 at 110.8 ms/token — `eb4dd5b`), (ii) `--url http://127.0.0.1:8081 --steps 256,256 --gen 16` against the **live** gateway (read-only, no stop, prefill-snapshot style) shows the REUSE lines from `~/glm53_server.log` covering the previous prompt on turn 2 — the HTTP driver is proven on the engine whose reuse signal exists before it meets one whose does not |
| **H2** | **The chain, `franken_chain.sh`**, launched only through `run_chain.sh` (lock; refused while the current benchmark holds it): stop gateway → `wait_no_engine` → warm GLM shards, assert ≥ 90 % → **A1** ladder `--steps 1024,1024,2048,4096,8192 --gen 128 --followups 2` (cumulative user text 1/2/4/8/16 Ki; the engine's own count lands near tonight's 18 055 point) → stop engine, assert VRAM free → **B1** start hipFire, throwaway request, ladder with the same steps **plus** `--steps …,8192,8192,8192,8192,8192,8192` to 64Ki (cheap for it; the 64Ki rows are B-only and labelled so; Colibri's 64Ki is the 9.6 h projection, not run), then the cold sweep → stop hipFire, assert VRAM free → **B2** identical, fresh process → assert VRAM free, re-warm GLM, assert ≥ 90 % → **A2** identical to A1 → **C** (`coli serve` on `qwen36`, `--sizes 512,2048`, 30 min cap) → re-warm GLM → restart gateway on every exit path → `accept_live.sh` (the request *after* the measurement is part of the measurement) | every rule in CLAUDE.md "How a change is measured"; MEASURING.md's factor-of-two on cache state | rig ≈ 70 min (A1) + ~20 (B1) + ~20 (B2) + 70 (A2) + ≤ 30 (C) + ~20 warm/restart ≈ **4 h**; schedule at night, tell the owner the gateway is down for it | Sonnet writes, Haiku runs | the chain exits 0 with four jsonl files whose A rows all carry `majflt=0`, every arm's pre-checks logged (DPM level, VRAM free, residency), `accept_live.sh` PASS at the end; **any A row with majflt > 0 or residency < 90 % invalidates that arm and the chain says so instead of averaging it** |
| **H3** | **The table.** `context_compare` output, plus D_B at empty context from the throwaway (the transfer check, §2.4 Q1), plus arm C's rows, plus the reuse ratio, into the record as §FRANKEN-H2 with the raw jsonl paths | — | one table | ½ day | Haiku | every cell either a measured number with its two samples or `NO VERDICT` / `REFUSED` from `gate_lib.sh`; no cell computed by hand |
| **H4** | **The owner's read (§2.5)**, not a gate: the same 12 prompts through both lanes, tabulated side by side, `finish_reason` and tool-call outcome per row | ninfer's 225-response audit is the pattern: throughput is not usable output | a page the owner reads | ½ day | Haiku | all 12 rows present for both lanes with `finish_reason=stop` counts; no scoring |

### 2.3 Why the ladder and not TG128, and why not a 64Ki cold prefill

TG128 at empty context is where hipFire's 253.3 lives and where nobody's
conversation lives: Open WebUI's tool block alone is ~4 000 tokens, and tonight's
curve loses 1.82× between 2.5k and 18k on GLM (hipFire's own multi-turn figures
lose 1.58× from 253 to 160). The ladder gives decode and incremental TTFT at
every depth in one pass, on the shape Open WebUI sends, and the same corpus on
every arm. A cold 64Ki prefill on GLM is 9.6 h projected and would add nothing:
the rate fit already covers 643–13 628 within 1.5 % and the ladder's 16Ki step
lands on the 18k point measured tonight; the B arm runs to 64Ki because it can.

### 2.4 The four numbers the chain must produce, and their thresholds

Let D_X(d) be decode tok/s at depth d, W_X(d) the incremental TTFT of the
follow-up turn at full depth (the two `--followups`), T_B^cold(d) hipFire's
cold-sweep TTFT, and T_B^lad(d) its ladder TTFT at the same depth.

- **Q1 — do hipFire's numbers transfer to this box?** D_B(≈0) from the
  throwaway. `≥ 150` (0.6 × the published 253.3): proceed. `80–150`: read the
  DPM level recorded (§Q13 found dev0 idling low and `high` changing numbers);
  do not change it mid-chain; a second chain with the level pinned is a
  separate, labelled run. `< 80`: **stop; something on this box is wrong**
  (driver, power, card) and no decision is taken on that number.
- **Q2 — does hipFire reuse prefixes?** r = T_B^lad(16k) / T_B^cold(16k).
  `r ≤ 0.25`: reuse. `r ≥ 0.8`: no reuse — every turn re-prefills.
  In between: NO VERDICT from the ratio; the server's own counters (H0)
  decide or the cell says unknown.
- **Q3 — the owner's wait at depth.** W_B(18k) vs W_A(18k). Projected
  W_A(18k) for a ~50-token follow-up = 50 × (113.9 + 0.01258 × 18 055) ms
  = 50 × 341 ms ≈ **17 s** (the chain measures it; the projection is for the
  reader). Without reuse W_B(18k) = T_B^cold(18k).
- **Q4 — decode at depth.** D_B(18k) vs D_A(18k) = 2.23 measured. The
  published 160 would be 72×. Anything ≥ 10× is the same decision.

### 2.6 The port and the same-weights pair (V1, H2b)

| id | item | evidence | expected | effort | tier | gate |
|---|---|---|---|---|---|---|
| **V1 — DONE 2026-09-16, GATE PASS (record §V1, `perf/v1-qwen36-vk-tier`)** | **A Vulkan expert tier for `qwen36`.** Port the tier `qwen38-vk` already has (heat-ranked preload from a histogram, per-device budget with a reserve, `COLI_VK_DEV2/3`, expert-group issue/take through `backend_vulkan.c`) behind `qwen36_tier.h`'s existing `qt_*` contract, so the engine's 28 call sites do not move; packed int4 experts (`g4/u4/d4`) in VRAM, ~20 GB on one card. **Step 0 (Sonnet, no rig):** read the CUDA tier's kernels for whether they take gs64 group scales; read `qwen38_core.h`'s `q38vk_*` path for the piece to copy; write the step list. **Step 1 (Opus):** tier off = the CPU engine, bit-identical. **Step 2 (Opus):** tier on. The model must first be fetched and converted (the published int4-gs64 container, or `convert_qwen36.py --gs 64` from the BF16 ~70 GB) — NVMe and page-cache work that waits for the lock, like everything else. **Sequenced after X1**, because Q7's measured gaps put a ≤ 28 tok/s ceiling on a 40-layer per-op-submit path (§3, branch 3), and a same-weights comparison taken under that ceiling would measure Colibri's submit model, not its backend | the pattern exists twice in the tree; `docs/qwen36-cuda-tier.md` measured the CUDA version at 9.2–11.3 tok/s on 8 GB cards, 83 % of experts resident on two cards — here all 256 × 40 fit on one | a Colibri-served VRAM-resident 35B with P7/P9/slots intact; tok/s **unknown** until X1 has a number (the ceiling above is the prior) | 1–3 weeks | Sonnet (step 0), Opus (1–2) | tier off: `teacher_forcing` and logits bit-identical to the CPU path; tier on: X2's KL bar against the CPU path, `tworeq.py` at 4 slots IDENTICAL, `[MAP]`/tier line asserts the residency it claims, decode A,B,B,A vs tier-off through `gate_ab_verdict` |
| **H2b** | **Same weights, same placement, two engines.** A′ = Colibri `qwen36` + V1 on int4-gs64, one card (dev3); B = hipFire on the same card, `-mq4r` **and** `-mq5` (the quant nearest a gs64 int4 by bits per weight; report both rows, do not average them). Same chain shape as H2 (A′,B,B,A′; fresh processes; VRAM and DPM asserted; ladder to 18k with `--gen 128`, cold sweep on both sides — affordable now on both). Colibri's arm keeps its REUSE-line contract; hipFire's keeps the sweep ratio | V1 | the first number that compares engines rather than model classes | ~2 h rig | Haiku runs | `context_compare` per depth through `gate_ab_verdict`; **plus** greedy-text agreement between A′ and B on the first 64 tokens of each turn reported as a count (a sanity on two different quantisations, not an oracle) |

Still not comparable in H2b, said plainly: the quantisation (int4-gs64 vs MQ4R
/ MQ5 — different formats, different error), and everything above the kernels
(scheduler, submit model, KV layout). H2b answers "which engine serves this
model better on this card"; only D-3 answers "which backend runs this op
faster". Both are asked; neither is asked of the other's number.

## 3. Decision tree (after H3), with what falsifies each branch

```
H0 fails on a named blocker ──────────────────────────────► O3: stop. Record the cause.
                                                             Vulkan stays. Nothing else changes.
H0 passes
 └─ Q1 < 80 ────────────────────────────────────────────────► O4: diagnose before deciding (driver /
 └─ Q1 ≥ 150 (or 80–150 with the DPM level explained)        DPM / card). No branch is taken on it.
     ├─ Q2 = no reuse AND W_B(18k) > W_A(18k) ───────────────► O2: the moat is real. hipFire wins decode
     │                                                          and loses the wait at depth. Do NOT
     │                                                          replace the lane. Go to branch 3 or wait
     │                                                          for hipFire to add prefix caching; re-run
     │                                                          H2 when it does.
     ├─ Q2 = reuse (or W_B(18k) ≤ W_A(18k)/2) AND Q4 ≥ 10× ──► O1: the fast lane is real. Two follow-ups:
     │                                                          L1 (two-lane cost on GLM) and the owner's
     │                                                          read (H4). Then branch 1 or 2, owner's call.
     └─ anything else (e.g. Q4 between 3× and 10×, W_B ≈ W_A) ► NO VERDICT on the product question.
                                                                Report the table. Fable arbitrates only
                                                                if the owner wants a call on partial data.
```

**Branch 1 — two lanes.** hipFire serves the 35B on dev3 as a second OpenAI
connection in Open WebUI (no gateway change; the daily driver on 8081 is
untouched); GLM-5.3 runs on **two** cards. Cost to GLM, **projected**: dev3
holds 1 695 of the 4 638 tier-resident experts (37 % of the tier); if hits
scaled with tier size the CPU share would go from ~21 % to ~21 + 0.37 × 79 ≈
50 % of routed calls, the CPU expert bucket from ~40 to ~95 ms/token, the
token from ~135 to ~190 ms — **about −29 % decode**. Heat ranking makes the
coldest third of the tier carry fewer hits than its share, so the bracket is
**−15 to −30 %**, and dev0's KDA pool and dense stream are untouched. Item
**L1** measures it: `tools/rome_bench.sh glm53 <name> COLI_VK_EXPERTS3=` with a
config name that says dev3 is skipped (an unset cap skips the device — CLAUDE.md;
verify the tier line says dev3 absent, or the row is invalid), interleaved
A,B,B,A against the three-card config, **and** `prefill_snapshot.sh` for the
TTFT side (both tracks, because the question is what the owner loses on GLM).
Haiku. Falsifier of the branch: L1 > −30 % or the owner rejects the 35B's
answers in H4.

**Branch 2 — replace.** The owner decides after H4 that the 35B is his daily
model; GLM-5.3 becomes on-demand. Then hipFire + Open WebUI is the stack and
the Colibri moat matters exactly as much as Q2 says: with reuse, P7/P9 are
redundant for that lane; without it, branch 2 is O2 and is not taken. This
branch is the owner's, not a measurement's; the plan only makes its cost
visible. Falsifier: O2, or H4.

**Branch 3 — the Colibri-native VRAM-resident lane: that is V1 (§2.6).** The
35B runs on Colibri with P7/P9/slots/gateway intact. Its ceiling before a line
is written, **projected from Q7's measured gaps**: 40 layers × ≥ 3 submits per
layer × ≥ 0.30 ms per gap ≥ **36 ms/token of submit gaps alone → ≤ 28 tok/s**
before any compute, against hipFire's published 160+. That is why graft X1
(retained command buffers) is a *precondition* of V1, not an optimisation of
it, and why V1's step 2 is not started before X1 has a measured number (V1's
step 0 needs no rig and can start now). Branch 3 is taken — V1 becomes the
serving lane rather than a measurement vehicle — when H2b shows
D_A′(18k) ≥ 0.6 × D_B(18k) and W_A′(18k) ≤ W_B(18k); below that, hipFire is
the better engine for this class on this card and V1 remains what makes H2b
and D-3 honest. Falsifier of the whole branch: X1 lands below 2× on the submit
buckets, in which case the ceiling stands and the lane is dead on arithmetic
before V1's step 2 is built.

> **Measured 2026-09-16 (record §X1): the falsifier fires, at 1.00×, and the
> ceiling's derivation needs correcting rather than the lane killing.** The
> 0.30 ms gap is not overhead a retained buffer can remove — §Q7 measured it as
> the engine's own CPU work between submits. Of that gap, the descriptor writes
> and the command re-record are **2.5–3.1 µs** (in-engine `VK_PROF` split),
> i.e. 0.2–0.9 %; the driver's `vkQueueSubmit` is **22–27 µs** and the round trip
> 331–348 µs. So the per-submit floor for a 40-layer, 3-submit-per-layer resident
> path is **120 × ~25 µs ≈ 2.8 ms/token of driver submit alone**, plus whatever
> CPU work sits in the gaps — and **X1 is not the lever that moves it. Fewer
> submits are** (the P2 / G12 / Q3 / Q7 pattern this fork already uses).
> V1's step 2 is therefore **not blocked on X1** and gets no relief from it.

**D-3 — the backend question, answered on its own terms.** One microbenchmark,
same card, same shape: hipFire's gfx11 MMQ int4 kernel vs `qmatmul_tile.spv`
at GLM's routed-expert GEMV shape (M = 1, and the tiled S-row case), each
warmed, each ≥ 5 repeats, interleaved. `tools/hot-expert/rome_vkbench.c` is the
Vulkan side's existing harness. **Migrating any Colibri op to HIP is justified
only if the HIP kernel is ≥ 1.5× on that op AND the op is ≥ 20 % of the token
it lives in.** On GLM the second condition is false by the profile (§0), so
the outcome can at most name a candidate for the *resident* path of branch 3.
Opus, 1 day, one short lock window. Falsifier of "the backend is not the
lever": ≥ 2× on the op — then X-HIP (a knob-gated HIP expert kernel) enters §4
at the bottom, priced by the op's share.

## 4. The graft list, by value per unit of risk

| rank | id | graft | source | what it replaces or adds in Colibri | expected | risk | tier | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | **X1 — DONE 2026-09-16, REJECTED on measurement (record §X1)** | **Retained command buffers for the per-token-invariant stream** — record once, re-submit per token; keep the MoE expert groups dynamic (their set changes every token). Built behind `COLI_VK_RETAIN_CB`, measured, **not merged** — `perf/x1-retained-cmdbuf` stays as the negative result and carries the `VK_PROF` per-submit phase split that killed it | hipFire "Redline" (record the kernel graph, retain invariant command state); CUDA-Graph analogue. Vulkan supports it natively: a command buffer recorded without `ONE_TIME_SUBMIT` is re-submittable | `backend_vulkan.c` resets and re-records on every submit (lines 1007–1009, 1113–1115, 1177–1179, 1268–1270). Targets: Q7's dense stream on qwen38-vk (`dn-proj`, `qsa-proj`, `lm-head`), G12's KDA path on glm53 | **projected**: a fraction of `vk-issue` 13.8 + `vk-take` 4.8 ms/token on qwen38-vk — the CPU-side re-record and validation, not the GPU wait; call it −3 to −8 ms/token (2–6 % of 138) and be pleased to be wrong upward; on glm53 smaller (its GPU stream is thinner). The larger value is strategic: it is the floor-remover for branch 3 | medium: shared file, both engines rebuild and re-measure; descriptor/buffer addresses must be stable across tokens | Opus | bit-identical (same kernels, same order) on `teacher_forcing`, `last_logits`, `tworeq.py` at 4 slots; `rome_bench.sh qwen38-vk` and `glm53` A,B,B,A through `gate_ab_verdict`; `[OPTIME] vk-issue` before/after in the commit body |
| 2 | **X2** | **KL oracle**: per-position logits over a fixed packet (450 rows, the hipEngine size), mean/max KL and top-1 agreement, as a third oracle beside greedy text and last-token cosine | hipEngine's gate methodology (it rejected a 5 % prefill win on this bar) | adds to `ttft_serve.compare_logits` / the `teacher_forcing` path a per-position dump (`GLM53_LOGIT_DUMP` today dumps the last token); a script that prints mean/max KL and top-1 % | a single scale on which G12 (cos 0.99992, text identical), G14's int8 (cos 0.98964, text changed) and G15 (cos 0.878) order themselves; the swiglu-clamp item (§G15's by-product) gets a number instead of "8 of 1232" | low | Sonnet | reproduces those three cases in the same order; refuses (via `gate_compare`) when either dump is empty |
| 3 | **X3** | **QSA block-sparse attention at depth for GLM's MLA** (pool 4 → 1 key, select 512 blocks; dense-identical below 2 052 tokens) — **profile first, then decide** | hipEngine QSA | GLM already has a DSA indexer with cached pooled keys (G5); this is a stricter selection at long context. Step 0: one 18k turn with `[OPTIME] mla split` on, from the A1 engine log — which bucket grows from 2.5k to 18k? | if attention/indexer is ≥ 30 % of the 18k token, a knob-gated selection with X2's bar; if the growth is KV read bandwidth, **dead at step 0** | medium (numerics change; ships off by default, per house rule) | Opus | bit-identical below 2 052 tokens by construction (checked, not assumed); KL within X2's bar at 18k; D_A(18k) A,B,B,A |
| 4 | **X4** | **The audit table** — N fixed prompts, `finish_reason`, tool-call outcome, length, per lane, kept as a page | ninfer's 225-response audit; this fork's own 256-cap incident (every reply truncated for a week and no gate saw it) | extends `accept_live.sh` check 5's idea into a table the owner reads; doubles as H4 | product visibility, not speed | very low | Haiku | all rows present, `finish_reason=stop` count reported, no scoring |
| 5 | **X5** | **MTP re-test, but only in a VRAM-resident regime** | ninfer 59–61 % acceptance on resident models; Q9's kill here (2.668) and hipEngine's 0.955× are host-resident results — the verify block's cost is linear in S when the routed experts come from DRAM per row, and sub-linear when the weights are read once per block from VRAM; that is the discriminator, and it does not transfer either way | nothing in Colibri today. If H0 found a speculative/MTP knob in hipFire: A/B it inside hipFire on the ladder (free). If branch 3 is built: Q9's spec applies to *that* engine, from step 0 | unknown; the point is that Q9's number is not evidence about this regime | low (measurement only) | Sonnet | interleaved knob on/off on the B ladder; `gate_ab_verdict`; text identical or divergences explained as near-ties |
| 6 | **X6** | **Bounded double-buffered pinned row-gather** (stream cold experts into the GPU instead of computing them on the CPU) | hipEngine's PLE streaming | would replace the CPU int4 path for tier misses on GLM | **projected, and it loses at batch 1**: one 14.1 MB expert over PCIe 4.0 x16 at a practical ~25 GB/s (of 32 nominal; not measured here) = 0.56 ms + a launch, vs 14.1 MB from DRAM at the measured 21.95 GB/s = 0.64 ms. Equal within the uncertainty, and the miss set changes every token so nothing amortises. Worth revisiting only for prefill chunks, where Q9 step 1 measured 29 % expert dedup across 32 rows on Qwen | medium | — | not scheduled; the derivation is the record |
| 7 | **X7** | Native artifact format / no load-time conversion | ninfer `.ninfer`, hipFire MQ4R | Colibri already has its own containers and maps shards; load time here is page cache, not conversion | none measurable | — | — | not scheduled |
| 8 | **X8** | Exact-batch decode / C1–C8 batching | ninfer | throughput under concurrency; this box has one user and the gateway serialises | none for the owner's latency | — | — | not scheduled unless the usage changes |

## 5. Multi-GPU: three gfx1100, and why "3" is the wrong question for speed

**M0 — the cheap proof (do it; it is information, not a decision).** Under the
lock with the gateway stopped (RCCL needs VRAM the tiers currently fill):
build rccl-tests against `/opt/rocm-6.2.0` (`make MPI=0 HIP_HOME=/opt/rocm-6.2.0
RCCL_HOME=/opt/rocm-6.2.0`), run `all_reduce_perf -b 8 -e 128M -f 2 -g 3` and
`-g 2`. Record: whether it runs at all; the small-message (8 B – 64 KB) latency
in µs; the large-message bus bandwidth in GB/s; and `NCCL_DEBUG=INFO`'s
transport line (P2P over PCIe vs host-staged). Sonnet, ½ day, a 15-minute lock
window. Outcomes:

- **does not run** ("hostcall not supported", atomics, or KFD access): the
  tonight's-lspci reading was necessary but not sufficient; TP/EP via RCCL is
  off the table on this box; only pipeline placement (no collectives) or one
  card. That is a finding worth the half day.
- **runs**: the small-message latency L is the number that matters for decode
  TP, and here is the arithmetic it is measured against. Tensor-parallel decode
  of the 35B places one all-reduce after attention and one after the MLP per
  layer: 40 layers × 2 = **80 collectives per token**. At 4 ms/token on one card
  (the published 250 tok/s) and compute split perfectly three ways, TP3 breaks
  even only if 4/3 + 80·L < 4 ms, i.e. **L < 33 µs** (projected; the proof step
  supplies L). Over PCIe without a GPU-to-GPU fabric that bar is the whole
  question, and the plan does not assume the answer.
- either way, **3 is an awkward TP degree** (head and expert counts rarely
  divide by 3; hipFire's documented TP/EP runs are 4–5 × gfx1201, never
  gfx1100), and **EP at batch 1 is what Colibri already does** across its three
  devices from the host — same latency structure, no new information.

**M1 — the honest use of the third card.** For a model that fits on one card,
the other two are *capacity*, not speed: a second lane (branch 1), or a larger
resident model across cards by pipeline placement (a model-class change again,
with its own H2). **TP3 for decode of a one-card model is not pursued** unless
M0 returns L well under 33 µs, and even then only as a measured item with the
ladder as its gate.

**M2 — not scheduled:** any Colibri multi-device change on the basis of M0.
Colibri's three-device expert groups are already measured (G2: round-based
issue-all/take-all; G9: CPU work in the gap) and nothing in M0 speaks to them.

## 6. Risks, and what this plan will not do

Risks, each with the check that catches it:

- **ROCm user-space on a RADV box.** HIP talks to `amdgpu` through `/dev/kfd`;
  Mesa through `/dev/dri`. They coexist, but `/dev/kfd` needs group membership
  the user may not have, and hipFire may want a newer ROCm than 6.2. Both are
  H0's first checks and both are owner decisions if they fail. **No system
  package is installed and no group is changed by a session.**
- **VRAM not released after a hipFire exit** would make the A2 arm's preload
  spill and the arm invalid. The chain asserts `mem_info_vram_used` < 1 GB on
  every card before every arm and refuses to proceed otherwise.
- **Page cache.** GLM's 182 GiB plus the MQ4R's 18.7 GB fit in 247 GiB; arm C's
  ~30 GB RSS may evict GLM pages, which is why C runs last and the chain
  re-warms GLM before the restart; the owner's first chat after the chain
  must not pay 100k major faults.
- **Clocks.** §Q13 showed the DPM level moves numbers; the chain records it
  before every arm and changes nothing. A run with a different level is a
  different, labelled run.
- **Two tokenizers.** Depth is matched by turn index and characters; token
  counts are printed beside every cell. A reader who compares tok/s across
  arms without the token counts is comparing two units.
- **Streaming granularity.** A server that batches deltas would misreport
  TTFT and decode; `usage.completion_tokens` is preferred and the row says
  which source it used. If neither exists the cell is REFUSED, not estimated.
- **The daily driver is down for ~4 h.** Once, at night, announced. The chain
  restarts it on every exit path and `accept_live.sh` proves it.
- **The current benchmark.** Nothing here runs until it releases the lock;
  `run_chain.sh` refuses otherwise.

This plan will **not**:

- run anything on the rig, stop the gateway, or take the lock before the
  running benchmark finishes;
- write a HIP kernel into Colibri, or open a backend port, on the strength of
  the head-to-head — only D-3 can put a HIP op on the list, and only for the
  resident path;
- run a 64Ki cold prefill on GLM (9.6 h projected for a number the fit
  already gives), or TP3 for decode;
- declare a quality verdict on the 35B — H4 puts the answers side by side and
  the owner reads them;
- merge `upstream/dev`, or change `~/start_glm53.sh`, the tier caps, the DPM
  level, or the histogram;
- treat any of hipFire's published numbers as this rig's until Q1 has been
  measured here.

## 7. Order of execution and effort

| step | what | tier | rig time | wall |
|---|---|---|---|---|
| H1 | ladder `--url`, cold sweep, `context_compare`; gate against tonight's ladder and the live gateway | Sonnet | 0 (live gateway, read-only) | 1 day |
| H0 | hipFire on the box; API/prefix-cache/MTP facts; device map | Sonnet (+ owner) | one short lock window | ½ day |
| M0 | rccl-tests, `-g 3` and `-g 2` | Sonnet | 15 min under lock | ½ day |
| H2 | the chain, A B B A + C, at night | Haiku runs | ~4 h | — |
| H3 | the table into the record | Haiku | 0 | ½ day |
| H4 / X4 | the audit page, both lanes | Haiku | ~1 h under lock (the B lane) | ½ day |
| D-3 | same-op microbench | Opus | 30 min under lock | 1 day |
| X1 | retained command buffers | Opus | ~3 h (both engines, A,B,B,A) | 3–5 days |
| X2 | KL oracle | Sonnet | ~1 h | 1 day |
| V1 step 0 | gs64-vs-per-row scales in the CUDA tier; the `q38vk_*` piece to copy; step list — **can start now, no rig** | Sonnet | 0 | 1 day |
| model fetch + convert | int4-gs64 container or BF16 → `--gs 64`; after the lock frees, never under a running ladder | Haiku | NVMe/page cache, ~1 h | — |
| V1 steps 1–2 | **DONE 2026-09-16, GATE PASS** — the Vulkan tier on dev3 (record §V1) | Opus | ~35 min of lock across three chains | 1 day |
| H2b | same weights, same card, A′,B,B,A′ with `-mq4r` and `-mq5` | Haiku runs | ~2 h | — |
| L1 | GLM on two cards, only under O1 | Haiku | ~2 h | — |
| X3 step 0 | which bucket grows at 18k (from A1's log; no extra rig time) | Opus | 0 | ½ day |

Fable's part ends here unless H3 lands in the NO VERDICT region and the owner
wants a call on partial data, or D-3 returns ≥ 2× and the backend question
reopens for the resident path.
