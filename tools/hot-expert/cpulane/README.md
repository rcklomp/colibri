# cpulane: can 5-6 CPU cores compute missed experts while three GPUs stream over PCIe?

Question: of the experts that miss VRAM (avg 3.4 of 8 per layer, 11.67 MB each), could the CPU compute some
straight from host RAM (~20 GB/s of weights, one expert in ~0.5-0.6 ms) *while* the three H2D streams run? The
CPU GEMV and the DMA reads share the DDR controllers, so part B measures the concurrent pair (FreeToken,
arXiv 2608.16157), not two independent bandwidths.

## What is measured

**Part A, `bench_cpu_gemv` (CPU only, host).** Batch-1 quantised GEMV with ggml's own x86 AVX2 kernels
(`ggml_vec_dot_{iq4_xs,iq3_s,iq2_s,q6_K}_q8_K`, nrc = 1, activation quantised to Q8_K first, as llama.cpp does).
Weight GB/s against 1..8 threads, each on a distinct physical core (`sched_setaffinity`, first hardware thread
of each core; order 3,4,5,6,7,0,1,2 so threads 1..5 sit on the cores part B gives the workers).
- `--mode whole`: every thread takes whole experts (task = gate + up + activation + down). `expert_ms` = median
  time of one task on one thread while the others run too.
- `--mode rows`: one expert at a time, rows split over all threads (gate|up rows, spin barrier, down rows, spin
  barrier). `expert_ms` = median time from dispatch (start barrier) to completion (end barrier), >= 200 experts
  (also >= 0.6 s). This is the latency of ONE expert across N threads.
- `cache`: ~2 MB piece re-read (whole: one private piece per thread; rows: one shared piece) = compute-bound
  ceiling. `piece_MB` is ~2, so its `expert_ms` is for a 2 MB mini-expert and is NOT comparable with `dram`.
- `dram`: 4 GB region cut into expert-sized pieces visited in random-permutation order (a piece comes back only
  after all others), each piece read once per visit. This is the real access pattern.
- types: `iq4_xs`, `iq3_s`, `iq2_s`, `q6_k` (pure expert, rows scaled so it is 11.67 MB), `glm` (the expert of the
  old IQ4_XS model: IQ3_S gate/up + IQ4_XS down, 11.67 MB) and `hyb` (the SERVED IQ3_XXS hybrid's routed expert:
  IQ2_S gate/up + IQ3_S down, at its real size 8 978 432 B = 8.98 MB, NOT scaled; in `cache` mode it is scaled to ~2 MB
  like every type). The default `--types` stays `iq4_xs,glm,iq3_s,q6_k`; `hyb` and `iq2_s` run only on request.

**Part B, `bench_h2d_cpu` (HIP, docker; STARTING IT INITIALISES THE GPUs).** Per card one host thread issues blocking
`hipMemcpyAsync` + `hipStreamSynchronize` of 11 665 408 B pieces, cycling through a 2 GB `hipHostMalloc(Portable)`
buffer into one VRAM piece (`hipSetDevice` at every thread entry and before every copy). H2D threads sit on
physical cores 0,1,2. Phases (each 2 s warm-up + 8 s measured, byte counters sampled at the window edges):
B1 each card alone; B2 all three; B4 N CPU workers alone (control, same process); B3 all three + N workers, for
N = 2, 4, 6 and for qtype `iq4_xs` (as specified) and `glm` (the real mix). Workers run the part-A whole-expert
GEMV (same `cpu_gemv.h`) on a separate 4 GB plain (mmap + THP advice) region, on physical cores 3..7.
**N = 6 does not fit on distinct physical cores** (only 5 are left): worker 6 goes on the SMT sibling of core 0
(`smt_shared=1` in the line); with blocking vs spinning stream sync that sibling is not idle. `--blocking-sync`
and `--cpu-pinned` (CPU region from `hipHostMalloc`) are available for follow-ups.

## Build (compile only, never starts a binary)

    cd ~/src/cpulane && make          # or: make a   /   make b ;  make clean
`bench_cpu_gemv`: host g++ -O3 -mavx2 -mfma -mf16c. `bench_h2d_cpu`: `hipcc --offload-arch=gfx1100 -O3 -std=c++17
-mavx2 -mfma -mf16c` in `rocm/dev-ubuntu-24.04:7.14.0-full` (franken-engine's recipe, `nice -n 19`, `-v /home/ronald`).
The three ggml objects (`ggml-quants.c`, `ggml-cpu/arch/x86/quants.c`, `ggml_stubs.c`) are built with docker's
gcc 13 (`-O3 -mavx2 -mfma -mf16c -mtune=znver2`) and linked into both programs, so both run the same kernel code.

## Run

    ~/src/cpulane/run_cpulane.sh      # started by run_chain.sh under the rig lock; ~8-10 min
Refuses (rc 3) unless every card's `mem_info_vram_used` < 1 GiB; runs A, then B in docker (flags copied from
`~/bench/glm_adapt_chain.sh`), prints `B3vsA`/`B4vsA` lines, ends with `=== cpulane exit rc=N`. Raw output is also
saved to `out/<timestamp>/{A,B,vsA}.txt`. Extra program arguments: `CPULANE_A_ARGS`, `CPULANE_B_ARGS`.

## Reading the result lines

    A,iq4_xs,rows,threads=5,dram,GBps=..,expert_ms=..,expert_ms_p90=..,experts_per_s=..,piece_MB=..,n=..,cpus=3+4+5+6+7
    B1,card=0,GBps=..
    B2,card0_GBps=..,card1_GBps=..,card2_GBps=..,agg_GBps=..,sum_B1_GBps=..,agg_vs_sum_B1=..,h2d_exp_per_s=..
    B4,N=4,cpu_GBps=..,cpu_exp_per_s=..,qtype=..,cpus=..,smt_shared=0
    B3,N=4,card0_GBps=..,card1_GBps=..,card2_GBps=..,agg_GBps=..,cpu_GBps=..,qtype=..,cpus=..,smt_shared=..,
       agg_vs_B2=..,cardX_vs_B2=..,cpu_vs_B4=..,h2d_exp_per_s=..,cpu_exp_per_s=..,total_exp_per_s=..,total_vs_B2=..
    B3vsA,N=4,qtype=iq4_xs,cpu_GBps=..,A_whole_dram_GBps=..,ratio=..
The answer is in `total_vs_B2` (experts/s of cards + CPU against the cards alone; > 1 means the CPU lane adds
throughput net of the DDR contention), `agg_vs_B2` (what the CPU does to the H2D streams), and `cpu_vs_B4`
(what the H2D streams do to the CPU). One "expert-equivalent" = 11.67 MB of weights moved or read.

## Assumptions

- `hyb` shapes read from the GGUF header of `~/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/...-00001-of-00001.gguf`
  (2026-10-13, `ckpt1006/gguf_tensor_types.py` parser): `blk.N.ffn_gate_exps` / `ffn_up_exps` = ne [4096, 2048, 288]
  IQ2_S (2048 rows x K 4096, 16 blocks x 82 B = 1312 B/row = 2 686 976 B per expert each), `ffn_down_exps` = ne
  [2048, 4096, 288] IQ3_S (4096 rows x K 2048, 8 x 110 = 880 B/row = 3 604 480 B): 8 978 432 B. This holds for layers
  3..43 except 11, 12 (39 layers). Not modelled: layer 11 (IQ3_S/IQ3_S/IQ4_XS = the `glm` expert, 11.67 MB), layers 12
  and 44 (IQ2_S/IQ2_S/IQ4_XS, 9.83 MB), layer 45 (NextN, Q3_K/Q3_K/Q4_K, unused by the text tower); layers 0-2 are
  dense. No Q6_K anywhere in the hybrid's experts.
- Shapes of `glm` read from `~/src/franken-engine/franken/decode/glm5_shapes.h` and `GLM5.md` section 4: N_EMBD 4096,
  N_FF_EXP 2048, 288 experts, 8 used; gate and up 2048 rows x K 4096 (IQ3_S, 1760 B/row = 3.60 MB each), down
  4096 rows x K 2048 (IQ4_XS, 1088 B/row = 4.46 MB): 11 665 408 B. Layer 11 (15.79 MB) and layers 12, 44 (Q6_K
  down, 14.09 MB) are not modelled for `glm`. Pure-type experts keep K and the 1 : 1 : 2 row ratio, rows scaled to ~11.67 MB.
- Weights are random bytes shaped like valid blocks with a finite fp16 scale (`d` in [2^-8, 2^-7)); outputs are
  checked finite. Timing does not depend on values. Every bit pattern is a valid payload for all four formats
  (read from `ggml_vec_dot_iq2_s_q8_K`): the IQ2_S grid index is `qs[i] | (qh bits << 8 & 0x300)` <= 1023 into the
  1024-entry `iq2s_grid`, the 32 sign bytes and the 4-bit scales take any value, and the grid tables are `static
  const` in `ggml-common.h` (no `iq2xs_init_impl`, which is quantiser-only). The random grid indices touch the 8 KB
  `iq2s_grid` uniformly, real weights concentrate on fewer entries: an L1 effect, not a DRAM one.
- ggml: `~/src/llama-glm53` @ 39931761a (the checkout franken-engine links); the x86 quant kernels are byte-identical in
  `~/src/llama.cpp` and `~/src/llama-latest`. Read only, built in this directory.
- Per task: gate and up share a pre-quantised Q8_K input (not timed); between up and down `h = clamp(g*u)` stands
  in for SwiGLU (no expf) and is quantised to Q8_K inside the timed task (a few us). Whole-expert tasks use
  `ggml_vec_dot` per row, no explicit prefetch (hardware prefetcher only), as llama.cpp's mul_mat does at batch 1.
- Spin barriers and pre-spun workers: the rows-mode latency excludes thread wake-up, which an engine pays unless it
  also keeps its workers spinning.
- The rig's CPU governor/boost is not pinned; results are at whatever clock the 7F32 runs under this load.

## Not verified

Part B was compiled, never started (no GPU run was allowed). Its fake-HIP logic (phases, counters, line format,
thread start/stop) was exercised on a Mac with stub HIP/ggml, which says nothing about real rates. Part A's full
matrix (128 configs, estimated ~2.5 min) was not run either, only a 1-thread smoke: IQ4_XS 10.9 GB/s cache and
10.3 GB/s DRAM, Q6_K 21.0, IQ3_S 2.6, glm mix 3.6 GB/s cache-resident per core.
