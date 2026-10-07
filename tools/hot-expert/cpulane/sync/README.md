# sync: what does the GPU -> host -> GPU synchronisation of a "CPU fourth lane" cost per MoE layer?

Question: a decode step runs ~42 MoE layers; in ~29 of them one expert could be computed on the host while the GPUs do
their own part. Per layer the owner card's single stream does (1) a SIGNAL kernel: plan packet (64 B) + activation x
(16 KB) into pinned host memory, then a sequence number (system-scope release); (2) a host thread on a dedicated core sees
the flag, reads x, computes (here: spins `tc` us), writes y (16 KB) and a sequence number; (3) meanwhile the GPU does its
own work (here: a kernel of `W` us); (4) a WAIT kernel on the same stream polls the result flag (system-scope acquire),
reads y and adds it into a device buffer. The fixed cost **d = (time of the whole layer pattern) - max(W, tc)** is what no
amount of overlap hides. Every 10 us of d costs ~0.3 ms per token (29 round trips); the model assumed d = 60-120 us.

This directory is a *measurement tool*. It was built (docker hipcc, gfx1100, clean) but **never started on a GPU**; the host
logic was exercised against a fake HIP (`make stub`). See "Not verified".

## Files

    bench_sync.hip   kernels + host code (one translation unit)       sync_proto.h  block layout, patterns, atomics, stats
    stub_hip.h       fake HIP for `make stub` (host-only test)        Makefile      make | make stub | make asm | make clean
    run_sync.sh      the script run_chain.sh launches (VRAM guard, docker, checks, headline)

## Build / run

    cd ~/src/cpulane/sync && make            # hipcc in rocm/dev-ubuntu-24.04:7.14.0-full under nice -n 19; compiles only
    ~/src/colibri/tools/hot-expert/run_chain.sh ~/src/cpulane/sync/run_sync.sh      # ~5-6 min, hard limit 690 s
Refuses (rc 3) unless every card's `mem_info_vram_used` < 1 GiB. Output also in `out/<timestamp>/sync.txt`.
Last lines: `SYNC_DONE rc=N` (program), `=== sync exit rc=N` (script). rc 0 ok, 1 data errors / unexpected line counts,
3 refused, 4 build failed, **5 abort** (a bounded wait timed out, a protocol check failed, a HIP error, the watchdog fired).
Env: `SYNC_ARGS` (replaces the default `--budget-s 560`; disables the line-count check), `SYNC_TIMEOUT`, `SYNC_VRAM_LIMIT`
(only to test the refusal). `./bench_sync --help` prints the options, **but starting the binary initialises the GPUs**.
Safe abort-path test with the GPU (orchestrator's call): `SYNC_ARGS="--phases s2 --inject-lane-stall 500 --w-list 300 --tc-list 300"`
makes the lane thread stop answering at iteration 500; the WAIT kernel must time out after 50 ms, the program must print
`SYNC_ABORT ...`, drain, and exit rc 5 within a second.

## Layout of the run (cores are physical cores of the 7F32, first hardware thread of each)

core 3 = lane thread (S1: the ping-pong host thread), core 4 = launch (main) thread, cores 0,1,2 = H2D threads of cards 0,1,2
(S3/S4). The HIP runtime is initialised on the unpinned main thread before any pinning, so its helper threads are not pinned.
The shared block is one `hipHostMalloc(hipHostMallocCoherent | hipHostMallocMapped)` allocation (68 KB) with the device
pointer from `hipHostGetDevicePointer`; every flag has a 128-byte line of its own, x / y / payload buffers are 128-byte
aligned 16 KB each (no unrelated hot data shares a line).

## Phases and result lines

All times in microseconds. `n` = measured iterations (>= 3000 expected in S2; the line carries it, and a `# WARN` is printed if fewer).
`warm` 1 s then `secs` 6 s per configuration (baselines 0.5 + 3 s); iteration counts are capped (40 000 measured).

**S1 ping-pong** (`--phases s1`). ONE persistent 1-workgroup kernel on card 0 (no launches in the loop), the host thread on core 3.
Host: `t0 = rdtsc`, store `s1a = seq` (release); kernel: poll `s1a >= seq` (bounded), optionally read 16 KB, optionally write 16 KB,
fence, store `s1b = seq`; host: spin on `s1b`, `t1 = rdtsc`. Four configs, payload = `0`, `g2h_16K` (kernel writes 16 KB
before the pong), `h2g_16K` (host writes 16 KB before the ping, kernel reads and verifies it), `both_16K`:

    S1,payload=..,n=..,rtt_us_med=..,rtt_us_p99=..,rtt_us_min=..,rtt_us_mean=..,rtt_read_us_med=..,rtt_read_us_p99=..,
       gpu_proc_us_med=..,gpu_proc_us_p99=..,wire_us_med=..,wire_us_p99=..,kernel_status=quit|max_pings,data_err=..
    S1delta,g2h_16K_minus_0_us=..,h2g_16K_minus_0_us=..,both_16K_minus_0_us=..
`rtt_us` = `t1 - t0` (until the flag is seen; the host's 16 KB payload write happens before `t0`, so it is excluded);
`rtt_read_us` additionally includes the host's read+verify of the g2h payload (what a consumer pays). `gpu_proc_us` = device clock
from "ping seen" to "pong store issued" (payload read/write, fences, barriers). `wire_us = rtt - gpu_proc` (per ping): both
transport directions plus the polling quantisation on both sides. **Directions separately: not possible honestly.** The GPU's
`wall_clock64()` (a 100 MHz reference counter on gfx1100) and the host TSC are different clocks with an unknown offset, so no
one-way latency can be derived without assuming symmetry; nothing is timestamped across the link. What is reported instead: the
marginal cost of a payload per direction (`S1delta`: g2h only, h2g only), the device-side share (`gpu_proc`) and the rest (`wire`).
Pings are capped at 1 M per config (if the cap is approached early, measuring starts after 25 % of the cap whatever the warm-up time).

**S2 layer pattern** (`--phases s2`; per work variant, per W in 0/300/600: first the baseline, then tc in 0/300/570).
Per iteration on card 0's stream: SIGNAL kernel, W kernel, WAIT kernel (spin variant). *Work variants*: `work=spin` (a kernel that
busy-waits W us of `wall_clock64`) and `work=mem` (streams a rotating window of a 512 MB device buffer, sized from a measured
streaming bandwidth so that it takes about W us; `S2base` shows what it really takes). The baseline = the same loop with only the
W kernel. Launch pacing: at most ~4 chunks of 8 iterations in flight (an event recorded every 8 iterations, waited 3 chunks later),
so the queue is deep enough to hide launch latency like a real engine's, but bounded.

    S2base,work=..,W=..,n=..,base_us=..,ev_us=..,it_us_med=..,it_us_p99=..
    S2,wait=spin,work=..,W=..,tc=..,n=..,pattern_us=..,ev_us=..,it_us_med=..,it_us_p99=..,base_us=..,d_us=..,d_ev_us=..,
       d_vs_base_us=..,spin_us_med=..,spin_us_p99=..,spin_us_max=..,polls_med=..,polls_p99=..,devrt_us_med=..,devrt_us_p99=..,
       lane_proc_us_med=..,lane_proc_us_p99=..,timeouts=..,data_err=..,lane_data_err=..
`pattern_us` = host-measured total / N: from the moment the window-start event was seen complete to the moment the window-end event
was (a final bounded stream drain), divided by the iterations in between. `ev_us` = the same from `hipEventElapsedTime` of the two
events (device timestamps); `it_us_med/p99` = per-iteration time from device clock stamps at the start of consecutive SIGNAL kernels
(base: W kernels). **d_us = pattern_us - max(W, tc)** (the definition in the task). **d_vs_base_us = pattern_us - max(base_us, tc)**:
the same with the *measured* W-kernel-only time instead of the nominal W, which removes the W kernel's own launch/start gap and,
for `work=mem`, the difference between nominal and real W. For `work=mem` read d_vs_base_us, not d_us (when the buffer is smaller
than W's worth of bandwidth a `# WARN` says so). `spin_us_*`, `polls_*` = how long (device clock) and how many polls the WAIT kernel
spent in its poll loop: the unhidden wait (about 0 when W > tc, about tc - W otherwise, plus the sync latency). `devrt_us` = device
clock from the SIGNAL flag store to the WAIT kernel leaving its loop (includes W). `lane_proc_us` = lane thread, signal seen ->
result posted (x read + check, tc spin, y write; excludes waiting for the signal).

**S2b** (`--phases s2b`): the same pattern (work = spin only) with the wait done by `hipStreamWaitValue32(stream, &res, (uint32_t)seq,
hipStreamWaitValueGte, 0xFFFFFFFF)` followed by a consume kernel (no polling; same read/verify/add). Lines `S2b,wait=streamwait,...`
as S2. Before use a bounded two-step test: attribute `hipDeviceAttributeCanUseStreamWaitValue`; an already satisfied wait must
complete within 2 s; an unsatisfied wait must still be blocked after 20 ms and complete within 2 s of one host flag write. Any
failure prints `S2b,unsupported,reason=...` and S2b (and the stream-wait half of S3/S4) is skipped; the flag is reset afterwards.

**S3 / S4 under H2D load** (`--phases s3,s4`): the S2 spin pattern at W=600, tc=570 (spin-kernel wait, and the stream-wait variant if
S2b works) while the part-B pattern of `bench_h2d_cpu.hip` runs: per card one host thread (cores 0,1,2) doing blocking
`hipMemcpyAsync` + `hipStreamSynchronize` of 11 665 408 B pieces from a 1 GB pinned buffer (`hipHostMallocPortable`) into one VRAM
piece, on a stream of its own. S3: cards 0, 1, 2 (card 0 = the waiting card). S4: cards 1 and 2 only (card 0 quiet; "the three
streams on cards 1 and 2" was read as one stream per card, 2 threads). First the streams alone for the same windows (reference),
then with the pattern:

    S3ref,card0_GBps=..,card1_GBps=..,card2_GBps=..          S4ref,card1_GBps=..,card2_GBps=..
    S3,wait=..,work=spin,W=600,tc=570,...(as S2)...,card0_GBps=..,card0_vs_ref=..,card1_GBps=..,card1_vs_ref=..,...,d_quiet_us=..,d_delta_us=..
`d_quiet_us` = d of the same config in S2 (no load, same run; `nan` if S2 did not run), `d_delta_us = d - d_quiet`: the cost of the load.
`cardN_vs_ref` = the stream's rate with the pattern running against the reference rate: the cost of the pattern to the H2D stream.
Comparing S3 with S4 separates "the stream on the waiting card's own link" from "the system (DDR, host) under load".
S3/S4 H2D rates are sampled at the same two instants as `pattern_us` (window start/end events seen complete).

## How to read it

* The number to compare with the model is `d_vs_base_us` (and `d_us`) of `S2,wait=spin,work=spin,W=600,tc=570` (all hidden but the
  sync) and `W=0,tc=0` (the bare round trip incl. the SIGNAL/WAIT kernels). Multiply by 29 for the per-token cost.
* d for W >= tc (the CPU finishes first) is signal-kernel + wait-kernel + extra launches + one poll. For tc > W it additionally
  contains the round-trip latency of the result flag (`spin_us`).
* `S1 rtt` is the floor of any single handshake; S2's d should be a few times it (two kernels, fences, launches).
* `ev_us` vs `pattern_us` disagreeing by more than the event overhead means the host window is skewed (look at `it_us_p99`).
* The events used for pacing and for the window edges are not free (~a few us per recorded event, one per 8 iterations): they are
  in the baseline too, so d_vs_base cancels them; d_us does not.

## Bounded waits (nothing may hang the card)

* Device: every poll loop (WAIT kernel, ping-pong kernel, poll-rate probe) stops after `--wait-ms` (50 ms of `wall_clock64`, 500 ms for
  the first S1 ping which includes the host thread's start) **or** a poll-count cap. The cap is derived: a 2 ms probe kernel runs the very
  same loop on an idle flag, the cap is 2x the polls that fit in the time bound (`--max-polls N` overrides), so the clock fires first
  unless the clock is broken. The clock is read every 16th poll only (on gfx11 it is a scalar message round trip). A timed-out kernel
  increments a device counter, stores 1 into the host-visible `dev_timeout` flag and exits without reading the result.
  The W kernels are bounded by construction (fixed ticks, plus a 2^28-iteration cap); the mem kernel is a fixed read.
* Host: every spin has a deadline (lane thread / S1 host: `--host-deadline-s` 1 s, first S1 pong 3 s); every wait for a stream (events:
  pacing, window edges, final drain, copies, calibration kernels, the S1 persistent kernel) is a `hipEventQuery` poll with a deadline
  (`--stream-deadline-s` 10 s, 5 s after a quit). The main loop checks `dev_timeout` every iteration.
* Abort: `abort_run()` prints `SYNC_ABORT reason=.. phase=..`, then writes `kAbortVal` (0xFFFFFFFF, above every real sequence number) into
  `res` and `s1a` and 1 into `cmd`: every device wait, including an unbounded `hipStreamWaitValue32`, is satisfied. It only writes host
  memory (no HIP call), so any thread can call it. The lane thread stores `res` with a monotonic max, so a racing store cannot undo the
  release. After an abort the run drains the stream (<= 5 s, re-releasing), joins the threads, prints `SYNC_DONE rc=5`, `_Exit(5)`.
* Watchdog thread (no HIP calls): if the heartbeat (lane iterations, pings, launched iterations) stands still for 3 s while a phase is
  active it calls `abort_run`; 25 s after an abort the process is forced out. It is what releases an S2b wait if the main thread were
  stuck in a HIP call.
* Time budget `--budget-s 560`: remaining configs are skipped with `S_SKIP` (the script then reports unexpected line counts, rc 1).

## Protocol correctness

Sequence numbers: process-global counters (`g_seq1` for s1a/s1b, `g_seq2` for sig/res) that only ever advance; a config uses
base+1..base+n and leaves the counter at base+n (S1: base+n+1, the quit value). A flag left behind by an earlier config is therefore
always below every later wait target and cannot satisfy it (the S2b test spends base+1..base+1000). Waits use `>=`. Each flag has one
writer. Iteration 0: the lane thread starts expecting `base+1` before the first launch; the first SIGNAL carries `base+1`. Last
iteration: the stream drains (`ev_b`), then the lane has served exactly the launched number (checked, else abort), then it is
stopped. The lane requires `sig == expected` exactly (the GPU cannot be ahead: the WAIT kernel of iteration k blocks the stream
until `res >= k`). Data: x, plan, y and the S1 payloads are patterns of the sequence number (`pat64`, `y_base + j`) written before the
flag and verified in full after it by the other side (device: `data_err`, host: `lane_data_err`); a mismatch means a flag was seen
before its data and sets rc 1. GPU stores to host memory: `__hip_atomic_store(.., RELAXED, SYSTEM)` for data, `__threadfence_system()` in every
thread, `__syncthreads()`, then thread 0 `__hip_atomic_store(.., RELEASE, SYSTEM)`; polling `__hip_atomic_load(.., ACQUIRE, SYSTEM)` (or
relaxed loads + one acquire fence with `--poll-relaxed`), `__builtin_amdgcn_s_sleep(1)` between polls, then `__threadfence_system()`
in every thread before reading the payload with relaxed system-scope loads. Host: `std::atomic<uint64_t>` acquire/release on the shared
words, data before flag, `_mm_pause()` in spins. `obj/bench_sync_gfx1100.s` (`make asm`) was read: the acquire loads are `global_load_b64 .. glc` +
`s_waitcnt vmcnt(0)` + `buffer_gl1_inv/gl0_inv`, the release path waits `s_waitcnt_vscnt 0` before the flag store, `wall_clock64()` is
`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME`, the polling loop has `s_sleep 1`.

## Assumptions

* Card 0 = HIP device 0. The 7F32's TSC is invariant (calibrated against steady_clock over 100 ms at start; used for all host timing).
* `wall_clock64` ticks at `hipDeviceAttributeWallClockRate` (printed); checked at start with 1000 us and 5000 us kernels timed by events
  (the rate is corrected if it is off by more than 20 %, with a WARN).
* hipHostMalloc(Coherent|Mapped) memory is cacheable write-back on the CPU side and fine-grained from the GPU (believed, not checked here:
  KFD maps coherent host memory with MTYPE_UC on gfx11, so GPU stores are written through and GPU loads are not served from a stale
  L2 line). If the runtime did not give that, the data check (not the latency) would catch it.
* The lane thread reads and checks all of x (16 KB) and writes all of y (16 KB) every iteration (~1-3 us at tc = 0): that is part of d
  by design (the real lane has to read x and produce y); the `tc` spin starts after x was read.
* The CPU expert is emulated by a TSC busy-wait; no real GEMV runs here (the DDR/cache pressure of a real expert is part B's job).
* `work=mem` bandwidth is measured with a 256 MB-per-pass kernel and 4 independent 16-byte loads per thread; MALL (96 MB) is defeated
  by the 512 MB buffer; the window rotates so consecutive layers read different lines.

## Not verified (everything that needs the GPU)

Whether `hipHostMalloc(Coherent|Mapped)` + system-scope atomics deliver sub-10 us handshakes on this stack (the numbers are the point of the
run); whether `hipStreamWaitValue32` accepts hipHostMalloc memory (the run tests it first and skips S2b if not); the real poll period of a system-scope
acquire load (printed by the probe); the tick rate of `wall_clock64` (printed and checked); that a 6 s persistent kernel (S1) is
tolerated by the compute queue (it is the same shape as an engine's persistent kernels, but was never run here); the `--poll-relaxed`
variant; the stub proves the host logic (sequence numbers, drain, window arithmetic, statistics, abort/watchdog/S2b-failure paths:
`--inject-lane-stall`, `--stub-nosig-seq`, `--stub-wv 1|2|3`) but its timings mean nothing. The x86 stub run on the rig exercised the TSC calibration,
pinning and topology code (no GPU touched).

## Known limitations / risks

* `hipStreamWaitValue32` has no hardware timeout; it is made safe only by the release path (abort value / watchdog) and by the support
  test. A process killed (SIGKILL) while such a wait is outstanding would leave the GPU waiting on freed host memory; the program never
  has one outstanding without a live lane thread and watchdog, and the script's `timeout` (690 s) is above the program's 560 s budget.
* A host stall > 50 ms of the lane thread (descheduling) makes a WAIT kernel time out: the run aborts instead of continuing (by design).
* `rtt_us` of S1 contains the polling quantisation of both sides (the kernel polls with `s_sleep 1` between system-scope loads).
* `d_us` for `work=mem` is not meaningful where the nominal W was not reached (see `d_vs_base_us`).
* `--hip-link` "argument unused" notice from `make asm` is hipcc's own flag with `-S`; the normal build prints nothing.
