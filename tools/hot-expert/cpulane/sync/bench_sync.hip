// bench_sync.hip -- what does the GPU -> host -> GPU synchronisation of a "CPU fourth lane" cost per MoE layer?
// See README.md. STARTING THIS PROGRAM INITIALISES THE GPUs: run it only through run_sync.sh (rig lock held, VRAM idle).
//
// Phases (--phases s1,s2,s2b,s3,s4):
//   S1   persistent-kernel ping-pong latency, no launches in the loop             S1,payload=..,rtt_us_med=..
//   S2   layer pattern SIGNAL kernel -> W kernel || host lane (tc) -> WAIT kernel  S2base,.. / S2,wait=spin,..
//   S2b  the same with hipStreamWaitValue32 instead of the spin kernel             S2b,wait=streamwait,.. | S2b,unsupported
//   S3   S2 (W=600, tc=570) while all three cards stream host->device              S3,.. S3ref,..
//   S4   same, streams on cards 1 and 2 only (card 0 quiet)                        S4,.. S4ref,..
// Every wait is bounded: kernels poll for at most --wait-ms (wall clock) / --max-polls, host spins have a deadline,
// every kernel launch is followed by a bounded wait for completion; any timeout calls abort_run(), which prints
// SYNC_ABORT, releases every flag (kAbortVal) so no GPU wait can outlive it, and ends the run with rc 5.
#ifdef SYNC_STUB
#include "stub_hip.h"
#else
#include <hip/hip_runtime.h>
#endif

#include "sync_proto.h"

#include <set>

using namespace sy;

#define ULL(x) ((unsigned long long)(x))

// ====================================================================== globals / abort infrastructure
struct alignas(128) Prog { std::atomic<uint64_t> v{0}; char pad[120]; };
static Prog g_prog;                         // heartbeat for the watchdog (lane iterations, pings, launched iterations)
static std::atomic<int>  g_abort{0};
static std::atomic<int>  g_active{0};       // 1 while a bounded GPU phase runs (arms the watchdog)
static std::atomic<int>  g_wd_stop{0};
static SyncBlock *       g_sbh = nullptr;   // host view of the block, for release_all()
static uint64_t          g_seq1 = 0, g_seq2 = 0;   // sequence counters: s1a/s1b, sig/res (see sync_proto.h)
static int               g_rc = 0;          // soft failures (data errors): rc 1
static std::mutex        g_phase_mu;
static std::string       g_phase = "init";
static int64_t           g_t_prog_start = 0;

static void set_phase(const char * s) { std::lock_guard<std::mutex> lk(g_phase_mu); g_phase = s; }
static std::string get_phase() { std::lock_guard<std::mutex> lk(g_phase_mu); return g_phase; }

// Writes the abort value into every flag a GPU wait could be spinning on. Host-memory writes only: no HIP call,
// callable from any thread (watchdog included).
static void release_all() {
    SyncBlock * sb = g_sbh;
    if (!sb) return;
    st_rel(&sb->cmd.v, 1);
    st_rel(&sb->res.v, kAbortVal);
    st_rel(&sb->s1a.v, kAbortVal);
}
static void abort_run(const char * fmt, ...) {
    int expected = 0;
    if (g_abort.compare_exchange_strong(expected, 1)) {
        char msg[400];
        va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
        printf("SYNC_ABORT reason=%s phase=%s\n", msg, get_phase().c_str());
        fflush(stdout);
    }
    release_all();
}
[[noreturn]] static void fatal(int rc, const char * fmt, ...) {
    char msg[400];
    va_list ap; va_start(ap, fmt); vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    abort_run("fatal: %s", msg);
    printf("SYNC_DONE rc=%d\n", rc);
    fflush(stdout); fflush(stderr);
    std::_Exit(rc);
}
#define HIP_OK(call)                                                                                              \
    do {                                                                                                          \
        hipError_t e_ = (call);                                                                                   \
        if (e_ != hipSuccess) fatal(2, "HIP %s failed at %s:%d: %s", #call, __FILE__, __LINE__, hipGetErrorString(e_)); \
    } while (0)

// watchdog: no HIP calls, host memory only. If the heartbeat does not move for 3 s while a phase is active it aborts
// (which releases every GPU wait); if the process is still alive 25 s after an abort it forces the exit.
static void watchdog_main() {
    uint64_t last = g_prog.v.load(); int64_t t_last = now_ns(), t_abort = 0;
    while (!g_wd_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int64_t t = now_ns();
        if (g_abort.load()) {
            if (!t_abort) t_abort = t;
            if (t - t_abort > 25000000000ll) {
                printf("SYNC_ABORT_HARD no clean exit 25 s after the abort, forcing exit\nSYNC_DONE rc=5\n");
                fflush(stdout); std::_Exit(5);
            }
            continue;
        }
        const uint64_t p = g_prog.v.load();
        if (!g_active.load() || p != last) { last = p; t_last = t; continue; }
        if (t - t_last > 3000000000ll) abort_run("watchdog: no progress for 3 s");
    }
}

// ====================================================================== options
struct Opts {
    std::string phases = "s1,s2,s2b,s3,s4";
    double secs = 6.0, warm = 1.0;               // measured window and warm-up of every pattern / ping-pong config
    double secs_base = 3.0, warm_base = 0.5;     // the baseline (W kernel only) runs shorter
    std::vector<int> W{0, 300, 600};             // synthetic GPU work, us
    std::vector<int> TC{0, 300, 570};            // emulated CPU expert time, us
    std::vector<std::string> s1_modes{"none", "g2h", "h2g", "both"};
    int cards = 3, lane_core = 3, main_core = 4;
    std::vector<int> h2d_cores{0, 1, 2};
    std::vector<int> s4_cards{1, 2};
    double h2d_host_gb = 1.0;
    int max_iters = 40000;                       // measured iterations per pattern config (cap)
    int poll_mode = 0;                           // 0: acquire load on every poll (as specified); 1: relaxed polls + one acquire fence
    double wait_ms = 50, first_wait_ms = 500;    // device wait bound (wall clock); the first S1 ping gets the longer bound
    double host_deadline_s = 1.0, stream_deadline_s = 10.0;
    unsigned max_polls = 0;                      // device poll-count cap; 0 = derive: 2x the polls that fit in --wait-ms, measured by the poll-rate kernel
    double budget_s = 600;                       // configs are skipped once this much time has passed
    int mem_mb = 512;
    bool no_mem = false;
    int inject_lane_stall = 0;                   // test hook: the lane thread stops answering at this iteration
    int stub_nosig_seq = 0, stub_wv = 0;         // stub-only test hooks
};
static Opts g_o;

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v; size_t i = 0;
    while (i <= s.size()) { size_t j = s.find(sep, i); if (j == std::string::npos) j = s.size(); if (j > i) v.push_back(s.substr(i, j - i)); i = j + 1; }
    return v;
}
static bool parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char * name) -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); } return argv[++i]; };
        if      (a == "--phases")        o.phases = val("--phases");
        else if (a == "--secs")          o.secs = atof(val("--secs"));
        else if (a == "--warm")          o.warm = atof(val("--warm"));
        else if (a == "--secs-base")     o.secs_base = atof(val("--secs-base"));
        else if (a == "--warm-base")     o.warm_base = atof(val("--warm-base"));
        else if (a == "--w-list")        o.W = parse_int_list(val("--w-list"));
        else if (a == "--tc-list")       o.TC = parse_int_list(val("--tc-list"));
        else if (a == "--s1-modes")      o.s1_modes = split(val("--s1-modes"), ',');
        else if (a == "--cards")         o.cards = atoi(val("--cards"));
        else if (a == "--lane-core")     o.lane_core = atoi(val("--lane-core"));
        else if (a == "--main-core")     o.main_core = atoi(val("--main-core"));
        else if (a == "--h2d-cores")     o.h2d_cores = parse_int_list(val("--h2d-cores"));
        else if (a == "--s4-cards")      o.s4_cards = parse_int_list(val("--s4-cards"));
        else if (a == "--h2d-host-gb")   o.h2d_host_gb = atof(val("--h2d-host-gb"));
        else if (a == "--max-iters")     o.max_iters = atoi(val("--max-iters"));
        else if (a == "--poll-relaxed")  o.poll_mode = 1;
        else if (a == "--wait-ms")       o.wait_ms = atof(val("--wait-ms"));
        else if (a == "--first-wait-ms") o.first_wait_ms = atof(val("--first-wait-ms"));
        else if (a == "--host-deadline-s")   o.host_deadline_s = atof(val("--host-deadline-s"));
        else if (a == "--stream-deadline-s") o.stream_deadline_s = atof(val("--stream-deadline-s"));
        else if (a == "--max-polls")     o.max_polls = (unsigned)atoll(val("--max-polls"));
        else if (a == "--budget-s")      o.budget_s = atof(val("--budget-s"));
        else if (a == "--mem-mb")        o.mem_mb = atoi(val("--mem-mb"));
        else if (a == "--no-mem")        o.no_mem = true;
        else if (a == "--inject-lane-stall") o.inject_lane_stall = atoi(val("--inject-lane-stall"));
        else if (a == "--stub-nosig-seq")    o.stub_nosig_seq = atoi(val("--stub-nosig-seq"));
        else if (a == "--stub-wv")           o.stub_wv = atoi(val("--stub-wv"));
        else {
            fprintf(stderr,
                "usage: %s [--phases s1,s2,s2b,s3,s4] [--secs 6] [--warm 1] [--secs-base 3] [--warm-base 0.5]\n"
                "          [--w-list 0,300,600] [--tc-list 0,300,570] [--s1-modes none,g2h,h2g,both] [--cards 3]\n"
                "          [--lane-core 3] [--main-core 4] [--h2d-cores 0,1,2] [--s4-cards 1,2] [--h2d-host-gb 1]\n"
                "          [--max-iters 40000] [--poll-relaxed] [--wait-ms 50] [--first-wait-ms 500] [--max-polls 0=auto]\n"
                "          [--host-deadline-s 1] [--stream-deadline-s 10] [--budget-s 600] [--mem-mb 512] [--no-mem]\n"
                "          [--inject-lane-stall N]   (test hook: the lane stops answering at iteration N -> exercises the abort path)\n"
                "NOTE: this initialises the GPUs; run it through run_sync.sh only.\n", argv[0]);
            return false;
        }
    }
    return true;
}

// ====================================================================== device side
// "DEV" primitives: real device intrinsics on the GPU, host atomics in the stub. The bounded poll is written once.
#ifdef SYNC_STUB
#define DEVFN static inline
static inline uint64_t d_ld_acq(const uint64_t * p) { return ld_acq(p); }
static inline uint64_t d_ld_rlx(const uint64_t * p) { return ld_rlx(p); }
static inline void     d_st_rel(uint64_t * p, uint64_t v) { st_rel(p, v); }
static inline void     d_st_rlx(uint64_t * p, uint64_t v) { st_rlx(p, v); }
static inline uint64_t d_clock() { return (uint64_t)(now_ns() / 10); }     // stub wall clock: 100 ticks per us
static inline void     d_nap() { cpu_relax(); }
#else
#define DEVFN __device__ __forceinline__
// GPU stores/loads of host memory: system-scope atomics (write-through / cache-bypassing on gfx11), see README
DEVFN uint64_t d_ld_acq(const uint64_t * p) { return __hip_atomic_load(const_cast<uint64_t *>(p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
DEVFN uint64_t d_ld_rlx(const uint64_t * p) { return __hip_atomic_load(const_cast<uint64_t *>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
DEVFN void     d_st_rel(uint64_t * p, uint64_t v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
DEVFN void     d_st_rlx(uint64_t * p, uint64_t v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
DEVFN uint64_t d_clock() { return (uint64_t)wall_clock64(); }
DEVFN void     d_nap() { __builtin_amdgcn_s_sleep(1); }
#endif

// Bounded poll of a host flag until *p >= want. Returns 0 = timed out, 1 = satisfied, 2 = released by the abort value.
// Bounds: wall-clock deadline (primary) and a poll-count cap (backstop if the clock were broken). One thread only.
DEVFN uint32_t d_poll_ge(const uint64_t * p, uint64_t want, uint64_t deadline_ticks, uint32_t max_polls, uint32_t poll_mode,
                         uint32_t * polls_out, uint64_t * t_begin, uint64_t * t_end) {
    const uint64_t tb = d_clock();
    uint32_t polls = 0, res = 0;
    for (;;) {
        const uint64_t v = poll_mode ? d_ld_rlx(p) : d_ld_acq(p);
        ++polls;
        if (v >= want) { res = (v == kAbortVal) ? 2u : 1u; break; }
        if (polls >= max_polls) break;
        // the wall clock is read every 16th poll only (on gfx11 it is a scalar message round trip, s_sendmsg_rtn GET_REALTIME):
        // the 50 ms bound is not that precise, and the poll period stays that of a bare load + s_sleep
        if ((polls & 15u) == 0 && d_clock() - tb >= deadline_ticks) break;
        d_nap();
    }
    const uint64_t te = d_clock();
#ifndef SYNC_STUB
    if (res && poll_mode) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");   // relaxed polling: one acquire after the flag was seen
#endif
    *polls_out = polls; *t_begin = tb; *t_end = te;
    return res;
}

#ifndef SYNC_STUB
// ---- SIGNAL: plan packet + x -> host, then sig = seq (release, system scope)
__global__ __launch_bounds__(kThreads) void k_signal(SyncBlock * sb, Rec * rec, uint64_t seq, uint32_t idx) {
    const uint32_t t = threadIdx.x;
    if (t == 0) rec->t_start[idx] = d_clock();
    uint64_t * x = reinterpret_cast<uint64_t *>(sb->x);
    for (uint32_t w = t; w < kXWords; w += kThreads) d_st_rlx(&x[w], pat64(seq, w));
    if (t < 8) d_st_rlx(&sb->plan.w[t], pat64(seq, 0x1000u + t));
    __threadfence_system();      // every thread: its stores are visible at system scope ...
    __syncthreads();             // ... before thread 0 publishes the flag
    if (t == 0) {
        d_st_rel(&sb->sig.v, seq);
        rec->t_sig[idx] = d_clock();
    }
}

// ---- W kernels: synthetic GPU work. spin: busy-wait `ticks` wall-clock ticks. mem: read nread 16-byte vectors from a big buffer.
__global__ __launch_bounds__(64) void k_work_spin(Rec * rec, uint32_t idx, uint64_t ticks) {
    if (threadIdx.x == 0) {
        const uint64_t t0 = d_clock();
        rec->t_ws[idx] = t0;
        uint32_t n = 0;
        while (d_clock() - t0 < ticks) { if (++n > (1u << 28)) break; }     // the cap only matters if the clock is broken
    }
}
__global__ __launch_bounds__(256) void k_work_mem(const uint4 * __restrict__ buf, uint64_t start, uint64_t nread, uint32_t * out, Rec * rec, uint32_t idx) {
    if (blockIdx.x == 0 && threadIdx.x == 0) rec->t_ws[idx] = d_clock();
    // nread is a multiple of 4 * gridDim.x * blockDim.x (host side), start + nread <= buffer size: no bounds tests, 4 independent loads
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint4 * p = buf + start;
    uint32_t acc = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < nread; i += 4 * stride) {
        const uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    if (acc == 0x9E3779B9u && nread == 0xFFFFFFFFFFFFFFFFull) out[0] = acc;   // never true: keeps the loads alive
}

// ---- WAIT: bounded poll of res >= seq (do_wait = 1) or no poll (do_wait = 0, the stream wait op already ran), then read y,
//      verify it and add it into acc. A timeout sets the device counter AND the host-visible dev_timeout flag, then exits.
__global__ __launch_bounds__(kThreads) void k_wait(SyncBlock * sb, Rec * rec, uint32_t * acc, uint64_t seq, uint32_t idx, uint32_t do_wait,
                                                    uint64_t deadline_ticks, uint32_t max_polls, uint32_t poll_mode) {
    __shared__ uint32_t s_ok;
    const uint32_t t = threadIdx.x;
    if (t == 0) {
        uint32_t ok = 1, polls = 0;
        uint64_t tb = d_clock(), te = tb;
        if (do_wait) ok = d_poll_ge(&sb->res.v, seq, deadline_ticks, max_polls, poll_mode, &polls, &tb, &te);
        rec->t_wb[idx] = tb; rec->t_we[idx] = te; rec->polls[idx] = polls;
        if (ok == 0) { atomicAdd(&rec->timeouts, 1u); d_st_rlx(&sb->dev_timeout.v, 1ull); }
        else if (ok == 2) atomicAdd(&rec->aborted, 1u);
        s_ok = ok;
    }
    __syncthreads();
    const uint32_t ok = s_ok;
    if (ok == 0) return;                // block-uniform: nobody reads y after a timeout
    __threadfence_system();             // acquire side in every thread: y is read after the flag was seen
    const uint64_t * y = reinterpret_cast<const uint64_t *>(sb->y);
    const uint32_t base = y_base(seq);
    uint32_t bad = 0;
    for (uint32_t w = t; w < kXWords; w += kThreads) {
        const uint64_t u = d_ld_rlx(&y[w]);
        const uint32_t lo = (uint32_t)u, hi = (uint32_t)(u >> 32);
        if (ok == 1) bad += (lo != base + 2u * w) + (hi != base + 2u * w + 1u);
        acc[2u * w] += lo; acc[2u * w + 1u] += hi;
    }
    if (bad) atomicAdd(&rec->data_err, bad);
}

// ---- S1: persistent ping-pong kernel (one workgroup). mode bit0: read the 16 KB host->GPU payload, bit1: write 16 KB GPU->host.
__global__ __launch_bounds__(kThreads) void k_pingpong(SyncBlock * sb, S1Rec * rec, uint64_t seq_base, uint32_t max_pings, uint32_t mode,
                                                        uint64_t first_deadline, uint64_t deadline, uint32_t first_polls, uint32_t max_polls, uint32_t poll_mode) {
    __shared__ uint32_t s_st;           // 0 timeout, 1 proceed, 2 abort value, 3 quit
    __shared__ uint64_t s_g1;
    const uint32_t t = threadIdx.x;
    uint32_t bad = 0, i = 0, final_status = 4;
    for (; i < max_pings; ++i) {
        const uint64_t seq = seq_base + i + 1;
        if (t == 0) {
            uint32_t polls; uint64_t tb, te;
            uint32_t r = d_poll_ge(&sb->s1a.v, seq, i == 0 ? first_deadline : deadline, i == 0 ? first_polls : max_polls, poll_mode, &polls, &tb, &te);
            uint32_t st = r == 0 ? 0u : r == 2 ? 2u : 1u;
            if (st == 1 && d_ld_rlx(&sb->cmd.v) != 0) st = 3;
            if (st == 0) { atomicAdd(&rec->timeouts, 1u); d_st_rlx(&sb->dev_timeout.v, 1ull); }
            s_g1 = te; s_st = st;
        }
        __syncthreads();
        const uint32_t st = s_st;
        if (st != 1) { final_status = st; break; }     // block-uniform
        if (mode) __threadfence_system();               // acquire side: payload is read after the flag was seen (mode 0: nothing to order)
        if (mode & 1u) {
            const uint64_t * p = reinterpret_cast<const uint64_t *>(sb->s1_h2g);
            for (uint32_t w = t; w < kXWords; w += kThreads) bad += (d_ld_rlx(&p[w]) != pat64(seq, 0x2000u + w));
        }
        if (mode & 2u) {
            uint64_t * q = reinterpret_cast<uint64_t *>(sb->s1_g2h);
            for (uint32_t w = t; w < kXWords; w += kThreads) d_st_rlx(&q[w], pat64(seq, 0x3000u + w));
        }
        if (mode) __threadfence_system();               // release side: payload visible before the pong flag
        __syncthreads();
        if (t == 0) {
            d_st_rel(&sb->s1b.v, seq);
            rec->proc[i] = (uint32_t)(d_clock() - s_g1);
        }
    }
    if (bad) atomicAdd(&rec->data_err, bad);
    if (t == 0) { rec->pings_done = i; rec->status = final_status; }
}

__global__ void k_marker(uint32_t * p) { if (threadIdx.x == 0) *p = 1u; }

// ---- poll-rate probe: runs the very same bounded poll loop as the WAIT kernel on an idle flag (s1b: nobody writes it now) with a target
//      that can never be reached, for `ticks` wall-clock ticks, and reports how many polls fit. Used to turn the "50 ms" bound into a
//      poll count. Bounded by its own deadline and by 2M polls.
__global__ void k_pollrate(SyncBlock * sb, uint32_t * out, uint64_t ticks, uint32_t poll_mode) {
    if (threadIdx.x == 0) {
        uint32_t polls; uint64_t tb, te;
        const uint32_t r = d_poll_ge(&sb->s1b.v, ~0ull, ticks, 2000000u, poll_mode, &polls, &tb, &te);
        out[0] = polls; out[1] = (uint32_t)(te - tb); out[2] = r;
    }
}
#else   // ------------------------------------------------------------ stub mirrors of the kernels (sequential, host memory)
static int g_stub_nosig_seq = 0;
static void k_signal(SyncBlock * sb, Rec * rec, uint64_t seq, uint32_t idx) {
    rec->t_start[idx] = d_clock();
    uint64_t * x = reinterpret_cast<uint64_t *>(sb->x);
    for (uint32_t w = 0; w < kXWords; ++w) d_st_rlx(&x[w], pat64(seq, w));
    for (uint32_t t = 0; t < 8; ++t) d_st_rlx(&sb->plan.w[t], pat64(seq, 0x1000u + t));
    if ((int)seq != g_stub_nosig_seq) d_st_rel(&sb->sig.v, seq);
    rec->t_sig[idx] = d_clock();
}
static void k_work_spin(Rec * rec, uint32_t idx, uint64_t ticks) {
    const uint64_t t0 = d_clock(); rec->t_ws[idx] = t0;
    while (d_clock() - t0 < ticks) cpu_relax();
}
static void k_work_mem(uint64_t nread, Rec * rec, uint32_t idx) {     // pretends to stream at 800 GB/s
    const uint64_t t0 = d_clock(); rec->t_ws[idx] = t0;
    const uint64_t ticks = (uint64_t)((double)nread * 16.0 / 800e9 * 1e6 * 100.0);
    while (d_clock() - t0 < ticks) cpu_relax();
}
static void k_wait(SyncBlock * sb, Rec * rec, uint32_t * acc, uint64_t seq, uint32_t idx, uint32_t do_wait, uint64_t deadline_ticks, uint32_t max_polls, uint32_t poll_mode) {
    uint32_t ok = 1, polls = 0; uint64_t tb = d_clock(), te = tb;
    if (do_wait) ok = d_poll_ge(&sb->res.v, seq, deadline_ticks, max_polls, poll_mode, &polls, &tb, &te);
    rec->t_wb[idx] = tb; rec->t_we[idx] = te; rec->polls[idx] = polls;
    if (ok == 0) { rec->timeouts++; d_st_rlx(&sb->dev_timeout.v, 1ull); return; }
    if (ok == 2) rec->aborted++;
    const uint64_t * y = reinterpret_cast<const uint64_t *>(sb->y);
    const uint32_t base = y_base(seq); uint32_t bad = 0;
    for (uint32_t w = 0; w < kXWords; ++w) {
        const uint64_t u = d_ld_rlx(&y[w]); const uint32_t lo = (uint32_t)u, hi = (uint32_t)(u >> 32);
        if (ok == 1) bad += (lo != base + 2u * w) + (hi != base + 2u * w + 1u);
        acc[2u * w] += lo; acc[2u * w + 1u] += hi;
    }
    rec->data_err += bad;
}
static void k_pingpong(SyncBlock * sb, S1Rec * rec, uint64_t seq_base, uint32_t max_pings, uint32_t mode, uint64_t first_deadline, uint64_t deadline, uint32_t first_polls, uint32_t max_polls, uint32_t poll_mode) {
    uint32_t bad = 0, i = 0, final_status = 4;
    for (; i < max_pings; ++i) {
        const uint64_t seq = seq_base + i + 1;
        uint32_t polls; uint64_t tb, te;
        const uint32_t r = d_poll_ge(&sb->s1a.v, seq, i == 0 ? first_deadline : deadline, i == 0 ? first_polls : max_polls, poll_mode, &polls, &tb, &te);
        uint32_t st = r == 0 ? 0u : r == 2 ? 2u : 1u;
        if (st == 1 && d_ld_rlx(&sb->cmd.v) != 0) st = 3;
        if (st == 0) { rec->timeouts++; d_st_rlx(&sb->dev_timeout.v, 1ull); }
        if (st != 1) { final_status = st; break; }
        if (mode & 1u) { const uint64_t * p = reinterpret_cast<const uint64_t *>(sb->s1_h2g); for (uint32_t w = 0; w < kXWords; ++w) bad += (d_ld_rlx(&p[w]) != pat64(seq, 0x2000u + w)); }
        if (mode & 2u) { uint64_t * q = reinterpret_cast<uint64_t *>(sb->s1_g2h); for (uint32_t w = 0; w < kXWords; ++w) d_st_rlx(&q[w], pat64(seq, 0x3000u + w)); }
        d_st_rel(&sb->s1b.v, seq);
        rec->proc[i] = (uint32_t)(d_clock() - te);
    }
    rec->data_err += bad; rec->pings_done = i; rec->status = final_status;
}
static void k_marker(uint32_t * p) { *p = 1u; }
static void k_pollrate(SyncBlock * sb, uint32_t * out, uint64_t ticks, uint32_t poll_mode) {
    uint32_t polls; uint64_t tb, te;
    const uint32_t r = d_poll_ge(&sb->s1b.v, ~0ull, ticks, 2000000u, poll_mode, &polls, &tb, &te);
    out[0] = polls; out[1] = (uint32_t)(te - tb); out[2] = r;
}
#endif

// ====================================================================== host: device context, bounded waits
struct Dev {
    hipStream_t st = nullptr;
    SyncBlock * sbh = nullptr; SyncBlock * sbd = nullptr;      // host view / device view of the pinned block
    Rec * rec = nullptr; S1Rec * rec1 = nullptr;                // device records
    Rec * rec_h = nullptr; S1Rec * rec1_h = nullptr;            // pinned staging copies
    uint32_t * acc = nullptr; uint32_t * dummy = nullptr;
    void * mem = nullptr; uint32_t * mem_out = nullptr; uint64_t nvec_total = 0; uint32_t mem_grid = 0; double mem_gbps = 0;
    double tpu = 100.0;                                         // wall_clock64 ticks per us (effective)
    uint64_t deadline_ticks = 0, first_deadline_ticks = 0; uint32_t max_polls = 0, first_max_polls = 0, poll_mode = 0;
    double poll_us = 0;                                         // measured period of one bounded-poll iteration
    hipEvent_t ring[4] = {}, ev_a = nullptr, ev_b = nullptr, ev_tmp = nullptr, ev_k = nullptr;
    int lane_cpu = -1;
    bool mem_ok = false;
};

// nread (16-byte vectors) is rounded down to a multiple of 4 * grid * 256 (see k_work_mem): ~12.6 MB steps, ~15 us of streaming
static uint64_t round_nread(const Dev & D, uint64_t nvec) {
    const uint64_t q = 4ull * D.mem_grid * 256ull;
    return std::min<uint64_t>(nvec, D.nvec_total) / q * q;
}

enum WaitRes { WR_OK, WR_TIMEOUT, WR_ABORT };
static bool event_done(hipEvent_t ev) {
    const hipError_t e = hipEventQuery(ev);
    if (e == hipSuccess) return true;
    (void)hipGetLastError();                                    // hipErrorNotReady may be recorded as the "last error"
    if (e != hipErrorNotReady) fatal(2, "hipEventQuery failed: %s", hipGetErrorString(e));
    return false;
}
// bounded wait for an event: spins on hipEventQuery. WR_TIMEOUT after `secs`; WR_ABORT as soon as abort_run() was called
// (unless honour_abort is false: the post-abort drain).
static WaitRes wait_event_ex(hipEvent_t ev, double secs, bool honour_abort) {
    const int64_t dl = now_ns() + (int64_t)(secs * 1e9);
    uint32_t spin = 0;
    for (;;) {
        if (event_done(ev)) return WR_OK;
        if (honour_abort && g_abort.load(std::memory_order_relaxed)) return WR_ABORT;
        cpu_relax();
        if ((++spin & 63) == 0) {
            if (honour_abort && g_sbh && ld_rlx(&g_sbh->dev_timeout.v)) { abort_run("a device wait kernel timed out (dev_timeout flag)"); return WR_ABORT; }
            if (!honour_abort && (spin & 1023) == 0 && g_abort.load(std::memory_order_relaxed)) release_all();   // post-abort drain only: keep every flag at the abort value
            if (now_ns() > dl) return WR_TIMEOUT;
        }
    }
}
static bool wait_event(hipEvent_t ev, double secs, const char * what) {
    const WaitRes r = wait_event_ex(ev, secs, true);
    if (r == WR_TIMEOUT) abort_run("timeout %.1f s waiting for %s", secs, what);
    return r == WR_OK;
}
static bool sync_stream(Dev & D, const char * what) {
    HIP_OK(hipEventRecord(D.ev_tmp, D.st));
    return wait_event(D.ev_tmp, g_o.stream_deadline_s, what);
}
static bool dev_memset(Dev & D, void * p, size_t n) { HIP_OK(hipMemsetAsync(p, 0, n, D.st)); return sync_stream(D, "device memset"); }
static bool d2h(Dev & D, void * dst, const void * src, size_t n) { HIP_OK(hipMemcpyAsync(dst, src, n, hipMemcpyDeviceToHost, D.st)); return sync_stream(D, "device-to-host copy"); }
// after an abort: let the (released) stream finish, ignoring the abort flag; true when it did
static bool drain_after_abort(Dev & D) {
    HIP_OK(hipEventRecord(D.ev_tmp, D.st));
    return wait_event_ex(D.ev_tmp, 5.0, false) == WR_OK;
}
static void launch_check() {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess && e != hipErrorNotReady) fatal(2, "kernel launch failed: %s", hipGetErrorString(e));
}

// ---- launch wrappers (real: hipLaunchKernelGGL, stub: closure on the stream thread)
#ifndef SYNC_STUB
static void launch_signal(Dev & D, uint64_t seq, uint32_t idx) { hipLaunchKernelGGL(k_signal, dim3(1), dim3(kThreads), 0, D.st, D.sbd, D.rec, seq, idx); }
static void launch_wait(Dev & D, uint64_t seq, uint32_t idx, uint32_t do_wait) {
    hipLaunchKernelGGL(k_wait, dim3(1), dim3(kThreads), 0, D.st, D.sbd, D.rec, D.acc, seq, idx, do_wait, D.deadline_ticks, D.max_polls, D.poll_mode);
}
static void launch_work_spin(Dev & D, uint32_t idx, uint64_t ticks) { hipLaunchKernelGGL(k_work_spin, dim3(1), dim3(64), 0, D.st, D.rec, idx, ticks); }
static void launch_work_mem(Dev & D, uint32_t idx, uint64_t start, uint64_t nread) {
    hipLaunchKernelGGL(k_work_mem, dim3(nread ? D.mem_grid : 1u), dim3(256), 0, D.st, (const uint4 *)D.mem, start, nread, D.mem_out, D.rec, idx);
}
static void launch_pingpong(Dev & D, uint64_t seq_base, uint32_t max_pings, uint32_t mode) {
    hipLaunchKernelGGL(k_pingpong, dim3(1), dim3(kThreads), 0, D.st, D.sbd, D.rec1, seq_base, max_pings, mode, D.first_deadline_ticks, D.deadline_ticks, D.first_max_polls, D.max_polls, D.poll_mode);
}
static void launch_marker(Dev & D) { hipLaunchKernelGGL(k_marker, dim3(1), dim3(64), 0, D.st, D.dummy); }
static void launch_pollrate(Dev & D, uint64_t ticks) { hipLaunchKernelGGL(k_pollrate, dim3(1), dim3(64), 0, D.st, D.sbd, D.dummy, ticks, D.poll_mode); }
#else
static void launch_signal(Dev & D, uint64_t seq, uint32_t idx) { SyncBlock * sb = D.sbd; Rec * r = D.rec; D.st->push([=] { k_signal(sb, r, seq, idx); }); }
static void launch_wait(Dev & D, uint64_t seq, uint32_t idx, uint32_t do_wait) {
    SyncBlock * sb = D.sbd; Rec * r = D.rec; uint32_t * acc = D.acc; uint64_t dl = D.deadline_ticks; uint32_t mp = D.max_polls, pm = D.poll_mode;
    D.st->push([=] { k_wait(sb, r, acc, seq, idx, do_wait, dl, mp, pm); });
}
static void launch_work_spin(Dev & D, uint32_t idx, uint64_t ticks) { Rec * r = D.rec; D.st->push([=] { k_work_spin(r, idx, ticks); }); }
static void launch_work_mem(Dev & D, uint32_t idx, uint64_t, uint64_t nread) { Rec * r = D.rec; D.st->push([=] { k_work_mem(nread, r, idx); }); }
static void launch_pingpong(Dev & D, uint64_t seq_base, uint32_t max_pings, uint32_t mode) {
    SyncBlock * sb = D.sbd; S1Rec * r = D.rec1; uint64_t fd = D.first_deadline_ticks, dl = D.deadline_ticks; uint32_t fp = D.first_max_polls, mp = D.max_polls, pm = D.poll_mode;
    D.st->push([=] { k_pingpong(sb, r, seq_base, max_pings, mode, fd, dl, fp, mp, pm); });
}
static void launch_marker(Dev & D) { uint32_t * p = D.dummy; D.st->push([=] { k_marker(p); }); }
static void launch_pollrate(Dev & D, uint64_t ticks) { SyncBlock * sb = D.sbd; uint32_t * o = D.dummy; uint32_t pm = D.poll_mode; D.st->push([=] { k_pollrate(sb, o, ticks, pm); }); }
#endif

// ====================================================================== calibration
static bool calibrate_wallclock(Dev & D) {
    int khz = 0;
    HIP_OK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
    double tpu = khz > 0 ? khz / 1000.0 : 100.0;
    printf("# wall_clock64 rate attribute = %d kHz (%.3f ticks/us)%s\n", khz, tpu, khz > 0 ? "" : "  [attribute missing: assuming 100 MHz]");
    auto timed = [&](double us, float * ms) -> bool {
        HIP_OK(hipEventRecord(D.ev_a, D.st));
        launch_work_spin(D, 0, (uint64_t)(us * tpu));
        HIP_OK(hipEventRecord(D.ev_b, D.st));
        launch_check();
        if (!wait_event(D.ev_b, g_o.stream_deadline_s, "wall-clock calibration kernel")) return false;
        HIP_OK(hipEventElapsedTime(ms, D.ev_a, D.ev_b));
        return true;
    };
    float m0 = 0, m1 = 0, m2 = 0;
    if (!timed(1000, &m0) || !timed(1000, &m1) || !timed(5000, &m2)) return false;
    const double r = ((double)(m2 - m1) * 1000.0) / 4000.0;       // measured us per nominal us (launch overhead cancels)
    printf("# wall_clock64 check: 1000 us kernel = %.1f us, 5000 us kernel = %.1f us -> measured/nominal = %.3f\n", m1 * 1000.0, m2 * 1000.0, r);
    if (std::fabs(r - 1.0) > 0.2 && r > 0.05 && r < 20.0) {
        tpu = tpu / r;
        printf("# WARN wall_clock64 does not tick at the attribute rate: using %.3f ticks/us\n", tpu);
    }
    D.tpu = tpu;
    D.deadline_ticks = (uint64_t)(g_o.wait_ms * 1e3 * tpu);
    D.first_deadline_ticks = (uint64_t)(g_o.first_wait_ms * 1e3 * tpu);
    D.poll_mode = (uint32_t)g_o.poll_mode;
    return true;
}

// the poll-count cap of every device wait: 2x the number of polls that fit in the wall-clock bound (so the clock bound fires first
// unless the clock is broken). Measured with a 2 ms probe kernel (it polls an idle flag, bounded by its own deadline).
static bool calibrate_poll(Dev & D) {
    HIP_OK(hipMemset(D.dummy, 0, 256));
    launch_pollrate(D, (uint64_t)(2000.0 * D.tpu));
    launch_check();
    if (!sync_stream(D, "poll-rate probe kernel")) return false;
    if (!d2h(D, D.rec_h, D.dummy, 16)) return false;
    const uint32_t * o = reinterpret_cast<const uint32_t *>(D.rec_h);
    if (o[0] == 0 || o[1] == 0) { abort_run("poll-rate probe returned nothing"); return false; }
    D.poll_us = ((double)o[1] / D.tpu) / (double)o[0];
    const double per_ms = 1000.0 / D.poll_us;
    auto cap = [&](double ms) { return (uint32_t)std::min(4e9, std::max(1000.0, 2.0 * ms * per_ms)); };
    D.max_polls = g_o.max_polls ? g_o.max_polls : cap(g_o.wait_ms);
    D.first_max_polls = g_o.max_polls ? g_o.max_polls : cap(g_o.first_wait_ms);
    printf("# device poll: one %s poll of a host flag = %.3f us (%u polls in %.0f us) -> poll cap %u (%.0f ms bound) / first %u (%.0f ms bound)\n",
           D.poll_mode ? "relaxed" : "acquire", D.poll_us, o[0], (double)o[1] / D.tpu, D.max_polls, g_o.wait_ms, D.first_max_polls, g_o.first_wait_ms);
    printf("# device wait bound: %.1f ms wall clock (%llu ticks), first S1 ping %.0f ms, poll_mode=%s\n", g_o.wait_ms, ULL(D.deadline_ticks),
           g_o.first_wait_ms, D.poll_mode ? "relaxed+fence" : "acquire");
    return true;
}

// streaming read bandwidth of the W=mem kernel: rotating reads over the whole buffer
static bool calibrate_mem(Dev & D) {
    const uint64_t nread = round_nread(D, (256ull << 20) / 16);
    if (nread == 0) { printf("# W=mem kernel: buffer too small for one pass\n"); return false; }
    uint64_t it = 0;
    auto run = [&](int reps, float * ms) -> bool {
        HIP_OK(hipEventRecord(D.ev_a, D.st));
        for (int r = 0; r < reps; ++r, ++it) launch_work_mem(D, 0, (it * nread) % (D.nvec_total - nread + 1), nread);
        HIP_OK(hipEventRecord(D.ev_b, D.st));
        launch_check();
        if (!wait_event(D.ev_b, g_o.stream_deadline_s, "memory-bandwidth calibration")) return false;
        HIP_OK(hipEventElapsedTime(ms, D.ev_a, D.ev_b));
        return true;
    };
    float m = 0;
    if (!run(2, &m) || !run(8, &m)) return false;
    D.mem_gbps = (double)nread * 16.0 * 8.0 / ((double)m * 1e-3) / 1e9;
    D.mem_ok = D.mem_gbps > 1.0;
    printf("# W=mem kernel: %.0f MB per pass over a %d MB buffer, streaming read %.1f GB/s -> 600 us = %.0f MB\n", nread * 16.0 / 1e6, g_o.mem_mb,
           D.mem_gbps, D.mem_gbps * 1e9 * 600e-6 / 1e6);
    return true;
}

// ====================================================================== CPU lane thread (the emulated fourth lane)
struct Lane {
    SyncBlock * sb = nullptr; int cpu = -1; uint64_t seq_base = 0, tc_ticks = 0;
    std::atomic<bool> stop{false}, ready{false};
    std::atomic<uint64_t> done{0};
    std::atomic<int> err{0};                 // 0 ok, 1 timeout waiting for sig, 2 protocol error, 3 abort seen
    uint64_t data_err = 0;
    std::vector<double> proc_us;             // per iteration: sig seen -> res posted
};

static void lane_main(Lane * L) {
    if (L->cpu >= 0 && !pin_to_cpu(L->cpu)) fprintf(stderr, "lane: sched_setaffinity(%d) failed\n", L->cpu);
    SyncBlock * sb = L->sb;
    const uint64_t dl_ticks = (uint64_t)(g_o.host_deadline_s * 1e6 * g_tsc_per_us);
    uint64_t next = L->seq_base + 1, count = 0;
    L->ready.store(true, std::memory_order_release);
    for (;;) {
        // ---- wait for the SIGNAL kernel of iteration `next` (bounded)
        uint64_t v, spin = 0;
        const uint64_t t_wait0 = rdtsc_();
        for (;;) {
            v = ld_acq(&sb->sig.v);
            if (v >= next) break;
            cpu_relax();
            if ((++spin & 127) == 0) {
                if (L->stop.load(std::memory_order_relaxed)) return;
                if (g_abort.load(std::memory_order_relaxed)) { L->err = 3; return; }
                if (rdtsc_() - t_wait0 > dl_ticks) {
                    L->err = 1;
                    abort_run("lane: no SIGNAL for seq %llu within %.1f s (sig=%llu)", ULL(next), g_o.host_deadline_s, ULL(v));
                    return;
                }
            }
        }
        const uint64_t t_seen = rdtsc_();
        if (v != next) {
            L->err = 2;
            abort_run("lane: protocol error, sig=%llu but expected exactly %llu", ULL(v), ULL(next));
            return;
        }
        if (g_o.inject_lane_stall > 0 && count == (uint64_t)g_o.inject_lane_stall) {    // test hook: go deaf
            while (!g_abort.load() && !L->stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            L->err = 3; return;
        }
        // ---- read the plan packet and x (16 KB) and check them
        uint64_t bad = 0;
        for (uint32_t t = 0; t < 8; ++t) bad += (sb->plan.w[t] != pat64(next, 0x1000u + t));
        const uint64_t * x = reinterpret_cast<const uint64_t *>(sb->x);
        for (uint32_t w = 0; w < kXWords; ++w) bad += (x[w] != pat64(next, w));
        L->data_err += bad;
        // ---- emulated expert compute: spin tc us (starts after x was read)
        if (L->tc_ticks) { const uint64_t tgt = rdtsc_() + L->tc_ticks; while (rdtsc_() < tgt) cpu_relax(); }
        // ---- result y (16 KB), then the sequence number (release)
        uint32_t * y = reinterpret_cast<uint32_t *>(sb->y);
        const uint32_t base = y_base(next);
        for (uint32_t j = 0; j < kYWords32; ++j) y[j] = base + j;
        st_max_rel(&sb->res.v, next);                              // monotonic: cannot undo an abort release
        const uint64_t t_post = rdtsc_();
        L->proc_us.push_back((double)(t_post - t_seen) / g_tsc_per_us);
        ++count; ++next;
        L->done.store(count, std::memory_order_release);
        g_prog.v.fetch_add(1, std::memory_order_relaxed);
    }
}

// ====================================================================== H2D streams (the part-B pattern of bench_h2d_cpu.hip)
struct H2DCard {
    int dev = 0, cpu = 0;
    uint8_t * host = nullptr; size_t npieces = 0; void * d = nullptr; hipStream_t st = nullptr;
    alignas(64) std::atomic<uint64_t> bytes{0};
    std::thread th; bool running = false;
};
struct H2D {
    std::vector<std::unique_ptr<H2DCard>> cards;
    std::atomic<bool> stop{false};
};
static void h2d_thread(H2D * H, H2DCard * c) {
    HIP_OK(hipSetDevice(c->dev));                       // per thread entry (and again per call below)
    if (!pin_to_cpu(c->cpu)) fprintf(stderr, "h2d card %d: sched_setaffinity(%d) failed\n", c->dev, c->cpu);
    size_t i = (size_t)c->dev * 13;
    while (!H->stop.load(std::memory_order_relaxed)) {
        HIP_OK(hipSetDevice(c->dev));
        HIP_OK(hipMemcpyAsync(c->d, c->host + (i % c->npieces) * kPiece, kPiece, hipMemcpyHostToDevice, c->st));
        HIP_OK(hipStreamSynchronize(c->st));
        c->bytes.fetch_add(kPiece, std::memory_order_relaxed);
        ++i;
    }
}
static void h2d_start(H2D & H, const std::vector<int> & active) {
    H.stop.store(false);
    for (int c : active) { H2DCard & cd = *H.cards[c]; cd.th = std::thread(h2d_thread, &H, &cd); cd.running = true; }
}
static void h2d_stop(H2D & H) {
    H.stop.store(true);
    for (auto & c : H.cards) if (c->running) { c->th.join(); c->running = false; }
}
static std::vector<uint64_t> h2d_bytes(H2D & H) { std::vector<uint64_t> v; for (auto & c : H.cards) v.push_back(c->bytes.load()); return v; }
static std::vector<double> h2d_rates(const std::vector<uint64_t> & a, const std::vector<uint64_t> & b, double dt_s) {
    std::vector<double> r;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) r.push_back(dt_s > 0 ? (double)(b[i] - a[i]) / dt_s / 1e9 : 0.0);
    return r;
}

// ====================================================================== S2: one layer-pattern (or baseline) configuration
struct PatCfg {
    bool sync = true;          // false: baseline = the W kernel only
    int wait_mode = 0;         // 0: spin kernel, 1: hipStreamWaitValue32 + consume kernel
    int work_mode = 0;         // 0: spin W us, 1: stream a big buffer for about W us
    double W_us = 0, tc_us = 0, warm = 1, secs = 6;
    H2D * h2d = nullptr;       // sample these byte counters at the window edges
};
struct PatRes {
    bool ok = false;
    uint64_t n = 0;
    double host_us = 0, ev_us = 0, window_s = 0;
    Stat it, spin, polls, devrt, lane;
    uint32_t timeouts = 0, aborted = 0, data_err = 0;
    uint64_t lane_err = 0, lane_data_err = 0;
    std::vector<double> gbps;
};

struct WorkParams { int mode = 0; uint64_t ticks = 0, nvec_read = 0; };
static WorkParams make_work(const Dev & D, int mode, double W_us) {
    WorkParams w; w.mode = mode;
    if (mode == 0) w.ticks = (uint64_t)(W_us * D.tpu);
    else w.nvec_read = round_nread(D, (uint64_t)(W_us * 1e-6 * D.mem_gbps * 1e9 / 16.0));
    return w;
}
static void launch_iteration(Dev & D, const PatCfg & c, const WorkParams & wp, uint64_t seq_base, uint64_t it) {
    const uint64_t seq = seq_base + it + 1;
    const uint32_t idx = (uint32_t)it;
    if (c.sync) launch_signal(D, seq, idx);
    if (wp.mode == 0) launch_work_spin(D, idx, wp.ticks);
    else launch_work_mem(D, idx, wp.nvec_read ? (it * wp.nvec_read) % (D.nvec_total - wp.nvec_read + 1) : 0, wp.nvec_read);   // rotating, never wrapping window
    if (c.sync) {
        if (c.wait_mode == 0) launch_wait(D, seq, idx, 1);
        else {
            HIP_OK(hipStreamWaitValue32(D.st, &D.sbd->res.v, (uint32_t)seq, hipStreamWaitValueGte, 0xFFFFFFFFu));
            launch_wait(D, seq, idx, 0);
        }
    }
    launch_check();
}

static PatRes run_pattern(Dev & D, const PatCfg & c) {
    PatRes R;
    char ph[160];
    snprintf(ph, sizeof ph, "%s wait=%s work=%s W=%.0f tc=%.0f", c.sync ? "pattern" : "baseline", c.wait_mode ? "streamwait" : "spin", c.work_mode ? "mem" : "spin", c.W_us, c.tc_us);
    set_phase(ph);
    HIP_OK(hipSetDevice(0));
    if (g_abort.load()) return R;
    st_rlx(&D.sbh->dev_timeout.v, 0);
    if (!dev_memset(D, D.rec, sizeof(Rec))) return R;
    const uint64_t seq_base = g_seq2;
    if (c.sync && seq_base + kRecCap + 64 >= kSeqLimit) fatal(2, "sequence numbers exhausted (%llu)", ULL(seq_base));
    const WorkParams wp = make_work(D, c.work_mode, c.W_us);

    Lane L; std::thread lane_th;
    if (c.sync) {
        L.sb = D.sbh; L.cpu = D.lane_cpu; L.seq_base = seq_base; L.tc_ticks = (uint64_t)(c.tc_us * g_tsc_per_us);
        L.proc_us.reserve(kRecCap + 64);
        lane_th = std::thread(lane_main, &L);
        const int64_t t0 = now_ns();
        while (!L.ready.load(std::memory_order_acquire)) {
            if (now_ns() - t0 > 2000000000ll) { abort_run("lane thread did not start"); break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    struct ActiveGuard { ActiveGuard() { g_active.store(1); } ~ActiveGuard() { g_active.store(0); } } guard;
    constexpr uint64_t kWarmCap = 20000, kMinWarm = 50, kChunk = 8;
    const uint64_t max_meas = std::min<uint64_t>((uint64_t)std::max(1, g_o.max_iters), kRecCap - 2 - kWarmCap);
    const int64_t warm_ns = (int64_t)(c.warm * 1e9), meas_ns = (int64_t)(c.secs * 1e9);
    bool ok = true, meas = false, a_pending = false;
    uint64_t it = 0, idx_a = 0, idx_b = 0;
    int64_t t_begin = now_ns(), t_meas0 = 0, t_a = 0, t_b = 0;
    std::vector<uint64_t> h2d_a, h2d_b;

    for (;;) {
        if (g_abort.load(std::memory_order_relaxed)) { ok = false; break; }
        if (ld_acq(&D.sbh->dev_timeout.v)) { abort_run("a device wait kernel timed out (dev_timeout flag)"); ok = false; break; }
        const int64_t now = now_ns();
        if (!meas) {
            if ((now - t_begin >= warm_ns && it >= kMinWarm) || it >= kWarmCap) {
                HIP_OK(hipEventRecord(D.ev_a, D.st));            // completes when iteration it-1 has completed
                idx_a = it; t_meas0 = now; meas = true; a_pending = true;
            }
        } else if (now - t_meas0 >= meas_ns || it - idx_a >= max_meas) {
            HIP_OK(hipEventRecord(D.ev_b, D.st));
            idx_b = it; break;
        }
        // pacing: at most ~4 chunks of kChunk iterations in flight (event ring); this also bounds the launch queue
        if (it > 0 && it % kChunk == 0) {
            const uint64_t k = it / kChunk - 1;
            HIP_OK(hipEventRecord(D.ring[k % 4], D.st));
            if (k >= 3 && !wait_event(D.ring[(k - 3) % 4], g_o.stream_deadline_s, "pacing event")) { ok = false; break; }
        }
        launch_iteration(D, c, wp, seq_base, it);
        ++it;
        g_prog.v.fetch_add(1, std::memory_order_relaxed);
        if (a_pending && event_done(D.ev_a)) { t_a = now_ns(); a_pending = false; if (c.h2d) h2d_a = h2d_bytes(*c.h2d); }
    }
    if (ok && a_pending) {
        if (wait_event(D.ev_a, g_o.stream_deadline_s, "window start event")) { t_a = now_ns(); if (c.h2d) h2d_a = h2d_bytes(*c.h2d); } else ok = false;
    }
    if (ok) {                                                      // the bounded wait for the final completion
        if (wait_event(D.ev_b, g_o.stream_deadline_s, "final stream drain")) { t_b = now_ns(); if (c.h2d) h2d_b = h2d_bytes(*c.h2d); } else ok = false;
    }
    if (!ok) {                                                     // abort path: the stream is released by release_all(); let it drain
        if (!drain_after_abort(D)) abort_run("stream did not drain 5 s after the abort");
    }
    if (c.sync) {
        L.stop.store(true);
        if (lane_th.joinable()) lane_th.join();
        if (ok && L.done.load() != it) { abort_run("lane served %llu iterations, %llu were launched", ULL(L.done.load()), ULL(it)); ok = false; }
        g_seq2 = seq_base + it;                                    // the counter only ever advances
    }
    R.lane_err = (uint64_t)L.err.load(); R.lane_data_err = L.data_err;
    if (!ok) return R;
    if (idx_b < idx_a + 2) { abort_run("window too short (%llu iterations)", ULL(idx_b - idx_a)); return R; }

    // ---- results: host total / N, device-event total / N, per-iteration device records
    R.n = idx_b - idx_a;
    R.window_s = (double)(t_b - t_a) * 1e-9;
    R.host_us = (double)(t_b - t_a) * 1e-3 / (double)R.n;
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, D.ev_a, D.ev_b));
    R.ev_us = (double)ms * 1e3 / (double)R.n;
    if (!d2h(D, D.rec_h, D.rec, sizeof(Rec))) return R;
    const Rec * r = D.rec_h;
    std::vector<double> v_it, v_spin, v_polls, v_rt, v_lane;
    const uint64_t * t_iter = c.sync ? r->t_start : r->t_ws;        // iteration start stamps (GPU clock)
    for (uint64_t i = idx_a; i + 1 < idx_b; ++i) v_it.push_back((double)(t_iter[i + 1] - t_iter[i]) / D.tpu);
    if (c.sync) {
        for (uint64_t i = idx_a; i < idx_b; ++i) {
            v_spin.push_back((double)(r->t_we[i] - r->t_wb[i]) / D.tpu);
            v_polls.push_back((double)r->polls[i]);
            v_rt.push_back((double)(r->t_we[i] - r->t_sig[i]) / D.tpu);
            if (i < L.proc_us.size()) v_lane.push_back(L.proc_us[i]);
        }
    }
    R.it = stat_of(v_it); R.spin = stat_of(v_spin); R.polls = stat_of(v_polls); R.devrt = stat_of(v_rt); R.lane = stat_of(v_lane);
    R.timeouts = r->timeouts; R.aborted = r->aborted; R.data_err = r->data_err;
    if (c.h2d) R.gbps = h2d_rates(h2d_a, h2d_b, R.window_s);
    R.ok = true;
    if (R.timeouts || R.aborted) { abort_run("device wait timeouts=%u aborted=%u", R.timeouts, R.aborted); R.ok = false; }
    if (R.data_err || R.lane_data_err) { g_rc = 1; printf("# ERROR data mismatches in %s: device=%u host(lane)=%llu\n", ph, R.data_err, ULL(R.lane_data_err)); }
    return R;
}

static std::map<std::pair<int, int>, double> g_base_us;                  // (work_mode, W) -> baseline host_us
static std::map<std::tuple<int, int, int, int>, double> g_pat_us;        // (wait, work, W, tc) -> pattern host_us

static void print_s2(const char * tag, const PatCfg & c, const PatRes & R, const char * extra) {
    const double mx = std::max(c.W_us, c.tc_us);
    auto it = g_base_us.find({c.work_mode, (int)c.W_us});
    const double base = it == g_base_us.end() ? NAN : it->second;
    printf("%s,wait=%s,work=%s,W=%.0f,tc=%.0f,n=%llu,pattern_us=%.2f,ev_us=%.2f,it_us_med=%.2f,it_us_p99=%.2f,base_us=%.2f,d_us=%.2f,d_ev_us=%.2f,d_vs_base_us=%.2f",
           tag, c.wait_mode ? "streamwait" : "spin", c.work_mode ? "mem" : "spin", c.W_us, c.tc_us, ULL(R.n), R.host_us, R.ev_us, R.it.med, R.it.p99, base,
           R.host_us - mx, R.ev_us - mx, std::isnan(base) ? NAN : R.host_us - std::max(base, c.tc_us));
    if (c.wait_mode == 0) printf(",spin_us_med=%.2f,spin_us_p99=%.2f,spin_us_max=%.2f,polls_med=%.0f,polls_p99=%.0f", R.spin.med, R.spin.p99, R.spin.max, R.polls.med, R.polls.p99);
    else printf(",spin_us_med=na,spin_us_p99=na,spin_us_max=na,polls_med=na,polls_p99=na");
    printf(",devrt_us_med=%.2f,devrt_us_p99=%.2f,lane_proc_us_med=%.2f,lane_proc_us_p99=%.2f,timeouts=%u,data_err=%u,lane_data_err=%llu",
           R.devrt.med, R.devrt.p99, R.lane.med, R.lane.p99, R.timeouts, R.data_err, ULL(R.lane_data_err));
    if (extra && *extra) printf(",%s", extra);
    printf("\n");
}
static void print_base(const PatCfg & c, const PatRes & R) {
    printf("S2base,work=%s,W=%.0f,n=%llu,base_us=%.2f,ev_us=%.2f,it_us_med=%.2f,it_us_p99=%.2f\n", c.work_mode ? "mem" : "spin", c.W_us, ULL(R.n), R.host_us, R.ev_us, R.it.med, R.it.p99);
}

// ====================================================================== S1: persistent-kernel ping-pong
struct S1Host {
    SyncBlock * sb = nullptr; int cpu = -1; uint64_t seq_base = 0; uint32_t mode = 0; double warm = 1, secs = 6;
    std::atomic<bool> ready{false}, go{false};
    std::vector<double> rtt, rtt_read;
    uint32_t n_warm = 0, n_total = 0; uint64_t bad = 0; int err = 0;
};
static void s1_host(S1Host * H) {
    if (H->cpu >= 0 && !pin_to_cpu(H->cpu)) fprintf(stderr, "s1 host: sched_setaffinity(%d) failed\n", H->cpu);
    SyncBlock * sb = H->sb;
    H->ready.store(true, std::memory_order_release);
    while (!H->go.load(std::memory_order_acquire)) { if (g_abort.load()) return; cpu_relax(); }
    const uint64_t dl_ticks = (uint64_t)(g_o.host_deadline_s * 1e6 * g_tsc_per_us), first_dl = (uint64_t)(3.0 * 1e6 * g_tsc_per_us);
    const uint64_t t_begin = rdtsc_();
    const uint64_t t_warm_end = t_begin + (uint64_t)(H->warm * 1e6 * g_tsc_per_us), t_end = t_warm_end + (uint64_t)(H->secs * 1e6 * g_tsc_per_us);
    H->rtt.reserve(kS1Cap); H->rtt_read.reserve(kS1Cap);
    uint32_t i = 0;
    for (; i < kS1Cap; ++i) {
        const uint64_t seq = H->seq_base + i + 1;
        if (H->mode & 1u) { uint64_t * p = reinterpret_cast<uint64_t *>(sb->s1_h2g); for (uint32_t w = 0; w < kXWords; ++w) p[w] = pat64(seq, 0x2000u + w); }
        const uint64_t t0 = rdtsc_();
        st_rel(&sb->s1a.v, seq);                                   // ping
        const uint64_t lim = (i == 0 ? first_dl : dl_ticks); uint64_t spin = 0, v;
        for (;;) {
            v = ld_acq(&sb->s1b.v);
            if (v >= seq) break;
            cpu_relax();
            if ((++spin & 127) == 0) {
                if (g_abort.load(std::memory_order_relaxed)) { H->err = 3; break; }
                if (ld_rlx(&sb->dev_timeout.v)) { H->err = 1; abort_run("S1: the kernel's bounded wait timed out (dev_timeout flag)"); break; }
                if (rdtsc_() - t0 > lim) { H->err = 1; abort_run("S1: no pong for seq %llu within %.1f s (s1b=%llu)", ULL(seq), i == 0 ? 3.0 : g_o.host_deadline_s, ULL(v)); break; }
            }
        }
        if (H->err) break;
        const uint64_t t1 = rdtsc_();
        uint64_t t2 = t1;
        if (v != seq) { H->err = 2; abort_run("S1: protocol error, s1b=%llu expected %llu", ULL(v), ULL(seq)); break; }
        if (H->mode & 2u) {                                        // consume the GPU->host payload
            const uint64_t * q = reinterpret_cast<const uint64_t *>(sb->s1_g2h);
            uint64_t bad = 0; for (uint32_t w = 0; w < kXWords; ++w) bad += (q[w] != pat64(seq, 0x3000u + w));
            H->bad += bad; t2 = rdtsc_();
        }
        if (t0 >= t_warm_end || i >= kS1Cap / 4) { H->rtt.push_back((double)(t1 - t0) / g_tsc_per_us); H->rtt_read.push_back((double)(t2 - t0) / g_tsc_per_us); }
        else H->n_warm++;
        if ((i & 63) == 63) g_prog.v.fetch_add(64, std::memory_order_relaxed);
        if (t1 >= t_end) { ++i; break; }
    }
    H->n_total = i;
    // quit: command first (release), then a flag value that wakes the kernel. After an abort s1a already holds kAbortVal and
    // cmd is 1 (release_all); a plain store of a smaller value would be harmless (the kernel only waits for seq <= this value)
    // but is skipped anyway.
    st_rel(&sb->cmd.v, 1);
    st_max_rel(&sb->s1a.v, H->seq_base + i + 1);
}

struct S1Out { bool ok = false; size_t n = 0; Stat rtt, rtt_read, gproc, wire; uint32_t kstatus = 99, dev_bad = 0; uint64_t host_bad = 0; };
static S1Out run_s1(Dev & D, uint32_t mode, double warm, double secs) {
    S1Out O;
    char ph[64]; snprintf(ph, sizeof ph, "S1 mode=%u", mode); set_phase(ph);
    HIP_OK(hipSetDevice(0));
    if (g_abort.load()) return O;
    st_rlx(&D.sbh->dev_timeout.v, 0);
    st_rel(&D.sbh->cmd.v, 0);
    if (!dev_memset(D, D.rec1, 16)) return O;                      // counters / status only; proc[] is written before it is read
    const uint64_t seq_base = g_seq1;
    if (seq_base + kS1Cap + 64 >= kSeqLimit) fatal(2, "sequence numbers exhausted (%llu)", ULL(seq_base));
    S1Host H; H.sb = D.sbh; H.cpu = D.lane_cpu; H.seq_base = seq_base; H.mode = mode; H.warm = warm; H.secs = secs;
    std::thread th(s1_host, &H);
    { const int64_t t0 = now_ns(); while (!H.ready.load(std::memory_order_acquire)) { if (now_ns() - t0 > 2000000000ll) { abort_run("S1 host thread did not start"); break; } std::this_thread::sleep_for(std::chrono::milliseconds(1)); } }
    struct ActiveGuard { ActiveGuard() { g_active.store(1); } ~ActiveGuard() { g_active.store(0); } } guard;
    launch_pingpong(D, seq_base, kS1Cap, mode);                    // ONE launch for the whole config
    HIP_OK(hipEventRecord(D.ev_k, D.st));
    launch_check();
    H.go.store(true, std::memory_order_release);
    th.join();                                                     // bounded: every host spin has a deadline
    g_seq1 = seq_base + H.n_total + 1;                             // includes the quit value written by s1_host
    bool ok = true;
    const WaitRes wr = wait_event_ex(D.ev_k, 5.0, false);          // the kernel exits on quit / abort value / its own timeout
    if (wr != WR_OK) { abort_run("S1 persistent kernel did not finish within 5 s of the quit"); ok = false; }
    if (!ok || g_abort.load()) { drain_after_abort(D); return O; }
    if (!d2h(D, D.rec1_h, D.rec1, sizeof(S1Rec))) return O;
    const S1Rec * r = D.rec1_h;
    O.kstatus = r->status; O.dev_bad = r->data_err; O.host_bad = H.bad;
    if (r->status == 0 || r->timeouts) { abort_run("S1 kernel wait timed out (status %u)", r->status); return O; }
    if (r->pings_done != H.n_total) { abort_run("S1 kernel served %u pings, host sent %u", r->pings_done, H.n_total); return O; }
    std::vector<double> gp, wire;                                  // measured pings are contiguous: rtt[k] pairs with proc[n_warm + k]
    for (uint32_t i = H.n_warm; i < H.n_total && i < r->pings_done; ++i) {
        const double g = (double)r->proc[i] / D.tpu;
        gp.push_back(g);
        if (i - H.n_warm < H.rtt.size()) wire.push_back(H.rtt[i - H.n_warm] - g);
    }
    O.rtt = stat_of(H.rtt); O.rtt_read = stat_of(H.rtt_read); O.gproc = stat_of(gp); O.wire = stat_of(wire); O.n = H.rtt.size(); O.ok = O.n > 0;
    if (O.dev_bad || O.host_bad) { g_rc = 1; printf("# ERROR S1 payload mismatches: device=%u host=%llu\n", O.dev_bad, ULL(O.host_bad)); }
    return O;
}

// ====================================================================== S2b: is hipStreamWaitValue32 usable on this host memory?
static void reset_res_flag(uint64_t v) { st_rel(&g_sbh->res.v, v); }
static bool test_wait_value(Dev & D, std::string & why) {
    set_phase("S2b support test");
    int can = 0;
    hipError_t e = hipDeviceGetAttribute(&can, hipDeviceAttributeCanUseStreamWaitValue, 0);
    if (e != hipSuccess) { (void)hipGetLastError(); why = std::string("attribute query failed: ") + hipGetErrorString(e); return false; }
    if (!can) { why = "hipDeviceAttributeCanUseStreamWaitValue=0"; return false; }
    st_rlx(&D.sbh->dev_timeout.v, 0);
    const uint64_t base = g_seq2;
    g_seq2 = base + 1000;                                          // this test spends seq values base+1 .. base+1000
    // (every wait below is bounded by this thread itself; the watchdog is not armed, its 3 s window is shorter than 2 x 2 s)
    // (1) an already satisfied wait must complete at once
    reset_res_flag(base + 8);
    e = hipStreamWaitValue32(D.st, &D.sbd->res.v, (uint32_t)(base + 3), hipStreamWaitValueGte, 0xFFFFFFFFu);
    if (e != hipSuccess) { (void)hipGetLastError(); why = std::string("hipStreamWaitValue32 returned ") + hipGetErrorString(e); return false; }
    launch_marker(D);
    launch_check();
    HIP_OK(hipEventRecord(D.ev_tmp, D.st));
    if (wait_event_ex(D.ev_tmp, 2.0, true) != WR_OK) {              // a satisfied wait that does not complete: release it by hand
        reset_res_flag(kAbortVal);
        const bool freed = wait_event_ex(D.ev_tmp, 2.0, false) == WR_OK;
        if (!freed) { abort_run("S2b test: a satisfied hipStreamWaitValue32 never completed, even after release"); return false; }
        reset_res_flag(base + 8);                                  // no waiter is in flight any more: the flag may be lowered again
        why = "an already satisfied hipStreamWaitValue32 did not complete (released by the host)";
        return false;
    }
    if (g_abort.load()) return false;
    // (2) an unsatisfied wait must block, and one host write must release it
    const uint64_t T = base + 100;
    e = hipStreamWaitValue32(D.st, &D.sbd->res.v, (uint32_t)T, hipStreamWaitValueGte, 0xFFFFFFFFu);
    if (e != hipSuccess) { (void)hipGetLastError(); why = std::string("hipStreamWaitValue32 returned ") + hipGetErrorString(e); return false; }
    launch_marker(D);
    launch_check();
    HIP_OK(hipEventRecord(D.ev_tmp, D.st));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (event_done(D.ev_tmp)) { why = "hipStreamWaitValue32 did not block on an unsatisfied value"; return false; }
    reset_res_flag(T);                                             // the release: the only thing the GPU is waiting for
    if (wait_event_ex(D.ev_tmp, 2.0, true) != WR_OK) {
        reset_res_flag(kAbortVal);
        if (wait_event_ex(D.ev_tmp, 2.0, false) != WR_OK) { abort_run("S2b test: hipStreamWaitValue32 not released even by the abort value"); return false; }
        reset_res_flag(T);
        why = "hipStreamWaitValue32 was not released by the host flag write (released by the abort value)";
        return false;
    }
    return !g_abort.load();
}

// ====================================================================== main
static bool time_left(const char * what) {
    if ((now_ns() - g_t_prog_start) * 1e-9 > g_o.budget_s) { printf("S_SKIP,%s,reason=time budget %.0f s used up\n", what, g_o.budget_s); return false; }
    return !g_abort.load();
}
static bool want(const std::set<std::string> & ph, const char * p) { return ph.count(p) > 0; }

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!parse_args(argc, argv, g_o)) return 2;
    g_t_prog_start = now_ns();
    calibrate_tsc();
    std::set<std::string> phases; for (auto & s : split(g_o.phases, ',')) phases.insert(s);
    for (int w : g_o.W) if (w < 0 || w > 20000) { fprintf(stderr, "W out of range 0..20000 us\n"); return 2; }
    for (int t : g_o.TC) if (t < 0 || t > 20000) { fprintf(stderr, "tc out of range 0..20000 us\n"); return 2; }
#ifdef SYNC_STUB
    g_stub_nosig_seq = g_o.stub_nosig_seq; stub::wv_mode = g_o.stub_wv;
#endif

    // ---- topology: owner-card lane thread, main (launch) thread and H2D threads each on their own physical core
    const Topo topo = read_topo();
    const std::vector<int> prim = topo.primary();
    auto cpu_of = [&](int core) -> int { return core >= 0 && core < (int)prim.size() ? prim[core] : -1; };
    std::vector<int> used{g_o.lane_core, g_o.main_core};
    const bool want_h2d = want(phases, "s3") || want(phases, "s4");
    if (want_h2d) for (int c : g_o.h2d_cores) used.push_back(c);
    for (size_t i = 0; i < used.size(); ++i) {
        if (cpu_of(used[i]) < 0) { fprintf(stderr, "SYNC_ERROR physical core %d does not exist (%zu cores)\n", used[i], prim.size()); return 2; }
        for (size_t j = i + 1; j < used.size(); ++j) if (used[i] == used[j]) { fprintf(stderr, "SYNC_ERROR physical core %d used twice\n", used[i]); return 2; }
    }
    printf("# bench_sync: lane thread -> cpu %d (core %d), launch thread -> cpu %d (core %d), H2D threads -> cpu %s\n", cpu_of(g_o.lane_core), g_o.lane_core,
           cpu_of(g_o.main_core), g_o.main_core, [&] { std::vector<int> v; for (int c : g_o.h2d_cores) v.push_back(cpu_of(c)); return join_plus(v); }().c_str());
    printf("# secs=%.1f warm=%.1f secs_base=%.1f W=%s tc=%s max_iters=%d tsc=%.1f MHz\n", g_o.secs, g_o.warm, g_o.secs_base, join_plus(g_o.W).c_str(), join_plus(g_o.TC).c_str(), g_o.max_iters, g_tsc_per_us);

    // ---- HIP setup on the unpinned main thread (HIP's own helper threads must not inherit a pin)
    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    if (ndev < 1) fatal(2, "no HIP device");
    const int ncards = std::min(g_o.cards, ndev);
    Dev D;
    D.lane_cpu = cpu_of(g_o.lane_core);
    HIP_OK(hipSetDevice(0));
    {
        hipDeviceProp_t prop; HIP_OK(hipGetDeviceProperties(&prop, 0));
        int mp = 0; HIP_OK(hipDeviceGetAttribute(&mp, hipDeviceAttributeMultiprocessorCount, 0));
        D.mem_grid = (uint32_t)std::max(1, mp) * 8u;
        printf("# owner card 0: %s %s pci_bus=%d vram=%.1f GiB mp=%d; %d device(s) visible\n", prop.name, prop.gcnArchName, prop.pciBusID, prop.totalGlobalMem / 1073741824.0, mp, ndev);
    }
    HIP_OK(hipHostMalloc((void **)&D.sbh, sizeof(SyncBlock), hipHostMallocCoherent | hipHostMallocMapped));
    if (((uintptr_t)D.sbh & 4095u) != 0) fatal(2, "sync block is not page aligned (%p)", (void *)D.sbh);
    HIP_OK(hipHostGetDevicePointer((void **)&D.sbd, D.sbh, 0));
    memset((void *)D.sbh, 0, sizeof(SyncBlock));
    g_sbh = D.sbh;
    printf("# sync block: %zu B at host %p / device %p (coherent+mapped), flags 128 B apart, x/y/payload 16 KB each\n", sizeof(SyncBlock), (void *)D.sbh, (void *)D.sbd);
    HIP_OK(hipMalloc((void **)&D.rec, sizeof(Rec)));
    HIP_OK(hipMalloc((void **)&D.rec1, sizeof(S1Rec)));
    HIP_OK(hipMalloc((void **)&D.acc, kYWords32 * sizeof(uint32_t)));
    HIP_OK(hipMemset(D.acc, 0, kYWords32 * sizeof(uint32_t)));
    HIP_OK(hipMalloc((void **)&D.dummy, 256));
    HIP_OK(hipMemset(D.dummy, 0, 256));
    HIP_OK(hipHostMalloc((void **)&D.rec_h, sizeof(Rec), hipHostMallocDefault));
    HIP_OK(hipHostMalloc((void **)&D.rec1_h, sizeof(S1Rec), hipHostMallocDefault));
    HIP_OK(hipStreamCreate(&D.st));
    for (int i = 0; i < 4; ++i) HIP_OK(hipEventCreateWithFlags(&D.ring[i], hipEventDisableTiming));
    HIP_OK(hipEventCreate(&D.ev_a)); HIP_OK(hipEventCreate(&D.ev_b));
    HIP_OK(hipEventCreateWithFlags(&D.ev_tmp, hipEventDisableTiming)); HIP_OK(hipEventCreateWithFlags(&D.ev_k, hipEventDisableTiming));
    const bool need_mem = !g_o.no_mem && want(phases, "s2");
    if (need_mem) {
        const size_t bytes = (size_t)g_o.mem_mb << 20;
        HIP_OK(hipMalloc(&D.mem, bytes));
        HIP_OK(hipMemset(D.mem, 0x5A, bytes));
        HIP_OK(hipMalloc((void **)&D.mem_out, 256));
        D.nvec_total = bytes / 16;
    }
    H2D H;
    if (want_h2d) {
        for (int c = 0; c < ncards; ++c) {
            auto cd = std::make_unique<H2DCard>();
            cd->dev = c; cd->cpu = c < (int)g_o.h2d_cores.size() ? cpu_of(g_o.h2d_cores[c]) : -1;
            HIP_OK(hipSetDevice(c));
            cd->npieces = std::max<size_t>(1, (size_t)(g_o.h2d_host_gb * 1073741824.0) / kPiece);
            HIP_OK(hipHostMalloc((void **)&cd->host, cd->npieces * kPiece, hipHostMallocPortable));
            memset(cd->host, 0x5A, cd->npieces * kPiece);
            HIP_OK(hipMalloc(&cd->d, kPiece));
            HIP_OK(hipStreamCreate(&cd->st));
            H.cards.push_back(std::move(cd));
        }
        HIP_OK(hipSetDevice(0));
        printf("# H2D: %zu cards, %zu pieces of %zu B pinned per card\n", H.cards.size(), H.cards.empty() ? (size_t)0 : H.cards[0]->npieces, kPiece);
    }
    std::thread wd(watchdog_main);
    {   // warm the runtime up (first launch, event, copy) on the unpinned thread, so no helper thread it may start inherits the pin
        launch_marker(D); launch_check();
        HIP_OK(hipEventRecord(D.ev_tmp, D.st));
        if (wait_event_ex(D.ev_tmp, 30.0, false) != WR_OK) fatal(5, "first kernel launch did not complete within 30 s");
    }
    if (!pin_to_cpu(cpu_of(g_o.main_core))) fprintf(stderr, "main: sched_setaffinity(%d) failed\n", cpu_of(g_o.main_core));

    // ---- clock + bandwidth calibration, then a short selftest of both protocols (also loads every kernel's code object)
    bool good = calibrate_wallclock(D);
    if (good) good = calibrate_poll(D);
    if (good && need_mem) good = calibrate_mem(D);
    if (good) {
        S1Out so = run_s1(D, 3, 0.1, 0.2);
        if (so.ok) printf("# selftest S1 (both payloads): n=%zu rtt_med=%.2f us, kernel status %u: ok\n", so.n, so.rtt.med, so.kstatus);
        good = so.ok;
    }
    if (good) {
        PatCfg c; c.sync = true; c.W_us = 0; c.tc_us = 0; c.warm = 0.1; c.secs = 0.2;
        PatRes pr = run_pattern(D, c);
        if (pr.ok) printf("# selftest S2 (spin kernel, W=0 tc=0): n=%llu pattern_us=%.2f data_err=%u: ok\n", ULL(pr.n), pr.host_us, pr.data_err);
        good = pr.ok;
    }
    if (!good) fatal(5, "selftest failed");

    // ================================================================= S1
    if (want(phases, "s1")) {
        double none_med = NAN; std::map<std::string, double> med;
        for (const std::string & m : g_o.s1_modes) {
            if (!time_left("S1")) break;
            const uint32_t mode = m == "none" ? 0u : m == "h2g" ? 1u : m == "g2h" ? 2u : m == "both" ? 3u : 99u;
            if (mode == 99u) { fprintf(stderr, "bad --s1-modes entry %s\n", m.c_str()); continue; }
            const char * label = mode == 0 ? "0" : mode == 1 ? "h2g_16K" : mode == 2 ? "g2h_16K" : "both_16K";
            S1Out o = run_s1(D, mode, g_o.warm, g_o.secs);
            if (!o.ok) { if (!g_abort.load()) { printf("# ERROR S1 payload=%s: no measured pings\n", label); g_rc = 1; } break; }
            printf("S1,payload=%s,n=%zu,rtt_us_med=%.2f,rtt_us_p99=%.2f,rtt_us_min=%.2f,rtt_us_mean=%.2f,rtt_read_us_med=%.2f,rtt_read_us_p99=%.2f,gpu_proc_us_med=%.2f,gpu_proc_us_p99=%.2f,wire_us_med=%.2f,wire_us_p99=%.2f,kernel_status=%s,data_err=%llu\n",
                   label, o.n, o.rtt.med, o.rtt.p99, o.rtt.min, o.rtt.mean, o.rtt_read.med, o.rtt_read.p99, o.gproc.med, o.gproc.p99, o.wire.med, o.wire.p99, o.kstatus == 3 ? "quit" : "max_pings", ULL(o.host_bad + o.dev_bad));
            med[label] = o.rtt_read.med; if (mode == 0) none_med = o.rtt.med;
        }
        if (!std::isnan(none_med)) {
            printf("S1delta,g2h_16K_minus_0_us=%.2f,h2g_16K_minus_0_us=%.2f,both_16K_minus_0_us=%.2f\n", med.count("g2h_16K") ? med["g2h_16K"] - none_med : NAN,
                   med.count("h2g_16K") ? med["h2g_16K"] - none_med : NAN, med.count("both_16K") ? med["both_16K"] - none_med : NAN);
        }
    }

    // ================================================================= S2
    if (want(phases, "s2")) {
        for (int work = 0; work < (g_o.no_mem ? 1 : 2) && !g_abort.load(); ++work) {
            if (work == 1 && !D.mem_ok) { printf("S2,work=mem,unsupported,reason=memory-bandwidth calibration failed\n"); break; }
            for (int W : g_o.W) {
                if (!time_left("S2")) break;
                if (work == 1 && W > 0 && W * 1e-6 * D.mem_gbps * 1e9 > (double)D.nvec_total * 16.0)
                    printf("# WARN S2 work=mem W=%d: %.0f MB needed at %.0f GB/s but the buffer is %d MB, the kernel streams the whole buffer once (see S2base base_us)\n", W, W * 1e-6 * D.mem_gbps * 1e3, D.mem_gbps, g_o.mem_mb);
                PatCfg b; b.sync = false; b.work_mode = work; b.W_us = W; b.warm = g_o.warm_base; b.secs = g_o.secs_base;
                PatRes br = run_pattern(D, b);
                if (!br.ok) break;
                g_base_us[{work, W}] = br.host_us;
                print_base(b, br);
                for (int tc : g_o.TC) {
                    if (!time_left("S2")) break;
                    PatCfg c; c.sync = true; c.wait_mode = 0; c.work_mode = work; c.W_us = W; c.tc_us = tc; c.warm = g_o.warm; c.secs = g_o.secs;
                    PatRes r = run_pattern(D, c);
                    if (!r.ok) break;
                    g_pat_us[{0, work, W, tc}] = r.host_us;
                    print_s2("S2", c, r, "");
                    if (r.n < 3000) printf("# WARN S2 W=%d tc=%d: only %llu iterations measured (< 3000)\n", W, tc, ULL(r.n));
                }
            }
        }
    }

    // ================================================================= S2b (and its support test, which S3/S4 reuse)
    bool wv_ok = false; std::string wv_why = "not tested";
    if ((want(phases, "s2b") || want(phases, "s3") || want(phases, "s4")) && !g_abort.load()) {
        wv_ok = test_wait_value(D, wv_why);
        if (!wv_ok && !g_abort.load()) printf("S2b,unsupported,reason=%s\n", wv_why.c_str());
        else if (wv_ok) printf("# hipStreamWaitValue32 on hipHostMalloc(coherent) memory: satisfied wait completes, unsatisfied wait blocks and is released by a host write\n");
    }
    if (want(phases, "s2b") && wv_ok) {
        for (int W : g_o.W) {
            if (!time_left("S2b")) break;
            if (!g_base_us.count({0, W})) {                       // baseline for this W (S2 may not have run)
                PatCfg b; b.sync = false; b.work_mode = 0; b.W_us = W; b.warm = g_o.warm_base; b.secs = g_o.secs_base;
                PatRes br = run_pattern(D, b); if (!br.ok) break;
                g_base_us[{0, W}] = br.host_us; print_base(b, br);
            }
            for (int tc : g_o.TC) {
                if (!time_left("S2b")) break;
                PatCfg c; c.sync = true; c.wait_mode = 1; c.work_mode = 0; c.W_us = W; c.tc_us = tc; c.warm = g_o.warm; c.secs = g_o.secs;
                PatRes r = run_pattern(D, c);
                if (!r.ok) break;
                g_pat_us[{1, 0, W, tc}] = r.host_us;
                print_s2("S2b", c, r, "");
            }
        }
    }

    // ================================================================= S3 / S4: the pattern under H2D load
    auto h2d_phase = [&](const char * tag, const std::vector<int> & active) {
        if (!want_h2d || H.cards.empty()) return;
        for (int c : active) if (c >= (int)H.cards.size()) { printf("%s,skipped,reason=card %d not present\n", tag, c); return; }
        if (!time_left(tag)) return;
        if (!g_base_us.count({0, 600})) {                          // baseline for d_vs_base (S2 may not have run)
            PatCfg b; b.sync = false; b.work_mode = 0; b.W_us = 600; b.warm = g_o.warm_base; b.secs = g_o.secs_base;
            PatRes br = run_pattern(D, b); if (!br.ok) return;
            g_base_us[{0, 600}] = br.host_us; print_base(b, br);
        }
        // reference: the same streams alone (no pattern), same window lengths
        set_phase((std::string(tag) + " h2d reference").c_str());
        h2d_start(H, active);
        std::this_thread::sleep_for(std::chrono::duration<double>(g_o.warm));
        const std::vector<uint64_t> a0 = h2d_bytes(H); const int64_t t0 = now_ns();
        std::this_thread::sleep_for(std::chrono::duration<double>(g_o.secs));
        const std::vector<uint64_t> a1 = h2d_bytes(H); const int64_t t1 = now_ns();
        h2d_stop(H);
        const std::vector<double> ref = h2d_rates(a0, a1, (t1 - t0) * 1e-9);
        printf("%sref", tag);
        for (int c : active) printf(",card%d_GBps=%.3f", c, ref[c]);
        printf("\n");
        for (int wait_mode = 0; wait_mode < 2; ++wait_mode) {
            if (wait_mode == 1 && !wv_ok) { printf("%s,wait=streamwait,unsupported,reason=%s\n", tag, wv_why.c_str()); continue; }
            if (!time_left(tag)) return;
            PatCfg c; c.sync = true; c.wait_mode = wait_mode; c.work_mode = 0; c.W_us = 600; c.tc_us = 570; c.warm = g_o.warm; c.secs = g_o.secs; c.h2d = &H;
            h2d_start(H, active);
            PatRes r = run_pattern(D, c);
            h2d_stop(H);
            if (!r.ok) return;
            const std::tuple<int, int, int, int> key{wait_mode, 0, 600, 570};
            const double quiet = g_pat_us.count(key) ? g_pat_us[key] - 600.0 : NAN;
            char extra[512]; int n = 0;
            for (int cidx = 0; cidx < (int)H.cards.size(); ++cidx) {
                const bool act = std::find(active.begin(), active.end(), cidx) != active.end();
                n += snprintf(extra + n, sizeof extra - n, "%scard%d_GBps=%.3f", n ? "," : "", cidx, act ? r.gbps[cidx] : 0.0);
                if (act) n += snprintf(extra + n, sizeof extra - n, ",card%d_vs_ref=%.3f", cidx, ref[cidx] > 0 ? r.gbps[cidx] / ref[cidx] : 0.0);
            }
            n += snprintf(extra + n, sizeof extra - n, ",d_quiet_us=%.2f,d_delta_us=%.2f", quiet, (r.host_us - 600.0) - quiet);
            print_s2(tag, c, r, extra);
        }
    };
    if (want(phases, "s3") && !g_abort.load()) { std::vector<int> all; for (int c = 0; c < (int)H.cards.size(); ++c) all.push_back(c); h2d_phase("S3", all); }
    if (want(phases, "s4") && !g_abort.load()) h2d_phase("S4", g_o.s4_cards);

    // ================================================================= teardown
    if (g_abort.load()) {                                         // clean shutdown after a timeout: stream already drained / released
        const bool drained = drain_after_abort(D);
        h2d_stop(H);
        g_wd_stop.store(1); wd.join();
        printf("# abort: stream %s\n", drained ? "drained" : "NOT drained");
        printf("SYNC_DONE rc=5\n"); fflush(stdout);
        std::_Exit(5);                                            // no HIP teardown after an abort
    }
    g_wd_stop.store(1); wd.join();
    h2d_stop(H);
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamSynchronize(D.st));
    for (auto & c : H.cards) { HIP_OK(hipSetDevice(c->dev)); HIP_OK(hipStreamDestroy(c->st)); HIP_OK(hipFree(c->d)); HIP_OK(hipHostFree(c->host)); }
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamDestroy(D.st));
    if (D.mem) { HIP_OK(hipFree(D.mem)); HIP_OK(hipFree(D.mem_out)); }
    HIP_OK(hipFree(D.rec)); HIP_OK(hipFree(D.rec1)); HIP_OK(hipFree(D.acc)); HIP_OK(hipFree(D.dummy));
    HIP_OK(hipHostFree(D.rec_h)); HIP_OK(hipHostFree(D.rec1_h));
    g_sbh = nullptr;
    HIP_OK(hipHostFree(D.sbh));
    printf("SYNC_DONE rc=%d\n", g_rc);
    return g_rc;
}
