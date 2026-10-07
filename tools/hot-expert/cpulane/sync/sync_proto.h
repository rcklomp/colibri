// sync_proto.h -- the shared-memory protocol of bench_sync.hip: block layout, sequence-number rules, payload
// patterns, host-side atomics, timing and topology helpers. Included by the host code and (for the layout and the
// patterns) by the kernels. Include it AFTER <hip/hip_runtime.h> (or stub_hip.h).
//
// PROTOCOL (one owner card, one stream, one pinned block `SyncBlock`):
//   S2 layer pattern   GPU SIGNAL kernel : writes x (16 KB) and plan (64 B), fence, then sig = seq   (release, system scope)
//                      host lane thread  : sees sig >= seq, checks plan+x, spins tc us, writes y (16 KB), res = seq (release)
//                      GPU WAIT kernel   : polls res >= seq (acquire, system scope, bounded), reads y, adds it into acc
//   S1 ping-pong       host writes payload, cmd = 0, s1a = seq; kernel sees s1a >= seq, replies s1b = seq
// SEQUENCE NUMBERS: every flag has ONE writer and a strictly increasing value. seq counters (g_seq1 for s1a/s1b,
//   g_seq2 for sig/res) are global to the process and never reset: a config starts at base = counter, uses base+1 ..
//   base+n, and sets counter = base+n. So a flag left behind by an earlier config is always < any later wait target
//   and can never satisfy it. Waits use >= (a satisfied wait stays satisfied). kAbortVal (< 2^32, > every real seq) is
//   written by the abort path to release every wait; nothing runs after an abort.
// 32-bit view: hipStreamWaitValue32 compares the LOW word of the 64-bit `res` flag (x86 little endian); real seq values
//   stay below kSeqLimit < 2^32 and the high word is always 0, so low word == value.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#ifdef SYNC_STUB
#define SYNC_HD inline
#else
#define SYNC_HD __host__ __device__ inline
#endif

namespace sy {

constexpr uint32_t kXBytes   = 16384;            // activation x and result y: 16 KB each
constexpr uint32_t kXWords   = kXBytes / 8;      // 2048 u64 words
constexpr uint32_t kYWords32 = kXBytes / 4;      // 4096 u32 words
constexpr uint32_t kThreads  = 256;              // workgroup size of the SIGNAL / WAIT / ping-pong kernels
constexpr uint64_t kAbortVal = 0xFFFFFFFFull;    // releases every wait; larger than any real seq
constexpr uint64_t kSeqLimit = 0xFFFFFF00ull;    // real seq values stay below this
constexpr uint32_t kRecCap   = 65536;            // per-iteration device records per pattern config
constexpr uint32_t kS1Cap    = 1u << 20;         // max pings per S1 config
constexpr size_t   kPiece    = 11665408;         // bytes of one H2D piece (bench_h2d_cpu part B)

// every flag sits alone on a 128-byte line (the GPU's and the CPU's cache lines are 64/128 B)
struct alignas(128) Flag { uint64_t v; uint8_t pad[120]; };
struct alignas(128) Plan { uint64_t w[8]; uint8_t pad[64]; };   // 64-byte plan packet, own 128 B line
static_assert(sizeof(Flag) == 128 && sizeof(Plan) == 128, "line padding");

// Pinned, coherent, mapped host memory. Writer of each member in the comment.
struct alignas(4096) SyncBlock {
    Flag sig;                         // GPU: SIGNAL kernel, last store (release)
    Flag res;                         // host: lane thread, last store (release); abort path writes kAbortVal
    Flag s1a;                         // host: S1 ping
    Flag s1b;                         // GPU: S1 pong
    Flag cmd;                         // host: S1 command (0 = ping, 1 = quit); written before s1a
    Flag dev_timeout;                 // GPU: set to 1 by a kernel whose bounded wait timed out
    Plan plan;                        // GPU: S2 plan packet
    alignas(128) uint8_t x[kXBytes];       // GPU: S2 activation
    alignas(128) uint8_t y[kXBytes];       // host: S2 result
    alignas(128) uint8_t s1_g2h[kXBytes];  // GPU: S1 payload GPU -> host
    alignas(128) uint8_t s1_h2g[kXBytes];  // host: S1 payload host -> GPU
};

// per-iteration records in DEVICE memory (written by the kernels, copied back after a config)
struct Rec {
    uint32_t timeouts, aborted, data_err, pad;
    uint64_t t_start[kRecCap];        // SIGNAL kernel start            (wall_clock64 ticks)
    uint64_t t_sig[kRecCap];          // after the sig release store
    uint64_t t_ws[kRecCap];           // W kernel start
    uint64_t t_wb[kRecCap];           // WAIT kernel: first poll
    uint64_t t_we[kRecCap];           // WAIT kernel: flag seen (or timeout)
    uint32_t polls[kRecCap];          // WAIT kernel: number of polls
};
struct S1Rec {
    uint32_t status;                  // 0 timeout, 2 abort value seen, 3 quit command, 4 all max_pings done
    uint32_t pings_done, data_err, timeouts;
    uint32_t proc[kS1Cap];            // per ping: ticks from "A seen" to "B store issued"
};

// payload patterns: one multiply per buffer, then +index, so the host check loops vectorise / cost ~1 us per 16 KB
SYNC_HD uint64_t pat64(uint64_t seq, uint32_t w) { return seq * 0x9E3779B97F4A7C15ull + w; }
SYNC_HD uint32_t y_base(uint64_t seq) { return (uint32_t)seq * 0x9E3779B1u; }      // y word j (u32) == y_base(seq) + j

// ----------------------------------------------------------------- host-side atomics on the shared block
static_assert(std::atomic<uint64_t>::is_always_lock_free, "need lock-free 64-bit atomics");
inline uint64_t ld_acq(const uint64_t * p) { return reinterpret_cast<const std::atomic<uint64_t> *>(p)->load(std::memory_order_acquire); }
inline uint64_t ld_rlx(const uint64_t * p) { return reinterpret_cast<const std::atomic<uint64_t> *>(p)->load(std::memory_order_relaxed); }
inline void st_rel(uint64_t * p, uint64_t v) { reinterpret_cast<std::atomic<uint64_t> *>(p)->store(v, std::memory_order_release); }
inline void st_rlx(uint64_t * p, uint64_t v) { reinterpret_cast<std::atomic<uint64_t> *>(p)->store(v, std::memory_order_relaxed); }
// monotonic release store: never lowers a flag. The lane thread posts `res` with this, so a store that races with the abort
// path (which writes kAbortVal) cannot undo the release of a GPU wait.
inline void st_max_rel(uint64_t * p, uint64_t v) {
    auto * a = reinterpret_cast<std::atomic<uint64_t> *>(p);
    uint64_t cur = a->load(std::memory_order_relaxed);
    while (cur < v && !a->compare_exchange_weak(cur, v, std::memory_order_release, std::memory_order_relaxed)) {}
}

// ----------------------------------------------------------------- time
using clk = std::chrono::steady_clock;
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count(); }
inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}
inline uint64_t rdtsc_() {
#if defined(__x86_64__)
    return __rdtsc();
#else
    return (uint64_t)now_ns();
#endif
}
inline double g_tsc_per_us = 1000.0;     // ticks of rdtsc_() per microsecond (non-x86: ns clock)
inline void calibrate_tsc() {
#if defined(__x86_64__)
    const int64_t t0 = now_ns(); const uint64_t c0 = __rdtsc();
    while (now_ns() - t0 < 100000000) cpu_relax();
    const int64_t t1 = now_ns(); const uint64_t c1 = __rdtsc();
    g_tsc_per_us = (double)(c1 - c0) / ((double)(t1 - t0) * 1e-3);
#else
    g_tsc_per_us = 1000.0;
#endif
}

// ----------------------------------------------------------------- statistics
struct Stat { size_t n = 0; double min = 0, med = 0, mean = 0, p99 = 0, max = 0; };
inline Stat stat_of(std::vector<double> v) {
    Stat s; s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front(); s.max = v.back();
    const size_t m = v.size() / 2;
    s.med = (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
    size_t k = (size_t)std::ceil(0.99 * (double)v.size()); if (k < 1) k = 1;
    s.p99 = v[k - 1];
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / (double)v.size();
    return s;
}

// ----------------------------------------------------------------- topology / pinning (same helpers as cpulane/cpu_gemv.h)
inline std::vector<int> parse_int_list(const std::string & s) {   // "0,8" "1-8" "0-1,8-9"
    std::vector<int> v; size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
        std::string tok = s.substr(i, j - i);
        size_t dash = tok.find('-');
        if (dash == std::string::npos) { if (!tok.empty()) v.push_back(atoi(tok.c_str())); }
        else { int a = atoi(tok.substr(0, dash).c_str()), b = atoi(tok.substr(dash + 1).c_str()); for (int k = a; k <= b; ++k) v.push_back(k); }
        i = j + 1;
    }
    return v;
}
struct Topo {
    std::vector<std::vector<int>> cores;   // hardware threads of each physical core, cores ordered by lowest cpu id
    std::vector<int> primary() const { std::vector<int> v; for (auto & c : cores) v.push_back(c[0]); return v; }
    int core_of(int cpu) const { for (size_t i = 0; i < cores.size(); ++i) for (int c : cores[i]) if (c == cpu) return (int)i; return -1; }
};
inline Topo read_topo() {
    std::map<int, std::vector<int>> m;
#if defined(__linux__)
    const long n = sysconf(_SC_NPROCESSORS_CONF);
    for (int cpu = 0; cpu < n; ++cpu) {
        char path[128]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE * f = fopen(path, "r"); if (!f) continue;
        char line[256] = {0}; if (!fgets(line, sizeof line, f)) { fclose(f); continue; } fclose(f);
        std::vector<int> v = parse_int_list(line);
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        m[v[0]] = v;
    }
#endif
    Topo t;
    if (m.empty()) { const unsigned n2 = std::max(1u, std::thread::hardware_concurrency()); for (unsigned cpu = 0; cpu < n2; ++cpu) t.cores.push_back({(int)cpu}); }
    else for (auto & kv : m) t.cores.push_back(kv.second);
    return t;
}
inline bool pin_to_cpu(int cpu) {
#if defined(__linux__)
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof s, &s) == 0;   // pid 0 = the calling thread
#else
    (void)cpu; return true;                           // no pinning off Linux (Mac test build)
#endif
}
inline std::string join_plus(const std::vector<int> & v) {
    std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += "+"; s += std::to_string(v[i]); } return s;
}

}  // namespace sy
