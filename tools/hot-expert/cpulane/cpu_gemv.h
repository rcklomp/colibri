// cpu_gemv.h -- shared by bench_cpu_gemv.cpp (part A) and bench_h2d_cpu.hip (part B).
//
// Batch-1 quantised GEMV with ggml's own AVX2 dot kernels (ggml_vec_dot_{iq4_xs,iq3_s,q6_K}_q8_K,
// nrc = 1, the shape llama.cpp uses for one decode token), a synthetic "expert" made of three
// such matrices, a piece-cycling DRAM region, physical-core topology + pinning helpers, a spin
// barrier, and the whole-expert worker loop that both programs run.
//
// Everything is header-only and `inline`; the three ggml kernels, quantize_row_q8_K_ref and
// cpulane_ggml_init() live in the objects built from ggml sources + ggml_stubs.c (see Makefile).

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
void ggml_vec_dot_iq4_xs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K  (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);   // ggml-quants.c: what llama.cpp's quantize_row_q8_K calls
void cpulane_ggml_init(void);                                       // ggml_stubs.c: fills ggml_table_f32_f16 & co
}

namespace cl {

using clk = std::chrono::steady_clock;
inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}
inline void cpu_relax() { __builtin_ia32_pause(); }

// ------------------------------------------------------------------ quant formats
enum QType : int { Q_IQ4_XS = 0, Q_IQ3_S = 1, Q_Q6_K = 2 };
constexpr int    QK_K = 256;            // weights per super-block, all three formats
constexpr size_t Q8K_BLOCK_BYTES = 292; // block_q8_K: float d + int8 qs[256] + int16 bsums[16]

inline size_t block_bytes(QType t) { return t == Q_IQ4_XS ? 136 : t == Q_IQ3_S ? 110 : 210; }
// offset of the fp16 super-scale `d` inside a block (block_iq4_xs / block_iq3_s: first; block_q6_K: last)
inline size_t d_offset(QType t) { return t == Q_Q6_K ? 208 : 0; }
inline const char * qname(QType t) { return t == Q_IQ4_XS ? "iq4_xs" : t == Q_IQ3_S ? "iq3_s" : "q6_k"; }

typedef void (*dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
inline dot_fn dot_for(QType t) {
    return t == Q_IQ4_XS ? ggml_vec_dot_iq4_xs_q8_K : t == Q_IQ3_S ? ggml_vec_dot_iq3_s_q8_K : ggml_vec_dot_q6_K_q8_K;
}
inline size_t q8_row_bytes(int K) { return (size_t)(K / QK_K) * Q8K_BLOCK_BYTES; }

// ------------------------------------------------------------------ synthetic expert
// GLM-5.3-Flash (glm5_shapes.h): N_EMBD = 4096, N_FF_EXP = 2048; an expert is
//   gate [2048 rows x K=4096] + up [2048 x 4096]  (IQ3_S, 1760 B/row)  + down [4096 rows x K=2048] (IQ4_XS, 1088 B/row)
// = 7 208 960 + 4 456 448 = 11 665 408 B ("11.67 MB", GLM5.md section 4).
constexpr size_t kExpertBytes = 11665408;
constexpr int    kKgu = 4096, kKdown = 2048;

struct Mat {
    QType  t = Q_IQ4_XS;
    int    rows = 0, K = 0;
    size_t rb = 0;    // bytes per row
    size_t off = 0;   // offset inside the piece
    size_t bytes() const { return rb * (size_t)rows; }
};

struct Expert {
    std::string name;
    Mat    gate, up, down;
    size_t bytes = 0;    // useful bytes of one expert
    size_t stride = 0;   // piece pitch in a region (bytes rounded up to 4 KiB)
};

// name: "glm" (the real GLM mix: IQ3_S gate/up, IQ4_XS down), or a pure "iq4_xs" / "iq3_s" / "q6_k".
// Row counts are chosen so that the expert is ~target_bytes (gate = up = r rows, down = 2r rows, i.e. the
// real 1 : 1 : 2 row ratio; for "glm" and target = kExpertBytes, r = 2048 exactly).
inline bool make_expert(const std::string & name, size_t target_bytes, Expert & e) {
    QType tg, td;
    if      (name == "glm")    { tg = Q_IQ3_S;  td = Q_IQ4_XS; }
    else if (name == "iq4_xs") { tg = td = Q_IQ4_XS; }
    else if (name == "iq3_s")  { tg = td = Q_IQ3_S; }
    else if (name == "q6_k")   { tg = td = Q_Q6_K; }
    else return false;
    const size_t rb_gu = (kKgu / QK_K) * block_bytes(tg), rb_dn = (kKdown / QK_K) * block_bytes(td);
    const double per_r = 2.0 * rb_gu + 2.0 * rb_dn;
    int r = (int)std::max(1.0, std::floor((double)target_bytes / per_r + 0.5));
    e.name = name;
    e.gate = Mat{tg, r, kKgu, rb_gu, 0};
    e.up   = Mat{tg, r, kKgu, rb_gu, e.gate.bytes()};
    e.down = Mat{td, 2 * r, kKdown, rb_dn, e.gate.bytes() + e.up.bytes()};
    e.bytes  = e.gate.bytes() + e.up.bytes() + e.down.bytes();
    e.stride = (e.bytes + 4095) & ~(size_t)4095;
    return true;
}

// ------------------------------------------------------------------ rng, fill
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) { if (!s) s = 1; for (int i = 0; i < 4; ++i) next(); }
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uniform() { return (float)(next() >> 40) * (1.0f / 8388608.0f) - 1.0f; }   // [-1, 1)
};

// Random bytes shaped like valid blocks: every bit pattern is a valid quant payload for these three
// formats (IQ3_S grid index <= 511, IQ4_XS nibbles index a 16-entry table, Q6_K any), only the fp16
// super-scale `d` must be finite, so it is overwritten with a normal value in [2^-8, 2^-7).
inline void fill_piece(const Expert & e, uint8_t * p, uint64_t seed) {
    Rng r(seed);
    uint64_t * q = (uint64_t *)p;
    const size_t n8 = e.bytes / 8;
    for (size_t i = 0; i < n8; ++i) q[i] = r.next();
    for (size_t i = n8 * 8; i < e.bytes; ++i) p[i] = (uint8_t)r.next();
    for (const Mat * m : {&e.gate, &e.up, &e.down}) {
        const size_t bb = block_bytes(m->t), doff = d_offset(m->t);
        const size_t nblk = (size_t)m->rows * (m->K / QK_K);
        uint8_t * b = p + m->off + doff;
        for (size_t i = 0; i < nblk; ++i, b += bb) {
            uint16_t d = (uint16_t)(0x2400 + (r.next() & 0x3ff));
            memcpy(b, &d, 2);
        }
    }
}

// ------------------------------------------------------------------ memory region (mmap + THP advice)
struct Region {
    void *  raw = nullptr;  size_t raw_bytes = 0;
    uint8_t * base = nullptr;  size_t bytes = 0;
};
inline bool region_alloc(Region & r, size_t bytes, bool thp) {
    const size_t al = (size_t)2 << 20;
    r.raw_bytes = bytes + al;
    r.raw = mmap(nullptr, r.raw_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (r.raw == MAP_FAILED) { r.raw = nullptr; return false; }
    r.base = (uint8_t *)(((uintptr_t)r.raw + al - 1) & ~(uintptr_t)(al - 1));
    r.bytes = bytes;
    if (thp) madvise(r.base, bytes, MADV_HUGEPAGE);
    return true;
}
inline void region_free(Region & r) { if (r.raw) munmap(r.raw, r.raw_bytes); r = Region(); }

// fill npieces pieces (piece i at base + i*stride) with nthreads unpinned threads
inline void fill_region(const Expert & e, uint8_t * base, size_t npieces, int nthreads) {
    nthreads = std::max(1, std::min<int>(nthreads, (int)npieces));
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t)
        th.emplace_back([&, t] { for (size_t i = t; i < npieces; i += nthreads) fill_piece(e, base + i * e.stride, 1000 + i); });
    for (auto & x : th) x.join();
}

inline long anon_huge_kb() {
    FILE * f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

// random-order piece sequence: `epochs` back-to-back random permutations of 0..npieces-1, so a piece is
// revisited only after every other piece (reuse distance = the whole region, far beyond any cache)
inline std::vector<uint32_t> make_seq(size_t npieces, int epochs, uint64_t seed) {
    std::vector<uint32_t> seq; seq.reserve(npieces * epochs);
    Rng r(seed);
    for (int ep = 0; ep < epochs; ++ep) {
        std::vector<uint32_t> p(npieces);
        for (size_t i = 0; i < npieces; ++i) p[i] = (uint32_t)i;
        for (size_t i = npieces; i > 1; --i) std::swap(p[i - 1], p[r.next() % i]);
        seq.insert(seq.end(), p.begin(), p.end());
    }
    return seq;
}

// ------------------------------------------------------------------ topology, pinning
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
    int sibling(int cpu) const { int k = core_of(cpu); if (k < 0) return -1; for (int c : cores[k]) if (c != cpu) return c; return -1; }
};
inline Topo read_topo() {
    std::map<int, std::vector<int>> m;
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
    Topo t;
    if (m.empty()) { for (int cpu = 0; cpu < n; ++cpu) t.cores.push_back({cpu}); }
    else for (auto & kv : m) t.cores.push_back(kv.second);
    return t;
}
inline bool pin_to_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    return sched_setaffinity(0, sizeof s, &s) == 0;   // pid 0 = the calling thread
}
inline std::string join_plus(const std::vector<int> & v) {
    std::string s; for (size_t i = 0; i < v.size(); ++i) { if (i) s += "+"; s += std::to_string(v[i]); } return s;
}

// ------------------------------------------------------------------ spin barrier (threads are pinned, one per core)
class SpinBarrier {
public:
    explicit SpinBarrier(int n) : n_(n) {}
    void wait() {
        const int g = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            while (gen_.load(std::memory_order_acquire) == g) cpu_relax();
        }
    }
private:
    const int n_;
    alignas(64) std::atomic<int> count_{0};
    alignas(64) std::atomic<int> gen_{0};
};

// ------------------------------------------------------------------ GEMV
template <class T> struct ABuf {
    T * p = nullptr;
    ABuf() = default;
    ABuf(const ABuf &) = delete;
    ABuf & operator=(const ABuf &) = delete;
    explicit ABuf(size_t n) { alloc(n); }
    void alloc(size_t n) {
        free(p); p = nullptr;
        if (posix_memalign((void **)&p, 64, std::max<size_t>(n * sizeof(T), 64)) != 0) { fprintf(stderr, "ABuf: out of memory\n"); std::_Exit(2); }
        memset(p, 0, std::max<size_t>(n * sizeof(T), 64));
    }
    ~ABuf() { free(p); }
};

// outputs of one expert: g, u [gate.rows], o [down.rows]
struct Outs {
    ABuf<float> g, u, o;
    explicit Outs(const Expert & e) { g.alloc(e.gate.rows); u.alloc(e.up.rows); o.alloc(e.down.rows); }
};
// activation vectors, quantised to Q8_K like llama.cpp does before every quantised mat-vec
struct Acts {
    ABuf<float> hf;
    ABuf<uint8_t> qx, qh;
    explicit Acts(const Expert & e) {
        hf.alloc(e.down.K); qx.alloc(q8_row_bytes(e.gate.K)); qh.alloc(q8_row_bytes(e.down.K));
        ABuf<float> x(e.gate.K);
        Rng r(77);
        for (int i = 0; i < e.gate.K; ++i) x.p[i] = r.uniform();
        quantize_row_q8_K_ref(x.p, qx.p, e.gate.K);
        for (int i = 0; i < e.down.K; ++i) hf.p[i] = r.uniform();
        quantize_row_q8_K_ref(hf.p, qh.p, e.down.K);
    }
};

// rows [r0, r1) of one matrix: out[r] = dot(W[r, :], q8)   (one ggml_vec_dot call per row, nrc = 1)
inline void gemv(const Mat & m, const uint8_t * piece, const void * q8, float * out, int r0, int r1) {
    const dot_fn f = dot_for(m.t);
    const uint8_t * w = piece + m.off + (size_t)r0 * m.rb;
    for (int r = r0; r < r1; ++r, w += m.rb) f(m.K, out + r, 0, w, 0, q8, 0, 1);
}

// the activation between gate/up and down: h = clamp(g * u) (stands in for SwiGLU, no expf), then Q8_K.
// Down's K is 2048; for pure-type experts with r != 2048 rows the gate/up outputs are wrapped.
inline void make_h(const Expert & e, const float * g, const float * u, Acts & a) {
    const int R = e.gate.rows, K = e.down.K;
    for (int i = 0; i < K; ++i) {
        const int j = i % R;
        a.hf.p[i] = std::min(64.0f, std::max(-64.0f, g[j] * u[j]));
    }
    quantize_row_q8_K_ref(a.hf.p, a.qh.p, K);
}

// one whole expert on the calling thread: gate, up, activation, down
inline void run_expert_whole(const Expert & e, const uint8_t * piece, Outs & o, Acts & a) {
    gemv(e.gate, piece, a.qx.p, o.g.p, 0, e.gate.rows);
    gemv(e.up,   piece, a.qx.p, o.u.p, 0, e.up.rows);
    make_h(e, o.g.p, o.u.p, a);
    gemv(e.down, piece, a.qh.p, o.o.p, 0, e.down.rows);
}

inline bool all_finite(const float * v, int n) { for (int i = 0; i < n; ++i) if (!std::isfinite(v[i])) return false; return true; }

// ------------------------------------------------------------------ whole-expert worker (used by A "whole" mode and by B3/B4)
struct alignas(64) WorkerStat {
    std::atomic<uint64_t> bytes{0};   // useful weight bytes of the completed experts
    std::vector<std::pair<int64_t, int64_t>> lat;   // (completion time ns, duration ns); only when WholeCfg::record
};

struct WholeCfg {
    const Expert *  ex = nullptr;
    const uint8_t * base = nullptr;          // region
    const uint32_t * seq = nullptr;          // random piece order (shared across the workers) ...
    size_t seq_len = 0;
    std::atomic<uint64_t> * next = nullptr;  // ... and the shared cursor into it
    bool private_piece = false;              // true: worker t re-reads piece t only (cache-resident working set)
    bool record = false;
    std::atomic<bool> * stop = nullptr;
    std::atomic<bool> * go = nullptr;
    std::atomic<int> *  ready = nullptr;
};

inline void whole_worker(int tid, int cpu, const WholeCfg & c, WorkerStat & st) {
    if (cpu >= 0 && !pin_to_cpu(cpu)) fprintf(stderr, "whole_worker %d: sched_setaffinity(%d) failed\n", tid, cpu);
    Outs outs(*c.ex);    // allocated after pinning: first touch on this core
    Acts acts(*c.ex);
    c.ready->fetch_add(1);
    while (!c.go->load(std::memory_order_acquire)) cpu_relax();
    while (!c.stop->load(std::memory_order_relaxed)) {
        const size_t pi = c.private_piece ? (size_t)tid : (size_t)c.seq[c.next->fetch_add(1, std::memory_order_relaxed) % c.seq_len];
        const uint8_t * piece = c.base + pi * c.ex->stride;
        const int64_t t0 = now_ns();
        run_expert_whole(*c.ex, piece, outs, acts);
        const int64_t t1 = now_ns();
        st.bytes.fetch_add(c.ex->bytes, std::memory_order_relaxed);
        if (c.record) st.lat.emplace_back(t1, t1 - t0);
    }
    if (!all_finite(outs.o.p, c.ex->down.rows)) fprintf(stderr, "whole_worker %d: non-finite outputs\n", tid);
}

inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const double idx = p * (v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

}  // namespace cl
