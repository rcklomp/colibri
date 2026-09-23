// m4_stream.hip -- M4: streaming prefill bandwidth on three cards under
// REAL contention (design section 3.3, section 4 row M4; record sections
// PCIE-STREAM, VK-STREAM, M5, M2).
//
// What M4 closes (design section 3.3, corrected in its own rev 2): the
// PCIE-STREAM measurement (61 GB/s aggregate, three cards each streaming
// their OWN block) is not the same measurement as two cards on a shared
// upstream link each independently missing and streaming AT THE SAME TIME,
// while a third card runs alone, while all of it may or may not overlap
// with GPU compute. Until this program's output is read, the design's
// engine gives the shared-link pair a joint budget assumed at one card's
// rate (28 GB/s), not two summed; this program measures the real number so
// that assumption can be replaced.
//
// Hardware fact this program depends on (record section PCIE-STREAM,
// verified there from hipDeviceGetPCIBusId, not assumed): of the three RX
// 7900 XTX cards, PCI 0000:83:00.0 and 0000:86:00.0 share an upstream link
// (each drops from ~28 to ~18 GB/s when both stream at once); PCI
// 0000:48:00.0 does not share that link (holds ~28 GB/s regardless of what
// the other two do). This program does NOT hardcode that pairing as fact:
// it re-derives it at run time from hipDeviceGetPCIBusId on THIS run (the
// bus-id strings are matched, with the previously-measured mapping used
// only as the match target and printed as a note either way), the same
// discipline PCIE-STREAM itself used ("verified this run, not assumed").
//
// Piece sizes: 7.5 MB and 11.7 MB, modeling one DeepSeek-V4 / GLM-5.3
// routed expert's quantised weight (design section 4 row M2: GLM streams
// ~1.0-1.4 GB of such pieces per token). Per token a device streams N
// pieces sized so N * piece_bytes ~= 1 GB (decimal); N is computed at run
// time, not hardcoded, and printed.
//
// Method: PER-OP BLOCKING (hipMemcpyAsync then hipStreamSynchronize on
// that one op, timed on the host around the pair), one thread per (device,
// stream) pulling from a start barrier so a multi-party trial is genuinely
// concurrent for its whole timed window. This is deliberately the same
// method as tools/hot-expert/pcie_stream_probe.cpp (record section
// PCIE-STREAM) and the P2P arm of m1/m5_bench.hip (record section M5),
// both already validated on this hardware, rather than an event-timestamp
// pipeline invented fresh for this file: it gives both the throughput
// number (bytes / wall time) and a genuine per-piece latency sample (the
// host-measured round trip of that one op) from the same loop, and for the
// two-copy-streams-per-card arm, two independent blocking threads on two
// streams of the same device produce real concurrent DMA-queue overlap
// (each thread blocks only on ITS OWN stream) without needing cross-stream
// event bookkeeping.
//
// Units: decimal GB = 1e9 bytes / decimal MB = 1e6 bytes throughout (the
// convention m1/m5_bench.hip states and uses; this box's GiB-vs-GB mixup
// has produced a real bug before, per CLAUDE.md).
//
// Rules this file honours (task brief): BUILD ONLY. It is not run by the
// agent that wrote it -- hipGetDeviceCount and hipSetDevice(0) alone would
// already touch /dev/kfd on all three cards. No docker --device flags, no
// rocminfo/amd-smi calls anywhere in the toolchain used to build this.
// Built with hipcc --offload-arch=gfx1100 in rocm/dev-ubuntu-24.04:7.14.0-full
// (offline codegen, no GPU needed to compile). Only the orchestrating
// session, under the rig lock with the gateway's engine idle, runs the
// resulting binary (see run_m4.sh).
//
// Output: key=value lines on stdout (grep-able), plus the same data
// written as JSON via --json <path> (same convention as m1/m5_bench.hip).
// Claims no measurement -- it prints numbers; reading them is a separate
// step, done by whoever runs the binary.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- utility --

#define HIP_CHECK(cmd) do { \
    hipError_t _e = (cmd); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d: %s\n", #cmd, __FILE__, __LINE__, hipGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)floor(idx), hi = (size_t)ceil(idx);
    if (lo == hi) return v[lo];
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}
static double median_of(std::vector<double> v) { return percentile(v, 0.5); }

// Collected key=value pairs, in order, for the JSON dump.
static std::vector<std::pair<std::string, std::string>> g_kv;

static void emit_num(const std::string& key, double val) {
    printf("%s=%.6f\n", key.c_str(), val);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", val);
    g_kv.emplace_back(key, std::string(buf));
}
static void emit_int(const std::string& key, long long val) {
    printf("%s=%lld\n", key.c_str(), val);
    g_kv.emplace_back(key, std::to_string(val));
}
static void emit_str(const std::string& key, const std::string& val) {
    printf("%s=%s\n", key.c_str(), val.c_str());
    g_kv.emplace_back(key, std::string("\"") + val + "\"");
}

static void write_json(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "could not open %s for write\n", path.c_str()); return; }
    fprintf(f, "{\n");
    for (size_t i = 0; i < g_kv.size(); i++) {
        fprintf(f, "  \"%s\": %s%s\n", g_kv[i].first.c_str(), g_kv[i].second.c_str(),
                (i + 1 < g_kv.size()) ? "," : "");
    }
    fprintf(f, "}\n");
    fclose(f);
}

// Sanitizes a PCI bus id ("0000:83:00.0") into a JSON-key-safe fragment
// ("83_00_0") -- strips the domain, replaces ':' and '.' with '_'.
static std::string sanitize_bus(const std::string& bus) {
    std::string s = bus;
    size_t colon = s.find(':');
    if (colon != std::string::npos && s.substr(0, colon) == "0000") {
        s = s.substr(colon + 1);
    }
    for (auto& c : s) {
        if (c == ':' || c == '.') c = '_';
    }
    return s;
}

// ------------------------------------------------------- device discovery --

struct DevInfo {
    int hip_idx;
    std::string bus_id;   // lowercased, as hipDeviceGetPCIBusId returns it
    std::string tag;      // sanitized, for JSON keys
};

static std::vector<DevInfo> g_devs;

static std::string pci_bus_id(int hip_idx) {
    char buf[64];
    HIP_CHECK(hipDeviceGetPCIBusId(buf, sizeof(buf), hip_idx));
    std::string s(buf);
    for (auto& c : s) c = (char)tolower((unsigned char)c);
    return s;
}

// Returns the g_devs index whose bus id contains `needle`, or -1.
static int find_dev_containing(const char* needle) {
    for (size_t i = 0; i < g_devs.size(); i++) {
        if (g_devs[i].bus_id.find(needle) != std::string::npos) return (int)i;
    }
    return -1;
}

// --------------------------------------------------- compute-side kernel --

// Plain read-reduce bandwidth kernel (same shape as m1/m5_bench.hip's
// read_reduce_kernel): grid-stride sum over a buffer. Used as the
// "VRAM-bandwidth-heavy compute" arm, not as a model op -- its only job is
// to occupy the memory controller the way a real decode/prefill kernel
// would, so the copy-vs-compute contention question (item e) has a signal
// to measure.
__global__ void read_reduce_kernel(const float* __restrict__ buf, size_t n, float* __restrict__ out) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (; i < n; i += stride) acc += buf[i];
    __shared__ float sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

// ------------------------------------------------------- persistent state --

static const int MAX_STREAMS = 2;

// One card's persistent resources for a run_group() call: a single 2 GB
// pinned host buffer (task spec: "pinned host buffers, one per card, 2 GB
// each"), sliced into piece-sized offsets that cycle through the buffer so
// consecutive pieces are not the exact same host address (closer to a real
// ring of distinct experts than one hot cache line), one small device
// buffer and one HIP stream PER copy-stream-slot (so the two-streams arm
// gets genuinely independent DMA queues), and one device buffer + stream
// for the optional compute kernel.
struct DevRes {
    int hip_idx;
    void* h_pool = nullptr;
    size_t pool_bytes = 0;
    size_t pool_slots = 0;      // pool_bytes / piece_bytes, >= 1
    void* d_piece[MAX_STREAMS] = {nullptr, nullptr};
    hipStream_t copy_stream[MAX_STREAMS] = {nullptr, nullptr};
    void* d_compute_buf = nullptr;
    float* d_compute_out = nullptr;
    hipStream_t compute_stream = nullptr;
};

static const size_t HOST_POOL_BYTES = 2000000000ULL;      // 2 GB decimal, per card
static const size_t COMPUTE_BUF_BYTES = 2000000000ULL;    // 2 GB decimal, matches m5_bench's d2d_read bound

static DevRes make_devres(int hip_idx, size_t piece_bytes, int n_streams, bool with_compute) {
    DevRes r;
    r.hip_idx = hip_idx;
    HIP_CHECK(hipSetDevice(hip_idx));
    HIP_CHECK(hipHostMalloc(&r.h_pool, HOST_POOL_BYTES, hipHostMallocPortable));
    memset(r.h_pool, 0x5a, HOST_POOL_BYTES);  // fault pages in before any timing
    r.pool_bytes = HOST_POOL_BYTES;
    r.pool_slots = std::max<size_t>(1, HOST_POOL_BYTES / piece_bytes);
    for (int s = 0; s < n_streams; s++) {
        HIP_CHECK(hipMalloc(&r.d_piece[s], piece_bytes));
        HIP_CHECK(hipStreamCreate(&r.copy_stream[s]));
    }
    if (with_compute) {
        HIP_CHECK(hipMalloc(&r.d_compute_buf, COMPUTE_BUF_BYTES));
        HIP_CHECK(hipMalloc(&r.d_compute_out, sizeof(float)));
        HIP_CHECK(hipMemset(r.d_compute_buf, 1, COMPUTE_BUF_BYTES));
        HIP_CHECK(hipStreamCreate(&r.compute_stream));
    }
    return r;
}

static void free_devres(DevRes& r, int n_streams, bool with_compute) {
    HIP_CHECK(hipSetDevice(r.hip_idx));
    for (int s = 0; s < n_streams; s++) {
        (void)hipStreamDestroy(r.copy_stream[s]);
        (void)hipFree(r.d_piece[s]);
    }
    if (with_compute) {
        (void)hipStreamDestroy(r.compute_stream);
        (void)hipFree(r.d_compute_buf);
        (void)hipFree(r.d_compute_out);
    }
    (void)hipHostFree(r.h_pool);
}

// --------------------------------------------------------- copy-corr check --

// One-time sanity check per device: a piece written with a known pattern,
// copied to the device and back, must come back identical. Cheap, and
// catches a misconfigured pointer/device pair before any timed number is
// trusted (same spirit as m1/m5_bench.hip's numerics gate, scaled to what
// a plain memcpy probe needs).
static void run_copy_correctness_check(int hip_idx) {
    HIP_CHECK(hipSetDevice(hip_idx));
    const size_t N = 4 * 1024 * 1024;
    std::vector<uint8_t> src(N), dst(N);
    for (size_t i = 0; i < N; i++) src[i] = (uint8_t)(i * 2654435761u >> 24);
    void* d_buf;
    HIP_CHECK(hipMalloc(&d_buf, N));
    HIP_CHECK(hipMemcpy(d_buf, src.data(), N, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dst.data(), d_buf, N, hipMemcpyDeviceToHost));
    (void)hipFree(d_buf);
    if (memcmp(src.data(), dst.data(), N) != 0) {
        fprintf(stderr, "CHECK_FAIL: copy round trip mismatch on hip_idx=%d\n", hip_idx);
        printf("CHECK_FAIL\n");
        exit(1);
    }
}

// ------------------------------------------------------------- copy thread --

// Result of one (device, stream-slot) blocking copy loop within one rep.
struct CopyThreadResult {
    long long bytes = 0;
    double secs = 0;
    std::vector<double> piece_us;   // one sample per timed piece
    std::vector<double> token_ms;   // one sample per timed token (n_per_token pieces summed)
};

// Blocking per-op H2D loop: warm-up pieces (untimed), a start barrier
// shared with every other thread in this rep (copy AND compute), then
// n_tokens * n_per_token timed pieces, each hipMemcpyAsync + hipStreamSynchronize
// timed on the host around the pair. Pieces cycle through the device's
// shared 2 GB pinned pool at consecutive offsets (own counter per stream so
// two streams on one device touch different slices, not the same one).
static void copy_thread_fn(DevRes* dr, int stream_slot, size_t piece_bytes,
                            int n_per_token, int n_tokens, int warmup_tokens,
                            size_t offset_stride_slots,
                            std::atomic<int>* ready, int nparty,
                            CopyThreadResult* out) {
    HIP_CHECK(hipSetDevice(dr->hip_idx));
    void* d_buf = dr->d_piece[stream_slot];
    hipStream_t stream = dr->copy_stream[stream_slot];
    uint8_t* h_pool = (uint8_t*)dr->h_pool;
    size_t pool_slots = dr->pool_slots;

    // Warm-up: untimed, also primes the SDMA path.
    size_t slot = (size_t)stream_slot * offset_stride_slots;
    for (int i = 0; i < warmup_tokens * n_per_token; i++) {
        size_t off = (slot % pool_slots) * piece_bytes;
        HIP_CHECK(hipMemcpyAsync(d_buf, h_pool + off, piece_bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        slot++;
    }

    // Start barrier: every thread in this rep (copy and compute) begins its
    // timed window together.
    ready->fetch_add(1, std::memory_order_acq_rel);
    while (ready->load(std::memory_order_acquire) < nparty) {
        std::this_thread::yield();
    }

    int total_pieces = n_tokens * n_per_token;
    out->piece_us.resize(total_pieces);
    double t0 = now_s();
    for (int i = 0; i < total_pieces; i++) {
        size_t off = (slot % pool_slots) * piece_bytes;
        double p0 = now_us();
        HIP_CHECK(hipMemcpyAsync(d_buf, h_pool + off, piece_bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        out->piece_us[i] = now_us() - p0;
        slot++;
    }
    double t1 = now_s();

    out->bytes = (long long)total_pieces * (long long)piece_bytes;
    out->secs = t1 - t0;
    out->token_ms.resize(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        double sum = 0.0;
        for (int k = 0; k < n_per_token; k++) sum += out->piece_us[t * n_per_token + k];
        out->token_ms[t] = sum / 1000.0;
    }
}

// ---------------------------------------------------------- compute thread --

struct ComputeThreadResult {
    long long bytes = 0;
    long long iters = 0;
    double secs_wall = 0;
    double secs_busy = 0;
};

// Duty-cycled read_reduce_kernel: launch, sync, measure the kernel's own
// duration, then sleep the same wall time scaled so the kernel is "busy"
// for `duty` of the window (0.5 => sleeps ~1x the kernel duration after
// each launch) -- a rough model of a real decode/prefill kernel that does
// not saturate the GPU continuously. Runs until `*stop` is set (contention
// arms: set by the caller once the copy threads it shares a rep with have
// finished) or `max_secs` elapses (baseline arm: no copy threads, no
// external stop signal, so a wall-clock cap ends it).
static void compute_thread_fn(DevRes* dr, double duty, double max_secs,
                               std::atomic<bool>* stop,
                               std::atomic<int>* ready, int nparty,
                               ComputeThreadResult* out) {
    HIP_CHECK(hipSetDevice(dr->hip_idx));
    size_t n = COMPUTE_BUF_BYTES / sizeof(float);
    float* d_buf = (float*)dr->d_compute_buf;
    float* d_out = dr->d_compute_out;
    hipStream_t stream = dr->compute_stream;
    int threads = 256, blocks = 4096;

    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
    hipLaunchKernelGGL(read_reduce_kernel, dim3(blocks), dim3(threads), 0, stream, d_buf, n, d_out);
    HIP_CHECK(hipStreamSynchronize(stream));

    ready->fetch_add(1, std::memory_order_acq_rel);
    while (ready->load(std::memory_order_acquire) < nparty) {
        std::this_thread::yield();
    }

    double t0 = now_s();
    long long iters = 0;
    double busy = 0.0;
    while (true) {
        bool stopped = stop && stop->load(std::memory_order_relaxed);
        if (stopped) break;
        if (now_s() - t0 >= max_secs) break;

        double k0 = now_s();
        HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));
        hipLaunchKernelGGL(read_reduce_kernel, dim3(blocks), dim3(threads), 0, stream, d_buf, n, d_out);
        HIP_CHECK(hipStreamSynchronize(stream));
        double kdur = now_s() - k0;
        busy += kdur;
        iters++;

        double sleep_s = (duty > 0.0 && duty < 1.0) ? kdur * (1.0 - duty) / duty : 0.0;
        if (sleep_s > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
    double t1 = now_s();

    out->iters = iters;
    out->bytes = iters * (long long)COMPUTE_BUF_BYTES;
    out->secs_busy = busy;
    out->secs_wall = t1 - t0;
}

// ------------------------------------------------------------ piece config --

struct PieceCfg {
    std::string label;   // JSON-key fragment, e.g. "p7_5mb"
    size_t bytes;
    int n_per_token;      // computed: round(1e9 / bytes)
};

static PieceCfg make_piece(const std::string& label, size_t bytes) {
    PieceCfg p;
    p.label = label;
    p.bytes = bytes;
    double target = 1.0e9;  // ~1 GB (decimal) per token, per design section 4 row M2
    p.n_per_token = (int)llround(target / (double)bytes);
    if (p.n_per_token < 1) p.n_per_token = 1;
    return p;
}

// ------------------------------------------------------------ run options --

struct RunOpts {
    int n_tokens = 6;
    int warmup_tokens = 1;
    int reps = 3;
    double compute_duty = 0.5;
    double compute_max_secs = 30.0;  // safety cap for the baseline (no-stop) arm
};

// --------------------------------------------------------------- run_group --

// Runs `hip_idxs` concurrently for `opts.reps` repetitions, `n_streams_per_dev`
// independent copy streams on each device, and (if with_compute) one
// duty-cycled compute thread per device sharing the same start barrier and
// the same timed window (the compute threads run until the copy threads in
// that rep finish). Emits per-device and aggregate JSON keys under prefix
// `tag`, and one row of metadata (n_streams, device count, piece label).
static void run_group(const std::string& tag, const std::vector<int>& hip_idxs,
                       const PieceCfg& pc, int n_streams_per_dev, bool with_compute,
                       const RunOpts& opts) {
    size_t ndev = hip_idxs.size();
    std::vector<DevRes> dres;
    dres.reserve(ndev);
    for (int idx : hip_idxs) dres.push_back(make_devres(idx, pc.bytes, n_streams_per_dev, with_compute));

    // Split n_per_token across the streams on one device as evenly as
    // possible; stream 0 gets any remainder.
    int base = pc.n_per_token / n_streams_per_dev;
    int rem = pc.n_per_token % n_streams_per_dev;

    // Per-device, per-rep aggregates.
    std::vector<std::vector<double>> dev_gbps_samples(ndev);
    std::vector<std::vector<double>> dev_lat_us_pool(ndev);
    std::vector<std::vector<double>> dev_token_ms_pool(ndev);
    std::vector<std::vector<double>> agg_gbps_samples_holder(1);  // [0] = aggregate across devices
    std::vector<double>& agg_gbps_samples = agg_gbps_samples_holder[0];

    std::vector<std::vector<double>> comp_gbps_busy_samples(ndev);
    std::vector<std::vector<double>> comp_gbps_wall_samples(ndev);
    std::vector<std::vector<double>> comp_duty_samples(ndev);

    for (int rep = 0; rep < opts.reps; rep++) {
        int n_copy_threads = (int)ndev * n_streams_per_dev;
        int nparty = n_copy_threads + (with_compute ? (int)ndev : 0);
        std::atomic<int> ready{0};
        std::atomic<bool> stop{false};

        std::vector<std::thread> copy_threads;
        std::vector<CopyThreadResult> copy_results(n_copy_threads);
        int ct = 0;
        for (size_t d = 0; d < ndev; d++) {
            for (int s = 0; s < n_streams_per_dev; s++) {
                int npt = base + (s == 0 ? rem : 0);
                copy_threads.emplace_back(copy_thread_fn, &dres[d], s, pc.bytes, npt,
                                           opts.n_tokens, opts.warmup_tokens,
                                           (size_t)(1000 + s * 997),  // offset stride between streams, arbitrary co-prime-ish
                                           &ready, nparty, &copy_results[ct]);
                ct++;
            }
        }

        std::vector<std::thread> compute_threads;
        std::vector<ComputeThreadResult> compute_results(ndev);
        if (with_compute) {
            for (size_t d = 0; d < ndev; d++) {
                compute_threads.emplace_back(compute_thread_fn, &dres[d], opts.compute_duty,
                                              opts.compute_max_secs, &stop,
                                              &ready, nparty, &compute_results[d]);
            }
        }

        for (auto& th : copy_threads) th.join();
        stop.store(true, std::memory_order_relaxed);
        for (auto& th : compute_threads) th.join();

        // Aggregate this rep's copy results per device (sum bytes across
        // that device's streams, max secs across them -> that device's
        // gbps for this rep), and pool latency/token samples.
        long long total_bytes_this_rep = 0;
        double max_secs_this_rep = 0.0;
        ct = 0;
        for (size_t d = 0; d < ndev; d++) {
            long long dev_bytes = 0;
            double dev_secs = 0.0;
            for (int s = 0; s < n_streams_per_dev; s++) {
                CopyThreadResult& r = copy_results[ct++];
                dev_bytes += r.bytes;
                dev_secs = std::max(dev_secs, r.secs);
                dev_lat_us_pool[d].insert(dev_lat_us_pool[d].end(), r.piece_us.begin(), r.piece_us.end());
                dev_token_ms_pool[d].insert(dev_token_ms_pool[d].end(), r.token_ms.begin(), r.token_ms.end());
            }
            double dev_gbps = (double)dev_bytes / dev_secs / 1e9;
            dev_gbps_samples[d].push_back(dev_gbps);
            total_bytes_this_rep += dev_bytes;
            max_secs_this_rep = std::max(max_secs_this_rep, dev_secs);

            if (with_compute) {
                ComputeThreadResult& cr = compute_results[d];
                double gbps_busy = cr.secs_busy > 0 ? (double)cr.bytes / cr.secs_busy / 1e9 : 0.0;
                double gbps_wall = cr.secs_wall > 0 ? (double)cr.bytes / cr.secs_wall / 1e9 : 0.0;
                comp_gbps_busy_samples[d].push_back(gbps_busy);
                comp_gbps_wall_samples[d].push_back(gbps_wall);
                comp_duty_samples[d].push_back(cr.secs_wall > 0 ? cr.secs_busy / cr.secs_wall : 0.0);
            }
        }
        agg_gbps_samples.push_back((double)total_bytes_this_rep / max_secs_this_rep / 1e9);
    }

    // ---- emit per-device rows ----
    for (size_t d = 0; d < ndev; d++) {
        const std::string& devtag = dres[d].hip_idx >= 0 ? std::to_string(dres[d].hip_idx) : "?";
        std::string bus_tag;
        for (auto& gd : g_devs) if (gd.hip_idx == dres[d].hip_idx) bus_tag = gd.tag;
        std::string prefix = "m4_" + tag + "_" + pc.label + "_dev" + bus_tag;

        emit_num(prefix + "_gbps", median_of(dev_gbps_samples[d]));
        emit_num(prefix + "_lat_us_p50", percentile(dev_lat_us_pool[d], 0.50));
        emit_num(prefix + "_lat_us_p90", percentile(dev_lat_us_pool[d], 0.90));
        emit_num(prefix + "_token_ms_p50", percentile(dev_token_ms_pool[d], 0.50));
        emit_num(prefix + "_token_ms_p10", percentile(dev_token_ms_pool[d], 0.10));
        emit_num(prefix + "_token_ms_p90", percentile(dev_token_ms_pool[d], 0.90));

        if (with_compute) {
            emit_num(prefix + "_compute_gbps_busy", median_of(comp_gbps_busy_samples[d]));
            emit_num(prefix + "_compute_gbps_wall", median_of(comp_gbps_wall_samples[d]));
            emit_num(prefix + "_compute_duty_achieved", median_of(comp_duty_samples[d]));
        }
    }

    // ---- emit aggregate across devices ----
    std::string agg_prefix = "m4_" + tag + "_" + pc.label + "_agg";
    emit_num(agg_prefix + "_gbps", median_of(agg_gbps_samples));
    emit_int(agg_prefix + "_ndev", (long long)ndev);
    emit_int(agg_prefix + "_nstreams_per_dev", (long long)n_streams_per_dev);

    for (size_t d = 0; d < ndev; d++) free_devres(dres[d], n_streams_per_dev, with_compute);
}

// --------------------------------------------------------------- baseline --

// Compute-alone baseline (no copy contention): one device, one compute
// thread, duty-cycled the same way, for a fixed wall-clock window. Gives
// the reference gbps_busy that arm (e)'s contended numbers are compared
// against.
static void run_compute_baseline(int hip_idx, const std::string& bus_tag,
                                  double duty, double window_secs) {
    DevRes dr = make_devres(hip_idx, /*piece_bytes unused*/ 1, /*n_streams*/ 0, /*with_compute*/ true);
    std::atomic<int> ready{0};
    ComputeThreadResult res;
    std::thread th(compute_thread_fn, &dr, duty, window_secs, (std::atomic<bool>*)nullptr,
                   &ready, 1, &res);
    th.join();
    double gbps_busy = res.secs_busy > 0 ? (double)res.bytes / res.secs_busy / 1e9 : 0.0;
    double gbps_wall = res.secs_wall > 0 ? (double)res.bytes / res.secs_wall / 1e9 : 0.0;
    emit_num("m4_compute_baseline_dev" + bus_tag + "_gbps_busy", gbps_busy);
    emit_num("m4_compute_baseline_dev" + bus_tag + "_gbps_wall", gbps_wall);
    emit_num("m4_compute_baseline_dev" + bus_tag + "_duty_achieved",
              res.secs_wall > 0 ? res.secs_busy / res.secs_wall : 0.0);
    free_devres(dr, 0, true);
}

// ------------------------------------------------------------------- main --

int main(int argc, char** argv) {
    std::string json_path;
    RunOpts opts;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) opts.n_tokens = atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) opts.warmup_tokens = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) opts.reps = atoi(argv[++i]);
        else if (a == "--compute-duty" && i + 1 < argc) opts.compute_duty = atof(argv[++i]);
        else if (a == "--compute-max-secs" && i + 1 < argc) opts.compute_max_secs = atof(argv[++i]);
    }
    emit_int("m4_opt_tokens", opts.n_tokens);
    emit_int("m4_opt_warmup_tokens", opts.warmup_tokens);
    emit_int("m4_opt_reps", opts.reps);
    emit_num("m4_opt_compute_duty", opts.compute_duty);

    // ---- device discovery ----
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    emit_int("device_count", ndev);
    if (ndev < 3) {
        fprintf(stderr, "FATAL: expected 3 HIP devices, hipGetDeviceCount=%d\n", ndev);
        return 1;
    }
    for (int i = 0; i < ndev; i++) {
        DevInfo d;
        d.hip_idx = i;
        d.bus_id = pci_bus_id(i);
        d.tag = sanitize_bus(d.bus_id);
        g_devs.push_back(d);
        emit_str("dev_" + std::to_string(i) + "_pci", d.bus_id);
    }

    // ---- correctness gate, before any timing is trusted ----
    for (auto& d : g_devs) run_copy_correctness_check(d.hip_idx);
    emit_str("copy_correctness_check", "PASS");

    // ---- identify the shared-link pair and the lone card ----
    // Matched at run time from hipDeviceGetPCIBusId (not assumed), against
    // the bus ids record section PCIE-STREAM measured as sharing a link.
    int idx_shared_a = find_dev_containing("83:00.0");
    int idx_shared_b = find_dev_containing("86:00.0");
    int idx_lone = find_dev_containing("48:00.0");
    bool fallback = false;
    if (idx_shared_a < 0 || idx_shared_b < 0 || idx_lone < 0) {
        fallback = true;
        // Fall back to HIP enumeration order: record section PCIE-STREAM
        // found HIP index 0/1 (dev0/dev3 = 83:00.0/86:00.0) share a link
        // and index 2 (dev2 = 48:00.0) does not, on this box's enumeration.
        // Falling back does not claim the same hardware; it lets the probe
        // still run and the fallback is loudly recorded either way.
        idx_shared_a = 0;
        idx_shared_b = 1;
        idx_lone = 2;
    }
    emit_int("bus_id_match_fallback", fallback ? 1 : 0);
    emit_str("shared_pair_dev_a", g_devs[idx_shared_a].bus_id);
    emit_str("shared_pair_dev_b", g_devs[idx_shared_b].bus_id);
    emit_str("lone_dev", g_devs[idx_lone].bus_id);
    int hip_shared_a = g_devs[idx_shared_a].hip_idx;
    int hip_shared_b = g_devs[idx_shared_b].hip_idx;
    int hip_lone = g_devs[idx_lone].hip_idx;

    // ---- piece configs ----
    PieceCfg p_deepseek = make_piece("p7_5mb", 7500000ULL);   // ~ DeepSeek expert
    PieceCfg p_glm = make_piece("p11_7mb", 11700000ULL);       // ~ GLM expert
    std::vector<PieceCfg> pieces = {p_deepseek, p_glm};
    for (auto& pc : pieces) {
        emit_int("piece_" + pc.label + "_bytes", (long long)pc.bytes);
        emit_int("piece_" + pc.label + "_n_per_token", pc.n_per_token);
        emit_int("piece_" + pc.label + "_token_bytes", (long long)pc.n_per_token * (long long)pc.bytes);
    }

    // ---- compute-alone baselines (one per device, before any contention) ----
    for (auto& d : g_devs) {
        run_compute_baseline(d.hip_idx, d.tag, opts.compute_duty, /*window_secs*/ 3.0);
    }

    // ---- (a) one card alone, each card in turn ----
    for (auto& d : g_devs) {
        for (auto& pc : pieces) {
            run_group("single_dev" + d.tag, {d.hip_idx}, pc, 1, /*with_compute*/ false, opts);
        }
    }

    // ---- (b) the shared-link pair together ----
    for (auto& pc : pieces) {
        run_group("pair_shared", {hip_shared_a, hip_shared_b}, pc, 1, false, opts);
    }

    // ---- (c) a non-shared pair (lone card + one shared-pair member) ----
    for (auto& pc : pieces) {
        run_group("pair_nonshared", {hip_lone, hip_shared_a}, pc, 1, false, opts);
    }

    // ---- (d) all three at once ----
    for (auto& pc : pieces) {
        run_group("triple", {hip_shared_a, hip_shared_b, hip_lone}, pc, 1, false, opts);
    }

    // ---- (e) concurrent compute kernel, ~50% duty, while copying ----
    // Scope note: run for the GLM-sized piece (11.7 MB) only, on (i) each
    // card alone with its own compute thread, and (ii) all three at once --
    // the two ends of the contention spectrum this decides between. The
    // single-card and triple copy-only numbers from (a)/(d) above are the
    // baselines these are read against; run_compute_baseline() above is the
    // baseline the *_compute_gbps_busy numbers here are read against.
    for (auto& d : g_devs) {
        run_group("single_compute_dev" + d.tag, {d.hip_idx}, p_glm, 1, /*with_compute*/ true, opts);
    }
    run_group("triple_compute", {hip_shared_a, hip_shared_b, hip_lone}, p_glm, 1, true, opts);

    // ---- (f) two copy streams per card vs one, all three cards ----
    // The 1-stream number is run_group("triple", ...) above; this adds the
    // 2-stream arm for direct comparison (same tag prefix family,
    // "_2streams" suffix distinguishes it).
    for (auto& pc : pieces) {
        run_group("triple_2streams", {hip_shared_a, hip_shared_b, hip_lone}, pc, 2, false, opts);
    }

    if (!json_path.empty()) write_json(json_path);
    printf("M4_DONE\n");
    return 0;
}
