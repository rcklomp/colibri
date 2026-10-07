// bench_h2d_cpu.hip -- part B of the cpulane microbenchmark: the CONCURRENT pair. Three cards stream pinned
// host RAM over PCIe (H2D) while CPU threads run the part-A expert GEMV straight from a separate host region;
// both draw on the same DDR controllers. See README.md.
//
// STARTING THIS PROGRAM INITIALISES THE GPUs. Run it only through run_cpulane.sh (rig lock held, VRAM idle).
//
// Phases (each = warm-up + measured window, per-thread byte counters sampled at the window edges):
//   B1  each card alone                         B1,card=<c>,GBps=..
//   B2  all cards at once                       B2,card0_GBps=..,card1_GBps=..,card2_GBps=..,agg_GBps=..
//   per --qtypes entry and per N in --workers:
//   B4  N CPU workers alone (control)           B4,N=<n>,cpu_GBps=..,qtype=..
//   B3  all cards + N CPU workers               B3,N=<n>,card0_GBps=..,card1_GBps=..,card2_GBps=..,agg_GBps=..,cpu_GBps=..
//
// H2D host threads: physical cores 0,1,2 (one per card). CPU workers: physical cores 3..7, then (N > 5 only)
// the SMT siblings of cores 0,1,2 (smt_shared=1 in the result line). Every pinned thread is created after all
// HIP initialisation, which happens on the unpinned main thread.
#include <hip/hip_runtime.h>

#include "cpu_gemv.h"

#ifndef GGML_REV
#define GGML_REV "unknown"
#endif

using namespace cl;

#define HIP_OK(call)                                                                                   \
    do {                                                                                               \
        hipError_t e_ = (call);                                                                        \
        if (e_ != hipSuccess) {                                                                        \
            fprintf(stderr, "B_ERROR HIP %s failed at %s:%d: %s\n", #call, __FILE__, __LINE__, hipGetErrorString(e_)); \
            fflush(stderr);                                                                            \
            std::_Exit(2);                                                                             \
        }                                                                                              \
    } while (0)

struct Opts {
    double secs = 8.0, warm = 2.0;
    double host_gb = 2.0;      // pinned host buffer per card
    double cpu_gb = 4.0;       // CPU-side expert region
    int    cards = 3;
    size_t piece = kExpertBytes;
    std::vector<std::string> qtypes{"iq4_xs", "glm"};
    std::vector<int> workers{2, 4, 6};
    bool cpu_pinned = false;   // CPU region from hipHostMalloc instead of mmap
    bool thp = true;
    bool blocking_sync = false;
    bool skip_b1 = false;
    int  fill_threads = 8;
};

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v; size_t i = 0;
    while (i <= s.size()) { size_t j = s.find(sep, i); if (j == std::string::npos) j = s.size(); if (j > i) v.push_back(s.substr(i, j - i)); i = j + 1; }
    return v;
}

static bool parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char * name) -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); } return argv[++i]; };
        if      (a == "--secs")          o.secs = atof(val("--secs"));
        else if (a == "--warm")          o.warm = atof(val("--warm"));
        else if (a == "--host-gb")       o.host_gb = atof(val("--host-gb"));
        else if (a == "--cpu-gb")        o.cpu_gb = atof(val("--cpu-gb"));
        else if (a == "--cards")         o.cards = atoi(val("--cards"));
        else if (a == "--piece-bytes")   o.piece = (size_t)atof(val("--piece-bytes"));
        else if (a == "--qtypes")        o.qtypes = split(val("--qtypes"), ',');
        else if (a == "--workers")       o.workers = parse_int_list(val("--workers"));
        else if (a == "--cpu-pinned")    o.cpu_pinned = true;
        else if (a == "--no-thp")        o.thp = false;
        else if (a == "--blocking-sync") o.blocking_sync = true;
        else if (a == "--skip-b1")       o.skip_b1 = true;
        else if (a == "--fill-threads")  o.fill_threads = atoi(val("--fill-threads"));
        else {
            fprintf(stderr,
                "usage: %s [--secs 8] [--warm 2] [--host-gb 2] [--cpu-gb 4] [--cards 3] [--piece-bytes 11665408]\n"
                "          [--qtypes iq4_xs,glm] [--workers 2,4,6] [--cpu-pinned] [--no-thp] [--blocking-sync] [--skip-b1]\n"
                "NOTE: this initialises the GPUs; run it through run_cpulane.sh only.\n", argv[0]);
            return false;
        }
    }
    return true;
}

struct Card {
    int          dev = 0;
    uint8_t *    host = nullptr;   // pinned, hipHostMallocPortable
    size_t       host_bytes = 0, npieces = 0;
    void *       d = nullptr;      // one piece of VRAM
    hipStream_t  st = nullptr;
    alignas(64) std::atomic<uint64_t> bytes{0};
};

// one host thread per card: blocking hipMemcpyAsync + stream sync of one piece at a time, cycling through the buffer
static void h2d_thread(Card * c, int cpu, size_t piece, std::atomic<bool> * stop, std::atomic<bool> * go, std::atomic<int> * ready) {
    HIP_OK(hipSetDevice(c->dev));                       // per thread entry (and again per call below)
    if (!pin_to_cpu(cpu)) fprintf(stderr, "h2d card %d: sched_setaffinity(%d) failed\n", c->dev, cpu);
    ready->fetch_add(1);
    while (!go->load(std::memory_order_acquire)) cpu_relax();
    size_t i = (size_t)c->dev * 13;
    while (!stop->load(std::memory_order_relaxed)) {
        HIP_OK(hipSetDevice(c->dev));                   // HIP's current device is not something to rely on across threads
        HIP_OK(hipMemcpyAsync(c->d, c->host + (i % c->npieces) * piece, piece, hipMemcpyHostToDevice, c->st));
        HIP_OK(hipStreamSynchronize(c->st));
        c->bytes.fetch_add(piece, std::memory_order_relaxed);
        ++i;
    }
}

struct CpuSide {
    Expert   ex;
    uint8_t * base = nullptr;
    size_t   npieces = 0;
    std::vector<uint32_t> seq;
    std::atomic<uint64_t> next{0};
};

struct PhaseResult {
    std::vector<double> card;   // GB/s per ACTIVE card, in the order of `active`
    double agg = 0, cpu = 0, cpu_eps = 0;
};

static PhaseResult run_phase(std::vector<Card> & cards, const std::vector<int> & active, const std::vector<int> & h2d_cpus,
                             CpuSide * cs, const std::vector<int> & wcpus, size_t piece, double warm, double secs) {
    std::atomic<bool> stop{false}, go{false};
    std::atomic<int> ready{0};
    const int nw = cs ? (int)wcpus.size() : 0;
    WholeCfg cfg;
    std::vector<WorkerStat> wst(nw);
    if (cs) {
        cfg.ex = &cs->ex; cfg.base = cs->base; cfg.seq = cs->seq.data(); cfg.seq_len = cs->seq.size(); cfg.next = &cs->next;
        cfg.private_piece = false; cfg.record = false; cfg.stop = &stop; cfg.go = &go; cfg.ready = &ready;
    }
    std::vector<std::thread> th;
    for (int c : active) th.emplace_back(h2d_thread, &cards[c], h2d_cpus[c], piece, &stop, &go, &ready);
    for (int t = 0; t < nw; ++t) th.emplace_back(whole_worker, t, wcpus[t], std::cref(cfg), std::ref(wst[t]));
    const int total = (int)active.size() + nw;
    while (ready.load() < total) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(warm));
    std::vector<uint64_t> cb0(active.size()), cb1(active.size()), wb0(nw), wb1(nw);
    for (size_t i = 0; i < active.size(); ++i) cb0[i] = cards[active[i]].bytes.load();
    for (int t = 0; t < nw; ++t) wb0[t] = wst[t].bytes.load();
    const int64_t t0 = now_ns();
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    for (size_t i = 0; i < active.size(); ++i) cb1[i] = cards[active[i]].bytes.load();
    for (int t = 0; t < nw; ++t) wb1[t] = wst[t].bytes.load();
    const int64_t t1 = now_ns();
    stop.store(true);
    for (auto & x : th) x.join();
    const double dt = (t1 - t0) * 1e-9;
    PhaseResult r;
    for (size_t i = 0; i < active.size(); ++i) { r.card.push_back((cb1[i] - cb0[i]) / dt / 1e9); r.agg += r.card.back(); }
    uint64_t wtot = 0; for (int t = 0; t < nw; ++t) wtot += wb1[t] - wb0[t];
    r.cpu = wtot / dt / 1e9;
    r.cpu_eps = cs ? wtot / (double)cs->ex.bytes / dt : 0;
    return r;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Opts o;
    if (!parse_args(argc, argv, o)) return 2;
    cpulane_ggml_init();
    int rc = 0;

    // ---------------- topology: H2D threads on physical cores 0..ncards-1, workers on the rest
    const Topo topo = read_topo();
    const std::vector<int> prim = topo.primary();
    int ndev = 0;
    HIP_OK(hipGetDeviceCount(&ndev));
    const int ncards = std::min(o.cards, ndev);
    if (ncards < 1) { fprintf(stderr, "B_ERROR no HIP device\n"); return 2; }
    if ((int)prim.size() < ncards + 1) { fprintf(stderr, "B_ERROR only %zu physical cores\n", prim.size()); return 2; }
    std::vector<int> h2d_cpus;
    for (int c = 0; c < ncards; ++c) h2d_cpus.push_back(prim[c]);
    std::vector<int> wcpu_all;                                       // worker slot -> hw thread
    for (size_t i = ncards; i < prim.size(); ++i) wcpu_all.push_back(prim[i]);          // distinct physical cores
    const int n_phys_workers = (int)wcpu_all.size();
    for (int c = 0; c < ncards; ++c) { const int s = topo.sibling(prim[c]); if (s >= 0) wcpu_all.push_back(s); }   // then SMT siblings of the H2D cores
    for (int c : h2d_cpus) for (int i = 0; i < n_phys_workers; ++i)
        if (topo.core_of(c) == topo.core_of(wcpu_all[i])) { fprintf(stderr, "B_ERROR H2D core %d shared with a worker core\n", c); return 2; }

    printf("# cpulane part B. ggml %s, expert GEMV = part A whole-expert worker (ggml AVX2 dot kernels, Q8_K activations)\n", GGML_REV);
    printf("# H2D host threads -> cpu %s ; CPU workers -> cpu %s (first %d are distinct physical cores, the rest SMT siblings of the H2D cores)\n",
           join_plus(h2d_cpus).c_str(), join_plus(wcpu_all).c_str(), n_phys_workers);
    printf("# secs=%.1f warm=%.1f host_gb/card=%.2f cpu_gb=%.2f piece=%zu B cpu_region=%s thp=%d blocking_sync=%d\n",
           o.secs, o.warm, o.host_gb, o.cpu_gb, o.piece, o.cpu_pinned ? "pinned" : "plain", (int)o.thp, (int)o.blocking_sync);

    // ---------------- HIP setup (main thread, unpinned): device flags, pinned buffers, VRAM piece, stream
    std::vector<Card> cards(ncards);
    for (int c = 0; c < ncards; ++c) {
        HIP_OK(hipSetDevice(c));
        if (o.blocking_sync) HIP_OK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));
        hipDeviceProp_t prop;
        HIP_OK(hipGetDeviceProperties(&prop, c));
        printf("# card %d: %s %s pci_bus=%d vram=%.1f GiB\n", c, prop.name, prop.gcnArchName, prop.pciBusID, prop.totalGlobalMem / 1073741824.0);
        Card & cd = cards[c];
        cd.dev = c;
        cd.npieces = std::max<size_t>(1, (size_t)(o.host_gb * 1073741824.0) / o.piece);
        cd.host_bytes = cd.npieces * o.piece;
        HIP_OK(hipHostMalloc((void **)&cd.host, cd.host_bytes, hipHostMallocPortable));
        HIP_OK(hipMalloc(&cd.d, o.piece));
        HIP_OK(hipStreamCreate(&cd.st));
        Rng r(500 + c);
        uint64_t * w = (uint64_t *)cd.host;
        for (size_t i = 0; i < cd.host_bytes / 8; ++i) w[i] = r.next();   // touch every page, non-zero data
    }
    for (int c = 0; c < ncards; ++c) {   // functional check + warm up the runtime on the main thread: piece 0 arrives intact
        HIP_OK(hipSetDevice(c));
        Card & cd = cards[c];
        HIP_OK(hipMemcpyAsync(cd.d, cd.host, o.piece, hipMemcpyHostToDevice, cd.st));
        HIP_OK(hipStreamSynchronize(cd.st));
        std::vector<uint8_t> back(4096);
        HIP_OK(hipMemcpy(back.data(), cd.d, back.size(), hipMemcpyDeviceToHost));
        const bool ok = memcmp(back.data(), cd.host, back.size()) == 0;
        printf("# card %d: %zu pieces of %zu B pinned (%.2f GB), H2D roundtrip check %s\n", c, cd.npieces, o.piece, cd.host_bytes / 1e9, ok ? "ok" : "FAILED");
        if (!ok) rc = 1;
    }

    std::vector<int> all_cards;
    for (int c = 0; c < ncards; ++c) all_cards.push_back(c);
    auto gap = [] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); };

    // ---------------- B1: each card alone
    std::vector<double> b1(ncards, 0.0);
    if (!o.skip_b1) {
        for (int c = 0; c < ncards; ++c) {
            PhaseResult r = run_phase(cards, {c}, h2d_cpus, nullptr, {}, o.piece, o.warm, o.secs);
            b1[c] = r.card[0];
            printf("B1,card=%d,GBps=%.3f\n", c, r.card[0]);
            if (!(r.card[0] > 0.5)) { fprintf(stderr, "B_ERROR card %d moved < 0.5 GB/s alone\n", c); rc = 1; }
            gap();
        }
    }

    // ---------------- B2: all cards at once
    PhaseResult b2 = run_phase(cards, all_cards, h2d_cpus, nullptr, {}, o.piece, o.warm, o.secs);
    {
        printf("B2,");
        for (int c = 0; c < ncards; ++c) printf("card%d_GBps=%.3f,", c, b2.card[c]);
        double sum1 = 0; for (double v : b1) sum1 += v;
        printf("agg_GBps=%.3f", b2.agg);
        if (!o.skip_b1) printf(",sum_B1_GBps=%.3f,agg_vs_sum_B1=%.3f", sum1, sum1 > 0 ? b2.agg / sum1 : 0.0);
        printf(",h2d_exp_per_s=%.1f\n", b2.agg * 1e9 / (double)o.piece);
        if (!(b2.agg > 0.5)) { fprintf(stderr, "B_ERROR B2 aggregate < 0.5 GB/s\n"); rc = 1; }
        gap();
    }

    // ---------------- CPU side: region + B4 / B3 per quant type and worker count
    Region plain;
    uint8_t * cpu_base = nullptr;
    size_t cpu_bytes = (size_t)(o.cpu_gb * 1073741824.0);
    if (o.cpu_pinned) {
        HIP_OK(hipSetDevice(0));
        HIP_OK(hipHostMalloc((void **)&cpu_base, cpu_bytes, hipHostMallocPortable));
    } else {
        if (!region_alloc(plain, cpu_bytes, o.thp)) { fprintf(stderr, "B_ERROR mmap of %zu bytes failed\n", cpu_bytes); return 3; }
        cpu_base = plain.base;
    }

    for (const std::string & qt : o.qtypes) {
        CpuSide cs;
        if (!make_expert(qt, kExpertBytes, cs.ex)) { fprintf(stderr, "B_ERROR unknown qtype %s\n", qt.c_str()); return 2; }
        cs.base = cpu_base;
        cs.npieces = cpu_bytes / cs.ex.stride;
        const int64_t f0 = now_ns();
        fill_region(cs.ex, cs.base, cs.npieces, o.fill_threads);
        cs.seq = make_seq(cs.npieces, 8, 2024);
        printf("# %s: CPU region filled: %zu pieces of %zu B in %.2f s, AnonHugePages=%ld kB\n", qt.c_str(), cs.npieces, cs.ex.bytes, (now_ns() - f0) * 1e-9, anon_huge_kb());
        {   // self-check: finite outputs
            ABuf<uint8_t> one(cs.ex.stride);
            fill_piece(cs.ex, one.p, 3);
            Outs outs(cs.ex); Acts acts(cs.ex);
            run_expert_whole(cs.ex, one.p, outs, acts);
            if (!all_finite(outs.o.p, cs.ex.down.rows)) { fprintf(stderr, "B_ERROR non-finite GEMV output for %s\n", qt.c_str()); rc = 1; }
        }
        for (int N : o.workers) {
            if (N < 1 || N > (int)wcpu_all.size()) { fprintf(stderr, "B_ERROR N=%d needs more than the %zu worker cpus available\n", N, wcpu_all.size()); rc = 1; continue; }
            const std::vector<int> wc(wcpu_all.begin(), wcpu_all.begin() + N);
            const int smt = N > n_phys_workers ? 1 : 0;

            cs.next.store(0);
            PhaseResult b4 = run_phase(cards, {}, h2d_cpus, &cs, wc, o.piece, o.warm, o.secs);
            printf("B4,N=%d,cpu_GBps=%.3f,cpu_exp_per_s=%.1f,qtype=%s,cpus=%s,smt_shared=%d\n", N, b4.cpu, b4.cpu_eps, qt.c_str(), join_plus(wc).c_str(), smt);
            gap();

            cs.next.store(0);
            PhaseResult b3 = run_phase(cards, all_cards, h2d_cpus, &cs, wc, o.piece, o.warm, o.secs);
            printf("B3,N=%d,", N);
            for (int c = 0; c < ncards; ++c) printf("card%d_GBps=%.3f,", c, b3.card[c]);
            const double h2d_eps = b3.agg * 1e9 / (double)o.piece, b2_eps = b2.agg * 1e9 / (double)o.piece;
            printf("agg_GBps=%.3f,cpu_GBps=%.3f,qtype=%s,cpus=%s,smt_shared=%d", b3.agg, b3.cpu, qt.c_str(), join_plus(wc).c_str(), smt);
            printf(",agg_vs_B2=%.3f", b2.agg > 0 ? b3.agg / b2.agg : 0.0);
            for (int c = 0; c < ncards; ++c) printf(",card%d_vs_B2=%.3f", c, b2.card[c] > 0 ? b3.card[c] / b2.card[c] : 0.0);
            printf(",cpu_vs_B4=%.3f", b4.cpu > 0 ? b3.cpu / b4.cpu : 0.0);
            printf(",h2d_exp_per_s=%.1f,cpu_exp_per_s=%.1f,total_exp_per_s=%.1f,total_vs_B2=%.3f\n",
                   h2d_eps, b3.cpu_eps, h2d_eps + b3.cpu_eps, b2_eps > 0 ? (h2d_eps + b3.cpu_eps) / b2_eps : 0.0);
            gap();
        }
    }

    // ---------------- teardown
    if (o.cpu_pinned) { HIP_OK(hipSetDevice(0)); HIP_OK(hipHostFree(cpu_base)); } else region_free(plain);
    for (int c = 0; c < ncards; ++c) {
        HIP_OK(hipSetDevice(c));
        HIP_OK(hipStreamDestroy(cards[c].st));
        HIP_OK(hipFree(cards[c].d));
        HIP_OK(hipHostFree(cards[c].host));
    }
    printf("B_DONE rc=%d\n", rc);
    return rc;
}
