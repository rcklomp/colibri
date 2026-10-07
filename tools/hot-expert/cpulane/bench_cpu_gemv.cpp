// bench_cpu_gemv.cpp -- part A of the cpulane microbenchmark: CPU-only batch-1 quantised GEMV, weight GB/s
// against thread count, for ggml's own AVX2 dot kernels. See README.md.
//
//   A,<type>,<mode>,threads=N,<ws>,GBps=..,expert_ms=..,...
//
// type: iq4_xs | iq3_s | q6_k (pure expert of that type, rows scaled to ~11.67 MB) | glm (IQ3_S gate/up +
//       IQ4_XS down, the real GLM-5.3-Flash expert, 11 665 408 B)
// mode: whole = every thread takes WHOLE experts (task = gate + up + act + down); expert_ms = median time of
//               one task on one thread while the others run too
//       rows  = ONE expert at a time, its rows split across all threads (gate|up rows, barrier, down rows,
//               barrier); expert_ms = median time from dispatch (start barrier) to completion (end barrier)
// ws:   cache = ~2 MB pieces, re-read (whole: one private piece per thread; rows: one shared piece), the
//               compute-bound ceiling; piece_MB says how small the "expert" is, so expert_ms is NOT comparable
//               with dram
//       dram  = a 4 GB region of expert-sized pieces visited in random order, each piece read once per visit
#include "cpu_gemv.h"

#ifndef GGML_REV
#define GGML_REV "unknown"
#endif

using namespace cl;

struct Opts {
    std::vector<std::string> types{"iq4_xs", "glm", "iq3_s", "q6_k"};
    std::string mode = "both", ws = "both";
    std::vector<int> threads{1, 2, 3, 4, 5, 6, 7, 8};
    double secs = 1.0, warm = 0.3;       // whole mode: warm-up and measured window per config
    double rows_secs = 0.6;              // rows mode: minimum measured time ...
    int    min_experts = 200;            // ... and minimum number of experts
    double region_gb = 4.0;              // DRAM-resident region
    size_t cache_bytes = (size_t)2 << 20;
    std::vector<int> order{3, 4, 5, 6, 7, 0, 1, 2};   // physical-core index per thread slot (see README)
    bool thp = true;
    int fill_threads = 8;
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
        if      (a == "--types")       o.types = split(val("--types"), ',');
        else if (a == "--mode")        o.mode = val("--mode");
        else if (a == "--ws")          o.ws = val("--ws");
        else if (a == "--threads")     o.threads = parse_int_list(val("--threads"));
        else if (a == "--secs")        o.secs = atof(val("--secs"));
        else if (a == "--warm")        o.warm = atof(val("--warm"));
        else if (a == "--rows-secs")   o.rows_secs = atof(val("--rows-secs"));
        else if (a == "--min-experts") o.min_experts = atoi(val("--min-experts"));
        else if (a == "--region-gb")   o.region_gb = atof(val("--region-gb"));
        else if (a == "--cache-bytes") o.cache_bytes = (size_t)atof(val("--cache-bytes"));
        else if (a == "--order")       o.order = parse_int_list(val("--order"));
        else if (a == "--no-thp")      o.thp = false;
        else if (a == "--fill-threads") o.fill_threads = atoi(val("--fill-threads"));
        else {
            fprintf(stderr,
                "usage: %s [--types iq4_xs,glm,iq3_s,q6_k] [--mode whole|rows|both] [--ws cache|dram|both] [--threads 1-8]\n"
                "          [--secs 1.0] [--warm 0.3] [--rows-secs 0.6] [--min-experts 200] [--region-gb 4] [--cache-bytes 2097152]\n"
                "          [--order 3,4,5,6,7,0,1,2] [--no-thp] [--fill-threads 8]\n", argv[0]);
            return false;
        }
    }
    if (o.mode != "whole" && o.mode != "rows" && o.mode != "both") { fprintf(stderr, "bad --mode\n"); return false; }
    if (o.ws != "cache" && o.ws != "dram" && o.ws != "both") { fprintf(stderr, "bad --ws\n"); return false; }
    return true;
}

struct Res { double gbps = 0, med_ms = 0, p90_ms = 0, eps = 0; size_t n = 0; };

// ---- whole mode: T workers take whole experts; rate = sum of completed bytes over the measured window
static Res run_whole(const Expert & ex, const uint8_t * base, size_t npieces, bool priv, const std::vector<int> & cpus,
                     double warm, double secs) {
    const int T = (int)cpus.size();
    std::vector<uint32_t> seq;
    if (!priv) seq = make_seq(npieces, 4, 4242);
    std::atomic<uint64_t> next{0};
    std::atomic<bool> stop{false}, go{false};
    std::atomic<int> ready{0};
    WholeCfg cfg;
    cfg.ex = &ex; cfg.base = base; cfg.seq = seq.data(); cfg.seq_len = seq.size(); cfg.next = &next;
    cfg.private_piece = priv; cfg.record = true; cfg.stop = &stop; cfg.go = &go; cfg.ready = &ready;
    std::vector<WorkerStat> st(T);
    for (auto & s : st) s.lat.reserve(1 << 16);
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(whole_worker, t, cpus[t], std::cref(cfg), std::ref(st[t]));
    while (ready.load() < T) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(warm));
    std::vector<uint64_t> b0(T), b1(T);
    for (int t = 0; t < T; ++t) b0[t] = st[t].bytes.load();
    const int64_t t0 = now_ns();
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    for (int t = 0; t < T; ++t) b1[t] = st[t].bytes.load();
    const int64_t t1 = now_ns();
    stop.store(true);
    for (auto & x : th) x.join();
    Res r;
    uint64_t tot = 0; for (int t = 0; t < T; ++t) tot += b1[t] - b0[t];
    const double dt = (t1 - t0) * 1e-9;
    r.gbps = tot / dt / 1e9;
    r.eps = tot / (double)ex.bytes / dt;
    std::vector<double> d;
    for (int t = 0; t < T; ++t) for (auto & l : st[t].lat) if (l.first >= t0 && l.first <= t1) d.push_back(l.second * 1e-6);
    r.n = d.size(); r.med_ms = percentile(d, 0.5); r.p90_ms = percentile(d, 0.9);
    return r;
}

// ---- rows mode: ONE expert at a time across all T threads
static void rows_pair(const Mat & a, const Mat & b, const uint8_t * piece, const void * q8, float * oa, float * ob, int lo, int hi) {
    const int R = a.rows;   // gate and up have the same rows; the work range [lo, hi) runs over gate rows then up rows
    if (lo < R) gemv(a, piece, q8, oa, lo, std::min(hi, R));
    if (hi > R) gemv(b, piece, q8, ob, std::max(lo, R) - R, hi - R);
}

static Res run_rows(const Expert & ex, const uint8_t * base, const std::vector<uint32_t> & seq, const std::vector<int> & cpus,
                    double min_secs, int min_experts) {
    const int T = (int)cpus.size();
    Outs outs(ex);                          // shared: every thread writes its own row range
    SpinBarrier bar(T);
    std::atomic<bool> done{false};
    std::vector<double> durs; durs.reserve(1 << 20);
    int64_t first_t0 = 0, last_t1 = 0;      // written by thread 0 only, read after join
    const int warm_iters = 10;
    const size_t cap = (size_t)1 << 20;
    auto body = [&](int tid) {
        if (!pin_to_cpu(cpus[tid])) fprintf(stderr, "rows thread %d: sched_setaffinity(%d) failed\n", tid, cpus[tid]);
        Acts acts(ex);
        const int total1 = 2 * ex.gate.rows;
        const int lo1 = (int)((int64_t)total1 * tid / T), hi1 = (int)((int64_t)total1 * (tid + 1) / T);
        const int lo2 = (int)((int64_t)ex.down.rows * tid / T), hi2 = (int)((int64_t)ex.down.rows * (tid + 1) / T);
        size_t it = 0;
        for (;;) {
            bar.wait();                                   // A: start of one expert
            if (done.load(std::memory_order_acquire)) break;
            int64_t t0 = 0; if (tid == 0) t0 = now_ns();
            const uint8_t * piece = base + (size_t)seq[it % seq.size()] * ex.stride;
            rows_pair(ex.gate, ex.up, piece, acts.qx.p, outs.g.p, outs.u.p, lo1, hi1);
            bar.wait();                                   // B: gate and up complete
            make_h(ex, outs.g.p, outs.u.p, acts);         // every thread builds the full activation (a few us)
            gemv(ex.down, piece, acts.qh.p, outs.o.p, lo2, hi2);
            bar.wait();                                   // C: down complete = expert finished
            if (tid == 0) {
                const int64_t t1 = now_ns();
                if ((int)it >= warm_iters) {
                    if (durs.empty()) first_t0 = t0;
                    durs.push_back((t1 - t0) * 1e-6);
                    last_t1 = t1;
                    if ((durs.size() >= (size_t)min_experts && (last_t1 - first_t0) * 1e-9 >= min_secs) || durs.size() >= cap)
                        done.store(true, std::memory_order_release);
                }
            }
            ++it;
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(body, t);
    for (auto & x : th) x.join();
    Res r;
    r.n = durs.size();
    const double wall = (last_t1 - first_t0) * 1e-9;
    r.gbps = r.n * (double)ex.bytes / wall / 1e9;
    r.eps = r.n / wall;
    r.med_ms = percentile(durs, 0.5); r.p90_ms = percentile(durs, 0.9);
    return r;
}

static void print_line(const char * type, const char * mode, int T, const char * ws, const Res & r, const Expert & ex,
                       const std::vector<int> & cpus) {
    printf("A,%s,%s,threads=%d,%s,GBps=%.3f,expert_ms=%.4f,expert_ms_p90=%.4f,experts_per_s=%.1f,piece_MB=%.3f,n=%zu,cpus=%s\n",
           type, mode, T, ws, r.gbps, r.med_ms, r.p90_ms, r.eps, ex.bytes / 1e6, r.n, join_plus(cpus).c_str());
    fflush(stdout);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Opts o;
    if (!parse_args(argc, argv, o)) return 2;
    cpulane_ggml_init();

    const Topo topo = read_topo();
    const std::vector<int> prim = topo.primary();
    std::vector<int> slot_cpu;                       // thread slot -> hardware thread, one per physical core
    for (int idx : o.order) if (idx >= 0 && idx < (int)prim.size()) slot_cpu.push_back(prim[idx]);
    int maxT = 0; for (int t : o.threads) maxT = std::max(maxT, t);
    if ((int)slot_cpu.size() < maxT) { fprintf(stderr, "need %d distinct physical cores, have %zu in --order\n", maxT, slot_cpu.size()); return 2; }
    for (int t : o.threads) if (t < 1) { fprintf(stderr, "bad thread count\n"); return 2; }

    printf("# cpulane part A. ggml %s (llama-glm53), kernels ggml_vec_dot_{iq4_xs,iq3_s,q6_K}_q8_K x86 AVX2, nrc=1, Q8_K activations\n", GGML_REV);
    printf("# physical cores (first hw thread of each):");
    for (size_t i = 0; i < topo.cores.size(); ++i) { printf(" core%zu=cpu", i); for (size_t k = 0; k < topo.cores[i].size(); ++k) printf("%s%d", k ? "/" : "", topo.cores[i][k]); }
    printf("\n# thread slots -> cpu:");
    for (size_t i = 0; i < slot_cpu.size(); ++i) printf(" %zu=%d", i + 1, slot_cpu[i]);
    printf("\n# whole: secs=%.2f warm=%.2f | rows: >=%d experts and >=%.2f s | dram region %.2f GB, cache piece %.2f MB, thp=%d\n",
           o.secs, o.warm, o.min_experts, o.rows_secs, o.region_gb, o.cache_bytes / 1e6, (int)o.thp);

    const bool do_cache = o.ws != "dram", do_dram = o.ws != "cache";
    const bool do_whole = o.mode != "rows", do_rows = o.mode != "whole";

    Region dram;
    if (do_dram) {
        const size_t want = (size_t)(o.region_gb * (double)((size_t)1 << 30));
        if (!region_alloc(dram, want, o.thp)) { fprintf(stderr, "mmap of %zu bytes failed\n", want); return 3; }
    }
    Region cache;
    {
        // maxT pieces of up to ~cache_bytes (+ stride rounding) each
        const size_t per = o.cache_bytes + (size_t)65536;
        if (!region_alloc(cache, per * (size_t)maxT, false)) { fprintf(stderr, "cache region mmap failed\n"); return 3; }
    }

    int rc = 0;
    for (const std::string & type : o.types) {
        Expert ed, ec;
        if (!make_expert(type, kExpertBytes, ed) || !make_expert(type, o.cache_bytes, ec)) { fprintf(stderr, "unknown type %s\n", type.c_str()); return 2; }

        {   // self-check on one filled piece: the kernels run, results are finite and non-trivial
            ABuf<uint8_t> one(ed.stride);
            fill_piece(ed, one.p, 1);
            Outs outs(ed); Acts acts(ed);
            run_expert_whole(ed, one.p, outs, acts);
            double s = 0; for (int i = 0; i < ed.down.rows; ++i) s += std::fabs(outs.o.p[i]);
            const bool fin = all_finite(outs.g.p, ed.gate.rows) && all_finite(outs.u.p, ed.up.rows) && all_finite(outs.o.p, ed.down.rows);
            printf("# selfcheck %-6s expert=%zu B (gate/up %d rows %s, down %d rows %s) g[0]=%.4g u[0]=%.4g o[0]=%.4g mean|o|=%.4g finite=%d\n",
                   type.c_str(), ed.bytes, ed.gate.rows, qname(ed.gate.t), ed.down.rows, qname(ed.down.t),
                   outs.g.p[0], outs.u.p[0], outs.o.p[0], s / ed.down.rows, (int)fin);
            if (!fin) { fprintf(stderr, "non-finite output in selfcheck of %s\n", type.c_str()); rc = 1; }
        }

        if (do_cache) fill_region(ec, cache.base, maxT, 4);
        size_t npieces = 0;
        std::vector<uint32_t> seq_dram;
        if (do_dram) {
            npieces = dram.bytes / ed.stride;
            const int64_t f0 = now_ns();
            fill_region(ed, dram.base, npieces, o.fill_threads);
            printf("# %s: filled %zu pieces of %zu B in %.2f s, AnonHugePages=%ld kB\n", type.c_str(), npieces, ed.bytes, (now_ns() - f0) * 1e-9, anon_huge_kb());
            seq_dram = make_seq(npieces, 8, 99);
        }
        const std::vector<uint32_t> seq_one{0};

        for (int pass = 0; pass < 2; ++pass) {
            const bool is_cache = pass == 0;
            if (is_cache ? !do_cache : !do_dram) continue;
            const Expert & ex = is_cache ? ec : ed;
            const char * wsn = is_cache ? "cache" : "dram";
            const uint8_t * base = is_cache ? cache.base : dram.base;
            for (int mpass = 0; mpass < 2; ++mpass) {
                if (mpass == 0 ? !do_whole : !do_rows) continue;
                for (int T : o.threads) {
                    std::vector<int> cpus(slot_cpu.begin(), slot_cpu.begin() + T);
                    Res r;
                    if (mpass == 0) r = run_whole(ex, base, is_cache ? (size_t)T : npieces, is_cache, cpus, o.warm, o.secs);
                    else            r = run_rows(ex, base, is_cache ? seq_one : seq_dram, cpus, o.rows_secs, o.min_experts);
                    print_line(type.c_str(), mpass == 0 ? "whole" : "rows", T, wsn, r, ex, cpus);
                }
            }
        }
    }
    printf("A_DONE rc=%d\n", rc);
    return rc;
}
