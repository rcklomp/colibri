// pcie_stream_probe.cpp -- FRANKEN-ENGINE-PLAN-2026-09-15.md item X6, the
// measurement the plan's row derived instead of measured: how many bytes/s
// can be streamed from host RAM into the three RX 7900 XTX cards
// concurrently, vs one card alone, vs the CPU int4 path's DRAM ceiling.
//
// X6's own row assumed one card at a practical ~25 GB/s (of 32 nominal, not
// measured) against DRAM at the measured 21.95 GB/s, and concluded a
// per-token expert stream loses to the CPU path at batch 1. This program
// measures instead of assuming: 1, 2 and 3 GPUs pulling *different* 14 MiB
// blocks (one GLM routed expert) concurrently, pinned and pageable host
// memory, plus a 256 MiB pinned block for the large-transfer asymptote, and
// an 8-thread DRAM-read control for comparison on the same box.
//
// No kernels beyond hipMemcpyAsync are needed -- this is a bandwidth probe,
// not a compute one.
//
// Build (venv ROCm 10 recipe, see tools/hot-expert/h0d_chain.sh /
// m0c_chain.sh headers -- this box's system GCC 16 has no libstdc++-16-dev,
// so clang's autodetected GCC-install must be steered off it):
//   ROCM_ROOT=$(~/venvs/rocm/bin/rocm-sdk path --root)
//   CPLUS_INCLUDE_PATH=/usr/include/c++/15:/usr/include/x86_64-linux-gnu/c++/15 \
//     "$ROCM_ROOT/bin/hipcc" --offload-arch=gfx1100 -O2 -fopenmp \
//     pcie_stream_probe.cpp -o pcie_stream_probe
// Run:
//   LD_LIBRARY_PATH="$ROCM_ROOT/lib" OMP_NUM_THREADS=8 OMP_PLACES=cores \
//     OMP_PROC_BIND=close ./pcie_stream_probe
//
// Output: a few header/info lines, then one `ROW ...` line per measurement,
// machine-readable (grep '^ROW'):
//   ROW cfg=<1|2|3|cpu> mem=<pinned|pageable> block=<MiB> dev=<busid|all|cpu> gbps=<..> n=<..>
//
// cfg is the number of GPUs pulling concurrently (or "cpu" for the DRAM
// control). For cfg=2/3, one row per participating device (dev=<pci busid>,
// that device's own share of the concurrent run) plus one dev=all row (the
// aggregate). For cfg=1, a single dev=<pci busid> row. gbps is the median of
// n repeats (n >= 5); GB = 1e9 bytes throughout, matching the record's
// existing DRAM/PCIe figures (e.g. "DRAM at 21.95 GB/s", "~25 GB/s of 32
// nominal").
#include <hip/hip_runtime.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(x)                                                        \
  do {                                                                      \
    hipError_t _e = (x);                                                    \
    if (_e != hipSuccess) {                                                 \
      fprintf(stderr, "HIP ERROR %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(_e));                                       \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static double MIN_SECONDS = 2.0;
static const int REPS_PER_VISIT = 3;  // sequence 1,2,3,3,2,1 -> n=6 per row

// ---- per-device HIP streaming worker --------------------------------------

struct DevResult {
  double gbps = 0.0;
  long long bytes = 0;
  double secs = 0.0;
};

// Runs on its own std::thread. Allocates its own host+device buffers, waits
// at a barrier so all participating devices in this trial start their timed
// window together, then streams `block` bytes H2D back-to-back for
// >= MIN_SECONDS.
static void stream_worker(int hip_idx, size_t block, bool pinned,
                           std::atomic<int> *ready, int nparty,
                           DevResult *out) {
  HIP_CHECK(hipSetDevice(hip_idx));

  void *host_buf = nullptr;
  if (pinned) {
    HIP_CHECK(hipHostMalloc(&host_buf, block, hipHostMallocPortable));
    memset(host_buf, 0x5a, block);
  } else {
    host_buf = malloc(block);
    if (!host_buf) {
      fprintf(stderr, "malloc(%zu) failed on hip_idx=%d\n", block, hip_idx);
      exit(1);
    }
    memset(host_buf, 0x5a, block);  // fault the pages in before timing
  }

  void *dev_buf = nullptr;
  HIP_CHECK(hipMalloc(&dev_buf, block));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Warm-up copy (JIT / page-table / first-touch cost, not timed).
  HIP_CHECK(hipMemcpyAsync(dev_buf, host_buf, block, hipMemcpyHostToDevice,
                            stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Barrier: all `nparty` threads in this trial begin their timed window
  // together, so the timed window is genuinely concurrent across devices
  // for its whole duration, not staggered by per-thread setup cost.
  ready->fetch_add(1, std::memory_order_acq_rel);
  while (ready->load(std::memory_order_acquire) < nparty) {
    std::this_thread::yield();
  }

  long long n = 0;
  double t0 = now_s();
  double t = t0;
  do {
    HIP_CHECK(hipMemcpyAsync(dev_buf, host_buf, block, hipMemcpyHostToDevice,
                              stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    n++;
    t = now_s();
  } while (t - t0 < MIN_SECONDS);

  out->secs = t - t0;
  out->bytes = n * (long long)block;
  out->gbps = (double)out->bytes / out->secs / 1e9;

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dev_buf));
  if (pinned) {
    HIP_CHECK(hipHostFree(host_buf));
  } else {
    free(host_buf);
  }
}

// Runs one trial: `devs` HIP indices concurrently, each streaming `block`
// bytes (pinned or pageable) for >= MIN_SECONDS. Returns per-device results
// (same order as `devs`) plus the aggregate (sum bytes / max secs, since all
// threads target the same wall-clock window via the barrier).
struct TrialResult {
  std::vector<DevResult> per_dev;
  double agg_gbps;
};

static TrialResult run_trial(const std::vector<int> &devs, size_t block,
                              bool pinned) {
  int n = (int)devs.size();
  std::vector<DevResult> results(n);
  std::atomic<int> ready{0};
  std::vector<std::thread> threads;
  threads.reserve(n);
  for (int i = 0; i < n; i++) {
    threads.emplace_back(stream_worker, devs[i], block, pinned, &ready, n,
                          &results[i]);
  }
  for (auto &th : threads) th.join();

  long long total_bytes = 0;
  double max_secs = 0.0;
  for (auto &r : results) {
    total_bytes += r.bytes;
    max_secs = std::max(max_secs, r.secs);
  }
  TrialResult tr;
  tr.per_dev = results;
  tr.agg_gbps = (double)total_bytes / max_secs / 1e9;
  return tr;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return 0.0;
  if (n % 2 == 1) return v[n / 2];
  return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// ---- CPU DRAM-read control --------------------------------------------

// 8 threads (OMP; OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND set by the
// caller, per CLAUDE.md's rule for 8-thread numbers on this box) summing
// 14 MiB blocks out of a 4 GiB buffer for >= MIN_SECONDS. Reports aggregate
// DRAM-read GB/s.
static double cpu_dram_gbps(size_t total, size_t block) {
  char *buf = (char *)malloc(total);
  if (!buf) {
    fprintf(stderr, "cpu buffer malloc(%zu) failed\n", total);
    exit(1);
  }
  size_t nblocks = total / block;

#pragma omp parallel for schedule(static)
  for (size_t b = 0; b < nblocks; b++) {
    memset(buf + b * block, 0x5a, block);  // fault the pages in
  }

  std::atomic<long long> total_bytes{0};
  double t_start = now_s();
  double t_end = t_start;

#pragma omp parallel reduction(max : t_end)
  {
    int tid = omp_get_thread_num();
    long long local_bytes = 0;
    volatile unsigned long long sink = 0;
    size_t idx = (size_t)tid;
    double t0 = now_s();
    double t = t0;
    do {
      const unsigned long long *p =
          (const unsigned long long *)(buf + (idx % nblocks) * block);
      size_t nwords = block / sizeof(unsigned long long);
      unsigned long long s = 0;
      for (size_t w = 0; w < nwords; w++) s += p[w];
      sink += s;
      local_bytes += (long long)block;
      idx++;
      t = now_s();
    } while (t - t0 < MIN_SECONDS);
    t_end = t;
    total_bytes.fetch_add(local_bytes, std::memory_order_relaxed);
  }

  double secs = t_end - t_start;
  double gbps = (double)total_bytes.load() / secs / 1e9;
  free(buf);
  return gbps;
}

// ---- device discovery / naming -----------------------------------------

struct DevInfo {
  int hip_idx;
  std::string bus_id;
};

static std::string pci_bus_id(int hip_idx) {
  char buf[64];
  HIP_CHECK(hipDeviceGetPCIBusId(buf, sizeof(buf), hip_idx));
  std::string s(buf);
  for (auto &c : s) c = (char)tolower((unsigned char)c);
  return s;
}

// ---- row printing --------------------------------------------------------

static void print_row(const char *cfg, const char *mem, size_t block_mib,
                       const std::string &dev, double gbps, int n) {
  printf("ROW cfg=%s mem=%s block=%zu dev=%s gbps=%.3f n=%d\n", cfg, mem,
         block_mib, dev.c_str(), gbps, n);
  fflush(stdout);
}

// One (cfg, mem, block) row, run across REPS_PER_VISIT trials in this visit;
// accumulates into `dev_samples`/`agg_samples` (indexed same as `devs`) so a
// second visit later in the interleave sequence can add more samples before
// the median is taken and printed.
static void collect_visit(const std::vector<int> &devs, size_t block,
                           bool pinned,
                           std::vector<std::vector<double>> &dev_samples,
                           std::vector<double> &agg_samples) {
  for (int r = 0; r < REPS_PER_VISIT; r++) {
    TrialResult tr = run_trial(devs, block, pinned);
    for (size_t i = 0; i < devs.size(); i++)
      dev_samples[i].push_back(tr.per_dev[i].gbps);
    agg_samples.push_back(tr.agg_gbps);
  }
}

// Single-device variant of collect_visit for cfg=1's three independent
// one-card groups (no aggregate needed: the one device IS the aggregate).
static void collect_visit_single(int hip_idx, size_t block, bool pinned,
                                  std::vector<double> &samples) {
  std::vector<int> one = {hip_idx};
  for (int r = 0; r < REPS_PER_VISIT; r++) {
    TrialResult tr = run_trial(one, block, pinned);
    samples.push_back(tr.per_dev[0].gbps);
  }
}

int main(int argc, char **argv) {
  if (argc > 1) MIN_SECONDS = atof(argv[1]);

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  printf("INFO device_count=%d min_seconds=%.1f reps_per_visit=%d\n", ndev,
         MIN_SECONDS, REPS_PER_VISIT);
  if (ndev < 3) {
    fprintf(stderr, "FATAL: expected 3 HIP devices, hipGetDeviceCount=%d\n",
            ndev);
    return 1;
  }

  std::vector<DevInfo> devs;
  for (int i = 0; i < ndev; i++) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, i));
    std::string bus = pci_bus_id(i);
    devs.push_back({i, bus});
    printf("INFO hip_idx=%d pci_bus_id=%s name=%s\n", i, bus.c_str(),
           prop.name);
  }

  const size_t MIB = 1024ull * 1024ull;
  const size_t SMALL = 14 * MIB;   // one GLM routed expert
  const size_t LARGE = 256 * MIB;  // large-transfer asymptote

  // Row state, keyed by cfg (1/2/3) x combo (0=pinned14, 1=pageable14,
  // 2=pinned256). For cfg=1 there are 3 independent single-device targets;
  // for cfg=2/3 there is one grouping (HIP idx 0..k-1) with per-device +
  // aggregate rows.
  struct Combo {
    const char *mem;
    size_t block;
    bool pinned;
  };
  std::vector<Combo> combos = {
      {"pinned", SMALL, true},
      {"pageable", SMALL, false},
      {"pinned", LARGE, true},
  };

  // Per-(cfg,combo) accumulators, indexed [combo][device-position].
  // cfg=1: one independent group per device (hip idx 0, 1, 2 alone); no
  // aggregate needed, the single device IS the aggregate.
  std::vector<int> cfg1_devs = {0, 1, 2};
  std::vector<std::vector<std::vector<double>>> cfg1_samples(
      combos.size(), std::vector<std::vector<double>>(cfg1_devs.size()));

  std::vector<int> cfg2_group = {0, 1};
  std::vector<std::vector<std::vector<double>>> cfg2_per_dev(
      combos.size(), std::vector<std::vector<double>>(cfg2_group.size()));
  std::vector<std::vector<double>> cfg2_agg(combos.size());

  std::vector<int> cfg3_group = {0, 1, 2};
  std::vector<std::vector<std::vector<double>>> cfg3_per_dev(
      combos.size(), std::vector<std::vector<double>>(cfg3_group.size()));
  std::vector<std::vector<double>> cfg3_agg(combos.size());

  const char *sequence[] = {"1", "2", "3", "3", "2", "1"};
  for (const char *cfg : sequence) {
    printf("INFO visit cfg=%s\n", cfg);
    if (strcmp(cfg, "1") == 0) {
      for (size_t g = 0; g < cfg1_devs.size(); g++) {
        for (size_t c = 0; c < combos.size(); c++) {
          collect_visit_single(cfg1_devs[g], combos[c].block,
                                combos[c].pinned, cfg1_samples[c][g]);
        }
      }
    } else if (strcmp(cfg, "2") == 0) {
      for (size_t c = 0; c < combos.size(); c++) {
        collect_visit(cfg2_group, combos[c].block, combos[c].pinned,
                      cfg2_per_dev[c], cfg2_agg[c]);
      }
    } else {  // "3"
      for (size_t c = 0; c < combos.size(); c++) {
        collect_visit(cfg3_group, combos[c].block, combos[c].pinned,
                      cfg3_per_dev[c], cfg3_agg[c]);
      }
    }
  }

  // --- print cfg=1 rows ---
  for (size_t g = 0; g < cfg1_devs.size(); g++) {
    int hip_idx = cfg1_devs[g];
    for (size_t c = 0; c < combos.size(); c++) {
      double m = median(cfg1_samples[c][g]);
      print_row("1", combos[c].mem, combos[c].block / MIB,
                 devs[hip_idx].bus_id, m, (int)cfg1_samples[c][g].size());
    }
  }

  // --- print cfg=2 rows (per-device + aggregate) ---
  for (size_t c = 0; c < combos.size(); c++) {
    for (size_t i = 0; i < cfg2_group.size(); i++) {
      double m = median(cfg2_per_dev[c][i]);
      print_row("2", combos[c].mem, combos[c].block / MIB,
                 devs[cfg2_group[i]].bus_id, m,
                 (int)cfg2_per_dev[c][i].size());
    }
    double magg = median(cfg2_agg[c]);
    print_row("2", combos[c].mem, combos[c].block / MIB, "all", magg,
               (int)cfg2_agg[c].size());
  }

  // --- print cfg=3 rows (per-device + aggregate) ---
  for (size_t c = 0; c < combos.size(); c++) {
    for (size_t i = 0; i < cfg3_group.size(); i++) {
      double m = median(cfg3_per_dev[c][i]);
      print_row("3", combos[c].mem, combos[c].block / MIB,
                 devs[cfg3_group[i]].bus_id, m,
                 (int)cfg3_per_dev[c][i].size());
    }
    double magg = median(cfg3_agg[c]);
    print_row("3", combos[c].mem, combos[c].block / MIB, "all", magg,
               (int)cfg3_agg[c].size());
  }

  // --- CPU DRAM control ---
  const size_t CPU_TOTAL = 4ull * 1024ull * 1024ull * 1024ull;  // 4 GiB
  std::vector<double> cpu_samples;
  for (int r = 0; r < 2 * REPS_PER_VISIT; r++) {  // n>=5, matches GPU rows' n=6
    cpu_samples.push_back(cpu_dram_gbps(CPU_TOTAL, SMALL));
  }
  double cpu_m = median(cpu_samples);
  print_row("cpu", "pageable", SMALL / MIB, "cpu", cpu_m,
             (int)cpu_samples.size());

  printf("INFO done\n");
  return 0;
}
