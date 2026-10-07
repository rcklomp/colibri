// stub_hip.h -- a HOST-ONLY stand-in for the few HIP runtime calls bench_sync.hip uses, so that its host logic
// (lane thread, pacing, drain, statistics, abort path, line format) can be exercised on a machine without a GPU
// (`make stub`). Streams are worker threads running queued closures; "device" memory is host memory. It says nothing
// about real latencies and never touches a GPU. Compiled ONLY with -DSYNC_STUB.
#pragma once
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef int hipError_t;
enum { hipSuccess = 0, hipErrorInvalidValue = 1, hipErrorNotReady = 600, hipErrorNotSupported = 801 };
#define hipHostMallocDefault 0x0
#define hipHostMallocPortable 0x1
#define hipHostMallocMapped 0x2
#define hipHostMallocCoherent 0x40000000
#define hipEventDisableTiming 0x2
#define hipStreamWaitValueGte 0x0
enum hipMemcpyKind { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };
enum hipDeviceAttribute_t { hipDeviceAttributeWallClockRate, hipDeviceAttributeMultiprocessorCount, hipDeviceAttributeCanUseStreamWaitValue };
struct dim3 { unsigned x = 1, y = 1, z = 1; dim3(unsigned a = 1) : x(a) {} };

struct hipDeviceProp_t { char name[256]; char gcnArchName[256]; int pciBusID; size_t totalGlobalMem; };

namespace stub {
inline std::atomic<int> wv_mode{0};   // 0 ok, 1 attribute says unsupported, 2 wait returns at once, 3 wait ignores normal values
inline std::atomic<int> cur_dev{0};
struct Stream {
    std::mutex m; std::condition_variable cv; std::deque<std::function<void()>> q; bool stop = false; bool busy = false;
    std::thread th;
    Stream() { th = std::thread([this] { loop(); }); }
    void loop() {
        for (;;) {
            std::function<void()> f;
            { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&] { return stop || !q.empty(); }); if (q.empty() && stop) return; f = std::move(q.front()); q.pop_front(); busy = true; }
            f();
            { std::lock_guard<std::mutex> lk(m); busy = false; }
        }
    }
    void push(std::function<void()> f) { { std::lock_guard<std::mutex> lk(m); q.push_back(std::move(f)); } cv.notify_one(); }
    ~Stream() { { std::lock_guard<std::mutex> lk(m); stop = true; } cv.notify_one(); th.join(); }
};
struct Event { std::atomic<int> done{1}; std::atomic<int64_t> ns{0}; };
inline int64_t nsnow() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace stub

typedef stub::Stream * hipStream_t;
typedef stub::Event * hipEvent_t;

inline const char * hipGetErrorString(hipError_t e) { return e == hipSuccess ? "success" : e == hipErrorNotReady ? "not ready" : e == hipErrorNotSupported ? "not supported" : "stub error"; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipGetDeviceCount(int * n) { *n = 3; return hipSuccess; }
inline hipError_t hipSetDevice(int d) { stub::cur_dev = d; return hipSuccess; }
inline hipError_t hipDeviceGetAttribute(int * v, hipDeviceAttribute_t a, int) {
    *v = a == hipDeviceAttributeWallClockRate ? 100000 : a == hipDeviceAttributeMultiprocessorCount ? 96 : (stub::wv_mode.load() == 1 ? 0 : 1);
    return hipSuccess;
}
inline hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int d) {
    snprintf(p->name, sizeof p->name, "StubGPU"); snprintf(p->gcnArchName, sizeof p->gcnArchName, "stub"); p->pciBusID = d; p->totalGlobalMem = 24ull << 30; return hipSuccess;
}
inline hipError_t hipHostMalloc(void ** p, size_t n, unsigned) { *p = aligned_alloc(4096, (n + 4095) & ~(size_t)4095); if (*p) memset(*p, 0, n); return *p ? hipSuccess : hipErrorInvalidValue; }
inline hipError_t hipHostFree(void * p) { free(p); return hipSuccess; }
inline hipError_t hipHostGetDevicePointer(void ** d, void * h, unsigned) { *d = h; return hipSuccess; }
inline hipError_t hipMalloc(void ** p, size_t n) { return hipHostMalloc(p, n, 0); }
inline hipError_t hipFree(void * p) { free(p); return hipSuccess; }
inline hipError_t hipMemset(void * p, int v, size_t n) { memset(p, v, n); return hipSuccess; }
inline hipError_t hipStreamCreate(hipStream_t * s) { *s = new stub::Stream(); return hipSuccess; }
inline hipError_t hipStreamDestroy(hipStream_t s) { delete s; return hipSuccess; }
inline hipError_t hipStreamSynchronize(hipStream_t s) {
    for (;;) { { std::lock_guard<std::mutex> lk(s->m); if (s->q.empty() && !s->busy) return hipSuccess; } std::this_thread::sleep_for(std::chrono::microseconds(50)); }
}
inline hipError_t hipMemsetAsync(void * p, int v, size_t n, hipStream_t s) { s->push([=] { memset(p, v, n); }); return hipSuccess; }
inline hipError_t hipMemcpyAsync(void * d, const void * src, size_t n, hipMemcpyKind, hipStream_t s) { s->push([=] { memcpy(d, src, n); }); return hipSuccess; }
inline hipError_t hipEventCreate(hipEvent_t * e) { *e = new stub::Event(); return hipSuccess; }
inline hipError_t hipEventCreateWithFlags(hipEvent_t * e, unsigned) { return hipEventCreate(e); }
inline hipError_t hipEventDestroy(hipEvent_t e) { delete e; return hipSuccess; }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t s) { e->done = 0; s->push([=] { e->ns = stub::nsnow(); e->done = 1; }); return hipSuccess; }
inline hipError_t hipEventQuery(hipEvent_t e) { return e->done.load() ? hipSuccess : hipErrorNotReady; }
inline hipError_t hipEventElapsedTime(float * ms, hipEvent_t a, hipEvent_t b) { *ms = (float)((b->ns.load() - a->ns.load()) * 1e-6); return hipSuccess; }
inline hipError_t hipStreamWaitValue32(hipStream_t s, void * ptr, uint32_t value, unsigned, uint32_t mask) {
    const int mode = stub::wv_mode.load();
    s->push([=] {
        if (mode == 2) return;
        const int64_t t0 = stub::nsnow();
        for (;;) {
            const uint32_t v = __atomic_load_n((volatile uint32_t *)ptr, __ATOMIC_ACQUIRE) & mask;
            if (mode == 3 ? v == 0xFFFFFFFFu : v >= value) return;
            if (stub::nsnow() - t0 > 20000000000ll) return;   // fail-safe of the stub itself
            std::this_thread::yield();
        }
    });
    return hipSuccess;
}
