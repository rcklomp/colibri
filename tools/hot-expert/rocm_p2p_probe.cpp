// rocm_p2p_probe.cpp -- FRANKEN-M0b: name the P2P setup failure without a
// gateway restart. Prints hipDeviceCanAccessPeer for every ordered device
// pair, three per-device attributes, and the result of hipIpcGetMemHandle on
// a 1 MB allocation per device. Read-only wrt the box: no engine touched, no
// sudo, no system files. Build:
//   HIPCC=~/amdclang_wrap.sh LD_LIBRARY_PATH=$HOME/compat/lib \
//     /opt/rocm-6.2.0/bin/hipcc rocm_p2p_probe.cpp -o rocm_p2p_probe
#include <hip/hip_runtime.h>
#include <cstdio>

static void attr(int dev, hipDeviceAttribute_t a, const char *name) {
    int v = -1;
    hipError_t e = hipDeviceGetAttribute(&v, a, dev);
    if (e == hipSuccess) printf("  dev%d %s = %d\n", dev, name, v);
    else printf("  dev%d %s = ERROR %s\n", dev, name, hipGetErrorString(e));
}

int main() {
    int n = 0;
    hipGetDeviceCount(&n);
    printf("device count: %d\n", n);

    printf("--- per-device attributes ---\n");
    for (int d = 0; d < n; d++) {
        attr(d, hipDeviceAttributeConcurrentManagedAccess, "ConcurrentManagedAccess");
        attr(d, hipDeviceAttributeVirtualMemoryManagementSupported, "VirtualMemoryManagementSupported");
        attr(d, hipDeviceAttributePciBusId, "PciBusId");
    }

    printf("--- hipDeviceCanAccessPeer (ordered pairs) ---\n");
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++) {
            if (a == b) continue;
            int can = -1;
            hipError_t e = hipDeviceCanAccessPeer(&can, a, b);
            printf("  %d -> %d : canAccessPeer=%d (%s)\n", a, b, can,
                   e == hipSuccess ? "ok" : hipGetErrorString(e));
        }

    printf("--- hipIpcGetMemHandle (1 MB per device) ---\n");
    for (int d = 0; d < n; d++) {
        hipSetDevice(d);
        void *p = nullptr;
        hipError_t e = hipMalloc(&p, 1 << 20);
        if (e != hipSuccess) { printf("  dev%d hipMalloc: ERROR %s\n", d, hipGetErrorString(e)); continue; }
        hipIpcMemHandle_t h;
        hipError_t ei = hipIpcGetMemHandle(&h, p);
        printf("  dev%d hipIpcGetMemHandle: %s\n", d,
               ei == hipSuccess ? "ok" : hipGetErrorString(ei));
        hipFree(p);
    }
    return 0;
}
