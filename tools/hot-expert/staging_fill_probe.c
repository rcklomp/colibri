/* staging_fill_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2 step 0:
 * F0/F0b (record §VK-STREAM) measured the GPU-side leg of F2's streamed
 * prefill -- host-visible staging read by the backend's own queue family at
 * 62-63 GB/s on three cards, via `shader-host-read` (a compute shader
 * reading a HOST_VISIBLE|HOST_COHERENT buffer directly as a storage
 * buffer, no vkCmdCopyBuffer). What F0/F0b did NOT measure is the CPU-side
 * leg: amdgpu refuses to import file-backed memory
 * (VK_EXT_external_memory_host, EACCES -- see §VK-STREAM), so every
 * non-resident expert must be memcpy'd (or pread) from its page-cached
 * shard file into that staging buffer before the GPU can touch it, and F0's
 * own single-thread memcpy managed only 15 GB/s. This probe measures that
 * fill, multi-threaded, in the SCATTERED (miss-pattern) access F2 will
 * actually have, and then the fill overlapped with the GPU read in a real
 * ring-buffer pipeline shape.
 *
 * Extends vk_stream_probe.c's setup verbatim in spirit (VKCHECK, find_memtype
 * with its avoid-ReBAR mask, load_spv, the reduce shader's descriptor/
 * pipeline shape, pick_backend_qfam, PCI-bus-id device matching, the same
 * `shader-host-read` compute shader vk_stream_reduce.comp UNCHANGED -- this
 * probe binds it to different byte ranges of a bigger buffer via descriptor
 * offsets, needing no shader change) rather than linking against it (that
 * probe has no header, only a .c file).
 *
 * Four measurements, `block=14` (MiB) throughout, matching F0's convention
 * (GLM's real expert is 14.16 MB; every shard file is treated as a pool of
 * 14 MiB-aligned blocks, which is what F0's own block size already used):
 *
 * (a) memcpy   T threads (1,2,4,8,16), each thread owns a private 64 MiB
 *              region of one dedicated 1 GiB HOST_VISIBLE|HOST_COHERENT
 *              "scratch" staging buffer (same memory type F0 found fastest
 *              -- see find_memtype's avoid-DEVICE_LOCAL mask) and its own
 *              disjoint slice of a 2048-entry, fixed-seed (42) random block
 *              list across the served GLM's 62 shard files -- the scatter
 *              pattern F2's non-resident-expert miss set actually has.
 *              Threads are pinned via sched_setaffinity (verified topology:
 *              cpu 0-7 are this box's 8 distinct physical cores, one thread
 *              each; cpu 8-15 are their SMT siblings in the SAME order --
 *              `cpu == thread index` already gives OMP_PLACES=cores,
 *              OMP_PROC_BIND=close packing for every T in the sweep with no
 *              remapping table needed).
 * (b) pread    identical sweep, `pread(fd, staging_ptr, 14MiB, file_off)`
 *              straight into the mapped staging pointer -- no mmap fault
 *              path at all.
 * (d) memcpy-seq  control: (a)'s memcpy, same T sweep, but each thread's
 *              block list is a CONTIGUOUS slice of a sequential block
 *              ordering (not interleaved across threads) instead of the
 *              random list, to isolate the scatter cost.
 * (c) pipeline the number F2 actually lives on: a per-card ring of 4 slots
 *              x 64 MiB in HOST_VISIBLE|HOST_COHERENT + STORAGE_BUFFER
 *              memory, filled by a SHARED pool of 8 persistent CPU worker
 *              threads (pinned to the 8 physical cores -- shared across
 *              however many cards are active in a config, so the "all
 *              three cards concurrently" case is genuine 8-core contention,
 *              not 24 threads on an 8-core box) while vk_stream_reduce.comp
 *              (F0b's winning shader-host-read path, unmodified) reads the
 *              PREVIOUS slot on the backend's own queue family
 *              (pick_backend_qfam, copied from vk_stream_probe.c). Run for
 *              each card alone and for all three concurrently.
 *
 *              Synchronisation, precisely: each ring driver thread runs
 *                  for i = 1, 2, 3, ...:
 *                    cur = i % 4; prev = (i-1) % 4
 *                    vkResetFences+vkQueueSubmit(prev)   // ASYNC, no wait
 *                    fill(cur)                            // CPU work now,
 *                                                          // concurrent
 *                                                          // with the GPU
 *                                                          // read of prev
 *                    vkWaitForFences(prev, timeout=5s)    // bounded
 *              "fill(cur)" splits the 64 MiB slot into 8 x 8 MiB chunk
 *              tasks pushed to a global mutex+condvar work queue (blocking
 *              producer/consumer, never a spin loop); the worker that
 *              completes the slot's 8th chunk broadcasts a per-ring condvar
 *              the driver blocks on in wait_slot_ready() -- also blocking,
 *              not spinning. The only spin in this file is the same
 *              bounded start barrier vk_stream_probe.c already uses
 *              (sched_yield until every participant has arrived, which is
 *              microseconds by construction, not an open-ended wait).
 *              Each worker reserves its own byte-range in its ring's
 *              virtual random-block stream via one atomic fetch-add before
 *              copying, so concurrent workers filling the same slot never
 *              overlap and never lock against each other for the copy
 *              itself.
 *
 * Source data: opened read-only (O_RDONLY, mmap PROT_READ|MAP_PRIVATE -- no
 * PROT_WRITE needed here, this probe never attempts the ext-host import
 * §VK-STREAM already found EACCES on this rig's amdgpu, only memcpy/pread
 * out of it); the shard set is page-cache resident (~95%, per the item's
 * own framing) and reading it is harmless. Residency is checked via
 * mincore() before and after (never fincore-spawned per file in the hot
 * path) and a WARN line fires if it drops more than 2 percentage points.
 *
 * Build:
 *   gcc -O2 -pthread staging_fill_probe.c -o staging_fill_probe -lvulkan -lm
 *   (vk_stream_reduce.spv reused unmodified from vk_stream_chain.sh's own
 *   build step -- this probe does not recompile the shader itself)
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
 *     ./staging_fill_probe 2.0 ~/models/GLM-5.3-Flash-colibri-int4-g64 \
 *     vk_stream_reduce.spv
 *
 * Output: INFO lines (setup/diagnostic) then ROW lines:
 *   ROW kind=<memcpy|pread|memcpy-seq> threads=<T> mem=coh cards=0 block=14 gbps=<..> n=<..>
 *   ROW kind=pipeline threads=8 mem=coh cards=<1|3> block=14 gbps=<..> n=<..> dev=<label|all>
 * gbps uses GB = 1e9 bytes, matching every other probe in this record.
 */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define VKCHECK(x, msg) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "VK ERROR %s:%d (%s): %d\n", __FILE__, __LINE__, (msg), (int)_r); \
        exit(1); \
    } \
} while (0)

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static double median(double *v, int n) {
    if (n <= 0) return 0.0;
    double *s = malloc((size_t)n * sizeof(double));
    memcpy(s, v, (size_t)n * sizeof(double));
    for (int i = 1; i < n; i++) { double k = s[i]; int j = i - 1; while (j >= 0 && s[j] > k) { s[j+1] = s[j]; j--; } s[j+1] = k; }
    double m = (n % 2) ? s[n/2] : (s[n/2 - 1] + s[n/2]) / 2.0;
    free(s);
    return m;
}

static const size_t BLOCK_BYTES = 14ull * 1024ull * 1024ull;  /* "expert" granularity, F0's convention */
#define NEXP 2048                       /* >= 2000 random expert offsets required */
static const uint64_t SEED = 42ull;
static double MIN_SECONDS = 2.0;
static const int REPS = 5;              /* median of >= 5 */

/* ---- generic memory-type search (copied from vk_stream_probe.c) -------- */
static int find_memtype(VkPhysicalDevice phys, uint32_t typeBits,
                         VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid,
                         int *relaxed) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    if (relaxed) *relaxed = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(typeBits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) == want && !(f & avoid)) return (int)i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(typeBits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) == want) { if (relaxed) *relaxed = 1; return (int)i; }
    }
    return -1;
}

static int pick_backend_qfam(VkPhysicalDevice phys) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return (int)i;
    return -1;
}

static VkShaderModule load_spv(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open shader %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) { fprintf(stderr, "FATAL: bad SPIR-V size %ld in %s\n", n, path); exit(1); }
    uint32_t *code = malloc((size_t)n);
    if (!code || fread(code, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "FATAL: read %s failed\n", path); exit(1); }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(dev, &si, NULL, &m);
    free(code);
    if (r != VK_SUCCESS) { fprintf(stderr, "FATAL: vkCreateShaderModule(%s) rc=%d\n", path, (int)r); exit(1); }
    return m;
}

static void create_buffer(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size,
                           VkBufferUsageFlags usage, VkMemoryPropertyFlags want,
                           VkMemoryPropertyFlags avoid, VkBuffer *buf,
                           VkDeviceMemory *mem, void **mapped, int *relaxed) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, *buf, &req);
    int mt = find_memtype(phys, req.memoryTypeBits, want, avoid, relaxed);
    if (mt < 0) { fprintf(stderr, "FATAL: no memory type for buffer (want=0x%x avoid=0x%x)\n", want, avoid); exit(1); }
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt};
    VKCHECK(vkAllocateMemory(dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (mapped) VKCHECK(vkMapMemory(dev, *mem, 0, size, 0, mapped), "vkMapMemory");
}

/* ---- shard pool ---------------------------------------------------------- */
typedef struct { int fd; uint8_t *map; size_t size; size_t nblocks; } ShardFile;
static ShardFile g_shard[256]; static int g_nshard;
static size_t *g_blk_file, *g_blk_off; static size_t g_total_blocks;

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void load_shards(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "FATAL: opendir(%s): %s\n", dir, strerror(errno)); exit(1); }
    char *names[1024]; int nn = 0;
    struct dirent *de;
    while ((de = readdir(d)) && nn < 1024) {
        size_t l = strlen(de->d_name);
        if (l > 12 && !strcmp(de->d_name + l - 12, ".safetensors")) names[nn++] = strdup(de->d_name);
    }
    closedir(d);
    qsort(names, (size_t)nn, sizeof(char *), cmp_str);
    g_nshard = 0; g_total_blocks = 0;
    for (int i = 0; i < nn; i++) {
        char path[1536]; snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "WARN open(%s): %s, skipping\n", path, strerror(errno)); free(names[i]); continue; }
        struct stat st;
        if (fstat(fd, &st) != 0) { fprintf(stderr, "WARN fstat(%s): %s, skipping\n", path, strerror(errno)); close(fd); free(names[i]); continue; }
        void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { fprintf(stderr, "WARN mmap(%s): %s, skipping\n", path, strerror(errno)); close(fd); free(names[i]); continue; }
        ShardFile *s = &g_shard[g_nshard++];
        s->fd = fd; s->map = (uint8_t *)m; s->size = (size_t)st.st_size; s->nblocks = s->size / BLOCK_BYTES;
        g_total_blocks += s->nblocks;
        free(names[i]);
    }
    if (g_total_blocks < 2000) {
        fprintf(stderr, "FATAL: only %zu x 14MiB blocks across %d shards, need >= 2000\n", g_total_blocks, g_nshard);
        exit(1);
    }
    g_blk_file = malloc(g_total_blocks * sizeof(size_t));
    g_blk_off = malloc(g_total_blocks * sizeof(size_t));
    size_t idx = 0;
    for (int i = 0; i < g_nshard; i++)
        for (size_t b = 0; b < g_shard[i].nblocks; b++) { g_blk_file[idx] = (size_t)i; g_blk_off[idx] = b * BLOCK_BYTES; idx++; }
    printf("INFO shards=%d total_14mib_blocks=%zu total_bytes_ge=%.1fGB\n",
           g_nshard, g_total_blocks, (double)(g_total_blocks * BLOCK_BYTES) / 1e9);
}

static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static size_t *g_rand_list, *g_seq_list;
static void build_lists(void) {
    g_rand_list = malloc(NEXP * sizeof(size_t));
    g_seq_list = malloc(NEXP * sizeof(size_t));
    uint64_t st = SEED;
    for (int i = 0; i < NEXP; i++) g_rand_list[i] = (size_t)(xorshift64s(&st) % g_total_blocks);
    for (int i = 0; i < NEXP; i++) g_seq_list[i] = (size_t)i % g_total_blocks;
    printf("INFO expert_lists n=%d seed=%llu\n", NEXP, (unsigned long long)SEED);
}

/* ---- residency / machine facts ------------------------------------------ */
static double residency_pct(void) {
    long pagesize = sysconf(_SC_PAGESIZE);
    unsigned long long resident = 0, total_pages = 0;
    unsigned char *vec = NULL; size_t vec_cap = 0;
    for (int i = 0; i < g_nshard; i++) {
        size_t npages = (g_shard[i].size + (size_t)pagesize - 1) / (size_t)pagesize;
        if (npages > vec_cap) { free(vec); vec = malloc(npages); vec_cap = npages; }
        if (!vec || mincore(g_shard[i].map, g_shard[i].size, vec) != 0) {
            fprintf(stderr, "WARN mincore(shard %d) failed: %s\n", i, strerror(errno));
            continue;
        }
        for (size_t p = 0; p < npages; p++) if (vec[p] & 1) resident++;
        total_pages += npages;
    }
    free(vec);
    return total_pages ? (100.0 * (double)resident / (double)total_pages) : 0.0;
}

static unsigned long long mem_available_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; unsigned long long v = 0;
    while (fgets(line, sizeof(line), f)) if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) break;
    fclose(f);
    return v;
}

/* ---- CPU pinning ---------------------------------------------------------
 * cpu 0-7 are the 8 physical cores (one hw thread each); 8-15 are their SMT
 * siblings in the SAME core order (/sys/devices/system/cpu/cpuN/topology/
 * core_id: cpu0/cpu8 both core_id=0, cpu1/cpu9 core_id=4, ...). cpu id ==
 * thread index therefore gives OMP_PLACES=cores,OMP_PROC_BIND=close packing
 * directly for T in {1,2,4,8} (T distinct physical cores) and T=16 (both
 * threads of all 8 cores), with no remapping table. */
static void pin_to_cpu(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        fprintf(stderr, "WARN sched_setaffinity(cpu=%d) failed: %s\n", cpu, strerror(errno));
}

/* ---- (a)/(b)/(d): T-thread fill sweep into a dedicated scratch buffer --- */
#define THREAD_REGION (64ull * 1024ull * 1024ull)   /* 64 MiB, own region per thread */

typedef enum { KIND_MEMCPY, KIND_PREAD } FillKind;

typedef struct {
    int tid;
    FillKind kind;
    uint8_t *dst_region;
    size_t *sub_list; int sub_n;
    double min_seconds;
    atomic_int *ready; int nparty;
    long long bytes; double secs;
} FillArg;

static void *fill_worker(void *argp) {
    FillArg *a = argp;
    pin_to_cpu(a->tid);
    const int NSUB = (int)(THREAD_REGION / BLOCK_BYTES);
    size_t k = 0;

    /* untimed warm-up copy */
    {
        size_t blk = a->sub_list[k % (size_t)a->sub_n];
        int fidx = (int)g_blk_file[blk]; size_t foff = g_blk_off[blk];
        uint8_t *dst = a->dst_region + (size_t)(k % (size_t)NSUB) * BLOCK_BYTES;
        if (a->kind == KIND_PREAD) { ssize_t rr = pread(g_shard[fidx].fd, dst, BLOCK_BYTES, (off_t)foff); (void)rr; }
        else memcpy(dst, g_shard[fidx].map + foff, BLOCK_BYTES);
        k++;
    }
    atomic_fetch_add_explicit(a->ready, 1, memory_order_acq_rel);
    while (atomic_load_explicit(a->ready, memory_order_acquire) < a->nparty) sched_yield();

    long long n = 0; double t0 = now_s(), t = t0;
    do {
        size_t blk = a->sub_list[k % (size_t)a->sub_n];
        int fidx = (int)g_blk_file[blk]; size_t foff = g_blk_off[blk];
        uint8_t *dst = a->dst_region + (size_t)(k % (size_t)NSUB) * BLOCK_BYTES;
        if (a->kind == KIND_PREAD) {
            ssize_t r = pread(g_shard[fidx].fd, dst, BLOCK_BYTES, (off_t)foff);
            if (r < 0) { fprintf(stderr, "WARN pread failed: %s\n", strerror(errno)); break; }
        } else {
            memcpy(dst, g_shard[fidx].map + foff, BLOCK_BYTES);
        }
        k++; n++; t = now_s();
    } while (t - t0 < a->min_seconds);
    a->bytes = n * (long long)BLOCK_BYTES;
    a->secs = t - t0;
    return NULL;
}

/* Splits `src`[0..n) across T threads: contig=1 gives each thread a
 * contiguous slice (the sequential control, (d)); contig=0 interleaves
 * (thread t gets indices t, t+T, t+2T, ... -- a clean partition of the
 * random list, (a)/(b)). */
static void partition_list(size_t *src, int n, int T, int contig, size_t **outs, int *outn) {
    if (contig) {
        int per = n / T;
        for (int t = 0; t < T; t++) {
            int start = t * per;
            int cnt = (t == T - 1) ? (n - start) : per;
            outs[t] = malloc((size_t)cnt * sizeof(size_t));
            memcpy(outs[t], src + start, (size_t)cnt * sizeof(size_t));
            outn[t] = cnt;
        }
    } else {
        for (int t = 0; t < T; t++) {
            int cnt = 0;
            for (int i = t; i < n; i += T) cnt++;
            outs[t] = malloc((size_t)cnt * sizeof(size_t));
            int k = 0;
            for (int i = t; i < n; i += T) outs[t][k++] = src[i];
            outn[t] = cnt;
        }
    }
}

static double run_fill_config(FillKind kind, int T, int contig, uint8_t *scratch, size_t *list) {
    pthread_t th[16]; FillArg args[16]; atomic_int ready = 0;
    size_t *subs[16]; int subn[16];
    partition_list(list, NEXP, T, contig, subs, subn);
    for (int i = 0; i < T; i++) {
        args[i] = (FillArg){i, kind, scratch + (size_t)i * THREAD_REGION, subs[i], subn[i], MIN_SECONDS, &ready, T, 0, 0};
        pthread_create(&th[i], NULL, fill_worker, &args[i]);
    }
    long long total = 0; double max_secs = 0;
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < T; i++) { total += args[i].bytes; if (args[i].secs > max_secs) max_secs = args[i].secs; free(subs[i]); }
    return max_secs > 0 ? (double)total / max_secs / 1e9 : 0.0;
}

static void print_row_fill(const char *kind, int threads, double gbps, int n) {
    printf("ROW kind=%s threads=%d mem=coh cards=0 block=14 gbps=%.3f n=%d\n", kind, threads, gbps, n);
    fflush(stdout);
}

/* ---- (c) pipeline: per-card ring, shared 8-thread fill pool ------------- */
#define RING_SLOTS 4
static const size_t SLOT_BYTES = 64ull * 1024ull * 1024ull;
static const uint32_t REDUCE_WORKGROUPS = 4096;
#define NFILL 8

typedef struct {
    VkPhysicalDevice phys; char bus_id[32]; const char *label;
    VkDevice dev; VkQueue queue_gfx; uint32_t qfam_gfx;
    VkCommandPool cpool; VkCommandBuffer cmd;
    VkFence fence[RING_SLOTS];
    VkBuffer ring_buf; VkDeviceMemory ring_mem; uint8_t *ring_ptr;
    VkBuffer dst_buf; VkDeviceMemory dst_mem;
    VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout plyt; VkPipeline pipe;
    VkDescriptorPool dpool; VkDescriptorSet dset[RING_SLOTS];

    atomic_int slot_done[RING_SLOTS];
    pthread_mutex_t mtx; pthread_cond_t cond;
    _Atomic size_t stream_pos;   /* rolling byte position into this ring's virtual random-block stream */
    size_t phase;                /* per-ring starting offset into g_rand_list, so 3 rings don't lock-step */
} Ring;

typedef struct { Ring *ring; int slot; int chunk; } Task;
#define QCAP 512
static Task g_q[QCAP]; static int g_qh = 0, g_qt = 0, g_qn = 0;
static pthread_mutex_t g_qmtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_qcond = PTHREAD_COND_INITIALIZER;
static int g_shutdown = 0;

static void q_push(Task t) {
    pthread_mutex_lock(&g_qmtx);
    while (g_qn == QCAP) pthread_cond_wait(&g_qcond, &g_qmtx);
    g_q[g_qt] = t; g_qt = (g_qt + 1) % QCAP; g_qn++;
    pthread_cond_broadcast(&g_qcond);
    pthread_mutex_unlock(&g_qmtx);
}
static int q_pop(Task *t) {
    pthread_mutex_lock(&g_qmtx);
    while (g_qn == 0 && !g_shutdown) pthread_cond_wait(&g_qcond, &g_qmtx);
    if (g_qn == 0) { pthread_mutex_unlock(&g_qmtx); return 0; }
    *t = g_q[g_qh]; g_qh = (g_qh + 1) % QCAP; g_qn--;
    pthread_cond_broadcast(&g_qcond);
    pthread_mutex_unlock(&g_qmtx);
    return 1;
}

/* Reserves [pos, pos+nbytes) in `ring`'s virtual stream (concatenation of
 * g_rand_list's 14 MiB blocks, phase-shifted per ring) via one atomic
 * fetch-add, then copies -- concurrent workers on the same ring never
 * overlap and never lock against each other for the memcpy itself. A span
 * can cross at most one block boundary since chunk (8 MiB) < block (14
 * MiB). */
static void copy_from_ring_stream(Ring *ring, uint8_t *dst, size_t nbytes) {
    size_t pos = atomic_fetch_add_explicit(&ring->stream_pos, nbytes, memory_order_relaxed);
    size_t done = 0;
    while (done < nbytes) {
        size_t abspos = pos + done;
        size_t block_i = abspos / BLOCK_BYTES;
        size_t off_in_block = abspos % BLOCK_BYTES;
        size_t blk = g_rand_list[(ring->phase + block_i) % NEXP];
        int fidx = (int)g_blk_file[blk]; size_t foff = g_blk_off[blk];
        size_t can = BLOCK_BYTES - off_in_block;
        size_t take = (nbytes - done) < can ? (nbytes - done) : can;
        memcpy(dst + done, g_shard[fidx].map + foff + off_in_block, take);
        done += take;
    }
}

static void *fill_pool_worker(void *argp) {
    int cpu = (int)(intptr_t)argp;
    pin_to_cpu(cpu);
    Task t;
    while (q_pop(&t)) {
        Ring *ring = t.ring;
        size_t chunk_bytes = SLOT_BYTES / NFILL;
        uint8_t *dst = ring->ring_ptr + (size_t)t.slot * SLOT_BYTES + (size_t)t.chunk * chunk_bytes;
        copy_from_ring_stream(ring, dst, chunk_bytes);
        int done = atomic_fetch_add_explicit(&ring->slot_done[t.slot], 1, memory_order_acq_rel) + 1;
        if (done == NFILL) {
            pthread_mutex_lock(&ring->mtx);
            pthread_cond_broadcast(&ring->cond);
            pthread_mutex_unlock(&ring->mtx);
        }
    }
    return NULL;
}

static void push_fill_tasks(Ring *ring, int slot) {
    atomic_store_explicit(&ring->slot_done[slot], 0, memory_order_release);
    for (int c = 0; c < NFILL; c++) q_push((Task){ring, slot, c});
}
static void wait_slot_ready(Ring *ring, int slot) {
    pthread_mutex_lock(&ring->mtx);
    while (atomic_load_explicit(&ring->slot_done[slot], memory_order_acquire) < NFILL)
        pthread_cond_wait(&ring->cond, &ring->mtx);
    pthread_mutex_unlock(&ring->mtx);
}

static void submit_read_async(Ring *ring, int slot) {
    VKCHECK(vkResetFences(ring->dev, 1, &ring->fence[slot]), "reset ring fence");
    VKCHECK(vkResetCommandBuffer(ring->cmd, 0), "reset ring cmd");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(ring->cmd, &bi), "begin ring cmd");
    vkCmdBindPipeline(ring->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ring->pipe);
    vkCmdBindDescriptorSets(ring->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ring->plyt, 0, 1, &ring->dset[slot], 0, NULL);
    uint32_t n_words = (uint32_t)(SLOT_BYTES / 4);
    vkCmdPushConstants(ring->cmd, ring->plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_words);
    vkCmdDispatch(ring->cmd, REDUCE_WORKGROUPS, 1, 1);
    VKCHECK(vkEndCommandBuffer(ring->cmd), "end ring cmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &ring->cmd};
    VKCHECK(vkQueueSubmit(ring->queue_gfx, 1, &si, ring->fence[slot]), "submit ring read");
}

typedef struct {
    Ring *ring;
    double min_seconds;
    atomic_int *start_barrier; int nparty;
    long long bytes; double secs;
} DriverArg;

static void *ring_driver(void *argp) {
    DriverArg *a = argp; Ring *ring = a->ring;

    push_fill_tasks(ring, 0); wait_slot_ready(ring, 0);           /* warm-up fill, uncounted */
    submit_read_async(ring, 0);
    VKCHECK(vkWaitForFences(ring->dev, 1, &ring->fence[0], VK_TRUE, 30000000000ULL), "warmup wait");

    atomic_fetch_add_explicit(a->start_barrier, 1, memory_order_acq_rel);
    while (atomic_load_explicit(a->start_barrier, memory_order_acquire) < a->nparty) sched_yield();

    long long cycles = 0; double t0 = now_s(), t = t0;
    int i = 1;
    do {
        int cur = i % RING_SLOTS, prev = (i - 1) % RING_SLOTS;
        submit_read_async(ring, prev);          /* async: GPU starts reading `prev` */
        push_fill_tasks(ring, cur);              /* CPU starts filling `cur`, concurrently */
        wait_slot_ready(ring, cur);
        VKCHECK(vkWaitForFences(ring->dev, 1, &ring->fence[prev], VK_TRUE, 5000000000ULL), "ring fence wait");
        cycles++; i++; t = now_s();
    } while (t - t0 < a->min_seconds);
    a->bytes = cycles * (long long)SLOT_BYTES;
    a->secs = t - t0;
    return NULL;
}

static void run_pipeline_config(Ring **rings, int nrings, DriverArg *outs) {
    atomic_int barrier = 0;
    pthread_t th[3];
    for (int i = 0; i < nrings; i++) {
        outs[i] = (DriverArg){rings[i], MIN_SECONDS, &barrier, nrings, 0, 0};
        pthread_create(&th[i], NULL, ring_driver, &outs[i]);
    }
    for (int i = 0; i < nrings; i++) pthread_join(th[i], NULL);
}

static void build_ring_pipeline(Ring *r, VkShaderModule shader) {
    VkDescriptorSetLayoutBinding b[2];
    for (int i = 0; i < 2; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(r->dev, &dsli, NULL, &r->dsl), "ring descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 4};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &r->dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(r->dev, &pli, NULL, &r->plyt), "ring pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = r->plyt};
    VKCHECK(vkCreateComputePipelines(r->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &r->pipe), "ring pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 * RING_SLOTS};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = RING_SLOTS, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(r->dev, &dpi, NULL, &r->dpool), "ring descPool");
    VkDescriptorSetLayout layouts[RING_SLOTS];
    for (int i = 0; i < RING_SLOTS; i++) layouts[i] = r->dsl;
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = r->dpool, .descriptorSetCount = RING_SLOTS, .pSetLayouts = layouts};
    VKCHECK(vkAllocateDescriptorSets(r->dev, &dsa, r->dset), "ring allocDescSets");
    for (int s = 0; s < RING_SLOTS; s++) {
        VkDescriptorBufferInfo dbi[2] = {
            {.buffer = r->ring_buf, .offset = (VkDeviceSize)s * SLOT_BYTES, .range = (VkDeviceSize)SLOT_BYTES},
            {.buffer = r->dst_buf, .offset = 0, .range = VK_WHOLE_SIZE}};
        VkWriteDescriptorSet wr[2];
        for (int i = 0; i < 2; i++) wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = r->dset[s], .dstBinding = (uint32_t)i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i]};
        vkUpdateDescriptorSets(r->dev, 2, wr, 0, NULL);
    }
}

static void setup_ring(Ring *r, VkPhysicalDevice phys, const char *bus_id, const char *label,
                        const char *reduce_spv_path, size_t phase) {
    memset(r, 0, sizeof(*r));
    r->phys = phys; r->label = label; strncpy(r->bus_id, bus_id, sizeof(r->bus_id) - 1);
    r->phase = phase;
    pthread_mutex_init(&r->mtx, NULL);
    pthread_cond_init(&r->cond, NULL);

    int qfam_gfx = pick_backend_qfam(phys);
    if (qfam_gfx < 0) { fprintf(stderr, "FATAL: %s: pick_backend_qfam found nothing\n", label); exit(1); }
    r->qfam_gfx = (uint32_t)qfam_gfx;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = r->qfam_gfx, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
    VKCHECK(vkCreateDevice(phys, &di, NULL, &r->dev), "vkCreateDevice");
    vkGetDeviceQueue(r->dev, r->qfam_gfx, 0, &r->queue_gfx);

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = r->qfam_gfx};
    VKCHECK(vkCreateCommandPool(r->dev, &cpci, NULL, &r->cpool), "ring cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = r->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(r->dev, &cbi, &r->cmd), "ring cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (int s = 0; s < RING_SLOTS; s++) VKCHECK(vkCreateFence(r->dev, &fi, NULL, &r->fence[s]), "ring fence");

    int relaxed;
    create_buffer(r->dev, phys, (VkDeviceSize)(RING_SLOTS * SLOT_BYTES), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &r->ring_buf, &r->ring_mem, (void **)&r->ring_ptr, &relaxed);
    create_buffer(r->dev, phys, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                  &r->dst_buf, &r->dst_mem, NULL, &relaxed);

    r->shader = load_spv(r->dev, reduce_spv_path);
    build_ring_pipeline(r, r->shader);

    fprintf(stderr, "INFO ring %s bus=%s qfam_gfx=%u ring_bytes=%zu slot_bytes=%zu\n",
            label, bus_id, r->qfam_gfx, (size_t)(RING_SLOTS * SLOT_BYTES), (size_t)SLOT_BYTES);
}

/* ---- main ----------------------------------------------------------------- */
int main(int argc, char **argv) {
    if (argc > 1) MIN_SECONDS = atof(argv[1]);
    const char *shard_dir = argc > 2 ? argv[2] : NULL;
    const char *reduce_spv_path = argc > 3 ? argv[3] : "vk_stream_reduce.spv";
    if (!shard_dir) { fprintf(stderr, "usage: %s <min_seconds> <shard_dir> <reduce.spv>\n", argv[0]); return 2; }

    load_shards(shard_dir);
    build_lists();

    unsigned long long mem_before = mem_available_kb();
    double res_before = residency_pct();
    printf("INFO machine mem_available_before_kb=%llu residency_before_pct=%.2f\n", mem_before, res_before);

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    VKCHECK(vkCreateInstance(&ici, NULL, &inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "FATAL: no Vulkan physical devices\n"); return 1; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(inst, &nd, devs);

    VkPhysicalDevice pdev0 = NULL, pdev2 = NULL, pdev3 = NULL;
    char bus0[32] = "", bus2[32] = "", bus3[32] = "";
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pci};
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        char bus[32]; snprintf(bus, sizeof(bus), "%04x:%02x:%02x.%01x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        printf("INFO discrete idx=%u bus_id=%s name=\"%s\"\n", i, bus, p.deviceName);
        if (!strcmp(bus, "0000:83:00.0")) { pdev0 = devs[i]; strcpy(bus0, bus); }
        else if (!strcmp(bus, "0000:48:00.0")) { pdev2 = devs[i]; strcpy(bus2, bus); }
        else if (!strcmp(bus, "0000:86:00.0")) { pdev3 = devs[i]; strcpy(bus3, bus); }
    }
    if (!pdev0 || !pdev2 || !pdev3) {
        fprintf(stderr, "FATAL: could not map all three PCI bus ids to dev0/dev2/dev3\n");
        return 1;
    }

    Ring ring0, ring2, ring3;
    setup_ring(&ring0, pdev0, bus0, "dev0", reduce_spv_path, 0);
    setup_ring(&ring2, pdev2, bus2, "dev2", reduce_spv_path, 700);
    setup_ring(&ring3, pdev3, bus3, "dev3", reduce_spv_path, 1400);

    /* scratch HV-coherent staging buffer for (a)/(b)/(d), carved on dev0's
     * VkDevice -- which physical card "owns" it is irrelevant here, no GPU
     * ever touches it; it's just a convenient HOST_VISIBLE|HOST_COHERENT
     * allocation of the SAME memory type F0 measured as fastest. */
    VkBuffer scratch_buf; VkDeviceMemory scratch_mem; uint8_t *scratch_ptr; int relaxed;
    size_t STAGE_TOTAL = 16ull * THREAD_REGION;
    create_buffer(ring0.dev, pdev0, (VkDeviceSize)STAGE_TOTAL, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &scratch_buf, &scratch_mem, (void **)&scratch_ptr, &relaxed);
    printf("INFO scratch_stage_bytes=%zu (relaxed_memtype=%d)\n", STAGE_TOTAL, relaxed);

    int Ts[] = {1, 2, 4, 8, 16};
    int NT = 5;

    for (int i = 0; i < NT; i++) {
        double s[REPS];
        for (int r = 0; r < REPS; r++) s[r] = run_fill_config(KIND_MEMCPY, Ts[i], 0, scratch_ptr, g_rand_list);
        print_row_fill("memcpy", Ts[i], median(s, REPS), REPS);
    }
    for (int i = 0; i < NT; i++) {
        double s[REPS];
        for (int r = 0; r < REPS; r++) s[r] = run_fill_config(KIND_PREAD, Ts[i], 0, scratch_ptr, g_rand_list);
        print_row_fill("pread", Ts[i], median(s, REPS), REPS);
    }
    for (int i = 0; i < NT; i++) {
        double s[REPS];
        for (int r = 0; r < REPS; r++) s[r] = run_fill_config(KIND_MEMCPY, Ts[i], 1, scratch_ptr, g_seq_list);
        print_row_fill("memcpy-seq", Ts[i], median(s, REPS), REPS);
    }

    /* (c) pipeline: shared 8-thread fill pool, started once, used by every
     * pipeline config below (alone and all-three both draw from it). */
    pthread_t pool[NFILL];
    for (int i = 0; i < NFILL; i++) pthread_create(&pool[i], NULL, fill_pool_worker, (void *)(intptr_t)i);

    Ring *alone[3] = {&ring0, &ring2, &ring3};
    const char *alone_label[3] = {"dev0", "dev2", "dev3"};
    for (int i = 0; i < 3; i++) {
        double s[REPS];
        for (int r = 0; r < REPS; r++) {
            DriverArg out[1];
            Ring *g[1] = {alone[i]};
            run_pipeline_config(g, 1, out);
            s[r] = out[0].secs > 0 ? (double)out[0].bytes / out[0].secs / 1e9 : 0.0;
        }
        printf("ROW kind=pipeline threads=%d mem=coh cards=1 block=14 gbps=%.3f n=%d dev=%s\n",
               NFILL, median(s, REPS), REPS, alone_label[i]);
        fflush(stdout);
    }

    {
        double s_all[REPS], s_dev0[REPS], s_dev2[REPS], s_dev3[REPS];
        Ring *g3[3] = {&ring0, &ring2, &ring3};
        for (int r = 0; r < REPS; r++) {
            DriverArg out[3];
            run_pipeline_config(g3, 3, out);
            double bsum = 0, maxsec = 0;
            for (int i = 0; i < 3; i++) { bsum += (double)out[i].bytes; if (out[i].secs > maxsec) maxsec = out[i].secs; }
            s_all[r] = maxsec > 0 ? bsum / maxsec / 1e9 : 0.0;
            s_dev0[r] = out[0].secs > 0 ? (double)out[0].bytes / out[0].secs / 1e9 : 0.0;
            s_dev2[r] = out[1].secs > 0 ? (double)out[1].bytes / out[1].secs / 1e9 : 0.0;
            s_dev3[r] = out[2].secs > 0 ? (double)out[2].bytes / out[2].secs / 1e9 : 0.0;
        }
        printf("ROW kind=pipeline threads=%d mem=coh cards=3 block=14 gbps=%.3f n=%d dev=dev0\n", NFILL, median(s_dev0, REPS), REPS);
        printf("ROW kind=pipeline threads=%d mem=coh cards=3 block=14 gbps=%.3f n=%d dev=dev2\n", NFILL, median(s_dev2, REPS), REPS);
        printf("ROW kind=pipeline threads=%d mem=coh cards=3 block=14 gbps=%.3f n=%d dev=dev3\n", NFILL, median(s_dev3, REPS), REPS);
        printf("ROW kind=pipeline threads=%d mem=coh cards=3 block=14 gbps=%.3f n=%d dev=all\n", NFILL, median(s_all, REPS), REPS);
        fflush(stdout);
    }

    pthread_mutex_lock(&g_qmtx); g_shutdown = 1; pthread_cond_broadcast(&g_qcond); pthread_mutex_unlock(&g_qmtx);
    for (int i = 0; i < NFILL; i++) pthread_join(pool[i], NULL);

    unsigned long long mem_after = mem_available_kb();
    double res_after = residency_pct();
    printf("INFO machine mem_available_after_kb=%llu residency_after_pct=%.2f\n", mem_after, res_after);
    if (res_before - res_after > 2.0)
        printf("WARN residency dropped %.2f -> %.2f pct during the run (> 2 pct)\n", res_before, res_after);

    printf("INFO done\n");
    return 0;
}
