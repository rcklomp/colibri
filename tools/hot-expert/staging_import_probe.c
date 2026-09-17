/* staging_import_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F2, step
 * 0b: does the copy go away entirely?
 *
 * Step 0 (record §F2-STEP0) measured a staging-fill-then-GPU-read pipeline
 * at 38.6 GB/s aggregate on three cards, below the plan's 60 GB/s bar, even
 * though CPU fill ALONE reached 59+ GB/s and F0's GPU read alone reached
 * 62-63 GB/s. Two explanations were offered: the ring pipeline's own
 * handshake overhead, or DRAM traffic roughly tripling (fill read + staging
 * write + PCIe read of the same bytes, close to a 91.6 GB/s DRAM-read
 * ceiling). Both explanations disappear if the memcpy-into-staging step is
 * removed. §VK-STREAM found `VK_EXT_external_memory_host` import fails
 * (EACCES) ONLY for file-backed (page-cache) memory on this rig's amdgpu;
 * an anonymous mapping imports fine. This probe measures the no-copy path:
 * import ANONYMOUS host memory (never a page-cache mapping) and read it in
 * place with F0's own `shader-host-read` dispatch.
 *
 * (1) For each of two variants -- `anon` (no THP hint) and `anon-huge`
 *     (`madvise(MADV_HUGEPAGE)` called immediately after `mmap`, before any
 *     write, so a THP collapse has a chance to apply from the first fault;
 *     this rig's THP mode is `madvise`, confirmed via
 *     /sys/kernel/mm/transparent_hugepage/enabled, so huge pages are used
 *     ONLY when asked -- this variant is a real A/B, not a no-op) -- mmap
 *     MAP_ANONYMOUS|MAP_PRIVATE, populate it ONCE via multi-threaded
 *     `pread` of >= 2000 random 14 MiB experts cycling to fill the whole
 *     region (the load-time copy F2 would pay once; its rate is reported
 *     as an INFO line, NOT a ROW/gate metric), import it per device via
 *     `VkImportMemoryHostPointerInfoEXT` (alignment from
 *     `VkPhysicalDeviceExternalMemoryHostPropertiesEXT.
 *     minImportedHostPointerAlignment`, queried per device, not assumed),
 *     bind a buffer, and run F0's `shader-host-read` shader
 *     (`vk_stream_reduce.comp`, UNCHANGED) over a 256 MiB window at a
 *     random 256 MiB-aligned offset within the region, re-chosen every
 *     submit via `vkUpdateDescriptorSets` (fixed-seed RNG, per device) --
 *     each card alone and all three concurrently, >= 2 s/trial, median of
 *     >= 5, `ROW kind=import-read src=anon[-huge] cards=<1|3> ...`.
 * (2) Scale: repeat at region sizes 8, 32, 64, 96 GiB, ONE region at a time
 *     (destroyed and unmapped before the next size, so peak extra RSS is
 *     one region, never a sum of them). Stops at the first size where
 *     mmap or import fails on ANY of the three devices (F2 needs every
 *     card to be able to read any expert) and prints the exact `VkResult`
 *     (no `journalctl` available in this environment) plus
 *     /proc/meminfo MemAvailable before/after that attempt.
 * (3) Prints /sys/module/amdgpu/parameters/gttsize (root-only on this rig;
 *     reports the permission failure honestly rather than guessing) and,
 *     per device, the HOST_VISIBLE|HOST_COHERENT heap's size from
 *     `VkPhysicalDeviceMemoryProperties` (userptr imports are commonly
 *     accounted against the GTT heap) as INFO lines.
 *
 * Build:
 *   gcc -O2 -pthread staging_import_probe.c -o staging_import_probe -lvulkan -lm
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
 *     ./staging_import_probe ~/models/GLM-5.3-Flash-colibri-int4-g64 vk_stream_reduce.spv
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

static const size_t GIB = 1024ull * 1024ull * 1024ull;
static const size_t BLOCK_BYTES = 14ull * 1024ull * 1024ull;
static const size_t WINDOW_BYTES = 256ull * 1024ull * 1024ull;   /* per-dispatch read window */
static const uint32_t REDUCE_WORKGROUPS = 4096;
#define NEXP 2048
static const uint64_t SEED = 42ull;
static double MIN_SECONDS = 2.0;
static const int REPS = 5;
#define NFILL 8

/* ---- copied verbatim in spirit from staging_fill_probe.c / vk_stream_probe.c ---- */
static int find_memtype(VkPhysicalDevice phys, uint32_t typeBits,
                         VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, int *relaxed) {
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
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return (int)i;
    return -1;
}
static VkShaderModule load_spv(VkDevice dev, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open shader %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)n);
    if (!code || fread(code, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "FATAL: read %s failed\n", path); exit(1); }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m; VkResult r = vkCreateShaderModule(dev, &si, NULL, &m);
    free(code);
    if (r != VK_SUCCESS) { fprintf(stderr, "FATAL: vkCreateShaderModule rc=%d\n", (int)r); exit(1); }
    return m;
}

/* ---- shard pool (same as staging_fill_probe.c) --------------------------- */
typedef struct { int fd; size_t size; size_t nblocks; } ShardFile;
static ShardFile g_shard[256]; static int g_nshard;
static size_t *g_blk_file, *g_blk_off; static size_t g_total_blocks;
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void load_shards(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "FATAL: opendir(%s): %s\n", dir, strerror(errno)); exit(1); }
    char *names[1024]; int nn = 0; struct dirent *de;
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
        if (fd < 0) { fprintf(stderr, "WARN open(%s): %s\n", path, strerror(errno)); free(names[i]); continue; }
        struct stat st; fstat(fd, &st);
        ShardFile *s = &g_shard[g_nshard++];
        s->fd = fd; s->size = (size_t)st.st_size; s->nblocks = s->size / BLOCK_BYTES;
        g_total_blocks += s->nblocks;
        free(names[i]);
    }
    if (g_total_blocks < 2000) { fprintf(stderr, "FATAL: only %zu blocks, need >= 2000\n", g_total_blocks); exit(1); }
    g_blk_file = malloc(g_total_blocks * sizeof(size_t));
    g_blk_off = malloc(g_total_blocks * sizeof(size_t));
    size_t idx = 0;
    for (int i = 0; i < g_nshard; i++)
        for (size_t b = 0; b < g_shard[i].nblocks; b++) { g_blk_file[idx] = (size_t)i; g_blk_off[idx] = b * BLOCK_BYTES; idx++; }
    printf("INFO shards=%d total_14mib_blocks=%zu\n", g_nshard, g_total_blocks);
}
static uint64_t xorshift64s(uint64_t *st) { uint64_t x = *st; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *st = x; return x * 2685821657736338717ull; }
static size_t *g_rand_list;
static void build_list(void) {
    g_rand_list = malloc(NEXP * sizeof(size_t));
    uint64_t st = SEED;
    for (int i = 0; i < NEXP; i++) g_rand_list[i] = (size_t)(xorshift64s(&st) % g_total_blocks);
    printf("INFO expert_list n=%d seed=%llu\n", NEXP, (unsigned long long)SEED);
}
static unsigned long long mem_available_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; unsigned long long v = 0;
    while (fgets(line, sizeof(line), f)) if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) break;
    fclose(f);
    return v;
}
static void pin_to_cpu(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

/* ---- populate: fill an anonymous region with real (cycled) expert bytes -- */
static _Atomic size_t g_pop_cursor;
static uint8_t *g_pop_dst; static size_t g_pop_total;
static void *populate_worker(void *argp) {
    int cpu = (int)(intptr_t)argp; pin_to_cpu(cpu);
    const size_t CHUNK = 8ull * 1024ull * 1024ull;
    for (;;) {
        size_t pos = atomic_fetch_add_explicit(&g_pop_cursor, CHUNK, memory_order_relaxed);
        if (pos >= g_pop_total) return NULL;
        size_t take = (g_pop_total - pos) < CHUNK ? (g_pop_total - pos) : CHUNK;
        size_t done = 0;
        while (done < take) {
            size_t abspos = pos + done;
            size_t block_i = (abspos / BLOCK_BYTES) % NEXP;
            size_t off_in_block = abspos % BLOCK_BYTES;
            size_t blk = g_rand_list[block_i];
            int fidx = (int)g_blk_file[blk]; size_t foff = g_blk_off[blk];
            size_t can = BLOCK_BYTES - off_in_block;
            size_t chunk = (take - done) < can ? (take - done) : can;
            ssize_t r = pread(g_shard[fidx].fd, g_pop_dst + pos + done, chunk, (off_t)(foff + off_in_block));
            if (r <= 0) { fprintf(stderr, "WARN populate pread failed: %s\n", strerror(errno)); return NULL; }
            done += (size_t)r;
        }
    }
}
static double populate_region(uint8_t *dst, size_t total) {
    g_pop_dst = dst; g_pop_total = total; atomic_store(&g_pop_cursor, 0);
    pthread_t th[NFILL];
    double t0 = now_s();
    for (int i = 0; i < NFILL; i++) pthread_create(&th[i], NULL, populate_worker, (void *)(intptr_t)i);
    for (int i = 0; i < NFILL; i++) pthread_join(th[i], NULL);
    double t = now_s();
    return (double)total / (t - t0) / 1e9;
}

/* ---- per-device persistent Vulkan context (built once, independent of
 * which import is currently bound) --------------------------------------- */
typedef struct {
    VkPhysicalDevice phys; char bus_id[32]; const char *label;
    VkDevice dev; VkQueue queue; uint32_t qfam;
    VkCommandPool cpool; VkCommandBuffer cmd; VkFence fence;
    PFN_vkGetMemoryHostPointerPropertiesEXT fn_getHostPtrProps;
    VkDeviceSize min_align;
    VkDeviceSize hv_heap_size; uint32_t hv_heap_idx;
    VkShaderModule shader; VkDescriptorSetLayout dsl; VkPipelineLayout plyt; VkPipeline pipe;
    VkDescriptorPool dpool; VkDescriptorSet dset;
    VkBuffer dst_buf; VkDeviceMemory dst_mem;
    uint64_t rng;
} Dev;

static void build_pipeline(Dev *d, VkShaderModule shader) {
    VkDescriptorSetLayoutBinding b[2];
    for (int i = 0; i < 2; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(d->dev, &dsli, NULL, &d->dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 4};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &d->dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(d->dev, &pli, NULL, &d->plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = d->plyt};
    VKCHECK(vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &cpi, NULL, &d->pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(d->dev, &dpi, NULL, &d->dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = d->dpool, .descriptorSetCount = 1, .pSetLayouts = &d->dsl};
    VKCHECK(vkAllocateDescriptorSets(d->dev, &dsa, &d->dset), "allocDescSet");
    /* dst binding (1) is fixed for the device's lifetime; src binding (0) is
     * rebound per trial to point at whichever import + offset is current. */
    VkDescriptorBufferInfo dbi1 = {.buffer = d->dst_buf, .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr1 = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = d->dset, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi1};
    vkUpdateDescriptorSets(d->dev, 1, &wr1, 0, NULL);
}

static void setup_dev(Dev *d, VkPhysicalDevice phys, const char *bus_id, const char *label, const char *reduce_spv_path) {
    memset(d, 0, sizeof(*d));
    d->phys = phys; d->label = label; strncpy(d->bus_id, bus_id, sizeof(d->bus_id) - 1);
    d->rng = SEED ^ (uint64_t)(uintptr_t)label;

    int qfam = pick_backend_qfam(phys);
    if (qfam < 0) { fprintf(stderr, "FATAL: %s: no compute-capable queue family\n", label); exit(1); }
    d->qfam = (uint32_t)qfam;

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
    int ext_host_avail = 0;
    if (ep) {
        vkEnumerateDeviceExtensionProperties(phys, NULL, &ne, ep);
        for (uint32_t i = 0; i < ne; i++) if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) ext_host_avail = 1;
        free(ep);
    }
    if (!ext_host_avail) { fprintf(stderr, "FATAL: %s: VK_EXT_external_memory_host not advertised\n", label); exit(1); }
    const char *dext[1] = {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = d->qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = dext};
    VKCHECK(vkCreateDevice(phys, &di, NULL, &d->dev), "vkCreateDevice");
    vkGetDeviceQueue(d->dev, d->qfam, 0, &d->queue);

    d->fn_getHostPtrProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(d->dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!d->fn_getHostPtrProps) { fprintf(stderr, "FATAL: %s: vkGetMemoryHostPointerPropertiesEXT resolved NULL\n", label); exit(1); }

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &emh};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    d->min_align = emh.minImportedHostPointerAlignment;

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    int hv_mt = find_memtype(phys, 0xFFFFFFFFu, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NULL);
    if (hv_mt >= 0) { d->hv_heap_idx = mp.memoryTypes[hv_mt].heapIndex; d->hv_heap_size = mp.memoryHeaps[d->hv_heap_idx].size; }

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = d->qfam};
    VKCHECK(vkCreateCommandPool(d->dev, &cpci, NULL, &d->cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = d->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(d->dev, &cbi, &d->cmd), "cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(d->dev, &fi, NULL, &d->fence), "fence");

    int relaxed;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(d->dev, &bi, NULL, &d->dst_buf), "dst buf");
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(d->dev, d->dst_buf, &req);
    int mt = find_memtype(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &relaxed);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt};
    VKCHECK(vkAllocateMemory(d->dev, &ai, NULL, &d->dst_mem), "dst mem");
    VKCHECK(vkBindBufferMemory(d->dev, d->dst_buf, d->dst_mem, 0), "bind dst");

    d->shader = load_spv(d->dev, reduce_spv_path);
    build_pipeline(d, d->shader);

    printf("INFO dev %s bus=%s qfam=%u min_imported_host_ptr_align=%llu hv_heap_idx=%u hv_heap_size_gib=%.2f\n",
           label, bus_id, d->qfam, (unsigned long long)d->min_align, d->hv_heap_idx, (double)d->hv_heap_size / GIB);
}

/* ---- per-size import state ------------------------------------------------ */
typedef struct {
    int ok; VkResult stage_rc; const char *stage;
    VkBuffer buf; VkDeviceMemory mem; VkDeviceSize import_size;
} Import;

static void try_import(Dev *d, void *hostptr, size_t region_bytes, Import *im) {
    memset(im, 0, sizeof(*im));
    VkDeviceSize align = d->min_align ? d->min_align : 4096;
    VkDeviceSize import_size = (VkDeviceSize)((region_bytes / align) * align);
    if (import_size == 0) { im->stage = "region smaller than alignment"; return; }

    VkMemoryHostPointerPropertiesEXT hostProps = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult hpr = d->fn_getHostPtrProps(d->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, hostptr, &hostProps);
    if (hpr != VK_SUCCESS) { im->stage = "vkGetMemoryHostPointerPropertiesEXT"; im->stage_rc = hpr; return; }

    VkExternalMemoryBufferCreateInfo extbi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &extbi,
        .size = import_size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult cbr = vkCreateBuffer(d->dev, &bi, NULL, &im->buf);
    if (cbr != VK_SUCCESS) { im->stage = "vkCreateBuffer"; im->stage_rc = cbr; return; }

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(d->dev, im->buf, &req);
    uint32_t bits = req.memoryTypeBits & hostProps.memoryTypeBits;
    int relaxed = 0;
    int mt = find_memtype(d->phys, bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, &relaxed);
    if (mt < 0 || req.size > (VkDeviceSize)region_bytes) {
        vkDestroyBuffer(d->dev, im->buf, NULL); im->buf = VK_NULL_HANDLE;
        im->stage = mt < 0 ? "no compatible memory type" : "buffer requirement exceeds mapped region"; return;
    }
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = hostptr};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp, .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt};
    VkResult amr = vkAllocateMemory(d->dev, &ai, NULL, &im->mem);
    if (amr != VK_SUCCESS) {
        vkDestroyBuffer(d->dev, im->buf, NULL); im->buf = VK_NULL_HANDLE;
        im->stage = "vkAllocateMemory(import)"; im->stage_rc = amr; return;
    }
    VkResult bbr = vkBindBufferMemory(d->dev, im->buf, im->mem, 0);
    if (bbr != VK_SUCCESS) {
        vkFreeMemory(d->dev, im->mem, NULL); vkDestroyBuffer(d->dev, im->buf, NULL);
        im->buf = VK_NULL_HANDLE; im->mem = VK_NULL_HANDLE;
        im->stage = "vkBindBufferMemory"; im->stage_rc = bbr; return;
    }
    im->ok = 1; im->import_size = import_size;
}
static void free_import(Dev *d, Import *im) {
    if (im->mem) vkFreeMemory(d->dev, im->mem, NULL);
    if (im->buf) vkDestroyBuffer(d->dev, im->buf, NULL);
    memset(im, 0, sizeof(*im));
}

/* ---- read-bandwidth trial: random 256 MiB window, rebound every submit -- */
typedef struct {
    Dev *d; Import *im;
    double min_seconds;
    atomic_int *ready; int nparty;
    long long bytes; double secs;
} ReadArg;

static void bind_window(Dev *d, VkBuffer buf, VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo dbi0 = {.buffer = buf, .offset = offset, .range = range};
    VkWriteDescriptorSet wr0 = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = d->dset, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi0};
    vkUpdateDescriptorSets(d->dev, 1, &wr0, 0, NULL);
}
static void submit_read(Dev *d, uint32_t n_words) {
    VKCHECK(vkResetFences(d->dev, 1, &d->fence), "reset fence");
    VKCHECK(vkResetCommandBuffer(d->cmd, 0), "reset cmd");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(d->cmd, &bi), "begin cmd");
    vkCmdBindPipeline(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipe);
    vkCmdBindDescriptorSets(d->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->plyt, 0, 1, &d->dset, 0, NULL);
    vkCmdPushConstants(d->cmd, d->plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_words);
    vkCmdDispatch(d->cmd, REDUCE_WORKGROUPS, 1, 1);
    VKCHECK(vkEndCommandBuffer(d->cmd), "end cmd");
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &d->cmd};
    VKCHECK(vkQueueSubmit(d->queue, 1, &si, d->fence), "submit");
    VKCHECK(vkWaitForFences(d->dev, 1, &d->fence, VK_TRUE, 30000000000ULL), "wait fence");
}
static void *read_worker(void *argp) {
    ReadArg *a = argp; Dev *d = a->d; Import *im = a->im;
    uint32_t n_words = (uint32_t)(WINDOW_BYTES / 4);
    VkDeviceSize nwin = im->import_size / WINDOW_BYTES;

    VkDeviceSize off = (VkDeviceSize)(xorshift64s(&d->rng) % nwin) * WINDOW_BYTES;
    bind_window(d, im->buf, off, WINDOW_BYTES);
    submit_read(d, n_words);   /* warm-up, uncounted */
    atomic_fetch_add_explicit(a->ready, 1, memory_order_acq_rel);
    while (atomic_load_explicit(a->ready, memory_order_acquire) < a->nparty) sched_yield();

    long long n = 0; double t0 = now_s(), t = t0;
    do {
        off = (VkDeviceSize)(xorshift64s(&d->rng) % nwin) * WINDOW_BYTES;
        bind_window(d, im->buf, off, WINDOW_BYTES);
        submit_read(d, n_words);
        n++; t = now_s();
    } while (t - t0 < a->min_seconds);
    a->bytes = n * (long long)WINDOW_BYTES;
    a->secs = t - t0;
    return NULL;
}
static double run_read_group(Dev **ds, Import **ims, int n, ReadArg *outs) {
    atomic_int ready = 0; pthread_t th[3];
    for (int i = 0; i < n; i++) {
        outs[i] = (ReadArg){ds[i], ims[i], MIN_SECONDS, &ready, n, 0, 0};
        pthread_create(&th[i], NULL, read_worker, &outs[i]);
    }
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    long long total = 0; double max_secs = 0;
    for (int i = 0; i < n; i++) { total += outs[i].bytes; if (outs[i].secs > max_secs) max_secs = outs[i].secs; }
    return max_secs > 0 ? (double)total / max_secs / 1e9 : 0.0;
}

/* ---- main ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    const char *shard_dir = argc > 1 ? argv[1] : NULL;
    const char *reduce_spv_path = argc > 2 ? argv[2] : "vk_stream_reduce.spv";
    if (!shard_dir) { fprintf(stderr, "usage: %s <shard_dir> <reduce.spv>\n", argv[0]); return 2; }

    load_shards(shard_dir);
    build_list();

    {
        FILE *f = fopen("/sys/module/amdgpu/parameters/gttsize", "r");
        if (f) { char buf[64] = ""; if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; fclose(f);
            buf[strcspn(buf, "\n")] = 0; printf("INFO gttsize_param=%s\n", buf); }
        else printf("INFO gttsize_param=unreadable (%s) -- root-only on this rig\n", strerror(errno));
        f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
        if (f) { char buf[128] = ""; if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; fclose(f);
            buf[strcspn(buf, "\n")] = 0; printf("INFO thp_enabled=%s\n", buf); }
    }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst; VKCHECK(vkCreateInstance(&ici, NULL, &inst), "vkCreateInstance");
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice pdevs[8]; if (nd > 8) nd = 8; vkEnumeratePhysicalDevices(inst, &nd, pdevs);

    VkPhysicalDevice pdev0 = NULL, pdev2 = NULL, pdev3 = NULL;
    char bus0[32] = "", bus2[32] = "", bus3[32] = "";
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pdevs[i], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pci};
        vkGetPhysicalDeviceProperties2(pdevs[i], &p2);
        char bus[32]; snprintf(bus, sizeof(bus), "%04x:%02x:%02x.%01x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        if (!strcmp(bus, "0000:83:00.0")) { pdev0 = pdevs[i]; strcpy(bus0, bus); }
        else if (!strcmp(bus, "0000:48:00.0")) { pdev2 = pdevs[i]; strcpy(bus2, bus); }
        else if (!strcmp(bus, "0000:86:00.0")) { pdev3 = pdevs[i]; strcpy(bus3, bus); }
    }
    if (!pdev0 || !pdev2 || !pdev3) { fprintf(stderr, "FATAL: could not map all three PCI bus ids\n"); return 1; }

    Dev d0, d2, d3;
    setup_dev(&d0, pdev0, bus0, "dev0", reduce_spv_path);
    setup_dev(&d2, pdev2, bus2, "dev2", reduce_spv_path);
    setup_dev(&d3, pdev3, bus3, "dev3", reduce_spv_path);
    Dev *devs[3] = {&d0, &d2, &d3};
    const char *labels[3] = {"dev0", "dev2", "dev3"};

    size_t sizes_gib[] = {8, 32, 64, 96};
    const char *variants[] = {"anon", "anon-huge"};
    int stop = 0;

    for (int si = 0; si < 4 && !stop; si++) {
        size_t S = sizes_gib[si] * GIB;
        for (int vi = 0; vi < 2 && !stop; vi++) {
            const char *variant = variants[vi];
            unsigned long long mem_b = mem_available_kb();
            printf("INFO attempt size_gib=%zu variant=%s mem_available_before_kb=%llu\n", sizes_gib[si], variant, mem_b);

            void *region = mmap(NULL, S, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (region == MAP_FAILED) {
                printf("INFO import size_gib=%zu variant=%s ok=0 stage=mmap errno=%d (%s)\n", sizes_gib[si], variant, errno, strerror(errno));
                printf("INFO attempt size_gib=%zu variant=%s mem_available_after_kb=%llu\n", sizes_gib[si], variant, mem_available_kb());
                stop = 1; break;
            }
            if (!strcmp(variant, "anon-huge")) { if (madvise(region, S, MADV_HUGEPAGE) != 0) fprintf(stderr, "WARN madvise(HUGEPAGE) failed: %s\n", strerror(errno)); }
            else { madvise(region, S, MADV_NOHUGEPAGE); }

            double pop_gbps = populate_region((uint8_t *)region, S);
            printf("INFO populate size_gib=%zu variant=%s gbps=%.3f (load-time copy, NOT a gate metric)\n", sizes_gib[si], variant, pop_gbps);

            Import ims[3]; int all_ok = 1;
            for (int i = 0; i < 3; i++) {
                try_import(devs[i], region, S, &ims[i]);
                if (!ims[i].ok) {
                    printf("INFO import size_gib=%zu variant=%s dev=%s ok=0 stage=%s vkresult=%d\n",
                           sizes_gib[si], variant, labels[i], ims[i].stage ? ims[i].stage : "?", (int)ims[i].stage_rc);
                    all_ok = 0;
                } else {
                    printf("INFO import size_gib=%zu variant=%s dev=%s ok=1 import_bytes=%llu\n",
                           sizes_gib[si], variant, labels[i], (unsigned long long)ims[i].import_size);
                }
            }
            printf("INFO concurrent_import size_gib=%zu variant=%s all_three_ok=%d\n", sizes_gib[si], variant, all_ok);

            if (all_ok) {
                for (int i = 0; i < 3; i++) {
                    double s[REPS];
                    for (int r = 0; r < REPS; r++) {
                        Dev *g[1] = {devs[i]}; Import *im[1] = {&ims[i]}; ReadArg out[1];
                        s[r] = run_read_group(g, im, 1, out);
                    }
                    printf("ROW kind=import-read src=%s cards=1 block=256 size_gib=%zu gbps=%.3f n=%d dev=%s\n",
                           variant, sizes_gib[si], median(s, REPS), REPS, labels[i]);
                }
                {
                    double s_all[REPS], s_d[3][REPS];
                    for (int r = 0; r < REPS; r++) {
                        Dev *g[3] = {&d0, &d2, &d3}; Import *im[3] = {&ims[0], &ims[1], &ims[2]}; ReadArg out[3];
                        double agg = run_read_group(g, im, 3, out);
                        s_all[r] = agg;
                        for (int i = 0; i < 3; i++) s_d[i][r] = out[i].secs > 0 ? (double)out[i].bytes / out[i].secs / 1e9 : 0.0;
                    }
                    for (int i = 0; i < 3; i++)
                        printf("ROW kind=import-read src=%s cards=3 block=256 size_gib=%zu gbps=%.3f n=%d dev=%s\n",
                               variant, sizes_gib[si], median(s_d[i], REPS), REPS, labels[i]);
                    printf("ROW kind=import-read src=%s cards=3 block=256 size_gib=%zu gbps=%.3f n=%d dev=all\n",
                           variant, sizes_gib[si], median(s_all, REPS), REPS);
                }
                fflush(stdout);
            } else {
                stop = 1;
            }

            for (int i = 0; i < 3; i++) free_import(devs[i], &ims[i]);
            munmap(region, S);
            printf("INFO attempt size_gib=%zu variant=%s mem_available_after_kb=%llu\n", sizes_gib[si], variant, mem_available_kb());
            fflush(stdout);
            if (stop) break;
        }
    }

    printf("INFO done\n");
    return 0;
}
