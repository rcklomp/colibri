/* vk_stream_probe.c -- FRANKEN-ENGINE-PLAN-2026-09-15.md item F0: does
 * Vulkan/RADV deliver the same host->device streaming rate that
 * §PCIE-STREAM measured on HIP (1 card 28.0 GB/s, three cards concurrent
 * 61.4-62.0 GB/s, dev0+dev3 sharing an upstream link)? Colibri's engines
 * are Vulkan (glm53, qwen38-vk), not HIP, so F2's projected expert-stream
 * rate has to be re-measured on the API the engine actually uses.
 *
 * One pthread per physical device (persistent for the program's life: the
 * VkDevice, its transfer-capable queue, command pool, fence and buffers are
 * all created once in dev_setup() and reused across every trial -- only the
 * ephemeral pthread that drives one trial's timed loop is spawned and
 * joined per trial, mirroring pcie_stream_probe.cpp's per-trial worker
 * without paying VkDevice teardown/rebuild cost 48+ times). A start barrier
 * (atomic counter, spin-wait) makes a multi-device trial's timed window
 * genuinely concurrent across its participating devices, exactly as
 * pcie_stream_probe.cpp does for HIP streams.
 *
 * Four source kinds per (config, block):
 *   hv-coherent     HOST_VISIBLE|HOST_COHERENT staging buffer (plain host
 *                   RAM, explicitly NOT DEVICE_LOCAL -- see find_memtype's
 *                   avoid mask) -> device-local dst via vkCmdCopyBuffer.
 *                   The "pinned" analogue.
 *   hv-cached       same, +HOST_CACHED, if the device offers that
 *                   combination (RADV usually does); "unavailable"
 *                   otherwise.
 *   memcpy+upload   the page-cache analogue: memcpy from an mmap'd region
 *                   of a GLM shard already resident in page cache (read
 *                   only, never written) into the hv-coherent staging
 *                   buffer, THEN the same vkCmdCopyBuffer upload. The
 *                   memcpy leg and the upload leg are timed separately and
 *                   reported via INFO breakdown lines; the ROW line's gbps
 *                   is the combined figure (bytes / (memcpy+upload secs)).
 *   ext-host        VK_EXT_external_memory_host: import the mmap'd host
 *                   pointer directly as device memory bound to a buffer,
 *                   and vkCmdCopyBuffer straight from it -- no memcpy leg
 *                   at all. "unavailable" if the extension, the function
 *                   pointer, a compatible memory type, or a large-enough
 *                   file window is missing (each reported with its own
 *                   reason on stderr).
 *
 * Every submit batches BATCH copies of `block` bytes (all into the same
 * device-local dst region -- content is never read back, only bytes/sec
 * matters, so overwrite hazards are harmless) so the record's own
 * complaint about 0.3-1.5 ms per-submit gaps (§Q7) is amortised; BATCH is
 * printed once via `INFO batch_size=`.
 *
 * Devices are matched to dev0/dev2/dev3 by PCI bus id (VK_EXT_pci_bus_info,
 * VkPhysicalDevicePCIBusInfoPropertiesEXT), not by enumeration order --
 * verified against the record's own HIP-probe mapping (0000:83:00.0=dev0,
 * 0000:86:00.0=dev3, 0000:48:00.0=dev2), not assumed. Config 2 is
 * deliberately the dev0+dev3 pair (the ones sharing a link on HIP), so the
 * two probes' cfg=2 rows are the same physical comparison.
 *
 * Build (see vk_stream_chain.sh step 2 for the exact recipe):
 *   gcc -O2 -pthread vk_stream_probe.c -o vk_stream_probe -lvulkan -lm
 * Run:
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
 *     ./vk_stream_probe 2.0 ~/models/GLM-5.3-Flash-colibri-int4-g64/model-00002-of-00062.safetensors
 *
 * Output: INFO lines (setup/diagnostic, human+grep readable) then one ROW
 * line per (cfg,src,block,dev) measurement:
 *   ROW cfg=<1|2|3> src=<hv-coherent|hv-cached|memcpy+upload|ext-host> block=<MiB> dev=<busid|all> gbps=<..> n=<..>
 * gbps uses GB = 1e9 bytes, matching every other probe in this record.
 */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
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

static double MIN_SECONDS = 2.0;
static const int REPS_PER_VISIT = 3;   /* sequence 1,2,3,3,2,1 -> n=6 per row */
static const int BATCH = 4;            /* copies per submit, amortises the 0.3-1.5 ms/submit gap (§Q7) */
static const size_t MIB = 1024ull * 1024ull;
static const size_t MAXBLOCK = 256ull * 1024ull * 1024ull;         /* dst buffer size */
static const size_t MAXSTAGE = 4ull * 256ull * 1024ull * 1024ull;  /* BATCH * MAXBLOCK */

/* ---- generic memory-type search --------------------------------------- */

/* Two passes: first insist `avoid` bits are absent (so a staging buffer is
 * genuine host RAM and not the ReBAR host-visible+device-local VRAM window
 * that backend_vulkan.c's pick_memtype() is happy to accept for weight
 * tiers -- accepting it here would let some "uploads" skip PCIe entirely
 * and read as impossibly fast). Falls back to ignoring `avoid` only if no
 * type satisfies `want` at all, and says so. */
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

static void flags_str(VkMemoryPropertyFlags f, char *out, size_t n) {
    out[0] = 0;
    #define ADD(bit, name) if (f & (bit)) { if (out[0]) strncat(out, "|", n - strlen(out) - 1); strncat(out, name, n - strlen(out) - 1); }
    ADD(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL")
    ADD(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE")
    ADD(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT")
    ADD(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED")
    #undef ADD
    if (!out[0]) strncpy(out, "(none)", n - 1);
}

/* ---- per-device persistent context -------------------------------------- */

typedef struct {
    int idx;
    VkPhysicalDevice phys;
    char bus_id[32];
    char name[256];
    const char *label;      /* "dev0" / "dev2" / "dev3" */

    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    int qfam_dedicated;     /* transfer-capable, not graphics-capable */

    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkFence fence;

    VkBuffer dst_buf; VkDeviceMemory dst_mem;
    VkBuffer stage_coh_buf; VkDeviceMemory stage_coh_mem; void *stage_coh_ptr;
    int has_cached;
    VkBuffer stage_cached_buf; VkDeviceMemory stage_cached_mem; void *stage_cached_ptr;

    int has_ext_host;
    VkDeviceSize ext_host_align;
    PFN_vkGetMemoryHostPointerPropertiesEXT fn_getHostPtrProps;
} DevCtx;

static VkInstance g_inst;

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

static void dev_setup(DevCtx *d) {
    /* queue family: prefer a transfer-capable family that is NOT
     * graphics-capable (this rig's async-compute family, which also
     * carries TRANSFER_BIT) over the universal graphics/compute/transfer
     * family -- see the probe header for the measured queueFlags. Falls
     * back to any TRANSFER_BIT family, then to any COMPUTE/GRAPHICS family
     * (which implies transfer support per the Vulkan spec even when the
     * bit isn't explicitly reported). */
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &nq, qf);
    int best = -1, best_rank = -1;
    for (uint32_t i = 0; i < nq; i++) {
        VkQueueFlags f = qf[i].queueFlags;
        int rank = -1;
        if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) rank = 3;
        else if (f & VK_QUEUE_TRANSFER_BIT) rank = 2;
        else if (f & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) rank = 1;
        if (rank > best_rank) { best_rank = rank; best = (int)i; }
    }
    if (best < 0) { fprintf(stderr, "FATAL: %s has no usable queue family\n", d->label); exit(1); }
    d->qfam = (uint32_t)best;
    d->qfam_dedicated = (qf[best].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf[best].queueFlags & VK_QUEUE_GRAPHICS_BIT);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(d->phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
    int ext_host_avail = 0;
    if (ep) {
        vkEnumerateDeviceExtensionProperties(d->phys, NULL, &ne, ep);
        for (uint32_t i = 0; i < ne; i++)
            if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) ext_host_avail = 1;
        free(ep);
    }

    const char *dext[2]; uint32_t ndext = 0;
    if (ext_host_avail) dext[ndext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
        .enabledExtensionCount = ndext, .ppEnabledExtensionNames = ndext ? dext : NULL};
    VKCHECK(vkCreateDevice(d->phys, &di, NULL, &d->dev), "vkCreateDevice");
    vkGetDeviceQueue(d->dev, d->qfam, 0, &d->queue);

    d->has_ext_host = 0;
    d->ext_host_align = 0;
    if (ext_host_avail) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &emh};
        vkGetPhysicalDeviceProperties2(d->phys, &p2);
        d->ext_host_align = emh.minImportedHostPointerAlignment;
        d->fn_getHostPtrProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)
            vkGetDeviceProcAddr(d->dev, "vkGetMemoryHostPointerPropertiesEXT");
        d->has_ext_host = d->fn_getHostPtrProps != NULL;
        if (!d->has_ext_host)
            fprintf(stderr, "INFO %s: VK_EXT_external_memory_host advertised but vkGetMemoryHostPointerPropertiesEXT resolved NULL\n", d->label);
    } else {
        fprintf(stderr, "INFO %s: VK_EXT_external_memory_host not advertised -- ext-host source unavailable\n", d->label);
    }

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = d->qfam};
    VKCHECK(vkCreateCommandPool(d->dev, &cpci, NULL, &d->cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = d->cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(d->dev, &cbi, &d->cmd), "cmdBuf");
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(d->dev, &fi, NULL, &d->fence), "fence");

    int relaxed_dst = 0, relaxed_coh = 0, relaxed_cached = 0;
    create_buffer(d->dev, d->phys, (VkDeviceSize)MAXBLOCK, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                  &d->dst_buf, &d->dst_mem, NULL, &relaxed_dst);
    create_buffer(d->dev, d->phys, (VkDeviceSize)MAXSTAGE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &d->stage_coh_buf, &d->stage_coh_mem, &d->stage_coh_ptr, &relaxed_coh);
    memset(d->stage_coh_ptr, 0x5a, MAXSTAGE);

    /* HOST_CACHED staging: probe for the memtype first without allocating,
     * so an absent combination is "unavailable", not a fatal exit. */
    {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (VkDeviceSize)MAXSTAGE, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer probe_buf; VKCHECK(vkCreateBuffer(d->dev, &bi, NULL, &probe_buf), "probe cached buf");
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(d->dev, probe_buf, &req);
        int relaxed = 0;
        int mt = find_memtype(d->phys, req.memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &relaxed);
        vkDestroyBuffer(d->dev, probe_buf, NULL);
        d->has_cached = mt >= 0;
        if (d->has_cached) {
            create_buffer(d->dev, d->phys, (VkDeviceSize)MAXSTAGE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          &d->stage_cached_buf, &d->stage_cached_mem, &d->stage_cached_ptr, &relaxed_cached);
            memset(d->stage_cached_ptr, 0x5a, MAXSTAGE);
        } else {
            fprintf(stderr, "INFO %s: no HOST_VISIBLE|HOST_COHERENT|HOST_CACHED memory type -- hv-cached source unavailable\n", d->label);
        }
    }

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(d->phys, &mp);
    VkMemoryRequirements dreq; vkGetBufferMemoryRequirements(d->dev, d->dst_buf, &dreq);
    VkMemoryRequirements creq; vkGetBufferMemoryRequirements(d->dev, d->stage_coh_buf, &creq);
    /* recover the memtype index actually bound, for the INFO line -- walk
     * the same find_memtype calls again (cheap, setup-time only) */
    int dst_mt = find_memtype(d->phys, dreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, NULL);
    int coh_mt = find_memtype(d->phys, creq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NULL);
    char dst_fl[128], coh_fl[128];
    flags_str(dst_mt >= 0 ? mp.memoryTypes[dst_mt].propertyFlags : 0, dst_fl, sizeof(dst_fl));
    flags_str(coh_mt >= 0 ? mp.memoryTypes[coh_mt].propertyFlags : 0, coh_fl, sizeof(coh_fl));
    fprintf(stderr, "INFO %s bus=%s qfam=%u dedicated_transfer=%d dst_memtype=%d(%s) stage_coh_memtype=%d(%s) "
            "has_cached=%d ext_host=%d ext_host_align=%llu\n",
            d->label, d->bus_id, d->qfam, d->qfam_dedicated, dst_mt, dst_fl, coh_mt, coh_fl,
            d->has_cached, d->has_ext_host, (unsigned long long)d->ext_host_align);
    if (d->has_cached) {
        VkMemoryRequirements creq2; vkGetBufferMemoryRequirements(d->dev, d->stage_cached_buf, &creq2);
        int cached_mt = find_memtype(d->phys, creq2.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NULL);
        char cached_fl[128];
        flags_str(cached_mt >= 0 ? mp.memoryTypes[cached_mt].propertyFlags : 0, cached_fl, sizeof(cached_fl));
        fprintf(stderr, "INFO %s stage_cached_memtype=%d(%s)\n", d->label, cached_mt, cached_fl);
    }
}

/* ---- barrier ------------------------------------------------------------ */

static void barrier_wait(atomic_int *ready, int nparty) {
    atomic_fetch_add_explicit(ready, 1, memory_order_acq_rel);
    while (atomic_load_explicit(ready, memory_order_acquire) < nparty) sched_yield();
}

/* ---- one trial per device ------------------------------------------------
 *
 * Records `BATCH` vkCmdCopyBuffer regions into the persistent command
 * buffer (same recorded command buffer resubmitted every iteration --
 * legal because each vkQueueSubmit is followed by a fence wait before the
 * next one, so there is never more than one submission of it in flight),
 * runs one untimed warm-up submit, waits at the barrier, then loops
 * submit+fence-wait until >= MIN_SECONDS have elapsed. */

typedef struct {
    double gbps;          /* headline: upload gbps for hv-coherent, hv-cached and ext-host; combined for memcpy+upload */
    double gbps_memcpy;   /* memcpy+upload only */
    double gbps_upload;   /* memcpy+upload only (upload leg alone) */
    long long bytes;
    double secs;
    int unavailable;
    char reason[160];
} TrialOut;

static void record_copy_cb(DevCtx *d, VkBuffer src, size_t block) {
    VKCHECK(vkResetCommandBuffer(d->cmd, 0), "reset cmd");
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(d->cmd, &bi), "begin cmd");
    VkBufferCopy regions[16];
    for (int i = 0; i < BATCH; i++) {
        regions[i].srcOffset = (VkDeviceSize)i * block;
        regions[i].dstOffset = 0;
        regions[i].size = (VkDeviceSize)block;
    }
    vkCmdCopyBuffer(d->cmd, src, d->dst_buf, BATCH, regions);
    VKCHECK(vkEndCommandBuffer(d->cmd), "end cmd");
}

static void submit_wait(DevCtx *d) {
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &d->cmd};
    VKCHECK(vkResetFences(d->dev, 1, &d->fence), "reset fence");
    VKCHECK(vkQueueSubmit(d->queue, 1, &si, d->fence), "submit");
    VKCHECK(vkWaitForFences(d->dev, 1, &d->fence, VK_TRUE, 30000000000ULL), "wait fence");
}

typedef struct {
    DevCtx *d;
    const char *src;      /* "hv-coherent" | "hv-cached" | "memcpy+upload" | "ext-host" */
    size_t block;
    const uint8_t *filebase;
    size_t filesize;
    size_t base_off;       /* per-device rotation seed, in units of `block` */
    atomic_int *ready;
    int nparty;
    TrialOut *out;
} TrialArg;

static void *trial_thread(void *argp) {
    TrialArg *a = argp;
    DevCtx *d = a->d;
    TrialOut *out = a->out;
    memset(out, 0, sizeof(*out));

    if (!strcmp(a->src, "hv-coherent")) {
        record_copy_cb(d, d->stage_coh_buf, a->block);
        submit_wait(d);                       /* warm-up, uncounted */
        barrier_wait(a->ready, a->nparty);
        long long n = 0; double t0 = now_s(), t = t0;
        do { submit_wait(d); n++; t = now_s(); } while (t - t0 < MIN_SECONDS);
        out->bytes = n * (long long)BATCH * a->block;
        out->secs = t - t0;
        out->gbps = out->gbps_upload = (double)out->bytes / out->secs / 1e9;
        return NULL;
    }
    if (!strcmp(a->src, "hv-cached")) {
        if (!d->has_cached) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "no HOST_VISIBLE|HOST_COHERENT|HOST_CACHED memory type"); barrier_wait(a->ready, a->nparty); return NULL; }
        record_copy_cb(d, d->stage_cached_buf, a->block);
        submit_wait(d);
        barrier_wait(a->ready, a->nparty);
        long long n = 0; double t0 = now_s(), t = t0;
        do { submit_wait(d); n++; t = now_s(); } while (t - t0 < MIN_SECONDS);
        out->bytes = n * (long long)BATCH * a->block;
        out->secs = t - t0;
        out->gbps = out->gbps_upload = (double)out->bytes / out->secs / 1e9;
        return NULL;
    }
    if (!strcmp(a->src, "memcpy+upload")) {
        if (!a->filebase) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "no shard file mmap'd"); barrier_wait(a->ready, a->nparty); return NULL; }
        size_t nwin = a->filesize / a->block;
        if (nwin < 1) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "shard smaller than one block"); barrier_wait(a->ready, a->nparty); return NULL; }
        record_copy_cb(d, d->stage_coh_buf, a->block);
        size_t cursor = a->base_off;
        for (int i = 0; i < BATCH; i++) { size_t w = cursor % nwin; memcpy((uint8_t *)d->stage_coh_ptr + (size_t)i * a->block, a->filebase + w * a->block, a->block); cursor++; }
        submit_wait(d);                       /* warm-up */
        barrier_wait(a->ready, a->nparty);
        long long n = 0; double t_mcpy = 0, t_up = 0, t0 = now_s(), t = t0;
        do {
            double a0 = now_s();
            for (int i = 0; i < BATCH; i++) { size_t w = cursor % nwin; memcpy((uint8_t *)d->stage_coh_ptr + (size_t)i * a->block, a->filebase + w * a->block, a->block); cursor++; }
            double a1 = now_s();
            submit_wait(d);
            double a2 = now_s();
            t_mcpy += a1 - a0; t_up += a2 - a1;
            n++; t = now_s();
        } while (t - t0 < MIN_SECONDS);
        out->bytes = n * (long long)BATCH * a->block;
        out->secs = t - t0;
        out->gbps_memcpy = (double)out->bytes / t_mcpy / 1e9;
        out->gbps_upload = (double)out->bytes / t_up / 1e9;
        out->gbps = (double)out->bytes / (t_mcpy + t_up) / 1e9;
        return NULL;
    }
    if (!strcmp(a->src, "ext-host")) {
        if (!d->has_ext_host) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "VK_EXT_external_memory_host unavailable on this device"); barrier_wait(a->ready, a->nparty); return NULL; }
        if (!a->filebase) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "no shard file mmap'd"); barrier_wait(a->ready, a->nparty); return NULL; }
        size_t win = (size_t)BATCH * a->block;
        size_t nwin = a->filesize / win;
        if (nwin < 1) { out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "shard smaller than one BATCH*block window (%zu)", win); barrier_wait(a->ready, a->nparty); return NULL; }
        size_t off = (a->base_off % nwin) * win;
        void *hostptr = (void *)(a->filebase + off);   /* page-aligned: win and off are both multiples of the 4 KiB page */

        VkExternalMemoryBufferCreateInfo extbi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT};
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &extbi,
            .size = (VkDeviceSize)win, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer ext_buf;
        if (vkCreateBuffer(d->dev, &bi, NULL, &ext_buf) != VK_SUCCESS) {
            out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "vkCreateBuffer(external) failed"); barrier_wait(a->ready, a->nparty); return NULL;
        }
        VkMemoryHostPointerPropertiesEXT hostProps = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        VkResult hpr = d->fn_getHostPtrProps(d->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT, hostptr, &hostProps);
        if (hpr != VK_SUCCESS) {
            vkDestroyBuffer(d->dev, ext_buf, NULL);
            out->unavailable = 1;
            snprintf(out->reason, sizeof(out->reason),
                     "vkGetMemoryHostPointerPropertiesEXT rc=%d ptr=%p off=%zu win=%zu align=%llu",
                     (int)hpr, hostptr, off, win, (unsigned long long)d->ext_host_align);
            barrier_wait(a->ready, a->nparty); return NULL;
        }
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(d->dev, ext_buf, &req);
        uint32_t bits = req.memoryTypeBits & hostProps.memoryTypeBits;
        int relaxed = 0;
        int mt = find_memtype(d->phys, bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, &relaxed);
        if (mt < 0 || req.size > win) {
            vkDestroyBuffer(d->dev, ext_buf, NULL);
            out->unavailable = 1;
            snprintf(out->reason, sizeof(out->reason), "%s", mt < 0 ? "no memory type compatible with imported host pointer" : "buffer memory requirement exceeds mapped window");
            barrier_wait(a->ready, a->nparty); return NULL;
        }
        VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT, .pHostPointer = hostptr};
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
            .allocationSize = req.size, .memoryTypeIndex = (uint32_t)mt};
        VkDeviceMemory ext_mem;
        if (vkAllocateMemory(d->dev, &ai, NULL, &ext_mem) != VK_SUCCESS) {
            vkDestroyBuffer(d->dev, ext_buf, NULL);
            out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "vkAllocateMemory(import) failed"); barrier_wait(a->ready, a->nparty); return NULL;
        }
        VKCHECK(vkBindBufferMemory(d->dev, ext_buf, ext_mem, 0), "bind imported");

        record_copy_cb(d, ext_buf, a->block);
        submit_wait(d);
        barrier_wait(a->ready, a->nparty);
        long long n = 0; double t0 = now_s(), t = t0;
        do { submit_wait(d); n++; t = now_s(); } while (t - t0 < MIN_SECONDS);
        out->bytes = n * (long long)BATCH * a->block;
        out->secs = t - t0;
        out->gbps = out->gbps_upload = (double)out->bytes / out->secs / 1e9;
        vkDestroyBuffer(d->dev, ext_buf, NULL);
        vkFreeMemory(d->dev, ext_mem, NULL);
        return NULL;
    }
    out->unavailable = 1; snprintf(out->reason, sizeof(out->reason), "unknown source kind %s", a->src);
    barrier_wait(a->ready, a->nparty);
    return NULL;
}

/* Runs one trial across `n` participating devices concurrently, returns
 * per-device TrialOut plus the aggregate gbps (sum bytes / max secs over
 * devices that were NOT unavailable -- matching pcie_stream_probe.cpp's
 * aggregate definition). */
static double run_trial_group(DevCtx **devs, int n, const char *src, size_t block,
                               const uint8_t *filebase, size_t filesize,
                               TrialOut *outs) {
    atomic_int ready = 0;
    pthread_t th[3]; TrialArg args[3];
    for (int i = 0; i < n; i++) {
        args[i] = (TrialArg){devs[i], src, block, filebase, filesize, (size_t)devs[i]->idx * 97, &ready, n, &outs[i]};
        pthread_create(&th[i], NULL, trial_thread, &args[i]);
    }
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    long long total_bytes = 0; double max_secs = 0.0; int any = 0;
    for (int i = 0; i < n; i++) {
        if (outs[i].unavailable) continue;
        any = 1;
        total_bytes += outs[i].bytes;
        if (outs[i].secs > max_secs) max_secs = outs[i].secs;
    }
    if (!any || max_secs <= 0) return 0.0;
    return (double)total_bytes / max_secs / 1e9;
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

/* ---- main ---------------------------------------------------------------- */

static void print_row(const char *cfg, const char *src, size_t block_mib, const char *dev, double gbps, int n) {
    printf("ROW cfg=%s src=%s block=%zu dev=%s gbps=%.3f n=%d\n", cfg, src, block_mib, dev, gbps, n);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc > 1) MIN_SECONDS = atof(argv[1]);
    const char *shard_path = argc > 2 ? argv[2] : NULL;

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &g_inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(g_inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "FATAL: no Vulkan physical devices\n"); return 1; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(g_inst, &nd, devs);

    DevCtx discs[8]; int ndisc = 0;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        DevCtx *d = &discs[ndisc++];
        memset(d, 0, sizeof(*d));
        d->idx = (int)i; d->phys = devs[i];
        strncpy(d->name, p.deviceName, sizeof(d->name) - 1);
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pci};
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        snprintf(d->bus_id, sizeof(d->bus_id), "%04x:%02x:%02x.%01x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        if (!strcmp(d->bus_id, "0000:83:00.0")) d->label = "dev0";
        else if (!strcmp(d->bus_id, "0000:86:00.0")) d->label = "dev3";
        else if (!strcmp(d->bus_id, "0000:48:00.0")) d->label = "dev2";
        else d->label = "unknown";
        printf("INFO discrete idx=%d bus_id=%s name=\"%s\" label=%s\n", d->idx, d->bus_id, d->name, d->label);
    }
    printf("INFO device_count=%d min_seconds=%.1f reps_per_visit=%d batch_size=%d\n", ndisc, MIN_SECONDS, REPS_PER_VISIT, BATCH);
    if (ndisc < 3) { fprintf(stderr, "FATAL: expected >= 3 discrete GPUs, found %d\n", ndisc); return 1; }

    DevCtx *dev0 = NULL, *dev2 = NULL, *dev3 = NULL;
    for (int i = 0; i < ndisc; i++) {
        if (!strcmp(discs[i].label, "dev0")) dev0 = &discs[i];
        else if (!strcmp(discs[i].label, "dev2")) dev2 = &discs[i];
        else if (!strcmp(discs[i].label, "dev3")) dev3 = &discs[i];
    }
    if (!dev0 || !dev2 || !dev3) {
        fprintf(stderr, "FATAL: could not map all three PCI bus ids to dev0/dev2/dev3 (dev0=%p dev2=%p dev3=%p)\n",
                (void *)dev0, (void *)dev2, (void *)dev3);
        return 1;
    }

    dev_setup(dev0); dev_setup(dev2); dev_setup(dev3);

    const uint8_t *filebase = NULL; size_t filesize = 0;
    if (shard_path) {
        int fd = open(shard_path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "WARN open(%s) failed: %s -- memcpy+upload and ext-host sources unavailable\n", shard_path, strerror(errno));
        } else {
            struct stat st;
            if (fstat(fd, &st) == 0) {
                filesize = (size_t)st.st_size;
                void *m = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
                if (m == MAP_FAILED) { fprintf(stderr, "WARN mmap(%s) failed: %s\n", shard_path, strerror(errno)); filesize = 0; }
                else filebase = (const uint8_t *)m;
            }
            close(fd);
        }
    }
    printf("INFO shard_path=%s shard_size=%zu mmap=%s\n", shard_path ? shard_path : "(none)", filesize, filebase ? "ok" : "unavailable");

    const char *combos_src[] = {"hv-coherent", "hv-coherent", "hv-cached", "hv-cached", "memcpy+upload", "memcpy+upload", "ext-host", "ext-host"};
    size_t combos_block[] = {14*MIB, 256*MIB, 14*MIB, 256*MIB, 14*MIB, 256*MIB, 14*MIB, 256*MIB};
    const int NCOMBO = 8;

    DevCtx *cfg1_devs[3] = {dev0, dev2, dev3};
    DevCtx *cfg2_group[2] = {dev0, dev3};
    DevCtx *cfg3_group[3] = {dev0, dev2, dev3};

    /* accumulators: [combo][slot] where slot indexes into the relevant device list */
    double cfg1_samp[8][3][64]; int cfg1_n[8][3];
    double cfg2_dev_samp[8][2][64]; int cfg2_dev_n[8][2];
    double cfg2_agg_samp[8][64]; int cfg2_agg_n[8];
    double cfg3_dev_samp[8][3][64]; int cfg3_dev_n[8][3];
    double cfg3_agg_samp[8][64]; int cfg3_agg_n[8];
    /* memcpy+upload breakdown, cfg1 only (per-device, single-device legs are unambiguous) */
    double cfg1_mcpy_samp[3][64], cfg1_up_samp[3][64]; int combo_is_mcpy[8];
    memset(cfg1_n, 0, sizeof(cfg1_n)); memset(cfg2_dev_n, 0, sizeof(cfg2_dev_n));
    memset(cfg2_agg_n, 0, sizeof(cfg2_agg_n)); memset(cfg3_dev_n, 0, sizeof(cfg3_dev_n));
    memset(cfg3_agg_n, 0, sizeof(cfg3_agg_n));
    int mcpy_n[3] = {0,0,0};
    for (int c = 0; c < NCOMBO; c++) combo_is_mcpy[c] = !strcmp(combos_src[c], "memcpy+upload");

    const char *sequence[] = {"1", "2", "3", "3", "2", "1"};
    for (int s = 0; s < 6; s++) {
        const char *cfg = sequence[s];
        printf("INFO visit cfg=%s\n", cfg);
        if (!strcmp(cfg, "1")) {
            for (int g = 0; g < 3; g++) {
                for (int c = 0; c < NCOMBO; c++) {
                    for (int r = 0; r < REPS_PER_VISIT; r++) {
                        TrialOut out[1];
                        DevCtx *one[1] = {cfg1_devs[g]};
                        double agg = run_trial_group(one, 1, combos_src[c], combos_block[c], filebase, filesize, out);
                        (void)agg;
                        if (!out[0].unavailable) {
                            cfg1_samp[c][g][cfg1_n[c][g]++] = out[0].gbps;
                            if (combo_is_mcpy[c] && g < 3) {
                                cfg1_mcpy_samp[g][mcpy_n[g]] = out[0].gbps_memcpy;
                                cfg1_up_samp[g][mcpy_n[g]] = out[0].gbps_upload;
                                mcpy_n[g]++;
                            }
                        } else if (cfg1_n[c][g] == 0) {
                            fprintf(stderr, "INFO cfg=1 src=%s block=%zu dev=%s unavailable: %s\n",
                                    combos_src[c], combos_block[c] / MIB, cfg1_devs[g]->label, out[0].reason);
                        }
                    }
                }
            }
        } else if (!strcmp(cfg, "2")) {
            for (int c = 0; c < NCOMBO; c++) {
                for (int r = 0; r < REPS_PER_VISIT; r++) {
                    TrialOut out[2];
                    double agg = run_trial_group(cfg2_group, 2, combos_src[c], combos_block[c], filebase, filesize, out);
                    int any = 0;
                    for (int i = 0; i < 2; i++) {
                        if (!out[i].unavailable) { cfg2_dev_samp[c][i][cfg2_dev_n[c][i]++] = out[i].gbps; any = 1; }
                        else if (cfg2_dev_n[c][i] == 0)
                            fprintf(stderr, "INFO cfg=2 src=%s block=%zu dev=%s unavailable: %s\n",
                                    combos_src[c], combos_block[c] / MIB, cfg2_group[i]->label, out[i].reason);
                    }
                    if (any) cfg2_agg_samp[c][cfg2_agg_n[c]++] = agg;
                }
            }
        } else { /* "3" */
            for (int c = 0; c < NCOMBO; c++) {
                for (int r = 0; r < REPS_PER_VISIT; r++) {
                    TrialOut out[3];
                    double agg = run_trial_group(cfg3_group, 3, combos_src[c], combos_block[c], filebase, filesize, out);
                    int any = 0;
                    for (int i = 0; i < 3; i++) {
                        if (!out[i].unavailable) { cfg3_dev_samp[c][i][cfg3_dev_n[c][i]++] = out[i].gbps; any = 1; }
                        else if (cfg3_dev_n[c][i] == 0)
                            fprintf(stderr, "INFO cfg=3 src=%s block=%zu dev=%s unavailable: %s\n",
                                    combos_src[c], combos_block[c] / MIB, cfg3_group[i]->label, out[i].reason);
                    }
                    if (any) cfg3_agg_samp[c][cfg3_agg_n[c]++] = agg;
                }
            }
        }
    }

    /* ---- print ROW lines ---- */
    for (int g = 0; g < 3; g++)
        for (int c = 0; c < NCOMBO; c++) {
            if (cfg1_n[c][g] == 0) continue;
            print_row("1", combos_src[c], combos_block[c] / MIB, cfg1_devs[g]->bus_id,
                      median(cfg1_samp[c][g], cfg1_n[c][g]), cfg1_n[c][g]);
        }
    for (int c = 0; c < NCOMBO; c++) {
        for (int i = 0; i < 2; i++)
            if (cfg2_dev_n[c][i] > 0)
                print_row("2", combos_src[c], combos_block[c] / MIB, cfg2_group[i]->bus_id,
                          median(cfg2_dev_samp[c][i], cfg2_dev_n[c][i]), cfg2_dev_n[c][i]);
        if (cfg2_agg_n[c] > 0)
            print_row("2", combos_src[c], combos_block[c] / MIB, "all",
                      median(cfg2_agg_samp[c], cfg2_agg_n[c]), cfg2_agg_n[c]);
    }
    for (int c = 0; c < NCOMBO; c++) {
        for (int i = 0; i < 3; i++)
            if (cfg3_dev_n[c][i] > 0)
                print_row("3", combos_src[c], combos_block[c] / MIB, cfg3_group[i]->bus_id,
                          median(cfg3_dev_samp[c][i], cfg3_dev_n[c][i]), cfg3_dev_n[c][i]);
        if (cfg3_agg_n[c] > 0)
            print_row("3", combos_src[c], combos_block[c] / MIB, "all",
                      median(cfg3_agg_samp[c], cfg3_agg_n[c]), cfg3_agg_n[c]);
    }

    /* memcpy+upload breakdown (cfg=1 only, per device, both blocks pooled per device
     * label into one INFO line each -- the ROW lines above already carry the
     * per-block combined figure; this is the "separately" requirement). */
    for (int g = 0; g < 3; g++) {
        if (mcpy_n[g] == 0) continue;
        printf("INFO breakdown cfg=1 src=memcpy+upload dev=%s memcpy_gbps=%.3f upload_gbps=%.3f n=%d\n",
               cfg1_devs[g]->bus_id, median(cfg1_mcpy_samp[g], mcpy_n[g]), median(cfg1_up_samp[g], mcpy_n[g]), mcpy_n[g]);
    }

    printf("INFO done\n");
    return 0;
}
