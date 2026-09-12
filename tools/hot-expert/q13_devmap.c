/* Q13 step 0 probe: which physical GPU is the engine's dev0/dev2/dev3?
 * Replicates c/backend_vulkan.c's selection EXACTLY:
 *   dev0  = coli_vk_init():      highest rank (discrete 4 > integrated 3 > virtual 2 > other 1 > cpu 0),
 *                                strict '>' so ties keep the FIRST in enumeration order.
 *   dev2  = coli_vk_init_dev2(-1, "auto"): best-ranked device that is not dev0, strict '>' again.
 *   dev3  = coli_vk_init_dev3(-1, "auto"): best-ranked device that is neither dev0 nor dev2.
 * Prints the PCI address of each enumeration index via VK_EXT_pci_bus_info.
 * Creates NO logical device and allocates NO memory. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>

static int rank_of(VkPhysicalDevice d) {
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
    return p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
           p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
           p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2 :
           p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER          ? 1 : 0;
}
static int rank2_of(VkPhysicalDevice d) { /* dev2/dev3 loop: only discrete/integrated count */
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
    return p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
           p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 : -1;
}

int main(void) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { fprintf(stderr, "no instance\n"); return 1; }
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(inst, &nd, devs);

    char pci[8][32];
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT bus = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &bus};
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        uint32_t ne = 0; int has_pci = 0;
        vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) { vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ep);
            for (uint32_t k = 0; k < ne; k++) if (!strcmp(ep[k].extensionName, "VK_EXT_pci_bus_info")) has_pci = 1;
            free(ep); }
        if (has_pci) snprintf(pci[i], sizeof pci[i], "%04x:%02x:%02x.%x",
                              bus.pciDomain, bus.pciBus, bus.pciDevice, bus.pciFunction);
        else snprintf(pci[i], sizeof pci[i], "(no pci_bus_info)");
        printf("enum[%u] rank=%d type=%u %-40s pci=%s\n", i, rank_of(devs[i]), p2.properties.deviceType,
               p2.properties.deviceName, pci[i]);
    }

    int i0 = 0, best = -1;
    for (uint32_t i = 0; i < nd; i++) { int r = rank_of(devs[i]); if (r > best) { best = r; i0 = (int)i; } }
    int i2 = -1; best = -1;
    for (uint32_t i = 0; i < nd; i++) { if ((int)i == i0) continue; int r = rank2_of(devs[i]); if (r > best) { best = r; i2 = (int)i; } }
    int i3 = -1; best = -1;
    for (uint32_t i = 0; i < nd; i++) { if ((int)i == i0 || (int)i == i2) continue; int r = rank2_of(devs[i]); if (r > best) { best = r; i3 = (int)i; } }
    printf("\nENGINE dev0 = enum[%d] pci=%s\n", i0, pci[i0]);
    if (i2 >= 0) printf("ENGINE dev2(auto) = enum[%d] pci=%s\n", i2, pci[i2]);
    if (i3 >= 0) printf("ENGINE dev3(auto) = enum[%d] pci=%s\n", i3, pci[i3]);

    /* fd-order cross-check: does DRM fd open order match enumeration order in THIS process? */
    printf("\nmy own /dev/dri fds (open order):\n");
    DIR *d = opendir("/proc/self/fd"); struct dirent *e;
    int fds[64], n = 0;
    while (d && (e = readdir(d))) { int f = atoi(e->d_name); if (f > 2) fds[n < 64 ? n++ : 63] = f; }
    if (d) closedir(d);
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (fds[b] < fds[a]) { int t = fds[a]; fds[a] = fds[b]; fds[b] = t; }
    for (int a = 0; a < n; a++) {
        char lp[64], tgt[256]; snprintf(lp, sizeof lp, "/proc/self/fd/%d", fds[a]);
        ssize_t r = readlink(lp, tgt, sizeof tgt - 1); if (r <= 0) continue; tgt[r] = 0;
        if (strncmp(tgt, "/dev/dri/", 9)) continue;
        char fi[64]; snprintf(fi, sizeof fi, "/proc/self/fdinfo/%d", fds[a]);
        FILE *f = fopen(fi, "r"); char line[256], pdev[64] = "?";
        while (f && fgets(line, sizeof line, f)) if (!strncmp(line, "drm-pdev", 8)) { char *c = strchr(line, ':'); if (c) { c++; while (*c == ' ' || *c == '\t') c++; line[strcspn(line, "\n")] = 0; snprintf(pdev, sizeof pdev, "%s", c); } }
        if (f) fclose(f);
        printf("  fd %d -> %s  drm-pdev=%s\n", fds[a], tgt, pdev);
    }
    return 0;
}
