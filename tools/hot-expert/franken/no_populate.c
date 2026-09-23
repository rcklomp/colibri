/* tools/hot-expert/franken/no_populate.c
 *
 * LD_PRELOAD shim for running libllama tools (franken_oracle_dump) against a
 * model that must NOT be read in full on this box.
 *
 * llama_model_loader::init_mappings() always maps with prefetch on
 * (llama-model.cpp: ml.init_mappings(true, ...)), which in llama-mmap.cpp
 * means MAP_POPULATE on the whole shard -- or, with lazy ranges, a
 * POSIX_MADV_WILLNEED over everything else -- and POSIX_FADV_SEQUENTIAL. On
 * DeepSeek-V4-Flash that is an 85 GB read into a page cache the served
 * GLM-5.3 lives in. This shim drops MAP_POPULATE from every mmap and turns
 * WILLNEED / SEQUENTIAL advice into no-ops, so the mapping stays lazy and a
 * run pages in exactly what the graph touches. It changes no data and no
 * arithmetic: the same bytes arrive through ordinary page faults.
 *
 * Build: gcc -shared -fPIC -O2 -o no_populate.so no_populate.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/types.h>

typedef void * (*mmap_fn)(void *, size_t, int, int, int, off_t);

void * mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    static mmap_fn real = 0;
    if (!real) real = (mmap_fn) dlsym(RTLD_NEXT, "mmap");
    return real(addr, len, prot, flags & ~MAP_POPULATE, fd, off);
}

void * mmap64(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    static mmap_fn real = 0;
    if (!real) real = (mmap_fn) dlsym(RTLD_NEXT, "mmap64");
    return real(addr, len, prot, flags & ~MAP_POPULATE, fd, off);
}

int posix_madvise(void * addr, size_t len, int advice) {
    static int (*real)(void *, size_t, int) = 0;
    if (advice == POSIX_MADV_WILLNEED || advice == POSIX_MADV_SEQUENTIAL) return 0;
    if (!real) real = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "posix_madvise");
    return real(addr, len, advice);
}

int madvise(void * addr, size_t len, int advice) {
    static int (*real)(void *, size_t, int) = 0;
    if (advice == MADV_WILLNEED || advice == MADV_SEQUENTIAL) return 0;
    if (!real) real = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
    return real(addr, len, advice);
}

int posix_fadvise(int fd, off_t off, off_t len, int advice) {
    static int (*real)(int, off_t, off_t, int) = 0;
    if (advice == POSIX_FADV_WILLNEED || advice == POSIX_FADV_SEQUENTIAL) return 0;
    if (!real) real = (int (*)(int, off_t, off_t, int)) dlsym(RTLD_NEXT, "posix_fadvise");
    return real(fd, off, len, advice);
}
