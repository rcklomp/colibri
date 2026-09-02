"""Serve expert weights straight from a file mapping instead of pread-into-malloc.

An expert is 14.16 MB in six pieces (gate/up/down, each weights + scales). On a
miss the engine malloc'd a 14.16 MB slot and pread the six pieces into it —
copying bytes the kernel was already holding in page cache. Those private copies
are also what forced the slot cache to compete with the page cache for RAM.

The pieces are NOT consecutive in the file (measured: 0 of 12096 experts are
contiguous), so a slot cannot point at one contiguous run. But nothing requires
the six pieces to be adjacent in memory either — expert_mats only ever builds
three read-only Mat views over them. So the slot becomes six pointers rather
than one buffer, and each piece points directly into its file mapping. No copy.

The codebase uses only unaligned SIMD loads (_mm*_loadu_*), so mapping at the
file's natural offsets is safe; we still require 4-byte alignment because the
scale pieces are dereferenced as float*.

Slots are reused and slot->base was never freed, so a recycled slot must never
be pread into while it still points at a read-only mapping. Hence `own`: the
malloc'd buffer, allocated only for experts that cannot be mapped.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("#define GLM53_EXPERT_PIECES 6",
    "#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n\n"
    "#define GLM53_EXPERT_PIECES 6",
    "includes")

rep("typedef struct { int eid; uint8_t *base; uint64_t used; } Slot;",
    "/* I sei pezzi non sono adiacenti nel file, ma non devono esserlo nemmeno in\n"
    " * memoria: expert_mats ci costruisce sopra solo tre viste in sola lettura. */\n"
    "typedef struct { int eid; uint8_t *piece[GLM53_EXPERT_PIECES]; uint8_t *own; uint64_t used; int mapped; } Slot;\n"
    "\n"
    "/* Una mappatura per file, non per esperto: gli offset dei pezzi non sono\n"
    " * allineati alla pagina, ma mappando il file intero ci pensa il kernel. */\n"
    "#define GLM53_MAXFD 4096\n"
    "static uint8_t *g_fmap[GLM53_MAXFD];\n"
    "static size_t   g_fmaplen[GLM53_MAXFD];\n"
    "static long     g_map_serve, g_map_copy;\n"
    "static int      g_map_active, g_map_all;",
    "slot pieces + map table")

rep("static void expert_cache_init(GModel *m) {\n    const Cfg *c = &m->c;",
    """/* Un pezzo e' mappabile se il suo file e' mappato, ci sta dentro, ed e'
 * allineato a 4 byte (le scale si leggono come float *). */
static int piece_mappable(const GModel *m, const ERef *ref, int q) {
    const int fd = ref->fd[q];
    if (fd <= 0 || fd >= GLM53_MAXFD || !g_fmap[fd]) return 0;
    if ((size_t)ref->off[q] + (size_t)m->e_len[q] > g_fmaplen[fd]) return 0;
    return (ref->off[q] & 3) == 0;
}

static void expert_map_init(GModel *m) {
    if (getenv("GLM53_NO_MMAP") || !m->eref) return;
    const Cfg *c = &m->c;
    int files = 0;
    for (int i = 0; i < c->n_layers; i++)
        for (int e = 0; e < c->n_experts; e++) {
            const ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            if (ref->fd[0] <= 0) continue;
            for (int q = 0; q < GLM53_EXPERT_PIECES; q++) {
                const int fd = ref->fd[q];
                if (fd <= 0 || fd >= GLM53_MAXFD || g_fmap[fd]) continue;
                struct stat st;
                if (fstat(fd, &st) != 0 || st.st_size <= 0) continue;
                void *pm = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (pm == MAP_FAILED) continue;
                g_fmap[fd] = (uint8_t *)pm;
                g_fmaplen[fd] = (size_t)st.st_size;
                files++;
            }
        }
    if (!files) return;
    g_map_active = 1;
    long full = 0, partial = 0;
    for (int i = 0; i < c->n_layers; i++)
        for (int e = 0; e < c->n_experts; e++) {
            const ERef *ref = &m->eref[(size_t)i * c->n_experts + e];
            if (ref->fd[0] <= 0) continue;
            int ok = 1;
            for (int q = 0; q < GLM53_EXPERT_PIECES; q++)
                if (!piece_mappable(m, ref, q)) { ok = 0; break; }
            if (ok) full++; else partial++;
        }
    g_map_all = (partial == 0 && full > 0);
    fprintf(stderr, "[MAP] %d file mappati, esperti mappabili %ld/%ld%s\\n",
            files, full, full + partial, g_map_all ? " (tutti: nessuna copia)" : "");
}

static void expert_cache_init(GModel *m) {
    const Cfg *c = &m->c;
    if (!g_map_active) expert_map_init(m);""",
    "expert_map_init")

# Map before the GPU tier preloads: otherwise the preload copies every resident
# expert through a private buffer for no reason.
rep("        expert_geometry(m);\n        expert_table_init(m);",
    "        expert_geometry(m);\n        expert_table_init(m);\n        expert_map_init(m);",
    "map before preload")

rep("    int cap = (int)((budget * 1e9) / ((double)m->e_slot * (sparse > 0 ? sparse : 1)));\n"
    "    if (g_cap_override > 0) cap = g_cap_override;      /* scelta esplicita: vince */",
    "    int cap = (int)((budget * 1e9) / ((double)m->e_slot * (sparse > 0 ? sparse : 1)));\n"
    "    /* Se ogni esperto arriva dalla mappatura, uno slot non costa memoria\n"
    "     * nostra: tenerne uno per esperto toglie di mezzo lo sfratto. */\n"
    "    if (g_map_all) cap = c->n_experts;\n"
    "    if (g_cap_override > 0) cap = g_cap_override;      /* scelta esplicita: vince */",
    "cap when fully mapped")

rep("""static void expert_read(GModel *m, int layer, int eid, Slot *slot) {
    const ERef *ref = &m->eref[(size_t)layer * m->c.n_experts + eid];
    if (!slot->base) {
        slot->base = malloc((size_t)m->e_slot);
        if (!slot->base) { fprintf(stderr, "OOM su uno slot esperto\\n"); exit(1); }
    }
    if (ref->contig) {
        st_pread_full(ref->fd[0], slot->base, m->e_slot, ref->off[0], "expert");
    } else {
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            st_pread_full(ref->fd[p], slot->base + m->e_at[p], m->e_len[p],
                          ref->off[p], "expert piece");
    }
    slot->eid = eid;""",
    """static void expert_read(GModel *m, int layer, int eid, Slot *slot) {
    const ERef *ref = &m->eref[(size_t)layer * m->c.n_experts + eid];
    if (g_map_active) {
        int ok = 1;
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            if (!piece_mappable(m, ref, p)) { ok = 0; break; }
        if (ok) {
            for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
                slot->piece[p] = g_fmap[ref->fd[p]] + ref->off[p];
            slot->mapped = 1;
            slot->eid = eid;
#ifdef _OPENMP
#pragma omp atomic
#endif
            g_map_serve++;
            return;
        }
    }
    /* Fallback: si torna a scrivere in memoria NOSTRA, non nella mappatura di
     * sola lettura che questo slot poteva star usando prima. */
    slot->mapped = 0;
    if (!slot->own) {
        slot->own = malloc((size_t)m->e_slot);
        if (!slot->own) { fprintf(stderr, "OOM su uno slot esperto\\n"); exit(1); }
    }
    for (int p = 0; p < GLM53_EXPERT_PIECES; p++) slot->piece[p] = slot->own + m->e_at[p];
#ifdef _OPENMP
#pragma omp atomic
#endif
    g_map_copy++;
    if (ref->contig) {
        st_pread_full(ref->fd[0], slot->own, m->e_slot, ref->off[0], "expert");
    } else {
        for (int p = 0; p < GLM53_EXPERT_PIECES; p++)
            st_pread_full(ref->fd[p], slot->own + m->e_at[p], m->e_len[p],
                          ref->off[p], "expert piece");
    }
    slot->eid = eid;""",
    "expert_read via mapping")

rep("""    const Mat shape[3] = {
        { 4, NULL, NULL, slot->base + m->e_at[0], (const float *)(slot->base + m->e_at[1]),
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->base + m->e_at[2], (const float *)(slot->base + m->e_at[3]),
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->base + m->e_at[4], (const float *)(slot->base + m->e_at[5]),
          hidden, inter, 64 },
    };""",
    """    const Mat shape[3] = {
        { 4, NULL, NULL, slot->piece[0], (const float *)slot->piece[1],
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->piece[2], (const float *)slot->piece[3],
          inter, hidden, 64 },
        { 4, NULL, NULL, slot->piece[4], (const float *)slot->piece[5],
          hidden, inter, 64 },
    };""",
    "expert_mats over pieces")

rep("    free(cand); if (tmp.base) free(tmp.base);",
    "    free(cand); if (tmp.own) free(tmp.own);",
    "preload frees own buffer")

# teardown must free only the buffer we allocated, never a mapping
rep("            for (int j = 0; j < cache->cap; j++) free(cache->s[j].base);",
    "            for (int j = 0; j < cache->cap; j++) free(cache->s[j].own);",
    "teardown frees own buffer")

rep('        printf("experts hits %ld miss %ld bytes %llu\\n",',
    '        if (g_map_active)\n'
    '            printf("[MAP] serviti da mmap %ld, copiati %ld\\n", g_map_serve, g_map_copy);\n'
    '        printf("experts hits %ld miss %ld bytes %llu\\n",',
    "map stats print")

open(p, "w").write(s)
print("MMAP EXPERT PIECES APPLIED")
