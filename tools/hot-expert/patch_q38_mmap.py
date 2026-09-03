"""Serve Qwen3.8 native-FP8 experts from a file mapping instead of copying them.

q38_load_native_fp8_ranges reserves a 3-matrix slab per slot and preads the
expert into it. With cache=512/layer (every expert) that is ~61 GB of private
copies of bytes the kernel is already holding in page cache -- measured on this
box: RSS 61 GB alongside 179 GB of page cache, on a 259 GB machine.

Q38Weight already distinguishes owned from borrowed storage (owns_data /
owns_scales, and q38_bind_borrowed_fp8 never claims ownership), so a slot can
point straight at mapped file bytes. FP8 is one byte per element and the scales
come from the separate per-layer scale bank, so there is no alignment or
dtype constraint on the mapped ranges at all.

Uses compat_map_readonly (st.h's mapping primitive, already used by kimi_k3.c)
rather than raw mmap, so this works on Windows too. Mapping is per SHARD, not
per tensor: 24576 experts x 3 matrices would otherwise mean 73728 mappings.

Q38_NO_MMAP=1 restores the copy path.
"""
p = "/home/ronald/src/colibri/c/qwen38_core.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# ---- shard mapping table, placed just before the slot binder ----
rep("static void q38_bind_borrowed_fp8(Q38Weight *weight,void *data,float *scales,",
    """/* Una mappatura per SHARD, non per tensore: 24576 esperti x 3 matrici
 * sarebbero 73728 mappature. compat_map_readonly gestisce l'allineamento e
 * funziona anche su Windows; le pagine restano file-backed e reclaimable, cioe'
 * non sono il working set anonimo che la cache degli slot creava copiandole. */
#define Q38_MAXFD 4096
static compat_ro_map q38_shard_map[Q38_MAXFD];
static const unsigned char *q38_shard_base[Q38_MAXFD];
static int64_t q38_shard_len[Q38_MAXFD];
static signed char q38_shard_tried[Q38_MAXFD];
static long q38_map_serve, q38_map_copy;

static int q38_mmap_enabled(void) {
    static int on = -1;
    if (on < 0) on = !(getenv("Q38_NO_MMAP") && atoi(getenv("Q38_NO_MMAP")));
    return on;
}

/* Ritorna la base mappata dello shard, o NULL. Un fallimento si ricorda: non
 * si ritenta una mmap per ogni singolo esperto. */
static const unsigned char *q38_shard_mapped(int fd) {
    if (fd < 0 || fd >= Q38_MAXFD || !q38_mmap_enabled()) return NULL;
    if (q38_shard_base[fd]) return q38_shard_base[fd];
    if (q38_shard_tried[fd]) return NULL;
    q38_shard_tried[fd] = 1;
    int64_t len = (int64_t)lseek(fd, 0, SEEK_END);
    if (len <= 0) return NULL;
    const void *data = NULL;
    if (compat_map_readonly(fd, 0, (size_t)len, &q38_shard_map[fd], &data) != 0) return NULL;
    q38_shard_base[fd] = (const unsigned char *)data;
    q38_shard_len[fd] = len;
    return q38_shard_base[fd];
}

/* Un intervallo e' servibile dalla mappatura se lo shard e' mappato e
 * l'intervallo ci sta dentro. FP8 e' un byte per elemento: nessun vincolo di
 * allineamento, e le scale arrivano dal banco per-layer, non dal file. */
static const unsigned char *q38_mapped_range(int fd, int64_t off, int64_t nbytes) {
    if (off < 0 || nbytes <= 0) return NULL;
    const unsigned char *base = q38_shard_mapped(fd);
    if (!base) return NULL;
    if (off > q38_shard_len[fd] - nbytes) return NULL;
    return base + off;
}

static void q38_unmap_shards(void) {
    for (int fd = 0; fd < Q38_MAXFD; fd++)
        if (q38_shard_base[fd]) {
            compat_unmap_readonly(&q38_shard_map[fd]);
            q38_shard_base[fd] = NULL;
            q38_shard_len[fd] = 0;
        }
}

static void q38_bind_borrowed_fp8(Q38Weight *weight,void *data,float *scales,""",
    "shard mapping table")

# ---- serve the three matrices from the mapping when possible ----
rep("""static void q38_load_native_fp8_ranges(Model *m,int layer,int expert,Slot *slot,
                                       st_tensor *weight[3]) {
    Cfg *c=&m->c;
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    float *scales=cache->values+(int64_t)expert*3*cache->scale_count;
    q38_bind_fp8_slot(slot,scales,(int)cache->scale_count,c->hidden,c->inter);""",
    """static void q38_load_native_fp8_ranges(Model *m,int layer,int expert,Slot *slot,
                                       st_tensor *weight[3]) {
    Cfg *c=&m->c;
    Q38ExpertScaleCache *cache=&m->expert_scales[layer];
    float *scales=cache->values+(int64_t)expert*3*cache->scale_count;
    /* Se i tre intervalli sono mappati, lo slot li PUNTA invece di copiarli:
     * niente slab, niente 14 MB per miss, e la residenza la gestisce il kernel. */
    {
        const unsigned char *pg=q38_mapped_range(weight[0]->fd,weight[0]->off,weight[0]->nbytes);
        const unsigned char *pu=q38_mapped_range(weight[1]->fd,weight[1]->off,weight[1]->nbytes);
        const unsigned char *pd=q38_mapped_range(weight[2]->fd,weight[2]->off,weight[2]->nbytes);
        if(pg&&pu&&pd){
            int sc=(int)cache->scale_count;
            q38_bind_borrowed_fp8(&slot->gate,(void*)pg,scales,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->up,(void*)pu,scales+sc,c->inter,c->hidden);
            q38_bind_borrowed_fp8(&slot->down,(void*)pd,scales+2*sc,c->hidden,c->inter);
            /* lo slab di questo slot non serve piu': e' esattamente la memoria
             * che stava duplicando la page cache. */
            if(slot->fp8_slab){free(slot->fp8_slab);slot->fp8_slab=NULL;slot->fp8_slab_bytes=0;}
            q38_map_serve++;
            return;
        }
    }
    q38_map_copy++;
    q38_bind_fp8_slot(slot,scales,(int)cache->scale_count,c->hidden,c->inter);""",
    "bind mapped ranges")

open(p, "w").write(s)
print("Q38 MMAP APPLIED")

# ---- report + teardown in qwen38.c ----
q = "/home/ronald/src/colibri/c/qwen38.c"
t = open(q).read()
OLD = 'rt_init("qwen38", m->c.layers, m->c.experts);'
assert t.count(OLD) == 1, "rt_init anchor"
t = t.replace(OLD, OLD + '\n    if (getenv("GLM53_VERBOSE") || getenv("Q38_VERBOSE"))\n'
              '        fprintf(stderr, "[MAP] expert mapping %s\\n",\n'
              '                q38_mmap_enabled() ? "on (Q38_NO_MMAP=1 to disable)" : "off");')
open(q, "w").write(t)
print("ok: qwen38.c map notice")
