"""Stop the expert slot cache from evicting the page cache that holds the model.

MemAvailable counts reclaimable page cache as free. Sizing the LRU expert-slot
cache from it makes the engine allocate ~MemTotal of anonymous memory, which
evicts the very model pages the slot cache then has to re-read from NVMe. Both
caches hold the SAME bytes. Measured on rome (247 GB RAM, 182 GB model,
GLM-5.3-Flash int4-g64, 2-GPU, --ngen 60):

    GLM53_EXPERT_GB   tok/s   miss   avg read
    default (~238)    1.387   9936   8.45 ms   <- page cache thrashes, reads hit NVMe
    24                2.013  15978   3.52 ms
    40                2.061  13625   4.46 ms   <- peak
    44                2.060  13167   4.79 ms
    56                1.971  11931   5.69 ms

Twice the misses but 48% faster, because each miss is served from page cache
instead of the drive. So: leave the model room in page cache, take the rest.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# 1. a MemTotal reader beside the MemAvailable one
rep("""static double memory_available_gb(void) {""",
    """static double memory_total_gb(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    char line[256];
    double gb = 0.0;
    while (fgets(line, sizeof(line), f)) {
        long kb;
        if (sscanf(line, "MemTotal: %ld kB", &kb) == 1) { gb = kb / 1048576.0; break; }
    }
    fclose(f);
    return gb;
}

static double memory_available_gb(void) {""",
    "memory_total_gb")

# 2. size the slot cache around the model instead of around MemAvailable
rep("""    double budget;
    if (setting) budget = atof(setting);
    else {
        const double free_now = memory_available_gb();
        budget = free_now - 3.0;
        if (budget < 1.0) budget = 1.0;
        if (getenv("GLM53_VERBOSE"))
            fprintf(stderr, "budget esperti: %.1f GB (%.1f disponibili, 3 di margine)\\n",
                    budget, free_now);
    }
    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    int sparse = m->layer_end - from;
    if (sparse < 0) sparse = 0;""",
    """    const int from = c->first_dense > m->layer_begin ? c->first_dense : m->layer_begin;
    int sparse = m->layer_end - from;
    if (sparse < 0) sparse = 0;
    double budget;
    if (setting) budget = atof(setting);
    else {
        /* MemAvailable conta la page cache come "libera". Fidarsi di quel
         * numero significa allocare in anonimo tutta la RAM e sfrattare
         * proprio le pagine del modello che poi si rileggono dal disco: le
         * due cache tengono gli stessi byte. Si lascia quindi al modello lo
         * spazio per restare in page cache e si prende solo l'avanzo. */
        const double free_now = memory_available_gb();
        const double total = memory_total_gb();
        /* gli esperti instradati dominano; il resto dei pesi e' un ~7% */
        const double model_gb = (double)sparse * c->n_experts * (double)m->e_slot / 1e9 * 1.07;
        double margin = total * 0.08;
        if (margin < 4.0) margin = 4.0;
        if (total <= 0.0 || model_gb >= total - margin) {
            /* il modello non ci sta comunque: la page cache non puo' aiutare
             * e tanto vale tenere piu' slot possibile, come prima. */
            budget = free_now - 3.0;
        } else {
            budget = total - model_gb - margin;
            if (budget > free_now - 3.0) budget = free_now - 3.0;
        }
        if (budget < 1.0) budget = 1.0;
        if (getenv("GLM53_VERBOSE"))
            fprintf(stderr, "budget esperti: %.1f GB (%.1f totali, %.1f modello, "
                            "%.1f margine, %.1f disponibili)\\n",
                    budget, total, model_gb, margin, free_now);
    }""",
    "page-cache-aware budget")

open(p, "w").write(s)
print("CACHE BUDGET PATCH APPLIED")
