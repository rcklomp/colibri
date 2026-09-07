"""Make the dense weights resident before the expert tier fills.

vk_preload_tier fills device 0 with heat-ranked experts until free VRAM drops
below a fixed 3 GB reserve. But the attention and shared-expert matrices upload
lazily on their first mv(), which happens AFTER the preload - so the tier takes
the VRAM first and the dense weights bounce off a full device for the rest of
the run.

Those matrices are needed by EVERY token; a resident routed expert is needed by
maybe one token in forty. Trading tier slots for them is strongly positive:

  dev0 experts   tok/s
  1600 (auto)    2.654
  1500           3.042
  1400           3.182   <- peak
  1300           3.175
  1000           3.106
  400            3.014

Rather than tune the reserve per model (it depends on hidden size, head count
and layer mix), upload the dense set first and let the tier have what is
actually left.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("static void vk_preload_tier(GModel *m) {",
    """/* Il denso prima del tier. Attenzione e shared expert servono a OGNI token,
 * un esperto instradato a circa un token su quaranta: lasciare che il tier si
 * prenda la VRAM per primo manda le matrici dense a rimbalzare sulla CPU e
 * costa molto piu' di quanto rendano gli esperti residenti in piu'. */
static void vk_warm_dense(GModel *m) {
    if (!m->layer) return;
    long warmed = 0, skipped = 0;
    double bytes = 0.0;
    for (int i = m->layer_begin; i < m->layer_end; i++) {
        GLayer *l = &m->layer[i];
        Mat *set[] = {
            &l->kq, &l->kk, &l->kv, &l->ko, &l->kga, &l->kgb, &l->kfa, &l->kfb, &l->kb,
            &l->qa, &l->qb, &l->kva, &l->kvb_kt, &l->kvb_v, &l->o,
            &l->iwq, &l->iwk, &l->iwp, &l->ikpg,
            &l->dg, &l->du, &l->dd, &l->rg, &l->ru, &l->rd,
        };
        for (size_t q = 0; q < sizeof(set) / sizeof(set[0]); q++) {
            Mat *w = set[q];
            if (w->fmt != 1 && w->fmt != 4) continue;
            const void *src = (w->fmt == 4) ? (const void *)w->q4 : (const void *)w->q8;
            if (!src || !w->s || w->rows < 1 || w->columns < 1) continue;
            if (coli_vk_tensor_ensure((ColiVkTensor **)&w->vk, src, w->s, w->fmt,
                                      w->columns, w->rows, w->gs)) {
                warmed++;
                bytes += (double)w->rows * (double)w->columns * 0.5;
            } else {
                skipped++;
            }
        }
    }
    fprintf(stderr, "[VK] denso residente: %ld matrici (~%.2f GB)%s\\n",
            warmed, bytes / 1e9, skipped ? " (alcune non entrate)" : "");
}

static void vk_preload_tier(GModel *m) {""",
    "vk_warm_dense")

rep("    Slot tmp; memset(&tmp, 0, sizeof(tmp)); tmp.eid = -1;\n    int loaded = 0;",
    "    if (!getenv(\"COLI_VK_NO_WARM_DENSE\")) vk_warm_dense(m);\n"
    "    Slot tmp; memset(&tmp, 0, sizeof(tmp)); tmp.eid = -1;\n    int loaded = 0;",
    "warm dense before tier fill")

open(p, "w").write(s)
print("WARM-DENSE APPLIED")
