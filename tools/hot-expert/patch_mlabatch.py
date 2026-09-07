"""Batch MLA's independent projections too.

mla_layer issues 8 mv() per token across 11 layers. Five of them (qa, kva, iwk,
ikpg, iwp) read the same input row; two more (qb, iwq) both read the normalised
q_a. So 8 round trips become 3, on the same coli_vk_matmul_multi the KDA path
already uses.

The reordering is dependency-safe: qn/here/kraw are produced by the first batch
and only normalised afterwards, and qb/iwq consume qn only after its rms.
"""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


# shared helper, placed before mla_layer
rep("/* ---------- MLA + indexer con k-pool ---------- */",
    """/* N matrici indipendenti contro lo stesso vettore d'ingresso, in un submit
 * solo. Ritorna 0 se il batch non e' servibile, e il chiamante ricade sui mv()
 * singoli. */
static int vk_batch_mv(const Mat *const *ws, float *const *os, int n,
                       const float *xin, int I) {
#ifdef COLI_VULKAN
    if (!g_vk_ready || n < 1 || n > VK_MM_MAX) return 0;
    ColiVkMM it[VK_MM_MAX];
    for (int q = 0; q < n; q++) {
        Mat *mw = (Mat *)ws[q];
        if (mw->fmt != 1 && mw->fmt != 4) return 0;
        it[q].tensor = (ColiVkTensor **)&mw->vk;
        it[q].weights = (mw->fmt == 4) ? (const void *)mw->q4 : (const void *)mw->q8;
        it[q].scales = mw->s;
        it[q].fmt = mw->fmt;
        it[q].O = mw->rows;
        it[q].gs = mw->gs;
        it[q].out = os[q];
    }
    return coli_vk_matmul_multi(it, n, xin, I);
#else
    (void)ws; (void)os; (void)n; (void)xin; (void)I;
    return 0;
#endif
}

/* ---------- MLA + indexer con k-pool ---------- */""",
    "vk_batch_mv helper")

OLD = """    for (int t = 0; t < tokens; t++) {
        const int at = base + t;          /* posizione assoluta nella cache */
        const float *row = x + (size_t)t * c->hidden;
        float *qn = qa + (size_t)t * c->q_lora;
        mv(qn, &l->qa, row);
        rms(qn, qn, l->qa_ln, c->q_lora, c->eps);
        mv(queries + (size_t)t * H * QK, &l->qb, qn);
        float *here = latent + (size_t)at * L;
        mv(here, &l->kva, row);
        rms(here, here, l->kva_ln, L, c->eps);
        /* la query entra nello spazio del latente una volta per testa, invece
         * che il latente nello spazio della query una volta per posizione */
        for (int h = 0; h < H; h++)
            mv_rows(absorbed + ((size_t)t * H + h) * L, &l->kvb_kt,
                    queries + ((size_t)t * H + h) * QK, h * L, L);
        /* indexer: le query vengono dal q_a normalizzato, le chiavi dall'hidden
         * con LayerNorm (con bias), e i pesi per testa sono scalati da IH^-0.5 */
        mv(iq + (size_t)t * IH * ID, &l->iwq, qn);
        float *kraw = ik + (size_t)at * ID;
        mv(kraw, &l->iwk, row);
        layer_norm(kraw, kraw, l->ik_nw, l->ik_nb, ID, 1e-5f);
        mv(gates + (size_t)at * ID, &l->ikpg, row);
        mv(head_w + (size_t)t * IH, &l->iwp, row);
        for (int h = 0; h < IH; h++) head_w[(size_t)t * IH + h] /= sqrtf((float)IH);
    }"""

NEW = """    for (int t = 0; t < tokens; t++) {
        const int at = base + t;          /* posizione assoluta nella cache */
        const float *row = x + (size_t)t * c->hidden;
        float *qn = qa + (size_t)t * c->q_lora;
        float *here = latent + (size_t)at * L;
        float *kraw = ik + (size_t)at * ID;
        /* qa/kva/iwk/ikpg/iwp leggono tutte la stessa riga: un submit invece di
         * cinque. Sul percorso denso si paga il round trip, non la matrice. */
        {
            const Mat *bw[5] = { &l->qa, &l->kva, &l->iwk, &l->ikpg, &l->iwp };
            float *bo[5] = { qn, here, kraw,
                             gates + (size_t)at * ID, head_w + (size_t)t * IH };
            if (!vk_batch_mv(bw, bo, 5, row, c->hidden)) {
                mv(qn, &l->qa, row);
                mv(here, &l->kva, row);
                mv(kraw, &l->iwk, row);
                mv(gates + (size_t)at * ID, &l->ikpg, row);
                mv(head_w + (size_t)t * IH, &l->iwp, row);
            }
        }
        rms(qn, qn, l->qa_ln, c->q_lora, c->eps);
        rms(here, here, l->kva_ln, L, c->eps);
        layer_norm(kraw, kraw, l->ik_nw, l->ik_nb, ID, 1e-5f);
        /* qb e iwq leggono entrambe il q_a normalizzato: altro submit unico. */
        {
            const Mat *bw[2] = { &l->qb, &l->iwq };
            float *bo[2] = { queries + (size_t)t * H * QK, iq + (size_t)t * IH * ID };
            if (!vk_batch_mv(bw, bo, 2, qn, c->q_lora)) {
                mv(queries + (size_t)t * H * QK, &l->qb, qn);
                mv(iq + (size_t)t * IH * ID, &l->iwq, qn);
            }
        }
        /* la query entra nello spazio del latente una volta per testa, invece
         * che il latente nello spazio della query una volta per posizione */
        for (int h = 0; h < H; h++)
            mv_rows(absorbed + ((size_t)t * H + h) * L, &l->kvb_kt,
                    queries + ((size_t)t * H + h) * QK, h * L, L);
        for (int h = 0; h < IH; h++) head_w[(size_t)t * IH + h] /= sqrtf((float)IH);
    }"""

rep(OLD, NEW, "batch mla projections")

open(p, "w").write(s)
print("MLA BATCH APPLIED")
