/* F3 step 1: the Vulkan expert tier ported from V1's ONE fixed device
 * (dev3) to a small per-device array (home(eid), per-device budget/used/cap,
 * planned/resident/refused counted per device), the same shape
 * qwen36_tier.c's CUDA tier already has and already tests
 * (tests/test_qwen36_tier_multidev.c) -- and its acceptance of a row-wise
 * int8 container (fmt=1, expert_gs absent), which V1 refused outright.
 *
 * Uses tests/qwen36_fake_vk.h (mirrors tests/qwen36_fake_cuda.h): every
 * coli_vk_* symbol qwen36_tier_vk.c links against is a recording stub, so
 * this runs with no GPU, no Vulkan loader and no SPIR-V shaders.
 *
 * Five scenarios, each its own qt_init/qt_shutdown cycle in one process
 * (ASan leak detection is off for this whole suite by policy -- c/Makefile,
 * "detect_leaks is OFF, deliberately" -- so the tier's own no-free-at-exit
 * convention, unchanged from V1/CUDA, does not need working around here):
 *
 *  A. Only COLI_VK_DEV3 set (COLI_VK_DEV2 unset): the DEFAULT must still be
 *     exactly one active device -- V1's own configuration, unchanged.
 *  B. Both devices active: home(eid)=eid%2 routes evens to dev2 and odds to
 *     dev3 (ported from test_qwen36_tier_multidev.c's odd/even routing
 *     check); qt_issue/qt_take batch per device and the input-replica
 *     blocks stay inside G.is_x and never overlap.
 *  C. Per-device budget/cap independence: COLI_VK_EXPERTS2 caps dev2 well
 *     below its share while dev3 is left uncapped -- dev2 must stop on its
 *     own cap and dev3 must keep filling past it (this is the "fill by
 *     budget, not blind modulo" requirement: a full device does not stall
 *     the whole plan).
 *  D. Row-wise int8 (fmt=1, expert_gs=0): accepted rather than refused,
 *     bytes-per-expert doubles (no int4 packing), scale counts are one per
 *     row, and the staged bytes are copied VERBATIM (no XOR -- that
 *     conversion is int4-nibble-only; ported from the CUDA tier's own
 *     wfmt==1 stage() branch, qwen36_tier.c).
 *  E. dev2 named but not available: graceful fallback to the single
 *     mandatory device (dev3), not a hard failure. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen36_fake_vk.h"

#include "../qwen36_tier_vk.c"

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  FAIL: %s\n", what); fails++; }
}

enum { MAX_ISSUE_REC = 8 };
static struct { int suffix, count; const float *x; } issue_rec[MAX_ISSUE_REC];
static int n_issue_rec;
static int record_issue(int suffix, int count, const float *x) {
    if (n_issue_rec < MAX_ISSUE_REC) {
        issue_rec[n_issue_rec].suffix = suffix;
        issue_rec[n_issue_rec].count = count;
        issue_rec[n_issue_rec].x = x;
        n_issue_rec++;
    }
    return 1;
}

static void fake_vk_reset(void) {
    fake_vk_uploads2 = fake_vk_uploads3 = 0;
    fake_vk_last_fmt2 = fake_vk_last_fmt3 = -1;
    fake_vk_last_bytes2 = fake_vk_last_bytes3 = 0;
    fake_vk_captured2_len = fake_vk_captured3_len = 0;
    fake_vk_dev2_avail = fake_vk_dev3_avail = 1;
    fake_vk_budget2_gb = fake_vk_budget3_gb = 4.0;
    fake_vk_used2_gb = fake_vk_used3_gb = 0.0;
    fake_vk_issue_hook = NULL;
}

/* Fills one synthetic expert's RAM-side bytes the way qwen36.c's slot would:
 * packed int4 (wfmt=4, mb=D*Ih/2 bytes/matrix) or plain int8 (wfmt=1,
 * mb=D*Ih). Distinct byte values per expert/matrix so a captured upload can
 * be told apart from another expert's. */
static void fill_expert(int wfmt, int D, int Ih, int eid,
                        unsigned char *g4, unsigned char *u4, unsigned char *d4,
                        float *sc, size_t sc_gu, size_t sc_d) {
    size_t mb = (wfmt == 4) ? (size_t)D * Ih / 2 : (size_t)D * Ih;
    memset(g4, (unsigned char)(eid * 3 + 1), mb);
    memset(u4, (unsigned char)(eid * 3 + 2), mb);
    memset(d4, (unsigned char)(eid * 3 + 3), mb);
    for (size_t i = 0; i < 2 * sc_gu + sc_d; i++) sc[i] = 1.0f;
}

/* Plans and uploads every expert of an NL=1 model via the same two calls
 * qwen36.c's warmstart makes (qt_plan_fill then qt_note_planned per planned
 * expert, qt_fill_wait at the end -- a no-op here, uploads are synchronous). */
static int warmstart_all(int ne, int wfmt, int D, int Ih, size_t sc_gu, size_t sc_d) {
    int *layers = malloc((size_t)ne * sizeof(int));
    int *eids = malloc((size_t)ne * sizeof(int));
    int n = qt_plan_fill(layers, eids, ne);
    size_t mb = (wfmt == 4) ? (size_t)D * Ih / 2 : (size_t)D * Ih;
    unsigned char *g4 = malloc(mb), *u4 = malloc(mb), *d4 = malloc(mb);
    float *sc = malloc((2 * sc_gu + sc_d) * sizeof(float));
    for (int i = 0; i < n; i++) {
        fill_expert(wfmt, D, Ih, eids[i], g4, u4, d4, sc, sc_gu, sc_d);
        qt_note_planned(layers[i], eids[i], g4, u4, d4, sc, sc + sc_gu, sc + 2 * sc_gu);
    }
    qt_fill_wait();
    free(g4); free(u4); free(d4); free(sc); free(layers); free(eids);
    return n;
}

int main(void) {
    enum { D = 64, IH = 32, TOPK = 32 };
    const size_t SC_GU = (size_t)IH;               /* gs=8 groups of 8 over D=64 -> below */
    const size_t SC_GU_GS8 = (size_t)IH * ((D + 7) / 8);
    const size_t SC_D_GS8 = (size_t)D * ((IH + 7) / 8);

    /* ---- A: default, only COLI_VK_DEV3 set -- must stay ONE device ---- */
    {
        enum { NL = 1, NE = 8 };
        fake_vk_reset();
        setenv("Q36_VULKAN", "1", 1);
        unsetenv("COLI_VK_DEV2");
        setenv("COLI_VK_DEV3", "auto", 1);
        unsetenv("COLI_VK_EXPERTS2");
        unsetenv("COLI_VK_EXPERTS3");
        unsetenv("HEAT_FILE");

        check(qt_init(NL, NE, D, IH, NE, TOPK, 8 /* gs */, 1 /* int4 */) == 1,
              "A: tier did not come up with only COLI_VK_DEV3 set");
        check(G.ndev == 1, "A: default must activate exactly one device");
        check(G.ndev >= 1 && G.phys[0] == 3, "A: the one device must be dev3");

        int n = warmstart_all(NE, 4, D, IH, SC_GU_GS8, SC_D_GS8);
        check(n == NE, "A: every expert should have been planned");
        for (int e = 0; e < NE; e++)
            check(qt_is_resident(0, e) && qs(0, e)->dev == 0,
                  "A: expert did not land resident on the sole device");
        check(fake_vk_uploads3 == 3 * NE && fake_vk_uploads2 == 0,
              "A: all uploads (3 tensors/expert: gate,up,down) must go to dev3, none to dev2");
        /* int4 nibble XOR: byte 1 of the first upload (an all-(eid*3+1)
         * gate matrix) must equal that byte XOR 0x88, not the raw byte. */
        check(fake_vk_captured3_len > 0 &&
              fake_vk_captured3[0] == (unsigned char)((0 * 3 + 1) ^ 0x88),
              "A: int4 upload must carry the offset-binary XOR, byte-exact");
        qt_stats();
        qt_shutdown();
    }

    /* ---- B: both devices, home(eid)=eid%2, issue/take geometry ---- */
    {
        enum { NL = 1, NE = 64 };
        fake_vk_reset();
        setenv("Q36_VULKAN", "1", 1);
        setenv("COLI_VK_DEV2", "auto", 1);
        setenv("COLI_VK_DEV3", "auto", 1);
        unsetenv("COLI_VK_EXPERTS2");
        unsetenv("COLI_VK_EXPERTS3");

        check(qt_init(NL, NE, D, IH, NE, TOPK, 8, 1) == 1,
              "B: tier did not come up with both devices set");
        check(G.ndev == 2, "B: both devices must be active");
        check(G.phys[0] == 2 && G.phys[1] == 3,
              "B: activation order must be dev2 then dev3");

        int n = warmstart_all(NE, 4, D, IH, SC_GU_GS8, SC_D_GS8);
        check(n == NE, "B: every expert should have been planned");
        for (int e = 0; e < NE; e++) {
            int want_idx = e % 2;   /* home(e) = e % G.ndev */
            check(qs(0, e)->dev == want_idx,
                  "B: home(eid) must route even/odd experts to dev2/dev3");
        }
        check(fake_vk_uploads2 == 3 * (NE / 2) && fake_vk_uploads3 == 3 * (NE / 2),
              "B: the 64 experts must split 32/32 across the two devices "
              "(3 tensor uploads/expert)");

        fake_vk_issue_hook = record_issue;
        float xin[D]; for (int i = 0; i < D; i++) xin[i] = (float)i;

        /* 32 odd-numbered experts -> home=1 -> all on dev3, one group call */
        {
            int eids[32]; for (int k = 0; k < 32; k++) eids[k] = 2 * k + 1;
            n_issue_rec = 0;
            uint32_t mask = qt_issue(0, eids, 32, xin);
            check(mask == 0xFFFFFFFFu, "B: issuing 32 resident odd experts must set all 32 bits");
            check(n_issue_rec == 1 && issue_rec[0].suffix == 3 && issue_rec[0].count == 32,
                  "B: 32 odd experts should issue as one 32-row group on dev3");
        }
        /* mixed batch: 16 even (dev2) + 16 odd (dev3) -> two group calls,
         * disjoint, inside the replica buffer */
        {
            int eids2[32];
            for (int k = 0; k < 16; k++) { eids2[k] = 2 * k; eids2[16 + k] = 2 * k + 1; }
            n_issue_rec = 0;
            uint32_t mask2 = qt_issue(0, eids2, 32, xin);
            check(mask2 == 0xFFFFFFFFu, "B: mixed even/odd batch must set all 32 bits");
            check(n_issue_rec == 2, "B: the mixed batch should issue once per device");
            const float *lo = G.is_x, *hi = G.is_x + (size_t)G.ndev * 32 * G.D;
            int inside = 1, disjoint = 1;
            for (int i = 0; i < n_issue_rec; i++)
                if (!(issue_rec[i].x >= lo && issue_rec[i].x + (size_t)issue_rec[i].count * G.D <= hi)) inside = 0;
            if (n_issue_rec == 2) {
                const float *a0 = issue_rec[0].x, *a1 = issue_rec[1].x;
                size_t c0 = (size_t)issue_rec[0].count, c1 = (size_t)issue_rec[1].count;
                disjoint = (a0 + c0 * G.D <= a1) || (a1 + c1 * G.D <= a0);
            }
            check(inside, "B: device blocks must stay inside the replica buffer");
            check(disjoint, "B: device blocks must not overlap");

            float val[32]; for (int k = 0; k < 32; k++) val[k] = 1.0f;
            float out[D]; memset(out, 0, sizeof out);
            qt_take(mask2, val, 32, out);   /* fake take() succeeds trivially */
        }
        qt_stats();
        qt_shutdown();
    }

    /* ---- C: per-device budget/cap independence ---- */
    {
        enum { NL = 1, NE = 64 };
        fake_vk_reset();
        setenv("Q36_VULKAN", "1", 1);
        setenv("COLI_VK_DEV2", "auto", 1);
        setenv("COLI_VK_DEV3", "auto", 1);
        setenv("COLI_VK_EXPERTS2", "5", 1);     /* dev2 gets 5 of its 32-expert share */
        unsetenv("COLI_VK_EXPERTS3");           /* dev3 uncapped: must keep filling */

        check(qt_init(NL, NE, D, IH, NE, TOPK, 8, 1) == 1, "C: tier did not come up");
        int n = warmstart_all(NE, 4, D, IH, SC_GU_GS8, SC_D_GS8);
        check(G.dev_planned[0] == 5, "C: dev2's count cap must stop it at exactly 5");
        check(G.dev_planned[1] == NE / 2, "C: dev3 must fill its whole share uncapped");
        check(n == 5 + NE / 2,
              "C: a full device must not stall the plan -- the other keeps filling");
        check(fake_vk_uploads2 == 3 * 5 && fake_vk_uploads3 == 3 * (NE / 2),
              "C: uploads per device must match the capped/uncapped plan "
              "(3 tensor uploads/expert)");
        qt_stats();
        qt_shutdown();
    }

    /* ---- D: row-wise int8 (fmt=1, gs absent) accepted, no XOR ---- */
    {
        enum { NL = 1, NE = 4 };
        fake_vk_reset();
        setenv("Q36_VULKAN", "1", 1);
        setenv("COLI_VK_DEV2", "auto", 1);
        setenv("COLI_VK_DEV3", "auto", 1);
        unsetenv("COLI_VK_EXPERTS2");
        unsetenv("COLI_VK_EXPERTS3");

        check(qt_init(NL, NE, D, IH, NE, 8, 0 /* gs absent */, 0 /* int8 */) == 1,
              "D: tier refused a row-wise int8 container");
        check(G.wfmt == 1, "D: wfmt must be 1 (int8) for expert_is_int4=0");
        check(G.sc_gu == (size_t)IH && G.sc_d == (size_t)D,
              "D: row-wise scales must be one per row (sc_gu=Ih, sc_d=D)");
        size_t want_bytes = 3 * (size_t)D * IH + (2 * (size_t)IH + (size_t)D) * sizeof(float);
        check(G.exp_bytes == want_bytes,
              "D: exp_bytes must use whole-byte (unpacked) matrices for int8");

        int n = warmstart_all(NE, 1, D, IH, (size_t)IH, (size_t)D);
        check(n == NE, "D: every int8 expert should have been planned");
        check(fake_vk_last_fmt2 == 1 || fake_vk_last_fmt3 == 1,
              "D: the upload must carry fmt=1, not the int4 fmt=4");
        /* No XOR on int8: the captured first byte of whichever device took
         * expert 0's gate matrix must equal the RAW byte (eid*3+1), not its
         * XOR 0x88 -- unlike scenario A's int4 check above. */
        /* Expert 0 is home()=0 -> dev2 (G.phys[0]==2), so dev2's first-upload
         * capture is expert 0's raw gate byte, verbatim -- not XOR 0x88 (that
         * conversion is int4-nibble-only, see stage()'s wfmt==1 branch). */
        int raw0 = (unsigned char)(0 * 3 + 1);
        check(fake_vk_captured2_len > 0 && fake_vk_captured2[0] == (unsigned char)raw0,
              "D: int8 staging must copy bytes verbatim, no int4-style XOR");
        qt_stats();
        qt_shutdown();
    }

    /* ---- E: dev2 named but unavailable -- graceful single-device fallback ---- */
    {
        enum { NL = 1, NE = 4 };
        fake_vk_reset();
        fake_vk_dev2_avail = 0;
        setenv("Q36_VULKAN", "1", 1);
        setenv("COLI_VK_DEV2", "auto", 1);
        setenv("COLI_VK_DEV3", "auto", 1);

        check(qt_init(NL, NE, D, IH, NE, 8, 8, 1) == 1,
              "E: tier must still come up on dev3 alone when dev2 is unavailable");
        check(G.ndev == 1 && G.phys[0] == 3,
              "E: an unavailable dev2 must fall back to dev3 only, not fail outright");
        qt_shutdown();
    }

    (void)SC_GU;
    if (fails) { printf("test_qwen36_tier_vk_multidev: %d fallimenti\n", fails); return 1; }
    printf("test_qwen36_tier_vk_multidev: ok\n");
    return 0;
}
