// tools/hot-expert/m1/m3_emul_check.cpp -- a HOST-ONLY emulation of the three
// depth kernels in m3_attn.hip (indexer scan, radix-select top-k, split-K
// flash attention), checked against the same CPU references the harness uses.
//
// Why it exists: the agent that wrote those kernels is not allowed to run
// anything on the GPUs (CLAUDE.md, "One benchmark at a time"; the M1/M3
// briefs' build-only discipline), so the only correctness evidence it can
// produce before the orchestrator runs the real binary is this: the same
// index arithmetic, the same LDS arrays and the same barrier structure
// (read phase / write phase, so a Hillis-Steele scan is exercised exactly as
// the GPU would exercise it), executed serially on the host. It caught
// nothing on the first run, which is a weak result on its own -- its value is
// that it covers the cases the harness's synthetic data never produces: all
// scores equal, all negative, mass ties at the cut, a 40-decade dynamic
// range, zeros and denormals, n_sel not a multiple of the attention chunk.
//
// It is NOT a measurement and NOT a substitute for the harness's oracle: it
// proves the algorithm and the indexing, not the generated code. A change to
// a kernel in m3_attn.hip must be mirrored here by hand or this file is worse
// than useless -- if the two ever disagree, m3_attn.hip is the truth.
//
// Build and run anywhere (no HIP, no GPU):
//   c++ -O2 -std=c++17 m3_emul_check.cpp -o m3_emul_check && ./m3_emul_check
//   (or `make m3_emul_check` in this directory, which does it in the same
//    docker image)
// Exit 0 = all checks passed, 1 = at least one failure, with the line named.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------- shared bits copied from m3_attn.hip ----------------
static uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    uint32_t lsb = (c.u >> 16) & 1u;
    c.u += 0x7fffu + lsb;
    return (uint16_t)(c.u >> 16);
}
static float bf16_to_f32(uint16_t b) { union { float f; uint32_t u; } c; c.u = ((uint32_t)b) << 16; return c.f; }
static float bits_to_f32(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }
static float bf16_lo(uint32_t w) { return bits_to_f32(w << 16); }
static float bf16_hi(uint32_t w) { return bits_to_f32(w & 0xffff0000u); }
static uint32_t ord_key(float f) {
    union { float f; uint32_t u; } c; c.f = f;
    return (c.u & 0x80000000u) ? ~c.u : (c.u | 0x80000000u);
}

#define IDX_N_HEADS 4
#define IDX_DIM     128
#define IDX_ELEMS   (IDX_N_HEADS * IDX_DIM)
#define IDX_U4_PER_BLK (IDX_ELEMS * 2 / 16)
#define IDX_LANES   32
#define IDX_U4_PER_LANE (IDX_U4_PER_BLK / IDX_LANES)

#define QSA_RATIO 4
#define QSA_TOP_K_TOKENS 2048
#define N_SEL_BLOCKS (QSA_TOP_K_TOKENS / QSA_RATIO)

#define TOPK_THREADS 1024
#define TOPK_HI_BITS 11
#define TOPK_HI_BINS (1 << TOPK_HI_BITS)
#define TOPK_HI_SHIFT (32 - TOPK_HI_BITS)
#define TOPK_LO_BITS 7
#define TOPK_LO_BINS (1 << TOPK_LO_BITS)
#define TOPK_LO_PASSES 3

#define N_Q_HEADS 24
#define HEAD_DIM  256
#define N_KV_HEADS 2
#define GQA_GROUP (N_Q_HEADS / N_KV_HEADS)
#define ATTN_CHUNK 64
#define ATTN_LANES 32
#define ATTN_DPL (HEAD_DIM / ATTN_LANES)
#define ATTN_WAVES GQA_GROUP
#define ATTN_THREADS (ATTN_WAVES * ATTN_LANES)
#define ATTN_UNR 4
#define ATTN_NEG_INF (-3.0e38f)

// ---------------- 1. idx_scan ----------------
static void idx_scan_cpu_ref(const std::vector<uint16_t>& kpool, const std::vector<float>& q,
                              std::vector<float>& scores, int n_blocks) {
    scores.assign(n_blocks, 0.f);
    for (int b = 0; b < n_blocks; b++) {
        double acc = 0.0;
        for (int h = 0; h < IDX_N_HEADS; h++)
            for (int d = 0; d < IDX_DIM; d++)
                acc += (double)q[h * IDX_DIM + d] * bf16_to_f32(kpool[((size_t)b * IDX_N_HEADS + h) * IDX_DIM + d]);
        scores[b] = (float)acc;
    }
}
static void idx_scan_emul(const std::vector<uint16_t>& kpool, const std::vector<float>& q,
                           std::vector<float>& scores, int n_blocks) {
    scores.assign(n_blocks, 0.f);
    const uint32_t* k32 = reinterpret_cast<const uint32_t*>(kpool.data());
    for (int b = 0; b < n_blocks; b++) {
        float lane_acc[IDX_LANES];
        for (int lane = 0; lane < IDX_LANES; lane++) {
            float qr[IDX_U4_PER_LANE][8];
            for (int r = 0; r < IDX_U4_PER_LANE; r++) {
                int e4 = (lane + r * IDX_LANES) * 2;      // float4 index
                for (int i = 0; i < 8; i++) qr[r][i] = q[e4 * 4 + i];
            }
            float acc = 0.f;
            for (int r = 0; r < IDX_U4_PER_LANE; r++) {
                // uint4 at index (lane + r*32) of this block = 4 uint32 words
                size_t w0 = (size_t)b * IDX_U4_PER_BLK * 4 + (size_t)(lane + r * IDX_LANES) * 4;
                uint32_t v[4] = { k32[w0], k32[w0 + 1], k32[w0 + 2], k32[w0 + 3] };
                acc = fmaf(qr[r][0], bf16_lo(v[0]), acc);
                acc = fmaf(qr[r][1], bf16_hi(v[0]), acc);
                acc = fmaf(qr[r][2], bf16_lo(v[1]), acc);
                acc = fmaf(qr[r][3], bf16_hi(v[1]), acc);
                acc = fmaf(qr[r][4], bf16_lo(v[2]), acc);
                acc = fmaf(qr[r][5], bf16_hi(v[2]), acc);
                acc = fmaf(qr[r][6], bf16_lo(v[3]), acc);
                acc = fmaf(qr[r][7], bf16_hi(v[3]), acc);
            }
            lane_acc[lane] = acc;
        }
        for (int off = IDX_LANES / 2; off > 0; off >>= 1) {
            float tmp[IDX_LANES];
            for (int l = 0; l < IDX_LANES; l++) tmp[l] = lane_acc[l] + lane_acc[l ^ off];
            memcpy(lane_acc, tmp, sizeof(tmp));
        }
        scores[b] = lane_acc[0];
    }
}

// ---------------- 2. radix select ----------------
static void topk_cpu_ref(const std::vector<float>& scores, int n_blocks, std::vector<int>& sel_sorted) {
    std::vector<int> idx(n_blocks);
    for (int i = 0; i < n_blocks; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + N_SEL_BLOCKS, idx.end(),
                       [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(N_SEL_BLOCKS);
    sel_sorted.clear();
    for (int b : idx) for (int r = 0; r < QSA_RATIO; r++) sel_sorted.push_back(b * QSA_RATIO + r);
    std::sort(sel_sorted.begin(), sel_sorted.end());
}

static void topk_emul(const std::vector<float>& scores, int n_blocks, int k_blocks, int ratio,
                       std::vector<int>& out_tokens, int& n_out) {
    std::vector<uint32_t> s_hist(TOPK_HI_BINS, 0u), s_scan(TOPK_THREADS, 0u);
    int s_found = 0, s_need = 0, s_cand = 0, s_out = 0, s_tie = 0;
    uint32_t s_above = 0u, s_prefix = 0u;
    std::vector<uint32_t> cand_key(n_blocks);
    std::vector<int> cand_idx(n_blocks);
    out_tokens.assign(k_blocks * ratio, -1);
    const int k = (k_blocks < n_blocks) ? k_blocks : n_blocks;

    // pass 0
    for (int i = 0; i < n_blocks; i++) s_hist[ord_key(scores[i]) >> TOPK_HI_SHIFT]++;
    // suffix scan, 2 bins/thread
    for (int t = 0; t < TOPK_THREADS; t++)
        s_scan[t] = s_hist[TOPK_HI_BINS - 1 - 2 * t] + s_hist[TOPK_HI_BINS - 2 - 2 * t];
    for (int off = 1; off < TOPK_THREADS; off <<= 1) {
        std::vector<uint32_t> v(TOPK_THREADS);
        for (int t = 0; t < TOPK_THREADS; t++) v[t] = (t >= off) ? s_scan[t - off] : 0u;
        for (int t = 0; t < TOPK_THREADS; t++) s_scan[t] += v[t];
    }
    for (int t = 0; t < TOPK_THREADS; t++) {
        uint32_t incl = s_scan[t], excl = (t == 0) ? 0u : s_scan[t - 1];
        if (incl >= (uint32_t)k && excl < (uint32_t)k) { s_found = t; s_above = excl; }
    }
    {
        uint32_t above = s_above;
        int bin = TOPK_HI_BINS - 1 - 2 * s_found;
        uint32_t c0 = s_hist[bin];
        if (above + c0 < (uint32_t)k) { above += c0; bin -= 1; }
        s_prefix = ((uint32_t)bin) << TOPK_HI_SHIFT;
        s_need = k - (int)above;
    }
    // pass 1
    const uint32_t hi = s_prefix >> TOPK_HI_SHIFT;
    for (int i = 0; i < n_blocks; i++) {
        uint32_t u = ord_key(scores[i]);
        uint32_t b = u >> TOPK_HI_SHIFT;
        if (b > hi) {
            int pos = s_out++;
            if (pos < k) for (int r = 0; r < ratio; r++) out_tokens[pos * ratio + r] = i * ratio + r;
        } else if (b == hi) {
            int c = s_cand++;
            if (c < n_blocks) { cand_key[c] = u; cand_idx[c] = i; }
        }
    }
    // pass 2
    for (int p = 0; p < TOPK_LO_PASSES; p++) {
        const int shift = TOPK_LO_BITS * (TOPK_LO_PASSES - 1 - p);
        const uint32_t himask = ~((1u << (shift + TOPK_LO_BITS)) - 1u);
        for (int i = 0; i < TOPK_LO_BINS; i++) s_hist[i] = 0u;
        for (int i = 0; i < s_cand; i++) {
            uint32_t u = cand_key[i];
            if ((u & himask) == (s_prefix & himask)) s_hist[(u >> shift) & (TOPK_LO_BINS - 1)]++;
        }
        for (int t = 0; t < TOPK_LO_BINS; t++) s_scan[t] = s_hist[TOPK_LO_BINS - 1 - t];
        for (int off = 1; off < TOPK_LO_BINS; off <<= 1) {
            std::vector<uint32_t> v(TOPK_LO_BINS);
            for (int t = 0; t < TOPK_LO_BINS; t++) v[t] = (t >= off) ? s_scan[t - off] : 0u;
            for (int t = 0; t < TOPK_LO_BINS; t++) s_scan[t] += v[t];
        }
        for (int t = 0; t < TOPK_LO_BINS; t++) {
            uint32_t incl = s_scan[t], excl = (t == 0) ? 0u : s_scan[t - 1];
            if (incl >= (uint32_t)s_need && excl < (uint32_t)s_need) { s_found = t; s_above = excl; }
        }
        int bin = TOPK_LO_BINS - 1 - s_found;
        s_prefix |= ((uint32_t)bin) << shift;
        s_need -= (int)s_above;
    }
    // pass 3
    {
        uint32_t thr = s_prefix;
        int need = s_need;
        for (int i = 0; i < s_cand; i++) {
            uint32_t u = cand_key[i];
            bool take = (u > thr);
            if (!take && u == thr) take = (s_tie++ < need);
            if (take) {
                int pos = s_out++;
                if (pos < k) {
                    int b = cand_idx[i];
                    for (int r = 0; r < ratio; r++) out_tokens[pos * ratio + r] = b * ratio + r;
                }
            }
        }
    }
    n_out = s_out;
}

// ---------------- 3. attention ----------------
static float dq(const int8_t* qs, const float* sc, int d) { return sc[d >> 5] * (float)qs[d]; }

static void attn_cpu_ref(const std::vector<int8_t>& Kqs, const std::vector<float>& Ksc,
                          const std::vector<int8_t>& Vqs, const std::vector<float>& Vsc,
                          const std::vector<float>& Q, const std::vector<float>& gate,
                          const std::vector<int>& sel, std::vector<float>& out) {
    int n_sel = (int)sel.size();
    std::vector<float> scores((size_t)N_Q_HEADS * n_sel);
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kvh = h / GQA_GROUP;
        for (int j = 0; j < n_sel; j++) {
            int tok = sel[j];
            const int8_t* qs = &Kqs[((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM];
            const float* sc = &Ksc[((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32)];
            double acc = 0.0;
            for (int d = 0; d < HEAD_DIM; d++) acc += (double)Q[(size_t)h * HEAD_DIM + d] * dq(qs, sc, d);
            scores[(size_t)h * n_sel + j] = (float)(acc / std::sqrt((double)HEAD_DIM));
        }
        float* row = &scores[(size_t)h * n_sel];
        float mx = *std::max_element(row, row + n_sel);
        double sum = 0.0;
        for (int j = 0; j < n_sel; j++) { row[j] = std::exp(row[j] - mx); sum += row[j]; }
        for (int j = 0; j < n_sel; j++) row[j] = (float)(row[j] / sum);
    }
    out.assign((size_t)N_Q_HEADS * HEAD_DIM, 0.f);
    for (int h = 0; h < N_Q_HEADS; h++) {
        int kvh = h / GQA_GROUP;
        const float* w = &scores[(size_t)h * n_sel];
        for (int d = 0; d < HEAD_DIM; d++) {
            double acc = 0.0;
            for (int j = 0; j < n_sel; j++) {
                int tok = sel[j];
                acc += (double)w[j] * dq(&Vqs[((size_t)tok * N_KV_HEADS + kvh) * HEAD_DIM],
                                          &Vsc[((size_t)tok * N_KV_HEADS + kvh) * (HEAD_DIM / 32)], d);
            }
            double g = 1.0 / (1.0 + std::exp(-(double)gate[(size_t)h * HEAD_DIM + d]));
            out[(size_t)h * HEAD_DIM + d] = (float)(acc * g);
        }
    }
}

static void q8_bytes(int lo, int hi, float sc, float* o) {
    for (int i = 0; i < 4; i++) o[i] = sc * (float)(int8_t)((lo >> (8 * i)) & 0xff);
    for (int i = 0; i < 4; i++) o[4 + i] = sc * (float)(int8_t)((hi >> (8 * i)) & 0xff);
}
static void load_i2(const int8_t* p, int& lo, int& hi) {
    memcpy(&lo, p, 4); memcpy(&hi, p + 4, 4);
}

static void attn_emul(const std::vector<int8_t>& Kqs, const std::vector<float>& Ksc,
                       const std::vector<int8_t>& Vqs, const std::vector<float>& Vsc,
                       const std::vector<float>& Q, const std::vector<float>& gate,
                       const std::vector<int>& sel, std::vector<float>& out) {
    const int n_sel = (int)sel.size();
    const int n_chunks = (n_sel + ATTN_CHUNK - 1) / ATTN_CHUNK;
    std::vector<float> partial((size_t)n_chunks * N_Q_HEADS * HEAD_DIM, 0.f);
    std::vector<float> mbuf((size_t)n_chunks * N_Q_HEADS, 0.f), lbuf((size_t)n_chunks * N_Q_HEADS, 0.f);

    for (int chunk = 0; chunk < n_chunks; chunk++)
    for (int kvh = 0; kvh < N_KV_HEADS; kvh++) {
        int sh_tok[ATTN_CHUNK];
        const int base = chunk * ATTN_CHUNK;
        for (int j = 0; j < ATTN_CHUNK; j++) sh_tok[j] = (base + j < n_sel) ? sel[base + j] : 0;
        float sh_p[ATTN_WAVES][ATTN_CHUNK];
        for (int w = 0; w < ATTN_WAVES; w++) {
            const int h = kvh * GQA_GROUP + w;
            const float qscale = 1.0f / std::sqrt((float)HEAD_DIM);
            // phase A
            for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR) {
                for (int u = 0; u < ATTN_UNR; u++) {
                    float lane_acc[ATTN_LANES];
                    for (int lane = 0; lane < ATTN_LANES; lane++) {
                        int d0 = lane * ATTN_DPL, sc_off = d0 >> 5;
                        float qr[ATTN_DPL];
                        for (int i = 0; i < ATTN_DPL; i++) qr[i] = Q[(size_t)h * HEAD_DIM + d0 + i];
                        size_t row = ((size_t)sh_tok[j0 + u] * N_KV_HEADS + kvh);
                        int lo, hi; load_i2(&Kqs[row * HEAD_DIM + d0], lo, hi);
                        float sc = Ksc[row * (HEAD_DIM / 32) + sc_off];
                        float kd[ATTN_DPL]; q8_bytes(lo, hi, sc, kd);
                        float acc = 0.f;
                        for (int i = 0; i < ATTN_DPL; i++) acc = fmaf(qr[i], kd[i], acc);
                        lane_acc[lane] = acc;
                    }
                    for (int off = ATTN_LANES / 2; off > 0; off >>= 1) {
                        float t[ATTN_LANES];
                        for (int l = 0; l < ATTN_LANES; l++) t[l] = lane_acc[l] + lane_acc[l ^ off];
                        memcpy(lane_acc, t, sizeof(t));
                    }
                    sh_p[w][j0 + u] = (base + j0 + u < n_sel) ? lane_acc[0] * qscale : ATTN_NEG_INF;
                }
            }
            // phase A2
            float mx = ATTN_NEG_INF, sum = 0.f;
            for (int j = 0; j < ATTN_CHUNK; j++) mx = std::fmax(mx, sh_p[w][j]);
            for (int j = 0; j < ATTN_CHUNK; j++) {
                float s = sh_p[w][j];
                float e = (s <= ATTN_NEG_INF) ? 0.f : std::exp(s - mx);
                sh_p[w][j] = e; sum += e;
            }
            mbuf[(size_t)chunk * N_Q_HEADS + h] = mx;
            lbuf[(size_t)chunk * N_Q_HEADS + h] = sum;
            // phase B
            for (int lane = 0; lane < ATTN_LANES; lane++) {
                int d0 = lane * ATTN_DPL, sc_off = d0 >> 5;
                float acc[ATTN_DPL] = {0};
                for (int j0 = 0; j0 < ATTN_CHUNK; j0 += ATTN_UNR)
                    for (int u = 0; u < ATTN_UNR; u++) {
                        size_t row = ((size_t)sh_tok[j0 + u] * N_KV_HEADS + kvh);
                        int lo, hi; load_i2(&Vqs[row * HEAD_DIM + d0], lo, hi);
                        float vd[ATTN_DPL]; q8_bytes(lo, hi, Vsc[row * (HEAD_DIM / 32) + sc_off], vd);
                        float p = sh_p[w][j0 + u];
                        for (int i = 0; i < ATTN_DPL; i++) acc[i] = fmaf(p, vd[i], acc[i]);
                    }
                for (int i = 0; i < ATTN_DPL; i++)
                    partial[((size_t)chunk * N_Q_HEADS + h) * HEAD_DIM + d0 + i] = acc[i];
            }
        }
    }
    // combine
    out.assign((size_t)N_Q_HEADS * HEAD_DIM, 0.f);
    for (int h = 0; h < N_Q_HEADS; h++)
        for (int d = 0; d < HEAD_DIM; d++) {
            float M = ATTN_NEG_INF;
            for (int c = 0; c < n_chunks; c++) M = std::fmax(M, mbuf[(size_t)c * N_Q_HEADS + h]);
            float acc = 0.f, denom = 0.f;
            for (int c = 0; c < n_chunks; c++) {
                float m = mbuf[(size_t)c * N_Q_HEADS + h];
                if (m <= ATTN_NEG_INF) continue;
                float s = std::exp(m - M);
                denom += s * lbuf[(size_t)c * N_Q_HEADS + h];
                acc += s * partial[((size_t)c * N_Q_HEADS + h) * HEAD_DIM + d];
            }
            float g = 1.0f / (1.0f + std::exp(-gate[(size_t)h * HEAD_DIM + d]));
            out[(size_t)h * HEAD_DIM + d] = (denom > 0.f ? acc / denom : 0.f) * g;
        }
}

// ---------------- driver ----------------
static std::mt19937 gen(12345);
static float randf(float lo = -1.f, float hi = 1.f) { return std::uniform_real_distribution<float>(lo, hi)(gen); }

static double cosine(const std::vector<float>& a, const std::vector<float>& b, double* maxabs) {
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        mx = std::max(mx, std::fabs((double)a[i] - b[i]));
    }
    *maxabs = mx;
    return (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
    printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) fails++;
}

int main() {
    // --- idx_scan ---
    for (int n_blocks : {8192, 32768, 65536}) {
        std::vector<float> q(IDX_ELEMS);
        for (auto& x : q) x = randf();
        std::vector<uint16_t> kp((size_t)n_blocks * IDX_ELEMS);
        for (auto& x : kp) x = f32_to_bf16(randf());
        std::vector<float> ref, got;
        idx_scan_cpu_ref(kp, q, ref, n_blocks);
        idx_scan_emul(kp, q, got, n_blocks);
        double mx; double c = cosine(got, ref, &mx);
        char buf[128]; snprintf(buf, sizeof buf, "idx_scan n=%d cos=%.9f maxabs=%.3g", n_blocks, c, mx);
        expect(c > 0.99999 && mx < 1e-2, buf);

        // --- topk on those scores (the harness's exact path) ---
        std::vector<int> out; int n_out = 0;
        topk_emul(got, n_blocks, N_SEL_BLOCKS, QSA_RATIO, out, n_out);
        std::vector<int> sorted = out; std::sort(sorted.begin(), sorted.end());
        std::vector<int> ref_sel; topk_cpu_ref(got, n_blocks, ref_sel);
        snprintf(buf, sizeof buf, "topk n=%d n_out=%d set-match (gpu scores)", n_blocks, n_out);
        expect(n_out == N_SEL_BLOCKS && sorted == ref_sel, buf);

        // and against the double-accumulated CPU scores, as the harness checks
        std::vector<int> ref_sel_cpu; topk_cpu_ref(ref, n_blocks, ref_sel_cpu);
        snprintf(buf, sizeof buf, "topk n=%d set-match vs CPU-ref scores", n_blocks);
        expect(sorted == ref_sel_cpu, buf);
    }

    // --- topk adversarial ---
    {
        struct Case { const char* name; int n; int kind; };
        for (Case cs : { Case{"all equal", 8192, 0}, Case{"two values", 8192, 1},
                         Case{"all negative", 8192, 2}, Case{"huge dynamic range", 8192, 3},
                         Case{"many ties at cut", 16384, 4}, Case{"exactly k blocks", 512, 5},
                         Case{"zeros and denormals", 8192, 6} }) {
            std::vector<float> s(cs.n);
            for (int i = 0; i < cs.n; i++) {
                switch (cs.kind) {
                    case 0: s[i] = 3.5f; break;
                    case 1: s[i] = (i % 2) ? 1.0f : -1.0f; break;
                    case 2: s[i] = -std::fabs(randf()) - 1.f; break;
                    case 3: s[i] = randf() * std::pow(10.f, (float)(i % 40) - 20.f); break;
                    case 4: s[i] = (float)(i % 32); break;
                    case 5: s[i] = randf(); break;
                    case 6: s[i] = (i % 3 == 0) ? 0.0f : (i % 3 == 1 ? -0.0f : randf() * 1e-40f); break;
                }
            }
            std::vector<int> out; int n_out = 0;
            topk_emul(s, cs.n, N_SEL_BLOCKS, QSA_RATIO, out, n_out);
            // exactly k selected, all distinct, all in range, and every selected
            // score >= every unselected score (the definition of a top-k set)
            std::vector<int> blocks;
            for (size_t i = 0; i < out.size(); i += QSA_RATIO) blocks.push_back(out[i] / QSA_RATIO);
            std::vector<char> sel(cs.n, 0);
            bool ok = (n_out == std::min(N_SEL_BLOCKS, cs.n)) && (int)blocks.size() == N_SEL_BLOCKS;
            for (int b : blocks) { if (b < 0 || b >= cs.n || sel[b]) ok = false; else sel[b] = 1; }
            float lo_sel = 1e30f, hi_uns = -1e30f;
            for (int i = 0; i < cs.n; i++) (sel[i] ? lo_sel = std::min(lo_sel, s[i]) : hi_uns = std::max(hi_uns, s[i]));
            if (lo_sel < hi_uns) ok = false;
            // token expansion must be right
            for (size_t i = 0; i < out.size(); i++)
                if (out[i] != blocks[i / QSA_RATIO] * QSA_RATIO + (int)(i % QSA_RATIO)) ok = false;
            char buf[128]; snprintf(buf, sizeof buf, "topk adversarial: %-22s n=%d", cs.name, cs.n);
            expect(ok, buf);
        }
    }

    // --- attention ---
    for (int n_sel : {2048, 128, 100}) {          // 100: a ragged tail chunk
        const int N = 4096;
        std::vector<float> Q((size_t)N_Q_HEADS * HEAD_DIM), gate((size_t)N_Q_HEADS * HEAD_DIM);
        for (auto& x : Q) x = randf();
        for (auto& x : gate) x = randf();
        std::vector<int8_t> Kq((size_t)N * N_KV_HEADS * HEAD_DIM), Vq(Kq.size());
        std::vector<float> Ks((size_t)N * N_KV_HEADS * (HEAD_DIM / 32)), Vs(Ks.size());
        for (size_t i = 0; i < Kq.size(); i++) { Kq[i] = (int8_t)(int)(randf(-127, 127)); Vq[i] = (int8_t)(int)(randf(-127, 127)); }
        for (size_t i = 0; i < Ks.size(); i++) { Ks[i] = std::fabs(randf(0.001f, 0.02f)); Vs[i] = std::fabs(randf(0.001f, 0.02f)); }
        std::vector<int> sel(n_sel);
        {
            std::vector<int> all(N); for (int i = 0; i < N; i++) all[i] = i;
            std::shuffle(all.begin(), all.end(), gen);
            for (int i = 0; i < n_sel; i++) sel[i] = all[i];
            std::sort(sel.begin(), sel.end());
        }
        std::vector<float> ref, got;
        attn_cpu_ref(Kq, Ks, Vq, Vs, Q, gate, sel, ref);
        attn_emul(Kq, Ks, Vq, Vs, Q, gate, sel, got);
        double mx; double c = cosine(got, ref, &mx);
        // relative to the magnitude of the output, not per-element: several
        // outputs are legitimately ~1e-8 and a per-element ratio there says
        // nothing (that is how the first run of this test "failed").
        double scale = 0; for (float v : ref) scale = std::max(scale, (double)std::fabs(v));
        double rel = mx / scale;
        char buf[160]; snprintf(buf, sizeof buf, "attn n_sel=%4d cos=%.9f maxabs=%.3g rel=%.3g", n_sel, c, mx, rel);
        expect(c > 0.9999999 && rel < 1e-4, buf);
    }

    printf("\n%s (%d failures)\n", fails ? "FAILURES" : "all checks passed", fails);
    return fails ? 1 : 0;
}
