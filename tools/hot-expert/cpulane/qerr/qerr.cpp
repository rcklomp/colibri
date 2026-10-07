// qerr.cpp -- CPU-only numerical experiment: how much does a routed expert's output (and the layer's MoE output)
// change on REAL GLM-5.3-Flash data when the expert is computed with ggml's INT8-activation chain
// (Q8_K(x) -> ggml_vec_dot_<type>_q8_K for gate/up -> swiglu -> Q8_K(h) -> ggml_vec_dot_<type>_q8_K for down)
// instead of the engine's FLOAT-activation chain?
//
// Inputs (read only): the split GGUF (expert weights, mmap'd), reference dumps of the GPU engine (float32 taps of
// the last token of a prompt: ffn_norm, router scores/weights, per-slot gate/up/swiglu/down, ffn_moe_out, ...).
// No GPU code, no threads unless --threads. See README.md for the method.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ggml.h"
#include "gguf.h"

extern "C" {
void quantize_row_q8_K(const float * x, void * y, int64_t k);   // ggml-cpu/arch/x86/quants.c (-> quantize_row_q8_K_ref)
void qerr_ggml_init(void);                                      // qerr_stubs.c: lookup tables
void dequantize_row_q8_K(const void * x, float * y, int64_t k); // ggml-quants.c (x = block_q8_K *)
typedef void (*vec_dot_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
#define QDECL(n) void ggml_vec_dot_##n##_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int)
QDECL(iq3_s); QDECL(iq4_xs); QDECL(q6_K); QDECL(q5_K); QDECL(q4_K); QDECL(q3_K); QDECL(q2_K);
QDECL(iq2_xxs); QDECL(iq2_xs); QDECL(iq2_s); QDECL(iq3_xxs); QDECL(iq1_s); QDECL(iq1_m);
}

static vec_dot_t vec_dot_for(ggml_type t) {   // every type whose vec_dot_type is Q8_K
    switch (t) {
        case GGML_TYPE_IQ3_S:   return ggml_vec_dot_iq3_s_q8_K;
        case GGML_TYPE_IQ4_XS:  return ggml_vec_dot_iq4_xs_q8_K;
        case GGML_TYPE_Q6_K:    return ggml_vec_dot_q6_K_q8_K;
        case GGML_TYPE_Q5_K:    return ggml_vec_dot_q5_K_q8_K;
        case GGML_TYPE_Q4_K:    return ggml_vec_dot_q4_K_q8_K;
        case GGML_TYPE_Q3_K:    return ggml_vec_dot_q3_K_q8_K;
        case GGML_TYPE_Q2_K:    return ggml_vec_dot_q2_K_q8_K;
        case GGML_TYPE_IQ2_XXS: return ggml_vec_dot_iq2_xxs_q8_K;
        case GGML_TYPE_IQ2_XS:  return ggml_vec_dot_iq2_xs_q8_K;
        case GGML_TYPE_IQ2_S:   return ggml_vec_dot_iq2_s_q8_K;
        case GGML_TYPE_IQ3_XXS: return ggml_vec_dot_iq3_xxs_q8_K;
        case GGML_TYPE_IQ1_S:   return ggml_vec_dot_iq1_s_q8_K;
        case GGML_TYPE_IQ1_M:   return ggml_vec_dot_iq1_m_q8_K;
        default:                return nullptr;
    }
}

constexpr float SWIGLU_LIMIT = 10.0f;   // glm5_shapes.h SWIGLU_CLAMP
constexpr int   N_USED = 8;

// ------------------------------------------------------------------ the engine's SwiGLU (glm5_cpu.cpp rows_for, "ds4 swiglu_clamp")
//   g = min(gate, limit); u = clamp(up, -limit, limit); h = g / (1 + exp(-g)) * u
static inline float swiglu_f(float gate, float up, float lim) {
    const float g = std::min(gate, lim);
    const float u = std::min(std::max(up, -lim), lim);
    return g / (1.f + expf(-g)) * u;
}
static inline double swiglu_d(double gate, double up, double lim) {
    const double g = std::min(gate, lim);
    const double u = std::min(std::max(up, -lim), lim);
    return g / (1.0 + std::exp(-g)) * u;
}

// ------------------------------------------------------------------ model (split GGUF, mmap'd read only)
struct Shard {
    std::string path;
    gguf_context * g = nullptr;
    ggml_context * c = nullptr;
    const uint8_t * map = nullptr;
    size_t size = 0, data_off = 0;
};
struct Tensor {   // a 3-D expert tensor [K, N, E]: expert e is a contiguous slab of N rows of row_bytes
    std::string name;
    const uint8_t * base = nullptr;
    ggml_type type = GGML_TYPE_F32;
    int64_t K = 0, N = 0, E = 0;
    size_t row_bytes = 0;
    const uint8_t * row(int e, int r) const { return base + ((size_t) e * (size_t) N + (size_t) r) * row_bytes; }
};
struct Model {
    std::vector<Shard> shards;
    bool open(const std::string & dir, std::string & err) {
        for (int i = 1; i <= 5; ++i) {
            char p[1024]; snprintf(p, sizeof p, "%s/GLM-5.3-Flash-UD-IQ4_XS-%05d-of-00005.gguf", dir.c_str(), i);
            Shard s; s.path = p;
            gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &s.c };
            s.g = gguf_init_from_file(p, ip);
            if (!s.g) { err = std::string("gguf_init_from_file failed: ") + p; return false; }
            const int fd = ::open(p, O_RDONLY);
            if (fd < 0) { err = std::string("open failed: ") + p; return false; }
            struct stat st; fstat(fd, &st);
            s.size = (size_t) st.st_size;
            void * m = mmap(nullptr, s.size, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) { err = std::string("mmap failed: ") + p; return false; }
            s.map = (const uint8_t *) m;
            s.data_off = gguf_get_data_offset(s.g);
            shards.push_back(s);
        }
        return true;
    }
    bool find(const std::string & name, Tensor & t, std::string & err) const {
        for (const Shard & s : shards) {
            const int64_t id = gguf_find_tensor(s.g, name.c_str());
            if (id < 0) continue;
            const ggml_tensor * gt = ggml_get_tensor(s.c, name.c_str());
            if (!gt) { err = "ggml_get_tensor failed: " + name; return false; }
            t.name = name;
            t.type = gguf_get_tensor_type(s.g, id);
            t.K = gt->ne[0]; t.N = gt->ne[1]; t.E = gt->ne[2];
            t.row_bytes = ggml_row_size(t.type, t.K);
            const size_t off = s.data_off + gguf_get_tensor_offset(s.g, id), nb = gguf_get_tensor_size(s.g, id);
            if (nb != t.row_bytes * (size_t) t.N * (size_t) t.E) { err = "size mismatch (not a plain 3-D tensor): " + name; return false; }
            if (off + nb > s.size) { err = "tensor beyond end of file: " + name; return false; }
            t.base = s.map + off;
            return true;
        }
        err = "tensor not found: " + name;
        return false;
    }
};

// ------------------------------------------------------------------ small numerics
static inline double dotd(const float * w, const double * x, int n) {   // float row . double vector, 8 independent double accumulators
    double a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        a0 += (double) w[i + 0] * x[i + 0]; a1 += (double) w[i + 1] * x[i + 1];
        a2 += (double) w[i + 2] * x[i + 2]; a3 += (double) w[i + 3] * x[i + 3];
        a4 += (double) w[i + 4] * x[i + 4]; a5 += (double) w[i + 5] * x[i + 5];
        a6 += (double) w[i + 6] * x[i + 6]; a7 += (double) w[i + 7] * x[i + 7];
    }
    for (; i < n; ++i) a0 += (double) w[i] * x[i];
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}
static double norm2(const double * a, int n) { double s = 0; for (int i = 0; i < n; ++i) s += a[i] * a[i]; return std::sqrt(s); }
static double rel_err(const double * a, const double * ref, int n) {   // ||a - ref|| / ||ref||
    double d = 0, r = 0;
    for (int i = 0; i < n; ++i) { const double e = a[i] - ref[i]; d += e * e; r += ref[i] * ref[i]; }
    return r > 0 ? std::sqrt(d / r) : (d > 0 ? INFINITY : 0.0);
}
static double cos_1m(const double * a, const double * b, int n) {   // 1 - cos(a, b) = ||a/|a| - b/|b|||^2 / 2 (no cancellation)
    const double na = norm2(a, n), nb = norm2(b, n);
    if (na == 0 || nb == 0) return 1.0;
    double s = 0;
    for (int i = 0; i < n; ++i) { const double e = a[i] / na - b[i] / nb; s += e * e; }
    return 0.5 * s;
}
static std::vector<double> to_d(const std::vector<float> & v) { return std::vector<double>(v.begin(), v.end()); }
static std::vector<double> slice(const std::vector<double> & v, int k, int n) { return std::vector<double>(v.begin() + (size_t) k * n, v.begin() + (size_t) (k + 1) * n); }

static bool read_f32(const std::string & path, size_t n, std::vector<float> & v, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if ((size_t) sz != n * 4) { fclose(f); err = "size of " + path + " is " + std::to_string(sz) + " B, expected " + std::to_string(n * 4); return false; }
    v.resize(n);
    const size_t got = fread(v.data(), 4, n, f);
    fclose(f);
    if (got != n) { err = "short read " + path; return false; }
    return true;
}

// ------------------------------------------------------------------ one expert, both chains
struct Weights { Tensor gate, up, down; };
struct ExpOut {
    int expert = -1;
    std::vector<double> gate_f, up_f, h_f, out_f;   // "exact": dequantised weights (ggml to_float), float x, double accumulation
    std::vector<double> gate_i, up_i, h_i, out_i;   // ggml int8 chain: Q8_K(x), vec_dot, swiglu, Q8_K(h), vec_dot
    std::vector<double> out_xq, out_hq;             // ablations: only x quantised (float down on h_i) / only h quantised (int8 down on h_f)
    double kern_g = 0, kern_u = 0, kern_d = 0;      // kernel self-check: vec_dot result vs (dequantised row) . (dequantised Q8_K activation), rel L2
};

static void compute_expert(const Weights & W, int e, const std::vector<float> & x, ExpOut & o) {
    const int K = (int) W.gate.K, F = (int) W.gate.N, D = (int) W.down.N;   // 4096, 2048, 4096; down.K == F
    const ggml_type_traits * tg = ggml_get_type_traits(W.gate.type), * tu = ggml_get_type_traits(W.up.type), * td = ggml_get_type_traits(W.down.type);
    const vec_dot_t dg = vec_dot_for(W.gate.type), du = vec_dot_for(W.up.type), dd = vec_dot_for(W.down.type);
    o.expert = e;
    o.gate_f.assign(F, 0); o.up_f.assign(F, 0); o.h_f.assign(F, 0); o.out_f.assign(D, 0);
    o.gate_i.assign(F, 0); o.up_i.assign(F, 0); o.h_i.assign(F, 0); o.out_i.assign(D, 0);
    o.out_xq.assign(D, 0); o.out_hq.assign(D, 0);

    std::vector<double> xd(x.begin(), x.end());
    std::vector<uint8_t> q8x(ggml_row_size(GGML_TYPE_Q8_K, K)), q8hf(ggml_row_size(GGML_TYPE_Q8_K, F)), q8hi(ggml_row_size(GGML_TYPE_Q8_K, F));
    std::vector<float> rowbuf((size_t) std::max(K, F));
    quantize_row_q8_K(x.data(), q8x.data(), K);
    std::vector<float> xhat32(K);   // what the int8 kernels actually multiply with: d * qs of the Q8_K blocks
    dequantize_row_q8_K(q8x.data(), xhat32.data(), K);
    const std::vector<double> xh(xhat32.begin(), xhat32.end());
    double kg_d = 0, kg_r = 0, ku_d = 0, ku_r = 0, kd_d = 0, kd_r = 0;

    for (int r = 0; r < F; ++r) {   // gate and up: float chain (dequantise + double dot) and int8 chain (vec_dot) on the same raw row
        float s;
        const uint8_t * pg = W.gate.row(e, r), * pu = W.up.row(e, r);
        tg->to_float(pg, rowbuf.data(), K);  o.gate_f[r] = dotd(rowbuf.data(), xd.data(), K);
        dg(K, &s, 0, pg, 0, q8x.data(), 0, 1);  o.gate_i[r] = s;
        { const double k = dotd(rowbuf.data(), xh.data(), K); kg_d += (s - k) * (s - k); kg_r += k * k; }
        tu->to_float(pu, rowbuf.data(), K);  o.up_f[r] = dotd(rowbuf.data(), xd.data(), K);
        du(K, &s, 0, pu, 0, q8x.data(), 0, 1);  o.up_i[r] = s;
        { const double k = dotd(rowbuf.data(), xh.data(), K); ku_d += (s - k) * (s - k); ku_r += k * k; }
    }
    std::vector<float> hf32(F), hi32(F);
    for (int r = 0; r < F; ++r) {
        o.h_f[r] = swiglu_d(o.gate_f[r], o.up_f[r], SWIGLU_LIMIT);
        hi32[r]  = swiglu_f((float) o.gate_i[r], (float) o.up_i[r], SWIGLU_LIMIT);
        hf32[r]  = (float) o.h_f[r];
        o.h_i[r] = hi32[r];
    }
    quantize_row_q8_K(hf32.data(), q8hf.data(), F);
    quantize_row_q8_K(hi32.data(), q8hi.data(), F);
    std::vector<float> hhat32(F);
    dequantize_row_q8_K(q8hi.data(), hhat32.data(), F);
    const std::vector<double> hh(hhat32.begin(), hhat32.end());
    for (int r = 0; r < D; ++r) {   // down: one dequantised row against h_f and h_i (double), and two vec_dots
        float s;
        const uint8_t * p = W.down.row(e, r);
        td->to_float(p, rowbuf.data(), F);
        o.out_f[r]  = dotd(rowbuf.data(), o.h_f.data(), F);
        o.out_xq[r] = dotd(rowbuf.data(), o.h_i.data(), F);
        dd(F, &s, 0, p, 0, q8hi.data(), 0, 1);  o.out_i[r]  = s;
        { const double k = dotd(rowbuf.data(), hh.data(), F); kd_d += (s - k) * (s - k); kd_r += k * k; }
        dd(F, &s, 0, p, 0, q8hf.data(), 0, 1);  o.out_hq[r] = s;
    }
    o.kern_g = std::sqrt(kg_d / kg_r); o.kern_u = std::sqrt(ku_d / ku_r); o.kern_d = std::sqrt(kd_d / kd_r);
}

// ------------------------------------------------------------------ per (dump, layer) result
struct SlotRec {
    int slot = 0, rank = 0, expert = 0;
    double idcos1m = 0;                              // 1 - cos(float reference, GPU ffn_moe_down slot) of the chosen candidate
    double gpu_rel = 0, gpu_g_rel = 0, gpu_u_rel = 0; // float reference vs the GPU taps (harness validation)
    double w = 0, onorm = 0, kern = 0;
    double e_rel = 0, e_c1m = 0, g_rel = 0, g_c1m = 0, u_rel = 0, u_c1m = 0, h_rel = 0, xq_rel = 0, hq_rel = 0;
};
struct LayerRec {
    int dump = 0, L = 0;
    std::string tg, tu, td, label;
    bool ok = true, fatal = false;
    std::string err, text;
    SlotRec s[N_USED];
    bool rank_identity = false;
    std::string wtap;
    double y_tap_rel = 0, y_ref_rel = 0, ffn_id_rel = 0, swiglu_chk = 0;
    int clamp_n = 0;
    double dY1[N_USED] = {}, dF1[N_USED] = {};
    double dYmin = 0, dYmax = 0, dFmin = 0, dFmax = 0, dYall = 0, dFall = 0;
    int kmin = 0, kmax = 0;
};

struct Cfg {
    std::string model_dir;
    std::vector<std::pair<std::string, std::string>> dumps;
    int l0 = 3, l1 = 44, threads = 1;
    bool keep_going = false, list_types = false;
};

static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char * f, ...) {
    char b[4096]; va_list ap; va_start(ap, f); vsnprintf(b, sizeof b, f, ap); va_end(ap); return b;
}

static LayerRec process_layer(const Model & M, const Cfg & cfg, int di, int L) {
    LayerRec R; R.dump = di; R.L = L; R.label = cfg.dumps[di].first;
    const std::string & dir = cfg.dumps[di].second;
    std::string err;
    Weights W;
    if (!M.find(fmt("blk.%d.ffn_gate_exps.weight", L), W.gate, err) || !M.find(fmt("blk.%d.ffn_up_exps.weight", L), W.up, err) ||
        !M.find(fmt("blk.%d.ffn_down_exps.weight", L), W.down, err)) { R.ok = false; R.err = err; return R; }
    R.tg = ggml_type_name(W.gate.type); R.tu = ggml_type_name(W.up.type); R.td = ggml_type_name(W.down.type);
    const int K = (int) W.gate.K, F = (int) W.gate.N, D = (int) W.down.N, E = (int) W.gate.E;
    auto tap = [&](const char * name, size_t n, std::vector<float> & v) { return read_f32(fmt("%s/%s-%d.f32", dir.c_str(), name, L), n, v, err); };
    std::vector<float> xf, probs, wt, wtn, wts, gate_t, up_t, swi_t, down_t, wtd_t, out_t, shexp_t, ffn_t;
    if (!tap("ffn_norm", K, xf) || !tap("ffn_moe_probs_biased", E, probs) || !tap("ffn_moe_weights", N_USED, wt) ||
        !tap("ffn_moe_weights_norm", N_USED, wtn) || !tap("ffn_moe_weights_scaled", N_USED, wts) ||
        !tap("ffn_moe_gate", (size_t) N_USED * F, gate_t) || !tap("ffn_moe_up", (size_t) N_USED * F, up_t) ||
        !tap("ffn_moe_swiglu_limited", (size_t) N_USED * F, swi_t) || !tap("ffn_moe_down", (size_t) N_USED * D, down_t) ||
        !tap("ffn_moe_weighted", (size_t) N_USED * D, wtd_t) || !tap("ffn_moe_out", D, out_t) || !tap("ffn_shexp", D, shexp_t) ||
        !tap("ffn_out", D, ffn_t)) { R.ok = false; R.err = err; return R; }
    const std::vector<double> gate_d = to_d(gate_t), up_d = to_d(up_t), swi_d = to_d(swi_t), down_d = to_d(down_t), wtd_d = to_d(wtd_t),
                              out_d = to_d(out_t), shexp_d = to_d(shexp_t), ffn_d = to_d(ffn_t);

    // swiglu formula check: engine formula on the GPU's own gate/up taps vs the GPU's swiglu_limited tap
    {
        double d = 0, r = 0;
        for (size_t i = 0; i < gate_t.size(); ++i) {
            const double h = swiglu_f(gate_t[i], up_t[i], SWIGLU_LIMIT), e = h - swi_t[i];
            d += e * e; r += (double) swi_t[i] * swi_t[i];
            if (gate_t[i] > SWIGLU_LIMIT || std::fabs(up_t[i]) > SWIGLU_LIMIT) R.clamp_n++;
        }
        R.swiglu_chk = std::sqrt(d / r);
    }

    // candidates: top-8 of the router's selection score (ties: lower index first)
    std::vector<int> idx(E);
    for (int i = 0; i < E; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
    int cand[N_USED];
    for (int i = 0; i < N_USED; ++i) cand[i] = idx[i];

    ExpOut eo[N_USED];
    for (int c = 0; c < N_USED; ++c) compute_expert(W, cand[c], xf, eo[c]);

    // slot identification: slot s = the candidate whose float reference output best matches the GPU's ffn_moe_down slot s
    int slot_cand[N_USED]; bool used[N_USED] = {false}, dup = false;
    R.rank_identity = true;
    for (int s = 0; s < N_USED; ++s) {
        const std::vector<double> ds = slice(down_d, s, D);
        int best = -1; double bc = 1e300;
        for (int c = 0; c < N_USED; ++c) {
            const double c1 = cos_1m(eo[c].out_f.data(), ds.data(), D);
            if (c1 < bc) { bc = c1; best = c; }
        }
        slot_cand[s] = best;
        if (used[best]) dup = true;
        used[best] = true;
        if (best != s) R.rank_identity = false;
        SlotRec & sr = R.s[s];
        sr.slot = s; sr.rank = best; sr.expert = cand[best]; sr.idcos1m = bc;
        if (bc > 1e-4) {   // cos < 0.9999: the harness does not reproduce the engine
            R.fatal = true;
            R.err += fmt("FATAL: slot %d best cosine %.8f (candidate rank %d, expert %d) < 0.9999: float reference does not match ffn_moe_down. ", s, 1.0 - bc, best, cand[best]);
        } else if (bc > 1e-5) {
            R.err += fmt("WARN: slot %d best cosine %.8f < 0.99999. ", s, 1.0 - bc);
        }
    }
    if (dup) { R.fatal = true; R.err += "FATAL: two slots matched the same candidate expert. "; }

    // weights: which tap reproduces ffn_moe_weighted = w_s * down_s ?
    const std::vector<float> * wcand[3] = { &wt, &wtn, &wts };
    const char * wname[3] = { "weights", "weights_norm", "weights_scaled" };
    int wbest = 0; double wbe = 1e300;
    for (int k = 0; k < 3; ++k) {
        std::vector<double> a((size_t) N_USED * D);
        for (int s = 0; s < N_USED; ++s) for (int i = 0; i < D; ++i) a[(size_t) s * D + i] = (double) (*wcand[k])[s] * down_d[(size_t) s * D + i];
        const double e = rel_err(a.data(), wtd_d.data(), N_USED * D);
        if (e < wbe) { wbe = e; wbest = k; }
    }
    R.wtap = wname[wbest];
    double w[N_USED];
    for (int s = 0; s < N_USED; ++s) w[s] = (*wcand[wbest])[s];

    // per-slot metrics
    for (int s = 0; s < N_USED; ++s) {
        const ExpOut & o = eo[slot_cand[s]];
        SlotRec & sr = R.s[s];
        sr.w = w[s]; sr.onorm = norm2(o.out_f.data(), D);
        sr.kern = std::max(o.kern_g, std::max(o.kern_u, o.kern_d));
        const std::vector<double> ds = slice(down_d, s, D), gs = slice(gate_d, s, F), us = slice(up_d, s, F);
        sr.gpu_rel = rel_err(o.out_f.data(), ds.data(), D);
        sr.gpu_g_rel = rel_err(o.gate_f.data(), gs.data(), F);
        sr.gpu_u_rel = rel_err(o.up_f.data(), us.data(), F);
        sr.e_rel = rel_err(o.out_i.data(), o.out_f.data(), D);  sr.e_c1m = cos_1m(o.out_i.data(), o.out_f.data(), D);
        sr.g_rel = rel_err(o.gate_i.data(), o.gate_f.data(), F); sr.g_c1m = cos_1m(o.gate_i.data(), o.gate_f.data(), F);
        sr.u_rel = rel_err(o.up_i.data(), o.up_f.data(), F);     sr.u_c1m = cos_1m(o.up_i.data(), o.up_f.data(), F);
        sr.h_rel = rel_err(o.h_i.data(), o.h_f.data(), F);
        sr.xq_rel = rel_err(o.out_xq.data(), o.out_f.data(), D);
        sr.hq_rel = rel_err(o.out_hq.data(), o.out_f.data(), D);
    }

    // layer level: Y = sum_s w_s out_s
    std::vector<double> Yf(D, 0.0), Ytap(D, 0.0), dAll(D, 0.0);
    for (int s = 0; s < N_USED; ++s) {
        const ExpOut & o = eo[slot_cand[s]];
        for (int i = 0; i < D; ++i) {
            Yf[i]   += w[s] * o.out_f[i];
            Ytap[i] += w[s] * down_d[(size_t) s * D + i];
            dAll[i] += w[s] * (o.out_i[i] - o.out_f[i]);
        }
    }
    R.y_tap_rel = rel_err(Ytap.data(), out_d.data(), D);
    R.y_ref_rel = rel_err(Yf.data(), out_d.data(), D);
    {   // ffn_out = Y + shexp ?
        std::vector<double> t(D);
        for (int i = 0; i < D; ++i) t[i] = out_d[i] + shexp_d[i];
        R.ffn_id_rel = rel_err(t.data(), ffn_d.data(), D);
    }
    std::vector<double> Ffn(D);   // the denominator for the ffn_out-level change: our float Y + the GPU's shared-expert output
    for (int i = 0; i < D; ++i) Ffn[i] = Yf[i] + shexp_d[i];
    const double nY = norm2(Yf.data(), D), nF = norm2(Ffn.data(), D);
    double mmin = 1e300, mmax = -1;
    R.kmin = R.kmax = 0;
    for (int s = 0; s < N_USED; ++s) {
        const ExpOut & o = eo[slot_cand[s]];
        std::vector<double> dl(D);
        for (int i = 0; i < D; ++i) dl[i] = w[s] * (o.out_i[i] - o.out_f[i]);
        const double nd = norm2(dl.data(), D);
        R.dY1[s] = nd / nY; R.dF1[s] = nd / nF;
        const double m = std::fabs(w[s]) * R.s[s].onorm;
        if (m < mmin) { mmin = m; R.kmin = s; }
        if (m > mmax) { mmax = m; R.kmax = s; }
    }
    R.dYmin = R.dY1[R.kmin]; R.dFmin = R.dF1[R.kmin];
    R.dYmax = R.dY1[R.kmax]; R.dFmax = R.dF1[R.kmax];
    R.dYall = norm2(dAll.data(), D) / nY; R.dFall = norm2(dAll.data(), D) / nF;

    // text
    for (int s = 0; s < N_USED; ++s) {
        const SlotRec & q = R.s[s];
        R.text += fmt("EXP dump=%s L=%d slot=%d rank=%d expert=%d types=%s/%s/%s idcos=%.9f gpu_rel=%.2e gpu_gate_rel=%.2e gpu_up_rel=%.2e kern_rel=%.2e w=%.5f onorm=%.4g "
                      "e_rel=%.4e e_1mcos=%.3e g_rel=%.4e g_1mcos=%.3e u_rel=%.4e u_1mcos=%.3e h_rel=%.4e xq_rel=%.4e hq_rel=%.4e\n",
                      R.label.c_str(), L, s, q.rank, q.expert, R.tg.c_str(), R.tu.c_str(), R.td.c_str(), 1.0 - q.idcos1m, q.gpu_rel, q.gpu_g_rel, q.gpu_u_rel, q.kern, q.w, q.onorm,
                      q.e_rel, q.e_c1m, q.g_rel, q.g_c1m, q.u_rel, q.u_c1m, q.h_rel, q.xq_rel, q.hq_rel);
    }
    R.text += fmt("LAY dump=%s L=%d types=%s/%s/%s slot_order_is_score_order=%d wtap=%s wtap_vs_weighted=%.2e y_tap_vs_moe_out=%.2e y_ref_vs_moe_out=%.2e "
                  "ffn_out_vs_Y+shexp=%.2e swiglu_chk=%.2e clamped=%d dY[0..7]=", R.label.c_str(), L, R.tg.c_str(), R.tu.c_str(), R.td.c_str(),
                  (int) R.rank_identity, R.wtap.c_str(), wbe, R.y_tap_rel, R.y_ref_rel, R.ffn_id_rel, R.swiglu_chk, R.clamp_n);
    for (int s = 0; s < N_USED; ++s) R.text += fmt("%s%.3e", s ? "," : "", R.dY1[s]);
    R.text += fmt(" dY_min(slot%d)=%.3e dY_max(slot%d)=%.3e dY_all=%.3e dFFN_min=%.3e dFFN_max=%.3e dFFN_all=%.3e\n",
                  R.kmin, R.dYmin, R.kmax, R.dYmax, R.dYall, R.dFmin, R.dFmax, R.dFall);
    if (!R.err.empty()) R.text += fmt("NOTE dump=%s L=%d %s\n", R.label.c_str(), L, R.err.c_str());
    return R;
}

// ------------------------------------------------------------------ statistics
static double pct(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const double idx = p * (v.size() - 1);
    const size_t lo = (size_t) idx, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}
static void dist_row(const char * label, const std::vector<double> & v) {
    if (v.empty()) { printf("  %-34s n=0\n", label); return; }
    double s = 0, mx = 0; for (double x : v) { s += x; mx = std::max(mx, x); }
    printf("  %-34s n=%-5zu mean=%.3e  median=%.3e  p90=%.3e  p99=%.3e  max=%.3e\n", label, v.size(), s / v.size(), pct(v, 0.5), pct(v, 0.9), pct(v, 0.99), mx);
}

typedef std::function<bool(const LayerRec &)> Pred;

static void expert_table(const std::vector<LayerRec> & R, const char * title, const Pred & pred) {
    struct M { const char * name; double SlotRec::*f; };
    static const M ms[] = {
        {"expert out  rel L2 ||int8-float||/||float||", &SlotRec::e_rel}, {"expert out  1-cos", &SlotRec::e_c1m},
        {"gate        rel L2", &SlotRec::g_rel}, {"gate        1-cos", &SlotRec::g_c1m},
        {"up          rel L2", &SlotRec::u_rel}, {"up          1-cos", &SlotRec::u_c1m},
        {"h (swiglu)  rel L2", &SlotRec::h_rel},
        {"ablation: only x quantised   (out rel)", &SlotRec::xq_rel}, {"ablation: only h quantised   (out rel)", &SlotRec::hq_rel},
        {"validation: float ref vs GPU down (rel)", &SlotRec::gpu_rel},
    };
    printf("\n--- expert level, %s ---\n", title);
    for (const M & m : ms) {
        std::vector<double> v;
        for (const LayerRec & r : R) if (r.ok && pred(r)) for (int s = 0; s < N_USED; ++s) v.push_back(r.s[s].*m.f);
        dist_row(m.name, v);
    }
}

static void layer_table(const std::vector<LayerRec> & R, const char * title, const Pred & pred) {
    printf("\n--- layer level, %s: relative change of Y (MoE output) and of ffn_out = Y + shexp ---\n", title);
    printf("  (one sample per (dump, layer); 'any slot' pools the 8 single-slot cases)\n");
    for (int k = 0; k < N_USED; ++k) {
        std::vector<double> a, b;
        for (const LayerRec & r : R) if (r.ok && pred(r)) { a.push_back(r.dY1[k]); b.push_back(r.dF1[k]); }
        dist_row(fmt("dY   one slot, slot index %d", k).c_str(), a);
        dist_row(fmt("dFFN one slot, slot index %d", k).c_str(), b);
    }
    std::vector<double> a1, b1, a2, b2, a3, b3, a4, b4, a5, b5;
    for (const LayerRec & r : R) if (r.ok && pred(r)) {
        a1.push_back(r.dYmin); b1.push_back(r.dFmin); a2.push_back(r.dYmax); b2.push_back(r.dFmax);
        for (int k = 0; k < N_USED; ++k) { a3.push_back(r.dY1[k]); b3.push_back(r.dF1[k]); }
        a4.push_back(r.dYall); b4.push_back(r.dFall);
    }
    dist_row("dY   one slot, smallest |w|*|out|", a1); dist_row("dFFN one slot, smallest |w|*|out|", b1);
    dist_row("dY   one slot, largest  |w|*|out|", a2); dist_row("dFFN one slot, largest  |w|*|out|", b2);
    dist_row("dY   one slot, any (pooled)", a3);       dist_row("dFFN one slot, any (pooled)", b3);
    dist_row("dY   ALL 8 slots (llama.cpp-like)", a4); dist_row("dFFN ALL 8 slots (llama.cpp-like)", b4);
}

struct Worst { double v; int dump, L, slot, expert; std::string what; };

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const char * home = getenv("HOME"); if (!home) home = "/home/ronald";
    Cfg cfg;
    cfg.model_dir = std::string(home) + "/models/GLM-5.3-Flash/UD-IQ4_XS";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--model-dir") cfg.model_dir = next();
        else if (a == "--dump") { const std::string v = next(); const size_t p = v.find('='); if (p == std::string::npos) { fprintf(stderr, "--dump LABEL=DIR\n"); return 2; } cfg.dumps.push_back({v.substr(0, p), v.substr(p + 1)}); }
        else if (a == "--layers") { const std::string v = next(); const size_t p = v.find('-'); cfg.l0 = atoi(v.c_str()); cfg.l1 = p == std::string::npos ? cfg.l0 : atoi(v.c_str() + p + 1); }
        else if (a == "--threads") cfg.threads = std::max(1, atoi(next().c_str()));
        else if (a == "--keep-going") cfg.keep_going = true;
        else if (a == "--list-types") cfg.list_types = true;
        else { fprintf(stderr, "usage: %s [--dump LABEL=DIR]... [--layers A-B] [--threads N] [--model-dir DIR] [--keep-going] [--list-types]\n", argv[0]); return 2; }
    }
    if (cfg.dumps.empty()) {
        cfg.dumps.push_back({"g136", std::string(home) + "/bench/franken/glm5/gpu5/g136_eager"});
        cfg.dumps.push_back({"ref512", std::string(home) + "/bench/franken/glm5/ref512_keep"});
    }
    qerr_ggml_init();
    printf("=== qerr: int8-activation chain (ggml %s) vs float-activation chain on real GLM-5.3-Flash data\n", GGML_REV);
    printf("model=%s layers=%d-%d threads=%d swiglu_limit=%.1f\n", cfg.model_dir.c_str(), cfg.l0, cfg.l1, cfg.threads, SWIGLU_LIMIT);
    for (auto & d : cfg.dumps) printf("dump %s = %s\n", d.first.c_str(), d.second.c_str());

    Model M; std::string err;
    if (!M.open(cfg.model_dir, err)) { printf("ERROR: %s\n", err.c_str()); return 2; }

    // types per layer
    std::map<std::string, std::vector<int>> combos;
    printf("\n--- expert tensor types per layer ---\n");
    for (int L = cfg.l0; L <= cfg.l1; ++L) {
        Weights W;
        if (!M.find(fmt("blk.%d.ffn_gate_exps.weight", L), W.gate, err) || !M.find(fmt("blk.%d.ffn_up_exps.weight", L), W.up, err) ||
            !M.find(fmt("blk.%d.ffn_down_exps.weight", L), W.down, err)) { printf("ERROR: layer %d: %s\n", L, err.c_str()); return 2; }
        const bool ok = vec_dot_for(W.gate.type) && vec_dot_for(W.up.type) && vec_dot_for(W.down.type) && ggml_get_type_traits(W.gate.type)->to_float &&
                        ggml_get_type_traits(W.up.type)->to_float && ggml_get_type_traits(W.down.type)->to_float;
        printf("TYPES L=%d gate=%s[K=%lld,N=%lld,E=%lld] up=%s down=%s[K=%lld,N=%lld] expert_bytes=%zu%s\n", L, ggml_type_name(W.gate.type), (long long) W.gate.K, (long long) W.gate.N,
               (long long) W.gate.E, ggml_type_name(W.up.type), ggml_type_name(W.down.type), (long long) W.down.K, (long long) W.down.N,
               (size_t) W.gate.N * W.gate.row_bytes + (size_t) W.up.N * W.up.row_bytes + (size_t) W.down.N * W.down.row_bytes, ok ? "" : "  UNSUPPORTED (no Q8_K vec_dot)");
        if (!ok) { printf("ERROR: layer %d uses a type without a Q8_K vec_dot in this build\n", L); return 2; }
        if (W.up.K != W.gate.K || W.up.N != W.gate.N || W.down.K != W.gate.N || W.gate.K % 256 || W.down.K % 256) { printf("ERROR: layer %d: unexpected shapes\n", L); return 2; }
        combos[std::string(ggml_type_name(W.gate.type)) + "/" + ggml_type_name(W.up.type) + "/" + ggml_type_name(W.down.type)].push_back(L);
    }
    for (auto & kv : combos) { printf("COMBO gate/up/down=%s layers:", kv.first.c_str()); for (int L : kv.second) printf(" %d", L); printf("\n"); }
    if (cfg.list_types) { printf("=== qerr finished (rc=0)\n"); return 0; }

    // tasks, in output order: (dump, layer)
    struct Task { int dump, L; };
    std::vector<Task> tasks;
    for (size_t d = 0; d < cfg.dumps.size(); ++d) for (int L = cfg.l0; L <= cfg.l1; ++L) tasks.push_back({(int) d, L});
    std::vector<LayerRec> res(tasks.size());
    std::vector<char> done(tasks.size(), 0);
    std::atomic<size_t> nexti{0};
    std::atomic<bool> stop{false};
    std::mutex mu; size_t printed = 0;
    printf("\n--- per-sample lines (EXP = one expert slot, LAY = one layer) ---\n");
    auto worker = [&]() {
        for (;;) {
            if (stop.load()) return;
            const size_t i = nexti.fetch_add(1);
            if (i >= tasks.size()) return;
            LayerRec r = process_layer(M, cfg, tasks[i].dump, tasks[i].L);
            std::lock_guard<std::mutex> lk(mu);
            res[i] = r; done[i] = 1;
            while (printed < tasks.size() && done[printed]) {
                const LayerRec & q = res[printed];
                if (!q.ok) { printf("ERROR dump=%s L=%d: %s\n", cfg.dumps[q.dump].first.c_str(), q.L, q.err.c_str()); stop = true; }
                else {
                    fputs(q.text.c_str(), stdout);
                    if (q.fatal && !cfg.keep_going) {
                        printf("\n!!!!!!!! HARNESS VALIDATION FAILED at dump=%s L=%d: the float reference does NOT reproduce the GPU's ffn_moe_down. STOPPING. !!!!!!!!\n%s\n",
                               cfg.dumps[q.dump].first.c_str(), q.L, q.err.c_str());
                        stop = true;
                    }
                }
                ++printed;
                if (stop.load()) break;
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < cfg.threads; ++t) th.emplace_back(worker);
    worker();
    for (auto & t : th) t.join();

    // anything not ok / fatal?
    int rc = 0;
    std::vector<LayerRec> good;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!done[i]) continue;
        if (!res[i].ok) rc = 2;
        else if (res[i].fatal && !cfg.keep_going) rc = 3;
        if (res[i].ok) good.push_back(res[i]);
    }
    if (rc != 0) {
        printf("\n!!!!!!!! RUN ABORTED (rc=%d): %zu of %zu (dump, layer) samples completed; tables below are PARTIAL and not to be trusted as the full result !!!!!!!!\n", rc, good.size(), tasks.size());
    }

    // ---------------------------------------------------------------- summaries
    printf("\n================ SUMMARY (%zu (dump, layer) samples = %zu expert slots) ================\n", good.size(), good.size() * N_USED);

    // harness validation
    {
        std::vector<double> idc, gr, gg, gu, y1, y2, ffi, sw, kn; size_t ident = 0, ncl = 0; std::map<std::string, int> wt;
        for (const LayerRec & r : good) {
            for (int s = 0; s < N_USED; ++s) { idc.push_back(r.s[s].idcos1m); gr.push_back(r.s[s].gpu_rel); gg.push_back(r.s[s].gpu_g_rel); gu.push_back(r.s[s].gpu_u_rel); kn.push_back(r.s[s].kern); }
            y1.push_back(r.y_tap_rel); y2.push_back(r.y_ref_rel); ffi.push_back(r.ffn_id_rel); sw.push_back(r.swiglu_chk);
            ident += r.rank_identity; ncl += r.clamp_n; wt[r.wtap]++;
        }
        printf("\n--- harness validation (should all be tiny) ---\n");
        dist_row("slot-id: 1-cos(float ref, GPU down)", idc);
        dist_row("float ref vs GPU down, rel L2", gr);
        dist_row("float ref vs GPU gate, rel L2", gg);
        dist_row("float ref vs GPU up, rel L2", gu);
        dist_row("vec_dot vs deq(w).deq(Q8_K act), max g/u/d", kn);
        dist_row("swiglu formula vs GPU swiglu tap", sw);
        dist_row("Y(GPU down taps) vs ffn_moe_out", y1);
        dist_row("Y(float ref) vs ffn_moe_out", y2);
        dist_row("ffn_out vs ffn_moe_out+shexp", ffi);
        printf("  slot order == descending router score in %zu of %zu (dump, layer) samples\n", ident, good.size());
        printf("  elements beyond the swiglu clamp (|gate>10| or |up|>10) in the GPU taps: %zu\n", ncl);
        printf("  router weight tap that reproduces ffn_moe_weighted:"); for (auto & kv : wt) printf(" %s x%d", kv.first.c_str(), kv.second); printf("\n");
    }

    expert_table(good, "ALL", [](const LayerRec &) { return true; });
    for (size_t d = 0; d < cfg.dumps.size(); ++d) expert_table(good, fmt("dump %s", cfg.dumps[d].first.c_str()).c_str(), [d](const LayerRec & r) { return r.dump == (int) d; });
    for (auto & kv : combos) {
        const std::string key = kv.first;
        expert_table(good, fmt("types gate/up/down = %s (%zu layers)", key.c_str(), kv.second.size()).c_str(),
                     [key](const LayerRec & r) { return r.tg + "/" + r.tu + "/" + r.td == key; });
    }
    layer_table(good, "ALL", [](const LayerRec &) { return true; });
    for (size_t d = 0; d < cfg.dumps.size(); ++d) layer_table(good, fmt("dump %s", cfg.dumps[d].first.c_str()).c_str(), [d](const LayerRec & r) { return r.dump == (int) d; });
    for (auto & kv : combos) {
        const std::string key = kv.first;
        layer_table(good, fmt("types gate/up/down = %s (%zu layers)", key.c_str(), kv.second.size()).c_str(),
                    [key](const LayerRec & r) { return r.tg + "/" + r.tu + "/" + r.td == key; });
    }

    // per layer
    printf("\n--- per layer (both dumps pooled): mean/max expert out rel L2; mean dY for one slot (pooled), for all 8 ---\n");
    printf("  %-3s %-24s %-12s %-12s %-12s %-12s\n", "L", "types g/u/d", "e_rel mean", "e_rel max", "dY1 mean", "dY8 mean");
    for (int L = cfg.l0; L <= cfg.l1; ++L) {
        double sm = 0, mx = 0, d1 = 0, d8 = 0; int n = 0, nl = 0; std::string ty;
        for (const LayerRec & r : good) if (r.L == L) {
            ty = r.tg + "/" + r.tu + "/" + r.td;
            for (int s = 0; s < N_USED; ++s) { sm += r.s[s].e_rel; mx = std::max(mx, r.s[s].e_rel); d1 += r.dY1[s]; n++; }
            d8 += r.dYall; nl++;
        }
        if (n) printf("  %-3d %-24s %-12.3e %-12.3e %-12.3e %-12.3e\n", L, ty.c_str(), sm / n, mx, d1 / n, d8 / nl);
    }

    // worst 5
    auto worst = [&](const char * title, const std::function<double(const LayerRec &, int)> & f, bool per_slot) {
        std::vector<Worst> w;
        for (const LayerRec & r : good) {
            if (per_slot) for (int s = 0; s < N_USED; ++s) w.push_back({f(r, s), r.dump, r.L, s, r.s[s].expert, ""});
            else w.push_back({f(r, 0), r.dump, r.L, -1, -1, ""});
        }
        std::sort(w.begin(), w.end(), [](const Worst & a, const Worst & b) { return a.v > b.v; });
        printf("\n--- worst 5: %s ---\n", title);
        for (size_t i = 0; i < std::min<size_t>(5, w.size()); ++i) {
            if (per_slot) printf("  %.4e  dump=%s L=%d slot=%d expert=%d\n", w[i].v, cfg.dumps[w[i].dump].first.c_str(), w[i].L, w[i].slot, w[i].expert);
            else printf("  %.4e  dump=%s L=%d\n", w[i].v, cfg.dumps[w[i].dump].first.c_str(), w[i].L);
        }
    };
    worst("expert out rel L2", [](const LayerRec & r, int s) { return r.s[s].e_rel; }, true);
    worst("layer dY, one slot (any slot)", [](const LayerRec & r, int s) { return r.dY1[s]; }, true);
    worst("layer dY, ALL 8 slots", [](const LayerRec & r, int) { return r.dYall; }, false);

    printf("\n=== qerr finished (rc=%d)\n", rc);
    return rc;
}
