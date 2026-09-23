// tools/hot-expert/franken/decode/moe_gather_test.cpp
//
// Does design 9.4 item 5's expert row-gather compute the same BITS as the
// per-assignment kernels it replaces?
//
// On 2026-09-22 the GPU said no: with the gather on, a chunk of 6 diverged
// from six single-token steps from layer 1 onwards. The bisection narrowed it
// to the DOWN stage (gate/up is exact), which leaves a question no GPU run
// can answer cheaply and no CPU gate could reach at all -- the gather only
// exists at T > 1 and only on the device.
//
// So this is the question asked on the host, where it can be answered in a
// second and kept: both paths are transcribed here from decode_gpu.hip,
// EMULATING A 32-LANE WAVE, over the same shared primitives from
// decode_quant.h / m1_native_decode.h that the kernels call. That is the
// point -- the primitives are not re-derived, so a disagreement can only be
// the surrounding loop, the indexing, the accumulator structure or the
// reduction, which is exactly the class of difference being hunted.
//
// Links no HIP. `make moe-gather-test` builds and runs it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) fk_test_half_to_f32(bits)
static inline float fk_test_half_to_f32(unsigned int bits);

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "decode_quant.h"

static inline float fk_test_half_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    const uint16_t u = (uint16_t) bits;
    std::memcpy(&h, &u, sizeof(u));
    return ggml_fp16_to_fp32(h);
}

using namespace fk;

static constexpr int WAVE       = 32;
static constexpr int MOE_TILE_E = 8;

// decode_gpu.hip's wave_sum, exactly: a butterfly over __shfl_xor, so the
// association is the kernel's and not a left-to-right sum.
static float wave_sum(const float * v_in) {
    float v[WAVE];
    for (int i = 0; i < WAVE; ++i) v[i] = v_in[i];
    for (int off = WAVE / 2; off > 0; off >>= 1) {
        float n[WAVE];
        for (int i = 0; i < WAVE; ++i) n[i] = v[i] + v[i ^ off];
        for (int i = 0; i < WAVE; ++i) v[i] = n[i];
    }
    return v[0];
}

static bool same_bits(float a, float b) {
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    return ua == ub;
}
static uint32_t bits(float a) { uint32_t u; std::memcpy(&u, &a, 4); return u; }

// ------------------------------------------------------- the counting sort --
//
// k_moe_sort, transcribed. Proven correct on the device by the gate/up arm
// (both stages share it), and kept here so a future change to it is caught
// without a GPU.
struct Tiles {
    std::vector<int> order, tile_exp, tile_off, tile_cnt;
    int n_tiles_max = 0;
};
static Tiles moe_sort(const std::vector<int> & ids, int n_expert) {
    const int n = (int) ids.size();
    Tiles T;
    T.n_tiles_max = (n < n_expert ? n : n_expert) + (n + MOE_TILE_E - 1) / MOE_TILE_E;
    T.order.assign(n, -1);
    T.tile_exp.assign(T.n_tiles_max, -1);
    T.tile_off.assign(T.n_tiles_max, -1);
    T.tile_cnt.assign(T.n_tiles_max, 0);

    std::vector<int> hist(n_expert, 0);
    for (int i = 0; i < n; ++i) ++hist[ids[i]];
    std::vector<int> base(n_expert), tbase(n_expert);
    int s = 0, ts = 0;
    for (int e = 0; e < n_expert; ++e) {
        s  += hist[e];                 base[e]  = s;
        ts += (hist[e] + MOE_TILE_E - 1) / MOE_TILE_E; tbase[e] = ts;
    }
    std::vector<int> cur(n_expert);
    for (int e = 0; e < n_expert; ++e) cur[e] = base[e] - hist[e];
    for (int i = 0; i < n; ++i) T.order[cur[ids[i]]++] = i;
    for (int e = 0; e < n_expert; ++e) {
        const int c = hist[e];
        const int tc = (c + MOE_TILE_E - 1) / MOE_TILE_E;
        const int t0 = tbase[e] - tc, start = base[e] - c;
        for (int k = 0; k * MOE_TILE_E < c; ++k) {
            const int idx = t0 + k;
            if (idx >= T.n_tiles_max) break;
            T.tile_exp[idx] = e;
            T.tile_off[idx] = start + k * MOE_TILE_E;
            T.tile_cnt[idx] = std::min(MOE_TILE_E, c - k * MOE_TILE_E);
        }
    }
    return T;
}

static bool check_sort(const std::vector<int> & ids, const Tiles & T) {
    const int n = (int) ids.size();
    std::vector<int> seen(n, 0);
    int covered = 0;
    for (int t = 0; t < T.n_tiles_max; ++t) {
        if (T.tile_cnt[t] <= 0) continue;
        for (int m = 0; m < T.tile_cnt[t]; ++m) {
            const int col = T.order[T.tile_off[t] + m];
            if (col < 0 || col >= n) { std::printf("SORT: tile %d slot %d out of range\n", t, m); return false; }
            if (seen[col]++) { std::printf("SORT: column %d covered twice\n", col); return false; }
            if (ids[col] != T.tile_exp[t]) {
                std::printf("SORT: column %d has expert %d but tile %d says %d\n",
                            col, ids[col], t, T.tile_exp[t]);
                return false;
            }
            ++covered;
        }
    }
    if (covered != n) { std::printf("SORT: %d of %d columns covered\n", covered, n); return false; }
    return true;
}

// ------------------------------------------------ the two DOWN code paths --
//
// Both transcribed from decode_gpu.hip. `A` is one output row of one expert.

// k_moe_down_iq4nl, per assignment.
static float down_iq4nl_ref(const unsigned char * A, const float * h_row, int K) {
    const int nchunk = (K >> 5) << 2;
    float part[WAVE];
    for (int tid = 0; tid < WAVE; ++tid) {
        float lo = 0.0f, hi = 0.0f;
        for (int w = tid; w < nchunk; w += WAVE)
            m1n_iq4nl_chunk_dot(A, kvalues_iq4nl, h_row, w, &lo, &hi);
        part[tid] = lo + hi;
    }
    return wave_sum(part);
}

// k_moe_down_iq4nl_gather, one tile. Since the ISA finding of 2026-09-23 the
// m loop is OUTER and the accumulators are scalars, so this body is literally
// the body of down_iq4nl_ref with a different h row -- which is what makes the
// device emit the same float ops for both. This transcription tracks that
// shape, not the old tiled one.
static void down_iq4nl_gather(const unsigned char * A, const float * h, const int * xb,
                              int cnt, int K, float * out) {
    const int nchunk = (K >> 5) << 2;
    float part[MOE_TILE_E][WAVE];
    for (int m = 0; m < cnt; ++m) {
        const float * hr = h + xb[m];
        for (int tid = 0; tid < WAVE; ++tid) {
            float lo = 0.0f, hi = 0.0f;
            for (int w = tid; w < nchunk; w += WAVE)
                m1n_iq4nl_chunk_dot(A, kvalues_iq4nl, hr, w, &lo, &hi);
            part[m][tid] = lo + hi;
        }
    }
    for (int m = 0; m < cnt; ++m) out[m] = wave_sum(part[m]);
}

// k_moe_down_q8_0, per assignment.
static float down_q8_0_ref(const unsigned char * A, const float * h_row, int K) {
    const int nblk = K / FK_Q8_0_BLOCK_WEIGHTS;
    float part[WAVE];
    for (int tid = 0; tid < WAVE; ++tid) {
        float acc = 0.0f;
        for (int b = 0; b < nblk; ++b)
            acc += fk_q8_0_lane_dot(A + (size_t) b * FK_Q8_0_BLOCK_BYTES,
                                    h_row + b * FK_Q8_0_BLOCK_WEIGHTS, tid);
        part[tid] = acc;
    }
    return wave_sum(part);
}

// k_moe_down_q8_0_gather, one tile. Since 2026-09-22 its inner statement is
// textually the per-assignment kernel's; this transcription tracks it.
static void down_q8_0_gather(const unsigned char * A, const float * h, const int * xb,
                             int cnt, int K, float * out) {
    const int nblk = K / FK_Q8_0_BLOCK_WEIGHTS;
    float part[MOE_TILE_E][WAVE];
    for (int tid = 0; tid < WAVE; ++tid) {
        float acc[MOE_TILE_E];
        for (int m = 0; m < MOE_TILE_E; ++m) acc[m] = 0.0f;
        for (int b = 0; b < nblk; ++b)
            for (int m = 0; m < MOE_TILE_E; ++m)
                if (m < cnt) acc[m] += fk_q8_0_lane_dot(A + (size_t) b * FK_Q8_0_BLOCK_BYTES,
                                                        h + xb[m] + b * FK_Q8_0_BLOCK_WEIGHTS, tid);
        for (int m = 0; m < cnt; ++m) part[m][tid] = acc[m];
    }
    for (int m = 0; m < cnt; ++m) out[m] = wave_sum(part[m]);
}

int main() {
    std::mt19937 rng(0xD08E4A17u);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);

    const int K = N_FF_EXP;                 // 640, the down stage's reduction
    const int T = 6, K_TOP = N_EXPERT_USED; // the chunk the GPU oracle runs
    const int n = T * K_TOP;

    // a plausible assignment list: ten DISTINCT experts a token, repeating
    // across tokens, which is the whole reason the gather exists
    std::vector<int> ids(n);
    for (int t = 0; t < T; ++t) {
        std::vector<int> pick;
        while ((int) pick.size() < K_TOP) {
            const int e = (int)(rng() % 64);     // a small pool, so tiles fill
            bool dup = false;
            for (int p : pick) dup |= (p == e);
            if (!dup) pick.push_back(e);
        }
        for (int k = 0; k < K_TOP; ++k) ids[t * K_TOP + k] = pick[k];
    }

    const Tiles tiles = moe_sort(ids, N_EXPERT);
    if (!check_sort(ids, tiles)) { std::printf("MOE-GATHER FAIL (sort)\n"); return 1; }
    std::printf("sort ok: %d assignments, %d tiles used of %d\n", n,
                (int) std::count_if(tiles.tile_cnt.begin(), tiles.tile_cnt.end(),
                                    [](int c) { return c > 0; }),
                tiles.n_tiles_max);

    // one output row of one expert, in each format, plus the activations
    std::vector<unsigned char> A_nl((size_t)(K / IQ4NL_BLOCK_WEIGHTS) * IQ4NL_BLOCK_BYTES);
    std::vector<unsigned char> A_q8((size_t)(K / FK_Q8_0_BLOCK_WEIGHTS) * FK_Q8_0_BLOCK_BYTES);
    for (auto & b : A_nl) b = (unsigned char)(rng() & 0xff);
    for (auto & b : A_q8) b = (unsigned char)(rng() & 0xff);
    // keep the f16 scales finite and ordinary
    for (size_t b = 0; b < A_nl.size(); b += IQ4NL_BLOCK_BYTES) {
        const uint16_t d = 0x3400 | (uint16_t)(rng() & 0xff);
        std::memcpy(&A_nl[b + IQ4NL_OFF_D], &d, 2);
    }
    for (size_t b = 0; b < A_q8.size(); b += FK_Q8_0_BLOCK_BYTES) {
        const uint16_t d = 0x3400 | (uint16_t)(rng() & 0xff);
        std::memcpy(&A_q8[b + FK_Q8_0_OFF_D], &d, 2);
    }
    std::vector<float> h((size_t) n * K);
    for (auto & v : h) v = uf(rng);

    int bad_nl = 0, bad_q8 = 0, compared = 0;
    for (int t = 0; t < tiles.n_tiles_max; ++t) {
        const int cnt = tiles.tile_cnt[t];
        if (cnt <= 0) continue;
        int cols[MOE_TILE_E] = {0}, xb[MOE_TILE_E] = {0};
        for (int m = 0; m < MOE_TILE_E; ++m) {
            cols[m] = (m < cnt) ? tiles.order[tiles.tile_off[t] + m] : 0;
            xb[m]   = cols[m] * K;
        }
        float g_nl[MOE_TILE_E], g_q8[MOE_TILE_E];
        down_iq4nl_gather(A_nl.data(), h.data(), xb, cnt, K, g_nl);
        down_q8_0_gather (A_q8.data(), h.data(), xb, cnt, K, g_q8);
        for (int m = 0; m < cnt; ++m) {
            const float r_nl = down_iq4nl_ref(A_nl.data(), h.data() + (size_t) cols[m] * K, K);
            const float r_q8 = down_q8_0_ref (A_q8.data(), h.data() + (size_t) cols[m] * K, K);
            ++compared;
            if (!same_bits(r_nl, g_nl[m])) {
                if (bad_nl++ < 3)
                    std::printf("IQ4_NL col %d (tile %d slot %d): ref %.9g [%08x] gather %.9g [%08x]\n",
                                cols[m], t, m, r_nl, bits(r_nl), g_nl[m], bits(g_nl[m]));
            }
            if (!same_bits(r_q8, g_q8[m])) {
                if (bad_q8++ < 3)
                    std::printf("Q8_0   col %d (tile %d slot %d): ref %.9g [%08x] gather %.9g [%08x]\n",
                                cols[m], t, m, r_q8, bits(r_q8), g_q8[m], bits(g_q8[m]));
            }
        }
    }
    std::printf("compared=%d iq4nl_mismatch=%d q8_0_mismatch=%d\n", compared, bad_nl, bad_q8);
    const bool ok = (bad_nl == 0 && bad_q8 == 0 && compared == n);
    std::printf("MOE-GATHER %s\n", ok ? "PASS (bit-identical)" : "FAIL");
    return ok ? 0 : 1;
}
