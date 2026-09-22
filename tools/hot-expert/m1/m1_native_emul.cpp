// tools/hot-expert/m1/m1_native_emul.cpp
//
// A host-only emulation of m1_native.hip's two GEMV kernels, to prove the
// IQ3_S / IQ4_NL decode and the lane-to-weight mapping WITHOUT a GPU and
// without the rig lock. Same role m3_emul_check.cpp plays for M3: it must
// never stand between a kernel edit and a build of the real binary, so it is
// not part of `all` and not a prerequisite of m1_native.
//
// It is not an emulation in the weak sense: the decode itself is
// m1_native_decode.h, the SAME source the kernels compile, included here
// with host qualifiers. What this file emulates is only the wave: the lane
// loop, the block/chunk partition and the __shfl_down butterfly.
//
// Three checks, in increasing strength:
//
//   1. WEIGHT EXTRACTION, bit-exact. Calling the shared per-lane decode with
//      a one-hot activation vector returns exactly one dequantised weight
//      (every other term is multiplied by 0.0f and 0.0f*w is exactly 0 for
//      finite w). Sweeping the one-hot position over a whole row therefore
//      reconstructs the row the kernel would see, weight by weight, and it
//      is compared to ggml's own dequantize_row_iq3_s / dequantize_row_iq4_nl
//      via ggml_get_type_traits(type)->to_float with EXACT float equality --
//      not a tolerance. A single wrong nibble, grid bit, sign bit or scale
//      nibble fails here, which a cosine over a 2560-long dot product can
//      hide. This is the check that says "bit-exactly ported".
//
//   2. COVERAGE. The lane partition must hit every weight of a row exactly
//      once. Check 1 proves each lane/slot maps to the right weight; it does
//      not prove the kernel's loops enumerate every (lane, block) pair. So
//      the extraction sweep also tallies how many times each position was
//      produced and demands a histogram of all ones.
//
//   3. FULL ROW DOT. The whole wave is run against a random fp32 activation
//      row, lane accumulators folded with the same __shfl_down butterfly the
//      kernels use, and compared to a float64 reference over ggml's
//      dequantised row. f32-vs-f64 summation order makes this a tolerance
//      check, not a bit check -- it is here to catch an error that check 1
//      and 2 could both pass, e.g. a wrong x index that still reads in range.
//
// Exit code 0 = all pass, 1 = a check failed. Prints key=value lines in the
// same shape as the rest of the M1 harness.
//
// No HIP, no GPU, no ggml backend: only ggml_quantize_chunk,
// ggml_get_type_traits()->to_float, ggml_row_size and ggml_fp16_to_fp32.

#include "ggml.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

// Host build of the shared decode. ggml_fp16_to_fp32 is the same IEEE
// conversion of the same 16 bits that the device's
// __half2float(__ushort_as_half(...)) performs -- every f16 is exactly
// representable in f32, so the two cannot differ.
#define M1N_QUAL inline
#define M1N_HALF_TO_F32(bits) ggml_fp16_to_fp32((ggml_fp16_t)(bits))
#include "m1_native_decode.h"

#include "m1_common.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static_assert(sizeof(block_iq3_s)  == IQ3S_BLOCK_BYTES,  "block_iq3_s size moved");
static_assert(sizeof(block_iq4_nl) == IQ4NL_BLOCK_BYTES, "block_iq4_nl size moved");
static_assert(offsetof(block_iq3_s, qs)     == IQ3S_OFF_QS,     "iq3_s qs offset moved");
static_assert(offsetof(block_iq3_s, qh)     == IQ3S_OFF_QH,     "iq3_s qh offset moved");
static_assert(offsetof(block_iq3_s, signs)  == IQ3S_OFF_SIGNS,  "iq3_s signs offset moved");
static_assert(offsetof(block_iq3_s, scales) == IQ3S_OFF_SCALES, "iq3_s scales offset moved");
static_assert(offsetof(block_iq4_nl, qs)    == IQ4NL_OFF_QS,    "iq4_nl qs offset moved");

namespace {

constexpr int D_MODEL = 2560;   // gate/up K
constexpr int N_FF    = 640;    // down K
constexpr int NROWS   = 8;      // rows of each kind to check

// The kernels' cross-lane fold: acc += __shfl_down(acc, 16/8/4/2/1), which
// leaves the wave total in lane 0. Reproduced exactly (same pairing, same
// order) so check 3 compares like with like.
float shfl_down_fold(float lane[32])
{
    float v[32];
    std::memcpy(v, lane, sizeof(v));
    for (int off = 16; off > 0; off >>= 1) {
        float nv[32];
        for (int i = 0; i < 32; i++) {
            // __shfl_down past the end of the wave returns the lane's own
            // value on AMD; lanes >= 32-off are dead for the final result
            // either way, so only lane 0's path matters.
            const int src = i + off;
            nv[i] = v[i] + (src < 32 ? v[src] : v[i]);
        }
        std::memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

// ---- IQ3_S: reconstruct a row through the shared per-lane decode. ----
void iq3s_extract_row(const unsigned char *row_ptr, int K,
                      std::vector<float> &out, std::vector<int> &hits)
{
    const int nblk = K >> 8;
    out.assign(K, 0.0f);
    hits.assign(K, 0);
    float onehot[8];
    for (int b = 0; b < nblk; b++) {
        const unsigned char *bp = row_ptr + (size_t)b * IQ3S_BLOCK_BYTES;
        for (int tid = 0; tid < 32; tid++) {
            const int ib32 = tid >> 2;
            const int l    = tid & 3;
            const int sh1  = 8 - 2 * l;
            const int sh2  = 7 - 2 * l;
            for (int j = 0; j < 8; j++) {
                for (int q = 0; q < 8; q++) onehot[q] = (q == j) ? 1.0f : 0.0f;
                const float w = m1n_iq3s_block_dot(bp, iq3s_grid, onehot, tid, ib32, sh1, sh2);
                const int pos = (b << 8) + (tid << 3) + j;
                out[pos] = w;
                hits[pos]++;
            }
        }
    }
}

// ---- IQ4_NL: the same, through the shared per-chunk decode. ----
void iq4nl_extract_row(const unsigned char *row_ptr, int K,
                       std::vector<float> &out, std::vector<int> &hits)
{
    const int nblk   = K >> 5;
    const int nchunk = nblk << 2;
    out.assign(K, 0.0f);
    hits.assign(K, 0);
    std::vector<float> onehot(K, 0.0f);
    for (int w = 0; w < nchunk; w++) {
        const int blk = w >> 2;
        const int off = (w & 3) << 2;
        for (int c = 0; c < 4; c++) {
            const int pos_lo = (blk << 5) + off + c;
            const int pos_hi = pos_lo + 16;
            for (int p : {pos_lo, pos_hi}) {
                onehot[p] = 1.0f;
                float lo = 0.0f, hi = 0.0f;
                m1n_iq4nl_chunk_dot(row_ptr, kvalues_iq4nl, onehot.data(), w, &lo, &hi);
                onehot[p] = 0.0f;
                out[p] = lo + hi;   // exactly one of the two is nonzero
                hits[p]++;
            }
        }
    }
}

// ---- Full-wave row dot, exactly as the kernels loop. ----
float iq3s_row_dot(const unsigned char *row_ptr, int K, const float *x)
{
    const int nblk = K >> 8;
    float lane[32];
    for (int tid = 0; tid < 32; tid++) {
        const int ib32 = tid >> 2;
        const int l    = tid & 3;
        const int sh1  = 8 - 2 * l;
        const int sh2  = 7 - 2 * l;
        const int xoff = tid << 3;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        int b = 0;
        for (; b + 3 < nblk; b += 4) {
            a0 += m1n_iq3s_block_dot(row_ptr + (size_t)(b+0) * IQ3S_BLOCK_BYTES, iq3s_grid, x + ((b+0) << 8) + xoff, tid, ib32, sh1, sh2);
            a1 += m1n_iq3s_block_dot(row_ptr + (size_t)(b+1) * IQ3S_BLOCK_BYTES, iq3s_grid, x + ((b+1) << 8) + xoff, tid, ib32, sh1, sh2);
            a2 += m1n_iq3s_block_dot(row_ptr + (size_t)(b+2) * IQ3S_BLOCK_BYTES, iq3s_grid, x + ((b+2) << 8) + xoff, tid, ib32, sh1, sh2);
            a3 += m1n_iq3s_block_dot(row_ptr + (size_t)(b+3) * IQ3S_BLOCK_BYTES, iq3s_grid, x + ((b+3) << 8) + xoff, tid, ib32, sh1, sh2);
        }
        for (; b < nblk; b++) {
            a0 += m1n_iq3s_block_dot(row_ptr + (size_t)b * IQ3S_BLOCK_BYTES, iq3s_grid, x + (b << 8) + xoff, tid, ib32, sh1, sh2);
        }
        lane[tid] = (a0 + a1) + (a2 + a3);
    }
    return shfl_down_fold(lane);
}

float iq4nl_row_dot(const unsigned char *row_ptr, int K, const float *h)
{
    const int nchunk = (K >> 5) << 2;
    float lane[32];
    for (int tid = 0; tid < 32; tid++) {
        float lo = 0.0f, hi = 0.0f;
        for (int w = tid; w < nchunk; w += 32) {
            m1n_iq4nl_chunk_dot(row_ptr, kvalues_iq4nl, h, w, &lo, &hi);
        }
        lane[tid] = lo + hi;
    }
    return shfl_down_fold(lane);
}

int check_format(const char *name, ggml_type type, int K, int nrows)
{
    const size_t row_bytes = ggml_row_size(type, K);
    std::vector<float> src = m1::randf_vec((size_t)nrows * K, -1.0f, 1.0f);
    std::vector<uint8_t> q(row_bytes * nrows);
    const size_t written = ggml_quantize_chunk(type, src.data(), q.data(), 0, nrows, K, nullptr);
    if (written != q.size()) {
        std::fprintf(stderr, "%s: quantize_chunk wrote %zu, expected %zu\n", name, written, q.size());
        return 1;
    }

    const ggml_type_traits *tt = ggml_get_type_traits(type);
    std::vector<float> ref(K), got, x = m1::randf_vec(K, -1.0f, 1.0f);
    std::vector<int> hits;

    long long exact_mismatch = 0, coverage_bad = 0;
    double worst_rel_dot = 0.0;
    double ref_l1_total = 0.0;

    for (int r = 0; r < nrows; r++) {
        const unsigned char *row_ptr = q.data() + (size_t)r * row_bytes;
        tt->to_float(row_ptr, ref.data(), K);
        ref_l1_total += m1::l1_norm(ref.data(), K);

        if (type == GGML_TYPE_IQ3_S) iq3s_extract_row(row_ptr, K, got, hits);
        else                          iq4nl_extract_row(row_ptr, K, got, hits);

        for (int i = 0; i < K; i++) {
            // EXACT: same bits, not a tolerance.
            if (got[i] != ref[i]) exact_mismatch++;
            if (hits[i] != 1)     coverage_bad++;
        }

        double dref = 0.0;
        for (int i = 0; i < K; i++) dref += (double)ref[i] * x[i];
        const float dgot = (type == GGML_TYPE_IQ3_S) ? iq3s_row_dot(row_ptr, K, x.data())
                                                     : iq4nl_row_dot(row_ptr, K, x.data());
        const double denom = std::fabs(dref) > 1e-6 ? std::fabs(dref) : 1e-6;
        const double rel = std::fabs((double)dgot - dref) / denom;
        if (rel > worst_rel_dot) worst_rel_dot = rel;
    }

    std::printf("emul_%s_rows=%d\n", name, nrows);
    std::printf("emul_%s_ref_l1=%.6g\n", name, ref_l1_total);
    std::printf("emul_%s_exact_mismatches=%lld\n", name, exact_mismatch);
    std::printf("emul_%s_coverage_bad=%lld\n", name, coverage_bad);
    std::printf("emul_%s_worst_rel_dot=%.6g\n", name, worst_rel_dot);

    int rc = 0;
    // A row that dequantises to all zeros would pass "exact" trivially --
    // the same trap m1_ggml.cpp's all-zero oracle fell into on 2026-09-22.
    if (ref_l1_total == 0.0) {
        std::fprintf(stderr, "%s: CHECK_FAIL reference is all zero\n", name);
        rc = 1;
    }
    if (exact_mismatch != 0) {
        std::fprintf(stderr, "%s: CHECK_FAIL %lld weights differ from ggml's to_float\n", name, exact_mismatch);
        rc = 1;
    }
    if (coverage_bad != 0) {
        std::fprintf(stderr, "%s: CHECK_FAIL %lld positions not covered exactly once\n", name, coverage_bad);
        rc = 1;
    }
    if (!(worst_rel_dot < 1e-4)) {
        std::fprintf(stderr, "%s: CHECK_FAIL wave row dot relative error %.6g\n", name, worst_rel_dot);
        rc = 1;
    }
    return rc;
}

} // namespace

int main()
{
    int rc = 0;
    for (int i = 0; i < 8; i++) {
        if (kmask_iq2xs[i] != (uint8_t)(1u << i)) {
            std::fprintf(stderr, "kmask_iq2xs is not 1<<m; the sign decode shortcut is invalid\n");
            return 1;
        }
    }
    std::printf("emul_kmask_iq2xs_is_bitm=1\n");

    rc |= check_format("iq3s",  GGML_TYPE_IQ3_S,  D_MODEL, NROWS);
    rc |= check_format("iq4nl", GGML_TYPE_IQ4_NL, N_FF,    NROWS);

    std::printf("emul_result=%s\n", rc == 0 ? "PASS" : "CHECK_FAIL");
    return rc;
}
