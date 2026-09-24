// tools/hot-expert/franken/decode/ds4_quant_emul.cpp
//
// The bit-exactness check for ds4_quant.h's six decoders (Q4_K, Q5_K,
// IQ2_XXS, IQ2_S, IQ3_XXS, MXFP4), in the shape of ../../m1/m1_native_emul.cpp:
// host only, no HIP, no GPU, seconds. It includes the SAME header the CPU
// backend (and later the kernels) compile, and walks the lanes the way a
// wave does.
//
// GLM-5.3-Flash UD-IQ4_XS (L5 GLM step 2, decode/GLM5.md section 4) needs no
// new decoder for its 45-layer text tower -- Q8_0, IQ3_S, IQ4_XS and Q6_K are
// decode_quant.h's / m1_native_decode.h's -- so those four are checked here
// the same way, on random blocks and (--glm-model) on real GLM rows.
//
// Checks, per format, on two block populations:
//   random  blocks of random bytes with a finite, normal scale field (every
//           grid index, sign pattern, scale nibble and qh bit gets exercised);
//   real    rows read out of the DeepSeek-V4-Flash GGUF itself (--model; a
//           few rows per tensor through the mmap, a few hundred KB in all).
//
//   1. EXTRACTION, bit-exact. A one-hot activation (x[w] = 1, every other
//      entry 0) makes the sum over the 32 lanes of a block equal to weight w
//      exactly (0*w is exactly 0 for finite w). Every w of every block is
//      compared with ggml's own dequantize (ggml_get_type_traits(t)->to_float,
//      i.e. libggml-base) with == on the float bits. A wrong nibble, grid
//      entry, sign bit, scale nibble, qh bit or lane mapping fails here; a
//      position read twice reads 2w and fails; one never read reads 0 and
//      fails wherever w != 0.
//   2. ROW DOT. A random activation row through the lane loop, against a
//      float64 dot over ggml's dequantised row: a tolerance check (summation
//      order), there to catch an x index that is wrong but in range.
//
// Exit 0 = all pass. key=value lines, like the rest of the harness.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

static inline float emul_half_to_f32(unsigned int bits) {
    ggml_fp16_t h;
    const uint16_t u = (uint16_t) bits;
    std::memcpy(&h, &u, sizeof(u));
    return ggml_fp16_to_fp32(h);
}
#define FK_QUAL inline
#define FK_HALF_TO_F32(bits) emul_half_to_f32(bits)
#include "decode_quant.h"      // includes ds4_quant.h

#include "../gguf_model.h"

static_assert(sizeof(block_q4_K)    == FK_Q4K_BLOCK_BYTES,    "q4_K");
static_assert(sizeof(block_q5_K)    == FK_Q5K_BLOCK_BYTES,    "q5_K");
static_assert(sizeof(block_iq2_xxs) == FK_IQ2XXS_BLOCK_BYTES, "iq2_xxs");
static_assert(sizeof(block_iq2_s)   == FK_IQ2S_BLOCK_BYTES,   "iq2_s");
static_assert(sizeof(block_iq3_xxs) == FK_IQ3XXS_BLOCK_BYTES, "iq3_xxs");
static_assert(sizeof(block_mxfp4)   == FK_MXFP4_BLOCK_BYTES,  "mxfp4");
static_assert(sizeof(block_q8_0)    == FK_Q8_0_BLOCK_BYTES,   "q8_0");
static_assert(sizeof(block_iq3_s)   == IQ3S_BLOCK_BYTES,      "iq3_s");
static_assert(sizeof(block_iq4_xs)  == FK_IQ4XS_BLOCK_BYTES,  "iq4_xs");
static_assert(sizeof(block_q6_K)    == FK_Q6K_BLOCK_BYTES,    "q6_K");

namespace {

struct Fmt {
    const char * name;
    ggml_type    gt;
    int          fk;
    int          wpb;     // weights per block
    int          bpb;     // bytes per block
};

const Fmt FMTS[] = {
    {"q4_K",    GGML_TYPE_Q4_K,    FK_Q_Q4_K,    256, FK_Q4K_BLOCK_BYTES},
    {"q5_K",    GGML_TYPE_Q5_K,    FK_Q_Q5_K,    256, FK_Q5K_BLOCK_BYTES},
    {"iq2_xxs", GGML_TYPE_IQ2_XXS, FK_Q_IQ2_XXS, 256, FK_IQ2XXS_BLOCK_BYTES},
    {"iq2_s",   GGML_TYPE_IQ2_S,   FK_Q_IQ2_S,   256, FK_IQ2S_BLOCK_BYTES},
    {"iq3_xxs", GGML_TYPE_IQ3_XXS, FK_Q_IQ3_XXS, 256, FK_IQ3XXS_BLOCK_BYTES},
    {"mxfp4",   GGML_TYPE_MXFP4,   FK_Q_MXFP4,    32, FK_MXFP4_BLOCK_BYTES},
    // the GLM-5.3 trunk and experts (decoders that predate DeepSeek)
    {"q8_0",    GGML_TYPE_Q8_0,    FK_Q_Q8_0,     32, FK_Q8_0_BLOCK_BYTES},
    {"iq3_s",   GGML_TYPE_IQ3_S,   FK_Q_IQ3_S,   256, IQ3S_BLOCK_BYTES},
    {"iq4_xs",  GGML_TYPE_IQ4_XS,  FK_Q_IQ4_XS,  256, FK_IQ4XS_BLOCK_BYTES},
    {"q6_K",    GGML_TYPE_Q6_K,    FK_Q_Q6_K,    256, FK_Q6K_BLOCK_BYTES},
};

// One block's lane loop: exactly what decode_cpu.cpp's row_dot does per
// block (and what a wave does), returned as the float sum of the lane
// partials in lane order.
double block_dot(int fk, const unsigned char * bp, const float * xblk) {
    double a = 0.0;
    for (int tid = 0; tid < 32; ++tid) {
        switch (fk) {
            case FK_Q_Q4_K:    a += fk_q4k_block_dot(bp, xblk + 8 * tid, tid); break;
            case FK_Q_Q5_K:    a += fk_q5k_block_dot(bp, xblk + 8 * tid, tid); break;
            case FK_Q_IQ2_XXS: a += fk_iq2xxs_block_dot(bp, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs, xblk + 8 * tid, tid); break;
            case FK_Q_IQ2_S:   a += fk_iq2s_block_dot(bp, iq2s_grid, kmask_iq2xs, xblk + 8 * tid, tid); break;
            case FK_Q_IQ3_XXS: a += fk_iq3xxs_block_dot(bp, iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs, xblk + 8 * tid, tid); break;
            case FK_Q_MXFP4:   a += fk_mxfp4_lane_dot(bp, kvalues_mxfp4, xblk, tid); break;
            // decode_cpu.cpp row_dot's lane loops, verbatim
            case FK_Q_Q8_0:    a += fk_q8_0_lane_dot(bp, xblk, tid); break;
            case FK_Q_IQ4_XS:  a += fk_iq4xs_block_dot(bp, kvalues_iq4nl, xblk + 8 * tid, tid); break;
            case FK_Q_Q6_K:    a += fk_q6k_block_dot(bp, xblk + 8 * tid, tid); break;
            case FK_Q_IQ3_S: {
                const int ib32 = tid >> 2, l = tid & 3;
                a += m1n_iq3s_block_dot(bp, iq3s_grid, xblk + 8 * tid, tid, ib32, 8 - 2 * l, 7 - 2 * l);
            } break;
        }
    }
    return a;
}

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct Result { long long weights = 0, mism = 0, zero_ref = 0; double max_rel_dot = 0.0; };

// Checks 1 and 2 over `n_blocks` contiguous blocks at `data`.
Result check_blocks(const Fmt & f, const unsigned char * data, int n_blocks, std::mt19937 & rng) {
    Result r;
    const int W = f.wpb;
    std::vector<float> ref((size_t) W * n_blocks);
    ggml_get_type_traits(f.gt)->to_float(data, ref.data(), (int64_t) W * n_blocks);

    std::vector<float> x((size_t) W, 0.0f);
    int shown = 0;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char * bp = data + (size_t) b * f.bpb;
        for (int w = 0; w < W; ++w) {
            x[w] = 1.0f;
            const float got = (float) block_dot(f.fk, bp, x.data());
            x[w] = 0.0f;
            const float want = ref[(size_t) b * W + w];
            ++r.weights;
            if (want == 0.0f) ++r.zero_ref;
            if (fbits(got) != fbits(want) && !(got == 0.0f && want == 0.0f)) {
                ++r.mism;
                if (shown++ < 5)
                    std::printf("  MISMATCH %s block %d weight %d: got %.9g (0x%08x) want %.9g (0x%08x)\n",
                                f.name, b, w, got, fbits(got), want, fbits(want));
            }
        }
    }
    // check 2: random activations, one row = all blocks
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xr((size_t) W * n_blocks);
    for (auto & v : xr) v = nd(rng);
    double got = 0.0, want = 0.0, l1 = 0.0;
    for (int b = 0; b < n_blocks; ++b)
        got += block_dot(f.fk, data + (size_t) b * f.bpb, xr.data() + (size_t) b * W);
    for (size_t i = 0; i < xr.size(); ++i) { want += (double) ref[i] * xr[i]; l1 += std::fabs((double) ref[i] * xr[i]); }
    r.max_rel_dot = l1 > 0 ? std::fabs(got - want) / l1 : 0.0;
    return r;
}

// Random blocks with a sane scale field: the payload bytes are uniform, the
// f16 d (and dmin) are a normal f16 in [2^-10, 2^-4), the E8M0 exponent in
// [110, 140].
std::vector<unsigned char> random_blocks(const Fmt & f, int n, std::mt19937 & rng) {
    std::vector<unsigned char> v((size_t) n * f.bpb);
    for (auto & c : v) c = (unsigned char) (rng() & 0xFF);
    std::uniform_real_distribution<float> ud(1.0f / 1024.0f, 1.0f / 16.0f);
    for (int b = 0; b < n; ++b) {
        unsigned char * bp = v.data() + (size_t) b * f.bpb;
        auto put_half = [&](unsigned char * p) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(ud(rng));
            std::memcpy(p, &h, 2);
        };
        switch (f.fk) {
            case FK_Q_Q4_K: case FK_Q_Q5_K: put_half(bp); put_half(bp + 2); break;
            case FK_Q_MXFP4: bp[0] = (unsigned char) (110 + rng() % 31); break;
            case FK_Q_Q6_K:  put_half(bp + FK_Q6K_OFF_D); break;
            default: put_half(bp); break;
        }
    }
    return v;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model, glm_model;
    int n_random = 512, rows_real = 4;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model = argv[++i];
        else if (a == "--glm-model" && i + 1 < argc) glm_model = argv[++i];
        else if (a == "--random" && i + 1 < argc) n_random = std::atoi(argv[++i]);
        else if (a == "--rows" && i + 1 < argc) rows_real = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s [--model shard.gguf] [--glm-model shard.gguf] [--random N] [--rows R]\n", argv[0]); return 2; }
    }
    std::mt19937 rng(0xD5E4C0DEu);
    bool ok = true;

    for (const Fmt & f : FMTS) {
        auto blocks = random_blocks(f, n_random, rng);
        const Result r = check_blocks(f, blocks.data(), n_random, rng);
        const bool pass = r.mism == 0 && r.max_rel_dot < 1e-5;
        ok &= pass;
        std::printf("emul fmt=%s set=random blocks=%d weights=%lld mismatches=%lld zero_ref=%lld rowdot_rel=%.3g %s\n",
                    f.name, n_random, r.weights, r.mism, r.zero_ref, r.max_rel_dot, pass ? "PASS" : "FAIL");
    }

    struct Real { const char * tensor; int expert; };
    auto check_reals = [&](const std::string & path, const std::vector<Real> & reals) {
        auto gm = franken::GgufModel::open(path);
        for (const Real & re : reals) {
            const franken::TensorInfo * t = gm->find(re.tensor);
            if (!t) { std::printf("emul real %s MISSING\n", re.tensor); ok = false; continue; }
            const Fmt * f = nullptr;
            for (const Fmt & c : FMTS) if (c.gt == t->type) f = &c;
            if (!f) { std::printf("emul real %s type %s not in the table\n", re.tensor, ggml_type_name(t->type)); ok = false; continue; }
            const size_t row = t->row_size();
            const size_t slice = re.expert >= 0 ? t->slice_bytes(t->ne2()) : 0;
            const unsigned char * base = t->data + (re.expert >= 0 ? (size_t) re.expert * slice : 0);
            const int blocks_per_row = (int) (t->ne0() / f->wpb);
            // rows spread over the matrix, not just the first ones
            Result tot;
            for (int k = 0; k < rows_real; ++k) {
                const int64_t r = (t->ne1() - 1) * k / std::max(1, rows_real - 1);
                std::vector<unsigned char> buf(base + (size_t) r * row, base + (size_t) (r + 1) * row);
                const Result rr = check_blocks(*f, buf.data(), blocks_per_row, rng);
                tot.weights += rr.weights; tot.mism += rr.mism; tot.zero_ref += rr.zero_ref;
                tot.max_rel_dot = std::max(tot.max_rel_dot, rr.max_rel_dot);
            }
            const bool pass = tot.mism == 0 && tot.max_rel_dot < 1e-5;
            ok &= pass;
            std::printf("emul fmt=%s set=real tensor=%s expert=%d rows=%d weights=%lld mismatches=%lld zero_ref=%lld rowdot_rel=%.3g %s\n",
                        f->name, re.tensor, re.expert, rows_real, tot.weights, tot.mism, tot.zero_ref,
                        tot.max_rel_dot, pass ? "PASS" : "FAIL");
        }
    };
    if (!model.empty()) {
        // One real tensor per format, as the DeepSeek-V4 file carries it
        // (DEEPSEEK4.md section 4). A few rows each: a few hundred KB.
        check_reals(model, {
            {"output.weight", -1},                    // Q4_K
            {"blk.0.attn_q_a.weight", -1},            // Q5_K
            {"blk.3.ffn_gate_exps.weight", 17},       // IQ2_XXS
            {"blk.26.ffn_up_exps.weight", 200},       // IQ2_S
            {"blk.3.ffn_down_exps.weight", 17},       // IQ3_XXS
            {"blk.42.ffn_down_exps.weight", 5},       // MXFP4
        });
    }
    if (!glm_model.empty()) {
        // Every format the GLM-5.3-Flash text tower carries (GLM5.md section 4).
        check_reals(glm_model, {
            {"token_embd.weight", -1},                // Q8_0 (the host gather's format)
            {"blk.3.attn_q_b.weight", -1},            // Q8_0 (trunk)
            {"blk.3.ffn_gate_exps.weight", 17},       // IQ3_S (41 layers)
            {"blk.11.ffn_up_exps.weight", 200},       // IQ4_XS (gate/up, layer 11)
            {"blk.3.ffn_down_exps.weight", 17},       // IQ4_XS (down, 39 layers)
            {"blk.12.ffn_down_exps.weight", 5},       // Q6_K (down, layers 11, 12, 44)
            {"output.weight", -1},                    // Q6_K (lm_head)
        });
    }
    std::printf("emul verdict=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
