// tools/hot-expert/m1/m1_common.h
//
// Small host-side helpers shared by the M1 candidates (tools/hot-expert/m1/
// m1_ggml.cpp, m1_hipfire.hip). Deterministic RNG, percentile/median timing
// stats, cosine/max-abs numerics comparison, and a key=value line printer so
// both candidates emit the same shape of record row
// (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, section M1).
//
// Pure host code -- no HIP/GPU calls here, safe to include from a .cpp or a
// .hip translation unit alike.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace m1 {

// Deterministic RNG: same seed => same weights/ids/activations across the
// GPU run and the CPU dequantised reference, and across candidates, so a
// diff in the printed check_* numbers is the kernel/layout, not the input.
inline std::mt19937& rng() {
    static std::mt19937 gen(0xC0FFEE42u);
    return gen;
}

inline float randf(float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng());
}

inline std::vector<float> randf_vec(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::vector<float> v(n);
    for (auto &x : v) x = randf(lo, hi);
    return v;
}

// n_expert_used DISTINCT experts in [0, n_expert), one draw per token --
// mirrors ggml_mul_mat_id's assumption that ids rows need not be distinct,
// but real top-k routing never repeats within one token.
inline std::vector<int32_t> random_distinct_ids(int n_expert, int n_used, int n_tokens) {
    std::vector<int32_t> out(static_cast<size_t>(n_used) * n_tokens);
    std::vector<int32_t> pool(n_expert);
    for (int i = 0; i < n_expert; i++) pool[i] = i;
    for (int t = 0; t < n_tokens; t++) {
        std::shuffle(pool.begin(), pool.end(), rng());
        for (int k = 0; k < n_used; k++) {
            out[static_cast<size_t>(t) * n_used + k] = pool[k];
        }
    }
    return out;
}

// Per-token softmax-normalised positive weights (sums to 1 per token), the
// same shape build_moe_ffn's norm_w path produces -- we skip the gating
// logits graph itself (negligible cost, out of scope per the brief) and
// synthesise the post-softmax result directly.
inline std::vector<float> random_norm_weights(int n_used, int n_tokens) {
    std::vector<float> out(static_cast<size_t>(n_used) * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        float sum = 0.0f;
        for (int k = 0; k < n_used; k++) {
            float v = std::exp(randf(-1.0f, 1.0f));
            out[static_cast<size_t>(t) * n_used + k] = v;
            sum += v;
        }
        for (int k = 0; k < n_used; k++) {
            out[static_cast<size_t>(t) * n_used + k] /= sum;
        }
    }
    return out;
}

struct TimingStats {
    double median_us = 0.0;
    double p10_us    = 0.0;
    double p90_us    = 0.0;
};

inline TimingStats stats_from_samples_us(std::vector<double> samples_us) {
    TimingStats s;
    if (samples_us.empty()) return s;
    std::sort(samples_us.begin(), samples_us.end());
    auto pick = [&](double pct) {
        size_t idx = static_cast<size_t>(pct * (samples_us.size() - 1));
        return samples_us[idx];
    };
    s.p10_us    = pick(0.10);
    s.median_us = pick(0.50);
    s.p90_us    = pick(0.90);
    return s;
}

inline double cosine_similarity(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

inline double max_abs_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return m;
}

inline double l1_norm(const float *a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += std::fabs((double)a[i]);
    return s;
}

// "label: v0 v1 v2 v3 ..." for up to the first `n` values (or fewer if the
// vector is shorter) -- a quick eyeball diagnostic, not a full dump.
inline void print_first_n(const char *label, const float *a, size_t total_n, size_t show_n = 4) {
    std::printf("%s:", label);
    size_t show = std::min(total_n, show_n);
    for (size_t i = 0; i < show; i++) std::printf(" %.6g", (double)a[i]);
    std::printf("\n");
}

inline void kv(const char *key, double value) {
    std::printf("%s=%g\n", key, value);
}
inline void kv(const char *key, long long value) {
    std::printf("%s=%lld\n", key, value);
}
inline void kv(const char *key, const char *value) {
    std::printf("%s=%s\n", key, value);
}

} // namespace m1
