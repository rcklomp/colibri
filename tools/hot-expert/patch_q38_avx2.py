"""Vectorize qwen38's BF16 dense matmul (49% of decode wall time), which the
scalar path left running 2-3x under the memory bandwidth it should hit.

Measured cold (weights round-robined through a 2 GB pool, so every call is a
genuine DRAM read, not an L3 hit -- this CPU's L3 is 128 MiB, big enough to
quietly cache a hot-reused benchmark and lie about the real number):

  I=2560 O=2560   scalar 29.15 GB/s   AVX2+FMA 81.87 GB/s   (2.8x)
  I=2560 O=5120   scalar 35.99 GB/s   AVX2+FMA 90.59 GB/s   (2.5x)
  I=2560 O=10240  scalar 39.20 GB/s   AVX2+FMA 72.67 GB/s   (1.9x)

The scalar loop calls bf16_to_f32 (a shift + memcpy) once per element inside
the reduction; the compiler is not auto-vectorizing that pattern. Widening 8
bf16 lanes to f32 via a shift + FMA is a direct AVX2 translation of the exact
same math -- same rounding, only the summation ORDER changes, since 8 partial
sums accumulate in parallel lanes before the final horizontal add instead of
one running scalar total.

NOT applied to the FP8 (e4m3) expert path: an AVX2 gather-based decode of that
LUT measured SLOWER than the scalar reference on this CPU (7-16 GB/s vs the
scalar path's 15-17 GB/s) -- vpgatherdd is notoriously slow on Zen 2, so this
one is a genuine case where the straightforward vectorization is a net loss.
Left as a separate, harder problem (a shuffle-based nibble LUT rather than
gather) rather than shipping a regression.
"""
p = "/home/ronald/src/colibri/c/qwen38_core.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("""static void q38_matmul_bf16(float *y,const float *x,const uint16_t *W,
                            int S,int I,int O) {
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;float a=0.f;
            for(int i=0;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}""",
    """#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline __m256 q38_bf16x8_to_f32x8(__m128i h) {
    __m256i widened=_mm256_cvtepu16_epi32(h);
    return _mm256_castsi256_ps(_mm256_slli_epi32(widened,16));
}
#endif

static void q38_matmul_bf16(float *y,const float *x,const uint16_t *W,
                            int S,int I,int O) {
#if defined(__AVX2__) && defined(__FMA__)
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m256 vacc=_mm256_setzero_ps();int i=0;
            for(;i+8<=I;i+=8){
                __m128i wh=_mm_loadu_si128((const __m128i*)(w+i));
                __m256 wf=q38_bf16x8_to_f32x8(wh);
                __m256 xf=_mm256_loadu_ps(xs+i);
                vacc=_mm256_fmadd_ps(xf,wf,vacc);
            }
            float buf[8];_mm256_storeu_ps(buf,vacc);
            float a=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
            for(;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;float a=0.f;
            for(int i=0;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
#endif
}""",
    "AVX2 q38_matmul_bf16")

open(p, "w").write(s)
print("Q38 AVX2 BF16 APPLIED")
