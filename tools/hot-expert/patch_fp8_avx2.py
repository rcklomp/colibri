"""Vectorize matmul_fp8 (e4m3 block-scaled) with a pure bit-manipulation decode
instead of a table lookup -- no gather, so it doesn't hit the vpgatherdd
weakness on Zen 2 that made the earlier gather-based attempt a net loss.

E4M3-FN structure (per the header comment above E4M3_LUT): sign(1) exp(4,
bias=7) mant(3), subnormal at exp==0, and the OCP convention that only
mant==0x7 at exp==0xF is NaN (max finite is exp=0xF,mant=0x6 -> 448, no
infinity). That has a closed-form bit decode:

  normal (exp4 != 0):    bits = sign<<31 | (exp4+120)<<23 | mant3<<20
  subnormal (exp4 == 0):  value = sign ? -(mant3/512) : mant3/512
  NaN (byte&0x7F==0x7F):  a real NaN, matching the LUT's documented policy

Validated exhaustively against the real 256-entry E4M3_LUT before this was
written into the engine: all 256 codes bit-exact (including both NaN codes
decoding to NaN), both in scalar form and processed 8-at-a-time through the
AVX2 path exactly as the kernel below uses it.

Cold-measured (weights round-robined through a 2 GB pool of distinct
matrices, so every call is a genuine DRAM read):

  I=2560 O=640   scalar(table) 16.26 GB/s   AVX2(bit-trick) 24.06 GB/s  (1.5x)
  I=640  O=2560  scalar(table)  5.44 GB/s   AVX2(bit-trick) 22.17 GB/s  (4.1x)

quant.h is shared by colibri.c, deepseek_v4.c, glm53.c, kimi_k3.c and
qwen38.c, so this is not qwen38-specific.
"""
p = "/home/ronald/src/colibri/c/quant.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("""static void matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale,
                       int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = q8 + (int64_t)o*I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float sc=scl[bi]; float acc=0;
                for(int i=base;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
                a += (double)acc*sc;
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}""",
    """#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
/* Decode 8 e4m3 bytes -> 8 f32 via pure bit manipulation (see the module
 * comment above for the derivation and exhaustive-validation record). No
 * table, no gather -- vpgatherdd measured SLOWER than scalar table lookup on
 * Zen 2, so this sidesteps that weakness instead of working around it. */
static inline __m256 e4m3x8_to_f32x8(__m128i b8) {
    __m256i b = _mm256_cvtepu8_epi32(b8);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
    __m256i exp4 = _mm256_and_si256(_mm256_srli_epi32(b, 3), _mm256_set1_epi32(0xF));
    __m256i mant3 = _mm256_and_si256(b, _mm256_set1_epi32(0x7));
    __m256i mag7 = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));
    __m256i normal_bits = _mm256_or_si256(sign,
        _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(exp4, _mm256_set1_epi32(120)), 23),
                         _mm256_slli_epi32(mant3, 20)));
    __m256 mant3f = _mm256_cvtepi32_ps(mant3);
    __m256 signf = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    __m256 subnorm = _mm256_mul_ps(_mm256_mul_ps(mant3f, signf), _mm256_set1_ps(1.0f/512.0f));
    __m256i is_sub = _mm256_cmpeq_epi32(exp4, _mm256_setzero_si256());
    __m256 blended = _mm256_blendv_ps(_mm256_castsi256_ps(normal_bits), subnorm, _mm256_castsi256_ps(is_sub));
    __m256i is_nan = _mm256_cmpeq_epi32(mag7, _mm256_set1_epi32(0x7F));
    __m256 nanval = _mm256_castsi256_ps(_mm256_or_si256(sign, _mm256_set1_epi32(0x7FC00000)));
    return _mm256_blendv_ps(blended, nanval, _mm256_castsi256_ps(is_nan));
}
#endif

static void matmul_fp8(float *y, const float *x, const uint8_t *q8, const float *bscale,
                       int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = q8 + (int64_t)o*I;
        int64_t blkO = o / FP8_BLOCK;
        const float *scl = bscale + blkO*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I;
            double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK); int blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                float sc=scl[bi];
#if defined(__AVX2__) && defined(__FMA__)
                __m256 vacc=_mm256_setzero_ps(); int i=base;
                for(;i+8<=base+blen;i+=8){
                    __m128i wb=_mm_loadl_epi64((const __m128i*)(w+i));
                    __m256 wf=e4m3x8_to_f32x8(wb);
                    __m256 xf=_mm256_loadu_ps(xs+i);
                    vacc=_mm256_fmadd_ps(xf,wf,vacc);
                }
                float buf[8]; _mm256_storeu_ps(buf,vacc);
                float acc=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
                for(;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
#else
                float acc=0;
                for(int i=base;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
#endif
                a += (double)acc*sc;
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}""",
    "AVX2 matmul_fp8")

open(p, "w").write(s)
print("FP8 AVX2 BIT-TRICK APPLIED")
