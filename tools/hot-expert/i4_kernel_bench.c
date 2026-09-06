/* G11 microbenchmark for matmul_i4_grouped.
 *
 * Fidelity notes, because a naive version of this benchmark lies:
 *  - EPYC 7F32 has 128 MB of L3 and one GLM expert is 14.2 MB, so hammering a
 *    single expert measures L3, not what the engine does. The engine streams
 *    ~988 MB/token (69.6 experts x 14.2 MB) cold from DRAM. So allocate enough
 *    experts to blow past L3 and cycle through them.
 *  - Real GLM shapes: gate/up are [O=2048, I=4096], down is [O=4096, I=2048],
 *    gs=64, S=1 (decode is one token at a time).
 *  - Every variant is checked against the baseline for bit-exactness before it
 *    is timed, and the checksum is printed so nothing gets optimised away.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "quant.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec/1e9; }

/* ---------------- variant A: bit-trick nibble -> float --------------------
 * Replaces cvtepu8_epi32 + sub_epi32 + cvtepi32_ps (3 ops incl. an int->float
 * convert) with or_si256 + sub_ps (2 ops, no convert): OR the nibble into the
 * mantissa of 2^23 and subtract 2^23+8. Produces EXACTLY the same float as
 * (float)(n-8) for n in 0..15, so this stays bit-identical.            */
#ifdef __AVX2__
static inline __m256 nib2ps_bittrick(__m128i nib8){
    const __m256i magic_i = _mm256_set1_epi32(0x4B000000);      /* 8388608.0f */
    const __m256  magic_f = _mm256_set1_ps(8388608.0f + 8.0f);
    __m256i w = _mm256_cvtepu8_epi32(nib8);
    return _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(w, magic_i)), magic_f);
}
#endif
static void mm_bittrick(float *y, const float *x, const uint8_t *q4, const float *scale,
                        int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g]; int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F);
                __m256 acc=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   nib2ps_bittrick(nib), acc);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), nib2ps_bittrick(_mm_srli_si128(nib,8)), acc); }
                a=fmaf(hsum256(acc),sc,a);
#endif
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---------------- variant B: 2 independent accumulators -------------------
 * Breaks the serial FMA dependency chain. NOT bit-identical: the two lane-wise
 * partial sums are combined in a different order.                          */
static void mm_acc2(float *y, const float *x, const uint8_t *q4, const float *scale,
                    int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g]; int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F);
                __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    acc0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   nib2ps_bittrick(nib), acc0);
                    acc1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), nib2ps_bittrick(_mm_srli_si128(nib,8)), acc1); }
                a=fmaf(hsum256(_mm256_add_ps(acc0,acc1)),sc,a);
#endif
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

/* ---------------- variant C: 4 accumulators, 32 nibbles per iteration -----
 * Also loads 16 B of weights at a time instead of 8, halving the mask/shift
 * setup per nibble. NOT bit-identical.                                     */
static void mm_acc4(float *y, const float *x, const uint8_t *q4, const float *scale,
                    int S, int I, int O, int gs){
    int rb=(I+1)/2; int ng=(I+gs-1)/gs;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I; float a=0;
            for(int g=0; g*gs<I; g++){
                int base=g*gs; int glen=gs; if(base+glen>I) glen=I-base;
                float sc=scl[g]; int i=base;
#ifdef __AVX2__
                const __m128i m4=_mm_set1_epi8(0x0F);
                __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                for(; i+32<=base+glen; i+=32){
                    __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i n0=_mm_unpacklo_epi8(lo,hi);     /* nibbles  0..15 */
                    __m128i n1=_mm_unpackhi_epi8(lo,hi);     /* nibbles 16..31 */
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),    nib2ps_bittrick(n0), a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),  nib2ps_bittrick(_mm_srli_si128(n0,8)), a1);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+16), nib2ps_bittrick(n1), a2);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+24), nib2ps_bittrick(_mm_srli_si128(n1,8)), a3); }
                for(; i+16<=base+glen; i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                    __m128i lo=_mm_and_si128(by,m4),hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                    __m128i nib=_mm_unpacklo_epi8(lo,hi);
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   nib2ps_bittrick(nib), a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), nib2ps_bittrick(_mm_srli_si128(nib,8)), a1); }
                __m256 s01=_mm256_add_ps(a0,a1), s23=_mm256_add_ps(a2,a3);
                a=fmaf(hsum256(_mm256_add_ps(s01,s23)),sc,a);
#endif
                for(; i<base+glen; i+=2){
                    if(i+1<base+glen){ uint8_t byte=w[i>>1];
                        a+=(xs[i]*(float)((int)(byte&0xF)-8)+xs[i+1]*(float)((int)(byte>>4)-8))*sc; }
                    else { uint8_t byte=w[i>>1]; a+=xs[i]*(float)((int)(byte&0xF)-8)*sc; }
                }
            }
            y[(int64_t)s*O+o]=a;
        }
    }
}

typedef void (*kern_t)(float*,const float*,const uint8_t*,const float*,int,int,int,int);


/* ---------------- variant D: INTEGER-DOMAIN dot product -------------------
 * The one thing G11 never tried. Variants A-C all keep the float FMA and only
 * make the nibble->float conversion cheaper; this removes the conversion from
 * the inner loop entirely.
 *
 *   - quantise the activation to int8 ONCE per matmul, per group of gs. That
 *     cost is O(I) against the matmul's O(I*O), i.e. 0.05% at GLM's shapes,
 *     and it is what makes the whole thing pay.
 *   - inner loop is maddubs_epi16 + madd_epi16 into int32. No cvt, no FMA.
 *   - nibbles stay UNSIGNED 0..15 (maddubs wants u8 x s8). The stored value is
 *     n-8, and sum((n-8)*q) = sum(n*q) - 8*sum(q), so the bias comes off with a
 *     per-group sum of the quantised activation, computed in the same pass.
 *   - saturation check: n<=15, q in [-128,127] -> |n*q| <= 1920, and maddubs
 *     sums two of them -> |.| <= 3840, well inside int16. No clamping needed.
 *
 * NOT bit-identical by construction: the activation is int8 now. That is the
 * trade being measured -- speed against a numerics change that would ship
 * behind a knob, off by default, exactly like COLI_KDA_GPU. */
static void mm_i8dot(float *y, const float *x, const uint8_t *q4, const float *scale,
                     int S, int I, int O, int gs){
    (void)S;
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    int8_t *xq=NULL; float *xs=NULL; int32_t *xsum=NULL;
    if(posix_memalign((void**)&xq,64,((size_t)I+63)&~(size_t)63)) return;
    if(posix_memalign((void**)&xs,64,(size_t)ng*sizeof(float))) { free(xq); return; }
    if(posix_memalign((void**)&xsum,64,(size_t)ng*sizeof(int32_t))) { free(xq); free(xs); return; }

    for(int g=0; g<ng; g++){
        int base=g*gs, glen=gs; if(base+glen>I) glen=I-base;
        float amax=0;
        for(int i=base;i<base+glen;i++){ float a=fabsf(x[i]); if(a>amax) amax=a; }
        float sc=amax/127.0f; if(sc<1e-12f) sc=1e-12f;
        xs[g]=sc;
        float inv=1.0f/sc; int32_t acc=0;
        for(int i=base;i<base+glen;i++){
            int v=(int)lrintf(x[i]*inv);
            if(v>127) v=127; if(v<-128) v=-128;
            xq[i]=(int8_t)v; acc+=v;
        }
        xsum[g]=acc;
    }

    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*rb;
        const float *scl=scale+(int64_t)o*ng;
        float a=0;
        for(int g=0; g<ng; g++){
            int base=g*gs, glen=gs; if(base+glen>I) glen=I-base;
            int32_t dot=0; int i=base;
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F);
            const __m256i ones=_mm256_set1_epi16(1);
            __m256i acc=_mm256_setzero_si256();
            for(; i+32<=base+glen; i+=32){
                __m128i by=_mm_loadu_si128((const __m128i*)(w+(i>>1)));   /* 32 nibbles */
                __m128i lo=_mm_and_si128(by,m4);
                __m128i hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i n0=_mm_unpacklo_epi8(lo,hi);    /* weights i   .. i+15 */
                __m128i n1=_mm_unpackhi_epi8(lo,hi);    /* weights i+16.. i+31 */
                __m256i wu=_mm256_set_m128i(n1,n0);
                __m256i xv=_mm256_loadu_si256((const __m256i*)(xq+i));
                acc=_mm256_add_epi32(acc,
                    _mm256_madd_epi16(_mm256_maddubs_epi16(wu,xv), ones));
            }
            __m128i s128=_mm_add_epi32(_mm256_castsi256_si128(acc),
                                       _mm256_extracti128_si256(acc,1));
            s128=_mm_hadd_epi32(s128,s128); s128=_mm_hadd_epi32(s128,s128);
            dot=_mm_cvtsi128_si32(s128);
#endif
            for(; i<base+glen; i++){
                uint8_t byte=w[i>>1];
                int nib=(i&1)?(byte>>4):(byte&0xF);
                dot += nib * (int)xq[i];
            }
            a += (float)(dot - 8*xsum[g]) * (scl[g]*xs[g]);
        }
        y[o]=a;
    }
    free(xq); free(xs); free(xsum);
}

int main(int argc, char **argv){
    int I = argc>1?atoi(argv[1]):4096;
    int O = argc>2?atoi(argv[2]):2048;
    int gs= argc>3?atoi(argv[3]):64;
    int reps=argc>4?atoi(argv[4]):40;

    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    size_t wbytes=(size_t)O*rb, sbytes=(size_t)O*ng*sizeof(float);
    size_t per = wbytes+sbytes;
    /* defeat the 128 MB L3: cycle over enough matrices to exceed it ~3x */
    int NM = (int)((384ull*1024*1024)/per) + 1;
    fprintf(stderr,"shape I=%d O=%d gs=%d | %.2f MB/matrix | %d matrices = %.0f MB | reps=%d\n",
            I,O,gs,per/1e6,NM,NM*per/1e6,reps);

    uint8_t **W=malloc(NM*sizeof(*W)); float **SC=malloc(NM*sizeof(*SC));
    srand(1234);
    for(int m=0;m<NM;m++){
        if(posix_memalign((void**)&W[m],64,wbytes)) return 1;
        if(posix_memalign((void**)&SC[m],64,sbytes)) return 1;
        for(size_t i=0;i<wbytes;i++) W[m][i]=(uint8_t)(rand()&0xFF);
        for(size_t i=0;i<sbytes/4;i++) SC[m][i]=((rand()/(float)RAND_MAX)-0.5f)*0.05f;
    }
    float *x=NULL,*y=NULL,*yref=NULL;
    if(posix_memalign((void**)&x,64,(size_t)I*sizeof(float))) return 1;
    if(posix_memalign((void**)&y,64,(size_t)O*sizeof(float))) return 1;
    if(posix_memalign((void**)&yref,64,(size_t)O*sizeof(float))) return 1;
    for(int i=0;i<I;i++) x[i]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;

    struct { const char *name; kern_t fn; int exact; } V[] = {
        {"baseline (in-tree)", matmul_i4_grouped, 1},
        {"A bittrick decode",  mm_bittrick,       1},
        {"B bittrick + 2 acc", mm_acc2,           0},
        {"C bittrick + 4 acc + 16B load", mm_acc4,0},
        {"D INTEGER-DOMAIN (int8 act)",   mm_i8dot,  0},
    };
    int nv=sizeof(V)/sizeof(V[0]);

    /* correctness first, against the in-tree kernel, on matrix 0 */
    matmul_i4_grouped(yref,x,W[0],SC[0],1,I,O,gs);
    for(int v=1;v<nv;v++){
        V[v].fn(y,x,W[0],SC[0],1,I,O,gs);
        double mx=0; int bitexact=1;
        double dot=0, na=0, nb=0, se=0, sr=0;
        for(int o=0;o<O;o++){ double d=fabs((double)y[o]-(double)yref[o]); if(d>mx) mx=d;
            if(y[o]!=yref[o]) bitexact=0;
            dot += (double)y[o]*yref[o]; na += (double)y[o]*y[o]; nb += (double)yref[o]*yref[o];
            se += d*d; sr += (double)yref[o]*yref[o]; }
        double cos = dot/(sqrt(na)*sqrt(nb)+1e-300);
        double rel = sqrt(se/(sr+1e-300));
        printf("  check %-32s bit-exact=%s  max_abs=%.3g  cos=%.9f  relL2=%.3g%s\n", V[v].name,
               bitexact?"YES":"no ", mx, cos, rel, V[v].exact&&!bitexact?"   <-- EXPECTED EXACT, IS NOT":"");
    }
    printf("\n");

    double macs = 2.0*I*O;   /* one matmul = I*O MACs, 2 flops each */
    printf("%-34s %10s %10s %10s %10s\n","variant","ms/call","GB/s","GMAC/s","vs base");
    double base_ms=0;
    for(int v=0;v<nv;v++){
        /* warm the code path, then time cold-streaming reps */
        V[v].fn(y,x,W[0],SC[0],1,I,O,gs);
        double t0=now_s(); double chk=0;
        for(int r=0;r<reps;r++){ int m=r%NM; V[v].fn(y,x,W[m],SC[m],1,I,O,gs); chk+=y[r%O]; }
        double dt=now_s()-t0;
        double ms=dt/reps*1000.0, gbs=per*reps/dt/1e9, gmac=(macs/2)*reps/dt/1e9;
        if(v==0) base_ms=ms;
        printf("%-34s %10.3f %10.2f %10.2f %9.2fx   [chk %.4f]\n",
               V[v].name, ms, gbs, gmac, base_ms/ms, chk);
    }
    return 0;
}
