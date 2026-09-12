/* Vulkan placement microbench against the real colibri backend (backend_vulkan.o).
 * Measures on THIS rig: submit+fence round trip, dense GEMV per format
 * (int8 / int4 / fp8-emulated), batched dense (matmul_multi), and the expert-group
 * primitive at Qwen3.8 shapes (D=2560, I=640, K=10) in fp8 vs int8 vs int4 with a
 * rotating pool of distinct experts (> Infinity Cache) so weights stream from VRAM.
 *
 * Build: gcc -O2 -fopenmp vkbench.c backend_vulkan.o -o vkbench -lvulkan -lm
 * Run:   ./vkbench shaders/qmatmul.spv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "backend_vulkan.h"

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static size_t rowbytes(int fmt,int I){return fmt==2?(size_t)(I+1)/2:fmt==9?(size_t)I*2:(size_t)I;} /* 1,8: 1 B/elem; 2: nibble; 9: bf16 */
static size_t nscales(int fmt,int I,int O){return fmt==8?(size_t)((O+127)/128)*((I+127)/128):fmt==9?(size_t)1:(size_t)O;}
static uint8_t rnd8(void){uint8_t v=rand()&0xff;return (v&0x7f)>=0x7e?0x38:v;}      /* no e4m3 NaN codes */
static float *randx(int n){float *x=malloc((size_t)n*4);for(int i=0;i<n;i++)x[i]=(rand()%200-100)/100.0f;return x;}
static uint8_t *randw(size_t n){uint8_t *w=malloc(n);for(size_t i=0;i<n;i++)w[i]=rnd8();return w;}
static float *consts(size_t n,float v){float *s=malloc(n*4);for(size_t i=0;i<n;i++)s[i]=v;return s;}
static const char *fname(int fmt){return fmt==1?"int8":fmt==2?"int4":fmt==8?"fp8-emul":fmt==9?"bf16":"?";}

/* ===================== Q7 step 0: the BF16 fmt=9 dense path =====================
 * Roadmap Q7 / tools/hot-expert/Q7-DENSE-GPU-SPEC-2026-09-12.md. Two jobs:
 *   (a) prove the GPU BF16 GEMV computes what the engine's q38_matmul_bf16
 *       computes -- exhaustively on the dequant itself, bit-exactly on data
 *       whose sums are order-independent, and to relL2 on random data;
 *   (b) time ONE SUBMIT at the five shapes the item dispatches, from pools
 *       larger than the 96 MB Infinity Cache, so the per-set ms/token the spec
 *       gates on is measured before a line of engine code is written.
 * Nothing here runs the model; this is the isolated harness. */

static inline float bf16_to_f32x(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}

/* THREE references, because "is the GPU right?" and "is the GPU as accurate as
 * what it replaces?" are different questions and only the second one matters.
 *
 *   ref_f64     the truth: the same products accumulated in double.
 *   ref_engine  what qwen38 computes TODAY, copied verbatim from
 *               qwen38_core.h's q38_matmul_bf16 default arm (8-wide AVX2
 *               accumulator + left-to-right horizontal sum). Q4's acc4 arm is
 *               deliberately not the reference: this item is measured against
 *               the knob-off engine.
 *   ref_scalar  a plain sequential f32 sum.
 *
 * An absolute relL2 bar against ref_scalar is the wrong gate and the first run
 * showed why: over I = 6144 random terms a SEQUENTIAL f32 sum carries ~sqrt(I)
 * roundings of its own, so it differs from the GPU's subgroup tree by ~1.4e-6
 * with the tree being the MORE accurate of the two. The gate that means what the
 * spec means -- "summation order only, not a precision loss" -- is: the GPU is
 * no further from the f64 truth than the CPU kernel it replaces. */
static void ref_f64(float *y,const float *x,const uint16_t *W,int I,int O){
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        double a=0.0;for(int i=0;i<I;i++)a+=(double)x[i]*(double)bf16_to_f32x(w[i]);
        y[o]=(float)a;
    }
}
static void ref_scalar(float *y,const float *x,const uint16_t *W,int I,int O){
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        float a=0.f;for(int i=0;i<I;i++)a+=x[i]*bf16_to_f32x(w[i]);
        y[o]=a;
    }
}
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static inline __m256 bf16x8(__m128i h){__m256i w=_mm256_cvtepu16_epi32(h);return _mm256_castsi256_ps(_mm256_slli_epi32(w,16));}
static void ref_engine(float *y,const float *x,const uint16_t *W,int I,int O){
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        __m256 vacc=_mm256_setzero_ps();int i=0;
        for(;i+8<=I;i+=8)
            vacc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),bf16x8(_mm_loadu_si128((const __m128i*)(w+i))),vacc);
        float buf[8];_mm256_storeu_ps(buf,vacc);
        float a=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
        for(;i<I;i++)a+=x[i]*bf16_to_f32x(w[i]);
        y[o]=a;
    }
}
#else
#define ref_engine ref_scalar
#endif
static double rel_l2(const float *a,const float *ref,int n){
    double num=0,den=0;
    for(int i=0;i<n;i++){double d=(double)a[i]-ref[i];num+=d*d;den+=(double)ref[i]*ref[i];}
    return den>0?sqrt(num/den):0.0;
}

static uint16_t *bf16_rand(size_t n){
    uint16_t *w=malloc(n*2);
    for(size_t i=0;i<n;i++){                 /* finite, O(1) magnitude, no NaN/Inf/denormal */
        int e=120+(rand()%9);                /* exponents 2^-7 .. 2^1 */
        w[i]=(uint16_t)(((rand()&1)<<15)|(e<<7)|(rand()&0x7f));
    }
    return w;
}
/* Integers in [-8,8]: every product and every partial sum over I<=10240 is exact
 * in f32 (|sum| <= 8*8*10240 = 655360 < 2^24), so ANY summation order gives the
 * SAME bits. A relL2 bar would hide a transposed or half-shifted column here; a
 * bit compare cannot. */
static uint16_t *bf16_smallint(size_t n){
    uint16_t *w=malloc(n*2);
    for(size_t i=0;i<n;i++){float v=(float)(rand()%17-8);uint32_t u;memcpy(&u,&v,4);w[i]=(uint16_t)(u>>16);}
    return w;
}
static float *smallint_x(int n){float *x=malloc((size_t)n*4);for(int i=0;i<n;i++)x[i]=(float)(rand()%17-8);return x;}
static float one_f=1.0f;

/* (a1) EXHAUSTIVE: every finite bf16 code, both lane parities, through the real
 * shader. I=2 so one word holds the pair; x picks the lane. This is the check
 * the spec asks for instead of the argument that "a shift is a shift".
 *
 * The expectation is the ENGINE'S OWN DOT PRODUCT over the same two-element row,
 * not the raw bf16_to_f32 of the code -- and that distinction is not pedantry:
 * compared against the raw dequant, code 0x8000 (negative zero) "disagreed",
 * because a sum that starts at +0.0 turns -0.0 into +0.0 by IEEE 754. The CPU
 * kernel does exactly the same thing. Comparing dot against dot asks the
 * question the item is actually about. */
static int bf16_dequant_exhaustive(void){
    const int O=65536,I=2;int bad=0,denorm=0,other=0;
    uint16_t *W=calloc((size_t)O*I,2);
    float *y=malloc((size_t)O*4),*yref=malloc((size_t)O*4);
    for(int parity=0;parity<2;parity++){
        for(int c=0;c<O;c++){W[(size_t)c*I+0]=0;W[(size_t)c*I+1]=0;W[(size_t)c*I+parity]=(uint16_t)c;}
        float x[2]={parity==0?1.f:0.f,parity==0?0.f:1.f};
        ColiVkTensor *t=NULL;
        if(!coli_vk_matmul(&t,y,x,W,&one_f,9,1,I,O,0)){printf("BF16 exhaustive: dispatch failed\n");return 1;}
        ref_engine(yref,x,W,I,O);
        for(int c=0;c<O;c++){
            if(((c>>7)&0xff)==0xff) continue;          /* Inf/NaN: no weight is one */
            float want=yref[c];
            uint32_t a,b;memcpy(&a,&y[c],4);memcpy(&b,&want,4);
            if(a==b) continue;
            bad++;
            /* A bf16 with exponent field 0 and a nonzero mantissa is an f32
             * DENORMAL (< 2^-126 ~ 1.2e-38). RADV runs compute with fp32
             * denormals flushed to zero, so these come back as +-0. Classified,
             * not counted: it is the only class this kernel is allowed to
             * disagree on, and the engine-side scan says whether the real
             * weights contain any (Q38_DENSE_GPU_SCAN=1). */
            if(((c>>7)&0xff)==0 && (c&0x7f)!=0) denorm++;
            else { if(other<5)printf("  bf16 code 0x%04x parity %d: gpu %.9g (0x%08x) want %.9g (0x%08x)\n",
                                     c,parity,y[c],a,want,b); other++; }
        }
        coli_vk_tensor_free(t);
    }
    printf("BF16 DEQUANT exhaustive: %d/131072 finite codes x parities disagree "
           "(%d denormal flush-to-zero, %d OTHER) -- %s\n",
           bad,denorm,other,
           other?"FAIL":"PASS (bit-identical on every normal code; denormals flushed)");
    free(W);free(y);free(yref);return other;
}

/* (a2) per shape, two legs:
 *   LAYOUT   bit-exact against the engine kernel on order-independent integer
 *            data. This is the leg that catches a transposed row, a wrong
 *            rowWords or a swapped column parity, and no tolerance can hide it.
 *   ACCURACY on random data, relL2 of the GPU and of the CPU kernel it replaces,
 *            both against an f64 accumulation of the same products. The gate is
 *            gpu <= engine: no less accurate than what ships today. */
static int bf16_check_shape(int I,int O){
    uint16_t *We=bf16_smallint((size_t)I*O);float *xe=smallint_x(I);
    float *yg=malloc((size_t)O*4),*ye=malloc((size_t)O*4),*yd=malloc((size_t)O*4),*ys=malloc((size_t)O*4);
    ColiVkTensor *t=NULL;int bad=0;
    if(!coli_vk_matmul(&t,yg,xe,We,&one_f,9,1,I,O,0)){printf("BF16 check %dx%d: dispatch failed\n",I,O);return 1;}
    ref_engine(ye,xe,We,I,O);
    for(int o=0;o<O;o++){uint32_t a,b;memcpy(&a,&yg[o],4);memcpy(&b,&ye[o],4);if(a!=b)bad++;}
    coli_vk_tensor_free(t);t=NULL;
    uint16_t *Wr=bf16_rand((size_t)I*O);float *xr=randx(I);
    if(!coli_vk_matmul(&t,yg,xr,Wr,&one_f,9,1,I,O,0)){printf("BF16 check %dx%d: dispatch failed\n",I,O);return 1;}
    ref_f64(yd,xr,Wr,I,O); ref_engine(ye,xr,Wr,I,O); ref_scalar(ys,xr,Wr,I,O);
    double rg=rel_l2(yg,yd,O),re=rel_l2(ye,yd,O),rs=rel_l2(ys,yd,O);
    double mx=0;for(int o=0;o<O;o++){double d=fabs((double)yg[o]-ye[o]);if(d>mx)mx=d;}
    int fail=bad||rg>re;
    printf("BF16 CHECK  I=%5d O=%6d: layout %d/%d rows differ (%s) | vs f64 truth: "
           "gpu %.3e  engine %.3e  scalar %.3e -> gpu is %.2fx the engine's error (%s) | "
           "gpu-vs-engine max-abs %.3e\n",
           I,O,bad,O,bad?"FAIL":"PASS",rg,re,rs,re>0?rg/re:0.0,rg<=re?"PASS":"FAIL",mx);
    coli_vk_tensor_free(t);
    free(We);free(xe);free(Wr);free(xr);free(yg);free(ye);free(yd);free(ys);
    return fail;
}

static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y?1:0;}

/* (b) ONE SUBMIT at a Q7 shape, `reps` repeats, MEDIAN. `copies` distinct tensor
 * sets rotate so the weights stream from VRAM and not from Infinity Cache; the
 * pool size is printed so the >96 MB claim is checkable and not assumed.
 * chain<0: all items independent (DeltaNet/QSA A, `out`, the head).
 * chain>=0: item[chain] reads item 0's device-side output (the gr pair). */
static void bench_q7(const char *name,int I,const int *Os,int n,int chain,int copies,int reps,double per_token){
    ColiVkTensor **t=calloc((size_t)copies*n,sizeof(*t));
    float *x=randx(I);
    uint16_t **w=malloc(n*sizeof(*w));float **y=malloc(n*sizeof(*y));
    size_t bytes=0;
    for(int j=0;j<n;j++){
        int Ij=(j==chain)?Os[0]:I;
        w[j]=bf16_rand((size_t)Ij*Os[j]);y[j]=malloc((size_t)Os[j]*4);
        bytes+=(size_t)Ij*Os[j]*2;
    }
    ColiVkMM items[VK_MM_MAX];
    #define Q7_FILL(c) for(int j=0;j<n;j++) items[j]=(ColiVkMM){&t[(size_t)(c)*n+j],w[j],&one_f,9,Os[j],0, \
        (j==0&&chain>0)?NULL:y[j], (j==chain)?Os[0]:I, (j==chain)?0:-1};
    for(int c=0;c<copies;c++){Q7_FILL(c) if(!coli_vk_matmul_multi(items,n,x,I)){printf("Q7 %s: dispatch failed\n",name);return;}}
    double *s=malloc((size_t)reps*sizeof(double));
    for(int r=0;r<reps;r++){
        int c=r%copies;Q7_FILL(c)
        double t0=now();coli_vk_matmul_multi(items,n,x,I);s[r]=(now()-t0)*1e3;
    }
    qsort(s,(size_t)reps,sizeof(double),dcmp);
    double med=s[reps/2];
    printf("Q7 %-13s %d tensor(s) I=%-5d %8.3f MB/submit  median %7.4f ms  (min %7.4f max %7.4f)  %6.1f GB/s  pool %.0f MB  -> x%.0f = %7.3f ms/token\n",
           name,n,I,bytes/1e6,med,s[0],s[reps-1],bytes/med/1e6,(double)copies*bytes/1e6,per_token,med*per_token);
    #undef Q7_FILL
    for(int c=0;c<copies*n;c++)coli_vk_tensor_free(t[c]);
    free(t);free(x);for(int j=0;j<n;j++){free(w[j]);free(y[j]);}free(w);free(y);free(s);
}

/* Q7 step 1 diagnosis: the SAME two DeltaNet submits, but with the engine's own
 * CPU gap between them.
 *
 * Step 0 predicted dn-proj at 19.6 ms/token from back-to-back submits; in the
 * engine the identical dispatches cost 26.96. The difference has to be either
 * the kernel (it is not -- same code, same shapes, same VRAM) or the DUTY CYCLE:
 * in the engine, submit A is followed by ~0.26 ms of CPU recurrence
 * (dn-recur 9.28 ms/token over 36 layers) and submit B by the rest of the layer,
 * so dev0 is idle in sub-millisecond bursts 72 times a token and never reaches
 * the clocks a back-to-back loop holds it at. This case reproduces that shape
 * with a busy-wait of the measured length and nothing else changed. If the
 * per-submit time rises to the in-engine figure, the attenuation is the duty
 * cycle and NOT something a wider load in the shader can fix. */
static void spin_ms(double ms){double t0=now();while((now()-t0)*1e3<ms){}}
static void bench_q7_dutycycle(double gap_ab,double gap_ba,int copies,int reps){
    const int I1=2560,I2=6144;
    int OsA[4]={10240,6144,48,48},OsB=2560;
    ColiVkTensor **ta=calloc((size_t)copies*4,sizeof(*ta)),**tb=calloc(copies,sizeof(*tb));
    float *x1=randx(I1),*x2=randx(I2);
    uint16_t *wa[4],*wb;float *ya[4],*yb=malloc((size_t)OsB*4);
    size_t bytes=0;
    for(int j=0;j<4;j++){wa[j]=bf16_rand((size_t)I1*OsA[j]);ya[j]=malloc((size_t)OsA[j]*4);bytes+=(size_t)I1*OsA[j]*2;}
    wb=bf16_rand((size_t)I2*OsB);bytes+=(size_t)I2*OsB*2;
    ColiVkMM it[4];
    #define FILL_A(c) for(int j=0;j<4;j++) it[j]=(ColiVkMM){&ta[(size_t)(c)*4+j],wa[j],&one_f,9,OsA[j],0,ya[j],I1,-1};
    #define FILL_B(c) it[0]=(ColiVkMM){&tb[c],wb,&one_f,9,OsB,0,yb,I2,-1};
    for(int c=0;c<copies;c++){FILL_A(c) if(!coli_vk_matmul_multi(it,4,x1,I1)){printf("dutycycle: A failed\n");return;}
                              FILL_B(c) if(!coli_vk_matmul_multi(it,1,x2,I2)){printf("dutycycle: B failed\n");return;}}
    double *s=malloc((size_t)reps*sizeof(double));
    for(int r=0;r<reps;r++){
        int c=r%copies;double acc=0,t0;
        FILL_A(c) t0=now(); coli_vk_matmul_multi(it,4,x1,I1); acc+=now()-t0;
        spin_ms(gap_ab);
        FILL_B(c) t0=now(); coli_vk_matmul_multi(it,1,x2,I2); acc+=now()-t0;
        spin_ms(gap_ba);
        s[r]=acc*1e3;
    }
    qsort(s,(size_t)reps,sizeof(double),dcmp);
    printf("Q7 dn-layer   gap A->B %.2f ms, B->A %.2f ms: median %7.4f ms/layer  "
           "(min %7.4f max %7.4f)  -> x36 = %7.3f ms/token   [engine measured 26.98]\n",
           gap_ab,gap_ba,s[reps/2],s[0],s[reps-1],s[reps/2]*36);
    #undef FILL_A
    #undef FILL_B
    for(int c=0;c<copies*4;c++)coli_vk_tensor_free(ta[c]);
    for(int c=0;c<copies;c++)coli_vk_tensor_free(tb[c]);
    free(ta);free(tb);free(x1);free(x2);free(yb);free(wb);
    for(int j=0;j<4;j++){free(wa[j]);free(ya[j]);}
    free(s);
}

static void q7_step0(void){
    printf("\n===== Q7 step 0: fmt=9 BF16 on dev0 (spec Q7-DENSE-GPU-SPEC-2026-09-12) =====\n");
    int bad=bf16_dequant_exhaustive();
    bad|=bf16_check_shape(2560,10240);      /* DeltaNet qkv          */
    bad|=bf16_check_shape(6144,2560);       /* DeltaNet/QSA out      */
    bad|=bf16_check_shape(2560,12288);      /* QSA q                 */
    bad|=bf16_check_shape(10240,320);       /* gr down (UNSTAGED: I > 6144) */
    bad|=bf16_check_shape(320,10240);       /* gr up                 */
    bad|=bf16_check_shape(2560,48);         /* DeltaNet b/a          */
    bad|=bf16_check_shape(2561,777);        /* odd I, odd O: the tail guard  */
    printf("BF16 CORRECTNESS: %s\n",bad?"*** FAIL -- do not time this ***":"ALL PASS");
    if(bad)return;
    printf("\n-- one submit per site, median of 10, pools > 96 MB Infinity Cache --\n");
    { int Os[4]={10240,6144,48,48};   bench_q7("deltanet-A",2560,Os,4,-1,4,10,36); }
    { int Os[1]={2560};               bench_q7("deltanet-B",6144,Os,1,-1,8,10,36); }
    { int Os[4]={12288,512,512,640};  bench_q7("qsa-A",     2560,Os,4,-1,4,10,12); }
    { int Os[1]={2560};               bench_q7("qsa-B",     6144,Os,1,-1,8,10,12); }
    { int Os[1]={248320};             bench_q7("lm-head",   2560,Os,1,-1,2,10, 1); }
    /* the gated-residual pair as ONE submit: down (-1, intermediate), inject (-1),
     * up (chained on down). No `act` flag yet -- that push constant is step 4; what
     * this times is the bytes and the single round trip, which is what the step-4
     * go/no-go (<= 9.2 ms/token) is about. */
    { int Os[3]={320,4,10240};        bench_q7("gr-pair",  10240,Os,3, 2,16,10,97); }
    /* Step 1's attenuation, isolated. 0.00 is the back-to-back control; 0.26 is
     * the engine's measured CPU recurrence between A and B (dn-recur 9.28
     * ms/token / 36); 1.5 is roughly the rest of a layer (the token is 139.7 ms
     * over 48 layers, minus the DeltaNet work itself). */
    printf("\n-- step 1 diagnosis: the same two submits with the engine's CPU gap between them --\n");
    bench_q7_dutycycle(0.00,0.00,4,20);
    bench_q7_dutycycle(0.26,0.00,4,20);
    bench_q7_dutycycle(0.26,1.50,4,20);
    bench_q7_dutycycle(0.26,3.00,4,20);
}

static void bench_dense(int fmt,int I,int O,int copies,int iters){
    size_t rb=rowbytes(fmt,I),ns=nscales(fmt,I,O);
    ColiVkTensor **t=calloc(copies,sizeof(*t));
    float *x=randx(I),*y=malloc((size_t)O*4);
    uint8_t *w=randw(rb*O);float *s=consts(ns,0.01f);
    for(int c=0;c<copies;c++) if(!coli_vk_matmul(&t[c],y,x,w,s,fmt,1,I,O,0)){printf("dense fmt=%d %dx%d unsupported\n",fmt,I,O);return;}
    for(int c=0;c<copies;c++)coli_vk_matmul(&t[c],y,x,w,s,fmt,1,I,O,0);
    double t0=now();for(int k=0;k<iters;k++)coli_vk_matmul(&t[k%copies],y,x,w,s,fmt,1,I,O,0);
    double ms=(now()-t0)*1e3/iters;
    printf("DENSE %-9s S=1 %5dx%-6d %7.3f ms/call  %6.1f GB/s (%d rotating copies, %.0f MB each)\n",fname(fmt),I,O,ms,rb*O/ms/1e6,copies,rb*O/1e6);
    for(int c=0;c<copies;c++)coli_vk_tensor_free(t[c]);free(t);free(x);free(y);free(w);free(s);
}

/* N dense tensors in ONE submit vs N separate calls (the DeltaNet qkv/z/b/a pattern) */
static void bench_multi(int fmt,int I,const int *Os,int n,int copies,int iters){
    size_t rb=rowbytes(fmt,I);
    ColiVkTensor **t=calloc((size_t)copies*n,sizeof(*t));
    float *x=randx(I);float **y=malloc(n*sizeof(float*));uint8_t **w=malloc(n*sizeof(uint8_t*));float **s=malloc(n*sizeof(float*));
    for(int j=0;j<n;j++){y[j]=malloc((size_t)Os[j]*4);w[j]=randw(rb*Os[j]);s[j]=consts(nscales(fmt,I,Os[j]),0.01f);}
    ColiVkMM items[VK_MM_MAX];
    for(int c=0;c<copies;c++){for(int j=0;j<n;j++){items[j]=(ColiVkMM){&t[c*n+j],w[j],s[j],fmt,Os[j],0,y[j],I,-1};}
        if(!coli_vk_matmul_multi(items,n,x,I)){printf("multi unsupported\n");return;}}
    double t0=now();
    for(int k=0;k<iters;k++){int c=k%copies;for(int j=0;j<n;j++)items[j]=(ColiVkMM){&t[c*n+j],w[j],s[j],fmt,Os[j],0,y[j],I,-1};coli_vk_matmul_multi(items,n,x,I);}
    double ms_multi=(now()-t0)*1e3/iters;
    t0=now();
    for(int k=0;k<iters;k++){int c=k%copies;for(int j=0;j<n;j++)coli_vk_matmul(&t[c*n+j],y[j],x,w[j],s[j],fmt,1,I,Os[j],0);}
    double ms_sep=(now()-t0)*1e3/iters;
    size_t bytes=0;for(int j=0;j<n;j++)bytes+=rb*Os[j];
    printf("MULTI %-9s %d tensors I=%d: one-submit %7.3f ms  vs separate %7.3f ms  (%.0f MB, %.1f GB/s batched)\n",fname(fmt),n,I,ms_multi,ms_sep,bytes/1e6,bytes/ms_multi/1e6);
    for(int c=0;c<copies*n;c++)coli_vk_tensor_free(t[c]);
}

/* expert group at Qwen shapes: K experts per call, rows per expert = R, pool P distinct */
static void bench_group(int fmt,int D,int I,int K,int R,int P,int iters,int dev){
    size_t gu_rb=rowbytes(fmt,D),d_rb=rowbytes(fmt,I),gu_sc=nscales(fmt,D,I),d_sc=nscales(fmt,I,D);
    ColiVkTensor **tg=calloc(P,sizeof(*tg)),**tu=calloc(P,sizeof(*tu)),**td=calloc(P,sizeof(*td));
    uint8_t *gw=randw(gu_rb*I),*uw=randw(gu_rb*I),*dw=randw(d_rb*D);
    float *gs=consts(gu_sc,0.01f),*us=consts(gu_sc,0.01f),*ds=consts(d_sc,0.01f);
    int (*ens)(ColiVkTensor**,const void*,const float*,int,int,int,int)=dev==0?coli_vk_tensor_ensure:dev==1?coli_vk_tensor_ensure2:coli_vk_tensor_ensure3;
    int (*grp)(ColiVkTensor*const*,ColiVkTensor*const*,ColiVkTensor*const*,const int*,int,float*,const float*)=dev==0?coli_vk_expert_group:dev==1?coli_vk_expert_group2:coli_vk_expert_group3;
    for(int p=0;p<P;p++){
        if(!ens(&tg[p],gw,gs,fmt,D,I,128)||!ens(&tu[p],uw,us,fmt,D,I,128)||!ens(&td[p],dw,ds,fmt,I,D,128)){printf("group fmt=%d upload failed at expert %d\n",fmt,p);return;}
    }
    float *x=randx(K*R*D),*y=malloc((size_t)K*R*D*4);
    ColiVkTensor *G[64],*U[64],*Dn[64];int rows[64];int rot=0;
    for(int k=-3;k<iters;k++){
        if(k==0)break;
        for(int z=0;z<K;z++){int e=(rot++)%P;G[z]=tg[e];U[z]=tu[e];Dn[z]=td[e];rows[z]=R;}
        if(!grp(G,U,Dn,rows,K,y,x)){printf("group fmt=%d failed\n",fmt);return;}
    }
    double t0=now();
    for(int k=0;k<iters;k++){
        for(int z=0;z<K;z++){int e=(rot++)%P;G[z]=tg[e];U[z]=tu[e];Dn[z]=td[e];rows[z]=R;}
        grp(G,U,Dn,rows,K,y,x);
    }
    double ms=(now()-t0)*1e3/iters;
    double bytes=(double)K*(2.0*gu_rb*I+d_rb*D);
    printf("GROUP dev%d %-9s K=%2d rows=%d D=%d I=%d: %7.3f ms/call  %5.3f ms/expert  %6.1f GB/s  -> x48 layers = %.1f ms/token\n",dev,fname(fmt),K,R,D,I,ms,ms/K,bytes/ms/1e6,ms*48);
    for(int p=0;p<P;p++){coli_vk_tensor_free(tg[p]);coli_vk_tensor_free(tu[p]);coli_vk_tensor_free(td[p]);}
    free(tg);free(tu);free(td);free(gw);free(uw);free(dw);free(gs);free(us);free(ds);free(x);free(y);
}

/* three devices issued concurrently vs sequentially (the engine currently does sequential) */
static void bench_group_3dev(int fmt,int D,int I,int K,int P,int iters){
    if(!coli_vk_dev2_available()||!coli_vk_dev3_available()){printf("3dev: dev2/dev3 not up\n");return;}
    size_t gu_rb=rowbytes(fmt,D),d_rb=rowbytes(fmt,I),gu_sc=nscales(fmt,D,I),d_sc=nscales(fmt,I,D);
    uint8_t *gw=randw(gu_rb*I),*uw=randw(gu_rb*I),*dw=randw(d_rb*D);
    float *gs=consts(gu_sc,0.01f),*us=consts(gu_sc,0.01f),*ds=consts(d_sc,0.01f);
    ColiVkTensor *T[3][3][128]={{{0}}};
    for(int dv=0;dv<3;dv++){
        int (*ens)(ColiVkTensor**,const void*,const float*,int,int,int,int)=dv==0?coli_vk_tensor_ensure:dv==1?coli_vk_tensor_ensure2:coli_vk_tensor_ensure3;
        for(int p=0;p<P;p++){ens(&T[dv][0][p],gw,gs,fmt,D,I,128);ens(&T[dv][1][p],uw,us,fmt,D,I,128);ens(&T[dv][2][p],dw,ds,fmt,I,D,128);}
    }
    float *x=randx(K*D),*y=malloc((size_t)K*D*4);
    ColiVkTensor *G[3][64],*U[3][64],*Dn[3][64];int rows[64];for(int z=0;z<64;z++)rows[z]=1;
    int rot=0;
    #define PICK for(int dv=0;dv<3;dv++)for(int z=0;z<K;z++){int e=(rot++)%P;G[dv][z]=T[dv][0][e];U[dv][z]=T[dv][1][e];Dn[dv][z]=T[dv][2][e];}
    PICK coli_vk_expert_group(G[0],U[0],Dn[0],rows,K,y,x);coli_vk_expert_group2(G[1],U[1],Dn[1],rows,K,y,x);coli_vk_expert_group3(G[2],U[2],Dn[2],rows,K,y,x);
    double t0=now();
    for(int k=0;k<iters;k++){PICK coli_vk_expert_group(G[0],U[0],Dn[0],rows,K,y,x);coli_vk_expert_group2(G[1],U[1],Dn[1],rows,K,y,x);coli_vk_expert_group3(G[2],U[2],Dn[2],rows,K,y,x);}
    double seq=(now()-t0)*1e3/iters;
    t0=now();
    for(int k=0;k<iters;k++){PICK coli_vk_expert_group_issue(G[0],U[0],Dn[0],rows,K,x);coli_vk_expert_group_issue2(G[1],U[1],Dn[1],rows,K,x);coli_vk_expert_group_issue3(G[2],U[2],Dn[2],rows,K,x);
        coli_vk_expert_group_take(y);coli_vk_expert_group_take2(y);coli_vk_expert_group_take3(y);}
    double conc=(now()-t0)*1e3/iters;
    printf("3DEV %-9s K=%d per device (%d experts/layer total): sequential %.3f ms  concurrent issue/take %.3f ms  -> x48 = %.0f vs %.0f ms/token\n",fname(fmt),K,3*K,seq,conc,seq*48,conc*48);
}

int main(int argc,char**argv){
    const char *spv=argc>1?argv[1]:"shaders/qmatmul.spv";
    if(!coli_vk_init(spv)){printf("vk init failed\n");return 1;}
    srand(7);
    double used,budget;if(coli_vk_mem_budget(&used,&budget))printf("dev0 budget %.1f GB used %.1f GB\n",budget,used);
    /* `vkbench <spv> q7` runs only Q7 step 0 -- the placement matrix below is
     * already in the record and re-running it costs minutes of rig time. */
    if(argc>2&&!strcmp(argv[2],"q7")){q7_step0();coli_vk_shutdown();return 0;}
    q7_step0();
    /* 1. round trip: tiny matmul, per-call cost is ~all overhead */
    bench_dense(1,512,512,1,400);
    bench_dense(1,256,64,1,400);
    /* 2. dense GEMV per format at Qwen shapes */
    bench_dense(1,2560,10240,8,60);  bench_dense(2,2560,10240,8,60);  bench_dense(8,2560,10240,8,60);
    bench_dense(1,6144,2560,12,60);  bench_dense(2,6144,2560,12,60);  bench_dense(8,6144,2560,12,60);
    bench_dense(1,2560,640,64,200);  bench_dense(8,2560,640,64,200);
    bench_dense(1,2560,248320,2,20); bench_dense(2,2560,248320,2,20);   /* LM head */
    /* 3. batching dense */
    { int Os[4]={10240,6144,48,48}; bench_multi(1,2560,Os,4,6,60); }
    /* 4. expert group, Qwen shapes */
    bench_group(8,2560,640,10,1,128,60,0);
    bench_group(1,2560,640,10,1,128,60,0);
    bench_group(2,2560,640,10,1,128,60,0);
    bench_group(8,2560,640,4,1,128,60,0);
    bench_group(8,2560,640,1,1,128,60,0);
    bench_group(8,2560,640,10,8,128,30,0);   /* prefill-ish rows=8 */
    bench_group(1,2560,640,10,8,128,30,0);
    /* 5. three devices */
    if(coli_vk_init_dev2(spv,-1))printf("dev2 up\n");
    if(coli_vk_init_dev3(spv,-1))printf("dev3 up\n");
    bench_group_3dev(8,2560,640,4,128,60);
    bench_group_3dev(8,2560,640,10,128,60);
    coli_vk_shutdown();
    return 0;
}
