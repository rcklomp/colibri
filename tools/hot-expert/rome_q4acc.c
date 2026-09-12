/* rome_q4acc.c -- the TRANSFORM ORACLE for roadmap item Q4 (four accumulators
 * in the BF16 dense GEMV).
 *
 * Q4 is not bit-identical, so the engine-level gate is greedy-text identity
 * plus a reported cosine. That gate cannot tell "the summation order moved"
 * apart from "the kernel reads the wrong lane on a tail", which is the bug this
 * class of change actually has: a 32-wide body plus an 8-wide tail plus a
 * scalar tail is three index expressions, and a wrong one on a shape whose I is
 * not a multiple of 32 would look exactly like reassociation on a cosine. So
 * this harness proves the stronger claim the code comment makes -- SAME
 * PRODUCTS, DIFFERENT ORDER -- before the model is ever run. rome_fp8test.c and
 * rome_i3sim.c are the pattern (§G15: an unverified transform's verdict is
 * worthless whichever way it comes out).
 *
 *   A  THE ONE THAT MATTERS. Over data where float addition is EXACT, the two
 *      kernels must be BIT-IDENTICAL. Weights and activations are small
 *      integers (bf16 is exact on integers to 256; |product| <= 64 and
 *      |sum| <= 64*I < 2^24), so every partial sum is exactly representable and
 *      reassociation provably cannot change the result. Any difference is a
 *      wrong index, a dropped element or a double-counted one -- not
 *      reassociation. Run over 36 inner dimensions chosen to hit every arm: the
 *      engine's own (2560, 6144, 10240, 320, 512, all %32==0), multiples of 8
 *      that are not multiples of 32, multiples of neither, I < 32 so the
 *      32-wide body never executes, and I < 8 so only the scalar tail does.
 *      Both kernels are also checked against the exact long-double sum, so a
 *      shared error could not cancel.
 *   B  a negative control: perturb one weight in the lane group a2 owns and
 *      confirm A's comparison notices. A test that cannot fail proves nothing.
 *   C  general float data: relL2 / cosine of acc4 against the 1-accumulator
 *      kernel, and of BOTH against a long-double reference. This is the size of
 *      the reassociation at the kernel level, and it says which of the two is
 *      the more accurate -- four partial sums of I/4 terms each accumulate less
 *      rounding than one chain of I.
 *   D  S>1: the engine calls both kernels with S=32 during prefill batching, so
 *      A is repeated there.
 *
 * Build (rig):
 *   gcc -O3 -march=native -fopenmp -o /tmp/rome_q4acc \
 *       tools/hot-expert/rome_q4acc.c -lm
 * Run:  OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/rome_q4acc
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <immintrin.h>

static inline float bf16_to_f32(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}
static inline uint16_t f32_to_bf16_rz(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)(u>>16);}
static inline __m256 q38_bf16x8_to_f32x8(__m128i h){
    __m256i w=_mm256_cvtepu16_epi32(h);return _mm256_castsi256_ps(_mm256_slli_epi32(w,16));
}

/* ---- the two kernels, copied from c/qwen38_core.h at the commit under test --- */
static void mm_1acc(float *y,const float *x,const uint16_t *W,int S,int I,int O){
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
}
static inline float q38_dot_bf16_acc4(const float *xs,const uint16_t *w,int I){
    __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;int i=0;
    for(;i+32<=I;i+=32){
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),
                           q38_bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i))),a0);
        a1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8),
                           q38_bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+8))),a1);
        a2=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+16),
                           q38_bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+16))),a2);
        a3=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+24),
                           q38_bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+24))),a3);
    }
    for(;i+8<=I;i+=8)
        a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),
                           q38_bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i))),a0);
    a0=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
    float buf[8];_mm256_storeu_ps(buf,a0);
    float a=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for(;i<I;i++)a+=xs[i]*bf16_to_f32(w[i]);
    return a;
}
static void mm_acc4(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++)
            y[(int64_t)s*O+o]=q38_dot_bf16_acc4(x+(int64_t)s*I,w,I);
    }
}

static uint64_t rs=0x243F6A8885A308D3ull;
static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (uint32_t)(rs>>32);}

/* An integer in [-8,8], exactly representable in bf16 and in float. */
static float small_int(void){ return (float)((int)(rnd()%17)-8); }
static float gauss(void){
    double u=(rnd()%1000000+1)/1000001.0,v=(rnd()%1000000+1)/1000001.0;
    return (float)(sqrt(-2*log(u))*cos(6.283185307179586*v));
}

static long double ref_dot_ld(const float *xs,const uint16_t *w,int I){
    long double a=0.0L;
    for(int i=0;i<I;i++) a+=(long double)xs[i]*(long double)bf16_to_f32(w[i]);
    return a;
}

/* aligned_alloc wants a size that is a multiple of the alignment (C11). */
static void *aalloc(size_t bytes){ return aligned_alloc(64,(bytes+63)/64*64); }

int main(void){
    /* ---------------- A: exact arithmetic, must be BIT-IDENTICAL ------------- */
    static const int Iset[]={2560,6144,10240,320,512,32,64,96,128,256,
                             8,16,24,40,72,1,2,3,7,9,15,17,31,33,63,65,
                             129,255,257,321,639,1023,1025,2559,2561,10241};
    const int nI=(int)(sizeof(Iset)/sizeof(Iset[0]));
    int failA=0; long tested=0;
    for(int k=0;k<nI;k++){
        int I=Iset[k],O=37;
        uint16_t *W=aalloc((size_t)I*O*2);
        float *x=aalloc((size_t)I*4);
        float *y1=malloc((size_t)O*4),*y4=malloc((size_t)O*4);
        for(int64_t i=0;i<(int64_t)I*O;i++)W[i]=f32_to_bf16_rz(small_int());
        for(int i=0;i<I;i++)x[i]=small_int();
        mm_1acc(y1,x,W,1,I,O); mm_acc4(y4,x,W,1,I,O);
        int bad=0;
        for(int o=0;o<O;o++){
            if(memcmp(&y1[o],&y4[o],4)){bad++;
                if(bad==1)fprintf(stderr,"A FAIL I=%d o=%d %.9g vs %.9g\n",I,o,y1[o],y4[o]);}
            /* and both must equal the exact integer sum */
            long double r=ref_dot_ld(x,W+(int64_t)o*I,I);
            if((long double)y1[o]!=r||(long double)y4[o]!=r){bad++;
                fprintf(stderr,"A FAIL(exact) I=%d o=%d ref=%.9Lg 1acc=%.9g acc4=%.9g\n",I,o,r,y1[o],y4[o]);}
        }
        tested+=O; failA+=bad;
        free(W);free(x);free(y1);free(y4);
    }
    printf("A  exact-arithmetic bit-identity, %d inner dims x 37 rows = %ld dots: %s (%d mismatches)\n",
           nI,tested,failA?"FAIL":"PASS",failA);

    /* ---------------- B: the test can fail --------------------------------- */
    {
        int I=2560,O=4;
        uint16_t *W=aalloc((size_t)I*O*2);
        float *x=aalloc((size_t)I*4);
        float *y1=malloc(O*4),*y4=malloc(O*4);
        for(int64_t i=0;i<(int64_t)I*O;i++)W[i]=f32_to_bf16_rz(small_int());
        for(int i=0;i<I;i++)x[i]=small_int();
        mm_1acc(y1,x,W,1,I,O);
        /* perturb ONE weight element in the a2 lane group of the first row, the
         * kind of damage a wrong offset in the 32-wide body would do */
        W[16]=f32_to_bf16_rz(bf16_to_f32(W[16])+1.0f);
        mm_acc4(y4,x,W,1,I,O);
        int diff=memcmp(&y1[0],&y4[0],4)!=0;
        printf("B  negative control (one weight perturbed in the a2 lane): %s\n",
               diff?"PASS (A would have caught it)":"FAIL (A is blind)");
        free(W);free(x);free(y1);free(y4);
    }

    /* ---------------- C: how big the reassociation is on real-shaped data --- */
    printf("C  general float data, acc4 vs 1-acc and both vs long double:\n");
    for(int k=0;k<5;k++){
        int I=Iset[k],O=1024;
        uint16_t *W=aalloc((size_t)I*O*2);
        float *x=aalloc((size_t)I*4);
        float *y1=malloc((size_t)O*4),*y4=malloc((size_t)O*4);
        for(int64_t i=0;i<(int64_t)I*O;i++)W[i]=f32_to_bf16_rz(gauss()*0.02f);
        for(int i=0;i<I;i++)x[i]=gauss();
        mm_1acc(y1,x,W,1,I,O); mm_acc4(y4,x,W,1,I,O);
        double dot=0,n1=0,n4=0,num=0,den=0,mx=0,e1=0,e4=0,eden=0;
        for(int o=0;o<O;o++){
            double a=y1[o],b=y4[o];
            dot+=a*b;n1+=a*a;n4+=b*b;num+=(b-a)*(b-a);den+=a*a;
            if(fabs(b-a)>mx)mx=fabs(b-a);
            double r=(double)ref_dot_ld(x,W+(int64_t)o*I,I);
            e1+=(a-r)*(a-r);e4+=(b-r)*(b-r);eden+=r*r;
        }
        printf("   I=%-6d O=%d  cos %.9f  relL2 %.3e  maxabs %.3e   |  err vs long double: 1acc %.3e  acc4 %.3e  (%.2fx)\n",
               I,O,dot/sqrt(n1*n4),sqrt(num/den),mx,sqrt(e1/eden),sqrt(e4/eden),
               sqrt(e4/eden)>0?sqrt(e1/eden)/sqrt(e4/eden):0.0);
        free(W);free(x);free(y1);free(y4);
    }

    /* ---------------- D: S>1, the prefill call shape ------------------------ */
    {
        int I=2560,O=64,S=32,bad=0;
        uint16_t *W=aalloc((size_t)I*O*2);
        float *x=aalloc((size_t)S*I*4);
        float *y1=malloc((size_t)S*O*4),*y4=malloc((size_t)S*O*4);
        for(int64_t i=0;i<(int64_t)I*O;i++)W[i]=f32_to_bf16_rz(small_int());
        for(int i=0;i<S*I;i++)x[i]=small_int();
        mm_1acc(y1,x,W,S,I,O); mm_acc4(y4,x,W,S,I,O);
        for(int i=0;i<S*O;i++) if(memcmp(&y1[i],&y4[i],4)) bad++;
        printf("D  S=32 exact-arithmetic bit-identity over %d outputs: %s (%d mismatches)\n",
               S*O,bad?"FAIL":"PASS",bad);
        failA+=bad;
        free(W);free(x);free(y1);free(y4);
    }

    printf("\nVERDICT: %s -- the four-accumulator kernel sums the SAME products as the\n"
           "         one-accumulator kernel; only the order differs.\n",
           failA?"FAILED":"PASSED");
    return failA?1:0;
}
