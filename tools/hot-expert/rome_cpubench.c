/* CPU placement microbench for Qwen3.8 on EPYC 7F32 (Zen 2, AVX2+FMA).
 * Uses the engine's real kernels: matmul_fp8 from quant.h (block-scaled E4M3)
 * and a copy of q38_matmul_bf16 from qwen38_core.h. Working sets are rotated
 * through pools larger than the 128 MB L3 so numbers are DRAM-streaming, which
 * is what one decode token sees.
 *
 * Build: gcc -O3 -march=native -fopenmp cpubench.c -o cpubench -lm
 * Run:   OMP_NUM_THREADS=16 ./cpubench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include "quant.h"

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static inline float bf16_to_f32x(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}

/* ---- engine kernel, verbatim from qwen38_core.h ---- */
static inline __m256 bf16x8_to_f32x8(__m128i h){__m256i w=_mm256_cvtepu16_epi32(h);return _mm256_castsi256_ps(_mm256_slli_epi32(w,16));}
static void matmul_bf16_engine(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m256 vacc=_mm256_setzero_ps();int i=0;
            for(;i+8<=I;i+=8){
                __m128i wh=_mm_loadu_si128((const __m128i*)(w+i));
                __m256 wf=bf16x8_to_f32x8(wh);
                __m256 xf=_mm256_loadu_ps(xs+i);
                vacc=_mm256_fmadd_ps(xf,wf,vacc);
            }
            float buf[8];_mm256_storeu_ps(buf,vacc);
            float a=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
            for(;i<I;i++)a+=xs[i]*bf16_to_f32x(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}
/* ---- candidate: 4 independent accumulators (breaks the FMA latency chain) ---- */
static void matmul_bf16_acc4(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;
        for(int s=0;s<S;s++){
            const float *xs=x+(int64_t)s*I;
            __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;int i=0;
            for(;i+32<=I;i+=32){
                a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i))),a0);
                a1=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+8))),a1);
                a2=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+16),bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+16))),a2);
                a3=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+24),bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i+24))),a3);
            }
            for(;i+8<=I;i+=8) a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i))),a0);
            a0=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            float buf[8];_mm256_storeu_ps(buf,a0);
            float a=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
            for(;i<I;i++)a+=xs[i]*bf16_to_f32x(w[i]);
            y[(int64_t)s*O+o]=a;
        }
    }
}
/* ---- candidate: prefill-oriented, 4 rows of x per weight pass (weights read once per 4 rows) ---- */
static void matmul_bf16_s4(float *y,const float *x,const uint16_t *W,int S,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint16_t *w=W+(int64_t)o*I;int s=0;
        for(;s+4<=S;s+=4){
            const float *x0=x+(int64_t)s*I,*x1=x0+I,*x2=x1+I,*x3=x2+I;
            __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;int i=0;
            for(;i+8<=I;i+=8){
                __m256 wf=bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i)));
                a0=_mm256_fmadd_ps(_mm256_loadu_ps(x0+i),wf,a0);
                a1=_mm256_fmadd_ps(_mm256_loadu_ps(x1+i),wf,a1);
                a2=_mm256_fmadd_ps(_mm256_loadu_ps(x2+i),wf,a2);
                a3=_mm256_fmadd_ps(_mm256_loadu_ps(x3+i),wf,a3);
            }
            float b[4][8];_mm256_storeu_ps(b[0],a0);_mm256_storeu_ps(b[1],a1);_mm256_storeu_ps(b[2],a2);_mm256_storeu_ps(b[3],a3);
            for(int r=0;r<4;r++){float a=0;for(int k=0;k<8;k++)a+=b[r][k];for(int j=i;j<I;j++)a+=x[(int64_t)(s+r)*I+j]*bf16_to_f32x(w[j]);y[(int64_t)(s+r)*O+o]=a;}
        }
        for(;s<S;s++){
            const float *xs=x+(int64_t)s*I;__m256 a0=_mm256_setzero_ps();int i=0;
            for(;i+8<=I;i+=8)a0=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),bf16x8_to_f32x8(_mm_loadu_si128((const __m128i*)(w+i))),a0);
            float buf[8];_mm256_storeu_ps(buf,a0);float a=0;for(int k=0;k<8;k++)a+=buf[k];for(;i<I;i++)a+=xs[i]*bf16_to_f32x(w[i]);y[(int64_t)s*O+o]=a;
        }
    }
}
/* ---- candidate: int8 per-row-scaled GEMV via AVX2 maddubs (Zen2 has no VNNI) ---- */
static void matmul_i8_maddubs(float *y,const uint8_t *xq,float xscale,const int8_t *W,const float *ws,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const int8_t *w=W+(int64_t)o*I;
        __m256i acc=_mm256_setzero_si256();const __m256i ones=_mm256_set1_epi16(1);int i=0;
        for(;i+32<=I;i+=32){
            __m256i xv=_mm256_loadu_si256((const __m256i*)(xq+i));
            __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
            __m256i p=_mm256_maddubs_epi16(xv,wv);       /* u8*s8 -> s16 pairs */
            acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
        }
        int32_t b[8];_mm256_storeu_si256((__m256i*)b,acc);int32_t s=0;for(int k=0;k<8;k++)s+=b[k];
        for(;i<I;i++)s+=(int32_t)xq[i]*w[i];
        y[o]=(float)s*xscale*ws[o];
    }
}

typedef struct{const char*name;int I,O;}Shape;

static void bench_bf16(const char *label,int I,int O,int S,int copies,void(*fn)(float*,const float*,const uint16_t*,int,int,int)){
    size_t per=(size_t)I*O;uint16_t *pool=aligned_alloc(64,per*copies*2);
    #pragma omp parallel for
    for(size_t i=0;i<per*copies;i++)pool[i]=(uint16_t)(0x3f00+(i%251));
    float *x=aligned_alloc(64,(size_t)S*I*4);for(int i=0;i<S*I;i++)x[i]=(i%7)*0.1f-0.3f;
    float *y=aligned_alloc(64,(size_t)S*O*4);
    for(int c=0;c<copies;c++)fn(y,x,pool+per*c,S,I,O);          /* warm */
    int reps=copies*3;double t0=now();
    for(int r=0;r<reps;r++)fn(y,x,pool+per*(r%copies),S,I,O);
    double ms=(now()-t0)*1e3/reps;
    printf("BF16 %-28s S=%2d %5dx%-6d %7.3f ms  %6.1f GB/s  %6.1f GFLOP/s\n",label,S,I,O,ms,per*2.0/ms/1e6,2.0*per*S/ms/1e6);
    free(pool);free(x);free(y);
}

static void bench_i8(int I,int O,int copies){
    size_t per=(size_t)I*O;int8_t *pool=aligned_alloc(64,per*copies);
    #pragma omp parallel for
    for(size_t i=0;i<per*copies;i++)pool[i]=(int8_t)(i%127)-63;
    float *ws=malloc(O*4);for(int o=0;o<O;o++)ws[o]=0.01f;
    uint8_t *xq=aligned_alloc(64,I);for(int i=0;i<I;i++)xq[i]=(uint8_t)(i%200);
    float *y=malloc(O*4);
    for(int c=0;c<copies;c++)matmul_i8_maddubs(y,xq,0.01f,pool+per*c,ws,I,O);
    int reps=copies*3;double t0=now();
    for(int r=0;r<reps;r++)matmul_i8_maddubs(y,xq,0.01f,pool+per*(r%copies),ws,I,O);
    double ms=(now()-t0)*1e3/reps;
    printf("INT8 %-28s S= 1 %5dx%-6d %7.3f ms  %6.1f GB/s\n","maddubs per-row scale",I,O,ms,per*1.0/ms/1e6);
    free(pool);free(ws);free(xq);free(y);
}

/* one decode token's routed experts: K experts x (gate,up,down), engine kernel, pool of P experts */
static void bench_fp8_experts(int H,int Iw,int K,int P,int fused_region){
    size_t mb=(size_t)H*Iw;                       /* bytes per matrix */
    size_t nsc=fp8_nblk(Iw)*fp8_nblk(H);
    uint8_t *pool=aligned_alloc(64,mb*3*P);
    #pragma omp parallel for
    for(size_t i=0;i<mb*3*P;i++){uint8_t v=(uint8_t)(i*2654435761u>>24);pool[i]=(v&0x7f)>=0x7e?0x38:v;} /* avoid NaN codes */
    float *sc=malloc(nsc*3*P*4);for(size_t i=0;i<nsc*3*P;i++)sc[i]=0.002f;
    float *x=malloc(H*4);for(int i=0;i<H;i++)x[i]=(i%5)*0.1f-0.2f;
    float *eg=malloc(Iw*4),*eu=malloc(Iw*4),*eh=malloc(Iw*4),*eo=malloc(H*4),*ys=calloc(H,4);
    float *egk=malloc((size_t)K*Iw*4),*euk=malloc((size_t)K*Iw*4),*eok=malloc((size_t)K*H*4);
    int rot=0;
    double t0=0;int reps=40;
    for(int r=-4;r<reps;r++){
        if(r==0)t0=now();
        if(!fused_region){
            for(int z=0;z<K;z++){int e=(rot++)%P;const uint8_t *g=pool+(size_t)e*3*mb,*u=g+mb,*d=u+mb;const float *sg=sc+(size_t)e*3*nsc,*su=sg+nsc,*sd=su+nsc;
                matmul_fp8(eg,x,g,sg,1,H,Iw);matmul_fp8(eu,x,u,su,1,H,Iw);
                for(int j=0;j<Iw;j++)eh[j]=eg[j]/(1+expf(-eg[j]))*eu[j];
                matmul_fp8(eo,eh,d,sd,1,Iw,H);for(int dd=0;dd<H;dd++)ys[dd]+=0.1f*eo[dd];}
        }else{
            /* candidate: all K experts' gate+up rows in ONE parallel region, then all K downs in one */
            int es[64];for(int z=0;z<K;z++)es[z]=(rot++)%P;
            #pragma omp parallel for schedule(static)
            for(int t=0;t<K*2*Iw;t++){int z=t/(2*Iw),rem=t%(2*Iw),which=rem/Iw,o=rem%Iw;int e=es[z];
                const uint8_t *w=pool+(size_t)e*3*mb+(size_t)which*mb+(size_t)o*H;const float *scl=sc+(size_t)e*3*nsc+(size_t)which*nsc+(o/FP8_BLOCK)*fp8_nblk(H);
                double a=0;for(int bi=0;bi*FP8_BLOCK<H;bi++){int base=bi*FP8_BLOCK;__m256 v=_mm256_setzero_ps();for(int i=base;i<base+FP8_BLOCK;i+=8)v=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),e4m3x8_to_f32x8(_mm_loadl_epi64((const __m128i*)(w+i))),v);float b[8];_mm256_storeu_ps(b,v);a+=(double)(b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+b[7])*scl[bi];}
                (which?euk:egk)[(size_t)z*Iw+o]=(float)a;}
            for(int z=0;z<K;z++)for(int j=0;j<Iw;j++)egk[(size_t)z*Iw+j]=egk[(size_t)z*Iw+j]/(1+expf(-egk[(size_t)z*Iw+j]))*euk[(size_t)z*Iw+j];
            #pragma omp parallel for schedule(static)
            for(int t=0;t<K*H;t++){int z=t/H,o=t%H;int e=es[z];const uint8_t *w=pool+(size_t)e*3*mb+2*mb+(size_t)o*Iw;const float *scl=sc+(size_t)e*3*nsc+2*nsc+(o/FP8_BLOCK)*fp8_nblk(Iw);const float *hx=egk+(size_t)z*Iw;
                double a=0;for(int bi=0;bi*FP8_BLOCK<Iw;bi++){int base=bi*FP8_BLOCK;__m256 v=_mm256_setzero_ps();for(int i=base;i<base+FP8_BLOCK;i+=8)v=_mm256_fmadd_ps(_mm256_loadu_ps(hx+i),e4m3x8_to_f32x8(_mm_loadl_epi64((const __m128i*)(w+i))),v);float b[8];_mm256_storeu_ps(b,v);a+=(double)(b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+b[7])*scl[bi];}
                eok[(size_t)z*H+o]=(float)a;}
            for(int z=0;z<K;z++)for(int dd=0;dd<H;dd++)ys[dd]+=0.1f*eok[(size_t)z*H+dd];
        }
    }
    double ms=(now()-t0)*1e3/reps;
    printf("FP8  routed experts K=%d %-14s %7.3f ms/token-layer  (%5.2f ms/expert, %6.1f GB/s) x48 layers = %.0f ms/token\n",K,fused_region?"fused-region":"engine-loop",ms,ms/K,mb*3.0*K/ms/1e6,ms*48);
    free(pool);free(sc);free(x);free(eg);free(eu);free(eh);free(eo);free(ys);free(egk);free(euk);free(eok);
}

int main(int argc,char**argv){
    int th=omp_get_max_threads();
    printf("=== cpubench threads=%d ===\n",th);
    /* Qwen3.8 dense shapes (I=input, O=output). copies sized > 128 MB L3 */
    bench_bf16("engine (1 acc)",2560,10240,1,8,matmul_bf16_engine);   /* dn_qkv 52 MB */
    bench_bf16("acc4",          2560,10240,1,8,matmul_bf16_acc4);
    bench_bf16("engine (1 acc)",2560,6144,1,12,matmul_bf16_engine);    /* dn_z / q-ish */
    bench_bf16("acc4",          2560,6144,1,12,matmul_bf16_acc4);
    bench_bf16("engine (1 acc)",6144,2560,1,12,matmul_bf16_engine);    /* dn_out / o_proj */
    bench_bf16("acc4",          6144,2560,1,12,matmul_bf16_acc4);
    bench_bf16("engine (1 acc)",2560,640,1,128,matmul_bf16_engine);    /* shared expert / small */
    bench_bf16("acc4",          2560,640,1,128,matmul_bf16_acc4);
    bench_bf16("engine (1 acc)",2560,248320,1,2,matmul_bf16_engine);   /* LM head 1.27 GB */
    bench_bf16("acc4",          2560,248320,1,2,matmul_bf16_acc4);
    /* prefill S=32 */
    bench_bf16("engine (1 acc)",2560,10240,32,4,matmul_bf16_engine);
    bench_bf16("acc4",          2560,10240,32,4,matmul_bf16_acc4);
    bench_bf16("s4 (x rows x4)",2560,10240,32,4,matmul_bf16_s4);
    /* int8 dense proxy */
    bench_i8(2560,10240,16);
    bench_i8(6144,2560,24);
    /* routed experts */
    bench_fp8_experts(2560,640,10,128,0);
    bench_fp8_experts(2560,640,10,128,1);
    /* The SHARED expert is one expert-shaped FP8 triple per layer through the
     * same kernel (shared_expert_intermediate_size == moe_intermediate_size ==
     * 640), so K=1 here IS the shared expert's per-layer cost. Added for Q3
     * (2026-09-11): the item moves this much CPU work into the GPU gap, and the
     * x48 column is the isolated prediction for the [OPTIME] shared counter.
     * Pool is 128 experts = 630 MB, well past the 128 MB L3 and 96 MB
     * Infinity Cache, so it streams from DRAM like the engine's does. */
    bench_fp8_experts(2560,640,1,128,0);
    return 0;
}
