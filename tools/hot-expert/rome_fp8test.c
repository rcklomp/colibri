/* Verifies the patched quant.h F16C e4m3 decode against the scalar LUT and the
 * previous bit-manipulation decode (exhaustive over 256 codes), checks matmul
 * outputs are bit-identical, and benchmarks old vs new decode inside the same
 * matmul_fp8 loop on Qwen3.8 expert shapes with a >L3 pool.
 * Build (against the PATCHED header): gcc -O3 -march=native -fopenmp -I<c> fp8test.c -o fp8test -lm
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
#ifndef E4M3_HAVE_F16C_DECODE
#error "patched quant.h with F16C decode expected"
#endif

/* the previous kernel body, with the previous decode */
static void matmul_fp8_bits(float *y, const float *x, const uint8_t *q8, const float *bscale, int S, int I, int O){
    int64_t nblkI = fp8_nblk(I);
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){
        const uint8_t *w = q8 + (int64_t)o*I; const float *scl = bscale + (o/FP8_BLOCK)*nblkI;
        for(int s=0;s<S;s++){
            const float *xs = x + (int64_t)s*I; double a=0;
            for(int64_t bi=0; bi*FP8_BLOCK<I; bi++){
                int base=(int)(bi*FP8_BLOCK), blen=FP8_BLOCK; if(base+blen>I) blen=I-base;
                __m256 vacc=_mm256_setzero_ps(); int i=base;
                for(;i+8<=base+blen;i+=8) vacc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), e4m3x8_to_f32x8_bits(_mm_loadl_epi64((const __m128i*)(w+i))), vacc);
                float buf[8]; _mm256_storeu_ps(buf,vacc); float acc=buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
                for(;i<base+blen;i++) acc += e4m3_decode(w[i])*xs[i];
                a += (double)acc*scl[bi];
            }
            y[(int64_t)s*O+o]=(float)a;
        }
    }
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

int main(void){
    int bad=0;
    for(int c=0;c<256;c+=8){
        uint8_t in[8]; for(int k=0;k<8;k++)in[k]=(uint8_t)(c+k);
        float a[8],b[8];
        _mm256_storeu_ps(a,e4m3x8_to_f32x8_bits(_mm_loadl_epi64((const __m128i*)in)));
        _mm256_storeu_ps(b,e4m3x8_to_f32x8(_mm_loadl_epi64((const __m128i*)in)));
        for(int k=0;k<8;k++){
            int code=c+k; float l=e4m3_decode((uint8_t)code);
            uint32_t ua,ub,ul; memcpy(&ua,&a[k],4); memcpy(&ub,&b[k],4); memcpy(&ul,&l,4);
            if((code&0x7f)==0x7f){ if(!isnan(b[k])){printf("NaN code 0x%02x not NaN: %a\n",code,b[k]);bad++;} continue; }
            if(ub!=ul){printf("MISMATCH vs LUT code 0x%02x: lut=%a f16c=%a\n",code,l,b[k]);bad++;}
            if(ua!=ul){printf("(old bits decode differs from LUT at 0x%02x: %a vs %a)\n",code,a[k],l);}
        }
    }
    printf("decode exhaustive vs LUT: %s (%d mismatches)\n",bad?"FAIL":"PASS",bad);
    srand(3);
    int shapes[2][2]={{2560,640},{640,2560}};
    for(int sh=0;sh<2;sh++){
        int I=shapes[sh][0],O=shapes[sh][1]; size_t n=(size_t)I*O, ns=fp8_nblk(I)*fp8_nblk(O);
        uint8_t *w=malloc(n); for(size_t i=0;i<n;i++){uint8_t v=rand()&0xff; w[i]=((v&0x7f)==0x7f)?0x38:v;}
        float *sc=malloc(ns*4); for(size_t i=0;i<ns;i++)sc[i]=0.001f+(rand()%1000)/1e5f;
        float *x=malloc(I*4); for(int i=0;i<I;i++)x[i]=(rand()%2000-1000)/300.0f;
        float *y1=malloc(O*4),*y2=malloc(O*4);
        for(int S=1;S<=4;S+=3){ float *xs=malloc((size_t)S*I*4); for(int i=0;i<S*I;i++)xs[i]=x[i%I]*(1+0.01f*(i/I));
            float *ya=malloc((size_t)S*O*4),*yb=malloc((size_t)S*O*4);
            matmul_fp8_bits(ya,xs,w,sc,S,I,O); matmul_fp8(yb,xs,w,sc,S,I,O);
            int diff=0; for(int o=0;o<S*O;o++) if(memcmp(&ya[o],&yb[o],4)) diff++;
            printf("matmul %dx%d S=%d bit-identical: %d/%d differ\n",I,O,S,diff,S*O); bad|=diff!=0; free(xs);free(ya);free(yb);}
        free(w);free(sc);free(x);free(y1);free(y2);
    }
    int H=2560,Iw=640,K=10,P=128; size_t mb=(size_t)H*Iw, nsc=fp8_nblk(Iw)*fp8_nblk(H);
    uint8_t *pool=aligned_alloc(64,mb*3*P);
    #pragma omp parallel for
    for(size_t i=0;i<mb*3*P;i++){uint8_t v=(uint8_t)(i*2654435761u>>24);pool[i]=(v&0x7f)>=0x7e?0x38:v;}
    float *sc=malloc(nsc*3*P*4);for(size_t i=0;i<nsc*3*P;i++)sc[i]=0.002f;
    float *x=malloc(H*4);for(int i=0;i<H;i++)x[i]=(i%5)*0.1f-0.2f;
    float *eg=malloc(Iw*4),*eu=malloc(Iw*4),*eh=malloc(Iw*4),*eo=malloc(H*4);
    for(int variant=0;variant<2;variant++){
        void (*mm)(float*,const float*,const uint8_t*,const float*,int,int,int)=variant?matmul_fp8:matmul_fp8_bits;
        int rot=0; double t0=0; int reps=60;
        for(int r=-5;r<reps;r++){ if(r==0)t0=now();
            for(int z=0;z<K;z++){int e=(rot++)%P;const uint8_t *g=pool+(size_t)e*3*mb,*u=g+mb,*d=u+mb;const float *sg=sc+(size_t)e*3*nsc,*su=sg+nsc,*sd=su+nsc;
                mm(eg,x,g,sg,1,H,Iw);mm(eu,x,u,su,1,H,Iw);for(int j=0;j<Iw;j++)eh[j]=eg[j]/(1+expf(-eg[j]))*eu[j];mm(eo,eh,d,sd,1,Iw,H);}
        }
        double ms=(now()-t0)*1e3/reps;
        printf("%-10s K=10 (threads=%d): %.3f ms/layer  %.3f ms/expert  %.1f GB/s  -> %.0f ms/token (48 layers)\n",variant?"f16c":"old-bits",omp_get_max_threads(),ms,ms/K,mb*3.0*K/ms/1e6,ms*48);
    }
    return bad?1:0;
}
