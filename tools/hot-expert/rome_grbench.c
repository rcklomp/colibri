/* rome_grbench.c -- the isolated microbenchmark for roadmap item Q2
 * (the gated residual's serial tail), plus its bit-identity oracle.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o /tmp/rome_grbench rome_grbench.c -lm
 * Run (8 threads, rig lock held, no engine up -- like every other row this
 * family of harnesses has produced):
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/rome_grbench
 *
 * WHAT IT MEASURES.  q38_gr_read at Qwen3.8-Flash-Next's real decode shapes
 * (hidden H=2560, hyper-columns C=4, hyper width W=C*H=10240, rank R=320,
 * S=1 row, 97 call sites per token = 48 layers x 2 + the final one), in the
 * engine's own order:
 *
 *   rms      for b < C=4: q38_rms0 over H=2560 -- a SERIAL double
 *            sum-of-squares (a dependent double add per element, ~4 cycles of
 *            latency each) and 2560 scalings.  Q2's first target.
 *   down     q38_matmul_bf16, W=10240 -> R=320.  6.55 MB of BF16 per site,
 *            8 threads.  NOT Q2's target: bandwidth-bound.
 *   lowsilu  for z < R=320: silu(low[z]/C).  Serial, 320 expf.
 *   up       q38_matmul_bf16, R=320 -> W=10240.  6.55 MB, 8 threads.  NOT
 *            Q2's target.
 *   mix      for d < H=2560: sum over b < C=4 of sigmoid(mix[b*H+d]) *
 *            norm[b*H+d], then /C.  SERIAL, and 10 240 expf per site =
 *            ~994k expf per token over 97 sites.  Q2's second target.
 *   inject   q38_matmul_bf16, W=10240 -> C=4 (96 of the 97 sites), then four
 *            2*sigmoid.  NOT Q2's target.
 *
 * The bodies are COPIED VERBATIM from c/qwen38_core.h (q38_gr_read, q38_rms0,
 * q38_sigmoid, q38_silu, q38_matmul_bf16's AVX2 arm) so the isolated figure is
 * the engine's own arithmetic and not a lookalike.
 *
 * WORKING SET -- and unlike Q1's rome_dnbench.c this harness does NOT have to
 * argue about the "pools bigger than L3" rule, because the real path streams
 * weights: 97 sites x (6.55 + 6.55 + 0.08 MB) = 1.28 GB of BF16 per token,
 * ten times the 128 MB L3, and the default configuration allocates exactly
 * that -- one distinct weight set per site, as in the engine.  The scratch the
 * target loops touch (norm, mix, hyper: 120 KB) is reused across sites, again
 * as in the engine, where three falloc/free pairs per call hand back the same
 * heap block.  A second configuration ("L3-hot weights") makes all 97 sites
 * share one 13.2 MB weight set; it is reported only to show how much of the
 * result is the weight stream.  The engine's own gr-* sub-timers (Q2 step 0)
 * are the arbiter.
 *
 * THE ORACLE.  Before any timing, every variant is run against the serial one
 * from identical inputs and compared with memcmp on all three outputs of all
 * 97 sites over 4 consecutive tokens -- the internal `norm` (which is what the
 * rms arm changes), `mixed` (the mix arm) and `inject`.  Bit-identical or the
 * variant does not ship: rome_fp8test.c's pattern, and Q2's gate.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;}

/* --- verbatim from c/qwen38_core.h ---------------------------------------- */
static inline float q38_sigmoid(float x) {
    if (x >= 0.f) { float z=expf(-x); return 1.f/(1.f+z); }
    float z=expf(x); return z/(1.f+z);
}
static inline float q38_silu(float x) { return x * q38_sigmoid(x); }
static void q38_rms0(float *out, const float *x, const float *w, int n, float eps) {
    double ss=0.0; for (int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for (int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}
static inline float bf16_to_f32(uint16_t h){uint32_t u=(uint32_t)h<<16;float f;memcpy(&f,&u,4);return f;}
#ifdef __AVX2__
static inline __m256 q38_bf16x8_to_f32x8(__m128i raw) {
    __m256i widened=_mm256_cvtepu16_epi32(raw);
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
}
/* -------------------------------------------------------------------------- */

enum { H=2560, C=4, W=C*H, R=320, SITES=97 };
#define EPS 1e-6f

typedef struct {          /* one call site's gated residual weights */
    float *norm;          /* [W]      the RMSNorm scale (1+w) */
    uint16_t *down;       /* [R,W]    BF16 */
    uint16_t *up;         /* [W,R]    BF16 */
    uint16_t *inject;     /* [C,W]    BF16 */
} Site;

static uint64_t rng=0x2026091200000002ull;
static inline double rnd(void){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;
    return ((double)((rng*0x2545F4914F6CDD1Dull)>>11)/9007199254740992.0);}
static inline float rndf(float lo,float hi){return (float)(lo+(hi-lo)*rnd());}
static uint16_t f32_to_bf16(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)(u>>16);}

/* ==== the arms ============================================================ */
/* Every arm computes the same values in the same order.  The only thing that
 * changes is which thread runs which iteration:
 *   - rms: parallel over b (4 ways at C=4).  Each q38_rms0 call keeps its own
 *     serial double accumulation, so no reduction order changes.
 *   - rms-split: phase 1 computes the four r's in parallel over b (the serial
 *     double sums, unchanged), phase 2 applies them over all W=10240 elements
 *     8 ways.  Element expressions are identical.
 *   - mix: parallel over d.  The inner `for b<C` sum stays serial per d, so
 *     every mixed[d] accumulates in the same order as today.
 * Writes: norm[b*H..] per b (disjoint), mixed[d] per d (disjoint).  Reads:
 * hyper, the norm weight, mix.  Nothing is written twice and nothing a loop
 * writes is read by the same loop. */

static void gr_rms_serial(float *norm,const float *hyper,const float *w){
    for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)b*H,hyper+(int64_t)b*H,w+(int64_t)b*H,H,EPS);
}
static void gr_rms_par_b(float *norm,const float *hyper,const float *w){
    #pragma omp parallel for schedule(static)
    for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)b*H,hyper+(int64_t)b*H,w+(int64_t)b*H,H,EPS);
}
static void gr_rms_par_split(float *norm,const float *hyper,const float *w){
    float rs[C];
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int b=0;b<C;b++){
            const float *x=hyper+(int64_t)b*H;
            double ss=0.0; for(int i=0;i<H;i++) ss+=(double)x[i]*x[i];
            rs[b]=1.f/sqrtf((float)(ss/H)+EPS);
        }
        #pragma omp for schedule(static)
        for(int z=0;z<W;z++) norm[z]=hyper[z]*rs[z/H]*(1.f+w[z]);
    }
}
static void gr_mix_serial(float *mixed,const float *mix,const float *norm){
    for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)b*H+d])*norm[(int64_t)b*H+d];
        mixed[d]=v/C;
    }
}
static void gr_mix_par_d(float *mixed,const float *mix,const float *norm){
    #pragma omp parallel for schedule(static)
    for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)b*H+d])*norm[(int64_t)b*H+d];
        mixed[d]=v/C;
    }
}

/* rms arm selector: 0 serial, 1 parallel over b, 2 parallel split */
/* mix arm selector: 0 serial, 1 parallel over d                   */
typedef struct { double rms,down,lowsilu,up,mix,inject,total; } Split;

static void gr_read(const Site *st,const float *hyper,float *norm,float *low,
                    float *mixbuf,float *mixed,float *inject,int do_inject,
                    int rms_arm,int mix_arm,Split *acc){
    double t=now(),t0=t;
    if(rms_arm==0)      gr_rms_serial(norm,hyper,st->norm);
    else if(rms_arm==1) gr_rms_par_b(norm,hyper,st->norm);
    else                gr_rms_par_split(norm,hyper,st->norm);
    if(acc){double n=now();acc->rms+=n-t;t=n;}
    q38_matmul_bf16(low,norm,st->down,1,W,R);
    if(acc){double n=now();acc->down+=n-t;t=n;}
    for(int z=0;z<R;z++) low[z]=q38_silu(low[z]/C);
    if(acc){double n=now();acc->lowsilu+=n-t;t=n;}
    q38_matmul_bf16(mixbuf,low,st->up,1,R,W);
    if(acc){double n=now();acc->up+=n-t;t=n;}
    if(mix_arm==0) gr_mix_serial(mixed,mixbuf,norm);
    else           gr_mix_par_d(mixed,mixbuf,norm);
    if(acc){double n=now();acc->mix+=n-t;t=n;}
    if(do_inject){
        q38_matmul_bf16(inject,norm,st->inject,1,W,C);
        for(int z=0;z<C;z++) inject[z]=2.f*q38_sigmoid(inject[z]/C);
        if(acc){double n=now();acc->inject+=n-t;t=n;}
    }
    if(acc) acc->total+=now()-t0;
}

/* ==== driver ============================================================== */
static Site *sites_alloc(int distinct){
    Site *s=(Site*)calloc(SITES,sizeof(Site));
    for(int i=0;i<distinct;i++){
        s[i].norm=(float*)malloc((size_t)W*sizeof(float));
        s[i].down=(uint16_t*)malloc((size_t)R*W*sizeof(uint16_t));
        s[i].up=(uint16_t*)malloc((size_t)W*R*sizeof(uint16_t));
        s[i].inject=(uint16_t*)malloc((size_t)C*W*sizeof(uint16_t));
        if(!s[i].norm||!s[i].down||!s[i].up||!s[i].inject){
            fprintf(stderr,"OOM building site %d\n",i);exit(1);}
        for(int z=0;z<W;z++) s[i].norm[z]=rndf(-0.2f,0.2f);
        for(int64_t z=0;z<(int64_t)R*W;z++) s[i].down[z]=f32_to_bf16(rndf(-0.04f,0.04f));
        for(int64_t z=0;z<(int64_t)W*R;z++) s[i].up[z]=f32_to_bf16(rndf(-0.04f,0.04f));
        for(int64_t z=0;z<(int64_t)C*W;z++) s[i].inject[z]=f32_to_bf16(rndf(-0.04f,0.04f));
    }
    for(int i=distinct;i<SITES;i++) s[i]=s[i%distinct];
    return s;
}

static void token(const Site *sites,const float *hyper_bank,float *norm,float *low,
                  float *mixbuf,float *mixed,float *inject,int rms_arm,int mix_arm,
                  Split *acc,float *cap_norm,float *cap_mixed,float *cap_inject){
    for(int i=0;i<SITES;i++){
        const float *hyper=hyper_bank+(int64_t)(i%8)*W;
        int do_inject = (i<SITES-1);
        gr_read(&sites[i],hyper,norm,low,mixbuf,mixed,inject,do_inject,
                rms_arm,mix_arm,acc);
        if(cap_norm){
            memcpy(cap_norm+(int64_t)i*W,norm,(size_t)W*sizeof(float));
            memcpy(cap_mixed+(int64_t)i*H,mixed,(size_t)H*sizeof(float));
            memcpy(cap_inject+(int64_t)i*C,inject,(size_t)C*sizeof(float));
        }
    }
}

int main(void){
    int threads=1;
#ifdef _OPENMP
    #pragma omp parallel
    { if(omp_get_thread_num()==0) threads=omp_get_num_threads(); }
#endif
    printf("rome_grbench: gated residual (q38_gr_read) at the engine's decode shapes\n");
    printf("  H=%d C=%d W=%d R=%d sites/token=%d threads=%d\n",H,C,W,R,SITES,threads);
    double wbytes=(double)SITES*((double)R*W+(double)W*R+(double)C*W)*2.0;
    printf("  weight stream per token (97 distinct sites): %.2f MB  (L3 is 128 MB)\n",
           wbytes/1048576.0);

    float *hyper=(float*)malloc((size_t)8*W*sizeof(float));
    for(int64_t z=0;z<(int64_t)8*W;z++) hyper[z]=rndf(-3.f,3.f);
    float *norm=(float*)malloc((size_t)W*sizeof(float));
    float *low=(float*)malloc((size_t)R*sizeof(float));
    float *mixbuf=(float*)malloc((size_t)W*sizeof(float));
    float *mixed=(float*)malloc((size_t)H*sizeof(float));
    float *inject=(float*)malloc((size_t)C*sizeof(float));

    printf("building 97 distinct weight sets (%.2f MB)...\n",wbytes/1048576.0);
    Site *big=sites_alloc(SITES);
    printf("building the L3-hot (1 shared weight set) configuration...\n");
    Site *hot=sites_alloc(1);

    /* ---- the oracle, before anything is timed --------------------------- */
    const char *arm_name[]={"serial (today)","rms par-b + mix par-d",
                            "rms par-split + mix par-d","mix par-d only",
                            "rms par-b only"};
    const int rms_arm[]={0,1,2,0,1}, mix_arm[]={0,1,1,1,0};
    const int NARM=5, TOK=4;
    size_t cn=(size_t)SITES*W, cm=(size_t)SITES*H, ci=(size_t)SITES*C;
    float *ref_n=(float*)malloc(cn*TOK*sizeof(float));
    float *ref_m=(float*)malloc(cm*TOK*sizeof(float));
    float *ref_i=(float*)malloc(ci*TOK*sizeof(float));
    float *got_n=(float*)malloc(cn*TOK*sizeof(float));
    float *got_m=(float*)malloc(cm*TOK*sizeof(float));
    float *got_i=(float*)malloc(ci*TOK*sizeof(float));
    if(!ref_n||!ref_m||!ref_i||!got_n||!got_m||!got_i){fprintf(stderr,"OOM oracle\n");return 1;}
    /* one fixed bank of inputs, built once and reused by every arm: the oracle
     * must compare arms on identical state, not on identical RNG bookkeeping */
    float *hb=(float*)malloc((size_t)TOK*8*W*sizeof(float));
    if(!hb){fprintf(stderr,"OOM oracle inputs\n");return 1;}
    for(int64_t z=0;z<(int64_t)TOK*8*W;z++) hb[z]=rndf(-3.f,3.f);
    for(int t=0;t<TOK;t++)
        token(big,hb+(size_t)t*8*W,norm,low,mixbuf,mixed,inject,0,0,NULL,
              ref_n+(size_t)t*cn,ref_m+(size_t)t*cm,ref_i+(size_t)t*ci);
    int all_ok=1;
    for(int a=1;a<NARM;a++){
        for(int t=0;t<TOK;t++)
            token(big,hb+(size_t)t*8*W,norm,low,mixbuf,mixed,inject,rms_arm[a],mix_arm[a],NULL,
                  got_n+(size_t)t*cn,got_m+(size_t)t*cm,got_i+(size_t)t*ci);
        int on=memcmp(ref_n,got_n,cn*TOK*sizeof(float))==0;
        int om=memcmp(ref_m,got_m,cm*TOK*sizeof(float))==0;
        int oi=memcmp(ref_i,got_i,ci*TOK*sizeof(float))==0;
        printf("ORACLE %-28s norm %s  mixed %s  inject %s  (%d tokens x %d sites)\n",
               arm_name[a],on?"IDENTICAL":"DIFFER",om?"IDENTICAL":"DIFFER",
               oi?"IDENTICAL":"DIFFER",TOK,SITES);
        if(!(on&&om&&oi)) all_ok=0;
    }
    printf("ORACLE verdict: %s\n",all_ok?"all arms BIT-IDENTICAL":"A VARIANT DIFFERS -- do not ship it");
    free(ref_n);free(ref_m);free(ref_i);free(got_n);free(got_m);free(got_i);free(hb);

    /* ---- timing -------------------------------------------------------- */
    const int WARM=2, MEAS=8, REPS=2;
    for(int cfg=0;cfg<2;cfg++){
        const Site *S_=cfg?hot:big;
        printf("\n=== %s ===\n",cfg?"L3-hot weights (one 13.2 MB set shared by all 97 sites)"
                                 :"97 distinct weight sets (1.28 GB/token -- the engine's stream)");
        printf("%-28s %9s %9s %9s %9s %9s %9s %9s\n","arm (ms/token)","rms","down",
               "lowsilu","up","mix","inject","TOTAL");
        for(int rep=0;rep<REPS;rep++){
            for(int a=0;a<NARM;a++){
                Split acc={0};
                for(int t=0;t<WARM;t++)
                    token(S_,hyper,norm,low,mixbuf,mixed,inject,rms_arm[a],mix_arm[a],NULL,NULL,NULL,NULL);
                for(int t=0;t<MEAS;t++)
                    token(S_,hyper,norm,low,mixbuf,mixed,inject,rms_arm[a],mix_arm[a],&acc,NULL,NULL,NULL);
                double k=1000.0/MEAS;
                printf("%-28s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f%s\n",
                       arm_name[a],acc.rms*k,acc.down*k,acc.lowsilu*k,acc.up*k,
                       acc.mix*k,acc.inject*k,acc.total*k,rep?"  (repeat)":"");
            }
        }
    }
    printf("\nread rms+mix together: they are the two loops Q2 names; down/up/inject\n"
           "are the matmuls it does not touch and are reported to show they do not move.\n");
    return 0;
}
