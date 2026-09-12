/* rome_grapplybench.c -- the isolated microbenchmark for roadmap item Q11
 * (the gated residual's WRITE-BACK, q38_gr_apply, plus the rms half that §Q2
 * rejected), its bit-identity oracle, and a direct measurement of rho, the
 * cost of one OpenMP region at this engine's cadence.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o /tmp/rome_grapplybench rome_grapplybench.c -lm
 * Run (8 threads, rig lock held, no engine up):
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/rome_grapplybench
 *
 * It is rome_grbench.c (Q2) with three things added, and nothing removed:
 *
 *  1. q38_gr_apply, the loop Q11 names -- `hyper[b*H+d] += inject[b]*block[d]`,
 *     96 sites per token, in the engine's own position: every gr_apply is
 *     immediately followed by a gr_read, which is the adjacency the fused arm
 *     exploits.
 *  2. the FUSED arms: one `omp parallel` containing the write-back's `omp for`
 *     and the rms's `omp for`, so the pair pays one fork/join instead of two,
 *     and (in the `b` variant) thread b reads back in the rms exactly the
 *     10 KB it has just written in the apply.
 *  3. a direct rho probe -- 96 empty regions per token, timed at the same
 *     cadence -- because Q11's whole verdict turns on the region cost and
 *     §Q11 step 0 refuses to import a constant from another item's bucket.
 *
 * WORKING SET.  97 distinct weight sets = 1.28 GB of BF16 streamed per token,
 * ten times the 128 MB L3, exactly as rome_grbench.c: the weights are never
 * cache-resident and the `hyper`/`norm`/`mix` scratch (120 KB) is reused
 * across sites as it is in the engine.  BALLAST_MB (default 24) streams that
 * many MB, eight-threaded, between a site's gr_read and its gr_apply, standing
 * in for the attention/moe the engine runs there; it is what decides whether
 * `hyper` is still in anybody's cache when the write-back arrives, so it is a
 * knob and both extremes are reported.
 *
 * THE ORACLE.  Every arm is run against the serial one from identical inputs
 * and compared with memcmp over 4 tokens x 97 sites on FOUR outputs: `hyper`
 * itself (which is what the write-back changes and what rome_grbench.c did not
 * have), `norm`, `mixed` and `inject`.  Bit-identical or it does not ship.
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

typedef struct { float *norm; uint16_t *down,*up,*inject; } Site;

static uint64_t rng=0x2026091200000011ull;
static inline double rnd(void){rng^=rng>>12;rng^=rng<<25;rng^=rng>>27;
    return ((double)((rng*0x2545F4914F6CDD1Dull)>>11)/9007199254740992.0);}
static inline float rndf(float lo,float hi){return (float)(lo+(hi-lo)*rnd());}
static uint16_t f32_to_bf16(float f){uint32_t u;memcpy(&u,&f,4);return (uint16_t)(u>>16);}

/* ==== the arms ============================================================ */
/* Element expressions are IDENTICAL in every arm.  The write-back writes each
 * hyper element exactly once (no accumulation across iterations), so any split
 * of (b,d) is bit-identical by construction; the rms keeps its own serial
 * double sum per column, so parallelising over b changes no summation order. */

static void gr_rms_serial(float *norm,const float *hyper,const float *w){
    for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)b*H,hyper+(int64_t)b*H,w+(int64_t)b*H,H,EPS);
}
static void gr_rms_par_b(float *norm,const float *hyper,const float *w){
    #pragma omp parallel for schedule(static)
    for(int b=0;b<C;b++)
        q38_rms0(norm+(int64_t)b*H,hyper+(int64_t)b*H,w+(int64_t)b*H,H,EPS);
}
static void gr_mix_par_d(float *mixed,const float *mix,const float *norm){
    #pragma omp parallel for schedule(static)
    for(int d=0;d<H;d++){
        float v=0.f;
        for(int b=0;b<C;b++) v+=q38_sigmoid(mix[(int64_t)b*H+d])*norm[(int64_t)b*H+d];
        mixed[d]=v/C;
    }
}
/* the write-back, exactly as c/qwen38_core.h has it (S=1 at decode) */
static void gr_apply_serial(float *hyper,const float *block,const float *inj){
    for(int b=0;b<C;b++){
        float a=inj[b];
        for(int d=0;d<H;d++) hyper[(int64_t)b*H+d]+=a*block[d];
    }
}
static void gr_apply_par_bd(float *hyper,const float *block,const float *inj){
    #pragma omp parallel for collapse(2) schedule(static)
    for(int b=0;b<C;b++) for(int d=0;d<H;d++)
        hyper[(int64_t)b*H+d]+=inj[b]*block[d];
}
/* the fused arm: ONE region, the write-back's `omp for` then the rms's.
 * split==0: both over b (four ways, identical static schedules, so thread b
 *           reads back in the rms the 10 KB it just wrote in the apply);
 * split==1: the apply over (b,d) eight ways, the rms over b four ways. */
static void gr_apply_rms_fused(float *hyper,const float *block,const float *inj,
                               int have_apply,float *norm,const float *w,int split,
                               double *t_apply_end){
    #pragma omp parallel
    {
        if(have_apply){
            if(split){
                #pragma omp for collapse(2) schedule(static)
                for(int b=0;b<C;b++) for(int d=0;d<H;d++)
                    hyper[(int64_t)b*H+d]+=inj[b]*block[d];
            } else {
                #pragma omp for schedule(static)
                for(int b=0;b<C;b++){
                    float a=inj[b];
                    for(int d=0;d<H;d++) hyper[(int64_t)b*H+d]+=a*block[d];
                }
            }
        }
        #pragma omp master
        { if(t_apply_end) *t_apply_end=now(); }
        #pragma omp for schedule(static)
        for(int b=0;b<C;b++)
            q38_rms0(norm+(int64_t)b*H,hyper+(int64_t)b*H,w+(int64_t)b*H,H,EPS);
    }
}

typedef struct { double apply,rms,down,lowsilu,up,mix,inject,ballast,total; } Split;

/* arm table */
enum { A_SERIAL=0, A_RMSPAR, A_APPLYPAR, A_BOTHPAR, A_FUSED_B, A_FUSED_BD, NARM };
static const char *arm_name[NARM]={
    "serial (today)",
    "rms par-b only (Q2 rejected)",
    "apply par-bd only",
    "rms par-b + apply par-bd",
    "FUSED one region, both par-b",
    "FUSED one region, apply par-bd",
};

static float *ballast=NULL; static int64_t ballast_n=0;
static volatile double ballast_sink=0;
static void run_ballast(void){
    if(ballast_n<=0) return;
    double s=0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for(int64_t z=0;z<ballast_n;z++) s+=ballast[z];
    ballast_sink=s;
}

/* One site, in the engine's order: the PENDING write-back from the previous
 * site (fused into this site's region, or run on its own), then the read. */
static void site_run(const Site *st,float *hyper,const float *block,const float *inj_prev,
                     int have_apply,float *norm,float *low,float *mixbuf,float *mixed,
                     float *inject,int do_inject,int arm,Split *acc){
    double t=now(),t0=t;
    if(arm==A_FUSED_B||arm==A_FUSED_BD){
        double tae=0;
        gr_apply_rms_fused(hyper,block,inj_prev,have_apply,norm,st->norm,
                           arm==A_FUSED_BD,&tae);
        if(acc){double n=now();
            /* the master's timestamp after the apply's implicit barrier is the
             * split point, the same instrument the engine's sub-timers use */
            if(have_apply){acc->apply+=tae-t; acc->rms+=n-tae;} else acc->rms+=n-t;
            t=n;}
    } else {
        if(have_apply){
            if(arm==A_APPLYPAR||arm==A_BOTHPAR) gr_apply_par_bd(hyper,block,inj_prev);
            else                                gr_apply_serial(hyper,block,inj_prev);
            if(acc){double n=now();acc->apply+=n-t;t=n;}
        }
        if(arm==A_RMSPAR||arm==A_BOTHPAR) gr_rms_par_b(norm,hyper,st->norm);
        else                              gr_rms_serial(norm,hyper,st->norm);
        if(acc){double n=now();acc->rms+=n-t;t=n;}
    }
    q38_matmul_bf16(low,norm,st->down,1,W,R);
    if(acc){double n=now();acc->down+=n-t;t=n;}
    for(int z=0;z<R;z++) low[z]=q38_silu(low[z]/C);
    if(acc){double n=now();acc->lowsilu+=n-t;t=n;}
    q38_matmul_bf16(mixbuf,low,st->up,1,R,W);
    if(acc){double n=now();acc->up+=n-t;t=n;}
    gr_mix_par_d(mixed,mixbuf,norm);
    if(acc){double n=now();acc->mix+=n-t;t=n;}
    if(do_inject){
        q38_matmul_bf16(inject,norm,st->inject,1,W,C);
        for(int z=0;z<C;z++) inject[z]=2.f*q38_sigmoid(inject[z]/C);
        if(acc){double n=now();acc->inject+=n-t;t=n;}
    }
    run_ballast();   /* stands in for the attention/moe between read and apply */
    if(acc){double n=now();acc->ballast+=n-t;t=n;}
    if(acc) acc->total+=now()-t0;
}

/* one token = 97 sites; `hyper` is carried, as it is in the engine */
static void token(const Site *sites,float *hyper,const float *hyper_seed,
                  float *blockbank,float *norm,float *low,float *mixbuf,float *mixed,
                  float *inject,int arm,Split *acc,
                  float *cap_h,float *cap_n,float *cap_m,float *cap_i){
    memcpy(hyper,hyper_seed,(size_t)W*sizeof(float));
    float inj_prev[C]={0,0,0,0};
    for(int i=0;i<SITES;i++){
        const float *block=blockbank+(int64_t)(i%8)*H;
        site_run(&sites[i],hyper,block,inj_prev,i>0,norm,low,mixbuf,mixed,inject,
                 i<SITES-1,arm,acc);
        if(cap_h){
            memcpy(cap_h+(int64_t)i*W,hyper,(size_t)W*sizeof(float));
            memcpy(cap_n+(int64_t)i*W,norm,(size_t)W*sizeof(float));
            memcpy(cap_m+(int64_t)i*H,mixed,(size_t)H*sizeof(float));
            memcpy(cap_i+(int64_t)i*C,inject,(size_t)C*sizeof(float));
        }
        if(i<SITES-1) memcpy(inj_prev,inject,sizeof inj_prev);
    }
}

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

/* rho: the cost of entering and leaving one omp region, at 96 regions/token */
static void rho_probe(void){
    volatile int sink=0;
    const int N=96, REP=200;
    double t0=now();
    for(int r=0;r<REP;r++) for(int i=0;i<N;i++){
        #pragma omp parallel
        { if(omp_get_thread_num()==0) sink++; }
    }
    double empty=(now()-t0)/REP;
    t0=now();
    for(int r=0;r<REP;r++) for(int i=0;i<N;i++){
        #pragma omp parallel for schedule(static)
        for(int b=0;b<C;b++) sink+=b;
    }
    double pfor=(now()-t0)/REP;
    t0=now();
    for(int r=0;r<REP;r++) for(int i=0;i<N;i++){
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for(int b=0;b<C;b++) sink+=b;
            #pragma omp for schedule(static)
            for(int b=0;b<C;b++) sink+=b;
        }
    }
    double two=(now()-t0)/REP;
    printf("rho probe (%d regions, the write-back's cadence):\n",N);
    printf("  empty `omp parallel`              %8.3f ms/token  = %6.2f us/region\n",
           empty*1000,empty*1e6/N);
    printf("  `omp parallel for` over C=4       %8.3f ms/token  = %6.2f us/region\n",
           pfor*1000,pfor*1e6/N);
    printf("  one region with TWO `omp for`s    %8.3f ms/token  = %6.2f us/region\n",
           two*1000,two*1e6/N);
    printf("  => fusing two loops into one region saves %.3f ms/token at 96 sites\n",
           (2*pfor-two)*1000);
}

int main(void){
    int threads=1;
#ifdef _OPENMP
    #pragma omp parallel
    { if(omp_get_thread_num()==0) threads=omp_get_num_threads(); }
#endif
    int ballast_mb=getenv("BALLAST_MB")?atoi(getenv("BALLAST_MB")):24;
    printf("rome_grapplybench: q38_gr_apply + q38_gr_read at the engine's decode shapes\n");
    printf("  H=%d C=%d W=%d R=%d sites/token=%d (apply at %d of them) threads=%d\n",
           H,C,W,R,SITES,SITES-1,threads);
    double wbytes=(double)SITES*((double)R*W+(double)W*R+(double)C*W)*2.0;
    printf("  weight stream per token: %.2f MB (L3 is 128 MB); ballast %d MB/site\n",
           wbytes/1048576.0,ballast_mb);

    if(ballast_mb>0){
        ballast_n=(int64_t)ballast_mb*1048576/4;
        ballast=(float*)malloc((size_t)ballast_n*sizeof(float));
        if(!ballast){fprintf(stderr,"OOM ballast\n");return 1;}
        for(int64_t z=0;z<ballast_n;z++) ballast[z]=(float)(z&1023);
    }

    float *hyper_seed=(float*)malloc((size_t)W*sizeof(float));
    for(int z=0;z<W;z++) hyper_seed[z]=rndf(-3.f,3.f);
    float *blockbank=(float*)malloc((size_t)8*H*sizeof(float));
    for(int64_t z=0;z<(int64_t)8*H;z++) blockbank[z]=rndf(-0.5f,0.5f);
    float *hyper=(float*)malloc((size_t)W*sizeof(float));
    float *norm=(float*)malloc((size_t)W*sizeof(float));
    float *low=(float*)malloc((size_t)R*sizeof(float));
    float *mixbuf=(float*)malloc((size_t)W*sizeof(float));
    float *mixed=(float*)malloc((size_t)H*sizeof(float));
    float *inject=(float*)malloc((size_t)C*sizeof(float));

    printf("building 97 distinct weight sets (%.2f MB)...\n",wbytes/1048576.0);
    Site *big=sites_alloc(SITES);

    printf("\n");
    rho_probe();

    /* ---- the oracle, before anything is timed --------------------------- */
    const int TOK=4;
    size_t ch=(size_t)SITES*W, cn=(size_t)SITES*W, cm=(size_t)SITES*H, ci=(size_t)SITES*C;
    float *ref_h=(float*)malloc(ch*TOK*sizeof(float)),*got_h=(float*)malloc(ch*TOK*sizeof(float));
    float *ref_n=(float*)malloc(cn*TOK*sizeof(float)),*got_n=(float*)malloc(cn*TOK*sizeof(float));
    float *ref_m=(float*)malloc(cm*TOK*sizeof(float)),*got_m=(float*)malloc(cm*TOK*sizeof(float));
    float *ref_i=(float*)malloc(ci*TOK*sizeof(float)),*got_i=(float*)malloc(ci*TOK*sizeof(float));
    if(!ref_h||!got_h||!ref_n||!got_n||!ref_m||!got_m||!ref_i||!got_i){
        fprintf(stderr,"OOM oracle\n");return 1;}
    float *seeds=(float*)malloc((size_t)TOK*W*sizeof(float));
    for(int64_t z=0;z<(int64_t)TOK*W;z++) seeds[z]=rndf(-3.f,3.f);
    int64_t save_ballast=ballast_n; ballast_n=0;   /* the oracle does not need it */
    printf("\n");
    for(int t=0;t<TOK;t++)
        token(big,hyper,seeds+(size_t)t*W,blockbank,norm,low,mixbuf,mixed,inject,
              A_SERIAL,NULL,ref_h+(size_t)t*ch,ref_n+(size_t)t*cn,
              ref_m+(size_t)t*cm,ref_i+(size_t)t*ci);
    int all_ok=1;
    for(int a=1;a<NARM;a++){
        for(int t=0;t<TOK;t++)
            token(big,hyper,seeds+(size_t)t*W,blockbank,norm,low,mixbuf,mixed,inject,
                  a,NULL,got_h+(size_t)t*ch,got_n+(size_t)t*cn,
                  got_m+(size_t)t*cm,got_i+(size_t)t*ci);
        int oh=memcmp(ref_h,got_h,ch*TOK*sizeof(float))==0;
        int on=memcmp(ref_n,got_n,cn*TOK*sizeof(float))==0;
        int om=memcmp(ref_m,got_m,cm*TOK*sizeof(float))==0;
        int oi=memcmp(ref_i,got_i,ci*TOK*sizeof(float))==0;
        printf("ORACLE %-32s hyper %s  norm %s  mixed %s  inject %s\n",
               arm_name[a],oh?"IDENTICAL":"DIFFER",on?"IDENTICAL":"DIFFER",
               om?"IDENTICAL":"DIFFER",oi?"IDENTICAL":"DIFFER");
        if(!(oh&&on&&om&&oi)) all_ok=0;
    }
    printf("ORACLE verdict: %s (%d tokens x %d sites)\n",
           all_ok?"all arms BIT-IDENTICAL":"A VARIANT DIFFERS -- do not ship it",TOK,SITES);
    ballast_n=save_ballast;
    free(ref_h);free(got_h);free(ref_n);free(got_n);free(ref_m);free(got_m);
    free(ref_i);free(got_i);free(seeds);

    /* ---- timing -------------------------------------------------------- */
    const int WARM=2, MEAS=8, REPS=3;
    printf("\n=== 97 distinct weight sets (1.28 GB/token), ballast %d MB/site ===\n",ballast_mb);
    printf("%-32s %8s %8s %8s %8s %8s %8s %8s\n","arm (ms/token)",
           "APPLY","RMS","A+R","down","up","mix","TOTAL");
    for(int rep=0;rep<REPS;rep++){
        for(int a=0;a<NARM;a++){
            Split acc={0};
            for(int t=0;t<WARM;t++)
                token(big,hyper,hyper_seed,blockbank,norm,low,mixbuf,mixed,inject,a,NULL,NULL,NULL,NULL,NULL);
            for(int t=0;t<MEAS;t++)
                token(big,hyper,hyper_seed,blockbank,norm,low,mixbuf,mixed,inject,a,&acc,NULL,NULL,NULL,NULL);
            double k=1000.0/MEAS;
            printf("%-32s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f%s\n",
                   arm_name[a],acc.apply*k,acc.rms*k,(acc.apply+acc.rms)*k,
                   acc.down*k,acc.up*k,acc.mix*k,acc.total*k,rep?"  (repeat)":"");
        }
    }
    printf("\nA+R is the column Q11 is judged on: the two loops together, in one\n"
           "binary, which is what the roadmap's gate requires.  down/up/mix are the\n"
           "neighbours this item does not touch and must not move.\n");
    return 0;
}
