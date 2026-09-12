/* rome_dnbench.c -- the isolated microbenchmark for roadmap item Q1
 * (DeltaNet's serial tail), plus its bit-identity oracle.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o /tmp/rome_dnbench rome_dnbench.c -lm
 * Run (8 threads, rig lock held, no engine up -- like every other row this
 * family of harnesses has produced):
 *   OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close /tmp/rome_dnbench
 *
 * WHAT IT MEASURES.  The three serial scalar loops inside q38_deltanet at
 * Qwen3.8-Flash-Next's real shapes (hidden 2560, 48 value heads x 128, 16 key
 * heads x 128, conv_dim 10240, conv kernel 4, 36 DeltaNet layers):
 *
 *   conv    for d < CD=10240: a 4-tap causal dot, a silu, and a 3-tap history
 *           shift.  36 layers => 368 640 iterations and 368 640 expf per token.
 *   qknorm  for h < VH=48: two 128-float memcpys, two double sums-of-squares,
 *           two rsqrts, 256 scalings.
 *   gnorm   for h < VH=48: q38_rmsg over VD=128 -- a double sum-of-squares and
 *           128 sigmoid-gated scalings, i.e. 6 144 expf per layer.
 *
 * The loop bodies are COPIED VERBATIM from c/qwen38_core.h (q38_deltanet, and
 * q38_silu/q38_sigmoid/q38_rmsg) so the isolated figure is the engine's own
 * arithmetic and not a lookalike.
 *
 * WORKING SET, and why this harness deviates from the "pools bigger than L3"
 * rule the record sets for the dense kernels.  Per layer the resident state is
 * dn_conv (10240*4 floats = 160 KB) + the conv ring (10240*3 = 120 KB) =
 * 280 KB; over 36 layers, 10.1 MB.  That is genuinely smaller than the 128 MB
 * L3 IN THE ENGINE TOO -- unlike the dense stream, this op has no large weight
 * to stream, and forcing it out of cache would model something the engine does
 * not do.  But the engine also pushes 6.81 GB/token of dense weights through
 * that same L3 between DeltaNet layers, so in practice these lines are cold
 * again by the time the next token wants them.  The two cases bracket the
 * truth, so BOTH are reported: `hot` (36 layers, 10.1 MB, L3-resident) and
 * `cold` (the same 36 layers replicated until the pool passes 512 MB, so every
 * token touches lines that have been evicted).  The engine's own sub-timers
 * (Q1 step 0) are the arbiter; these rows say what speedup is available and
 * whether the in-engine figure is attenuated.
 *
 * THE ORACLE.  Before any timing, every parallel variant is run against the
 * serial one from identical state and compared with memcmp on all four outputs
 * AND on the mutated conv ring, over 64 consecutive tokens (the ring carries
 * state forward, so a single-token check would not see a shift-order bug).
 * Bit-identical or the variant does not ship -- rome_fp8test.c's pattern.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
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
static void q38_rmsg(float *out,const float *x,const float *gate,const float *w,
                     int n,float eps,int sigmoid_gate) {
    double ss=0.0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*w[i]*(sigmoid_gate?q38_sigmoid(gate[i]):q38_silu(gate[i]));
}
/* -------------------------------------------------------------------------- */

enum { H=2560, VH=48, KH=16, KD=128, VD=128, CK=4 };
#define K  (KH*KD)          /* 2048 */
#define V  (VH*VD)          /* 6144 */
#define CD (2*K + V)        /* 10240 */
#define EPS 1e-6f

typedef struct {            /* one DeltaNet layer's conv weights + ring state */
    float *dn_conv;         /* CD*CK   */
    float *ring;            /* CD*(CK-1) */
    float *dn_norm;         /* VD      */
} DNLayer;

static float frand(uint64_t *s){
    *s = *s*6364136223846793005ull + 1442695040888963407ull;
    return (float)((int32_t)(*s>>33))/(float)(1u<<30);   /* ~[-2,2] */
}

/* ---- the three loops, serial: verbatim from q38_deltanet ------------------ */
static void dn_conv_serial(const DNLayer *L,const float *qkv_row,float *conv){
    for(int d=0;d<CD;d++) {
        float value=L->dn_conv[(int64_t)d*CK+CK-1]*qkv_row[d];
        float *history=L->ring+(int64_t)d*(CK-1);
        for(int tap=0;tap<CK-1;tap++)
            value+=L->dn_conv[(int64_t)d*CK+tap]*history[tap];
        conv[d]=q38_silu(value);
        for(int tap=0;tap<CK-2;tap++)history[tap]=history[tap+1];
        history[CK-2]=qkv_row[d];
    }
}
static void dn_qknorm_serial(const float *conv,float *q,float *k){
    const float *qi=conv,*ki=conv+K;
    const int rep=VH/KH;
    for(int h=0;h<VH;h++) {
        float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD;
        memcpy(qh,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
        memcpy(kh,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
        double qsum=1e-6,ksum=1e-6;
        for(int d=0;d<KD;d++) { qsum+=(double)qh[d]*qh[d]; ksum+=(double)kh[d]*kh[d]; }
        float qscale=1.f/sqrtf((float)qsum)/sqrtf((float)KD);
        float kscale=1.f/sqrtf((float)ksum);
        for(int d=0;d<KD;d++){qh[d]*=qscale;kh[d]*=kscale;}
    }
}
static void dn_gnorm_serial(const DNLayer *L,float *norm_row,const float *core,
                            const float *z_row){
    for(int h=0;h<VH;h++)
        q38_rmsg(norm_row+(int64_t)h*VD,core+(int64_t)h*VD,
                 z_row+(int64_t)h*VD,L->dn_norm,VD,EPS,1);
}

/* ---- the same three loops, parallel: identical bodies, one pragma each ---- */
static void dn_conv_par(const DNLayer *L,const float *qkv_row,float *conv){
    #pragma omp parallel for schedule(static)
    for(int d=0;d<CD;d++) {
        float value=L->dn_conv[(int64_t)d*CK+CK-1]*qkv_row[d];
        float *history=L->ring+(int64_t)d*(CK-1);
        for(int tap=0;tap<CK-1;tap++)
            value+=L->dn_conv[(int64_t)d*CK+tap]*history[tap];
        conv[d]=q38_silu(value);
        for(int tap=0;tap<CK-2;tap++)history[tap]=history[tap+1];
        history[CK-2]=qkv_row[d];
    }
}
static void dn_qknorm_par(const float *conv,float *q,float *k){
    const float *qi=conv,*ki=conv+K;
    const int rep=VH/KH;
    #pragma omp parallel for schedule(static)
    for(int h=0;h<VH;h++) {
        float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD;
        memcpy(qh,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
        memcpy(kh,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
        double qsum=1e-6,ksum=1e-6;
        for(int d=0;d<KD;d++) { qsum+=(double)qh[d]*qh[d]; ksum+=(double)kh[d]*kh[d]; }
        float qscale=1.f/sqrtf((float)qsum)/sqrtf((float)KD);
        float kscale=1.f/sqrtf((float)ksum);
        for(int d=0;d<KD;d++){qh[d]*=qscale;kh[d]*=kscale;}
    }
}
static void dn_gnorm_par(const DNLayer *L,float *norm_row,const float *core,
                         const float *z_row){
    #pragma omp parallel for schedule(static)
    for(int h=0;h<VH;h++)
        q38_rmsg(norm_row+(int64_t)h*VD,core+(int64_t)h*VD,
                 z_row+(int64_t)h*VD,L->dn_norm,VD,EPS,1);
}

/* ---- and fused into ONE region: three omp fors, three implicit barriers,
 *      one fork/join.  The recurrence loop (already parallel in the engine) is
 *      modelled by a fourth `omp for` that only touches its own head slice, so
 *      the fork/join count of this variant matches the engine's TODAY (one per
 *      layer per token) instead of quadrupling it. ------------------------- */
static void dn_fused(const DNLayer *L,const float *qkv_row,float *conv,
                     float *q,float *k,float *norm_row,const float *core,
                     const float *z_row){
    const int rep=VH/KH;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int d=0;d<CD;d++) {
            float value=L->dn_conv[(int64_t)d*CK+CK-1]*qkv_row[d];
            float *history=L->ring+(int64_t)d*(CK-1);
            for(int tap=0;tap<CK-1;tap++)
                value+=L->dn_conv[(int64_t)d*CK+tap]*history[tap];
            conv[d]=q38_silu(value);
            for(int tap=0;tap<CK-2;tap++)history[tap]=history[tap+1];
            history[CK-2]=qkv_row[d];
        }
        const float *qi=conv,*ki=conv+K;
        #pragma omp for schedule(static)
        for(int h=0;h<VH;h++) {
            float *qh=q+(int64_t)h*KD,*kh=k+(int64_t)h*KD;
            memcpy(qh,qi+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
            memcpy(kh,ki+(int64_t)(h/rep)*KD,(size_t)KD*sizeof(float));
            double qsum=1e-6,ksum=1e-6;
            for(int d=0;d<KD;d++) { qsum+=(double)qh[d]*qh[d]; ksum+=(double)kh[d]*kh[d]; }
            float qscale=1.f/sqrtf((float)qsum)/sqrtf((float)KD);
            float kscale=1.f/sqrtf((float)ksum);
            for(int d=0;d<KD;d++){qh[d]*=qscale;kh[d]*=kscale;}
        }
        #pragma omp for schedule(static)
        for(int h=0;h<VH;h++)
            q38_rmsg(norm_row+(int64_t)h*VD,core+(int64_t)h*VD,
                     z_row+(int64_t)h*VD,L->dn_norm,VD,EPS,1);
    }
}

/* -------------------------------------------------------------------------- */
static DNLayer *make_layers(int n,uint64_t *seed){
    DNLayer *L=calloc((size_t)n,sizeof(DNLayer));
    for(int i=0;i<n;i++){
        L[i].dn_conv=aligned_alloc(64,(size_t)CD*CK*sizeof(float));
        L[i].ring   =aligned_alloc(64,(size_t)CD*(CK-1)*sizeof(float));
        L[i].dn_norm=aligned_alloc(64,(size_t)VD*sizeof(float));
        for(int j=0;j<CD*CK;j++)      L[i].dn_conv[j]=frand(seed)*0.25f;
        for(int j=0;j<CD*(CK-1);j++)  L[i].ring[j]=frand(seed);
        for(int j=0;j<VD;j++)         L[i].dn_norm[j]=1.f+frand(seed)*0.1f;
    }
    return L;
}
static void copy_rings(DNLayer *dst,const DNLayer *src,int n){
    for(int i=0;i<n;i++) memcpy(dst[i].ring,src[i].ring,(size_t)CD*(CK-1)*sizeof(float));
}
static DNLayer *clone_layers(const DNLayer *src,int n){
    DNLayer *L=calloc((size_t)n,sizeof(DNLayer));
    for(int i=0;i<n;i++){
        L[i].dn_conv=src[i].dn_conv; L[i].dn_norm=src[i].dn_norm;   /* shared, read-only */
        L[i].ring=aligned_alloc(64,(size_t)CD*(CK-1)*sizeof(float));
        memcpy(L[i].ring,src[i].ring,(size_t)CD*(CK-1)*sizeof(float));
    }
    return L;
}

typedef enum { V_SERIAL, V_PAR, V_FUSED } Variant;

/* one token through `layers` DeltaNet layers */
static void token(Variant v,DNLayer *L,int layers,const float *qkv,const float *z,
                  const float *core,float *conv,float *q,float *k,float *norm){
    for(int i=0;i<layers;i++){
        const float *qkv_row=qkv+(int64_t)(i%8)*CD;   /* vary the input a little */
        switch(v){
        case V_SERIAL:
            dn_conv_serial(&L[i],qkv_row,conv);
            dn_qknorm_serial(conv,q,k);
            dn_gnorm_serial(&L[i],norm,core,z);
            break;
        case V_PAR:
            dn_conv_par(&L[i],qkv_row,conv);
            dn_qknorm_par(conv,q,k);
            dn_gnorm_par(&L[i],norm,core,z);
            break;
        case V_FUSED:
            dn_fused(&L[i],qkv_row,conv,q,k,norm,core,z);
            break;
        }
    }
}

int main(void){
    uint64_t seed=0x51ed270bULL;
    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    printf("rome_dnbench: %d threads, CD=%d VH=%d KD=%d VD=%d CK=%d, 36 DeltaNet layers\n",
           threads,CD,VH,KD,VD,CK);

    float *qkv=aligned_alloc(64,(size_t)8*CD*sizeof(float));
    float *z  =aligned_alloc(64,(size_t)V*sizeof(float));
    float *core=aligned_alloc(64,(size_t)V*sizeof(float));
    for(int i=0;i<8*CD;i++) qkv[i]=frand(&seed);
    for(int i=0;i<V;i++){ z[i]=frand(&seed); core[i]=frand(&seed); }

    float *conv_a=aligned_alloc(64,(size_t)CD*sizeof(float));
    float *conv_b=aligned_alloc(64,(size_t)CD*sizeof(float));
    float *q_a=aligned_alloc(64,(size_t)VH*KD*sizeof(float));
    float *q_b=aligned_alloc(64,(size_t)VH*KD*sizeof(float));
    float *k_a=aligned_alloc(64,(size_t)VH*KD*sizeof(float));
    float *k_b=aligned_alloc(64,(size_t)VH*KD*sizeof(float));
    float *n_a=aligned_alloc(64,(size_t)V*sizeof(float));
    float *n_b=aligned_alloc(64,(size_t)V*sizeof(float));

    /* ---- ORACLE: 64 consecutive tokens, state carried, memcmp everything --- */
    const int LAY=36, TOK=64;
    DNLayer *base=make_layers(LAY,&seed);
    const char *vname[3]={"serial","par(3 pragmas)","fused(1 region)"};
    int all_ok=1;
    for(int vi=1;vi<3;vi++){
        DNLayer *A=clone_layers(base,LAY), *B=clone_layers(base,LAY);
        int ok=1; int first_tok=-1; const char *what="";
        for(int t=0;t<TOK && ok;t++){
            token(V_SERIAL,A,LAY,qkv,z,core,conv_a,q_a,k_a,n_a);
            token((Variant)vi,B,LAY,qkv,z,core,conv_b,q_b,k_b,n_b);
            if(memcmp(conv_a,conv_b,(size_t)CD*4)){ok=0;what="conv";}
            else if(memcmp(q_a,q_b,(size_t)VH*KD*4)){ok=0;what="q";}
            else if(memcmp(k_a,k_b,(size_t)VH*KD*4)){ok=0;what="k";}
            else if(memcmp(n_a,n_b,(size_t)V*4)){ok=0;what="norm";}
            else for(int i=0;i<LAY;i++)
                if(memcmp(A[i].ring,B[i].ring,(size_t)CD*(CK-1)*4)){ok=0;what="ring";break;}
            if(!ok) first_tok=t;
        }
        printf("ORACLE  %-16s vs serial, %d layers x %d tokens: %s%s",
               vname[vi],LAY,TOK,ok?"BIT-IDENTICAL":"DIFFER on ",ok?"":what);
        if(!ok) printf(" at token %d",first_tok);
        printf("\n");
        all_ok = all_ok && ok;
        for(int i=0;i<LAY;i++) free(A[i].ring), free(B[i].ring);
        free(A); free(B);
    }
    if(!all_ok){ printf("STOP: a parallel variant is not bit-identical.\n"); return 1; }

    /* ---- TIMING ----------------------------------------------------------- */
    /* hot: 36 layers, 10.1 MB of state -- what the engine holds.
     * cold: the same 36 layers replicated to pass 512 MB, so the lines are
     *       evicted between touches, which is what the dense stream does to
     *       them in the engine.  Per-token cost is reported per 36 layers in
     *       both cases so the two rows are directly comparable. */
    struct { const char *tag; int layers; int stride; } cases[]={
        {"hot  (10.1 MB, L3-resident)", LAY, 1},
        {"cold (>512 MB pool)",         LAY*64, 1},
    };
    for(unsigned ci=0;ci<sizeof cases/sizeof cases[0];ci++){
        int nl=cases[ci].layers;
        DNLayer *L=make_layers(nl,&seed);
        double mb=(double)nl*(CD*CK+CD*(CK-1)+VD)*4.0/1048576.0;
        printf("--- %s : %d layer-states, %.1f MB\n",cases[ci].tag,nl,mb);
        for(int vi=0;vi<3;vi++){
            /* warm/settle, then time enough tokens for a stable number */
            int reps=(ci==0)?200:8;
            int per=(ci==0)?LAY:nl;              /* layers touched per rep */
            for(int w=0;w<2;w++)
                for(int i=0;i<per;i+=LAY)
                    token((Variant)vi,L+i,(per-i<LAY)?per-i:LAY,qkv,z,core,
                          conv_a,q_a,k_a,n_a);
            double t0=now();
            for(int r=0;r<reps;r++)
                for(int i=0;i<per;i+=LAY)
                    token((Variant)vi,L+i,(per-i<LAY)?per-i:LAY,qkv,z,core,
                          conv_a,q_a,k_a,n_a);
            double dt=now()-t0;
            double tokens=(double)reps*(double)per/(double)LAY;
            printf("    %-16s %8.3f ms/token (36 layers)   %6.3f us/layer\n",
                   vname[vi],dt/tokens*1000.0,dt/tokens/LAY*1e6);
        }
        for(int i=0;i<nl;i++){free(L[i].dn_conv);free(L[i].ring);free(L[i].dn_norm);}
        free(L);
    }
    (void)copy_rings;
    return 0;
}
