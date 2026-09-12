/* rome_dnbench.c -- the isolated microbenchmark for roadmap item Q1
 * (DeltaNet's serial tail), plus its bit-identity oracle.
 *
 * EXTENDED 2026-09-12 for roadmap item Q10 (the DeltaNet recurrence's state
 * traffic).  Part 2 at the bottom of main() adds the 48-head recurrence itself
 * -- the loop Q1 measured at 9.19 ms/token and did not touch -- with its own
 * oracle (memcmp on the FULL 3.146 MB/layer state, not just the output, over
 * 36 layers x 64 consecutive tokens, because the state persists across tokens
 * and a single-token check cannot see a wrong reassociation) and three cache
 * regimes chosen to separate the bandwidth reading from the latency one.
 * Run `rome_dnbench q1` or `rome_dnbench recur` for one part only.
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o /tmp/rome_dnbench rome_dnbench.c -lm
 *   (the engine's own flags: c/Makefile's Linux x86-64 block is
 *    -O3 -march=native -fopenmp -pthread, no -std=, so -ffp-contract=fast is
 *    in force in BOTH and the contraction the oracle sees is the engine's.)
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

/* ==========================================================================
 * PART 2 -- roadmap item Q10: the 48-head DeltaNet recurrence.
 *
 * Verbatim from c/qwen38_core.h's `#pragma omp for` over h<VH inside
 * q38_deltanet (the [OPTIME] dn-recur bucket, 9.19 ms/token per Q1 step 0).
 * Per head and token it makes FOUR traversals of a KD*VD = 128*128 f32 state
 * (64 KB):
 *   1. decay        state[cell] *= alpha              (read + write)
 *   2. k.state      previous = sum_d kh[d]*state[d,v] (read, d-inner, stride
 *                                                      VD*4 = 512 B)
 *   3. rank-1       state[d,v] += kh[d]*delta[v]      (read + write)
 *   4. q.state      current  = sum_d qh[d]*state[d,v] (read, d-inner, 512 B)
 * = 6 traffic units (4 reads + 2 writes) of 64 KB per head.
 *
 * THE TWO FOLDINGS UNDER TEST, and exactly what each claims.
 *
 * (a) REC_A -- drop the decay pass.  fl(s*alpha) is recomputed on the fly
 *     inside pass 2 and again inside pass 3.  Both are the same single float
 *     multiply of the same two floats, so they round to the same bits; the
 *     only thing that can break it is FMA CONTRACTION, and only in pass 3,
 *     where `s*alpha + kh[d]*delta[v]` has TWO multiplies feeding ONE add and
 *     the compiler picks one to fuse.  The base's pass 3 is
 *     `state[c] += kh[d]*delta[v]` with state[c] already holding fl(s*alpha),
 *     i.e. one multiply feeding one add -- under -ffp-contract=fast that is
 *     fma(kh[d], delta[v], fl(s*alpha)).  So the candidate is bit-identical
 *     iff it fuses the SAME multiply.  This harness does not assume that: it
 *     builds BOTH spellings --
 *        REC_A / REC_AB      : plain `s*alpha + kd*delta[v]`, compiler's pick
 *        REC_A_F / REC_AB_F  : explicit fmaf(kd, delta[v], s*alpha)
 *     -- and lets the memcmp oracle say which one (or both) holds.  Pass 2's
 *     `previous += kh[d]*(s*alpha)` is not at risk: the inner `s*alpha` feeds
 *     a multiply, not an add, so it is never a contraction candidate.
 *     Traffic: 6 -> 4 units.
 *
 * (b) REC_AB -- additionally fold q.state into the rank-1 pass with `d` OUTER
 *     and `value` INNER, so every current[value] still accumulates in
 *     ascending d (same summation order, same addends, same rounding), and
 *     do the same to the `previous` reduction, which costs nothing extra and
 *     preserves its d-order for the same reason.  Traffic: 4 -> 3 units, and
 *     BOTH d-inner strided reductions become unit-stride and vectorisable
 *     across `value` -- which is the change that matters if the loop is
 *     latency-bound rather than bandwidth-bound.
 *
 * Note both foldings recompute fl(s*alpha), so the state is READ twice and
 * WRITTEN once per head per token in (b): 3 units against the base's 6.
 * ========================================================================== */

typedef struct {                /* one DeltaNet layer's recurrent state */
    float *state;               /* VH*KD*VD floats = 3.146 MB */
    float *alog;                /* VH */
    float *dtbias;              /* VH */
} RecLayer;

/* verbatim from c/qwen38_core.h */
static inline float q38_softplus(float x){ return x>20.f?x:log1pf(expf(x)); }

/* ---- per-head kernels ----------------------------------------------------- */
/* BASE: copied character for character out of q38_deltanet's h-loop body. */
static void rec_base(float *state,const float *qh,const float *kh,
                     const float *vh,float *core_h,float alpha,float beta){
    float delta[512];
    int64_t state_cells=(int64_t)KD*VD;
    for(int64_t cell=0;cell<state_cells;cell++)state[cell]*=alpha;
    for(int value=0;value<VD;value++) {
        float previous=0.f;
        for(int d=0;d<KD;d++)
            previous+=kh[d]*state[(int64_t)d*VD+value];
        delta[value]=(vh[value]-previous)*beta;
    }
    for(int d=0;d<KD;d++)for(int value=0;value<VD;value++)
        state[(int64_t)d*VD+value]+=kh[d]*delta[value];
    for(int value=0;value<VD;value++) {
        float current=0.f;
        for(int d=0;d<KD;d++)
            current+=qh[d]*state[(int64_t)d*VD+value];
        core_h[value]=current;
    }
}

/* (a) only: decay folded away, both reductions still d-inner/strided. */
#define REC_A_BODY(FUSE)                                                      \
    float delta[512];                                                         \
    for(int value=0;value<VD;value++) {                                       \
        float previous=0.f;                                                   \
        for(int d=0;d<KD;d++)                                                 \
            previous+=kh[d]*(state[(int64_t)d*VD+value]*alpha);               \
        delta[value]=(vh[value]-previous)*beta;                               \
    }                                                                         \
    for(int d=0;d<KD;d++)for(int value=0;value<VD;value++){                   \
        int64_t c=(int64_t)d*VD+value;                                        \
        state[c]=FUSE(kh[d],delta[value],state[c]*alpha);                     \
    }                                                                         \
    for(int value=0;value<VD;value++) {                                       \
        float current=0.f;                                                    \
        for(int d=0;d<KD;d++)                                                 \
            current+=qh[d]*state[(int64_t)d*VD+value];                        \
        core_h[value]=current;                                                \
    }
#define PLAIN_FMA(a,b,c) ((c)+(a)*(b))
static void rec_a(float *state,const float *qh,const float *kh,
                  const float *vh,float *core_h,float alpha,float beta){
    REC_A_BODY(PLAIN_FMA)
}
static void rec_a_f(float *state,const float *qh,const float *kh,
                    const float *vh,float *core_h,float alpha,float beta){
    REC_A_BODY(fmaf)
}

/* (a)+(b): every pass d-outer / value-inner.  prev[] and cur[] accumulate in
 * ascending d exactly as `previous` and `current` do in the base. */
#define REC_AB_BODY(FUSE)                                                     \
    float prev[VD],delta[VD],cur[VD];                                         \
    for(int value=0;value<VD;value++) prev[value]=0.f;                        \
    for(int d=0;d<KD;d++){                                                    \
        float kd=kh[d]; const float *sr=state+(int64_t)d*VD;                  \
        for(int value=0;value<VD;value++)                                     \
            prev[value]+=kd*(sr[value]*alpha);                                \
    }                                                                         \
    for(int value=0;value<VD;value++)                                         \
        delta[value]=(vh[value]-prev[value])*beta;                            \
    for(int value=0;value<VD;value++) cur[value]=0.f;                         \
    for(int d=0;d<KD;d++){                                                    \
        float kd=kh[d],qd=qh[d]; float *sr=state+(int64_t)d*VD;               \
        for(int value=0;value<VD;value++){                                    \
            float s=FUSE(kd,delta[value],sr[value]*alpha);                    \
            sr[value]=s;                                                      \
            cur[value]+=qd*s;                                                 \
        }                                                                     \
    }                                                                         \
    for(int value=0;value<VD;value++) core_h[value]=cur[value];
static void rec_ab(float *state,const float *qh,const float *kh,
                   const float *vh,float *core_h,float alpha,float beta){
    REC_AB_BODY(PLAIN_FMA)
}
static void rec_ab_f(float *state,const float *qh,const float *kh,
                     const float *vh,float *core_h,float alpha,float beta){
    REC_AB_BODY(fmaf)
}

/* --- WHY THE TWO SPELLINGS ABOVE ARE NOT ENOUGH, from the base's own asm ---
 * gcc 15.2 -O3 -march=native (the engine's flags) compiles the BASE's two
 * reductions by OUTER-LOOP vectorisation -- 8 `value`s at a time with `d`
 * inner and a 512 B stride -- and it emits them as vmulps + vaddps, i.e. it
 * does NOT contract them:
 *      .L4:  vbroadcastss (%rdx),%ymm0
 *            vmulps  -512(%rax),%ymm0,%ymm0
 *            vaddps  %ymm0,%ymm2,%ymm2
 * while it DOES contract the rank-1 update (vfmadd213ps at .L10).  So the
 * roadmap's premise that these reductions are "unvectorised across value" is
 * wrong -- they are vectorised -- and the bit-identity requirement is the
 * exact opposite of what the item assumes: the rank-1 addend must be FUSED
 * (hence fmaf) and the two reductions must NOT be.  Writing (b)'s reductions
 * as a plain `acc[v] += x*y` in a d-outer/value-inner loop is a textbook
 * vectorisable statement and gcc contracts it, which is why both (a)+(b)
 * spellings above fail the oracle.  The fix is to spell the contraction
 * explicitly in both directions: fmaf() where the base fuses, and
 * `#pragma GCC optimize ("fp-contract=off")` around the function where it
 * does not.  The two diagnostics below localise the failure statement by
 * statement, and REC_AB_OFF is the corrected form. */

/* diagnostic: (a)+fmaf, with (b) applied to the `previous` reduction ONLY */
static void rec_b_prev(float *state,const float *qh,const float *kh,
                       const float *vh,float *core_h,float alpha,float beta){
    float prev[VD],delta[VD];
    for(int value=0;value<VD;value++) prev[value]=0.f;
    for(int d=0;d<KD;d++){
        float kd=kh[d]; const float *sr=state+(int64_t)d*VD;
        for(int value=0;value<VD;value++) prev[value]+=kd*(sr[value]*alpha);
    }
    for(int value=0;value<VD;value++) delta[value]=(vh[value]-prev[value])*beta;
    for(int d=0;d<KD;d++)for(int value=0;value<VD;value++){
        int64_t c=(int64_t)d*VD+value;
        state[c]=fmaf(kh[d],delta[value],state[c]*alpha);
    }
    for(int value=0;value<VD;value++) {
        float current=0.f;
        for(int d=0;d<KD;d++) current+=qh[d]*state[(int64_t)d*VD+value];
        core_h[value]=current;
    }
}
/* diagnostic: (a)+fmaf, with (b) applied to the `current` reduction ONLY */
static void rec_b_cur(float *state,const float *qh,const float *kh,
                      const float *vh,float *core_h,float alpha,float beta){
    float delta[512],cur[VD];
    for(int value=0;value<VD;value++) {
        float previous=0.f;
        for(int d=0;d<KD;d++) previous+=kh[d]*(state[(int64_t)d*VD+value]*alpha);
        delta[value]=(vh[value]-previous)*beta;
    }
    for(int value=0;value<VD;value++) cur[value]=0.f;
    for(int d=0;d<KD;d++){
        float kd=kh[d],qd=qh[d]; float *sr=state+(int64_t)d*VD;
        for(int value=0;value<VD;value++){
            float s=fmaf(kd,delta[value],sr[value]*alpha);
            sr[value]=s; cur[value]+=qd*s;
        }
    }
    for(int value=0;value<VD;value++) core_h[value]=cur[value];
}

/* THE CORRECTED (a)+(b): contraction spelled out in both directions. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
static void rec_ab_off(float *state,const float *qh,const float *kh,
                       const float *vh,float *core_h,float alpha,float beta){
    REC_AB_BODY(fmaf)
}
/* CONTROL: (b)'s loop order WITHOUT (a) -- the decay pass is kept, so the
 * traffic is 5 units instead of (a)+(b)'s 3.  This is the row that separates
 * "traffic removed" from "loop order fixed": (a) alone gives the traffic and
 * not the order, this gives the order and not the traffic. */
static void rec_b_off(float *state,const float *qh,const float *kh,
                      const float *vh,float *core_h,float alpha,float beta){
    float prev[VD],delta[VD],cur[VD];
    int64_t state_cells=(int64_t)KD*VD;
    for(int64_t cell=0;cell<state_cells;cell++)state[cell]*=alpha;
    for(int value=0;value<VD;value++) prev[value]=0.f;
    for(int d=0;d<KD;d++){
        float kd=kh[d]; const float *sr=state+(int64_t)d*VD;
        for(int value=0;value<VD;value++) prev[value]+=kd*sr[value];
    }
    for(int value=0;value<VD;value++) delta[value]=(vh[value]-prev[value])*beta;
    for(int value=0;value<VD;value++) cur[value]=0.f;
    for(int d=0;d<KD;d++){
        float kd=kh[d],qd=qh[d]; float *sr=state+(int64_t)d*VD;
        for(int value=0;value<VD;value++){
            float s=fmaf(kd,delta[value],sr[value]);
            sr[value]=s; cur[value]+=qd*s;
        }
    }
    for(int value=0;value<VD;value++) core_h[value]=cur[value];
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

typedef enum { R_BASE, R_A, R_A_F, R_AB, R_AB_F,
               R_BPREV, R_BCUR, R_AB_OFF, R_B_OFF, R_NVAR } RVariant;
static const char *rvname[R_NVAR]={
    "base (4 passes)","(a) plain +","(a) fmaf()","(a)+(b) plain +",
    "(a)+(b) fmaf()","diag (b) prev only","diag (b) cur only",
    "(a)+(b) contract-off","(b) only, decay kept"};

/* one layer of the recurrence, parallel over heads exactly as the engine is */
static void rec_layer(RVariant v,RecLayer *L,const float *q,const float *k,
                      const float *vi,const float *a_row,const float *b_row,
                      float *core){
    #pragma omp parallel for schedule(static)
    for(int h=0;h<VH;h++){
        float *state=L->state+(int64_t)h*KD*VD;
        const float *qh=q+(int64_t)h*KD;
        const float *kh=k+(int64_t)h*KD;
        const float *vh=vi+(int64_t)h*VD;
        float alpha=expf(-expf(L->alog[h])*q38_softplus(a_row[h]+L->dtbias[h]));
        float beta=q38_sigmoid(b_row[h]);
        float *core_h=core+(int64_t)h*VD;
        switch(v){
        case R_BASE: rec_base(state,qh,kh,vh,core_h,alpha,beta); break;
        case R_A:    rec_a   (state,qh,kh,vh,core_h,alpha,beta); break;
        case R_A_F:  rec_a_f (state,qh,kh,vh,core_h,alpha,beta); break;
        case R_AB:   rec_ab  (state,qh,kh,vh,core_h,alpha,beta); break;
        case R_AB_F: rec_ab_f(state,qh,kh,vh,core_h,alpha,beta); break;
        case R_BPREV:rec_b_prev(state,qh,kh,vh,core_h,alpha,beta); break;
        case R_BCUR: rec_b_cur (state,qh,kh,vh,core_h,alpha,beta); break;
        case R_B_OFF:rec_b_off (state,qh,kh,vh,core_h,alpha,beta); break;
        default:     rec_ab_off(state,qh,kh,vh,core_h,alpha,beta); break;
        }
    }
}

#define REC_BYTES ((size_t)VH*KD*VD*sizeof(float))
static RecLayer *make_rec(int n,uint64_t *seed){
    RecLayer *L=calloc((size_t)n,sizeof(RecLayer));
    for(int i=0;i<n;i++){
        L[i].state =aligned_alloc(64,REC_BYTES);
        L[i].alog  =aligned_alloc(64,(size_t)VH*sizeof(float));
        L[i].dtbias=aligned_alloc(64,(size_t)VH*sizeof(float));
        for(size_t j=0;j<REC_BYTES/4;j++) L[i].state[j]=frand(seed)*0.1f;
        for(int j=0;j<VH;j++){ L[i].alog[j]=frand(seed)*0.5f-1.f;
                               L[i].dtbias[j]=frand(seed)*0.5f; }
    }
    return L;
}
static void free_rec(RecLayer *L,int n){
    for(int i=0;i<n;i++){free(L[i].state);free(L[i].alog);free(L[i].dtbias);}
    free(L);
}
static RecLayer *clone_rec(const RecLayer *src,int n){
    RecLayer *L=calloc((size_t)n,sizeof(RecLayer));
    for(int i=0;i<n;i++){
        L[i].alog=src[i].alog; L[i].dtbias=src[i].dtbias;   /* shared, read-only */
        L[i].state=aligned_alloc(64,REC_BYTES);
        memcpy(L[i].state,src[i].state,REC_BYTES);
    }
    return L;
}
static void free_clone(RecLayer *L,int n){
    for(int i=0;i<n;i++) free(L[i].state);
    free(L);
}

static int part_recur(uint64_t *seed);

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

int main(int argc,char **argv){
    uint64_t seed=0x51ed270bULL;
    int threads=1;
    int do_q1=1, do_rec=1;
    if(argc>1){
        if(!strcmp(argv[1],"q1"))         { do_rec=0; }
        else if(!strcmp(argv[1],"recur")) { do_q1=0;  }
        else { fprintf(stderr,"usage: %s [q1|recur]\n",argv[0]); return 2; }
    }
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    printf("rome_dnbench: %d threads, CD=%d VH=%d KD=%d VD=%d CK=%d, 36 DeltaNet layers\n",
           threads,CD,VH,KD,VD,CK);
    if(!do_q1) goto part2;

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
part2:
    if(do_rec) return part_recur(&seed);
    return 0;
}

/* ==========================================================================
 * PART 2 -- Q10 step 0: the recurrence's oracle and its three cache regimes.
 *
 * THE ORACLE is deliberately stricter than the rest of this track's harness
 * oracles, because the recurrent state PERSISTS ACROSS TOKENS: a reassociation
 * that is invisible in one token's `core` output would drift the state and
 * show up dozens of tokens later.  So every variant is run from an identical
 * cloned state for 36 layers x 64 consecutive tokens and, at EVERY token,
 * memcmp'd against the base on (i) the `core` output and (ii) all 36 layers'
 * FULL 3.146 MB state -- 4.08 GB of memcmp per variant.
 *
 * THE THREE REGIMES separate the two competing readings of the 9.19 ms/token:
 *   l2   1 layer's state (3.146 MB; 8 threads x 6 heads x 64 KB = 384 KB per
 *        core, inside this part's 512 KB L2) re-used every token.
 *   l3   36 layers (110.6 MB) -- what the engine holds; fits the 128 MB L3 but
 *        not any L2, so every token's first traversal of a head comes from L3.
 *   dram 288 layers (884 MB) -- past L3, so the first traversal is a DRAM read
 *        and the write-back is a DRAM write.  The engine pushes 6.81 GB/token
 *        of dense weights through L3 between DeltaNet layers, so this row is
 *        the closer model of the engine even though the engine's own state is
 *        110.6 MB.
 * If the loop is BANDWIDTH-bound the three rows separate sharply.  If it is
 * LATENCY-bound (128 dependent adds per output, 512 B apart, unvectorised)
 * they are close, and the fix is (b)'s loop order rather than (a)'s traffic.
 * ========================================================================== */
static int part_recur(uint64_t *seed){
    const int LAY=36, TOK=64;
    printf("\n================ Q10: the 48-head recurrence ================\n");
    printf("state %d heads x %dx%d f32 = %.3f MB/layer, %.1f MB over %d layers\n",
           VH,KD,VD,(double)REC_BYTES/1048576.0,
           (double)REC_BYTES*LAY/1048576.0,LAY);

    /* per-token inputs; 8 rotating sets so the arithmetic is not degenerate */
    float *q=aligned_alloc(64,(size_t)8*VH*KD*sizeof(float));
    float *k=aligned_alloc(64,(size_t)8*VH*KD*sizeof(float));
    float *vv=aligned_alloc(64,(size_t)8*V*sizeof(float));
    float *ar=aligned_alloc(64,(size_t)8*VH*sizeof(float));
    float *br=aligned_alloc(64,(size_t)8*VH*sizeof(float));
    for(size_t i=0;i<(size_t)8*VH*KD;i++){ q[i]=frand(seed)*0.1f; k[i]=frand(seed)*0.1f; }
    for(size_t i=0;i<(size_t)8*V;i++)  vv[i]=frand(seed);
    for(size_t i=0;i<(size_t)8*VH;i++){ ar[i]=frand(seed); br[i]=frand(seed); }
    float *core_a=aligned_alloc(64,(size_t)V*sizeof(float));
    float *core_b=aligned_alloc(64,(size_t)V*sizeof(float));

    /* Q10_ONLY / Q10_REG / Q10_NOORACLE: one variant in one regime with no
     * oracle, so `perf stat` can bracket exactly one kernel's cycles and
     * bytes.  Everything else is unchanged; the default run is the full one. */
    const char *e_only=getenv("Q10_ONLY"), *e_reg=getenv("Q10_REG");
    int only=e_only?atoi(e_only):-1, onereg=e_reg?atoi(e_reg):-1;
    int no_oracle=getenv("Q10_NOORACLE")!=NULL;

    /* ---- ORACLE ---------------------------------------------------------- */
    RecLayer *base=make_rec(LAY,seed);
    int ok_var[R_NVAR]; for(int i=0;i<R_NVAR;i++) ok_var[i]=1;
    for(int vi=1;vi<R_NVAR && !no_oracle;vi++){
        RecLayer *A=clone_rec(base,LAY), *B=clone_rec(base,LAY);
        int ok=1,first=-1; long dcore=0,dstate=0; double maxabs=0.0;
        for(int t=0;t<TOK;t++){
            int s=t%8;
            for(int i=0;i<LAY;i++){
                rec_layer(R_BASE,&A[i],q+(int64_t)s*VH*KD,k+(int64_t)s*VH*KD,
                          vv+(int64_t)s*V,ar+(int64_t)s*VH,br+(int64_t)s*VH,core_a);
                rec_layer((RVariant)vi,&B[i],q+(int64_t)s*VH*KD,k+(int64_t)s*VH*KD,
                          vv+(int64_t)s*V,ar+(int64_t)s*VH,br+(int64_t)s*VH,core_b);
            }
            int bad=0;
            if(memcmp(core_a,core_b,(size_t)V*4)){
                bad=1;
                for(int j=0;j<V;j++) if(core_a[j]!=core_b[j]){ dcore++;
                    double e=fabs((double)core_a[j]-(double)core_b[j]);
                    if(e>maxabs)maxabs=e; }
            }
            for(int i=0;i<LAY;i++)
                if(memcmp(A[i].state,B[i].state,REC_BYTES)){
                    bad=1;
                    for(size_t j=0;j<REC_BYTES/4;j++)
                        if(A[i].state[j]!=B[i].state[j]) dstate++;
                }
            if(bad && ok){ ok=0; first=t; }
            /* keep both sides on the BASE's state so later tokens still compare
             * the transformation and not the accumulated drift */
            if(bad) for(int i=0;i<LAY;i++) memcpy(B[i].state,A[i].state,REC_BYTES);
        }
        printf("ORACLE  %-20s vs base, %d layers x %d tokens, core+FULL state: %s",
               rvname[vi],LAY,TOK,ok?"BIT-IDENTICAL":"DIFFER");
        if(!ok) printf(" from token %d: %ld/%d core elems, %ld/%lld state elems, max|d| %.3e",
                       first,dcore,V*TOK,dstate,
                       (long long)(REC_BYTES/4)*LAY*TOK,maxabs);
        printf("\n");
        ok_var[vi]=ok;
        free_clone(A,LAY); free_clone(B,LAY);
    }
    free_rec(base,LAY);

    /* ---- TIMING ---------------------------------------------------------- */
    struct { const char *tag; int layers; int reps; } regs[]={
        {"l2   (1 layer, 3.1 MB -- 384 KB/core)", 1,   400},
        {"l3   (36 layers, 110.6 MB)",            LAY, 40 },
        {"dram (288 layers, 884.7 MB)",           LAY*8, 5 },
    };
    for(unsigned ri=0;ri<sizeof regs/sizeof regs[0];ri++){
        if(onereg>=0 && (int)ri!=onereg) continue;
        int nl=regs[ri].layers, reps=regs[ri].reps;
        RecLayer *L=make_rec(nl,seed);
        printf("--- %s\n",regs[ri].tag);
        double res[R_NVAR][2];
        for(int pass=0;pass<2;pass++)
        for(int vi=0;vi<R_NVAR;vi++){
            res[vi][pass]=-1.0;
            if(vi && !ok_var[vi]) continue;
            if(only>=0 && vi!=only) continue;
            for(int w=0;w<2;w++)
                for(int i=0;i<nl;i++)
                    rec_layer((RVariant)vi,&L[i],q,k,vv,ar,br,core_a);
            double t0=now();
            for(int r=0;r<reps;r++){
                int s=r%8;
                for(int i=0;i<nl;i++)
                    rec_layer((RVariant)vi,&L[i],q+(int64_t)s*VH*KD,k+(int64_t)s*VH*KD,
                              vv+(int64_t)s*V,ar+(int64_t)s*VH,br+(int64_t)s*VH,core_a);
            }
            double dt=now()-t0;
            res[vi][pass]=dt/((double)reps*(double)nl)*LAY*1000.0;
        }
        for(int vi=0;vi<R_NVAR;vi++){
            if(only>=0 && vi!=only) continue;
            if(vi && !ok_var[vi]){
                printf("    %-20s (not bit-identical -- not timed)\n",rvname[vi]);
                continue;
            }
            printf("    %-20s %8.3f / %8.3f ms/token (36 layers)   %+7.3f vs base\n",
                   rvname[vi],res[vi][0],res[vi][1],
                   (res[vi][0]+res[vi][1])/2.0-(res[0][0]+res[0][1])/2.0);
        }
        free_rec(L,nl);
    }
    free(q);free(k);free(vv);free(ar);free(br);free(core_a);free(core_b);
    return 0;
}
