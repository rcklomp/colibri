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
static size_t rowbytes(int fmt,int I){return fmt==2?(size_t)(I+1)/2:(size_t)I;}   /* 1,8: 1 B/elem; 2: nibble */
static size_t nscales(int fmt,int I,int O){return fmt==8?(size_t)((O+127)/128)*((I+127)/128):(size_t)O;}
static uint8_t rnd8(void){uint8_t v=rand()&0xff;return (v&0x7f)>=0x7e?0x38:v;}      /* no e4m3 NaN codes */
static float *randx(int n){float *x=malloc((size_t)n*4);for(int i=0;i<n;i++)x[i]=(rand()%200-100)/100.0f;return x;}
static uint8_t *randw(size_t n){uint8_t *w=malloc(n);for(size_t i=0;i<n;i++)w[i]=rnd8();return w;}
static float *consts(size_t n,float v){float *s=malloc(n*4);for(size_t i=0;i<n;i++)s[i]=v;return s;}
static const char *fname(int fmt){return fmt==1?"int8":fmt==2?"int4":fmt==8?"fp8-emul":"?";}

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
