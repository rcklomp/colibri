"""Replace q38_moe_decode's all-or-nothing single-device GPU dispatch with
per-device buckets: each of the K selected experts is placed on whichever
device (dev0/dev2/dev3) has room, each non-empty bucket dispatches through
its own coli_vk_expert_group call, and only experts that could not be placed
on ANY device fall back to CPU -- individually, not as a whole group.

Three cases, kept as separate straight-line code rather than a shared helper,
so each is easy to verify by inspection:
  - Q38_VK_TIER not compiled: original single CPU loop, byte-for-byte
    unchanged.
  - Compiled but GPU placed nothing this round (disabled, or every device
    full/failed): falls through to the SAME original batch-loading CPU loop,
    unchanged -- zero regression versus the pre-GPU-tier code path.
  - Compiled and GPU placed some/most of K: the small remainder is fetched
    one expert at a time (no batch-loading) -- simpler, and should be rare
    once dev0-3 have warmed up, so the lost batch-load parallelism doesn't
    matter in practice.

If a device's dispatch call itself fails (not a placement/budget issue --
an actual runtime failure), that device's bucket is treated as unplaced and
added to the CPU fallback list; nothing is double-counted or corrupted.
"""
import sys
p = sys.argv[1] if len(sys.argv) > 1 else "/home/ronald/src/colibri/c/qwen38_core.h"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


OLD = """#ifdef Q38_VK_TIER
        int q38vk_handled=0;
        if(g_q38vk_ready&&K<=64){
            static ColiVkTensor *q38vk_gt[64],*q38vk_ut[64],*q38vk_dt[64];
            static int q38vk_rows[64];
            static float *q38vk_x=NULL,*q38vk_y=NULL;
            if(!q38vk_x){q38vk_x=falloc(64*(int64_t)H);q38vk_y=falloc(64*(int64_t)H);}
            int ok=1;
            for(int z=0;z<K;z++){
                if(!q38vk_expert_ensure(m,layer,idx[z])){ok=0;break;}
                ColiVkTensor **reg=q38vk_reg_at(layer,idx[z]);
                q38vk_gt[z]=reg[0];q38vk_ut[z]=reg[1];q38vk_dt[z]=reg[2];
                q38vk_rows[z]=1;
                memcpy(q38vk_x+(int64_t)z*H,xs,(size_t)H*sizeof(float));
            }
            if(ok){
                phase_started=now_s();
                if(coli_vk_expert_group(q38vk_gt,q38vk_ut,q38vk_dt,q38vk_rows,K,q38vk_y,q38vk_x)){
                    for(int z=0;z<K;z++)
                        for(int d=0;d<H;d++)ys[d]+=route_gates[z]*q38vk_y[(int64_t)z*H+d];
                    q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
                    q38vk_handled=1;
                }
            }
        }
        if(!q38vk_handled){
#endif
        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=q38_expert_get_batch(m,layer,idx,K,selected);
        for(int z=0;z<K;z++){
            Slot *ex=loaded_batch?selected[z]:q38_expert_get(m,layer,idx[z]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
            q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
        }
#ifdef Q38_VK_TIER
        }
#endif"""

NEW = """#ifdef Q38_VK_TIER
        int q38vk_cpu_idx[Q38_MAX_TOPK], q38vk_cpu_n=K;
        for(int z=0;z<K;z++) q38vk_cpu_idx[z]=z;
        if(g_q38vk_ready&&K<=64){
            static ColiVkTensor *bufG[3][64],*bufU[3][64],*bufD[3][64];
            static int bufRows[3][64], bufZ[3][64], bufN[3];
            static float *bufX[3], *bufY[3];
            if(!bufX[0]) for(int dv=0;dv<3;dv++){ bufX[dv]=falloc(64*(int64_t)H); bufY[dv]=falloc(64*(int64_t)H); }
            bufN[0]=bufN[1]=bufN[2]=0;
            int placed_mask[Q38_MAX_TOPK]; for(int z=0;z<K;z++) placed_mask[z]=0;
            for(int z=0;z<K;z++){
                int dev=q38vk_expert_ensure(m,layer,idx[z]);
                if(dev<0||dev>2) continue;
                ColiVkTensor **reg=q38vk_reg_at(layer,idx[z]);
                int n=bufN[dev]++;
                bufG[dev][n]=reg[0]; bufU[dev][n]=reg[1]; bufD[dev][n]=reg[2];
                bufRows[dev][n]=1; bufZ[dev][n]=z;
                memcpy(bufX[dev]+(int64_t)n*H,xs,(size_t)H*sizeof(float));
            }
            phase_started=now_s();
            int any_gpu=0;
            for(int dev=0;dev<3;dev++){
                if(!bufN[dev]) continue;
                int rc;
                if(dev==0) rc=coli_vk_expert_group(bufG[0],bufU[0],bufD[0],bufRows[0],bufN[0],bufY[0],bufX[0]);
                else if(dev==1) rc=coli_vk_expert_group2(bufG[1],bufU[1],bufD[1],bufRows[1],bufN[1],bufY[1],bufX[1]);
                else rc=coli_vk_expert_group3(bufG[2],bufU[2],bufD[2],bufRows[2],bufN[2],bufY[2],bufX[2]);
                if(!rc) continue;   /* this device's bucket stays unplaced -> CPU fallback below */
                for(int n=0;n<bufN[dev];n++){
                    int z=bufZ[dev][n];
                    for(int d=0;d<H;d++)ys[d]+=route_gates[z]*bufY[dev][(int64_t)n*H+d];
                    placed_mask[z]=1; any_gpu=1;
                }
            }
            if(any_gpu){
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
                q38vk_cpu_n=0;
                for(int z=0;z<K;z++) if(!placed_mask[z]) q38vk_cpu_idx[q38vk_cpu_n++]=z;
            }
        }
        if(q38vk_cpu_n==K){
#endif
        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=q38_expert_get_batch(m,layer,idx,K,selected);
        for(int z=0;z<K;z++){
            Slot *ex=loaded_batch?selected[z]:q38_expert_get(m,layer,idx[z]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
            q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
        }
#ifdef Q38_VK_TIER
        } else if(q38vk_cpu_n>0){
            for(int zi=0;zi<q38vk_cpu_n;zi++){
                int z=q38vk_cpu_idx[zi];
                Slot *ex=q38_expert_get(m,layer,idx[z]);phase_started=now_s();
                q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
                for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
                for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
                q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
            }
        }
#endif"""

rep(OLD, NEW, "per-device bucket dispatch")

open(p, "w").write(s)
print("QWEN38 MULTI-GPU DISPATCH APPLIED")
