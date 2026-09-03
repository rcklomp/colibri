"""Add the GPU expert-batch fast path to q38_moe_decode -- the ACTUAL
single-token serving path (distinct from the prefill/chunk function patched
earlier, which only engages for multi-row prompt processing and is never
hit during normal chat/serve decode). This is why the first attempt showed
zero VRAM usage and no speedup: the code was correct but attached to a
function that decode never calls.

All K selected experts in q38_moe_decode read the SAME input row `xs` by
construction (single-token decode), so they batch directly -- no row/offset
bookkeeping needed, simpler than the prefill case.
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


OLD = """        Slot *selected[Q38_MAX_TOPK];
        int loaded_batch=q38_expert_get_batch(m,layer,idx,K,selected);
        for(int z=0;z<K;z++){
            Slot *ex=loaded_batch?selected[z]:q38_expert_get(m,layer,idx[z]);phase_started=now_s();
            q38_weight_matmul(eg,xs,&ex->gate,1,H,I);q38_weight_matmul(eu,xs,&ex->up,1,H,I);
            for(int j=0;j<I;j++)eh[j]=q38_silu(eg[j])*eu[j];q38_weight_matmul(eo,eh,&ex->down,1,I,H);
            for(int d=0;d<H;d++)ys[d]+=route_gates[z]*eo[d];
            q38_tm_add(m,Q38_TM_ROUTED_EXPERT,phase_started);
        }"""

NEW = """#ifdef Q38_VK_TIER
        int q38vk_handled=0;
        if(g_q38vk_ready&&K<=64){
            static ColiVkTensor *q38vk_gt[64],*q38vk_ut[64],*q38vk_dt[64];
            static int q38vk_rows[64];
            static float *q38vk_y=NULL;
            if(!q38vk_y) q38vk_y=falloc(64*(int64_t)H);
            int ok=1;
            for(int z=0;z<K;z++){
                if(!q38vk_expert_ensure(m,layer,idx[z])){ok=0;break;}
                ColiVkTensor **reg=q38vk_reg_at(layer,idx[z]);
                q38vk_gt[z]=reg[0];q38vk_ut[z]=reg[1];q38vk_dt[z]=reg[2];
                q38vk_rows[z]=1;
            }
            if(ok){
                phase_started=now_s();
                if(coli_vk_expert_group(q38vk_gt,q38vk_ut,q38vk_dt,q38vk_rows,K,q38vk_y,xs)){
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

rep(OLD, NEW, "q38_moe_decode GPU fast path")

open(p, "w").write(s)
print("QWEN38 DECODE GPU PATH APPLIED")
