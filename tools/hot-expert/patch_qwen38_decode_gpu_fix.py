"""Fix: coli_vk_expert_group reads K*H floats starting at the x pointer it is
given (K virtual rows concatenated), but the decode-path GPU patch passed
`xs` directly -- a single H-float row. Experts z=1..K-1 read past the row
into adjacent memory. Replicate xs into a K-row scratch buffer first, exactly
as the (correct) prefill-path patch already does.
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


rep("""            static ColiVkTensor *q38vk_gt[64],*q38vk_ut[64],*q38vk_dt[64];
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
                if(coli_vk_expert_group(q38vk_gt,q38vk_ut,q38vk_dt,q38vk_rows,K,q38vk_y,xs)){""",
    """            static ColiVkTensor *q38vk_gt[64],*q38vk_ut[64],*q38vk_dt[64];
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
                if(coli_vk_expert_group(q38vk_gt,q38vk_ut,q38vk_dt,q38vk_rows,K,q38vk_y,q38vk_x)){""",
    "replicate xs into K-row batch buffer")

open(p, "w").write(s)
print("QWEN38 DECODE GPU X-BUFFER FIX APPLIED")
