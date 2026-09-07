"""Dump the qwen38 phase timers and the mapping counters to stderr.

q38_format_prof only emits a PROF frame over the serve wire protocol, which the
HTTP path never surfaces. This prints the same breakdown plus the mmap
serve/copy counters after every request, so the actual bottleneck is visible.
"""
p = "/home/ronald/src/colibri/c/qwen38.c"
s = open(p).read()

OLD = """    int count=snprintf(out,capacity,
        "PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\\n",
        wall_s,prompt_tokens,completion_tokens,expert_read,
        expert_read+fp8_expand,expert_compute,attention,
        timers->seconds[Q38_TM_LM_HEAD],
        (unsigned long long)timers->forwards);"""

NEW = """    if(getenv("Q38_PROF")){
        double total=wall_s>0?wall_s:1.0;
        fprintf(stderr,
          "[Q38PROF] wall=%.1fs gen=%d | expert_read=%.2fs(%.0f%%) fp8_expand=%.2fs(%.0f%%) "
          "expert_compute=%.2fs(%.0f%%) attention=%.2fs(%.0f%%) lm_head=%.2fs(%.0f%%)\\n",
          wall_s,completion_tokens,
          expert_read,100*expert_read/total,
          fp8_expand,100*fp8_expand/total,
          expert_compute,100*expert_compute/total,
          attention,100*attention/total,
          timers->seconds[Q38_TM_LM_HEAD],100*timers->seconds[Q38_TM_LM_HEAD]/total);
        fprintf(stderr,
          "[Q38PROF] deltanet=%.2fs qsa_index=%.2fs qsa_attn=%.2fs dense_matmul=%.2fs ple=%.2fs "
          "| map_serve=%ld map_copy=%ld\\n",
          timers->seconds[Q38_TM_DELTANET],timers->seconds[Q38_TM_QSA_INDEX],
          timers->seconds[Q38_TM_QSA_ATTENTION],timers->seconds[Q38_TM_DENSE_MATMUL],
          timers->seconds[Q38_TM_PLE],q38_map_serve,q38_map_copy);
    }
    int count=snprintf(out,capacity,
        "PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\\n",
        wall_s,prompt_tokens,completion_tokens,expert_read,
        expert_read+fp8_expand,expert_compute,attention,
        timers->seconds[Q38_TM_LM_HEAD],
        (unsigned long long)timers->forwards);"""

assert s.count(OLD) == 1, "prof anchor"
s = s.replace(OLD, NEW)
open(p, "w").write(s)
print("Q38 PROF APPLIED")
