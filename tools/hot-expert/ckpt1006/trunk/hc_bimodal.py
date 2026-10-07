import csv, re, sys, statistics as st, collections
ks=[]
for r in csv.DictReader(open(sys.argv[1])):
    try: ks.append((int(r["Start_Timestamp"]),int(r["End_Timestamp"]),r["Agent_Id"],r["Kernel_Name"],int(r["Grid_Size_X"]),int(r["Grid_Size_Y"])))
    except: pass
ks.sort()
ag=sorted(set(k[2] for k in ks)); by={a:[k for k in ks if k[2]==a] for a in ag}
heads=[k for k in by[ag[-1]] if "k_glm5_hc_mean" in k[3]]; ends=[h[1] for h in heads]
toks=range(len(ends)-12,len(ends))
pos=collections.defaultdict(list); prevs=collections.defaultdict(list)
for ti in toks:
    ta,tb=ends[ti-1],ends[ti]
    for a in ag:
        w=[k for k in by[a] if k[0]>=ta and k[1]<=tb]
        hcs=[(i,k) for i,k in enumerate(w) if "k_gemm_batch" in k[3] and k[4]==131072]
        for j,(i,k) in enumerate(hcs):
            d=(k[1]-k[0])/1e3
            pos[(a, "first-on-card" if j==0 else ("attn" if j%2==0 else "ffn"))].append(d)
            # idle before
            idle=(k[0]-w[i-1][1])/1e3 if i>0 else -1
            prevs["slow" if d>15 else "fast"].append(re.search(r"k_[A-Za-z0-9_]+",w[i-1][3]).group(0) if i>0 else "none")
for key,v in sorted(pos.items()): print(key, len(v), "med %.1f p90 %.1f  >15us: %d"%(st.median(v), sorted(v)[int(.9*(len(v)-1))], sum(1 for x in v if x>15)))
for k,v in prevs.items(): print(k, collections.Counter(v).most_common(4))
