#!/usr/bin/env python3
"""Compare two S-format routing traces positionally.
S <layer> <token> <rank> <eid> <score>   -- 16 ranks per decision point.
Token index t repeats across decode steps, so we compare in FILE ORDER."""
import sys, collections

TOPK = 8

def load(path):
    recs = []   # list of (layer, token, [(eid,score)]*16)
    cur = []
    for line in open(path):
        p = line.split()
        if not p or p[0] != "S": continue
        lay, tok, rank, eid, sc = int(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5])
        if rank == 0 and cur:
            recs.append(cur); cur = []
        if rank == 0:
            cur = [lay, tok, []]
        cur[2].append((eid, sc))
    if cur: recs.append(cur)
    return recs

def hist_topn(recs):
    """global (layer,eid) usage histogram from top-8 selections"""
    h = collections.Counter()
    for lay, tok, rs in recs:
        for eid, sc in rs[:TOPK]:
            h[(lay, eid)] += 1
    return h

def topn_set(h, n):
    return set(k for k, _ in h.most_common(n))

def hitrate(recs, resident):
    hit = tot = 0
    for lay, tok, rs in recs:
        for eid, sc in rs[:TOPK]:
            tot += 1
            if (lay, eid) in resident: hit += 1
    return hit, tot

def main(pa, pb, na, nb):
    A, B = load(pa), load(pb)
    print("=" * 72)
    print("TRACE COMPARISON: %s (%s) vs %s (%s)" % (na, pa, nb, pb))
    print("=" * 72)
    print("records: %s=%d  %s=%d" % (na, len(A), nb, len(B)))
    n = min(len(A), len(B))
    if len(A) != len(B):
        print("!! record count differs - comparing first %d" % n)

    misaligned = sum(1 for i in range(n) if A[i][0] != B[i][0] or A[i][1] != B[i][1])
    print("positional (layer,token) misalignment: %d / %d" % (misaligned, n))

    # ---- per-rank differences ----
    buckets = {"tie<1e-6": 0, "1e-6..1e-4": 0, "1e-4..1e-3": 0, "1e-3..1e-2": 0, ">=1e-2": 0}
    diff_rows = []
    rank_diffs = collections.Counter()
    layer_diffs = collections.Counter()
    ndec = 0
    for i in range(n):
        lay, tok, ra = A[i][0], A[i][1], A[i][2]
        rb = B[i][2]
        for k in range(min(len(ra), len(rb))):
            ndec += 1
            if ra[k][0] != rb[k][0]:
                d = abs(ra[k][1] - rb[k][1])
                # closeness of the local tie in run A (gap to neighbour rank)
                gap = abs(ra[k][1] - ra[k+1][1]) if k+1 < len(ra) else float("nan")
                if d < 1e-6: buckets["tie<1e-6"] += 1
                elif d < 1e-4: buckets["1e-6..1e-4"] += 1
                elif d < 1e-3: buckets["1e-4..1e-3"] += 1
                elif d < 1e-2: buckets["1e-3..1e-2"] += 1
                else: buckets[">=1e-2"] += 1
                rank_diffs[k] += 1
                layer_diffs[lay] += 1
                diff_rows.append((lay, i, k, ra[k][0], ra[k][1], rb[k][0], rb[k][1], d, gap))
    print("\n-- per-rank routing differences --")
    print("differing (rank slots): %d / %d  (%.2f%%)" % (len(diff_rows), ndec, 100.0*len(diff_rows)/max(ndec,1)))
    print("score-delta buckets at differing slots:")
    for k, v in buckets.items():
        print("   %-12s %6d  (%.1f%%)" % (k, v, 100.0*v/max(len(diff_rows),1)))
    print("by rank position:", dict(sorted(rank_diffs.items())))

    # ---- what actually matters: the top-8 SET ----
    setdiff_tot = setdiff_pts = 0
    for i in range(n):
        sa = set(e for e, _ in A[i][2][:TOPK])
        sb = set(e for e, _ in B[i][2][:TOPK])
        d = len(sa - sb)
        if d: setdiff_pts += 1
        setdiff_tot += d
    print("\n-- top-%d SET differences (what the engine actually computes) --" % TOPK)
    print("decision points with any set change: %d / %d (%.2f%%)" % (setdiff_pts, n, 100.0*setdiff_pts/max(n,1)))
    print("experts swapped: %d / %d selections (%.2f%%)" % (setdiff_tot, n*TOPK, 100.0*setdiff_tot/max(n*TOPK,1)))

    if layer_diffs:
        print("\n-- layer concentration (top 10 by differing slots) --")
        for lay, c in layer_diffs.most_common(10):
            print("   layer %2d: %d" % (lay, c))

    print("\n-- sample differing decisions (first 15) --")
    print("%5s %7s %4s | %5s %12s | %5s %12s | %11s %11s" %
          ("layer","recIdx","rank","eidA","scoreA","eidB","scoreB","|delta|","tieGapA"))
    for r in diff_rows[:15]:
        print("%5d %7d %4d | %5d %12.7f | %5d %12.7f | %11.3e %11.3e" % r)

    # ---- heat-map transferability ----
    hA, hB = hist_topn(A), hist_topn(B)
    print("\n-- heat-map transferability (global top-N over (layer,eid)) --")
    print("%8s %10s %10s %12s %12s" % ("N","overlap%","jaccard%","hit(A-set on B)","hit(B-set on B)"))
    for N in (100, 500, 1000, 1695, 3295, 3391):
        sa, sb = topn_set(hA, N), topn_set(hB, N)
        inter = len(sa & sb)
        ov = 100.0*inter/max(min(len(sa),len(sb)),1)
        ja = 100.0*inter/max(len(sa|sb),1)
        ha, ta = hitrate(B, sa)
        hb, tb = hitrate(B, sb)
        print("%8d %9.1f%% %9.1f%% %11.1f%% %11.1f%%" %
              (N, ov, ja, 100.0*ha/max(ta,1), 100.0*hb/max(tb,1)))
    print("(hit(A-set on B) = mismatched heat map; hit(B-set on B) = config-matched ceiling)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else "A", sys.argv[4] if len(sys.argv)>4 else "B")
