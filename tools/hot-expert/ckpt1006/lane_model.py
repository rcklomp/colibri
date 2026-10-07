# E4: offline bound for the fetch levers, from GLM5.md "Optimal fetch assignment" (links: lone 417 us/slab, shared pair 678 us/slab when both fetch, 417 when one does).
import itertools
P = [.011,.067,.178,.268,.253,.152,.058,.012,.001]; LAYERS = 42
def t_cards(cl, ca, cb):
    return max(cl*417, ca*(678 if cb>0 else 417), cb*(678 if ca>0 else 417))
def best(n, tc=None, delta=0.0):
    b = 1e9
    for cl in range(n+1):
        for ca in range(n-cl+1):
            for cb in range(n-cl-ca+1):
                cc = n-cl-ca-cb
                if cc and tc is None: continue
                t = t_cards(cl,ca,cb)
                if cc: t = max(t, cc*tc + delta)
                b = min(b, t)
    return b
def avg(tc=None, delta=0.0):
    return sum(p*best(n, tc, delta) for n,p in enumerate(P))
base = avg()
print("n   : " + " ".join("%5d"%n for n in range(9)))
print("opt : " + " ".join("%5.0f"%best(n) for n in range(9)), " avg/layer %.0f us -> %.1f ms/token" % (base, base*LAYERS/1000))
# continuous bound. ERRATUM 2026-10-07: this used 52 GB/s, which is M4's all-three figure for an EQUAL split (3 GB over the slowest card's time, the lone card idles
# after 35 ms); with the loads balanced all three run at once (lone 28 + the shared pair 18 + 18 = 62-64 GB/s, the engine's own 62.4), and THAT is the bound.
for agg_gbs, why in ((52.0, "WRONG: M4's equal-split artifact"), (62.4, "balanced: 28 + 17.2 + 17.2, the engine's own figure")):
    agg = 11.67/(agg_gbs*1e3)*1e6   # us per slab
    cont = sum(p*n*agg for n,p in enumerate(P))
    print("continuous 3-link bound at %.1f GB/s (%s): %.0f us/layer -> %.1f ms/token (%.1f ms below the best slab split)" % (agg_gbs, why, cont, cont*LAYERS/1000, (base-cont)*LAYERS/1000))
mean_n = sum(n*p for n,p in enumerate(P)); print("mean slabs/layer %.2f" % mean_n)
print()
print("CPU lane (4th lane), per slab time tc and fixed overhead delta per layer that uses it; ms/token saved vs best slab split:")
print("   tc(us)  GB/s |  d=0   d=60  d=120  d=200")
for tc in (350, 450, 600, 800, 1000, 1400):
    row = []
    for d in (0, 60, 120, 200):
        row.append((base - avg(tc, d))*LAYERS/1000)
    print("  %6d %5.1f | %5.1f %5.1f %5.1f %5.1f" % (tc, 11.67/tc*1e3, *row))
# upper bound with CPU + a continuous split of the rest
print()
print("share of layers where a CPU slab helps (tc=600,d=60):", sum(p for n,p in enumerate(P) if best(n,600,60) < best(n)-1))
