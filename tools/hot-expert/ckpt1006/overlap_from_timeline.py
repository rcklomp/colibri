#!/usr/bin/env python3
"""PF3 step 1a (record §L5-PF3 point 2): from the in-place device timeline (PF9, chunks 16-21), how much do a card's MAIN stream and its HELPER streams already overlap in time,
and does a main-stream interval get longer when helper compute runs beside it?

    python3 -I overlap_from_timeline.py timeline32.csv

Per card:
  sum / union     main busy + helper busy against the union of both (seconds per chunk): sum - union = time both streams were active
  stretch         for each main-stream work interval (layer_a, moe_b, moe_c) the fraction f of its length covered by helper compute (help_batch / help_writeback)
                  on the same card; per (card, layer, class) the intervals of the 6 chunks are normalised by the smallest of them, and the normalised length is
                  averaged in bins of f. A flat line = the helper work costs the main stream nothing; a rising line = they share the card.
"""
import csv, sys, collections, statistics

rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith('#'))]
dev = [r for r in rows if r['stream'] != 'host']
for r in dev:
    r['t0'] = float(r['t_start_us']); r['t1'] = float(r['t_end_us']); r['chunk'] = int(r['chunk'])
    r['card'] = int(r['card']); r['layer'] = int(r['layer']); r['cls'] = r['class']

def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def total(m): return sum(b - a for a, b in m)

def cover(a, b, m):
    s = 0.0
    for x, y in m:
        if y <= a: continue
        if x >= b: break
        s += min(b, y) - max(a, x)
    return s

MAINW = {'layer_a', 'moe_b', 'moe_c'}
HELPW = {'help_batch', 'help_writeback'}
cs = sorted({r['chunk'] for r in dev})
WIN = {0: (18, 21), 1: (17, 20), 2: (16, 19)}   # main-stream chunks whose neighbours' (helper) chunks are all recorded
for card in (0, 1, 2):
    lo, hi = WIN[card]; nchunk = hi - lo + 1
    mw = [r for r in dev if r['card'] == card and r['stream'] == 'main' and lo <= r['chunk'] <= hi and r['cls'] in {'layer_a','moe_b','moe_c','join_wait'}]
    W0 = min(r['t0'] for r in mw); W1 = max(r['t1'] for r in mw); Wl = W1 - W0
    def clipw(iv): return [(max(a, W0), min(b, W1)) for a, b in iv if b > W0 and a < W1]
    mi = clipw([(r['t0'], r['t1']) for r in dev if r['card'] == card and r['stream'] == 'main' and r['cls'] in MAINW])
    hv = clipw([(r['t0'], r['t1']) for r in dev if r['card'] == card and r['stream'] == 'help' and r['cls'] in HELPW])
    mm, hm = merge(mi), merge(hv)
    both = merge(mi + hv)
    both_active = total(mm) + total(hm) - total(both)
    print('card %d: steady window = main chunks %d-%d, %.2f s (%.3f s per chunk)' % (card, lo, hi, Wl / 1e6, Wl / 1e6 / nchunk))
    print('   main compute active %.1f %%   helper compute active %.1f %%   both %.1f %%   either %.1f %%   neither %.1f %%   (of the window)' % (
        100 * total(mm) / Wl, 100 * total(hm) / Wl, 100 * both_active / Wl, 100 * total(both) / Wl, 100 * (Wl - total(both)) / Wl))

    # stretch of the main stream's intervals against helper cover
    hm_full = merge([(r['t0'], r['t1']) for r in dev if r['card'] == card and r['stream'] == 'help' and r['cls'] in HELPW])
    groups = collections.defaultdict(list)
    for r in dev:
        if r['card'] == card and r['stream'] == 'main' and r['cls'] in MAINW and lo <= r['chunk'] <= hi:
            L = r['t1'] - r['t0']
            f = cover(r['t0'], r['t1'], hm_full) / L if L > 0 else 0.0
            groups[(r['layer'], r['cls'])].append((L, f))
    for cl in sorted(MAINW):
        bins = collections.defaultdict(list)
        for (lay, c), v in groups.items():
            if c != cl or len(v) < 3: continue
            base = min(x[0] for x in v)
            for L, f in v:
                bins[min(int(f * 5), 4)].append(L / base)
        print('   %-8s normalised length by helper cover 0-20-40-60-80-100 %%: %s' % (cl, '  '.join(
            '%d%%:%.3f(n=%d)' % (b * 20, statistics.mean(bins[b]), len(bins[b])) if bins[b] else '%d%%:-' % (b * 20) for b in range(5))))
    # the same, only for the longest class (moe_b) in absolute ms
    tot = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for (lay, c), v in groups.items():
        for L, f in v:
            t = tot[c]; t[0] += L; t[1] += f * L; t[2] += 1
    print('   share of each main class covered by helper compute: ' + '  '.join('%s %.0f %%' % (c, 100 * t[1] / t[0]) for c, t in sorted(tot.items())))
