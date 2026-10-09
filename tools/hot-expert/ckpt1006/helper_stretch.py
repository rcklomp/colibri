#!/usr/bin/env python3
"""PF3 step 1a (record §L5-PF3 point 2): the other half of the natural experiment. Does a HELPER's expert interval (help_batch: a card reading and computing its share of another
card's layer) get longer when the card's MAIN stream runs a trunk interval (layer_a) or its own expert interval (moe_b) beside it?

    python3 -I helper_stretch.py timeline32.csv

Per card and per (owner, layer): the 6 chunks' help_batch lengths are normalised by the smallest; the mean normalised length is printed for 4 states of the
main stream during the interval: IDLE (< 10 % covered), TRUNK (layer_a covers >= 50 %), OWN-EXPERTS (moe_b covers >= 50 %), MIXED (the rest). Only chunks whose
neighbours are all recorded count (the same steady windows as overlap_from_timeline.py).
"""
import csv, sys, collections, statistics
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith('#'))]
dev = [r for r in rows if r['stream'] != 'host']
for r in dev:
    r['t0'] = float(r['t_start_us']); r['t1'] = float(r['t_end_us']); r['chunk'] = int(r['chunk'])
    r['card'] = int(r['card']); r['layer'] = int(r['layer']); r['cls'] = r['class']; r['owner'] = int(r['owner'])
def merge(iv):
    iv = sorted(iv); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
def cover(a, b, m):
    s = 0.0
    for x, y in m:
        if y <= a: continue
        if x >= b: break
        s += min(b, y) - max(a, x)
    return s
# the helper chunk window: the card's main works on chunk m; it helps owner o's chunk about m + (card - o)*(-1)... use the recorded steady chunks of the MAIN stream
# windows as in overlap_from_timeline.py: a help interval counts if it lies inside the card's main-stream steady window
WIN = {0: (18, 21), 1: (17, 20), 2: (16, 19)}
for card in (0, 1, 2):
    lo, hi = WIN[card]
    mw = [r for r in dev if r['card'] == card and r['stream'] == 'main' and lo <= r['chunk'] <= hi]
    W0 = min(r['t0'] for r in mw); W1 = max(r['t1'] for r in mw)
    A = merge([(r['t0'], r['t1']) for r in dev if r['card'] == card and r['stream'] == 'main' and r['cls'] == 'layer_a'])
    B = merge([(r['t0'], r['t1']) for r in dev if r['card'] == card and r['stream'] == 'main' and r['cls'] == 'moe_b'])
    hb = [r for r in dev if r['card'] == card and r['stream'] == 'help' and r['cls'] == 'help_batch']
    grp = collections.defaultdict(list)
    for r in hb: grp[(r['owner'], r['layer'])].append(r)
    res = collections.defaultdict(list); ab = collections.defaultdict(float)
    for (o, lay), v in grp.items():
        if len(v) < 3: continue
        base = min(x['t1'] - x['t0'] for x in v)
        for x in v:
            if not (W0 <= x['t0'] and x['t1'] <= W1): continue
            L = x['t1'] - x['t0']; fa = cover(x['t0'], x['t1'], A) / L; fb = cover(x['t0'], x['t1'], B) / L
            st = 'IDLE' if fa + fb < 0.1 else ('TRUNK' if fa >= 0.5 else ('OWN-EXPERTS' if fb >= 0.5 else 'MIXED'))
            res[st].append(L / base); ab[st] += L
    print('card %d (steady window %.1f s): help_batch normalised length by what the main stream runs beside it' % (card, (W1 - W0) / 1e6))
    for st in ('IDLE', 'TRUNK', 'OWN-EXPERTS', 'MIXED'):
        if res[st]: print('   %-12s n=%3d  mean %.3f  median %.3f   (%.2f s of helper time)' % (st, len(res[st]), statistics.mean(res[st]), statistics.median(res[st]), ab[st] / 1e6))
