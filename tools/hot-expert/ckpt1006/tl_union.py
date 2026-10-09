#!/usr/bin/env python3
"""tl_union.py -- PF0: what the prefill device timeline (--timeline CSV, glm5_timeline.py's input) says about pipelining in the IN-PLACE config (no DMA records exist there).

    python3 -I tl_union.py timeline.csv [--label NAME]

Prints (chunks after the first recorded one, the window = end of the first recorded chunk .. end of the last):
  period       per chunk: end of its last device work minus the previous chunk's (the throughput number) and the SPAN (first to last device work of the chunk)
  in flight    mean number of recorded chunks in flight over the window (span-weighted); the engine's two banks cap it at 2
  busy         per card, over the WINDOW (biased for upstream cards at a full pipeline, see the per-chunk lines below it): main-stream busy, helper-stream busy (help_batch, writeback, packet), their union, the union of the compute classes only
  packet       the longest help_packet of each chunk (two hipMemcpyPeerAsync: ids/plan + x rows) and where it sits: a >100 ms packet is a copy stuck behind another copy
  stall        for each such packet: the device event that ended within 30 ms of it (what it queued behind)
  chain        the owner's layer chain (layer_a start .. moe_c end) per layer, median over the chunks: the sum is the chunk latency
"""
import csv, collections, statistics, sys

def main():
    path = sys.argv[1]
    label = sys.argv[sys.argv.index('--label') + 1] if '--label' in sys.argv else path
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith('#'))]
    dev = [r for r in rows if r['stream'] != 'host']
    for r in dev:
        r['t0'] = float(r['t_start_us']); r['t1'] = float(r['t_end_us']); r['chunk'] = int(r['chunk'])
        r['card'] = int(r['card']); r['layer'] = int(r['layer']); r['owner'] = int(r['owner']); r['cls'] = r['class']
    WORK = {'layer_a', 'moe_b', 'moe_c', 'help_batch', 'help_writeback', 'embed_upload', 'head'}
    COPY = {'help_packet', 'boundary_copy', 'hout_copy'}
    def union(iv):
        iv = sorted(iv); tot = 0.0; cur = None
        for a, b in iv:
            if cur is None: cur = [a, b]
            elif a <= cur[1]: cur[1] = max(cur[1], b)
            else: tot += cur[1] - cur[0]; cur = [a, b]
        if cur: tot += cur[1] - cur[0]
        return tot
    cs = sorted({r['chunk'] for r in dev})
    span = {}
    for c in cs:
        rr = [r for r in dev if r['chunk'] == c and r['cls'] in WORK | COPY]
        span[c] = (min(r['t0'] for r in rr), max(r['t1'] for r in rr))
    ends = [span[c][1] for c in cs]
    print('== %s: chunks %s' % (label, cs))
    per = [(ends[i] - ends[i - 1]) / 1e6 for i in range(1, len(cs))]
    print('period s per chunk: ' + ' '.join('%d:%.3f' % (cs[i], per[i - 1]) for i in range(1, len(cs))) + '   mean %.3f s' % (sum(per) / len(per)))
    print('span s per chunk:   ' + ' '.join('%d:%.2f' % (c, (span[c][1] - span[c][0]) / 1e6) for c in cs))
    w0, w1 = ends[0], ends[-1]; W = w1 - w0
    inf = sum(max(0, min(span[c][1], w1) - max(span[c][0], w0)) for c in cs)
    print('window %.2f s; mean chunks in flight %.2f' % (W / 1e6, inf / W))
    def clip(r): return (max(r['t0'], w0), min(r['t1'], w1))
    for card in sorted({r['card'] for r in dev if r['card'] >= 0}):
        sel = lambda st, cl: [clip(r) for r in dev if r['card'] == card and r['stream'] == st and r['cls'] in cl and r['t1'] > w0 and r['t0'] < w1]
        m = sel('main', WORK | COPY); h = sel('help', WORK | COPY)
        comp = [clip(r) for r in dev if r['card'] == card and r['cls'] in WORK and r['t1'] > w0 and r['t0'] < w1]
        print('card %d: main-stream busy %.1f %%  helper-stream busy %.1f %%  union %.1f %%  union(compute) %.1f %%' % (
            card, 100 * union(m) / W, 100 * union(h) / W, 100 * union(m + h) / W, 100 * union(comp) / W))
    # PF9 (record §L5-PF9): the window shares above are BIASED for the upstream cards -- a chunk's span is far longer than the window when the pipeline is
    # full (23 s against 17 s at 32 chunks), so card 0 does its share of the recorded chunks BEFORE the window opens and reads ~0 %. The unbiased number
    # is per chunk: the card's work on that chunk (union of its intervals) against the steady period.
    steady = cs[1:]
    if len(steady) >= 2:
        pmean = statistics.mean(per[1:]) * 1e6 if len(per) > 1 else (w1 - w0) / (len(cs) - 1)
        print('per chunk (chunks %s), seconds of device work and share of the %.3f s steady period (chunks after the first two):' % (steady, pmean / 1e6))
        for card in sorted({r['card'] for r in dev if r['card'] >= 0}):
            acc = []
            for c in steady:
                f = lambda st: union([(r['t0'], r['t1']) for r in dev if r['chunk'] == c and r['card'] == card and r['cls'] in WORK | COPY and (st is None or r['stream'] == st)])
                acc.append((f('main'), f('help'), f(None)))
            m = [statistics.mean(a[i] for a in acc) / 1e6 for i in range(3)]
            print('  card %d: main-stream work %.2f s (%.0f %%)  helper streams %.2f s (%.0f %%)  union %.2f s' % (card, m[0], 100e6 * m[0] / pmean, m[1], 100e6 * m[1] / pmean, m[2]))
        print('  (a main stream at 100 % binds the period; the helper streams run beside it, so the shares do not add)')
    print('longest help_packet per chunk (ms) and what it ended behind:')
    for c in cs:
        pk = sorted([r for r in dev if r['chunk'] == c and r['cls'] == 'help_packet'], key=lambda r: r['t0'] - r['t1'])[:2]
        for p in pk:
            d = (p['t1'] - p['t0']) / 1e3
            if d < 100: continue
            near = sorted([r for r in dev if r is not p and r['cls'] not in ('help_wait', 'join_wait', 'upstream_wait') and abs(r['t1'] - p['t1']) < 30000],
                          key=lambda r: abs(r['t1'] - p['t1']))[:2]
            print('  chunk %d card %d owner %d L%d: %.0f ms; ended with ' % (c, p['card'], p['owner'], p['layer'], d) +
                  '; '.join('card%d %s L%d chunk %d (%+.1f ms)' % (r['card'], r['cls'], r['layer'], r['chunk'], (r['t1'] - p['t1']) / 1e3) for r in near))
    by = collections.defaultdict(dict)
    for r in dev:
        if r['stream'] == 'main' and r['cls'] in ('layer_a', 'moe_c'): by[(r['chunk'], r['layer'])][r['cls']] = (r['t0'], r['t1'])
    lat = collections.defaultdict(list)
    for (c, l), d in by.items():
        if c <= cs[1] and False: continue
        if 'layer_a' in d: lat[l].append(((d.get('moe_c') or d['layer_a'])[1] - d['layer_a'][0]) / 1e3)
    med = {l: statistics.median(v) for l, v in lat.items()}
    big = {l: round(v) for l, v in med.items() if v > 400}
    print('owner chain: sum of per-layer medians %.0f ms; layers over 400 ms: %s; median of the rest %.0f ms' % (
        sum(med.values()), big, statistics.median([v for v in med.values() if v <= 400])))

if __name__ == '__main__':
    main()
