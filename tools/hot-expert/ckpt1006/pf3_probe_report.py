#!/usr/bin/env python3
"""pf3_probe_report.py -- the table behind PF3 route B's step 1 (build_pf3probe.sh, glm_pf3_probe_chain.sh).

    python3 -I pf3_probe_report.py gate_run.log

Input: lines `pf3 card=C il=L dsa=D cond=NAME run=a|b trunk_ms=X expert_ms=Y` (ms a rep; the forward pass is run=a, the reverse run=b), grouped by the `=== gate-plan config NAME` header.
For every (config, card, layer) it prints, in ms a rep (mean of a and b):
  alone         T = the trunk alone, X = the expert phase alone (all 48 WGPs)
  CU scaling    T on 40 / 32 / 24 / 16 of the card's 48 units (WGPs) and X on 8 / 16 / 24 / 32, alone: how much of each side needs the CUs (flat = it waits for something else)
  partitions    B_part_Xa_Tb = the expert stream masked to a units and the trunk stream to the other b
  both at once  per condition: the trunk stream's and the expert stream's finish time, the total (the later of the two), the SAVING against running them one after the other
                (T + X - total) and the overlap = saving / min(T, X)  (0 = they add up, 1 = the shorter one is hidden completely), and the stretch of each side against its alone time.
The plan's stop rule (section 0d): overlap < 20 % of the shorter side, or a predicted gain < 0.3 ms/token -> route B stops.
The prediction line scales the saving of the best condition to a chunk: card layers x saving, against the 3.375 s steady period (an ESTIMATE: it assumes the other chunk's trunk is always there
to hide the expert phase, which route B would have to arrange).
"""
import re, sys, collections

LINE = re.compile(r'^pf3 card=(\d+) il=(\d+) dsa=(\d) cond=(\S+) run=([ab]) trunk_ms=([\d.]+) expert_ms=([\d.]+)')
data = collections.OrderedDict()
cfg = '?'
for l in open(sys.argv[1], errors='replace'):
    if l.startswith('=== gate-plan config'):
        cfg = l.split()[3]
    m = LINE.match(l)
    if not m:
        continue
    card, il, dsa, cond, run, tr, ex = int(m[1]), int(m[2]), int(m[3]), m[4], m[5], float(m[6]), float(m[7])
    data.setdefault((cfg, card, il, dsa), collections.defaultdict(list))[cond].append((tr, ex))

if not data:
    print('no pf3 lines in', sys.argv[1]); sys.exit(1)

def mean(v): return sum(v) / len(v) if v else float('nan')
OUT = []      # runs dropped as stalls (one run in 120 took 3.9 s a rep against 0.07: a one-off GPU stall, probe_rec card 2 layer 30 B_trunk_hi run a)
PERIOD = 3375.0   # ms a 1 024-row chunk, PF9 steady state
LAYERS = 15       # layers a card owns
best_all = []
for (cfg, card, il, dsa), c in data.items():
    ref = max(mean([a for a, b in c['T_alone']]), mean([b for a, b in c['X_alone']]))
    for k in list(c):
        keep = [(a, b) for a, b in c[k] if max(a, b) <= 10 * ref]
        if len(keep) != len(c[k]):
            OUT.append('%s card %d layer %d %s: dropped %d run(s) of %s ms' % (cfg, card, il, k, len(c[k]) - len(keep), ','.join('%.0f' % max(a, b) for a, b in c[k] if max(a, b) > 10 * ref)))
            c[k] = keep
    T = mean([a for a, b in c['T_alone']]); X = mean([b for a, b in c['X_alone']])
    print('== %s card %d layer %d (%s): trunk alone %.1f ms, expert phase alone %.1f ms  (a/b spread: T %.1f X %.1f)' % (
        cfg, card, il, 'DSA' if dsa else 'KDA', T, X,
        abs(c['T_alone'][0][0] - c['T_alone'][-1][0]) if len(c['T_alone']) > 1 else 0, abs(c['X_alone'][0][1] - c['X_alone'][-1][1]) if len(c['X_alone']) > 1 else 0))
    def sc(ks, col): return ' / '.join('%.1f' % mean([x[col] for x in c[k]]) if c[k] else '-' for k in ks)
    print('   CU scaling, alone, ms a rep (48 units = the whole card = the alone time above):')
    print('     trunk on 40 / 32 / 24 / 16 units: %s     expert on 8 / 16 / 24 / 32 units: %s' % (
        sc(('T_on_40', 'T_on_32', 'T_on_24', 'T_on_16'), 0), sc(('X_on_8', 'X_on_16', 'X_on_24', 'X_on_32'), 1)))
    print('   %-17s %9s %9s %9s %9s %8s %9s %9s' % ('both at once', 'trunk', 'expert', 'total', 'saving', 'overlap', 'T stretch', 'X stretch'))
    best = None
    LBL = {}
    for k in ('B_default', 'B_trunk_hi', 'B_expert_hi', 'B_part_X8_T40', 'B_part_X16_T32', 'B_part_X24_T24', 'B_part_X32_T16'):
        if not c[k]:
            continue
        t = mean([a for a, b in c[k]]); x = mean([b for a, b in c[k]]); tot = max(t, x)
        sav = T + X - tot; ov = sav / min(T, X)
        print('   %-17s %9.1f %9.1f %9.1f %9.1f %7.0f %% %9.2f %9.2f' % (LBL.get(k, k), t, x, tot, sav, 100 * ov, t / T, x / X))
        if best is None or sav > best[1]:
            best = (k, sav, ov)
    if best:
        gain_s = LAYERS * best[1] / 1000.0
        print('   best: %s saves %.1f ms a layer pair = %.2f s on %d layers = %.2f ms/token of the 1 024-row chunk (estimate, ceiling of route B on this card; period %.3f s = %.2f ms/token)' % (
            best[0], best[1], gain_s, LAYERS, gain_s * 1000 / 1024.0, PERIOD / 1000, PERIOD / 1024.0))
        best_all.append((cfg, card, il, dsa, best))
# prediction per config and card: the saving of a layer pair x the card's layers of that kind, for B_default (what a second stream gives with no tuning) and for the best unmasked condition
NMOE = {0: (9, 3), 1: (11, 4), 2: (11, 4)}   # (KDA MoE layers, DSA MoE layers) a card owns (placement: 12 / 15 / 15 MoE layers, 3 / 4 / 4 DSA)
byc = collections.defaultdict(dict)
for (cfg, card, il, dsa), c in data.items():
    T = mean([a for a, b in c['T_alone']]); X = mean([b for a, b in c['X_alone']])
    sv = {}
    for k in ('B_default', 'B_trunk_hi', 'B_expert_hi'):
        if c[k]:
            sv[k] = T + X - max(mean([a for a, b in c[k]]), mean([b for a, b in c[k]]))
    byc[(cfg, card)][dsa] = sv
print('== predicted saving per 1 024-row chunk on a card (an ESTIMATE; layer pairs by kind x the layers of that kind a card owns; the binding card 2 sets the period 3.375 s = 3.30 ms/token):')
for (cfg, card), d in byc.items():
    if 0 in d and 1 in d:
        nk, nd = NMOE[card]
        dflt = nk * d[0].get('B_default', 0) + nd * d[1].get('B_default', 0)
        best = nk * max(d[0].values()) + nd * max(d[1].values())
        print('   %-12s card %d: default %.0f ms = %.3f ms/token ; best unmasked condition %.0f ms = %.3f ms/token' % (cfg, card, dflt, dflt / 1024.0, best, best / 1024.0))
for o in OUT: print('   (outlier)', o)
print('== summary (best condition per layer): ' + '; '.join('%s c%d l%d %s %s overlap %.0f %%' % (cfg, card, il, 'DSA' if dsa else 'KDA', b[0], 100 * b[2]) for cfg, card, il, dsa, b in best_all))
