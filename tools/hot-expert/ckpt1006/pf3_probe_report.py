#!/usr/bin/env python3
"""pf3_probe_report.py -- the table behind PF3 route B's step 1 (build_pf3probe.sh, glm_pf3_probe_chain.sh).

    python3 -I pf3_probe_report.py gate_run.log

Input: lines `pf3 card=C il=L dsa=D cond=NAME run=a|b trunk_ms=X expert_ms=Y` (ms a rep; the forward pass is run=a, the reverse run=b), grouped by the `=== gate-plan config NAME` header.
For every (config, card, layer) it prints, in ms a rep (mean of a and b):
  alone         T = the trunk alone, X = the expert phase alone (all 48 WGPs)
  CU scaling    T on 40 / 32 / 24 WGPs and X on 8 / 16 / 24 WGPs, alone: how much of each side needs the CUs (flat = it waits for something else)
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
PERIOD = 3375.0   # ms a 1 024-row chunk, PF9 steady state
LAYERS = 15       # layers a card owns
best_all = []
for (cfg, card, il, dsa), c in data.items():
    T = mean([a for a, b in c['T_alone']]); X = mean([b for a, b in c['X_alone']])
    print('== %s card %d layer %d (%s): trunk alone %.1f ms, expert phase alone %.1f ms  (a/b spread: T %.1f X %.1f)' % (
        cfg, card, il, 'DSA' if dsa else 'KDA', T, X,
        abs(c['T_alone'][0][0] - c['T_alone'][-1][0]) if len(c['T_alone']) > 1 else 0, abs(c['X_alone'][0][1] - c['X_alone'][-1][1]) if len(c['X_alone']) > 1 else 0))
    print('   CU scaling (alone, ms a rep; 48 WGPs = all):  trunk on 40 / 32 / 24 WGPs: %s    expert on 8 / 16 / 24 WGPs: %s' % (
        ' / '.join('%.1f' % mean([a for a, b in c[k]]) for k in ('T_on_40w', 'T_on_32w', 'T_on_24w')),
        ' / '.join('%.1f' % mean([b for a, b in c[k]]) for k in ('X_on_8w', 'X_on_16w', 'X_on_24w'))))
    print('   %-14s %9s %9s %9s %9s %8s %9s %9s' % ('both at once', 'trunk', 'expert', 'total', 'saving', 'overlap', 'T stretch', 'X stretch'))
    best = None
    for k in ('B_default', 'B_trunk_hi', 'B_expert_hi', 'B_part_8x40', 'B_part_16x32', 'B_part_24x24'):
        if not c[k]:
            continue
        t = mean([a for a, b in c[k]]); x = mean([b for a, b in c[k]]); tot = max(t, x)
        sav = T + X - tot; ov = sav / min(T, X)
        print('   %-14s %9.1f %9.1f %9.1f %9.1f %7.0f %% %9.2f %9.2f' % (k, t, x, tot, sav, 100 * ov, t / T, x / X))
        if best is None or sav > best[1]:
            best = (k, sav, ov)
    if best:
        gain_s = LAYERS * best[1] / 1000.0
        print('   best: %s saves %.1f ms a layer pair = %.2f s on %d layers = %.2f ms/token of the 1 024-row chunk (estimate, ceiling of route B on this card; period %.3f s = %.2f ms/token)' % (
            best[0], best[1], gain_s, LAYERS, gain_s * 1000 / 1024.0, PERIOD / 1000, PERIOD / 1024.0))
        best_all.append((cfg, card, il, dsa, best))
print('== summary (best condition per layer): ' + '; '.join('%s c%d l%d %s %s overlap %.0f %%' % (cfg, card, il, 'DSA' if dsa else 'KDA', b[0], 100 * b[2]) for cfg, card, il, dsa, b in best_all))
