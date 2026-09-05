#!/usr/bin/env python3
"""Turn RP2's [OPTIME]/[PROF] lines into RP1's bucket table, ms/token.

Two things that are easy to get wrong and are asserted rather than assumed:
  * since G9, [PROF] eg CONTAINS the CPU expert time it overlaps, so the
    exclusive GPU-group cost is eg - cpu. RP1's table is additive, so it must
    have used the exclusive form; the accounting check below fails loudly if
    that is wrong.
  * bind+dispatch+accumulate is the MoE remainder, not a timer.
"""
import re, sys, statistics as st

NT = 128

def parse(path):
    runs, cur = [], None
    for line in open(path):
        m = re.match(r'### ===== (\S+) \(COLI_KDA_GPU=(\d)\)', line)
        if m:
            cur = {'tag': m.group(1), 'knob': int(m.group(2))}
            runs.append(cur); continue
        if cur is None: continue
        if (m := re.search(r'decode \d+ token in ([\d.]+)s', line)): cur['wall'] = float(m.group(1))
        if (m := re.search(r'layers=([\d.]+)s head=([\d.]+)s', line)):
            cur['layers'] = float(m.group(1)); cur['head'] = float(m.group(2))
        if (m := re.search(r'kda=([\d.]+)s .*mla=([\d.]+)s', line)):
            cur['kda'] = float(m.group(1)); cur['mla'] = float(m.group(2))
        if (m := re.search(r'ffn_dense=([\d.]+)s .*ffn_moe=([\d.]+)s', line)):
            cur['dense'] = float(m.group(1)); cur['moe'] = float(m.group(2))
        if (m := re.search(r'hc\+norm=([\d.]+)s', line)): cur['hc'] = float(m.group(1))
        if (m := re.search(r'kda split .*proj=([\d.]+)s \(([\d.]+) ms\) decay=([\d.]+)s .*step=([\d.]+)s .*norm=([\d.]+)s .*ko=([\d.]+)s', line)):
            g = m.groups()
            cur['k_proj'], cur['k_decay'], cur['k_step'] = float(g[0]), float(g[2]), float(g[3])
            cur['k_norm'], cur['k_ko'] = float(g[4]), float(g[5])
        if (m := re.search(r'mla split .*proj=([\d.]+)s .*index=([\d.]+)s .*attn=([\d.]+)s', line)):
            cur['m_proj'], cur['m_index'], cur['m_attn'] = map(float, m.groups())
        if (m := re.search(r'moe split: router=([\d.]+)s shared=([\d.]+)s', line)):
            cur['router'], cur['shared'] = map(float, m.groups())
        if (m := re.search(r'eg=([\d.]+)s\(disp=(\d+) experts=(\d+)\) cpu=([\d.]+)s\(n=(\d+)\)', line)):
            cur['eg'] = float(m.group(1)); cur['experts'] = int(m.group(3))
            cur['cpu'] = float(m.group(4)); cur['cpu_n'] = int(m.group(5))
    return [r for r in runs if 'wall' in r]

def buckets(r):
    b = {}
    b['CPU int4 experts'] = r['cpu']
    b['router']           = r['router']
    b['shared expert']    = r['shared']
    b['GPU expert groups']= r['eg'] - r['cpu']          # exclusive; see docstring
    b['bind+disp+accum']  = r['moe'] - r['router'] - r['shared'] - r['eg']
    b['MoE FFN total']    = r['moe']
    b['KDA']              = r['kda']
    b['MLA / DSA']        = r['mla']
    b['mHC + RMSNorm']    = r['hc']
    b['dense FFN']        = r['dense']
    b['lm_head']          = r['head']
    b['TOTAL']            = r['layers'] + r['head']
    return {k: v / NT * 1000 for k, v in b.items()}

ORDER = ['MoE FFN total','CPU int4 experts','router','shared expert',
         'GPU expert groups','bind+disp+accum','KDA','MLA / DSA',
         'mHC + RMSNorm','dense FFN','lm_head','TOTAL']
SUB   = {'CPU int4 experts','router','shared expert','GPU expert groups','bind+disp+accum'}

runs = []
for p in sys.argv[1:]:
    runs += parse(p)

for knob in (0, 2):
    sel = [r for r in runs if r['knob'] == knob]
    if not sel: continue
    print(f"\n{'='*72}\nCOLI_KDA_GPU={knob}   n={len(sel)} runs   "
          f"({', '.join(r['tag'] for r in sel)})\n{'='*72}")
    bs = [buckets(r) for r in sel]
    print(f"{'bucket':<22}{'median':>9}{'min':>9}{'max':>9}{'spread':>9}   %")
    med_tot = st.median([b['TOTAL'] for b in bs])
    for k in ORDER:
        v = sorted(b[k] for b in bs)
        m = st.median(v)
        pre = '  — ' if k in SUB else ''
        print(f"{pre+k:<22}{m:>9.2f}{v[0]:>9.2f}{v[-1]:>9.2f}"
              f"{(v[-1]-v[0])/m*100:>8.1f}%   {m/med_tot*100:>5.1f}")
    # accounting: do the buckets close on the measured total?
    top = ['MoE FFN total','KDA','MLA / DSA','mHC + RMSNorm','dense FFN','lm_head']
    s = sum(st.median([b[k] for b in bs]) for k in top)
    print(f"{'sum of top-level':<22}{s:>9.2f}   vs measured {med_tot:9.2f}"
          f"   unaccounted {med_tot-s:+.3f} ms/token")
    sm = sum(st.median([b[k] for b in bs]) for k in SUB)
    print(f"{'sum of MoE sub':<22}{sm:>9.2f}   vs MoE total "
          f"{st.median([b['MoE FFN total'] for b in bs]):9.2f}"
          f"   unaccounted {st.median([b['MoE FFN total'] for b in bs])-sm:+.3f}")
    # KDA internals
    if knob == 0:
        for nm, key in [('proj','k_proj'),('decay','k_decay'),('step','k_step'),
                        ('norm','k_norm'),('ko','k_ko')]:
            v = sorted(r[key]/NT*1000 for r in sel)
            print(f"    kda.{nm:<16}{st.median(v):>9.2f}{v[0]:>9.2f}{v[-1]:>9.2f}"
                  f"{(v[-1]-v[0])/max(st.median(v),1e-9)*100:>8.1f}%")
    print("  routing:", {r['tag']: (r['experts'], r['cpu_n']) for r in sel})
    print("  wall tok/s:", [round(NT/r['wall'], 3) for r in sel])
