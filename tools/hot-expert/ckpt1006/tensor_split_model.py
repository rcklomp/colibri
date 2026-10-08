#!/usr/bin/env python3
"""tensor_split_model.py -- D9a of DECODE-OPEN-ITEMS-PLAN-2026-10-08.md: can the tensor-level split of a missed expert (gate+up on one card, down on another, bit-exact) reach >= 1.5 ms a token?
Offline model, no GPU, python3 stdlib only. Extends lane_model.py (GLM5.md "Optimal fetch assignment") with
  * the corrected balanced link bound (record §L5-GLM-FETCH-FLOOR-ERRATUM: lone card 28 GB/s, the shared pair 17.2 GB/s each when BOTH fetch and ~28 when only one does = 62.4 GB/s balanced),
  * the miss distribution after the placement policy: P(n) of the old table IS Binomial(8, 0.43) (P0 = 0.57^8 = .011); hit 0.66 -> miss 0.34, 1 300-1 600 MB missed a token -> miss 0.34-0.41,
  * the cost of a cut expert: a hand-off of the gate+up result (2 048 floats) to the card that holds `down`, ~30 us (cross-card event + P2P), put on the layer's critical path (conservative) or 0 (optimistic).
Sizes (IQ3_S gate/up 2048x4096 at 3.4375 bpw, IQ4_XS down 4096x2048 at 4.25 bpw): gate+up 7.205 MB, down 4.463 MB, slab 11.668 MB.
Layer fetch time = max over the three cards of bytes / rate; rate of a pair card = 17.2 GB/s if the partner also fetches, else 28. Whole slab = both pieces on one card; a CUT expert has them on two cards.
Enumerates, per n missed experts (0..8), every placement of the n gate+up pieces and n down pieces on (lone, A, B); cuts = n - sum_c min(gu_c, dn_c) (pair the pieces on a card as whole experts first).
"""
import itertools, sys

GU, DN = 7.2054, 4.4630            # MB
SLAB = GU + DN
LONE, SH_BOTH, SH_ONE = 28.0, 17.2, 28.0   # GB/s
LAYERS = 42

def binom(n, k, p):
    from math import comb
    return comb(n, k) * p ** k * (1 - p) ** (n - k)

def card_us(mb, partner_active):   # MB / (GB/s) = ms/1000... MB/GBps = 1e-3 s = 1000 us
    return 0.0 if mb <= 0 else mb / (SH_BOTH if partner_active else SH_ONE) * 1000.0

def layer_us(gu, dn):
    """gu, dn = (lone, A, B) piece counts; returns fetch time of the layer in us (no hand-off)"""
    b = [gu[i] * GU + dn[i] * DN for i in range(3)]
    t_lone = b[0] / LONE * 1000.0
    t_a = card_us(b[1], b[2] > 0)
    t_b = card_us(b[2], b[1] > 0)
    return max(t_lone, t_a, t_b)

def comps(n):
    for a in range(n + 1):
        for b in range(n - a + 1):
            yield (a, b, n - a - b)

def best(n, whole_only, handoff_us):
    if n == 0:
        return 0.0, 0
    best_t, best_cuts = 1e18, 0
    for gu in comps(n):
        for dn in (comps(n) if not whole_only else [gu]):
            cuts = n - sum(min(gu[i], dn[i]) for i in range(3))
            t = layer_us(gu, dn) + (handoff_us if cuts > 0 else 0.0)
            if t < best_t - 1e-9 or (abs(t - best_t) < 1e-9 and cuts < best_cuts):
                best_t, best_cuts = t, cuts
    return best_t, best_cuts

def token_ms(p_miss, whole_only, handoff_us):
    tot, cuts = 0.0, 0.0
    for n in range(9):
        pr = binom(8, n, p_miss)
        t, c = best(n, whole_only, handoff_us)
        tot += pr * t
        cuts += pr * c
    return tot * LAYERS / 1000.0, cuts * LAYERS

def cont_ms(p_miss, agg_gbs=62.4):
    mean_n = 8 * p_miss
    return mean_n * SLAB / agg_gbs * LAYERS   # MB / (GB/s) = ms

PAT = [0, 1, 2, 0, 1, 0, 2, 0, 1, 2, 0]      # GLM_LINK_PAT: 0 = the lone card, 1 / 2 = the shared pair (GLM5.md "Optimal fetch assignment")

def pattern_layer_us(n):
    """the fixed pattern: n missed slabs take a window of the 11-cycle at (salt + j) % 11; mean over the 11 salts, whole slabs"""
    if n == 0:
        return 0.0
    tot = 0.0
    for salt in range(11):
        c = [0, 0, 0]
        for j in range(n):
            c[PAT[(salt + j) % 11]] += 1
        tot += layer_us(c, c)
    return tot / 11.0

def pattern_ms(p_miss):
    return sum(binom(8, n, p_miss) * pattern_layer_us(n) for n in range(9)) * LAYERS / 1000.0

if __name__ == "__main__":
    print("slab %.3f MB = gate+up %.3f + down %.3f ; links lone %.1f, shared pair %.1f each when both fetch / %.1f alone GB/s" % (SLAB, GU, DN, LONE, SH_BOTH, SH_ONE))
    print("sanity (old table, miss 0.43, whole slabs only): %.1f ms a token (GLM5.md says best whole-slab split 32.6 ms, pattern 35.7)" % token_ms(0.43, True, 0)[0])
    print()
    print("miss  MB/token | whole-slab ms | perfect-continuous@62.4 | cut: handoff 0us  30us  60us  100us (ms, cuts/token) | gain over whole slabs (ms): 0us 30us 60us 100us")
    for pm in (0.34, 0.37, 0.41, 0.43):
        w, _ = token_ms(pm, True, 0)
        cm = cont_ms(pm)
        row, gains = [], []
        for h in (0, 30, 60, 100):
            t, c = token_ms(pm, False, h)
            row.append("%.2f/%.1f" % (t, c)); gains.append(w - t)
        print("%.2f %7.0f | %6.2f | %6.2f | %s | %s" % (pm, 8 * pm * SLAB * LAYERS, w, cm, "  ".join(row), "  ".join("%.2f" % g for g in gains)))
    print()
    print("CALIBRATION of the realization factor at the miss rate of the traces: pattern model vs optimal model vs MEASURED (record: critical fetch 32.63 pattern -> 31.68 optimal = -0.97 ms)")
    print("  sanity: pattern at miss 0.43 = %.1f ms (GLM5.md: 35.7)" % pattern_ms(0.43))
    pm_tr = min((abs(pattern_ms(x / 1000.0) - 32.63), x / 1000.0) for x in range(250, 460))[1]
    mod_gain = pattern_ms(pm_tr) - token_ms(pm_tr, True, 0)[0]
    print("  the miss rate at which the pattern model gives the traced 32.63 ms: %.3f (%.0f MB a token)" % (pm_tr, 8 * pm_tr * SLAB * LAYERS))
    print("  modeled gain of the optimal whole-slab table there: %.2f ms; measured 0.97 ms -> realization factor %.2f" % (mod_gain, 0.97 / mod_gain))
    RF = 0.97 / mod_gain
    print()
    print("Reading: 'gain over whole slabs' is MODEL; the whole-slab optimum (--fetch-assign optimal) modeled -2.6..-3.0 ms and MEASURED -0.97 ms (a third, record §L5-GLM-FETCH-FLOOR-ERRATUM addendum).")
    print("A realization factor of 1/3 applied to the gain of the 30 us column is printed below; D9 goes only if that is >= 1.5 ms (plan section 5).")
    print()
    print("miss | modeled gain @30us | x calibrated realization %.2f | x 1/3 | x 1/2 | cut gain at 60 us x calibrated" % RF)
    for pm in (0.34, 0.37, 0.41, 0.43):
        w, _ = token_ms(pm, True, 0); t, _ = token_ms(pm, False, 30); t60, _ = token_ms(pm, False, 60)
        print("%.2f | %.2f ms | %.2f ms | %.2f ms | %.2f ms | %.2f ms" % (pm, w - t, (w - t) * RF, (w - t) / 3, (w - t) / 2, (w - t60) * RF))
    print()
    print("per-n detail at miss 0.37 (whole-slab us vs cut us at 30 us hand-off, cuts):")
    for n in range(1, 9):
        w, _ = best(n, True, 0); c, k = best(n, False, 30)
        print("  n=%d P=%.3f whole %5.0f us  cut %5.0f us (cuts %d)  saves %4.0f us a layer" % (n, binom(8, n, 0.37), w, c, k, w - c))
