# gate_lib.sh -- source this in any A/B or oracle script.
#
# Two failure shapes cost real credibility on 2026-09-15 and neither was caught
# by reading the output; both are mechanical and both are refused here.
#
# 1. AN EMPTY COMPARISON READING AS A PASS.
#    `diff <(grep X a) <(grep X b)` with X absent from BOTH files compares two
#    empty streams and prints IDENTICAL. That is how a gate reported a passing
#    numerics oracle from two runs that had produced no output at all (the
#    engine had been invoked with the wrong CLI). CLAUDE.md already warns about
#    this shape; the warning did not stop it happening. gate_compare refuses.
#
# 2. AN ORDER-CONFOUNDED A/B.
#    Run arm A then arm B once each and the second arm inherits a warm page
#    cache. On this box that is worth more than most optimisations: an
#    uninterleaved pair read +20% and +11% on two changes whose real effects
#    were +3.9% and zero. gate_ab_verdict refuses a verdict when the arms
#    overlap, and refuses one outright from a single unpaired sample.

# gate_compare <label> <fileA> <fileB> <pattern>
#   Compares the lines matching <pattern> in both files. REFUSES (exit 2) if
#   the pattern is absent from either -- an absent oracle line is not a pass.
gate_compare() {
  local label="$1" a="$2" b="$3" pat="$4"
  local na nb
  na=$(grep -c -- "$pat" "$a" 2>/dev/null || true); na=${na:-0}
  nb=$(grep -c -- "$pat" "$b" 2>/dev/null || true); nb=${nb:-0}
  if [ "$na" -eq 0 ] || [ "$nb" -eq 0 ]; then
    printf '  %-34s REFUSED: pattern absent (A=%s B=%s) -- an empty comparison is NOT a pass\n' \
           "$label" "$na" "$nb"
    return 2
  fi
  if diff <(grep -- "$pat" "$a") <(grep -- "$pat" "$b") >/dev/null; then
    printf '  %-34s IDENTICAL (%s line(s) each)\n' "$label" "$na"; return 0
  fi
  printf '  %-34s DIFFERS\n' "$label"; return 1
}

# gate_ab_verdict <label> "<A samples>" "<B samples>"
#   Samples are whitespace-separated numbers, at least two per arm, and the
#   caller is expected to have produced them INTERLEAVED (A,B,B,A). Reports a
#   verdict only when the arms do not overlap.
gate_ab_verdict() {
  local label="$1" as="$2" bs="$3"
  python3 - "$label" "$as" "$bs" <<'PY'
import sys
label, a_s, b_s = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
a = [float(x) for x in a_s]; b = [float(x) for x in b_s]
if len(a) < 2 or len(b) < 2:
    print(f"  {label:<34} REFUSED: {len(a)} A and {len(b)} B sample(s) -- a single "
          f"pair cannot separate the change from cache warmth"); sys.exit(2)
amin, amax, bmin, bmax = min(a), max(a), min(b), max(b)
am, bm = sum(a)/len(a), sum(b)/len(b)
overlap = not (amin > bmax or bmin > amax)
delta = (bm - am) / am * 100.0
print(f"  {label:<34} A={am:.3f} [{amin:.3f}-{amax:.3f}]  B={bm:.3f} [{bmin:.3f}-{bmax:.3f}]")
if overlap:
    print(f"  {'':<34} OVERLAPPING ranges -- NO VERDICT. The means differ by "
          f"{delta:+.1f}% but the arms do not separate; report 'did not move'.")
    sys.exit(1)
# conservative: worst B against best A (or vice versa), never the means
cons = ((bmin - amax) / amax * 100.0) if bm > am else ((bmax - amin) / amin * 100.0)
print(f"  {'':<34} SEPARATED: {delta:+.1f}% on means, {cons:+.1f}% conservative "
      f"(worst-against-best) -- quote the conservative figure")
PY
}
