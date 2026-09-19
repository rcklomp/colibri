#!/bin/bash
# f2_kl_report.sh -- the KL half of F2's oracle, on dumps that already exist.
#
# X2's KL bar is part of F2's gate (FRANKEN-ENGINE-PLAN-2026-09-15.md §8.3, F2
# row): mean KL < 0.0284 AND top-1 agreement >= 99.0 %, the numbers the record
# states for V1. `kl_compare.py` computes them and REFUSES an empty or
# mismatched comparison; it does not apply the bar, so this does.
#
# It runs NO engine and takes NO rig lock, deliberately. A KL pass is a few
# minutes of pure Python per comparison over a 6 327 x 154 880 float32 dump,
# and holding the gateway down for that is a waste of the owner's day --
# f2_gate_chain.sh's phase 1c produces the dumps under the lock and stops, and
# this reads them afterwards with the gateway back up.
#
#   ~/src/colibri-f2/tools/hot-expert/f2_kl_report.sh [out_dir]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-~/bench/f2_out}
KL="$HERE/kl_compare.py"
BAR_KL=0.0284
BAR_TOP1=99.0

echo "=== f2_kl_report $(date -Is)  dumps in $OUT"
echo "=== bar: mean KL < $BAR_KL AND top-1 >= $BAR_TOP1 %"
[ -x "$KL" ] || [ -f "$KL" ] || { echo "FATAL: $KL missing"; exit 2; }

fails=0
# kl_line <label> <ref.f32> <cand.f32> <gated:0|1>
kl_line() {
  local label=$1 ref=$2 cand=$3 gated=$4 tmp rc
  tmp=$(mktemp)
  python3 "$KL" "$label" "$ref" "$cand" > "$tmp" 2>&1; rc=$?
  cat "$tmp"
  if [ "$rc" -ne 0 ]; then
    echo "     VERDICT             : REFUSED (kl_compare exit $rc) -- not a pass"
    fails=$((fails + 1)); rm -f "$tmp"; return 2
  fi
  python3 - "$tmp" "$BAR_KL" "$BAR_TOP1" "$gated" <<'PY'
import sys, re
txt = open(sys.argv[1]).read()
bar_kl, bar_t1, gated = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1"
m_kl = re.search(r"mean KL\(ref\|\|cand\)\s*:\s*([0-9eE.+-]+)", txt)
m_t1 = re.search(r"top-1 agreement\s*:\s*([0-9.]+)%", txt)
if not m_kl or not m_t1:
    print("     VERDICT             : REFUSED -- kl_compare printed no mean KL / top-1")
    sys.exit(2)
kl, t1 = float(m_kl.group(1)), float(m_t1.group(1))
ok = kl < bar_kl and t1 >= bar_t1
if not gated:
    print("     VERDICT             : REPORTED (not gated)  mean KL %.6g, top-1 %.2f%%" % (kl, t1))
    sys.exit(0)
print("     VERDICT             : %s  mean KL %.6g %s %.4g, top-1 %.2f%% %s %.1f%%"
      % ("MEETS THE BAR" if ok else "FAILS THE BAR",
         kl, "<" if kl < bar_kl else ">=", bar_kl,
         t1, ">=" if t1 >= bar_t1 else "<", bar_t1))
sys.exit(0 if ok else 1)
PY
  rc=$?
  [ "$rc" -eq 0 ] || fails=$((fails + 1))
  rm -f "$tmp"
  return $rc
}

echo
echo "--- C1: the clamp fix's own correctness line (clamp on, stream off) against"
echo "---     the CLAMPED all-CPU reference. Both sides clamp; only placement and"
echo "---     summation order are left. Shallow, because EXPERTS_CPU=1 is ~9x."
kl_line "C1 clamp vs EXPERTS_CPU=1 (shallow)" \
        "$OUT/shallow_dump_cpu1.f32" "$OUT/shallow_dump_clamp.f32" 1

echo
echo "--- C2: F2's KL GATE. Clamp on both sides; streaming at the DEFAULT chunk"
echo "---     against no streaming. This is the number that decides whether F2"
echo "---     can be served."
kl_line "C2 clamp+stream@128 vs clamp (deep)" \
        "$OUT/deep_dump_clamp.f32" "$OUT/deep_dump_clamp_s128.f32" 1

echo
echo "--- C3: the chunk, on top of clamp+streaming. Reported, not gated: the"
echo "---     chunk is an engine property that predates F2 (GLM53_PREFILL_CHUNK)."
kl_line "C3 clamp+stream@512 vs @128 (deep)" \
        "$OUT/deep_dump_clamp_s128.f32" "$OUT/deep_dump_clamp_s512.f32" 0

echo
if [ "${F2_KL_LEGACY:-0}" != 1 ]; then
  echo "--- the UNCLAMPED rows this replaces are already in the record; set"
  echo "---     F2_KL_LEGACY=1 to recompute them (a few minutes each)."
else
echo "--- for the record, the UNCLAMPED rows this replaces (recomputed only if"
echo "---     the dumps are still there; these are what failed the bar)"
for t in "O2 stream+chunk512 vs off (deep):deep_dump_off.f32:deep_dump_on.f32" \
         "O4 chunk512 alone vs off (deep):deep_dump_off.f32:deep_dump_chunk.f32" \
         "O5 stream@128 vs off (deep):deep_dump_off.f32:deep_dump_stream128.f32"; do
  lbl=${t%%:*}; rest=${t#*:}; r=${rest%%:*}; c=${rest#*:}
  [ -s "$OUT/$r" ] && [ -s "$OUT/$c" ] && kl_line "$lbl" "$OUT/$r" "$OUT/$c" 0 || \
    echo "  $lbl  SKIPPED (dump gone)"
done
fi

echo
echo "=== f2_kl_report: $fails gated line(s) failed $(date -Is)"
exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
