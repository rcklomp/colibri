#!/bin/bash
# f8_kl_report.sh [out_dir] -- F8's gv-vs-off KL bar, OUTSIDE the rig lock.
#
# Modelled on f7_kl_report.sh and there for the same reason: kl_compare.py
# is minutes of pure Python per comparison and needs no engine, no GPU and
# no lock. Run it after f8_gate_chain.sh's phase 1 has produced the dumps,
# with the gateway back up. `nice` so it never competes with a served
# request.
#
# The bar (record §F7-VERDICT, the jitter-arm scale this item's own gv
# kernel is judged against -- see f8_gate_chain.sh's header comment): mean
# KL <= 0.003 on the shallow packet, gv vs off, same binary. `REFUSED` from
# kl_compare.py is a failure, never a pass.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-~/bench/f8_out}
OUT=$(eval echo "$OUT")
KL="$HERE/kl_compare.py"

echo "=== f8_kl_report $(date -Is)  out=$OUT"
rc=0
a="$OUT/shallow_dump_off.f32"; b="$OUT/shallow_dump_gv.f32"
if [ ! -s "$a" ] || [ ! -s "$b" ]; then
  echo "  shallow: REFUSED -- missing dump ($a / $b)"; rc=1
else
  echo "--- shallow: $(stat -c %s "$a") vs $(stat -c %s "$b") bytes"
  nice -n 19 python3 "$KL" "gv vs off (shallow)" "$a" "$b" | tee "$OUT/kl_shallow.txt"
  [ "${PIPESTATUS[0]}" = 2 ] && rc=1
fi

python3 - "$OUT" <<'PY'
import re, sys, os
out = sys.argv[1]
bar_kl = 0.003
path = os.path.join(out, "kl_shallow.txt")
if not os.path.exists(path):
    print("  shallow    NO FILE"); sys.exit(1)
txt = open(path).read()
if "REFUSED" in txt:
    print("  shallow    REFUSED -- an empty comparison is not a pass"); sys.exit(1)
m = re.search(r"mean KL[^0-9\-]*([0-9.eE+\-]+)", txt)
t = re.search(r"top-1[^0-9]*([0-9.]+)\s*%", txt)
if not m or not t:
    print("  shallow    could not read mean KL / top-1 -- REFUSING"); sys.exit(1)
kl, t1 = float(m.group(1)), float(t.group(1))
ok = kl <= bar_kl
print("=== F8 gv-vs-off KL bar: mean KL <= %s (record F7-VERDICT jitter-arm scale)" % bar_kl)
print("  shallow    mean KL=%.6g  top-1=%.3f%%  %s" % (kl, t1, "MET" if ok else "NOT MET"))
sys.exit(0 if ok else 1)
PY
bar=$?
echo "=== f8_kl_report exit bar=$bar read_rc=$rc"
[ "$rc" = 0 ] || exit "$rc"
exit "$bar"
