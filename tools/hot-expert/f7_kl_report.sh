#!/bin/bash
# f7_kl_report.sh [out_dir] -- F7's O2 KL bar, OUTSIDE the rig lock.
#
# Modelled on f2_kl_report.sh and there for the same reason: kl_compare.py is
# minutes to an hour of pure Python per comparison (a 9 115-position GLM-5.3
# dump is 1.4 G float32 a side) and it needs no engine, no GPU and no lock. Run
# it after f7_gate_chain.sh's phase 1 has produced the dumps, with the gateway
# back up. `nice` so it never competes with a served request.
#
# The bar (F7 design note §5, from X2): mean KL < 0.0284 AND top-1 >= 99.0 %,
# knob-on against knob-off, same binary, on BOTH packets. Exit 0 only if both
# are MET; `REFUSED` from kl_compare.py is a failure, never a pass.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-~/bench/f7_out}
OUT=$(eval echo "$OUT")
KL="$HERE/kl_compare.py"

echo "=== f7_kl_report $(date -Is)  out=$OUT"
rc=0
for pkt in shallow deep; do          # shallow first: it is the cheap one
  a="$OUT/${pkt}_dump_off.f32"; b="$OUT/${pkt}_dump_gpu.f32"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    echo "  $pkt: REFUSED -- missing dump ($a / $b)"; rc=1; continue
  fi
  echo "--- $pkt: $(stat -c %s "$a") vs $(stat -c %s "$b") bytes"
  nice -n 19 python3 "$KL" "O2 $pkt (off || gpu)" "$a" "$b" | tee "$OUT/kl_${pkt}.txt"
  [ "${PIPESTATUS[0]}" = 2 ] && rc=1
done

python3 - "$OUT" <<'PY'
import re, sys, os
out = sys.argv[1]
bar_kl, bar_t1 = 0.0284, 99.0
bad = 0
print("=== F7 O2 KL bar: mean KL < %s AND top-1 >= %s %%" % (bar_kl, bar_t1))
for pkt in ("shallow", "deep"):
    path = os.path.join(out, "kl_%s.txt" % pkt)
    if not os.path.exists(path):
        print("  %-10s NO FILE" % pkt); bad = 1; continue
    txt = open(path).read()
    if "REFUSED" in txt:
        print("  %-10s REFUSED -- an empty comparison is not a pass" % pkt); bad = 1; continue
    m = re.search(r"mean KL[^0-9\-]*([0-9.eE+\-]+)", txt)
    t = re.search(r"top-1[^0-9]*([0-9.]+)\s*%", txt)
    if not m or not t:
        print("  %-10s could not read mean KL / top-1 -- REFUSING" % pkt); bad = 1; continue
    kl, t1 = float(m.group(1)), float(t.group(1))
    ok = kl < bar_kl and t1 >= bar_t1
    print("  %-10s mean KL=%.6g  top-1=%.3f%%  %s" % (pkt, kl, t1, "MET" if ok else "NOT MET"))
    if not ok: bad = 1
sys.exit(bad)
PY
bar=$?
echo "=== f7_kl_report exit bar=$bar read_rc=$rc"
[ "$rc" = 0 ] || exit "$rc"
exit "$bar"
