#!/bin/bash
# build_m4_span.sh -- three M4 variants for a quick screen of "does the host-source SPAN / allocation FLAGS lower the H2D rate":
#   m4_big40   40 GB pool a card, Portable (M4's flags)           -> span only
#   m4_big40m  40 GB pool a card, Mapped|Portable (engine flags)   -> span + engine flags
#   m4_small2m 2 GB pool a card, Mapped|Portable                   -> engine flags only
# Build only, in a temporary worktree, outputs ~/bench/m4span/. No GPU is touched.
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-m4span; OUT=$HOME/bench/m4span; mkdir -p "$OUT"
git -C "$SRC" worktree add --detach "$WT" main || exit 1
cd "$WT/m1" || exit 1
mk() {  # mk <name> <bytes> <flags>
  cp m4_stream.hip "m4_$1.hip"
  python3 - "$1" "$2" "$3" <<'PY'
import sys
n, b, f = sys.argv[1:4]
s = open("m4_%s.hip" % n).read()
a1 = "static const size_t HOST_POOL_BYTES = 2000000000ULL;"
assert a1 in s; s = s.replace(a1, "static const size_t HOST_POOL_BYTES = %sULL;" % b)
a2 = "hipHostMalloc(&r.h_pool, HOST_POOL_BYTES, hipHostMallocPortable)"
assert a2 in s; s = s.replace(a2, "hipHostMalloc(&r.h_pool, HOST_POOL_BYTES, %s)" % f)
open("m4_%s.hip" % n, "w").write(s)
PY
  nice -n 19 make -j2 build-m4 M4_SRC="m4_$1.hip" M4_BIN="m4_$1" 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -3
  [ -x "m4_$1" ] && cp -p "m4_$1" "$OUT/" && echo "built m4_$1 $(sha256sum "$OUT/m4_$1" | cut -c1-16)"
}
mk big40 40000000000 hipHostMallocPortable
mk big40m 40000000000 "hipHostMallocMapped | hipHostMallocPortable"
mk small2m 2000000000 "hipHostMallocMapped | hipHostMallocPortable"
cp -p "$SRC/m1/m4_stream" "$OUT/m4_stock" 2>/dev/null
git -C "$SRC" worktree remove --force "$WT"; echo done
