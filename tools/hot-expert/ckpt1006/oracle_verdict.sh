#!/bin/bash
# oracle_verdict.sh GATE_RUN.log -- per config of a --gate-plan log: taps compared, taps NOT bit-exact (maxabs != 0), the oracle's own summary line.
awk '/^=== gate-plan config/ { c = $4 } /^oracle [^ ]+ cos=/ { n[c]++; split($4, a, "="); if (a[2] != "0") bad[c]++ } /^oracle summary/ { s[c] = $0 }
     END { for (c in s) printf "%-12s taps=%d not_exact=%d | %s\n", c, n[c], bad[c] + 0, s[c] }' "$1" | cut -c1-230
