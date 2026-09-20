#!/bin/bash
# f9a_serve.sh -- put the gated F9a binary (indexer head-lane SIMD, bit-identical, on by default) in service.
set -u
T=$HOME/src/colibri-f9a/tools/hot-expert
CAND=$HOME/bench/glm53.f9a; PRIS=$HOME/bench/glm53.f8
. "$T/rig_lock.sh"
rig_lock_take "f9a_serve" || exit 3
trap "rig_lock_release" EXIT INT TERM
echo "=== f9a_serve $(date -u -Is)"
cp -p $HOME/src/colibri-f9a/c/glm53 $CAND || exit 4
sha256sum $CAND $PRIS $HOME/src/colibri/c/glm53 | cut -c1-16,65-
for s in $HOME/src/colibri-f9a/c/shaders/*.spv; do cmp -s $s $HOME/src/colibri/c/shaders/$(basename $s) || { echo "shader differs: $s"; exit 6; }; done; echo "shaders identical"
"$T/serve_candidate.sh" $CAND $PRIS
rc=$?
echo "=== serve_candidate rc=$rc $(date -u -Is)"
exit $rc
