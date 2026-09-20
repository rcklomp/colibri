#!/bin/bash
# f8_serve.sh -- put the gated F7 binary (GPU prefill attention core) in service.
set -u
T=$HOME/src/colibri-f8e/tools/hot-expert
CAND=$HOME/bench/glm53.f8; PRIS=$HOME/bench/glm53.f7
. "$T/rig_lock.sh"
rig_lock_take "f8_serve" || exit 3
trap 'rig_lock_release' EXIT INT TERM
echo "=== f8_serve $(date -u -Is)"
cp -p $HOME/src/colibri-f8e/c/glm53 $CAND || exit 4
sha256sum $CAND $PRIS $HOME/src/colibri/c/glm53 | cut -c1-16,65-
S=$HOME/start_glm53.sh
cp -p $S $HOME/bench/start_glm53.sh.pre-f8
if ! grep -q GLM53_MOE_ONE_TEAM $S; then
  awk '/^export GLM53_MLA_ATTN_GPU=1/ { print; print "# F8 (2026-09-20): batch-1 CPU experts -- one OpenMP team per MoE layer (bit-identical) + the group-vector int4 row kernel (reassociation, relL2 1e-7)."; print "# Gate: decode +6.4..+9.6 % conservative at every depth to 19k, greedy text identical."; print "export GLM53_MOE_ONE_TEAM=1 GLM53_I4_FAST=2"; next } { print }' $S > $S.new && chmod +x $S.new && bash -n $S.new && mv -f $S.new $S || exit 5
fi
grep -n "MOE_ONE_TEAM" $S || exit 5
"$T/serve_candidate.sh" $CAND $PRIS
rc=$?
echo "=== serve_candidate rc=$rc $(date -u -Is)"
if [ $rc -ne 0 ]; then cp -p $HOME/bench/start_glm53.sh.pre-f8 $S.revert && mv -f $S.revert $S; echo "start script reverted"; fi
grep -a "\[MLA\]\|mla_attn\|MLA attn" $HOME/glm53_server.log | head -5
exit $rc
