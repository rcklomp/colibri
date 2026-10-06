#!/bin/bash
# glm_ckpt_accept.sh -- run accept_live.sh against the Franken GLM gateway while the rig is
# development-reserved. accept_live SKIPS under ~/bench/.dev_reserved, so the flag is moved aside
# for the duration and ALWAYS put back (trap). The rig lock is held by serve_alt's keeper
# (serve_alt-franken-glm), so gateway_watchdog.sh cannot restart anything meanwhile.
F=$HOME/bench/.dev_reserved
restore() { [ -e "$F.accept-off" ] && mv "$F.accept-off" "$F"; echo "=== flag restored: $(ls -la "$F" 2>&1 | cut -c1-80) $(date -Is)"; }
trap restore EXIT INT TERM
echo "=== glm_ckpt_accept start $(date -Is) engine=$(ps -eo args | grep -m1 '[f]ranken_dec_glm' | awk '{print $NF}')"
mv "$F" "$F.accept-off" || { echo "no flag to move"; exit 2; }
GLM53_LOG=$HOME/bench/serve_alt_franken_glm.log "$HOME/src/colibri/tools/hot-expert/accept_live.sh"
rc=$?
echo "=== accept_live rc=$rc $(date -Is)"
exit $rc
