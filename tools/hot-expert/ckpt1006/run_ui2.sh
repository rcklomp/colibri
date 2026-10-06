#!/bin/bash
# run_ui2.sh -- the Mac side of the acceptance: accept_ui.sh (real Chromium through Open WebUI), then release the rig chain. accept_ui.sh's own rig-side step reads a stale log for a gateway that
# serve_alt did not start and can wait up to an hour: kill it the moment it starts (glm_accept_chain.sh already ran accept_live with the right log) and touch the chain's done-marker.
cd /Users/ronald/Projects/colibri
OUT=${1:-/tmp/accept_ui_run.out}
tools/hot-expert/accept_ui.sh > "$OUT" 2>&1 &
UI=$!
until grep -q "rig-side acceptance" "$OUT" 2>/dev/null || ! kill -0 $UI 2>/dev/null; do sleep 3; done
ssh rome 'pkill -f "owui_ui_turn[.]sh"; pkill -f "tools/hot-expert/accept_live[.]sh"; touch ~/bench/.accept_ui_done'
wait $UI 2>/dev/null
echo "ACCEPT_UI_DONE (rig-side phase killed on purpose)" >> "$OUT"
