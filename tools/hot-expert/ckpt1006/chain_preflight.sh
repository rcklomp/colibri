#!/bin/bash
# chain_preflight.sh -- SOURCE this at the top of a rig chain. One copy of the checks that cost this session three timing processes, an idle engine for 45 minutes and a failed arm.
#   rig_quiet_wait [max_seconds=900]  waits until: every card holds < 1024 MiB (the previous engine has finished unpinning 144 GB, which takes minutes in D state), no `franken_engine`
#                                     container exists (a second `docker run --name franken_engine` fails with "name already in use"), nothing is compiling on the host (a build competes for the
#                                     engine's issue thread). Prints the blocker and returns 1 on timeout. NEVER refuse at once: waiting is the correct response.
#   rig_stop_serving                  stops the gateway, the engine and the container and waits until the container is gone.
#   tool_refuses <script> <pattern>   greps the tool's own refusal lines so the caller reads what it refuses BEFORE launching (accept_ui.sh refuses while a process named run_chain.sh is alive and
#                                     takes the gateway log from `serve_alt.sh status`; accept_live.sh skips under ~/bench/.dev_reserved and reads GLM53_LOG).
rig_vram_max() { local m=0 u d; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
rig_quiet_wait() {
  local max=${1:-900} t=0 why=""
  while [ $t -lt $max ]; do
    why=""
    [ "$(rig_vram_max)" -ge 1024 ] && why="a card holds $(rig_vram_max) MiB"
    [ -n "$(docker ps -aq -f name='^franken_engine$' 2>/dev/null)" ] && why="$why; the franken_engine container still exists"
    pgrep -f "[m]ake .*glm-serve|[m]ake .*gpu|[h]ipcc|[c]c1plus|[c]lang.*offload" > /dev/null && why="$why; a compile is running"
    [ -z "$why" ] && return 0
    sleep 5; t=$((t+5))
  done
  echo "rig not quiet after ${max}s:$why"; return 1
}
rig_stop_serving() {
  pkill -f "openai_[s]erver.py" 2>/dev/null; sleep 2; pkill -9 -f "openai_[s]erver.py" 2>/dev/null
  pkill -9 -f "franken_dec_[g]lm" 2>/dev/null
  docker stop -t 5 franken_engine >/dev/null 2>&1 || true
  for _ in $(seq 1 120); do [ -z "$(docker ps -aq -f name='^franken_engine$' 2>/dev/null)" ] && break; sleep 5; done
}
tool_refuses() { grep -nE "REFUSED|refus|exit [1-9]|dev_reserved|run_\[c\]hain|GLM53_LOG|serve_alt" "$1" | grep -E "${2:-.}" | cut -c1-200; }
