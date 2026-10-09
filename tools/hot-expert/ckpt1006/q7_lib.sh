# Shared preamble for the Q7 chains. Sourced, not executed.
#
# The gateway rule, learned the hard way at 07:39 on 2026-09-12: a chain must
# restart the gateway ONLY if that same chain stopped it. The first version of
# q7_step0.sh restarted it on EVERY exit path, including the one where
# rig_lock_take had just REFUSED because a peer session held the lock -- so it
# started a second openai_server.py and a glm53 underneath another session s
# reserved rig. That is the 2026-09-10 incident in CLAUDE.md, reproduced.
# I_STOPPED_GATEWAY is the whole fix: nothing is restarted that this chain did
# not take down itself.
set -u
WT=${WT:-/home/ronald/q7wt}
OUT=${OUT:-~/bench/q7_out}; mkdir -p "$OUT"
GATEWAY_UP=0
I_STOPPED_GATEWAY=0
log(){ echo "[$(date -Is)] $*"; }
. /home/ronald/src/colibri/tools/hot-expert/rig_lock.sh
wait_no_engine(){ for _ in $(seq 1 240); do pgrep -x "$1" >/dev/null || return 0; sleep 1; done
  echo "REFUSED: $1 still running"; return 1; }
start_gateway(){
  log "restarting the gateway"
  SKIP_WARM=1 setsid nohup ~/start_glm53.sh > ~/glm53_server.log 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    pgrep -f "openai_[s]erver.py" >/dev/null && pgrep -x glm53 >/dev/null && break
    sleep 10
  done
  local svc eng code
  svc=$(pgrep -f "openai_[s]erver.py" | wc -l); eng=$(pgrep -x glm53 | wc -l)
  log "gateway processes: server=$svc engine=$eng"
  if [ "$svc" -ge 1 ] && [ "$eng" -ge 1 ]; then
    GATEWAY_UP=1
    code=$(curl -s -o /dev/null -m 30 -w "%{http_code}" -H "Authorization: Bearer $(cat ~/.colibri_api_key)" http://127.0.0.1:8081/v1/models) || true
    log "curl /v1/models -> ${code:-?}"
  else log "WARNING: gateway did not come up cleanly"; fi
}
stop_gateway(){
  for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; return 1; }; done
  if pgrep -x glm53 >/dev/null; then
    log "stopping the owner gateway (this chain holds the rig lock)"
    I_STOPPED_GATEWAY=1
    pkill -f "openai_[s]erver.py"; sleep 3; pkill -9 -x glm53
    wait_no_engine glm53 || { log "REFUSED: glm53 would not die"; return 1; }
  fi
  return 0
}
q7_on_exit(){ rc=$?; log "=== exiting rc=$rc"
  if [ "$I_STOPPED_GATEWAY" = 1 ] && [ "$GATEWAY_UP" = 0 ]; then start_gateway
  else log "gateway not touched by this chain -- leaving it alone"; fi
  rig_lock_release; log "=== done"; }
# Wait politely for a peer session instead of fighting it. Minutes, not forever.
q7_take_lock(){
  local name="$1" mins="${2:-90}" i=0
  while [ $i -lt $((mins*2)) ]; do
    rig_lock_take "$name" && return 0
    i=$((i+1)); sleep 30
  done
  log "REFUSED: rig lock still held after ${mins}m by $(rig_lock_holder)"; return 3
}
# The tracked c/shaders/qmatmul.spv and its .comp get the SAME mtime from a fresh
# `git worktree add`, so make keeps the PRISTINE shader and the run silently
# measures the wrong kernel (Q7 step 0, first attempt: every bf16 code came back
# as the int4 branch s -8). Always rebuild it from source.
q7_build_shader(){
  rm -f "$WT/c/shaders/qmatmul.spv"
  make -C "$WT/c" shaders/qmatmul.spv VK=1 || return 1
  sha256sum "$WT/c/shaders/qmatmul.spv"
}

# Rebuild the ENGINES, not just the shader. Q7 step 0 ran q7_probe.sh against a
# binary built two commits earlier and the denormal scan silently did not exist
# in it -- the chain had rebuilt only qmatmul.spv. A measurement chain rebuilds
# everything it is about to measure.
q7_build_engines(){
  q7_build_shader || return 1
  make -C "$WT/c" qwen38 qwen38-vk glm53 VK=1 -j8 > /tmp/q7_build.log 2>&1 || {
    log "engine build FAILED"; tail -20 /tmp/q7_build.log; return 1; }
  sha256sum "$WT/c/qwen38-vk" "$WT/c/glm53" "$WT/c/shaders/qmatmul.spv"
}
