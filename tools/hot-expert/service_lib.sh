#!/bin/bash
# service_lib.sh -- which engine serves :8081, whether it is alive, and how to bring it back.
# SOURCED (never run) by gateway_watchdog.sh and run_chain.sh, so the two cannot disagree (2026-10-06: run_chain.sh
# tested liveness with `ps -C glm53` only and restarted the OLD Colibri engine after every chain, which would have
# killed a healthy Franken service; the watchdog had the same blind spot).
#
# The service engine is named by ONE word in ~/bench/service_engine:
#   franken-glm   (default when the file is absent) the Franken engine, through `serve_alt.sh franken-glm` -- the same safe
#                 sequence a person uses (lock, VRAM check, warm, start, one real chat, acceptance, keeper lock), run with
#                 SERVE_ALT_NO_OLD_FALLBACK=1: if the start fails, everything is stopped and :8081 stays EMPTY rather than
#                 silently becoming the old, ~3x slower Colibri engine.
#   colibri-glm   the OLD Colibri engine, `~/start_glm53.sh`.
# "Engine alive" means EITHER engine's process is alive: callers restore availability, they do not enforce which engine a
# person chose to run.
#
# The caller defines `svc_say <text>` (default: echo) and sets KEY (the API key) before calling service_restore.
# Hooks for tests (never set in production): SERVICE_ENGINE_FILE, SERVICE_SERVE_ALT, SERVICE_FRANKEN_LOG,
# SERVICE_WAIT_STEPS (polls while a Franken start runs, default 180), SERVICE_POLL (seconds per poll, default 15).

SERVICE_ENGINE_FILE="${SERVICE_ENGINE_FILE:-$HOME/bench/service_engine}"
SERVICE_SERVE_ALT="${SERVICE_SERVE_ALT:-$HOME/src/colibri/tools/hot-expert/serve_alt.sh}"
SERVICE_FRANKEN_LOG="${SERVICE_FRANKEN_LOG:-$HOME/bench/service_restore_franken_glm.log}"
SERVICE_WAIT_STEPS="${SERVICE_WAIT_STEPS:-180}"
SERVICE_POLL="${SERVICE_POLL:-15}"
SERVICE_ENGINE_NAME=franken-glm
type svc_say >/dev/null 2>&1 || svc_say() { echo "$*"; }

# sets SERVICE_ENGINE_NAME (a global, not a command substitution: svc_say may write to stdout)
service_engine_select() {
  local w; w=$(tr -d ' \t\r\n' < "$SERVICE_ENGINE_FILE" 2>/dev/null); w=${w:-franken-glm}
  case "$w" in
    franken-glm|colibri-glm) SERVICE_ENGINE_NAME=$w ;;
    *) svc_say "unknown engine '$w' in $SERVICE_ENGINE_FILE -- using franken-glm"; SERVICE_ENGINE_NAME=franken-glm ;;
  esac
}

service_models_code() { curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer ${KEY:-}" http://127.0.0.1:8081/v1/models 2>/dev/null; }

# A killed engine the server has not reaped is a ZOMBIE and `pgrep -x glm53` still matches it
# (2026-09-20: 13 minutes of 500s with that check passing). Count only non-Z processes.
colibri_engine_alive() { ps -C glm53 -o stat= 2>/dev/null | grep -qv "^Z"; }
# The Franken GLM engine runs inside docker (the host comm is not its name): the bracket pgrep -f form, as serve_alt.sh's
# franken_glm_alive does; again only a non-zombie counts.
franken_glm_alive() { local p; p=$(pgrep -f "franken_dec_[g]lm" 2>/dev/null | head -1); [ -n "$p" ] && ps -p "$p" -o stat= 2>/dev/null | grep -qv "^Z"; }
engine_alive() { colibri_engine_alive || franken_glm_alive; }

service_restore_colibri() {
  local i code
  svc_say "restarting the OLD Colibri engine (service_engine=colibri-glm)"
  pkill -f "openai_[s]erver.py" 2>/dev/null; sleep 3
  pkill -9 -x glm53 2>/dev/null
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
  for i in $(seq 1 60); do
    sleep 10
    code=$(service_models_code)
    [ "$code" = 200 ] && { svc_say "gateway back after $((i*10))s"; return 0; }
  done
  svc_say "gateway did NOT come back within 10 min — needs a person"
  return 1
}

service_restore_franken() {
  local t0 pid rc code i
  t0=$(date +%s)
  svc_say "restarting the Franken engine through serve_alt.sh franken-glm (log $SERVICE_FRANKEN_LOG)"
  SERVE_ALT_NO_OLD_FALLBACK=1 setsid nohup "$SERVICE_SERVE_ALT" franken-glm > "$SERVICE_FRANKEN_LOG" 2>&1 < /dev/null &
  pid=$!
  for i in $(seq 1 "$SERVICE_WAIT_STEPS"); do kill -0 "$pid" 2>/dev/null || break; sleep "$SERVICE_POLL"; done
  if kill -0 "$pid" 2>/dev/null; then
    svc_say "Franken start still running after $(( $(date +%s) - t0 ))s — leaving it to finish (it holds the rig lock)"
    return 1
  fi
  wait "$pid" 2>/dev/null; rc=$?
  code=$(service_models_code)
  if [ "$code" = 200 ] && franken_glm_alive; then
    svc_say "gateway back (Franken engine) after $(( $(date +%s) - t0 ))s"
    return 0
  fi
  svc_say "Franken restart FAILED (serve_alt rc=$rc, /v1/models=$code) after $(( $(date +%s) - t0 ))s — the old engine was NOT started; :8081 is empty; see $SERVICE_FRANKEN_LOG"
  return 1
}

# restore the service engine; 0 = the gateway answers again. The caller must NOT hold the rig lock (serve_alt takes it).
service_restore() {
  service_engine_select
  case "$SERVICE_ENGINE_NAME" in
    colibri-glm) service_restore_colibri ;;
    *)           service_restore_franken ;;
  esac
}
