#!/bin/bash
# serve_alt.sh -- let the owner judge an alternative model's ANSWERS in his normal Open WebUI,
# without touching webui.db, and hand the port straight back to GLM-5.3 when he is done.
#
# Franken plan Rev 41 (tools/hot-expert/FRANKEN-ENGINE-PLAN-2026-09-15.md), record
# §F11-STEP0/§F11-DEPTH (tools/hot-expert/ROME-3x7900XTX-2026-09-04.md): "Next: (1) the
# one-command swap so he can judge Qwen3.8 IQ4_XS in Open WebUI." This is that command, plus
# the DeepSeek-V4-Flash arm the same rev measured alongside it, plus the way back.
#
#   serve_alt.sh qwen38    -> GLM gateway out, llama-server + Qwen3.8-Flash-Next UD-IQ4_XS in
#   serve_alt.sh deepseek  -> GLM gateway out, llama-server + DeepSeek-V4-Flash-0731 UD-IQ2_M in
#   serve_alt.sh franken   -> GLM gateway out, Colibri's OWN gateway with the FRANKEN ENGINE in,
#                             on the same Qwen3.8 GGUF the `qwen38` arm serves (L0 step 4,
#                             2026-09-22). This one is not a container: it is openai_server.py
#                             with franken_decode as its child, so it is the arm that measures
#                             the new engine on the owner's own path.
#   serve_alt.sh franken-ds4 -> same shape as `franken`, on DeepSeek-V4-Flash-0731 UD-IQ2_M
#                             instead (franken_decode_ds4 as the gateway's child, M1). Not run
#                             or measured on a GPU by the session that wrote this: see
#                             tools/hot-expert/franken/decode/DEEPSEEK4.md and the M1 task body.
#   serve_alt.sh glm       -> alt server out, GLM gateway back, accept_live.sh must PASS
#   serve_alt.sh status    -> what is serving, since when, VRAM per card, lock holder
#
# HOW OPEN WEBUI REACHES WHICHEVER MODEL IS RUNNING (found read-only, 2026-09-21; see this
# branch's commit body for the exact queries):
#   - open-webui-new's OpenAI connection #1 is http://host.docker.internal:8081/v1, bearer key
#     = the content of ~/.colibri_api_key (verified: sha256 of that file's stripped content
#     matches openai.api_keys[1] in webui.db's config table; connection #0, :8080 with key
#     "none", is disabled). host.docker.internal resolves via an ExtraHosts entry
#     ("host.docker.internal:host-gateway") to the docker0 bridge IP (172.17.0.1) -- a port
#     PUBLISHED on the host (0.0.0.0, not 127.0.0.1) is reachable there from another container.
#   - the one configured model is id "glm-5.3-flash" (webui.db `model` table), matching
#     ~/start_glm53.sh's --model-id.
#   So: run llama-server on the SAME port (8081), with --api-key-file pointing at the SAME key
#   file, and --alias "glm-5.3-flash" -- the model shows up as the same entry Open WebUI already
#   has, whichever backend is actually behind it. webui.db is never opened for a write.
#   NB the model picker in the UI will still say "glm-5.3-flash" while an alt model answers --
#   this script prints plainly, every time, which model that name currently means.
#
# LAUNCH (requirement 1): exactly f11_ladder_chain.sh's start_f11 in "fit" mode -- same docker
# image, same build/bin dir, --fit on --fit-target 1024 --fit-ctx 262144 --ctx-size 262144
# --split-mode layer --device ROCm0,ROCm1,ROCm2 -fa on -ctk q8_0 -ctv q8_0 -t 16 -tb 8
# --parallel 1 -- NEVER -ngl/--tensor-split/--n-cpu-moe (fit aborts if any of those is set and
# an even layer split has left a card at 5.8/24 GB before, record §F11-STEP0). Differences from
# the benchmark chain, for real use:
#   * --jinja: NOT added. ~/src/llama-glm53/common/common.h:638 `bool use_jinja = true;` --
#     this build's default is already jinja-templated chat rendering (needed for tool calls);
#     the F11 chain's own flag list has no --jinja either, for the same reason.
#   * --reasoning-effort low: NOT added. openai_server.py (this box's own gateway) forces no
#     reasoning-effort override of its own for GLM (grep of openai_server.py: the only forced
#     default is COLI_THINK, which ~/start_glm53.sh does not set); the F11 chain's "low" was a
#     benchmark-speed choice, not something the served GLM setup mirrors. Leaving it unset keeps
#     each model's own chat-template default (Qwen3.8: xhigh unless the client asks otherwise;
#     DeepSeek-V4: template default), which is what the owner should be judging.
#   * a detached, named container (`docker run -d --rm --name serve_alt --restart no ...`), not
#     the chain's own backgrounded-foreground pattern -- the chain tears its container down at
#     the end of the arm, this one is meant to keep running after the script exits.
#   * logs to ~/bench/serve_alt_<model>.log (a `docker logs -f` tail, since -d does not stream
#     to this script's own stdout).
#
# THE RIG LOCK ACROSS A SCRIPT THAT EXITS WHILE THE SERVER KEEPS RUNNING (requirement 3).
# rig_lock.sh's contract: a directory holding an "owner" file "<name> <pid> <started>";
# `rig_lock_holder` calls it live iff `kill -0 <pid>` succeeds, and `rig_lock_take` binds pid to
# $$ of whatever process calls it. That is exactly right for a chain that runs start-to-finish
# under one pid, and exactly wrong for a service meant to outlive the script that started it --
# taken with $$ as usual, the lock would go stale the moment this script's own process exits,
# and gateway_watchdog.sh (which only skips its own work while `rig_lock_maintenance` is true)
# would treat the alt server as an ordinary "gateway down" and restart GLM underneath it.
# rig_lock.sh itself is NOT modified -- both of its rules (mkdir is the only way to create the
# lock; a holder is live iff kill -0 succeeds) are honoured as written. What this script adds on
# top, entirely in its own functions:
#   1. `serve_alt_lock_take <pid> <name>` acquires the lock the same way rig_lock_take does
#      (atomic mkdir; on EEXIST, clear it if `rig_lock_holder` finds it stale), but ALSO
#      recognises the one case a single-process design has no name for: the lock is already
#      held, live, by an EARLIER invocation of this same script (holder name starts with
#      "serve_alt"). That is not a foreign holder to refuse and not a stale one to clear --
#      it is self, taking over its own resource across a process boundary -- so it rewrites the
#      owner file rather than refusing. Anything else live and non-"serve_alt*" is refused,
#      unchanged from rig_lock_take.
#   2. Once the alt server is verified (health + one real chat), `rig_lock_rebind_pid` rewrites
#      the SAME owner file to a pid that lives exactly as long as the server does: a
#      backgrounded `docker wait serve_alt`, which blocks until the container exits and then
#      exits itself. So the lock's liveness becomes tied 1:1 to the thing it is protecting --
#      if the container is stopped by this script, by a crash, or by hand, the lock goes stale
#      on its own and the existing staleness rule in rig_lock_holder (not a new one) lets the
#      watchdog reclaim the box. This script still releases the lock explicitly on the way back
#      to GLM; the self-healing property is a backstop, not the primary path.
#   3. `serve_alt_lock_release` only removes the lock directory when the current holder's name
#      starts with "serve_alt" (rig_lock_release itself cannot be called this way -- it is
#      guarded by an in-process flag that does not survive across invocations). It refuses,
#      like rig_lock_release's own spirit, to ever clear a lock it does not recognise as its own.
# rig_lock.sh and gateway_watchdog.sh are UNCHANGED by this branch -- no separate diff was
# needed; see the branch commit body for why a watchdog-side marker file was considered and
# rejected (this mechanism satisfies requirement 3 within rig_lock.sh's existing rules).
#
# THE DAILY CANARY (requirement 4, `accept_live.sh --canary`, cron 05:00 UTC). Read in full:
# its very first check is `pgrep -f "openai_[s]erver.py" >/dev/null || { echo "REFUSED: gateway
# not running"; exit 2; }`, BEFORE it reads the ledger log, BEFORE any chat request, BEFORE any
# checkpoint touches the model. Serving an alt model always means the gateway process is fully
# stopped (this script's stop_gateway, same as every chain's), so the canary's first line is the
# only thing that runs: it prints REFUSED and exits 2 into ~/bench/accept_live.log, with no
# side effect on either server. That is already harmless and already distinguishable from a
# real FAIL in the log, so nothing further was added here -- adding a guard would be solving a
# problem that reading accept_live.sh shows does not exist.
#
# SAFETY ORDER for qwen38/deepseek (requirement 5): refuse if a benchmark engine or a foreign
# llama-server is running, or the rig lock is held by someone else; take the lock; stop_gateway
# the hardened way (server gone before the engine is killed, zombie-safe engine_alive, both
# copied verbatim from f11_ladder_chain.sh); assert VRAM free; warm the model two passes,
# verify >=90% resident with fincore (warm_and_verify, copied verbatim); start the container;
# wait for /health; send ONE real chat completion through the served port/key and print the
# reply and tok/s; on ANY failure from here on, fall back to GLM automatically, the same path
# `serve_alt.sh glm` uses.
#
# NOT RUN, NOT STARTED by writing this file: no engine, no docker container, no gateway stop --
# per the Franken track addendum, this script is launched by the orchestrating session, never
# by the agent that wrote it, not even with `status`.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/rig_lock.sh"
KEY_FILE="$HOME/.colibri_api_key"
GLOG="$HOME/glm53_server.log"
GLM_SNAP=${GLM_SNAP:-~/models/GLM-5.3-Flash-colibri-int4-g64}
STATE_FILE="$HOME/bench/serve_alt.state"
mkdir -p "$HOME/bench" 2>/dev/null || true

# --- models (byte-identical paths/constants to f11_ladder_chain.sh -- CLAUDE.md: "do not
# re-derive anything that is in it") -----------------------------------------------------------
Q_MODEL=/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
Q_SNAP_DIR=/home/ronald/models/Qwen3.8-Flash-Next/UD-IQ4_XS
Q_LABEL="Qwen3.8-Flash-Next UD-IQ4_XS"

D_MODEL=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
D_SNAP_DIR=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M
D_LABEL="DeepSeek-V4-Flash-0731 UD-IQ2_M"

# GLM-5.3-Flash on the SAME llama.cpp: 149 GB, ~half of it stays in host RAM under fit (2.8 tok/s
# on the 09-16 ladder). Not for daily use; it exists so the three models can be compared on one
# engine (2026-09-22).
G_MODEL=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
G_SNAP_DIR=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS
G_LABEL="GLM-5.3-Flash UD-IQ4_XS (llama.cpp)"

# --- the Franken engine (L0 step 4, 2026-09-22) ------------------------------------------------
# The one arm here that is NOT llama-server in a container: Colibri's own gateway
# (openai_server.py) with tools/hot-expert/franken/decode/franken_decode behind it, speaking the
# line protocol GATEWAY-PROTOCOL.md specs. Same port, same key, same model id as every other arm,
# so Open WebUI is unchanged; the chat template is Qwen3.8's, because the model is Qwen3.8's
# (start_franken.sh's header says why --arch qwen38 and --model-id glm-5.3-flash are not in
# conflict). It runs the SAME GGUF as the qwen38 arm, which is what makes the two comparable:
# one command apart, the only difference is which engine is behind the port.
F_START=$HOME/src/colibri/tools/hot-expert/franken/start_franken.sh
F_SNAP_DIR=$Q_SNAP_DIR
F_LABEL="Qwen3.8-Flash-Next UD-IQ4_XS (Franken engine)"
F_LOG=$HOME/bench/serve_alt_franken.log

# --- the Franken engine on DeepSeek-V4-Flash (M1: same shape as `franken` above) ------------
# The DS4 runner behind the same gateway, binary franken_decode_ds4 (a distinct process name:
# franken_decode_ds4 is 19 characters and a Linux comm field truncates at 15 -- `ps -C
# franken_decode_ds4` therefore matches NOTHING; every check below uses `pgrep -f
# "franken_decode_[d]s4"` (bracket self-match guard, CLAUDE.md) against the full argv instead,
# the same trick the gateway's own "openai_[s]erver.py" pattern uses and for the same reason).
D4_START=$HOME/src/colibri/tools/hot-expert/franken/start_franken_ds4.sh
D4_SNAP_DIR=$D_SNAP_DIR
D4_LABEL="DeepSeek-V4-Flash-0731 UD-IQ2_M (Franken engine)"
D4_LOG=$HOME/bench/serve_alt_franken_ds4.log

# --- the Franken engine on GLM-5.3-Flash (same shape as `franken-ds4` above) -------------------
# The GLM serve runner behind the same gateway, binary franken_dec_glm -- a SHORT process name,
# 15 characters, deliberately chosen (see that binary's own header comment,
# franken_dec_glm_main.cpp in frankenstack/franken-engine) so it does NOT truncate into the same
# 15-byte comm franken_decode_ds4 already collides with ("franken_decode_"); it still truncates
# to itself under ps -C, but every check below uses the same bracket pgrep -f form the D4_* arm
# uses, for the same reason (it works whether the engine runs natively or inside the
# franken_decode_glm_docker.sh wrapper, where the host-visible process is `docker run`, not the
# binary -- the wrapper's argv still carries the binary's path, which pgrep -f matches).
FG_START=$HOME/src/colibri/tools/hot-expert/franken/start_franken_glm.sh
FG_SNAP_DIR=$G_SNAP_DIR
FG_LABEL="GLM-5.3-Flash UD-IQ4_XS (Franken engine)"
FG_LOG=$HOME/bench/serve_alt_franken_glm.log

# --- backend: HIP docker, same bin dir/image the F11 chain used --------------------------------
ALT_BIN_DIR=/home/ronald/src/llama-glm53/build-hip/bin
ALT_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
ALT_CTX=262144
ALT_FIT_MARGIN_MIB=1024
ALT_PORT=8081                 # the gateway's own port -- Open WebUI already points here
ALT_ALIAS=glm-5.3-flash       # the gateway's own model id (webui.db `model` table) -- so
                               # Open WebUI needs NO change to reach whichever model is behind it
ALT_NAME=serve_alt            # fixed docker container name
ALT_READY_TIMEOUT=1200        # 20 min -- matches f11_ladder_chain.sh's READY_TIMEOUT

# ------------------------------------------------------------- primitives (verbatim ports) -----
# VRAM/engine_alive/wait_no_glm53/wait_no_container/warm_glm/assert_glm_resident/warm_and_verify/
# assert_vram_free/stop_gateway all copied from f11_ladder_chain.sh (CLAUDE.md: "do not
# re-derive anything that is in it" -- proven live on this box 2026-09-21).
VRAM() { cat "/sys/class/drm/card$1/device/mem_info_vram_used" 2>/dev/null || echo -1; }

engine_alive() { ps -C glm53 -o stat= 2>/dev/null | grep -qv '^Z'; }

# The same zombie-safe test for the Franken engine. `pgrep -x` would answer YES for a process
# that has been killed and not yet reaped, which is the bug that left five chains believing an
# engine was alive on 2026-09-20 (CLAUDE.md); `ps -C <name> -o stat=` plus the ^Z filter is the
# fix that went into run_chain.sh and gateway_watchdog.sh, reused verbatim.
franken_alive() { ps -C franken_decode -o stat= 2>/dev/null | grep -qv '^Z'; }

wait_no_franken() {   # bounded 240 s, 2 s steps
  for _ in $(seq 1 120); do franken_alive || return 0; sleep 2; done
  echo "FATAL: franken_decode still alive (non-zombie) after 240 s"; return 1
}

# franken_decode_ds4's comm truncates to "franken_decode_" (15 chars) at the kernel's
# TASK_COMM_LEN, which `ps -C franken_decode_ds4` cannot match (see the D4_* header comment
# above) -- pgrep -f matches the untruncated argv instead, with the bracket trick so this
# check (and the ssh command carrying it) never matches itself.
franken_ds4_alive() {
  local pid
  pid=$(pgrep -f "franken_decode_[d]s4" 2>/dev/null | head -1)
  [ -n "$pid" ] && ps -p "$pid" -o stat= 2>/dev/null | grep -qv '^Z'
}

wait_no_franken_ds4() {   # bounded 240 s, 2 s steps
  for _ in $(seq 1 120); do franken_ds4_alive || return 0; sleep 2; done
  echo "FATAL: franken_decode_ds4 still alive (non-zombie) after 240 s"; return 1
}

# franken_dec_glm: same pgrep -f shape as franken_ds4_alive, for the same reason (see the FG_*
# header comment above) -- it also covers the docker-wrapped case ds4's own comment names.
franken_glm_alive() {
  local pid
  pid=$(pgrep -f "franken_dec_[g]lm" 2>/dev/null | head -1)
  [ -n "$pid" ] && ps -p "$pid" -o stat= 2>/dev/null | grep -qv '^Z'
}

wait_no_franken_glm() {   # bounded 240 s, 2 s steps
  for _ in $(seq 1 120); do franken_glm_alive || return 0; sleep 2; done
  echo "FATAL: franken_dec_glm still alive (non-zombie) after 240 s"; return 1
}

wait_no_glm53() {   # bounded 240 s, 2 s steps; zombie-safe (a killed engine child is a ZOMBIE
                     # and still matches `pgrep -x glm53` -- CLAUDE.md, 2026-09-20)
  for _ in $(seq 1 120); do engine_alive || return 0; sleep 2; done
  echo "FATAL: glm53 still alive (non-zombie) after 240 s"; return 1
}

wait_no_container() {   # wait_no_container <name> -- bounded 60 s, 2 s steps
  local name=$1
  for _ in $(seq 1 30); do
    docker ps -aq -f "name=^/${name}\$" 2>/dev/null | grep -q . || return 0
    sleep 2
  done
  echo "FATAL: container $name still present after 60 s"
  return 1
}

assert_vram_free() {   # assert_vram_free <label> -- bounded 60 s, 2 s steps
  local label=$1 v c bad attempt
  for attempt in $(seq 1 30); do
    bad=0
    for c in 0 1 2; do
      v=$(VRAM "$c")
      [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ] && bad=1
    done
    [ "$bad" = 0 ] && { [ "$attempt" -gt 1 ] && echo "[$label] VRAM free after ${attempt} checks (~$(( (attempt-1) * 2 ))s)"; return 0; }
    sleep 2
  done
  for c in 0 1 2; do
    v=$(VRAM "$c")
    if [ "$v" -lt 0 ] || [ "$v" -ge 1073741824 ]; then
      echo "FATAL [$label]: card$c VRAM $v >= 1 GiB (or unreadable) after 60s of retries"
    fi
  done
  return 1
}

warm_glm() {
  echo "--- warming GLM shards (one model at a time; 182 GiB)"
  cat "$GLM_SNAP"/*.safetensors > /dev/null 2>&1 || true
}

assert_glm_resident() {   # assert_glm_resident <label>
  local label=$1 pct
  pct=$(fincore --bytes --output SIZE,RES "$GLM_SNAP"/*.safetensors 2>/dev/null \
        | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
  echo "[resid $label] resident=${pct}%"
  awk -v p="$pct" 'BEGIN{exit !(p>=90)}'
}

# warm_and_verify <dir> <label> -- two passes, fincore-verified >=90% each (record bug 19,
# 2026-09-16: one `cat` under a nearly-full page cache can leave a file ~85% resident).
warm_and_verify() {
  local dir=$1 label=$2 pass pct
  for pass in 1 2; do
    echo "--- warming $label ($dir) pass $pass $(date -Is)"
    cat "$dir"/*.gguf > /dev/null 2>&1 || true
    pct=$(fincore --bytes --output SIZE,RES "$dir"/*.gguf 2>/dev/null \
          | tail -n +2 | awk '{ts+=$1; rs+=$2} END{if (ts>0) printf "%.4f", 100*rs/ts; else print 0}')
    echo "[resid $label-p$pass] resident=${pct}%"
    awk -v p="$pct" 'BEGIN{exit !(p>=90)}' && return 0
  done
  echo "WARNING: $label not >=90% resident after 2 warm passes -- continuing (cold reads only cost time, not correctness)"
  return 1
}

stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  # The server must be GONE before the engine is killed: a server that survives its SIGTERM
  # with a dead engine answers /v1/models 200 and every chat 500.
  for _ in $(seq 1 10); do pgrep -f "openai_[s]erver.py" >/dev/null || break; sleep 1; done
  pkill -9 -f "openai_[s]erver.py" 2>/dev/null || true
  pkill -9 -x glm53 2>/dev/null || true
  # The gateway may have had the FRANKEN engine behind it rather than glm53. Killing only glm53
  # would leave franken_decode holding ~22 GB on each card, and the very next step
  # (assert_vram_free) would then refuse to start anything at all -- with nothing saying why.
  pkill -9 -x franken_decode 2>/dev/null || true
  # franken_decode_ds4: pkill -x cannot match it either (comm truncation, see franken_ds4_alive);
  # the bracket form is the self-match-safe pgrep -f equivalent of pkill -9 -x.
  pkill -9 -f "franken_decode_[d]s4" 2>/dev/null || true
  # franken_dec_glm: pkill -x cannot match it either when it runs inside the docker wrapper (the
  # host process is `docker run`, not the binary) -- the same bracket pgrep -f equivalent.
  pkill -9 -f "franken_dec_[g]lm" 2>/dev/null || true
  # The engine runs inside the ROCm 7.14 image (~/bench/franken_decode_docker.sh /
  # franken_decode_glm_docker.sh, container `franken_engine` either way -- one engine at a time,
  # CLAUDE.md): a host-side pkill by name did not reach it twice on 2026-09-23 and
  # the swap back found 24 GB still on the cards. Stop the container explicitly.
  docker stop -t 5 franken_engine >/dev/null 2>&1 || true
  wait_no_glm53 && wait_no_franken && wait_no_franken_ds4 && wait_no_franken_glm
}

start_gateway() {   # requirement 6: exactly how every chain restarts the gateway
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$GLOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$KEY_FILE")" \
         -w '%{http_code}' "http://127.0.0.1:${ALT_PORT}/v1/models" 2>/dev/null)" = 200 ] && engine_alive; then
      return 0
    fi
    sleep 5
  done
  echo "FATAL: gateway did not come up (server 200 + live engine) within 600 s"
  return 1
}

# ------------------------------------------------------- rig lock, extended for a service ------
# See the header for why rig_lock.sh's own rig_lock_take/rig_lock_release (pid = $$, in-process
# flag) do not fit a process that must keep the lock after it exits. rig_lock.sh is sourced
# unmodified; these three functions are additive, built only from its own primitives
# (rig_lock_holder, $RIG_LOCK_DIR) and its own on-disk format ("<name> <pid> <started>").
serve_alt_lock_take() {   # serve_alt_lock_take <pid> <name>
  local pid=$1 name=$2 holder
  for _ in 1 2 3; do
    if mkdir "$RIG_LOCK_DIR" 2>/dev/null; then
      echo "$name $pid $(date +%Y-%m-%dT%H:%M:%S%z)" > "$RIG_LOCK_DIR/owner"
      return 0
    fi
    holder=$(rig_lock_holder 2>/dev/null || true)
    if [ -z "$holder" ]; then
      echo "rig_lock: clearing a stale lock ($(cat "$RIG_LOCK_DIR/owner" 2>/dev/null))"
      rm -rf "$RIG_LOCK_DIR"; continue
    fi
    case "$holder" in
      serve_alt*)   # ours, from an earlier invocation of this script -- take it over
        echo "$name $pid $(date +%Y-%m-%dT%H:%M:%S%z)" > "$RIG_LOCK_DIR/owner"
        return 0 ;;
      *) break ;;
    esac
  done
  echo "rig_lock: REFUSED, held by $(rig_lock_holder 2>/dev/null || echo unknown)"
  return 3
}

rig_lock_rebind_pid() {   # rig_lock_rebind_pid <name> <pid> -- only call while WE are the
                           # current holder (checked by the caller before this runs)
  local name=$1 pid=$2
  echo "$name $pid $(date +%Y-%m-%dT%H:%M:%S%z)" > "$RIG_LOCK_DIR/owner"
}

serve_alt_lock_release() {
  local holder; holder=$(rig_lock_holder 2>/dev/null || true)
  case "$holder" in
    serve_alt*) rm -rf "$RIG_LOCK_DIR"; return 0 ;;
    "") return 0 ;;   # already gone
    *) echo "rig_lock: NOT releasing -- held by '$holder', not ours"; return 1 ;;
  esac
}

# ---------------------------------------------------------------------- alt-server lifecycle ---
ensure_alt_stopped() {
  # The Franken engine's container (started by ~/bench/franken_decode_docker.sh) is not
  # $ALT_NAME and a host pkill by name does not reach it: three swaps back on 2026-09-23
  # found 24 GB still on the cards. Stop it here, on every path that must empty the cards.
  if docker ps -q -f "name=^/franken_engine\$" 2>/dev/null | grep -q .; then
    echo "--- stopping the franken_engine container"
    docker stop -t 5 franken_engine >/dev/null 2>&1 || true
    wait_no_container franken_engine
  fi
  if docker ps -aq -f "name=^/${ALT_NAME}\$" 2>/dev/null | grep -q .; then
    echo "--- stopping existing '$ALT_NAME' container"
    docker stop -t 10 "$ALT_NAME" >/dev/null 2>&1 || true
    docker rm -f "$ALT_NAME" >/dev/null 2>&1 || true
    wait_no_container "$ALT_NAME"
  fi
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    . "$STATE_FILE" 2>/dev/null || true
    [ -n "${SERVE_ALT_KEEPER_PID:-}" ] && kill "${SERVE_ALT_KEEPER_PID}" 2>/dev/null || true
  fi
}

# refuse_if_busy <label> -- owner-safety, mirrors f11_ladder_chain.sh's own preflight checks
refuse_if_busy() {
  local label=$1 llama_procs foreign holder
  if pgrep -x qwen38 >/dev/null || pgrep -x qwen38-vk >/dev/null; then
    echo "REFUSED [$label]: a qwen38/qwen38-vk benchmark process is running -- not touching it"
    return 1
  fi
  llama_procs=$(pgrep -f "llama-[s]erver" 2>/dev/null || true)
  if [ -n "$llama_procs" ] && ! docker ps -q -f "name=^/${ALT_NAME}\$" 2>/dev/null | grep -q .; then
    echo "REFUSED [$label]: a llama-server is running outside this script's own '$ALT_NAME' container -- not ours, not touching it:"
    pgrep -af "llama-[s]erver"
    return 1
  fi
  foreign=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i llama | grep -vx "$ALT_NAME" || true)
  if [ -n "$foreign" ]; then
    echo "REFUSED [$label]: a foreign docker container with 'llama' in its name is running -- not ours:"
    echo "$foreign"
    return 1
  fi
  holder=$(rig_lock_holder 2>/dev/null || true)
  if [ -n "$holder" ]; then
    case "$holder" in
      serve_alt*) : ;;   # ours (possibly from an earlier invocation) -- fine
      *) echo "REFUSED [$label]: rig lock held by '$holder' -- not ours"; return 1 ;;
    esac
  fi
  return 0
}

# start_alt_container <model_path> <logname: qwen38|deepseek> <label> -- docker run -d, then a
# log tail into ~/bench/serve_alt_<logname>.log (requirement 1)
start_alt_container() {
  local model=$1 label=$3 logfile="$HOME/bench/serve_alt_${2}.log"
  [ -s "$KEY_FILE" ] || { echo "FATAL: $KEY_FILE missing or empty"; return 1; }
  echo "--- starting llama-server ($label) in container '$ALT_NAME' on port $ALT_PORT, log $logfile"
  docker run -d --rm --name "$ALT_NAME" --restart no \
    -p "${ALT_PORT}:${ALT_PORT}" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    -e "LD_LIBRARY_PATH=/opt/rocm/lib:${ALT_BIN_DIR}" \
    -v /home/ronald:/home/ronald \
    "$ALT_IMAGE" \
    "${ALT_BIN_DIR}/llama-server" \
    -m "$model" \
    --fit on --fit-target "$ALT_FIT_MARGIN_MIB" --fit-ctx "$ALT_CTX" \
    --split-mode layer --device ROCm0,ROCm1,ROCm2 \
    -fa on -ctk q8_0 -ctv q8_0 -t 16 -tb 8 \
    --host 0.0.0.0 --port "$ALT_PORT" \
    --parallel 1 --ctx-size "$ALT_CTX" \
    --alias "$ALT_ALIAS" \
    --api-key-file "$KEY_FILE" \
    > /dev/null 2>"$logfile" || return 1
  ( docker logs -f "$ALT_NAME" >> "$logfile" 2>&1 < /dev/null & ) 2>/dev/null
  return 0
}

wait_ready_alt() {   # wait_ready_alt <cap_s>
  local cap=$1 steps i running
  steps=$(( (cap + 4) / 5 ))
  for i in $(seq 1 "$steps"); do
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$KEY_FILE")" \
         -w '%{http_code}' "http://127.0.0.1:${ALT_PORT}/health" 2>/dev/null)" = 200 ]; then
      return 0
    fi
    running=$(docker inspect -f '{{.State.Running}}' "$ALT_NAME" 2>/dev/null || echo false)
    [ "$running" = true ] || { echo "FATAL: container $ALT_NAME exited before /health=200"; return 1; }
    sleep 5
  done
  echo "FATAL: /health never returned 200 after ${cap}s"
  return 1
}

send_test_chat() {   # send_test_chat <label> -- ONE real chat, prints reply + tok/s
  local label=$1 resp_file dt t0 t1
  resp_file=$(mktemp)
  t0=$(date +%s.%N)
  curl -s -m 300 -H "Authorization: Bearer $(cat "$KEY_FILE")" -H "Content-Type: application/json" \
    "http://127.0.0.1:${ALT_PORT}/v1/chat/completions" \
    -d "{\"model\":\"$ALT_ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"In one short sentence, name the model you are and answer: what is 12 plus 30?\"}],\"max_tokens\":96}" \
    > "$resp_file"
  t1=$(date +%s.%N)
  dt=$(python3 -c "print($t1 - $t0)")
  python3 - "$resp_file" "$dt" "$label" <<'PY'
import json, sys
path, dt, label = sys.argv[1], float(sys.argv[2]), sys.argv[3]
try:
    d = json.load(open(path))
    choice = d["choices"][0]["message"]["content"]
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens")
except Exception as e:
    print(f"FAIL [{label}]: could not parse a chat reply ({e}); raw response follows:")
    print(open(path).read()[:2000])
    sys.exit(1)
print(f"--- {label} reply:")
print(choice.strip())
if ct:
    print(f"--- completion_tokens={ct} wall_s={dt:.2f} tok/s={ct/dt:.2f}")
else:
    print(f"--- wall_s={dt:.2f} (no completion_tokens in usage)")
PY
  local rc=$?
  rm -f "$resp_file"
  return $rc
}

write_state() {   # write_state <model_key> <label>
  cat > "$STATE_FILE" <<EOF
SERVE_ALT_MODEL=$1
SERVE_ALT_LABEL="$2"
SERVE_ALT_STARTED=$(date -Is)
SERVE_ALT_KEEPER_PID=${SERVE_ALT_KEEPER_PID:-}
EOF
}

# restore_glm <reason> -- shared by `serve_alt.sh glm` and the automatic fallback on failure.
# Warms GLM (two passes), starts the gateway exactly as start_glm53.sh is started elsewhere,
# runs accept_live.sh, and releases the rig lock ONLY on a PASS (requirement 6).
restore_glm() {
  local reason=$1
  echo "=== restoring GLM-5.3 ($reason)"
  ensure_alt_stopped
  if [ -e "$HOME/bench/.dev_reserved" ]; then
    # the owner reserved the rig for development (2026-09-25): stop the alternative, free the
    # lock, and leave port 8081 empty instead of loading GLM for nobody. The alternative's
    # GATEWAY goes too: ensure_alt_stopped above stops the engine container only, and on
    # 2026-10-06 the Franken/GLM openai_server.py stayed up on 8081 with no engine behind it
    # (/v1/models answering, every chat a 500) until it was killed by hand.
    stop_gateway
    serve_alt_lock_release
    rm -f "$STATE_FILE"
    echo "=== rig reserved for development (~/bench/.dev_reserved): alternative stopped, GLM NOT restored, port $ALT_PORT empty"
    return 0
  fi
  echo "[1/4] confirming VRAM is free on all three cards"
  if ! assert_vram_free "pre-glm"; then
    echo "FATAL: VRAM stuck -- NOT starting GLM. Rig lock left HELD on purpose so the watchdog does not fight you; investigate by hand, then re-run 'serve_alt.sh glm'."
    return 1
  fi
  echo "[2/4] warming GLM shards (182 GiB, two passes)"
  warm_glm; warm_glm
  assert_glm_resident final || echo "WARNING: GLM <90% resident at restart time -- proceeding, first request pays the cold read"
  echo "[3/4] starting the gateway"
  stop_gateway   # idempotent: clears anything stale before we start fresh
  if ! start_gateway; then
    echo "FATAL: gateway did not come up. Rig lock left HELD; investigate, then re-run 'serve_alt.sh glm'."
    return 1
  fi
  echo "[4/4] accept_live.sh (the request after the restart is part of the measurement)"
  if "$HERE/accept_live.sh"; then
    echo "PASS: GLM gateway is back in service"
    serve_alt_lock_release
    rm -f "$STATE_FILE"
    echo "=== now serving: GLM-5.3 (model id 'glm-5.3-flash') on port $ALT_PORT"
    return 0
  else
    echo "FAIL: accept_live.sh did not pass -- GLM is up but NOT verified. Rig lock left HELD on purpose; investigate, then re-run 'serve_alt.sh glm'."
    return 1
  fi
}

# cmd_alt <model_key: Q|D> -- the qwen38/deepseek entry point (requirement 5's order)
cmd_alt() {
  local model_key=$1 model snap label logname
  case "$model_key" in
    Q) model=$Q_MODEL; snap=$Q_SNAP_DIR; label=$Q_LABEL; logname=qwen38 ;;
    D) model=$D_MODEL; snap=$D_SNAP_DIR; label=$D_LABEL; logname=deepseek ;;
    G) model=$G_MODEL; snap=$G_SNAP_DIR; label=$G_LABEL; logname=glm-llama ;;
  esac
  echo "=== serve_alt: switching to $label $(date -Is)"
  refuse_if_busy "$logname" || exit 1
  serve_alt_lock_take "$$" "serve_alt-setup-$logname" || exit 3
  ensure_alt_stopped
  echo "[1/7] stopping the GLM gateway"
  if ! stop_gateway; then restore_glm "stop_gateway failed"; exit 1; fi
  echo "[2/7] confirming VRAM is free on all three cards"
  if ! assert_vram_free "pre-$logname"; then restore_glm "VRAM not free before $label"; exit 1; fi
  echo "[3/7] warming $label in page cache (two passes, target >=90% resident)"
  warm_and_verify "$snap" "$logname" || true   # warning only, see the function's own comment
  echo "[4/7] starting llama-server ($label)"
  if ! start_alt_container "$model" "$logname" "$label"; then restore_glm "$label container failed to start"; exit 1; fi
  echo "[5/7] waiting for /health"
  if ! wait_ready_alt "$ALT_READY_TIMEOUT"; then restore_glm "$label did not become ready"; exit 1; fi
  for c in 0 1 2; do echo "  card$c VRAM used: $(VRAM "$c") bytes"; done
  echo "[6/7] sending one real chat completion to verify"
  if ! send_test_chat "$label"; then restore_glm "$label test chat failed"; exit 1; fi
  echo "[7/7] handing the rig lock to the running server (safe to exit this script now)"
  # A keeper tied 1:1 to the container's own lifetime (see the header's lock design): this
  # blocks until the container exits, then exits itself, so the lock's liveness IS the
  # server's liveness.
  nohup docker wait "$ALT_NAME" > /dev/null 2>&1 < /dev/null &   # must survive this script and its ssh session
  SERVE_ALT_KEEPER_PID=$!
  disown "$SERVE_ALT_KEEPER_PID" 2>/dev/null || true
  rig_lock_rebind_pid "serve_alt-$logname" "$SERVE_ALT_KEEPER_PID"
  write_state "$model_key" "$label"
  echo "=== now serving: $label on port $ALT_PORT (Open WebUI still shows it as model 'glm-5.3-flash')"
  echo "=== to go back: $HERE/serve_alt.sh glm"
}

# start_franken -- the gateway with the Franken engine behind it, waiting the way start_gateway
# waits for GLM's (server answering /v1/models 200 AND a live engine process, never one of the
# two: a server that outlives its engine answers 200 on /v1/models and 500 on every chat, which
# is exactly the failure five F3 chains produced on 2026-09-20).
start_franken() {
  [ -x "$F_START" ] || { echo "FATAL: $F_START is not executable"; return 1; }
  # FRANKEN_LOG is what start_franken.sh echoes back as `gateway log <path>`,
  # so the file and the thing that names it can never drift apart.
  FRANKEN_LOG="$F_LOG" SKIP_WARM=1 setsid nohup "$F_START" > "$F_LOG" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 240); do
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$KEY_FILE")" \
         -w '%{http_code}' "http://127.0.0.1:${ALT_PORT}/v1/models" 2>/dev/null)" = 200 ] \
       && franken_alive; then
      return 0
    fi
    # A boot that has already failed should not cost 20 minutes of polling.
    if ! pgrep -f "openai_[s]erver.py" >/dev/null && [ "$i" -gt 6 ]; then
      echo "FATAL: the gateway process is gone; last lines of $F_LOG:"; tail -20 "$F_LOG"; return 1
    fi
    sleep 5
  done
  echo "FATAL: the Franken gateway did not come up within 1200 s; last lines of $F_LOG:"
  tail -20 "$F_LOG"
  return 1
}

# cmd_franken -- same safety order as cmd_alt (requirement 5), with the container steps replaced
# by the gateway ones. The weights are the qwen38 arm's own GGUF, so the warm step is identical.
cmd_franken() {
  echo "=== serve_alt: switching to $F_LABEL $(date -Is)"
  refuse_if_busy franken || exit 1
  serve_alt_lock_take "$$" "serve_alt-setup-franken" || exit 3
  ensure_alt_stopped
  echo "[1/7] stopping whatever the gateway is serving now"
  if ! stop_gateway; then restore_glm "stop_gateway failed"; exit 1; fi
  echo "[2/7] confirming VRAM is free on all three cards"
  if ! assert_vram_free "pre-franken"; then restore_glm "VRAM not free before the Franken engine"; exit 1; fi
  echo "[3/7] warming the GGUF in page cache (two passes, target >=90% resident)"
  warm_and_verify "$F_SNAP_DIR" franken || true
  echo "[4/7] starting the gateway with the Franken engine ($F_LOG)"
  if ! start_franken; then restore_glm "the Franken gateway did not come up"; exit 1; fi
  for c in 0 1 2; do echo "  card$c VRAM used: $(VRAM "$c") bytes"; done
  echo "[5/7] sending one real chat completion to verify"
  if ! send_test_chat "$F_LABEL"; then restore_glm "the Franken test chat failed"; exit 1; fi
  echo "[6/7] accept_live.sh -- the owner's own path, including the request AFTER the one it tests"
  # GLM53_LOG IS NOT OPTIONAL HERE. accept_live.sh and owui_ui_turn.sh both read
  # "${GLM53_LOG:-$HOME/glm53_server.log}" -- GLM's log -- and every number they report comes out
  # of it: the [req] line they wait for, and the " REUSE <id> " line they take `reused` from. Run
  # against this gateway without it, check 1 polls a file nothing is writing to and waits its full
  # 60 minutes (18 of them spent on 2026-09-23 before it was killed). It is exported, not just set,
  # because accept_live.sh calls owui_ui_turn.sh as a child and that one reads it too.
  # Not fatal, and that is deliberate: a FAIL here is a result to read, not a reason to throw the
  # box back to GLM automatically -- the owner asked to judge this engine's answers.
  GLM53_LOG="$F_LOG" "$HERE/accept_live.sh" || echo "NOTE: accept_live.sh did not pass -- see its output above; the engine is still serving"
  echo "[7/7] handing the rig lock to the running engine (safe to exit this script now)"
  # The keeper's liveness IS the engine's liveness: tail --pid exits when that pid does, so a
  # crashed engine leaves a stale lock that rig_lock_holder's existing kill -0 rule clears.
  local epid keeper
  epid=$(pgrep -x franken_decode | head -1)
  nohup tail --pid="$epid" -f /dev/null > /dev/null 2>&1 < /dev/null &
  keeper=$!
  disown "$keeper" 2>/dev/null || true
  SERVE_ALT_KEEPER_PID=$keeper
  rig_lock_rebind_pid "serve_alt-franken" "$keeper"
  write_state F "$F_LABEL"
  echo "=== now serving: $F_LABEL on port $ALT_PORT (Open WebUI still shows it as 'glm-5.3-flash')"
  echo "=== engine pid $epid, gateway log $F_LOG"
  echo "=== to go back: $HERE/serve_alt.sh glm"
}

# start_franken_ds4 -- the gateway with the Franken engine on DeepSeek-V4-Flash behind it.
# Same readiness rule as start_franken (server 200 AND a live engine), against the DS4 process
# name (franken_ds4_alive, not franken_alive -- see the D4_* header comment).
start_franken_ds4() {
  [ -x "$D4_START" ] || { echo "FATAL: $D4_START is not executable"; return 1; }
  FRANKEN_LOG="$D4_LOG" SKIP_WARM=1 setsid nohup "$D4_START" > "$D4_LOG" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 240); do
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$KEY_FILE")" \
         -w '%{http_code}' "http://127.0.0.1:${ALT_PORT}/v1/models" 2>/dev/null)" = 200 ] \
       && franken_ds4_alive; then
      return 0
    fi
    if ! pgrep -f "openai_[s]erver.py" >/dev/null && [ "$i" -gt 6 ]; then
      echo "FATAL: the gateway process is gone; last lines of $D4_LOG:"; tail -20 "$D4_LOG"; return 1
    fi
    sleep 5
  done
  echo "FATAL: the Franken/DS4 gateway did not come up within 1200 s; last lines of $D4_LOG:"
  tail -20 "$D4_LOG"
  return 1
}

# cmd_franken_ds4 -- same shape and same safety order as cmd_franken (requirement 5), for the
# DeepSeek-V4-Flash arm. The weights are D_SNAP_DIR (the same GGUF `deepseek` serves via
# llama-server), so the two arms are comparable the same way `franken`/`qwen38` are.
cmd_franken_ds4() {
  echo "=== serve_alt: switching to $D4_LABEL $(date -Is)"
  refuse_if_busy franken-ds4 || exit 1
  serve_alt_lock_take "$$" "serve_alt-setup-franken-ds4" || exit 3
  ensure_alt_stopped
  echo "[1/7] stopping whatever the gateway is serving now"
  if ! stop_gateway; then restore_glm "stop_gateway failed"; exit 1; fi
  echo "[2/7] confirming VRAM is free on all three cards"
  if ! assert_vram_free "pre-franken-ds4"; then restore_glm "VRAM not free before the Franken/DS4 engine"; exit 1; fi
  echo "[3/7] warming the GGUF in page cache (two passes, target >=90% resident)"
  warm_and_verify "$D4_SNAP_DIR" franken-ds4 || true
  echo "[4/7] starting the gateway with the Franken/DS4 engine ($D4_LOG)"
  if ! start_franken_ds4; then restore_glm "the Franken/DS4 gateway did not come up"; exit 1; fi
  for c in 0 1 2; do echo "  card$c VRAM used: $(VRAM "$c") bytes"; done
  echo "[5/7] sending one real chat completion to verify"
  if ! send_test_chat "$D4_LABEL"; then restore_glm "the Franken/DS4 test chat failed"; exit 1; fi
  echo "[6/7] accept_live.sh -- the owner's own path, including the request AFTER the one it tests"
  GLM53_LOG="$D4_LOG" "$HERE/accept_live.sh" || echo "NOTE: accept_live.sh did not pass -- see its output above; the engine is still serving"
  echo "[7/7] handing the rig lock to the running engine (safe to exit this script now)"
  local epid keeper
  epid=$(pgrep -f "franken_decode_[d]s4" | head -1)
  nohup tail --pid="$epid" -f /dev/null > /dev/null 2>&1 < /dev/null &
  keeper=$!
  disown "$keeper" 2>/dev/null || true
  SERVE_ALT_KEEPER_PID=$keeper
  rig_lock_rebind_pid "serve_alt-franken-ds4" "$keeper"
  write_state D4 "$D4_LABEL"
  echo "=== now serving: $D4_LABEL on port $ALT_PORT (Open WebUI still shows it as 'glm-5.3-flash')"
  echo "=== engine pid $epid, gateway log $D4_LOG"
  echo "=== to go back: $HERE/serve_alt.sh glm"
}

# start_franken_glm -- the gateway with the Franken engine on GLM-5.3-Flash behind it. Same
# readiness rule as start_franken/start_franken_ds4 (server 200 AND a live engine), against the
# GLM serve process name (franken_glm_alive -- see the FG_* header comment).
start_franken_glm() {
  [ -x "$FG_START" ] || { echo "FATAL: $FG_START is not executable"; return 1; }
  FRANKEN_LOG="$FG_LOG" SKIP_WARM=1 setsid nohup "$FG_START" > "$FG_LOG" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 240); do
    if [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $(cat "$KEY_FILE")" \
         -w '%{http_code}' "http://127.0.0.1:${ALT_PORT}/v1/models" 2>/dev/null)" = 200 ] \
       && franken_glm_alive; then
      return 0
    fi
    if ! pgrep -f "openai_[s]erver.py" >/dev/null && [ "$i" -gt 6 ]; then
      echo "FATAL: the gateway process is gone; last lines of $FG_LOG:"; tail -20 "$FG_LOG"; return 1
    fi
    sleep 5
  done
  echo "FATAL: the Franken/GLM gateway did not come up within 1200 s; last lines of $FG_LOG:"
  tail -20 "$FG_LOG"
  return 1
}

# cmd_franken_glm -- same shape and same safety order as cmd_franken_ds4 (requirement 5), for
# the GLM-5.3-Flash arm. The weights are FG_SNAP_DIR ($G_SNAP_DIR, the same GGUF the `glm-llama`
# arm serves via llama-server), so this arm is comparable the same way `franken`/`qwen38` are.
cmd_franken_glm() {
  echo "=== serve_alt: switching to $FG_LABEL $(date -Is)"
  refuse_if_busy franken-glm || exit 1
  serve_alt_lock_take "$$" "serve_alt-setup-franken-glm" || exit 3
  ensure_alt_stopped
  echo "[1/7] stopping whatever the gateway is serving now"
  if ! stop_gateway; then restore_glm "stop_gateway failed"; exit 1; fi
  echo "[2/7] confirming VRAM is free on all three cards"
  if ! assert_vram_free "pre-franken-glm"; then restore_glm "VRAM not free before the Franken/GLM engine"; exit 1; fi
  echo "[3/7] warming the GGUF in page cache (two passes, target >=90% resident)"
  warm_and_verify "$FG_SNAP_DIR" franken-glm || true
  echo "[4/7] starting the gateway with the Franken/GLM engine ($FG_LOG)"
  if ! start_franken_glm; then restore_glm "the Franken/GLM gateway did not come up"; exit 1; fi
  for c in 0 1 2; do echo "  card$c VRAM used: $(VRAM "$c") bytes"; done
  echo "[5/7] sending one real chat completion to verify"
  if ! send_test_chat "$FG_LABEL"; then restore_glm "the Franken/GLM test chat failed"; exit 1; fi
  echo "[6/7] accept_live.sh -- the owner's own path, including the request AFTER the one it tests"
  GLM53_LOG="$FG_LOG" "$HERE/accept_live.sh" || echo "NOTE: accept_live.sh did not pass -- see its output above; the engine is still serving"
  echo "[7/7] handing the rig lock to the running engine (safe to exit this script now)"
  local epid keeper
  epid=$(pgrep -f "franken_dec_[g]lm" | head -1)
  nohup tail --pid="$epid" -f /dev/null > /dev/null 2>&1 < /dev/null &
  keeper=$!
  disown "$keeper" 2>/dev/null || true
  SERVE_ALT_KEEPER_PID=$keeper
  rig_lock_rebind_pid "serve_alt-franken-glm" "$keeper"
  write_state FG "$FG_LABEL"
  echo "=== now serving: $FG_LABEL on port $ALT_PORT (Open WebUI still shows it as 'glm-5.3-flash')"
  echo "=== engine pid $epid, gateway log $FG_LOG"
  echo "=== to go back: $HERE/serve_alt.sh glm"
}

cmd_glm() {
  echo "=== serve_alt: switching back to GLM-5.3 $(date -Is)"
  refuse_if_busy glm || exit 1
  # serve_alt_lock_take takes over cleanly whether the lock is already ours (from the alt
  # session, held by its keeper pid) or not held at all; refuse_if_busy above already refused
  # if it is held by anyone else.
  serve_alt_lock_take "$$" "serve_alt-torestore" || exit 3
  restore_glm "explicit 'glm' command"
  exit $?
}

cmd_status() {
  echo "=== serve_alt status $(date -Is)"
  local holder started
  holder=$(rig_lock_holder 2>/dev/null || echo "(none)")
  echo "rig lock holder: $holder"
  if docker ps -q -f "name=^/${ALT_NAME}\$" 2>/dev/null | grep -q .; then
    started=$(docker inspect -f '{{.State.StartedAt}}' "$ALT_NAME" 2>/dev/null)
    if [ -f "$STATE_FILE" ]; then
      # shellcheck disable=SC1090
      . "$STATE_FILE" 2>/dev/null || true
    fi
    echo "serving: ${SERVE_ALT_LABEL:-an alt model (label unknown, see $STATE_FILE)} on port $ALT_PORT (Open WebUI shows it as 'glm-5.3-flash')"
    echo "container '$ALT_NAME' started: $started"
  elif pgrep -f "openai_[s]erver.py" >/dev/null && engine_alive; then
    echo "serving: GLM-5.3 (model id 'glm-5.3-flash') via the Colibri gateway on port $ALT_PORT"
  elif pgrep -f "openai_[s]erver.py" >/dev/null && franken_alive; then
    echo "serving: $F_LABEL via the Colibri gateway on port $ALT_PORT (Open WebUI shows it as 'glm-5.3-flash')"
    # The launcher prints `gateway log <path>` into its own log; read it back rather than
    # assuming, so a run started with a different FRANKEN_LOG still reports the truth.
    local flog
    flog=$(grep -h "gateway log " "$F_LOG" 2>/dev/null | tail -1 | sed 's/.*gateway log //')
    echo "engine pid: $(pgrep -x franken_decode | head -1), gateway log ${flog:-$F_LOG}"
    echo "accept_live against it: GLM53_LOG=${flog:-$F_LOG} $HERE/accept_live.sh"
  elif pgrep -f "openai_[s]erver.py" >/dev/null && franken_ds4_alive; then
    echo "serving: $D4_LABEL via the Colibri gateway on port $ALT_PORT (Open WebUI shows it as 'glm-5.3-flash')"
    local d4log
    d4log=$(grep -h "gateway log " "$D4_LOG" 2>/dev/null | tail -1 | sed 's/.*gateway log //')
    echo "engine pid: $(pgrep -f "franken_decode_[d]s4" | head -1), gateway log ${d4log:-$D4_LOG}"
    echo "accept_live against it: GLM53_LOG=${d4log:-$D4_LOG} $HERE/accept_live.sh"
  elif pgrep -f "openai_[s]erver.py" >/dev/null && franken_glm_alive; then
    echo "serving: $FG_LABEL via the Colibri gateway on port $ALT_PORT (Open WebUI shows it as 'glm-5.3-flash')"
    local fglog
    fglog=$(grep -h "gateway log " "$FG_LOG" 2>/dev/null | tail -1 | sed 's/.*gateway log //')
    echo "engine pid: $(pgrep -f "franken_dec_[g]lm" | head -1), gateway log ${fglog:-$FG_LOG}"
    echo "accept_live against it: GLM53_LOG=${fglog:-$FG_LOG} $HERE/accept_live.sh"
  else
    echo "serving: NOTHING recognizable on port $ALT_PORT -- run '$HERE/serve_alt.sh glm' to restore GLM"
  fi
  for c in 0 1 2; do
    local v; v=$(VRAM "$c")
    if [ "$v" -ge 0 ]; then
      printf 'card%s VRAM used: %.2f GB\n' "$c" "$(python3 -c "print($v/1e9)")"
    else
      echo "card$c VRAM used: unknown"
    fi
  done
}

# --------------------------------------------------------------------------------- dispatch ----
case "${1:-}" in
  qwen38)   cmd_alt Q ;;
  deepseek) cmd_alt D ;;
  glm-llama) cmd_alt G ;;
  franken)  cmd_franken ;;
  franken-ds4) cmd_franken_ds4 ;;
  franken-glm) cmd_franken_glm ;;
  glm)      cmd_glm ;;
  status)   cmd_status ;;
  *)
    echo "usage: $0 {qwen38|deepseek|glm-llama|franken|franken-ds4|franken-glm|glm|status}"
    exit 2
    ;;
esac
