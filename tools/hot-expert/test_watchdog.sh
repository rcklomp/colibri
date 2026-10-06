#!/bin/bash
# test_watchdog.sh -- regression test for gateway_watchdog.sh, run_chain.sh's post-chain restore, service_lib.sh and serve_alt.sh's
# SERVE_ALT_NO_OLD_FALLBACK, run ON THE RIG.
# No GPU, no model, no engine: the real watchdog runs in a throw-away HOME against a fake serve_alt.sh, a fake ~/start_glm53.sh,
# fake gateway / engine processes and a fake :8081. It refuses to run if the rig is not idle (an openai_server.py, a glm53 or a
# franken_dec_glm process, or anything listening on 8081), so it can never touch the real service.
#   ~/src/colibri/tools/hot-expert/test_watchdog.sh        exit 0 = all pass
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if pgrep -f "openai_[s]erver.py|franken_dec_[g]lm" >/dev/null || pgrep -x glm53 >/dev/null || ss -ltn 2>/dev/null | grep -q ':8081 '; then
  echo "REFUSED: the rig is not idle (gateway / engine process or :8081 listener present)"; exit 2
fi
ROOT=$(mktemp -d /tmp/wdtest.XXXXXX); PASS=0; FAILN=0; PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done; rm -rf "$ROOT"; }
trap cleanup EXIT
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAILN=$((FAILN+1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1  [$2]"; fi; }

# A fresh fake HOME with the watchdog (as cron sees it, next to rig_lock.sh), a fake serve_alt and a fake old start script.
newhome() {
  T=$ROOT/$1; mkdir -p "$T/bench"; cp "$HERE/gateway_watchdog.sh" "$HERE/rig_lock.sh" "$HERE/service_lib.sh" "$HERE/run_chain.sh" "$T/bench/"; echo k > "$T/.colibri_api_key"
  # fake serve_alt.sh: records its arguments and the fallback switch; FAKE_SA_RC decides its exit status; FAKE_SA_UP=1 brings the fakes up
  cat > "$T/fake_serve_alt.sh" <<EOF
#!/bin/bash
echo "serve_alt \$* NO_OLD_FALLBACK=\${SERVE_ALT_NO_OLD_FALLBACK:-unset}" >> $T/calls
[ -e $T/bench/.rig.lock ] && echo LOCK_HELD_AT_CALL >> $T/calls
if [ "\${FAKE_SA_UP:-0}" = 1 ]; then       # a successful start: a fake :8081 that answers 200 and a fake franken_dec_glm process
  python3 $T/fake_http.py > /dev/null 2>&1 < /dev/null & echo \$! >> $T/up_pids
  bash -c 'exec -a franken_dec_glm sleep 60' > /dev/null 2>&1 < /dev/null & echo \$! >> $T/up_pids
  sleep 1
fi
exit \${FAKE_SA_RC:-1}
EOF
  cat > "$T/fake_http.py" <<'EOF'
import http.server, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 8081), H); s.timeout = 1; end = time.time() + 30
while time.time() < end: s.handle_request()
EOF
  cat > "$T/start_glm53.sh" <<EOF
#!/bin/bash
echo "start_glm53" >> $T/calls
python3 $T/fake_http.py > /dev/null 2>&1 < /dev/null & echo \$! >> $T/up_pids      # a fake :8081 that answers 200
exit 0
EOF
  chmod +x "$T/fake_serve_alt.sh" "$T/start_glm53.sh"
}
wd() {   # wd <extra env...> -- run the watchdog once in the current fake HOME
  env HOME="$T" WATCHDOG_LOG="$T/wd.log" SERVICE_SERVE_ALT="$T/fake_serve_alt.sh" SERVICE_FRANKEN_LOG="$T/franken.log" WATCHDOG_GRACE=0 SERVICE_WAIT_STEPS=3 SERVICE_POLL=1 "$@" \
      bash "$T/bench/gateway_watchdog.sh" > "$T/out" 2>&1; echo $? > "$T/rc"
}
calls() { cat "$T/calls" 2>/dev/null; }
nocalls() { [ ! -s "$T/calls" ]; }
logq() { grep -q -- "$1" "$T/wd.log" 2>/dev/null; }
fake_server() { python3 -c 'import time; time.sleep(300)' openai_server.py & PIDS+=($!); FSRV=$!; }
fake_engine() { bash -c 'exec -a franken_dec_glm sleep 300' & PIDS+=($!); FENG=$!; }

echo "== 1. reservation flag present: does nothing"
newhome t1; touch "$T/bench/.dev_reserved"; wd
check "no call, no log" 'nocalls && [ ! -s "$T/wd.log" ]'

echo "== 2. a live lock holder (a chain / the keeper): does nothing"
newhome t2; mkdir -p "$T/bench/.rig.lock"; sleep 300 & PIDS+=($!); echo "chain $! 2026-10-06T00:00:00+0000" > "$T/bench/.rig.lock/owner"; wd
check "no call" 'nocalls'

echo "== 3. down, default engine: starts the Franken engine through serve_alt, without the old fallback"
newhome t3; wd FAKE_SA_RC=1
check "serve_alt franken-glm, NO_OLD_FALLBACK=1" 'calls | grep -qx "serve_alt franken-glm NO_OLD_FALLBACK=1"'
check "the old start script was NOT run" '! calls | grep -q start_glm53'
check "failure logged, old engine not started" 'logq "Franken restart FAILED" && logq "old engine was NOT started"'
check "one attempt recorded" '[ "$(wc -l < "$T/bench/.watchdog_attempts")" = 1 ]'

echo "== 4. attempt cap: 3 attempts in 3 h -> no more tries"
newhome t4; n=$(date +%s); printf '%s\n%s\n%s\n' $((n-600)) $((n-300)) $((n-60)) > "$T/bench/.watchdog_attempts"; wd
check "no call" 'nocalls'
check "gives up, says needs a person" 'logq "needs a person"'

echo "== 5. old attempts (older than 3 h) do not count"
newhome t5; n=$(date +%s); printf '%s\n%s\n%s\n' $((n-20000)) $((n-19000)) $((n-18000)) > "$T/bench/.watchdog_attempts"; wd FAKE_SA_RC=1
check "serve_alt called" 'calls | grep -q "serve_alt franken-glm"'
check "the stale lines were pruned (1 line left)" '[ "$(wc -l < "$T/bench/.watchdog_attempts")" = 1 ]'

echo "== 6. a successful Franken start: logged, attempt counter cleared"
newhome t6; wd FAKE_SA_RC=0 FAKE_SA_UP=1
for p in $(cat "$T/up_pids" 2>/dev/null); do PIDS+=("$p"); done
check "serve_alt franken-glm was called" 'calls | grep -q "serve_alt franken-glm NO_OLD_FALLBACK=1"'
check "success logged" 'logq "gateway back (Franken engine)"'
check "attempt counter cleared" '[ ! -e "$T/bench/.watchdog_attempts" ]'
for p in $(cat "$T/up_pids" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 1

echo "== 7. service_engine=colibri-glm: the OLD start script is used, serve_alt is not"
newhome t7; echo colibri-glm > "$T/bench/service_engine"; wd
check "start_glm53 called, serve_alt not" 'calls | grep -qx start_glm53 && ! calls | grep -q serve_alt'
check "logged as the OLD engine" 'logq "OLD Colibri engine"'
check "and it came back" 'logq "gateway back after"'
for p in $(cat "$T/up_pids" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 1

echo "== 8. a gateway server whose Franken engine is alive is NOT killed"
newhome t8; fake_server; fake_engine; sleep 2; wd
check "server still alive" 'kill -0 $FSRV 2>/dev/null'
check "no restart" 'nocalls'
kill -9 "$FSRV" "$FENG" 2>/dev/null; wait "$FSRV" "$FENG" 2>/dev/null; sleep 1

echo "== 9. a gateway server with NO engine behind it is killed and the Franken engine is started"
newhome t9; fake_server; sleep 2; wd FAKE_SA_RC=1
check "server killed" '! kill -0 $FSRV 2>/dev/null'
check "serve_alt franken-glm called" 'calls | grep -q "serve_alt franken-glm NO_OLD_FALLBACK=1"'

echo "== 10. unknown engine name: warned, falls back to franken-glm (never to the old engine)"
newhome t10; echo bogus > "$T/bench/service_engine"; wd
check "warned and used franken-glm" 'logq "unknown engine" && calls | grep -q "serve_alt franken-glm"'

echo "== 11. run_chain.sh: what it leaves in service after a chain (the chain itself is a script that exits 0)"
rc_run() {
  printf '#!/bin/bash\nexit 0\n' > "$T/chain_ok.sh"; chmod +x "$T/chain_ok.sh"
  env HOME="$T" SERVICE_SERVE_ALT="$T/fake_serve_alt.sh" SERVICE_FRANKEN_LOG="$T/franken.log" SERVICE_WAIT_STEPS=3 SERVICE_POLL=1 "$@" \
      bash "$T/bench/run_chain.sh" "$T/chain_ok.sh" > "$T/out" 2>&1; echo $? > "$T/rc"
}
outq() { grep -q -- "$1" "$T/out" 2>/dev/null; }
fake_http() { python3 "$T/fake_http.py" > /dev/null 2>&1 < /dev/null & PIDS+=($!); FHTTP=$!; sleep 1; }
newhome r1; touch "$T/bench/.dev_reserved"; rc_run
check "flag present: gateway left down, nothing restored" 'outq "gateway left down" && nocalls'
newhome r2; fake_http; fake_server; fake_engine; sleep 1; rc_run
check "a healthy Franken service (200 + server + franken_dec_glm) is left alone" 'nocalls && outq "engine: franken-glm, alive: yes"'
kill -9 "$FHTTP" "$FSRV" "$FENG" 2>/dev/null; wait "$FHTTP" "$FSRV" "$FENG" 2>/dev/null; sleep 1
newhome r3; rc_run FAKE_SA_RC=1
check "nothing alive: the Franken engine is restored through serve_alt, no old fallback, lock already released" \
  'calls | grep -qx "serve_alt franken-glm NO_OLD_FALLBACK=1" && ! calls | grep -q LOCK_HELD_AT_CALL && ! calls | grep -q start_glm53'
check "a failed restore is reported, not hidden" 'outq "NOT in service"'
newhome r4; echo colibri-glm > "$T/bench/service_engine"; rc_run
check "service_engine=colibri-glm: the old start script is used" 'calls | grep -qx start_glm53 && ! calls | grep -q serve_alt'
for p in $(cat "$T/up_pids" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 1
newhome r5; fake_http; fake_server; sleep 1; rc_run FAKE_SA_RC=1
check "200 but no engine behind the server: treated as down and restored" 'outq "treating as down" && calls | grep -q "serve_alt franken-glm"'
kill -9 "$FHTTP" "$FSRV" 2>/dev/null; wait "$FHTTP" "$FSRV" 2>/dev/null; sleep 1

echo "== 12. serve_alt.sh restore_glm honours SERVE_ALT_NO_OLD_FALLBACK (functions sourced, every effect stubbed)"
SA=$ROOT/serve_alt_funcs.sh; sed '/^case "\${1:-}" in/,$d' "$HERE/serve_alt.sh" > "$SA"; cp "$HERE/rig_lock.sh" "$ROOT/"
unit() {   # unit <home-name> <flag 0|1> <env fallback 0|1> -> prints the call log
  local H=$ROOT/$1; mkdir -p "$H/bench"; [ "$2" = 1 ] && touch "$H/bench/.dev_reserved"; : > "$H/calls"
  ( export HOME=$H; [ "$3" = 1 ] && export SERVE_ALT_NO_OLD_FALLBACK=1
    . "$SA" 2>/dev/null; set +e
    ensure_alt_stopped() { echo ENSURE_STOPPED >> "$H/calls"; }; stop_gateway() { echo STOP_GATEWAY >> "$H/calls"; }
    serve_alt_lock_release() { echo LOCK_RELEASE >> "$H/calls"; }; assert_vram_free() { return 0; }
    warm_glm() { echo WARM >> "$H/calls"; }; assert_glm_resident() { return 0; }; start_gateway() { echo START_OLD_GATEWAY >> "$H/calls"; return 1; }
    restore_glm "unit test" > "$H/out" 2>&1; echo "rc=$?" >> "$H/calls" )
  cat "$H/calls"
}
r=$(unit u1 0 1); check "no flag, switch set: gateway stopped, lock freed, rc=1, old engine NOT warmed or started" \
  'echo "$r" | grep -q STOP_GATEWAY && echo "$r" | grep -q LOCK_RELEASE && echo "$r" | grep -q "rc=1" && ! echo "$r" | grep -qE "WARM|START_OLD_GATEWAY"'
r=$(unit u2 0 0); check "no flag, switch NOT set: old behaviour intact (warms, tries the old gateway)" 'echo "$r" | grep -q WARM && echo "$r" | grep -q START_OLD_GATEWAY'
r=$(unit u3 1 1); check "flag present: reserved path, rc=0, nothing started" 'echo "$r" | grep -q "rc=0" && ! echo "$r" | grep -qE "WARM|START_OLD_GATEWAY"'

echo; echo "test_watchdog: $PASS passed, $FAILN failed"; [ "$FAILN" = 0 ]
