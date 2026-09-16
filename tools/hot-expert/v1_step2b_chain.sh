#!/bin/bash
# v1_step2b_chain.sh -- FRANKEN-ENGINE-PLAN §2.6, item V1, STEP 2, part B:
# what the tier is WORTH, in the regime arm C's floor was taken in.
#
#   setsid nohup ~/src/colibri-v1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-v1/tools/hot-expert/v1_step2b_chain.sh \
#       >> ~/bench/v1_step2b.log 2>&1 < /dev/null &
#
# THE FLOOR, and why this harness and not another. Arm C of the franken smoke
# (~/bench/ctx_ladder_out/fk09152356_C.jsonl, 2026-09-15) is `coli serve` on
# qwen36 over HTTP, driven by context_ladder.py at --steps 256 --gen 16
# --followups 1 --cap 256: turn 1 413 prompt tokens, TTFT 32.85 s, decode
# 14.84 tok/s; turn 2 459 tokens, 20.67 s, 14.38 tok/s. Those exact arguments
# are reproduced here -- a different --steps or --gen would be a different
# number wearing the same name (MEASURING.md's whole subject).
#
# ONE binary, ONE knob: arm A is qwen36-vk with Q36_VULKAN unset (= the CPU
# engine, proved bit-identical in step 1), arm B is the same file with
# Q36_VULKAN=1. COLI_ENGINE points `coli serve` at it.
#
# ORDER A,B,B,A, per CLAUDE.md: on this box an uninterleaved pair read +20%
# and +11% on changes whose real effects were +3.9% and zero, because the
# second arm inherits a warm page cache. Each arm is a FRESH `coli serve`
# process, so no arm inherits the previous one's KV or expert cache.
#
# SLOTS: qwen36 has none. c/qwen36.c's serve_read_req parses the SUBMIT line's
# slot field and discards it ((void)slot), serve_loop reads and serves exactly
# one request at a time, and serve_one resets the recurrent state, the KV
# length, first_step, the resident set and the router EMA on EVERY request. So
# tworeq.py's four-slot identity check has nothing to attach to here. What
# replaces it is the invariant that actually exists: request P1, then P2, then
# P1 AGAIN in the same server process -- the two P1 replies must be identical,
# or something (the tier's issue/take state above all) is leaking across
# requests. That is run on the tier-ON arm, where the leak would be.
set -u
TAG=v1s2b$(date +%m%d%H%M)
OUT=~/bench/v1_step2b_out; mkdir -p "$OUT"
SRC=~/src/colibri-v1
HERE="$SRC/tools/hot-expert"
M=${V1_MODEL:-/home/ronald/models/qwen36_i4_gs64}
CAP=${V1_CAP:-256}
DEV3=${V1_DEV3:-2}
PORT=${V1_PORT:-8600}
STEPS=${V1_STEPS:-256}
GEN=${V1_GEN:-16}
FOLLOWUPS=${V1_FOLLOWUPS:-1}
ARM_TIMEOUT=${V1_ARM_TIMEOUT:-900}
BIN=~/bench/qwen36-vk.v1
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
RESTART_GATEWAY=0
SRV_PID=""

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  local code=""
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[v1s2b] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}
wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[v1s2b] FATAL: a $n is still alive after 240s"; return 1
}
stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[v1s2b] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine glm53
}
stop_srv() {
  [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null && {
    kill "$SRV_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null || true
  }
  # setsid detaches `coli serve` into its own session, so killing the subshell
  # above does not reach it. Kill the engine (which is what holds dev3's VRAM
  # and makes coli exit) and then coli itself. Bracket form throughout: a bare
  # `pkill -f "coli serve"` over ssh matches the ssh command line and kills the
  # session (CLAUDE.md).
  pkill -9 -x qwen36-vk 2>/dev/null || true
  pkill -f "col[i] serve" 2>/dev/null || true
  SRV_PID=""
  wait_no_engine qwen36-vk
  for _ in $(seq 1 30); do pgrep -f "col[i] serve" >/dev/null 2>&1 || break; sleep 1; done
}
on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  stop_srv
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== v1_step2b_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== v1_step2b_chain $TAG $(date -Is)"
echo "=== floor (arm C, fk09152356): turn1 413 tok, TTFT 32.85 s, decode 14.84 tok/s"
echo "===                            turn2 459 tok, TTFT 20.67 s, decode 14.38 tok/s"
for e in glm53 qwen38 qwen38-vk qwen36 qwen36-vk; do
  [ "$e" = glm53 ] && continue
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[v1s2b] $e is running -- refusing"; exit 1; }
done
# Another session's KL job would steal cores from an 8-thread measurement.
if pgrep -f "kl_[c]ompare.py" >/dev/null 2>&1; then
  echo "[v1s2b] REFUSED: a kl_compare.py is running -- it would contend for the 8 cores"; exit 1
fi
stop_gateway || exit 1

echo "[v1s2b] refreshing $SRC"; git -C "$SRC" pull --ff-only || exit 1
git -C "$SRC" log -1 --oneline
GITSHA=$(git -C "$SRC" rev-parse --short HEAD)
[ -x "$BIN" ] || { echo "[v1s2b] FATAL: $BIN missing -- run v1_step2a_chain.sh first"; exit 1; }
echo "[v1s2b] qwen36-vk.v1 sha256=$(sha256sum $BIN | cut -c1-16)"

echo "--- DPM level per card, recorded and NOT changed (record §Q13: it moves numbers)"
for d in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  [ -r "$d" ] && echo "  $d = $(cat "$d")"
done

precheck() {                       # precheck <label>
  echo "--- precheck $1"
  for c in /sys/class/drm/card*/device/mem_info_vram_used; do
    [ -r "$c" ] || continue
    used=$(cat "$c"); echo "    $c = $((used/1048576)) MB"
    [ "$used" -le 1073741824 ] || { echo "[v1s2b] REFUSED at $1: $c holds $((used/1048576)) MB"; return 1; }
  done
  local pct
  pct=$(fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors | python3 -c "
import sys
PAGE=4096; tot=res=0
for line in sys.stdin.read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    size,r=int(f[-2]),int(f[-1]); tot+=-(-size//PAGE)*PAGE; res+=r
print(f'{res/tot*100 if tot else 0:.1f}')
")
  echo "    container resident: $pct%"
  python3 -c "import sys; sys.exit(0 if float('$pct')>=90.0 else 1)" \
    || { echo "[v1s2b] REFUSED at $1: residency $pct% < 90%"; return 1; }
  return 0
}

echo "--- warming the qwen36 container ($M)"
find "$M" -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true

# arm <name> <on|off>
arm() {
  local name="$1" tier="$2"
  local clog="$OUT/${TAG}_${name}_serve.log" json="$OUT/${TAG}_${name}.jsonl"
  echo
  echo "=== arm $name (tier $tier) $(date -Is)"
  precheck "$name" || return 1
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_ENGINE="$BIN" COLI_MODEL="$M"
    export COLI_VK_SHADERS="$SRC/c/shaders" COLI_VK_DEV3="$DEV3"
    if [ "$tier" = on ]; then export Q36_VULKAN=1; fi
    setsid "$SRC/c/coli" serve --host 127.0.0.1 --port "$PORT" --cap "$CAP" \
      >> "$clog" 2>&1 < /dev/null ) &
  SRV_PID=$!
  local up=0 code
  for _ in $(seq 1 120); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[v1s2b] arm $name FAILED: coli serve never answered"; tail -30 "$clog"; stop_srv; return 1; }
  echo "  server up, pid=$SRV_PID"
  if [ "$tier" = on ]; then
    if grep -qE '^\[qtier-vk\] resident|Vulkan VRAM expert tier active' "$clog"; then
      grep -E '^\[qtier-vk\]' "$clog" | tail -3 | sed 's/^/    /'
    else
      echo "[v1s2b] arm $name FAILED: no [qtier-vk] line -- the tier did not come up"
      tail -30 "$clog"; stop_srv; return 1
    fi
  else
    grep -qE '^\[qtier-vk\]' "$clog" && { echo "[v1s2b] arm $name FAILED: the tier spoke in a tier-OFF arm"; stop_srv; return 1; }
  fi
  timeout "$ARM_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$PORT" \
      --steps "$STEPS" --gen "$GEN" --followups "$FOLLOWUPS" \
      --snap "$M" --min-resident 90 --warm \
      --arm "$name" --tag "$TAG" --json "$json" --server-log "$clog"
  local rc=$?
  echo "  ladder rc=$rc json=$json"
  python3 -c "
import json,sys
try:
    for line in open('$json'):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        print('    turn=%s kind=%s prompt_tokens=%s decode_tps=%s ttft_s=%s' %
              (r.get('turn'), r.get('kind'), r.get('prompt_tokens'),
               r.get('decode_tps'), r.get('ttft_s')))
except FileNotFoundError:
    print('    (no json)')
"
  # the identity invariant, on the tier-on arm only (see the header)
  if [ "$tier" = on ] && [ "$name" = B1 ]; then
    echo "  --- sequential identity: P1, P2, P1 again in ONE server process"
    idq() {                         # idq <file> <prompt>
      curl -s -m 300 -H 'Content-Type: application/json' \
        -d "{\"model\":\"qwen36\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":24,\"temperature\":0}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > "$1" 2>&1
      python3 -c "
import json,sys
try: print(json.load(open('$1'))['choices'][0]['message']['content'])
except Exception as e: print('PARSE-FAIL', e)
" > "$1.txt"
    }
    idq "$OUT/${TAG}_id_p1a.json" "List the days of the week in order."
    idq "$OUT/${TAG}_id_p2.json"  "Name three prime numbers greater than one hundred."
    idq "$OUT/${TAG}_id_p1b.json" "List the days of the week in order."
    local n1
    n1=$(tr -d '[:space:]' < "$OUT/${TAG}_id_p1a.txt" | wc -c)
    if [ "${n1:-0}" -lt 8 ]; then
      echo "      REFUSED: P1's reply is ${n1:-0} non-space chars -- an empty identity check is not a pass"
      IDENT=REFUSED
    elif cmp -s "$OUT/${TAG}_id_p1a.txt" "$OUT/${TAG}_id_p1b.txt"; then
      echo "      IDENTICAL ($n1 non-space chars) -- no state leaks from the request in between"
      IDENT=PASS
    else
      echo "      DIFFERS -- state leaks across requests"
      echo "      first : $(head -c 160 "$OUT/${TAG}_id_p1a.txt" | tr '\n' ' ')"
      echo "      repeat: $(head -c 160 "$OUT/${TAG}_id_p1b.txt" | tr '\n' ' ')"
      IDENT=FAIL
    fi
    echo "      P2 (the request in between): $(head -c 120 "$OUT/${TAG}_id_p2.txt" | tr '\n' ' ')"
  fi
  stop_srv
  return 0
}

IDENT=NOTRUN
RC=0
arm A1 off || RC=1
arm B1 on  || RC=1
arm B2 on  || RC=1
arm A2 off || RC=1

echo
echo "=== V1 step 2b: decode and TTFT, A,B,B,A ==="
. "$HERE/gate_lib.sh"
vals() {                  # vals <field> <kind> <arms...>
  local field="$1" kind="$2"; shift 2
  local out=""
  for a in "$@"; do
    local v
    v=$(python3 -c "
import json
for line in open('$OUT/${TAG}_${a}.jsonl'):
    line=line.strip()
    if not line: continue
    r=json.loads(line)
    if r.get('kind')=='$kind' and r.get('$field') is not None:
        print('%.4f' % float(r['$field'])); break
" 2>/dev/null)
    out="$out $v"
  done
  echo "$out"
}
for kind in ladder followup; do
  for field in decode_tps ttft_s; do
    A=$(vals "$field" "$kind" A1 A2); B=$(vals "$field" "$kind" B1 B2)
    echo "  -- $kind $field: A(off)=[$A ] B(on)=[$B ]"
    gate_ab_verdict "$kind $field (off -> on)" "$A" "$B" || true
  done
done

echo
echo "=== against the arm C floor (same harness, same arguments) ==="
python3 -c "
import json, glob
floor={'ladder':(14.84,32.85),'followup':(14.38,20.67)}
for kind in ('ladder','followup'):
    for arm in ('A1','A2','B1','B2'):
        try:
            rows=[json.loads(l) for l in open('$OUT/${TAG}_%s.jsonl'%arm) if l.strip()]
        except FileNotFoundError:
            continue
        for r in rows:
            if r.get('kind')!=kind: continue
            d,t=r.get('decode_tps'),r.get('ttft_s')
            fd,ft=floor[kind]
            if d is None or t is None: continue
            print('  %-8s %-9s prompt=%-5s decode %6.2f tok/s (%+.1f%% vs floor %.2f)  TTFT %7.2f s (%+.1f%% vs floor %.2f)'
                  % (arm, kind, r.get('prompt_tokens'), d, (d-fd)/fd*100, fd, t, (t-ft)/ft*100, ft))
"

echo
echo "=== multi-request identity (tier on): $IDENT"
echo "    qwen36 has NO KV slots: serve_read_req discards the SUBMIT slot field,"
echo "    serve_loop serves one request at a time, and serve_one resets the"
echo "    recurrent state, kv_len, first_step, the resident set and the router EMA"
echo "    per request. tworeq.py's four-slot check does not apply to this engine."
[ "$IDENT" = PASS ] || [ "$IDENT" = NOTRUN ] || RC=1

echo
if [ "$RC" -eq 0 ]; then echo "=== V1 step 2b: chain completed (tree $GITSHA)"
else echo "=== V1 step 2b: chain had a FAILING arm (tree $GITSHA)"; fi

echo
echo "[v1s2b] re-warming the GLM container before the gateway returns"
GLM=${V1_GLM_MODEL:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
cat "$GLM"/*.safetensors > /dev/null 2>&1 || true
start_gateway
RESTART_GATEWAY=0
echo "[v1s2b] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[v1s2b] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
