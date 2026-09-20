#!/bin/bash
# f3s1_perf_chain.sh -- FRANKEN-ENGINE-PLAN item F3, step 1, performance: what
# does the ported two-device tier over dev2+dev3, serving the row-wise int8
# container, cost or buy against V1's own served arm.
#
#   setsid nohup ~/src/colibri-f3s1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f3s1/tools/hot-expert/f3s1_perf_chain.sh \
#       >> ~/bench/f3s1_perf.log 2>&1 < /dev/null &
#
# Pattern: V1 step 2b's ladder (`coli serve` + context_ladder.py --steps 256
# --gen 16 --followups 1 --cap 256, arm C's own harness), interleaved A,B,B,A,
# judged by gate_ab_verdict. Requires f3s1_kl_chain.sh (chain a) to have
# already built and copied ~/bench/qwen36-vk.f3s1 -- same dependency V1's
# step 2b had on step 2a's binary.
#
#   A = ~/bench/qwen36-vk.v1 (pristine)  + int4-gs64 container, COLI_VK_DEV3=2
#       only -- V1's own served-candidate arm (record V1: 21.6 tok/s).
#   B = ~/bench/qwen36-vk.f3s1 (this branch) + row-wise int8 container,
#       COLI_VK_DEV2=auto COLI_VK_DEV3=auto -- F3 step 1's arm.
#
# THE FIFTH RUN IS A SEPARATE, CLI-MODE PROFILE, NOT COLI_TIMERS=1 INSIDE THE
# LADDER, for two independent reasons found by reading the code before
# assuming either:
#   1. qwen36.c's own per-phase timers (M-PROF, g_tm_dec[]/g_tm_pre[],
#      "COLI_TIMERS=1 for the detailed ... phase breakdown") are NOT free to
#      enable: the main decode/attention/moe/lm_head clock_gettime() calls
#      already run unconditionally (tm_add() is called with no tm_on() guard
#      at every one of those call sites), but the DeltaNet SUB-phase timers
#      (g_dn_sub[], proj/conv/l2n+rec/norm+out) are gated `if (tm_on() &&
#      S==1)` and add up to four EXTRA clock_gettime() calls PER DeltaNet
#      LAYER per decode token (30 layers) that do not happen at all with
#      COLI_TIMERS unset. Small, but not nothing, and not verified negligible
#      on this box -- so it does not go in the A,B,B,A ladder.
#   2. Even if it were free, tm_report() -- the function that actually PRINTS
#      the "[timers]" block -- is called once, at the very end of main()'s
#      CLI-mode return path (c/qwen36.c:3168), and `coli serve` mode returns
#      immediately after serve_loop() (line ~3070: "serve_loop(&m); return
#      0;") WITHOUT ever reaching that line. This is the exact shape
#      CLAUDE.md records for glm53's OPTIME table under SERVE (an instrument
#      that was assumed to fire and never checked): under `coli serve`,
#      qwen36's own per-phase report NEVER PRINTS, timers on or off. So a
#      COLI_TIMERS=1 ladder arm would not even produce the profile it was
#      meant to capture.
# The fix for both is the same: run the profile as a DIRECT CLI invocation
# (not `coli serve`), which reaches tm_report() normally, and run it AFTER
# the A,B,B,A ladder so nothing it measures leaks into the verdict.
set -u
TAG=f3s1perf$(date +%m%d%H%M)
OUT=~/bench/f3s1_perf_out; mkdir -p "$OUT"
SRC=~/src/colibri-f3s1
HERE="$SRC/tools/hot-expert"
I4=${F3S1_I4:-/home/ronald/models/qwen36_i4_gs64}
I8=${F3S1_I8:-/home/ronald/models/qwen36_i8_row}
CAP=${F3S1_CAP:-256}
DEV3_PIN=${F3S1_DEV3_PIN:-2}
PORT=${F3S1_PORT:-8601}
STEPS=${F3S1_STEPS:-256}
GEN=${F3S1_GEN:-16}
FOLLOWUPS=${F3S1_FOLLOWUPS:-1}
ARM_TIMEOUT=${F3S1_ARM_TIMEOUT:-900}
NEXPERTS=${F3S1_NEXPERTS:-10240}
A_BIN=~/bench/qwen36-vk.v1
B_BIN=~/bench/qwen36-vk.f3s1
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
  echo "[f3s1perf] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}
wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[f3s1perf] FATAL: a $n is still alive after 240s"; return 1
}
stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[f3s1perf] stopping the owner's gateway for the duration of this run"
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
  # bracket form throughout: a bare `pkill -f "coli serve"` over ssh matches
  # the ssh command line itself and kills the session (CLAUDE.md).
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
  pkill -9 -x qwen36 2>/dev/null || true
  # Kill glm53 ONLY when the gateway is about to be restarted here. On the normal
  # path the body has already restarted it (RESTART_GATEWAY=0) and that glm53 IS the
  # owner's served engine: killing it left openai_server.py up with a defunct engine,
  # /v1/models=200 and every chat a 500 "engine dispatcher stopped" (2026-09-20).
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    pkill -9 -x glm53 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== f3s1_perf_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== f3s1_perf_chain $TAG $(date -Is)"
echo "=== floor (arm C, fk09152356): turn1 413 tok, TTFT 32.85 s, decode 14.84 tok/s"
echo "===                            turn2 459 tok, TTFT 20.67 s, decode 14.38 tok/s"
echo "=== V1's own served arm (record V1): turn1 decode 21.637 tok/s, TTFT 10.673 s"
for e in glm53 qwen38 qwen38-vk qwen36 qwen36-vk; do
  [ "$e" = glm53 ] && continue
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[f3s1perf] $e is running -- refusing"; exit 1; }
done
if pgrep -f "kl_[c]ompare.py" >/dev/null 2>&1; then
  echo "[f3s1perf] REFUSED: a kl_compare.py is running -- it would contend for the 8 cores"; exit 1
fi
[ -x "$A_BIN" ] || { echo "[f3s1perf] FATAL: $A_BIN (pristine V1) missing"; exit 1; }
[ -x "$B_BIN" ] || { echo "[f3s1perf] FATAL: $B_BIN missing -- run f3s1_kl_chain.sh (chain a) first"; exit 1; }
stop_gateway || exit 1

echo "[f3s1perf] tree at $SRC: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITSHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[f3s1perf] A_BIN=$A_BIN sha256=$(sha256sum "$A_BIN" | cut -c1-16)"
echo "[f3s1perf] B_BIN=$B_BIN sha256=$(sha256sum "$B_BIN" | cut -c1-16)"

echo "--- DPM level per card, recorded and NOT changed (record §Q13: it moves numbers)"
for d in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  [ -r "$d" ] && echo "  $d = $(cat "$d")"
done

resident_pct() {
  local dir="$1"
  fincore --bytes --output FILE,SIZE,RES "$dir"/*.safetensors | python3 -c "
import sys
PAGE=4096; tot=res=0
for line in sys.stdin.read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    size,r=int(f[-2]),int(f[-1]); tot+=-(-size//PAGE)*PAGE; res+=r
print(f'{res/tot*100 if tot else 0:.1f}')
"
}
warm_two_pass() {
  local dir="$1"
  for _ in 1 2; do
    find "$dir" -maxdepth 1 -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true
  done
}

echo "--- warming $I4 (arm A) and $I8 (arm B), two passes each (f3s0's own finding:"
echo "    one pass can leave a container under-resident, run f3s0kl09201409)"
warm_two_pass "$I4"
warm_two_pass "$I8"
PCT4=$(resident_pct "$I4"); echo "[f3s1perf] i4 resident: $PCT4%"
PCT8=$(resident_pct "$I8"); echo "[f3s1perf] i8 resident: $PCT8%"
python3 -c "import sys; sys.exit(0 if float('$PCT4')>=90.0 and float('$PCT8')>=90.0 else 1)" \
  || { echo "[f3s1perf] REFUSED: a container is under 90% resident"; exit 1; }

precheck() {                       # precheck <label>
  # WAIT for VRAM release, do not just read it once (V1 step2b's own finding:
  # the driver frees a multi-GB tier over several seconds after the process
  # exits, and a snapshot taken too early reports gigabytes still falling).
  local waited=0 busy=1
  while [ "$waited" -lt 180 ]; do
    busy=0
    for c in /sys/class/drm/card*/device/mem_info_vram_used; do
      [ -r "$c" ] || continue
      [ "$(cat "$c")" -gt 1073741824 ] && busy=1
    done
    [ "$busy" = 0 ] && break
    sleep 5; waited=$((waited+5))
  done
  [ "$waited" -gt 0 ] && echo "    (waited ${waited}s for VRAM to be released)"
  for c in /sys/class/drm/card*/device/mem_info_vram_used; do
    [ -r "$c" ] || continue
    used=$(cat "$c"); echo "    $c = $((used/1048576)) MB"
    [ "$used" -le 1073741824 ] || { echo "[f3s1perf] REFUSED at $1: $c holds $((used/1048576)) MB after ${waited}s"; return 1; }
  done
  return 0
}

# arm <name> <A|B>
arm() {
  local name="$1" which="$2"
  local bin snap tier_env clog json
  if [ "$which" = A ]; then bin="$A_BIN"; snap="$I4"
  else bin="$B_BIN"; snap="$I8"; fi
  clog="$OUT/${TAG}_${name}_serve.log"; json="$OUT/${TAG}_${name}.jsonl"
  echo
  echo "=== arm $name (which=$which bin=$(basename "$bin") snap=$(basename "$snap")) $(date -Is)"
  precheck "$name" || return 1
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_ENGINE="$bin" COLI_MODEL="$snap"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    if [ "$which" = A ]; then
      export Q36_VULKAN=1 COLI_VK_DEV3="$DEV3_PIN"
      unset COLI_VK_DEV2
    else
      export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    fi
    setsid "$SRC/c/coli" serve --host 127.0.0.1 --port "$PORT" --cap "$CAP" \
      >> "$clog" 2>&1 < /dev/null ) &
  SRV_PID=$!
  local up=0 code
  for _ in $(seq 1 120); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[f3s1perf] arm $name FAILED: coli serve never answered"; tail -30 "$clog"; stop_srv; return 1; }
  echo "  server up, pid=$SRV_PID"
  grep -E '^\[qtier-vk\]|^\[qtier\] warmstart|^\[VK\]' "$clog" | tail -8 | sed 's/^/    /'
  local invram
  invram=$(grep -oE '[0-9]+ in VRAM' "$clog" | tail -1 | grep -oE '^[0-9]+')
  if [ -z "$invram" ]; then
    echo "[f3s1perf] arm $name FAILED: no warmstart line -- the tier did not come up"
    tail -30 "$clog"; stop_srv; return 1
  fi
  if [ "$invram" != "$NEXPERTS" ]; then
    echo "[f3s1perf] arm $name FAILED: $invram of $NEXPERTS experts in VRAM"
    stop_srv; return 1
  fi
  echo "    residency asserted: $invram/$NEXPERTS experts in VRAM"
  timeout "$ARM_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$PORT" \
      --steps "$STEPS" --gen "$GEN" --followups "$FOLLOWUPS" \
      --snap "$snap" --min-resident 90 --warm \
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
  stop_srv
  return 0
}

RC=0
arm A1 A || RC=1
arm B1 B || RC=1
arm B2 B || RC=1
arm A2 A || RC=1

echo
echo "=== F3 step 1 perf: decode and TTFT, A,B,B,A ==="
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
    echo "  -- $kind $field: A(v1/int4/dev3)=[$A ] B(f3s1/int8/dev2+3)=[$B ]"
    gate_ab_verdict "$kind $field (A -> B)" "$A" "$B" || true
  done
done

echo
echo "=== against V1's own served arm and the arm C floor ==="
python3 -c "
import json
v1={'ladder':(21.637,10.673),'followup':(21.331,11.790)}
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
            vd,vt=v1[kind]; fd,ft=floor[kind]
            if d is None or t is None: continue
            print('  %-8s %-9s prompt=%-5s decode %6.2f tok/s (%+.1f%% vs V1 %.2f, %+.1f%% vs floor %.2f)  '
                  'TTFT %7.2f s (%+.1f%% vs V1 %.2f, %+.1f%% vs floor %.2f)'
                  % (arm, kind, r.get('prompt_tokens'), d, (d-vd)/vd*100, vd, (d-fd)/fd*100, fd,
                     t, (t-vt)/vt*100, vt, (t-ft)/ft*100, ft))
"

echo
echo "=== fifth run: CLI-mode profile of arm B (int8, dev2+dev3), COLI_TIMERS=1 ==="
echo "    (separate from the ladder -- see this file's header for why)"
PACKET="$HERE/x2_packet_450.txt"
PROFILE_OUT="$OUT/${TAG}_profile.txt"
if [ -s "$PACKET" ]; then
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$I8" N_NEW=128 NOSTREAM=1 COLI_TIMERS=1
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    /usr/bin/time -v "$B_BIN" "$CAP" 8 "$PACKET" ) \
      > "$OUT/${TAG}_profile.out" 2> "$OUT/${TAG}_profile.err"
  {
    echo "# F3 step 1 fifth run: per-bucket profile, arm B (int8, dev2+dev3)"
    echo "# built from tree $GITSHA, binary $B_BIN"
    echo "# packet prompt, N_NEW=128, COLI_TIMERS=1, CLI mode (not coli serve -- see"
    echo "# f3s1_perf_chain.sh's header for why serve mode never prints this)"
    echo
    grep -E '^\[timers\]|^\[qtier-vk\]|^TTFT:|^PEAK RSS:|^Speed:' "$OUT/${TAG}_profile.err"
  } > "$PROFILE_OUT"
  echo "  profile written to $PROFILE_OUT:"
  sed 's/^/    /' "$PROFILE_OUT"
  if ! grep -q '^\[timers\]' "$PROFILE_OUT"; then
    echo "  FAIL: no [timers] block -- the profile run did not produce what step 2 needs"
    RC=1
  fi
else
  echo "  FATAL: $PACKET missing -- no profile taken"; RC=1
fi

echo
if [ "$RC" -eq 0 ]; then echo "=== F3 step 1 perf chain: completed (tree $GITSHA)"
else echo "=== F3 step 1 perf chain: had a FAILING arm or leg (tree $GITSHA)"; fi

echo
echo "[f3s1perf] restarting the gateway (SKIP_WARM=1 is enough -- CLAUDE.md: do not"
echo "           re-warm GLM's 184 GB by cat before restarting the gateway)"
start_gateway
RESTART_GATEWAY=0
echo "[f3s1perf] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[f3s1perf] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
