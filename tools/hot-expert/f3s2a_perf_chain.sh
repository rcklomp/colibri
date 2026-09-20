#!/bin/bash
# f3s2a_perf_chain.sh -- FRANKEN-ENGINE-PLAN item F3, step 2a, performance:
# what does moving lm_head and the DeltaNet input projections from the CPU
# to dev0 (Q36_VK_TRUNK=1) cost or buy against the SAME binary with the knob
# off, both arms serving the int8 container over dev2+dev3 (F3 step 1's own
# tier, unchanged by this item).
#
#   setsid nohup ~/src/colibri-f3s2a/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f3s2a/tools/hot-expert/f3s2a_perf_chain.sh \
#       >> ~/bench/f3s2a_perf.log 2>&1 < /dev/null &
#
# Pattern: f3s1_perf_chain.sh's ladder (`coli serve` + context_ladder.py
# --steps 256 --gen 16 --followups 1 --cap 256), interleaved A,B,B,A, judged
# by gate_ab_verdict. Requires f3s2a_kl_chain.sh (chain a) to have already
# built and copied ~/bench/qwen36-vk.f3s2a -- same dependency f3s1's own perf
# chain had on its own step's KL chain.
#
#   A = ~/bench/qwen36-vk.f3s2a, Q36_VK_TRUNK=0 (or unset) -- step 1's own
#       served shape: lm_head and DeltaNet projections on the CPU.
#   B = the SAME binary, Q36_VK_TRUNK=1 -- this item's arm: lm_head and every
#       DeltaNet layer's fused qkv++z projection on dev0.
#
# Gate: decode >= +20% conservative (gate_ab_verdict's own SEPARATED band).
# This is a REPORT -- it says whether the bar was met; the landing decision
# is the owner's, per this item's own brief and CLAUDE.md ("Report what the
# gate says, in numbers").
#
# THE FIFTH RUN IS A SEPARATE, CLI-MODE PROFILE, NOT COLI_TIMERS=1 INSIDE THE
# LADDER -- f3s1_perf_chain.sh's own reasoning applies verbatim here (its
# header, reasons 1 and 2: the DeltaNet sub-phase timers cost four EXTRA
# clock_gettime() calls per layer per token when COLI_TIMERS=1, not free
# enough for the A,B,B,A arms; and `coli serve` never reaches tm_report()
# at all, so a served-mode profile would print nothing). Both the lm_head
# timer (qwen36.c: g_tm_dec[5], wrapped around `if (!qt_lmhead_matmul(...))
# matmul_d(...)`) and the DeltaNet projection timer (g_dn_sub[0], wrapped
# around `if (!qt_dnproj_matmul(...)) {matmul_d...} matmul(b...);
# matmul(a...)`) time the WHOLE call, GPU or CPU -- qt_lmhead_matmul and
# qt_dnproj_matmul are synchronous (coli_vk_matmul blocks on the fence before
# returning), so with Q36_VK_TRUNK=1 these buckets already include the
# submit+fence+readback round trip with no engine change needed. This run
# checks that reading holds on real hardware, not just by inspection.
set -u
TAG=f3s2aperf$(date +%m%d%H%M)
OUT=~/bench/f3s2a_perf_out; mkdir -p "$OUT"
SRC=~/src/colibri-f3s2a
HERE="$SRC/tools/hot-expert"
I8=${F3S2A_I8:-/home/ronald/models/qwen36_i8_row}
CAP=${F3S2A_CAP:-256}
PORT=${F3S2A_PORT:-8601}
STEPS=${F3S2A_STEPS:-256}
GEN=${F3S2A_GEN:-16}
FOLLOWUPS=${F3S2A_FOLLOWUPS:-1}
ARM_TIMEOUT=${F3S2A_ARM_TIMEOUT:-900}
NEXPERTS=${F3S2A_NEXPERTS:-10240}
BIN=~/bench/qwen36-vk.f3s2a
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
  echo "[f3s2aperf] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}
wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[f3s2aperf] FATAL: a $n is still alive after 240s"; return 1
}
stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[f3s2aperf] stopping the owner's gateway for the duration of this run"
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
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== f3s2a_perf_chain exit rc=$rc tag=$TAG $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== f3s2a_perf_chain $TAG $(date -Is)"
echo "=== F3 step 1's own served arm (record §F3-STEP1): ladder decode 21.40/21.63 tok/s,"
echo "===                                                 TTFT 10.78/10.73 s"
echo "=== profile bucket to beat (record §F3-STEP1, 47.0 ms/token): lm_head 6.3 ms (13%),"
echo "===   DeltaNet projections 10.8 of 19.1 ms; the arithmetic in the plan's rev 36 entry"
echo "===   projects ~27 tok/s if the per-call round trip (~0.25 ms measured on the expert"
echo "===   tier) does not eat the CPU time saved -- report what was actually measured"
for e in glm53 qwen38 qwen38-vk qwen36 qwen36-vk; do
  [ "$e" = glm53 ] && continue
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[f3s2aperf] $e is running -- refusing"; exit 1; }
done
if pgrep -f "kl_[c]ompare.py" >/dev/null 2>&1; then
  echo "[f3s2aperf] REFUSED: a kl_compare.py is running -- it would contend for the 8 cores"; exit 1
fi
[ -x "$BIN" ] || { echo "[f3s2aperf] FATAL: $BIN missing -- run f3s2a_kl_chain.sh (chain a) first"; exit 1; }
stop_gateway || exit 1

echo "[f3s2aperf] tree at $SRC: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITSHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[f3s2aperf] BIN=$BIN sha256=$(sha256sum "$BIN" | cut -c1-16)"

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

echo "--- warming $I8 (both arms use it), two passes (f3s0's own finding: one pass can"
echo "    leave a container under-resident)"
warm_two_pass "$I8"
PCT8=$(resident_pct "$I8"); echo "[f3s2aperf] i8 resident: $PCT8%"
python3 -c "import sys; sys.exit(0 if float('$PCT8')>=90.0 else 1)" \
  || { echo "[f3s2aperf] REFUSED: i8 residency $PCT8% < 90%"; exit 1; }

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
    [ "$used" -le 1073741824 ] || { echo "[f3s2aperf] REFUSED at $1: $c holds $((used/1048576)) MB after ${waited}s"; return 1; }
  done
  return 0
}

# arm <name> <trunk 0|1>
arm() {
  local name="$1" trunk="$2"
  local clog json
  clog="$OUT/${TAG}_${name}_serve.log"; json="$OUT/${TAG}_${name}.jsonl"
  echo
  echo "=== arm $name (trunk=$trunk bin=$(basename "$BIN") snap=$(basename "$I8")) $(date -Is)"
  precheck "$name" || return 1
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_ENGINE="$BIN" COLI_MODEL="$I8"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export Q36_VK_TRUNK="$trunk"
    setsid "$SRC/c/coli" serve --host 127.0.0.1 --port "$PORT" --cap "$CAP" \
      >> "$clog" 2>&1 < /dev/null ) &
  SRV_PID=$!
  local up=0 code
  for _ in $(seq 1 120); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { up=1; break; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[f3s2aperf] arm $name FAILED: coli serve never answered"; tail -30 "$clog"; stop_srv; return 1; }
  echo "  server up, pid=$SRV_PID"
  grep -E '^\[qtier-vk\]|^\[trunk-vk\]|^\[qtier\] warmstart|^\[VK\]' "$clog" | tail -10 | sed 's/^/    /'
  local invram
  invram=$(grep -oE '[0-9]+ in VRAM' "$clog" | tail -1 | grep -oE '^[0-9]+')
  if [ -z "$invram" ]; then
    echo "[f3s2aperf] arm $name FAILED: no warmstart line -- the tier did not come up"
    tail -30 "$clog"; stop_srv; return 1
  fi
  if [ "$invram" != "$NEXPERTS" ]; then
    echo "[f3s2aperf] arm $name FAILED: $invram of $NEXPERTS experts in VRAM"
    stop_srv; return 1
  fi
  echo "    residency asserted: $invram/$NEXPERTS experts in VRAM"
  if [ "$trunk" = 1 ]; then
    grep -q '^\[trunk-vk\] lm_head' "$clog" \
      || { echo "[f3s2aperf] arm $name FAILED: trunk=1 but no [trunk-vk] lm_head line -- the trunk never came up"; stop_srv; return 1; }
    echo "    trunk asserted: lm_head resident on dev0"
  fi
  timeout "$ARM_TIMEOUT" python3 "$HERE/context_ladder.py" \
      --url "http://127.0.0.1:$PORT" \
      --steps "$STEPS" --gen "$GEN" --followups "$FOLLOWUPS" \
      --snap "$I8" --min-resident 90 --warm \
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
arm A1 0 || RC=1
arm B1 1 || RC=1
arm B2 1 || RC=1
arm A2 0 || RC=1

echo
echo "=== F3 step 2a perf: decode and TTFT, A(trunk=0),B(trunk=1),B,A ==="
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
    echo "  -- $kind $field: A(trunk=0)=[$A ] B(trunk=1)=[$B ]"
    gate_ab_verdict "$kind $field (A -> B)" "$A" "$B" || true
  done
done

echo
echo "=== gate: decode >= +20% conservative (this is a REPORT, not a landing decision) ==="
python3 -c "
import json
def load(a):
    try: return [json.loads(l) for l in open('$OUT/${TAG}_%s.jsonl'%a) if l.strip()]
    except FileNotFoundError: return []
for kind in ('ladder','followup'):
    def dec(rows):
        for r in rows:
            if r.get('kind')==kind and r.get('decode_tps') is not None:
                return float(r['decode_tps'])
        return None
    a1,a2 = dec(load('A1')), dec(load('A2'))
    b1,b2 = dec(load('B1')), dec(load('B2'))
    if None in (a1,a2,b1,b2):
        print('  %-9s missing a sample, cannot judge the +20%% bar' % kind); continue
    a_lo, b_hi = min(a1,a2), max(b1,b2)   # conservative: worst A vs best B
    a_hi, b_lo = max(a1,a2), min(b1,b2)   # and the other way, for the honest range
    cons = (b_lo - a_hi) / a_hi * 100.0   # worst-case B vs best-case A
    opt  = (b_hi - a_lo) / a_lo * 100.0
    met = 'MET' if cons >= 20.0 else 'NOT MET'
    print('  %-9s conservative %+.1f%% (worst B vs best A), optimistic %+.1f%% -- +20%% bar: %s'
          % (kind, cons, opt, met))
"

echo
echo "=== fifth run: CLI-mode profile of trunk=1, COLI_TIMERS=1 ==="
echo "    (separate from the ladder -- see this file's header for why)"
PACKET="$HERE/x2_packet_450.txt"
PROFILE_OUT="$OUT/${TAG}_profile.txt"
if [ -s "$PACKET" ]; then
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$I8" N_NEW=128 NOSTREAM=1 COLI_TIMERS=1
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto Q36_VK_TRUNK=1
    /usr/bin/time -v "$BIN" "$CAP" 8 "$PACKET" ) \
      > "$OUT/${TAG}_profile.out" 2> "$OUT/${TAG}_profile.err"
  {
    echo "# F3 step 2a fifth run: per-bucket profile, trunk=1 (lm_head + DeltaNet"
    echo "# projections on dev0), int8 tier on dev2+dev3"
    echo "# built from tree $GITSHA, binary $BIN"
    echo "# packet prompt, N_NEW=128, COLI_TIMERS=1, CLI mode (not coli serve -- see"
    echo "# this file's header for why serve mode never prints this)"
    echo
    grep -E '^\[timers\]|^\[qtier-vk\]|^\[trunk-vk\]|^TTFT:|^PEAK RSS:|^Speed:' "$OUT/${TAG}_profile.err"
  } > "$PROFILE_OUT"
  echo "  profile written to $PROFILE_OUT:"
  sed 's/^/    /' "$PROFILE_OUT"
  if ! grep -q '^\[timers\]' "$PROFILE_OUT"; then
    echo "  FAIL: no [timers] block -- the profile run did not produce what this item needs"
    RC=1
  fi
  echo "  --- sanity: lm_head (bucket 5) and DeltaNet proj (dn-sub 0) must have DROPPED"
  echo "      from record §F3-STEP1's CPU numbers (lm_head 6.3 ms, proj 10.8 ms of 19.1"
  echo "      DeltaNet ms/token) if the GPU round trip is cheaper than the CPU matmul it"
  echo "      replaced -- read the printed bucket lines above against those, do not just"
  echo "      check they are present"
else
  echo "  FATAL: $PACKET missing -- no profile taken"; RC=1
fi

echo
if [ "$RC" -eq 0 ]; then echo "=== F3 step 2a perf chain: completed (tree $GITSHA)"
else echo "=== F3 step 2a perf chain: had a FAILING arm or leg (tree $GITSHA)"; fi

echo
echo "[f3s2aperf] restarting the gateway (SKIP_WARM=1 is enough -- CLAUDE.md: do not"
echo "            re-warm GLM's 184 GB by cat before restarting the gateway)"
start_gateway
RESTART_GATEWAY=0
echo "[f3s2aperf] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[f3s2aperf] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
