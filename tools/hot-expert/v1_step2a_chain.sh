#!/bin/bash
# v1_step2a_chain.sh -- FRANKEN-ENGINE-PLAN §2.6, item V1, STEP 2, part A:
# turn the Vulkan expert tier ON and ask whether it computes the same model.
#
#   setsid nohup ~/src/colibri-v1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-v1/tools/hot-expert/v1_step2a_chain.sh \
#       >> ~/bench/v1_step2a.log 2>&1 < /dev/null &
#
# ONE binary (`qwen36-vk`), ONE knob (`Q36_VULKAN`), four runs: tier off and
# tier on, on the ladder packet and on a short prompt. The tier holds every
# expert of the 40 x 256 set on dev3 (PCI 0000:86:00.0, Vulkan enumeration
# index 2 -- record §Q13), budgeted with glm53's G6 rule and its 1.0 GB
# reserve.
#
# WHAT THIS CHAIN DOES *NOT* DO: it does not compute the KL. kl_compare.py is
# pure Python over 625 x 248 320 floats per side and takes ~15 minutes a pair
# -- that is fifteen minutes of the owner's gateway held down for arithmetic
# that needs no rig state at all. The chain writes the dumps, restarts the
# gateway, runs accept_live.sh and RELEASES THE LOCK; the KL is computed
# afterwards, outside the lock, by v1_step2a_kl.sh over the same files.
#
# The refusals that matter here:
#   - the tier must SAY what it holds. `[qtier-vk] resident N/10240` must read
#     10240 and the VRAM hit rate must be >= 99%, or the tier-on arm is
#     measuring the CPU path with extra steps and the comparison is vacuous.
#     This is the "tier line asserts the residency it claims" gate leg, and it
#     is the one that catches a silent fallback -- which would otherwise show
#     up as a *perfect* KL of 0 and read as a pass.
#   - both dumps must exist, be the same length, and differ from each other's
#     absence (kl_compare's own REFUSED path).
set -u
TAG=v1s2a$(date +%m%d%H%M)
OUT=~/bench/v1_step2a_out; mkdir -p "$OUT"
SRC=~/src/colibri-v1
HERE="$SRC/tools/hot-expert"
M=${V1_MODEL:-/home/ronald/models/qwen36_i4_gs64}
CAP=${V1_CAP:-256}
DEV3=${V1_DEV3:-2}                 # Vulkan enumeration index of PCI 0000:86:00.0
RESERVE=${V1_RESERVE_GB:-1.0}
LOG=~/glm53_server.log
KEY=$(cat ~/.colibri_api_key 2>/dev/null)
RESTART_GATEWAY=0

start_gateway() {
  SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  local code=""
  for _ in $(seq 1 90); do
    sleep 10
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
             http://127.0.0.1:8081/v1/models 2>/dev/null) || true
    [ "${code:-}" = 200 ] && break
  done
  echo "[v1s2a] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[v1s2a] FATAL: a $n is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[v1s2a] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine glm53
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x qwen36-vk 2>/dev/null || true
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
  echo "=== v1_step2a_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== dumps: $OUT/  -- now run v1_step2a_kl.sh OUTSIDE the lock"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== v1_step2a_chain $TAG $(date -Is)"
for e in qwen38 qwen38-vk qwen36 qwen36-vk; do
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[v1s2a] $e is running -- refusing"; exit 1; }
done
stop_gateway || exit 1

echo "[v1s2a] refreshing $SRC"; git -C "$SRC" pull --ff-only || exit 1
git -C "$SRC" log -1 --oneline
GITSHA=$(git -C "$SRC" rev-parse --short HEAD)

echo "--- VRAM after the gateway stopped (the tier needs dev3 empty)"
for c in /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -r "$c" ] || continue
  used=$(cat "$c"); echo "  $c = $((used/1048576)) MB"
  [ "$used" -le 1073741824 ] || { echo "[v1s2a] REFUSED: $c holds $((used/1048576)) MB"; exit 1; }
done

echo "--- building qwen36-vk"
( cd "$SRC/c" && make qwen36-vk VK=1 ) > "$OUT/${TAG}_build.log" 2>&1 \
  || { echo "[v1s2a] build FAILED"; tail -30 "$OUT/${TAG}_build.log"; exit 1; }
cp -f "$SRC/c/qwen36-vk" ~/bench/qwen36-vk.v1
BIN=~/bench/qwen36-vk.v1
echo "[v1s2a] qwen36-vk.v1 sha256=$(sha256sum $BIN | cut -c1-16)"

echo "--- warming the qwen36 container ($M)"
find "$M" -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true
PCT=$(fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors | python3 -c "
import sys
PAGE=4096; tot=res=0
for line in sys.stdin.read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    size,r=int(f[-2]),int(f[-1]); tot+=-(-size//PAGE)*PAGE; res+=r
print(f'{res/tot*100 if tot else 0:.1f}')
")
echo "[v1s2a] container resident: $PCT%"
python3 -c "import sys; sys.exit(0 if float('$PCT')>=90.0 else 1)" \
  || { echo "[v1s2a] REFUSED: residency $PCT% < 90%"; exit 1; }

PACKET="$HERE/x2_packet_450.txt"
SHORT=/tmp/v1_short.txt
printf '%s' 'What comes after Tuesday, and why is the answer the same in every calendar that has a seven-day week?' > "$SHORT"
[ -s "$PACKET" ] || { echo "[v1s2a] FATAL: $PACKET missing"; exit 1; }

# run <tag> <on|off> <promptfile> <n_new>
run() {
  local tag="$1" tier="$2" pf="$3" nnew="$4"
  echo "[v1s2a] $(date +%H:%M:%S) run $tag (tier $tier)"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$M" N_NEW="$nnew" NOSTREAM=1
    export Q36_TEACHER_FORCING=1
    export Q36_LOGIT_DUMP_ALL="$OUT/$tag.dumpall"
    export DUMP="$OUT/$tag.last"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export COLI_VK_DEV3="$DEV3" COLI_VK_TIER_RESERVE_GB="$RESERVE"
    # HEAT_FILE deliberately unset: qwen36 has no routing history on this box,
    # so the fill is natural order and identical between runs by construction.
    if [ "$tier" = on ]; then export Q36_VULKAN=1; fi
    /usr/bin/time -v "$BIN" "$CAP" 4 "$pf" ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  grep -v '^teacher_forcing' "$OUT/$tag.out" > "$OUT/$tag.text"
  local pos maj ttft rss
  pos=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  maj=$(awk -F': ' '/Major .*page faults/{print $2}' "$OUT/$tag.err")
  ttft=$(awk '/^TTFT:/{print $2}' "$OUT/$tag.err")
  rss=$(awk -F': ' '/Maximum resident set size/{printf "%.1f", $2/1048576}' "$OUT/$tag.err")
  echo "[v1s2a] $(date +%H:%M:%S) run $tag rc=$rc positions=${pos:-0} majflt=${maj:-?}" \
       "ttft=${ttft:-?}s peakRSS=${rss:-?}GB dumpall=$(wc -c < "$OUT/$tag.dumpall" 2>/dev/null || echo 0)B"
  grep -E '^\[qtier-vk\]|^\[gpu\]' "$OUT/$tag.err" | sed 's/^/        /'
  return $rc
}

run off_sh off "$SHORT"  16
run on_sh  on  "$SHORT"  16
run off_pk off "$PACKET" 8
run on_pk  on  "$PACKET" 8

echo
echo "=== V1 step 2a: the tier must say what it holds ==="
RC=0
NEXP=$(python3 -c "print(40*256)")
for t in on_sh on_pk; do
  line=$(grep -E '^\[qtier-vk\] resident' "$OUT/$t.err" | tail -1)
  hit=$(grep -E 'VRAM hit rate' "$OUT/$t.err" | tail -1 | sed -E 's/.*rate: *([0-9.]+).*/\1/')
  res=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\1/')
  tot=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\2/')
  if [ -z "$line" ]; then
    printf '  %-22s REFUSED: no [qtier-vk] line at all -- the tier never came up\n' "$t"; RC=1; continue
  fi
  printf '  %-22s resident %s/%s  hit rate %s%%\n' "$t" "${res:-?}" "${tot:-?}" "${hit:-?}"
  [ "${res:-0}" = "${NEXP}" ] || { echo "     FAIL: expected all $NEXP experts resident"; RC=1; }
  python3 -c "import sys; sys.exit(0 if float('${hit:-0}')>=99.0 else 1)" \
    || { echo "     FAIL: VRAM hit rate ${hit:-?}% < 99% -- the tier-on arm is largely CPU"; RC=1; }
  grep -E '^\[qtier-vk\] dev3 planned' "$OUT/$t.err" | tail -1 | sed 's/^/     /'
done
for t in off_sh off_pk; do
  if grep -qE '^\[qtier-vk\]' "$OUT/$t.err"; then
    printf '  %-22s FAIL: the tier spoke in a tier-OFF run\n' "$t"; RC=1
  else
    printf '  %-22s silent, as a tier-off run must be\n' "$t"
  fi
done

echo
echo "=== greedy text, tier-on vs tier-off ==="
. "$HERE/gate_lib.sh"
for p in sh pk; do
  a="$OUT/off_$p.text"; b="$OUT/on_$p.text"
  na=$(tr -d '[:space:]' < "$a" 2>/dev/null | wc -c); nb=$(tr -d '[:space:]' < "$b" 2>/dev/null | wc -c)
  if [ "${na:-0}" -lt 8 ] || [ "${nb:-0}" -lt 8 ]; then
    printf '  %-22s REFUSED: %s / %s non-space chars\n' "$p greedy text" "$na" "$nb"; RC=1
  elif cmp -s "$a" "$b"; then
    printf '  %-22s IDENTICAL (%s non-space chars)\n' "$p greedy text" "$na"
  else
    printf '  %-22s DIFFERS -- reported, not failed; the KL is the scale\n' "$p greedy text"
    echo "     off: $(head -c 160 "$a" | tr '\n' ' ')"
    echo "     on : $(head -c 160 "$b" | tr '\n' ' ')"
  fi
done

echo
echo "=== teacher_forcing argmax agreement (the cheap half of the KL oracle) ==="
for p in sh pk; do
  gate_compare "$p teacher_forcing" "$OUT/off_$p.out" "$OUT/on_$p.out" "^teacher_forcing" \
    || python3 - "$OUT/off_$p.out" "$OUT/on_$p.out" <<'PY'
import sys
def tf(p):
    for line in open(p):
        if line.startswith("teacher_forcing"):
            return line.split()[1:]
    return []
a, b = tf(sys.argv[1]), tf(sys.argv[2])
if not a or not b or len(a) != len(b):
    print("     REFUSED: %d vs %d positions" % (len(a), len(b))); sys.exit(0)
d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
print("     %d of %d positions differ (%.2f%% agree)%s"
      % (len(d), len(a), 100.0*(len(a)-len(d))/len(a),
         "; first at %d" % d[0] if d else ""))
PY
done

echo
echo "=== dumps ready for the KL (run OUTSIDE the lock) ==="
for f in off_sh on_sh off_pk on_pk; do
  printf '  %-10s %s bytes\n' "$f" "$(wc -c < "$OUT/$f.dumpall" 2>/dev/null || echo 0)"
done
echo "  next:  ~/src/colibri-v1/tools/hot-expert/v1_step2a_kl.sh"

echo
if [ "$RC" -eq 0 ]; then echo "=== V1 step 2a residency/plumbing: PASS (tree $GITSHA)"
else echo "=== V1 step 2a residency/plumbing: FAIL (tree $GITSHA)"; fi

echo
echo "[v1s2a] re-warming the GLM container before the gateway returns"
GLM=${V1_GLM_MODEL:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
cat "$GLM"/*.safetensors > /dev/null 2>&1 || true
start_gateway
RESTART_GATEWAY=0
echo "[v1s2a] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[v1s2a] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
