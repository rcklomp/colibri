#!/bin/bash
# f3s1_kl_chain.sh -- FRANKEN-ENGINE-PLAN item F3, step 1: correctness of the
# Vulkan expert tier ported from V1's ONE fixed device (dev3) to a per-device
# array over dev2+dev3, and of its new acceptance of a row-wise int8
# container (fmt=1) -- which V1 refused outright.
#
#   setsid nohup ~/src/colibri-f3s1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f3s1/tools/hot-expert/f3s1_kl_chain.sh \
#       >> ~/bench/f3s1_kl.log 2>&1 < /dev/null &
#
# Pattern: v1_step2a_chain.sh's tier-on/tier-off correctness shape, plus
# f3s0_kl_chain.sh's two-container warming (two `cat` passes -- one pass left
# a 35 GB container at 84.9% resident, run f3s0kl09201409, because a page
# read once sits on the inactive list and is reclaimed ahead of GLM's cache).
#
# TWO LEGS, both on ONE binary (`qwen36-vk` built from perf/f3-step1):
#
#   Leg 1 (the port's own oracle): the int8 container
#   (~/models/qwen36_i8_row), tier OFF (CPU) vs tier ON over dev2+dev3
#   (COLI_VK_DEV2=auto COLI_VK_DEV3=auto, glm53's own convention). Same
#   reasoning as V1's tier-on/off KL: this is a REASSOCIATION of the same
#   int8 products (row-wise scales applied per row on both sides, summed in
#   a different order), so the bar is V1's own floor, not the loose
#   "different quantization" bar f3s0_kl.sh used for int4-vs-int8. Residency
#   must read 10240/10240 with CPU misses 0, or the tier-on arm is quietly
#   running the CPU path and the KL would read as a false pass.
#
#   Leg 2 (default behaviour unchanged): the int4-gs64 container
#   (~/models/qwen36_i4_gs64), COLI_VK_DEV3=2 only (V1's own device pin,
#   COLI_VK_DEV2 unset) -- run FRESH here on both this branch's binary and
#   the pristine ~/bench/qwen36-vk.v1, same env, same prompt. `cmp` on the
#   two dumps must be IDENTICAL: with only COLI_VK_DEV3 set, G.ndev==1 and
#   every arithmetic path below is V1's own, unchanged (see qwen36_tier_vk.c's
#   file header). Bit-identical, not a KL bar -- if this leg is not
#   byte-identical, F3 step 1 changed V1's default numerics and that is a
#   defect to explain before anything downstream is trusted.
#
# WHAT THIS CHAIN DOES NOT DO: it does not compute leg 1's per-position KL.
# kl_compare.py over 625 x vocab floats is pure Python and takes minutes a
# pair -- gateway-downtime time held down for arithmetic that needs no rig
# state. The chain writes the dumps, restarts the gateway, runs
# accept_live.sh and RELEASES THE LOCK; f3s1_kl.sh computes the KL
# afterwards, outside the lock, over the same files.
set -u
TAG=f3s1kl$(date +%m%d%H%M)
OUT=~/bench/f3s1_kl_out; mkdir -p "$OUT"
SRC=~/src/colibri-f3s1
HERE="$SRC/tools/hot-expert"
I4=${F3S1_I4:-/home/ronald/models/qwen36_i4_gs64}
I8=${F3S1_I8:-/home/ronald/models/qwen36_i8_row}
CAP=${F3S1_CAP:-256}                # cache/layer: every expert resident, as V1
DEV3_PIN=${F3S1_DEV3_PIN:-2}         # V1's own device pin (PCI 0000:86:00.0)
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
  echo "[f3s1kl] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[f3s1kl] FATAL: a $n is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[f3s1kl] stopping the owner's gateway for the duration of this run"
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
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== f3s1_kl_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== dumps: $OUT/  -- now run f3s1_kl.sh OUTSIDE the lock"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== f3s1_kl_chain $TAG $(date -Is)"
for e in qwen38 qwen38-vk qwen36 qwen36-vk; do
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[f3s1kl] $e is running -- refusing"; exit 1; }
done

echo "--- int8 container completeness: $I8"
if [ ! -f "$I8/qwen36_meta.json" ]; then
  echo "[f3s1kl] REFUSED: $I8/qwen36_meta.json missing -- the int8 conversion may be incomplete"
  exit 1
fi
python3 -c "
import json
m = json.load(open('$I8/qwen36_meta.json'))
assert m.get('ebits') == 8, 'meta.ebits=%r, expected 8' % m.get('ebits')
assert not m.get('expert_gs'), 'meta.expert_gs=%r, expected 0/absent (row-wise)' % m.get('expert_gs')
print('[f3s1kl] i8 meta OK: ebits=8 expert_gs=%r num_experts=%s topk=%s' %
      (m.get('expert_gs'), m.get('num_experts'), m.get('topk')))
" || { echo "[f3s1kl] REFUSED: $I8/qwen36_meta.json failed the ebits=8/per-row check"; exit 1; }
[ -x ~/bench/qwen36-vk.v1 ] || { echo "[f3s1kl] REFUSED: ~/bench/qwen36-vk.v1 (V1's pristine binary) missing"; exit 1; }

stop_gateway || exit 1

echo "--- VRAM after the gateway stopped (both tier devices need to be empty)"
for c in /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -r "$c" ] || continue
  used=$(cat "$c"); echo "  $c = $((used/1048576)) MB"
  [ "$used" -le 1073741824 ] || { echo "[f3s1kl] REFUSED: $c holds $((used/1048576)) MB"; exit 1; }
done

echo "[f3s1kl] tree at $SRC: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITSHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[f3s1kl] NOTE: $SRC is a detached checkout of a branch relayed from the Mac"
echo "         (perf/f3-step1 -> rome:f3s1-sync -> git checkout --detach), not a tracking"
echo "         branch -- there is nothing to 'git pull' here, the sha above is what runs"

echo "--- building qwen36-vk"
( cd "$SRC/c" && make qwen36-vk VK=1 ) > "$OUT/${TAG}_build.log" 2>&1 \
  || { echo "[f3s1kl] build FAILED"; tail -30 "$OUT/${TAG}_build.log"; exit 1; }
cp -f "$SRC/c/qwen36-vk" ~/bench/qwen36-vk.f3s1
BIN=~/bench/qwen36-vk.f3s1
echo "[f3s1kl] qwen36-vk.f3s1 sha256=$(sha256sum "$BIN" | cut -c1-16)"
echo "[f3s1kl] qwen36-vk.v1   sha256=$(sha256sum ~/bench/qwen36-vk.v1 | cut -c1-16) (pristine)"

resident_pct() {
  local dir="$1"
  # Two passes: a page read once sits on the INACTIVE list and is reclaimed
  # ahead of GLM's long-lived cache (f3s0_kl_chain.sh's own finding,
  # run f3s0kl09201409: 84.9% after one pass, refused).
  for _ in 1 2; do
    find "$dir" -maxdepth 1 -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true
  done
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

echo "--- warming $I4"
PCT4=$(resident_pct "$I4")
echo "[f3s1kl] i4 container resident: $PCT4%"
python3 -c "import sys; sys.exit(0 if float('$PCT4')>=90.0 else 1)" \
  || { echo "[f3s1kl] REFUSED: i4 residency $PCT4% < 90%"; exit 1; }

echo "--- warming $I8"
PCT8=$(resident_pct "$I8")
echo "[f3s1kl] i8 container resident: $PCT8%"
python3 -c "import sys; sys.exit(0 if float('$PCT8')>=90.0 else 1)" \
  || { echo "[f3s1kl] REFUSED: i8 residency $PCT8% < 90%"; exit 1; }

PACKET="$HERE/x2_packet_450.txt"
SHORT=/tmp/f3s1_short.txt
printf '%s' 'What comes after Tuesday, and why is the answer the same in every calendar that has a seven-day week?' > "$SHORT"
[ -s "$PACKET" ] || { echo "[f3s1kl] FATAL: $PACKET missing"; exit 1; }

# run <tag> <binary> <container> <tier=off|on2|on23> <promptfile> <n_new>
#   off   -> Q36_VULKAN unset (CPU path)
#   on23  -> Q36_VULKAN=1, COLI_VK_DEV2=auto COLI_VK_DEV3=auto (F3 step 1)
#   on3   -> Q36_VULKAN=1, COLI_VK_DEV3=<pin>, COLI_VK_DEV2 unset (V1's shape)
run() {
  local tag="$1" bin="$2" snap="$3" tier="$4" pf="$5" nnew="$6"
  echo "[f3s1kl] $(date +%H:%M:%S) run $tag (bin=$(basename "$bin") tier=$tier snap=$(basename "$snap"))"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$snap" N_NEW="$nnew" NOSTREAM=1
    export Q36_TEACHER_FORCING=1
    export Q36_LOGIT_DUMP_ALL="$OUT/$tag.dumpall"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    case "$tier" in
      on23) export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto ;;
      on3)  export Q36_VULKAN=1 COLI_VK_DEV3="$DEV3_PIN"; unset COLI_VK_DEV2 ;;
      off)  unset Q36_VULKAN COLI_VK_DEV2 COLI_VK_DEV3 ;;
    esac
    local ebits=4; [ "$snap" = "$I8" ] && ebits=8
    /usr/bin/time -v "$bin" "$CAP" "$ebits" "$pf" ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  grep -v '^teacher_forcing' "$OUT/$tag.out" > "$OUT/$tag.text"
  local pos maj rss
  pos=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  maj=$(awk -F': ' '/Major .*page faults/{print $2}' "$OUT/$tag.err")
  rss=$(awk -F': ' '/Maximum resident set size/{printf "%.1f", $2/1048576}' "$OUT/$tag.err")
  echo "[f3s1kl] $(date +%H:%M:%S) run $tag rc=$rc positions=${pos:-0} majflt=${maj:-?}" \
       "peakRSS=${rss:-?}GB dumpall=$(wc -c < "$OUT/$tag.dumpall" 2>/dev/null || echo 0)B"
  grep -E '^\[qtier-vk\]|^\[qwen36\]' "$OUT/$tag.err" | sed 's/^/        /'
  return $rc
}

echo
echo "=== leg 1: int8 container, tier OFF (CPU) vs tier ON over dev2+dev3 ==="
run i8_off_sh "$BIN" "$I8" off  "$SHORT"  16
run i8_on_sh  "$BIN" "$I8" on23 "$SHORT"  16
run i8_off_pk "$BIN" "$I8" off  "$PACKET" 8
run i8_on_pk  "$BIN" "$I8" on23 "$PACKET" 8

echo
echo "=== leg 1: residency must read 10240/10240 with CPU misses 0 (both tier-on runs) ==="
RC=0
for t in i8_on_sh i8_on_pk; do
  line=$(grep -E '^\[qtier-vk\] resident' "$OUT/$t.err" | tail -1)
  hitline=$(grep -E 'VRAM hit rate' "$OUT/$t.err" | tail -1)
  if [ -z "$line" ]; then
    printf '  %-12s REFUSED: no [qtier-vk] resident line at all -- the tier never came up\n' "$t"
    RC=1; continue
  fi
  res=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\1/')
  tot=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\2/')
  miss=$(echo "$hitline" | sed -E 's/.*CPU misses ([0-9]+).*/\1/')
  printf '  %-12s resident %s/%s  CPU misses %s\n' "$t" "${res:-?}" "${tot:-?}" "${miss:-?}"
  [ "${res:-0}" = "${tot:-1}" ] && [ "${res:-0}" = "10240" ] \
    || { echo "     FAIL: expected 10240/10240 resident"; RC=1; }
  [ "${miss:-1}" = "0" ] || { echo "     FAIL: CPU misses ${miss:-?} != 0 -- the tier-on arm is partly on the CPU"; RC=1; }
  grep -E '^\[qtier-vk\] dev2: resident|^\[qtier-vk\] dev3: resident' "$OUT/$t.err" | sed 's/^/     /'
done
for t in i8_off_sh i8_off_pk; do
  if grep -qE '^\[qtier-vk\]' "$OUT/$t.err"; then
    printf '  %-12s FAIL: the tier spoke in a tier-OFF run\n' "$t"; RC=1
  else
    printf '  %-12s silent, as a tier-off run must be\n' "$t"
  fi
done

echo
echo "=== leg 1: greedy text and teacher_forcing, tier-on vs tier-off (int8) ==="
. "$HERE/gate_lib.sh"
for p in sh pk; do
  a="$OUT/i8_off_$p.text"; b="$OUT/i8_on_$p.text"
  na=$(tr -d '[:space:]' < "$a" 2>/dev/null | wc -c); nb=$(tr -d '[:space:]' < "$b" 2>/dev/null | wc -c)
  if [ "${na:-0}" -lt 8 ] || [ "${nb:-0}" -lt 8 ]; then
    printf '  %-30s REFUSED: %s / %s non-space chars\n' "$p greedy text" "$na" "$nb"; RC=1
  elif cmp -s "$a" "$b"; then
    printf '  %-30s IDENTICAL (%s non-space chars)\n' "$p greedy text" "$na"
  else
    printf '  %-30s DIFFERS -- reported, the KL (outside the lock) is the scale\n' "$p greedy text"
  fi
  gate_compare "$p teacher_forcing (int8 on/off)" "$OUT/i8_off_$p.out" "$OUT/i8_on_$p.out" "^teacher_forcing" || RC=1
done

echo
echo "=== leg 2: default behaviour unchanged -- int4, COLI_VK_DEV3=$DEV3_PIN only,"
echo "===        this branch's binary vs the pristine V1 binary, bit-identical dumps ==="
run reg_new "$BIN" "$I4" on3 "$SHORT" 16
run reg_v1  ~/bench/qwen36-vk.v1 "$I4" on3 "$SHORT" 16
if [ ! -s "$OUT/reg_new.dumpall" ] || [ ! -s "$OUT/reg_v1.dumpall" ]; then
  echo "  REFUSED: one of the regression dumps is empty"; RC=1
elif cmp -s "$OUT/reg_new.dumpall" "$OUT/reg_v1.dumpall"; then
  echo "  IDENTICAL: $(wc -c < "$OUT/reg_new.dumpall") bytes -- the single-device (dev3-only)"
  echo "  path is bit-exact with V1, as the file header claims"
else
  echo "  DIFFERS -- default behaviour changed. cmp -l (first mismatches):"
  cmp -l "$OUT/reg_new.dumpall" "$OUT/reg_v1.dumpall" 2>/dev/null | head -5 | sed 's/^/    /'
  RC=1
fi
gate_compare "reg teacher_forcing (new vs V1, dev3-only)" "$OUT/reg_new.out" "$OUT/reg_v1.out" "^teacher_forcing" || RC=1
a="$OUT/reg_new.text"; b="$OUT/reg_v1.text"
if cmp -s "$a" "$b"; then
  echo "  greedy text IDENTICAL ($(tr -d '[:space:]' < "$a" | wc -c) non-space chars)"
else
  echo "  greedy text DIFFERS"; RC=1
fi

echo
echo "=== dumps ready for leg 1's KL (run OUTSIDE the lock) ==="
for f in i8_off_sh i8_on_sh i8_off_pk i8_on_pk; do
  printf '  %-10s %s bytes\n' "$f" "$(wc -c < "$OUT/$f.dumpall" 2>/dev/null || echo 0)"
done
echo "  next:  ~/src/colibri-f3s1/tools/hot-expert/f3s1_kl.sh $OUT"

echo
if [ "$RC" -eq 0 ]; then echo "=== f3s1 KL chain plumbing: PASS (tree $GITSHA)"
else echo "=== f3s1 KL chain plumbing: FAIL (tree $GITSHA)"; fi

echo
echo "[f3s1kl] restarting the gateway (SKIP_WARM=1 is enough -- CLAUDE.md: do not"
echo "         re-warm GLM's 184 GB by cat before restarting the gateway)"
start_gateway
RESTART_GATEWAY=0
echo "[f3s1kl] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[f3s1kl] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
