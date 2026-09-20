#!/bin/bash
# f3s2a_kl_chain.sh -- FRANKEN-ENGINE-PLAN item F3, step 2a: correctness of
# porting the CUDA tier's trunk placement (qt_lmhead_init/matmul,
# qt_dnproj_init/matmul, qt_place_of) onto this backend's dev0 (the
# unsuffixed coli_vk_matmul/coli_vk_tensor_ensure glm53 already drives its
# own dense matrices and lm_head through), behind Q36_VK_TRUNK -- moving
# lm_head and every DeltaNet layer's fused qkv++z input projection off the
# CPU, dev2/dev3's expert tier untouched.
#
#   setsid nohup ~/src/colibri-f3s2a/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f3s2a/tools/hot-expert/f3s2a_kl_chain.sh \
#       >> ~/bench/f3s2a_kl.log 2>&1 < /dev/null &
#
# Pattern: f3s1_kl_chain.sh's two-leg shape (one binary, both legs run here,
# the packet-sized KL computed afterwards outside the lock by f3s2a_kl.sh).
# int8 container in EVERY arm (~/models/qwen36_i8_row), tier on dev2+dev3 in
# EVERY arm (COLI_VK_DEV2=auto COLI_VK_DEV3=auto) -- step 2a only moves the
# trunk, it does not touch the expert tier this item builds on.
#
# THREE LEGS, one binary (`qwen36-vk` built from perf/f3-step2a):
#
#   Leg 1 (default behaviour unchanged): Q36_VK_TRUNK unset on the new binary
#   vs ~/bench/qwen36-vk.f3s1 (step 1's own pristine, no knob exists on that
#   binary at all), same env otherwise (int8, dev2+dev3, short prompt).
#   `cmp` on the dumps must be IDENTICAL: with the knob off, qt_place_of
#   returns QT_PLACE_CPU exactly as the pre-2a stub did and qt_lmhead_init/
#   qt_dnproj_init are no-ops, so nothing downstream can differ.
#
#   Leg 2 (the port's own oracle): the SAME binary, Q36_VK_TRUNK=0 vs
#   Q36_VK_TRUNK=1, int8, dev2+dev3, short prompt AND the 625-position ladder
#   packet. This REASSOCIATES the same row-wise int8 dot products qwen36.c's
#   dense-i8 quantization already computes for the CPU path (AVX2 lanes vs
#   the GPU's subgroup reduction) -- the same argument F3 step 1 made for its
#   own tier-on/off pair, so the bar is that pair's own floor (record
#   §F3-STEP1: mean KL ~1e-10, top-1 100%), not the looser "different
#   quantization" bar. Residency must still read 10240/10240 with CPU misses
#   0 in both trunk arms -- step 2a must not perturb the expert tier this
#   sits on top of.
#
#   Leg 3: dev0 VRAM. Nothing else of qwen36 lives on dev0 (it is not part of
#   the expert pool, dev2/dev3 only) -- the chain reads the "[trunk-vk] ..."
#   startup lines (lm_head resident, N DeltaNet layers placed) plus a raw
#   sysfs mem_info_vram_used snapshot after the trunk-on run into the log for
#   a human to read (CLAUDE.md: amd-smi/nvtop number the three cards
#   differently and this chain does not resolve which physical card is dev0;
#   the [trunk-vk] lines are the authoritative source for what THIS run put
#   there).
#
# WHAT THIS CHAIN DOES NOT DO: it does not compute leg 2's per-position KL
# (kl_compare.py over 625 x vocab floats is pure Python, minutes/pair --
# gateway-downtime time held down for arithmetic that needs no rig state).
# The chain writes the dumps, restarts the gateway, runs accept_live.sh and
# RELEASES THE LOCK; f3s2a_kl.sh computes the KL afterwards, outside the
# lock, over the same files.
set -u
TAG=f3s2akl$(date +%m%d%H%M)
OUT=~/bench/f3s2a_kl_out; mkdir -p "$OUT"
SRC=~/src/colibri-f3s2a
HERE="$SRC/tools/hot-expert"
I8=${F3S2A_I8:-/home/ronald/models/qwen36_i8_row}
CAP=${F3S2A_CAP:-256}                # cache/layer: every expert resident, as step 1
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
  echo "[f3s2akl] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[f3s2akl] FATAL: a $n is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[f3s2akl] stopping the owner's gateway for the duration of this run"
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
  echo "=== f3s2a_kl_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== dumps: $OUT/  -- now run f3s2a_kl.sh OUTSIDE the lock"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== f3s2a_kl_chain $TAG $(date -Is)"
for e in qwen38 qwen38-vk qwen36 qwen36-vk; do
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[f3s2akl] $e is running -- refusing"; exit 1; }
done

echo "--- int8 container completeness: $I8"
if [ ! -f "$I8/qwen36_meta.json" ]; then
  echo "[f3s2akl] REFUSED: $I8/qwen36_meta.json missing -- the int8 conversion may be incomplete"
  exit 1
fi
python3 -c "
import json
m = json.load(open('$I8/qwen36_meta.json'))
assert m.get('ebits') == 8, 'meta.ebits=%r, expected 8' % m.get('ebits')
assert not m.get('expert_gs'), 'meta.expert_gs=%r, expected 0/absent (row-wise)' % m.get('expert_gs')
print('[f3s2akl] i8 meta OK: ebits=8 expert_gs=%r num_experts=%s topk=%s' %
      (m.get('expert_gs'), m.get('num_experts'), m.get('topk')))
" || { echo "[f3s2akl] REFUSED: $I8/qwen36_meta.json failed the ebits=8/per-row check"; exit 1; }
[ -x ~/bench/qwen36-vk.f3s1 ] || { echo "[f3s2akl] REFUSED: ~/bench/qwen36-vk.f3s1 (step 1's pristine binary) missing"; exit 1; }

stop_gateway || exit 1

echo "--- VRAM after the gateway stopped (all cards need to be empty)"
for c in /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -r "$c" ] || continue
  used=$(cat "$c"); echo "  $c = $((used/1048576)) MB"
  [ "$used" -le 1073741824 ] || { echo "[f3s2akl] REFUSED: $c holds $((used/1048576)) MB"; exit 1; }
done

echo "[f3s2akl] tree at $SRC: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITSHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "[f3s2akl] NOTE: $SRC is a detached checkout of a branch relayed from the Mac"
echo "          (perf/f3-step2a -> rome:f3s2a-sync -> git checkout --detach), not a"
echo "          tracking branch -- there is nothing to 'git pull' here"

echo "--- building qwen36-vk"
( cd "$SRC/c" && make qwen36-vk VK=1 ) > "$OUT/${TAG}_build.log" 2>&1 \
  || { echo "[f3s2akl] build FAILED"; tail -30 "$OUT/${TAG}_build.log"; exit 1; }
cp -f "$SRC/c/qwen36-vk" ~/bench/qwen36-vk.f3s2a
BIN=~/bench/qwen36-vk.f3s2a
echo "[f3s2akl] qwen36-vk.f3s2a sha256=$(sha256sum "$BIN" | cut -c1-16)"
echo "[f3s2akl] qwen36-vk.f3s1  sha256=$(sha256sum ~/bench/qwen36-vk.f3s1 | cut -c1-16) (step 1 pristine)"

resident_pct() {
  local dir="$1"
  # Two passes: a page read once sits on the INACTIVE list and is reclaimed
  # ahead of GLM's long-lived cache (f3s0_kl_chain.sh's own finding).
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

echo "--- warming $I8"
PCT8=$(resident_pct "$I8")
echo "[f3s2akl] i8 container resident: $PCT8%"
python3 -c "import sys; sys.exit(0 if float('$PCT8')>=90.0 else 1)" \
  || { echo "[f3s2akl] REFUSED: i8 residency $PCT8% < 90%"; exit 1; }

PACKET="$HERE/x2_packet_450.txt"
SHORT=/tmp/f3s2a_short.txt
printf '%s' 'What comes after Tuesday, and why is the answer the same in every calendar that has a seven-day week?' > "$SHORT"
[ -s "$PACKET" ] || { echo "[f3s2akl] FATAL: $PACKET missing"; exit 1; }

# run <tag> <binary> <trunk=0|1> <promptfile> <n_new>
# Tier is ALWAYS int8 over dev2+dev3 (COLI_VK_DEV2=auto COLI_VK_DEV3=auto).
run() {
  local tag="$1" bin="$2" trunk="$3" pf="$4" nnew="$5"
  echo "[f3s2akl] $(date +%H:%M:%S) run $tag (bin=$(basename "$bin") trunk=$trunk pf=$(basename "$pf"))"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$I8" N_NEW="$nnew" NOSTREAM=1
    export Q36_TEACHER_FORCING=1
    export Q36_LOGIT_DUMP_ALL="$OUT/$tag.dumpall"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export Q36_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export Q36_VK_TRUNK="$trunk"
    /usr/bin/time -v "$bin" "$CAP" 8 "$pf" ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  grep -v '^teacher_forcing' "$OUT/$tag.out" > "$OUT/$tag.text"
  local pos maj rss
  pos=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  maj=$(awk -F': ' '/Major .*page faults/{print $2}' "$OUT/$tag.err")
  rss=$(awk -F': ' '/Maximum resident set size/{printf "%.1f", $2/1048576}' "$OUT/$tag.err")
  echo "[f3s2akl] $(date +%H:%M:%S) run $tag rc=$rc positions=${pos:-0} majflt=${maj:-?}" \
       "peakRSS=${rss:-?}GB dumpall=$(wc -c < "$OUT/$tag.dumpall" 2>/dev/null || echo 0)B"
  grep -E '^\[qtier-vk\]|^\[trunk-vk\]|^\[dnp\]|^\[lmh\]' "$OUT/$tag.err" | sed 's/^/        /'
  return $rc
}
# reg run: uses the PRISTINE f3s1 binary, which has no Q36_VK_TRUNK knob at
# all -- calling it with the env var set is harmless (getenv of an unused
# name), so the SAME run() helper works for both binaries.

echo
echo "=== leg 1: default behaviour unchanged -- knob OFF on the new binary vs step 1's pristine ==="
run reg_new "$BIN" 0 "$SHORT" 16
run reg_f3s1 ~/bench/qwen36-vk.f3s1 0 "$SHORT" 16
RC=0
if [ ! -s "$OUT/reg_new.dumpall" ] || [ ! -s "$OUT/reg_f3s1.dumpall" ]; then
  echo "  REFUSED: one of the regression dumps is empty"; RC=1
elif cmp -s "$OUT/reg_new.dumpall" "$OUT/reg_f3s1.dumpall"; then
  echo "  IDENTICAL: $(wc -c < "$OUT/reg_new.dumpall") bytes -- knob-off is bit-exact with step 1"
else
  echo "  DIFFERS -- default behaviour changed. cmp -l (first mismatches):"
  cmp -l "$OUT/reg_new.dumpall" "$OUT/reg_f3s1.dumpall" 2>/dev/null | head -5 | sed 's/^/    /'
  RC=1
fi
. "$HERE/gate_lib.sh"
gate_compare "reg teacher_forcing (new trunk=0 vs f3s1)" "$OUT/reg_new.out" "$OUT/reg_f3s1.out" "^teacher_forcing" || RC=1
if cmp -s "$OUT/reg_new.text" "$OUT/reg_f3s1.text"; then
  echo "  greedy text IDENTICAL ($(tr -d '[:space:]' < "$OUT/reg_new.text" | wc -c) non-space chars)"
else
  echo "  greedy text DIFFERS"; RC=1
fi

echo
echo "=== leg 2: the port's own oracle -- Q36_VK_TRUNK=0 vs =1, int8, dev2+dev3 ==="
run trunk_off_sh "$BIN" 0 "$SHORT"  16
run trunk_on_sh  "$BIN" 1 "$SHORT"  16
run trunk_off_pk "$BIN" 0 "$PACKET" 8
run trunk_on_pk  "$BIN" 1 "$PACKET" 8

echo
echo "=== leg 2: residency must still read 10240/10240 with CPU misses 0 (every run above) ==="
for t in reg_new reg_f3s1 trunk_off_sh trunk_on_sh trunk_off_pk trunk_on_pk; do
  line=$(grep -E '^\[qtier-vk\] resident' "$OUT/$t.err" | tail -1)
  hitline=$(grep -E 'VRAM hit rate' "$OUT/$t.err" | tail -1)
  if [ -z "$line" ]; then
    printf '  %-14s REFUSED: no [qtier-vk] resident line -- the expert tier never came up\n' "$t"
    RC=1; continue
  fi
  res=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\1/')
  tot=$(echo "$line" | sed -E 's/.*resident ([0-9]+)\/([0-9]+).*/\2/')
  miss=$(echo "$hitline" | sed -E 's/.*CPU misses ([0-9]+).*/\1/')
  printf '  %-14s resident %s/%s  CPU misses %s\n' "$t" "${res:-?}" "${tot:-?}" "${miss:-?}"
  [ "${res:-0}" = "${tot:-1}" ] && [ "${res:-0}" = "10240" ] \
    || { echo "     FAIL: expected 10240/10240 resident"; RC=1; }
  [ "${miss:-1}" = "0" ] || { echo "     FAIL: CPU misses ${miss:-?} != 0"; RC=1; }
done

echo
echo "=== leg 2: greedy text and teacher_forcing, trunk-on vs trunk-off ==="
for p in sh pk; do
  a="$OUT/trunk_off_$p.text"; b="$OUT/trunk_on_$p.text"
  na=$(tr -d '[:space:]' < "$a" 2>/dev/null | wc -c); nb=$(tr -d '[:space:]' < "$b" 2>/dev/null | wc -c)
  if [ "${na:-0}" -lt 8 ] || [ "${nb:-0}" -lt 8 ]; then
    printf '  %-30s REFUSED: %s / %s non-space chars\n' "$p greedy text" "$na" "$nb"; RC=1
  elif cmp -s "$a" "$b"; then
    printf '  %-30s IDENTICAL (%s non-space chars)\n' "$p greedy text" "$na"
  else
    printf '  %-30s DIFFERS -- reported, the KL (outside the lock) is the scale\n' "$p greedy text"
  fi
  gate_compare "$p teacher_forcing (trunk on/off)" "$OUT/trunk_off_$p.out" "$OUT/trunk_on_$p.out" "^teacher_forcing" || RC=1
done

echo
echo "=== leg 3: dev0 VRAM -- nothing else of qwen36 lives there (expert pool is dev2/dev3 only) ==="
echo "--- [trunk-vk]/[dnp]/[lmh] startup lines from the trunk-ON runs:"
grep -E '^\[trunk-vk\]' "$OUT/trunk_on_sh.err" "$OUT/trunk_on_pk.err" | sed 's/^/    /'
echo "--- raw sysfs mem_info_vram_used per card, right after a trunk-ON run (not resolved to"
echo "    dev0/dev2/dev3 -- CLAUDE.md: amd-smi and nvtop number the three cards differently;"
echo "    the [trunk-vk] lines above are the authoritative source for what THIS run placed)"
for c in /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -r "$c" ] || continue
  echo "    $c = $(( $(cat "$c") /1048576)) MB"
done

echo
echo "=== dumps ready for leg 2's KL (run OUTSIDE the lock) ==="
for f in trunk_off_sh trunk_on_sh trunk_off_pk trunk_on_pk; do
  printf '  %-14s %s bytes\n' "$f" "$(wc -c < "$OUT/$f.dumpall" 2>/dev/null || echo 0)"
done
echo "  next:  ~/src/colibri-f3s2a/tools/hot-expert/f3s2a_kl.sh $OUT"

echo
if [ "$RC" -eq 0 ]; then echo "=== f3s2a KL chain plumbing: PASS (tree $GITSHA)"
else echo "=== f3s2a KL chain plumbing: FAIL (tree $GITSHA)"; fi

echo
echo "[f3s2akl] restarting the gateway (SKIP_WARM=1 is enough -- CLAUDE.md: do not"
echo "          re-warm GLM's 184 GB by cat before restarting the gateway)"
start_gateway
RESTART_GATEWAY=0
echo "[f3s2akl] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[f3s2akl] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
