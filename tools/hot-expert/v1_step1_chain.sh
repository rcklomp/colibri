#!/bin/bash
# v1_step1_chain.sh -- FRANKEN-ENGINE-PLAN section 2.6, item V1, STEP 1:
# prove that `qwen36-vk` with the Vulkan expert tier COMPILED IN but OFF is the
# CPU engine, bit for bit.
#
# Launch only through run_chain.sh, which takes the rig lock and refuses a
# second chain rather than interleaving:
#
#   setsid nohup ~/src/colibri-v1/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-v1/tools/hot-expert/v1_step1_chain.sh \
#       >> ~/bench/v1_step1.log 2>&1 < /dev/null &
#
# Shape taken from x2_chain.sh: refuse if any engine is alive -> stop the
# gateway -> wait for it to die -> warm the container and assert page-cache
# residency -> build -> measure -> restart the gateway on EVERY exit path ->
# accept_live.sh, because the request AFTER the measurement is part of it.
#
# This chain never touches the serving tree (~/src/colibri); it works inside
# the dedicated clone ~/src/colibri-v1 (perf/v1-qwen36-vk-tier).
#
# Step 1 does not start a Vulkan device at all: with Q36_VULKAN unset,
# qwen36_tier_vk.c's qt_init returns before coli_vk_init. The gateway still
# comes down, because 8 pinned threads and ~30 GB of RSS beside the owner's
# engine is not a measurement either side would recognise.
#
# THREE binaries, all from this clone:
#   qwen36.v1cpu   `make qwen36`             -- no VK at all: the reference,
#                                               the same build arm C measured.
#   qwen36.v1vkoff `make qwen36-vk VK=1`     -- tier compiled in, left OFF.
#   (the same file is what step 2 will run with Q36_VULKAN=1.)
#
# THE GATE (all four legs must pass):
#   1. thirteen qwen36 C tests pass (count re-derived from the Makefile here,
#      not copied from a document -- CLAUDE.md's qwen38 lesson)
#   2. teacher_forcing identical on the ladder packet and on a short prompt
#   3. last-token logits BYTE-identical (cmp, not a cosine -- step 1 claims
#      bit-identity, so anything weaker would be a claim about a claim)
#   4. greedy text identical on both prompts
# plus a self-check on the new oracle itself: the last row of the
# per-position dump must equal the DUMP= last-token vector byte for byte,
# or the dump is measuring a different head pass than the engine's own.
set -u
TAG=v1s1$(date +%m%d%H%M)
OUT=~/bench/v1_step1_out; mkdir -p "$OUT"
SRC=~/src/colibri-v1
HERE="$SRC/tools/hot-expert"
M=${V1_MODEL:-/home/ronald/models/qwen36_i4_gs64}
CAP=${V1_CAP:-256}          # cache/layer = n_experts: every expert in RAM, as arm C
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
  echo "[v1s1] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {          # <name>
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[v1s1] FATAL: a $n is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[v1s1] stopping the owner's gateway for the duration of this run"
    pkill -f "openai_[s]erver.py" || true
    RESTART_GATEWAY=1
  fi
  pkill -9 -x glm53 2>/dev/null || true
  wait_no_engine glm53
}

on_exit() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x qwen36 2>/dev/null || true
  pkill -9 -x qwen36-vk 2>/dev/null || true
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ] || ! pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    start_gateway
  fi
  echo "=== v1_step1_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== results: $OUT/$TAG.*"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== v1_step1_chain $TAG $(date -Is)"

for e in glm53 qwen38 qwen38-vk qwen36 qwen36-vk; do
  if pgrep -x "$e" >/dev/null 2>&1 && [ "$e" != glm53 ]; then
    echo "[v1s1] $e is running -- refusing"; exit 1
  fi
done

stop_gateway || exit 1

echo "[v1s1] refreshing $SRC"
git -C "$SRC" pull --ff-only || exit 1
git -C "$SRC" log -1 --oneline
GITSHA=$(git -C "$SRC" rev-parse --short HEAD)

# ---- VRAM assertion: nothing of ours, and nothing of the gateway's, is left
# on any card before we measure. Step 1 opens no device, so a non-zero reading
# here means something else is still holding VRAM and the box is not idle.
echo "--- VRAM after the gateway stopped"
for c in /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -r "$c" ] || continue
  used=$(cat "$c"); echo "  $c = $((used/1048576)) MB"
  if [ "$used" -gt 1073741824 ]; then
    echo "[v1s1] REFUSED: $c holds $((used/1048576)) MB (> 1 GB) -- the box is not idle"; exit 1
  fi
done

echo "--- building (qwen36 without VK, qwen36-vk with VK=1)"
( cd "$SRC/c" && make qwen36 ) > "$OUT/${TAG}_build_cpu.log" 2>&1 \
  || { echo "[v1s1] qwen36 build FAILED"; tail -30 "$OUT/${TAG}_build_cpu.log"; exit 1; }
cp -f "$SRC/c/qwen36" ~/bench/qwen36.v1cpu
( cd "$SRC/c" && make qwen36-vk VK=1 ) > "$OUT/${TAG}_build_vk.log" 2>&1 \
  || { echo "[v1s1] qwen36-vk build FAILED"; tail -30 "$OUT/${TAG}_build_vk.log"; exit 1; }
cp -f "$SRC/c/qwen36-vk" ~/bench/qwen36-vk.v1
CPU_BIN=~/bench/qwen36.v1cpu
VK_BIN=~/bench/qwen36-vk.v1
echo "[v1s1] qwen36.v1cpu   sha256=$(sha256sum $CPU_BIN | cut -c1-16)"
echo "[v1s1] qwen36-vk.v1   sha256=$(sha256sum $VK_BIN  | cut -c1-16)"
echo "[v1s1] qwen36-vk links: $(ldd $VK_BIN | grep -c vulkan) vulkan, qwen36 links: $(ldd $CPU_BIN | grep -c vulkan)"

# ---- leg 1: the C tests, count derived here -------------------------------
echo
echo "--- leg 1: qwen36 C tests"
TESTS=$(grep -oE '^tests/(test_qwen36[A-Za-z_0-9]*)\$\(EXE\):' "$SRC/c/Makefile" \
        | sed -E 's|^tests/||; s|\$\(EXE\):||' | sort -u)
NTESTS=$(echo "$TESTS" | wc -w)
echo "[v1s1] Makefile declares $NTESTS qwen36 test targets: $(echo $TESTS)"
( cd "$SRC/c" && make $(echo "$TESTS" | sed 's|^|tests/|') ) > "$OUT/${TAG}_tests_build.log" 2>&1 \
  || { echo "[v1s1] test build FAILED"; tail -30 "$OUT/${TAG}_tests_build.log"; exit 1; }
TPASS=0; TFAIL=0
for t in $TESTS; do
  if ( cd "$SRC/c" && OMP_NUM_THREADS=2 "./tests/$t" ) > "$OUT/${TAG}_t_$t.log" 2>&1; then
    echo "  PASS $t"; TPASS=$((TPASS+1))
  else
    echo "  FAIL $t"; tail -4 "$OUT/${TAG}_t_$t.log"; TFAIL=$((TFAIL+1))
  fi
done
echo "[v1s1] C tests: $TPASS passed, $TFAIL failed (of $NTESTS)"

# ---- warm the container ----------------------------------------------------
echo
echo "--- warming the qwen36 container ($M)"
find "$M" -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true
resid() {
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/v1_fincore.txt
  python3 - /tmp/v1_fincore.txt <<'PY'
import sys
PAGE = 4096
tot = res = 0
for line in open(sys.argv[1]).read().splitlines()[1:]:
    f = line.split()
    if len(f) < 3: continue
    size, r = int(f[-2]), int(f[-1])
    tot += -(-size // PAGE) * PAGE
    res += r
pct = res / tot * 100 if tot else 0.0
print(f"{pct:.1f}")
sys.exit(0 if pct >= 90.0 else 1)
PY
}
PCT=$(resid) || { echo "[v1s1] REFUSED: page-cache residency $PCT% < 90%"; exit 1; }
echo "[v1s1] container resident: $PCT%"

# ---- the two prompts -------------------------------------------------------
PACKET="$HERE/x2_packet_450.txt"
SHORT=/tmp/v1_short.txt
printf '%s' 'What comes after Tuesday, and why is the answer the same in every calendar that has a seven-day week?' > "$SHORT"
[ -s "$PACKET" ] || { echo "[v1s1] FATAL: $PACKET missing"; exit 1; }
echo "[v1s1] prompts: packet $(wc -c < "$PACKET") chars, short $(wc -c < "$SHORT") chars"

# ---- the runs --------------------------------------------------------------
# One CLI run per (binary, prompt). Both binaries see byte-identical
# environments; the ONLY difference is which file is executed.
run() {                     # run <tag> <binary> <promptfile> <n_new>
  local tag="$1" bin="$2" pf="$3" nnew="$4"
  echo "[v1s1] $(date +%H:%M:%S) run $tag"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$M" N_NEW="$nnew" NOSTREAM=1
    export Q36_TEACHER_FORCING=1
    export Q36_LOGIT_DUMP_ALL="$OUT/$tag.dumpall"
    export DUMP="$OUT/$tag.last"
    export COLI_VK_SHADERS="$SRC/c/shaders"
    # Q36_VULKAN deliberately UNSET: this is the tier-off arm.
    /usr/bin/time -v "$bin" "$CAP" 4 "$pf" ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  local pos maj
  pos=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  maj=$(awk -F': ' '/Major .page faults/{print $2}' "$OUT/$tag.err")
  echo "[v1s1] $(date +%H:%M:%S) run $tag rc=$rc positions=${pos:-0} majflt=${maj:-?} dumpall=$(wc -c < "$OUT/$tag.dumpall" 2>/dev/null || echo 0)B last=$(wc -c < "$OUT/$tag.last" 2>/dev/null || echo 0)B"
  grep -E '^\[qtier' "$OUT/$tag.err" | head -3
  return $rc
}

# Short prompt first: it is the cheap one, and if the new oracle knobs produce
# nothing there is no point paying for two packet prefills to find out.
run sh_cpu "$CPU_BIN" "$SHORT"  16
if ! grep -q '^teacher_forcing' "$OUT/sh_cpu.out"; then
  echo "[v1s1] FATAL: no teacher_forcing line from the reference binary -- aborting before"
  echo "       the packet runs. An absent oracle line is not a pass (gate_lib.sh's rule)."
  tail -20 "$OUT/sh_cpu.err"; exit 1
fi
run sh_vk  "$VK_BIN"  "$SHORT"  16
run pk_cpu "$CPU_BIN" "$PACKET" 8
run pk_vk  "$VK_BIN"  "$PACKET" 8

# ---- the gate --------------------------------------------------------------
echo
echo "=== V1 step 1 gate: qwen36-vk (tier OFF) vs qwen36 ==="
. "$HERE/gate_lib.sh"
RC=0

echo "-- leg 2: teacher_forcing (per-position argmax over the whole prefill)"
gate_compare "packet teacher_forcing" "$OUT/pk_cpu.out" "$OUT/pk_vk.out" "^teacher_forcing" || RC=1
gate_compare "short  teacher_forcing" "$OUT/sh_cpu.out" "$OUT/sh_vk.out" "^teacher_forcing" || RC=1

echo "-- leg 3: last-token logits, byte-identical"
for p in pk sh; do
  a="$OUT/${p}_cpu.last"; b="$OUT/${p}_vk.last"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    printf '  %-34s REFUSED: a dump is missing or empty (A=%s B=%s bytes)\n' \
      "$p last_logits" "$(wc -c < "$a" 2>/dev/null || echo 0)" "$(wc -c < "$b" 2>/dev/null || echo 0)"
    RC=1
  elif cmp -s "$a" "$b"; then
    printf '  %-34s IDENTICAL (%s bytes)\n' "$p last_logits" "$(wc -c < "$a")"
  else
    printf '  %-34s DIFFERS\n' "$p last_logits"; RC=1
  fi
done

echo "-- leg 4: greedy text"
gate_compare "packet greedy text" "$OUT/pk_cpu.err" "$OUT/pk_vk.err" "^Text" || RC=1
gate_compare "short  greedy text" "$OUT/sh_cpu.err" "$OUT/sh_vk.err" "^Text" || RC=1

echo "-- leg 5: the per-position dump agrees with the engine's own last-token logits"
python3 - "$OUT" <<'PY'
import sys, os, struct
out = sys.argv[1]
bad = 0
for p in ("pk_cpu", "pk_vk", "sh_cpu", "sh_vk"):
    da, la = f"{out}/{p}.dumpall", f"{out}/{p}.last"
    if not (os.path.exists(da) and os.path.exists(la)):
        print(f"  {p:<32} REFUSED: dump missing"); bad += 1; continue
    with open(da, "rb") as f:
        magic, ver, count, vocab = struct.unpack("<4I", f.read(16))
        if magic != 0x444b4c47 or ver != 1 or count <= 0 or vocab <= 0:
            print(f"  {p:<32} REFUSED: bad GLKD header"); bad += 1; continue
        f.seek(16 + (count - 1) * vocab * 4)
        last_of_dump = f.read(vocab * 4)
    with open(la, "rb") as f:
        last = f.read()
    if len(last) != vocab * 4:
        print(f"  {p:<32} REFUSED: DUMP= is {len(last)}B, expected {vocab*4}B"); bad += 1
    elif last_of_dump == last:
        print(f"  {p:<32} IDENTICAL ({count} positions x {vocab})")
    else:
        print(f"  {p:<32} DIFFERS -- the dump's head pass is not the engine's"); bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || RC=1

echo "-- leg 1 recap: C tests $TPASS/$NTESTS passed"
[ "$TFAIL" -eq 0 ] || RC=1

echo
if [ "$RC" -eq 0 ]; then echo "=== V1 STEP 1 GATE: PASS  (tree $GITSHA)"
else echo "=== V1 STEP 1 GATE: FAIL (tree $GITSHA)"; fi

echo
echo "[v1s1] re-warming the GLM container before the gateway comes back -- this run put"
echo "       ~22 GB of qwen36 into a page cache that holds the owner's 182 GiB model, and"
echo "       his first chat afterwards must not pay 100k major faults for our measurement"
GLM=${V1_GLM_MODEL:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
cat "$GLM"/*.safetensors > /dev/null 2>&1 || true

echo "[v1s1] restoring the gateway before accept_live.sh"
start_gateway
RESTART_GATEWAY=0
echo "[v1s1] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ACCEPT_RC=$?
echo "[v1s1] accept_live.sh rc=$ACCEPT_RC"

[ "$RC" -eq 0 ] || exit 1
exit $ACCEPT_RC
