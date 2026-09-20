#!/bin/bash
# f3s0_kl_chain.sh -- FRANKEN-ENGINE-PLAN item F3, step 0, part A: what does
# int4-gs64 cost against row-wise int8 on the CPU engine `qwen36`, same
# prompts, teacher-forced full-row logits.
#
#   setsid nohup ~/src/colibri-f3s0/tools/hot-expert/run_chain.sh \
#       ~/src/colibri-f3s0/tools/hot-expert/f3s0_kl_chain.sh \
#       >> ~/bench/f3s0_kl.log 2>&1 < /dev/null &
#
# Pattern copied from v1_step2a_chain.sh (V1 step 2a, the tier-on/tier-off
# KL chain), the closest thing this repo has to "compare two qwen36
# containers on the same binary". The two differences from that pattern,
# both because this is a container comparison and not a bit-identity one:
#
#   - ONE binary (`qwen36`, CPU, no Vulkan -- this question has nothing to
#     do with the tier), TWO containers: A = int4-gs64
#     (~/models/qwen36_i4_gs64, the one V1 already gated), B = row-wise int8
#     (~/models/qwen36_i8_row, produced by THIS repo's own
#     convert_qwen36.py --ebits 8, no --gs -- a container V1's KL bar was
#     never run against, because it did not exist until 2026-09-20).
#   - int4 and int8 are NOT expected to be bit-identical or even KL-floor
#     identical the way tier-on/tier-off was: they are different
#     quantizations of the same weights. The question is whether int8 buys
#     quality over int4, and by how much -- there is no PASS/FAIL bar here,
#     only a report (f3s0_kl.sh, run OUTSIDE the lock, prints the numbers
#     next to the bars this track has already used to accept or reject a
#     quantization: V1's own floor, 3.5e-05 / 99.89%, "same"; the swiglu
#     clamp gap, 0.0284 / 92.73%, a known DEFECT).
#
# WHAT THIS CHAIN DOES NOT DO: it does not compute the KL. kl_compare.py is
# pure Python and takes minutes per pair on the 625-position packet -- that
# is rig-lock time held down for arithmetic that needs no rig state. The
# chain writes the dumps, restarts the gateway, runs accept_live.sh and
# RELEASES THE LOCK; f3s0_kl.sh computes the KL and the flip-margin view
# afterwards, over the same files, with no lock held.
#
# Refusals that matter here:
#   - ~/models/qwen36_i8_row must be a COMPLETE container: qwen36_meta.json
#     is written by convert_qwen36.py only after every layer has been
#     converted (c/tools/convert_qwen36.py, the meta.write_text() call is
#     the second-to-last thing main() does), so its mere presence is a
#     completeness marker; CONVERT_DONE in the conversion log is the second,
#     independent signal. Both are required, or the chain refuses before
#     touching the gateway.
#   - each container must be resident in the page cache (>= 90%, fincore)
#     before ITS OWN arm runs -- both containers fit in RAM at once here
#     (22 GB + ~37 GB, well inside 247 GiB), so unlike the GLM/Qwen3.8
#     mutual-exclusion case (CLAUDE.md) there is no need to evict between
#     arms, only to warm each one once.
#   - qwen36 has no KV slots and no prefix checkpointing (grep for
#     COLI_CKPT_DIR / PREFIX_CKPT in c/qwen36.c: neither exists), so unlike
#     glm53 gate chains there is no private checkpoint dir to set here.
set -u
TAG=f3s0kl$(date +%m%d%H%M)
OUT=~/bench/f3s0_kl_out; mkdir -p "$OUT"
SRC=~/src/colibri-f3s0
HERE="$SRC/tools/hot-expert"
I4=${F3S0_I4:-/home/ronald/models/qwen36_i4_gs64}
I8=${F3S0_I8:-/home/ronald/models/qwen36_i8_row}
CONVLOG=${F3S0_CONVLOG:-$HOME/bench/f3s0_convert.log}
CAP=${F3S0_CAP:-256}                # cache/layer: every expert resident, as V1's arm C
NNEW=${F3S0_NNEW:-128}
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
  echo "[f3s0kl] gateway back, /v1/models=${code:-?} engine=$(pgrep -x glm53 | wc -l)"
}

wait_no_engine() {
  local n="$1"
  for _ in $(seq 1 120); do pgrep -x "$n" >/dev/null 2>&1 || return 0; sleep 2; done
  echo "[f3s0kl] FATAL: a $n is still alive after 240s"; return 1
}

stop_gateway() {
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[f3s0kl] stopping the owner's gateway for the duration of this run"
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
  echo "=== f3s0_kl_chain exit rc=$rc tag=$TAG $(date -Is)"
  echo "=== dumps: $OUT/  -- now run f3s0_kl.sh OUTSIDE the lock"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP

echo "=== f3s0_kl_chain $TAG $(date -Is)"
for e in qwen38 qwen38-vk qwen36 qwen36-vk; do
  pgrep -x "$e" >/dev/null 2>&1 && { echo "[f3s0kl] $e is running -- refusing"; exit 1; }
done

echo "--- container completeness: $I8 (the i8 container is the one still being produced)"
if ! grep -q '^CONVERT_DONE' "$CONVLOG" 2>/dev/null; then
  echo "[f3s0kl] REFUSED: no CONVERT_DONE in $CONVLOG -- the int8 conversion has not finished"
  exit 1
fi
if [ ! -f "$I8/qwen36_meta.json" ]; then
  echo "[f3s0kl] REFUSED: $I8/qwen36_meta.json missing -- convert_qwen36.py writes this file"
  echo "         only after every layer is converted (c/tools/convert_qwen36.py); its"
  echo "         absence means the container is a partial write, not a finished one"
  exit 1
fi
for f in config.json tokenizer.json; do
  [ -f "$I8/$f" ] || { echo "[f3s0kl] REFUSED: $I8/$f missing"; exit 1; }
done
NSHI4=$(find "$I4" -maxdepth 1 -name '*.safetensors' | wc -l)
NSHI8=$(find "$I8" -maxdepth 1 -name '*.safetensors' | wc -l)
echo "[f3s0kl] i4 shards=$NSHI4 i8 shards=$NSHI8 (need not match: --low-disk shard counts can"
echo "         differ between containers; qwen36_meta.json + CONVERT_DONE are the completeness"
echo "         signal, not the shard count)"
python3 -c "
import json
m = json.load(open('$I8/qwen36_meta.json'))
assert m.get('ebits') == 8, 'meta.ebits=%r, expected 8' % m.get('ebits')
assert not m.get('expert_gs'), 'meta.expert_gs=%r, expected 0/absent (row-wise)' % m.get('expert_gs')
print('[f3s0kl] i8 meta OK: ebits=8 expert_gs=%r num_experts=%s topk=%s' %
      (m.get('expert_gs'), m.get('num_experts'), m.get('topk')))
" || { echo "[f3s0kl] REFUSED: $I8/qwen36_meta.json failed the ebits=8/per-row check"; exit 1; }

stop_gateway || exit 1

echo "[f3s0kl] tree at $SRC: $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?')"
GITSHA=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)

echo "--- building qwen36 (CPU, no VK -- this question has nothing to do with the tier)"
( cd "$SRC/c" && make qwen36 ) > "$OUT/${TAG}_build.log" 2>&1 \
  || { echo "[f3s0kl] build FAILED"; tail -30 "$OUT/${TAG}_build.log"; exit 1; }
cp -f "$SRC/c/qwen36" ~/bench/qwen36.f3s0
BIN=~/bench/qwen36.f3s0
echo "[f3s0kl] qwen36.f3s0 sha256=$(sha256sum "$BIN" | cut -c1-16)"
if [ -x ~/bench/qwen36.v1cpu ]; then
  if cmp -s "$BIN" ~/bench/qwen36.v1cpu; then
    echo "[f3s0kl] identical to ~/bench/qwen36.v1cpu (V1's gated CPU binary) -- as expected,"
    echo "         c/qwen36.c has not changed since V1 and includes no shared header (quant.h"
    echo "         etc) that F2-F9a touched"
  else
    echo "[f3s0kl] NOTE: differs from ~/bench/qwen36.v1cpu -- c/qwen36.c or a header it includes"
    echo "         has changed since V1; this run uses the freshly built binary regardless"
  fi
fi

resident_pct() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.safetensors' -print0 | xargs -0 cat > /dev/null 2>&1 || true
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
echo "[f3s0kl] i4 container resident: $PCT4%"
python3 -c "import sys; sys.exit(0 if float('$PCT4')>=90.0 else 1)" \
  || { echo "[f3s0kl] REFUSED: i4 residency $PCT4% < 90%"; exit 1; }

echo "--- warming $I8"
PCT8=$(resident_pct "$I8")
echo "[f3s0kl] i8 container resident: $PCT8%"
python3 -c "import sys; sys.exit(0 if float('$PCT8')>=90.0 else 1)" \
  || { echo "[f3s0kl] REFUSED: i8 residency $PCT8% < 90%"; exit 1; }

PACKET="$HERE/x2_packet_450.txt"
SHORT=/tmp/f3s0_short.txt
printf '%s' 'What comes after Tuesday, and why is the answer the same in every calendar that has a seven-day week?' > "$SHORT"
[ -s "$PACKET" ] || { echo "[f3s0kl] FATAL: $PACKET missing"; exit 1; }

# run <tag> <container> <ebits-arg-for-the-cli-label> <promptfile> <n_new>
run() {
  local tag="$1" snap="$2" ebits="$3" pf="$4" nnew="$5"
  echo "[f3s0kl] $(date +%H:%M:%S) run $tag (SNAP=$snap ebits-arg=$ebits)"
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export SNAP="$snap" N_NEW="$nnew" NOSTREAM=1
    export Q36_TEACHER_FORCING=1
    export Q36_LOGIT_DUMP_ALL="$OUT/$tag.dumpall"
    /usr/bin/time -v "$BIN" "$CAP" "$ebits" "$pf" ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  grep -v '^teacher_forcing' "$OUT/$tag.out" > "$OUT/$tag.text"
  local pos maj ttft rss
  pos=$(awk '/^teacher_forcing/{print NF-1}' "$OUT/$tag.out")
  maj=$(awk -F': ' '/Major .*page faults/{print $2}' "$OUT/$tag.err")
  ttft=$(awk '/^TTFT:/{print $2}' "$OUT/$tag.err")
  rss=$(awk -F': ' '/Maximum resident set size/{printf "%.1f", $2/1048576}' "$OUT/$tag.err")
  echo "[f3s0kl] $(date +%H:%M:%S) run $tag rc=$rc positions=${pos:-0} majflt=${maj:-?}" \
       "ttft=${ttft:-?}s peakRSS=${rss:-?}GB dumpall=$(wc -c < "$OUT/$tag.dumpall" 2>/dev/null || echo 0)B"
  grep -E '^\[qwen36\]|^== qwen36' "$OUT/$tag.err" | sed 's/^/        /'
  return $rc
}

run sh_i4 "$I4" 4 "$SHORT"  "$NNEW"
run sh_i8 "$I8" 8 "$SHORT"  "$NNEW"
run pk_i4 "$I4" 4 "$PACKET" "$NNEW"
run pk_i8 "$I8" 8 "$PACKET" "$NNEW"

echo
echo "=== both containers loaded correctly (report, not a bit-identity gate) ==="
RC=0
for t in sh_i4 pk_i4; do
  n=$(wc -c < "$OUT/$t.dumpall" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] || { echo "  $t REFUSED: empty dump"; RC=1; }
  printf '  %-8s dumpall %s bytes\n' "$t" "$n"
done
for t in sh_i8 pk_i8; do
  n=$(wc -c < "$OUT/$t.dumpall" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] || { echo "  $t REFUSED: empty dump"; RC=1; }
  printf '  %-8s dumpall %s bytes\n' "$t" "$n"
  # the int8 slot path must actually read per-row scales, not silently fall
  # back to the group path with a bogus group size -- container_layer_is_int4
  # decides this by on-disk SIZE, so a wrong read would show up as either an
  # "int4 packed weights detected" line (wrong container) or a crash.
  if grep -qi 'int4 packed weights detected' "$OUT/$t.err"; then
    echo "  $t FAIL: engine detected int4-packed weights while loading the int8 container"
    RC=1
  fi
done

echo
echo "=== greedy text, both containers (report; DIFFERS is expected -- int4 vs int8"
echo "===  are different quantizations, not a reassociation of the same one) ==="
. "$HERE/gate_lib.sh"
for p in sh pk; do
  a="$OUT/${p}_i4.text"; b="$OUT/${p}_i8.text"
  na=$(tr -d '[:space:]' < "$a" 2>/dev/null | wc -c); nb=$(tr -d '[:space:]' < "$b" 2>/dev/null | wc -c)
  if [ "${na:-0}" -lt 8 ] || [ "${nb:-0}" -lt 8 ]; then
    printf '  %-22s REFUSED: %s / %s non-space chars\n' "$p greedy text" "$na" "$nb"; RC=1
  elif cmp -s "$a" "$b"; then
    printf '  %-22s IDENTICAL (%s non-space chars)\n' "$p greedy text" "$na"
  else
    printf '  %-22s DIFFERS -- reported, the KL/flip-margin view (outside the lock) is the scale\n' "$p greedy text"
    echo "     i4: $(head -c 160 "$a" | tr '\n' ' ')"
    echo "     i8: $(head -c 160 "$b" | tr '\n' ' ')"
  fi
done

echo
echo "=== teacher_forcing argmax agreement (the cheap half of the KL question) ==="
for p in sh pk; do
  gate_compare "$p teacher_forcing" "$OUT/${p}_i4.out" "$OUT/${p}_i8.out" "^teacher_forcing" \
    || python3 - "$OUT/${p}_i4.out" "$OUT/${p}_i8.out" <<'PY'
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
for f in sh_i4 sh_i8 pk_i4 pk_i8; do
  printf '  %-10s %s bytes\n' "$f" "$(wc -c < "$OUT/$f.dumpall" 2>/dev/null || echo 0)"
done
echo "  next:  ~/src/colibri-f3s0/tools/hot-expert/f3s0_kl.sh $OUT"

echo
if [ "$RC" -eq 0 ]; then echo "=== f3s0 KL chain plumbing: PASS (tree $GITSHA)"
else echo "=== f3s0 KL chain plumbing: FAIL (tree $GITSHA)"; fi

echo
echo "[f3s0kl] re-warming the GLM container before the gateway returns"
GLM=${F3S0_GLM_MODEL:-$HOME/models/GLM-5.3-Flash-colibri-int4-g64}
cat "$GLM"/*.safetensors > /dev/null 2>&1 || true
start_gateway
RESTART_GATEWAY=0
echo "[f3s0kl] accept_live.sh"
"$HOME/src/colibri/tools/hot-expert/accept_live.sh"
ARC=$?
echo "[f3s0kl] accept_live.sh rc=$ARC"
[ "$RC" -eq 0 ] || exit 1
exit $ARC
