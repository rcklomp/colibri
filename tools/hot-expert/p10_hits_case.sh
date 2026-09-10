#!/bin/bash
# p10_hits_case.sh -- the HITS half of P10, isolated.
#
#   p10_hits_case.sh [tag]
#
# Step 3 of p10_gate.sh floods the cache with SHORT prefixes, and a short prefix
# is beaten by `len` alone: that arm proves the policy is not LRU, but it would
# pass just as well for a length-only rule. The roadmap's actual claim is
# stronger and is the reason a length-only rule was rejected:
#
#   "the tool-block capture must survive a long one-off paste, which a
#    length-only rule does not guarantee (that paste is longer)"
#
# So this case puts the tool block in a cache where it is ALREADY the shortest
# thing -- two slots, seeded with the tool block and a LONGER stale capture --
# and then changes exactly one thing between the two arms: how many times the
# tool block has been hit. Same binary, same knob (GLM53_CKPT_VALUE_EVICT=1),
# same seed, same trigger.
#
#   arm "warm"  -- $WARM UI chats first, so the tool block reaches hits >= 5
#   arm "cold"  -- no UI chats first, so the tool block stays at hits = 1,
#                  which IS the length-only rule
#
# One capture then arrives and someone has to go:
#   warm: the STALE one  (4712 x 1 = 4712  <  4418 x 5 = 22090)  -> tool block lives
#   cold: the TOOL BLOCK (4418 x 1 = 4418  <  4712 x 1 = 4712)   -> tool block dies
#
# Note what is and is not being ranked: the policy scores the OCCUPANTS, never
# the incoming capture, so the trigger only has to be a capture -- its length is
# irrelevant and it is deliberately small (~400 tokens, ~70 s) instead of the
# multi-minute paste the roadmap tells the story with. The story is unchanged:
# a length-only rule puts the tool block last in this cache, and the roadmap's
# long paste is simply one way to fill it.
#
# The `CKPT store ... evicted=<len>/<hits>` line added by P10 names the victim,
# so the verdict is read from the engine's own accounting, and then confirmed by
# whether the next UI chat still restores >= 4 300. The cold arm's final chat is
# a real ~12-minute cold prefill: that is the cost being avoided, measured.
#
# Run it under the rig lock, through run_chain.sh, like everything else that
# stops the gateway.
set -u
TAG=${1:-p10hits$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p10_hits_$TAG
LOG=${GLM53_LOG:-$HOME/glm53_server.log}
WARM=${P10_WARM:-4}
PASTE_TOK=${P10_PASTE_TOK:-400}
TOOLBLOCK_MIN=${P10_TOOLBLOCK_MIN:-4300}
K=$(cat "$HOME/.colibri_api_key")
URL=http://127.0.0.1:8081
mkdir -p "$OUT"
say() { printf '  %-42s %s\n' "$1" "$2"; }
count() { local n; n=$(grep -ac "$1" "$LOG" 2>/dev/null) || true; echo "${n:-0}"; }

wait_no_engine() {
  for _ in $(seq 1 240); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running: $(pgrep -x glm53 | tr '\n' ' ')"; return 1
}
gw_stop() { "$HOME/bench/p7_stop.sh"; wait_no_engine; }
ARM_SCRIPT="$OUT/start_glm53_arm.sh"
make_arm_script() {
  sed -e 's/^export GLM53_PREFIX_CKPT_MIN=1024$/export GLM53_PREFIX_CKPT_MIN=${GLM53_PREFIX_CKPT_MIN:-1024}/' \
      -e 's/^export GLM53_PREFIX_CKPT_MIN=128$/export GLM53_PREFIX_CKPT_MIN=${GLM53_PREFIX_CKPT_MIN:-128}/' \
      "$HOME/start_glm53.sh" > "$ARM_SCRIPT"
  grep -q 'GLM53_PREFIX_CKPT_MIN:-' "$ARM_SCRIPT" || { echo "the arm script did not take"; return 1; }
  chmod +x "$ARM_SCRIPT"
}
# Every arm runs the gateway on a PRIVATE checkpoint dir and arm knobs. Whatever
# happens, the owner's own service must be what is running when this exits --
# run_chain.sh only restarts a gateway that is DOWN, and an arm's gateway is up.
restore_service() {
  echo "--- restoring the owner's serving configuration"
  "$HOME/bench/p7_stop.sh"; wait_no_engine || echo "  (engine still up)"
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_PREFIX_CKPT_MIN \
      -u GLM53_PREFIX_CKPT_SLOTS -u GLM53_CKPT_VALUE_EVICT -u COLI_MLA_POOL \
      -u COLI_REPLY_PIN -u COLI_LEDGER -u COLI_PREFIX_PIN \
      SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $K" \
         -w '%{http_code}' $URL/v1/models 2>/dev/null)" = 200 ] && break
    sleep 5
  done
  echo "    gateway: $(pgrep -f "openai_[s]erver.py" | wc -l) engine: $(pgrep -x glm53 | wc -l)"
  echo "    serving: $(grep -o 'GLM53_PREFIX_CKPT_MIN=[0-9]*' "$HOME/start_glm53.sh")"
}
trap 'rc=$?; restore_service; echo "=== p10_hits_case done rc=$rc $(date -Is)"' EXIT

gw_start() {
  make_arm_script || return 1
  env -u COLI_CKPT_DIR -u GLM53_PREFIX_CKPT -u GLM53_PREFIX_CKPT_MIN \
      -u GLM53_CKPT_VALUE_EVICT -u COLI_MLA_POOL -u COLI_REPLY_PIN \
      -u COLI_LEDGER -u COLI_PREFIX_PIN \
      "$@" SKIP_WARM=1 setsid nohup "$ARM_SCRIPT" > "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -m 5 -H "Authorization: Bearer $K" \
         -w '%{http_code}' $URL/v1/models 2>/dev/null)" = 200 ] && { sleep 2; return 0; }
    sleep 5
  done
  echo "gateway did not answer /v1/models within 10 min"; return 1
}

LIVE_CKPT=${P10_LIVE_CKPT:-$(ls -d "$HOME"/models/*/.coli_ckpt 2>/dev/null | head -1)}
[ -d "$LIVE_CKPT" ] || { echo "REFUSED: no live checkpoint directory (set P10_LIVE_CKPT)"; exit 2; }

# --- the seed: exactly two slots, the tool block and a LONGER stale capture ----
# WHICH file is the tool block is not a guess and not "the shortest": it is the
# prefix the engine has actually been RESTORING. Seeding the wrong two files
# would make this case measure nothing while looking like it measured something.
#
# The evidence is the gateway log's own `CKPT hit prefix=` lines -- but every
# restart truncates that log (`> $LOG`), so a run that follows one finds none
# (measured 2026-09-10 08:14, this case refusing on an empty log). When there is
# no evidence, MAKE some: one UI-shaped turn against the gateway that is already
# up, on the live directory, which is exactly the request whose prefix is worth
# defending. Five seconds, and it is a better answer than any heuristic.
echo "=== p10_hits_case $TAG $(date -Is)"
if ! grep -aq "CKPT hit prefix=" "$LOG" 2>/dev/null; then
  echo "  no restore in the current log -- driving one UI turn to find out which prefix is the one"
  if pgrep -f "openai_[s]erver.py" >/dev/null; then
    "$HERE/owui_ui_turn.sh" "Which day comes after Tuesday? One word. [$TAG-discover]" \
      > "$OUT/discover.txt" 2>&1
    echo "  $(grep '^RESULT' "$OUT/discover.txt" || echo 'no RESULT')"
    echo "  $(grep -a 'CKPT hit prefix=' "$LOG" | tail -1 | sed 's/.*CKPT/CKPT/')"
  else
    echo "  REFUSED: no gateway is up, so the tool block cannot be identified"; exit 2
  fi
fi
python3 - "$LIVE_CKPT" "$OUT/seed" "$TOOLBLOCK_MIN" "$LOG" <<'PY' | tee "$OUT/seed.txt"
import struct, sys, os, shutil, glob, re, collections
src, dst, tbmin, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)
rows = []
for p in sorted(glob.glob(os.path.join(src, "*.bin"))):
    with open(p, "rb") as f:
        head = f.read(28)
    if len(head) < 28 or head[:8] != b"G53CKPT1":
        continue
    ln, kind = struct.unpack_from("<ii", head, 12)
    rows.append((ln, kind, p, os.path.getsize(p)))
for ln, kind, p, sz in rows:
    print("  candidate %s prefix=%d kind=%d %.0f MB" % (os.path.basename(p), ln, kind, sz / 1e6))
hits = collections.Counter()
try:
    with open(log, "rb") as f:
        for line in f:
            m = re.search(rb"CKPT hit prefix=(\d+)", line)
            if m:
                hits[int(m.group(1))] += 1
except OSError:
    pass
print("  restores seen in the log: " +
      (", ".join("%dx%d" % (k, v) for k, v in hits.most_common(5)) or "none"))
on_disk = dict((r[0], r) for r in rows)
defend = next((ln for ln, _ in hits.most_common() if ln in on_disk and ln >= tbmin), None)
if defend is None:
    print("REFUSED: no >= %d-token capture that the log shows being restored" % tbmin)
    sys.exit(2)
longer = [r for r in rows if r[0] > defend]
if not longer:
    print("REFUSED: nothing on disk is longer than %d, so length-only and len x hits would"
          " pick the SAME victim and the two arms could not differ" % defend)
    sys.exit(2)
stale = max(longer)
base = os.path.basename(on_disk[defend][2]).rsplit("_", 1)[0]
shutil.copy2(on_disk[defend][2], os.path.join(dst, base + "_0.bin"))
shutil.copy2(stale[2], os.path.join(dst, base + "_1.bin"))
# The size this file had BEFORE P10 appended its counter: 28-byte header,
# then the ids, then the blob. Truncating to it turns the copy back into a
# file written by the pre-P10 binary, which loads at hits = 1.


with open(on_disk[defend][2], "rb") as f:
    f.seek(20)
    nbytes = struct.unpack("<Q", f.read(8))[0]
base_size = 28 + defend * 4 + nbytes
print("SEED defend=%d stale=%d base_size=%d" % (defend, stale[0], base_size))
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 2
DEFEND=$(sed -n 's/^SEED defend=\([0-9]*\).*/\1/p' "$OUT/seed.txt")
STALE=$(sed -n 's/^SEED .*stale=\([0-9]*\) .*/\1/p' "$OUT/seed.txt")
BASE_SIZE=$(sed -n 's/^SEED .*base_size=\([0-9]*\).*/\1/p' "$OUT/seed.txt")
[ -n "$BASE_SIZE" ] || { echo "REFUSED: the seed did not report the pre-P10 file size"; exit 2; }
echo "  defending prefix=$DEFEND (hit in the log) against the stale $STALE, trigger=$PASTE_TOK tokens"

# The trigger: one unique system block, big enough to be captured and no bigger.
python3 - "$PASTE_TOK" "$TAG" > "$OUT/paste.json" <<'PY'
import sys, json, random
ntok, tag = int(sys.argv[1]), sys.argv[2]
r = random.Random(tag + "-paste")
words = ["ledger","prefix","expert","router","tensor","kernel","window","segment","adapter",
         "residue","channel","budget","harness","canary","rotate","shard","latent","gate"]
filler = " ".join(r.choice(words) for _ in range(int(ntok * 3.6 / 6.5)))
print(json.dumps({"model": "glm-5.3-flash", "max_tokens": 8, "messages": [
    {"role": "system", "content": f"Pasted reference document {tag}. Do not summarise: {filler}"},
    {"role": "user", "content": f"Reply with the single word OK. [{tag}]"}]}))
PY

arm() {   # arm <label> <warmups>
  # Two statements, not one: under `set -u` bash creates every name in a single
  # `local` before assigning them, so `local label=$1 dir="$OUT/ckpt-$label"`
  # reads an unbound `label` and aborts (measured 2026-09-10 08:11).
  local label=$1 n=$2 i
  local dir="$OUT/ckpt-$label"
  echo
  echo "--- arm $label: $n warm-up UI chats before the paste (GLM53_CKPT_VALUE_EVICT=1)"
  rm -rf "$dir"; mkdir -p "$dir"; cp -p "$OUT/seed"/*.bin "$dir"/ || return 2
  # P10 PERSISTS the counter, so skipping the warm-ups does NOT make an arm cold:
  # the first attempt at this case (2026-09-10 08:18) seeded a file that already
  # carried hits=4 and both arms kept the tool block, discriminating nothing.
  # A genuinely cold arm needs a genuinely pre-P10 file -- so truncate the
  # counter off, which is byte-for-byte what the old binary wrote, and loads at
  # hits = 1: the length-only rule, which is the thing being argued against.
  if [ "$n" = 0 ]; then
    local f0; f0=$(ls "$dir"/*_0.bin)
    local was; was=$(stat -c %s "$f0")
    truncate -s "$BASE_SIZE" "$f0" || return 2
    say "cold arm: counter truncated" "$was -> $(stat -c %s "$f0") bytes (a pre-P10 file, hits=1)"
  fi
  gw_stop || return 2
  gw_start COLI_CKPT_DIR="$dir" GLM53_PREFIX_CKPT_MIN=128 GLM53_CKPT_VALUE_EVICT=1 \
           GLM53_PREFIX_CKPT_SLOTS=2 || return 2
  grep -a "CKPT disk load" "$LOG" | tail -2 | sed 's/^/      /' | cut -c1-140
  for i in $(seq 1 "$n"); do
    "$HERE/owui_ui_turn.sh" "Which day comes after Tuesday? One word. [$TAG-$label-w$i]" \
      > "$OUT/$label-warm$i.txt" 2>&1
    say "warm-up $i/$n" "$(grep '^RESULT' "$OUT/$label-warm$i.txt" || echo 'no RESULT')"
  done
  say "tool block now" "$(grep -a 'CKPT hit prefix=' "$LOG" | tail -1 | cut -c1-90)"
  local mark; mark=$(count "CKPT store")
  say "the paste ($PASTE_TOK tokens, one cold prefill)" "submitting $(date -u +%H:%M:%S)"
  curl -s -m 3600 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
    $URL/v1/chat/completions -d @"$OUT/paste.json"
  local stores ev
  stores=$(( $(count "CKPT store") - mark ))
  ev=$(grep -a "CKPT store" "$LOG" | tail -1)
  say "CKPT store lines" "+$stores"
  echo "      ${ev:0:150}"
  if [ "$stores" -lt 1 ]; then
    echo "    REFUSED: the paste stored nothing -- it tested nothing"; return 2
  fi
  local evicted; evicted=$(echo "$ev" | sed -n 's/.*evicted=\([0-9]*\)\/.*/\1/p')
  # The verdict the arm exists for: whom did the policy choose?
  local mark_hit; mark_hit=$(count "CKPT hit prefix=")
  "$HERE/owui_ui_turn.sh" "Name one prime number greater than ten. One word. [$TAG-$label-after]" \
    > "$OUT/$label-after.txt" 2>&1
  local res big
  res=$(grep '^RESULT' "$OUT/$label-after.txt" || echo "RESULT error")
  big=$(grep -a "CKPT hit prefix=" "$LOG" | tail -n +$((mark_hit + 1)) \
        | sed -n 's/.*CKPT hit prefix=\([0-9]*\).*/\1/p' | awk -v m="$TOOLBLOCK_MIN" '$1>=m' | tail -1)
  say "UI chat after the paste" "$res"
  say "verdict" "evicted=$evicted, tool-block restore=${big:-none}"
  ARM_RESULT="evicted=$evicted restore=${big:-none} ${res#RESULT }"
  [ -n "$big" ] && return 0 || return 1
}

ARM_RESULT=""; arm warm "$WARM"; warm_rc=$?; warm_res=$ARM_RESULT
ARM_RESULT=""; arm cold 0;       cold_rc=$?; cold_res=$ARM_RESULT

echo
echo "  | arm | tool block's hits | after one $PASTE_TOK-token paste |"
echo "  |---|---|---|"
echo "  | warm ($WARM UI chats first) | >= $((WARM + 1)) | $warm_res |"
echo "  | cold (none)                 | 1 (= the length-only rule) | $cold_res |"
if [ "$warm_rc" = 2 ] || [ "$cold_rc" = 2 ]; then
  echo "  REFUSED: an arm did not test anything (warm=$warm_rc cold=$cold_rc)"; exit 2
fi
if [ "$warm_rc" != 0 ]; then
  echo "  FAIL: a long paste evicted the tool block even after it had earned its hits"
  echo "        -- the hits term is not doing what the spec claims"
  exit 1
fi
if [ "$cold_rc" = 0 ]; then
  echo "  FAIL: the cold arm ALSO kept the tool block, so this paste does not discriminate"
  echo "        len from len x hits and the warm arm passed for an unknown reason."
  echo "        Raise P10_PASTE_TOK above the STALE capture ($STALE), not just above $DEFEND."
  exit 1
fi
echo "  PASS: with hits the tool block survives a longer paste; at hits=1 (length only) it does not"
exit 0
