#!/bin/bash
# P6b — the executable gate for per-slot KDA device state.
#
#   p6b_gate.sh <pristine-glm53> <candidate-glm53> [tag]
#
# Exit 0 = the item is done and its tables belong in the commit body.
# Exit 1 = an oracle failed (tworeq across slots, teacher forcing, logits).
# Exit 2 = a harness refusal (residency, another engine running, a fixture).
# Exit 3 = a speed check failed (neutrality at one slot, decode at four,
#          or the device->host sync still on the write-combined path).
#
# The six steps are P6B-KDA-SLOT-STATE-SPEC-2026-09.md §4, in its order:
#
#  (1) tworeq.py at TWOREQ_SLOTS=4 with COLI_KDA_GPU=2 -- IDENTICAL, and no
#      "forcing COLI_KDA_GPU=0" line, which is the whole point of the item:
#      before P6b that guard fired and the run silently measured the CPU
#      recurrence. Then again at =0. This is THE oracle for the change: state
#      with a new home is exactly the class tworeq exists for (G12 stage 2c),
#      and the teacher-forcing CLI never opens two sessions.
#  (2) prefill_gate.sh with MIN_SPEEDUP=0.97 -- one slot, one state set, so
#      the candidate must be neutral against the served P7 binary.
#  (3) Decode at four slots. Three rows, one engine at a time:
#        candidate --kv-slots 4 (GPU, the pool)
#        pristine  --kv-slots 1 (GPU, one state set -- the number to match)
#        pristine  --kv-slots 4 (the guard forces the CPU recurrence)
#      Candidate within 1 % of the pristine 1-slot row and >= 8 % above the
#      pristine 4-slot row.
#  (4) The live side-request case through the SERVING gateway. It needs a
#      gateway, so it runs only when P6B_URL is set -- the chain calls this
#      script again with GATE_STEPS=4 after it has served the candidate.
#  (5) Sync speed: capture a checkpoint at the GPU knob and read `sync=<ms>`
#      off the engine's own `CKPT store` line. Under P6B_SYNC_MS (1000);
#      the write-combined read path would print ~11 000.
#  (6) The pool's price and its consequence: `[VK] preload: N heat-ranked
#      experts` on both binaries (the pool is allocated BEFORE the preload, so
#      N drops by the pool's worth of experts) and the DONE STAT expert-cache
#      hit rate within 0.5 points of the pristine's.
#
# GATE_TOOLS overrides the tool block. Iterate with ~/bench/owui_tools_4.json;
# the final gate uses the real 34-tool dump, whose cold prefill is ~24 min.
# COLI_CKPT_DIR is forced to this run's own directory: the serving checkpoint
# store under the snapshot is the gateway's, and step 5 must capture, not hit.
set -u
PRISTINE=${1:?pristine binary}; CAND=${2:?candidate binary}; TAG=${3:-p6bgate$(date +%m%d%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/bench/p6b_gate_$TAG
TOOLS=${GATE_TOOLS:-$HOME/bench/owui_tools.json}
SYSTEM=${GATE_SYSTEM:-$HOME/bench/p6_system.txt}
SLOTS=${P6B_SLOTS:-4}
SYNC_MS=${P6B_SYNC_MS:-1000}
mkdir -p "$OUT" "$OUT/ckpt"

# A step that leaves its engine behind makes the NEXT step refuse, and a revert
# that copies over a still-running binary fails with ETXTBSY -- which is how a
# failed gate left the candidate in service on 2026-09-07 (P7). Wait, every time.
wait_no_engine() {
  for _ in $(seq 1 180); do pgrep -x glm53 >/dev/null || return 0; sleep 1; done
  echo "REFUSED: a glm53 is still running after 180 s: $(pgrep -x glm53 | tr '\n' ' ')"
  return 1
}

# Step 4 talks to a live gateway; every other step needs the box to itself.
STEPS=${GATE_STEPS:-12356}
run_step() { case "$STEPS" in *"$1"*) return 0;; *) return 1;; esac; }
if [ "$STEPS" != 4 ] && pgrep -x glm53 >/dev/null; then
  echo "REFUSED: a glm53 is running -- stop the gateway first"; exit 2
fi
for f in "$TOOLS" "$SYSTEM"; do
  [ -s "$f" ] || { echo "REFUSED: missing fixture $f"; exit 2; }
done
[ -x "$PRISTINE" ] || { echo "REFUSED: no pristine binary at $PRISTINE"; exit 2; }
[ -x "$CAND" ]     || { echo "REFUSED: no candidate binary at $CAND"; exit 2; }

SHADERS=$HERE/../../c/shaders
# A COPY of the histogram, never the canonical file: the engine rewrites
# COLI_USAGE_PATH at exit and a benchmark must not teach the serving tier.
cp -f "$HOME/.glm53_explain.bin" "$OUT/hist.bin" 2>/dev/null || true

echo "=== p6b_gate $TAG $(date -Is)"
echo "    pristine=$PRISTINE ($(sha256sum "$PRISTINE" | cut -c1-16))"
echo "    candidate=$CAND ($(sha256sum "$CAND" | cut -c1-16))"
echo "    tools=$TOOLS ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$TOOLS") entries) system=$SYSTEM"
echo "    slots=$SLOTS steps=$STEPS ckpt dir=$OUT/ckpt sync bound=${SYNC_MS} ms"

rc1=0; rc2=0; rc3=0; rc4=0; rc5=0; rc6=0

# ---------------------------------------------------------------- (1)
echo
echo "### step 1 — tworeq at $SLOTS slots, both KDA knobs, and the guard must stay silent"
if run_step 1; then
for knob in 2 0; do
  echo "--- tworeq COLI_KDA_GPU=$knob TWOREQ_SLOTS=$SLOTS"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
      COLI_CKPT_DIR="$OUT/ckpt" \
      COLI_KDA_GPU=$knob TWOREQ_SLOTS=$SLOTS TWOREQ_EXE="$CAND" GLM53_VERBOSE=1 \
      python3 "$HERE/tworeq.py" > "$OUT/tworeq_kda$knob.txt" 2>&1
  r=$?
  forced=$(grep -c "forcing COLI_KDA_GPU=0" "$OUT/tworeq_kda$knob.txt")
  pool=$(grep -m1 "KDA slot pool" "$OUT/tworeq_kda$knob.txt" || echo "(no pool line)")
  echo "tworeq COLI_KDA_GPU=$knob: $(grep '^RESULT:' "$OUT/tworeq_kda$knob.txt" || echo 'NO RESULT') (rc=$r)"
  echo "    forcing lines: $forced   pool: $pool"
  [ "$r" = 0 ] || rc1=1
  # The guard firing at knob 2 means the run measured the CPU recurrence and
  # its IDENTICAL proves nothing about per-slot device state.
  if [ "$knob" = 2 ] && [ "$forced" != 0 ]; then
    echo "    FAIL: the pool did not cover $SLOTS slots -- this is the P6 behaviour, not P6b"
    rc1=1
  fi
  wait_no_engine || exit 2
done
else echo "(skipped)"; fi
echo "step 1 rc=$rc1"

# ---------------------------------------------------------------- (2)
echo
echo "### step 2 — prefill_gate.sh at one slot (the pool must be neutral)"
# GLM53_PREFIX_CKPT=0, and not for tidiness. prefill_gate runs the PRISTINE side
# first and the CANDIDATE second against ONE checkpoint directory, and its sizes
# 30/300/1000 share a prefix: with checkpoints on, the 1 000-token run plans an
# LCP capture at ~300 tokens, writes it to disk, and the candidate's FIRST
# 1 000-token run then RESTORES what the pristine's runs stored. The candidate
# would come out faster because it went second. P6b changes no prefill path, so
# the honest comparison is with the feature off -- the same reason p7_gate.sh
# step 1 turns it off.
if run_step 2; then
COLI_CKPT_DIR="$OUT/ckpt" GLM53_PREFIX_CKPT=0 MIN_SPEEDUP=${MIN_SPEEDUP:-0.97} \
  "$HERE/prefill_gate.sh" "$PRISTINE" "$CAND" "$TAG-one" 2>&1 | tee "$OUT/step2.txt"
rc2=${PIPESTATUS[0]}
wait_no_engine || exit 2
else echo "(skipped)"; fi
echo "step 2 rc=$rc2"

# ---------------------------------------------------------------- (3)
echo
echo "### step 3 — decode at $SLOTS slots vs the pristine at 1 and at $SLOTS"
decode_row() {   # <label> <binary> <slots>
  local label=$1 bin=$2 slots=$3
  echo "--- $label ($(basename "$bin"), --kv-slots $slots, COLI_KDA_GPU=2)"
  env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
      COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
      COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
      COLI_CKPT_DIR="$OUT/ckpt" COLI_KDA_GPU=2 GLM53_VERBOSE=1 \
      GLM53_PREFIX_CKPT=0 \
      python3 "$HERE/ttft_serve.py" --engine "$bin" --kv-slots "$slots" \
        --sizes 30 --repeat 2 --gen 64 --warm --min-resident 96 \
        --tag "$TAG-$label" --json "$OUT/decode.jsonl" \
        --engine-log "$OUT/engine_$label.log" 2>&1 | tee "$OUT/decode_$label.txt"
  local r=${PIPESTATUS[0]}
  grep -m1 "KDA slot pool\|forcing COLI_KDA_GPU" "$OUT/engine_$label.log" || true
  wait_no_engine || exit 2
  return $r
}
if run_step 3; then
decode_row cand4 "$CAND" "$SLOTS"     || rc3=2
decode_row pris1 "$PRISTINE" 1        || rc3=2
decode_row pris4 "$PRISTINE" "$SLOTS" || rc3=2
[ "$rc3" = 2 ] && { echo "REFUSED: a decode row did not run"; exit 2; }
python3 - "$OUT/decode.jsonl" "$TAG" <<'PY' && rc3=0 || rc3=3
import json, sys, statistics as st
recs = [json.loads(l) for l in open(sys.argv[1])]
tag = sys.argv[2]
def dec(label):
    return [r["decode_tps"] for r in recs
            if r["tag"] == f"{tag}-{label}" and r["kind"] == "ttft" and r.get("decode_tps")]
rows = {k: dec(k) for k in ("cand4", "pris1", "pris4")}
print("%-34s %-22s %10s" % ("row", "decode tok/s (both runs)", "median"))
for k, name in (("cand4", "candidate, 4 slots, GPU"),
                ("pris1", "pristine, 1 slot, GPU"),
                ("pris4", "pristine, 4 slots (CPU forced)")):
    v = rows[k]
    print("%-34s %-22s %10s" % (name, " / ".join("%.3f" % x for x in v) or "-",
                                ("%.3f" % st.median(v)) if v else "-"))
if not all(rows.values()):
    print("step 3: FAIL (a row is missing)"); sys.exit(1)
m = {k: st.median(v) for k, v in rows.items()}
vs1 = m["cand4"] / m["pris1"]; vs4 = m["cand4"] / m["pris4"]
print("candidate/pristine-1slot = %.3fx (need >= 0.99)" % vs1)
print("candidate/pristine-4slot = %.3fx (need >= 1.08)" % vs4)
ok = vs1 >= 0.99 and vs4 >= 1.08
print("step 3:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
else echo "(skipped)"; fi
echo "step 3 rc=$rc3"

# ---------------------------------------------------------------- (4)
echo
echo "### step 4 — the live side-request case through the gateway"
if run_step 4; then
  if [ -z "${P6B_URL:-}" ]; then
    echo "step 4 needs a serving gateway: set P6B_URL (the chain runs it after serving)"
    rc4=2
  else
    rm -f "$OUT/live.jsonl"
    python3 "$HERE/ttft_serve.py" --url "$P6B_URL" --multiturn --side-request \
        --system "$SYSTEM" --sizes 300 --repeat 0 --gen 16 \
        --warm --min-resident 96 --tag "$TAG-live" --json "$OUT/live.jsonl" \
        2>&1 | tee "$OUT/step4.txt"
    rc4=${PIPESTATUS[0]}
    # --multiturn PRINTS its ratio but appends no verdict, so the harness exits
    # 0 even when turn 2 re-prefilled the whole history. The gate judges the
    # engine's own REUSE count, which is the authoritative answer (P1).
    python3 - "$OUT/live.jsonl" "$TAG" <<'PY' || rc4=1
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
tag = sys.argv[2]
def one(kind):
    r = [x for x in recs if x["tag"] == f"{tag}-live" and x["kind"] == kind]
    return r[-1] if r else None
t1, sd, t2 = one("turn1"), one("side"), one("turn2")
print("%-28s %8s %8s %10s" % ("request", "prompt", "reused", "ttft"))
for name, r in (("turn 1 [A]", t1), ("side (title)", sd), ("turn 2 [A, reply, B]", t2)):
    if r: print("%-28s %8s %8s %9.2fs" % (name, r["prompt_tokens"], r["reused"], r["ttft_s"] or 0))
ok = bool(t1 and t2 and t2.get("reused") and t2["prompt_tokens"])
if ok:
    frac = t2["reused"] / t2["prompt_tokens"]
    ratio = (t2["ttft_s"] / t1["ttft_s"]) if t1["ttft_s"] and t2["ttft_s"] else 9.9
    print("turn 2 reused %d/%d = %.0f%% of its prompt; ttft(2)/ttft(1) = %.3f"
          % (t2["reused"], t2["prompt_tokens"], 100 * frac, ratio))
    ok = frac >= 0.8 and ratio <= 0.35
else:
    print("turn 2 reused nothing -- the side request took the slot, which is what P6 fixed")
print("step 4 reuse:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
    # The server log must not carry the guard: serving at 4 slots with
    # COLI_KDA_GPU=2 and a "forcing" line means the pool did not cover the
    # gateway's slots and the whole item silently did not ship.
    forced=$(grep -c "forcing COLI_KDA_GPU=0" "${P6B_SERVER_LOG:-$HOME/glm53_server.log}" || true)
    echo "server log 'forcing COLI_KDA_GPU=0' lines: $forced"
    grep -E "REUSE|KV_SLOTS=|KDA slot pool" \
        "${P6B_SERVER_LOG:-$HOME/glm53_server.log}" | tail -12 || true
    [ "$forced" = 0 ] || rc4=1
  fi
else echo "(skipped)"; fi
echo "step 4 rc=$rc4"

# ---------------------------------------------------------------- (5)
echo
echo "### step 5 — device->host sync speed at the GPU knob (CKPT store sync=…)"
# A fresh, private checkpoint dir, so the tool-block prefix is CAPTURED here
# and not restored from the gateway's store: only a capture runs the sync.
if run_step 5; then
rm -f "$OUT/ckpt"/*.bin
env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
    COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto \
    COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
    COLI_VK_SHADERS="$SHADERS" COLI_USAGE_PATH="$OUT/hist.bin" \
    COLI_CKPT_DIR="$OUT/ckpt" COLI_KDA_GPU=2 GLM53_PREFIX_CKPT=1 GLM53_VERBOSE=1 \
    python3 "$HERE/ttft_serve.py" --engine "$CAND" --prefix-ckpt \
      --tools "$TOOLS" --system "$SYSTEM" --kv-slots "$SLOTS" \
      --sizes "" --repeat 0 --warm --min-resident 96 \
      --tag "$TAG-sync" --json "$OUT/sync.jsonl" \
      --engine-log "$OUT/engine_sync.log" 2>&1 | tee "$OUT/step5.txt"
r5=${PIPESTATUS[0]}
wait_no_engine || exit 2
grep "CKPT store" "$OUT/engine_sync.log" || true
python3 - "$OUT/engine_sync.log" "$SYNC_MS" <<'PY' && rc5=0 || rc5=3
import re, sys
pat = re.compile(r"CKPT store prefix=(\d+) (\d+) MB kind=(\d+) slot=(\d+) sync=(\d+) ms copy=(\d+) ms")
rows = [m.groups() for m in (pat.search(l) for l in open(sys.argv[1])) if m]
bound = float(sys.argv[2])
print("%10s %8s %10s %10s %14s" % ("prefix", "MB", "sync ms", "copy ms", "MB/s (sync)"))
ok = bool(rows)
for p, mb, kind, slot, sync, copy in rows:
    r = (float(mb) / (float(sync) / 1000.0)) if float(sync) > 0 else float("inf")
    print("%10s %8s %10s %10s %14.0f" % (p, mb, sync, copy, r))
    if float(sync) > bound: ok = False
if not rows: print("no CKPT store line -- nothing was captured, so nothing was measured")
print("step 5:", "PASS" if ok else "FAIL (sync must be under %.0f ms)" % bound)
sys.exit(0 if ok else 1)
PY
[ "$r5" = 0 ] || echo "(note: ttft_serve --prefix-ckpt returned $r5; step 5 judges the sync line only)"
else echo "(skipped)"; fi
echo "step 5 rc=$rc5"

# ---------------------------------------------------------------- (6)
echo
echo "### step 6 — what the pool cost: preload count and expert-cache hit rate"
if run_step 6; then
python3 - "$OUT" "$TAG" <<'PY' && rc6=0 || rc6=3
import json, os, re, sys, statistics as st
out, tag = sys.argv[1], sys.argv[2]
pre = re.compile(r"\[VK\] preload: (\d+) heat-ranked experts")
pool = re.compile(r"\[VK\] KDA slot pool: (\d+) x ([0-9.]+) MB")
def scan(name):
    p = os.path.join(out, name)
    if not os.path.exists(p): return None, None
    n = s = None
    for line in open(p, errors="replace"):
        m = pre.search(line)
        if m: n = int(m.group(1))
        m = pool.search(line)
        if m: s = (int(m.group(1)), float(m.group(2)))
    return n, s
cn, cp = scan("engine_cand4.log")
pn, _  = scan("engine_pris1.log")
print("%-34s %10s %s" % ("engine", "preload", "KDA slot pool"))
print("%-34s %10s %s" % ("pristine, 1 slot", pn, "-"))
print("%-34s %10s %s" % ("candidate, 4 slots", cn,
                         ("%d x %.0f MB" % cp) if cp else "(none)"))
ok = cn is not None and pn is not None and cp is not None
if ok:
    print("experts given up for the pool: %d (%.1f%% of the pristine tier)"
          % (pn - cn, 100.0 * (pn - cn) / pn))
    # The pool is 4 x 156 MB = 625 MB against ~14.2 MB per expert = ~44.
    if not (0 <= pn - cn <= 120): print("  unexpected: the drop is not the pool's worth"); ok = False
try:
    recs = [json.loads(l) for l in open(os.path.join(out, "decode.jsonl"))]
except OSError:
    recs = []
def hits(label):
    return [r["hit_pct"] for r in recs
            if r["tag"] == f"{tag}-{label}" and r.get("hit_pct") is not None]
hc, hp = hits("cand4"), hits("pris1")
print("%-34s %10s" % ("hit rate, candidate 4 slots", ("%.2f%%" % st.median(hc)) if hc else "-"))
print("%-34s %10s" % ("hit rate, pristine 1 slot", ("%.2f%%" % st.median(hp)) if hp else "-"))
if hc and hp:
    d = st.median(hc) - st.median(hp)
    print("hit-rate delta %+.2f points (bound: 0.5)" % d)
    if abs(d) > 0.5: ok = False
else:
    print("hit rate not recorded -- step 6 cannot judge routing"); ok = False
print("step 6:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
else echo "(skipped)"; fi
echo "step 6 rc=$rc6"

# ---------------------------------------------------------------- verdict
echo
echo "=== p6b_gate $TAG verdict $(date -Is)"
printf "  step 1 tworeq %d slots, both knobs, no guard  rc=%s\n" "$SLOTS" "$rc1"
printf "  step 2 prefill_gate, one slot, neutral        rc=%s\n" "$rc2"
printf "  step 3 decode at %d slots                     rc=%s\n" "$SLOTS" "$rc3"
printf "  step 4 live side-request through the gateway  rc=%s%s\n" "$rc4" \
       "$(run_step 4 || echo ' (not selected; the chain runs it)')"
printf "  step 5 device->host sync speed                rc=%s\n" "$rc5"
printf "  step 6 preload count and hit rate             rc=%s\n" "$rc6"
echo "  outputs in $OUT"
[ "$rc2" = 2 ] && exit 2
[ "$rc4" = 2 ] && exit 2
[ "$rc1" = 0 ] || exit 1
[ "$rc2" = 1 ] && exit 1
[ "$rc4" = 1 ] && exit 1
[ "$rc4" = 0 ] || exit 1
[ "$rc2" = 0 ] || exit 3
[ "$rc3" = 0 ] || exit 3
[ "$rc5" = 0 ] || exit 3
[ "$rc6" = 0 ] || exit 3
exit 0
