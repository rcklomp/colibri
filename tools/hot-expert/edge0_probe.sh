#!/bin/bash
# edge0_probe.sh -- EDGE0 evaluation, technique #1 (prerouter) re-framed as a
# CORRECTNESS question: does GLM-5.3's output depend on the TIER CONFIGURATION,
# and could a routing predictor that pins experts make it not depend on it?
#
# §G15 measured residency-dependence with the PLACEMENT-FORCING knob
# (GLM53_EXPERTS_CPU=1/2: all experts CPU-clamped vs all experts CPU-unclamped).
# It never varied the tier itself. This probe does: same binary, same weights,
# same prompt, same frozen histogram -- only COLI_VK_EXPERTS2/3 changes, which
# is the one variable that decides which experts are GPU-resident and therefore
# which ones take the UNCLAMPED routed kernel (c/shaders/qmatmul_gate_up.comp)
# instead of the CPU's swiglu_clamped(limit=10.0).
#
# NO ENGINE CODE CHANGES. Every run is the in-service binary; the probe is
# entirely an environment A/B. That is deliberate: if a pure configuration
# change moves teacher_forcing, the gap is a property of the shipped engine and
# not of anything this evaluation built.
#
#   A   COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695  -- the recorded standard
#   B   COLI_VK_EXPERTS2=0    COLI_VK_EXPERTS3=0     -- dev2/dev3 skipped
#                                                       entirely (what an unset
#                                                       cap gives you, per
#                                                       CLAUDE.md), dev0 only
#   C   COLI_VK_EXPERTS2=848  COLI_VK_EXPERTS3=848   -- half tier, the midpoint
#
# NOTE ON COMPARABILITY: B and C are deliberately OFF the recorded baseline.
# CLAUDE.md says always pass 1695/1695 because changing the tier changes
# routing -- that is exactly the effect under test here. NONE of these runs
# produces a tok/s number and none may be compared to a throughput row.
#
# Pre-committed decision rule, written before the first run:
#   * null control sA1 vs sA2 MUST be bit-identical (cos 1.000000000, relL2 0,
#     teacher_forcing identical). If it is not, the engine is not deterministic
#     under a frozen histogram and the whole probe is void -- stop.
#   * the A-vs-B gap MUST NOT exceed §G15's extreme (EXPERTS_CPU=1 vs =2:
#     6 of 42 short, 8 of 1232 long, cos 0.992324 / 0.981143), because A-vs-B
#     moves a STRICT SUBSET of the activations across the clamp boundary that
#     §G15's knob moved. If it does exceed it, the clamp is not the whole
#     mechanism, this probe's model is wrong, and the result is reported as a
#     contradiction rather than tuned.
#   * falsification arm for the pinning idea: A and B re-run with
#     GLM53_EXPERTS_CPU=2. Under that knob the tier is not consulted at all, so
#     A and B must be bit-identical. If they are NOT, there is a residency-set
#     effect independent of the clamp and edge0 #1 earns a second look.
#
# Oracles, per CLAUDE.md: greedy text over 128 tokens, the teacher_forcing line,
# last_logits (cosine / relL2 / max-abs / argmax). No speed is measured.
#
# Also captured per run, because it is the mediating variable:
#   [PROF] eg=...(experts=N) cpu=...(n=K)  ->  N/(N+K) is the fraction of
#   routed-expert evaluations that took the UNCLAMPED GPU kernel.
#
# Runs the owner's gateway down for its duration: takes ~/bench/.rig.lock first
# so gateway_watchdog.sh does not race back in, and restores the exact binary
# that was in service on every exit path.
#
# Usage (on the rig):  ~/bench/edge0_probe.sh [short|long|control|all|a,b]
set -u

PHASE="${1:-all}"
SRC="$HOME/src/colibri"
M="$HOME/models/GLM-5.3-Flash-colibri-int4-g64"
OUT="$HOME/bench/edge0_probe_out"

mkdir -p "$OUT"
. "$SRC/tools/hot-expert/rig_lock.sh"

RESTART_GATEWAY=0
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  pkill -9 -x glm53 2>/dev/null || true
  for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
  if [ "$RESTART_GATEWAY" = 1 ]; then
    echo "[edge0] restarting the owner's gateway"
    SKIP_WARM=1 setsid nohup "$HOME/start_glm53.sh" > "$HOME/glm53_server.log" 2>&1 < /dev/null &
    KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
    for _ in $(seq 1 90); do
      sleep 10
      code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -H "Authorization: Bearer $KEY" \
               http://127.0.0.1:8081/v1/models 2>/dev/null) || true
      [ "${code:-}" = 200 ] && break
    done
    echo "[edge0] gateway back, /v1/models=${code:-?}"
  fi
  rig_lock_release
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

rig_lock_take edge0probe || exit 3

for _eng in qwen38 qwen38-vk; do
  if pgrep -x "$_eng" >/dev/null 2>&1; then echo "[edge0] $_eng is running -- refusing"; exit 1; fi
done
if pgrep -x glm53 >/dev/null 2>&1; then
  if pgrep -f "openai_[s]erver.py" >/dev/null 2>&1; then
    echo "[edge0] stopping the owner's gateway for the duration of this probe"
    pkill -f "openai_[s]erver.py" || true
    sleep 2
    pkill -9 -x glm53 || true
    for _ in $(seq 1 20); do pgrep -x glm53 >/dev/null 2>&1 || break; sleep 1; done
    RESTART_GATEWAY=1
  else
    echo "[edge0] a glm53 is running that is not the gateway -- refusing"; exit 1
  fi
fi

# One binary for every run. No build, no candidate: the whole point is that the
# SHIPPED engine is what shows the effect.
cp -f "$SRC/c/glm53" "$HOME/bench/glm53.edge0base"
BIN="$HOME/bench/glm53.edge0base"
echo "[edge0] binary sha256=$(sha256sum "$BIN" | cut -c1-16)  (the in-service glm53)"

# GLM must own the page cache. Qwen work evicts it; a cold run is not a result
# (CLAUDE.md: 100k+ major faults, not a baseline). This probe compares two runs
# of the same binary, so a cold run does not bias the ORACLE the way it biases a
# timing -- but a run that spends 10 minutes in major faults per configuration
# turns a 3-hour probe into a 6-hour one, so it is checked and reported.
resid() {
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/edge0_fincore.txt 2>/dev/null || return 0
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/edge0_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[edge0 resid $1] shards={n} resident={res/tot*100:.3f}% short={short}')
"
}
resid before

P_SHORT=$(cat "$HOME/bench/prompt_glm.txt")
P_LONG=$(python3 -c "print((open('$HOME/bench/prompt_glm.txt').read().strip()+' ')*30)")

run() {                      # run <tag> <greedy> <prompt> <e2> <e3> [ENV=V ...]
  local tag="$1" greedy="$2" prompt="$3" e2="$4" e3="$5"; shift 5
  echo "[edge0] $(date +%H:%M:%S) run $tag  (EXPERTS2=$e2 EXPERTS3=$e3 $*)"
  cp -f "$HOME/.glm53_explain.bin" "/tmp/edge0_hist.bin"   # frozen per run: the
  rm -rf /tmp/edge0_ckpt; mkdir -p /tmp/edge0_ckpt         # engine rewrites it at exit
  ( export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
    export COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
    export COLI_VK_SHADERS="$SRC/c/shaders"
    export COLI_VK_EXPERTS2="$e2" COLI_VK_EXPERTS3="$e3"
    export COLI_USAGE_PATH=/tmp/edge0_hist.bin
    export GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=/tmp/edge0_ckpt
    export COLI_TIMERS=1 GLM53_VERBOSE=1
    for kv in "$@"; do export "${kv?}"; done
    "$BIN" --model "$M" --prompt "$prompt" --greedy "$greedy" --logits 512 ) \
      > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  local rc=$?
  echo "[edge0] $(date +%H:%M:%S) run $tag rc=$rc  $(wc -c < "$OUT/$tag.out") bytes"
  # the tier that actually came up -- a config that silently did not apply is
  # the single most likely way this probe reports a fake zero or a fake gap
  grep -E '^\[VK\] preload' "$OUT/$tag.err" | tail -4
  # the mediating variable: how many routed-expert evaluations took the
  # UNCLAMPED GPU kernel versus the clamped CPU one
  grep -E '^\[PROF\] eg=' "$OUT/$tag.err" | tail -1
  return $rc
}

want() { case ",$PHASE," in *,"$1",*|*,all,*) return 0;; esac; return 1; }

A2=1695; A3=1695
B2=0;    B3=0
C2=848;  C3=848

if want short; then
  run sA1 128 "$P_SHORT" $A2 $A3
  run sA2 128 "$P_SHORT" $A2 $A3          # null control: must equal sA1 exactly
  run sB1 128 "$P_SHORT" $B2 $B3
  run sB2 128 "$P_SHORT" $B2 $B3          # headline repeat
  run sC  128 "$P_SHORT" $C2 $C3
fi

if want long; then
  run lA 0 "$P_LONG" $A2 $A3
  run lB 0 "$P_LONG" $B2 $B3
  run lC 0 "$P_LONG" $C2 $C3
fi

# The falsification arm. GLM53_EXPERTS_CPU=2 puts every routed expert on the
# CPU with the swiglu clamp DISABLED, i.e. every expert computes what the GPU
# kernel computes and the tier is never consulted. If the tier configuration
# still changes the output under this knob, the effect is NOT the clamp and the
# pinning idea is back on the table.
if want control; then
  run sA_nc 128 "$P_SHORT" $A2 $A3 GLM53_EXPERTS_CPU=2
  run sB_nc 128 "$P_SHORT" $B2 $B3 GLM53_EXPERTS_CPU=2
fi

echo
echo "=== EDGE0 #1 correctness probe ==="
python3 "$SRC/tools/hot-expert/edge0_compare.py" "$OUT"
