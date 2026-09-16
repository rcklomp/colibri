#!/bin/bash
# x1_probe.sh -- graft X1 (FRANKEN-ENGINE-PLAN-2026-09-15.md §4): retained,
# re-submittable Vulkan command buffers for the per-token-invariant stream.
#
# The knob is COLI_VK_RETAIN_CB (backend_vulkan.c, off by default). Both engines
# build from that file, so both are measured here.
#
#   step0   the isolated microbenchmark, no model: `rome_vkbench <spv> x1` runs
#           the SAME sites with the knob off and on, interleaved inside one
#           process, at the engine's own shapes and inter-submit gaps. This is
#           the item's ceiling: if the per-submit saving here does not reach the
#           projected -3..-8 ms/token, the engine cannot either.
#   qwen    qwen38-vk oracle + [OPTIME]. The retained sites on this engine are
#           Q7's dense stream, which exists only under Q38_DENSE_GPU, so the A/B
#           runs at mask 7 (the Q7 arm). Mask 0 is checked too, to prove the
#           default path is untouched.
#   glm     glm53 oracle + [OPTIME] at COLI_KDA_GPU=2 (the serving knob, and the
#           only setting under which coli_vk_kda_layer is called at all).
#
# Every leg is bit-identity: same shaders, same push constants, same dispatch
# order, same buffers. A knob-on run that is not byte-identical to the pristine
# is a failure of the item, not a numerics trade to be priced.
#
# Usage (on the rig):  x1_probe.sh [step0|qwen|glm|all]
#   WT=<worktree>    candidate tree   (default ~/src/colibri-x1)
#   PT=<worktree>    pristine tree    (default ~/src/colibri-x1pris)
set -u
WT=${WT:-/home/ronald/src/colibri-x1}
PT=${PT:-/home/ronald/src/colibri-x1pris}
OUT=${OUT:-~/bench/x1_out}; mkdir -p "$OUT"
. /home/ronald/bench/q7_lib.sh          # log, rig lock, gateway stop/restart
. "$WT/tools/hot-expert/gate_lib.sh"    # gate_compare, gate_ab_verdict

PHASE="${1:-all}"
QSNAP=~/models/Qwen3.8-Flash-Next-FP8
GSNAP=~/models/GLM-5.3-Flash-colibri-int4-g64

trap q7_on_exit EXIT INT TERM HUP
if [ -n "${X1_PARENT_LOCK:-}" ]; then
  holder=$(rig_lock_holder) || { log "REFUSED: X1_PARENT_LOCK set but no live lock holder"; exit 3; }
  hpid=$(echo "$holder" | awk '{print $2}')
  [ "$hpid" = "$X1_PARENT_LOCK" ] || { log "REFUSED: rig lock held by [$holder], not by $X1_PARENT_LOCK"; exit 3; }
  log "rig lock and gateway delegated by parent pid $X1_PARENT_LOCK"
else
  q7_take_lock "x1-$PHASE" 240 || exit 3
fi
for e in qwen38 qwen38-vk; do pgrep -x "$e" >/dev/null && { log "REFUSED: $e running"; exit 1; }; done
stop_gateway || exit 1

log "candidate $(git -C "$WT" log --oneline -1)"
log "pristine  $(git -C "$PT" log --oneline -1)"
for t in "$WT" "$PT"; do
  sha256sum "$t/c/glm53" "$t/c/qwen38-vk" "$t/c/shaders/qmatmul.spv" | sed 's/^/  /'
done

# ---------------------------------------------------------------- step 0 ----
phase_step0(){
  gcc -O2 -march=native -fopenmp -I"$WT/c" -o /tmp/x1_vkbench \
      "$WT/tools/hot-expert/rome_vkbench.c" "$WT/c/backend_vulkan.o" -lvulkan -lm \
    || { log "vkbench build failed"; return 1; }
  for r in 1 2; do
    log "--- rome_vkbench x1 repeat $r"
    /tmp/x1_vkbench "$WT/c/shaders/qmatmul.spv" x1 2>&1 | tee "$OUT/vkbench-x1-$r.log"
  done
  # The same run with the phase split on: how many microseconds of a submit are
  # OUR descriptor writes and recording (what a retained buffer can remove) and
  # how many are the driver's submit and the GPU wait (what it cannot). This is
  # the item's ceiling, measured directly instead of inferred from an A/B at the
  # resolution limit.
  log "--- rome_vkbench x1 with VK_PROF=1 (phase split)"
  VK_PROF=1 /tmp/x1_vkbench "$WT/c/shaders/qmatmul.spv" x1 2>&1 | tee "$OUT/vkbench-x1-prof.log"
}

# ----------------------------------------------------------------- qwen ----
phase_qwen(){
  local SHORT=/tmp/x1_prompt_short.txt LONG=/tmp/x1_prompt_long.txt
  cat > "$SHORT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example with two transactions and two rows, and compare two practical prevention strategies in detail.
EOF
  : > "$LONG"; for _ in $(seq 1 40); do cat "$SHORT" >> "$LONG"; done

  python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(GLM) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/GLM-5.3-Flash-colibri-int4-g64")))
PY
  log "warming Qwen"
  find "$QSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
  fincore --bytes --output FILE,SIZE,RES "$QSNAP"/*.safetensors > /tmp/x1_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/x1_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
print(f'[resid qwen] shards={n} resident={res/tot*100:.4f}% short={short}')
assert n==131 and res/tot>0.9, 'NOT RESIDENT'
" || { log "REFUSED: Qwen not resident"; return 1; }

  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
  qrun(){   # qrun <tree> <tag> <prompt> <n_new> [KEY=VAL ...]
    local tree="$1" tag="$2" prompt="$3" nn="$4"; shift 4
    cp -f "$QSNAP/.coli_usage" /tmp/x1_hist.bin
    log "--- $tag n_new=$nn $*"
    env SNAP="$QSNAP" COLI_USAGE=/tmp/x1_hist.bin N_NEW="$nn" NOSTREAM=1 \
        COLI_VK_SHADERS="$tree/c/shaders" \
        Q38_TF=1 Q38_TF_DUMP="$OUT/$tag.tf.f32" DUMP="$OUT/$tag.last.f32" \
        "$@" "$tree/c/qwen38-vk" 512 8 "$prompt" > "$OUT/$tag.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -20 "$OUT/$tag.log"; exit 1; }
    grep -E "Vulkan tier preloaded|Q7 dense|RCB|^Speed:" "$OUT/$tag.log" | sed 's/^/  /'
  }
  # oracle: mask 7 (where the retained sites are) and mask 0 (the default path)
  qrun "$PT" q-pris-m7  "$SHORT" 128 Q38_DENSE_GPU=7
  qrun "$WT" q-off-m7   "$SHORT" 128 Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=0
  qrun "$WT" q-on-m7    "$SHORT" 128 Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=1
  qrun "$PT" q-pris-m0  "$SHORT" 128
  qrun "$WT" q-on-m0    "$SHORT" 128 COLI_VK_RETAIN_CB=1
  qrun "$PT" q-pris-lng "$LONG"  1   Q38_DENSE_GPU=7
  qrun "$WT" q-on-lng   "$LONG"  1   Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=1
  # [OPTIME], three repeats a side, interleaved
  for r in 1 2 3; do
    qrun "$WT" q-t-off-$r "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=0
    qrun "$WT" q-t-on-$r  "$SHORT" 32 COLI_TIMERS=1 Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=1
  done
  log "--- qwen oracle"
  for leg in q-off-m7 q-on-m7; do
    cmp -s "$OUT/q-pris-m7.last.f32" "$OUT/$leg.last.f32" \
      && echo "  $leg last_logits   BIT-IDENTICAL to pristine ($(stat -c%s "$OUT/$leg.last.f32") B)" \
      || echo "  $leg last_logits   DIFFERS"
    cmp -s "$OUT/q-pris-m7.tf.f32" "$OUT/$leg.tf.f32" \
      && echo "  $leg teacher_force BIT-IDENTICAL to pristine" || echo "  $leg teacher_force DIFFERS"
    # the GENERATED TEXT, not a line that carries a timing: the first version of
    # this compared "^Speed:" and reported DIFFERS on two byte-identical runs.
    sed -n '/^Generated (/,/^TTFT:/p' "$OUT/q-pris-m7.log" | grep -v '^TTFT:' > /tmp/x1_txt_a
    sed -n '/^Generated (/,/^TTFT:/p' "$OUT/$leg.log"      | grep -v '^TTFT:' > /tmp/x1_txt_b
    if [ ! -s /tmp/x1_txt_a ] || [ ! -s /tmp/x1_txt_b ]; then
      echo "  $leg greedy text  REFUSED: one side produced no text"
    elif cmp -s /tmp/x1_txt_a /tmp/x1_txt_b; then
      echo "  $leg greedy text  IDENTICAL ($(wc -l < /tmp/x1_txt_a) lines)"
    else
      echo "  $leg greedy text  DIFFERS"
    fi
  done
  cmp -s "$OUT/q-pris-m0.last.f32" "$OUT/q-on-m0.last.f32" \
    && echo "  q-on-m0  last_logits   BIT-IDENTICAL to pristine (default path)" || echo "  q-on-m0 DIFFERS"
  cmp -s "$OUT/q-pris-lng.last.f32" "$OUT/q-on-lng.last.f32" \
    && echo "  q-on-lng last_logits   BIT-IDENTICAL to pristine (long prompt)" || echo "  q-on-lng DIFFERS"
  # tworeq: three requests in ONE persistent engine, knob on
  log "--- tworeq qwen38, knob on"
  TWOREQ_ARCH=qwen38 TWOREQ_SNAP="$QSNAP" TWOREQ_EXE="$WT/c/qwen38-vk" \
    COLI_VK_SHADERS="$WT/c/shaders" Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=1 \
    python3 "$WT/tools/hot-expert/tworeq.py" 2>&1 | tail -12 | tee "$OUT/q-tworeq.log"
}

# ------------------------------------------------------------------ glm ----
phase_glm(){
  python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.expanduser("~/src/colibri/c/tools"))
import datapoint
print("evict_cache(Qwen) ->", datapoint.evict_cache(247.0, snap_dir=os.path.expanduser("~/models/Qwen3.8-Flash-Next-FP8")))
PY
  log "warming GLM"
  find "$GSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
  local PROMPT=$HOME/bench/prefill_prompt_600.txt
  export COLI_CKPT_DIR=/tmp/x1_ckpt; mkdir -p "$COLI_CKPT_DIR"
  export GLM53_PREFIX_CKPT=0
  grun(){   # grun <tree> <tag> [KEY=VAL ...]
    local tree="$1" tag="$2"; shift 2
    log "--- $tag $*"
    env "$@" COLI_VK_SHADERS="$tree/c/shaders" \
        "$WT/tools/hot-expert/prefill_profile.sh" "$tree/c/glm53" "$PROMPT" "x1-$tag" \
        > "$OUT/$tag.txt" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || { log "REFUSED: $tag exited $rc"; tail -20 "$OUT/$tag.txt"; exit 1; }
    grep -E "^teacher_forcing" "$HOME/bench/prefill_profile_x1-$tag.log" > "$OUT/$tag.tf.txt"
    grep -E "^last_logits"     "$HOME/bench/prefill_profile_x1-$tag.log" > "$OUT/$tag.lg.txt"
    grep -E "RCB|OPTIME" "$HOME/bench/prefill_profile_x1-$tag.log" | tail -4 | sed 's/^/  /'
    wait_no_engine glm53 || exit 1
  }
  grun "$PT" g-pris  COLI_KDA_GPU=2
  grun "$WT" g-off   COLI_KDA_GPU=2 COLI_VK_RETAIN_CB=0
  grun "$WT" g-on    COLI_KDA_GPU=2 COLI_VK_RETAIN_CB=1
  grun "$WT" g-on-c0 COLI_KDA_GPU=0 COLI_VK_RETAIN_CB=1     # the CPU-recurrence default path
  grun "$PT" g-pris-c0 COLI_KDA_GPU=0
  log "--- glm oracle (pristine vs candidate, KDA chain on)"
  for leg in g-off g-on; do
    gate_compare "$leg teacher_forcing" "$OUT/g-pris.tf.txt" "$OUT/$leg.tf.txt" "teacher_forcing"
    gate_compare "$leg last_logits"     "$OUT/g-pris.lg.txt" "$OUT/$leg.lg.txt" "last_logits"
  done
  gate_compare "g-on-c0 teacher_forcing" "$OUT/g-pris-c0.tf.txt" "$OUT/g-on-c0.tf.txt" "teacher_forcing"
  gate_compare "g-on-c0 last_logits"     "$OUT/g-pris-c0.lg.txt" "$OUT/g-on-c0.lg.txt" "last_logits"
  log "--- tworeq glm53, 4 slots, knob on and off"
  for k in 0 1; do
    TWOREQ_SLOTS=4 TWOREQ_EXE="$WT/c/glm53" COLI_VK_SHADERS="$WT/c/shaders" \
      COLI_KDA_GPU=2 COLI_VK_RETAIN_CB=$k COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 \
      python3 "$WT/tools/hot-expert/tworeq.py" 2>&1 | tail -8 | tee "$OUT/g-tworeq-$k.log"
    wait_no_engine glm53 || exit 1
  done
}

# ----------------------------------------------------------------- prof ----
# The phase split IN THE ENGINE, where the CPU is busy and its caches are not
# the microbenchmark's. Three questions, all answered by VK_PROF=1:
#   [VK_PROF mm] / [VK_PROF kda]  what X1 removes, per submit, in microseconds;
#   [VK_PROF]                     the expert group's own desc/record share --
#                                 the bucket the -3..-8 ms/token projection was
#                                 taken from, and the one X1 deliberately leaves
#                                 dynamic;
#   n per token                   how many retained submits a token really has.
phase_prof(){
  export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
  if [ "${1:-both}" != glm ]; then
    cp -f "$QSNAP/.coli_usage" /tmp/x1_hist.bin
    log "--- qwen38-vk VK_PROF, mask 7, 32 tokens"
    env SNAP="$QSNAP" COLI_USAGE=/tmp/x1_hist.bin N_NEW=32 NOSTREAM=1 VK_PROF=1 \
        Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto COLI_VK_SHADERS="$WT/c/shaders" \
        COLI_TIMERS=1 Q38_DENSE_GPU=7 COLI_VK_RETAIN_CB=0 \
        "$WT/c/qwen38-vk" 512 8 /tmp/x1_prompt_short.txt > "$OUT/prof-qwen-off.log" 2>&1
    grep -E "VK_PROF|OPTIME" "$OUT/prof-qwen-off.log" | tail -25 | sed 's/^/  /'
  fi
  if [ "${1:-both}" != qwen ]; then
    # glm53 has its OWN cli (--model/--prompt/--greedy/--logits); the first
    # version of this leg passed qwen38's positional form and the engine exited
    # on "unknown argument". And it must DECODE, not just prefill: the retained
    # KDA site is coli_vk_kda_layer, which is the single-token path (prefill
    # chunks go through coli_vk_kda_step_rows instead).
    find "$GSNAP" -type f -name "*.safetensors" -print0 | while IFS= read -r -d '' f; do cat "$f" >/dev/null; done
    cp -f ~/.glm53_explain.bin /tmp/x1_glmhist.bin
    for k in 0 1; do
      log "--- glm53 VK_PROF decode, KDA chain on, COLI_VK_RETAIN_CB=$k"
      env OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
          COLI_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto COLI_VK_SHADERS="$WT/c/shaders" \
          COLI_VK_EXPERTS2=1695 COLI_VK_EXPERTS3=1695 COLI_USAGE_PATH=/tmp/x1_glmhist.bin \
          COLI_KDA_GPU=2 COLI_VK_RETAIN_CB=$k VK_PROF=1 COLI_TIMERS=1 GLM53_VERBOSE=1 \
          GLM53_PREFIX_CKPT=0 COLI_CKPT_DIR=/tmp/x1_ckpt DUMP="$OUT/prof-glm-$k.last.f32" \
          "$WT/c/glm53" --model "$GSNAP" --prompt "$(cat "$HOME/bench/prefill_prompt_600.txt")" \
          --greedy 64 --logits 512 > "$OUT/prof-glm-$k.log" 2>&1
      grep -E "VK_PROF (kda|mm)" "$OUT/prof-glm-$k.log" | tail -8 | sed 's/^/  /'
      grep -E "^\[VK_PROF\] memcpy_x" "$OUT/prof-glm-$k.log" | tail -2 | sed 's/^/  /'
      echo "  [RCB] lines: $(grep -c RCB "$OUT/prof-glm-$k.log")   kda split: $(grep -o 'kda split[^)]*)' "$OUT/prof-glm-$k.log" | tail -1)"
      wait_no_engine glm53 || exit 1
    done
    # greedy text across the knob, in a mode where the retained path is live
    sed -n '/^Generated (/,/^TTFT:/p' "$OUT/prof-glm-0.log" | grep -v '^TTFT:' > /tmp/x1_g0
    sed -n '/^Generated (/,/^TTFT:/p' "$OUT/prof-glm-1.log" | grep -v '^TTFT:' > /tmp/x1_g1
    if [ ! -s /tmp/x1_g0 ]; then echo "  glm decode text REFUSED: no text"; \
    elif cmp -s /tmp/x1_g0 /tmp/x1_g1; then echo "  glm decode text IDENTICAL across the knob"; \
    else echo "  glm decode text DIFFERS"; fi
  fi
}

case "$PHASE" in
  step0) phase_step0 ;;
  prof)  phase_prof both ;;
  profq) phase_prof qwen ;;
  profg) phase_prof glm ;;
  qwen)  phase_qwen ;;
  glm)   phase_glm ;;
  all)   phase_step0; phase_qwen; phase_glm ;;
  *) log "unknown phase $PHASE"; exit 2 ;;
esac
log "x1_probe $PHASE done; outputs in $OUT"
