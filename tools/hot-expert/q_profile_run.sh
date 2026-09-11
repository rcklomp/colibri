#!/bin/bash
# The per-op profiling campaign for qwen38-vk on rome (record §Q-PROFILE).
#
# The Qwen-track twin of profile_run.sh, which does the same job for glm53.
# Everything above the run list is the procedure that makes the numbers
# comparable; edit the run list at the bottom.
#
# Three things differ from the GLM script and each one is a trap that has
# already cost this project a wrong number on the other engine:
#
#  1. 131 shards, not 62, and the checkpoint is 173 GiB rather than 182. The
#     residency assert counts them, so a partially-warmed model cannot be
#     profiled by accident (§RP1 checked the wrong files, §RP2 checked at the
#     wrong time; both are in the record).
#  2. qwen38 reads COLI_USAGE, NOT COLI_USAGE_PATH, and defaults it to
#     <snap>/.coli_usage -- which it REWRITES at exit with the run's own
#     counts. Every run therefore changes what the next run's GPU tier
#     preloads from unless the history is frozen to a copy. §RP1 found 23.5
#     ms/token of apparent GLM gain that was only a warmer histogram; nothing
#     had ever frozen it for Qwen. This script freezes it for the whole
#     campaign.
#  3. qwen38's CLI is positional: `qwen38-vk <cap> <bits> <prompt-file>` with
#     N_NEW for the token count. There is no --greedy/--prompt.
#
# There is no parser: unlike glm53's [OPTIME] lines, this engine prints its own
# accounting residual ("unaccounted ... sum X vs step Y"), so the check
# profile_parse.py exists to perform is already in the output.
set -u
cd "$HOME/src/colibri/c"
M=$HOME/models/Qwen3.8-Flash-Next-FP8
BIN=${Q_PROFILE_BIN:-./qwen38-vk}

# SNAP is not optional and not defaulted: qwen38 prints "set SNAP=<snapshot
# directory>" and exits 1 without it. The first draft of this script omitted it
# and seven runs died in six seconds each.
export SNAP="$M"
export OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close
export Q38_VULKAN=1 COLI_VK_DEV2=auto COLI_VK_DEV3=auto
export COLI_VK_SHADERS="$HOME/src/colibri/c/shaders"
export COLI_TIMERS=1
export N_NEW=${N_NEW:-80}

# Freeze the routing history for the WHOLE campaign (see note 2 above).
HIST=/tmp/q_profile_hist.bin
if [ -f "$M/.coli_usage" ]; then cp "$M/.coli_usage" "$HIST"; else : > "$HIST"; fi
export COLI_USAGE="$HIST"

# datapoint.py's rotating prompt 1, verbatim: the profile and the serving gate
# then talk about the same kind of request.
PROMPT=/tmp/q_profile_prompt.txt
cat > "$PROMPT" <<'EOF'
Explain how a database transaction can deadlock, give a concrete example, and compare two practical prevention strategies in detail.
EOF

resid() {   # $1 = label, $2 = "assert" | "report"
  fincore --bytes --output FILE,SIZE,RES "$M"/*.safetensors > /tmp/q_profile_fincore.txt
  python3 -c "
PAGE=4096; tot=res=0; short=0; n=0
for line in open('/tmp/q_profile_fincore.txt').read().splitlines()[1:]:
    f=line.split()
    if len(f)<3: continue
    n+=1; size=int(f[-2]); r=int(f[-1]); want=-(-size//PAGE)*PAGE
    tot+=want; res+=r
    if r<want: short+=1
pct=res/tot*100
print(f'[resid $1] shards={n} resident={pct:.4f}% short={short}')
if '$2'=='assert':
    assert n==131 and short==0, 'NOT 100% RESIDENT'
"
}

warm() { find "$M" -type f -name "*.safetensors" -print0 \
         | while IFS= read -r -d '' f; do cat "$f" > /dev/null; done; }

run() {   # $1 = tag
  warm
  resid "$1-pre" assert || { echo "ABORT before $1"; exit 1; }
  echo "### ===== $1 ====="
  "$BIN" 512 8 "$PROMPT" 2>&1 \
    | grep -E "^\[OPTIME\]|^\[qwen38 timers\]|^Speed:|^TTFT:|Vulkan tier preloaded|^\[USAGE\]|^Expert cache hit"
  resid "$1-post" report
}

# perf mode: the independent cross-check on the timers, as §G3 used for GLM.
# The engine is started in the background, given time to load the checkpoint and
# run prefill, and only then sampled -- otherwise the flat profile is mostly the
# loader. cycles:u, flat (no call graph): the question is which symbol burns the
# decode token, and -g on a 16-thread libgomp process buries that in unwinding.
# NOTE: "TTFT:" is printed AFTER generate() returns, so it cannot be the signal
# that prefill is done. The tier-preload line is printed before generation
# begins; wait for that, then give prefill a fixed head start.
perfrun() {   # $1 = tag, $2 = seconds to sample, $3 = seconds to wait first
  warm
  resid "$1-pre" assert || { echo "ABORT before $1"; exit 1; }
  echo "### ===== $1 (perf) ====="
  "$BIN" 512 8 "$PROMPT" > /tmp/q_perf_engine.log 2>&1 &
  epid=$!
  sleep "$3"
  if ! kill -0 "$epid" 2>/dev/null; then
    echo "engine exited before sampling started -- raise the wait"; tail -5 /tmp/q_perf_engine.log; return 1
  fi
  perf record -e cycles:u -F 999 -p "$epid" -o /tmp/q_perf.data -- sleep "$2" 2>&1 | tail -2
  wait "$epid"
  grep -E "^\[OPTIME\]|^Speed:|^TTFT:|Vulkan tier preloaded|^Expert cache hit" /tmp/q_perf_engine.log
  echo "--- perf report (dso) ---"
  perf report -i /tmp/q_perf.data --stdio --sort=dso --percent-limit 0.3 2>/dev/null | head -20
  echo "--- perf report (symbol) ---"
  perf report -i /tmp/q_perf.data --stdio --sort=symbol --percent-limit 0.3 2>/dev/null | head -40
  resid "$1-post" report
}

case "${1:-run}" in
  perf) perfrun "${2:-q-prof-perf}" "${3:-25}" "${4:-150}" ;;
  *)    run "${1:-q-prof-1}" ;;
esac
echo "### done"
