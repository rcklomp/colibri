#!/bin/bash
# glm_d10_exact_chain.sh -- D10 (ACTION-LIST row 8): is the CPU lane's branch (franken-engine d10-cpu-lane 0436ded; CLI ~/bench/franken_bin/franken_decode_glm_d10l) BIT-EXACT to main (fe7607a = the installed .pf15
# code 890b75f + docs) with the lane OFF (`--cpu-lane 0`, the default: no lane object, no lane kernel, the plan's CPU count 0), and does `--cpu-lane 6` decode sane text? Exactness and sanity only; the task-level
# quality of the lane (MMLU-Pro, needle) is glm_d10_quality_chain.sh.
# METHOD, and why not an existing reference: ref512_keep and gpu5/{g136_eager,g2200} are IQ4_XS dumps (another model); the one hybrid dump, hyb/g136_eager, is chunk 1 / 136 tokens from the pf14 build of 2026-10-09 at
#   non-shipped flags, and nothing in the records compares .pf15 with it. The shipped prefill (--gemm-lds 3 + min-rows 16 + splitk 1, --glm-attn-wmma 64) is NOT bit-exact to any dump older than PF2 / PF15 (GLM5.md 25, 26, 29), so
#   no old reference can be met at the shipped flags. The reference is therefore MAIN'S OWN CLI on the very same plan: --dump in one process, --oracle in the next (how ref512_keep served every candidate; the
#   glm_q8f_chain / fa2 recipe: 64 greedy tokens at depth 1 500, plus a deep pair past the indexer's 2 048 threshold). Per config the oracle compares ~1 800 taps of the last prompt token (maxabs must be 0), the 64 greedy
#   ids (equal) and the logits of the last greedy step (maxabs 0): a lane-off decode that differs anywhere moves those logits.
# Stages (D10X_ONLY, default 1,2,3,4):
#   1 build main's CLI: worktree ~/src/franken-engine-d10x at fe7607a, `make gpu GPU_BIN=franken_decode_glm_main` (compile only, no GPU), worktree removed at the end
#   2 process A = main, one --gate-plan: ref1500 (--dump), ctl1500 (--oracle: main against main = the CONTROL), ref2300 (--dump), ctl2300
#   3 process B = the lane build, started only after A is gone (rig_quiet_wait + MemAvailable): l0_1500 FIRST (--cpu-lane 0 --oracle ref1500), l0_2300, l0_ad (swap every token), l0_pat (--fetch-assign
#     pattern), l6_1500 (--cpu-lane 6 --oracle ref1500: NOT expected exact -- the positive control of the comparison and the sanity run), l0_after (lane off again after a lane run), l0_g (--hip-graph 1: last,
#     it is the config most likely to hang and must not cost the ones above)
#   4 the verdict (no GPU; re-runnable alone)
#   5 automatic only when 4 finds lane-off differing while the control is exact: main's CLI again in a FRESH process against the same dumps (rules out cross-process noise of main itself); `D10X_ONLY=1,5` by hand
# Every config: the shipped flags (PF + DEC0 below = glm_d10_lane_chain.sh's plus --glm-attn-wmma 64), --ctx 262144, --expert-gb 16.85, the hybrid GGUF, the first 1 500 / 2 300 ids of prose8400.txt (neither leaves a
# one-token last chunk, so no prompt step takes the lane's T == 1 path), --greedy 64.
# Launch (rig idle, service OFF, reservation flag in place; while the D10 quality run (franken_quality_chain.sh) holds the rig lock run_chain.sh exits 3 at once: launch again when its log shows its exit marker):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d10_exact_chain.sh > ~/bench/glm_d10_exact_chain.log 2>&1 < /dev/null &
# Watch: ckpt1006/watch_chain.sh ~/bench/glm_d10_exact_chain.log ~/bench/franken/glm5/d10x/lane/gate_run.log   (process A's engine tail is printed into the chain log on a bad end; a "DEAD" line during the build or
#   between the two processes is the gap, not a death)
# Judge: the chain prints `=== VERDICT exactness: PASS|FAIL|INCONCLUSIVE|NO VERDICT` and `=== VERDICT lane-on sanity: OK|NOT OK`. Exit: 0 both fine; 1 lane-off NOT exact (control exact); 4 exact, lane-on sanity failed;
#   5 inconclusive (control not exact, regime broke, or cross-process noise); 6 a dump, an engine process or a config is missing / did not compare (NOT-RUN, oracle none, coverage); 2 setup; 3 rig busy.
#   Re-read without a GPU:  ssh rome 'D10X_ONLY=4 ~/bench/glm_d10_exact_chain.sh'      The standard tool alone:  ssh rome '~/bench/oracle_verdict.sh ~/bench/franken/glm5/d10x/lane/gate_run.log'
#   (it counts taps with maxabs != 0 but not greedy_ids and cannot see an absent oracle: this chain's judge does both).
# Env: D10X_OUT (~/bench/franken/glm5/d10x), D10X_ONLY, D10X_MAIN_REV (fe7607a), D10X_MAIN_BIN (an existing main CLI -- franken_dec_glm.pf15 is one too -- skips stage 1), D10X_LANE_BIN, D10X_KEEP_MAIN=1 (copy the built
#   franken_decode_glm_main to ~/bench/franken_bin), D10X_KEEP_DUMPS=1 (the two 50 MB root-owned dumps are deleted on rc 0 / 4), D10X_NO_X=1 (no automatic stage 5), D10X_TIMEOUT (3000 s a process), D10X_PF (prefill flags;
#   if the CONTROL is not exact because the f16 GEMM is not run-to-run reproducible, rerun with D10X_PF="--chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 1").
# About 10-15 min with a warm page cache (build 1-2 min; each process ~85 s load + ~15 s a config; the 2026-10-12 d10l run did 5 configs in 2 min). Times in the logs are UTC (Amsterdam = UTC+2).
set -u
. "$HOME/bench/chain_preflight.sh"
REPO=$HOME/src/franken-engine
MAIN_REV=${D10X_MAIN_REV:-fe7607a}; LANE_BR=d10-cpu-lane; LANE_REV=0436ded
LANE_BIN=${D10X_LANE_BIN:-$HOME/bench/franken_bin/franken_decode_glm_d10l}
MBIN=franken_decode_glm_main; WT=$HOME/src/franken-engine-d10x; WD=$WT/franken/decode
MAIN_EXE=${D10X_MAIN_BIN:-$WD/$MBIN}
O=${D10X_OUT:-$HOME/bench/franken/glm5/d10x}
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; EG=16.85; PR=$HOME/bench/franken/glm5/prose8400.txt
NG=64; TMO=${D10X_TIMEOUT:-3000}
ONLY=${D10X_ONLY:-1,2,3,4}
has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
MADE_WT=0
cleanup() {
  if [ "$MADE_WT" = 1 ] && [ -d "$WT" ]; then
    if [ "${D10X_KEEP_MAIN:-0}" = 1 ] && [ -x "$WD/$MBIN" ]; then cp -p "$WD/$MBIN" "$HOME/bench/franken_bin/$MBIN" && echo "kept $HOME/bench/franken_bin/$MBIN"; fi
    git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1; rm -rf "$WT"; git -C "$REPO" worktree prune
    echo "removed the build worktree $WT"
  fi
  MADE_WT=0
}
say_end() { cleanup; echo "=== glm_d10x exit rc=$1 $(date -Is)"; exit "$1"; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

# ---- the shipped flags (glm_d10_lane_chain.sh's, plus the PF15 attention) ----
ENVS="-e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=3 -e FRANKEN_GEMM_LDS_MIN_ROWS=16 -e FRANKEN_GEMM_LDS_SPLITK=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal \
 -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1"
DKO="--rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 $ENVS \
 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench"
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
PF_DEF="--chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 --glm-attn-wmma 64"
PF=${D10X_PF:-$PF_DEF}
DEC0="--gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1"
FA="--fetch-assign optimal"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
base() { echo "--tokens-file $O/p$1.txt --ctx 262144 $PF $DEC0 --greedy $NG"; }   # base <1500|2300>

# ---- helpers ----
fresh() {   # fresh <dir>: remove it (the container wrote root-owned files: remove through a container), recreate it empty
  local d=$1
  if [ -e "$d" ]; then docker run --rm -v "$(dirname "$d")":/w "$IMG" rm -rf "/w/$(basename "$d")" >/dev/null 2>&1; fi
  mkdir -p "$d" && [ -z "$(ls -A "$d")" ]
}
mem_wait() {   # an engine pins a ~110 GB host mirror: never start one while another's is still pinned (two hung the rig for 2.5 h on 2026-10-12)
  local t=0 a
  while :; do
    a=$(awk '/^MemAvailable:/ { print int($2 / 1048576) }' /proc/meminfo)
    [ "${a:-0}" -ge 150 ] && return 0
    [ "$t" -ge 900 ] && break
    sleep 10; t=$((t + 10))
  done
  echo "MemAvailable ${a:-?} GiB < 150 after 900 s: another engine's host mirror is still pinned"; return 1
}
mkprompts() {   # the first 1 500 / 2 300 ids of prose8400.txt (1 500 = 1 024 + 476, 2 300 = 2 x 1 024 + 252: no one-token last chunk)
  local n
  for n in 1500 2300; do
    tr -s ' \n' '\n\n' < "$PR" | head -n "$n" | tr '\n' ' ' > "$O/p$n.txt"
    [ "$(wc -w < "$O/p$n.txt" | tr -d ' ')" = "$n" ] || { echo "FATAL: bad prompt file $O/p$n.txt"; return 1; }
  done
}
section() { awk -v c="$2" '/^=== gate-plan config / { p = ($4 == c) } p' "$1"; }   # one configuration's part of a --gate-plan log
plan_main() {
  { echo "ref1500 $(base 1500) $FA --dump $O/main/ref1500"
    echo "ctl1500 $(base 1500) $FA --oracle $O/main/ref1500"
    echo "ref2300 $(base 2300) $FA --dump $O/main/ref2300"
    echo "ctl2300 $(base 2300) $FA --oracle $O/main/ref2300"; } > "$O/main/plan.txt"
}
plan_lane() {   # the same lines as plan_main's oracles, plus --cpu-lane
  local R15=$O/main/ref1500 R23=$O/main/ref2300
  { echo "l0_1500 $(base 1500) $FA --cpu-lane 0 --oracle $R15"
    echo "l0_2300 $(base 2300) $FA --cpu-lane 0 --oracle $R23"
    echo "l0_ad $(base 1500) $FA --cpu-lane 0 $AD --oracle $R15"
    echo "l0_pat $(base 1500) --fetch-assign pattern --cpu-lane 0 --oracle $R15"
    echo "l6_1500 $(base 1500) $FA --cpu-lane 6 --oracle $R15"
    echo "l0_after $(base 1500) $FA --cpu-lane 0 --oracle $R15"
    echo "l0_g $(base 1500) $FA --cpu-lane 0 --hip-graph 1 --oracle $R15"; } > "$O/lane/plan.txt"
}
plan_x() {
  { echo "xctl1500 $(base 1500) $FA --oracle $O/main/ref1500"
    echo "xctl2300 $(base 2300) $FA --oracle $O/main/ref2300"; } > "$O/xmain/plan.txt"
}
run_engine() {   # run_engine <tag> <binary> <dir>: plan <dir>/plan.txt, log <dir>/gate_run.log; the process is gone when this returns
  local tag=$1 bin=$2 d=$3 rc
  rig_quiet_wait 1800 || { echo "[$tag] the rig did not become quiet"; return 3; }
  mem_wait || return 3
  echo "=== [$tag] engine start $(date -Is) bin=$(sha256sum "$bin" | cut -c1-16) configs=$(grep -c . "$d/plan.txt")"
  timeout -k 120 "$TMO" docker run $DKO $IMG "$bin" --model "$M" --placement "$P" --expert-gb "$EG" --gate-plan "$d/plan.txt" > "$d/gate_run.log" 2>&1
  rc=$?
  echo "[$tag] engine rc=$rc $(date -Is)   (0 = every config ok, 1 = some config failed -- the verdict judges them one by one, 124 = timeout)"
  grep -aE "gate_plan_summary|HIP error|Memory access|out of memory|FATAL|REFUSED|\[cpu-lane\] (OFF|FATAL)|gemm-lds 3:|error:" "$d/gate_run.log" | sort | uniq -c | head -12 | cut -c1-220
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then echo "--- [$tag] engine log tail"; tail -n 14 "$d/gate_run.log" | cut -c1-200; fi
  return $rc
}

# ---- the judges ----
# judge <log> <config...>: one line a config (OK / BAD + why); returns 0 when every config is OK, 1 when a BAD one has a MEASURED difference (a tap, the ids or the logits), 2 when the BAD ones only did not
# run / did not compare (no log, rc != 0, oracle PENDING, partial coverage). A config is OK only when ALL hold: it ran (rc 0), the oracle printed PASS, >= 1000 taps are
# bit-exact (maxabs 0) and none is not, greedy_ids exact=1 with 64 of 64, greedy_last_logits maxabs 0, missing / refused 0, and compared + incomparable == the number of entries in the DUMP's index.txt (an
# oracle that PENDs or compares part of a dump leaves rc 0 in the engine: this is the guard). The dump's index is read from <dir>/index.txt or, after the dump is deleted, <dir>.index.txt.
judge() {
  local log=$1; shift
  [ -s "$log" ] || { echo "  (no log $log)"; return 2; }
  awk -v want="$*" -v nt="$NG" '
    BEGIN { nw = split(want, W, " "); wids = "exact=1 mine=" nt " ref=" nt }
    /^=== gate-plan config / { c = $4; next }
    /^gate_plan_result name=/ { split($2, a, "="); split($3, b, "="); rc[a[2]] = b[2]; next }
    /^GLM5 GPU ORACLE / { orc[c] = $4; next }
    /^GLM5 ADAPT VERIFY FAIL/ { avf[c]++; next }
    /HIP error|Memory access fault|out of memory/ { herr[c]++ }
    /^oracle summary: / { for (i = 3; i <= NF; i++) { split($i, kv, "="); k = kv[1]; v = kv[2]
         if (k == "compared") cmp[c] = v; else if (k == "missing") mis[c] = v; else if (k == "refused") rfs[c] = v
         else if (k == "incomparable") inc[c] = v; else if (k == "dir") dir[c] = v }
         next }
    /^oracle greedy_ids / { ids[c] = $3 " " $4 " " $5; next }
    /^oracle [^ ]+ cos=/ { n[c]++; split($4, m, "="); if (m[2] != "0") { bad[c]++; if (bad[c] <= 4) fb[c] = fb[c] " " $2 "(" $4 ")" }
         if ($2 == "greedy_last_logits") lg[c] = $4
         next }
    /^oracle [^ ]+ (MISSING|UNREADABLE|REFUSED)/ { odd[c]++; next }
    /^oracle / && / FAIL$/ { ofail[c]++; next }
    END {
      nbad = 0; ndiff = 0
      for (i = 1; i <= nw; i++) {
        c = W[i]; why = ""
        if (!(c in rc)) why = why " NOT-RUN"; else if (rc[c] != "0") why = why " rc=" rc[c]
        if (orc[c] != "PASS") why = why " oracle=" (orc[c] == "" ? "none(PENDING?)" : orc[c])
        if (n[c] < 1000) why = why " taps=" (n[c] + 0)
        if (bad[c] > 0) why = why " not_exact=" bad[c]
        if (ids[c] != wids) why = why " ids[" ids[c] "]"
        if (lg[c] != "maxabs=0") why = why " logits[" lg[c] "]"
        if (mis[c] != "0" || rfs[c] != "0") why = why " missing=" mis[c] " refused=" rfs[c]
        if (odd[c] > 0 || ofail[c] > 0) why = why " odd=" (odd[c] + 0) " failed=" (ofail[c] + 0)
        if (avf[c] > 0 || herr[c] > 0) why = why " adapt_verify_fail=" (avf[c] + 0) " hip_errors=" (herr[c] + 0)
        ent = 0
        if (dir[c] != "") {
          f = dir[c] "/index.txt"; while ((getline ln < f) > 0) ent++; close(f)
          if (ent == 0) { f = dir[c] ".index.txt"; while ((getline ln < f) > 0) ent++; close(f) }
        }
        if (ent == 0) why = why " dump-index-unreadable"
        else if (cmp[c] + inc[c] != ent) why = why " coverage=" (cmp[c] + 0) "+" (inc[c] + 0) "!=" ent
        printf "  %-9s %s taps=%d not_exact=%d compared=%s incomparable=%s dump_entries=%d ids[%s] logits[%s]%s\n", c, (why == "" ? "OK " : "BAD"), n[c] + 0, bad[c] + 0, cmp[c], inc[c], ent, ids[c], lg[c], (why == "" ? "" : "   <-" why "   first-not-exact:" fb[c])
        if (why != "") nbad++
        if (bad[c] > 0 || (ids[c] != "" && ids[c] != wids) || (lg[c] != "" && lg[c] != "maxabs=0")) ndiff++
      }
      exit (ndiff > 0 ? 1 : (nbad > 0 ? 2 : 0))
    }' "$log"
}
judge_ref() {   # judge_ref <log> <config> <dumpdir>: the --dump config ran, dumped as many taps as its index lists, and printed 64 greedy ids
  local log=$1 c=$2 d=$3 sec rc nd ent ids f
  sec=$(section "$log" "$c")
  [ -n "$sec" ] || { echo "  $c  BAD  did not run"; return 1; }
  rc=$(echo "$sec" | sed -n 's/^gate_plan_result name=[^ ]* rc=\([0-9]*\).*/\1/p' | tail -1)
  nd=$(echo "$sec" | sed -n 's/^dumped \([0-9]*\) taps to .*/\1/p' | tail -1)
  f=$d/index.txt; [ -s "$f" ] || f=$d.index.txt
  ent=$(wc -l < "$f" 2>/dev/null | tr -d ' ')
  ids=$(( $(echo "$sec" | grep -a '^greedy_ids:' | tail -1 | wc -w) - 1 ))
  if [ "${rc:-x}" = 0 ] && [ "${nd:-0}" -ge 1000 ] && [ "${nd:-0}" = "${ent:-x}" ] && [ "$ids" = "$NG" ]; then
    echo "  $c  OK   rc=0 dumped=$nd index_entries=$ent greedy_ids=$ids"
  else
    echo "  $c  BAD  rc=${rc:-none} dumped=${nd:-none} index_entries=${ent:-0} greedy_ids=$ids"; return 1
  fi
}
fp() {   # a checksum of the engine's own settings lines of one configuration
  local s; s=$(section "$1" "$2" | grep -aE "^(chunk=|glm5_prefill_stage=|adapt_snapshot_dma=|adapt=|gemv_fused_reduce=)")
  if [ -z "$s" ]; then echo absent; else echo "$s" | cksum | cut -d' ' -f1; fi
}
fpcheck() {   # fpcheck <label> <log:config>...: configurations of one prompt must have run the same flags (absent ones are not compared)
  local label=$1 first="" x lc f ok=1; shift
  for x in "$@"; do
    lc=${x#*:}; f=$(fp "${x%%:*}" "$lc"); printf ' %s=%s' "$lc" "$f"
    [ "$f" = absent ] && continue
    [ -z "$first" ] && first=$f
    [ "$f" = "$first" ] || ok=0
  done
  echo
  [ "$ok" = 1 ] || { echo "  BAD: the engine's settings lines differ between configurations that must run the same flags ($label)"; return 1; }
}
lane_sanity() {   # --cpu-lane 6: it ran, the lane served slots, the tally and the lane agree, 64 ids that look like text, the prompt taps untouched; how many of the 64 match lane-off
  local LL=$O/lane/gate_run.log s6 line slots tally served ids0 ids6 n n0 m first i d0 d6 pre pn pb lgl bad=0
  local -a A B
  [ -s "$LL" ] || { echo "  (no lane log)"; return 1; }
  s6=$(section "$LL" l6_1500)
  [ -n "$s6" ] || { echo "  l6_1500 did not run (the process ended before it: see the engine tail above)"; return 1; }
  echo "$s6" | grep -aE "^cpu_lane=|\[cpu-lane\] (started|REFUSED|OFF|FATAL)" | cut -c1-200 | sed 's/^/  /'
  echo "$s6" | grep -aqE "\[cpu-lane\] (REFUSED|OFF|FATAL)" && bad=1
  line=$(echo "$s6" | grep -a "^\[cpu-lane\] N=" | head -1)
  if [ -z "$line" ]; then echo "  BAD: no '[cpu-lane] N=' counter line (the lane did not run)"; bad=1
  else
    echo "  $line" | cut -c1-340
    slots=$(echo "$line" | sed -n 's/.*slots\/token=\([0-9.]*\).*/\1/p')
    tally=$(echo "$line" | sed -n 's/.*device tally \([0-9]*\) CPU slots.*/\1/p')
    served=$(echo "$line" | sed -n 's/.*the lane served [0-9]* layers, \([0-9]*\) slots.*/\1/p')
    awk -v x="${slots:-0}" 'BEGIN { exit !(x > 0) }' || { echo "  BAD: slots/token=${slots:-?}: the lane served nothing"; bad=1; }
    # the device tally is cumulative over the process (it was 2 x in the 2026-10-12 d10l run); l6_1500 is the first lane config here, so the two must agree
    if [ -z "$tally" ] || [ "$tally" != "$served" ]; then echo "  BAD: device tally ${tally:-?} CPU slots != the lane's own count ${served:-?}"; bad=1; fi
  fi
  ids0=$(section "$LL" l0_1500 | grep -a '^greedy_ids:' | tail -1); ids6=$(echo "$s6" | grep -a '^greedy_ids:' | tail -1)
  set -- ${ids0#greedy_ids:}; n0=$#; A=("$@"); set -- ${ids6#greedy_ids:}; n=$#; B=("$@")
  if [ "$n0" != "$NG" ] || [ "$n" != "$NG" ]; then echo "  BAD: greedy ids: lane-off $n0, lane-on $n (expected $NG each)"; bad=1
  else
    m=0; first=-1; i=0
    while [ $i -lt $n ]; do
      if [ "${A[$i]}" = "${B[$i]}" ]; then m=$((m + 1)); elif [ $first -lt 0 ]; then first=$i; fi
      i=$((i + 1))
    done
    d0=$(printf '%s\n' "${A[@]}" | sort -u | wc -l | tr -d ' '); d6=$(printf '%s\n' "${B[@]}" | sort -u | wc -l | tr -d ' ')
    echo "  LANE-ON vs LANE-OFF, same prompt: $m of $n greedy ids equal at the same position, first difference at position $first (0-based, -1 = none), distinct ids $d6 (lane-off $d0)"
    echo "  lane-off: $(echo "${A[*]}" | cut -c1-200)..."
    echo "  lane-on : $(echo "${B[*]}" | cut -c1-200)..."
    [ "$d6" -ge 8 ] || { echo "  BAD: only $d6 distinct ids in 64: degenerate output"; bad=1; }
  fi
  pre=$(echo "$s6" | awk '/^oracle [^ ]+ cos=/ { if ($2 == "greedy_last_logits") next; n++; split($4, m, "="); if (m[2] != "0") b++ } END { print n + 0, b + 0 }')
  set -- $pre; pn=$1; pb=$2
  echo "  prompt taps (chunks never use the lane): $pn compared against main, $pb not bit-exact (expected 0)"
  [ "$pb" = 0 ] && [ "$pn" -ge 1000 ] || { echo "  BAD: the lane changed (or did not compare) the prompt's taps"; bad=1; }
  lgl=$(echo "$s6" | grep -a '^oracle greedy_last_logits' | head -1 | cut -c1-120)
  echo "  last-step logits vs main (the positive control: must NOT be maxabs=0 when the lane served slots): $lgl"
  case "$lgl" in *"maxabs=0 "*) echo "  BAD: the lane served slots but the logits are identical to main's: the lane is not in the data path"; bad=1;; esac
  return $bad
}
steady() { section "$1" "$2" | awk '/^vram_dev[0-9]_steady/ { split($2, a, "="); printf " %s", a[2] } END { print "" }'; }

# verdict: prints everything, returns 0 PASS / 1 lane-off NOT exact / 4 lane-on sanity failed / 5 inconclusive / 6 something missing
verdict() {
  local rc=0 ctl=0 l0=0 san=0 reg=0 l
  echo "--- standard tool oracle_verdict.sh (taps with maxabs != 0; it does not count greedy_ids)"
  for l in main lane xmain; do [ -s "$O/$l/gate_run.log" ] && { echo "[$l]"; "$HOME/bench/oracle_verdict.sh" "$O/$l/gate_run.log"; }; done
  echo "--- regime: the f16 GEMM must have stayed on in every process ('gemm-lds 3:' = it fell back to mode 1)"
  for l in main lane xmain; do
    [ -s "$O/$l/gate_run.log" ] && grep -aq "gemm-lds 3:" "$O/$l/gate_run.log" && { echo "  BAD [$l]: $(grep -a 'gemm-lds 3:' "$O/$l/gate_run.log" | head -1 | cut -c1-200)"; reg=1; }
  done
  echo "--- settings checksum of each configuration (configs of one prompt must agree)"
  fpcheck 1500 "$O/main/gate_run.log:ref1500" "$O/main/gate_run.log:ctl1500" "$O/lane/gate_run.log:l0_1500" "$O/lane/gate_run.log:l6_1500" "$O/lane/gate_run.log:l0_after" || reg=1
  fpcheck 2300 "$O/main/gate_run.log:ref2300" "$O/main/gate_run.log:ctl2300" "$O/lane/gate_run.log:l0_2300" || reg=1
  l=$(section "$O/lane/gate_run.log" l0_1500 | grep -a "^chunk=" | head -1 | cut -c1-140); echo "  lane l0_1500: $l"
  l=$(section "$O/lane/gate_run.log" l0_1500 | grep -a "^adapt_snapshot_dma=" | head -1 | cut -c1-160); echo "  lane l0_1500: $l"
  echo "--- steady VRAM, free MB on dev0 dev1 dev2: main ref1500 $(steady "$O/main/gate_run.log" ref1500) | lane l0_1500 $(steady "$O/lane/gate_run.log" l0_1500) | lane l6_1500 $(steady "$O/lane/gate_run.log" l6_1500)"
  echo "--- the dumps of main (process A)"
  judge_ref "$O/main/gate_run.log" ref1500 "$O/main/ref1500" || rc=6
  judge_ref "$O/main/gate_run.log" ref2300 "$O/main/ref2300" || rc=6
  echo "--- CONTROL: main against main in the same process (a difference here is the engine's own run-to-run noise, not the lane)"
  judge "$O/main/gate_run.log" ctl1500 ctl2300; ctl=$?
  echo "--- LANE OFF against main: l0_1500 (first config of a fresh process) l0_2300 l0_ad (swap every token) l0_pat (pattern assignment) l0_g (graph replay, last in the plan)"
  judge "$O/lane/gate_run.log" l0_1500 l0_2300 l0_ad l0_pat l0_g; l0=$?
  echo "--- LANE ON (--cpu-lane 6), 64 greedy tokens from the same prompt (not a gate: answers may change); then lane off again (l0_after)"
  lane_sanity || san=1
  judge "$O/lane/gate_run.log" l0_after || san=1
  echo
  if [ $rc -ne 0 ]; then echo "=== VERDICT exactness: NO VERDICT -- a reference dump of main is missing or inconsistent (see above)"; return 6; fi
  if [ $reg -ne 0 ] || [ $ctl -eq 1 ]; then
    echo "=== VERDICT exactness: INCONCLUSIVE -- the control (main against main, in process) or the settings are not clean: the comparison cannot separate the lane from the engine's own noise"
    echo "    (if only the control fails: prefill is not run-to-run exact -- rerun with D10X_PF set to --gemm-lds 1 and no --glm-attn-wmma, see the header)"
    return 5
  fi
  if [ $ctl -eq 2 ] || [ $l0 -eq 2 ]; then echo "=== VERDICT exactness: NO VERDICT -- a control or lane-off configuration did not run or did not compare (see the BAD lines: NOT-RUN / oracle none / coverage)"; return 6; fi
  if [ $l0 -ne 0 ]; then echo "=== VERDICT exactness: FAIL -- lane OFF (--cpu-lane 0) differs from main $MAIN_REV while main reproduces itself in process (a fresh-process control follows when stage 5 runs)"; return 1; fi
  echo "=== VERDICT exactness: PASS -- lane OFF (--cpu-lane 0) is bit-exact to main $MAIN_REV on 5 configs (1500 / 2300 tokens, plain, swap every token, graph replay, pattern): every tap maxabs 0, 64 greedy ids, last logits maxabs 0; control exact"
  if [ $san -ne 0 ]; then echo "=== VERDICT lane-on sanity: NOT OK (see the LANE ON block)"; return 4; fi
  echo "=== VERDICT lane-on sanity: OK -- --cpu-lane 6 served slots, 64 ids, prompt taps untouched, logits differ from main's; lane off again afterwards is exact"
  return 0
}

# ---- run ----
echo "=== glm_d10x start $(date -Is) stages=$ONLY main=$MAIN_REV lane=$LANE_BR($LANE_REV) out=$O"
mkdir -p "$O" || say_end 2
vrc=""
if has 2 || has 3 || has 5; then
  for f in "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
  mkprompts || say_end 2
fi

if has 1 && [ -z "${D10X_MAIN_BIN:-}" ]; then
  echo "=== 1. build main's CLI $(date +%T)"
  git -C "$REPO" cat-file -e "$MAIN_REV^{commit}" 2>/dev/null || { echo "FATAL: no commit $MAIN_REV in $REPO"; say_end 2; }
  git -C "$REPO" rev-parse --verify -q "refs/heads/$LANE_BR" >/dev/null || { echo "FATAL: no branch $LANE_BR in $REPO"; say_end 2; }
  if [ -e "$WT" ]; then git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1; rm -rf "$WT"; git -C "$REPO" worktree prune; fi
  git -C "$REPO" worktree add --detach "$WT" "$MAIN_REV" >/dev/null 2>&1 || { echo "FATAL: git worktree add $WT $MAIN_REV failed"; say_end 2; }
  MADE_WT=1
  echo "main : $(git -C "$WT" log --oneline -1 | cut -c1-110)"
  echo "lane : $(git -C "$REPO" log --oneline -1 "$LANE_BR" | cut -c1-110)   (its parent $(git -C "$REPO" rev-parse --short "$LANE_BR~1"); main here $(git -C "$WT" rev-parse --short HEAD))"
  [ "$(git -C "$REPO" rev-parse --short=7 "$LANE_BR")" = "$LANE_REV" ] || echo "WARN: the branch tip is not $LANE_REV: is $LANE_BIN built from it? (binary mtime $(stat -c %y "$LANE_BIN" 2>/dev/null | cut -c1-19) UTC)"
  ( cd "$WD" && nice -n 19 make -j4 gpu GPU_BIN=$MBIN > "$O/build_main.log" 2>&1; echo "make rc=$?" >> "$O/build_main.log" )
  grep -vE "^docker run|^\s+\.\./|^\s*$" "$O/build_main.log" | tail -4 | cut -c1-200
  [ -x "$WD/$MBIN" ] || { echo "FATAL: no $WD/$MBIN (build log $O/build_main.log)"; say_end 2; }
  echo "main CLI: $MBIN sha256 $(sha256sum "$WD/$MBIN" | cut -c1-16)"
fi

if has 2; then
  [ -x "$MAIN_EXE" ] || { echo "FATAL: no main CLI $MAIN_EXE (stage 1 builds it; or set D10X_MAIN_BIN)"; say_end 2; }
  fresh "$O/main" || { echo "FATAL: cannot clear $O/main"; say_end 2; }
  plan_main
  echo "=== 2. process A: main ($MAIN_EXE) $(date +%T)"
  run_engine A "$MAIN_EXE" "$O/main"; rc=$?
  [ "$rc" -eq 3 ] && say_end 3
  for r in ref1500 ref2300; do
    if [ -s "$O/main/$r/index.txt" ]; then cp "$O/main/$r/index.txt" "$O/main/$r.index.txt"; else echo "FATAL: process A left no dump $O/main/$r (engine rc=$rc)"; say_end 6; fi
  done
fi

if has 3; then
  [ -x "$LANE_BIN" ] || { echo "FATAL: no lane CLI $LANE_BIN"; say_end 2; }
  for r in ref1500 ref2300; do [ -s "$O/main/$r/index.txt" ] || { echo "FATAL: no reference dump $O/main/$r (stage 2 makes it)"; say_end 2; }; done
  fresh "$O/lane" || { echo "FATAL: cannot clear $O/lane"; say_end 2; }
  plan_lane
  echo "=== 3. process B: the lane build ($LANE_BIN, $LANE_BR) $(date +%T)"
  run_engine B "$LANE_BIN" "$O/lane"; rc=$?
  [ "$rc" -eq 3 ] && say_end 3
  [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ] && echo "process B ended abnormally (rc=$rc): the verdict reads what it wrote"
fi

if has 4; then
  echo "=== 4. verdict $(date +%T)"
  verdict 2>&1 | tee "$O/verdict.txt"; vrc=${PIPESTATUS[0]}
  if [ "$vrc" = 1 ] && [ "${D10X_NO_X:-0}" != 1 ] && ! has 5; then ONLY="$ONLY,5"; echo "lane off differs from main: stage 5 (main again, fresh process) follows"; fi
fi

if has 5; then
  echo "=== 5. process C: main again in a fresh process against process A's dumps $(date +%T)"
  if [ ! -x "$MAIN_EXE" ] || [ ! -s "$O/main/ref1500/index.txt" ] || [ ! -s "$O/main/ref2300/index.txt" ]; then
    echo "FATAL: stage 5 needs the main CLI ($MAIN_EXE) and process A's dumps"; say_end 2
  fi
  fresh "$O/xmain" || { echo "FATAL: cannot clear $O/xmain"; say_end 2; }
  plan_x
  run_engine C "$MAIN_EXE" "$O/xmain"; rc=$?
  [ "$rc" -eq 3 ] && say_end 3
  judge "$O/xmain/gate_run.log" xctl1500 xctl2300 | tee -a "$O/verdict.txt"; xrc=${PIPESTATUS[0]}
  if [ "$xrc" = 0 ]; then
    echo "=== VERDICT cross-process control: main reproduces itself across fresh processes" | tee -a "$O/verdict.txt"
    [ "$vrc" = 1 ] && echo "=== the lane-off difference is therefore the lane build's own: NOT exact (rc 1)" | tee -a "$O/verdict.txt"
    [ -z "$vrc" ] && vrc=0
  else
    echo "=== VERDICT cross-process control: main differs from its own dump in a fresh process: the comparison is INCONCLUSIVE (rc 5), the lane is not the cause" | tee -a "$O/verdict.txt"
    vrc=5
  fi
fi

# the dumps (2 x 50 MB, root-owned) only while a failure is being looked at; the index copies, the plans and the logs stay
if { [ "$vrc" = 0 ] || [ "$vrc" = 4 ]; } && [ "${D10X_KEEP_DUMPS:-0}" != 1 ] && has 4; then
  if docker run --rm -v "$O/main":/w "$IMG" rm -rf /w/ref1500 /w/ref2300 >/dev/null 2>&1; then echo "dumps deleted (exactness passed; the verdict can be re-read with D10X_ONLY=4)"; else echo "could not delete $O/main/ref1500 and ref2300: remove them through a container"; fi
fi
say_end "${vrc:-0}"
