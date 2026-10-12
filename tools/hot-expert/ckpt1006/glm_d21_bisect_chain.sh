#!/bin/bash
# glm_d21_bisect_chain.sh -- D21 (record §L5-D15): which franken-engine commit made card 0's decode helper start late? On 2026-10-07 (main 5b47ef9) card 0's stage
# kernel started ~19 us after the owner's plan, like the other cards; on today's binary ~350 us (IQ4_XS and the hybrid alike, every decode flag on or off).
# Builds the CLI at each commit of BISECT_COMMITS in ONE detached worktree (~/src/fe-bisect, removed at the end), copies each binary to ~/bench/franken_bin/bisect/,
# then one rocprofv3 decode trace per binary on the IQ4_XS file (every commit since 5b47ef9 reads it): 2 048 prompt tokens, 24 settle, 12 traced, only
# --fetch-assign optimal (older binaries know no later flag). Read each: python3 ckpt1006/d15_wake_report.py ~/bench/franken/glm5/dtrace_bis_<sha>/out --tokens 12
# Launch (rig idle, service OFF): setsid nohup ~/src/colibri/tools/hot-expert/ckpt1006/glm_d21_bisect_chain.sh > ~/bench/glm_d21_bisect.log 2>&1 < /dev/null &
# Env: BISECT_COMMITS (default below, oldest first), BISECT_ONLY (build,trace; default both).
set -u
. "$HOME/bench/chain_preflight.sh"
C=${BISECT_COMMITS:-5b47ef9 e10f196 1d12b52 900d38e a7bf81a 384e7b5 ff2dc5d 63a5c0e 1f432a8 7a5d2dc}
ONLY=${BISECT_ONLY:-build,trace}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
W=$HOME/src/fe-bisect; OUTB=$HOME/bench/franken_bin/bisect; mkdir -p "$OUTB"
say_end() { git -C "$HOME/src/franken-engine" worktree remove --force "$W" 2>/dev/null; echo "=== glm_d21_bisect exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d21_bisect start $(date -Is) commits=$C"
if has build; then
  [ -d "$W" ] || git -C "$HOME/src/franken-engine" worktree add -q --detach "$W" main || say_end 2
  for c in $C; do
    git -C "$W" checkout -q --detach "$c" || { echo "FATAL: no commit $c"; say_end 2; }
    echo "--- build $c $(date +%T): $(git -C "$W" log --oneline -1 | cut -c1-90)"
    ( cd "$W/franken/decode" && nice -n 19 make -j8 gpu GPU_BIN=franken_decode_glm_bis 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -3 )
    [ -x "$W/franken/decode/franken_decode_glm_bis" ] || { echo "FATAL: build $c failed"; say_end 2; }
    cp -p "$W/franken/decode/franken_decode_glm_bis" "$OUTB/franken_decode_glm_$c"; rm -f "$W/franken/decode/franken_decode_glm_bis"
  done
fi
rc_all=0
if has trace; then
  for c in $C; do
    rig_quiet_wait 600 || say_end 3
    echo "--- trace $c $(date +%T)"
    TRACE_BIN="$OUTB/franken_decode_glm_$c" TRACE_OUT="$HOME/bench/franken/glm5/dtrace_bis_$c" TRACE_TOKENS=12 TRACE_EXTRA="--fetch-assign optimal" \
      "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_decode_trace_chain.sh" > "$HOME/bench/glm_d21_bisect_$c.log" 2>&1
    rc=$?; echo "--- trace $c done rc=$rc $(grep -a -o 'glm5_decode_ms_median=[0-9.]*' "$HOME/bench/franken/glm5/dtrace_bis_$c/gate_run.log" 2>/dev/null)"
    [ $rc -ne 0 ] && { rc_all=$rc; tail -n 5 "$HOME/bench/glm_d21_bisect_$c.log" | cut -c1-200; }
  done
fi
say_end $rc_all
