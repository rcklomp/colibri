#!/bin/bash
# glm_g8_chain.sh -- FRANKEN_GLM_MOE_G=8 at the SHIPPED prefill config: is it bit-exact, and is it faster than the default 4?
# (handoff 06b section 4 item 4; record section L5-GLM-CHUNK measured G=8 -1.9 % at chunk 1024 in the staged config and never gated it for exactness. The knob is read once a
#  process and only the chunked routed-expert kernel k_glm5_moe_blk uses it -- a decode token keeps the old kernel -- so the question is prefill only.)
# VRAM budget (CLAUDE.md "Before a build that allocates device memory"): G changes the kernel's VGPR count (IQ4_XS 89 -> 131, LDS unchanged) and allocates NOTHING on a card, so the
#  shipped-config steady free VRAM (732 / 818 / 458 MiB) is unchanged; the timing configs are the shipped flags at the largest shapes (262 144 cells, chunk 1024, in place).
# Stages (one fresh process each: a second chunk size in one process diverges, record section L5-GLM-CHUNK):
#   P1 g8_full   G=8: ex1024u = the FULL model, 2 200 prose tokens + 8 greedy, chunk 1024, stage 0, FIRST config, against the saved stock chunk-512 reference ref512_keep
#   P2 g8_small  G=8: c32u / c512u (136 tokens + 8 greedy against gpu5/g136_eager) and long512u (layers 0-3, 2 200 tokens against gpu5/g2200): the small chunks run the remainder
#                paths (groups of 4 / 2 / 1 after the groups of 8), P1 the long-run mix. Every tap must read maxabs=0.
#   Only when P1 AND P2 are exact (and their engines exited 0): timing, prefill ms/token of 8 192 tokens at 262 144 cells, chunk 1024, in place, two configs a process:
#   P3 t_g4, P4 t_g8, P5 t_g4 (the two G=4 processes bracket the G=8 one). Verdict: G=8 mean vs G=4 mean; adopt only if exact AND faster than the G=4 spread.
# Launch through run_chain.sh and watch with ckpt1006/watch_chain.sh <chain log> $G8_OUT/engine.log (every stage's engine output is appended to it; the per-stage logs are $G8_OUT/<stage>/gate_run.log):
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_g8_chain.sh > ~/bench/glm_g8_chain.log 2>&1 < /dev/null &
# Env: G8_BIN (default franken-engine main's franken_decode_glm), G8_OUT (default ~/bench/franken/glm5/g8).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${G8_BIN:-$HOME/src/franken-engine/franken/decode/franken_decode_glm}
O=${G8_OUT:-$HOME/bench/franken/glm5/g8}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; KEEP=$B/ref512_keep; REF5=$B/gpu5; P2200=$B/prose2200.txt; T136=$B/t136.txt; PR=$B/prose8400.txt
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0"
say_end() { echo "=== glm_g8 exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
[ -d "$KEEP" ] && [ -d "$REF5/g136_eager" ] && [ -d "$REF5/g2200" ] && [ -f "$PR" ] || { echo "FATAL: references or prompt missing"; say_end 2; }

run() {   # tag G plan-file  -> engine rc in $RC; logs in $O/<tag>
  local tag=$1 g=$2 plan=$3 D=$O/$1; mkdir -p "$D"; cp "$plan" "$D/plan.txt"
  rig_quiet_wait 1800 || { RC=3; return 3; }
  echo "=== $tag (G=$g) start $(date -Is)"
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=$g \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
    rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$D/plan.txt" 2>&1 | tee -a "$O/engine.log" > "$D/gate_run.log"
  RC=${PIPESTATUS[0]}; echo "gpu rc=$RC"
  grep -aE "HIP error|Memory access|out of memory|gate_plan_summary" "$D/gate_run.log" | head -2 | cut -c1-170
  grep -aE "vram_dev[0-9]_steady" "$D/gate_run.log" | tail -3 | cut -c1-120
  [ "$RC" -ne 0 ] && { echo "--- engine log tail:"; tail -n 8 "$D/gate_run.log" | cut -c1-200; }
  return 0
}
exact_ok() {   # gate_run.log -> 0 when every config has taps > 0 and none is not bit-exact
  $HOME/bench/oracle_verdict.sh "$1" | tee -a "$O/verdict.txt" | awk '{ split($2, a, "="); split($3, b, "="); n++; if (a[2] + 0 <= 0 || b[2] + 0 != 0) bad = 1 } END { exit (n == 0 || bad) }'
}
timing() {   # gate_run.log tag -> prints the prefill rows
  awk -v t="$2" '/^=== gate-plan config/ { c = $4 } /^prefill_tokens=/ { split($5, a, "="); printf "%-8s %-5s %8.4f ms/token = %6.1f tok/s\n", t, c, a[2], 1000 / a[2] }' "$1"
}

echo "=== glm_g8 start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) main=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
: > "$O/verdict.txt"
echo "ex1024u --tokens-file $P2200 --ctx 4096 --greedy 8 --adapt-prefill 0 --gemm-lds 1 --chunk 1024 --glm-prefill-stage 0 --oracle $KEEP" > "$O/p1.plan"
run g8_full 8 "$O/p1.plan"; rc1=$RC; ok1=1; exact_ok "$O/g8_full/gate_run.log" || ok1=0
{ echo "c32u --tokens-file $T136 --ctx 4096 --greedy 8 --chunk 32 --glm-prefill-stage 0 --adapt-prefill 0 --oracle $REF5/g136_eager"
  echo "c512u --tokens-file $T136 --ctx 4096 --greedy 8 --chunk 512 --glm-prefill-stage 0 --adapt-prefill 0 --oracle $REF5/g136_eager"
  echo "long512u --layers 0-3 --no-head --tokens-file $P2200 --ctx 4096 --chunk 512 --glm-prefill-stage 0 --oracle $REF5/g2200"; } > "$O/p2.plan"
run g8_small 8 "$O/p2.plan"; rc2=$RC; ok2=1; exact_ok "$O/g8_small/gate_run.log" || ok2=0
echo "=== EXACTNESS G=8: g8_full engine rc=$rc1 exact=$ok1 | g8_small engine rc=$rc2 exact=$ok2"
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ] || [ "$ok1" -ne 1 ] || [ "$ok2" -ne 1 ]; then
  echo "=== VERDICT: G=8 is NOT adopted (exactness gate failed or an engine died); timing skipped"
  say_end 1
fi
echo "=== G=8 is bit-exact at chunk 1024 / 512 / 32 in place: timing next"
{ echo "t_a $TP"; echo "t_b $TP"; } > "$O/pt.plan"
run t_g4a 4 "$O/pt.plan"; rt1=$RC
run t_g8  8 "$O/pt.plan"; rt2=$RC
run t_g4b 4 "$O/pt.plan"; rt3=$RC
echo "=== prefill ms/token, 8 192 tokens, 262 144 cells, chunk 1024, in place (G=4, G=8, G=4)"
for t in t_g4a t_g8 t_g4b; do timing "$O/$t/gate_run.log" "$t"; done | tee "$O/timing.txt"
awk '{ v[$1] = v[$1] " " $3; s[$1] += $3; n[$1]++ }
     END { if (n["t_g4a"] == 2 && n["t_g8"] == 2 && n["t_g4b"] == 2) {
             g4 = (s["t_g4a"] + s["t_g4b"]) / 4; g8 = s["t_g8"] / 2
             printf "means: G=4 %.4f  G=8 %.4f  ratio G8/G4 %.4f (%.1f %%)\n", g4, g8, g8 / g4, (g8 / g4 - 1) * 100
           } else print "incomplete: a timing config did not finish" }' "$O/timing.txt" | tee -a "$O/timing.txt"
say_end $(( rt1 + rt2 + rt3 > 0 ? 1 : 0 ))
