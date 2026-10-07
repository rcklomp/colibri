#!/bin/bash
# glm_adapt_chain.sh -- does the placement adaptation's POLICY leave decode speed on the table? (2026-10-07; handoff 06b section 4 item 2; record section L5-GLM-DECODE-TRACE)
# Why: the decode trace says the critical fetch (35.2 ms of ~62) sits at ~48 GB/s = 92 % of M4's 52 GB/s aggregate, so splitting the missed slabs finer cannot help (offline model
#  ckpt1006/lane_model.py: a perfect continuous split at 52 GB/s is 0.2 ms better than the shipped slab table). What is left is FEWER MISSED BYTES. The 2026-10-07 stack chain log shows,
#  in the timed window after an 8 192-token prefill, hit 0.62-0.65 and 1 400-1 500 MB missed a token, with swaps down to 2-7 per 16 tokens, while a 136-token run reached hit 0.875 and
#  492 MB a token: the EMA (half-life 2048 tokens, prefill delta folded in as one snapshot of 8 192 tokens) is the PROMPT's routing histogram and decode barely moves it.
#  Placement changes no bit (glm5_adapt.h), so this is a policy question only, answered by knobs that already exist: --adapt-halflife / -every / -mb-per-token / -margin / -hyst.
# ONE process, one load, shipped flags (ctx 262 144, chunk 1024, in place, adaptation on). Every config starts from the same generic placement (hit ~0.34 at prefill end), so the arms are
#  independent. S1 = the 696-token chat prompt (a coding question, reasoning on: an assistant-style answer, the owner's shape), 256 settle + 256 timed decode tokens;
#  S2 = prose8400 prefill (the benchmark shape), 256 settle + 128 timed. Per config: the hit / missed-MB curve in four equal parts of the decode, the swap volume, the decode median.
# VRAM budget (CLAUDE.md "Before a build that allocates device memory"): no new allocation: swaps copy into the existing slots, the shipped steady free VRAM (732 / 818 / 458 MiB) stands.
# Exactness lines are not needed (placement-only; the engine's own --adapt-verify checks every table entry against the mirror at the end of a run: added to d0 and the last arm).
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> $AD_OUT/gate_run.log:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_adapt_chain.sh > ~/bench/glm_adapt_chain.log 2>&1 < /dev/null &
# Env: AD_BIN (default franken-engine main's franken_decode_glm), AD_OUT (default ~/bench/franken/glm5/adapt), AD_CFGS (override the config list, one "tag|flags" per line; default below).
set -u
BIN=${AD_BIN:-$HOME/src/franken-engine/franken/decode/franken_decode_glm}
VERDICT=${AD_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${AD_OUT:-$HOME/bench/franken/glm5/adapt}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
S1="--tokens-file $B/chat_lookahead_ids.txt --ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --time 256 --time-settle 256"
S2="--tokens-file $B/prose8400.txt --ctx 262144 --chunk 1024 --time-prefill 8192 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --time 128 --time-settle 256"
pol() { echo "--adapt 1 --adapt-every $1 --adapt-mb-per-token $2 --adapt-halflife $3 --adapt-margin $4 --adapt-hyst $5"; }
: > "$O/plan.txt"; : > "$O/checks.txt"; : > "$O/order.txt"
cfg() { local n=$1; shift; echo "$n $*" >> "$O/plan.txt"; echo "$n none" >> "$O/checks.txt"; echo "$n" >> "$O/order.txt"; }
#   tag          scenario policy: every  mb/token  halflife  margin  hyst
cfg ad_c_d0      $S1 $(pol 16  64 2048 1.0 0.25) --adapt-verify 1
cfg ad_c_h512    $S1 $(pol 16  64  512 1.0 0.25)
cfg ad_c_h128    $S1 $(pol 16  64  128 1.0 0.25)
cfg ad_c_h32     $S1 $(pol 16  64   32 1.0 0.25)
cfg ad_c_e4h128  $S1 $(pol  4  64  128 1.0 0.25)
cfg ad_c_m256    $S1 $(pol 16 256  128 1.0 0.25)
cfg ad_c_loose   $S1 $(pol 16  64  128 0.5 0.05)
cfg ad_c_fast    $S1 $(pol  4 256   64 0.5 0.05)
cfg ad_p_d0      $S2 $(pol 16  64 2048 1.0 0.25)
cfg ad_p_h128    $S2 $(pol 16  64  128 1.0 0.25)
cfg ad_p_h32     $S2 $(pol 16  64   32 1.0 0.25)
cfg ad_c_d0b     $S1 $(pol 16  64 2048 1.0 0.25) --adapt-verify 1
N=$(grep -c . "$O/checks.txt")
echo "=== glm_adapt start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) configs=$N"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_adapt exit rc=2 $(date -Is)"; exit 2; }
[ -f "$B/chat_lookahead_ids.txt" ] && [ -f "$B/prose8400.txt" ] || { echo "FATAL: prompts missing"; echo "=== glm_adapt exit rc=2 $(date -Is)"; exit 2; }
. "$HOME/bench/chain_preflight.sh"
rig_quiet_wait 1800 || { echo "=== glm_adapt exit rc=3 $(date -Is)"; exit 3; }   # waits for the previous engine to release its VRAM (minutes), never refuses at once
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
cat > "$O/report.awk" <<'EOF'
# per config: the decode windows (adapt_all lines with tokens=16) in four equal parts -> mean hit and mean missed MB a token; swap MB over the whole decode; the decode median
/^=== gate-plan config/ { c = $4; nw[c] = 0 }
/^adapt_all / { split($3, t, "="); if (t[2] + 0 == 16 && c != "") { n = ++nw[c]; split($4, h, "="); split($5, m, "="); split($7, s, "="); H[c, n] = h[2] + 0; MB[c, n] = m[2] + 0; SW[c] += s[2] } }
/^glm5_decode_ms_median=/ { split($1, a, "="); med[c] = a[2] }
END {
  while ((getline l < ord) > 0) {
    c = l; n = nw[c]
    if (n < 4) { printf "%-12s incomplete (%d decode windows)\n", c, n; continue }
    line = ""
    for (q = 0; q < 4; q++) {
      lo = int(q * n / 4) + 1; hi = int((q + 1) * n / 4); sh = 0; sm = 0
      for (i = lo; i <= hi; i++) { sh += H[c, i]; sm += MB[c, i] }
      line = line sprintf("  q%d hit %.3f miss %6.0f", q + 1, sh / (hi - lo + 1), sm / (hi - lo + 1))
    }
    printf "%-12s%s | swap %6.0f MB | decode median %s ms\n", c, line, SW[c], (c in med) ? med[c] : "?"
  }
}
EOF
echo "=== decode windows in four equal parts (hit, missed MB a token), swap volume, decode median ms (timed part = the last half of the windows)"
awk -v ord="$O/order.txt" -f "$O/report.awk" "$O/gate_run.log" | tee "$O/report.txt"
grep -aE "^adapt_total|adapt-verify|verify" "$O/gate_run.log" | tail -8 | cut -c1-180
echo "=== glm_adapt exit rc=$rc $(date -Is)"
exit $rc
