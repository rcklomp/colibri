#!/bin/bash
# glm_adapt2_chain.sh -- the SECOND placement-policy sweep (2026-10-07): --adapt-prefill-cap N (franken-engine branch adapt-cap, commit 38757b6) against the controls.
# Round 1 (glm_adapt_chain.sh, record section L5-GLM-ADAPT): the placement average is dominated by a long prefill (the held span enters as one 8 192-token snapshot), decode barely moves it;
#  a short half-life fixes that (prose8192: 1 760 -> 671 MB missed a token, decode 65.8 -> 50.3 ms with half-life 32) but moves 2-3x the swap bytes, and the real service keeps its average
#  for hours, where a long half-life may be right. The cap makes a held span count as at most N tokens: the prompt seeds the placement without swamping it, at any half-life.
# ONE process, one load, shipped flags, every config from the same generic placement (as round 1). S1 = the 696-token chat prompt, 256 settle + 256 timed decode tokens; S2 = prose8400
#  prefilled to 8 192, 256 settle + 256 timed. Per config: the hit / missed-MB curve in four parts of the decode, the swap volume, the decode median; adapt_verify on the controls and the last arm.
# Arms: d0 (default), h128 (round 1's best moderate), h32 (S2 only: round 1's best), then cap 64 / 256 x half-life 128 / 512 / 2048, and cap 64 + half-life 128 with a 512 MB/token swap budget
#  (the cap shrinks the first round's budget too: this arm separates the two effects).
# VRAM budget: no new allocation (the cap is host arithmetic in the adapter); the steady free VRAM of the shipped config (796 / 882 / 522 MiB without decode) stands.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> $AD_OUT/gate_run.log:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_adapt2_chain.sh > ~/bench/glm_adapt2_chain.log 2>&1 < /dev/null &
# Env: AD_BIN (default the adapt-cap worktree's franken_decode_glm_adcap), AD_OUT (default ~/bench/franken/glm5/adapt2).
set -u
BIN=${AD_BIN:-$HOME/src/franken-engine-adcap/franken/decode/franken_decode_glm_adcap}
VERDICT=${AD_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${AD_OUT:-$HOME/bench/franken/glm5/adapt2}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
S1="--tokens-file $B/chat_lookahead_ids.txt --ctx 262144 --chunk 1024 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --time 256 --time-settle 256"
S2="--tokens-file $B/prose8400.txt --ctx 262144 --chunk 1024 --time-prefill 8192 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --time 256 --time-settle 256"
pol() { echo "--adapt 1 --adapt-every $1 --adapt-mb-per-token $2 --adapt-halflife $3 --adapt-margin $4 --adapt-hyst $5 --adapt-prefill-cap ${6:-0}"; }
: > "$O/plan.txt"; : > "$O/checks.txt"; : > "$O/order.txt"
cfg() { local n=$1; shift; echo "$n $*" >> "$O/plan.txt"; echo "$n none" >> "$O/checks.txt"; echo "$n" >> "$O/order.txt"; }
#   tag             scenario policy: every  mb/token  halflife  margin  hyst  cap
cfg ad2_c_d0        $S1 $(pol 16  64 2048 1.0 0.25   0) --adapt-verify 1
cfg ad2_c_h128      $S1 $(pol 16  64  128 1.0 0.25   0)
cfg ad2_c_c64h128   $S1 $(pol 16  64  128 1.0 0.25  64)
cfg ad2_c_c64h512   $S1 $(pol 16  64  512 1.0 0.25  64)
cfg ad2_c_c64h2048  $S1 $(pol 16  64 2048 1.0 0.25  64)
cfg ad2_c_c256h128  $S1 $(pol 16  64  128 1.0 0.25 256)
cfg ad2_c_c256h512  $S1 $(pol 16  64  512 1.0 0.25 256)
cfg ad2_c_c256h2048 $S1 $(pol 16  64 2048 1.0 0.25 256)
cfg ad2_c_c64m512   $S1 $(pol 16 512  128 1.0 0.25  64)
cfg ad2_p_d0        $S2 $(pol 16  64 2048 1.0 0.25   0)
cfg ad2_p_h128      $S2 $(pol 16  64  128 1.0 0.25   0)
cfg ad2_p_h32       $S2 $(pol 16  64   32 1.0 0.25   0)
cfg ad2_p_c64h128   $S2 $(pol 16  64  128 1.0 0.25  64)
cfg ad2_p_c64h512   $S2 $(pol 16  64  512 1.0 0.25  64)
cfg ad2_p_c64h2048  $S2 $(pol 16  64 2048 1.0 0.25  64)
cfg ad2_p_c256h128  $S2 $(pol 16  64  128 1.0 0.25 256)
cfg ad2_p_c256h512  $S2 $(pol 16  64  512 1.0 0.25 256)
cfg ad2_p_c256h2048 $S2 $(pol 16  64 2048 1.0 0.25 256)
cfg ad2_p_c64m512   $S2 $(pol 16 512  128 1.0 0.25  64)
cfg ad2_p_c64h128b  $S2 $(pol 16  64  128 1.0 0.25  64) --adapt-verify 1
N=$(grep -c . "$O/checks.txt")
echo "=== glm_adapt2 start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) configs=$N"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_adapt2 exit rc=2 $(date -Is)"; exit 2; }
[ -f "$B/chat_lookahead_ids.txt" ] && [ -f "$B/prose8400.txt" ] || { echo "FATAL: prompts missing"; echo "=== glm_adapt2 exit rc=2 $(date -Is)"; exit 2; }
. "$HOME/bench/chain_preflight.sh"
rig_quiet_wait 1800 || { echo "=== glm_adapt2 exit rc=3 $(date -Is)"; exit 3; }   # waits for the previous engine to release its VRAM (minutes), never refuses at once
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
echo "=== glm_adapt2 exit rc=$rc $(date -Is)"
exit $rc
