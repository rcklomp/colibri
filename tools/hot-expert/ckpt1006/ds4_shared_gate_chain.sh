#!/bin/bash
# ds4_shared_gate_chain.sh -- the DeepSeek gate for the shared ds4_gpu.inc change of franken-engine
# 473c5cc (the adapt counter read-back as a kernel into host-mapped memory instead of a D2H
# hipMemcpyAsync; GLM prefill needed it, DeepSeek links the same file). HANDOFF-2026-09-26 §4 item 2.
#   A = franken_decode_ds4.pre473  (franken-engine 5f409c2, the commit before the change)
#   B = franken_decode_ds4.tip     (franken-engine b5cf4e0, the tip; the diff A..B in DS4-relevant
#                                   code is ds4_gpu.inc + ds4_ops.h only: checked with git diff --stat)
# Both built by ~/bench/build_ds4_ab.sh with the same toolchain. Launch ONLY through run_chain.sh:
#   ssh -n -f rome 'setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/ds4gate/ds4_shared_gate_chain.sh \
#       > ~/bench/ds4gate/chain.log 2>&1 < /dev/null &'
# Part 1 (identity): B --oracle against A --dump, every tap must be cos=1 maxabs=0, in four
# configurations that reach the changed code: eager, graph replay, adaptation with a swap every
# round (--adapt-verify), and 64 prose tokens (the hot set moves most) with adaptation.
# Part 2 (timing): the depth probe 64 / 8192 / 64 with adaptation on (the read-back runs every 64
# tokens in decode and every chunk in prefill), A,B,B,A, verdict by gate_lib.sh.
set -u
OUT=$HOME/bench/ds4gate; mkdir -p "$OUT"
BINDIR=$HOME/bench/franken_bin
A_BIN=$BINDIR/franken_decode_ds4.pre473
B_BIN=$BINDIR/franken_decode_ds4.tip
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
HE=$HOME/src/colibri/tools/hot-expert
. "$HE/gate_lib.sh"
M=$HOME/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
DSDIR=$(dirname "$M")
P=$HOME/bench/m2/deepseek_b60            # the identity configs' placement (ds4_gpu_gate.sh's default)
MIX=$HOME/bench/m2/deepseek_mix          # the timing placement (as §L5-DS4-STEP5 / §BASELINE-7.0.0-38-DS4-GLM)
AD_DIR=$HOME/bench/franken/ds4/adapt
MM=$AD_DIR/mmlu_8k_ids.txt
T="0 671 6102 294 8760 344"              # BOS + "The capital of France is"
G="--model $M --tokens $T --placement $P --expert-gb 20"
AD="--adapt 1 --adapt-verify 1"

D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
        -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; echo "vram still $(vram_max) MiB"; }
nb() { echo "not_bitexact=$(grep '^oracle .* cos=' "$1" | grep -v 'cos=1.000000 maxabs=0 ' | wc -l) taps=$(grep -c '^oracle .* cos=' "$1")"; }
FAILS=0
fail() { echo "GATE FAIL: $*"; FAILS=$((FAILS + 1)); }

echo "=== ds4_shared_gate start $(date -Is) kernel=$(uname -r)"
for f in "$A_BIN" "$B_BIN" "$MM" "$P/layer_3.csv" "$MIX/layer_3.csv" "$M"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }
done
echo "A=$(sha256sum "$A_BIN" | cut -c1-16) B=$(sha256sum "$B_BIN" | cut -c1-16)"
[ "$(vram_max)" -lt 1024 ] || { echo "FATAL: VRAM in use"; exit 2; }

echo "=== warm $(date +%T)"; cat "$DSDIR"/*.gguf > /dev/null
echo "=== preflight smoke (B) $(date +%T)"
PREFLIGHT_OWN_LOCK=1 "$HE/preflight.sh" bash -c '"$@"; rc=$?; [ $rc -eq 4 ] && rc=0; exit $rc' _ \
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1 -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin \
  -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" \
  "$B_BIN" $G --ctx 512 --greedy 4 || { echo "FATAL: smoke failed"; exit 2; }

# ---------------------------------------------------------------- part 1: identity
ident() {   # ident <label> <ref-args> <cand-args>   (A --dump, then B --oracle)
  local label=$1 ra=$2 ca=$3
  vram_wait; D "$A_BIN" $ra --dump "$OUT/ref_$label" > "$OUT/ref_$label.log"; local rc1=$?
  vram_wait; D "$B_BIN" $ca --oracle "$OUT/ref_$label" > "$OUT/cand_$label.log"; local rc2=$?
  local n; n=$(grep -c '^oracle .* cos=' "$OUT/cand_$label.log")
  local bad; bad=$(grep '^oracle .* cos=' "$OUT/cand_$label.log" | grep -v 'cos=1.000000 maxabs=0 ' | wc -l)
  echo "=== identity $label: rc(A)=$rc1 rc(B)=$rc2 taps=$n not_bitexact=$bad"
  grep -E "greedy_ids|^adapt_total|^adapt_verify" "$OUT/cand_$label.log" | cut -c1-200 | head -8
  if [ "$n" -lt 100 ] || [ "$bad" -ne 0 ] || [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then fail "identity $label (taps=$n bad=$bad rc=$rc1/$rc2)"; fi
  rm -rf "$OUT/ref_$label"       # the dump is large; the verdict is in the log
}
ident eager      "$G --ctx 512 --greedy 16 --adapt 0"                                   "$G --ctx 512 --greedy 16 --adapt 0"
ident graph      "$G --ctx 512 --greedy 16 --adapt 0"                                   "$G --ctx 512 --greedy 16 --adapt 0 --hip-graph 1"
ident adapt      "$G --ctx 512 --greedy 16 --adapt 0" "$G --ctx 512 --greedy 16 $AD --adapt-every 1 --adapt-mb-per-token 512"
P64="--model $M --tokens-file $MM --max-tokens 64 --placement $P --expert-gb 20 --ctx 512 --greedy 16"
ident adapt64    "$P64 --adapt 0" "$P64 $AD --adapt-every 1 --adapt-mb-per-token 512"
ident adapt_graph "$G --ctx 512 --greedy 16 --adapt 0" "$G --ctx 512 --greedy 16 $AD --adapt-every 1 --adapt-mb-per-token 512 --hip-graph 1"

# ---------------------------------------------------------------- part 2: timing A,B,B,A
DP="--model $M --tokens-file $MM --placement $MIX --expert-gb 20 --ctx 262144 --chunk 256 --probe-at 64,8192,64 --probe-n 32 --hip-graph 1 --adapt 1 --adapt-every 64 --adapt-mb-per-token 64 --adapt-halflife 2048"
declare -A S64 S8K
for arm in A B B A; do
  vram_wait
  bin=$A_BIN; [ $arm = B ] && bin=$B_BIN
  tag="time_${arm}_$(date +%H%M%S)"
  echo "=== timing arm $arm $(date +%T)"
  D "$bin" $DP > "$OUT/$tag.log"; rc=$?
  grep -E "^ds4_probe|load_s=|HIP error|fault" "$OUT/$tag.log" | cut -c1-200
  echo "rc=$rc"
  [ $rc -eq 0 ] || fail "timing arm $arm rc=$rc"
  m64=$(grep '^ds4_probe depth=64 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')
  m8k=$(grep '^ds4_probe depth=8192 ' "$OUT/$tag.log" | sed -n 's/.*ms_median=\([0-9.]*\).*/\1/p' | tr '\n' ' ')
  S64[$arm]="${S64[$arm]:-}$m64"; S8K[$arm]="${S8K[$arm]:-}$m8k"
done
echo "=== verdicts (decode ms/token, medians of 32; A = pre473, B = tip)"
gate_ab_verdict "decode at depth 64"   "${S64[A]}" "${S64[B]}"
gate_ab_verdict "decode at depth 8192" "${S8K[A]}" "${S8K[B]}"
echo "=== ds4_shared_gate end $(date -Is) failures=$FAILS"
[ "$FAILS" -eq 0 ] && echo "DS4 SHARED-CHANGE GATE: PASS" || echo "DS4 SHARED-CHANGE GATE: FAIL ($FAILS)"
exit $(( FAILS > 0 ))
