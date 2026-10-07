#!/bin/bash
# glm_rowsplit_ab_chain.sh -- the SPEED half of the --gemv-rowsplit gate, designed for the noise the first gate showed (2026-10-07). glm_rowsplit_chain.sh proved the flag BIT-EXACT (18 configs,
# 1 788 taps, not_exact=0) but its palindrome could not separate the effect from noise: the FIRST timing arm read 64.77 ms, its repeat 59.62 (the same first-arm penalty as the 2026-10-07 stack
# chain: 70.35 / 66.11), and rowsplit-on arms spread 58.8-62.2. So: ONE process, one load, two blocks, each = a discarded WARM-UP arm (rowsplit 0), then A B B A A B B A
# (A rowsplit 0 = main, B rowsplit 1 waves auto), each arm 8 192 prefill tokens at 262 144 cells + 64 settle + 48 timed decode tokens at the shipped flags:
#   block 0: --adapt 0 (placement FIXED at the generic start: no adaptation noise, a slower token with the fetch heavier, the same absolute GEMV saving expected);
#   block 1: --adapt 1 (the shipped adaptation: what the owner sees, noisier).
# The report prints every arm, the mean of A and B per block and the PAIRED differences (adjacent A/B pairs: the palindrome's neighbours), so a first-arm effect cannot hide.
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <engine log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_rowsplit_ab_chain.sh > ~/bench/glm_rowsplit_ab_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_rowsplit_ab_chain.log ~/bench/franken/glm5/rowsplit_ab/gate_run.log
# Env: RB_BIN (default ~/bench/franken_bin/franken_decode_glm.rowsplit = franken-engine gemv-rowsplit e10f196, now in main: `make -C franken/decode gpu`), RB_OUT (default ~/bench/franken/glm5/rowsplit_ab).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${RB_BIN:-$HOME/bench/franken_bin/franken_decode_glm.rowsplit}
VERDICT=${RB_VERDICT:-$HOME/src/franken-engine/franken/decode/glm5_gate_verdict.sh}
O=${RB_OUT:-$HOME/bench/franken/glm5/rowsplit_ab}; rm -rf "$O"; mkdir -p "$O"
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5; PR=$B/prose8400.txt
head -c 100000 $B/prose2200.txt | tr ' ' '\n' | head -1500 | tr '\n' ' ' > "$O/prose1500.txt"
C136="--tokens-file $B/t136.txt --ctx 4096 --greedy 8"
D1500="--tokens-file $O/prose1500.txt --ctx 4096 --chunk 512 --greedy 64"
AD="--adapt 1 --adapt-every 1 --adapt-mb-per-token 512 --adapt-verify 1"
TP="--tokens-file $PR --ctx 262144 --chunk 1024 --time-prefill 8192 --time 48 --time-settle 64 --adapt-prefill 0"
NEW="--fetch-assign optimal --gemv-wave-reduce 1"
: > "$O/plan.txt"; : > "$O/checks.txt"
cfg() { local n=$1 c=$2; shift 2; echo "$n $*" >> "$O/plan.txt"; echo "$n $c" >> "$O/checks.txt"; }
for blk in 0 1; do
  AF="--adapt $blk"
  cfg ab${blk}_w   none  $TP $NEW $AF --gemv-rowsplit 0
  for k in a1 b1 b2 a2 a3 b3 b4 a4; do
    case $k in a*) R=0;; *) R=1;; esac
    cfg ab${blk}_$k none $TP $NEW $AF --gemv-rowsplit $R
  done
done
echo "=== glm_rowsplit_ab start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; echo "=== glm_rowsplit_ab exit rc=2 $(date -Is)"; exit 2; }
[ -d "$B/gpu5/g136_eager" ] && [ -f "$PR" ] && [ -f "$B/t136.txt" ] || { echo "FATAL: references or prompts missing"; echo "=== glm_rowsplit_ab exit rc=2 $(date -Is)"; exit 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; echo "=== glm_rowsplit_ab exit rc=3 $(date -Is)"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?
echo "engine rc=$rc"
grep -aE "HIP error|Memory access|out of memory" "$O/gate_run.log" | head -2 | cut -c1-170
"$VERDICT" "$O/gate_run.log" "$O/checks.txt" $rc 2>&1 | cut -c1-200 | tee "$O/verdict.txt"
grep -aE "vram_dev[0-9]_steady" "$O/gate_run.log" | tail -3 | cut -c1-120
cat > "$O/report.py" <<'PYEOF'
import re, sys
log = open(sys.argv[1], errors="replace").read()
cur = None; res = {}
for line in log.splitlines():
    m = re.match(r"=== gate-plan config (\S+)", line)
    if m: cur = m.group(1)
    m = re.match(r"glm5_decode_ms_median=([0-9.]+)", line)
    if m and cur: res[cur] = float(m.group(1))
for blk, lab in ((0, "adapt 0 (placement fixed)"), (1, "adapt 1 (shipped)")):
    print(f"--- block {blk}: {lab}; decode ms/token, median of 48 at depth ~8 290")
    order = ["w", "a1", "b1", "b2", "a2", "a3", "b3", "b4", "a4"]
    row = [(k, res.get(f"ab{blk}_{k}")) for k in order]
    print("  " + "  ".join(f"{k}={v:.2f}" if v else f"{k}=?" for k, v in row))
    A = [res.get(f"ab{blk}_{k}") for k in ("a1", "a2", "a3", "a4")]
    B = [res.get(f"ab{blk}_{k}") for k in ("b1", "b2", "b3", "b4")]
    if None in A or None in B:
        print("  incomplete"); continue
    mA = sum(A) / 4; mB = sum(B) / 4
    pairs = list(zip(A, B))      # the palindrome's neighbours: (a1 b1) (b2 a2) (a3 b3) (b4 a4), each as (A, B); d = B - A
    d = [b - a for a, b in pairs]
    sd = (sum((x - sum(d) / 4) ** 2 for x in d) / 3) ** 0.5
    sA = (sum((x - mA) ** 2 for x in A) / 3) ** 0.5; sB = (sum((x - mB) ** 2 for x in B) / 3) ** 0.5
    print(f"  A (rowsplit 0) mean {mA:.2f} sd {sA:.2f} | B (rowsplit 1) mean {mB:.2f} sd {sB:.2f} | B-A {mB - mA:+.2f} ms ({100 * (mB / mA - 1):+.1f} %)")
    print(f"  paired B-A over the four neighbour pairs: " + " ".join(f"{x:+.2f}" for x in d) + f" | mean {sum(d) / 4:+.2f} sd {sd:.2f}; warm-up arm w={res.get(f'ab{blk}_w')}")
PYEOF
python3 "$O/report.py" "$O/gate_run.log" | tee "$O/report.txt"
echo "=== glm_rowsplit_ab exit rc=$rc $(date -Is)"
exit $rc
