#!/usr/bin/env bash
# run_cpulane.sh -- the cpulane microbenchmark: part A (CPU GEMV GB/s vs threads, host) then part B (three H2D
# streams + CPU GEMV workers, in docker). Meant to be started through tools/hot-expert/run_chain.sh, which holds
# the rig lock. Needs: every GPU's VRAM use < 1 GiB, nothing else heavy on the CPU.
#
# Result lines (grep-friendly, one per measurement):
#   A,<type>,<mode>,threads=N,<ws>,GBps=..,expert_ms=..,...
#   B1,card=<c>,GBps=..          B2,card0_GBps=..,...,agg_GBps=..
#   B4,N=..,cpu_GBps=..,qtype=.. B3,N=..,card0_GBps=..,card1_GBps=..,card2_GBps=..,agg_GBps=..,cpu_GBps=..,...
#   B3vsA / B4vsA                (this script: B3/B4 CPU rate against part A whole-mode dram at the same N)
# last line: === cpulane exit rc=N
#
# Env overrides: CPULANE_A_ARGS, CPULANE_B_ARGS (extra arguments for the two programs; see their usage).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
START=$(date +%s)
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
OUT="$HERE/out/$(date +%Y%m%d-%H%M%S)"
CNAME="cpulane_b_$$"

finish() {
    local rc=$1
    echo "=== cpulane exit rc=$rc elapsed=$(( $(date +%s) - START ))s $(date -Is)"
    exit "$rc"
}

echo "=== cpulane start $(date -Is) host=$(hostname) out=$OUT"

# ---- 1. refuse unless every GPU is idle (< 1 GiB VRAM in use)
LIMIT=${CPULANE_VRAM_LIMIT:-$((1 << 30))}   # bytes; the override exists only to test the refusal path
shopt -s nullglob
vram_files=(/sys/class/drm/card[0-9]/device/mem_info_vram_used)
shopt -u nullglob
if [ "${#vram_files[@]}" -eq 0 ]; then
    echo "REFUSE: no /sys/class/drm/card*/device/mem_info_vram_used found, cannot verify the GPUs are idle"
    finish 3
fi
for f in "${vram_files[@]}"; do
    used=$(cat "$f" 2>/dev/null || echo 999999999999)
    echo "vram_used $(basename "$(dirname "$(dirname "$f")")")=$used"
    if [ "$used" -ge "$LIMIT" ]; then
        echo "REFUSE: $f = $used B >= 1 GiB, a GPU is in use (engine still loaded?)"
        finish 3
    fi
done

mkdir -p "$OUT"

# ---- 2. environment record
echo "--- environment"
uptime
lscpu -e=CPU,CORE,SOCKET,ONLINE,MAXMHZ 2>/dev/null | head -20
echo "thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
free -g | head -2

# ---- 3. build (no-op when up to date; compiles only, never starts a binary)
echo "--- build"
nice -n 19 make 2>&1 | tail -5
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -x ./bench_cpu_gemv ] || [ ! -x ./bench_h2d_cpu ]; then
    echo "FAIL: build"
    finish 4
fi

RC=0

# ---- 4. part A: CPU only, host
echo "=== part A (CPU GEMV, host) $(date -Is)"
# shellcheck disable=SC2086
timeout -k 10 480 ./bench_cpu_gemv ${CPULANE_A_ARGS:-} 2>&1 | tee "$OUT/A.txt"
rcA=${PIPESTATUS[0]}
echo "part A rc=$rcA"
[ "$rcA" -ne 0 ] && RC=1
if [ -z "${CPULANE_A_ARGS:-}" ]; then
    nA=$(grep -c '^A,' "$OUT/A.txt")
    if [ "$nA" -ne 128 ]; then echo "FAIL: part A printed $nA result lines, expected 128"; RC=1; fi
fi
grep -q '^A_DONE rc=0' "$OUT/A.txt" || { echo "FAIL: part A did not end with A_DONE rc=0"; RC=1; }

# ---- 5. part B: concurrent pair, docker (same device flags as ~/bench/glm_adapt_chain.sh)
echo "=== part B (3x H2D + CPU GEMV, docker) $(date -Is)"
# shellcheck disable=SC2086
timeout -k 20 900 docker run --rm --name "$CNAME" \
    --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib -e HOME=/home/ronald -v /home/ronald:/home/ronald -w "$HERE" \
    "$IMG" ./bench_h2d_cpu ${CPULANE_B_ARGS:-} 2>&1 | tee "$OUT/B.txt"
rcB=${PIPESTATUS[0]}
docker kill "$CNAME" >/dev/null 2>&1 || true
echo "part B rc=$rcB"
[ "$rcB" -ne 0 ] && RC=1
grep -q '^B_DONE rc=0' "$OUT/B.txt" || { echo "FAIL: part B did not end with B_DONE rc=0"; RC=1; }
if [ -z "${CPULANE_B_ARGS:-}" ]; then
    n1=$(grep -c '^B1,' "$OUT/B.txt"); n2=$(grep -c '^B2,' "$OUT/B.txt"); n3=$(grep -c '^B3,' "$OUT/B.txt"); n4=$(grep -c '^B4,' "$OUT/B.txt")
    if [ "$n1" -ne 3 ] || [ "$n2" -ne 1 ] || [ "$n3" -ne 6 ] || [ "$n4" -ne 6 ]; then
        echo "FAIL: part B result lines B1=$n1 B2=$n2 B3=$n3 B4=$n4, expected 3/1/6/6"; RC=1
    fi
fi

# ---- 6. B3 / B4 CPU rate against part A (whole mode, dram, same thread count)
echo "=== B vs A (cpu GBps, whole mode, DRAM-resident; N workers vs N threads in part A) $(date -Is)"
awk -F, '
    FNR == NR { if ($1 == "A" && $3 == "whole" && $5 == "dram") { split($4, t, "="); split($6, g, "="); A[$2 SUBSEP t[2]] = g[2] } next }
    $1 == "B3" || $1 == "B4" {
        n = ""; cg = ""; q = "";
        for (i = 2; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "N") n = kv[2]; else if (kv[1] == "cpu_GBps") cg = kv[2]; else if (kv[1] == "qtype") q = kv[2] }
        a = ((q SUBSEP n) in A) ? A[q SUBSEP n] : "";
        printf "%svsA,N=%s,qtype=%s,cpu_GBps=%s,A_whole_dram_GBps=%s,ratio=%s\n", $1, n, q, cg, (a == "" ? "na" : a), (a == "" || a + 0 == 0 ? "na" : sprintf("%.3f", cg / a))
    }' "$OUT/A.txt" "$OUT/B.txt" | tee "$OUT/vsA.txt"

echo "results saved in $OUT (A.txt B.txt vsA.txt)"
finish "$RC"
