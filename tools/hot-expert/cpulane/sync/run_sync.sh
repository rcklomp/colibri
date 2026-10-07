#!/usr/bin/env bash
# run_sync.sh -- the GPU -> host -> GPU synchronisation microbenchmark (S1 ping-pong, S2/S2b layer pattern, S3/S4 under H2D load).
# Meant to be started through ~/src/colibri/tools/hot-expert/run_chain.sh, which holds the rig lock. Needs: every GPU's
# VRAM use < 1 GiB, nothing else heavy on the CPU (cores 0-4 of the 7F32 are used by the benchmark's threads).
#
# Result lines (grep-friendly, one per measurement; see README.md):
#   S1,payload=..,n=..,rtt_us_med=..,rtt_us_p99=..,rtt_us_min=..,...         S1delta,...
#   S2base,work=..,W=..,base_us=..       S2,wait=spin,work=..,W=..,tc=..,pattern_us=..,d_us=..,d_vs_base_us=..,spin_us_med=..
#   S2b,wait=streamwait,...   or  S2b,unsupported,reason=..
#   S3ref,card0_GBps=..   S3,wait=..,W=600,tc=570,...,d_us=..,d_quiet_us=..,card0_GBps=..,card0_vs_ref=..   S4ref / S4 (cards 1,2 only)
# Program end:  SYNC_DONE rc=N       Any bounded wait that timed out:  SYNC_ABORT reason=...   (program rc 5)
# Last line of this script:  === sync exit rc=N      rc 0 ok | 1 data errors / failed checks | 3 refused (GPU busy) | 4 build failed | 5 abort
#
# Env: SYNC_ARGS (extra program arguments, replaces the default budget), SYNC_VRAM_LIMIT (bytes; only to test the refusal path),
#      SYNC_TIMEOUT (seconds, hard limit of the program, default 690).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
START=$(date +%s)
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
OUT="$HERE/out/$(date +%Y%m%d-%H%M%S)"
CNAME="cpulane_sync_$$"

finish() {
    local rc=$1
    echo "=== sync exit rc=$rc elapsed=$(( $(date +%s) - START ))s $(date -Is)"
    exit "$rc"
}

echo "=== sync start $(date -Is) host=$(hostname) out=$OUT"

# ---- 1. refuse unless every GPU is idle (< 1 GiB VRAM in use)
LIMIT=${SYNC_VRAM_LIMIT:-$((1 << 30))}   # bytes; the override exists only to test the refusal path
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
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -x ./bench_sync ]; then
    echo "FAIL: build"
    finish 4
fi

RC=0

# ---- 4. the benchmark, in docker (device flags as in ../run_cpulane.sh / ~/bench/glm_adapt_chain.sh)
echo "=== bench_sync $(date -Is)"
DEFAULT_ARGS="--budget-s 560"
TMO=${SYNC_TIMEOUT:-690}
# shellcheck disable=SC2086
timeout -k 20 "$TMO" docker run --rm --name "$CNAME" \
    --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
    -e LD_LIBRARY_PATH=/opt/rocm/lib -e HOME=/home/ronald -v /home/ronald:/home/ronald -w "$HERE" \
    "$IMG" ./bench_sync ${SYNC_ARGS:-$DEFAULT_ARGS} 2>&1 | tee "$OUT/sync.txt"
rcB=${PIPESTATUS[0]}
docker kill "$CNAME" >/dev/null 2>&1 || true
echo "bench_sync rc=$rcB"

# ---- 5. checks
if grep -q '^SYNC_ABORT' "$OUT/sync.txt"; then
    echo "FAIL: SYNC_ABORT -- a bounded wait timed out or a protocol check failed:"
    grep '^SYNC_ABORT' "$OUT/sync.txt" | head -3
    RC=5
elif [ "$rcB" -eq 124 ] || [ "$rcB" -eq 137 ]; then
    echo "FAIL: bench_sync hit the hard time limit (${TMO}s) and was killed"
    RC=5
elif [ "$rcB" -ne 0 ]; then
    echo "FAIL: bench_sync rc=$rcB"
    RC=1
fi
grep -q '^SYNC_DONE rc=0' "$OUT/sync.txt" || { echo "FAIL: no 'SYNC_DONE rc=0' line"; [ "$RC" -eq 0 ] && RC=1; }
if grep -q 'data mismatches\|payload mismatches' "$OUT/sync.txt"; then
    echo "FAIL: data mismatches reported (a flag was seen before its data, or the reverse)"
    [ "$RC" -eq 0 ] && RC=1
fi
if [ -z "${SYNC_ARGS:-}" ] && [ "$RC" -eq 0 ]; then
    n1=$(grep -c '^S1,' "$OUT/sync.txt");  nb=$(grep -c '^S2base,' "$OUT/sync.txt"); n2=$(grep -c '^S2,' "$OUT/sync.txt")
    n2b=$(grep -c '^S2b,' "$OUT/sync.txt"); n3=$(grep -c '^S3,' "$OUT/sync.txt");   n4=$(grep -c '^S4,' "$OUT/sync.txt")
    echo "lines: S1=$n1 S2base=$nb S2=$n2 S2b=$n2b S3=$n3 S4=$n4 (expected 4 / 6 / 18 / 9 or 1 (unsupported) / 2 / 2)"
    if [ "$n1" -ne 4 ] || [ "$nb" -ne 6 ] || [ "$n2" -ne 18 ] || { [ "$n2b" -ne 9 ] && [ "$n2b" -ne 1 ]; } || [ "$n3" -ne 2 ] || [ "$n4" -ne 2 ]; then
        echo "FAIL: unexpected number of result lines (a phase was skipped: time budget or a failed config?)"
        grep '^S_SKIP' "$OUT/sync.txt" | head -5
        RC=1
    fi
fi

# ---- 6. the headline numbers, for the chain log
echo "=== headline (d = per-layer fixed cost, us; see README.md)"
grep -E '^S2,wait=spin,work=spin,W=(0|300|600),tc=(0|570),' "$OUT/sync.txt" 2>/dev/null |
    awk -F, '{ for (i = 1; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "d_us" || kv[1] == "d_vs_base_us" || kv[1] == "pattern_us" || kv[1] == "spin_us_med") v[kv[1]] = kv[2] }
               printf "%s %s %s %s pattern_us=%s d_us=%s d_vs_base_us=%s spin_us_med=%s\n", $1, $2, $4, $5, v["pattern_us"], v["d_us"], v["d_vs_base_us"], v["spin_us_med"] }' | head -12
grep -E '^S1,|^S1delta' "$OUT/sync.txt" | cut -c1-200

echo "results saved in $OUT (sync.txt)"
finish "$RC"
