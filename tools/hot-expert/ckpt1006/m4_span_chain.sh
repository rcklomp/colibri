#!/bin/bash
# m4_span_chain.sh -- does the pinned host SOURCE span / allocation flags lower host->card DMA (record §L5-GLM-EMBED)?
# M4 variants (build_m4_span.sh): stock (2 GB Portable) | small2m (2 GB Mapped|Portable) | big40 (40 GB Portable) | big40m (40 GB Mapped|Portable),
# one run each in palindrome order S s B M M B s S. Launch through run_chain.sh. Memory: up to 3 x 40 GB pinned.
set -u
OUT=$HOME/bench/m4span; mkdir -p "$OUT"
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
D() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
        -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -v /home/ronald:/home/ronald -w /home/ronald/bench "$IMG" "$@" 2>&1; }
vram_max() { m=0; for d in /sys/class/drm/card[0-9]/device; do u=$(( $(cat $d/mem_info_vram_used)/1048576 )); [ $u -gt $m ] && m=$u; done; echo $m; }
vram_wait() { for _ in $(seq 1 60); do [ "$(vram_max)" -lt 1024 ] && return 0; sleep 2; done; return 1; }
echo "=== m4_span start $(date -Is) free: $(free -g | sed -n 2p | awk '{print $7" GB available"}')"
n=0
for v in stock small2m big40 big40m big40m big40 small2m stock; do
  n=$((n+1)); bin=$OUT/m4_$v; [ -x "$bin" ] || { echo "FATAL: $bin"; exit 2; }
  vram_wait || { echo "FATAL: VRAM"; exit 2; }
  echo "=== run $n: $v $(date +%T)"
  t0=$(date +%s); D "$bin" --json "$OUT/run${n}_$v.json" > "$OUT/run${n}_$v.txt"; rc=$?; echo "rc=$rc in $(( $(date +%s) - t0 )) s"
done
python3 - "$OUT" <<'PY'
import sys, re, glob, collections
out = sys.argv[1]; data = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(out + "/run*_*.txt")):
    v = re.search(r"run\d+_(\w+)\.txt", f).group(1)
    for l in open(f, errors="replace"):
        m = re.match(r"(m4_\S+_agg_gbps)=([0-9.]+)", l)
        if m: data[m.group(1)][v].append(float(m.group(2)))
print("=== GB/s by arm and variant (mean of the two runs of each; ratio to stock)")
print("%-62s %8s %8s %8s %8s | %6s %6s %6s" % ("arm", "stock", "small2m", "big40", "big40m", "s2m/st", "b40/st", "b40m/st"))
for k in sorted(data):
    d = data[k]; mean = lambda v: sum(d[v]) / len(d[v]) if d.get(v) else float("nan")
    st = mean("stock")
    print("%-62s %8.2f %8.2f %8.2f %8.2f | %6.2f %6.2f %6.2f" % (k[3:-9], st, mean("small2m"), mean("big40"), mean("big40m"), mean("small2m")/st, mean("big40")/st, mean("big40m")/st))
PY
echo "=== m4_span end $(date -Is)"
