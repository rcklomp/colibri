#!/bin/bash
# glm_q8bw_micro_chain.sh -- D7 of DECODE-OPEN-ITEMS-PLAN-2026-10-08.md: the Q8_0 LOAD-PATTERN BANDWIDTH MICROBENCHMARK on ONE card, no model load (2026-10-08; franken-engine branch d7-q8bw).
# Would a 16-byte-per-lane load read the big Q8_0 trunk matrices materially faster than the shipped one-byte-per-lane pattern (34-byte blocks, one wave a row, k_gemm_batch)?
# bench_q8bw (franken/decode/bench_q8bw.hip) #includes the real decode_gpu.hip and times, per size (1536x4096 6.7 MB; 8192x4096 and 4096x8192 35.7 MB; 4096x16384 71 MB; 24576x4096 107 MB):
#   A  the engine's k_gemm_batch (8 waves, 1 byte a lane)         Acopy  a line-for-line copy of A's loop at 4 / 8 / 16 waves and 8 / 16 / 32 blocks in flight (the depth control)
#   B  4 bytes a lane (dword)     C / Cnt  16 bytes a lane (global_load_dwordx4, plain / non-temporal)     CL  16 bytes a lane through an LDS ring with A's EXACT arithmetic
#   D  the streaming ceiling (float4 grid-stride, plain / non-temporal / one-shot grid) and Dcopy (hipMemcpy D2D)
# in the hot / cold / dep modes of bench_group (2 000 launches after 200 warm-up; cold = rotating copies, >= 256 MiB between reuses of a byte), us per launch AND GB/s, median of 3 interleaved
# passes. Checks (EXACT / MISMATCH): A against a host replay of its summation order; Acopy and CL bit for bit against A; B, C, Cnt against a host integer reference (x is a vector of small
# integers there, so every sum is exact and a byte missed or read twice shows); D against the host dword sum. Then one summary line a size: `q8bw SIZE: A x GB/s, B ..., C ..., CL ..., D ... ; C/A = ...`
# (cold; q8bw-dep and q8bw-hot in the same format). PLAN GATE: build D7b/D8 only if a wide variant is >= 8 % faster than A at 35 MB (C/A or CL/A >= 1.08 on 8192x4096 and 4096x8192; CL/A is the one
# that keeps A's arithmetic). The bench exits 0 whatever it finds; this chain's rc is the bench's rc (2 = no binary, 3 = rig not quiet, 124-ish/137 = the watchdog killed a hung kernel).
# Allocates up to ~1 GB on card 0 (the cold copies). Expected run time 6-12 minutes.
# Setup (once, from the Mac):  scp ckpt1006/glm_q8bw_micro_chain.sh rome:bench/ && ssh rome chmod +x bench/glm_q8bw_micro_chain.sh
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <bench log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_q8bw_micro_chain.sh > ~/bench/glm_q8bw_micro_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_q8bw_micro_chain.log ~/bench/franken/glm5/q8bw_micro/bench.log
# A first, short look (one small size, one pass, 400 launches; about 30 s) before the full run:
#   Q8W_ARGS="--only-size 1536x4096 --reps 1 --iters 400" setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_q8bw_micro_chain.sh > ~/bench/glm_q8bw_micro_chain.log 2>&1 < /dev/null &
# Env: Q8W_ENV (extra docker -e options, e.g. "-e AMD_LOG_LEVEL=3"), Q8W_ARGS (bench arguments: --only-size NAME, --reps N, --iters N, --no-cl, --no-d2d, --list), Q8W_TIMEOUT (s, default 1800;
# the bench container is killed after it), Q8W_BIN (default ~/bench/franken_bin/bench_q8bw, `make -C franken/decode bench-q8bw`), Q8W_OUT (default ~/bench/franken/glm5/q8bw_micro),
# Q8W_SAMPLE (1 = sample the cards' shader / memory clocks and power once a second beside the bench with gpu_sampler.py arm mode, if the rig's copy has it; default 1), Q8W_SAMPLER (default ~/bench/gpu_sampler.py).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${Q8W_BIN:-$HOME/bench/franken_bin/bench_q8bw}
O=${Q8W_OUT:-$HOME/bench/franken/glm5/q8bw_micro}; rm -rf "$O"; mkdir -p "$O"
SAMPLE=${Q8W_SAMPLE:-1}; SAMPLER=${Q8W_SAMPLER:-$HOME/bench/gpu_sampler.py}; SAMP=""
say_end() { [ -n "$SAMP" ] && kill "$SAMP" 2>/dev/null; echo "=== glm_q8bw_micro exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_q8bw_micro start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$HOME/src/franken-engine-d7" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker rm -f q8bw_micro >/dev/null 2>&1
if [ "$SAMPLE" = 1 ] && [ -f "$SAMPLER" ] && grep -q 'def arm_run' "$SAMPLER"; then
  python3 -I "$SAMPLER" arm "$O/clk" q8bw 1 > "$O/sampler.log" 2>&1 & SAMP=$!
  echo "clock sampler: $SAMPLER arm, 1 s, pid $SAMP -> $O/clk.gpu.tsv"
else
  echo "clock sampler: off ($( [ "$SAMPLE" = 1 ] && echo "$SAMPLER has no arm mode: copy ckpt1006/gpu_sampler.py to ~/bench" || echo "Q8W_SAMPLE=$SAMPLE" ))"
fi
( sleep ${Q8W_TIMEOUT:-1800}; docker kill q8bw_micro >/dev/null 2>&1 && echo "TIMEOUT: bench killed after ${Q8W_TIMEOUT:-1800} s (a hung kernel?)" ) &
WD=$!
docker run --name q8bw_micro --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 ${Q8W_ENV:-} \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" ${Q8W_ARGS:-} > "$O/bench.log" 2>&1
rc=$?
kill $WD 2>/dev/null
[ -n "$SAMP" ] && { kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null; SAMP=""; }
echo "bench rc=$rc"
grep -aE "HIP error|Memory access|out of memory|Segmentation|Aborted|error state|launch error" "$O/bench.log" | head -3 | cut -c1-200
head -n 3 "$O/bench.log" | cut -c1-260
echo "=== checks: $(grep -ac ' EXACT' "$O/bench.log") EXACT lines, $(grep -ac 'MISMATCH' "$O/bench.log") MISMATCH lines, $(grep -ac ' SKIP' "$O/bench.log") SKIP lines"
grep -a 'MISMATCH' "$O/bench.log" | head -20 | cut -c1-240
grep -a ' SKIP\|did NOT contract' "$O/bench.log" | head -12 | cut -c1-240
grep -aE "^  check (A|A1) " "$O/bench.log" | cut -c1-200
awk '/^=== SIZE/ { sz = $3 } /^  check / { if ($0 ~ / EXACT/) e[sz]++; else if ($0 ~ /MISMATCH/) m[sz]++; else o[sz]++; seen[sz] = 1 } END { for (z in seen) printf "checks %s %d EXACT, %d MISMATCH, %d other (skipped)\n", z, e[z], m[z], o[z] }' "$O/bench.log" | sort
grep -aE "^=== SIZE|^  A geometry" "$O/bench.log" | cut -c1-240
echo "--- timing tables (hot / cold / dep us per launch of the whole matrix; cold/A = A's cold time over the variant's):"
grep -aE "^  (family|A|A1|Acopy|B|C|Cnt|CL|D|Dnt|D1|D1nt|Dcopy) " "$O/bench.log" | grep -av "A geometry" | cut -c1-200
sed -n '/^=== SUMMARY/,$p' "$O/bench.log" | cut -c1-300
if [ -s "$O/clk.gpu.tsv" ]; then
  echo "--- clocks while the cards were busy (>= 50 %), mean over the samples: dev sclk_MHz mclk_MHz avg_sclk_MHz power_W hotspot_C n"
  awk -F'\t' '!/^#/ && $14 ~ /^[0-9.]+$/ && $14+0 >= 50 { n[$4]++; s[$4]+=$5; m[$4]+=$6; a[$4]+=$16; p[$4]+=$10; t[$4]+=$11 } END { for (d in n) printf "  %s %.0f %.0f %.0f %.0f %.0f %d\n", d, s[d]/n[d], m[d]/n[d], a[d]/n[d], p[d]/n[d], t[d]/n[d], n[d] }' "$O/clk.gpu.tsv"
fi
[ "$rc" -ne 0 ] && { echo "--- bench log tail:"; tail -n 8 "$O/bench.log" | cut -c1-200; }
say_end $rc
