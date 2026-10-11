#!/bin/bash
# thermal_load_chain.sh -- O7 (ACTION-LIST, record §L5-PF2 items 10 / 14): a sustained GPU load with a 2-second thermal sampler beside it, to judge the fan curve (`fan_od.sh curve`).
# Why: the cards reached 107-112 C junction under hour-long decode loads with the stock curve and zero-RPM on (soak max 97 / 112 / 112 C on cards 0 / 1 / 2, quality run 107-109 on card 1);
#  on 2026-10-11 the owner unlocked OverDrive and set a curve (40 % at 40 C -> 100 % at 85 C hotspot, zero-RPM off; all fans seen spinning). This measures the same kind of load.
# Load: ONE engine process (the served hybrid, the shipped flags, 262 144 cells), THERM_ROUNDS plan lines, each = 8 192 prose tokens of timed prefill + THERM_DECODE timed decode tokens
#  (decode keeps all three cards busy; ~2 min a line). Sampler (bash, sysfs, every 2 s): per card edge / junction / memory C, fan rpm, pwm, power W -> $O/samples.tsv.
# Safety: any junction >= THERM_KILL C (default 108; crit 110, emergency 115) -> the engine container is killed and the chain says so.
# Launch (rig idle, service OFF):  setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/thermal_load_chain.sh > ~/bench/thermal_load_chain.log 2>&1 < /dev/null &
# Env: THERM_GPU (CLI binary; build: make -C ~/src/franken-engine/franken/decode gpu GPU_BIN=franken_decode_glm_therm), THERM_OUT, THERM_ROUNDS (16), THERM_DECODE (2000), THERM_KILL (108).
set -u
. "$HOME/bench/chain_preflight.sh"
D=$HOME/src/franken-engine/franken/decode
GPU=${THERM_GPU:-$D/franken_decode_glm_therm}
O=${THERM_OUT:-$HOME/bench/thermal_$(date +%Y%m%d_%H%M)}; mkdir -p "$O"
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
P=$HOME/bench/m2/glm; PR=$HOME/bench/franken/glm5/prose8400.txt
ROUNDS=${THERM_ROUNDS:-16}; DEC=${THERM_DECODE:-2000}; KILL=${THERM_KILL:-108}
NAME=thermal_load_$$
say_end() { echo "=== thermal_load exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$GPU" "$M" "$PR"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== thermal_load start $(date -Is) gpu=$(sha256sum "$GPU" | cut -c1-16) rounds=$ROUNDS decode=$DEC kill_at=${KILL}C out=$O"
rig_quiet_wait 1800 || say_end 3

sample() {   # one tsv row a card: time card edge junction mem rpm pwm watts
  for c in 0 1 2; do
    h=$(ls -d /sys/class/drm/card$c/device/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    printf '%s\t%d\t%d\t%d\t%d\t%s\t%s\t%d\n' "$(date +%s)" "$c" $(( $(cat $h/temp1_input) / 1000 )) $(( $(cat $h/temp2_input) / 1000 )) $(( $(cat $h/temp3_input) / 1000 )) \
      "$(cat $h/fan1_input 2>/dev/null)" "$(cat $h/pwm1 2>/dev/null)" $(( $(cat $h/power1_average 2>/dev/null || echo 0) / 1000000 ))
  done
}
printf 'time\tcard\tedge\tjunction\tmem\trpm\tpwm\tW\n' > "$O/samples.tsv"
(
  while true; do
    rows=$(sample); echo "$rows" >> "$O/samples.tsv"
    hot=$(echo "$rows" | awk -v k="$KILL" '$4 >= k {print "card" $2 " junction " $4 "C"}' | head -1)
    if [ -n "$hot" ]; then echo "THERMAL KILL: $hot >= ${KILL}C at $(date +%T)" | tee -a "$O/kill.txt"; docker kill "$NAME" > /dev/null 2>&1; fi
    sleep 2
  done
) &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

: > "$O/plan.txt"
for r in $(seq 1 "$ROUNDS"); do
  echo "r$r --tokens-file $PR --ctx 262144 --chunk 1024 --adapt-prefill 0 --adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --fetch-assign optimal --gemv-fused-reduce 0 --glm-gemv-group 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 --time-prefill 8192 --time $DEC --time-settle 64" >> "$O/plan.txt"
done
echo "=== load $(date +%T): $ROUNDS x (8 192 prefill + $DEC decode)"
docker run --rm --name "$NAME" --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full \
  "$GPU" --model "$M" --placement "$P" --expert-gb 16.85 --gate-plan "$O/plan.txt" > "$O/engine.log" 2>&1
rc=$?; echo "engine rc=$rc $(date +%T)"
grep -aE "^prefill_tokens=|glm5_decode_ms_median" "$O/engine.log" | sed -E 's/^prefill_tokens=.*prefill_ms_per_token=([0-9.]+).*/prefill_ms_per_token=\1/' | paste -sd' ' | cut -c1-600
kill $SPID 2>/dev/null; sleep 3; sample >> "$O/samples.tsv"
echo "=== cool-down 120 s"; for i in $(seq 1 60); do sample >> "$O/samples.tsv"; sleep 2; done
# per card over the LOADED samples (power >= 150 W): n, edge median / max, junction median / p95 / max, mem max, rpm median, pwm median
awk -F'\t' 'NR > 1 && $8 >= 150 { n[$2]++; e[$2, n[$2]] = $3; j[$2, n[$2]] = $4; m[$2] = ($5 > m[$2] ? $5 : m[$2]); r[$2, n[$2]] = $6; p[$2, n[$2]] = $7 }
  function srt(a, c, k,   i, x, y, t) { for (i = 1; i <= k; i++) x[i] = a[c, i]; for (i = 2; i <= k; i++) { t = x[i]; y = i - 1; while (y > 0 && x[y] > t) { x[y + 1] = x[y]; y-- }; x[y + 1] = t }; for (i = 1; i <= k; i++) S[i] = x[i] }
  END { for (c = 0; c <= 2; c++) { k = n[c]; if (!k) { print "card" c ": no loaded samples"; continue }
          srt(e, c, k); em = S[int((k + 1) / 2)]; ex = S[k]; srt(j, c, k); jm = S[int((k + 1) / 2)]; jp = S[int(k * 0.95) > 0 ? int(k * 0.95) : 1]; jx = S[k]
          srt(r, c, k); rm = S[int((k + 1) / 2)]; srt(p, c, k); pm = S[int((k + 1) / 2)]
          printf "card%d loaded_samples=%d edge_med=%d edge_max=%d junction_med=%d junction_p95=%d junction_max=%d mem_max=%d rpm_med=%s pwm_med=%s\n", c, k, em, ex, jm, jp, jx, m[c], rm, pm } }' "$O/samples.tsv"
[ -s "$O/kill.txt" ] && echo "NOTE: the thermal kill fired: $(head -1 "$O/kill.txt")"
say_end "$rc"
