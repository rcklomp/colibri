#!/bin/bash
# clock_test.sh -- ROOT, run ONCE:  sudo bash ~/bench/clock_test.sh start      (prompts for the password, returns at once; log ~/bench/clock_test.log)
# Does the in-flight rate loss of the staged GLM DMA follow the GPUs' power management (§M7-SKIPCLASS)?  Runs glm_skipclock_chain.sh (masks 31 / 29 / 30 / 0,
# A,B,B,A, 8 192 tokens, gpu_sampler.py clocks) under two settings and compares with the earlier default-profile run (skipclock/):
#   peak     power_dpm_force_performance_level=profile_peak            (clocks held at the peak standard levels: the cleanest "are clocks the cause" test)
#   compute  level=manual + pp_power_profile_mode=5 (COMPUTE)           (the documented compute profile; the form a service-start setting would take)
# Every exit path (end, error, signal) restores each card's saved level and profile and then sets `auto`. ~35 min, ~2 model loads. Needs the rig idle
# (run_chain.sh takes the lock and refuses otherwise). Only writes: power_dpm_force_performance_level, pp_power_profile_mode of /sys/class/drm/card*/device.
set -u
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HOMEDIR=/home/ronald; LOG=$HOMEDIR/bench/clock_test.log; SAVE=/root/clock_test.saved
if [ "${1:-}" = start ]; then
  setsid nohup bash "$0" run > "$LOG" 2>&1 < /dev/null & disown
  echo "started, pid $!; log $LOG ; restores the power settings itself at the end"; exit 0
fi
[ "${1:-}" = run ] || { echo "usage: sudo bash $0 start"; exit 1; }
cards() { for d in /sys/class/drm/card[0-9]/device; do [ -w $d/power_dpm_force_performance_level ] && echo $d; done; }
: > $SAVE
for d in $(cards); do
  prof=$(grep -E "^ *[0-9]+ .*\*" $d/pp_power_profile_mode | head -1 | awk '{print $1}')
  echo "$d $(cat $d/power_dpm_force_performance_level) ${prof:-0}" >> $SAVE
done
restore() {
  echo "=== restoring $(date -Is)"
  while read d lvl prof; do
    echo manual > $d/power_dpm_force_performance_level; echo "$prof" > $d/pp_power_profile_mode 2>/dev/null
    echo "$lvl" > $d/power_dpm_force_performance_level
    echo "  $(basename $(readlink -f $d)): level=$(cat $d/power_dpm_force_performance_level)"
  done < $SAVE
}
trap restore EXIT
echo "=== clock_test start $(date -Is); saved:"; cat $SAVE
run_arm() {   # $1 tag
  echo "=== arm $1 $(date -Is)"
  sudo -u ronald -H env HOME=$HOMEDIR bash -c "$HOMEDIR/src/colibri/tools/hot-expert/run_chain.sh $HOMEDIR/bench/glm_skipclock_chain.sh $1" > $HOMEDIR/bench/glm_skipclock_chain_$1.log 2>&1
  echo "arm $1 rc=$? $(date -Is)"
}
for d in $(cards); do echo profile_peak > $d/power_dpm_force_performance_level; done
echo "peak set: $(for d in $(cards); do cat $d/power_dpm_force_performance_level; done | tr '\n' ' ')"
run_arm peak
for d in $(cards); do echo manual > $d/power_dpm_force_performance_level; echo 5 > $d/pp_power_profile_mode; done
echo "compute set: $(for d in $(cards); do grep -E '^ *[0-9]+ .*\*' $d/pp_power_profile_mode | head -1 | tr -s ' '; done | tr '\n' ' ')"
run_arm compute
echo "=== clock_test end $(date -Is)"
