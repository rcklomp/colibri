#!/bin/bash
# fan_od.sh -- unlock and set the fan control of the three RX 7900 XTX on rome (written 2026-10-10, ACTION-LIST O7, record §L5-PF2 item 14).
# Why: the cards run at 100-112 C junction under a long load (crit 110) and two of them report a fan tach of 0 rpm. The owner's reading (2026-10-10): the 0 rpm is the known amdgpu / RDNA3 sensor artifact, the heat is
# stacked cards + the stock fan curve, not dead fans. The sampler data agrees (record item 14): card 0 reads 0 rpm on every loaded sample and has the coolest edge, junction minus edge is 21-28 C on all three (no mounting signature).
# What is missing is a way to see and set the fans: gpu_od/ is EMPTY on every card because the kernel parameter amdgpu.ppfeaturemask has the OverDrive bit (0x4000) cleared (0xfff7bfff, the stock value). With the bit set the kernel
# (7.0, the RDNA3 fan interface exists since 6.7 / 6.13) lists gpu_od/fan_ctrl/{fan_curve, fan_zero_rpm_enable, fan_zero_rpm_stop_temperature, fan_target_temperature, acoustic_*}.
# NOT verified on the rig: the files do not exist until the module is loaded with the bit, so `curve` and `auto` are untested against the real driver (they are tested against a fake tree, see FAN_OD_ROOT). `status` reads back everything.
#
#   fan_od.sh status   read only, no root: per card the temperatures (edge / junction / memory), fan rpm and pwm, the gpu_od/fan_ctrl files, the ppfeaturemask in use and the one configured.
#   fan_od.sh enable   ROOT: writes /etc/modprobe.d/amdgpu-od.conf (`options amdgpu ppfeaturemask=<current | 0x4000>`) and runs update-initramfs. Needs a REBOOT to take effect (three busy cards: no module reload). Idempotent.
#   fan_od.sh curve    ROOT, after the reboot: per card a fan curve that is fan-friendly (about 40 % at 40 C, 100 % at 85 C of the curve's own axis, clamped to the driver's OD_RANGE) and the zero-RPM stop OFF
#                      (the fans never stop); reads the result back. Only ever raises fan speed against stock; noise goes up. Does not persist over a reboot (re-run it, or add it to a boot unit once it has proven itself).
#   fan_od.sh auto     ROOT: back to the stock curve and zero-RPM on ('r' + commit on every file).
#   fan_od.sh target   ROOT, THE ONE TO USE (probe 2026-10-12): no custom curve, zero-RPM off, acoustic thresholds 3600, fan target temperature FAN_TARGET_C (60). `curve` does NOT work
#                      on this firmware (any committed curve stops the fans at idle and capped card 1 at ~55 % under load).
#   fan_od.sh probe    ROOT, ~1 min a card, LOUD: per card a flat 100 % curve, then + acoustic thresholds 3600, then + target temperature 50, the fan command / rpm after each; ends with every
#                      fan_ctrl file reset to stock. Why: under load (2026-10-12) the curve was obeyed fully by card 2, scaled to ~55 % by card 1 and ignored by card 0.
# The script asks for sudo itself (you type the password; nothing is stored). Run:  ssh rome 'cd ~/src/colibri/tools/hot-expert/ckpt1006 && ./fan_od.sh status'   and for the root steps:  ssh -t rome '.../fan_od.sh enable'
# Env: FAN_OD_ROOT (test hook: a fake tree prefix for /sys, /etc and /proc; with it set nothing runs update-initramfs and no sudo is used), FAN_T0 / FAN_P0 (first point, default 40 C / 40 %), FAN_T1 / FAN_P1 (last point, default 85 C / 100 %).
set -u
R=${FAN_OD_ROOT:-}
MODE=${1:-status}
OD_BIT=$((0x4000))
CONF=$R/etc/modprobe.d/amdgpu-od.conf
PARAM=$R/sys/module/amdgpu/parameters/ppfeaturemask

wr() {   # wr FILE VALUE: one write, reported (the driver wants one line per write; a rejected write is an error we must see)
  if { printf '%s\n' "$2" 2>/dev/null > "$1"; } 2>/dev/null; then echo "  wrote '$2' -> ${1#$R}"; else echo "  FAILED to write '$2' -> ${1#$R}" >&2; return 1; fi
}
need_root() { [ -n "$R" ] || [ "$(id -u)" = 0 ] || { echo "(root needed: asking sudo)"; exec sudo "$0" "$@"; }; }
cards() { for d in "$R"/sys/class/drm/card[0-9]/device; do [ -e "$d/pp_dpm_sclk" ] && echo "$d"; done; }
hw() { ls -d "$1"/hwmon/hwmon* 2>/dev/null | head -1; }
mask_in_use() { cat "$PARAM" 2>/dev/null || echo "?"; }

status() {
  echo "kernel $(uname -r)  ppfeaturemask in use: $(mask_in_use)  (OverDrive bit 0x4000 $( [ "$(mask_in_use)" != "?" ] && [ $(( $(mask_in_use) & OD_BIT )) -ne 0 ] && echo SET || echo CLEARED ))"
  [ -f "$CONF" ] && echo "configured for the next boot: $(grep '^options' "$CONF")" || echo "configured for the next boot: nothing ($CONF absent)"
  for d in $(cards); do
    h=$(hw "$d"); n=$(basename "$(dirname "$d")")
    printf '%s  edge %s  junction %s  mem %s  C   fan %s rpm  pwm %s (enable %s)  power %s W\n' "$n" \
      "$(( $(cat "$h/temp1_input" 2>/dev/null || echo 0) / 1000 ))" "$(( $(cat "$h/temp2_input" 2>/dev/null || echo 0) / 1000 ))" "$(( $(cat "$h/temp3_input" 2>/dev/null || echo 0) / 1000 ))" \
      "$(cat "$h/fan1_input" 2>/dev/null)" "$(cat "$h/pwm1" 2>/dev/null)" "$(cat "$h/pwm1_enable" 2>/dev/null)" "$(( $(cat "$h/power1_average" 2>/dev/null || echo 0) / 1000000 ))"
    if [ -d "$d/gpu_od/fan_ctrl" ]; then
      for f in "$d"/gpu_od/fan_ctrl/*; do echo "    $(basename "$f"):"; sed 's/^/        /' "$f"; done
    else
      echo "    gpu_od/fan_ctrl: absent (OverDrive off: run 'enable' as root and reboot)"
    fi
  done
}

enable() {
  need_root "$@"
  cur=$(mask_in_use); [ "$cur" = "?" ] && { echo "cannot read $PARAM: is amdgpu loaded?" >&2; exit 1; }
  new=$(printf '0x%x' $(( cur | OD_BIT )))
  mkdir -p "$(dirname "$CONF")"
  printf '# written by fan_od.sh enable (%s): OverDrive bit on, for gpu_od/fan_ctrl (fan curve, zero-RPM)\noptions amdgpu ppfeaturemask=%s\n' "$(date +%F)" "$new" > "$CONF"
  echo "wrote ${CONF#$R}: ppfeaturemask $cur -> $new"
  if [ -z "$R" ]; then update-initramfs -u -k all 2>&1 | tail -3; fi
  echo "DONE. Reboot the rig (service is off, the reservation flag stays) and then run:  ./fan_od.sh status   (gpu_od/fan_ctrl must be listed), then  sudo ./fan_od.sh curve"
}

axis_range() {   # axis_range FILE "FAN_CURVE(hotspot temp)" -> "lo hi" (numbers only) from the OD_RANGE block of fan_curve
  awk -v key="$2" 'index($0, key) { for (i = 1; i <= NF; i++) { v = $i; gsub(/[^0-9]/, "", v); if (v != "") printf "%s ", v }; exit }' "$1"
}

curve() {
  need_root "$@"
  T0=${FAN_T0:-40}; P0=${FAN_P0:-40}; T1=${FAN_T1:-85}; P1=${FAN_P1:-100}
  n_done=0
  for d in $(cards); do
    fc=$d/gpu_od/fan_ctrl; n=$(basename "$(dirname "$d")")
    [ -f "$fc/fan_curve" ] || { echo "$n: no gpu_od/fan_ctrl/fan_curve (OverDrive off): run 'enable' and reboot"; continue; }
    echo "== $n"; cat "$fc/fan_curve" | sed 's/^/   before: /'
    pts=$(grep -c -E '^[0-9]+:' "$fc/fan_curve"); [ "$pts" -ge 2 ] || { echo "$n: cannot read the number of curve points" >&2; continue; }
    tkey=$(grep -o 'FAN_CURVE([^)]*temp)' "$fc/fan_curve" | head -1); pkey=$(grep -o 'FAN_CURVE(fan speed)' "$fc/fan_curve" | head -1)
    set -- $(axis_range "$fc/fan_curve" "$tkey"); tlo=${1:-25}; thi=${2:-100}
    set -- $(axis_range "$fc/fan_curve" "$pkey"); plo=${1:-15}; phi=${2:-100}
    off=0; case "$tkey" in *edge*) off=20;; esac   # an edge axis runs ~20 C below the junction: shift so the fans reach full speed before the junction limit
    echo "   $pts points, temperature axis '$tkey' $tlo..$thi C, speed axis $plo..$phi %$( [ "$off" -gt 0 ] && echo " (edge axis: points shifted -$off C)")"
    i=0; ok=1
    while [ "$i" -lt "$pts" ]; do
      t=$(( T0 + (T1 - T0) * i / (pts - 1) - off )); p=$(( P0 + (P1 - P0) * i / (pts - 1) ))
      [ "$t" -lt "$tlo" ] && t=$tlo; [ "$t" -gt "$thi" ] && t=$thi; [ "$p" -lt "$plo" ] && p=$plo; [ "$p" -gt "$phi" ] && p=$phi
      wr "$fc/fan_curve" "$i $t $p" || { ok=0; break; }
      i=$((i + 1))
    done
    if [ "$ok" = 0 ]; then echo "   $n: the driver rejected a point: curve NOT committed, nothing changed"; continue; fi
    wr "$fc/fan_curve" "c" || { echo "   $n: commit rejected: curve NOT set"; continue; }
    if [ -f "$fc/fan_zero_rpm_enable" ]; then wr "$fc/fan_zero_rpm_enable" "0"; wr "$fc/fan_zero_rpm_enable" "c"; else echo "   (no fan_zero_rpm_enable on this kernel)"; fi
    sed 's/^/   after:  /' "$fc/fan_curve"; [ -f "$fc/fan_zero_rpm_enable" ] && sed 's/^/   zero-rpm: /' "$fc/fan_zero_rpm_enable"
    n_done=$((n_done + 1))
  done
  echo "curve set on $n_done card(s). Check the effect under load:  ./fan_od.sh status   (fan rpm must now be non-zero above ~40 C, edge/junction lower than before)"
}

readback() {   # readback CARD_DEVICE_DIR LABEL: the fan command, tach and the zero-RPM setting the driver reports, after the firmware had time to move
  local h fc; h=$(hw "$1"); fc=$1/gpu_od/fan_ctrl; sleep "${FAN_SETTLE:-20}"
  printf '   %-34s pwm %3s/255  rpm %5s  junction %s C  zero-rpm %s\n' "$2" "$(cat "$h/pwm1" 2>/dev/null)" "$(cat "$h/fan1_input" 2>/dev/null)" \
    "$(( $(cat "$h/temp2_input" 2>/dev/null || echo 0) / 1000 ))" "$(sed -n 2p "$fc/fan_zero_rpm_enable" 2>/dev/null | tr -d ' ')"
}
zr_off() { [ -f "$1/fan_zero_rpm_enable" ] && { wr "$1/fan_zero_rpm_enable" "0" > /dev/null; wr "$1/fan_zero_rpm_enable" "c" > /dev/null; }; }   # zero-RPM off (re-applied after every commit: 2026-10-12 a curve commit turned it back on)

probe() {   # which knob lets each card reach full fan speed? (2026-10-12: under load card 1 capped at pwm ~140 / 2 200 rpm with the curve, card 0 flat at ~105)
  need_root "$@"
  for d in $(cards); do
    fc=$d/gpu_od/fan_ctrl; n=$(basename "$(dirname "$d")")
    [ -f "$fc/fan_curve" ] || { echo "$n: no fan_curve (OverDrive off)"; continue; }
    echo "== $n"
    readback "$d" "now"
    zr_off "$fc"
    readback "$d" "0 zero-RPM off only"
    i=0; for t in 25 35 45 55 65; do wr "$fc/fan_curve" "$i $t 100" > /dev/null || break; i=$((i + 1)); done; wr "$fc/fan_curve" "c" > /dev/null
    readback "$d" "A1 flat 100 % curve (as committed)"
    zr_off "$fc"
    readback "$d" "A2 flat 100 % + zero-RPM off again"
    for f in acoustic_limit_rpm_threshold acoustic_target_rpm_threshold; do [ -f "$fc/$f" ] && { wr "$fc/$f" "3600" > /dev/null; wr "$fc/$f" "c" > /dev/null; }; done
    zr_off "$fc"
    readback "$d" "B + acoustic limit/target 3600"
    [ -f "$fc/fan_target_temperature" ] && { wr "$fc/fan_target_temperature" "50" > /dev/null; wr "$fc/fan_target_temperature" "c" > /dev/null; }
    zr_off "$fc"
    readback "$d" "C + target temperature 50"
    for f in fan_curve acoustic_limit_rpm_threshold acoustic_target_rpm_threshold fan_target_temperature fan_zero_rpm_enable; do
      [ -f "$fc/$f" ] && { wr "$fc/$f" "r" > /dev/null; wr "$fc/$f" "c" > /dev/null; }
    done
    readback "$d" "stock again (reset)"
  done
  echo "probe done: every card is back on its STOCK fan settings."
}

target() {   # the setting that works on this firmware (probe 2026-10-12): NO custom curve (any committed curve stopped the fans, a flat 100 % one too), zero-RPM off,
             # the acoustic thresholds at their maximum, and a LOW target temperature (the firmware steers the fans toward it; stock 85 C / 75 C let the stacked cards run hot)
  need_root "$@"
  TT=${FAN_TARGET_C:-60}; AC=${FAN_ACOUSTIC:-3600}
  for d in $(cards); do
    fc=$d/gpu_od/fan_ctrl; n=$(basename "$(dirname "$d")")
    [ -f "$fc/fan_curve" ] || { echo "$n: no fan_ctrl (OverDrive off)"; continue; }
    echo "== $n"
    wr "$fc/fan_curve" "r" > /dev/null; wr "$fc/fan_curve" "c" > /dev/null
    for f in acoustic_limit_rpm_threshold acoustic_target_rpm_threshold; do [ -f "$fc/$f" ] && { wr "$fc/$f" "$AC"; wr "$fc/$f" "c" > /dev/null; }; done
    [ -f "$fc/fan_target_temperature" ] && { wr "$fc/fan_target_temperature" "$TT"; wr "$fc/fan_target_temperature" "c" > /dev/null; }
    zr_off "$fc"
    readback "$d" "target ${TT} C, acoustic ${AC}, zero-RPM off"
  done
  echo "target set. Not persistent over a reboot (re-run it). Back to stock: $0 auto"
}

auto() {
  need_root "$@"
  for d in $(cards); do
    fc=$d/gpu_od/fan_ctrl; n=$(basename "$(dirname "$d")"); [ -f "$fc/fan_curve" ] || { echo "$n: nothing to reset"; continue; }
    echo "== $n"; wr "$fc/fan_curve" "r"; wr "$fc/fan_curve" "c"
    [ -f "$fc/fan_zero_rpm_enable" ] && { wr "$fc/fan_zero_rpm_enable" "r"; wr "$fc/fan_zero_rpm_enable" "c"; }
  done
}

case "$MODE" in
  status) status;;
  enable) enable "$@";;
  curve)  curve "$@";;
  auto)   auto "$@";;
  probe)  probe "$@";;
  target) target "$@";;
  *) sed -n 2,3p "$0" | cut -c1-200; echo "usage: $0 status|enable|curve|auto|probe|target" >&2; exit 2;;
esac
