#!/usr/bin/env bash
# Prepare the Radxa ROCK 5B+ (RK3588, kernel 6.1.84) for timing until the next reboot (protocol P29, supplement of 10-25).
# Replaces the vendor's fix_freq_rk3588.sh in the board kit, which does nothing on this kernel: neither cpufreq nor devfreq
# offers the userspace governor it writes to, and its DDR rate of 2112 MHz is not an operating point of this board.
# Here the CPU clusters run the performance governor (1.8 / 2.352 / 2.304 GHz, the maxima of this chip), the NPU, GPU and
# DDR are held at their highest operating point through min_freq, and the deep CPU idle state is disabled as in the vendor
# script. The NPU runtime of the kit (librknnrt 2.3.2, the version the detector and vision models were exported with)
# is copied to /run and bind-mounted over the system library (2.3.0), so nothing on the disk changes.
#   sudo bash bench_setup_rock5bp.sh           apply, then print and check the state (exit 1 if a check fails)
#   sudo bash bench_setup_rock5bp.sh restore   back to the boot defaults (a reboot does the same)
#   bash bench_setup_rock5bp.sh show           print and check the state only; needs no sudo
set -u
KIT=${PB_KIT:-$(eval echo ~${SUDO_USER:-$USER})/pb_edge/board_kit}
RT_KIT=$KIT/vision/librknnrt.so
RT_MD5=a37ee1d5d664c79836bf6e35b7ef6289            # librknnrt 2.3.2 of the kit (same file as on the server, 10-24)
RT_RUN=/run/pb_bench/librknnrt.so.2.3.2
RT_SYS=$(readlink -f /usr/lib/librknnrt.so)          # /usr/lib/aarch64-linux-gnu/librknnrt.so.2.3.0 on this image
C=/sys/devices/system/cpu; F=/sys/class/devfreq
DEVS="fdab0000.npu fb000000.gpu dmc"
top() { tr ' ' '\n' < "$F/$1/available_frequencies" | grep . | sort -n | tail -1; }
low() { tr ' ' '\n' < "$F/$1/available_frequencies" | grep . | sort -n | head -1; }
put() { echo "$1" > "$2" 2>/dev/null || { echo "FAILED  echo $1 > $2"; BAD=1; }; }

apply() {
  for s in $C/cpu[0-9]*/cpuidle/state1/disable; do put 1 "$s"; done
  for p in $C/cpufreq/policy*; do put performance "$p/scaling_governor"; done
  for d in $DEVS; do t=$(top $d); put "$t" "$F/$d/max_freq"; put "$t" "$F/$d/min_freq"; done
  [ "$(md5sum < "$RT_KIT" | cut -d' ' -f1)" = "$RT_MD5" ] || { echo "FAILED  $RT_KIT is not the expected 2.3.2 file"; BAD=1; return; }
  findmnt -n "$RT_SYS" > /dev/null && umount "$RT_SYS"   # a second run would otherwise leave the mount on a deleted copy
  install -D -m 0755 -o root -g root "$RT_KIT" "$RT_RUN"
  mount --bind "$RT_RUN" "$RT_SYS" || { echo "FAILED  bind mount"; BAD=1; }
  # /run is mounted noexec and a bind mount inherits that flag, so the library could not be mapped; clear it on this mount
  mount -o remount,bind,exec "$RT_SYS" || { echo "FAILED  remount exec"; BAD=1; }
}

restore() {
  for s in $C/cpu[0-9]*/cpuidle/state1/disable; do put 0 "$s"; done
  for p in $C/cpufreq/policy*; do put ondemand "$p/scaling_governor"; done   # the boot default seen on 10-24
  for d in $DEVS; do put "$(low $d)" "$F/$d/min_freq"; done
  findmnt -n "$RT_SYS" > /dev/null && umount "$RT_SYS"
  rm -f "$RT_RUN"
}

show() {
  tr -d '\0' < /proc/device-tree/model; echo "  (kernel $(uname -r))"
  for p in $C/cpufreq/policy*; do
    g=$(cat $p/scaling_governor); cur=$(cat $p/scaling_cur_freq); max=$(cat $p/cpuinfo_max_freq)
    ok=ok; [ "$g" = performance ] && [ "$cur" = "$max" ] || { ok=NOT_FIXED; BAD=1; }
    printf "%-9s %-8s %-12s cur %8s kHz  max %8s kHz\n" "$ok" "${p##*/}" "$g" "$cur" "$max"
  done
  for d in $DEVS; do
    cur=$(cat $F/$d/cur_freq); mn=$(cat $F/$d/min_freq); t=$(top $d)
    ok=ok; [ "$cur" = "$t" ] && [ "$mn" = "$t" ] || { ok=NOT_FIXED; BAD=1; }
    printf "%-9s %-12s %-15s cur %10s Hz  min %10s  top %10s\n" "$ok" "$d" "$(cat $F/$d/governor)" "$cur" "$mn" "$t"
  done
  n=0; for s in $C/cpu[0-9]*/cpuidle/state1/disable; do [ "$(cat $s)" = 1 ] || n=$((n+1)); done
  [ $n = 0 ] && echo "ok        cpuidle state1 disabled on all CPUs" || { echo "NOT_FIXED cpuidle state1 enabled on $n CPUs"; BAD=1; }
  v=$(strings /usr/lib/librknnrt.so | grep -m1 -o 'librknnrt version: [0-9.]*')
  case "$v" in *2.3.2) echo "ok        $v";; *) echo "NOT_FIXED $v (want 2.3.2)"; BAD=1;; esac
  for z in /sys/class/thermal/thermal_zone*; do printf "%s %.1f C  " "$(cat $z/type)" "$(($(cat $z/temp)/100))e-1"; done; echo
  for c in /sys/class/thermal/cooling_device*; do printf "%s %s/%s  " "$(cat $c/type)" "$(cat $c/cur_state)" "$(cat $c/max_state)"; done; echo
}

BAD=0
case "${1:-apply}" in
  apply)   [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 2; }; apply; show ;;
  restore) [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 2; }; restore; show; exit 0 ;;
  show)    show ;;
  *) echo "usage: sudo bash bench_setup_rock5bp.sh [apply|restore|show]"; exit 2 ;;
esac
exit $BAD
