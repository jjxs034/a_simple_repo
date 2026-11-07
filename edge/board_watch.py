#!/usr/bin/env python3
"""Check and log the clocks and temperatures of the RK3588 board around a timing run (protocol P29, supplement of 10-25).

  python3 board_watch.py check                         is the board ready for timing? exit 1 if not
  python3 board_watch.py run --out DIR --tag T -- CMD  check, wait until the board is idle and cool, log once a second
                                                       while CMD runs, then write DIR/watch_T.csv and DIR/watch_T.json

A run is valid when every clock stays at its fixed value in at least 99% of the samples and no CPU or devfreq cooling
device engages (no thermal throttling). Crossing the first passive trip (60 C) is only reported, since the clocks are
fixed. Needs no sudo; the clocks are fixed beforehand with bench_setup_rock5bp.sh. Standard library only.
A devfreq domain (NPU, GPU, DDR) counts as fixed when its min_freq equals its max_freq: at the top operating point after
bench_setup_rock5bp.sh, or at a lower one set by board_p31.sh for the operating-point sweep of protocol P31.
"""
import argparse, csv, datetime, glob, json, os, re, subprocess, sys, threading, time

CPU, DF, TH = '/sys/devices/system/cpu/cpufreq', '/sys/class/devfreq', '/sys/class/thermal'
DEVS = ['fdab0000.npu', 'fb000000.gpu', 'dmc']
MIN_AT_TARGET, START_MAX_C, IDLE_MAX_BUSY = 0.99, 40.0, 0.05


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def targets():
    """The fixed clock of every domain and whether it is held there: CPU performance governor, devfreq min = max."""
    t, problems = {}, []
    for p in sorted(glob.glob(f'{CPU}/policy*')):
        name = os.path.basename(p)
        t[name] = int(rd(f'{p}/cpuinfo_max_freq'))
        if rd(f'{p}/scaling_governor') != 'performance':
            problems.append(f'{name}: governor {rd(f"{p}/scaling_governor")}, not performance')
    for d in DEVS:
        lo, hi = int(rd(f'{DF}/{d}/min_freq')), int(rd(f'{DF}/{d}/max_freq'))
        t[d] = lo
        if lo != hi:
            problems.append(f'{d}: min_freq {lo} and max_freq {hi} differ, so the clock is not fixed')
    return t, problems


def zones():
    return {rd(f'{z}/type'): f'{z}/temp' for z in sorted(glob.glob(f'{TH}/thermal_zone*'))}


def coolers():
    return {rd(f'{c}/type'): f'{c}/cur_state' for c in sorted(glob.glob(f'{TH}/cooling_device*'))}


def sample(zs, cs):
    s = {'t': round(time.time(), 2)}
    for p in sorted(glob.glob(f'{CPU}/policy*')):
        s[os.path.basename(p)] = int(rd(f'{p}/scaling_cur_freq') or 0)
    for d in DEVS:
        s[d] = int(rd(f'{DF}/{d}/cur_freq') or 0)
    for name, path in zs.items():
        s[name] = int(rd(path) or 0) / 1000
    for name, path in cs.items():
        s['cool:' + name] = int(rd(path) or 0)
    s['load1'] = float(rd('/proc/loadavg').split()[0])
    s['mem_avail_mb'] = mem_available()
    return s


def mem_available():
    for line in (rd('/proc/meminfo') or '').splitlines():
        if line.startswith('MemAvailable:'):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def busy(seconds=2.0):
    """Share of CPU time spent outside idle and iowait over a short window, from /proc/stat."""
    def snap():
        v = [int(x) for x in rd('/proc/stat').splitlines()[0].split()[1:]]
        return sum(v), v[3] + v[4]
    a = snap(); time.sleep(seconds); b = snap()
    return 1 - (b[1] - a[1]) / max(b[0] - a[0], 1)


def runtime_version():
    try:
        with open('/usr/lib/librknnrt.so', 'rb') as f:
            m = re.search(rb'librknnrt version: ([0-9.]+)', f.read())
        return m.group(1).decode() if m else None
    except OSError:
        return None


def check(verbose=True):
    t, problems = targets()
    s = sample(zones(), coolers())
    for k, v in t.items():
        if s[k] != v:
            problems.append(f'{k}: at {s[k]}, fixed value {v}')
    for k, v in s.items():
        if k.startswith('cool:') and not k.endswith('pwm-fan') and v:
            problems.append(f'{k[5:]}: cooling state {v} (throttling)')
    if runtime_version() != '2.3.2':
        problems.append(f'librknnrt {runtime_version()}, want 2.3.2')
    if verbose:
        print('clocks  ' + '  '.join(f'{k} {s[k]}' for k in t))
        print('temps   ' + '  '.join(f'{k} {s[k]:.1f}' for k in zones()))
        print('cooling ' + '  '.join(f'{k[5:]} {v}' for k, v in s.items() if k.startswith('cool:')))
        print('runtime librknnrt', runtime_version(), '  load1', s['load1'])
        print('READY' if not problems else 'NOT READY\n  ' + '\n  '.join(problems))
    return t, problems


def summarize(rows, t, zs, cs, extra):
    out = dict(extra, samples=len(rows), clocks={}, temps={}, cooling_max={}, valid=True, reasons=[], warnings=[])
    if not rows:
        out.update(valid=False, reasons=['no samples'])
        return out
    for k, v in t.items():
        vals = [r[k] for r in rows]
        share = sum(x == v for x in vals) / len(vals)
        out['clocks'][k] = {'fixed': v, 'min': min(vals), 'max': max(vals), 'at_fixed': round(share, 4)}
        if share < MIN_AT_TARGET:
            out['valid'] = False
            out['reasons'].append(f'{k} at its fixed value in only {share:.1%} of the samples')
    for k in zs:
        vals = [r[k] for r in rows]
        out['temps'][k] = {'start': vals[0], 'min': min(vals), 'max': max(vals), 'end': vals[-1]}
    for k in cs:
        m = max(r['cool:' + k] for r in rows)
        out['cooling_max'][k] = m
        if m and k != 'pwm-fan':
            out['valid'] = False
            out['reasons'].append(f'cooling device {k} engaged (state {m})')
    avail = [r['mem_avail_mb'] for r in rows if r.get('mem_avail_mb') is not None]
    if avail and extra.get('mem_avail_before_mb'):
        out['memory'] = {'avail_before_mb': extra['mem_avail_before_mb'], 'avail_min_mb': min(avail),
                         'peak_used_by_run_mb': round(extra['mem_avail_before_mb'] - min(avail), 1)}
    soc = out['temps'].get('soc-thermal', {}).get('max', 0)
    if soc >= 60:
        out['warnings'].append(f'soc-thermal reached {soc:.1f} C, above the first passive trip (60 C)')
    return out


def run(args):
    t, problems = check()
    if problems and not args.force:
        sys.exit('not ready; fix the clocks with: sudo bash bench_setup_rock5bp.sh')
    zs, cs = zones(), coolers()
    t0 = time.time()
    while True:                                   # wait until the board is idle and cool, so that runs start alike
        hot = max(sample(zs, cs)[k] for k in zs)
        b = busy()
        if hot <= args.start_max_c and b <= IDLE_MAX_BUSY:
            break
        if time.time() - t0 > args.wait:
            sys.exit(f'board not idle or cool after {args.wait} s (hottest zone {hot:.1f} C, CPU busy {b:.0%})')
        print(f'waiting: hottest zone {hot:.1f} C, CPU busy {b:.0%}', flush=True)
        time.sleep(10)
    rows, stop = [], threading.Event()

    def loop():
        while True:
            rows.append(sample(zs, cs))
            if stop.wait(args.interval):
                break
    th = threading.Thread(target=loop, daemon=True)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    avail_before = mem_available()   # the memory the command takes = this minus the lowest value seen while it runs
    th.start(); t1 = time.time(); rc = None
    try:
        rc = subprocess.run(args.cmd).returncode
    except KeyboardInterrupt:
        rc = 'interrupted'
    finally:
        stop.set(); th.join()
    extra = {'tag': args.tag, 'cmd': args.cmd, 'start_utc': start, 'seconds': round(time.time() - t1, 1),
             'returncode': rc, 'librknnrt': runtime_version(), 'kernel': os.uname().release, 'mem_avail_before_mb': avail_before,
             'model': (rd('/proc/device-tree/model') or '').rstrip('\x00')}
    summ = summarize(rows, t, zs, cs, extra)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f'watch_{args.tag}')
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['t'])
        w.writeheader(); w.writerows(rows)
    with open(base + '.json', 'w') as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    soc = summ['temps'].get('soc-thermal', {})
    print(f"WATCH {args.tag}: conditions {'VALID' if summ['valid'] else 'INVALID'}, command exit {rc}, "
          f"{summ['seconds']} s, {summ['samples']} samples, soc {soc.get('start', 0):.1f} -> max {soc.get('max', 0):.1f} C")
    for r in summ['reasons'] + summ['warnings']:
        print('  ' + r)
    sys.exit(0 if summ['valid'] and rc == 0 else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    sub.add_parser('check')
    r = sub.add_parser('run')
    r.add_argument('--out', required=True)
    r.add_argument('--tag', required=True)
    r.add_argument('--interval', type=float, default=1.0)
    r.add_argument('--start-max-c', type=float, default=START_MAX_C)
    r.add_argument('--wait', type=float, default=900, help='seconds to wait for an idle, cool board before giving up')
    r.add_argument('--force', action='store_true', help='run even if the clocks are not fixed (the summary says so)')
    r.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.mode == 'check':
        sys.exit(1 if check()[1] else 0)
    if a.cmd and a.cmd[0] == '--':
        a.cmd = a.cmd[1:]
    if not a.cmd:
        ap.error('give the command after --')
    run(a)


if __name__ == '__main__':
    main()
