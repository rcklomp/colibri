#!/usr/bin/env python3
"""gpu_sampler.py -- passive per-card clock / power / PCIe-state sampler for a gate-plan run (read-only sysfs, python3 stdlib).

    gpu_sampler.py run OUT.tsv GATE_RUN.log [period_s=0.25]   sample until killed; the current config is read off GATE_RUN.log
    gpu_sampler.py summary OUT.tsv [GATE_RUN.log]             per config and card: means of every metric (10 % trimmed at each end)
    gpu_sampler.py arm PREFIX ARM [period_s=5] [MARKFILE]     serve-path A/B mode (D6, 2026-10-08): every period write PREFIX.gpu.tsv (one row per card, labelled by PCI
                                                              address) and PREFIX.cpu.tsv (CPU effective MHz of the busy hardware threads), stamped with epoch seconds and
                                                              tagged with ARM and with the request name found in MARKFILE (serve_ab_driver.py writes it); stops on SIGTERM/SIGINT/SIGHUP
                                                              and when its parent process is gone (the chain died without running its trap)
    gpu_sampler.py check                                      one sample of the arm mode to stdout, with its cost (read-only; run before a chain to see the paths work)

Why: record §M7-HOSTSRC / the skip-bisect (2026-10-06) showed that compute running beside the staged DMA slows the copies themselves
(per-link rate while a copy is in flight 13/13/23 GB/s with nothing running, 9.6/9.7/16.5 with only the trunk GEMMs, 8/8/13 normal).
Candidates this can see: the SoC / fabric / memory clock dropping under load, a PCIe DPM downshift, the power cap. It cannot see
host-side effects. Reading amdgpu sysfs is passive (CLAUDE.md: amd-smi / nvtop do not disturb a gate); the period is 4 Hz to keep it so.
"""
import glob, os, re, signal, struct, sys, time

ROOT = os.environ.get("GPU_SAMPLER_ROOT", "")      # test hook only: a fake /sys + /proc tree; empty = the real machine


def cards():
    out = []
    for d in sorted(glob.glob(ROOT + "/sys/class/drm/card[0-9]/device")):
        if not os.path.exists(d + "/pp_dpm_sclk"):
            continue
        pci = os.path.basename(os.path.realpath(d))
        hw = glob.glob(d + "/hwmon/hwmon*")
        out.append((os.path.basename(os.path.dirname(d)), pci, d, hw[0] if hw else None))
    return out


def rd(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def star(txt):                       # the line marked '*' of a pp_dpm_* table: (level index, MHz or the raw rate string)
    for ln in txt.splitlines():
        if ln.rstrip().endswith("*"):
            m = re.match(r"\s*(\w+):\s*(\S+)", ln)          # "S: 0Mhz *" is the sclk table's live line on this kernel
            if m:
                v = re.match(r"([\d.]+)", m.group(2))
                return (int(m.group(1)) if m.group(1).isdigit() else -1), (float(v.group(1)) if v else float("nan")), m.group(2)
    return -1, float("nan"), ""


FIELDS = ["sclk", "mclk", "fclk", "socclk", "pcie_lvl", "power_w", "busy", "memb", "temp_c"]


def sample(c):
    _, _, d, hw = c
    s = star(rd(d + "/pp_dpm_sclk"))[1], star(rd(d + "/pp_dpm_mclk"))[1], star(rd(d + "/pp_dpm_fclk"))[1], \
        star(rd(d + "/pp_dpm_socclk"))[1], float(star(rd(d + "/pp_dpm_pcie"))[0])
    pw = rd(hw + "/power1_average").strip() if hw else ""
    p = float(pw) / 1e6 if pw.isdigit() else float("nan")
    b = rd(d + "/gpu_busy_percent").strip(); mb = rd(d + "/mem_busy_percent").strip()
    t = rd(hw + "/temp2_input").strip() if hw else ""        # junction (hotspot)
    return s + (p, float(b) if b.isdigit() else float("nan"), float(mb) if mb.isdigit() else float("nan"),
                float(t) / 1000 if t.isdigit() else float("nan"))


def cfg_of(log):
    try:
        return rd(log).count("=== gate-plan config")
    except Exception:
        return -1


def run(out, log, period):
    cs = cards()
    with open(out, "w") as f:
        f.write("# cards: " + " ".join("%s=%s" % (c[0], c[1]) for c in cs) + "\n")
        f.write("t_s\tcfg\tcard\tpci\t" + "\t".join(FIELDS) + "\n")
        t0 = time.time(); n = 0
        while True:
            k = cfg_of(log) if n % 4 == 0 else k           # the log read is the dearest call: once a second
            now = time.time() - t0
            for c in cs:
                f.write("%.2f\t%d\t%s\t%s\t%s\n" % (now, k, c[0], c[1], "\t".join("%.1f" % v for v in sample(c))))
            f.flush(); n += 1
            time.sleep(period)


# ---- arm mode (serve-path A/B, D6) -------------------------------------------------------------------------------------------------------------------
# Sources, all passive and readable without root (checked on the rig 2026-10-08): the same amdgpu sysfs files as `run` (pp_dpm_sclk / pp_dpm_mclk show the live clock
# on their starred line while the GPU works), hwmon temp1 = edge, temp2 = junction (HOTSPOT), temp3 = mem, power1_average; gpu_metrics (binary v1.3, 120 bytes, the
# 7900 XTX's layout on kernel 7.0) for the firmware's average GFX clock and the throttle bits; per hardware thread /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq
# (acpi-cpufreq on this kernel: the APERF/MPERF effective rate, so it shows the boost, 3.9 GHz seen against scaling_max 3.7) and /proc/stat to tell the busy threads.
# amd-smi is NOT used: one call took 0.12-0.33 s and a python start (~0.14 s CPU) on a box whose 8 busy cores the engine owns.
GPU_COLS = ["sclk", "mclk", "fclk", "socclk", "pcie_lvl", "power_w", "temp_hot", "temp_edge", "temp_mem", "busy", "memb", "avg_sclk", "thr"]
CPU_COLS = ["n_busy", "mhz_busy", "mhz_busy_min", "mhz_busy_max", "mhz_all", "util_all", "tctl_c", "mhz_cpus", "util_cpus"]
BUSY_UTIL = 0.5                      # a hardware thread counts as busy when it ran >= 50 % of the interval since the previous sample
NAN = float("nan")


def num(txt, scale=1.0):
    t = txt.strip()
    return float(t) / scale if re.match(r"^-?\d+$", t) else NAN


def hw_temps(hw):                    # label -> degrees C, from hwmon temp*_label (edge / junction / mem)
    out = {}
    for lf in glob.glob(hw + "/temp[0-9]*_label"):
        out[rd(lf).strip()] = num(rd(lf.replace("_label", "_input")), 1000.0)
    return out


def metrics(d):                      # (average GFX clock MHz, independent throttle bits as hex) from gpu_metrics v1.3; (nan, "-") for any other layout
    try:
        with open(d + "/gpu_metrics", "rb") as f:
            b = f.read()
    except OSError:
        return NAN, "-"
    if len(b) != 120 or b[2] != 1 or b[3] != 3:
        return NAN, "-"
    avg = struct.unpack_from("<H", b, 40)[0]          # average_gfxclk_frequency
    thr = struct.unpack_from("<Q", b, 112)[0]         # indep_throttle_status
    return (float(avg) if avg != 0xFFFF else NAN), "%x" % thr


def gpu_row(c):
    _, _, d, hw = c
    t = hw_temps(hw) if hw else {}
    pw = num(rd(hw + "/power1_average"), 1e6) if hw else NAN
    avg, thr = metrics(d)
    v = [star(rd(d + "/pp_dpm_sclk"))[1], star(rd(d + "/pp_dpm_mclk"))[1], star(rd(d + "/pp_dpm_fclk"))[1], star(rd(d + "/pp_dpm_socclk"))[1],
         float(star(rd(d + "/pp_dpm_pcie"))[0]), pw, t.get("junction", NAN), t.get("edge", NAN), t.get("mem", NAN),
         num(rd(d + "/gpu_busy_percent")), num(rd(d + "/mem_busy_percent")), avg]
    return "\t".join("%.1f" % x for x in v) + "\t" + thr


def cpu_stat():                      # {cpu index: (busy jiffies, total jiffies)} from /proc/stat
    out = {}
    for ln in rd(ROOT + "/proc/stat").splitlines():
        m = re.match(r"cpu(\d+)\s+(.*)", ln)
        if m:
            f = [int(x) for x in m.group(2).split()[:8]]
            out[int(m.group(1))] = (sum(f) - f[3] - f[4], sum(f))      # user..steal minus idle and iowait
    return out


def cpu_mhz():                       # {cpu index: MHz}
    out = {}
    for p in glob.glob(ROOT + "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        i = int(re.search(r"cpu(\d+)/", p).group(1)); k = num(rd(p), 1000.0)
        if k == k:
            out[i] = k
    if not out:                      # no cpufreq: /proc/cpuinfo "cpu MHz" (the same number on most kernels)
        i = -1
        for ln in rd(ROOT + "/proc/cpuinfo").splitlines():
            if ln.startswith("processor"):
                i += 1
            elif ln.startswith("cpu MHz"):
                out[i] = float(ln.split(":")[1])
    return out


def tctl():
    for h in glob.glob(ROOT + "/sys/class/hwmon/hwmon*"):
        if rd(h + "/name").strip() == "k10temp":
            return num(rd(h + "/temp1_input"), 1000.0)
    return NAN


def cpu_row(prev, cur):
    mhz = cpu_mhz(); ids = sorted(set(cur) & set(mhz))
    util = {}
    for i in ids:
        if i in prev and cur[i][1] - prev[i][1] >= 10:             # at least 0.1 s of jiffies between the two reads
            util[i] = (cur[i][0] - prev[i][0]) / float(cur[i][1] - prev[i][1])
    busy = [mhz[i] for i in ids if util.get(i, 0.0) >= BUSY_UTIL]
    mean = lambda x: sum(x) / len(x) if x else NAN
    f = lambda x: "%.1f" % x
    nb = len(busy) if util else -1
    return "\t".join([str(nb), f(mean(busy)), f(min(busy) if busy else NAN), f(max(busy) if busy else NAN), f(mean([mhz[i] for i in ids])),
                      f(100 * mean(list(util.values()))), f(tctl()),
                      ",".join("%d" % mhz[i] for i in ids), ",".join(("%d" % round(100 * util[i])) if i in util else "-" for i in ids)])


def mark_of(path):
    s = rd(path).strip() if path else ""
    return s.split()[0] if s.split() else "-"


def arm_run(prefix, arm, period, markf):
    try:
        os.nice(19)
    except OSError:
        pass
    stop = []
    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sg, lambda *a: stop.append(1))
    ppid = os.getppid(); cs = cards()
    gf = open(prefix + ".gpu.tsv", "w", buffering=1); cf = open(prefix + ".cpu.tsv", "w", buffering=1)
    hdr = "# gpu_sampler arm=%s period_s=%g tz_offset_s=%d started=%s pid=%d\n" % (arm, period, time.localtime().tm_gmtoff, time.strftime("%Y-%m-%dT%H:%M:%S%z"), os.getpid())
    gf.write(hdr + "# cards (dev column = PCI address; amd-smi orders by it, nvtop/DRM do not): " + " ".join("%s=%s" % (c[0], c[1]) for c in cs) + "\n")
    gf.write("t_epoch\tarm\treq\tdev\t" + "\t".join(GPU_COLS) + "\n")
    cf.write(hdr + "# n_busy = hardware threads that ran >= %d %% of the interval since the previous row (-1: first row, no interval); mhz_* = scaling_cur_freq (effective MHz)\n" % (100 * BUSY_UTIL))
    cf.write("t_epoch\tarm\treq\t" + "\t".join(CPU_COLS) + "\n")
    prev = cpu_stat(); nxt = time.time()
    while not stop:
        t = time.time(); req = mark_of(markf); cur = cpu_stat()
        for c in cs:
            gf.write("%.2f\t%s\t%s\t%s\t%s\n" % (t, arm, req, c[1], gpu_row(c)))
        cf.write("%.2f\t%s\t%s\t%s\n" % (t, arm, req, cpu_row(prev, cur)))
        prev = cur; nxt += period
        while not stop and time.time() < nxt:
            time.sleep(min(0.5, max(0.01, nxt - time.time())))
            if os.getppid() != ppid:                                 # the chain is gone: do not outlive it
                stop.append(1)
        if nxt < time.time():
            nxt = time.time()
    gf.close(); cf.close()


def arm_check():
    cs = cards(); p = cpu_stat(); time.sleep(1.0)
    t0 = time.time(); g = ["%s\t%s" % (c[1], gpu_row(c)) for c in cs]; q = cpu_row(p, cpu_stat()); dt = time.time() - t0
    print("gpu\tdev\t" + "\t".join(GPU_COLS)); print("\n".join("gpu\t" + x for x in g))
    print("cpu\t" + "\t".join(CPU_COLS)); print("cpu\t" + q)
    print("# one sample of %d cards + %d threads cost %.1f ms" % (len(cs), len(cpu_stat()), dt * 1e3))


def summary(path, log):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")][1:]
    names = {}
    for l in open(path):
        if l.startswith("# cards:"):
            names = dict(x.split("=") for x in l[8:].split())
    # config id -> name, in the order the log prints them
    cfgs = re.findall(r"^=== gate-plan config\s+(\S+)", rd(log), re.M) if log else []
    by = {}
    for r in rows:
        by.setdefault((int(r[1]), r[2]), []).append([float(x) for x in r[4:]])
    print("%-10s %-6s %-12s " % ("config", "card", "pci") + " ".join("%8s" % x for x in FIELDS) + "   samples")
    for (k, card), v in sorted(by.items()):
        if k < 1:
            continue
        lo, hi = int(len(v) * 0.1), int(len(v) * 0.9)
        v = v[lo:hi] or v
        mean = [sum(x[i] for x in v if x[i] == x[i]) / max(1, sum(1 for x in v if x[i] == x[i])) for i in range(len(FIELDS))]
        nm = cfgs[k - 1] if k - 1 < len(cfgs) else "cfg%d" % k
        print("%-10s %-6s %-12s " % (nm, card, names.get(card, "")) + " ".join("%8.1f" % m for m in mean) + "   %d" % len(v))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.25)
    elif len(sys.argv) >= 4 and sys.argv[1] == "arm":
        arm_run(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 5.0, sys.argv[5] if len(sys.argv) > 5 else "")
    elif len(sys.argv) == 2 and sys.argv[1] == "check":
        arm_check()
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
