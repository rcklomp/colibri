#!/usr/bin/env python3
"""gpu_sampler.py -- passive per-card clock / power / PCIe-state sampler for a gate-plan run (read-only sysfs, python3 stdlib).

    gpu_sampler.py run OUT.tsv GATE_RUN.log [period_s=0.25]   sample until killed; the current config is read off GATE_RUN.log
    gpu_sampler.py summary OUT.tsv [GATE_RUN.log]             per config and card: means of every metric (10 % trimmed at each end)

Why: record §M7-HOSTSRC / the skip-bisect (2026-10-06) showed that compute running beside the staged DMA slows the copies themselves
(per-link rate while a copy is in flight 13/13/23 GB/s with nothing running, 9.6/9.7/16.5 with only the trunk GEMMs, 8/8/13 normal).
Candidates this can see: the SoC / fabric / memory clock dropping under load, a PCIe DPM downshift, the power cap. It cannot see
host-side effects. Reading amdgpu sysfs is passive (CLAUDE.md: amd-smi / nvtop do not disturb a gate); the period is 4 Hz to keep it so.
"""
import glob, os, re, sys, time


def cards():
    out = []
    for d in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
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
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
