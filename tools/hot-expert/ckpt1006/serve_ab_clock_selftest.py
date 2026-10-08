#!/usr/bin/env python3
"""serve_ab_clock_selftest.py OUTDIR   -- synthetic self-test of serve_ab_clock_report.py and of gpu_sampler.py's arm mode (python3 stdlib, no GPU, no rig; run with `python3 -I`).

Writes fabricated chain output into OUTDIR (4 arms old/new/old/new of the serve A/B; arm 3_old is the planted ODD arm: tok/s -6 %, GPU 83:00.0 shader clock -5 % and
hotspot +9 C; everything else is noise), feeds it to the report, and checks the report's numbers against an independent computation. Also checks the fallbacks (an old driver
without t_end -> times from the engine log; an unsampled arm) and runs the REAL sampler against a fake sysfs tree (GPU_SAMPLER_ROOT) to prove its rows parse in the report, that it
stops on SIGTERM and that it exits by itself when its parent is killed. Exit 0 and a final PASS line, else an assertion.
"""
import importlib.util, io, json, math, os, random, signal, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)


def load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


R, S = load("serve_ab_clock_report"), load("gpu_sampler")
DEVS = ["0000:86:00.0", "0000:83:00.0", "0000:48:00.0"]
TAGS = [("W1_long_doc", 16.3, 18.4, 300), ("W2_hash", 14.0, 28.8, 400), ("W3_bash", 13.5, 29.6, 400), ("W4_btree", 14.2, 28.2, 400), ("W5_hash_again", 14.0, 28.6, 400)]
rng = random.Random(7)
T0 = 1790000000


def fabricate(root):
    os.makedirs(root, exist_ok=True)
    truth = {}                                              # (arm, tag) -> list of raw sample dicts inside the decode window
    for k, (arm, build) in enumerate([("1_old", "old"), ("2_new", "new"), ("3_old", "old"), ("4_new", "new")]):
        odd = arm == "3_old"; a0 = T0 + k * 1500
        rows, t = [], a0 + 700
        for ti, (tag, tps, dec, em) in enumerate(TAGS):
            x = tps * (1.025 if build == "new" else 1.0) * (0.94 if odd else 1.0) * (1 + rng.gauss(0, 0.004))
            pre = 154.0 if ti == 0 else 0.8
            d = round(em / x, 2); t_start = t; t_end = float(round(t_start + pre + d)); t = t_end + 0.5
            rows.append({"tag": tag, "http_s": round(pre + d + 0.3, 2), "chars": 1500, "adapt_windows": 0, "t_start": t_start, "t_end": t_end, "req": ti + 1,
                         "prompt": 30, "reused": 0, "emitted": em, "prefill_s": pre, "decode_s": d, "tok_s": round(x, 2)})
        json.dump(rows, open(os.path.join(root, arm + ".res.json"), "w"), indent=1)
        gl, cl, ts = [], [], a0
        while ts < t + 6:
            req = next((r["tag"] for r in rows if r["t_start"] <= ts <= r["t_end"]), "-")
            busy = req != "-"
            for dv in DEVS:
                hot = 80 + rng.gauss(0, 3) if busy else 40 + rng.gauss(0, 1); sclk = 2350 + rng.gauss(0, 25) if busy else 0.0
                if dv == "0000:83:00.0" and odd and busy:
                    sclk -= 118; hot += 9
                gl.append((ts, arm, req, dv, [sclk, 1249.0 if busy else 96.0, 2301.0, 1500.0, 2.0, 250 + rng.gauss(0, 15) if busy else 12.0, hot, hot - 12, hot - 6,
                                              100.0 if busy else 0.0, 3.0, sclk, "0"]))
            nb = 5 if busy else 0; mb = 3500 + rng.gauss(0, 120) if busy else float("nan")
            cl.append((ts, arm, req, [nb, mb, mb - 150 if busy else float("nan"), mb + 120 if busy else float("nan"), 2600.0, 40.0, 55.0, "2500,2500", "5,95"]))
            ts += 5 + rng.uniform(-0.3, 0.3)
        hdr = "# gpu_sampler arm=%s period_s=5 tz_offset_s=0 started=x pid=1\n" % arm
        with open(os.path.join(root, arm + ".gpu.tsv"), "w") as f:
            f.write(hdr + "# cards: card0=0000:86:00.0\nt_epoch\tarm\treq\tdev\t" + "\t".join(S.GPU_COLS) + "\n")
            for ts, a, q, dv, v in gl:
                f.write("%.2f\t%s\t%s\t%s\t%s\n" % (ts, a, q, dv, "\t".join(("%.1f" % x) if not isinstance(x, str) else x for x in v)))
        with open(os.path.join(root, arm + ".cpu.tsv"), "w") as f:
            f.write(hdr + "t_epoch\tarm\treq\t" + "\t".join(S.CPU_COLS) + "\n")
            for ts, a, q, v in cl:
                f.write("%.2f\t%s\t%s\t%s\n" % (ts, a, q, "\t".join(("%.1f" % x) if not isinstance(x, str) else x for x in v)))
        for r in rows:
            lo, hi = r["t_end"] - r["decode_s"], r["t_end"]
            truth[(arm, r["tag"])] = {"gpu": [(ts, dv, v) for ts, a, q, dv, v in gl if lo <= round(ts, 2) <= hi], "cpu": [(ts, v) for ts, a, q, v in cl if lo <= round(ts, 2) <= hi], "tok": r["tok_s"]}
    # arm 5_new: copy of 2_new written by an OLD driver (no t_start / t_end), times only in the engine log, whose clock is UTC+2 (the sampler header says so)
    rows = json.load(open(os.path.join(root, "2_new.res.json")))
    with open(os.path.join(root, "5_new.gw.log"), "w") as f:
        f.write("[start] engine log header line\n")
        for r in rows:
            f.write("%s [serve-glm5] req=%d slot=0 prompt=30 reused=0 from=0 chunks=1 emitted=%d limited=1 cancelled=0 prefill_s=%s decode_s=%s tok/s=%s ckpts=8\n" % (
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r["t_end"] + 7200)), r["req"], r["emitted"], r["prefill_s"], r["decode_s"], r["tok_s"]))
            r.pop("t_start"); r.pop("t_end")
    json.dump(rows, open(os.path.join(root, "5_new.res.json"), "w"))
    for ext in ("gpu", "cpu"):
        txt = open(os.path.join(root, "2_new.%s.tsv" % ext)).read().replace("arm=2_new", "arm=5_new").replace("tz_offset_s=0", "tz_offset_s=7200").replace("\t2_new\t", "\t5_new\t")
        open(os.path.join(root, "5_new.%s.tsv" % ext), "w").write(txt)
    # arm 6_old: the chain ran it without sampling (SAB_SAMPLE=0 / sampler failed): only the result file
    json.dump([{"tag": "W2_hash", "tok_s": 13.9, "decode_s": 28.8, "emitted": 400}], open(os.path.join(root, "6_old.res.json"), "w"))
    return truth


def pear(xs, ys):                                           # independent two-pass implementation
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def check_report(root, truth):
    arms, notes, gq, cq = R.load(root, False)
    by = {a["arm"]: a for a in arms}
    assert [a["arm"] for a in arms] == ["1_old", "2_new", "3_old", "4_new", "5_new", "6_old"], [a["arm"] for a in arms]
    assert any("6_old" in n and "unsampled" in n for n in notes), notes
    # 1. per-request mean / min / max against the independent window selection
    for (arm, tag), tr in truth.items():
        rec = next(r for r in by[arm]["reqs"] if r["tag"] == tag)
        assert rec["n"] == len({ts for ts, _, _ in tr["gpu"]}) and rec["n"] >= 3, (arm, tag, rec["n"])
        g = [v[0] for ts, dv, v in tr["gpu"] if dv == "0000:83:00.0"]
        m, lo, hi = rec["q"][("0000:83:00.0", "sclk")]
        assert abs(m - sum(g) / len(g)) < 0.06 and abs(lo - min(g)) < 0.06 and abs(hi - max(g)) < 0.06, (arm, tag, m, lo, hi)
        c = [v for ts, v in tr["cpu"]]
        cm, clo, chi = rec["q"][("cpu", "mhz_busy")]
        assert abs(cm - sum(v[1] for v in c) / len(c)) < 0.06 and abs(clo - min(v[2] for v in c)) < 0.06 and abs(chi - max(v[3] for v in c)) < 0.06, (arm, tag)
    # 2. the engine-log fallback reproduces the driver-time windows exactly (times were whole seconds)
    for r5, r2 in zip(by["5_new"]["reqs"], by["2_new"]["reqs"]):
        assert r5["src"].startswith("engine log") and r2["src"] == "driver" and r5["n"] == r2["n"] and r5["q"] == r2["q"], (r5["tag"], r5["n"], r2["n"])
    # 3. Pearson, raw and within (group = request tag x build), recomputed independently over the driver-timed arms 1-4
    qs = R.quantities([a for a in arms if a["arm"] in ("1_old", "2_new", "3_old", "4_new")], gq, cq)
    cor = R.correlations([a for a in arms if a["arm"] in ("1_old", "2_new", "3_old", "4_new")], qs)
    recs = [r for a in arms if a["arm"] in ("1_old", "2_new", "3_old", "4_new") for r in a["reqs"]]
    for c, (nm, key, stat) in zip(cor, qs):
        xs = [R.pick(r, key, stat) for r in recs]; ys = [r["tok_s"] for r in recs]
        if not isinstance(c["raw"], float):
            continue
        assert abs(c["raw"] - pear(xs, ys)) < 1e-9, (nm, stat)
        dx, dy = [], []
        for tag, build in {(r["tag"], r["build"]) for r in recs}:
            g = [r for r in recs if r["tag"] == tag and r["build"] == build]
            mx, my = sum(R.pick(r, key, stat) for r in g) / len(g), sum(r["tok_s"] for r in g) / len(g)
            dx += [R.pick(r, key, stat) - mx for r in g]; dy += [r["tok_s"] - my for r in g]
        assert abs(c["win"] - pear(dx, dy)) < 1e-9 and c["nw"] == 20, (nm, stat, c["win"], c["nw"])
    # 4. the planted cause is found: 83:00.0 shader clock or hotspot leads the within-group list, and the unrelated CPU clock does not
    top = sorted([c for c in cor if isinstance(c["win"], float)], key=lambda c: -abs(c["win"]))
    assert top[0]["name"].startswith("83:00.0"), [(c["name"], c["stat"], round(c["win"], 2)) for c in top[:4]]
    sc = next(c for c in cor if c["name"] == "83:00.0 sclk MHz" and c["stat"] == "mean")
    assert sc["win"] > 0.8 and sc["raw"] > 0.2, sc
    cpu = next(c for c in cor if c["name"] == "cpu busy MHz" and c["stat"] == "mean")
    assert abs(cpu["win"]) < 0.5, cpu
    # 5. the odd arm shows in vs-grp (-3 % against the mean of its pair, more where 6_old joins the median of W2), its sibling is not odd
    assert all(-7.0 < r["vs_grp"] < -1.5 for r in by["3_old"]["reqs"]) and all(abs(r["vs_grp"]) < 3.5 for r in by["1_old"]["reqs"]), [r["vs_grp"] for r in by["3_old"]["reqs"]]
    # 6. the printed report and the csv
    buf = io.StringIO(); csvp = os.path.join(root, "per_request.csv")
    assert R.main([root, "--csv", csvp], buf) == 0
    txt = buf.getvalue()
    for need in ("--- arm 3_old", "83:00.0 sclk", "cpu     busy-thread MHz", "Pearson r", "largest |r within|", "times from the engine log", "ran unsampled", "hotspot C at arm start"):
        assert need in txt, need
    assert len(open(csvp).read().splitlines()) == 1 + sum(len(a["reqs"]) for a in arms)
    assert R.main([root, "--all"], io.StringIO()) == 0
    open(os.path.join(root, "report.txt"), "w").write(txt)
    return txt


def check_sampler(root):
    """The real sampler on a fake /sys + /proc: rows parse in the report; SIGTERM stops it; killing its parent stops it."""
    fk = os.path.join(root, "fakeroot"); os.makedirs(fk, exist_ok=True)
    for i, pci in enumerate(["0000:86:00.0", "0000:83:00.0", "0000:48:00.0"]):
        dev = os.path.join(fk, "sys/devices/pci0000:80", pci); hw = os.path.join(dev, "hwmon/hwmon%d" % (i + 2)); os.makedirs(hw, exist_ok=True)
        os.makedirs(os.path.join(fk, "sys/class/drm/card%d" % i), exist_ok=True)
        if not os.path.islink(os.path.join(fk, "sys/class/drm/card%d/device" % i)):
            os.symlink(dev, os.path.join(fk, "sys/class/drm/card%d/device" % i))
        w = lambda p, t: open(os.path.join(dev, p), "w").write(t)
        w("pp_dpm_sclk", "S: 2300Mhz *\n1: 500Mhz \n2: 2371Mhz \n"); w("pp_dpm_mclk", "0: 96Mhz \n1: 456Mhz \n3: 1249Mhz *\n"); w("pp_dpm_fclk", "1: 1000Mhz \n7: 2301Mhz *\n")
        w("pp_dpm_socclk", "2: 1500Mhz *\n"); w("pp_dpm_pcie", "0: 2.5GT/s, x16 \n2: 16.0GT/s, x16 *\n"); w("gpu_busy_percent", "100\n"); w("mem_busy_percent", "7\n")
        for nm, v in (("temp1_input", 70000 + i * 1000), ("temp2_input", 85000 + i * 1000), ("temp3_input", 90000), ("power1_average", 250000000 + i * 1000000)):
            open(os.path.join(hw, nm), "w").write("%d\n" % v)
        for nm, v in (("temp1_label", "edge"), ("temp2_label", "junction"), ("temp3_label", "mem")):
            open(os.path.join(hw, nm), "w").write(v + "\n")
    for c in range(4):
        d = os.path.join(fk, "sys/devices/system/cpu/cpu%d/cpufreq" % c); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "scaling_cur_freq"), "w").write("%d\n" % (3400000 if c < 2 else 2500000))
    hw = os.path.join(fk, "sys/class/hwmon/hwmon1"); os.makedirs(hw, exist_ok=True); open(os.path.join(hw, "name"), "w").write("k10temp\n"); open(os.path.join(hw, "temp1_input"), "w").write("61500\n")
    os.makedirs(os.path.join(fk, "proc"), exist_ok=True)
    stop = threading.Event()

    def feed():                                             # /proc/stat that moves: cpu0, cpu1 busy, cpu2, cpu3 idle (USER_HZ 100)
        n = 0
        while not stop.is_set():
            n += 20
            lines = ["cpu  %d 0 0 %d 0 0 0 0" % (2 * n, 2 * n)] + ["cpu%d %d 0 0 %d 0 0 0 0" % (c, (n if c < 2 else 0) + 100, (0 if c < 2 else n) + 100) for c in range(4)]
            open(os.path.join(fk, "proc/stat.tmp"), "w").write("\n".join(lines) + "\n"); os.replace(os.path.join(fk, "proc/stat.tmp"), os.path.join(fk, "proc/stat")); time.sleep(0.1)
    th = threading.Thread(target=feed, daemon=True); th.start(); time.sleep(0.3)
    sd = os.path.join(root, "sampler"); os.makedirs(sd, exist_ok=True)
    mark = os.path.join(sd, "9_new.mark"); open(mark, "w").write("W2_hash\n")
    env = dict(os.environ, GPU_SAMPLER_ROOT=fk)
    p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "gpu_sampler.py"), "arm", os.path.join(sd, "9_new"), "9_new", "0.3", mark], env=env)
    time.sleep(2.2); t_mid = time.time(); p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=5) == 0, "sampler did not stop cleanly on SIGTERM"
    gm, gr = R.read_tsv(os.path.join(sd, "9_new.gpu.tsv")); cm, cr = R.read_tsv(os.path.join(sd, "9_new.cpu.tsv"))
    assert gm["arm"] == "9_new" and len(gr) >= 15 and {r["dev"] for r in gr} == {"0000:86:00.0", "0000:83:00.0", "0000:48:00.0"} and all(r["req"] == "W2_hash" for r in gr)
    r83 = [r for r in gr if r["dev"] == "0000:83:00.0"][3]
    assert (r83["sclk"], r83["mclk"], r83["temp_hot"], r83["temp_edge"], r83["temp_mem"], r83["power_w"], r83["pcie_lvl"], r83["busy"]) == ("2300.0", "1249.0", "86.0", "71.0", "90.0", "251.0", "2.0", "100.0"), r83
    assert cr[0]["n_busy"] == "-1" and any(r["n_busy"] == "2" and r["mhz_busy"] == "3400.0" and r["tctl_c"] == "61.5" for r in cr[2:]), cr[:4]
    # the report reads the sampler's own files: a result file whose decode window is the middle second
    t_hi = max(float(r["t_epoch"]) for r in gr)
    json.dump([{"tag": "W2_hash", "tok_s": 14.0, "decode_s": 1.5, "emitted": 20, "t_end": t_hi - 0.2}], open(os.path.join(sd, "9_new.res.json"), "w"))
    arms = R.load(sd, True)[0]; rec = arms[0]["reqs"][0]
    assert rec["n"] >= 3 and rec["ntag"] == rec["n"] and rec["q"][("0000:83:00.0", "sclk")][0] == 2300.0 and rec["q"][("cpu", "mhz_busy")][0] == 3400.0, rec
    # orphan check: a shell starts the sampler in the background and is then killed with -9 (no trap runs); the sampler must notice within ~1 s
    sh = subprocess.Popen(["sh", "-c", '%s -I %s arm %s o_arm 0.3 "" & echo $!; sleep 60' % (sys.executable, os.path.join(HERE, "gpu_sampler.py"), os.path.join(sd, "o"))], stdout=subprocess.PIPE, env=env)
    pid = int(sh.stdout.readline()); time.sleep(1.0); os.kill(pid, 0); sh.kill(); sh.wait()
    for _ in range(40):
        try:
            os.kill(pid, 0); time.sleep(0.1)
        except OSError:
            break
    else:
        os.kill(pid, signal.SIGKILL); raise AssertionError("the sampler outlived its parent")
    stop.set()
    return len(gr)


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = fabricate(OUT)
    txt = check_report(OUT, truth)
    nrows = check_sampler(OUT)
    blk = txt.split("--- arm 3_old")[1].split("--- arm 4_new")[0].splitlines()
    print("--- arm 3_old" + blk[0] + "   <- the planted odd arm; first request shown\n" + "\n".join(blk[1:9]) + "\n...")
    print("\n".join(l for l in txt.splitlines() if l.startswith("    largest") or "83:00.0 sclk MHz" in l or l.startswith("    cpu busy")))
    print("(full report: %s/report.txt)" % OUT)
    print("PASS: report numbers match the independent computation (6 arms, %d requests), planted cause found, fallbacks work, sampler wrote %d rows, stops on SIGTERM and on a dead parent" % (
        sum(len(a["reqs"]) for a in R.load(OUT, False)[0]), nrows))


main()
