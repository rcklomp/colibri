#!/usr/bin/env python3
# cpu_probe.py SECONDS OUT [PATTERN] -- once a second: busy % of every hardware thread (/proc/stat), and of every thread of the processes whose command line contains PATTERN
# (default "franken_dec_glm": the serving engine, seen from the host through its container), with last CPU and allowed-CPU list; plus the five busiest OTHER processes.
# Stdlib only, read-only. Output: one line a second, `t=<epoch> cpu=<16 busy %> eng=<tid:comm:busy:psr:affinity ...> other=<name:busy ...>`; summarise with cpu_probe_summary.py.
import os, sys, time
secs = int(sys.argv[1]); out = open(sys.argv[2], "w"); pat = sys.argv[3] if len(sys.argv) > 3 else "franken_dec_glm"
HZ = os.sysconf("SC_CLK_TCK"); ncpu = os.cpu_count()
def cpu_times():
    r = {}
    for l in open("/proc/stat"):
        if l.startswith("cpu") and l[3].isdigit():
            f = l.split(); v = list(map(int, f[1:9])); r[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))   # busy = total - idle - iowait
    return r
def procs():
    r = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open("/proc/%s/cmdline" % p, "rb") as f: cmd = f.read().replace(b"\0", b" ").decode("utf8", "replace")
                with open("/proc/%s/stat" % p) as f: st = f.read()
                comm = st[st.index("(") + 1:st.rindex(")")]; fs = st[st.rindex(")") + 2:].split()
                r[int(p)] = (comm, cmd, int(fs[11]) + int(fs[12]))
            except Exception: pass
    return r
def threads(pid):
    r = {}
    try:
        for t in os.listdir("/proc/%d/task" % pid):
            try:
                with open("/proc/%d/task/%s/stat" % (pid, t)) as f: st = f.read()
                comm = st[st.index("(") + 1:st.rindex(")")]; fs = st[st.rindex(")") + 2:].split()
                aff = ""
                with open("/proc/%d/task/%s/status" % (pid, t)) as f:
                    for l in f:
                        if l.startswith("Cpus_allowed_list"): aff = l.split()[1]; break
                r[int(t)] = (comm, int(fs[11]) + int(fs[12]), int(fs[36]), aff)
            except Exception: pass
    except Exception: pass
    return r
prev_c = cpu_times(); prev_p = procs(); eng = [p for p, v in prev_p.items() if pat in v[1] and "cpu_probe" not in v[1]]
prev_t = {p: threads(p) for p in eng}; t_prev = time.time()
for _ in range(secs):
    time.sleep(1.0 - ((time.time() - t_prev) % 1.0) if False else 1.0)
    now = time.time(); dt = now - t_prev; t_prev = now
    c = cpu_times(); p = procs()
    cpu = " ".join("%.0f" % (100.0 * (c[i][0] - prev_c[i][0]) / max(1, c[i][1] - prev_c[i][1])) for i in sorted(c))
    eng = [q for q, v in p.items() if pat in v[1] and "cpu_probe" not in v[1]]
    et = []
    for q in eng:
        th = threads(q)
        for tid, (comm, tm, psr, aff) in th.items():
            pv = prev_t.get(q, {}).get(tid)
            busy = 100.0 * (tm - (pv[1] if pv else tm)) / HZ / dt
            if busy >= 1.0: et.append((busy, "%d:%s:%.0f:%d:%s" % (tid, comm, busy, psr, aff)))
        prev_t[q] = th
    et.sort(reverse=True)
    others = []
    for q, v in p.items():
        if q in eng or q not in prev_p: continue
        d = (v[2] - prev_p[q][2]) * 100.0 / HZ / dt
        if d >= 3.0: others.append((d, "%s:%.0f" % (v[0], d)))
    others.sort(reverse=True)
    out.write("t=%.2f cpu=%s eng=%s other=%s\n" % (now, cpu, " ".join(x[1] for x in et[:24]), " ".join(x[1] for x in others[:5]))); out.flush()
    prev_c, prev_p = c, p
