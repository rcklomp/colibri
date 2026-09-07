"""Time the expert slot read path (pread) so the unaccounted token time is attributed."""
p = "/home/ronald/src/colibri/c/glm53.c"
s = open(p).read()

def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)

rep("static void expert_read(GModel *m, int layer, int eid, Slot *slot) {",
    "static double prof_now_s(void);\n"
    "static double g_t_read = 0.0; static long g_n_read = 0;\n"
    "static void expert_read(GModel *m, int layer, int eid, Slot *slot) {\n"
    "    const double _tr0 = prof_now_s();",
    "read timer start")

rep("    slot->eid = eid;\n    /* expert_read gira dentro a un ciclo parallelo",
    "    slot->eid = eid;\n"
    "    {\n"
    "        const double _dt = prof_now_s() - _tr0;\n"
    "#ifdef _OPENMP\n#pragma omp atomic\n#endif\n"
    "        g_t_read += _dt;\n"
    "#ifdef _OPENMP\n#pragma omp atomic\n#endif\n"
    "        g_n_read++;\n"
    "    }\n"
    "    /* expert_read gira dentro a un ciclo parallelo",
    "read timer stop")

OLD_PRINT = '        printf("experts hits %ld miss %ld bytes %llu\\n",'
NEW_PRINT = ('        printf("[IO] expert_read wall=%.3fs n=%ld avg=%.2fms\\n",\n'
             '               g_t_read, g_n_read, g_n_read ? 1e3 * g_t_read / g_n_read : 0.0);\n'
             + OLD_PRINT)
rep(OLD_PRINT, NEW_PRINT, "io print")

open(p, "w").write(s)
print("IO PATCH APPLIED")
