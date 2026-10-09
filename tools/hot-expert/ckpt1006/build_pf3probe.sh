#!/bin/bash
# build_pf3probe.sh -- PF3 route B, step 1 (ACTION-LIST row 5d, plan Rev 104 section 0d): a DEBUG-ONLY build of franken-engine `main` + `--pf3-probe REPS`, a per-config run option.
# Question: on ONE card, does the link-bound expert phase of one chunk run beside the trunk work of another? (route B = a second stream per bank).
# What it does, after the prompt of the config has been prefilled (so routing, the resident set and the KV state are realistic): for one KDA and one DSA MoE layer `il` of every card
# (with il + 1 a MoE layer of the same card) it runs
#   trunk  = layer_a(il)                         attention / KDA, norms, hyper-connection, router, plan   (the compute-bound side)
#   expert = moe_compute_own(layer il + 1)       the card's own share of a chunk's routed experts, resident from VRAM + missed in place over the link, with the plan layer_a(il + 1) left behind
#            (bank (il + 1) & 1: never the bank layer_a(il) writes, so the expert kernels read a frozen plan)
# on two probe streams, REPS reps each, under 14 conditions (each run twice, forward then reverse): alone, alone on a CU subset, both at once (default; trunk stream high priority; expert
# stream high priority; the CUs split in WGPs 8/40, 16/32, 24/24 between the expert and the trunk stream). The output is TIMING ONLY (the hidden state is re-fed to the same layer).
# Lines:  pf3 card= il= dsa= cond= run= trunk_ms= expert_ms=    (ms a rep, from a common start to the last kernel of that stream)
# The patch asserts every anchor once; a stale anchor stops the build. Output binary: franken_decode_glm_pf3probe (mask-free: `--pf3-probe 0` = the shipped path).
set -u
SRC=$HOME/src/franken-engine; WT=$HOME/src/franken-engine-pf3probe; OUT=$HOME/bench/franken_bin
NAME=${PF3_NAME:-franken_decode_glm_pf3probe}
git -C "$SRC" worktree add --detach "$WT" main || exit 1
cd "$WT/franken/decode" || exit 1
python3 - <<'PY'
def sub(p, pairs):
    s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, 'w').write(s)

sub('glm5_ops.h', [
 ("#include <cstddef>\n", "#include <cstddef>\n#include <functional>\n"),
 ("    virtual void set_help_copy(int on) { (void) on; }",
  "    virtual void set_help_copy(int on) { (void) on; }\n"
  "    // PF3 probe (DEBUG build, build_pf3probe.sh; GPU only): run `trunk` and `expert` (closures that launch on this card's main stream) `reps` times each on two probe streams\n"
  "    // (mode bit 0 = trunk, bit 1 = expert; *_hi = the greatest stream priority; xs_wgps > 0 = the expert stream gets that many WGPs and the trunk stream the rest),\n"
  "    // from a common start; ms[0] / ms[1] = when the trunk / expert stream's last rep finished. false when the stream could not be made.\n"
  "    virtual bool pf3_probe(const std::function<void()> & trunk, const std::function<void()> & expert, int reps, int mode,\n"
  "                           int ts_hi, int xs_hi, int xs_wgps, double * ms) {\n"
  "        (void) trunk; (void) expert; (void) reps; (void) mode; (void) ts_hi; (void) xs_hi; (void) xs_wgps; (void) ms; return false;\n"
  "    }"),
])
sub('decode_gpu.hip', [
 ("#include <hip/hip_fp16.h>\n", "#include <hip/hip_fp16.h>\n#include <hip/hip_ext.h>\n"),
])
sub('glm5_gpu.inc', [
 ("    void set_prefill_stage(int on, int ring_mb) override {",
  """    // ---- PF3 probe (DEBUG; build_pf3probe.sh) --------------------------------------------------------------------------------------------
    hipStream_t pr_ts_ = nullptr, pr_xs_ = nullptr;
    int pr_key_ = -1;
    hipEvent_t pr_e0_ = nullptr, pr_eT_ = nullptr, pr_eX_ = nullptr;
    bool pf3_probe(const std::function<void()> & trunk, const std::function<void()> & expert, int reps, int mode, int ts_hi, int xs_hi,
                   int xs_wgps, double * ms) override {
        dev_ensure(g_.dev_);
        HIP_CHECK(hipDeviceSynchronize());
        if (!pr_e0_) for (hipEvent_t * e : {&pr_e0_, &pr_eT_, &pr_eX_}) HIP_CHECK(hipEventCreate(e));
        const int key = ts_hi | (xs_hi << 1) | (xs_wgps << 2);
        if (key != pr_key_) {
            for (hipStream_t * s : {&pr_ts_, &pr_xs_}) if (*s) { HIP_CHECK(hipStreamDestroy(*s)); *s = nullptr; }
            int least = 0, greatest = 0;
            HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
            hipDeviceProp_t prop{};
            HIP_CHECK(hipGetDeviceProperties(&prop, g_.dev_));
            const int ncu = prop.multiProcessorCount, nw = ncu / 2;
            std::vector<int> xw, tw;
            if (xs_wgps > 0) {           // a spread of WGPs (a stride coprime with nw), the first xs_wgps to the expert stream
                auto gcd = [](int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; };
                int stride = (int) (0.618 * nw); while (gcd(stride, nw) != 1) ++stride;
                for (int j = 0; j < nw; ++j) (j < xs_wgps ? xw : tw).push_back((j * stride) % nw);
            }
            auto mk = [&](hipStream_t * s, int hi, const std::vector<int> & wgps) -> bool {
                if (wgps.empty()) return hipStreamCreateWithPriority(s, hipStreamNonBlocking, hi ? greatest : least) == hipSuccess;
                std::vector<uint32_t> m((size_t) (ncu + 31) / 32, 0u);       // a CU mask has no priority argument
                for (int w : wgps) for (int c = 2 * w; c < 2 * w + 2; ++c) m[(size_t) c / 32] |= 1u << (c % 32);
                return hipExtStreamCreateWithCUMask(s, (uint32_t) m.size(), m.data()) == hipSuccess;
            };
            if (!mk(&pr_ts_, ts_hi, tw) || !mk(&pr_xs_, xs_hi, xw)) { std::printf("pf3_probe dev=%d: stream create failed\\n", g_.dev_); pr_key_ = -1; return false; }
            pr_key_ = key;
            std::printf("pf3_probe dev=%d streams: priority least=%d greatest=%d, %d CUs, expert stream %d WGPs, trunk stream %d WGPs (0 = unmasked)\\n",
                        g_.dev_, least, greatest, ncu, (int) xw.size(), (int) tw.size());
        }
        struct Guard { hipStream_t & s; hipStream_t keep; ~Guard() { s = keep; } } guard{g_.stream_, g_.stream_};
        hipStream_t main = g_.stream_;
        HIP_CHECK(hipEventRecord(pr_e0_, main));
        HIP_CHECK(hipStreamWaitEvent(pr_ts_, pr_e0_, 0));
        HIP_CHECK(hipStreamWaitEvent(pr_xs_, pr_e0_, 0));
        for (int i = 0; i < reps; ++i) {
            if (mode & 1) { g_.stream_ = pr_ts_; trunk(); }
            if (mode & 2) { g_.stream_ = pr_xs_; expert(); }
        }
        g_.stream_ = main;
        HIP_CHECK(hipEventRecord(pr_eT_, pr_ts_));
        HIP_CHECK(hipEventRecord(pr_eX_, pr_xs_));
        HIP_CHECK(hipEventSynchronize(pr_eT_));
        HIP_CHECK(hipEventSynchronize(pr_eX_));
        float a = 0.f, b = 0.f;
        HIP_CHECK(hipEventElapsedTime(&a, pr_e0_, pr_eT_));
        HIP_CHECK(hipEventElapsedTime(&b, pr_e0_, pr_eX_));
        ms[0] = (mode & 1) ? a : 0.0;
        ms[1] = (mode & 2) ? b : 0.0;
        HIP_CHECK(hipDeviceSynchronize());
        return true;
    }
    void set_prefill_stage(int on, int ring_mb) override {"""),
])
sub('glm5_graph.h', [
 ("    int step(const int32_t * tokens, int T, Recorder & rec, bool flush = true);",
  "    int step(const int32_t * tokens, int T, Recorder & rec, bool flush = true);\n"
  "    // PF3 probe (DEBUG build, build_pf3probe.sh): trunk work against the expert phase on one card, two streams. TIMING ONLY.\n"
  "    void pf3_probe(Recorder & rec, int reps);"),
])
sub('glm5_graph.cpp', [
 ("    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;",
  "    int prefill_stage = 0, stage_mb = 0, adapt_prefill = 1;\n    int pf3_probe = 0;               // --pf3-probe REPS (DEBUG build, build_pf3probe.sh)"),
 ('    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }',
  '    else if (a == "--glm-stage-mb")  { if (!(v = val())) return -1; r.stage_mb = std::atoi(v); }\n'
  '    else if (a == "--pf3-probe")     { if (!(v = val())) return -1; r.pf3_probe = std::atoi(v); }'),
 ("    const std::vector<int> last_ids = run.routed_ids();",
  "    if (r.pf3_probe > 0) { for (int d = 0; d < n_report; ++d) devs[d]->sync(); run.pf3_probe(rec, r.pf3_probe); }\n"
  "    const std::vector<int> last_ids = run.routed_ids();"),
 ("// ------------------------------------------------------------------- CLI --\n",
  r'''// PF3 probe (DEBUG build, build_pf3probe.sh). Per card, a KDA and a DSA MoE layer `il` whose successor is a MoE layer of the same card: the trunk (layer_a(il)) and the card's own
// expert share of layer il + 1 (a frozen plan: layer_a(il + 1) ran once, and layer_a(il) writes the other plan bank) on two probe streams, 14 conditions, forward then reverse.
void Glm5Runner::pf3_probe(Recorder & rec, int reps) {
    const int T = T_;
    std::printf("pf3_probe T=%d pos=%d reps=%d (TIMING ONLY: the hidden state is re-fed to the same layer)\n", T, pos_, reps);
    struct Cond { const char * name; int mode, ts_hi, xs_hi, xw; };
    static const Cond conds[] = {
        {"T_alone", 1, 0, 0, 0},       {"X_alone", 2, 0, 0, 0},
        {"T_on_40w", 1, 0, 0, 8},      {"T_on_32w", 1, 0, 0, 16},     {"T_on_24w", 1, 0, 0, 24},
        {"X_on_8w", 2, 0, 0, 8},       {"X_on_16w", 2, 0, 0, 16},     {"X_on_24w", 2, 0, 0, 24},
        {"B_default", 3, 0, 0, 0},     {"B_trunk_hi", 3, 1, 0, 0},    {"B_expert_hi", 3, 0, 1, 0},
        {"B_part_8x40", 3, 0, 0, 8},   {"B_part_16x32", 3, 0, 0, 16}, {"B_part_24x24", 3, 0, 0, 24},
    };
    const int NC = (int) (sizeof(conds) / sizeof(conds[0]));
    for (int c = 0; c < model_.n_devices(); ++c) {
        std::vector<std::pair<int, bool>> pairs;
        bool got_kda = false, got_dsa = false;
        for (int il = il0_; il < il1_; ++il) {
            const LayerWeights & L = model_.layer(il);
            const LayerWeights & L1 = model_.layer(il + 1);
            if (L.dev != c || L1.dev != c || !L.moe || !L1.moe) continue;
            if (L.dsa && !got_dsa) { pairs.push_back({il, true}); got_dsa = true; }
            if (!L.dsa && !got_kda) { pairs.push_back({il, false}); got_kda = true; }
        }
        for (const auto & pr : pairs) {
            const int il = pr.first;
            const LayerWeights & L1 = model_.layer(il + 1);
            Glm5Ops & g = gop(il);
            Scratch & s = S(il);
            layer_a(il + 1, rec);                      // the expert layer's router + plan, once (bank (il + 1) & 1, x = s.xn)
            std::function<void()> trunk = [&]() { layer_a(il, rec); };
            std::function<void()> expert = [&]() { g.moe_compute_own(L1.et, L1.dev, (il + 1) & 1, s.yg, s.yu, s.yh, s.yd, SWIGLU_CLAMP, T); };
            double ms[2] = {0, 0};
            bool ok = true;
            for (int w = 0; w < 2 && ok; ++w) ok = g.pf3_probe(trunk, expert, 2, 3, 0, 0, 0, ms);   // warm-up: both kinds of kernels, grown scratch
            for (int pass = 0; pass < 2 && ok; ++pass)
                for (int k = 0; k < NC && ok; ++k) {
                    const Cond & cd = conds[pass == 0 ? k : NC - 1 - k];     // forward, then reverse
                    ok = g.pf3_probe(trunk, expert, reps, cd.mode, cd.ts_hi, cd.xs_hi, cd.xw, ms);
                    if (ok) std::printf("pf3 card=%d il=%d dsa=%d cond=%s run=%c trunk_ms=%.2f expert_ms=%.2f\n", c, il, pr.second ? 1 : 0,
                                        cd.name, pass == 0 ? 'a' : 'b', ms[0] / reps, ms[1] / reps);
                }
            if (!ok) std::printf("pf3 card=%d il=%d: probe stopped (stream create failed)\n", c, il);
        }
    }
    std::fflush(stdout);
}

// ------------------------------------------------------------------- CLI --
'''),
])
PY
[ $? -eq 0 ] || { echo "PATCH FAILED"; git -C "$SRC" worktree remove --force "$WT"; exit 1; }
nice -n 19 make -j4 gpu GPU_BIN=$NAME 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -12
[ -x $NAME ] && cp -p $NAME "$OUT/" && sha256sum "$OUT/$NAME" | cut -c1-16
git -C "$WT" rev-parse --short HEAD
git -C "$SRC" worktree remove --force "$WT"; echo done
