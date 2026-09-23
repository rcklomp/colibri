// tools/hot-expert/franken/decode/franken_decode.cpp
//
// L0 step 2's CLI (L0-STEP2-BRIEF-2026-09-22.md, deliverable B; design rev 8
// section 9.5 step 2):
//
//   franken_decode --model <shard> --tokens <ids...> [--layers 0-15]
//                  [--oracle <dir>] [--cpu] [--time N] [--ctx N] [--threads N]
//
// Default backend is device 0. `--cpu` runs the SAME graph on the host, which
// is how the math is proved before a GPU minute is spent -- and the only
// thing the agent that wrote this could run at all.
//
// GPU DISCIPLINE: without `--cpu` this binary calls hipSetDevice and uploads
// ~22 GB. It must be run under the rig lock, with the gateway stopped and the
// cards empty (CLAUDE.md, "One benchmark at a time"). Nothing in the Makefile
// runs it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fstream>

#include "../ple.h"
#include "decode_graph.h"
#include "decode_model.h"
#include "decode_oracle.h"
#include "franken_serve.h"

using namespace fk;

namespace {

void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s --model <path-to-any-shard.gguf> --tokens <id> [<id> ...]\n"
        "          [--layers A-B]   layer span (default 0-47, the whole model)\n"
        "          [--devices N]    split the span over N cards (default 3)\n"
        "          [--no-head]      skip the final mixer and lm_head\n"
        "          [--greedy N]     append the argmax and decode N more tokens\n"
        "          [--expect-ids F] compare the greedy ids with F, one id a line\n"
        "          [--routing DIR]  compare the routed experts with DIR/moe_ids.txt\n"
        "          [--oracle DIR]   compare every tap against llama.cpp's dump\n"
        "          [--cpu]          run the same graph on the host (no GPU at all)\n"
        "          [--time N]       decode N further tokens and time the layer body\n"
        "          [--chunk C]      feed the prompt C tokens at a time (default 256,\n"
        "                           cap 512). C=1 is the decode path. A chunk produces\n"
        "                           exactly what C single-token steps produce, so\n"
        "                           `--chunk 6 --oracle A` against a `--chunk 1 --dump A`\n"
        "                           run of this binary must be cos 1.000000 on EVERY tap\n"
        "          [--time-prefill N] feed N synthetic ids (fixed seed) in chunks of C\n"
        "                           and print prefill_ms_per_token / prefill_tokens_per_s\n"
        "          [--prefill-pipeline 0|1] overlap the chunks across the cards\n"
        "                           (default 1): chunk n+1 starts on card 0 as soon as\n"
        "                           card 0 has handed chunk n to card 1. Numerics are\n"
        "                           UNCHANGED -- the residual gets a second bank and the\n"
        "                           host stops waiting, no kernel moves. Forced off for\n"
        "                           a chunk the recorder is capturing (a tap is a\n"
        "                           download) and for --jitter / --sync-debug.\n"
        "          [--gemm-lds M]   trunk GEMM kernel at T > 1: 0 the wave-per-row\n"
        "                           k_gemm_batch (DEFAULT, bit-identical to decode),\n"
        "                           1 the LDS-tiled GEMM (weight block decoded once per\n"
        "                           64 token columns, not per 8), 2 the same with the\n"
        "                           activation tile quantised to int8 and RDNA3's\n"
        "                           v_dot4_i32_iu8. 1 and 2 reassociate K, so they are\n"
        "                           a knob and their divergence is measured.\n"
        "          [--snap-every N] with --time-prefill: take a SERVING CHECKPOINT of\n"
        "                           the recurrent state every N tokens, exactly as\n"
        "                           franken_serve.cpp does -- stream-ordered into pinned\n"
        "                           host memory, never awaited. This is the measurement\n"
        "                           that says whether checkpoints cost the pipeline:\n"
        "                           run --time-prefill twice, with 0 and with 512, and\n"
        "                           compare prefill_ms_per_token.\n"
        "          [--ctx N]        cells the QSA caches are sized for (default 512)\n"
        "          [--threads N]    CPU backend threads (default 4)\n"
        "          [--min-cos X]    oracle bar (default 0.999)\n"
        "          [--quant-act]    CPU only: quantise activations to the weight's\n"
        "                           ggml vec_dot_type before the dot, as llama.cpp's\n"
        "                           CPU mul_mat does. A DIAGNOSTIC ARM, never the engine.\n"
        "          [--verbose]      print the routed experts each layer picks\n"
        "                           (device-side; read back AFTER the body)\n"
        "          [--profile]      per-kernel-class device time for one token, plus\n"
        "                           prof_host_syncs_per_token and prof_launches_per_token\n"
        "          [--gemv-lds 0|1] stage the GEMV activation in LDS (default 1);\n"
        "                           both arms exist so the profile can decide\n"
        "          [--gemv-min-rows N] split K below this many output rows (default\n"
        "                           1024); sweep it against prof_gemv_*_gbs\n"
        "          [--expert-gather M] design 9.4 item 5's device-side sort + row-gather\n"
        "                           per stage: bit 0 gate/up, bit 1 down. DEFAULT 1 --\n"
        "                           gate/up is MEASURED bit-exact, down is measured NOT\n"
        "                           to be and is off until it is (decode_backend.h).\n"
        "                           --no-expert-gather is the same as --expert-gather 0\n"
        "          [--sync-debug]   drain and check after every launch and copy;\n"
        "                           names the failing op, its class, layer and device\n"
        "          [--dump DIR]     write this run's taps in the oracle's own format,\n"
        "                           so one run can be the oracle of another\n"
        "          [--jitter X]     perturb every block input by +-X (oracle only):\n"
        "                           how far apart two equally correct summation\n"
        "                           orders of this architecture land\n"
        "          [--serve]        speak c/openai_server.py's line protocol on\n"
        "                           stdin/stdout instead of running a CLI turn\n"
        "                           (GATEWAY-PROTOCOL.md; SERVE=1 does the same).\n"
        "                           Everything it takes comes from the environment:\n"
        "                           SNAP/FRANKEN_GGUF, KV_SLOTS, FRANKEN_CTX,\n"
        "                           FRANKEN_CHUNK, FRANKEN_GEMM_LDS, FRANKEN_DEVICES.\n"
        "          [--serve-test]   --serve on the CPU backend with a short layer\n"
        "                           span and a small context, for serve_conformance.py\n", p);
}

bool parse_range(const std::string & s, int & a, int & b) {
    const size_t d = s.find('-');
    if (d == std::string::npos) return false;
    a = std::atoi(s.substr(0, d).c_str());
    b = std::atoi(s.substr(d + 1).c_str());
    return a >= 0 && b >= a;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char ** argv) {
    // SERVE MODE FIRST, before anything reads argv or writes a byte: under the
    // gateway this process is launched as `[binary, <cap>]` with SERVE=1 in the
    // environment (GATEWAY-PROTOCOL.md section 1), which the CLI parser below
    // would reject as an unrecognised argument -- and stdout is the wire, so
    // nothing may print on it. franken_serve.cpp owns the process from here.
    {
        bool serve = false, serve_test = false;
        for (int i = 1; i < argc; ++i) {
            if      (!std::strcmp(argv[i], "--serve"))      serve = true;
            else if (!std::strcmp(argv[i], "--serve-test")) serve = serve_test = true;
        }
        const char * s = std::getenv("SERVE");
        if (s && !std::strcmp(s, "1")) serve = true;
        if (serve) return fk::serve_main(argc, argv, serve_test);
    }

    // Line-buffered, so the progress lines survive an abort: a GPU memory
    // fault kills the process and a full stdout buffer goes with it, which
    // is how the first three-card run produced no output at all.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string model_path, oracle_dir, dump_dir;
    float jitter = 0.0f;
    std::vector<int32_t> tokens;
    int il0 = 0, il1 = N_LAYER - 1;
    int n_devices = 3, greedy_n = 0;
    bool with_head = true;
    std::string expect_ids_path, routing_dir;
    int time_n = 0, ctx = 512, threads = 4, snap_every = 0;
    int chunk = 256, time_prefill = 0;
    // franken_decode_cpu is built with -DFRANKEN_NO_HIP and has no GPU
    // backend to fall back to, so --cpu is its only mode and its default.
#ifdef FRANKEN_NO_HIP
    bool use_cpu = true;
#else
    bool use_cpu = false;
#endif
    double min_cos = 0.999;
    bool quant_act = false;
    bool verbose = false;
    bool profile = false;
    int  gemv_lds = 1;
    int  gemv_min_rows = 1024;
    bool sync_debug = false;
    int expert_gather = 3;
    int prefill_pipeline = 1;
    int gemm_lds = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle_dir = argv[++i];
        else if (a == "--cpu")                    use_cpu = true;
        else if (a == "--quant-act")              quant_act = true;
        else if (a == "--verbose")                verbose = true;
        else if (a == "--profile")                profile = true;
        else if (a == "--devices" && i + 1 < argc) n_devices = std::atoi(argv[++i]);
        else if (a == "--no-head")                with_head = false;
        else if (a == "--greedy"  && i + 1 < argc) greedy_n = std::atoi(argv[++i]);
        else if (a == "--expect-ids" && i + 1 < argc) expect_ids_path = argv[++i];
        else if (a == "--routing" && i + 1 < argc) routing_dir = argv[++i];
        else if (a == "--gemv-lds" && i + 1 < argc) gemv_lds = std::atoi(argv[++i]);
        else if (a == "--gemv-min-rows" && i + 1 < argc) gemv_min_rows = std::atoi(argv[++i]);
        else if (a == "--sync-debug")             sync_debug = true;
        else if (a == "--no-expert-gather")       expert_gather = 0;
        else if (a == "--expert-gather" && i + 1 < argc) expert_gather = std::atoi(argv[++i]);
        else if (a == "--dump"   && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--jitter" && i + 1 < argc) jitter = (float) std::atof(argv[++i]);
        else if (a == "--time"   && i + 1 < argc) time_n = std::atoi(argv[++i]);
        else if (a == "--chunk"  && i + 1 < argc) chunk = std::atoi(argv[++i]);
        else if (a == "--time-prefill" && i + 1 < argc) time_prefill = std::atoi(argv[++i]);
        else if (a == "--prefill-pipeline" && i + 1 < argc) prefill_pipeline = std::atoi(argv[++i]);
        else if (a == "--gemm-lds" && i + 1 < argc) gemm_lds = std::atoi(argv[++i]);
        else if (a == "--snap-every" && i + 1 < argc) snap_every = std::atoi(argv[++i]);
        else if (a == "--ctx"    && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--threads"&& i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--min-cos"&& i + 1 < argc) min_cos = std::atof(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) {
            if (!parse_range(argv[++i], il0, il1)) { usage(argv[0]); return 2; }
        } else if (a == "--tokens") {
            while (i + 1 < argc && argv[i + 1][0] != '-') tokens.push_back((int32_t) std::atoll(argv[++i]));
        } else { std::fprintf(stderr, "unrecognised argument: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || tokens.empty()) { usage(argv[0]); return 2; }
    // The cap is the MoE scratch: T x 10 x 3 840 floats a device is 78 MB at
    // 512 (PREFILL.md section 5), and everything else is under 20 MB.
    if (chunk < 1)   chunk = 1;
    if (chunk > 512) { std::fprintf(stderr, "--chunk capped at 512 (was %d)\n", chunk); chunk = 512; }

    // The synthetic prefill feed. A fixed seed so two runs on two binaries
    // see the same ids, and ids well inside the vocabulary so the embedding
    // gather is the real one.
    if (time_prefill > 0) {
        std::mt19937 rng(0x5EEDF00Du);
        tokens.clear();
        for (int i = 0; i < time_prefill; ++i) tokens.push_back((int32_t)(rng() % 100000u));
        greedy_n = 0;
    }
    if ((int) tokens.size() + time_n > ctx) {
        std::fprintf(stderr, "--ctx %d is too small for %zu + %d tokens\n", ctx, tokens.size(), time_n);
        return 2;
    }

    try {
        if (n_devices < 1) n_devices = 1;
        // One CpuBackend serves every range: it is all the same host memory,
        // so the "devices" only decide which layers a range holds.
        std::vector<std::unique_ptr<Backend>> owned;
        std::vector<Backend *> devs;
        if (use_cpu) {
            owned.emplace_back(make_cpu_backend(threads));
            for (int d = 0; d < n_devices; ++d) devs.push_back(owned[0].get());
        } else {
            enable_peer_access(n_devices);          // before anything is placed
            for (int d = 0; d < n_devices; ++d) {
                owned.emplace_back(make_gpu_backend(d));
                devs.push_back(owned.back().get());
            }
        }
        for (auto * b : devs) {
            if (quant_act) b->set_quant_act(true);
            if (profile)   b->set_profile(true);
            b->set_gemv_lds(gemv_lds);
            b->set_gemv_min_rows(gemv_min_rows);
            b->set_sync_debug(sync_debug);
            b->set_expert_gather(expert_gather);
            b->set_gemm_lds(gemm_lds);
            if (profile) b->prof_defer(prefill_pipeline != 0);
        }
        Backend * be = devs.front();
        std::printf("backend=%s devices=%d layers=%d-%d ctx=%d tokens=%zu chunk=%d head=%d quant_act=%d\n",
                    be->name(), n_devices, il0, il1, ctx, tokens.size(), chunk,
                    (int) with_head, (int) quant_act);
        // Two lines a gate greps for. `gemm_kernel` names the kernel a T > 1
        // trunk GEMM will take, so an oracle log says which arm produced it.
        std::printf("prefill_pipeline=%d gemm_lds=%d gemm_kernel=%s\n",
                    prefill_pipeline, gemm_lds,
                    gemm_lds == 2 ? "lds_i8_dot4" : (gemm_lds == 1 ? "lds_f32" : "batch_tile"));

        const int n_report = use_cpu ? 1 : n_devices;
        // BEFORE the weights, so "the card was empty" is a measurement
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "before placement");

        DecodeModel model(model_path, devs, il0, il1, with_head);
        std::printf("placed=%.2f GB rms_eps=%g\n", model.placed_bytes() / 1e9, model.rms_eps());
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "after placement");

        // Fail HERE, not a dozen layers into the next allocation with a
        // message about the wrong tensor: if a card holds far less than it was
        // asked to place, the allocations went somewhere else.
        bool placement_ok = true;
        for (int d = 0; d < n_report; ++d) placement_ok &= devs[d]->verify_placement(stdout);
        if (!placement_ok) { std::printf("STEP3 FAIL (placement)\n"); return 1; }

        // The PLE gather is host work by design (9.1) and depends on the whole
        // token prefix, so it is computed once for the sequence. ../ple.cpp's
        // hash was cross-checked against llama.cpp's own (record §L0-STEP1).
        std::vector<float> ple_all;
        const bool need_ple = (PLE_LAYER >= il0 && PLE_LAYER <= il1);
        if (need_ple) {
            const auto r = franken::ple_gather(model.gguf(), tokens);
            ple_all = r.emb;
            std::printf("ple_gather: %zu rows of %d, %zu floats\n",
                        tokens.size() * PLE_N_HEADS, PLE_HEAD_DIM, ple_all.size());
        }

        DecodeConfig cfg;
        cfg.ctx = ctx;
        cfg.max_tokens = chunk;
        cfg.verbose = verbose;
        cfg.jitter  = jitter;
        cfg.log_routing = !routing_dir.empty();
        cfg.sync_debug  = sync_debug;
        // weights first, then the caches and scratch, each on its owning card
        DecodeRunner run(model, cfg);
        run.report_cache_bytes(stdout);
        std::printf("caches+scratch allocated\n");
        for (int d = 0; d < n_report; ++d) devs[d]->vram_report(stdout, "after caches+scratch");

        std::printf("starting token 0\n");
        Recorder rec;
        RoutingOracle routing;
        if (!routing_dir.empty()) {
            std::string why;
            if (!routing.load(routing_dir, why)) std::printf("routing PENDING: %s\n", why.c_str());
        }

        std::vector<int> greedy_ids;
        int last_greedy = -1;
        int greedy_first_mismatch = -1;
        // The prompt in chunks of `chunk`. Every tap the recorder keeps is
        // the LAST ROW of the LAST chunk -- the column the oracle dump
        // writes -- whatever the chunking, which is what makes two runs at
        // different --chunk comparable tap for tap.
        // A timing run measures the ENGINE: a tap is a download and the
        // recorder would put 1 600 of them on the last chunk.
        if (time_prefill > 0) run.set_capture(false);
        // Serving checkpoints, as the serve loop takes them: pinned host
        // memory and Backend::download_async, so the copy is ordered inside
        // the stream and the host never waits. A ring of four, because a fifth
        // would only ever be needed if a copy took four chunks to land.
        std::vector<void *> snap_bufs;
        int snaps_taken = 0, snap_next = 0;
        long long snap_bytes = 0;
        if (snap_every > 0) {
            for (int i = 0; i < 4; ++i) {
                void * b = devs.front()->alloc_pinned(run.rec_bytes());
                if (!b) { std::printf("snapshot buffers are NOT pinned; copies will block\n"); b = std::malloc(run.rec_bytes()); }
                snap_bufs.push_back(b);
            }
            std::printf("snap_every=%d snap_bytes_each=%.1f MB\n",
                        snap_every, run.rec_bytes() / 1e6);
        }
        int last_snap_pos = 0;
        std::vector<double> chunk_ms, chunk_issue;
        long long pf_launches = 0;
        int n_chunks = 0;
        const auto t_pf0 = std::chrono::steady_clock::now();
        for (size_t t = 0; t < tokens.size(); t += (size_t) chunk) {
            const int T = (int) std::min((size_t) chunk, tokens.size() - t);
            const bool is_last = (t + (size_t) T == tokens.size());
            rec.enable(time_prefill == 0 && is_last);
            const float * ple = need_ple ? ple_all.data() + t * N_EMBD : nullptr;
            // Only the LAST chunk is awaited: every other one is enqueued and
            // left to run while the host builds the next. The cards stay
            // correct on events alone (decode_backend.h boundary_recv).
            const int g = run.step(tokens.data() + t, T, ple, rec,
                                   (prefill_pipeline == 0) || is_last);
            if (run.last_flushed()) {
                last_greedy = g;
                if (routing.loaded()) routing.observe(run.pos() - 1, il0, run.routed_ids());
            }
            ++n_chunks;
            if (snap_every > 0 && run.pos() - last_snap_pos >= snap_every) {
                run.save_rec_async(snap_bufs[(size_t)(snap_next++ % (int) snap_bufs.size())]);
                last_snap_pos = run.pos();
                ++snaps_taken;
                snap_bytes += (long long) run.rec_bytes();
            }
            if (run.last_body_ms() > 0.0) chunk_ms.push_back(run.last_body_ms());
            chunk_issue.push_back(run.last_issue_ms());
            // The FIRST chunk pays every one-time allocation this run makes
            // (the split-K partials and the expert sort tables grow with T),
            // so the profile is reset after it rather than before: otherwise
            // a hipMalloc and a hipFree land in the first chunk's numbers and
            // hipFree synchronises the device.
            if (time_prefill > 0 && n_chunks == 1 &&
                t + (size_t) T < tokens.size())            // only if more follow
                for (auto * b : devs) b->prof_reset();
            if (time_prefill == 0 || t == 0 || t + (size_t) T == tokens.size()) {
                std::printf("chunk at %zu T=%d id=%d pos=%d done\n", t, T, tokens[t], run.pos() - 1);
                std::fflush(stdout);
            }
        }
        if (time_prefill > 0) {
            run.set_capture(true);
            // The last chunk was awaited inside step(), and every card is
            // drained here as well, so the wall time below is the whole
            // prefill and not a queue depth. With --prefill-pipeline 1 that
            // is the ONLY thing that makes the number honest.
            for (int d = 0; d < n_report; ++d) devs[d]->sync();
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_pf0).count();
            std::printf("prefill_snapshots=%d snap_bytes_total=%.1f MB\n",
                        snaps_taken, snap_bytes / 1e6);
            std::printf("prefill_tokens=%d chunk=%d chunks=%d prefill_ms=%.1f "
                        "prefill_ms_per_token=%.4f prefill_tokens_per_s=%.2f\n",
                        time_prefill, chunk, n_chunks, ms, ms / (double) time_prefill,
                        1000.0 * (double) time_prefill / ms);
            // Where the wall clock goes at the chunk level, before the
            // per-class breakdown: body is the last device's GPU time for a
            // chunk, issue is the host's cost to enqueue one.
            for (int d = 0; d < n_report; ++d) pf_launches += devs[d]->launches_last_token();
            const double bms = median(chunk_ms), ims = median(chunk_issue);
            // body is a device-timer read and only an AWAITED chunk has one,
            // so with the pipeline on this is the last chunk alone (n=1) and
            // it is a span, not a busy time -- prof_busy_us is the busy time.
            std::printf("prefill_chunk_body_ms_median=%.3f (n=%zu) "
                        "prefill_chunk_issue_ms_median=%.3f "
                        "prefill_launches_last_chunk_all_devices=%lld "
                        "prefill_cpu_per_launch_us=%.2f\n",
                        bms, chunk_ms.size(), ims, pf_launches,
                        pf_launches > 0 ? ims * 1000.0 / (double) pf_launches : 0.0);
            // The hyper-connection residual is the ONLY thing that crosses a
            // card, and a chunk makes it T times bigger: 40 KB a token, so
            // 10.5 MB at T = 256 against M5's 16-24 GB/s and ~30 us of
            // latency. Printed because it is a term that did not exist at
            // T = 1 and prof_boundary_p2p_us should be held against it.
            const double bnd = (double) (n_devices - 1) * (double) chunk * HC_DIM * sizeof(float);
            std::printf("prefill_boundary_bytes_per_chunk=%.0f (%.2f MB, %d crossings of T x 40 KB)\n",
                        bnd, bnd / 1e6, n_devices - 1);
            if (profile) {
                // per token, so every number can be held against
                // prefill_ms_per_token directly
                const int skipped = (n_chunks > 1) ? std::min(time_prefill, chunk) : 0;
                const int counted = time_prefill - skipped;
                for (int d = 0; d < n_report; ++d) {
                    std::printf("--- device %d (per prompt token, %d tokens over chunks %d..%d) ---\n",
                                d, counted, n_chunks > 1 ? 2 : 1, n_chunks);
                    devs[d]->prof_report(stdout, counted);
                }
            }
        }

        // ---- greedy continuation -------------------------------------------
        // Each new token's PLE row depends on the whole prefix, so the gather
        // is redone for the appended id (host work, design 9.1).
        if (greedy_n > 0) {
            if (!model.have_head()) {
                std::fprintf(stderr, "--greedy needs the head; drop --no-head\n");
                return 2;
            }
            std::vector<int32_t> seq = tokens;
            for (int g = 0; g < greedy_n; ++g) {
                if (last_greedy < 0) break;
                seq.push_back(last_greedy);
                greedy_ids.push_back(last_greedy);
                std::vector<float> ple_row;
                const float * ple = nullptr;
                if (need_ple) {
                    const auto r = franken::ple_gather(model.gguf(), seq);
                    ple_row.assign(r.emb.end() - N_EMBD, r.emb.end());
                    ple = ple_row.data();
                }
                Recorder off;
                last_greedy = run.step(seq.back(), ple, off);
                if (routing.loaded()) routing.observe(run.pos() - 1, il0, run.routed_ids());
            }
            std::printf("greedy_ids:");
            for (int id : greedy_ids) std::printf(" %d", id);
            std::printf("\n");
        }

        // A cheap, always-printed sanity number: a zero or non-finite residual
        // means nothing below is worth reading.
        {
            const auto res = run.residual_host();
            double l1 = 0.0; bool finite = true;
            for (float v : res) { l1 += std::fabs((double) v); finite = finite && std::isfinite(v); }
            std::printf("residual l1=%.6g finite=%d n=%zu\n", l1, (int) finite, res.size());
            if (!finite || l1 == 0.0) {
                std::printf("STEP2 FAIL (residual is zero or non-finite)\n");
                return 1;
            }
        }

        if (!dump_dir.empty()) {
            std::string why;
            if (!dump_taps(rec, dump_dir, why)) { std::fprintf(stderr, "dump: %s\n", why.c_str()); return 1; }
            std::printf("dumped %zu taps to %s\n", rec.all().size(), dump_dir.c_str());
        }

        int rc = 0;
        if (!expect_ids_path.empty()) {
            std::vector<int> want;
            std::string why;
            if (!load_expected_ids(expect_ids_path, want, why)) {
                std::printf("expect-ids PENDING: %s\n", why.c_str());
                rc = 4;
            } else {
                const int first = first_mismatch(greedy_ids, want);
                greedy_first_mismatch = first;
                const size_t n = std::min(greedy_ids.size(), want.size());
                if (first < 0) std::printf("GREEDY MATCH %zu of %zu ids\n", n, want.size());
                else std::printf("GREEDY MISMATCH at position %d: mine=%d ref=%d (of %zu compared)\n",
                                 first, greedy_ids[first], want[first], n);
                if (first >= 0) rc = 1;
            }
        }
        if (routing.loaded()) {
            // Positions at and after the first greedy mismatch are fed a
            // DIFFERENT token than the reference was, so only the positions
            // below it compare like with like.
            const int valid_end = greedy_first_mismatch < 0
                                ? (int) tokens.size() + (int) greedy_ids.size()
                                : (int) tokens.size() + greedy_first_mismatch;
            routing.report(stdout, il0, il1, valid_end);
        }
        if (!oracle_dir.empty()) {
            Oracle oracle;
            std::string why;
            if (!oracle.load(oracle_dir, why)) {
                std::printf("oracle PENDING: %s\n", why.c_str());
                std::printf("STEP2 PENDING (no oracle to compare against)\n");
                rc = 4;
            } else {
                std::vector<int> pass_layers{ il1 };
                const bool ok = oracle.compare(rec, pass_layers, min_cos, stdout);
                std::printf("STEP2 %s\n", ok ? "PASS" : "FAIL");
                rc = ok ? 0 : 1;
            }
        } else {
            std::printf("STEP2 NO-ORACLE (pass --oracle <dir> for the verdict)\n");
            rc = 4;
        }

        if (time_n > 0) {
            Recorder off;                       // no taps: a tap is a download
            std::vector<double> ms;
            const int32_t filler = tokens.back();
            // The PLE row of a filler token is not its own (the n-gram hash
            // depends on the prefix), which is fine for timing: the work is
            // identical either way. Nothing else about the step differs.
            const float * ple = need_ple ? ple_all.data() + (tokens.size() - 1) * N_EMBD : nullptr;
            // A timing loop measures the ENGINE: no taps, and no --verbose or
            // --routing capture either (the log costs nothing now, but its
            // read-back is a sync a token).
            run.set_capture(false);
            run.step(filler, ple, off);         // one warm-up, not counted
            for (auto * b : devs) b->prof_reset();
            std::vector<double> issue;
            for (int i = 0; i < time_n; ++i) {
                run.step(filler, ple, off);
                ms.push_back(run.last_body_ms());
                issue.push_back(run.last_issue_ms());
            }
            run.set_capture(true);
            std::printf("layers%d_%d_ms_median=%.4f n=%d\n", il0, il1, median(ms), time_n);
            // The host cost of a token, and per launch over all devices. A
            // card whose work is pre-queued runs at its own speed; a card
            // being fed one op at a time runs at THIS rate.
            long long lt = 0;
            for (int d = 0; d < n_report; ++d) lt += devs[d]->launches_last_token();
            const double iss = median(issue);
            std::printf("issue_ms=%.4f prof_cpu_per_launch_us=%.2f launches_all_devices=%lld\n",
                        iss, lt > 0 ? iss * 1000.0 / (double) lt : 0.0, lt);
            if (profile) {
                for (int d = 0; d < n_report; ++d) {
                    std::printf("--- device %d ---\n", d);
                    devs[d]->prof_report(stdout, time_n);
                }
            }
        }
        return rc;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
