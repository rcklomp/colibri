// tools/hot-expert/franken/decode/ds4_serve.cpp -- see ds4_serve.h.
//
// THE CONTRACT THIS FILE IMPLEMENTS is GATEWAY-PROTOCOL.md, exactly as
// franken_serve.cpp implements it for Qwen3.8 -- section numbers below are
// that file's. Everything in this file that is not DS4-specific (the wire
// framing, the tokenizer, the sampler, the checkpoint bookkeeping's SHAPE)
// is a close port of franken_serve.cpp; read that file's own header comment
// for the five things a new engine most easily gets wrong. What differs:
//
//   * Construction needs a `std::vector<Ds4Ops *>` alongside the backends
//     (Ds4Model/Ds4Runner both take it) and a `Placement` (M2 histogram +
//     VRAM budget for resident experts) -- neither has a Qwen equivalent.
//   * No PLE: DeepSeek-V4-Flash's per-token gathers (token_embd, the three
//     hash layers' tid2eid) are computed INSIDE Ds4Runner::step() from the
//     model's own embed_row()/hash_ids(), not fed in by the caller the way
//     Qwen's PLE row has to be (decode_model.h's design 9.1 rule still
//     applies -- Ds4Model keeps both host-resident, ds4_model.h's own header
//     comment -- there is just no serve-side gather loop to write).
//   * The checkpoint split is the same SHAPE (recurrent vs positional) but
//     different STATE: DS4's "recurrent" half is the raw sliding-window ring
//     plus the open compressor/indexer accumulation rings (ds4_graph.h's own
//     comment on Ds4Runner::state_pieces/kv_plan has the detail); Qwen's is
//     the GDN conv window and state. Ds4Runner exposes the identical method
//     set DecodeRunner does (rec_bytes/save_rec/load_rec/kv_bytes/save_kv/
//     load_kv/reset_state/sync_devices/logits_host, plus the pre-existing
//     pos()), so franken_serve.cpp's Slot/Snapshot/plan_reuse/protect_cells/
//     take_snapshot/snap_points/evict machinery ports over unchanged in
//     shape -- only the type names change.
//   * Adaptive placement (design L2, DEEPSEEK4.md section 12): FRANKEN_ADAPT
//     defaults ON here (unlike the CLI's off-by-default), because a served
//     conversation's own routing is exactly the traffic the M2 histogram
//     placement cannot see in advance.

#include "ds4_serve.h"

#include <dlfcn.h>
#include <poll.h>
#include <unistd.h>

#ifndef FRANKEN_LLAMA_SO_DEFAULT
#define FRANKEN_LLAMA_SO_DEFAULT "libllama.so"
#endif
#ifndef FRANKEN_LLAMA_PRELOAD_DEFAULT
#define FRANKEN_LLAMA_PRELOAD_DEFAULT \
    "/home/ronald/venvs/rocm/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/librocblas.so.5:" \
    "/home/ronald/venvs/rocm/lib/python3.14/site-packages/_rocm_sdk_libraries/lib/libhipblas.so.3"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include "llama.h"

#include "ds4_adapt.h"
#include "ds4_graph.h"
#include "ds4_model.h"
#include "ds4_ops.h"
#include "ds4_shapes.h"

namespace fk {
namespace ds4 {
namespace {

// ---------------------------------------------------------------- the wire --
// Byte-identical to franken_serve.cpp's own (GATEWAY-PROTOCOL.md section 1);
// duplicated rather than shared so this file links against nothing in the
// Qwen path (see ds4_serve.h's header comment).

const char READY_SENTINEL[] = "\x01\x01READY\x01\x01\n";

FILE * g_wire = nullptr;

void wire_open() {
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0) throw std::runtime_error("cannot dup stdout");
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot redirect stdout");
    g_wire = fdopen(fd, "w");
    if (!g_wire) throw std::runtime_error("cannot fdopen the wire");
    setvbuf(g_wire, nullptr, _IONBF, 0);
    setvbuf(stdin,  nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IOLBF, 0);
}

void wire(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_wire, fmt, ap);
    va_end(ap);
    std::fflush(g_wire);
}

void wire_data(const std::string & id, const char * bytes, int n) {
    std::fprintf(g_wire, "DATA %s %d\n", id.c_str(), n);
    if (n > 0) std::fwrite(bytes, 1, (size_t) n, g_wire);
    std::fputc('\n', g_wire);
    std::fflush(g_wire);
}

double rss_gb() {
    FILE * f = std::fopen("/proc/self/status", "r");
    if (!f) return 0.0;
    char line[256];
    double gb = 0.0;
    while (std::fgets(line, sizeof(line), f)) {
        long kb = 0;
        if (std::sscanf(line, "VmRSS: %ld kB", &kb) == 1) { gb = (double) kb / 1e6; break; }
    }
    std::fclose(f);
    return gb;
}

void logf(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[serve-ds4] ");
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

int env_int(const char * name, int dflt) {
    const char * v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atoi(v);
}

double env_double(const char * name, double dflt) {
    const char * v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atof(v);
}

std::string env_str(const char * name, const std::string & dflt = std::string()) {
    const char * v = std::getenv(name);
    return (v && *v) ? std::string(v) : dflt;
}

std::vector<std::string> split_paths(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ':') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ------------------------------------------------------------ the frames --
// Identical to franken_serve.cpp's FrameIO/Req (GATEWAY-PROTOCOL.md section
// 2): the wire protocol does not depend on which model is behind it.

struct Req {
    std::string id;
    int   slot = 0;
    int   plen = 0;
    int   max_tokens = 0;
    float temp = 0.0f;
    float top_p = 1.0f;
    int   prefix_bytes = 0;
    std::string payload;
};

constexpr int MAX_QUEUED = 16;

class FrameIO {
public:
    bool next(std::string & verb, Req & q) {
        if (!queue_.empty()) {
            verb = queue_.front().first;
            q    = queue_.front().second;
            queue_.pop_front();
            return true;
        }
        std::string header;
        if (pushback_full_) { header = pushback_; pushback_full_ = false; }
        else if (!read_line(header))  return false;
        read_body(header, verb, q);
        return true;
    }

    bool drain(const std::string & inflight) {
        if (cancelled_) return true;
        while (!pushback_full_ && stdin_ready()) {
            std::string line;
            if (!read_line(line)) {
                cancelled_ = true;
                eof_ = true;
                break;
            }
            char cmd[32] = {0}, who[64] = {0};
            if (std::sscanf(line.c_str(), "%31s %63s", cmd, who) < 1) continue;
            if (!std::strcmp(cmd, "CANCEL")) {
                if (inflight == who) cancelled_ = true;
                else wire("ERROR %s NOT_FOUND\n", who);
                continue;
            }
            if (!std::strcmp(cmd, "STOP")) continue;
            if (!std::strcmp(cmd, "SUBMIT") && (int) queue_.size() < MAX_QUEUED) {
                std::string verb;
                Req q;
                read_body(line, verb, q);
                queue_.emplace_back(verb, q);
                continue;
            }
            pushback_ = line;
            pushback_full_ = true;
            break;
        }
        return cancelled_;
    }

    void arm(const std::string &) { cancelled_ = false; }
    bool cancelled() const { return cancelled_; }
    bool eof() const { return eof_; }

private:
    static bool stdin_ready() {
        struct pollfd p { STDIN_FILENO, POLLIN, 0 };
        return poll(&p, 1, 0) > 0 && (p.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
    }

    static bool read_line(std::string & out) {
        char buf[1024];
        if (!std::fgets(buf, sizeof(buf), stdin)) return false;
        out = buf;
        return true;
    }

    void read_body(const std::string & header, std::string & verb, Req & q) {
        q = Req();
        char v[32] = {0};
        if (std::sscanf(header.c_str(), "%31s", v) != 1) { verb = "NONE"; return; }
        verb = v;
        if (verb == "STOP" || verb == "CANCEL") {
            char who[64] = {0};
            if (std::sscanf(header.c_str(), "%*s %63s", who) == 1) q.id = who;
            return;
        }
        if (verb == "IMAGE") {
            char id[64] = {0};
            int bytes = 0, gh = 0, gw = 0;
            if (std::sscanf(header.c_str(), "IMAGE %63s %d %d %d", id, &bytes, &gh, &gw) != 4 ||
                bytes < 0 || bytes > (1 << 28)) { verb = "BAD_FRAME"; return; }
            q.id = id;
            std::vector<char> discard((size_t) bytes);
            if (bytes && std::fread(discard.data(), 1, (size_t) bytes, stdin) != (size_t) bytes) {
                verb = "BAD_FRAME";
                return;
            }
            (void) std::fgetc(stdin);
            return;
        }
        if (verb != "SUBMIT") return;
        char id[64] = {0};
        int xlen = 0, hint = 0;
        const int fields = std::sscanf(header.c_str(), "SUBMIT %63s %d %d %d %f %f %d %d",
                                       id, &q.slot, &q.plen, &q.max_tokens,
                                       &q.temp, &q.top_p, &xlen, &hint);
        if (fields < 6) { verb = "BAD_FRAME"; return; }
        q.id = id;
        if (fields == 8 && xlen == 0 && hint > 0) q.prefix_bytes = hint;
        if (q.plen < 0 || q.plen > (1 << 24)) { verb = "BAD_FRAME"; return; }
        q.payload.resize((size_t) q.plen);
        if (q.plen && std::fread(&q.payload[0], 1, (size_t) q.plen, stdin) != (size_t) q.plen) {
            verb = "BAD_FRAME";
            return;
        }
        if (xlen > 0) {
            std::vector<char> ext((size_t) xlen);
            if (std::fread(ext.data(), 1, (size_t) xlen, stdin) != (size_t) xlen) {
                verb = "BAD_FRAME";
                return;
            }
        }
        (void) std::fgetc(stdin);
    }

    std::deque<std::pair<std::string, Req>> queue_;
    std::string pushback_;
    bool pushback_full_ = false;
    bool cancelled_ = false;
    bool eof_ = false;
};

// --------------------------------------------------------------- tokenizer --
// Identical in mechanism to franken_serve.cpp's LlamaApi/Vocab (same reasons:
// dlopen, not link -- see that file's comment on why); duplicated rather than
// shared, see ds4_serve.h.

struct LlamaApi {
    llama_model_params  (*model_default_params)() = nullptr;
    llama_model *       (*model_load_from_file)(const char *, llama_model_params) = nullptr;
    void                (*model_free)(llama_model *) = nullptr;
    const llama_vocab * (*model_get_vocab)(const llama_model *) = nullptr;
    int32_t             (*vocab_n_tokens)(const llama_vocab *) = nullptr;
    int32_t             (*tokenize)(const llama_vocab *, const char *, int32_t,
                                    llama_token *, int32_t, bool, bool) = nullptr;
    int32_t             (*token_to_piece)(const llama_vocab *, llama_token, char *,
                                          int32_t, int32_t, bool) = nullptr;
    bool                (*vocab_is_eog)(const llama_vocab *, llama_token) = nullptr;
    void                (*log_set)(ggml_log_callback, void *) = nullptr;

    void open(const std::string & so) {
        void * h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            const char * first = dlerror();
            logf("dlopen %s: %s -- trying the preload", so.c_str(), first ? first : "?");
            for (const std::string & dep : split_paths(env_str("FRANKEN_LLAMA_PRELOAD",
                                                               FRANKEN_LLAMA_PRELOAD_DEFAULT))) {
                if (dep.empty() || dep == "none") continue;
                if (!dlopen(dep.c_str(), RTLD_LAZY | RTLD_GLOBAL)) {
                    const char * e = dlerror();
                    logf("preload %s: %s (continuing)", dep.c_str(), e ? e : "?");
                }
            }
            h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
        if (!h) {
            const char * err = dlerror();
            throw std::runtime_error("dlopen " + so + ": " + (err ? err : "?"));
        }
        auto sym = [&](const char * n) {
            void * p = dlsym(h, n);
            if (!p) throw std::runtime_error(std::string("libllama has no ") + n);
            return p;
        };
        *(void **) &model_default_params = sym("llama_model_default_params");
        *(void **) &model_load_from_file = sym("llama_model_load_from_file");
        *(void **) &model_free           = sym("llama_model_free");
        *(void **) &model_get_vocab      = sym("llama_model_get_vocab");
        *(void **) &vocab_n_tokens       = sym("llama_vocab_n_tokens");
        *(void **) &tokenize             = sym("llama_tokenize");
        *(void **) &token_to_piece       = sym("llama_token_to_piece");
        *(void **) &vocab_is_eog         = sym("llama_vocab_is_eog");
        *(void **) &log_set              = sym("llama_log_set");
    }
};

class Vocab {
public:
    explicit Vocab(const std::string & gguf) {
        api_.open(env_str("FRANKEN_LLAMA_SO", FRANKEN_LLAMA_SO_DEFAULT));
        api_.log_set([](ggml_log_level, const char * text, void *) {
            std::fputs(text, stderr);
        }, nullptr);
        llama_model_params mp = api_.model_default_params();
        mp.vocab_only = true;
        model_ = api_.model_load_from_file(gguf.c_str(), mp);
        if (!model_) throw std::runtime_error("cannot load the tokenizer from " + gguf);
        vocab_ = api_.model_get_vocab(model_);
        n_     = api_.vocab_n_tokens(vocab_);
    }
    ~Vocab() { if (model_) api_.model_free(model_); }

    std::vector<int32_t> encode(const std::string & text) const {
        std::vector<int32_t> out((size_t) text.size() + 8);
        int n = api_.tokenize(vocab_, text.data(), (int32_t) text.size(),
                              out.data(), (int32_t) out.size(), false, true);
        if (n < 0) {
            out.resize((size_t)(-n));
            n = api_.tokenize(vocab_, text.data(), (int32_t) text.size(),
                              out.data(), (int32_t) out.size(), false, true);
        }
        if (n < 0) throw std::runtime_error("tokenizer overflow");
        out.resize((size_t) n);
        return out;
    }

    std::string piece(int32_t tok) const {
        char buf[256];
        int n = api_.token_to_piece(vocab_, tok, buf, (int32_t) sizeof(buf), 0, true);
        if (n >= 0) return std::string(buf, (size_t) n);
        std::vector<char> big((size_t)(-n));
        n = api_.token_to_piece(vocab_, tok, big.data(), (int32_t) big.size(), 0, true);
        if (n < 0) return std::string();
        return std::string(big.data(), (size_t) n);
    }

    bool is_eog(int32_t tok) const {
        if (tok < 0 || tok >= n_) return true;
        if (api_.vocab_is_eog(vocab_, tok)) return true;
        return std::find(extra_stops_.begin(), extra_stops_.end(), tok) != extra_stops_.end();
    }

    void add_stop(int32_t tok) { extra_stops_.push_back(tok); }
    int  n_tokens() const { return n_; }

private:
    LlamaApi             api_;
    llama_model *        model_ = nullptr;
    const llama_vocab *  vocab_ = nullptr;
    int                  n_ = 0;
    std::vector<int32_t> extra_stops_;
};

// ---------------------------------------------------------------- sampling --
// Identical to franken_serve.cpp's sample_token.

int32_t sample_token(const std::vector<float> & logits, float temp, float top_p,
                     int n_cand, std::mt19937 & rng) {
    const int n = (int) logits.size();
    if (n <= 0) return -1;
    if (temp <= 0.0f) {
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const int k = std::min(n, std::max(1, n_cand));
    std::vector<int> idx((size_t) n);
    for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
    std::nth_element(idx.begin(), idx.begin() + k - 1, idx.end(),
                     [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    idx.resize((size_t) k);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });

    const float maxl = logits[(size_t) idx[0]];
    std::vector<double> p((size_t) k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        p[(size_t) i] = std::exp((double)(logits[(size_t) idx[(size_t) i]] - maxl) / (double) temp);
        sum += p[(size_t) i];
    }
    for (auto & v : p) v /= sum;

    double cum = 0.0;
    int keep = k;
    const double bar = (top_p > 0.0f && top_p <= 1.0f) ? (double) top_p : 1.0;
    for (int i = 0; i < k; ++i) {
        cum += p[(size_t) i];
        if (cum >= bar) { keep = i + 1; break; }
    }
    std::uniform_real_distribution<double> u(0.0, cum > 0.0 ? cum : 1.0);
    const double r = u(rng);
    double acc = 0.0;
    for (int i = 0; i < keep; ++i) {
        acc += p[(size_t) i];
        if (r <= acc) return (int32_t) idx[(size_t) i];
    }
    return (int32_t) idx[0];
}

// ------------------------------------------------------------ checkpoints --
// Same split as franken_serve.cpp's (that file's own comment explains the
// "why", ds4_graph.h's explains the DS4-specific "what": raw window + open
// compressor/indexer rings = recurrent, closed compressor/indexer blocks =
// positional). Ds4Runner exposes the identical method set DecodeRunner does,
// so this section is the same code with DecodeRunner -> Ds4Runner.

struct HostBuf {
    Backend * be = nullptr;
    void *    p = nullptr;
    size_t    bytes = 0;
    bool      pinned = false;
};

HostBuf host_alloc(Backend * be, size_t bytes) {
    HostBuf b;
    b.be = be;
    b.bytes = bytes;
    b.p = be->alloc_pinned(bytes);
    b.pinned = (b.p != nullptr);
    if (!b.p) b.p = std::malloc(bytes);
    if (!b.p) throw std::runtime_error("out of host memory for a checkpoint");
    return b;
}

void host_free(HostBuf & b) {
    if (!b.p) return;
    if (b.pinned) b.be->free_pinned(b.p);
    else          std::free(b.p);
    b.p = nullptr;
}

struct KvImage {
    int     len = 0;
    HostBuf buf;
    ~KvImage() { host_free(buf); }
};

struct Snapshot {
    std::vector<int32_t>     seq;
    void *                   rec = nullptr;
    std::shared_ptr<KvImage> kv;
    long long                stamp = 0;
    int    pos() const { return (int) seq.size(); }
};

struct Slot {
    std::unique_ptr<Ds4Runner> run;
    std::vector<int32_t>  seq;
    std::vector<Snapshot> snaps;
    long long             stamp = 0;
    std::vector<HostBuf>  rec_all;
    std::vector<void *>   rec_free;
    size_t rec_bytes = 0;
};

bool is_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// --------------------------------------------------------------- the engine --

std::string find_gguf(const std::string & snap) {
    const std::string explicit_path = env_str("FRANKEN_GGUF");
    if (!explicit_path.empty()) return explicit_path;
    if (snap.empty()) throw std::runtime_error("neither FRANKEN_GGUF nor SNAP is set");
    struct stat st {};
    if (stat(snap.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return snap;
    DIR * d = opendir(snap.c_str());
    if (!d) throw std::runtime_error("cannot open SNAP directory " + snap);
    std::string first, shard1;
    while (struct dirent * e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 6 || n.substr(n.size() - 5) != ".gguf") continue;
        if (first.empty() || n < first) first = n;
        if (n.find("00001-of-") != std::string::npos) shard1 = n;
    }
    closedir(d);
    const std::string pick = !shard1.empty() ? shard1 : first;
    if (pick.empty()) throw std::runtime_error("no .gguf under " + snap);
    return snap + "/" + pick;
}

bool parse_span(const std::string & s, int & a, int & b) {
    const size_t d = s.find('-');
    if (d == std::string::npos) return false;
    a = std::atoi(s.substr(0, d).c_str());
    b = std::atoi(s.substr(d + 1).c_str());
    return a >= 0 && b >= a;
}

class Server {
public:
    Server(bool test_mode) : test_(test_mode) {
        chunk_ = std::min(512, std::max(1, env_int("FRANKEN_CHUNK", test_ ? 8 : 256)));
        int ctx = env_int("FRANKEN_CTX", 0);
        if (ctx <= 0) ctx = env_int("CTX", 0);   // deepseek_v4's own context_env, family_registry.py
        if (ctx <= 0) ctx = test_ ? 512 : 262144;
        ctx_ = ctx;
        n_slots_ = std::max(1, std::min(16, env_int("KV_SLOTS", 1)));
        snap_every_ = std::max(1, env_int("FRANKEN_SNAP_EVERY", test_ ? chunk_ : 512));
        snap_turns_ = std::max(1, env_int("FRANKEN_SNAP_TURNS", 3));
        snap_keep_  = std::max(1, env_int("FRANKEN_SNAP_KEEP", 8));
        budget_bytes_ = (size_t) std::max(256, env_int("FRANKEN_SNAP_BUDGET_MB", 6144))
                      * (size_t) 1024 * 1024;
        n_cand_     = std::max(1, env_int("FRANKEN_TOPK_CAND", 4096));
        const int seed = env_int("FRANKEN_SEED", 0);
        rng_.seed(seed > 0 ? (uint32_t) seed
                           : (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());

        const std::string snap = env_str("SNAP");
        const std::string gguf = find_gguf(snap);

        // Placement (DEEPSEEK4.md section 5/12): the M2 histogram plus the
        // per-card VRAM budget for resident experts, and adaptive placement
        // on top of it. Defaults match the served instruction's own choice
        // (~/bench/m2/deepseek_mix, 20 GB, --adapt 1) -- all three are env
        // overrides, not knobs a request can touch.
        Placement pl;
        pl.hist_dir  = env_str("FRANKEN_PLACEMENT", "/home/ronald/bench/m2/deepseek_mix");
        pl.expert_gb = env_double("FRANKEN_EXPERT_GB", 20.0);
        AdaptConfig adapt;
        adapt.on            = env_int("FRANKEN_ADAPT", test_ ? 0 : 1);
        adapt.every         = env_int("FRANKEN_ADAPT_EVERY", 16);
        adapt.mb_per_token   = env_double("FRANKEN_ADAPT_MB_PER_TOKEN", 64.0);
        adapt.halflife       = env_double("FRANKEN_ADAPT_HALFLIFE", 2048.0);
        adapt.margin         = env_double("FRANKEN_ADAPT_MARGIN", 1.0);
        adapt.hyst           = env_double("FRANKEN_ADAPT_HYST", 0.25);
        adapt.verify         = env_int("FRANKEN_ADAPT_VERIFY", 0);
        pl.adapt = adapt.on != 0;
        adapt_cfg_ = adapt;

        logf("gguf=%s ctx=%d chunk=%d kv_slots=%d snap_every=%d snap_keep=%d "
             "placement=%s expert_gb=%.1f adapt=%d",
             gguf.c_str(), ctx_, chunk_, n_slots_, snap_every_, snap_keep_,
             pl.hist_dir.empty() ? "(none)" : pl.hist_dir.c_str(), pl.expert_gb, adapt.on);

        vocab_ = std::make_unique<Vocab>(gguf);
        logf("tokenizer: %d tokens (vocab_only, no weights of its own)", vocab_->n_tokens());
        for (const std::string & t : split_csv(env_str("FRANKEN_STOP_IDS")))
            vocab_->add_stop((int32_t) std::atoi(t.c_str()));

        int il0 = 0, il1 = N_LAYER - 1;
        const std::string span = env_str("FRANKEN_LAYERS", test_ ? "0-3" : "");
        if (!span.empty() && !parse_span(span, il0, il1))
            throw std::runtime_error("FRANKEN_LAYERS must be A-B");
        const bool cpu = test_ || env_int("FRANKEN_CPU", 0) == 1;
        int n_dev = std::max(1, env_int("FRANKEN_DEVICES", 3));
        if (cpu) {
            owned_.emplace_back(make_cpu_backend(std::max(1, env_int("FRANKEN_THREADS", 8))));
            for (int d = 0; d < n_dev; ++d) devs_.push_back(owned_[0].get());
        } else {
            enable_peer_access(n_dev);
            for (int d = 0; d < n_dev; ++d) {
                owned_.emplace_back(make_gpu_backend(d));
                devs_.push_back(owned_.back().get());
            }
        }
        const int gemm_lds = env_int("FRANKEN_GEMM_LDS", 1);
        for (auto * b : devs_) {
            if (gemm_lds) b->set_gemm_lds(gemm_lds);
            owned_ops_.emplace_back(cpu ? make_ds4_cpu_ops(*b) : make_ds4_gpu_ops(*b));
            ops_.push_back(owned_ops_.back().get());
        }
        // ds4_main's own order (ds4_graph.cpp): the Ds4Ops runtime knobs
        // (staged_loads/stage_wgs/expert_gather/attn_mb) are set AFTER the
        // model places its experts and BEFORE any Ds4Runner is built -- model
        // placement itself does not read them, but matching the CLI's own
        // order rather than an unverified reordering is the safer bet.
        const bool with_head = true;
        model_ = std::make_unique<Ds4Model>(gguf, devs_, ops_, il0, il1, with_head, pl);
        logf("placed=%.2f GB host_experts=%.2f GB layers=%d-%d devices=%d backend=%s",
             model_->placed_bytes() / 1e9, model_->host_expert_bytes() / 1e9,
             il0, il1, (int) devs_.size(), devs_.front()->name());
        for (int d = 0; d < (int) owned_.size(); ++d) owned_[d]->vram_report(stderr, "after weights");
        for (Ds4Ops * o : ops_) {
            o->set_profile(false);
            o->set_staged_loads(env_int("FRANKEN_STAGED_LOADS", 1));
            o->set_stage_wgs(env_int("FRANKEN_STAGE_WGS", 64));
            o->set_expert_gather(env_int("FRANKEN_DS4_EXPERT_GATHER", 1));
            o->set_attn_mb(env_int("FRANKEN_DS4_ATTN_MB", 256));
        }
        slots_.resize((size_t) n_slots_);

        // The template's turn boundaries, as TOKEN IDS (franken_serve.cpp's
        // snap_points comment explains why this, not the interval, is the
        // schedule that matters for accept_live.sh check 2). DeepSeek-V4's
        // render_chat_v4 (c/openai_server.py) opens EVERY user turn with
        // "<｜User｜>" and every assistant turn with
        // "<｜Assistant｜>" -- unlike Qwen's single <|im_start|>,
        // there is no one marker common to every role, so both are boundary
        // tokens by default.
        for (const std::string & tok :
                 split_csv(env_str("FRANKEN_BOUNDARY_TOKENS",
                                   "<｜User｜>,<｜Assistant｜>"))) {
            const std::vector<int32_t> ids = vocab_->encode(tok);
            if (ids.size() == 1) boundary_ids_.push_back(ids[0]);
            else logf("boundary token %s is %zu tokens, ignored", tok.c_str(), ids.size());
        }
        {
            std::string b;
            for (int32_t id : boundary_ids_) b += " " + std::to_string(id);
            logf("checkpoint boundaries:%s (interval floor %d tokens, last %d turns)",
                 b.empty() ? " none" : b.c_str(), snap_every_, snap_turns_);
        }
    }

    void run() {
        std::fwrite(READY_SENTINEL, 1, sizeof(READY_SENTINEL) - 1, g_wire);
        std::fflush(g_wire);
        wire("STAT 0 0.00 0.0 %.1f\n", rss_gb());
        logf("READY");

        for (;;) {
            std::string verb;
            Req q;
            if (!io_.next(verb, q)) break;
            if (verb == "SUBMIT")          serve_one(q);
            else if (verb == "BAD_FRAME")  wire("ERROR %s BAD_FRAME\n", q.id.empty() ? "0" : q.id.c_str());
            else if (verb == "CANCEL")     { if (!q.id.empty()) wire("ERROR %s NOT_FOUND\n", q.id.c_str()); }
            else if (verb == "IMAGE")      wire("ERROR %s BAD_REQUEST\n", q.id.c_str());
            if (io_.eof()) break;
        }
        logf("stdin closed, exiting");
    }

private:
    struct ReusePlan { int pos = 0; int snap = -1; bool live = false; };

    ReusePlan plan_reuse(Slot & sl, const std::vector<int32_t> & toks) {
        ReusePlan p;
        const int cap = (int) toks.size() - 1;
        if (cap <= 0) return p;
        if (!sl.seq.empty() && (int) sl.seq.size() <= cap && is_prefix(sl.seq, toks)) {
            p.pos = (int) sl.seq.size();
            p.live = true;
        }
        for (int i = 0; i < (int) sl.snaps.size(); ++i) {
            const Snapshot & s = sl.snaps[(size_t) i];
            const int sp = s.pos();
            if (sp <= p.pos || sp > cap)        continue;
            if (!is_prefix(s.seq, toks))        continue;
            if (!s.kv && !is_prefix(s.seq, sl.seq)) continue;
            p.pos  = sp;
            p.snap = i;
            p.live = false;
        }
        return p;
    }

    double protect_cells(Slot & sl, int from) {
        std::vector<int> want;
        for (const Snapshot & s : sl.snaps)
            if (!s.kv && s.pos() > from && is_prefix(s.seq, sl.seq)) want.push_back(s.pos());
        if (want.empty()) return 0.0;
        std::sort(want.begin(), want.end());
        int need = want.back();
        while (need > 0 && sl.run->kv_bytes(need) > budget_bytes_) {
            while (!want.empty() && want.back() >= need) want.pop_back();
            need = want.empty() ? 0 : want.back();
        }
        if (need <= 0) {
            logf("protect: %zu checkpoint(s) dropped -- their cells do not fit "
                 "FRANKEN_SNAP_BUDGET_MB", sl.snaps.size());
            sl.snaps.erase(std::remove_if(sl.snaps.begin(), sl.snaps.end(),
                              [&](const Snapshot & s) {
                                  return !s.kv && s.pos() > from;
                              }), sl.snaps.end());
            return 0.0;
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto img = std::make_shared<KvImage>();
        img->len = need;
        img->buf = host_alloc(devs_.front(), sl.run->kv_bytes(need));
        if (img->buf.pinned) sl.run->save_kv_async(img->buf.p, need);
        else                 sl.run->save_kv(img->buf.p, need);
        for (Snapshot & sn : sl.snaps) {
            if (sn.kv || sn.pos() <= from || !is_prefix(sn.seq, sl.seq)) continue;
            if (sn.pos() <= need) sn.kv = img;
        }
        sl.snaps.erase(std::remove_if(sl.snaps.begin(), sl.snaps.end(),
                          [&](const Snapshot & s) { return !s.kv && s.pos() > need; }),
                       sl.snaps.end());
        const double ms = ms_since(t0);
        logf("protect: %d cells, %.0f MB, %.0f ms issue (%s) -- a request is about to "
             "overwrite from %d", need, img->buf.bytes / 1e6, ms,
             img->buf.pinned ? "async" : "blocking", from);
        return ms;
    }

    void * rec_take(Slot & sl) {
        if (sl.rec_free.empty()) {
            if ((int) sl.rec_all.size() >= snap_keep_ + 8) {
                sl.run->sync_devices();
                evict(sl);
            }
            if (sl.rec_free.empty()) {
                sl.rec_all.push_back(host_alloc(devs_.front(), sl.rec_bytes));
                sl.rec_free.push_back(sl.rec_all.back().p);
                if (sl.rec_all.size() == 1)
                    logf("checkpoint buffers are %s (%.0f MB each)",
                         !devs_.front()->is_gpu() ? "host memory -- copies are memcpy"
                         : sl.rec_all.back().pinned ? "PINNED -- copies are async"
                                                    : "pageable -- copies BLOCK",
                         sl.rec_bytes / 1e6);
            }
        }
        void * p = sl.rec_free.back();
        sl.rec_free.pop_back();
        return p;
    }

    bool rec_pinned(const Slot & sl) const {
        return !sl.rec_all.empty() && sl.rec_all.front().pinned;
    }

    double take_snapshot(Slot & sl) {
        const auto t0 = std::chrono::steady_clock::now();
        Snapshot s;
        s.seq.assign(sl.seq.begin(), sl.seq.begin() + sl.run->pos());
        s.rec = rec_take(sl);
        s.stamp = ++sl.stamp;
        if (rec_pinned(sl)) sl.run->save_rec_async(s.rec);
        else                sl.run->save_rec(s.rec);
        sl.snaps.push_back(std::move(s));
        return ms_since(t0);
    }

    size_t snaps_bytes(const Slot & sl) const {
        size_t n = 0;
        std::vector<const KvImage *> seen;
        for (const Snapshot & s : sl.snaps) {
            n += sl.rec_bytes;
            if (!s.kv) continue;
            const KvImage * p = s.kv.get();
            if (std::find(seen.begin(), seen.end(), p) != seen.end()) continue;
            seen.push_back(p);
            n += p->buf.bytes;
        }
        return n;
    }

    void evict(Slot & sl) {
        auto drop = [&](Snapshot & s) { if (s.rec) sl.rec_free.push_back(s.rec); s.rec = nullptr; };
        for (Snapshot & s : sl.snaps)
            if (!s.kv && !is_prefix(s.seq, sl.seq)) drop(s);
        sl.snaps.erase(std::remove_if(sl.snaps.begin(), sl.snaps.end(),
                          [&](const Snapshot & s) { return s.rec == nullptr; }), sl.snaps.end());
        while ((int) sl.snaps.size() > snap_keep_ ||
               (snaps_bytes(sl) > budget_bytes_ && sl.snaps.size() > 1)) {
            long long a = -1, b = -1;
            for (const Snapshot & s : sl.snaps) {
                if (s.stamp > a) { b = a; a = s.stamp; }
                else if (s.stamp > b) b = s.stamp;
            }
            int victim = -1, low = INT32_MAX;
            for (int i = 0; i < (int) sl.snaps.size(); ++i) {
                const Snapshot & s = sl.snaps[(size_t) i];
                if (s.stamp == a || s.stamp == b) continue;
                if (s.pos() < low) { low = s.pos(); victim = i; }
            }
            if (victim < 0) break;
            drop(sl.snaps[(size_t) victim]);
            sl.snaps.erase(sl.snaps.begin() + (long) victim);
        }
    }

    std::vector<int> snap_points(const std::vector<int32_t> & toks, int from, int n) {
        std::vector<int> pts;
        if (!boundary_ids_.empty()) {
            std::vector<int> structural;
            for (int i = from + 1; i < n; ++i)
                if (std::find(boundary_ids_.begin(), boundary_ids_.end(), toks[(size_t) i])
                        != boundary_ids_.end())
                    structural.push_back(i);
            const int keep = std::max(1, snap_turns_);
            if ((int) structural.size() > keep)
                structural.erase(structural.begin(), structural.end() - keep);
            pts.insert(pts.end(), structural.begin(), structural.end());
        }
        for (int p = ((from / snap_every_) + 1) * snap_every_; p < n; p += snap_every_)
            pts.push_back(p);
        std::sort(pts.begin(), pts.end());
        pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
        std::vector<int> out;
        int last = from;
        for (int p : pts) {
            if (p - last < chunk_ / 2 || p <= from || p >= n) continue;
            out.push_back(p);
            last = p;
        }
        return out;
    }

    // Every slot's Ds4Runner gets its own Adapter (Ds4Config::adapt, when
    // FRANKEN_ADAPT=1) but they all mutate the SAME model_'s placement
    // (Ds4Model::set_resident) -- untested with more than one slot ever
    // live at once. Not exercised today: the deepseek_v4 family caps
    // max_kv_slots at 1 (c/family_registry.py), same as qwen38, so the
    // gateway never asks for a second slot -- but a future relaxation of
    // that cap would need this looked at first.
    bool ensure_slot(int s, std::string & why) {
        if (slots_[(size_t) s].run) return true;
        try {
            Ds4Config cfg;
            cfg.ctx             = ctx_;
            cfg.max_chunk       = chunk_;
            cfg.log_routing     = false;             // stdout is the wire; no per-token sync for it
            cfg.miss_stage      = env_int("FRANKEN_MISS_STAGE", 1);
            cfg.hip_graph       = env_int("FRANKEN_HIP_GRAPH", 0);
            cfg.hip_graph_bucket= env_int("FRANKEN_HIP_GRAPH_BUCKET", 1024);
            cfg.all_ops         = 0;
            cfg.prefill_pipeline= env_int("FRANKEN_PREFILL_PIPELINE", 1);
            cfg.adapt           = adapt_cfg_;
            slots_[(size_t) s].run = std::make_unique<Ds4Runner>(*model_, ops_, cfg);
            slots_[(size_t) s].rec_bytes = slots_[(size_t) s].run->rec_bytes();
            slots_[(size_t) s].run->report_cache_bytes(stderr);
            for (int d = 0; d < (int) owned_.size(); ++d)
                owned_[d]->vram_report(stderr, "after slot");
            logf("slot %d allocated (checkpoint: %.1f MB recurrent + %.2f kB a token of cells)",
                 s, slots_[(size_t) s].run->rec_bytes() / 1e6,
                 slots_[(size_t) s].run->kv_bytes(1024) / 1024.0 / 1024.0);
            return true;
        } catch (const std::exception & e) {
            why = e.what();
            slots_[(size_t) s].run.reset();
            return false;
        }
    }

    void serve_one(Req & q) {
        io_.arm(q.id);
        if (q.plen == 0) { wire("ERROR %s EMPTY_PROMPT\n", q.id.c_str()); return; }
        if (q.slot < 0 || q.slot >= n_slots_) { wire("ERROR %s BAD_REQUEST\n", q.id.c_str()); return; }

        std::vector<int32_t> toks;
        try {
            toks = vocab_->encode(q.payload);
        } catch (const std::exception & e) {
            logf("tokenizer: %s", e.what());
            wire("ERROR %s BAD_REQUEST\n", q.id.c_str());
            return;
        }
        if (toks.empty()) { wire("ERROR %s EMPTY_PROMPT\n", q.id.c_str()); return; }

        const int prompt_tokens = (int) toks.size();
        const int budget = q.max_tokens > 0 ? q.max_tokens : 256;
        if (prompt_tokens + budget > ctx_) {
            wire("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",
                 q.id.c_str(), prompt_tokens, budget, ctx_);
            return;
        }
        std::string why;
        if (!ensure_slot(q.slot, why)) {
            logf("slot %d does not fit: %s", q.slot, why.c_str());
            wire("ERROR %s SLOT_UNAVAILABLE\n", q.id.c_str());
            return;
        }
        wire("ACCEPT %s %d\n", q.id.c_str(), prompt_tokens);

        Slot & sl = slots_[(size_t) q.slot];
        Ds4Runner & run = *sl.run;

        const ReusePlan plan = plan_reuse(sl, toks);
        const int reused = plan.pos;
        double protect_ms = protect_cells(sl, reused);
        double restore_ms = 0.0;
        {
            const auto tr = std::chrono::steady_clock::now();
            if (plan.snap >= 0) {
                const Snapshot & s = sl.snaps[(size_t) plan.snap];
                run.load_rec(s.rec, reused);
                if (s.kv) run.load_kv(s.kv->buf.p, s.kv->len, reused, reused);
            } else if (!plan.live) {
                run.reset_state();
            }
            restore_ms = ms_since(tr);
        }
        sl.seq.assign(toks.begin(), toks.begin() + reused);

        std::fprintf(stderr, "[serve-ds4] REUSE %s %d %d\n", q.id.c_str(), reused, prompt_tokens);
        std::fflush(stderr);

        const auto t0 = std::chrono::steady_clock::now();
        Recorder rec;
        bool cancelled = false;
        int  last_snap = reused;
        int32_t greedy = -1;
        int  emitted = 0;
        bool limited = false;
        auto t_prefill = t0;
        double snap_ms = 0.0;
        int    chunks = 0;

        try {
        const std::vector<int> points = snap_points(toks, reused, prompt_tokens);
        size_t next_pt = 0;
        for (int t = reused; t < prompt_tokens && !cancelled; ) {
            int limit = prompt_tokens;
            while (next_pt < points.size() && points[next_pt] <= t) ++next_pt;
            if (next_pt < points.size()) limit = std::min(limit, points[next_pt]);
            const int T = std::min(chunk_, limit - t);
            sl.seq.insert(sl.seq.end(), toks.begin() + t, toks.begin() + t + T);
            ++chunks;
            const bool is_last = (t + T == prompt_tokens);
            const int g = run.step(toks.data() + t, T, rec, is_last);
            if (g >= 0) greedy = g;
            t += T;
            if (next_pt < points.size() && t == points[next_pt] && !is_last) {
                snap_ms += take_snapshot(sl);
                last_snap = t;
                ++next_pt;
            }
            cancelled = io_.drain(q.id);
        }
        t_prefill = std::chrono::steady_clock::now();

        if (!cancelled) {
            int32_t next = (q.temp <= 0.0f) ? greedy
                                            : sample_token(run.logits_host(), q.temp, q.top_p,
                                                           n_cand_, rng_);
            for (;;) {
                if (next < 0 || vocab_->is_eog(next)) break;
                const std::string piece = vocab_->piece(next);
                wire_data(q.id, piece.data(), (int) piece.size());
                ++emitted;
                if (emitted >= budget) { limited = true; break; }
                if (run.pos() + 1 > ctx_)  { limited = true; break; }
                sl.seq.push_back(next);
                const int g = run.step(next, rec);
                next = (q.temp <= 0.0f) ? (int32_t) g
                                        : sample_token(run.logits_host(), q.temp, q.top_p,
                                                       n_cand_, rng_);
                if (io_.drain(q.id)) { cancelled = true; break; }
            }
        }
        if (run.pos() > last_snap) snap_ms += take_snapshot(sl);
        run.sync_devices();
        evict(sl);
        } catch (const std::exception & e) {
            logf("req=%s FAILED: %s", q.id.c_str(), e.what());
            try { run.reset_state(); } catch (const std::exception &) {}
            sl.seq.clear();
            sl.snaps.clear();
            wire("ERROR %s ENGINE_ERROR\n", q.id.c_str());
            return;
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double dec_s = std::chrono::duration<double>(t1 - t_prefill).count();
        const double tps   = dec_s > 0.0 ? emitted / dec_s : 0.0;
        const double hit   = prompt_tokens > 0 ? 100.0 * reused / prompt_tokens : 0.0;
        logf("req=%s slot=%d prompt=%d reused=%d from=%d chunks=%d emitted=%d limited=%d "
             "cancelled=%d prefill_s=%.2f [restore=%.0f protect=%.0f snap=%.0f ms] "
             "decode_s=%.2f tok/s=%.2f ckpts=%zu ckpt_mb=%.0f",
             q.id.c_str(), q.slot, prompt_tokens, reused, reused, chunks, emitted,
             (int) limited, (int) cancelled,
             std::chrono::duration<double>(t_prefill - t0).count(),
             restore_ms, protect_ms, snap_ms, dec_s, tps,
             sl.snaps.size(), snaps_bytes(sl) / 1e6);

        wire("DONE %s STAT %d %.2f %.1f %.1f %d %d %d\n",
             q.id.c_str(), emitted, tps, hit, rss_gb(), prompt_tokens, (int) limited, reused);
        if (cancelled) wire("ERROR %s CANCELLED\n", q.id.c_str());
    }

    bool test_;
    int  ctx_ = 262144, chunk_ = 256, n_slots_ = 1;
    int  snap_every_ = 512, snap_keep_ = 8, n_cand_ = 4096, snap_turns_ = 3;
    std::vector<int32_t> boundary_ids_;
    size_t budget_bytes_ = (size_t) 6144 * 1024 * 1024;
    AdaptConfig adapt_cfg_;
    std::vector<std::unique_ptr<Backend>> owned_;
    std::vector<Backend *>                devs_;
    std::vector<std::unique_ptr<Ds4Ops>>  owned_ops_;
    std::vector<Ds4Ops *>                 ops_;
    std::unique_ptr<Ds4Model>             model_;
    std::unique_ptr<Vocab>                vocab_;
    std::vector<Slot>                     slots_;
    FrameIO                               io_;
    std::mt19937                          rng_;
};

} // namespace

int serve_main(int argc, char ** argv, bool test_mode) {
    (void) argc;
    (void) argv;
    try {
        wire_open();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "serve: %s\n", e.what());
        return 1;
    }
    try {
        Server s(test_mode);
        s.run();
    } catch (const std::exception & e) {
        logf("fatal: %s", e.what());
        return 1;
    }
    return 0;
}

} // namespace ds4
} // namespace fk
