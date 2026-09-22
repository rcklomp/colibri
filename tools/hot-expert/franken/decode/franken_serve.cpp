// tools/hot-expert/franken/decode/franken_serve.cpp -- see franken_serve.h.
//
// THE CONTRACT THIS FILE IMPLEMENTS is GATEWAY-PROTOCOL.md, which was read out
// of c/openai_server.py and c/glm53.c rather than out of any description of
// them. Section numbers below are that file's. Nothing here is negotiable by
// taste: every frame, every field order and the one ordering rule that is easy
// to get backwards (DONE before ERROR CANCELLED, section 3) come from there.
//
// The five things a new engine most easily gets wrong (that file's closing
// list), and where each is handled here:
//
//   1. "the SUBMIT payload is already tokenized" -- it is NOT, it is UTF-8 chat
//      text. `Vocab` below loads the GGUF's own tokenizer through libllama with
//      vocab_only=true (no weights, no backend, no GPU -- llama.cpp only calls
//      ggml_backend_load_all() from llama_backend_init(), which this file never
//      calls, and llama_model_load_from_file_impl's device enumeration is
//      guarded by `!params.vocab_only`; src/llama.cpp:405).
//   2. "reused is advisory" -- it is not; see report_reused() and section 5.
//      What is reported is the position the slot's recurrent state was actually
//      rolled back to, which is by construction the number of tokens this turn
//      did not run through the model.
//   3. "forgetting to flush" -- every wire write ends in fflush, and the wire
//      is an UNBUFFERED FILE* on a dup of the original fd 1, while fd 1 itself
//      is pointed at stderr for the whole run (wire_open()). A stray printf
//      from the loader or a kernel cannot corrupt a frame even in principle.
//   4. "not queuing SUBMIT frames that arrive mid-turn" -- FrameIO::drain()
//      reads whole frames into a queue exactly as glm53.c's g_queue does, so a
//      CANCEL sitting behind them in the pipe is still seen.
//   5. "CANCEL/DONE/ERROR ordering" -- serve_one() emits DONE with the partial
//      counts and only then ERROR <id> CANCELLED.
//
// WHAT THIS ENGINE DOES NOT DO, deliberately: no prefix CHECKPOINTS (a side
// cache that restores a prefix into a fresh slot -- section 5 says plain
// per-slot reuse passes every accept_live.sh check on its own), no TOOL/ECHO
// frames (the gateway parses tool calls out of the DATA text for every family),
// no IMAGE (text-only model; the frame is consumed to keep the stream in sync
// and refused), and STOP is a no-op (section 2: it is one on glm53 too).

#include "franken_serve.h"

#include <dlfcn.h>
#include <poll.h>
#include <unistd.h>

// Where libllama lives on this rig; the Makefile passes the build's own path.
// FRANKEN_LLAMA_SO overrides it at run time.
#ifndef FRANKEN_LLAMA_SO_DEFAULT
#define FRANKEN_LLAMA_SO_DEFAULT "libllama.so"
#endif
// The two ROCm 7 libraries libggml-hip.so.0 needs and this host does not have
// on its path (LlamaApi::open explains why they are loaded by absolute path).
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

#include "../ple.h"
#include "decode_graph.h"
#include "decode_model.h"

namespace fk {
namespace {

// ---------------------------------------------------------------- the wire --

// The gateway's boot sentinel, byte for byte (openai_server.py:45).
const char READY_SENTINEL[] = "\x01\x01READY\x01\x01\n";

FILE * g_wire = nullptr;

// stdout IS the protocol. So the protocol gets its own file descriptor and
// fd 1 gets stderr: DecodeRunner's progress lines, the backend's VRAM report,
// libllama's loader chatter and anything else that ever prints then land in
// the log where they belong instead of between two frames.
void wire_open() {
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0) throw std::runtime_error("cannot dup stdout");
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot redirect stdout");
    g_wire = fdopen(fd, "w");
    if (!g_wire) throw std::runtime_error("cannot fdopen the wire");
    setvbuf(g_wire, nullptr, _IONBF, 0);
    setvbuf(stdin,  nullptr, _IONBF, 0);   // so poll() on fd 0 cannot miss bytes
                                           // that stdio has already swallowed
    setvbuf(stdout, nullptr, _IOLBF, 0);
}

void wire(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_wire, fmt, ap);
    va_end(ap);
    std::fflush(g_wire);
}

// DATA <id> <n>\n<n bytes>\n -- one generated token's UTF-8 bytes (section 3).
// The bytes are raw: a piece that is half a codepoint is legal and the gateway
// runs an incremental UTF-8 decoder over the stream for exactly that reason.
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
    std::fprintf(stderr, "[serve] ");
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

// ------------------------------------------------------------ the frames --

struct Req {
    // A STRING id, not %llu. The gateway sends a decimal integer today, but
    // c/serve_codec.h (qwen38/qwen36, "the newer dialect", section 2) types it
    // as a string and this engine serves under the qwen38 family -- so it is
    // echoed back verbatim, whatever it is, and never reformatted.
    std::string id;
    int   slot = 0;
    int   plen = 0;
    int   max_tokens = 0;
    float temp = 0.0f;
    float top_p = 1.0f;
    int   prefix_bytes = 0;     // the 8th field: a HINT, unused here (see below)
    std::string payload;
};

constexpr int MAX_QUEUED = 16;   // the gateway admits at most kv_slots at once

class FrameIO {
public:
    // A whole frame, from the mid-turn queue if one is waiting there, else from
    // the pipe. False on EOF.
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

    // The drain (section 2's CANCEL): see a CANCEL for the turn in flight
    // WITHOUT ever stopping generation to wait for one. Returns true once the
    // in-flight request is cancelled. Every frame it reads it either answers
    // (a CANCEL for someone else) or queues whole (a SUBMIT), so the CANCEL
    // that follows one of those in the pipe is not lost either.
    bool drain(const std::string & inflight) {
        if (cancelled_) return true;
        while (!pushback_full_ && stdin_ready()) {
            std::string line;
            if (!read_line(line)) {
                // EOF. On a closed pipe "ready" stays true forever, so this
                // loop must end -- and it ends as a cancellation: generating
                // for a reader that is gone is the only outcome worse than
                // stopping.
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
            if (!std::strcmp(cmd, "STOP")) continue;      // a no-op here, section 2
            if (!std::strcmp(cmd, "SUBMIT") && (int) queue_.size() < MAX_QUEUED) {
                std::string verb;
                Req q;
                read_body(line, verb, q);                 // the WHOLE frame, body included
                queue_.emplace_back(verb, q);
                continue;
            }
            // IMAGE, or a full queue: the bytes that follow belong to THAT
            // frame, so the header goes back and the drain stops here.
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

    // The header is already read; the body is read by COUNTED BYTES, never by
    // lines -- it can contain them (section 2).
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
            (void) std::fgetc(stdin);                 // the frame's closing '\n'
            return;                                   // refused by the caller
        }
        if (verb != "SUBMIT") return;                 // unknown verbs are ignored
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
        // The extension payload (grammar/audio). The gateway never sends one to
        // this family -- FamilyCapabilities.grammar_payload is false and the
        // 400 is raised before the SUBMIT is written -- but if one ever arrives
        // it is consumed rather than left to desync the next header.
        if (xlen > 0) {
            std::vector<char> ext((size_t) xlen);
            if (std::fread(ext.data(), 1, (size_t) xlen, stdin) != (size_t) xlen) {
                verb = "BAD_FRAME";
                return;
            }
        }
        (void) std::fgetc(stdin);                     // the frame's closing '\n'
    }

    std::deque<std::pair<std::string, Req>> queue_;
    std::string pushback_;
    bool pushback_full_ = false;
    bool cancelled_ = false;
    bool eof_ = false;
};

// --------------------------------------------------------------- tokenizer --

// libllama is DLOPENED, not linked, and that is a deliberate choice rather
// than a convenience.
//
// `ldd libggml.so.0` on this box lists libggml-hip.so.0 as a DT_NEEDED, which
// lists libamdhip64. Linking -lllama would therefore put a HIP runtime in
// franken_decode_cpu's own dependency list and break the one claim that
// binary exists to make -- `make ldd-check`, "the CPU binary links no HIP
// runtime". Resolved at run time, in serve mode only, the link contract of
// both binaries is unchanged and `ldd` still says what it said.
//
// It does NOT make the CPU conformance run HIP-free by itself: dlopen pulls
// the same NEEDED chain into the process. What keeps it off a card is that
// nothing here ever calls a ggml_backend_* entry point -- llama.cpp builds its
// backend registry lazily, on the first such call, and the vocab_only load
// path has none (src/llama.cpp:405 short-circuits before ggml_backend_reg_count(),
// and make_cpu_buft_list's device walk is inside load_tensors, which
// vocab_only returns before) -- and that serve_conformance.py runs the driver
// with HIP_VISIBLE_DEVICES set to empty as a second, independent guard.
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
        // libggml.so.0 NEEDs libggml-hip.so.0, which NEEDs libhipblas.so.3 and
        // librocblas.so.5 -- and THOSE ARE NOT ON THIS HOST's library path:
        // /opt/rocm here is 6.2 (libhipblas.so.2, librocblas.so.4), while the
        // build-hip build was made in the ROCm 7.14 docker image. `ldd
        // libllama.so` says "not found" for both, so a plain dlopen of it fails
        // outright on the host even though nothing in this process will ever
        // call a BLAS entry point. The ROCm 7 pair does exist on the box, in
        // the rocm SDK wheels under ~/venvs/rocm; loading them BY ABSOLUTE PATH
        // and RTLD_GLOBAL first satisfies the soname for the loader (it checks
        // what is already loaded before it searches), so libllama then opens
        // with no LD_LIBRARY_PATH from the caller. LD_LIBRARY_PATH is
        // deliberately NOT how start_franken.sh solves it: that directory also
        // holds a libamdhip64 of a different ROCm than the one this box runs,
        // and putting it in front of the system path would swap the HIP
        // runtime under this engine's own kernels to fix a dependency of a
        // library it never calls. A missing preload is not an error by itself
        // (the caller may have arranged the path some other way); the dlopen
        // below is what decides.
        for (const std::string & dep : split_paths(env_str("FRANKEN_LLAMA_PRELOAD",
                                                           FRANKEN_LLAMA_PRELOAD_DEFAULT))) {
            if (dep.empty()) continue;
            if (!dlopen(dep.c_str(), RTLD_LAZY | RTLD_GLOBAL))
                logf("preload %s: %s (continuing)", dep.c_str(), dlerror());
        }
        void * h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            // dlerror() CLEARS the error, so it is read exactly once. Calling
            // it twice in one expression (`dlerror() ? dlerror() : "?"`) hands
            // the second call's nullptr to std::string and segfaults -- which
            // is precisely how this engine's first boot died, with no message
            // at all where the real one ("libhipblas.so.3: cannot open shared
            // object file") was waiting.
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

// The GGUF's own tokenizer, with vocab_only=true: the header and the vocab are
// read, not one weight tensor, and no ggml backend is registered. The engine's
// own loader (DecodeModel) keeps doing the weights.
class Vocab {
public:
    explicit Vocab(const std::string & gguf) {
        api_.open(env_str("FRANKEN_LLAMA_SO", FRANKEN_LLAMA_SO_DEFAULT));
        api_.log_set([](ggml_log_level, const char * text, void *) {
            std::fputs(text, stderr);
        }, nullptr);
        llama_model_params mp = api_.model_default_params();
        // vocab_only is the whole point: the header and the vocab, no tensor.
        // (This build's llama_model_params has no use_mmap field -- load_mode
        // carries that now -- so the default is left alone.)
        mp.vocab_only = true;
        model_ = api_.model_load_from_file(gguf.c_str(), mp);
        if (!model_) throw std::runtime_error("cannot load the tokenizer from " + gguf);
        vocab_ = api_.model_get_vocab(model_);
        n_     = api_.vocab_n_tokens(vocab_);
    }
    ~Vocab() { if (model_) api_.model_free(model_); }

    // The payload is chat text the GATEWAY has already templated (section 4),
    // so the role markers in it are real special tokens and must be parsed as
    // such -- parse_special=true. add_special is false: the template already
    // carries whatever the model wants at the front, and letting the tokenizer
    // add a second BOS would shift every position by one against the prefix the
    // slot already holds.
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

    // special=true: a special token that is NOT an end-of-generation token has
    // to reach the gateway as the text it is, because the gateway's own
    // stop_filter matches role markers over the decoded stream (section 4) and
    // cannot match what it never sees.
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

// Sampling is the ENGINE's job -- the gateway forwards temperature and top_p
// and samples nothing (section 4). temperature 0 is greedy, which is what the
// device-side argmax already produced, so the greedy path never downloads a
// logit vector at all; only this function does.
int32_t sample_token(const std::vector<float> & logits, float temp, float top_p,
                     int n_cand, std::mt19937 & rng) {
    const int n = (int) logits.size();
    if (n <= 0) return -1;
    if (temp <= 0.0f) {
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    // The nucleus never reaches past the head of the distribution, so the tail
    // is cut by rank first: nth_element is O(n) where a full sort of 248 320
    // logits a token would not be. FRANKEN_TOPK_CAND sets the rank; the mass
    // beyond it is dropped rather than renormalised away, which is the one
    // deliberate approximation in this file.
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

// ------------------------------------------------------------------ slots --

// A rolled-back-to position and the recurrent state that belongs to it.
struct Snapshot {
    int               pos = 0;
    std::vector<char> blob;
};

struct Slot {
    std::unique_ptr<DecodeRunner> run;
    // The tokens the live state corresponds to: seq.size() == run->pos(), always.
    std::vector<int32_t>  seq;
    std::vector<Snapshot> snaps;
};

// --------------------------------------------------------------- the engine --

// SNAP is the model DIRECTORY the gateway resolved a family from (it needs a
// config.json, so it is the FP8 checkpoint's directory), while this engine
// reads a GGUF. FRANKEN_GGUF names the shard; failing that, the first
// *-00001-of-*.gguf (or any *.gguf) under SNAP is used, so a GGUF-only
// directory also works when the gateway is not the launcher.
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
        // The family's own context env (Q38_MAXT, family_registry.py) is
        // honoured so an operator who sets the documented knob is not ignored;
        // FRANKEN_CTX wins when both are set.
        int ctx = env_int("FRANKEN_CTX", 0);
        if (ctx <= 0) ctx = env_int("Q38_MAXT", 0);
        if (ctx <= 0) ctx = test_ ? 512 : 262144;
        ctx_ = ctx;
        n_slots_ = std::max(1, std::min(16, env_int("KV_SLOTS", 1)));
        snap_every_ = std::max(1, env_int("FRANKEN_SNAP_EVERY", chunk_));
        snap_keep_  = std::max(1, env_int("FRANKEN_SNAP_KEEP", 8));
        n_cand_     = std::max(1, env_int("FRANKEN_TOPK_CAND", 4096));
        const int seed = env_int("FRANKEN_SEED", 0);
        rng_.seed(seed > 0 ? (uint32_t) seed
                           : (uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());

        const std::string snap = env_str("SNAP");
        const std::string gguf = find_gguf(snap);
        logf("gguf=%s ctx=%d chunk=%d kv_slots=%d snap_every=%d snap_keep=%d",
             gguf.c_str(), ctx_, chunk_, n_slots_, snap_every_, snap_keep_);

        auto load_vocab = [&] {
            vocab_ = std::make_unique<Vocab>(gguf);
            logf("tokenizer: %d tokens (vocab_only, no weights of its own)", vocab_->n_tokens());
            for (const std::string & t : split_csv(env_str("FRANKEN_STOP_IDS")))
                vocab_->add_stop((int32_t) std::atoi(t.c_str()));
        };
        const bool vocab_first = env_int("FRANKEN_TOKENIZER_FIRST", 0) == 1;
        if (vocab_first) load_vocab();

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
        for (auto * b : devs_) {
            b->set_gemv_lds(env_int("FRANKEN_GEMV_LDS", 1));
            b->set_gemv_min_rows(env_int("FRANKEN_GEMV_MIN_ROWS", 1024));
            b->set_expert_gather(env_int("FRANKEN_EXPERT_GATHER", 1));
            // 0 is bit-exact against the decode kernel; 1 and 2 reassociate K
            // (franken_decode.cpp's --gemm-lds). Serving defaults to 0.
            b->set_gemm_lds(env_int("FRANKEN_GEMM_LDS", 0));
        }
        model_ = std::make_unique<DecodeModel>(gguf, devs_, il0, il1, /*with_head=*/true);
        logf("placed=%.2f GB layers=%d-%d devices=%d backend=%s",
             model_->placed_bytes() / 1e9, il0, il1, (int) devs_.size(), devs_.front()->name());
        for (int d = 0; d < (int) owned_.size(); ++d) owned_[d]->vram_report(stderr, "after weights");
        need_ple_ = (PLE_LAYER >= il0 && PLE_LAYER <= il1);
        slots_.resize((size_t) n_slots_);

        // THE TOKENIZER LOADS LAST, and that order is measured, not stylistic.
        // The conformance run's own log shows `ggml_cuda_init: failed to
        // initialize ROCm` while libllama was loading a VOCAB -- so something
        // on that path does construct ggml's backend registry, which on a box
        // with cards visible means ggml-hip enumerating all three and taking a
        // primary context on each. Doing it AFTER this engine's own
        // hipSetDevice/placement means those contexts already exist and ggml
        // finds them rather than creating them in front of a 22 GB upload.
        // The VRAM report on either side is what makes the claim checkable on
        // the first GPU run; FRANKEN_TOKENIZER_FIRST=1 puts it back in front.
        if (!vocab_first) {
            load_vocab();
            for (int d = 0; d < (int) owned_.size(); ++d)
                owned_[d]->vram_report(stderr, "after tokenizer");
        }
    }

    void run() {
        // Boot handshake, in this order and nothing between them (section 1).
        std::fwrite(READY_SENTINEL, 1, sizeof(READY_SENTINEL) - 1, g_wire);
        std::fflush(g_wire);
        wire("STAT 0 0.00 0.0 %.1f\n", rss_gb());
        logf("READY");

        for (;;) {
            std::string verb;
            Req q;
            if (!io_.next(verb, q)) break;             // EOF: the gateway is gone
            if (verb == "SUBMIT")          serve_one(q);
            else if (verb == "BAD_FRAME")  wire("ERROR %s BAD_FRAME\n", q.id.empty() ? "0" : q.id.c_str());
            else if (verb == "CANCEL")     { if (!q.id.empty()) wire("ERROR %s NOT_FOUND\n", q.id.c_str()); }
            else if (verb == "IMAGE")      wire("ERROR %s BAD_REQUEST\n", q.id.c_str());
            // STOP and anything unrecognised: ignored, exactly as glm53 does.
            // The drain turns a closed pipe into a cancellation; serving what
            // it had already queued would only write into it.
            if (io_.eof()) break;
        }
        logf("stdin closed, exiting");
    }

private:
    static std::vector<std::string> split_csv(const std::string & s) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : s) {
            if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    // The PLE gather for positions [start, start+T) of `seq`.
    //
    // ple_gather() takes a token LIST and hashes each position against its
    // PLE_NGRAM-1 predecessors and nothing else (ple.h: "t[1..n-1] are its
    // 1..n-1 predecessors"), so the rows for a window that carries those
    // predecessors are the rows the whole-prefix gather would have produced --
    // at O(T) instead of O(pos). The CLI re-gathers the whole prefix for every
    // appended token, which is quadratic and unusable at a 256k prompt.
    // FRANKEN_PLE_SELFCHECK=1 checks the two against each other (serve-test
    // turns it on by default; it is O(pos) and only for the test).
    std::vector<float> ple_rows(const std::vector<int32_t> & seq, int start, int T) {
        if (!need_ple_) return {};
        const int back = std::min(start, PLE_NGRAM - 1);
        std::vector<int32_t> win(seq.begin() + (start - back), seq.begin() + (start + T));
        const auto r = franken::ple_gather(model_->gguf(), win);
        std::vector<float> out((size_t) T * N_EMBD);
        std::copy(r.emb.begin() + (size_t) back * N_EMBD,
                  r.emb.begin() + (size_t)(back + T) * N_EMBD, out.begin());
        if (env_int("FRANKEN_PLE_SELFCHECK", test_ ? 1 : 0) == 1) {
            const auto full = franken::ple_gather(model_->gguf(),
                                  std::vector<int32_t>(seq.begin(), seq.begin() + (start + T)));
            const bool same = std::equal(out.begin(), out.end(),
                                         full.emb.begin() + (size_t) start * N_EMBD);
            logf("ple_selfcheck start=%d T=%d window=%s", start, T, same ? "IDENTICAL" : "DIFFERENT");
        }
        return out;
    }

    // Lazily, because a slot is 1.21 GB a card at 256k and the cards have
    // 2.4-3.3 GB free after the weights: at the served context exactly one
    // fits, and a slot the gateway never uses must not be what takes the card
    // out. A failure here is an error for THIS request, not a dead engine.
    bool ensure_slot(int s, std::string & why) {
        if (slots_[(size_t) s].run) return true;
        try {
            DecodeConfig cfg;
            cfg.ctx        = ctx_;
            cfg.max_tokens = chunk_;
            cfg.verbose    = false;
            cfg.log_routing= false;
            cfg.progress   = false;         // stdout is the wire
            slots_[(size_t) s].run = std::make_unique<DecodeRunner>(*model_, cfg);
            slots_[(size_t) s].run->report_cache_bytes(stderr);
            for (int d = 0; d < (int) owned_.size(); ++d)
                owned_[d]->vram_report(stderr, "after slot");
            logf("slot %d allocated (state snapshot = %.1f MB a copy)",
                 s, slots_[(size_t) s].run->state_bytes() / 1e6);
            return true;
        } catch (const std::exception & e) {
            why = e.what();
            slots_[(size_t) s].run.reset();
            return false;
        }
    }

    // Section 5's rule, as arithmetic. Candidates are the snapshots plus the
    // LIVE position (which needs no restore at all and is the continuation
    // case: the slot already sits exactly where the next turn's prefix ends).
    // The cap at n_prompt-1 is not cosmetic: a turn must run at least one token
    // through the model or it has no logits to sample from.
    int plan_reuse(Slot & sl, const std::vector<int32_t> & toks, int & snap_idx) {
        snap_idx = -1;
        const int live = (int) sl.seq.size();
        int lcp = 0;
        const int n = std::min(live, (int) toks.size());
        while (lcp < n && sl.seq[(size_t) lcp] == toks[(size_t) lcp]) ++lcp;
        const int cap = std::min(lcp, (int) toks.size() - 1);
        if (cap <= 0) return 0;
        int best = 0;
        if (live <= cap) best = live;                 // the live state is usable as it stands
        for (int i = 0; i < (int) sl.snaps.size(); ++i) {
            const int p = sl.snaps[(size_t) i].pos;
            if (p <= cap && p > best) { best = p; snap_idx = i; }
        }
        if (best == live) snap_idx = -1;
        return best;
    }

    void take_snapshot(Slot & sl) {
        Snapshot s;
        s.pos = sl.run->pos();
        s.blob.resize(sl.run->state_bytes());
        sl.run->save_state(s.blob.data());
        sl.snaps.push_back(std::move(s));
        // Decimation rather than a FIFO: what a rollback needs is a snapshot
        // just BELOW the shared prefix's end, and that end is anywhere. Evict
        // the one whose removal leaves the smallest hole, keeping the oldest
        // and the newest, so the survivors stay spread over the sequence.
        while ((int) sl.snaps.size() > snap_keep_) {
            size_t victim = 1;
            int    hole = INT32_MAX;
            for (size_t i = 1; i + 1 < sl.snaps.size(); ++i) {
                const int h = sl.snaps[i + 1].pos - sl.snaps[i - 1].pos;
                if (h < hole) { hole = h; victim = i; }
            }
            sl.snaps.erase(sl.snaps.begin() + (long) victim);
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
        // glm53's own fallback, kept for the same reason (section 2): a caller
        // that sends 0 gets a reply rather than an empty one.
        const int budget = q.max_tokens > 0 ? q.max_tokens : 256;
        if (prompt_tokens + budget > ctx_) {
            // The spelling the gateway turns into a clean 400 -- the qwen38 one,
            // since this engine serves under that family (section 3, and
            // openai_server.py's _engine_error reads both).
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
        // The early acknowledgement (section 3). Optional for glm53, sent here
        // because it is what lets a context rejection reach the client as a 400
        // before anything is committed.
        wire("ACCEPT %s %d\n", q.id.c_str(), prompt_tokens);

        Slot & sl = slots_[(size_t) q.slot];
        DecodeRunner & run = *sl.run;

        int snap_idx = -1;
        const int reused = plan_reuse(sl, toks, snap_idx);
        if (reused == 0) {
            run.reset_state();
            sl.snaps.clear();
        } else if (snap_idx >= 0) {
            run.load_state(sl.snaps[(size_t) snap_idx].blob.data(), reused);
        }
        // Everything above the reuse point belonged to a sequence this prompt
        // has just diverged from.
        sl.snaps.erase(std::remove_if(sl.snaps.begin(), sl.snaps.end(),
                                      [&](const Snapshot & s) { return s.pos > reused; }),
                       sl.snaps.end());
        sl.seq.assign(toks.begin(), toks.begin() + reused);

        const auto t0 = std::chrono::steady_clock::now();
        Recorder rec;                                  // taps off: a tap is a download
        bool cancelled = false;
        int  last_snap = reused;
        int32_t greedy = -1;
        int  emitted = 0;
        bool limited = false;
        auto t_prefill = t0;

        try {
        // ---- prefill ------------------------------------------------------
        for (int t = reused; t < prompt_tokens && !cancelled; ) {
            const int T = std::min(chunk_, prompt_tokens - t);
            sl.seq.insert(sl.seq.end(), toks.begin() + t, toks.begin() + t + T);
            const std::vector<float> ple = ple_rows(sl.seq, t, T);
            const bool is_last  = (t + T == prompt_tokens);
            const bool snap_due = (t + T - last_snap >= snap_every_);
            // A chunk that is neither the last nor about to be snapshotted is
            // ENQUEUED: the pipeline is what makes the prompt 1.63 ms a token,
            // and the only thing that forces a wait is a host read of the
            // state (record §L0-PREFILL-2).
            const int g = run.step(toks.data() + t, T, ple.empty() ? nullptr : ple.data(),
                                   rec, is_last || snap_due);
            if (run.last_flushed()) greedy = g;
            t += T;
            if (snap_due && !is_last) { take_snapshot(sl); last_snap = t; }
            cancelled = io_.drain(q.id);
        }
        t_prefill = std::chrono::steady_clock::now();

        // ---- decode -------------------------------------------------------
        if (!cancelled) {
            int32_t next = (q.temp <= 0.0f) ? greedy
                                            : sample_token(run.logits_host(), q.temp, q.top_p,
                                                           n_cand_, rng_);
            for (;;) {
                if (next < 0 || vocab_->is_eog(next)) break;   // the stop token is never sent
                const std::string piece = vocab_->piece(next);
                wire_data(q.id, piece.data(), (int) piece.size());
                ++emitted;
                if (emitted >= budget) { limited = true; break; }
                if (run.pos() + 1 > ctx_)  { limited = true; break; }
                // Feed it, THEN sample the next one. That order is what makes
                // the slot's final position prompt+completion on a normal stop
                // and prompt+completion-1 on a budget stop -- which is exactly
                // what the gateway's ledger predicts for the next turn's
                // `reused` (section 5, ledger_expect_reuse).
                sl.seq.push_back(next);
                const std::vector<float> ple = ple_rows(sl.seq, run.pos(), 1);
                const int g = run.step(next, ple.empty() ? nullptr : ple.data(), rec);
                next = (q.temp <= 0.0f) ? (int32_t) g
                                        : sample_token(run.logits_host(), q.temp, q.top_p,
                                                       n_cand_, rng_);
                if (io_.drain(q.id)) { cancelled = true; break; }
            }
        }
        // The state the next turn will want to roll back to is the one this
        // turn ends in: one snapshot here is what makes a continuation reuse
        // EXACTLY prompt+completion tokens instead of the nearest chunk below.
        if (run.pos() > last_snap) take_snapshot(sl);
        } catch (const std::exception & e) {
            // One request must not take the engine down, and a slot whose
            // state got as far as an exception is not trustworthy: it is
            // cleared so the next turn re-prefills from zero rather than
            // reusing something half-written.
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
        logf("req=%s slot=%d prompt=%d reused=%d emitted=%d limited=%d cancelled=%d "
             "prefill_s=%.2f decode_s=%.2f tok/s=%.2f",
             q.id.c_str(), q.slot, prompt_tokens, reused, emitted, (int) limited,
             (int) cancelled, std::chrono::duration<double>(t_prefill - t0).count(), dec_s, tps);

        // DONE first, ALWAYS -- including for a cancelled turn. The gateway's
        // dispatcher pops the pending entry on ERROR, so a DONE after it would
        // be dropped with its counts (section 3).
        wire("DONE %s STAT %d %.2f %.1f %.1f %d %d %d\n",
             q.id.c_str(), emitted, tps, hit, rss_gb(), prompt_tokens, (int) limited, reused);
        if (cancelled) wire("ERROR %s CANCELLED\n", q.id.c_str());
    }

    bool test_;
    int  ctx_ = 262144, chunk_ = 256, n_slots_ = 1;
    int  snap_every_ = 256, snap_keep_ = 8, n_cand_ = 4096;
    bool need_ple_ = false;
    std::vector<std::unique_ptr<Backend>> owned_;
    std::vector<Backend *>                devs_;
    std::unique_ptr<DecodeModel>          model_;
    std::unique_ptr<Vocab>                vocab_;
    std::vector<Slot>                     slots_;
    FrameIO                               io_;
    std::mt19937                          rng_;
};

} // namespace

int serve_main(int argc, char ** argv, bool test_mode) {
    (void) argc;
    (void) argv;      // the gateway passes one positional cap and nothing else
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
        // Before READY this is a boot failure and the gateway's own
        // RuntimeError is the right outcome; after it, the process dying is
        // what Engine.close() is written for either way.
        logf("fatal: %s", e.what());
        return 1;
    }
    return 0;
}

} // namespace fk
