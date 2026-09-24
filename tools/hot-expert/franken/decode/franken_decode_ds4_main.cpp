// tools/hot-expert/franken/decode/franken_decode_ds4_main.cpp -- the whole
// entry point of the `franken_decode_ds4` binary (M1). Deliberately not
// franken_decode.cpp: that file's main() owns the Qwen3.8 CLI dispatch and
// argv-sniffs --model to route a deepseek4 GGUF into fk::ds4::ds4_main()
// already (L5 step 1, DEEPSEEK4.md) -- but that sniff only works when --model
// is an argv token, and the gateway launches every engine as `[binary, <cap>]`
// with the model named by SNAP/FRANKEN_GGUF in the ENVIRONMENT instead
// (GATEWAY-PROTOCOL.md section 1). A binary whose main() always means "this
// process is DeepSeek-V4" sidesteps that: no sniffing, no ambiguity, and the
// Qwen path (franken_decode) is untouched by this file existing.
//
// The SERVE dispatch lives HERE, not inside ds4_main() (ds4_graph.cpp):
// ds4_graph.cpp is the file every L5 step's GPU gate (ds4_gpu_gate.sh) and
// the CLI measurement/oracle tooling depend on, and gateway serving is a
// concern only THIS binary has -- adding it there would mean every future
// reader of ds4_main() has to know a --serve/SERVE=1 branch exists that the
// combined `franken_decode`/`franken_decode_cpu` binary can never actually
// reach (its own main(), franken_decode.cpp, intercepts SERVE=1 before ever
// sniffing --model's architecture, so ds4_main() only ever runs there in
// plain CLI mode). Checking it here instead keeps ds4_graph.cpp exactly the
// file DEEPSEEK4.md already describes, and makes this binary's own contract
// self-contained: SERVE/--serve/--serve-test -> fk::ds4::serve_main()
// (ds4_serve.cpp); anything else -> ds4_main()'s own CLI (measurement,
// oracle, --dump, ...), unchanged, the same code L5 steps 1-5 already gated.
#include <cstring>
#include <cstdlib>

#include "ds4_graph.h"
#include "ds4_serve.h"

int main(int argc, char ** argv) {
    bool serve = false, serve_test = false;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--serve"))      serve = true;
        else if (!std::strcmp(argv[i], "--serve-test")) serve = serve_test = true;
    }
    const char * s = std::getenv("SERVE");
    if (s && !std::strcmp(s, "1")) serve = true;
    if (serve) return fk::ds4::serve_main(argc, argv, serve_test);
    return fk::ds4::ds4_main(argc, argv);
}
