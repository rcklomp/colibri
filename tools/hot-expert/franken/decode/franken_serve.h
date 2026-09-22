// tools/hot-expert/franken/decode/franken_serve.h
//
// L0 step 4: the Franken engine as a gateway engine. `franken_decode --serve`
// (or SERVE=1 in the environment, which is what c/openai_server.py sets) hands
// main() over to serve_main(), which speaks the line protocol GATEWAY-PROTOCOL.md
// specs -- SUBMIT/CANCEL/STOP in, ACCEPT/DATA/DONE/ERROR out -- over stdin and
// stdout, and nothing else.
//
// `test_mode` is --serve-test: the same loop with the CPU backend and a short
// layer span, so serve_conformance.py can drive a real round trip without a
// GPU and without minutes per request. It changes no frame and no field.

#pragma once

namespace fk {

int serve_main(int argc, char ** argv, bool test_mode);

} // namespace fk
