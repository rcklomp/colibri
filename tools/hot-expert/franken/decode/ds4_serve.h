// tools/hot-expert/franken/decode/ds4_serve.h -- DeepSeek-V4-Flash as a
// gateway engine (M1: "put DeepSeek-V4-Flash behind Colibri's gateway
// exactly as Qwen3.8 is", GATEWAY-PROTOCOL.md, record L0-STEP4).
//
// This is franken_serve.cpp's `serve_main` ported to the Ds4Model/Ds4Runner
// API instead of Qwen's DecodeModel/DecodeRunner -- same wire protocol, same
// checkpoint design (Ds4Runner's rec/kv split mirrors DecodeRunner's own,
// ds4_graph.h's file comment on the split), deliberately a SEPARATE
// self-contained file rather than a shared abstraction over franken_serve.cpp:
// the two engines' construction (Ds4Ops/Placement/AdaptConfig have no Qwen
// equivalent, and Qwen's PLE gather has no DS4 equivalent) differ enough that
// sharing would mean parameterizing franken_serve.cpp's Server class for a
// second, differently-shaped model -- a bigger, riskier change to the engine
// already in service than standing up a parallel file. `franken_decode_ds4`
// (Makefile) links this instead of franken_decode.cpp, so the Qwen path is
// untouched.
//
// `test_mode` is --serve-test, exactly as it is for Qwen: the CPU backend, a
// short layer span, for serve_conformance.py to drive a real round trip with
// no GPU and no minutes per request.

#pragma once

namespace fk {
namespace ds4 {

int serve_main(int argc, char ** argv, bool test_mode);

} // namespace ds4
} // namespace fk
