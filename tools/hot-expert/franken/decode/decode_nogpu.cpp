// tools/hot-expert/franken/decode/decode_nogpu.cpp
//
// The GPU backend's stand-in for `franken_decode_cpu`, the binary that links
// NO HIP at all. It exists so the `--cpu` check is provably GPU-free rather
// than GPU-free by inspection: `ldd franken_decode_cpu` shows no libamdhip64,
// so the binary cannot initialise a device even by accident, and the task
// rule ("nothing you run may touch a GPU") holds by construction.

#include <stdexcept>

#include "decode_backend.h"

namespace fk {

Backend * make_gpu_backend(int) {
    throw std::runtime_error(
        "this binary was built without HIP (franken_decode_cpu); use franken_decode for device 0");
}

} // namespace fk
