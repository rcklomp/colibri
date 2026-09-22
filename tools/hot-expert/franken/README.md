# L0 step 1: the loader

What is here (design doc `FRANKEN-ENGINE-DESIGN-2026-09-22.md` section 9, build order step 1): a
GGUF split reader for Qwen3.8-Flash-Next (`gguf_model.{h,cpp}`, ggml's own `gguf.h` API, no
`libllama`, mmap per shard, no GPU); the layer-range device placement table and the
per-layer/per-expert device-pointer table from design 9.1 (`placement.{h,cpp}`, plus an
`upload_placement()` that would `hipMalloc`/`hipMemcpy` everything but is never called here); and
the CPU-side per-layer n-gram (PLE) gather ported from llama.cpp's `qwen4exp.cpp` (`ple.{h,cpp}`,
including a `load_ple_table_pinned()` that would pull the whole 28.8 GB table into a
`hipHostMalloc`'d buffer but is likewise never called). `franken_load.cpp` is the CLI over all
three (`--plan`, `--ple-check <ids>`, `--upload`). What is **not** here: any device-side kernel,
graph, or forward pass (build order steps 2-4), the KV-cache-backed version of the PLE gather that
resumes a prefix checkpoint (`ple.h`'s header comment says exactly what the mmap-based
`compute_ple_indices()` here does and does not reproduce), and anything that requantises the
routed-expert tensors into the L0 4.25 bpw format design 9.3 calls for (`--upload` copies each
tensor's own file bytes verbatim, no conversion). `--upload` and `load_ple_table_pinned()` were
built (hipcc, no `--device`) but never run, per this task's hard rule against touching a GPU;
`--upload` additionally refuses at runtime unless `FRANKEN_ALLOW_UPLOAD=1` is set.

Build: `make` (see `Makefile`; runs `hipcc` inside `rocm/dev-ubuntu-24.04:7.14.0-full`, linking
`~/src/llama-glm53/build-hip/bin`'s already-built ggml libraries). Run the CPU-only modes directly
on the rig, e.g. `./franken_load --model ~/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --plan`.
