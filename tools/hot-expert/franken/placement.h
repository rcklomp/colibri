// tools/hot-expert/franken/placement.h
//
// L0 step 1's layer-range placement (design doc section 9.1): three
// devices, layer ranges 0-15 / 16-31 / 32-47, lm_head plus the final
// hyper-connection mixer on the last device, the token embedding lookup on
// the first, and the one tensor the design keeps off every device at all --
// the PLE n-gram table, a per-token row gather that belongs in host RAM
// (record §PLE-GATHER). This file only computes and prints where bytes
// would go and builds the per-expert device-pointer table the kernels index
// by (layer, expert); the actual --upload path (hipMalloc + hipMemcpy) is
// here too but is never invoked except by an explicit --upload command-line
// flag, which this build is not allowed to pass (CLAUDE.md/task rule: no
// GPU touch of any kind).

#pragma once

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "gguf_model.h"

namespace franken {

// design 9.3's tensor kinds by source/behaviour, mapped from a tensor name.
enum class TensorClass {
    TRUNK = 0,      // read wholesale per token: attn/ssm/hc projections, ffn_gate_inp, token_embd, lm_head, ple_key/value/conv1d
    ROUTED_EXPERTS, // ffn_{gate,up,down}_exps -- the 512-expert MoE tensors
    SHARED_EXPERTS, // ffn_{gate,up,down}_shexp, ffn_gate_inp_shexp
    INDEXER,        // blk.N.indexer.*
    HC,             // hc_attn_*, hc_ffn_*, output_hc_* (hyper-connection mixers)
    NORMS,          // *_norm tensors not already claimed by indexer/hc above
    HOST_GATHER,    // per_layer_token_embd.weight -- the ONE tensor design 9.1 keeps off every device
    N_CLASSES
};

const char * tensor_class_name(TensorClass c);
TensorClass classify_tensor(const std::string & name);

// -1 for a HOST_GATHER tensor (design 9.1's rule: a tensor read by a
// per-token row gather lives in host RAM, never on a device). Otherwise the
// device index 0..n_devices-1 a "blk.<il>." tensor's layer maps to
// (il / layers_per_device, clamped to the last device); token_embd.weight
// starts the token on device 0, everything else non-layered (the output
// projection, the final HC mixer) runs after the last layer and lives with
// it on the last device.
int device_for_tensor(const TensorInfo & t, int n_devices, int layers_per_device);

struct DeviceClassTotals {
    std::array<std::array<size_t, (size_t) TensorClass::N_CLASSES>, 8> bytes{}; // bytes[dev][class], dev < 8
    std::array<size_t, 8> total{};
    size_t host_bytes = 0; // HOST_GATHER tensors, not counted against any device
    int    n_devices  = 0;
};

DeviceClassTotals compute_placement(const GgufModel & model, int n_devices = 3, int layers_per_device = 16);
void print_placement_table(const DeviceClassTotals & totals, FILE * out = stdout);

// The per-layer, per-expert device pointer table design 9.3 says the MoE
// kernels index by (kind, layer, expert) -- see tools/hot-expert/m1/hipfire_src/*.hip
// for the indexed-batched launch shape this feeds. In --plan mode these
// pointers are into the mmap (host); in --upload mode (upload_placement(),
// below) they would be into the per-device allocation, but that path is
// never run here.
struct ExpertPtrTable {
    // expert_ptrs[kind][layer][expert]; kind 0=gate 1=up 2=down.
    std::vector<std::vector<std::vector<const void *>>> expert_ptrs;
    void build(const GgufModel & model);
};

// Actually hipMallocs and hipMemcpys every tensor to the device
// device_for_tensor() assigns it to, in the file's own format (no
// conversion), and prints per-device VRAM used and the wall-clock load
// time. THIS INITIALISES THE HIP RUNTIME AND TOUCHES EVERY GPU IT TARGETS.
// It is compiled (the task's hard rule allows building HIP code without
// --device) but this repository never calls it: franken_load.cpp requires
// both the --upload flag AND FRANKEN_ALLOW_UPLOAD=1 in the environment
// before it will, and neither was set for anything run while producing this
// directory's deliverable.
void upload_placement(const GgufModel & model, int n_devices = 3, int layers_per_device = 16);

} // namespace franken
