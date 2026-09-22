// tools/hot-expert/franken/placement.cpp -- see placement.h for the design.

#include "placement.h"

#include <chrono>
#include <cstdlib>
#include <stdexcept>

#include <hip/hip_runtime.h>

namespace franken {

namespace {

bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

// Parses the layer index out of a "blk.<N>.<...>" tensor name. Returns
// false (leaving il untouched) for a non-layer tensor.
bool layer_of(const std::string & name, int & il) {
    if (name.rfind("blk.", 0) != 0) return false;
    size_t p = 4;
    size_t q = name.find('.', p);
    if (q == std::string::npos || q == p) return false;
    for (size_t i = p; i < q; ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    il = std::atoi(name.substr(p, q - p).c_str());
    return true;
}

} // namespace

const char * tensor_class_name(TensorClass c) {
    switch (c) {
        case TensorClass::TRUNK:          return "trunk";
        case TensorClass::ROUTED_EXPERTS: return "routed_experts";
        case TensorClass::SHARED_EXPERTS: return "shared_experts";
        case TensorClass::INDEXER:        return "indexer";
        case TensorClass::HC:             return "hc";
        case TensorClass::NORMS:          return "norms";
        case TensorClass::HOST_GATHER:    return "host_gather";
        default:                          return "?";
    }
}

TensorClass classify_tensor(const std::string & name) {
    // design 9.1's rule, applied to the one tensor in this model it applies to.
    if (name == "per_layer_token_embd.weight") return TensorClass::HOST_GATHER;

    if (contains(name, "ffn_gate_exps") || contains(name, "ffn_up_exps") || contains(name, "ffn_down_exps")) {
        return TensorClass::ROUTED_EXPERTS;
    }
    if (contains(name, "shexp")) return TensorClass::SHARED_EXPERTS; // ffn_{gate,up,down}_shexp, ffn_gate_inp_shexp
    if (contains(name, "indexer")) return TensorClass::INDEXER;
    if (contains(name, "hc_") || name.rfind("output_hc_", 0) == 0) return TensorClass::HC;
    if (contains(name, "norm")) return TensorClass::NORMS; // attn_{q,k}_norm, ssm_norm, ple_norm_*
    return TensorClass::TRUNK;
}

int device_for_tensor(const TensorInfo & t, int n_devices, int layers_per_device) {
    if (classify_tensor(t.name) == TensorClass::HOST_GATHER) return -1;

    int il;
    if (layer_of(t.name, il)) {
        int dev = il / layers_per_device;
        if (dev >= n_devices) dev = n_devices - 1;
        return dev;
    }
    if (t.name == "token_embd.weight") return 0; // design 9.2: "Card 0: embed ... -> layers 0-15"
    return n_devices - 1;                        // output.weight, output_hc_{norm,down,up}: after the last layer
}

DeviceClassTotals compute_placement(const GgufModel & model, int n_devices, int layers_per_device) {
    if (n_devices > 8) throw std::runtime_error("compute_placement: n_devices > 8 not supported by this table");

    DeviceClassTotals out;
    out.n_devices = n_devices;
    for (const auto & t : model.tensors()) {
        TensorClass cls = classify_tensor(t.name);
        int dev = device_for_tensor(t, n_devices, layers_per_device);
        if (dev < 0) {
            out.host_bytes += t.nbytes;
            continue;
        }
        out.bytes[dev][(size_t) cls] += t.nbytes;
        out.total[dev] += t.nbytes;
    }
    return out;
}

void print_placement_table(const DeviceClassTotals & t, FILE * out) {
    std::fprintf(out, "%-16s", "class (GB)");
    for (int d = 0; d < t.n_devices; ++d) std::fprintf(out, "  %10s", ("dev" + std::to_string(d)).c_str());
    std::fprintf(out, "\n");

    for (size_t c = 0; c < (size_t) TensorClass::N_CLASSES; ++c) {
        if (c == (size_t) TensorClass::HOST_GATHER) continue; // reported separately, it is on no device
        std::fprintf(out, "%-16s", tensor_class_name((TensorClass) c));
        for (int d = 0; d < t.n_devices; ++d) {
            std::fprintf(out, "  %10.3f", t.bytes[d][c] / 1e9);
        }
        std::fprintf(out, "\n");
    }

    std::fprintf(out, "%-16s", "TOTAL");
    for (int d = 0; d < t.n_devices; ++d) std::fprintf(out, "  %10.3f", t.total[d] / 1e9);
    std::fprintf(out, "\n");

    std::fprintf(out, "host RAM (%s, pinned, never on a device): %.3f GB\n",
                 tensor_class_name(TensorClass::HOST_GATHER), t.host_bytes / 1e9);
}

void ExpertPtrTable::build(const GgufModel & model) {
    const auto & hp = model.hparams();
    static const char * suffixes[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };

    expert_ptrs.assign(3, {});
    for (auto & kind : expert_ptrs) kind.assign(hp.block_count, {});

    for (int k = 0; k < 3; ++k) {
        for (uint32_t il = 0; il < hp.block_count; ++il) {
            const TensorInfo * t = model.find_layer(il, suffixes[k]);
            expert_ptrs[k][il].assign(hp.expert_count, nullptr);
            if (!t) continue; // defensive: every layer has these three in this model, but don't assume it elsewhere
            size_t bytes_per_expert = t->slice_bytes(t->ne2()); // ne2 = expert count, the tensor's slowest dim
            for (uint32_t e = 0; e < hp.expert_count; ++e) {
                expert_ptrs[k][il][e] = t->data + (size_t) e * bytes_per_expert;
            }
        }
    }
}

void upload_placement(const GgufModel & model, int n_devices, int layers_per_device) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<size_t> dev_bytes(n_devices, 0);

    for (int dev = 0; dev < n_devices; ++dev) {
        hipError_t rc = hipSetDevice(dev);
        if (rc != hipSuccess) throw std::runtime_error("hipSetDevice(" + std::to_string(dev) + ") failed");

        for (const auto & t : model.tensors()) {
            if (device_for_tensor(t, n_devices, layers_per_device) != dev) continue;

            void * d_ptr = nullptr;
            rc = hipMalloc(&d_ptr, t.nbytes);
            if (rc != hipSuccess) throw std::runtime_error("hipMalloc failed for " + t.name);

            rc = hipMemcpy(d_ptr, t.data, t.nbytes, hipMemcpyHostToDevice);
            if (rc != hipSuccess) throw std::runtime_error("hipMemcpy failed for " + t.name);

            dev_bytes[dev] += t.nbytes;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    for (int dev = 0; dev < n_devices; ++dev) {
        std::fprintf(stdout, "dev%d: %.3f GB uploaded\n", dev, dev_bytes[dev] / 1e9);
    }
    std::fprintf(stdout, "load time: %.2f s\n", secs);
}

} // namespace franken
