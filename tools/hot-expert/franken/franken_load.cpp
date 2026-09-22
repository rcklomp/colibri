// tools/hot-expert/franken/franken_load.cpp
//
// L0 step 1's CLI (design doc section 9.5, step 1's loader). Three modes:
//
//   --plan       CPU only, safe to run: prints the layer-range placement
//                table (placement.h) and builds the per-expert pointer
//                table over the mmap.
//   --ple-check <id> [id ...]
//                CPU only, safe to run: the CPU-side PLE gather (ple.h) for
//                a short token-id list, printed in a stable text format.
//   --upload     TOUCHES EVERY GPU (hipMalloc/hipMemcpy per tensor). Refused
//                unless FRANKEN_ALLOW_UPLOAD=1 is set in the environment --
//                this build must never pass that, and nothing in this
//                directory's Makefile or its own use of the binary does.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_model.h"
#include "ple.h"
#include "placement.h"

using namespace franken;

namespace {

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model <path-to-any-shard.gguf> --plan\n"
        "       %s --model <path> --ple-check <token_id> [token_id ...]\n"
        "       %s --model <path> --upload      (touches every GPU -- requires FRANKEN_ALLOW_UPLOAD=1)\n",
        prog, prog, prog);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string mode;
    std::vector<int32_t> ids;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--plan" || a == "--upload" || a == "--ple-check") {
            mode = a;
        } else if (mode == "--ple-check") {
            ids.push_back((int32_t) std::atoll(a.c_str()));
        } else {
            std::fprintf(stderr, "unrecognised argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || mode.empty()) {
        usage(argv[0]);
        return 2;
    }

    try {
        auto model = GgufModel::open(model_path);
        const auto & hp = model->hparams();

        std::fprintf(stdout,
            "model: arch=%s block_count=%u embedding_length=%u expert_used_count=%u/%u "
            "shards=%zu tensors=%zu ple_layers=%zu ple_n_heads=%u ple_head_dim=%u\n",
            hp.arch.c_str(), hp.block_count, hp.embedding_length,
            hp.expert_used_count, hp.expert_count,
            model->num_shards(), model->tensors().size(),
            hp.ple_layers.size(), hp.ple_n_heads(), hp.n_embd_per_layer);

        if (mode == "--plan") {
            auto totals = compute_placement(*model, 3, 16);
            print_placement_table(totals);

            ExpertPtrTable ept;
            ept.build(*model);
            std::fprintf(stdout, "expert pointer table: %zu kinds x %u layers x %u experts (gate,up,down)\n",
                         ept.expert_ptrs.size(), hp.block_count, hp.expert_count);

        } else if (mode == "--upload") {
            if (!std::getenv("FRANKEN_ALLOW_UPLOAD")) {
                std::fprintf(stderr,
                    "--upload touches every GPU (hipMalloc/hipMemcpy); refusing because "
                    "FRANKEN_ALLOW_UPLOAD is not set in the environment.\n");
                return 3;
            }
            upload_placement(*model, 3, 16);

        } else if (mode == "--ple-check") {
            if (ids.empty()) {
                std::fprintf(stderr, "--ple-check needs at least one token id\n");
                return 2;
            }
            auto r = ple_gather(*model, ids);
            print_ple_check(hp, ids, r, stdout);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
