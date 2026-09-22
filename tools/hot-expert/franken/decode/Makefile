# Makefile for tools/hot-expert/franken/decode -- L0 step 2's one-card decode
# loop (L0-STEP2-BRIEF-2026-09-22.md deliverable B, design rev 8 section 9.5
# step 2).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image, the same
# recipe as ../Makefile and ../../m1/Makefile. NO --device flags are passed to
# docker, so the container cannot see the GPUs; hipcc compiling for
# --offload-arch=gfx1100 does not need them (offline codegen), and
# `__builtin_amdgcn_sdot4` is not used anywhere here -- it does not compile for
# gfx1100 on this toolchain (see ../../m1/Makefile's note).
#
# `make` builds and runs nothing. Two binaries:
#
#   franken_decode_cpu   host C++ only, links no HIP at all (decode_nogpu.cpp
#                        stands in for the GPU backend). This is what `--cpu`
#                        runs, and `ldd` proves it cannot touch a card.
#   franken_decode       the real thing: the HIP kernels plus the same graph.
#                        Running it uploads ~22 GB to device 0 and MUST be
#                        done under the rig lock with the gateway stopped.
#
# `make resources` prints the per-kernel VGPR/SGPR/LDS/spill report for every
# kernel in decode_gpu.hip. Compile-only, no binary, no GPU.

DOCKER_IMAGE := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH := gfx1100
DIR := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(DOCKER_IMAGE)

LLAMA_DIR    := /home/ronald/src/llama-glm53
GGML_INC     := $(LLAMA_DIR)/ggml/include
GGML_SRC_INC := $(LLAMA_DIR)/ggml/src          # ggml-common.h: the block structs and decode tables
GGML_LIB_DIR := $(LLAMA_DIR)/build-hip/bin

# ../gguf_model.cpp and ../ple.cpp are L0 step 1's loader and CPU gather,
# reused unchanged (record §L0-STEP1). ../placement.cpp is NOT built here:
# its upload_placement() targets three cards, and step 2 is one card by way
# of DecodeModel::place_mat -> Backend::place.
COMMON_SRC := ../gguf_model.cpp ../ple.cpp \
              decode_model.cpp decode_graph.cpp decode_oracle.cpp \
              decode_cpu.cpp franken_decode.cpp
HDRS := decode_shapes.h decode_quant.h decode_backend.h decode_model.h \
        decode_graph.h decode_oracle.h ../gguf_model.h ../ple.h \
        ../../m1/m1_native_decode.h

CPU_BIN := franken_decode_cpu
GPU_BIN := franken_decode
GPU_SRC := decode_gpu.hip

INCS     := -I$(GGML_INC) -I$(GGML_SRC_INC) -I..
# No -lggml-hip on either binary: no ggml backend is initialised in this
# directory, the kernels are ours, and the CPU binary must not gain a HIP
# dependency through the back door.
LDLIBS   := -L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(GGML_LIB_DIR)

.PHONY: all clean resources cpu gpu ldd-check moe-gather-test
all: $(CPU_BIN) $(GPU_BIN)

cpu: $(CPU_BIN)
$(CPU_BIN): $(COMMON_SRC) decode_nogpu.cpp $(HDRS)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall -Wno-unused-function -DFRANKEN_NO_HIP $(INCS) \
		$(COMMON_SRC) decode_nogpu.cpp -o $(CPU_BIN) $(LDLIBS) -lpthread

gpu: $(GPU_BIN)
$(GPU_BIN): $(COMMON_SRC) $(GPU_SRC) $(HDRS)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 -Wall -Wno-unused-function $(INCS) \
		$(COMMON_SRC) $(GPU_SRC) -o $(GPU_BIN) $(LDLIBS) -lpthread

# Per-kernel VGPR/SGPR/LDS/spill report. Compile-only (-c to /dev/null), so it
# produces no binary and touches nothing; the remarks go to stderr.
resources:
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 \
		-Rpass-analysis=kernel-resource-usage $(INCS) \
		-c $(GPU_SRC) -o /dev/null

# The one claim worth verifying mechanically: the CPU binary links no HIP.
ldd-check: $(CPU_BIN)
	@ldd $(CPU_BIN) | grep -qi 'amdhip\|hsa-runtime' \
		&& { echo "FAIL: $(CPU_BIN) links a HIP runtime"; exit 1; } \
		|| echo "OK: $(CPU_BIN) links no HIP runtime"

# Does the expert row-gather compute the same BITS as the per-assignment
# kernels? Host only, no HIP, seconds. See moe_gather_test.cpp.
moe-gather-test: franken_moe_gather_test
	./franken_moe_gather_test

franken_moe_gather_test: moe_gather_test.cpp $(HDRS)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall $(INCS) moe_gather_test.cpp \
		-o franken_moe_gather_test $(LDLIBS)

clean:
	rm -f $(CPU_BIN) $(GPU_BIN) franken_moe_gather_test
