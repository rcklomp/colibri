# Makefile for tools/hot-expert/franken/decode -- L0 step 2's one-card decode
# loop (L0-STEP2-BRIEF-2026-09-22.md deliverable B, design rev 8 section 9.5
# step 2).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image, the same
# recipe as ../Makefile and ../../m1/Makefile. NO --device flags are passed to
# docker, so the container cannot see the GPUs; hipcc compiling for
# --offload-arch=gfx1100 does not need them (offline codegen).
#
# ON THE INT8 DOT PRODUCT, since ../../m1/Makefile's note has been read as
# "gfx1100 has none": `__builtin_amdgcn_sdot4` indeed does not compile here
# (it wants the `dot1-insts` feature), but `__builtin_amdgcn_sudot4` DOES,
# and emits `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` -- the real packed int8 dot.
# `__builtin_amdgcn_udot4`, `__builtin_amdgcn_fdot2` and inline
# `v_dot4_i32_iu8` / `v_dot4_i32_i8` also compile. decode_gpu.hip's
# k_gemm_lds_i8 (--gemm-lds 2) uses sudot4; `make resources` shows 128 of
# them in that kernel.
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
              decode_cpu.cpp franken_serve.cpp franken_decode.cpp \
              ds4_model.cpp ds4_graph.cpp ds4_cpu.cpp ds4_adapt.cpp \
              glm5_model.cpp glm5_graph.cpp glm5_cpu.cpp glm5_adapt.cpp
HDRS := decode_shapes.h decode_quant.h decode_backend.h decode_model.h \
        decode_graph.h decode_oracle.h franken_serve.h ../gguf_model.h ../ple.h \
        ../../m1/m1_native_decode.h \
        ds4_shapes.h ds4_quant.h ds4_ops.h ds4_model.h ds4_graph.h ds4_gpu.inc ds4_adapt.h \
        glm5_shapes.h glm5_ops.h glm5_model.h glm5_graph.h glm5_adapt.h glm5_gpu.inc \
        ds4_serve.h

CPU_BIN := franken_decode_cpu
GPU_BIN := franken_decode
GPU_SRC := decode_gpu.hip

# franken_decode_ds4 (M1): DeepSeek-V4-Flash behind the gateway
# (GATEWAY-PROTOCOL.md), its own main() (franken_decode_ds4_main.cpp) so the
# process is unconditionally DeepSeek-V4 with no argv --model to sniff an
# architecture from (the gateway launches `[binary, <cap>]` with the model
# named by SNAP/FRANKEN_GGUF in the environment, not argv -- see that file's
# header comment). Reuses COMMON_SRC wholesale minus franken_decode.cpp (whose
# own main() would conflict with this one) plus ds4_serve.cpp: the Qwen code
# it drags in (DecodeModel/DecodeRunner/franken_serve.cpp) is linked but never
# called, which costs compile time and binary size and changes no behaviour --
# COMMON_SRC is the already-proven-working set, and trimming it to a minimal
# DS4-only source list is a separate, lower-priority cleanup, not a
# correctness question.
DS4_SRC := $(filter-out franken_decode.cpp,$(COMMON_SRC)) ds4_serve.cpp franken_decode_ds4_main.cpp
DS4_HDRS := $(HDRS) ds4_serve.h
DS4_CPU_BIN := franken_decode_ds4_cpu
DS4_GPU_BIN := franken_decode_ds4

# llama.h is included for its TYPES only: franken_serve.cpp dlopens libllama at
# run time rather than linking it, because `ldd libggml.so.0` lists
# libggml-hip.so.0 and linking -lllama would put a HIP runtime in
# franken_decode_cpu's dependency list -- exactly what `make ldd-check` exists
# to forbid. -DFRANKEN_LLAMA_SO_DEFAULT is where the dlopen looks by default.
LLAMA_INC := $(LLAMA_DIR)/include
INCS     := -I$(GGML_INC) -I$(GGML_SRC_INC) -I$(LLAMA_INC) -I.. \
            -DFRANKEN_LLAMA_SO_DEFAULT='"$(GGML_LIB_DIR)/libllama.so"'
# No -lggml-hip on either binary: no ggml backend is initialised in this
# directory, the kernels are ours, and the CPU binary must not gain a HIP
# dependency through the back door.
LDLIBS   := -L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu -Wl,-rpath,$(GGML_LIB_DIR) -ldl

.PHONY: all clean resources cpu gpu ldd-check moe-gather-test ds4-emul ds4-cpu ds4-gpu ds4-ldd-check
all: $(CPU_BIN) $(GPU_BIN)

cpu: $(CPU_BIN)
$(CPU_BIN): $(COMMON_SRC) decode_nogpu.cpp $(HDRS)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall -Wno-unused-function -DFRANKEN_NO_HIP $(INCS) \
		$(COMMON_SRC) decode_nogpu.cpp -o $(CPU_BIN) $(LDLIBS) -lpthread

gpu: $(GPU_BIN)
$(GPU_BIN): $(COMMON_SRC) $(GPU_SRC) $(HDRS)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 -Wall -Wno-unused-function $(INCS) \
		$(COMMON_SRC) $(GPU_SRC) -o $(GPU_BIN) $(LDLIBS) -lpthread

# franken_decode_ds4[_cpu] (M1): same recipe, DS4_SRC/DS4_HDRS/DS4_*_BIN above.
ds4-cpu: $(DS4_CPU_BIN)
$(DS4_CPU_BIN): $(DS4_SRC) decode_nogpu.cpp $(DS4_HDRS)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall -Wno-unused-function -DFRANKEN_NO_HIP $(INCS) \
		$(DS4_SRC) decode_nogpu.cpp -o $(DS4_CPU_BIN) $(LDLIBS) -lpthread

ds4-gpu: $(DS4_GPU_BIN)
$(DS4_GPU_BIN): $(DS4_SRC) $(GPU_SRC) $(DS4_HDRS)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 -Wall -Wno-unused-function $(INCS) \
		$(DS4_SRC) $(GPU_SRC) -o $(DS4_GPU_BIN) $(LDLIBS) -lpthread

ds4-ldd-check: $(DS4_CPU_BIN)
	@ldd $(DS4_CPU_BIN) | grep -qi 'amdhip\|hsa-runtime' \
		&& { echo "FAIL: $(DS4_CPU_BIN) links a HIP runtime"; exit 1; } \
		|| echo "OK: $(DS4_CPU_BIN) links no HIP runtime"

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

# ds4_quant.h's six decoders against ggml's own dequantisers, bit for bit
# (ds4_quant_emul.cpp). Host only, no HIP, seconds. DS4_MODEL=<shard> adds the
# real-rows set (a few hundred KB read through the mmap); GLM_MODEL=<shard> the
# GLM-5.3 formats real rows (Q8_0, IQ3_S, IQ4_XS, Q6_K; GLM5.md section 4).
ds4-emul: franken_ds4_quant_emul
	./franken_ds4_quant_emul $(if $(DS4_MODEL),--model $(DS4_MODEL),) $(if $(GLM_MODEL),--glm-model $(GLM_MODEL),)

franken_ds4_quant_emul: ds4_quant_emul.cpp ../gguf_model.cpp $(HDRS)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall $(INCS) ds4_quant_emul.cpp ../gguf_model.cpp \
		-o franken_ds4_quant_emul $(LDLIBS)

clean:
	rm -f $(CPU_BIN) $(GPU_BIN) $(DS4_CPU_BIN) $(DS4_GPU_BIN) franken_moe_gather_test franken_ds4_quant_emul
