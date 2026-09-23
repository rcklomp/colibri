# Makefile for tools/hot-expert/franken -- L0 step 1's loader
# (FRANKEN-ENGINE-DESIGN-2026-09-22.md, section 9), and L0 step 2 deliverable
# A, the CPU-only oracle dump tool (L0-STEP2-BRIEF-2026-09-22.md).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image via
# hipcc, the same recipe as tools/hot-expert/m1/Makefile. No --device flags
# are passed to docker, so the container cannot see the GPUs; hipcc
# compiling for --offload-arch=gfx1100 does not need them either (offline
# codegen). `make` only builds; it does not run anything, and nothing in
# this directory's own code calls a HIP runtime function unless
# franken_load is run with --upload AND FRANKEN_ALLOW_UPLOAD=1 is set in
# the environment (franken_load.cpp, placement.cpp's upload_placement()).
# franken_oracle_dump links against libllama (built without any tensors
# ever touching a GPU: it loads with n_gpu_layers=0, and running it in this
# same --device-less image means ggml-hip enumerates no ROCm device, so
# every op lands on the CPU backend).
#
# moe_hist (M2, FRANKEN-ENGINE-DESIGN-2026-09-22.md section 4 row M2) is
# built the same CPU-only way, plus one exception: it declares (moe_hist.cpp
# includes common/fit.h) and links against common_fit_params()/
# common_fit_print(), which are already compiled into the prebuilt
# build-hip/bin/libllama-common.so -- no common/*.cpp is recompiled here.

DOCKER_IMAGE := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH := gfx1100
DIR := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(DOCKER_IMAGE)

LLAMA_DIR    := /home/ronald/src/llama-glm53
LLAMA_INC    := $(LLAMA_DIR)/include
GGML_INC     := $(LLAMA_DIR)/ggml/include
GGML_LIB_DIR := $(LLAMA_DIR)/build-hip/bin

SRCS := gguf_model.cpp placement.cpp ple.cpp franken_load.cpp
HDRS := gguf_model.h placement.h ple.h
BIN  := franken_load

ORACLE_SRC := oracle_dump.cpp
ORACLE_BIN := franken_oracle_dump

MOE_HIST_SRCS := moe_hist.cpp gguf_model.cpp
MOE_HIST_BIN  := moe_hist

CXXFLAGS := --offload-arch=$(ARCH) -O2 -std=c++17 -Wall -I$(GGML_INC)
LDFLAGS  := -L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$(GGML_LIB_DIR)

ORACLE_CXXFLAGS := --offload-arch=$(ARCH) -O2 -std=c++17 -Wall -I$(LLAMA_INC) -I$(GGML_INC)
ORACLE_LDFLAGS  := -L$(GGML_LIB_DIR) -lllama -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$(GGML_LIB_DIR)

MOE_HIST_CXXFLAGS := --offload-arch=$(ARCH) -O2 -std=c++17 -Wall -I$(LLAMA_INC) -I$(GGML_INC) -I$(LLAMA_DIR)/common
MOE_HIST_LDFLAGS  := -L$(GGML_LIB_DIR) -lllama -lllama-common -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$(GGML_LIB_DIR)

.PHONY: all clean
all: $(BIN) $(ORACLE_BIN) $(MOE_HIST_BIN)

$(BIN): $(SRCS) $(HDRS)
	$(DOCKER_RUN) hipcc $(CXXFLAGS) $(SRCS) -o $(BIN) $(LDFLAGS)

$(ORACLE_BIN): $(ORACLE_SRC)
	$(DOCKER_RUN) hipcc $(ORACLE_CXXFLAGS) $(ORACLE_SRC) -o $(ORACLE_BIN) $(ORACLE_LDFLAGS)

$(MOE_HIST_BIN): $(MOE_HIST_SRCS) gguf_model.h
	$(DOCKER_RUN) hipcc $(MOE_HIST_CXXFLAGS) $(MOE_HIST_SRCS) -o $(MOE_HIST_BIN) $(MOE_HIST_LDFLAGS)

clean:
	rm -f $(BIN) $(ORACLE_BIN) $(MOE_HIST_BIN)
