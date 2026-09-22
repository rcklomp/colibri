# Makefile for tools/hot-expert/franken -- L0 step 1's loader
# (FRANKEN-ENGINE-DESIGN-2026-09-22.md, section 9).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image via
# hipcc, the same recipe as tools/hot-expert/m1/Makefile. No --device flags
# are passed to docker, so the container cannot see the GPUs; hipcc
# compiling for --offload-arch=gfx1100 does not need them either (offline
# codegen). `make` only builds; it does not run anything, and nothing in
# this directory's own code calls a HIP runtime function unless
# franken_load is run with --upload AND FRANKEN_ALLOW_UPLOAD=1 is set in
# the environment (franken_load.cpp, placement.cpp's upload_placement()).

DOCKER_IMAGE := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH := gfx1100
DIR := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(DOCKER_IMAGE)

LLAMA_DIR    := /home/ronald/src/llama-glm53
GGML_INC     := $(LLAMA_DIR)/ggml/include
GGML_LIB_DIR := $(LLAMA_DIR)/build-hip/bin

SRCS := gguf_model.cpp placement.cpp ple.cpp franken_load.cpp
HDRS := gguf_model.h placement.h ple.h
BIN  := franken_load

CXXFLAGS := --offload-arch=$(ARCH) -O2 -std=c++17 -Wall -I$(GGML_INC)
LDFLAGS  := -L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$(GGML_LIB_DIR)

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(DOCKER_RUN) hipcc $(CXXFLAGS) $(SRCS) -o $(BIN) $(LDFLAGS)

clean:
	rm -f $(BIN)
