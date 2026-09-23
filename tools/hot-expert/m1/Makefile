# Makefile for the M1/M5 microbenchmark harness
# (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image via hipcc.
# No --device flags are passed to docker, so the container cannot see the
# GPUs; hipcc compiling for --offload-arch=gfx1100 does not need them either
# (offline codegen). `make all` only builds; it does not run anything.

DOCKER_IMAGE := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH := gfx1100
DIR := $(shell pwd)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(DOCKER_IMAGE)

# --- M5: launch/sync-floor microbenchmark ----------------------------------

M5_SRC := m5_bench.hip
M5_BIN := m5_bench

# Dot-product intrinsic: __builtin_amdgcn_sdot4 does NOT build for gfx1100
# with this hipcc/ROCm 7.14 toolchain -- it requires Clang target feature
# "dot1-insts" (Sema-level gate on the builtin), and gfx1100 in this
# toolchain does not recognize dot1-insts (nor dot2..dot12-insts) as a
# valid feature at all ("not a recognized feature for this target");
# forcing it in via -Xclang -target-feature is Sema-only (the backend
# silently drops it with the same message), so the resulting codegen is
# unverified and was not trusted. DOT=scalar (default) is what actually
# builds and is what m5_bench ships with; set DOT=sdot4 only to reproduce
# the failure for further investigation.
DOT ?= scalar
ifeq ($(DOT),scalar)
DOTFLAG := -DM5_DOT_SCALAR=1
else
DOTFLAG :=
endif

# --- M1 candidate 1: llama.cpp's ggml MoE path ------------------------------
#
# Links against the already-built ~/src/llama-glm53/build-hip HIP backend
# (see M1-M5-BRIEF-2026-09-22.md, M1 candidate 1). No __global__ code of our
# own here, so a plain hipcc host-C++ compile+link is enough.

LLAMA_DIR    := /home/ronald/src/llama-glm53
GGML_INC     := $(LLAMA_DIR)/ggml/include
GGML_LIB_DIR := $(LLAMA_DIR)/build-hip/bin

M1_GGML_SRC := m1_ggml.cpp
M1_GGML_BIN := m1_ggml

# --- M1 candidate 2: hipFire's MoE kernels ----------------------------------
#
# The kernel sources are copied verbatim into hipfire_src/ (see that
# directory's README-in-comment headers for the exact source path + commit)
# and #included directly by m1_hipfire.hip, so this is a single hipcc TU.

M1_HIPFIRE_SRC := m1_hipfire.hip
M1_HIPFIRE_BIN := m1_hipfire

# --- M1 candidate 3: the experts in their NATIVE GGUF formats --------------
#
# m1_native.hip: hipFire's kernel SHAPE, but decoding ggml's own IQ3_S
# (gate/up) and IQ4_NL (down) blocks in place -- no requantisation, 19 %
# fewer bytes than the 4.25 bpw container (design rev 7 section 9.3, L0 step 1).
#
# It is a single hipcc TU for the kernels, but it links ggml's CPU libs the
# way m1_ggml does, for two reasons: the weights are quantised on the host
# with ggml_quantize_chunk() and the oracle dequantises with
# ggml_get_type_traits(type)->to_float(), so the bytes the kernel reads are
# byte-identical to what llama.cpp would hold; and ggml-common.h is included
# for block_iq3_s / block_iq4_nl / iq3s_grid / kvalues_iq4nl verbatim rather
# than transcribed. That needs ggml/src on the include path as well as
# ggml/include. Same libs and flags as build-m1-ggml, minus -lggml-hip (no
# ggml backend is initialised here; the kernels are ours).
GGML_SRC_INC := $(LLAMA_DIR)/ggml/src

M1_NATIVE_SRC := m1_native.hip
M1_NATIVE_BIN := m1_native

# 0 = iq3s_grid in __constant__ (default), 1 = staged into LDS per workgroup.
IQ3S_GRID_LDS ?= 0
M1_HIPFIRE_DEPS := m1_common.h \
	hipfire_src/gfx12_weight_cache_policy.inc \
	hipfire_src/gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip \
	hipfire_src/gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded.hip \
	hipfire_src/moe_down_combine_k8_batched.hip \
	hipfire_src/silu_mul.hip

# Phony convenience aliases are named build-<x> so they never collide with
# the binary file targets themselves (a `foo: $(FOO_BIN)` alias where
# FOO_BIN also equals "foo" is a circular self-dependency to make).
.PHONY: all clean build-m5 build-m1-ggml build-m1-hipfire build-m1-native \
        resources-m1-native
all: $(M5_BIN) $(M1_GGML_BIN) $(M1_HIPFIRE_BIN) $(M1_NATIVE_BIN)

build-m5: $(M5_BIN)
$(M5_BIN): $(M5_SRC)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 $(DOTFLAG) $(M5_SRC) -o $(M5_BIN)

build-m1-ggml: $(M1_GGML_BIN)
$(M1_GGML_BIN): $(M1_GGML_SRC) m1_common.h
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 \
		-I$(GGML_INC) \
		$(M1_GGML_SRC) -o $(M1_GGML_BIN) \
		-L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu -lggml-hip \
		-Wl,-rpath,$(GGML_LIB_DIR)

build-m1-hipfire: $(M1_HIPFIRE_BIN)
$(M1_HIPFIRE_BIN): $(M1_HIPFIRE_SRC) $(M1_HIPFIRE_DEPS)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 \
		$(M1_HIPFIRE_SRC) -o $(M1_HIPFIRE_BIN)

build-m1-native: $(M1_NATIVE_BIN)
$(M1_NATIVE_BIN): $(M1_NATIVE_SRC) m1_native_decode.h m1_common.h
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 \
		-DNATIVE_IQ3S_GRID_LDS=$(IQ3S_GRID_LDS) \
		-I$(GGML_INC) -I$(GGML_SRC_INC) \
		$(M1_NATIVE_SRC) -o $(M1_NATIVE_BIN) \
		-L$(GGML_LIB_DIR) -lggml -lggml-base -lggml-cpu \
		-Wl,-rpath,$(GGML_LIB_DIR)

# Per-kernel VGPR/SGPR/LDS/spill report. Compile-only (-c to /dev/null), so
# it touches nothing and produces no binary; the remarks go to stderr.
resources-m1-native:
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 \
		-DNATIVE_IQ3S_GRID_LDS=$(IQ3S_GRID_LDS) \
		-Rpass-analysis=kernel-resource-usage \
		-I$(GGML_INC) -I$(GGML_SRC_INC) \
		-c $(M1_NATIVE_SRC) -o /dev/null

# Host-only emulation of m1_native's two GEMV kernels: it compiles the SAME
# m1_native_decode.h the kernels compile, with host qualifiers, and checks
# every dequantised weight of a row against ggml's own to_float with exact
# float equality, plus lane coverage and a full-wave row dot. No HIP, no GPU,
# no rig lock -- so a session that may not touch the cards can still prove
# the format port. Deliberately NOT part of `all` and NOT a prerequisite of
# $(M1_NATIVE_BIN): it must never stand between a kernel edit and a build of
# the real binary (same rule as check-m3-emul).
M1_NATIVE_EMUL_SRC := m1_native_emul.cpp
M1_NATIVE_EMUL_BIN := m1_native_emul

.PHONY: build-m1-native-emul check-m1-native-emul
build-m1-native-emul: $(M1_NATIVE_EMUL_BIN)
$(M1_NATIVE_EMUL_BIN): $(M1_NATIVE_EMUL_SRC) m1_native_decode.h m1_common.h
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall \
		-I$(GGML_INC) -I$(GGML_SRC_INC) \
		$(M1_NATIVE_EMUL_SRC) -o $(M1_NATIVE_EMUL_BIN) \
		-L$(GGML_LIB_DIR) -lggml-base \
		-Wl,-rpath,$(GGML_LIB_DIR)

check-m1-native-emul: $(M1_NATIVE_EMUL_BIN)
	./$(M1_NATIVE_EMUL_BIN)

clean:
	rm -f $(M5_BIN) $(M1_GGML_BIN) $(M1_HIPFIRE_BIN) $(M1_NATIVE_BIN) $(M1_NATIVE_EMUL_BIN)

# --- M3: QSA attention + Gated DeltaNet decode-step microbenchmark --------
#
# tools/hot-expert/M3-BRIEF-2026-09-22.md. Single hipcc TU, no external
# deps (no ggml/hipFire linkage needed).

M3_SRC := m3_attn.hip
M3_BIN := m3_attn

.PHONY: build-m3
build-m3: $(M3_BIN)
$(M3_BIN): $(M3_SRC)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 $(M3_SRC) -o $(M3_BIN)

# The host-only emulation of m3_attn.hip's three depth kernels
# (m3_emul_check.cpp's own header says what it is and what it is not). No HIP
# and no GPU: an ordinary C++ compile, run straight from `make`, so a session
# that may not touch the cards can still check the index arithmetic, the LDS
# scans and the tail handling. Deliberately NOT part of `all` and not a
# prerequisite of $(M3_BIN): it must never stand between a kernel edit and a
# build of the real binary.
M3_EMUL_SRC := m3_emul_check.cpp
M3_EMUL_BIN := m3_emul_check

.PHONY: build-m3-emul check-m3-emul
build-m3-emul: $(M3_EMUL_BIN)
$(M3_EMUL_BIN): $(M3_EMUL_SRC)
	$(DOCKER_RUN) c++ -O2 -std=c++17 -Wall $(M3_EMUL_SRC) -o $(M3_EMUL_BIN)

check-m3-emul: $(M3_EMUL_BIN)
	./$(M3_EMUL_BIN)

# --- M4: streaming prefill bandwidth under real contention -----------------
#
# tools/hot-expert/FRANKEN-ENGINE-DESIGN-2026-09-22.md section 4 row M4.
# Single hipcc TU, no external deps (host->device copy + a plain read-reduce
# compute kernel, both defined in the file).

M4_SRC := m4_stream.hip
M4_BIN := m4_stream

.PHONY: build-m4
build-m4: $(M4_BIN)
$(M4_BIN): $(M4_SRC)
	$(DOCKER_RUN) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 $(M4_SRC) -o $(M4_BIN)
