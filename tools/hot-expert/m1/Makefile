# Makefile for the M5 launch/sync-floor microbenchmark
# (tools/hot-expert/M1-M5-BRIEF-2026-09-22.md, section M5).
#
# Builds inside the rocm/dev-ubuntu-24.04:7.14.0-full docker image via hipcc.
# No --device flags are passed to docker, so the container cannot see the
# GPUs; hipcc compiling for --offload-arch=gfx1100 does not need them either
# (offline codegen). `make all` only builds; it does not run anything.

DOCKER_IMAGE := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH := gfx1100
SRC := m5_bench.hip
BIN := m5_bench
DIR := $(shell pwd)

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

.PHONY: all clean
all:
	docker run --rm --user $(shell id -u):$(shell id -g) -v /home/ronald:/home/ronald -w $(DIR) $(DOCKER_IMAGE) \
		hipcc --offload-arch=$(ARCH) -O3 -std=c++17 $(DOTFLAG) $(SRC) -o $(BIN)

clean:
	rm -f $(BIN)
