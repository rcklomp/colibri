# Makefile for ~/src/cpulane -- builds and runs NOTHING. Two programs:
#   bench_cpu_gemv   part A, host g++, CPU only (links no HIP)
#   bench_h2d_cpu    part B, hipcc inside docker (same recipe as franken-engine's decode/Makefile); STARTING it
#                    initialises the GPUs: only run it through run_cpulane.sh under the rig lock.
#
# The ggml pieces (three objects) are compiled with docker's gcc, so that part A (host) and part B (docker)
# link the very same machine code for the dot kernels. ggml sources are read from LLAMA_DIR, never written.

LLAMA_DIR := /home/ronald/src/llama-glm53
IMG       := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH      := gfx1100
DIR       := $(shell pwd)
UID       := $(shell id -u)
GID       := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(IMG)
NICE      := nice -n 19

GGML_REV  := $(shell git -C $(LLAMA_DIR) rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
GGML_INC  := -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src -I$(LLAMA_DIR)/ggml/src/ggml-cpu
GGML_CFLAGS := -O3 -mavx2 -mfma -mf16c -mtune=znver2 -DNDEBUG -D_GNU_SOURCE -DGGML_AVX2 -DGGML_FMA -DGGML_F16C \
               -DGGML_USE_CPU -std=gnu11 -fno-math-errno
OBJS := obj/ggml_quants.o obj/ggml_x86_quants.o obj/ggml_stubs.o

.PHONY: all a b clean
all: bench_cpu_gemv bench_h2d_cpu
a: bench_cpu_gemv
b: bench_h2d_cpu

obj/ggml_quants.o: $(LLAMA_DIR)/ggml/src/ggml-quants.c
	@mkdir -p obj
	$(NICE) $(DOCKER_RUN) $(NICE) gcc $(GGML_CFLAGS) $(GGML_INC) -c $< -o $@

obj/ggml_x86_quants.o: $(LLAMA_DIR)/ggml/src/ggml-cpu/arch/x86/quants.c
	@mkdir -p obj
	$(NICE) $(DOCKER_RUN) $(NICE) gcc $(GGML_CFLAGS) $(GGML_INC) -c $< -o $@

obj/ggml_stubs.o: ggml_stubs.c
	@mkdir -p obj
	$(NICE) $(DOCKER_RUN) $(NICE) gcc $(GGML_CFLAGS) $(GGML_INC) -c $< -o $@

bench_cpu_gemv: bench_cpu_gemv.cpp cpu_gemv.h $(OBJS)
	$(NICE) g++ -O3 -mavx2 -mfma -mf16c -std=c++17 -Wall -Wno-unused-function -DGGML_REV='"$(GGML_REV)"' \
		bench_cpu_gemv.cpp $(OBJS) -o $@ -lpthread -lm

# two steps: hipcc's `-x hip` would otherwise be applied to the .o files on the same command line
obj/bench_h2d_cpu.o: bench_h2d_cpu.hip cpu_gemv.h
	@mkdir -p obj
	$(NICE) $(DOCKER_RUN) $(NICE) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 -mavx2 -mfma -mf16c -Wall -Wno-unused-function \
		-DGGML_REV='"$(GGML_REV)"' -c bench_h2d_cpu.hip -o $@

bench_h2d_cpu: obj/bench_h2d_cpu.o $(OBJS)
	$(NICE) $(DOCKER_RUN) $(NICE) hipcc --offload-arch=$(ARCH) -O3 obj/bench_h2d_cpu.o $(OBJS) -o $@ -lpthread -lm

clean:
	rm -rf obj bench_cpu_gemv bench_h2d_cpu
