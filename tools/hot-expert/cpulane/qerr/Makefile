# Makefile for ~/src/cpulane/qerr -- CPU only, host gcc/g++, builds ggml's quantisation + gguf pieces from
# ~/src/llama-glm53 (read only) next to qerr.cpp. Builds and runs nothing else; no GPU code is involved.
#
#   make            # builds ./qerr
#   make clean
#
# Same flags as ../Makefile (cpulane part A): -O3 -mavx2 -mfma -mf16c -mtune=znver2, GGML_AVX2/FMA/F16C, so the
# ggml_vec_dot_*_q8_K kernels are the very ones the cpulane benchmark measures (ggml-cpu/arch/x86/quants.c).
# Objects: ggml.c (type traits -> to_float), ggml-quants.c (dequantize_row_*, quantize_row_q8_K_ref),
# ggml-cpu/arch/x86/quants.c (ggml_vec_dot_*_q8_K, AVX2), gguf.cpp (reader), ggml-threading.cpp (ggml.c needs it),
# and qerr_stubs.c (the 3 lookup tables ggml-cpu.c would define, plus init).

LLAMA_DIR := /home/ronald/src/llama-glm53
GGML_REV  := $(shell git -C $(LLAMA_DIR) rev-parse --short=9 HEAD 2>/dev/null || echo unknown)
GGML_INC  := -I$(LLAMA_DIR)/ggml/include -I$(LLAMA_DIR)/ggml/src -I$(LLAMA_DIR)/ggml/src/ggml-cpu
DEFS      := -DNDEBUG -D_GNU_SOURCE -DGGML_AVX2 -DGGML_FMA -DGGML_F16C -DGGML_USE_CPU \
             -DGGML_VERSION='"qerr"' -DGGML_COMMIT='"$(GGML_REV)"'
ARCHF     := -O3 -mavx2 -mfma -mf16c -mtune=znver2 -fno-math-errno
CFLAGS    := $(ARCHF) $(DEFS) -std=gnu11
CXXFLAGS  := $(ARCHF) $(DEFS) -std=c++17
OBJS      := obj/ggml.o obj/ggml_quants.o obj/ggml_x86_quants.o obj/gguf.o obj/ggml_threading.o obj/qerr_stubs.o

.PHONY: all clean
all: qerr

obj/ggml.o: $(LLAMA_DIR)/ggml/src/ggml.c
	@mkdir -p obj
	gcc $(CFLAGS) $(GGML_INC) -c $< -o $@
obj/ggml_quants.o: $(LLAMA_DIR)/ggml/src/ggml-quants.c
	@mkdir -p obj
	gcc $(CFLAGS) $(GGML_INC) -c $< -o $@
obj/ggml_x86_quants.o: $(LLAMA_DIR)/ggml/src/ggml-cpu/arch/x86/quants.c
	@mkdir -p obj
	gcc $(CFLAGS) $(GGML_INC) -c $< -o $@
obj/gguf.o: $(LLAMA_DIR)/ggml/src/gguf.cpp
	@mkdir -p obj
	g++ $(CXXFLAGS) $(GGML_INC) -c $< -o $@
obj/ggml_threading.o: $(LLAMA_DIR)/ggml/src/ggml-threading.cpp
	@mkdir -p obj
	g++ $(CXXFLAGS) $(GGML_INC) -c $< -o $@
obj/qerr_stubs.o: qerr_stubs.c
	@mkdir -p obj
	gcc $(CFLAGS) $(GGML_INC) -c $< -o $@

qerr: qerr.cpp $(OBJS)
	g++ -O3 -mavx2 -mfma -mf16c -std=c++17 -Wall -Wno-unused-function $(GGML_INC) -DGGML_REV='"$(GGML_REV)"' \
		qerr.cpp $(OBJS) -o $@ -lpthread -lm

clean:
	rm -rf obj qerr
