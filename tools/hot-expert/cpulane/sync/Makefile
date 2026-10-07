# Makefile for ~/src/cpulane/sync -- builds and runs NOTHING that touches a GPU.
#   make            bench_sync       hipcc inside docker (same recipe as ../Makefile / franken-engine's decode/Makefile);
#                                    STARTING it initialises the GPUs: only run it through run_sync.sh under the rig lock.
#   make stub       bench_sync_stub  host-only build against stub_hip.h (fake HIP, no GPU): exercises the host logic,
#                                    the statistics, the abort path. Safe anywhere; it spins a few cores for ~1 min.
#   make asm        obj/bench_sync_gfx1100.s (device assembly, compile only)
#   make clean

IMG       := rocm/dev-ubuntu-24.04:7.14.0-full
ARCH      := gfx1100
DIR       := $(shell pwd)
UID       := $(shell id -u)
GID       := $(shell id -g)
DOCKER_RUN := docker run --rm --user $(UID):$(GID) -v /home/ronald:/home/ronald -w $(DIR) $(IMG)
NICE      := nice -n 19

.PHONY: all stub asm clean
all: bench_sync

bench_sync: bench_sync.hip sync_proto.h
	$(NICE) $(DOCKER_RUN) $(NICE) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 -Wall -pthread bench_sync.hip -o $@

stub: bench_sync_stub
bench_sync_stub: bench_sync.hip sync_proto.h stub_hip.h
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -DSYNC_STUB -x c++ bench_sync.hip -o $@ -pthread

# device assembly of all kernels (compile only), to review the memory-model instructions the compiler emits
asm: obj/bench_sync_gfx1100.s
obj/bench_sync_gfx1100.s: bench_sync.hip sync_proto.h
	@mkdir -p obj
	$(NICE) $(DOCKER_RUN) $(NICE) hipcc --offload-arch=$(ARCH) -O3 -std=c++17 --cuda-device-only -S bench_sync.hip -o $@

clean:
	rm -rf bench_sync bench_sync_stub obj
