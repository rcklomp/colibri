#!/bin/bash
# isa.sh <decode dir> <out.s> -- device-only gfx1100 assembly of decode_gpu.hip (compile only, no GPU, no binary)
set -eu
D=$1; OUT=$2
cd "$D"
L=/home/ronald/src/llama-glm53
nice -n 10 docker run --rm --user $(id -u):$(id -g) -v /home/ronald:/home/ronald -w "$D" rocm/dev-ubuntu-24.04:7.14.0-full \
  hipcc --offload-arch=gfx1100 -O3 -std=c++17 -Wall -Wno-unused-function -I$L/ggml/include -I$L/ggml/src -I$L/include -I.. \
  -DFRANKEN_LLAMA_SO_DEFAULT="\"$L/build-hip/bin/libllama.so\"" --cuda-device-only -S ${EXTRA:-decode_gpu.hip} -o "$OUT"
