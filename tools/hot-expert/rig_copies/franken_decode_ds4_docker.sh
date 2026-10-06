#!/bin/bash
# franken_decode_docker.sh -- run the Franken engine inside the ROCm 7.14 image it was
# built in (the host's ROCm is 6.2: libhipblas.so.3 / librocblas.so.5 that libllama's
# tokenizer pulls in are not on the host). stdin/stdout pass through `docker run -i`
# unchanged, so the gateway's line protocol works as with a native binary.
# Usage: same argv as franken_decode; env SNAP SERVE SERVE_BATCH NGEN KV_SLOTS FRANKEN_* pass through.
BIN=${FRANKEN_DOCKER_BIN:-/home/ronald/bench/franken_bin/franken_decode_ds4}
ENVS=()
for v in SNAP SERVE SERVE_BATCH NGEN KV_SLOTS COLI_REQ_LOG FRANKEN_GGUF FRANKEN_CTX FRANKEN_CHUNK FRANKEN_GEMM_LDS FRANKEN_TOPK_CAND FRANKEN_SNAP_EVERY FRANKEN_SNAP_KEEP FRANKEN_SNAP_TURNS FRANKEN_SNAP_BUDGET_MB FRANKEN_PLE_RESIDENT FRANKEN_PLE_PINNED FRANKEN_BOUNDARY_TOKENS FRANKEN_LOG FRANKEN_DEVICES COLI_PREFIX_PIN COLI_THINK COLI_LEDGER OMP_NUM_THREADS FRANKEN_LAYERS FRANKEN_CPU FRANKEN_THREADS FRANKEN_SEED FRANKEN_STOP_IDS FRANKEN_PLACEMENT FRANKEN_EXPERT_GB FRANKEN_ADAPT FRANKEN_ADAPT_EVERY FRANKEN_ADAPT_MB_PER_TOKEN FRANKEN_ADAPT_HALFLIFE FRANKEN_ADAPT_MARGIN FRANKEN_ADAPT_HYST FRANKEN_ADAPT_VERIFY FRANKEN_MISS_STAGE FRANKEN_HIP_GRAPH FRANKEN_HIP_GRAPH_BUCKET FRANKEN_PREFILL_PIPELINE FRANKEN_STAGED_LOADS FRANKEN_STAGE_WGS FRANKEN_DS4_EXPERT_GATHER FRANKEN_DS4_ATTN_MB; do
  [ -n "${!v+x}" ] && ENVS+=(-e "$v=${!v}")
done
exec docker run --rm -i --name franken_engine --user $(id -u):$(id -g) --group-add 991 \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin "${ENVS[@]}" \
  -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" "$@"
