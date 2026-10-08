#!/bin/bash
# franken_decode_glm_docker.sh -- run the Franken GLM-5.3-Flash serve engine (glm5_serve.cpp,
# binary franken_dec_glm) inside the ROCm 7.14 image it was built in, exactly
# franken_decode_ds4_docker.sh's own shape (that file's own header comment explains why a
# container: the host's ROCm 6.2 cannot satisfy libllama's tokenizer dependency). stdin/stdout
# pass through `docker run -i` unchanged, so the gateway's line protocol works as with a
# native binary.
# Usage: same argv as franken_dec_glm; env SNAP SERVE SERVE_BATCH NGEN KV_SLOTS FRANKEN_* pass through.
BIN=${FRANKEN_DOCKER_BIN:-/home/ronald/bench/franken_bin/franken_dec_glm}
ENVS=()
for v in SNAP SERVE SERVE_BATCH NGEN KV_SLOTS COLI_REQ_LOG FRANKEN_GGUF FRANKEN_CTX \
         FRANKEN_GLM_CHUNK FRANKEN_GEMM_LDS FRANKEN_TOPK_CAND FRANKEN_LOG FRANKEN_DEVICES \
         COLI_PREFIX_PIN COLI_THINK COLI_LEDGER OMP_NUM_THREADS FRANKEN_LAYERS FRANKEN_CPU \
         FRANKEN_THREADS FRANKEN_SEED FRANKEN_STOP_IDS FRANKEN_PLACEMENT FRANKEN_EXPERT_GB \
         FRANKEN_ADAPT FRANKEN_ADAPT_EVERY FRANKEN_ADAPT_MB_PER_TOKEN FRANKEN_ADAPT_HALFLIFE \
         FRANKEN_ADAPT_MARGIN FRANKEN_ADAPT_HYST FRANKEN_ADAPT_VERIFY \
         FRANKEN_ADAPT_SNAPSHOT_DURING_HOLD FRANKEN_ADAPT_SNAPSHOT_DMA FRANKEN_ADAPT_PREFILL \
         FRANKEN_HIP_GRAPH FRANKEN_HIP_GRAPH_BUCKET FRANKEN_PREFILL_PIPELINE FRANKEN_ALL_OPS \
         FRANKEN_GLM_PREFILL_STAGE FRANKEN_GLM_STAGE_MB FRANKEN_GLM_ATTN_MB \
         FRANKEN_GLM_LINKS FRANKEN_GLM_LONE_DEV \
         FRANKEN_GLM_MOE_G FRANKEN_GLM_HELP_COPY FRANKEN_GEMV_FUSED_REDUCE FRANKEN_GEMV_WAVE_REDUCE FRANKEN_GEMV_FUSED_MAX_SPLIT FRANKEN_GLM_FETCH_ASSIGN FRANKEN_ADAPT_PREFILL_CAP \
         FRANKEN_GEMV_ROWSPLIT FRANKEN_GEMV_ROWSPLIT_WAVES FRANKEN_GEMV_LDS FRANKEN_GLM_GEMV_GROUP FRANKEN_GEMV_Q8FAST \
         FRANKEN_SNAP_EVERY FRANKEN_SNAP_TURNS FRANKEN_SNAP_KEEP FRANKEN_SNAP_BUDGET_MB FRANKEN_BOUNDARY_TOKENS; do
  [ -n "${!v+x}" ] && ENVS+=(-e "$v=${!v}")
done
exec docker run --rm -i --name franken_engine --user $(id -u):$(id -g) --group-add 991 \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin "${ENVS[@]}" \
  -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" "$@"
