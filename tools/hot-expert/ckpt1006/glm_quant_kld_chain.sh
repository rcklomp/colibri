#!/bin/bash
# glm_quant_kld_chain.sh -- how far is a smaller expert quant of GLM-5.3-Flash from the quant we serve (UD-IQ4_XS)?
#
# Why (PF4, 2026-10-09): chunk 2048 needs +2.2-2.8 GB of VRAM a card that only resident experts could pay, at a decode price; a smaller expert quant shrinks every expert
# (UD-IQ3_XXS: 11.67 -> 8.98 MB in the 39 main layers), which frees that VRAM at the same resident-expert count AND cuts the missed bytes of decode and the link bytes of prefill
# by itself. Unsloth's own table (vs BF16): UD-IQ4_XS top-1 88.18 % / mean KLD 0.117, UD-IQ3_XXS 81.63 % / 0.284. This chain measures the quant against OUR served quant, on
# text of the kind the rig is used for, with llama.cpp's KL-divergence tool (the same llama-glm53 HIP build the quality harness uses, in the same container image).
#
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_quant_kld_chain.sh > ~/bench/quant_kld_chain.log 2>&1 < /dev/null &
#
# Env: CAND (default UD-IQ3_XXS), CAND_N (shard count, default 4), CHUNKS (default 10), CTX (default 4096).
# Steps: (1) the corpus; (2) IQ4_XS writes its logits (--kl-divergence-base); (3) wait for the candidate's download (~/bench/dl_glm_iq3xxs.log DONE); (4) the candidate
# is scored against them (--kl-divergence); (5) summary.txt. Output ~/bench/quant_kld/. The logits file is ~CHUNKS*CTX/2 * 155 k * 2 bytes (about 6 GB at the defaults).
set -u
OUT=$HOME/bench/quant_kld; mkdir -p "$OUT"
BIN=/home/ronald/src/llama-glm53/build-hip/bin
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
CAND=${CAND:-UD-IQ3_XXS}; CAND_N=${CAND_N:-4}; CHUNKS=${CHUNKS:-10}; CTX=${CTX:-4096}
BASE_GGUF=$HOME/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
CAND_GGUF=$HOME/models/GLM-5.3-Flash/$CAND/GLM-5.3-Flash-$CAND-00001-of-0000$CAND_N.gguf
CORPUS=$OUT/corpus.txt; LOGITS=$OUT/base_iq4xs.kld
say() { echo "$(date +%H:%M:%S) quant_kld: $*"; }

# ---- (1) the corpus: technical prose, source code, project docs (the kinds of text the rig is asked about) ----
if [ ! -s "$CORPUS" ]; then
  { head -c 60000 "$HOME/bench/prompt_64ki.txt"; echo; echo
    head -c 50000 "$HOME/src/franken-engine/franken/decode/glm5_gpu.inc"; echo; echo
    head -c 40000 "$HOME/src/colibri/CLAUDE.md"; echo; echo
    head -c 30000 "$HOME/src/colibri/tools/hot-expert/HANDOFF-2026-10-06b.md"; } > "$CORPUS"
fi
say "corpus $(wc -c < "$CORPUS") bytes"

run_ppl() {   # run_ppl <name> <gguf> <extra args…>
  local name=$1 gguf=$2; shift 2
  local log=$OUT/$name.log
  say "start $name ($gguf)"
  timeout 9000 docker run --rm --name "quantkld-$name" --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host \
    -e "LD_LIBRARY_PATH=/opt/rocm/lib:$BIN" -v /home/ronald:/home/ronald "$IMG" "$BIN/llama-perplexity" \
    -m "$gguf" -f "$CORPUS" -c "$CTX" -b "$CTX" -ub 2048 --chunks "$CHUNKS" \
    --fit on --fit-target 1024,1024,1024 --fit-ctx "$CTX" --split-mode layer --device ROCm0,ROCm1,ROCm2 -fa on -t 16 -tb 8 "$@" > "$log" 2>&1
  local rc=$?
  [ $rc -eq 124 ] && docker kill "quantkld-$name" >/dev/null 2>&1
  say "end $name rc=$rc"
  return $rc
}

# ---- (2) the base: our served quant writes its logits ----
if [ ! -s "$LOGITS" ]; then
  run_ppl base_iq4xs "$BASE_GGUF" --kl-divergence-base "$LOGITS" || { say "FATAL: the base run failed (see $OUT/base_iq4xs.log)"; tail -20 "$OUT/base_iq4xs.log"; exit 1; }
fi
ls -la "$LOGITS"

# ---- (3) the candidate's download ----
for i in $(seq 1 360); do grep -q DONE "$HOME/bench/dl_glm_${CAND#UD-}.log" 2>/dev/null && break; sleep 10; done
sz=$(cat "$HOME/models/GLM-5.3-Flash/$CAND"/*.gguf 2>/dev/null | wc -c)
say "candidate files: $sz bytes"
[ "$sz" -gt 100000000000 ] || { say "FATAL: the candidate download is not complete"; exit 1; }

# ---- (4) the candidate against the base ----
run_ppl "cand_${CAND}" "$CAND_GGUF" --kl-divergence-base "$LOGITS" --kl-divergence || { say "FATAL: the candidate run failed"; tail -20 "$OUT/cand_${CAND}.log"; exit 1; }

# ---- (5) summary ----
{
  echo "== $CAND against UD-IQ4_XS (ours), $CHUNKS chunks of $CTX tokens ($((CHUNKS * CTX / 2)) scored tokens), corpus $(wc -c < "$CORPUS") bytes"
  grep -E "Mean PPL|Mean    KLD|Mean    Δp|RMS Δp|Same top p|Maximum KLD|99.9%|Median KLD|99.0%|90.0%|Minimum KLD|Mean    ln\(PPL" "$OUT/cand_${CAND}.log"
  grep -E "load_tensors:.*(buffer|model buffer)|llama_kv_cache|system_info" "$OUT/cand_${CAND}.log" | head -8
} > "$OUT/summary_${CAND}.txt"
cat "$OUT/summary_${CAND}.txt"
say "done"
