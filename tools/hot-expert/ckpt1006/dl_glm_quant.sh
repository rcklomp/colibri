#!/bin/bash
# dl_glm_quant.sh QUANT NSHARDS -- resumable, lowest-priority download of unsloth/GLM-5.3-Flash-GGUF/QUANT (e.g. UD-Q3_K_XL 4) to ~/models/GLM-5.3-Flash/QUANT; log ~/bench/dl_glm_<quant, lower case, no _ or ->.log
Q=${1:?quant}; N=${2:?shards}
LOG=$HOME/bench/dl_glm_$(echo "${Q#UD-}" | tr -d '_-' | tr A-Z a-z).log
mkdir -p "$HOME/models/GLM-5.3-Flash/$Q" && cd "$HOME/models/GLM-5.3-Flash/$Q" || exit 1
for i in $(seq 1 "$N"); do
  f=$(printf "GLM-5.3-Flash-%s-%05d-of-%05d.gguf" "$Q" "$i" "$N")
  for try in 1 2 3 4 5; do
    nice -n 19 ionice -c3 curl -sS -L -C - --retry 3 -o "$f" "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/$Q/$f" && break
    sleep 20
  done
  echo "$(date -u +%FT%TZ) $f $(stat -c %s "$f")"
done >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) DONE" >> "$LOG"
