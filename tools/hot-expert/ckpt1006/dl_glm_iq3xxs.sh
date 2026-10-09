#!/bin/bash
# download the four shards of unsloth/GLM-5.3-Flash-GGUF UD-IQ3_XXS (resumable), lowest priority; log ~/bench/dl_glm_iq3xxs.log
cd ~/models/GLM-5.3-Flash/UD-IQ3_XXS || exit 1
for i in 1 2 3 4; do
  f=$(printf "GLM-5.3-Flash-UD-IQ3_XXS-%05d-of-00004.gguf" $i)
  for try in 1 2 3 4 5; do
    nice -n 19 ionice -c3 curl -sS -L -C - --retry 3 -o "$f" "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-IQ3_XXS/$f" && break
    sleep 20
  done
  echo "$(date -u +%FT%TZ) $f $(stat -c %s "$f")"
done
echo "$(date -u +%FT%TZ) DONE"
