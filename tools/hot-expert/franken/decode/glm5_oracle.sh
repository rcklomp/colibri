#!/bin/bash
# tools/hot-expert/franken/decode/glm5_oracle.sh -- L5 GLM step 3, the CPU oracle for
# GLM-5.3-Flash, no GPU anywhere:
#   (1) llama.cpp dumps layers 0-$STOP of the prompt (franken_oracle_dump, --stop-after-layer,
#       lazy mmap via no_populate.so, no repacking, --ctx 4096 so the indexer SCORES:
#       glm5next.cpp runs the pool scores and the top-k only when n_ctx > 2 051);
#   (2) franken_decode (CPU binary) runs layers 0-$STOP at the same ctx and compares every
#       tap and the routing.
# A watchdog kills both if the GLM page cache grows past 10 GB (the model is served on
# this box; a CPU oracle must not push it out).
#   TOK="..." STOP=N O=<dir> [DUMP=0 to reuse an existing dump] [EXTRA="--quant-act"] ./glm5_oracle.sh
set -u
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
F=/home/ronald/src/colibri-m1/tools/hot-expert/franken
TOK="${TOK:-154822 785 6722 315 9621 374}"
STOP=${STOP:-3}
CTX=${CTX:-4096}
O=${O:-/home/ronald/bench/franken/glm5/oracle6}
DUMPER=${DUMPER:-$F/franken_oracle_dump}
MINE=${MINE:-$F/decode/franken_decode_cpu}
LLAMA_LIB=/home/ronald/src/llama-glm53/build-hip/bin
mkdir -p $O
res() { fincore -b -n -o RES /home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/*.gguf | awk '{s+=$1} END {print s}'; }
R0=$(res); echo "glm_page_cache_before=$R0"
( while sleep 3; do r=$(res); if [ $((r - R0)) -gt 10000000000 ]; then echo "WATCHDOG: +$((r - R0)) bytes, killing"; pkill -f "franken_oracle_[d]ump"; pkill -f "franken_decode_[g]lm_cpu"; pkill -f "franken_decode_[c]pu --model /home/ronald/models/GLM"; fi; done ) & W=$!
if [ "${DUMP:-1}" = 1 ]; then
  echo "=== (1) llama.cpp dump, layers 0-$STOP, ctx $CTX, tokens: $(echo $TOK | wc -w)"
  docker run --rm --user $(id -u):$(id -g) -v /home/ronald:/home/ronald -w $F \
    -e LD_LIBRARY_PATH=/opt/rocm/lib:$LLAMA_LIB -e LD_PRELOAD=$F/no_populate.so \
    rocm/dev-ubuntu-24.04:7.14.0-full $DUMPER --model $M --tokens $TOK \
    --out $O --threads 4 --stop-after-layer $STOP --no-extra-bufts --ctx $CTX > $O/dump.log 2>&1
  echo "dump rc=$? taps=$(wc -l < $O/index.txt 2>/dev/null)"
  grep -E "stopping|decode done|NOT FOUND|error|failed|WATCHDOG" $O/dump.log | head -20
  echo "glm_page_cache_after_dump=$(res) delta=$(( $(res) - R0 ))"
fi
echo "=== (2) franken_decode (CPU), layers 0-$STOP, ctx $CTX"
L=$O/../mine_$(basename $O)
$MINE --model $M --tokens $TOK --cpu --layers 0-$STOP --no-head --ctx $CTX \
  --threads 4 --oracle $O --routing $O --dump $L ${EXTRA:-} > $L.log 2>&1
echo "decode rc=$?"
grep -E "slot_order|routing|GLM5 CPU|oracle summary|FAIL|MISSING|REQUIRED" $L.log | head -30
echo "min_cos=$(grep '^oracle .* cos=' $L.log | sed 's/.* cos=\([-0-9.]*\).*/\1/' | sort -g | head -1) taps=$(grep -c '^oracle .* cos=' $L.log)"
echo "glm_page_cache_after=$(res) delta=$(( $(res) - R0 ))"
kill $W 2>/dev/null
echo "=== DONE"
