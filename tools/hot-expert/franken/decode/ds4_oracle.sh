#!/bin/bash
# tools/hot-expert/franken/decode/ds4_oracle.sh -- L5 step 1 oracle for DeepSeek-V4-Flash, CPU only, no GPU anywhere:
#   (1) llama.cpp dumps layers 0-3 of a 6-token prompt (franken_oracle_dump,
#       --stop-after-layer 3, lazy mmap via no_populate.so, no repacking);
#   (2) franken_decode_cpu runs layers 0-3 and compares every tap + routing.
# A watchdog kills the dump if the DeepSeek page cache grows past 10 GB.
set -u
M=/home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf
F=/home/ronald/src/colibri-m1/tools/hot-expert/franken
TOK="${TOK:-0 671 6102 294 8760 344}"
STOP=${STOP:-3}
O=${O:-/home/ronald/bench/franken/ds4/oracle6}
LLAMA_LIB=/home/ronald/src/llama-glm53/build-hip/bin
mkdir -p $O
res() { fincore -b -n -o RES /home/ronald/models/DeepSeek-V4-Flash-0731-UD-IQ2_M/UD-IQ2_M/*.gguf | awk '{s+=$1} END {print s}'; }
R0=$(res); echo "ds4_page_cache_before=$R0"
( while sleep 3; do r=$(res); if [ $((r - R0)) -gt 10000000000 ]; then echo "WATCHDOG: +$((r - R0)) bytes, killing"; pkill -f "franken_oracle_[d]ump"; pkill -f "franken_decode_[c]pu"; fi; done ) & W=$!
echo "=== (1) llama.cpp dump, layers 0-$STOP, tokens: $TOK"
docker run --rm --user $(id -u):$(id -g) -v /home/ronald:/home/ronald -w $F \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:$LLAMA_LIB -e LD_PRELOAD=$F/no_populate.so \
  rocm/dev-ubuntu-24.04:7.14.0-full ./franken_oracle_dump --model $M --tokens $TOK \
  --out $O --threads 4 --stop-after-layer $STOP --no-extra-bufts > $O/dump.log 2>&1
echo "dump rc=$? taps=$(wc -l < $O/index.txt 2>/dev/null)"
grep -E "stopping|decode done|NOT FOUND|load_mode|error|failed" $O/dump.log | head -20
echo "ds4_page_cache_after_dump=$(res) delta=$(( $(res) - R0 ))"
echo "=== (2) franken_decode_cpu, layers 0-$STOP"
$F/decode/franken_decode_cpu --model $M --tokens $TOK --cpu --layers 0-$STOP --no-head --ctx 512 \
  --threads 4 --oracle $O --routing $O --dump $O/../mine_$(basename $O) > $O/../mine_$(basename $O).log 2>&1
echo "decode rc=$?"
tail -3 $O/../mine_$(basename $O).log
echo "ds4_page_cache_after=$(res) delta=$(( $(res) - R0 ))"
kill $W 2>/dev/null
echo "=== DONE"
