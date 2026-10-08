#!/bin/bash
# glm_d14_chain.sh -- D14 of DECODE-OPEN-ITEMS-PLAN-2026-10-08: is the prefill of a prompt past 2 051 tokens independent of the chunking, and does decode through a HIP graph equal eager decode?
# Hypothesis (record section L5-GLM-D14): k_glm5_attn_part's grid is sized from the LAST row of a launch; in the sparse regime a row's key count is 2 048 + (p + 1) % 4, not monotonic in p, so a chunk
# ending on a pool boundary has too few chunks for the rows with a tail; the combine kernel folds in the stale partials of the missing chunk. A chunk-1 prefill (every row its own launch) is the reference.
# ONE process (one load), shipped decode flags, adaptation OFF, prompt = the first $DEPTH ids of prose8400.txt (default 2 300), --ctx 4096:
#   ref1    --chunk 1   --greedy 1 --dump      the reference: each prompt token its own launch
#   c512    --chunk 512 --greedy 1 --oracle    the shipped chunking          } bit-exact against ref1 once the grid is right,
#   c256    --chunk 256 --greedy 1 --oracle                                  } not before (D14); a fixed binary must be exact in all four
#   c333    --chunk 333 --greedy 1 --oracle    (chunk ends that are not multiples of 4 either)
#   c512b   --chunk 512 --greedy 1 --oracle    the same again (repeatability)
#   e8      --chunk 512 --greedy 8 --dump      eager decode, 8 tokens
#   g8      --chunk 512 --greedy 8 --hip-graph 1 --oracle   the same through HIP graphs (greedy_ids + last logits + prompt taps against e8)
# rc of the chain: 0 when the process ran to its summary (a non-exact oracle config is the finding, not a failure), else the process's rc.
# Env: D14_BIN (binary, default franken_decode_glm_d14), D14_OUT (default ~/bench/franken/glm5/d14), D14_DEPTH (default 2300), D14_EXTRA (more flags for every config).
# Launch: ~/src/colibri/tools/hot-expert/preflight.sh && D14_BIN=... setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d14_chain.sh > ~/bench/glm_d14_chain.log 2>&1 < /dev/null &
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${D14_BIN:-$HOME/bench/franken_bin/franken_decode_glm_d14}
O=${D14_OUT:-$HOME/bench/franken/glm5/d14}
DEPTH=${D14_DEPTH:-2300}
say_end() { echo "=== glm_d14 exit rc=$1 $(date -Is)"; exit "$1"; }
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
FLAGS="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1 --adapt 0 ${D14_EXTRA:-}"
echo "=== glm_d14 start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) depth=$DEPTH flags=[$FLAGS]"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
[ -f "$B/prose8400.txt" ] || { echo "FATAL: prose8400.txt missing"; say_end 2; }
if [ -d "$O" ]; then docker run --rm -v "$(dirname "$O")":/w rocm/dev-ubuntu-24.04:7.14.0-full rm -rf "/w/$(basename "$O")" >/dev/null 2>&1; fi
mkdir -p "$O"
tr ' ' '\n' < "$B/prose8400.txt" | grep -v '^$' | head -$DEPTH | tr '\n' ' ' > "$O/prompt_deep.txt"
echo "prompt ids: deep=$(wc -w < "$O/prompt_deep.txt")"
TOK="--tokens-file $O/prompt_deep.txt --ctx 4096"
: > "$O/plan.txt"
echo "ref1  $TOK --chunk 1   --greedy 1 $FLAGS --dump $O/d_ref" >> "$O/plan.txt"
echo "c512  $TOK --chunk 512 --greedy 1 $FLAGS --oracle $O/d_ref" >> "$O/plan.txt"
echo "c256  $TOK --chunk 256 --greedy 1 $FLAGS --oracle $O/d_ref" >> "$O/plan.txt"
echo "c333  $TOK --chunk 333 --greedy 1 $FLAGS --oracle $O/d_ref" >> "$O/plan.txt"
echo "c512b $TOK --chunk 512 --greedy 1 $FLAGS --oracle $O/d_ref" >> "$O/plan.txt"
echo "e8    $TOK --chunk 512 --greedy 8 $FLAGS --dump $O/d_e8" >> "$O/plan.txt"
echo "g8    $TOK --chunk 512 --greedy 8 --hip-graph 1 $FLAGS --oracle $O/d_e8" >> "$O/plan.txt"
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/run.log" 2>&1; rc=$?
echo "process rc=$rc $(date -Is)"; grep -aE "HIP error|Memory access|out of memory" "$O/run.log" | head -2 | cut -c1-170
if [ $rc -ge 2 ] || ! grep -aq "gate_plan_summary" "$O/run.log"; then echo "process failed (rc=$rc or no summary)"; tail -n 5 "$O/run.log" | cut -c1-200; say_end $rc; fi
python3 -I "$HOME/bench/d14_report.py" "$O/run.log" 2>&1 | cut -c1-240
say_end 0
