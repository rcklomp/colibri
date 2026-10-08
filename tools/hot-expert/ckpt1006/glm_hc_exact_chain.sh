#!/bin/bash
# glm_hc_exact_chain.sh -- is --glm-help-copy 1 bit-exact, in the PIPELINED prefill? (plan Rev 102, franken-engine branch pf-helpcopy, binary franken_decode_glm_pfhc2 with --pipeline-taps)
# The kernel copy removes a runtime-imposed ordering of peer copies; chunks now overlap (2.88 chunks in flight against 1.59, timeline hc1_t), so a data race that the old serialisation hid could show here. The earlier
# gates could not see it: with --time-prefill 0 every chunk is flushed before the next, so no gate ever ran the overlapped schedule the service uses. --pipeline-taps 1 keeps the chunks pipelined (no flush) and still
# taps the last chunk. THE REFERENCE IS A SERIALISED RUN OF THE SAME FLAGS, NOT A CHUNK-1 PREFILL: --gemm-lds 1 (served) is a last-bits order change against the T = 1 path, so a chunked prefill is not bit-equal to a
# chunk-1 one under it (the first version of this chain compared against chunk 1 and every chunked config read cos 0.96-0.99: a flaw of the gate, not of the change; the D14 gate has no --gemm-lds and is exact).
# ONE process, shipped decode flags, adaptation OFF, in place, G=8 (env), --gemm-lds 1, prompt = the first $DEPTH ids of prose8400 (default 6 144 = six 1 024-row chunks):
#   ref     --chunk 1024 --prefill-pipeline 0 --glm-help-copy 0            serialised, shipped copy: the reference (dump)
#   p0      --chunk 1024 --pipeline-taps 1 --glm-help-copy 0               pipelined, shipped copy         } each against ref, every tap bit-exact
#   p1      --chunk 1024 --pipeline-taps 1 --glm-help-copy 1               pipelined, kernel copy (THE CANDIDATE)
#   p1b     the same again                                                 repeatability: a race differs run to run
#   s512    --chunk 512 --prefill-pipeline 0 --glm-help-copy 0             serialised at another chunk size: rows are independent of the chunking under --gemm-lds 1 (so the chunk-512 / 333 / 256 lines below can use ref)
#   p1_512  --chunk 512 --pipeline-taps 1 ... 1   p1_333  --chunk 333 ...   p1_256  --chunk 256 ...   (more chunks in flight, chunk ends that are not multiples of 4)
# Verdict: oracle_verdict.sh per config = taps compared and taps NOT bit-exact; every config must read not_exact=0. A non-exact config is the finding, not a chain failure.
# Launch through run_chain.sh.  Env: HX_BIN, HX_OUT, HX_DEPTH.
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${HX_BIN:-$HOME/bench/franken_bin/franken_decode_glm_pfhc2}
O=${HX_OUT:-$HOME/bench/franken/glm5/hc_exact}; DEPTH=${HX_DEPTH:-6144}
M=/home/ronald/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
P=/home/ronald/bench/m2/glm; EG=17; B=/home/ronald/bench/franken/glm5
say_end() { echo "=== glm_hc_exact exit rc=$1 $(date -Is)"; exit "$1"; }
[ -x "$BIN" ] && [ -f "$B/prose8400.txt" ] || { echo "FATAL: missing $BIN or prose8400.txt"; say_end 2; }
if [ -d "$O" ]; then docker run --rm -v "$(dirname "$O")":/w rocm/dev-ubuntu-24.04:7.14.0-full rm -rf "/w/$(basename "$O")" >/dev/null 2>&1; fi
mkdir -p "$O"
tr ' ' '\n' < "$B/prose8400.txt" | grep -v '^$' | head -$DEPTH | tr '\n' ' ' > "$O/prompt.txt"
echo "=== glm_hc_exact start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16) depth=$(wc -w < "$O/prompt.txt")"
FLAGS="--fetch-assign optimal --gemv-wave-reduce 1 --gemv-rowsplit 1 --glm-gemv-group 1 --adapt 0 --adapt-prefill 0 --gemm-lds 1 --glm-prefill-stage 0 --greedy 4"
TOK="--tokens-file $O/prompt.txt --ctx 16384"
{ echo "ref    $TOK --chunk 1024 --prefill-pipeline 0 --glm-help-copy 0 $FLAGS --dump $O/d_ref"
  echo "p0     $TOK --chunk 1024 --pipeline-taps 1 --glm-help-copy 0 $FLAGS --oracle $O/d_ref"
  echo "p1     $TOK --chunk 1024 --pipeline-taps 1 --glm-help-copy 1 $FLAGS --oracle $O/d_ref"
  echo "p1b    $TOK --chunk 1024 --pipeline-taps 1 --glm-help-copy 1 $FLAGS --oracle $O/d_ref"
  echo "s512   $TOK --chunk 512  --prefill-pipeline 0 --glm-help-copy 0 $FLAGS --oracle $O/d_ref"
  echo "p1_512 $TOK --chunk 512  --pipeline-taps 1 --glm-help-copy 1 $FLAGS --oracle $O/d_ref"
  echo "p1_333 $TOK --chunk 333  --pipeline-taps 1 --glm-help-copy 1 $FLAGS --oracle $O/d_ref"
  echo "p1_256 $TOK --chunk 256  --pipeline-taps 1 --glm-help-copy 1 $FLAGS --oracle $O/d_ref"; } > "$O/plan.txt"
rig_quiet_wait 1800 || say_end 3
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e FRANKEN_GLM_MOE_G=8 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" --model $M --placement $P --expert-gb $EG --gate-plan "$O/plan.txt" > "$O/gate_run.log" 2>&1
rc=$?; echo "engine rc=$rc"; grep -aE "gate_plan_summary|HIP error|Memory access|out of memory" "$O/gate_run.log" | head -3 | cut -c1-200
[ "$rc" -ne 0 ] && tail -n 8 "$O/gate_run.log" | cut -c1-200
echo "=== exactness against the chunk-1 reference (taps compared / not bit-exact)"
bash "$HOME/bench/oracle_verdict.sh" "$O/gate_run.log"
echo "=== greedy ids per config"
awk '/^=== gate-plan config/ { c = $4 } /^greedy_ids:/ { printf "%-7s %s\n", c, $0 }' "$O/gate_run.log" | cut -c1-120
awk '/^=== gate-plan config/ { c = $4 } /^prompt_tokens=/ { printf "%-7s %s\n", c, $0 }' "$O/gate_run.log" | cut -c1-120
say_end "$rc"
