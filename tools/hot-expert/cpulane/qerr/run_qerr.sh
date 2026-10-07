#!/bin/bash
# run_qerr.sh -- the full qerr experiment: both reference dumps (g136_eager: 136-token prompt, ref512_keep: 2200-token
# prompt) x all 42 MoE layers (3..44) x 8 slots, int8-activation chain (ggml kernels) vs float chain, on real
# GLM-5.3-Flash data. CPU only; reads the GGUF shards and the dump dirs read-only; touches no GPU, no git repo.
# Everything goes to stdout; the LAST line is "=== qerr exit rc=N" (0 ok, 2 setup/IO error, 3 harness validation failed).
#
#   ~/src/cpulane/qerr/run_qerr.sh                      # 2 threads, ~1.5 min when the shards are in the page cache
#   QERR_THREADS=1 ~/src/cpulane/qerr/run_qerr.sh       # one thread
#   QERR_PREFIX="nice -n 19 taskset -c 14,15" QERR_THREADS=2 ~/src/cpulane/qerr/run_qerr.sh   # polite, 2 hardware threads
#   extra qerr arguments are passed through, e.g.  run_qerr.sh --layers 7-8 --dump g136=$HOME/bench/franken/glm5/gpu5/g136_eager
cd "$(dirname "$0")" || exit 2
if [ ! -x ./qerr ]; then ${QERR_PREFIX} make qerr >&2 || { echo "=== qerr exit rc=2"; exit 2; }; fi
${QERR_PREFIX} ./qerr --threads "${QERR_THREADS:-2}" "$@"
rc=$?
echo "=== qerr exit rc=$rc"
exit $rc
