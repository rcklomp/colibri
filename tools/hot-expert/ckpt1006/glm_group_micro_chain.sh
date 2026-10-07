#!/bin/bash
# glm_group_micro_chain.sh -- the --glm-gemv-group MICROBENCHMARK on ONE card, no model load (2026-10-08; franken-engine branch gemv-group, GLM5.md section 21).
# A decode token launches each trunk GEMV on its own, and many of them read the SAME activation vector. Backend::gemv_group runs the ones that do as ONE launch -- KDA wq|wk|wv
# (G1, 3 x 8192 rows, solo nsplit 1), KDA f_a|beta|g_a (G2, 128+64+128 rows, solo nsplit 128), the indexer's idx_k|idx_gate|idx_proj (G3, 128+128 Q8_0 + 32 F32, solo nsplit 128) --
# with EVERY member at the K-split count it has ALONE (nsplit decides the summation order; gemv_batch's own formula would take the concatenation's rows: 52 for G2, not 128), and only
# when all members' solo counts are equal. Is every member's output BIT-IDENTICAL to main's SOLO gemv of the same matrix, and how many us does the merge save per launch sequence?
# bench_group (franken/decode/bench_group.hip) #includes the real decode_gpu.hip and links main's own backend (fa1c233) as the reference: one line a case, EXACT or MISMATCH
# (memcmp, every member, 40 configurations: rowsplit 0/1, waves auto/4/8/16, gemv-lds 0/1, fused reduce 0/1, wave reduce 0/1; the group off; T = 13, which must take the solo path;
# the path each call took is asserted from the backend's own counters), then the timing table (hot / cold / dependent-chain us per launch sequence, merged against the sum of the solo
# launches) and the estimated saving a token (G1 x 34, G2 x 34, G3 x 11). Exit 1 on any mismatch. No Q6_K case (a synthetic Q6_K nsplit-16 case stalls the card in main's own backend).
# Allocates up to ~1 GB on card 0 (the cold copies).
# Launch through run_chain.sh, watch with ckpt1006/watch_chain.sh <chain log> <bench log>:
#   ~/src/colibri/tools/hot-expert/preflight.sh && setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_group_micro_chain.sh > ~/bench/glm_group_micro_chain.log 2>&1 < /dev/null &
#   watch: ckpt1006/watch_chain.sh ~/bench/glm_group_micro_chain.log ~/bench/franken/glm5/group_micro/bench.log
# Env: RSM_ENV (extra docker -e options, e.g. "-e AMD_LOG_LEVEL=3"), RSM_ARGS (bench arguments: --exact-only, --verbose, --skip-case NAME, --only-case NAME), RSM_TIMEOUT (s, default 900; the bench
# container is killed after it), RSM_BIN (default ~/bench/franken_bin/bench_group, `make -C franken/decode bench-group`), RSM_OUT (default ~/bench/franken/glm5/group_micro).
set -u
. "$HOME/bench/chain_preflight.sh"
BIN=${RSM_BIN:-$HOME/bench/franken_bin/bench_group}
O=${RSM_OUT:-$HOME/bench/franken/glm5/group_micro}; rm -rf "$O"; mkdir -p "$O"
say_end() { echo "=== glm_group_micro exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_group_micro start $(date -Is) bin=$(sha256sum "$BIN" 2>/dev/null | cut -c1-16) branch=$(git -C "$(dirname "$BIN")" log --oneline -1 2>/dev/null)"
[ -x "$BIN" ] || { echo "FATAL: no binary $BIN"; say_end 2; }
rig_quiet_wait 1800 || { echo "FATAL: the rig did not become quiet"; say_end 3; }
docker rm -f group_micro >/dev/null 2>&1
( sleep ${RSM_TIMEOUT:-900}; docker kill group_micro >/dev/null 2>&1 && echo "TIMEOUT: bench killed after ${RSM_TIMEOUT:-900} s (a hung kernel?)" ) &
WD=$!
docker run --name group_micro --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 -e HIP_VISIBLE_DEVICES=0 ${RSM_ENV:-} \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \
  rocm/dev-ubuntu-24.04:7.14.0-full "$BIN" ${RSM_ARGS:-} > "$O/bench.log" 2>&1
rc=$?
kill $WD 2>/dev/null
echo "bench rc=$rc"
grep -aE "HIP error|Memory access|out of memory|Segmentation|Aborted" "$O/bench.log" | head -3 | cut -c1-200
echo "=== exactness: $(grep -ac ' EXACT ' "$O/bench.log") EXACT lines, $(grep -ac 'MISMATCH' "$O/bench.log") MISMATCH lines"
grep -a 'MISMATCH' "$O/bench.log" | head -20 | cut -c1-200
grep -aE "^[A-Za-z0-9_]+: [0-9]+ matrices" "$O/bench.log" | cut -c1-200
grep -aE "^  [A-Za-z0-9_]+ +[0-9]+ configs" "$O/bench.log" | cut -c1-200
sed -n '/^=== TIMING/,$p' "$O/bench.log" | cut -c1-220
grep -a "^  timing " "$O/bench.log" | cut -c1-260
[ "$rc" -ne 0 ] && { echo "--- bench log tail:"; tail -n 8 "$O/bench.log" | cut -c1-200; }
say_end $rc
