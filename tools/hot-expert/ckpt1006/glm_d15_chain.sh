#!/bin/bash
# glm_d15_chain.sh -- D15 (ACTION-LIST row 6, DECODE-OPEN-ITEMS-PLAN section 8): the refined fetch table (franken-engine branch d15-fetch-table, GLM5.md section 31) on the SERVED
# IQ3_XXS hybrid. Offline (record §L5-D15) the 2026-10-07 traces showed why --fetch-assign optimal gained a third of its model: a shared-pair card with more slabs than its partner
# runs alone at 417 us once the partner is done, so the old model overpriced unequal pair splits; the refined model fits the measured gain (0.96 vs 0.94 ms) and moves one row
# (n = 5: (2,0,3) -> (2,1,2), modeled -1 ms a token on IQ4_XS). Stages:
#   1 = build the CLI from the worktree ~/src/franken-engine-d15 (franken_decode_glm_d15) + the host unit test (test-fetch-assign);
#   2 = exactness (glm5_gpu_gate.sh fa2, one process): 64 greedy tokens at depth 1500 dumped with optimal-v1, the new table held to it bit for bit (plain, swap every token,
#       graph replay; the pattern as a third placement), shipped decode flags;
#   3 = the paired kernel trace, one process per arm, order v1 new new v1 (glm_arms_trace_seq.sh), 2 048 prompt tokens + 48 settle + TOKENS traced, shipped decode flags as CLI flags.
# Launch (rig idle, service OFF, reservation flag in place):  setsid nohup ~/src/colibri/tools/hot-expert/ckpt1006/glm_d15_chain.sh > ~/bench/glm_d15_chain.log 2>&1 < /dev/null &
#   (stages 2 and 3 take the rig lock through run_chain.sh themselves; stage 1 only builds)
# Watch: ckpt1006/watch_chain.sh ~/bench/glm_d15_chain.log ~/bench/franken/glm5/d15_gate/gate_run.log
# Report: python3 ckpt1006/d15_pair_report.py ~/bench/franken/glm5/dtrace_v1d15_1 ~/bench/franken/glm5/dtrace_newd15_2 ~/bench/franken/glm5/dtrace_newd15_3 ~/bench/franken/glm5/dtrace_v1d15_4
# Env: D15_ONLY (1,2,3), D15_TOKENS (decode tokens traced, default 24).
set -u
. "$HOME/bench/chain_preflight.sh"
W=$HOME/src/franken-engine-d15; D=$W/franken/decode; BIN=franken_decode_glm_d15
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf; EG=16.85
G=$HOME/bench/franken/glm5/d15_gate
ONLY=${D15_ONLY:-1,2,3}; has() { case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }
say_end() { echo "=== glm_d15 exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d15 start $(date -Is) stages=$ONLY engine=$(git -C "$W" log --oneline -1 2>/dev/null)"
[ -d "$D" ] || { echo "FATAL: no worktree $W"; say_end 2; }
if has 1; then
  echo "=== 1. build + unit test $(date +%T)"
  ( cd "$D" && nice -n 19 make -j8 gpu GPU_BIN=$BIN 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -5 )
  ( cd "$D" && make test-fetch-assign 2>&1 | tail -3 ) | tee /dev/stderr | grep -q "test_fetch_assign PASS" || { echo "FATAL: test_fetch_assign did not pass"; say_end 2; }
fi
[ -x "$D/$BIN" ] || { echo "FATAL: no $D/$BIN"; say_end 2; }
echo "bin=$BIN sha=$(sha256sum "$D/$BIN" | cut -c1-16)"
if has 2; then
  echo "=== 2. exactness gate fa2 $(date +%T)"
  rig_quiet_wait 1800 || say_end 3
  mkdir -p "$G"
  cat > "$HOME/bench/glm_d15_gate_inner.sh" <<IN
#!/bin/bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \\
  -e GLM5_GPU_OK=1 -e GLM5_D="$D" -e GLM5_GPU_BIN="$BIN" -e GLM5_GATE_OUT="$G" -e GLM5_MODEL="$M" -e GLM5_EXPERT_GB=$EG \\
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench \\
  rocm/dev-ubuntu-24.04:7.14.0-full bash "$D/glm5_gpu_gate.sh" fa2 > "$G/gate.txt" 2>&1
rc=\$?; echo "gate rc=\$rc (0 PASS, 1 FAIL, 3 the engine died)"; tail -n 20 "$G/gate.txt" | cut -c1-210
echo "=== glm_d15_gate exit rc=\$rc \$(date -Is)"; exit \$rc
IN
  chmod +x "$HOME/bench/glm_d15_gate_inner.sh"
  "$HOME/src/colibri/tools/hot-expert/run_chain.sh" "$HOME/bench/glm_d15_gate_inner.sh"; rc=$?
  rm -f "$HOME/bench/glm_d15_gate_inner.sh"
  [ $rc -ne 0 ] && { echo "exactness gate rc=$rc: no trace"; say_end $rc; }
fi
if has 3; then
  echo "=== 3. paired kernel trace v1 new new v1 $(date +%T)"
  TS_BIN="$D/$BIN" TS_TOKENS=${D15_TOKENS:-24} TS_TAG=d15 \
  TS_SHIPPED="--adapt-prefill-cap 64 --adapt-halflife 512 --glm-help-copy 1 --gemm-lds 3 --gemm-lds-min-rows 16 --gemm-lds-splitk 1 --gemv-rowsplit 1 --gemv-rowsplit-waves 1 --gemv-q8fast 2 --glm-gemv-group 1" \
  TS_ARMS="v1=--fetch-assign optimal-v1|new=--fetch-assign optimal|new=--fetch-assign optimal|v1=--fetch-assign optimal-v1" \
  TRACE_MODEL="$M" TRACE_EG=$EG TRACE_SETTLE=48 \
    "$HOME/src/colibri/tools/hot-expert/ckpt1006/glm_arms_trace_seq.sh"; rc=$?
  say_end $rc
fi
say_end 0
