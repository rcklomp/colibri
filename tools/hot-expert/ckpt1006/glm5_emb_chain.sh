#!/bin/bash
# glm5_emb_chain.sh -- GLM5.md section 15: the embedding rows by KERNEL instead of a copy-engine
# upload (--embed-upload-dma), one GPU process (glm5_gpu_gate.sh emb: one load, then emb_smoke,
# pf2's four exact configs against the gpu5 references, A,B,B,A prefill + decode timing with
# A = --embed-upload-dma 1 and B = the default, and one --timeline run on the new path). The shape
# of glm5_timeline_chain.sh: stop the gateway, run the gate in the ROCm 7.14 image with the cards
# (--ulimit memlock=-1: the pinned expert mirror and the embedding banks), read the timeline with
# glm5_timeline.py on the host, then the usual exit (gateway left down while ~/bench/.dev_reserved
# exists, else left down for run_chain.sh / gateway_watchdog.sh to restore -- see on_exit).
#
# Launch (on the rig; NOT run by the agent that wrote it -- it initialises the GPUs):
#   ~/src/colibri/tools/hot-expert/preflight.sh && \
#     setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm5_emb_chain.sh \
#       > ~/bench/glm5_emb_chain.log 2>&1 < /dev/null &
# Env: EMB_WT the worktree whose build runs (default ~/src/franken-engine), EMB_OUT the
# output dir (default ~/bench/franken/glm5/pfemb), HWQ (optional) GPU_MAX_HW_QUEUES for the
# container (GLM5.md 14.3 H1b). Outputs: gate_run.log, gate.txt, verdict.txt, timeline.csv,
# timeline_smoke.csv, timeline_report.txt (the gate's), timeline_report_host.txt.
set -u
HERE=~/src/colibri/tools/hot-expert
WT=${EMB_WT:-/home/ronald/src/franken-engine}
D=$WT/franken/decode
O=${EMB_OUT:-/home/ronald/bench/franken/glm5/pfemb}
KEY=$(cat ~/.colibri_api_key); GLOG=~/glm53_server.log
engine_alive() { ps -C glm53 -o stat= 2>/dev/null | grep -qv '^Z'; }
franken_alive() { [ -n "$(docker ps -q -f name='^franken_engine$' 2>/dev/null)" ]; }
wait_engines_gone() {
  for _ in $(seq 1 120); do engine_alive || franken_alive || return 0; sleep 2; done
  echo "an engine is still alive after 240 s (glm53: $(engine_alive && echo yes || echo no), franken_engine container: $(franken_alive && echo yes || echo no))"; return 1
}
stop_gateway() {
  pkill -f "openai_[s]erver.py" 2>/dev/null || true
  for _ in $(seq 1 10); do pgrep -f "openai_[s]erver.py" >/dev/null || break; sleep 1; done
  pkill -9 -f "openai_[s]erver.py" 2>/dev/null || true
  pkill -9 -x glm53 2>/dev/null || true
  wait_engines_gone
}
on_exit() {
  rc=$?; trap - EXIT INT TERM HUP
  if [ -e "$HOME/bench/.dev_reserved" ]; then
    echo "rig reserved for development: gateway left down"
    "$HERE/accept_live.sh"; a=$?; echo "--- accept_live exit=$a (SKIPPED under the reservation)"; [ "$rc" -eq 0 ] && rc=$a
  else
    # 2026-10-06: this handler used to restart the OLD Colibri engine (~/start_glm53.sh) here, which would replace a Franken
    # service and made run_chain.sh see "in service". The chain now leaves the gateway down: run_chain.sh (which launches it)
    # releases the rig lock and restores the SERVICE engine named in ~/bench/service_engine; outside run_chain.sh the
    # gateway_watchdog.sh does it within 5 minutes. Judge the restored service with accept_live.sh afterwards.
    echo "gateway left down: run_chain.sh / gateway_watchdog.sh restore the service engine (~/bench/service_engine) after this exit"
  fi
  echo "=== glm5_emb_chain exit rc=$rc $(date -Is)"
  exit "$rc"
}
trap on_exit EXIT INT TERM HUP
[ -x $D/franken_decode_glm ] || { echo "no $D/franken_decode_glm (make -C franken/decode gpu GPU_BIN=franken_decode_glm)"; exit 2; }
mkdir -p $O
echo "=== glm5_emb_chain start $(date -Is) worktree $WT ($(git -C $WT log --oneline -1 2>/dev/null))"
stop_gateway || exit 1
sleep 5
m=0
for d in /sys/class/drm/card[0-9]/device; do
  u=$(( $(cat $d/mem_info_vram_used) / 1048576 )); echo "vram_used_before=$u MiB"; [ $u -gt $m ] && m=$u
done
[ $m -gt 1024 ] && { echo "REFUSED: a card still holds $m MiB (another engine?) -- not timing on busy cards"; exit 3; }
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host \
  --ulimit memlock=-1 ${HWQ:+-e GPU_MAX_HW_QUEUES=$HWQ} -e GLM5_GPU_OK=1 -e GLM5_GATE_OUT=$O -e GLM5_D=$D \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald \
  -v /home/ronald:/home/ronald -w $D rocm/dev-ubuntu-24.04:7.14.0-full bash ./glm5_gpu_gate.sh emb 2>&1 \
  | tee $O/gate.txt; rc=${PIPESTATUS[0]}
echo "=== glm5 emb gate rc=$rc"
# the report again on the host (the gate wrote it from inside the image when it had python3)
if [ -f $O/timeline.csv ]; then
  # (the gate's own copy, timeline_report.txt, is root's: the image runs as root)
  python3 -I $D/glm5_timeline.py $O/timeline.csv --tokens 512 --layers --gantt 4 > $O/timeline_report_host.txt 2>&1
  echo "--- glm5_timeline.py rc=$? -> $O/timeline_report_host.txt"
  sed -n '/^=== steady state/,/^  gantt/p' $O/timeline_report_host.txt
fi
exit $rc
