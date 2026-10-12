#!/bin/bash
# glm_d18_accept_chain.sh -- D18 READ part 2, step 4 (DECODE-OPEN-ITEMS-PLAN section 9): how often is the NextN head's ONE-token draft right on our model
# (the served IQ3_XXS hybrid, which carries blk.45) and our kinds of text? llama.cpp's glm5next code loads the head and drafts with it
# (--spec-type draft-mtp); depth 1 (--spec-draft-n-max 1) is what a Maya-style pipeline uses. Greedy, so the acceptance is the head's top-1 agreement.
# ONE process only (an engine process beside it would exhaust host RAM, record §L5-D18-PART2). Speed is not measured here.
# Launch (rig idle, service OFF, reservation flag in place):
#   setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_d18_accept_chain.sh > ~/bench/glm_d18_accept_chain.log 2>&1 < /dev/null &
# Env: ACC_OUT, ACC_GEN (tokens a request, 256), ACC_NGL (18).
set -u
. "$HOME/bench/chain_preflight.sh"
O=${ACC_OUT:-$HOME/bench/franken/glm5/d18acc}; mkdir -p "$O"
BIN=/home/ronald/src/llama-glm53/build-hip/bin; IMG=rocm/dev-ubuntu-24.04:7.14.0-full
M=$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf
PR=$HOME/bench/franken/glm5/prose8400.txt; PORT=8093; GEN=${ACC_GEN:-256}
say_end() { docker rm -f d18acc >/dev/null 2>&1; echo "=== glm_d18_accept exit rc=$1 $(date -Is)"; exit "$1"; }
echo "=== glm_d18_accept start $(date -Is) llama=$(git -C ~/src/llama-glm53 log --oneline -1 | cut -c1-60)"
rig_quiet_wait 1800 || say_end 3
docker rm -f d18acc >/dev/null 2>&1
docker run --rm --name d18acc -p 127.0.0.1:$PORT:$PORT --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:$BIN -v /home/ronald:/home/ronald $IMG \
  $BIN/llama-server -m "$M" -ngl ${ACC_NGL:-18} --tensor-split 1,1,1 --split-mode layer --device ROCm0,ROCm1,ROCm2 -fa on -t 16 -tb 8 \
  --spec-type draft-mtp --spec-draft-n-max 1 --host 0.0.0.0 --port $PORT --parallel 1 --ctx-size 16384 --temp 0 > "$O/server.log" 2>&1 < /dev/null &
for i in $(seq 1 180); do curl -sf -m 5 http://127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
curl -sf -m 5 http://127.0.0.1:$PORT/health >/dev/null || { echo "server never became ready"; grep -aiE "error|mtp|nextn|draft" "$O/server.log" | tail -15 | cut -c1-200; say_end 4; }
grep -aiE "mtp|nextn|draft|spec" "$O/server.log" | head -12 | cut -c1-200
python3 -I - "$PORT" "$GEN" "$PR" "$O" <<'EOF'
import json, sys, urllib.request
port, gen, pr, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
prose = [int(x) for x in open(pr).read().split()[:2000]]   # the file holds TOKEN IDS (the engine's --tokens-file), not text
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:%s%s" % (port, path), json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
cases = [
  ("prose", "/completion", {"prompt": prose, "n_predict": gen, "temperature": 0, "cache_prompt": False}),
  ("chat", "/v1/chat/completions", {"messages": [{"role": "user", "content": "I am planning a week of cycling in the Dolomites in late September. What should I pack, how fit do I need to be, and which passes are worth it? Answer in a few paragraphs."}], "max_tokens": gen, "temperature": 0}),
  ("code", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M into total seconds, with unit tests using unittest. Code only."}], "max_tokens": gen, "temperature": 0}),
  ("chat_nothink", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Explain to a new colleague how a write-ahead log makes a database crash-safe. Be concrete."}], "max_tokens": gen, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}),
]
tot_n = tot_a = 0
for name, path, body in cases:
    try:
        r = post(path, body)
    except Exception as e:
        print("%-13s ERROR %s" % (name, e)); continue
    t = r.get("timings", {})
    n, a = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    tot_n += n; tot_a += a
    text = r.get("content") or (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    print("%-13s predicted %4s  drafts %4d  accepted %4d  rate %.3f  tok/s %.1f  | %s" % (name, t.get("predicted_n"), n, a, a / n if n else float("nan"),
          t.get("predicted_per_second", 0), text[:80].replace("\n", " ")))
    json.dump(r, open("%s/%s.json" % (out, name), "w"))
print("ALL drafts %d accepted %d rate %.3f" % (tot_n, tot_a, tot_a / tot_n if tot_n else float("nan")))
EOF
grep -a "draft acceptance" "$O/server.log" | tail -6 | cut -c1-200
say_end 0
