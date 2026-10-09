#!/bin/bash
# Check 4 in isolation, N cycles: a client abandons a ~1300-token request (curl -m 8), then a
# short request must be answered. Prints the wait and whether the engine logged a CANCEL.
set -u
N=${1:-3}; K=$(cat ~/.colibri_api_key); URL=http://127.0.0.1:8081
BIG=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 130)")
for i in $(seq 1 $N); do
  # wait for an idle engine so the run measures the cancel path, not a backlog
  for _ in $(seq 1 60); do
    p=$(grep -c "POST /v1/chat/completions" ~/glm53_server.log); r=$(grep -c "\[req\] " ~/glm53_server.log)
    [ "$p" -le "$r" ] && break; sleep 5
  done
  before=$(grep -c "CANCEL" ~/glm53_server.log)
  body=$(python3 -c "import json,sys; print(json.dumps({'model':'glm-5.3-flash','messages':[{'role':'user','content':sys.argv[1]+' Summarise in one word. [run $i]'}],'max_tokens':32}))" "$BIG")
  curl -s -m 8 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" $URL/v1/chat/completions -d "$body" || true
  t0=$(date +%s.%N)
  curl -s -m 400 -o /dev/null -H "Authorization: Bearer $K" -H "Content-Type: application/json" $URL/v1/chat/completions \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'
  t=$(python3 -c "print(round($(date +%s.%N) - $t0, 1))")
  after=$(grep -c "CANCEL" ~/glm53_server.log)
  echo "run $i: next request answered in ${t}s, engine CANCEL lines +$((after - before))"
done
