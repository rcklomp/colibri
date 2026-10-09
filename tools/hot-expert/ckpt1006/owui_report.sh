#!/bin/bash
# Per-request view of the gateway log: time, slot, prompt tokens, how many were REUSED, ttft, generated, total.
#   owui_report.sh [N]   (last N requests, default 20)
N=${1:-20}
awk '
  /REUSE [0-9]+ [0-9]+ [0-9]+/ { reused[$4]=$5 }
  /\[req\] id=/ {
    for (i=1;i<=NF;i++) { split($i,kv,"="); v[kv[1]]=kv[2] }
    id=v["id"]; r=(id in reused)?reused[id]:"?"
    printf "%s %s  req %-4s slot %s  prompt %6s tok  reused %6s  ttft %8s  gen %4s  total %8s  %s\n", $1, $2, id, v["slot"], v["prompt_tokens"], r, v["ttft"], v["gen"], v["total"], $NF
  }
' ~/glm53_server.log | tail -n "$N"
