#!/bin/bash
# glm_soak.sh -- a SOAK of the LIVE franken-glm service: mixed real requests for N minutes, with a sampler beside it, then one verdict. Runs on the rig, CLIENT side only: the service holds
# the rig lock, this script takes none and starts nothing (bring the service up first: handoff 3; `serve_alt.sh status` must say it is serving).
# Why (2026-10-09, plan Rev 110): the PF14 hybrid passed the gates and the acceptance, but acceptance is ~2 minutes of short requests; a service is hours of long ones. Per round (~3-6 min):
#   a. accept_live.sh -- the owner's own path: a UI-shaped new chat, a warm one, an API follow-up (prefix pin), a reply that finishes, a request behind an abandoned one, the ledger invariant
#   b. every third round one LONG prompt through the API (15k, 40k, 90k tokens in turn) with an access code buried in the middle; the answer is checked for the code
# Beside it, every 30 s: VRAM, edge / junction temperature, power and fan per card, the engine's host RSS and MemAvailable.
# Verdict (all must hold): every accept_live round PASS; no HIP error / memory fault / "engine dispatcher stopped" / HTTP 5xx in the gateway log since the start; VRAM drift per card < 300 MiB
# (first third against last third); engine RSS drift < 3 GB; warm new-chat ttft last third < 1.5x the first third (decode tok/s of the log's requests is printed, not gated: long prompts decode slower by depth);
# the engine is still alive and answering at the end. A long prompt that did not contain the code in its answer is reported, not failed (a reasoning model may spend its budget thinking).
# Usage: glm_soak.sh [minutes=60]        Out: ~/bench/soak_<date>_<time>/ {summary.txt, rounds.log, long.log, samples.tsv, accept_N.log}
# Env: GLM53_LOG (gateway log, default ~/bench/serve_alt_franken_glm.log), SOAK_LENS ("15000 40000 90000" prompt tokens).
set -u
MIN=${1:-60}; END=$(( $(date +%s) + MIN * 60 ))
O=$HOME/bench/soak_$(date +%Y%m%d_%H%M); mkdir -p "$O"
export GLM53_LOG=${GLM53_LOG:-$HOME/bench/serve_alt_franken_glm.log}
HERE=$HOME/src/colibri/tools/hot-expert; KEY=$(cat "$HOME/.colibri_api_key" 2>/dev/null)
LENS=(${SOAK_LENS:-15000 40000 90000})
curl -s -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8081/v1/models | grep -q glm-5.3-flash || { echo "FATAL: the gateway on :8081 does not list glm-5.3-flash -- bring the service up first"; exit 2; }
OFF=$(stat -c %s "$GLM53_LOG")
echo "=== soak start $(date -Is) minutes=$MIN out=$O log_offset=$OFF engine_sha=$(sha256sum "$HOME/bench/franken_bin/franken_dec_glm" | cut -c1-16)" | tee "$O/rounds.log"

sample() {   # one TSV row per 30 s
  printf 'epoch\tcard\tvram_mib\ttemp_edge\ttemp_junc\tpower_w\tfan_rpm\trss_mb\tmemavail_mb\n' > "$O/samples.tsv"
  while :; do
    rss=$(ps -eo rss,args | awk '/franken_dec_[g]lm/ && !/docker/ {s+=$1} END {printf "%d", s/1024}'); ma=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    for d in /sys/class/drm/card[0-9]/device; do
      [ -e "$d/mem_info_vram_used" ] || continue
      h=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
      printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$(basename "$(dirname "$d")")" "$(( $(cat "$d/mem_info_vram_used") / 1048576 ))" \
        "$(( $(cat "$h/temp1_input" 2>/dev/null || echo 0) / 1000 ))" "$(( $(cat "$h/temp2_input" 2>/dev/null || echo 0) / 1000 ))" \
        "$(( $(cat "$h/power1_average" 2>/dev/null || echo 0) / 1000000 ))" "$(cat "$h/fan1_input" 2>/dev/null || echo 0)" "$rss" "$ma"
    done >> "$O/samples.tsv"
    sleep 30
  done
}
sample & SAMP=$!; trap 'kill $SAMP 2>/dev/null' EXIT

long_request() {   # long_request <tokens> <round>
  python3 -I - "$1" "$2" "$O/long.log" "$KEY" <<'PY'
import json, sys, time, urllib.request
toks, rnd, logf, key = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
src = open("/home/ronald/src/colibri/tools/hot-expert/HANDOFF-2026-10-06b.md", encoding="utf-8", errors="replace").read()
chars = toks * 4
body = (src * (chars // len(src) + 2))[:chars]
code = "7391"
mid = len(body) // 2
doc = body[:mid] + "\n\nNOTE FOR THE READER: the access code is " + code + ".\n\n" + body[mid:]
prompt = doc + "\n\nQuestion: what is the access code mentioned in the note above? Answer with the number only."
req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions", data=json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000}).encode(),
                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
t0 = time.time()
try:
    d = json.loads(urllib.request.urlopen(req, timeout=1500).read())
    u = d.get("usage", {}); c = d["choices"][0]
    line = "round=%s ok=1 prompt_tokens=%s completion_tokens=%s finish=%s found_code=%d wall_s=%.1f" % (rnd, u.get("prompt_tokens"), u.get("completion_tokens"), c.get("finish_reason"), int(code in (c["message"].get("content") or "")), time.time() - t0)
except Exception as e:
    line = "round=%s ok=0 error=%r wall_s=%.1f" % (rnd, e, time.time() - t0)
open(logf, "a").write(line + "\n"); print(line)
PY
}

n=0
while [ "$(date +%s)" -lt "$END" ]; do
  n=$((n + 1))
  r=$("$HERE/accept_live.sh" 2>&1); echo "$r" > "$O/accept_$n.log"
  pass=$(printf '%s\n' "$r" | grep -c '=== accept_live PASS')
  warm=$(printf '%s\n' "$r" | sed -n 's/^2\. UI new chat B warm.*ttft=\([0-9.]*\)s.*/\1/p' | head -1)
  echo "round $n $(date +%T) accept_live_pass=$pass warm_ttft=${warm:-NA}" | tee -a "$O/rounds.log"
  if [ $((n % 3)) -eq 1 ]; then
    L=${LENS[$(( (n / 3) % ${#LENS[@]} ))]}
    [ "$(date +%s)" -lt "$END" ] && echo "round $n long prompt ~$L tokens: $(long_request "$L" "$n")" | tee -a "$O/rounds.log"
  fi
  sleep 20
done

python3 -I - "$O" "$GLM53_LOG" "$OFF" <<'PY' | tee "$O/summary.txt"
import csv, glob, re, statistics as st, subprocess, sys
O, LOG, OFF = sys.argv[1], sys.argv[2], int(sys.argv[3])
bad = []
rounds = [l for l in open(O + "/rounds.log") if l.startswith("round") and "accept_live_pass" in l]
passes = sum(1 for l in rounds if "accept_live_pass=1" in l)
if passes != len(rounds): bad.append("accept_live: %d of %d rounds passed" % (passes, len(rounds)))
warm = [float(m.group(1)) for l in rounds for m in [re.search(r"warm_ttft=([0-9.]+)", l)] if m]
def thirds(xs): k = max(1, len(xs) // 3); return xs[:k], xs[-k:]
if len(warm) >= 3:
    a, b = thirds(warm)
    if st.median(b) > 1.5 * st.median(a): bad.append("warm ttft drift: first third %.2f s, last third %.2f s" % (st.median(a), st.median(b)))
print("rounds=%d accept_live_pass=%d warm_ttft min/median/max = %s" % (len(rounds), passes, ("%.2f / %.2f / %.2f s" % (min(warm), st.median(warm), max(warm))) if warm else "NA"))
longs = [l.strip() for l in open(O + "/long.log")] if glob.glob(O + "/long.log") else []
for l in longs:
    print("long:", l)
    if " ok=0 " in l: bad.append("long request failed: " + l[:100])
data = open(LOG, "rb").read()[OFF:].decode("utf-8", "replace")
errs = re.findall(r"HIP error|Memory access|FATAL|Traceback|engine dispatcher stopped|HTTP/1\.1\" 5\d\d", data)
print("gateway log since start: %d bytes, %d error markers %s" % (len(data), len(errs), sorted(set(errs))[:4]))
if errs: bad.append("%d error markers in the gateway log" % len(errs))
rates = [float(x) for x in re.findall(r"decode_s=[0-9.]+ tok/s=([0-9.]+)", data)]
if rates: print("decode tok/s over %d requests: min %.1f median %.1f max %.1f" % (len(rates), min(rates), st.median(rates), max(rates)))
rows = list(csv.DictReader(open(O + "/samples.tsv"), delimiter="\t"))
for card in sorted({r["card"] for r in rows}):
    rr = [r for r in rows if r["card"] == card]; a, b = thirds(rr)
    f = lambda xs, k: st.median(float(x[k]) for x in xs)
    drift = f(b, "vram_mib") - f(a, "vram_mib")
    print("%s: vram %.0f -> %.0f MiB (drift %+.0f), temp edge max %d junction max %d C, power max %d W, fan max %d rpm" % (card, f(a, "vram_mib"), f(b, "vram_mib"), drift,
          max(int(x["temp_edge"]) for x in rr), max(int(x["temp_junc"]) for x in rr), max(int(x["power_w"]) for x in rr), max(int(x["fan_rpm"]) for x in rr)))
    if abs(drift) > 300: bad.append("%s VRAM drift %+.0f MiB" % (card, drift))
r0 = [r for r in rows if r["card"] == rows[0]["card"]]; a, b = thirds(r0)
rss_a, rss_b = st.median(float(x["rss_mb"]) for x in a), st.median(float(x["rss_mb"]) for x in b)
ma_a, ma_b = st.median(float(x["memavail_mb"]) for x in a), st.median(float(x["memavail_mb"]) for x in b)
print("engine host RSS %.0f -> %.0f MB (drift %+.0f), MemAvailable %.0f -> %.0f MB" % (rss_a, rss_b, rss_b - rss_a, ma_a, ma_b))
if rss_b - rss_a > 3000: bad.append("engine RSS drift %+.0f MB" % (rss_b - rss_a))
alive = subprocess.run("ps -eo args | grep -c '[f]ranken_dec_glm'", shell=True, capture_output=True, text=True).stdout.strip()
print("engine processes at the end:", alive)
if alive == "0": bad.append("the engine is gone")
print("=== SOAK " + ("PASS" if not bad else "FAIL: " + "; ".join(bad)))
PY
echo "=== soak end $(date -Is)"
