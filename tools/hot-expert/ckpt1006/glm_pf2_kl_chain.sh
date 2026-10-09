#!/bin/bash
# glm_pf2_kl_chain.sh -- PF2 (ACTION-LIST row 5g): how far does the f16 trunk GEMM (`--gemm-lds 3`) move the NEXT-TOKEN DISTRIBUTION of the hybrid, against two yardsticks?
#   mode 0 = k_gemm_batch (the decode kernel's arithmetic, bit-exact to the eager one-token path), mode 1 = the shipped LDS f32 GEMM (a reassociated sum), mode 3 = the f16 arm.
# 24 prompt windows (16 x 1 024 tokens: the prose file and the technical file, plus 8 x 512 at other offsets), each prefilled at chunk 1 024 in ONE process per mode, with the engine's dump of the
# final logits (--dump). Then, per window, in float64: KL(p_ref || p_mode), the top-1 token and whether it is the same, the cosine of the logits and the largest logit difference, for
# KL(0||1) (what the shipped arm already costs) and KL(0||3) and KL(1||3). The stop rule in words: mode 3 must cost no more than a small multiple of what mode 1 costs, and keep the top-1.
# Launch (rig idle, service stopped): setsid nohup ~/src/colibri/tools/hot-expert/run_chain.sh ~/bench/glm_pf2_kl_chain.sh > ~/bench/glm_pf2_kl_chain.log 2>&1 < /dev/null &
# Env: PF2K_D, PF2K_BIN (franken_decode_glm_f16), PF2K_MODEL, PF2K_OUT (default ~/bench/franken/glm5/pf2_kl; the dumps are ~50 MB a window and are deleted at the end (through docker: they are root-owned) unless PF2K_KEEP=1).
set -u
. "$HOME/bench/chain_preflight.sh"
D=${PF2K_D:-$HOME/src/franken-engine/franken/decode}; BIN=$D/${PF2K_BIN:-franken_decode_glm_f16}
O=${PF2K_OUT:-$HOME/bench/franken/glm5/pf2_kl}; mkdir -p "$O"
M=${PF2K_MODEL:-$HOME/models/GLM-5.3-Flash/hybrid-HYB-IQ3XXS/GLM-5.3-Flash-HYB-IQ3XXS-00001-of-00001.gguf}
P=$HOME/bench/m2/glm; B=$HOME/bench/franken/glm5; PR=$B/prose8400.txt; REC=$B/rec_depth/rec_ids.txt
say_end() { echo "=== glm_pf2_kl exit rc=$1 $(date -Is)"; exit "$1"; }
for f in "$BIN" "$M" "$PR" "$REC"; do [ -e "$f" ] || { echo "FATAL: missing $f"; say_end 2; }; done
echo "=== glm_pf2_kl start $(date -Is) bin=$(sha256sum "$BIN" | cut -c1-16)"
rig_quiet_wait 1800 || say_end 3
python3 -I - "$PR" "$REC" "$O" <<'PY'
import sys
pr, rec, o = sys.argv[1:4]
src = {"p": open(pr).read().split(), "t": open(rec).read().split()}
n = 0
for k, ids in src.items():
    for i in range(8):                                   # 8 x 1 024, evenly spread
        off = i * ((len(ids) - 1024) // 7)
        open(f"{o}/win{n:02d}.txt", "w").write(" ".join(ids[off:off + 1024])); n += 1
    for i in range(4):                                   # 4 x 512 at odd offsets
        off = 333 + i * ((len(ids) - 1000) // 4)
        open(f"{o}/win{n:02d}.txt", "w").write(" ".join(ids[off:off + 512])); n += 1
print("windows:", n)
PY
NW=$(ls "$O"/win*.txt | wc -l)
dk() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 \
         -e FRANKEN_GLM_MOE_G=8 -e FRANKEN_GEMM_LDS=1 -e FRANKEN_GEMV_FUSED_REDUCE=0 -e FRANKEN_GLM_FETCH_ASSIGN=optimal -e FRANKEN_GLM_GEMV_GROUP=1 -e FRANKEN_GEMV_ROWSPLIT=1 \
         -e FRANKEN_GEMV_ROWSPLIT_WAVES=1 -e FRANKEN_GEMV_Q8FAST=2 -e FRANKEN_GLM_HELP_COPY=1 \
         -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w /home/ronald/bench rocm/dev-ubuntu-24.04:7.14.0-full "$@"; }
for G in 0 1 3; do
  PL=$O/plan_g$G.txt; CK=$O/checks_g$G.txt; : > "$PL"; : > "$CK"
  for W in "$O"/win*.txt; do n=$(basename "$W" .txt); echo "${n}_g$G --tokens-file $W --ctx 4096 --chunk 1024 --adapt-prefill 0 --glm-help-copy 1 --gemm-lds $G --greedy 1 --dump $O/dump_g${G}_$n" >> "$PL"; echo "${n}_g$G ref" >> "$CK"; done
  echo "=== mode $G: $NW windows $(date +%T)"
  dk "$BIN" --model "$M" --placement "$P" --expert-gb 17 --gate-plan "$PL" > "$O/gate_g$G.log" 2>&1
  rc=$?; echo "[g$G] engine rc=$rc"
  grep -aE "HIP error|Memory access|out of memory|FATAL|gemm-lds 3:|rocBLAS|rocblas" "$O/gate_g$G.log" | sort -u | head -4 | cut -c1-200
done
python3 -I - "$O" <<'PY'
import glob, math, os, struct, sys, array, statistics as st
O = sys.argv[1]
def load(d):
    fs = glob.glob(d + "/result_output*.f32")
    if not fs: return None
    a = array.array("f"); a.frombytes(open(fs[0], "rb").read()); return a
def lsm(a):
    m = max(a); s = math.log(sum(math.exp(x - m) for x in a)) + m
    return [x - s for x in a]
def cmp(ra, rb):
    la, lb = lsm(ra), lsm(rb)
    kl = sum(math.exp(x) * (x - y) for x, y in zip(la, lb))
    ia = max(range(len(ra)), key=ra.__getitem__); ib = max(range(len(rb)), key=rb.__getitem__)
    dot = sum(x * y for x, y in zip(ra, rb)); na = math.sqrt(sum(x * x for x in ra)); nb = math.sqrt(sum(y * y for y in rb))
    return kl, ia == ib, dot / (na * nb), max(abs(x - y) for x, y in zip(ra, rb))
wins = sorted(os.path.basename(f)[:-4] for f in glob.glob(O + "/win*.txt"))
rows = {"0|1": [], "0|3": [], "1|3": []}
for w in wins:
    L = {g: load(f"{O}/dump_g{g}_{w}") for g in (0, 1, 3)}
    if any(v is None for v in L.values()): print("missing logits for", w); continue
    for k in rows:
        a, b = k.split("|"); rows[k].append(cmp(L[int(a)], L[int(b)]))
print("=== final-position logits, %d windows (1 024 or 512 tokens), float64 softmax over the full vocabulary" % len(wins))
print("%-8s %12s %12s %12s  %s  %s %s" % ("pair", "KL mean", "KL median", "KL max", "top-1 same", "cos min", "max |dlogit|"))
for k, r in rows.items():
    if not r: continue
    kls = [x[0] for x in r]
    print("%-8s %12.3e %12.3e %12.3e  %d/%d        %.8f  %.3f" % ("KL(%s)" % k, st.mean(kls), st.median(kls), max(kls), sum(1 for x in r if x[1]), len(r), min(x[2] for x in r), max(x[3] for x in r)))
PY
[ "${PF2K_KEEP:-0}" = 1 ] || docker run --rm -v /home/ronald:/home/ronald rocm/dev-ubuntu-24.04:7.14.0-full bash -c "rm -rf $O/dump_g*"   # the engine container wrote them as root
say_end 0
