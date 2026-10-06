#!/bin/bash
# build_ds4_ab.sh -- build franken_decode_ds4 at the commit BEFORE the shared ds4_gpu.inc change
# (473c5cc, the counter read-back kernel) and at the tip (b5cf4e0), same toolchain, for an A,B,B,A
# gate. Build only: no GPU is touched. Worktree is temporary and removed at the end.
set -u
SRC=$HOME/src/franken-engine
WT=$HOME/src/franken-engine-ds4ab
IMG=rocm/dev-ubuntu-24.04:7.14.0-full
OUT=$HOME/bench/franken_bin
log() { echo "[build_ds4_ab] $(date +%T) $*"; }
git -C "$SRC" worktree add --detach "$WT" 5f409c2 || exit 1
build() {   # build <rev> <outname>
  git -C "$WT" checkout -q --detach "$1" || return 1
  log "building $1 -> $2"
  ( cd "$WT/franken/decode" && nice -n 19 make -j4 ds4-gpu DS4_GPU_BIN=$2 2>&1 | grep -vE "^docker run|^\s+\.\./|^\s*$" | tail -6 )
  [ -x "$WT/franken/decode/$2" ] && cp -p "$WT/franken/decode/$2" "$OUT/$2" && sha256sum "$OUT/$2" | cut -c1-16
}
build 5f409c2 franken_decode_ds4.pre473
git -C "$WT" clean -fdxq
build b5cf4e0 franken_decode_ds4.tip
git -C "$SRC" worktree remove --force "$WT"
log "done"; ls -la "$OUT" | grep ds4
