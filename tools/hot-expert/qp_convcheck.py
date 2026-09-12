#!/usr/bin/env python3
"""qp_convcheck.py -- close the loop from QP's simulations to the CONVERTER.

rome_q4sim.c proves that c/q38_sim.h computes what the tree's own int4-g64
packer and quantize_rows compute. That leaves one link: is the tree's packer the
same quantiser the CONVERTER would use when Q8/Q7 actually writes the weights?
This script answers it by IMPORTING the converter's own functions --
quant_int4_grouped and quant_int8 from c/tools/convert_fp8_to_int4.py -- and
running them on the exact float arrays the C harness quantised, then comparing
levels and scales bit for bit.

That is the step §G15's method insists on: "against the tree's OWN pipeline
rather than against a restatement of the same formula". A restatement in this
file would prove nothing.

Usage (on the rig, after `rome_q4sim /tmp/qpdump`):
    python3 tools/hot-expert/qp_convcheck.py /tmp/qpdump 256 2560
Exit 0 = every level and every scale identical.
"""
import importlib.util
import os
import sys

import numpy as np

O = int(sys.argv[2]) if len(sys.argv) > 2 else 256
I = int(sys.argv[3]) if len(sys.argv) > 3 else 2560
GS = 64
d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qpdump"

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv_path = os.path.join(repo, "c", "tools", "convert_fp8_to_int4.py")
spec = importlib.util.spec_from_file_location("coli_conv", conv_path)
conv = importlib.util.module_from_spec(spec)
# The converter is a CLI script; importing it must not run main(). It guards on
# __name__, so a plain exec of the module body is safe and gives us the real
# functions rather than copies of them.
spec.loader.exec_module(conv)
print(f"imported quant_int4_grouped / quant_int8 from {conv_path}")

fails = 0


def rd(name, dtype):
    return np.fromfile(os.path.join(d, name), dtype=dtype)


deq = rd("qp_deq_f32.bin", np.float32).reshape(O, I)
sim_lev = rd("qp_i4_levels_i8.bin", np.int8).reshape(O, I).astype(np.int32)
sim_sc = rd("qp_i4_scales_f32.bin", np.float32).reshape(O, I // GS)

# --- (a) int4-g64: the converter's own quant_int4_grouped at gs=64 -----------
qb, sc = conv.quant_int4_grouped(deq, 4, gs=GS)
qb = np.asarray(qb).reshape(O, (I + 1) // 2)
sc = np.asarray(sc, dtype=np.float32).reshape(O, I // GS)
# unpack the converter's nibbles back to levels: LOW nibble = even element
conv_lev = np.empty((O, I), dtype=np.int32)
conv_lev[:, 0::2] = (qb & 0xF).astype(np.int32) - 8
conv_lev[:, 1::2] = (qb >> 4).astype(np.int32) - 8

bad_lev = int((conv_lev != sim_lev).sum())
bad_sc = int((sc.view(np.uint32) != sim_sc.view(np.uint32)).sum())
print(f"(a) int4-g64 vs converter quant_int4_grouped(gs=64): "
      f"levels {bad_lev} of {O*I} differ, scales {bad_sc} of {O*(I//GS)} differ")
if bad_lev or bad_sc:
    fails += 1
    wr = np.argwhere(conv_lev != sim_lev)[:5]
    for r, c in wr:
        print(f"    [{r},{c}] deq={deq[r,c]!r} sim={sim_lev[r,c]} conv={conv_lev[r,c]} "
              f"step_sim={sim_sc[r,c//GS]!r} step_conv={sc[r,c//GS]!r}")

# --- (b)/(c) int8 per row: the converter's own quant_int8 -------------------
W = rd("qp_bf16_u16.bin", np.uint16).reshape(O, I)
wf = (W.astype(np.uint32) << 16).view(np.float32)
sim8 = rd("qp_i8row_levels_i8.bin", np.int8).reshape(O, I).astype(np.int32)
sim8s = rd("qp_i8row_scales_f32.bin", np.float32).reshape(O, 1)

q8, s8 = conv.quant_int8(wf, 8)
# quant_int8 returns its int8 levels VIEWED AS uint8 (that is the on-disk byte
# order for fmt=1); view them back before comparing, or every negative level
# reads as 256+level and about half the array "differs" for no reason.
q8 = np.asarray(q8).view(np.int8).reshape(O, I).astype(np.int32)
s8 = np.asarray(s8, dtype=np.float32).reshape(O, 1)
bad8 = int((q8 != sim8).sum())
bad8s = int((s8.view(np.uint32) != sim8s.view(np.uint32)).sum())
print(f"(b/c) int8 per-row vs converter quant_int8(8): "
      f"levels {bad8} of {O*I} differ, scales {bad8s} of {O} differ")
if bad8 or bad8s:
    fails += 1

# --- (b)/(c) int8 g64: quant_int4_grouped's grouping with int8's qmax -------
# The converter has no grouped int8 entry point, so this arm is checked against
# numpy written to the SAME shape as quant_int4_grouped's body (amax over the
# group along the input dim, step = max(amax/qmax, 1e-8), rint, clip) with
# qmax=127. Stated as a limitation rather than glossed: the per-row arm above is
# the one pinned to the converter itself.
sim8g = rd("qp_i8g64_levels_i8.bin", np.int8).reshape(O, I).astype(np.int32)
sim8gs = rd("qp_i8g64_scales_f32.bin", np.float32).reshape(O, I // GS)
wr = wf.reshape(O, I // GS, GS)
amax = np.abs(wr).max(axis=2, keepdims=True)
step = np.maximum(amax / np.float32(127.0), np.float32(1e-8)).astype(np.float32)
ref = np.clip(np.rint(wr / step), -128, 127).astype(np.int32).reshape(O, I)
badg = int((ref != sim8g).sum())
badgs = int((step.reshape(O, I // GS).view(np.uint32) != sim8gs.view(np.uint32)).sum())
print(f"(b/c) int8 g64 vs quant_int4_grouped's grouping at qmax=127: "
      f"levels {badg} of {O*I} differ, scales {badgs} of {O*(I//GS)} differ")
if badg or badgs:
    fails += 1

print("CONVCHECK " + ("FAILED" if fails else "PASSED"))
sys.exit(1 if fails else 0)
