#!/bin/bash
# isa_blocks.sh <kernel.s> -- per basic block that loads int8 (a Q8_0 block loop) or is a loop with fp32 loads: the float-op mix
awk '
function flush() { if (blk != "" && (c["global_load_i8"] > 0 || c["global_load_d16_u8"]>0 || c["v_fma_f32"]+c["v_fmac_f32_e32"]+c["v_dual_fmac_f32"] > 0)) {
    printf "%-10s i8=%d d16=%d ld32=%d cvt_i32=%d fma_mix=%d mul=%d dmul=%d fmac=%d fma=%d dfmac=%d add=%d dadd=%d bperm=%d br=%s\n", blk, c["global_load_i8"]+c["global_load_d16_u8"], c["global_load_d16_b16"], c["global_load_b32"], c["v_cvt_f32_i32_e32"], c["v_fma_mix_f32"], c["v_mul_f32_e32"], c["v_dual_mul_f32"], c["v_fmac_f32_e32"], c["v_fma_f32"], c["v_dual_fmac_f32"], c["v_add_f32_e32"], c["v_dual_add_f32"], c["ds_bpermute_b32"], br }
  delete c; br="" }
/^\.LBB/ { flush(); blk=$1; next }
/^\t[a-z_0-9]+/ { c[$1]++; if ($1 ~ /s_cbranch/) br=br $1 ":" $2 " " }
END { flush() }' "$1"
