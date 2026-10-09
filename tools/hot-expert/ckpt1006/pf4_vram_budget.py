#!/usr/bin/env python3
"""pf4_vram_budget.py -- PF4's feasibility arithmetic: what a bigger prefill chunk costs in VRAM a card (no GPU, stdlib).

    python3 -I pf4_vram_budget.py [ctx] [free_mb_card0 free_mb_card1 free_mb_card2]

Sizes come from franken-engine `glm5_shapes.h` and the allocations in `Glm5Runner::Glm5Runner` (glm5_graph.cpp, every `Scratch` buffer is [Tm][width]) and
`moe_reserve` (glm5_gpu.inc, the two helper contexts a card keeps for the other two owners: x, g, u, h, d and the plan packet). Prints, for chunk 512 / 1024 / 2048,
the scratch and the helper buffers a card holds, the step from the previous chunk size, and the shortfall against the free VRAM of the shipped config
(default: the steady `vram_dev?_steady free_mb` lines of ~/bench/franken/glm5/pf9_main/gate_run.log, chunk 1024, 262 144 cells, --expert-gb 17: 796 / 882 / 522 MB).
Calibration (record §L5-PF4): the step 512 -> 1024 computed here is +1.4 GB; the free VRAM measured in §L5-GLM-CHUNK fell 0.91-0.98 GB across it with a different
staging ring, so the arithmetic may run up to ~25 % high; the report prints both the arithmetic and the arithmetic x 0.8.
"""
import sys

N_EMBD, HC, N_FF_DENSE, N_FF_EXP, N_EXPERT, K = 4096, 4, 12288, 2048, 288, 8
HC_DIM, HC_MIX = HC * N_EMBD, (2 + HC) * HC
KDA_INNER, KDA_QKV = 64 * 128, 3 * 64 * 128
Q_LORA, Q_WIDTH, KV_LORA, N_HEAD, O_WIDTH = 1536, 64 * 256, 512, 64, 64 * 256
IDX_DIM, IDX_N_HEAD, IDX_TOP_POOLS, KPOOL = 128, 32, 2048 // 4, 4

def scratch_floats_per_row(ctx):
    """the words a chunk row holds in every card's Scratch (4 B each; `isel` and `ids` are ints), by who uses them"""
    n_pool = ctx // KPOOL + 1
    common = [HC_DIM, HC_DIM, HC_MIX, HC, HC, HC * HC, N_EMBD, N_EMBD, N_EMBD, N_EMBD,       # H Hn mixes pre post comb x xn ao fo
              N_EXPERT * 3, K * 3, N_EMBD, N_FF_EXP * 3, N_EMBD, K, 2 * HC_DIM]              # rlog probs probs_b, wraw wnorm wsc, moe, sg su sh, sd, ids, hout x2
    kda = [KDA_QKV, KDA_QKV, 128, KDA_INNER, KDA_INNER, N_HEAD, 128, KDA_INNER, KDA_INNER, KDA_INNER]
    dsa = [Q_LORA, Q_WIDTH, KV_LORA, N_HEAD * KV_LORA, N_HEAD * KV_LORA, O_WIDTH, IDX_DIM, IDX_DIM, IDX_N_HEAD * IDX_DIM, IDX_N_HEAD, n_pool, IDX_TOP_POOLS]
    dense = [N_FF_DENSE] * 3
    moe_rows = [K * N_FF_EXP * 3, K * N_EMBD * 2]                                            # yg yu yh, yd ywt
    return {"common": sum(common), "kda only": sum(kda), "dsa only": sum(dsa), "dense ffn only": sum(dense), "moe rows": sum(moe_rows)}

def helper_bytes(T):
    """one helper context (a card keeps two: one per other owner): plan packet x2 banks, x rows, g/u/h (assignments x 2048), d (assignments x 4096)"""
    na, cpn = T * K, 864 + 2 * T * K
    return 2 * cpn * 4 + T * N_EMBD * 4 + 3 * na * N_FF_EXP * 4 + na * N_EMBD * 4

if __name__ == "__main__":
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
    free = [float(x) for x in sys.argv[2:5]] if len(sys.argv) >= 5 else [796.0, 882.0, 522.0]
    parts = scratch_floats_per_row(ctx)
    row = sum(parts.values()) * 4
    print(f"ctx {ctx}: scratch {row / 1e6:.3f} MB a chunk row")
    for k, v in parts.items():
        print(f"    {k:15s} {v * 4 / 1e6:6.3f} MB a row")
    alias = (parts["common"] + max(parts["kda only"], parts["dsa only"]) + max(parts["dense ffn only"], parts["moe rows"])) * 4
    print(f"    (if the attention-kind buffers and the ffn-kind buffers shared storage: {alias / 1e6:.3f} MB a row, -{(row - alias) / 1e6:.3f})")
    prev = None
    for T in (512, 1024, 2048):
        tot = row * T + 2 * helper_bytes(T)
        step = "" if prev is None else f"  step from {prev[0]}: +{(tot - prev[1]) / 1e9:.2f} GB (x0.8: +{0.8 * (tot - prev[1]) / 1e9:.2f})"
        print(f"chunk {T:4d}: scratch {row * T / 1e9:.2f} GB + helpers {2 * helper_bytes(T) / 1e9:.2f} GB = {tot / 1e9:.2f} GB a card{step}")
        prev = (T, tot)
    d = prev[1] - (row * 1024 + 2 * helper_bytes(1024))
    print(f"\nfree at the shipped config (chunk 1024), MB a card: {free}")
    for f, name in zip(free, ("card 0", "card 1", "card 2")):
        print(f"  {name}: chunk 2048 needs {0.8 * d / 1e6:.0f}..{d / 1e6:.0f} MB more, {f:.0f} MB free: short by {0.8 * d / 1e6 - f:.0f}..{d / 1e6 - f:.0f} MB")
