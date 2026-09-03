"""Add fmt=8 (E4M3 FP8, block-scaled 128x128) to the Vulkan dense matmul shader
and backend, mirroring the CPU bit-trick decode (quant.h) exactly.

Byte layout is IDENTICAL to fmt=1 (int8): one byte per weight element, so
rowwords()/cpu_rb reuse fmt=1's branch. Only the decode function and the
scale layout differ: fmt=8's scale is a 2D grid of [ceil(O/128), ceil(I/128)]
floats, shared across 128 consecutive OUTPUT rows AND 128 consecutive INPUT
columns (unlike fmt=4/7's per-row grouping), matching quant.h's matmul_fp8.
"""
comp = "/home/ronald/src/colibri/c/shaders/qmatmul.comp"
s = open(comp).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("""float mx4(uint word, int lane) {
    uint n = (word >> (uint(lane) * 4u)) & 0xfu;
    const float lut[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    float v = lut[n & 7u];
    return (n & 8u) != 0u ? -v : v;
}""",
    """float mx4(uint word, int lane) {
    uint n = (word >> (uint(lane) * 4u)) & 0xfu;
    const float lut[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    float v = lut[n & 7u];
    return (n & 8u) != 0u ? -v : v;
}
// fmt=8: E4M3-FN (Qwen3.8 experts) via pure bit manipulation, no table --
// matches quant.h's matmul_fp8/e4m3x8_to_f32x8 bit-for-bit (see that file's
// derivation and its exhaustive 256-code validation record). sign(1) exp(4,
// bias=7) mant(3); subnormal at exp==0 -> mant/512; NaN only at byte&0x7F==0x7F.
float e4m3(uint word, int lane) {
    uint b = (word >> (uint(lane) * 8u)) & 0xffu;
    uint sign = (b & 0x80u) << 24;
    uint exp4 = (b >> 3) & 0xFu;
    uint mant3 = b & 0x7u;
    if ((b & 0x7Fu) == 0x7Fu) return uintBitsToFloat(sign | 0x7FC00000u);
    if (exp4 == 0u) {
        float mag = float(mant3) * (1.0 / 512.0);
        return (sign != 0u) ? -mag : mag;
    }
    uint bits = sign | ((exp4 + 120u) << 23) | (mant3 << 20);
    return uintBitsToFloat(bits);
}""",
    "e4m3 decode")

rep("""        } else { // int4
            int words = (p.I + 7) / 8;""",
    """        } else if (p.fmt == 8) { // E4M3 FP8, 128x128-blocked scale
            int words = (p.I + 3) / 4;                 // 4 bytes/word, 1 byte/element
            int nblkI = (p.I + 127) / 128;
            int blkO = o / 128;
            uint sb = uint(blkO) * uint(nblkI);
            for (int wi = lane; wi < words; wi += sgsize) {
                uint pk = w[rowBase + uint(wi)]; int i0 = wi * 4;
                for (int k = 0; k < 4; k++) {
                    int i = i0 + k;
                    if (i < p.I) {
                        float xv = staged ? xsh[i] : x[xoff + i];
                        sum += xv * e4m3(pk, k) * scale[sb + uint(i / 128)];
                    }
                }
            }
        } else { // int4
            int words = (p.I + 7) / 8;""",
    "fmt=8 branch")

rep("""        float tot = subgroupAdd(sum);
        if (lane == 0) y[s * p.O + o] = (p.fmt == 5 || p.fmt == 4 || p.fmt == 7) ? tot : tot * scale[o];""",
    """        float tot = subgroupAdd(sum);
        if (lane == 0) y[s * p.O + o] = (p.fmt == 5 || p.fmt == 4 || p.fmt == 7 || p.fmt == 8) ? tot : tot * scale[o];""",
    "skip per-row scale for fmt=8")

open(comp, "w").write(s)
print("SHADER FP8 APPLIED")

# ------------------------------------------------------------------ backend
b = "/home/ronald/src/colibri/c/backend_vulkan.c"
bs = open(b).read()


def brep(old, new, tag):
    global bs
    assert old in bs, "MISS: " + tag
    assert bs.count(old) == 1, "NOT UNIQUE (%d): %s" % (bs.count(old), tag)
    bs = bs.replace(old, new)
    print("ok:", tag)


brep("""static int rowwords(int fmt, int I) {
    size_t rb = fmt == 1 ? (size_t)I                         // bytes/row on CPU side
              : fmt == 5 ? ((size_t)I + 63) / 64 * 24        // int3-g64: 24B per 64-group
              : (size_t)(I + 1) / 2;
    return (int)((rb + 3) / 4);                              // padded to uint32 (24|4: exact)
}""",
    """static int rowwords(int fmt, int I) {
    size_t rb = (fmt == 1 || fmt == 8) ? (size_t)I            // bytes/row on CPU side
              : fmt == 5 ? ((size_t)I + 63) / 64 * 24        // int3-g64: 24B per 64-group
              : (size_t)(I + 1) / 2;
    return (int)((rb + 3) / 4);                              // padded to uint32 (24|4: exact)
}""",
    "rowwords fmt=8")

brep("""static size_t scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (fmt == 4 || fmt == 7)
        return (size_t)O * (((size_t)I + gs - 1) / gs);   // per-group [O,ng]
    return (size_t)O;
}""",
    """static size_t scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (fmt == 4 || fmt == 7)
        return (size_t)O * (((size_t)I + gs - 1) / gs);   // per-group [O,ng]
    if (fmt == 8)                                          // 128x128-blocked [ceil(O/128),ceil(I/128)]
        return (((size_t)O + 127) / 128) * (((size_t)I + 127) / 128);
    return (size_t)O;
}""",
    "scale_floats fmt=8")

brep("""    if (fmt != 1 && fmt != 2 && fmt != 5 &&              /* fmt=4/7: word-aligned groups only */
        !((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0)) return 0;""",
    """    if (fmt != 1 && fmt != 2 && fmt != 5 && fmt != 8 &&  /* fmt=4/7: word-aligned groups only */
        !((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0)) return 0;""",
    "upload_tensor accepts fmt=8")

brep("""    if (fmt != 1 && fmt != 2 && fmt != 5 && fmt != 8 &&  /* fmt=4/7: word-aligned groups only */
        !((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    """    if (fmt != 1 && fmt != 2 && fmt != 5 && fmt != 8 &&  /* fmt=4/7: word-aligned groups only */
        !((fmt == 4 || fmt == 7) && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = (fmt == 1 || fmt == 8) ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    "upload_tensor cpu_rb fmt=8")

open(b, "w").write(bs)
print("BACKEND FP8 APPLIED")
