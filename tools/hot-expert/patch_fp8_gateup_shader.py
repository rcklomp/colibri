"""Add fmt=8 (E4M3 FP8) to the fused gate+up+SiLU shader, mirroring the plain
qmatmul.comp fmt=8 branch added earlier (same bit-decode formula, same
128x128-blocked scale layout). This is the shader eg_prepare_submit's phase 1
uses for every expert in a batch; phase 2 (down) already reuses the plain
qmatmul.comp shader, which already has fmt=8 support.
"""
comp = "/home/ronald/src/colibri/c/shaders/qmatmul_gate_up.comp"
s = open(comp).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("""float i8(uint w, int l) { int b = int((w >> (uint(l) * 8u)) & 0xffu); if (b >= 128) b -= 256; return float(b); }
float i4(uint w, int l) { return float(int((w >> (uint(l) * 4u)) & 0xfu) - 8); }""",
    """float i8(uint w, int l) { int b = int((w >> (uint(l) * 8u)) & 0xffu); if (b >= 128) b -= 256; return float(b); }
float i4(uint w, int l) { return float(int((w >> (uint(l) * 4u)) & 0xfu) - 8); }
// fmt=8: E4M3-FN, identical formula to qmatmul.comp's e4m3() -- see that file
// for the derivation and the exhaustive 256-code validation record.
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
    "e4m3 decode in gate_up shader")

rep("""        } else { // int4
            int words = (p.I + 7) / 8;
            for (int wi = lane; wi < words; wi += sgsize) {
                uint gp = wg[rb + uint(wi)], upk = wu[rb + uint(wi)]; int i0 = wi * 8;
                for (int k = 0; k < 8; k++) { int i = i0 + k; if (i < p.I) { float xv = xsh[i]; g += xv * i4(gp, k); u += xv * i4(upk, k); } }
            }
        }""",
    """        } else if (p.fmt == 8) { // E4M3 FP8, 128x128-blocked scale, gate/up share the layout
            int words = (p.I + 3) / 4;
            int nblkI = (p.I + 127) / 128;
            int blkO = o / 128;
            uint sb = uint(blkO) * uint(nblkI);
            for (int wi = lane; wi < words; wi += sgsize) {
                uint gp = wg[rb + uint(wi)], upk = wu[rb + uint(wi)]; int i0 = wi * 4;
                for (int k = 0; k < 4; k++) {
                    int i = i0 + k;
                    if (i < p.I) {
                        float xv = xsh[i];
                        float sc = gscale[sb + uint(i / 128)];   // gate/up scale grids are the same shape
                        g += xv * e4m3(gp, k) * sc;
                        u += xv * e4m3(upk, k) * uscale[sb + uint(i / 128)];
                    }
                }
            }
        } else { // int4
            int words = (p.I + 7) / 8;
            for (int wi = lane; wi < words; wi += sgsize) {
                uint gp = wg[rb + uint(wi)], upk = wu[rb + uint(wi)]; int i0 = wi * 8;
                for (int k = 0; k < 8; k++) { int i = i0 + k; if (i < p.I) { float xv = xsh[i]; g += xv * i4(gp, k); u += xv * i4(upk, k); } }
            }
        }""",
    "fmt=8 branch in gate_up")

rep('        if (p.fmt != 5 && p.fmt != 4) { gt *= gscale[o]; ut *= uscale[o]; }   // per-row formats only',
    '        if (p.fmt != 5 && p.fmt != 4 && p.fmt != 8) { gt *= gscale[o]; ut *= uscale[o]; }   // per-row formats only',
    "skip per-row scale for fmt=8 in gate_up")

open(comp, "w").write(s)
print("GATE_UP FP8 APPLIED")

# ------------------------------------------------------------------ backend
# eg_prepare_submit needs no fmt-specific gate beyond what upload_tensor
# already validates (it checks D<=6144 and cross-expert consistency, both
# fmt-agnostic), so no backend change is required here -- only the shader.
