"""Add fmt=8 (E4M3 FP8) support to upload_tensor_d2/d3 -- they are separate
copy-pasted functions from upload_tensor (dev0), and the earlier fmt=8 patch
(commit 6da3910) only touched dev0's copy. rowwords()/scale_floats() are
shared helpers and already handle fmt=8 correctly for all three devices;
only the per-device validation gate and cpu_rb computation were missed.

This is the root cause of qwen38's 3-GPU expert cache placing nothing on
dev2/dev3: coli_vk_tensor_ensure2/3 call upload_tensor_d2/3, whose own gate
rejected fmt=8 immediately, so every upload attempt failed at the first
check regardless of available VRAM.
"""
p = "/home/ronald/src/colibri/c/backend_vulkan.c"
s = open(p).read()


def rep(old, new, tag):
    global s
    assert old in s, "MISS: " + tag
    assert s.count(old) == 1, "NOT UNIQUE (%d): %s" % (s.count(old), tag)
    s = s.replace(old, new)
    print("ok:", tag)


rep("""static int upload_tensor_d2(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 &&
        !(fmt == 4 && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    t->dev = 1;
    size_t stride = (size_t)t->rowWords * 4;
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    """static int upload_tensor_d2(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 && fmt != 8 &&
        !(fmt == 4 && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    t->dev = 1;
    size_t stride = (size_t)t->rowWords * 4;
    size_t cpu_rb = (fmt == 1 || fmt == 8) ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    "dev2 fmt=8 gate + cpu_rb")

rep("""static int upload_tensor_d3(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 &&
        !(fmt == 4 && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    t->dev = 2;
    size_t stride = (size_t)t->rowWords * 4;
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    """static int upload_tensor_d3(ColiVkTensor **out, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 && fmt != 8 &&
        !(fmt == 4 && gs >= 8 && gs % 8 == 0)) return 0;
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 7) ? gs : 0;
    t->dev = 2;
    size_t stride = (size_t)t->rowWords * 4;
    size_t cpu_rb = (fmt == 1 || fmt == 8) ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;""",
    "dev3 fmt=8 gate + cpu_rb")

open(p, "w").write(s)
print("DEV2/DEV3 FP8 UPLOAD APPLIED")
