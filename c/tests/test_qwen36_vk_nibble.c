/* test_qwen36_vk_nibble.c -- V1 step 1: the int4 staging transformation, proved
 * exhaustively rather than reasoned about.
 *
 * qwen36's packed experts live in RAM as TWO'S-COMPLEMENT nibbles: qwen36.c's
 * unpack_int4_to_int8 sign-extends (`(int8_t)(byte<<4)>>4`), and the CPU expert
 * GEMV consumes that. Every Colibri Vulkan shader decodes fmt=2/4 weights as
 * OFFSET BINARY instead:
 *
 *     float i4(uint w, int l) { return float(int((w >> (uint(l)*4u)) & 0xfu) - 8); }
 *         c/shaders/qmatmul.comp:47, qmatmul_tile.comp:48,
 *         qmatmul_gate_up.comp:33, qmatmul_gate_up_tile.comp:27
 *
 * and c/backend_vulkan.c's upload_tensor copies the bytes verbatim, so the
 * conversion has to happen host-side. That is what qwen36_tier_vk.c's stage()
 * XOR 0x88 does -- the same XOR the CUDA tier does for the same reason (CUDA's
 * upload runs offset_to_signed_s4 on the device, backend_cuda.cu:1428, so its
 * API boundary is offset binary too).
 *
 * tools/hot-expert/V1-STEP0-2026-09-16.md sections (c)/(d)/(e) instruct the
 * opposite -- "the stage() XOR step must be DROPPED, not ported" -- from
 * reading i4()'s `- 8` as a sign extension. It is not. This test is the
 * refutation, and it is written so that dropping the XOR fails it: case 2
 * asserts the un-XOR'd path disagrees.
 *
 * Exhaustive over all 256 byte values (both nibbles), plus one full-size
 * random matrix through the real unpack path. No model, no GPU, no Vulkan.
 */
#define QWEN36_NO_MAIN 1
#include "../qwen36.c"

#include <stdio.h>
#include <stdlib.h>

/* The GLSL i4() above, transcribed for one byte's low/high nibble. `lane` is
 * the element index within the byte: LOW nibble = even element, exactly the
 * packing order qwen36.c and c/tools/convert_qwen36.py use. */
static int i4_shader(unsigned char byte, int lane) {
    unsigned w = (unsigned)byte;
    return (int)((w >> (unsigned)(lane * 4)) & 0xfu) - 8;
}

static int fails = 0;
static void check(int cond, const char *what, int a, int b, int byte) {
    if (!cond) {
        if (fails < 10)
            fprintf(stderr, "FAIL %s: byte 0x%02x -> cpu %d, shader %d\n", what, byte, a, b);
        fails++;
    }
}

int main(void) {
    /* ---- case 1: exhaustive, XOR 0x88 then the shader decode == CPU unpack */
    for (int b = 0; b < 256; b++) {
        unsigned char raw = (unsigned char)b;
        int8_t out[2];
        unpack_int4_to_int8(out, &raw, 2);          /* the engine's own unpack */
        unsigned char staged = raw ^ 0x88u;          /* what stage() ships to VRAM */
        check(out[0] == i4_shader(staged, 0), "low nibble",  out[0], i4_shader(staged, 0), b);
        check(out[1] == i4_shader(staged, 1), "high nibble", out[1], i4_shader(staged, 1), b);
    }
    if (fails) { fprintf(stderr, "case 1 FAILED (%d mismatches)\n", fails); return 1; }
    printf("case 1 PASS: XOR 0x88 + shader i4() == unpack_int4_to_int8 over all 256 bytes\n");

    /* ---- case 2: the un-XOR'd path must NOT agree, or this test proves nothing.
     * If someone follows V1-STEP0's instruction and drops the XOR, case 1 above
     * is the leg that fires; this leg is here so a future refactor cannot make
     * case 1 vacuously true. */
    int disagreements = 0;
    for (int b = 0; b < 256; b++) {
        unsigned char raw = (unsigned char)b;
        int8_t out[2];
        unpack_int4_to_int8(out, &raw, 2);
        if (out[0] != i4_shader(raw, 0)) disagreements++;
        if (out[1] != i4_shader(raw, 1)) disagreements++;
    }
    if (disagreements != 512) {
        fprintf(stderr, "case 2 FAILED: expected all 512 nibbles to disagree without the "
                        "XOR, got %d\n", disagreements);
        return 1;
    }
    printf("case 2 PASS: without the XOR all 512 nibbles disagree (offset by 8 quanta)\n");

    /* ---- case 3: a full-size expert matrix, same identity end to end.
     * 2048 x 512 is this model's gate/up shape; the packed row is 1024 bytes. */
    const long n = 2048L * 512L;                     /* elements */
    const long nb = n / 2;                           /* packed bytes */
    unsigned char *raw = malloc((size_t)nb);
    int8_t *ref = malloc((size_t)n);
    if (!raw || !ref) { fprintf(stderr, "OOM\n"); return 1; }
    unsigned seed = 12345u;
    for (long i = 0; i < nb; i++) { seed = seed * 1103515245u + 12345u; raw[i] = (unsigned char)(seed >> 16); }
    unpack_int4_to_int8(ref, raw, n);
    long bad = 0;
    for (long i = 0; i < nb; i++) {
        unsigned char staged = raw[i] ^ 0x88u;
        if (ref[2*i]   != i4_shader(staged, 0)) bad++;
        if (ref[2*i+1] != i4_shader(staged, 1)) bad++;
    }
    free(raw); free(ref);
    if (bad) { fprintf(stderr, "case 3 FAILED: %ld of %ld elements differ\n", bad, n); return 1; }
    printf("case 3 PASS: %ld elements of a full-size expert matrix agree\n", n);

    printf("test_qwen36_vk_nibble: ALL PASS\n");
    return 0;
}
