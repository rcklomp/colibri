/* ckpt_disk_touch() must prove the file on disk is the capture the slot holds
 * before it writes the hit counter into it.
 *
 * The counter is four bytes at an offset computed from the IN-MEMORY slot
 * (28 + len*4 + bytes). The old guard was only "the file is at least that
 * long", and memory and disk can legitimately disagree: ckpt_disk_write()
 * returns on a failed write -- ENOSPC on a ~305 MB file, or a failing fclose --
 * BEFORE remove(path)/rename(tmp,path), so the slot holds the new capture while
 * the path still holds the previous one. Writing four bytes at the new
 * capture's offset then lands INSIDE the old capture's blob.
 *
 * Nothing downstream notices: the file's header is untouched, so it still loads
 * cleanly, and ckpt_restore compares only span byte TOTALS, never content. The
 * greedy oracle does not see it either, because the oracle runs on fresh
 * prefills, not on restores. Silent KV corruption is the worst failure this
 * cache can have, so it gets a test rather than a comment.
 *
 * Case C below is the regression: it FAILS on the pre-P11 code (four bytes of
 * the foreign blob overwritten) and passes once the header is re-read and the
 * size is required to match exactly. */
#define _GNU_SOURCE
#define GLM53_NO_MAIN
#include "../glm53.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x); return 1; \
} } while(0)

/* The on-disk format, written here by hand so the test does not depend on the
 * writer being correct: magic[8] fp[4] len[4] kind[4] bytes[8] = 28, then
 * len*4 bytes of ids, then `bytes` of blob, then optionally 4 bytes of hits. */
static int write_ckpt_file(const char *path, uint32_t fp, int32_t len,
                           int32_t kind, uint64_t bytes, unsigned char fill,
                           const uint32_t *hits /* NULL = pre-P10 file */) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const char magic[8] = { 'G','5','3','C','K','P','T','1' };
    const int32_t head[2] = { len, kind };
    int ok = fwrite(magic, 8, 1, f) == 1 &&
             fwrite(&fp, sizeof(fp), 1, f) == 1 &&
             fwrite(head, sizeof(head), 1, f) == 1 &&
             fwrite(&bytes, sizeof(bytes), 1, f) == 1;
    for (int32_t i = 0; ok && i < len; i++) { int v = i; ok = fwrite(&v, sizeof(int), 1, f) == 1; }
    for (uint64_t i = 0; ok && i < bytes; i++) ok = fputc(fill, f) != EOF;
    if (ok && hits) ok = fwrite(hits, sizeof(*hits), 1, f) == 1;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

static long file_size(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n;
}

/* A cheap content fingerprint: sum of every byte, so any single-byte change
 * anywhere in the file shows up. */
static unsigned long long file_sum(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return 0;
    unsigned long long s = 0; int c;
    while ((c = fgetc(f)) != EOF) s += (unsigned)c + 1u;
    fclose(f); return s;
}

static uint32_t read_hits_at(const char *p, long off) {
    FILE *f = fopen(p, "rb"); if (!f) return 0;
    uint32_t h = 0;
    if (fseek(f, off, SEEK_SET) == 0) { if (fread(&h, sizeof(h), 1, f) != 1) h = 0; }
    fclose(f); return h;
}

/* Point the module's globals at a scratch directory and install one slot. */
static void set_slot(int index, int32_t len, uint64_t bytes, int32_t kind, unsigned hits) {
    PrefixCkpt *s = &g_ckpt[index];
    free(s->ids); free(s->blob);
    s->ids = calloc((size_t)len, sizeof(int));
    s->blob = calloc((size_t)bytes, 1);
    s->len = len; s->bytes = (size_t)bytes; s->kind = kind; s->hits = hits; s->used = 1;
}

int main(void) {
    setenv("GLM53_PREFIX_CKPT", "1", 1);
    setenv("GLM53_PREFIX_CKPT_DISK", "1", 1);

    char dir[] = "/tmp/glm53_ckpt_touch_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    snprintf(g_ckpt_dir, sizeof(g_ckpt_dir), "%s", dir);
    g_ckpt_fp = 0xABCD1234u;

    const int32_t LEN = 16, KIND = 0;
    const uint64_t BYTES = 64;
    const long off = 28L + (long)LEN * (long)sizeof(int) + (long)BYTES;
    char path[1200];

    /* ---- A: the file IS this capture, already carrying a counter -> update it */
    set_slot(0, LEN, BYTES, KIND, 7u);
    ckpt_disk_path(path, sizeof(path), 0);
    uint32_t one = 1u;
    CHECK(write_ckpt_file(path, g_ckpt_fp, LEN, KIND, BYTES, 0xAA, &one) == 0);
    CHECK(file_size(path) == off + 4);
    ckpt_disk_touch(0);
    CHECK(read_hits_at(path, off) == 7u);
    CHECK(file_size(path) == off + 4);
    printf("A ok: matching file, counter updated 1 -> 7\n");

    /* ---- B: a pre-P11/P10 file with NO counter -> the counter is appended */
    set_slot(0, LEN, BYTES, KIND, 5u);
    CHECK(write_ckpt_file(path, g_ckpt_fp, LEN, KIND, BYTES, 0xAA, NULL) == 0);
    CHECK(file_size(path) == off);
    ckpt_disk_touch(0);
    CHECK(read_hits_at(path, off) == 5u);
    printf("B ok: pre-P10 file, counter appended = 5\n");

    /* ---- C: THE REGRESSION. The path holds a DIFFERENT, longer capture than
     * the slot -- exactly what a failed ckpt_disk_write leaves behind. The old
     * code saw only "long enough" and wrote four bytes into that file's blob.
     * Nothing may be written. */
    set_slot(0, LEN, BYTES, KIND, 9u);
    CHECK(write_ckpt_file(path, g_ckpt_fp, LEN * 2, KIND, BYTES * 4, 0x5C, NULL) == 0);
    const long foreign_size = file_size(path);
    CHECK(foreign_size > off);                 /* "at least off" -- the old guard passed */
    const unsigned long long before = file_sum(path);
    ckpt_disk_touch(0);
    const unsigned long long after = file_sum(path);
    CHECK(file_size(path) == foreign_size);
    CHECK(before == after);                    /* fails on the pre-P11 binary */
    printf("C ok: foreign capture at this path left byte-for-byte untouched\n");

    /* ---- D: a truncated file is not this capture either */
    set_slot(0, LEN, BYTES, KIND, 3u);
    CHECK(write_ckpt_file(path, g_ckpt_fp, LEN, KIND, BYTES / 2, 0x11, NULL) == 0);
    const unsigned long long dbefore = file_sum(path);
    ckpt_disk_touch(0);
    CHECK(file_sum(path) == dbefore);
    printf("D ok: short file untouched\n");

    /* ---- E: right size, WRONG fingerprint (a different model) -> untouched */
    set_slot(0, LEN, BYTES, KIND, 4u);
    CHECK(write_ckpt_file(path, g_ckpt_fp ^ 0xFFFFu, LEN, KIND, BYTES, 0x77, &one) == 0);
    const unsigned long long ebefore = file_sum(path);
    ckpt_disk_touch(0);
    CHECK(file_sum(path) == ebefore);
    printf("E ok: foreign fingerprint untouched\n");

    unlink(path); rmdir(dir);
    printf("test_glm53_ckpt_touch: all cases passed\n");
    return 0;
}
