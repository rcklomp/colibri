/* GLM-5.3 must see a CANCEL for the turn in flight even when the gateway has
 * already written the NEXT request into the same pipe.
 *
 * This is not a hypothetical ordering. `openai_server.py` builds its scheduler
 * with `capacity = kv_slots` (four in service), so up to four SUBMIT frames can
 * be in the pipe at once; the engine serves one and the others wait there. On
 * 2026-09-08 the live `curl -m 8` check hit exactly that: the client hung up,
 * the shell's next request was admitted on another slot and its SUBMIT reached
 * the pipe first, the CANCEL landed behind it, and the engine ran the whole
 * 132-second turn — twice out of two. An earlier round of the same check, where
 * the race went the other way, had passed in 7 s. A fix that depends on winning
 * a millisecond race is not a fix, and a test is how it stops depending on one.
 *
 * The frames are byte-counted, so "skip the line we do not understand" is not
 * available: the payload would stay in the stream and desync the parser. The
 * engine reads the queued frame in full instead and keeps looking. Everything
 * below is that behaviour, against the real stdin (a pipe, unbuffered, with
 * the same select() the engine uses). */
#define _GNU_SOURCE
#define GLM53_NO_MAIN
#include "../glm53.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x); return 1; \
} } while(0)

/* stdin becomes the read end of a pipe, exactly as under the gateway, so
 * coli_serve_stdin_ready()'s select(2) is the one being tested and not a
 * stand-in for it. */
static int write_fd = -1;

static int pipe_stdin(void) {
    int fds[2];
    if (pipe(fds)) return -1;
    if (dup2(fds[0], STDIN_FILENO) < 0) return -1;
    close(fds[0]);
    write_fd = fds[1];
    clearerr(stdin);
    setvbuf(stdin, NULL, _IONBF, 0);
    return 0;
}

static void feed(const char *bytes) {
    size_t n = strlen(bytes);
    if (write(write_fd, bytes, n) != (ssize_t)n) { perror("write"); exit(1); }
}

/* stdout goes to a temp file so the ERROR line the drain writes can be read
 * back: "answer NOT_FOUND for somebody else's id" is part of the contract. */
static FILE *capture_stdout(void) {
    FILE *f = tmpfile();
    if (!f) return NULL;
    fflush(stdout);
    if (dup2(fileno(f), STDOUT_FILENO) < 0) return NULL;
    setvbuf(stdout, NULL, _IONBF, 0);
    return f;
}

static int stdout_contains(FILE *f, const char *needle) {
    fflush(stdout);
    char buf[4096];
    long at = ftell(f);
    (void)at;
    rewind(f);
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fseek(f, 0, SEEK_END);
    return strstr(buf, needle) != NULL;
}

int main(void) {
    CHECK(pipe_stdin() == 0);
    FILE *out = capture_stdout();
    CHECK(out != NULL);

    /* 1. The case the live run failed: request 4 is in flight, request 5's
     *    whole frame is already in the pipe, and CANCEL 4 is behind it. The
     *    drain must reach the CANCEL. */
    feed("SUBMIT 5 0 5 8 0 1\nHello\nCANCEL 4\n");
    ServeCancel c = { 4, 0, 0 };
    CHECK(serve_cancel_pending(&c) == 1);
    CHECK(c.cancelled == 1);
    CHECK(g_queue_n == 1);

    /* 2. And the queued request is intact: same id, same slot, same payload,
     *    served next, in order. Losing it here would trade one hang for a
     *    dropped request, which is worse. */
    ServeReq q; char verb[16];
    CHECK(serve_read_req(&q, verb, sizeof(verb)) == 1);
    CHECK(!strcmp(verb, "SUBMIT"));
    CHECK(q.id == 5);
    CHECK(q.slot == 0);
    CHECK(q.plen == 5);
    CHECK(q.payload && !strcmp(q.payload, "Hello"));
    CHECK(q.max_tokens == 8);
    free(q.payload);
    CHECK(g_queue_n == 0);

    /* 3. A CANCEL for a request that is not the one in flight is answered
     *    NOT_FOUND, as serve_loop always did, and does not stop this turn. */
    feed("CANCEL 99\n");
    ServeCancel c2 = { 7, 0, 0 };
    CHECK(serve_cancel_pending(&c2) == 0);
    CHECK(c2.cancelled == 0);
    CHECK(stdout_contains(out, "ERROR 99 NOT_FOUND"));

    /* 4. STOP is dropped, exactly as serve_loop drops it between requests. */
    feed("STOP 7\n");
    CHECK(serve_cancel_pending(&c2) == 0);
    CHECK(g_queue_n == 0);

    /* 5. An empty pipe must never block and never invent a cancel. */
    CHECK(serve_cancel_pending(&c2) == 0);

    /* 6. An IMAGE frame is NOT queued: the engine holds one announced image at
     *    a time, so a second would erase the first before its request ran. Its
     *    header is put back and the drain stops there — the old behaviour, on
     *    purpose. The frame is then read normally, body and all. */
    unsigned char patch[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    feed("IMAGE 11 8 1 1\n");
    if (write(write_fd, patch, sizeof(patch)) != (ssize_t)sizeof(patch)) return 1;
    feed("\nCANCEL 7\n");
    CHECK(serve_cancel_pending(&c2) == 0);   /* stopped at the IMAGE, as designed */
    CHECK(g_pushback_full == 1);
    CHECK(g_queue_n == 0);
    CHECK(serve_read_req(&q, verb, sizeof(verb)) == 1);
    CHECK(!strcmp(verb, "IMAGE"));
    CHECK(q.id == 11);
    CHECK(g_pushback_full == 0);
    /* and the CANCEL that was behind it is still there, for the next read */
    CHECK(serve_cancel_pending(&c2) == 1);
    pending_clear();

    /* 7. Two queued requests in a row, then the cancel: the drain does not stop
     *    at the first one. Four slots means up to three can be waiting. */
    ServeCancel c3 = { 20, 0, 0 };
    feed("SUBMIT 21 1 2 4 0 1\nhi\nSUBMIT 22 2 3 4 0 1\nbye\nCANCEL 20\n");
    CHECK(serve_cancel_pending(&c3) == 1);
    CHECK(g_queue_n == 2);
    CHECK(serve_read_req(&q, verb, sizeof(verb)) == 1);
    CHECK(q.id == 21 && q.slot == 1 && !strcmp(q.payload, "hi"));
    free(q.payload);
    CHECK(serve_read_req(&q, verb, sizeof(verb)) == 1);
    CHECK(q.id == 22 && q.slot == 2 && !strcmp(q.payload, "bye"));
    free(q.payload);
    CHECK(g_queue_n == 0);

    /* 8. EOF is a cancel: on a closed pipe "readable" stays true forever, so a
     *    drain that only broke out of the loop would spin, and a turn that kept
     *    generating would be generating for nobody. */
    close(write_fd);
    ServeCancel c4 = { 30, 0, 0 };
    CHECK(serve_cancel_pending(&c4) == 1);

    fprintf(stderr, "test_glm53_cancel_frames: all checks passed\n");
    return 0;
}
