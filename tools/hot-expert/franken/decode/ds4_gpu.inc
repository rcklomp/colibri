// tools/hot-expert/franken/decode/ds4_gpu.inc
//
// Ds4GpuOps: the DeepSeek-V4 operations of ds4_ops.h as HIP kernels (L5
// step 2). #included by decode_gpu.hip at the end of its anonymous namespace,
// AFTER GpuBackend, so every launch goes to the backend's own stream through
// its own mark() (device selection, launch count, --profile, --sync-debug).
// Nothing above this include is changed by it: the qwen4exp kernels and
// their host code compile exactly as before (the device ISA of every
// pre-existing kernel was diffed, DEEPSEEK4.md section 9).
//
// First cut, correctness first, every kernel a transcription of the CPU op
// it replaces (ds4_cpu.cpp); where the order of a sum differs it says so.
// Per the HIP multi-GPU rule (CLAUDE.md) every entry point that allocates or
// launches selects this card first -- mark() does it for the launches.

namespace D = ::fk::ds4;

// ---------------------------------------------------------------- tables --
__constant__ uint64_t      c_ds4_iq2xxs[256];
__constant__ uint64_t      c_ds4_iq2s[1024];
__constant__ uint32_t      c_ds4_iq3xxs[256];
__constant__ unsigned char c_ds4_ksigns[128];
__constant__ unsigned char c_ds4_kmask[8];
__constant__ signed char   c_ds4_kvfp4[16];

__device__ __forceinline__ float ds4_wave_max(float v) {
#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, WAVE));
    return v;
}
__device__ __forceinline__ float ds4_h2f(unsigned int u) {
    return __half2float(__ushort_as_half((unsigned short) u));
}
__device__ __forceinline__ float ds4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
// The token's position: the device int a captured graph advances, or the
// argument when the op is issued eagerly (L5 step 3b).
__device__ __forceinline__ int ds4_pos(const int * dpos, int pos) { return dpos ? *dpos : pos; }

// ------------------------------------------------------------ the GEMVs --
//
// One wave per output row; each lane runs the ds4_quant.h lane decoder on
// every block of the row (lane tid owns weights 8*tid..8*tid+7 of a
// 256-block, one weight of a 32-block for MXFP4), then a wave sum. The
// decoders are the ones ds4_quant_emul.cpp proved bit-exact against ggml.
template <int TY>
__device__ __forceinline__ float ds4_row_lane(const unsigned char * __restrict__ row,
                                              const float * __restrict__ x, int K, int lane) {
    float a = 0.0f;
    if constexpr (TY == FK_Q_MXFP4) {
        const int nb = K / FK_MXFP4_BLOCK_WEIGHTS;
        for (int b = 0; b < nb; ++b)
            a += fk_mxfp4_lane_dot(row + (size_t) b * FK_MXFP4_BLOCK_BYTES, c_ds4_kvfp4,
                                   x + b * FK_MXFP4_BLOCK_WEIGHTS, lane);
    } else {
        constexpr int BB = TY == FK_Q_Q4_K    ? FK_Q4K_BLOCK_BYTES
                         : TY == FK_Q_Q5_K    ? FK_Q5K_BLOCK_BYTES
                         : TY == FK_Q_IQ2_XXS ? FK_IQ2XXS_BLOCK_BYTES
                         : TY == FK_Q_IQ2_S   ? FK_IQ2S_BLOCK_BYTES
                                              : FK_IQ3XXS_BLOCK_BYTES;
        const int nb = K / 256;
        for (int b = 0; b < nb; ++b) {
            const unsigned char * bp = row + (size_t) b * BB;
            const float * xb = x + b * 256 + 8 * lane;
            if constexpr (TY == FK_Q_Q4_K)         a += fk_q4k_block_dot(bp, xb, lane);
            else if constexpr (TY == FK_Q_Q5_K)    a += fk_q5k_block_dot(bp, xb, lane);
            else if constexpr (TY == FK_Q_IQ2_XXS) a += fk_iq2xxs_block_dot(bp, c_ds4_iq2xxs, c_ds4_ksigns, c_ds4_kmask, xb, lane);
            else if constexpr (TY == FK_Q_IQ2_S)   a += fk_iq2s_block_dot(bp, c_ds4_iq2s, c_ds4_kmask, xb, lane);
            else                                   a += fk_iq3xxs_block_dot(bp, c_ds4_iq3xxs, c_ds4_ksigns, c_ds4_kmask, xb, lane);
        }
    }
    return a;
}

constexpr int DS4_GEMV_THREADS = 256;                 // 8 waves = 8 rows a workgroup
constexpr int DS4_GEMV_ROWS    = DS4_GEMV_THREADS / WAVE;

// WIDE LOADS THROUGH LDS (L5 step 3). The lane decoders read their block
// with BYTE loads -- ~30 of them per block per lane, and on a row that lives
// in pinned host memory each one is its own PCIe read. So a wave first copies
// its whole row into LDS with 16-byte loads, 32 lanes side by side (a row of
// every format here is a multiple of 16 bytes, and every row starts 16-byte
// aligned: the host checks both and passes `staged`), then runs the SAME
// decoder on the LDS copy. Same bytes, same expressions, same order: the
// result is bit-identical to reading the row in place. The widest row is
// Q5_K at K = 4096, 2 816 B; the LDS is sized per format.
template <int TY> constexpr int ds4_row_max() {
    return TY == FK_Q_Q5_K ? 2816 : TY == FK_Q_Q4_K ? 2304 : TY == FK_Q_MXFP4 ? 2176
         : TY == FK_Q_IQ3_XXS ? 1568 : TY == FK_Q_IQ2_S ? 1312 : 1056;   // K <= 4096
}
template <int TY>
__device__ __forceinline__ float ds4_row_dot(const unsigned char * __restrict__ row, size_t row_bytes,
                                             int staged, uint4 * sl, const float * __restrict__ x,
                                             int K, int lane) {
    if (!staged) return ds4_row_lane<TY>(row, x, K, lane);
    const uint4 * src = (const uint4 *) row;
    const int n16 = (int) (row_bytes >> 4);
    for (int i = lane; i < n16; i += WAVE) sl[i] = src[i];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    return ds4_row_lane<TY>((const unsigned char *) sl, x, K, lane);
}

// The wave's row, staged into LDS with 16-byte loads when `staged` (the copy
// half of ds4_row_dot), else read in place: the address the decoder reads.
__device__ __forceinline__ const unsigned char * ds4_stage_row(const unsigned char * __restrict__ row,
                                                             size_t row_bytes, int staged, uint4 * sl,
                                                             int lane) {
    if (!staged) return row;
    const uint4 * src = (const uint4 *) row;
    const int n16 = (int) (row_bytes >> 4);
    for (int i = lane; i < n16; i += WAVE) sl[i] = src[i];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    return (const unsigned char *) sl;
}

// ROWS (L5 step 5): a chunk's T columns in a loop the compiler must not
// unroll, so the body -- ds4_row_lane on the staged row, the wave sum -- is
// ONE compiled sequence whatever T is: column t of a chunk is bit for bit the
// T == 1 GEMV of that column (PREFILL.md 8.1: an unrolled or tiled loop lets
// the backend fuse differently). The row is staged once for all T columns.
template <int TY>
__global__ __launch_bounds__(DS4_GEMV_THREADS)
void k_ds4_gemv(const unsigned char * __restrict__ W, size_t row_bytes, int rows, int K,
                const float * __restrict__ x, float * __restrict__ y, int staged, int T) {
    __shared__ uint4 lds[DS4_GEMV_ROWS][ds4_row_max<TY>() / 16];
    const int lane = threadIdx.x & (WAVE - 1);
    const int wv = threadIdx.x / WAVE;
    const int r = blockIdx.x * DS4_GEMV_ROWS + wv;
    if (r >= rows) return;                                   // wave-uniform
    const unsigned char * src = ds4_stage_row(W + (size_t) r * row_bytes, row_bytes, staged, lds[wv], lane);
#pragma clang loop unroll(disable)
    for (int t = 0; t < T; ++t) {
        float a = ds4_row_lane<TY>(src, x + (size_t) t * K, K, lane);
        a = wave_sum(a);
        if (lane == 0) y[(size_t) t * rows + r] = a;
    }
}

// The routed experts, as a list of ITEMS, each an expert row read once and a
// list of ASSIGNMENTS (a = t*6 + k: row t of the chunk, its slot k) it
// serves. grid.y is the item.
//   direct (`order` null): item = assignment a, its expert ids[a] -- the
//     decode token (T == 1, grid.y = 6, exactly step 3's launch) and the
//     per-assignment path of a chunk;
//   sorted: item = expert e, its ecnt[e] assignments order[eoff[e] ..] in
//     ascending order (k_ds4_moe_sort) -- a chunk's expert-major path, the
//     row read from VRAM (or over PCIe) once for all of them.
// The expert's slice comes from the layer's TABLE by id (`sp` null: VRAM when
// resident, the device-mapped pinned host copy when not) or, direct at
// T == 1, from the staging slot's pointers `sp[k*3 + which]` (L5 step 3).
// Every assignment runs the SAME compiled loop body (unroll disabled), so a
// dot is bit-identical in either mode and at any T. `filter` (profiling and
// the resident-first split): 1 = resident experts only, 2 = missed only.
// x: `x_per_slot` 1 reads x + a*K (down), 0 reads x + (a/6)*K (gate/up).
template <int TY>
__global__ __launch_bounds__(DS4_GEMV_THREADS)
void k_ds4_moe(const void * const * __restrict__ tab, const int * __restrict__ ids,
               const void * const * __restrict__ sp, int which, const int * __restrict__ miss,
               int filter, size_t row_bytes, int rows, int K, const float * __restrict__ x,
               int x_per_slot, float * __restrict__ y, int staged,
               const int * __restrict__ order, const int * __restrict__ ecnt,
               const int * __restrict__ eoff) {
    __shared__ uint4 lds[DS4_GEMV_ROWS][ds4_row_max<TY>() / 16];
    const int item = blockIdx.y;
    const int lane = threadIdx.x & (WAVE - 1);
    const int wv = threadIdx.x / WAVE;
    const int r = blockIdx.x * DS4_GEMV_ROWS + wv;
    if (r >= rows) return;
    int e, n, first;
    if (order) { e = item; n = ecnt[e]; first = eoff[e]; }
    else       { e = ids[item]; n = 1; first = item; }
    if (n == 0) return;
    if (filter && ((miss[e] != 0) != (filter == 2))) return;
    const unsigned char * W = (const unsigned char *) (sp ? sp[item * 3 + which] : tab[e]);
    const unsigned char * src = ds4_stage_row(W + (size_t) r * row_bytes, row_bytes, staged, lds[wv], lane);
#pragma clang loop unroll(disable)
    for (int j = 0; j < n; ++j) {
        const int a = order ? order[first + j] : first;
        const float * xs = x + (size_t) (x_per_slot ? a : a / D::N_EXPERT_USED) * K;
        float acc = ds4_row_lane<TY>(src, xs, K, lane);
        acc = wave_sum(acc);
        if (lane == 0) y[(size_t) a * rows + r] = acc;
    }
}

// A chunk's T*6 assignments sorted by expert, ascending assignment index
// within an expert (deterministic). One workgroup, a thread an expert.
constexpr int DS4_MOE_SORT_MAX = 512 * D::N_EXPERT_USED;
__global__ __launch_bounds__(D::N_EXPERT)
void k_ds4_moe_sort(const int * __restrict__ ids, int n_assign, int * __restrict__ ecnt,
                    int * __restrict__ eoff, int * __restrict__ order) {
    __shared__ int sid[DS4_MOE_SORT_MAX];
    __shared__ int scnt[D::N_EXPERT];
    const int e = threadIdx.x;
    for (int i = e; i < n_assign; i += D::N_EXPERT) sid[i] = ids[i];
    __syncthreads();
    int c = 0;
    for (int i = 0; i < n_assign; ++i) c += sid[i] == e;
    scnt[e] = c;
    __syncthreads();
    int off = 0;
    for (int j = 0; j < e; ++j) off += scnt[j];
    ecnt[e] = c; eoff[e] = off;
    for (int i = 0; i < n_assign; ++i) if (sid[i] == e) order[off++] = i;
}

// THE STAGING COPY (L5 step 3), on the side stream. grid = (chunks, 6):
// slot k's expert e = ids[k]; a resident one is only POINTED at (sp gets its
// three table addresses), a missed one is copied gate|up|down into
// dst + k*slot_bytes with 16-byte loads -- coalesced, many in flight, which is
// what a PCIe read wants and what the GEMV's byte loads were not. Block 0 of
// the slot publishes the pointers and counts the bytes.
constexpr int DS4_STAGE_THREADS = 256;
// grid = (wgs, 6): workgroup x of slot k copies 16-byte words x, x + wgs*256, ...
// of the missed expert's gate|up|down -- grid-stride, so a SMALL grid (the
// --stage-wgs knob, 64 by default: ~11 a slot) moves the whole slab while
// most of the card stays free for the main stream's kernels. Block 0 of the
// slot publishes the three pointers.
__global__ __launch_bounds__(DS4_STAGE_THREADS)
void k_ds4_stage(const void * const * __restrict__ tg, const void * const * __restrict__ tu,
                 const void * const * __restrict__ td, const int * __restrict__ ids,
                 const int * __restrict__ miss, unsigned long long sg, unsigned long long su,
                 unsigned long long sd, unsigned char * __restrict__ dst, unsigned long long slot_bytes,
                 const void ** __restrict__ sp, unsigned long long * __restrict__ counter) {
    const int k = blockIdx.y;
    const int e = ids[k];
    const bool missed = miss[e] != 0;
    unsigned char * d = dst + (size_t) k * slot_bytes;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sp[k * 3 + 0] = missed ? (const void *) d             : tg[e];
        sp[k * 3 + 1] = missed ? (const void *) (d + sg)      : tu[e];
        sp[k * 3 + 2] = missed ? (const void *) (d + sg + su) : td[e];
        if (missed && counter) atomicAdd(counter, sg + su + sd);
    }
    if (!missed) return;
    const unsigned long long total = sg + su + sd;
    const unsigned long long step = 16ull * DS4_STAGE_THREADS * gridDim.x;
    for (unsigned long long o = 16ull * ((unsigned long long) blockIdx.x * DS4_STAGE_THREADS + threadIdx.x);
         o < total; o += step) {
        const unsigned char * s = o < sg      ? (const unsigned char *) tg[e] + o
                                : o < sg + su ? (const unsigned char *) tu[e] + (o - sg)
                                              : (const unsigned char *) td[e] + (o - sg - su);
        *(uint4 *) (d + o) = *(const uint4 *) s;
    }
}

// --profile: the six chosen experts' miss bytes, added on the device.
__global__ void k_ds4_miss_count(const int * __restrict__ ids, const int * __restrict__ miss,
                                 unsigned long long * __restrict__ counter, int n_assign) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n_assign) {
        const int m = miss[ids[k]];
        if (m) atomicAdd(counter, (unsigned long long) m);
    }
}

// -------------------------------------------------------- hyper-connections --
// 24 inputs and a 4x4 Sinkhorn loop. Four threads (L5 step 3; it was one):
// thread o owns column o in a column pass and row o in a row pass, and sums
// it in the same order the single thread did, so every element sees the same
// operations -- bit-identical to the one-thread version and to the CPU op's
// order. 39 dependent normalisations a call, 86 calls a token.
__global__ void k_ds4_hc_split(const float * __restrict__ m, const float * __restrict__ s,
                               const float * __restrict__ b, float * __restrict__ pre,
                               float * __restrict__ post, float * __restrict__ comb,
                               float eps, int iters) {
    constexpr int H = D::HC;
    __shared__ float c[H * H];
    const int o = threadIdx.x;                       // launched with H threads
    {   // rows (L5 step 5): grid.x is the row
        const int row = blockIdx.x;
        m += (size_t) row * D::HC_MIX; pre += (size_t) row * H; post += (size_t) row * H;
        comb += (size_t) row * H * H;
    }
    pre[o]  = ds4_sigmoid(m[o] * s[0] + b[o]) * 1.0f + eps;
    post[o] = ds4_sigmoid(m[H + o] * s[1] + b[H + o]) * 2.0f;
    {   // softmax over dst for source isrc = o
        const int isrc = o;
        float v[H], mx = -INFINITY;
        for (int idst = 0; idst < H; ++idst) {
            const int idx = idst + H * isrc;
            v[idst] = m[2 * H + idx] * s[2] + b[2 * H + idx];
            mx = fmaxf(mx, v[idst]);
        }
        float sum = 0.0f;
        for (int idst = 0; idst < H; ++idst) { v[idst] = expf(v[idst] - mx); sum += v[idst]; }
        const float inv = 1.0f / sum;
        for (int idst = 0; idst < H; ++idst) c[idst + H * isrc] = v[idst] * inv + eps;
    }
    __syncthreads();
    for (int it = 0; it < 2 * iters - 1; ++it) {
        const bool cols = (it % 2 == 0);             // cols, then (rows, cols) x (iters-1)
        float sum = eps;
        for (int i = 0; i < H; ++i) sum += cols ? c[o + H * i] : c[i + H * o];
        const float inv = 1.0f / sum;
        for (int i = 0; i < H; ++i) { if (cols) c[o + H * i] *= inv; else c[i + H * o] *= inv; }
        __syncthreads();
    }
    for (int i = 0; i < H; ++i) comb[o + H * i] = c[o + H * i];
}

__global__ void k_ds4_hc_head_pre(const float * __restrict__ m, const float * __restrict__ s,
                                  const float * __restrict__ b, float * __restrict__ pre, float eps) {
    const int h = threadIdx.x;
    if (h < D::HC) pre[h] = ds4_sigmoid(m[h] * s[0] + b[h]) * 1.0f + eps;
}

__global__ void k_ds4_hc_init(const float * __restrict__ x, float * __restrict__ H, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D::HC_DIM * T) H[i] = x[(size_t) (i / D::HC_DIM) * D::N_EMBD + (i % D::HC_DIM) % D::N_EMBD];
}

__global__ void k_ds4_hc_wsum(const float * __restrict__ H, const float * __restrict__ w,
                              float * __restrict__ out, int T) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= D::N_EMBD * T) return;
    const int row = gi / D::N_EMBD, i = gi % D::N_EMBD;
    H += (size_t) row * D::HC_DIM; w += (size_t) row * D::HC; out += (size_t) row * D::N_EMBD;
    float sum = 0.0f;
    for (int h = 0; h < D::HC; ++h) sum += H[(size_t) h * D::N_EMBD + i] * w[h];
    out[i] = sum;
}

__global__ void k_ds4_hc_post(const float * __restrict__ x, const float * __restrict__ H,
                              const float * __restrict__ post, const float * __restrict__ comb,
                              float * __restrict__ Ho, int T) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= D::HC_DIM * T) return;
    const int row = gi / D::HC_DIM, idx = gi % D::HC_DIM;
    x += (size_t) row * D::N_EMBD; H += (size_t) row * D::HC_DIM; post += (size_t) row * D::HC;
    comb += (size_t) row * D::HC * D::HC; Ho += (size_t) row * D::HC_DIM;
    const int d = idx / D::N_EMBD, i = idx % D::N_EMBD;
    float sum = x[i] * post[d];
    for (int s = 0; s < D::HC; ++s) sum += H[(size_t) s * D::N_EMBD + i] * comb[d + D::HC * s];
    Ho[idx] = sum;
}

// --------------------------------------------------------------- rope ----
// One thread a rotated pair. theta_scale, the yarn corr dims and the
// effective mscale come from the HOST (the CPU arm's own powf/logf), and
// theta is built by the same repeated multiply as ggml's cache, so theta is
// bit-identical to the CPU arm; cos/sin are the device's.
__global__ void k_ds4_rope_tail(float * __restrict__ x, int row_len, int pos_arg,
                                const int * __restrict__ dpos, int block_ratio, float theta_scale,
                                float freq_scale, float ext, float mscale, float c0, float c1,
                                float sin_sign, int rows_per_pos) {
    const int i = threadIdx.x;                  // pair, 0..N_ROT/2-1
    int p = ds4_pos(dpos, pos_arg) + (rows_per_pos > 0 ? (int) blockIdx.x / rows_per_pos : 0);
    if (block_ratio > 0) p = (p / block_ratio) * block_ratio;
    float theta = (float) p;
    for (int j = 0; j < i; ++j) theta *= theta_scale;
    const float ti = freq_scale * theta;
    float th = ti;
    if (ext != 0.0f) {
        const float y = ((float) i - c0) / fmaxf(0.001f, c1 - c0);
        const float rm = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext;
        th = ti * (1.0f - rm) + theta * rm;
    }
    const float c = cosf(th) * mscale, s = sinf(th) * mscale * sin_sign;
    float * v = x + (size_t) blockIdx.x * row_len + (row_len - D::N_ROT) + 2 * i;
    const float x0 = v[0], x1 = v[1];
    v[0] = x0 * c - x1 * s;
    v[1] = x0 * s + x1 * c;
}

// ggml's FWHT: scale first, then the butterflies (u+v, u-v), n = 128.
__global__ __launch_bounds__(D::IDX_DIM)
void k_ds4_fwht(float * __restrict__ x, float scale) {
    __shared__ float v[D::IDX_DIM];
    float * row = x + (size_t) blockIdx.x * D::IDX_DIM;
    const int t = threadIdx.x;
    v[t] = row[t] * scale;
    __syncthreads();
    for (int len = 1; len < D::IDX_DIM; len <<= 1) {
        float a = 0.0f, b = 0.0f;
        const bool lo = (t & len) == 0;
        if (lo) { a = v[t]; b = v[t + len]; }
        __syncthreads();
        if (lo) { v[t] = a + b; v[t + len] = a - b; }
        __syncthreads();
    }
    row[t] = v[t];
}

// f16 store into a positional cache. ratio 0: the raw window, row pos % N_SWA;
// ratio R > 0: row pos / R, and only when a block ends at pos.
// Rows: grid.y is the row t, at position pos + t, reading x + t*width.
// ratio 0: the raw window, slot p % raw_rows.
__global__ void k_ds4_store_f16(const float * __restrict__ x, unsigned short * __restrict__ cache,
                                int width, int ratio, int pos_arg, const int * __restrict__ dpos,
                                int raw_rows) {
    const int p = ds4_pos(dpos, pos_arg) + (int) blockIdx.y;
    x += (size_t) blockIdx.y * width;
    int row;
    if (ratio == 0) row = p % raw_rows;
    else { if ((p + 1) % ratio != 0) return; row = p / ratio; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) cache[(size_t) row * width + i] = __half_as_ushort(__float2half_rn(x[i]));
}
__global__ void k_ds4_ring_put(float * __restrict__ ring, int ring_rows, int width, int pos_arg,
                               const int * __restrict__ dpos, const float * __restrict__ src) {
    const int p = ds4_pos(dpos, pos_arg) + (int) blockIdx.y;
    src += (size_t) blockIdx.y * width;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) ring[(size_t) (p % ring_rows) * width + i] = src[i];
}
__global__ void k_ds4_add_row(float * __restrict__ x, const float * __restrict__ table, int width,
                              int ratio, int pos_arg, const int * __restrict__ dpos) {
    const int p = ds4_pos(dpos, pos_arg) + (int) blockIdx.y;
    x += (size_t) blockIdx.y * width;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) x[i] = x[i] + table[(size_t) (p % ratio) * width + i];
}
// dst[r*ds + i] = src[r*ss + i]: grid (ceil(row_len/BLK), n_rows)
__global__ void k_ds4_copy_rows(const float * __restrict__ src, size_t ss, float * __restrict__ dst,
                                size_t ds, int row_len) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row_len) dst[(size_t) blockIdx.y * ds + i] = src[(size_t) blockIdx.y * ss + i];
}

// ------------------------------------------------------ compressor pools --
// One thread a dimension: softmax over the candidates, then the weighted sum
// (the CPU arm sums in double; here in float -- a summation-order difference).
__global__ void k_ds4_comp_pool(const float * __restrict__ rk, const float * __restrict__ rs,
                                int ring, int R, int d_out, int pos_arg, const int * __restrict__ dpos,
                                int overlap, float * __restrict__ out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_out) return;
    const int p = ds4_pos(dpos, pos_arg) + (int) blockIdx.y;   // rows: grid.y
    out += (size_t) blockIdx.y * d_out;
    if ((p + 1) % R != 0) return;                // no block ends here: out untouched
    const int blk = p / R;
    const int n = overlap ? 2 * R : R;
    const int width = overlap ? 2 * d_out : d_out;
    auto get = [&](int j, float & v, float & s) {
        if (!overlap) {
            const int slot = (blk * R + j) % ring;
            v = rk[(size_t) slot * width + d]; s = rs[(size_t) slot * width + d];
        } else if (j < R) {
            if (blk == 0) { v = 0.0f; s = -INFINITY; return; }
            const int slot = ((blk - 1) * R + j) % ring;
            v = rk[(size_t) slot * width + d]; s = rs[(size_t) slot * width + d];
        } else {
            const int slot = (blk * R + j - R) % ring;
            v = rk[(size_t) slot * width + d_out + d]; s = rs[(size_t) slot * width + d_out + d];
        }
    };
    float mx = -INFINITY, v, s;
    for (int j = 0; j < n; ++j) { get(j, v, s); mx = fmaxf(mx, s); }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) { get(j, v, s); sum += expf(s - mx); }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) { get(j, v, s); acc += v * (expf(s - mx) * inv); }
    out[d] = acc;
}

// ------------------------------------------------- the lightning indexer --
//
// score[b] = sum_h relu(q_h . k_b) * w_h: 64 heads x 128 dims = 8 192 MACs
// per 256-byte f16 key. That is ARITHMETIC-bound on this card, not
// bandwidth-bound: at 256k, 65 536 keys a layer are 16.8 MB (21 us at
// 800 GB/s) but 537 M MACs (~27 us at a ~20 TFMA/s fp32 peak that needs the
// dual issue this code cannot count on, ~55 us at the single-issue rate).
//
// FOUR lanes own one key, 32 dims each, and the head loop keeps ONE running
// dot per key: 32 FMAs, a two-step exchange within the four, relu, times w_h.
// q (32 KB) and w are in LDS; each group of four reads four distinct q
// addresses, the rest of the wave the same four (broadcast). Heads are summed
// in ascending order; a dot is ((p0+p1)+(p2+p3)) of the four partials. (The
// first shapes tried -- one thread a key with 64 or 16 head sums in
// registers -- asked for 256 / 246 VGPR and spilled.)
//
// L5 step 5 (DEEPSEEK4.md section 13): each group of four now carries
// DS4_LID_KPT = 4 keys, so one LDS read of q feeds four FMAs instead of one
// -- step 2's shape was LDS-bound (8 ds_load_b128 + 2 ds_bpermute per 32
// FMAs a head), about twice its FMA time -- the keys stay packed f16 in
// registers (a conversion is exact, so fma(q, (float) k, a) is the same number
// however the compiler forms it), the exchange is DPP instead of LDS
// bpermute, and the grid is PERSISTENT (at most DS4_LID_MAX_WGS workgroups
// walk the tiles), so q is staged into LDS by a few hundred workgroups rather
// than one per 64 keys (1 024 of them at 256k: 32 MB of q against 16.8 MB of
// keys). Per key the operations and their order are exactly step 2's --
// the dot's 32 fmaf in dim order, the two adds, fmaxf, the fused
// sc = fma(relu, w_h, sc) the step-2 ISA shows (v_fmac) -- so the scores are
// bit-identical. grid.y is the query row of a chunk.
constexpr int DS4_LID_THREADS = 256;
constexpr int DS4_LID_LANES   = 4;                                  // lanes a key
constexpr int DS4_LID_GROUPS  = DS4_LID_THREADS / DS4_LID_LANES;    // 64 groups of four
constexpr int DS4_LID_KPT     = 4;                                  // keys a group
constexpr int DS4_LID_TILE    = DS4_LID_GROUPS * DS4_LID_KPT;       // 256 keys a tile
constexpr int DS4_LID_MAX_WGS = 384;                                // persistent grid cap
__device__ __forceinline__ float ds4_quad_xor1(float v) {           // lane ^ 1 within a quad
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false));
}
__device__ __forceinline__ float ds4_quad_xor2(float v) {           // lane ^ 2 within a quad
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false));
}
__global__ __launch_bounds__(DS4_LID_THREADS)
void k_ds4_lid_scores(const float * __restrict__ q, const float * __restrict__ w,
                      const unsigned short * __restrict__ keys, int pos_arg,
                      const int * __restrict__ dpos, float * __restrict__ scores, int stride) {
    const int row = blockIdx.y;
    const int n = (ds4_pos(dpos, pos_arg) + row + 1) / D::CSA_RATIO;
    if ((int) blockIdx.x * DS4_LID_TILE >= n) return;        // uniform, before any barrier
    q += (size_t) row * D::IDX_N_HEAD * D::IDX_DIM;
    w += (size_t) row * D::IDX_N_HEAD;
    scores += (size_t) row * stride;
    __shared__ float sq[D::IDX_N_HEAD * D::IDX_DIM];
    __shared__ float sw[D::IDX_N_HEAD];
    for (int i = threadIdx.x; i < D::IDX_N_HEAD * D::IDX_DIM / 4; i += DS4_LID_THREADS)
        ((float4 *) sq)[i] = ((const float4 *) q)[i];
    if (threadIdx.x < D::IDX_N_HEAD) sw[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    constexpr int PD = D::IDX_DIM / DS4_LID_LANES;                  // 32 dims a lane
    const int part = threadIdx.x & (DS4_LID_LANES - 1);
    const int g = threadIdx.x / DS4_LID_LANES;
    for (int tile = blockIdx.x; tile * DS4_LID_TILE < n; tile += gridDim.x) {
        // key k of this group: tile*256 + k*64 + g -- a wave's eight groups
        // read eight consecutive 256-byte keys, fully coalesced
        unsigned kp[DS4_LID_KPT][PD / 2];
        int b[DS4_LID_KPT];
#pragma unroll
        for (int k = 0; k < DS4_LID_KPT; ++k) {
            b[k] = tile * DS4_LID_TILE + k * DS4_LID_GROUPS + g;
            const uint4 * k4 = (const uint4 *) (keys + (size_t) b[k] * D::IDX_DIM + part * PD);
#pragma unroll
            for (int j = 0; j < PD / 8; ++j) {
                const uint4 u = (b[k] < n) ? k4[j] : make_uint4(0u, 0u, 0u, 0u);
                kp[k][4 * j] = u.x; kp[k][4 * j + 1] = u.y; kp[k][4 * j + 2] = u.z; kp[k][4 * j + 3] = u.w;
            }
        }
        float sc[DS4_LID_KPT];
#pragma unroll
        for (int k = 0; k < DS4_LID_KPT; ++k) sc[k] = 0.0f;
        for (int h = 0; h < D::IDX_N_HEAD; ++h) {
            const float * qh = sq + h * D::IDX_DIM + part * PD;
            float a[DS4_LID_KPT];
#pragma unroll
            for (int k = 0; k < DS4_LID_KPT; ++k) a[k] = 0.0f;
#pragma unroll
            for (int i = 0; i < PD / 2; ++i) {
                const float q0 = qh[2 * i], q1 = qh[2 * i + 1];
#pragma unroll
                for (int k = 0; k < DS4_LID_KPT; ++k) {
                    a[k] = fmaf(q0, ds4_h2f(kp[k][i] & 0xFFFFu), a[k]);
                    a[k] = fmaf(q1, ds4_h2f(kp[k][i] >> 16), a[k]);
                }
            }
            const float wh = sw[h];
#pragma unroll
            for (int k = 0; k < DS4_LID_KPT; ++k) {
                a[k] = a[k] + ds4_quad_xor1(a[k]);
                a[k] = a[k] + ds4_quad_xor2(a[k]);
                sc[k] = fmaf(fmaxf(a[k], 0.0f), wh, sc[k]);
            }
        }
        if (part == 0) {
#pragma unroll
            for (int k = 0; k < DS4_LID_KPT; ++k) if (b[k] < n) scores[b[k]] = sc[k];
        }
    }
}

// Top-k as a SET: one workgroup a query row, a 4-pass 8-bit MSB-first radix
// select on the order-preserving uint image of the scores (n <= 65 536 at
// 256k), then everything above the threshold plus as many ties as the count
// needs.
//
// L5 step 5: (1) the histogram's LDS atomics are aggregated per wave -- the
// scores of one query share their top bits, so step 2's kernel put nearly all
// of a pass's n atomics on ONE bin, serialised on one CU (~400 us a layer at
// 256k, projected); now a wave adds each distinct bin once. (2) The bin is
// chosen by a parallel suffix scan instead of one thread walking 256 bins.
// (3) The output is in ASCENDING INDEX ORDER, from a block scan: step 2 wrote
// it through an atomic counter, so the order -- and with it the attention's
// summation order over the selected rows -- changed from run to run past 2 048
// tokens. The SET is the same (same threshold, ties taken lowest index first
// instead of first-come), so this is deterministic where step 2 was not; it is
// not bit-comparable to step 2 past 2 048 tokens because step 2 was not
// comparable to itself.
constexpr int DS4_TOPK_THREADS = 1024;
__device__ __forceinline__ unsigned ds4_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// exclusive prefix sum over the workgroup (blockDim.x a multiple of WAVE, <= 1024)
__device__ __forceinline__ unsigned ds4_block_excl_scan(unsigned v, unsigned * sh) {
    const int lane = threadIdx.x & (WAVE - 1), wv = threadIdx.x / WAVE;
    unsigned x = v;
#pragma unroll
    for (int off = 1; off < WAVE; off <<= 1) {
        const unsigned y = __shfl_up(x, off, WAVE);
        if (lane >= off) x += y;
    }
    if (lane == WAVE - 1) sh[wv] = x;
    __syncthreads();
    if (wv == 0) {
        unsigned z = (lane < (int) (blockDim.x / WAVE)) ? sh[lane] : 0u;
#pragma unroll
        for (int off = 1; off < WAVE; off <<= 1) {
            const unsigned y = __shfl_up(z, off, WAVE);
            if (lane >= off) z += y;
        }
        sh[lane] = z;
    }
    __syncthreads();
    const unsigned r = (wv ? sh[wv - 1] : 0u) + x - v;
    __syncthreads();
    return r;
}
__global__ __launch_bounds__(DS4_TOPK_THREADS)
void k_ds4_topk(const float * __restrict__ s, int pos_arg, const int * __restrict__ dpos, int k,
                int * __restrict__ out, int stride) {
    __shared__ unsigned hist[256], suf[256], scan_sh[WAVE];
    __shared__ unsigned sh_prefix, sh_mask;
    __shared__ int sh_kr;
    const int row = blockIdx.x;
    const int t = threadIdx.x, lane = t & (WAVE - 1);
    const int n = (ds4_pos(dpos, pos_arg) + row + 1) / D::CSA_RATIO;
    s += (size_t) row * stride;
    out += (size_t) row * k;
    if (n <= k) {                                // the identity, as ggml_top_k's
        for (int i = t; i < n; i += DS4_TOPK_THREADS) out[i] = i;
        return;
    }
    unsigned prefix = 0, mask = 0;
    int kr = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int base = 0; base < n; base += DS4_TOPK_THREADS) {
            const int i = base + t;
            bool m = false;
            unsigned bin = 0;
            if (i < n) {
                const unsigned u = ds4_key(s[i]);
                m = (u & mask) == prefix;
                bin = (u >> shift) & 255u;
            }
            unsigned act = (unsigned) __ballot(m);
            while (act) {                            // wave-uniform
                const int leader = __ffs(act) - 1;
                const unsigned lb = __shfl(bin, leader, WAVE);
                const unsigned same = (unsigned) __ballot(m && bin == lb) & act;
                if (lane == leader) atomicAdd(&hist[lb], (unsigned) __popc(same));
                act &= ~same;
            }
        }
        __syncthreads();
        // suf[b] = hist[b] + ... + hist[255]; the chosen bin is the one with
        // suf[b] >= kr > suf[b+1] -- step 2's descending walk, in parallel
        if (t < 256) suf[t] = hist[t];
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            const unsigned v = (t < 256 && t + off < 256) ? suf[t + off] : 0u;
            __syncthreads();
            if (t < 256) suf[t] += v;
            __syncthreads();
        }
        if (t < 256) {
            const unsigned above = (t + 1 < 256) ? suf[t + 1] : 0u;
            if (suf[t] >= (unsigned) kr && above < (unsigned) kr) {
                sh_kr = kr - (int) above;
                sh_prefix = prefix | ((unsigned) t << shift);
                sh_mask = mask | (255u << shift);
            }
        }
        __syncthreads();
        prefix = sh_prefix; mask = sh_mask; kr = sh_kr;
        __syncthreads();
    }
    // ascending index order: thread t owns the contiguous range [lo, hi)
    const int c = (n + DS4_TOPK_THREADS - 1) / DS4_TOPK_THREADS;
    const int lo = min(n, t * c), hi = min(n, lo + c);
    unsigned gt = 0, eq = 0;
    for (int i = lo; i < hi; ++i) {
        const unsigned u = ds4_key(s[i]);
        gt += u > prefix;
        eq += u == prefix;
    }
    const unsigned gte = ds4_block_excl_scan(gt, scan_sh);
    const unsigned eqe = ds4_block_excl_scan(eq, scan_sh);
    unsigned wr = gte + min(eqe, (unsigned) kr), er = eqe;
    for (int i = lo; i < hi; ++i) {
        const unsigned u = ds4_key(s[i]);
        if (u > prefix) out[wr++] = i;
        else if (u == prefix) { if (er < (unsigned) kr) out[wr++] = i; ++er; }
    }
}

// -------------------------------------------------------------- attention --
//
// Flash-style split: a workgroup takes a CHUNK of 32 keys (raw-window slots
// then compressed rows, the order the CPU arm uses) into LDS once, and 16 of
// the 64 heads (grid.y = 4 head groups; 8 waves x 2 heads). Lane j scores key
// j; the value pass reuses the same LDS rows (V = K). Per (chunk, head) it
// leaves the max, the sum of exp and the unnormalised output; the combine
// folds the chunks together with the head's SINK in the denominator.
// Rows are padded to 257 u32 so lane j's reads of row j fall in distinct banks.
constexpr int DS4_ATT_CHUNK   = 32;
constexpr int DS4_ATT_HGROUPS = 4;
constexpr int DS4_ATT_STRIDE  = D::HEAD_DIM / 2 + 1;       // u32 per LDS row
__device__ __forceinline__ void ds4_attn_extent(int p, int ratio, int cap, int & raw0, int & n_raw,
                                                int & n_comp) {
    raw0 = p - D::N_SWA + 1 > 0 ? p - D::N_SWA + 1 : 0;
    n_raw = p - raw0 + 1;
    n_comp = ratio ? min((p + 1) / ratio, cap) : 0;
}
// Rows (L5 step 5): grid.z is the row of a block of query rows starting at
// row0 (row0 + z of the chunk, at position pos + row0 + z); its partials sit
// at local row z of a buffer of `chunk_cap` chunks a row. The per-row work
// -- the chunking, the order, the combine -- is the T == 1 kernel's.
__global__ __launch_bounds__(256)
void k_ds4_attn_part(const float * __restrict__ q, const unsigned short * __restrict__ raw,
                     int raw_rows, int pos_arg, const int * __restrict__ dpos,
                     const unsigned short * __restrict__ comp,
                     int ratio, int cap, const int * __restrict__ ids, float scale,
                     float * __restrict__ po, float * __restrict__ pm, float * __restrict__ pl,
                     int row0, int chunk_cap) {
    __shared__ unsigned kl[DS4_ATT_CHUNK * DS4_ATT_STRIDE];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = threadIdx.x;
    const int row = row0 + (int) blockIdx.z;
    q += (size_t) row * D::N_HEAD * D::HEAD_DIM;
    if (ids) ids += (size_t) row * D::IDX_TOP_K;
    {
        const size_t lb = (size_t) blockIdx.z * chunk_cap * D::N_HEAD;
        po += lb * D::HEAD_DIM; pm += lb; pl += lb;
    }
    int raw0, n_raw, n_comp;
    ds4_attn_extent(ds4_pos(dpos, pos_arg) + row, ratio, cap, raw0, n_raw, n_comp);
    const int n_keys = n_raw + n_comp;
    const int j0 = chunk * DS4_ATT_CHUNK;
    if (j0 >= n_keys) return;                    // a graph sized for its class's last position
    const int nk = min(DS4_ATT_CHUNK, n_keys - j0);
    for (int idx = t; idx < DS4_ATT_CHUNK * (D::HEAD_DIM / 2); idx += 256) {
        const int r = idx / (D::HEAD_DIM / 2), c = idx % (D::HEAD_DIM / 2);
        if (r < nk) {
            const int g = j0 + r;
            const unsigned short * row = (g < n_raw)
                ? raw + (size_t) ((raw0 + g) % raw_rows) * D::HEAD_DIM
                : comp + (size_t) (ids ? ids[g - n_raw] : g - n_raw) * D::HEAD_DIM;
            kl[r * DS4_ATT_STRIDE + c] = ((const unsigned *) row)[c];
        }
    }
    __syncthreads();
    // the wave index made scalar, so q's head address is uniform and the
    // compiler may load it with scalar loads (same values either way)
    const int wave = __builtin_amdgcn_readfirstlane(t / WAVE), lane = t & (WAVE - 1);
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * (D::N_HEAD / DS4_ATT_HGROUPS) + wave * 2 + hh;
        const float * qh = q + (size_t) h * D::HEAD_DIM;
        float sc = -INFINITY;
        if (lane < nk) {
            float a = 0.0f;
            for (int c = 0; c < D::HEAD_DIM / 2; ++c) {
                const unsigned u = kl[lane * DS4_ATT_STRIDE + c];
                a = fmaf(qh[2 * c],     ds4_h2f(u & 0xFFFFu), a);
                a = fmaf(qh[2 * c + 1], ds4_h2f(u >> 16), a);
            }
            sc = a * scale;
        }
        const float m = ds4_wave_max(sc);
        const float p = (lane < nk) ? expf(sc - m) : 0.0f;
        const float l = wave_sum(p);
        float acc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
        for (int j = 0; j < nk; ++j) {
            const float pj = __shfl(p, j, WAVE);
#pragma unroll
            for (int tt = 0; tt < 8; ++tt) {
                const unsigned u = kl[j * DS4_ATT_STRIDE + lane + 32 * tt];
                acc[2 * tt]     = fmaf(pj, ds4_h2f(u & 0xFFFFu), acc[2 * tt]);
                acc[2 * tt + 1] = fmaf(pj, ds4_h2f(u >> 16),     acc[2 * tt + 1]);
            }
        }
        float * o = po + ((size_t) chunk * D::N_HEAD + h) * D::HEAD_DIM;
#pragma unroll
        for (int tt = 0; tt < 8; ++tt) {
            o[2 * (lane + 32 * tt)]     = acc[2 * tt];
            o[2 * (lane + 32 * tt) + 1] = acc[2 * tt + 1];
        }
        if (lane == 0) { pm[chunk * D::N_HEAD + h] = m; pl[chunk * D::N_HEAD + h] = l; }
    }
}

__global__ __launch_bounds__(D::HEAD_DIM)
void k_ds4_attn_combine(const float * __restrict__ po, const float * __restrict__ pm,
                        const float * __restrict__ pl, int pos_arg, const int * __restrict__ dpos,
                        int ratio, int cap, const float * __restrict__ sinks, float * __restrict__ out,
                        int row0, int chunk_cap) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int row = row0 + (int) blockIdx.y;
    out += (size_t) row * D::N_HEAD * D::HEAD_DIM;
    {
        const size_t lb = (size_t) blockIdx.y * chunk_cap * D::N_HEAD;
        po += lb * D::HEAD_DIM; pm += lb; pl += lb;
    }
    int raw0, n_raw, n_comp;
    ds4_attn_extent(ds4_pos(dpos, pos_arg) + row, ratio, cap, raw0, n_raw, n_comp);
    const int n_chunks = (n_raw + n_comp + DS4_ATT_CHUNK - 1) / DS4_ATT_CHUNK;
    float M = sinks[h];
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, pm[c * D::N_HEAD + h]);
    float L = expf(sinks[h] - M), o = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float f = expf(pm[c * D::N_HEAD + h] - M);
        L += pl[c * D::N_HEAD + h] * f;
        o += po[((size_t) c * D::N_HEAD + h) * D::HEAD_DIM + d] * f;
    }
    out[(size_t) h * D::HEAD_DIM + d] = o / L;
}

// ------------------------------------------------------------------ MoE ----
// One thread an expert; the top-6 by six block argmax passes (value
// descending, lower index first on a tie -- the CPU arm's partial_sort rule).
__global__ __launch_bounds__(D::N_EXPERT)
void k_ds4_router(const float * __restrict__ logits, const float * __restrict__ bias,
                  const int * __restrict__ hash, float * __restrict__ probs,
                  float * __restrict__ pb, int * __restrict__ ids, float * __restrict__ wr,
                  float * __restrict__ wn, float * __restrict__ ws, const int * __restrict__ miss,
                  unsigned * __restrict__ st) {
    __shared__ float sp[D::N_EXPERT], sv[D::N_EXPERT];
    __shared__ int   si[D::N_EXPERT], sel[D::N_EXPERT_USED];
    const int e = threadIdx.x;
    {   // rows (L5 step 5): grid.x is the row
        const size_t re = (size_t) blockIdx.x * D::N_EXPERT, ru = (size_t) blockIdx.x * D::N_EXPERT_USED;
        logits += re; probs += re; pb += re;
        if (hash) hash += ru;
        ids += ru; wr += ru; wn += ru; ws += ru;
    }
    const float l = logits[e];
    const float p = sqrtf(l > 20.0f ? l : logf(1.0f + expf(l)));
    sp[e] = p; probs[e] = p;
    if (hash) {
        pb[e] = p;
        if (e < D::N_EXPERT_USED) sel[e] = hash[e];
    } else {
        const float v = p + bias[e];
        pb[e] = v;
        bool taken = false;
        for (int k = 0; k < D::N_EXPERT_USED; ++k) {
            sv[e] = taken ? -INFINITY : v; si[e] = e;
            __syncthreads();
            for (int st = D::N_EXPERT / 2; st > 0; st >>= 1) {
                if (e < st) {
                    const float a = sv[e], b = sv[e + st];
                    if (b > a || (b == a && si[e + st] < si[e])) { sv[e] = b; si[e] = si[e + st]; }
                }
                __syncthreads();
            }
            if (e == 0) sel[k] = si[0];
            __syncthreads();
            if (e == sel[k]) taken = true;
            __syncthreads();
        }
    }
    __syncthreads();
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < D::N_EXPERT_USED; ++k) sum += sp[sel[k]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int k = 0; k < D::N_EXPERT_USED; ++k) {
            const float w = sp[sel[k]];
            ids[k] = sel[k]; wr[k] = w; wn[k] = w / sum; ws[k] = (w / sum) * D::EXPERT_WEIGHTS_SCALE;
        }
        // --adapt: the route counters (DEEPSEEK4.md section 12). One thread
        // a workgroup; a chunk's rows are workgroups running at once, so the
        // increments are atomic (integers: exact in any order). Nothing
        // above reads them.
        if (st)
            for (int k = 0; k < D::N_EXPERT_USED; ++k) {
                const int x = sel[k];
                atomicAdd(&st[x], 1u);
                if (miss[x] != 0) atomicAdd(&st[D::ADAPT_MISS], 1u);
                atomicAdd(&st[D::ADAPT_PICKS], 1u);
            }
    }
}

// --adapt: rewrite up to ADAPT_MAX_EDITS entries of one layer's table (the
// three address arrays and the miss bytes). One tiny launch on the main
// stream between tokens: every kernel queued before it read the old entries,
// every kernel queued after it reads the new ones (DEEPSEEK4.md section 12).
__global__ void k_ds4_table_edit(const void ** __restrict__ tg, const void ** __restrict__ tu,
                                 const void ** __restrict__ td, int * __restrict__ miss,
                                 const D::TableEdit ed) {
    const int i = threadIdx.x;
    if (i >= ed.n) return;
    const int e = ed.e[i];
    tg[e] = ed.g[i]; tu[e] = ed.u[i]; td[e] = ed.d[i]; miss[e] = ed.miss[i];
}

// --adapt-verify: 16-byte words of a (a VRAM slot) that differ from b (the
// device view of the host mirror), counted into *bad. Grid-stride.
__global__ void k_ds4_compare(const uint4 * __restrict__ a, const uint4 * __restrict__ b,
                              unsigned long long n16, unsigned long long * __restrict__ bad) {
    unsigned long long n = 0;
    for (unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += (unsigned long long) gridDim.x * blockDim.x) {
        const uint4 x = a[i], y = b[i];
        n += (x.x != y.x) | (x.y != y.y) | (x.z != y.z) | (x.w != y.w);
    }
    if (n) atomicAdd(bad, n);
}

__global__ void k_ds4_swiglu_clamp(const float * __restrict__ g, const float * __restrict__ u,
                                   float * __restrict__ h, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gg = fminf(g[i], limit);
    const float uu = fminf(fmaxf(u[i], -limit), limit);
    h[i] = gg / (1.f + expf(-gg)) * uu;
}

__global__ void k_ds4_moe_accum(const float * __restrict__ y, const float * __restrict__ w,
                                int n_used, int n, float * __restrict__ weighted, float * __restrict__ out,
                                int T) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= n * T) return;
    const int row = gi / n, i = gi % n;
    y += (size_t) row * n_used * n; weighted += (size_t) row * n_used * n;
    w += (size_t) row * n_used; out += (size_t) row * n;
    float a = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const float v = y[(size_t) e * n + i] * w[e];
        weighted[(size_t) e * n + i] = v;
        a = (e == 0) ? v : a + v;
    }
    out[i] = a;
}

// ------------------------------------------------------------- the host side --
// How a routed-expert launch addresses its items: direct (n assignments, one
// item each -- the decode token's six) or sorted by expert (256 items, the
// sort's tables).
struct Ds4MoeItems { int n = D::N_EXPERT_USED; const int * order = nullptr, * cnt = nullptr, * off = nullptr; };

class Ds4GpuOps : public D::Ds4Ops {
public:
    explicit Ds4GpuOps(GpuBackend & g) : g_(g) {
        dev_ensure(g_.dev_);
        // __constant__ memory is per device: copy the tables into THIS card's
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_iq2xxs), iq2xxs_grid,  sizeof(uint64_t) * 256));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_iq2s),   iq2s_grid,    sizeof(uint64_t) * 1024));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_iq3xxs), iq3xxs_grid,  sizeof(uint32_t) * 256));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_ksigns), ksigns_iq2xs, 128));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_kmask),  kmask_iq2xs,  8));
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ds4_kvfp4),  kvalues_mxfp4, 16));
        HIP_CHECK(hipMalloc(&miss_ctr_, sizeof(unsigned long long)));
        HIP_CHECK(hipMemset(miss_ctr_, 0, sizeof(unsigned long long)));
    }
    ~Ds4GpuOps() override {
        dev_ensure(g_.dev_);
        HIP_IGNORE(hipFree(po_)); HIP_IGNORE(hipFree(pm_)); HIP_IGNORE(hipFree(pl_));
        HIP_IGNORE(hipFree(stage_)); HIP_IGNORE(hipFree(sp_)); HIP_IGNORE(hipFree(miss_ctr_));
        HIP_IGNORE(hipFree(sort_order_)); HIP_IGNORE(hipFree(sort_cnt_)); HIP_IGNORE(hipFree(sort_off_));
        for (auto & e : ev_route_) HIP_IGNORE(hipEventDestroy(e));
        for (auto & e : ev_done_)  HIP_IGNORE(hipEventDestroy(e));
        if (up_ev_) HIP_IGNORE(hipEventDestroy(up_ev_));
        if (side_) HIP_IGNORE(hipStreamDestroy(side_));
        if (copy_) { HIP_IGNORE(hipStreamSynchronize(copy_)); HIP_IGNORE(hipStreamDestroy(copy_)); }
        for (hipEvent_t e : {ev_snap_, ev_fence_, ev_copied_}) if (e) HIP_IGNORE(hipEventDestroy(e));
        HIP_IGNORE(hipFree(stats_)); HIP_IGNORE(hipFree(cmp_ctr_));
        if (snap_host_) HIP_IGNORE(hipHostFree(snap_host_));
        for (void * h : host_) HIP_IGNORE(hipHostFree(h));
    }

    void gemv(const Mat & W, const float * x, float * y, int T) override {
        switch (W.type) {
            case FK_Q_Q4_K:    return launch_gemv<FK_Q_Q4_K>(W, x, y, T);
            case FK_Q_Q5_K:    return launch_gemv<FK_Q_Q5_K>(W, x, y, T);
            case FK_Q_IQ2_XXS: return launch_gemv<FK_Q_IQ2_XXS>(W, x, y, T);
            case FK_Q_IQ2_S:   return launch_gemv<FK_Q_IQ2_S>(W, x, y, T);
            case FK_Q_IQ3_XXS: return launch_gemv<FK_Q_IQ3_XXS>(W, x, y, T);
            case FK_Q_MXFP4:   return launch_gemv<FK_Q_MXFP4>(W, x, y, T);
            default:
                // Q8_0, Q6_K, BF16, F32: the tuned L0 kernel, charged to its own
                // class. T > 1 is its GEMM (k_gemm_batch at TILE 8, the qwen
                // prefill's; the split-K count is the matrix's alone, so a
                // column is the T == 1 GEMV's, PREFILL.md section 9).
                g_.trunk_cls_override_ = PC_DS4_TRUNK_L0;
                if (T <= 1) g_.gemv(W, x, y);
                else {
                    GemvJob j; j.W = &W; j.out = y;
                    g_.gemv_batch(&j, 1, x, -1, GX_PLAIN, nullptr, T);
                }
                g_.trunk_cls_override_ = -1;
                return;
        }
    }

    void hc_split(const float * m, const float * s, const float * b, float * pre, float * post,
                  float * comb, float eps, int iters, int T) override {
        g_.mark(PC_DS4_HC); g_.last_op_ = "ds4_hc_split";
        k_ds4_hc_split<<<T, D::HC, 0, g_.stream_>>>(m, s, b, pre, post, comb, eps, iters);
    }
    void hc_head_pre(const float * m, const float * s, const float * b, float * pre, float eps) override {
        g_.mark(PC_DS4_HC); g_.last_op_ = "ds4_hc_head_pre";
        k_ds4_hc_head_pre<<<1, WAVE, 0, g_.stream_>>>(m, s, b, pre, eps);
    }
    void hc_init(const float * x, float * H, int T) override {
        g_.mark(PC_DS4_HC); g_.last_op_ = "ds4_hc_init";
        k_ds4_hc_init<<<ceil_div(D::HC_DIM * T, BLK), BLK, 0, g_.stream_>>>(x, H, T);
    }
    void hc_weighted_sum(const float * H, const float * w, float * out, int T) override {
        g_.mark(PC_DS4_HC); g_.last_op_ = "ds4_hc_wsum";
        k_ds4_hc_wsum<<<ceil_div(D::N_EMBD * T, BLK), BLK, 0, g_.stream_>>>(H, w, out, T);
    }
    void hc_post(const float * x, const float * H, const float * post, const float * comb,
                 float * Ho, int T) override {
        g_.mark(PC_DS4_HC); g_.last_op_ = "ds4_hc_post";
        k_ds4_hc_post<<<ceil_div(D::HC_DIM * T, BLK), BLK, 0, g_.stream_>>>(x, H, post, comb, Ho, T);
    }

    void rope_tail(float * x, int n_rows, int row_len, int pos, int block_ratio,
                   const D::RopeParams & rp, bool inverse, int rows_per_pos) override {
        // the host half of ggml's rope cache, with the CPU arm's own libm
        const float theta_scale = powf(rp.freq_base, -2.0f / D::N_ROT);
        auto corr = [&](float beta) {
            return D::N_ROT * logf(rp.n_ctx_orig / (beta * 2 * (float) M_PI)) / (2 * logf(rp.freq_base));
        };
        const float c0 = std::max(0.0f, floorf(corr(rp.beta_fast)));
        const float c1 = std::min((float) (D::N_ROT - 1), ceilf(corr(rp.beta_slow)));
        float ms = rp.attn_factor;
        if (rp.ext_factor != 0.0f) ms *= 1.0f + 0.1f * logf(1.0f / rp.freq_scale);
        g_.mark(PC_DS4_ATTN_MISC); g_.last_op_ = "ds4_rope_tail";
        k_ds4_rope_tail<<<n_rows, D::N_ROT / 2, 0, g_.stream_>>>(x, row_len, pos, dp(), block_ratio,
            theta_scale, rp.freq_scale, rp.ext_factor, ms, c0, c1, inverse ? -1.0f : 1.0f, rows_per_pos);
    }
    void fwht(float * x, int n_rows, int n) override {
        if (n != D::IDX_DIM) { std::fprintf(stderr, "ds4 fwht: n=%d, the kernel is 128-wide\n", n); std::abort(); }
        g_.mark(PC_DS4_ATTN_MISC); g_.last_op_ = "ds4_fwht";
        k_ds4_fwht<<<n_rows, D::IDX_DIM, 0, g_.stream_>>>(x, 1.0f / sqrtf((float) n));
    }
    void store_raw(const float * kv, uint16_t * raw, int raw_rows, int pos, int T) override {
        g_.mark(PC_DS4_ATTN_MISC); g_.last_op_ = "ds4_store_raw";
        k_ds4_store_f16<<<dim3((unsigned) ceil_div(D::HEAD_DIM, BLK), (unsigned) T), BLK, 0, g_.stream_>>>(
            kv, (unsigned short *) raw, D::HEAD_DIM, 0, pos, dp(), raw_rows);
    }
    void store_block(const float * x, uint16_t * cache, int width, int ratio, int pos, int T) override {
        g_.mark(PC_DS4_POOL); g_.last_op_ = "ds4_store_block";
        k_ds4_store_f16<<<dim3((unsigned) ceil_div(width, BLK), (unsigned) T), BLK, 0, g_.stream_>>>(
            x, (unsigned short *) cache, width, ratio, pos, dp(), 1);
    }
    void ring_put(float * ring, int ring_rows, int width, int pos, const float * src, int T) override {
        g_.mark(PC_DS4_POOL); g_.last_op_ = "ds4_ring_put";
        k_ds4_ring_put<<<dim3((unsigned) ceil_div(width, BLK), (unsigned) T), BLK, 0, g_.stream_>>>(
            ring, ring_rows, width, pos, dp(), src);
    }
    void add_row(float * x, const float * table, int width, int ratio, int pos, int T) override {
        g_.mark(PC_DS4_POOL); g_.last_op_ = "ds4_add_row";
        k_ds4_add_row<<<dim3((unsigned) ceil_div(width, BLK), (unsigned) T), BLK, 0, g_.stream_>>>(
            x, table, width, ratio, pos, dp());
    }
    void copy_rows(const float * src, size_t ss, float * dst, size_t ds, int n_rows, int row_len) override {
        if (n_rows <= 0 || row_len <= 0) return;
        g_.mark(PC_DS4_ATTN_MISC); g_.last_op_ = "ds4_copy_rows";
        k_ds4_copy_rows<<<dim3((unsigned) ceil_div(row_len, BLK), (unsigned) n_rows), BLK, 0, g_.stream_>>>(
            src, ss, dst, ds, row_len);
    }

    void comp_pool(const float * rk, const float * rs, int ring, int R, int d_out, int pos,
                   float * out, int T) override {
        g_.mark(PC_DS4_POOL); g_.last_op_ = "ds4_comp_pool";
        k_ds4_comp_pool<<<dim3((unsigned) ceil_div(d_out, 128), (unsigned) T), 128, 0, g_.stream_>>>(
            rk, rs, ring, R, d_out, pos, dp(), 0, out);
    }
    void comp_pool_overlap(const float * rk, const float * rs, int ring, int R, int d_out, int pos,
                           float * out, int T) override {
        g_.mark(PC_DS4_POOL); g_.last_op_ = "ds4_comp_pool_overlap";
        k_ds4_comp_pool<<<dim3((unsigned) ceil_div(d_out, 128), (unsigned) T), 128, 0, g_.stream_>>>(
            rk, rs, ring, R, d_out, pos, dp(), 1, out);
    }

    void lid_scores(const float * q, const float * w, const uint16_t * keys, int pos,
                    float * scores, int T, int stride) override {
        // grid for the last row's position (a graph: the class's last); the
        // tiles past a row's own extent return at once
        const int n = (bpos(pos) + T) / D::CSA_RATIO;
        if (n <= 0) return;
        const int wgs = std::min(ceil_div(n, DS4_LID_TILE), DS4_LID_MAX_WGS);
        g_.mark(PC_DS4_LID); g_.last_op_ = "ds4_lid_scores";
        k_ds4_lid_scores<<<dim3((unsigned) wgs, (unsigned) T), DS4_LID_THREADS, 0, g_.stream_>>>(
            q, w, (const unsigned short *) keys, pos, dp(), scores, stride);
    }
    void topk(const float * s, int pos, int k, int * out, int T, int stride) override {
        g_.mark(PC_DS4_TOPK); g_.last_op_ = "ds4_topk";
        k_ds4_topk<<<T, DS4_TOPK_THREADS, 0, g_.stream_>>>(s, pos, dp(), k, out, stride);
    }

    void attn(const float * q, const uint16_t * raw, int raw_rows, int pos, const uint16_t * comp,
              int ratio, int cap, const int * ids, const float * sinks, float scale, float * out,
              int T) override {
        const int bp = bpos(pos) + T - 1;                          // the last row's position
        const int n_raw = bp - std::max(0, bp - D::N_SWA + 1) + 1;
        const int n_comp = ratio ? std::min((bp + 1) / ratio, cap) : 0;
        const int n_chunks = ceil_div(n_raw + n_comp, DS4_ATT_CHUNK);
        if (n_chunks > cap_chunks_ || rows_blk_ < 1) {
            std::fprintf(stderr, "ds4 attn: %d chunks > the %d reserved (ctx too small for reserve())\n",
                         n_chunks, cap_chunks_);
            std::abort();
        }
        const int cls = ratio == D::CSA_RATIO ? PC_DS4_ATTN_CSA : ratio == D::HCA_RATIO ? PC_DS4_ATTN_HCA
                                                                                         : PC_DS4_ATTN;
        for (int r0 = 0; r0 < T; r0 += rows_blk_) {
            const int nr = std::min(rows_blk_, T - r0);
            g_.mark(cls); g_.last_op_ = "ds4_attn_part";
            k_ds4_attn_part<<<dim3((unsigned) n_chunks, DS4_ATT_HGROUPS, (unsigned) nr), 256, 0, g_.stream_>>>(
                q, (const unsigned short *) raw, raw_rows, pos, dp(), (const unsigned short *) comp, ratio,
                cap, ids, scale, po_, pm_, pl_, r0, cap_chunks_);
            g_.mark(cls); g_.last_op_ = "ds4_attn_combine";
            k_ds4_attn_combine<<<dim3(D::N_HEAD, (unsigned) nr), D::HEAD_DIM, 0, g_.stream_>>>(
                po_, pm_, pl_, pos, dp(), ratio, cap, sinks, out, r0, cap_chunks_);
        }
    }
    // The partials: `cap_chunks_` chunks a row (the widest attention at this
    // context: the raw window plus every HCA row, or the CSA top-512), for a
    // block of rows_blk_ query rows bounded by --ds4-attn-mb (a 256k context
    // is 9 MB of partials a row, so a 256-row chunk runs in blocks).
    void reserve(int ctx, int t_max) override {
        const int max_keys = D::N_SWA + std::max(D::IDX_TOP_K, ctx / D::HCA_RATIO + 1);
        const int chunks = ceil_div(max_keys, DS4_ATT_CHUNK);
        const size_t per_row = (size_t) chunks * D::N_HEAD * (D::HEAD_DIM + 2) * sizeof(float);
        int rows = (int) std::max<size_t>(1, ((size_t) attn_mb_ << 20) / per_row);
        rows = std::min(rows, std::max(1, t_max));
        if (chunks <= cap_chunks_ && rows <= rows_blk_) return;
        dev_ensure(g_.dev_);
        HIP_IGNORE(hipFree(po_)); HIP_IGNORE(hipFree(pm_)); HIP_IGNORE(hipFree(pl_));
        cap_chunks_ = std::max(chunks, cap_chunks_);
        rows_blk_ = std::max(rows, rows_blk_);
        const size_t nch = (size_t) cap_chunks_ * rows_blk_;
        HIP_CHECK(hipMalloc(&po_, nch * D::N_HEAD * D::HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMalloc(&pm_, nch * D::N_HEAD * sizeof(float)));
        HIP_CHECK(hipMalloc(&pl_, nch * D::N_HEAD * sizeof(float)));
        ++g_.epoch_;              // a graph captured before this holds the freed pointers
        // the expert sort of a chunk (--ds4-expert-gather)
        if (t_max > 1 && !sort_order_) {
            HIP_CHECK(hipMalloc(&sort_order_, (size_t) DS4_MOE_SORT_MAX * sizeof(int)));
            HIP_CHECK(hipMalloc(&sort_cnt_, D::N_EXPERT * sizeof(int)));
            HIP_CHECK(hipMalloc(&sort_off_, D::N_EXPERT * sizeof(int)));
        }
    }

    void router(const float * logits, const float * bias, const int32_t * hash, float * probs,
                float * pb, int * ids, float * wr, float * wn, float * ws, const int * miss,
                uint32_t * stats, int T) override {
        g_.mark(PC_ROUTER); g_.last_op_ = "ds4_router";
        k_ds4_router<<<T, D::N_EXPERT, 0, g_.stream_>>>(logits, bias, hash, probs, pb, ids, wr, wn, ws,
                                                        miss, (unsigned *) stats);
    }

    // ---- adaptive placement (--adapt 1; DEEPSEEK4.md section 12) ---------------
    // Every entry sets THIS card current first (CLAUDE.md's HIP multi-GPU
    // rule): the streams, events and allocations below are per card.
    uint32_t * adapt_stats_alloc(size_t n) override {
        dev_ensure(g_.dev_);
        uint32_t * p = nullptr;
        HIP_CHECK(hipMalloc(&p, n * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(p, 0, n * sizeof(uint32_t)));
        HIP_CHECK(hipHostMalloc(&snap_host_, n * sizeof(uint32_t), hipHostMallocDefault));
        HIP_CHECK(hipEventCreateWithFlags(&ev_snap_, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&ev_fence_, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&ev_copied_, hipEventDisableTiming));
        // the swap copies: their own stream, at the LOWEST priority the
        // device offers, so the main and staging streams' work goes first
        int least = 0, greatest = 0;
        HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
        HIP_CHECK(hipStreamCreateWithPriority(&copy_, hipStreamNonBlocking, least));
        stats_ = p;
        return p;
    }
    void adapt_snapshot(const uint32_t * st, size_t n) override {
        g_.mark(PC_DS4_ADAPT); g_.last_op_ = "ds4_adapt_snapshot";
        HIP_CHECK(hipMemcpyAsync(snap_host_, st, n * sizeof(uint32_t), hipMemcpyDeviceToHost, g_.stream_));
        HIP_CHECK(hipEventRecord(ev_snap_, g_.stream_));
        snap_pending_ = true;
    }
    const uint32_t * adapt_snapshot_poll() override {
        if (!snap_pending_) return nullptr;
        dev_ensure(g_.dev_);
        const hipError_t r = hipEventQuery(ev_snap_);
        if (r == hipErrorNotReady) return nullptr;
        HIP_CHECK(r);
        snap_pending_ = false;
        return snap_host_;
    }
    void adapt_table_edit(const D::ExpertTable & t, const D::TableEdit & ed) override {
        if (ed.n <= 0) return;
        g_.mark(PC_DS4_ADAPT); g_.last_op_ = "ds4_table_edit";
        k_ds4_table_edit<<<1, WAVE, 0, g_.stream_>>>(const_cast<const void **>(t.gate),
            const_cast<const void **>(t.up), const_cast<const void **>(t.down),
            const_cast<int *>(t.miss_bytes), ed);
    }
    void adapt_copy_begin() override {
        dev_ensure(g_.dev_);
        HIP_CHECK(hipEventRecord(ev_fence_, g_.stream_));        // behind everything queued so far
        HIP_CHECK(hipStreamWaitEvent(copy_, ev_fence_, 0));
    }
    void adapt_copy(void * dst, const void * src, size_t n) override {
        dev_ensure(g_.dev_);
        HIP_CHECK(hipMemcpyAsync(dst, src, n, hipMemcpyHostToDevice, copy_));
    }
    void adapt_copy_end() override {
        dev_ensure(g_.dev_);
        HIP_CHECK(hipEventRecord(ev_copied_, copy_));
        copy_pending_ = true;
    }
    bool adapt_copy_landed() override {
        if (!copy_pending_) return true;
        dev_ensure(g_.dev_);
        const hipError_t r = hipEventQuery(ev_copied_);
        if (r == hipErrorNotReady) return false;
        HIP_CHECK(r);
        return true;
    }
    void adapt_copy_join() override {
        if (!copy_pending_) return;
        g_.mark(PC_DS4_ADAPT); g_.last_op_ = "ds4_adapt_join";
        // the batch has landed (adapt_copy_landed), so this wait costs
        // nothing; it is the edge that orders the copy before the install
        // edit for the device, not only for the host that observed it
        HIP_CHECK(hipStreamWaitEvent(g_.stream_, ev_copied_, 0));
        copy_pending_ = false;
    }
    long long adapt_compare(const void * a, const void * b, size_t n) override {
        if (n % 16) { std::fprintf(stderr, "ds4 adapt_compare: %zu bytes, not 16-byte words\n", n); std::abort(); }
        if (!cmp_ctr_) { dev_ensure(g_.dev_); HIP_CHECK(hipMalloc(&cmp_ctr_, sizeof(unsigned long long))); }
        dev_ensure(g_.dev_);
        HIP_CHECK(hipMemsetAsync(cmp_ctr_, 0, sizeof(unsigned long long), g_.stream_));
        g_.mark(PC_DS4_ADAPT); g_.last_op_ = "ds4_compare";
        k_ds4_compare<<<256, 256, 0, g_.stream_>>>((const uint4 *) a, (const uint4 *) b, n / 16, cmp_ctr_);
        unsigned long long v = 0;
        g_.download(&v, cmp_ctr_, sizeof(v));
        return (long long) v * 16;       // an upper bound in bytes: whole 16-byte words
    }

    // ---- the miss path --------------------------------------------------------
    void reserve_moe(size_t max_slab, int n_slots) override {
        dev_ensure(g_.dev_);
        if (n_slots <= 0) return;
        max_slab = (max_slab + 255) & ~(size_t) 255;
        slot_bytes_ = max_slab;
        n_slots_ = n_slots;
        HIP_CHECK(hipMalloc(&stage_, (size_t) n_slots * D::N_EXPERT_USED * max_slab));
        HIP_CHECK(hipMalloc(&sp_, (size_t) n_slots * D::N_EXPERT_USED * 3 * sizeof(void *)));
        // the side stream and its events, created with THIS card current
        HIP_CHECK(hipStreamCreateWithFlags(&side_, hipStreamNonBlocking));
        ev_route_.resize((size_t) n_slots); ev_done_.resize((size_t) n_slots);
        for (int i = 0; i < n_slots; ++i) {
            HIP_CHECK(hipEventCreateWithFlags(&ev_route_[(size_t) i], hipEventDisableTiming));
            HIP_CHECK(hipEventCreateWithFlags(&ev_done_[(size_t) i], hipEventDisableTiming));
        }
        staged_bytes_ = (size_t) n_slots * D::N_EXPERT_USED * max_slab;
    }
    // Fork after the router (or, for a token-id-routed layer, after the ids'
    // upload), copy on the side stream, and leave an event to join on.
    void moe_stage(const D::ExpertTable & t, const int * ids, int slot) override {
        if (slot < 0 || slot >= n_slots_) return;
        const size_t need = t.sz_g + t.sz_u + t.sz_d;
        if (need > slot_bytes_) { std::fprintf(stderr, "ds4 stage: slab %zu > slot %zu\n", need, slot_bytes_); std::abort(); }
        g_.mark(PC_DS4_MISS_WAIT); g_.last_op_ = "ds4_stage_fork";
        HIP_CHECK(hipEventRecord(ev_route_[(size_t) slot], g_.stream_));
        HIP_CHECK(hipStreamWaitEvent(side_, ev_route_[(size_t) slot], 0));
        const unsigned wgs = (unsigned) std::max(1, stage_wgs_ / D::N_EXPERT_USED);
        k_ds4_stage<<<dim3(wgs, D::N_EXPERT_USED), DS4_STAGE_THREADS, 0, side_>>>(
            t.gate, t.up, t.down, ids, t.miss_bytes, t.sz_g, t.sz_u, t.sz_d,
            stage_ + (size_t) slot * D::N_EXPERT_USED * slot_bytes_, slot_bytes_,
            sp_ + (size_t) slot * D::N_EXPERT_USED * 3, nullptr);   // counted by count_misses
        HIP_CHECK(hipEventRecord(ev_done_[(size_t) slot], side_));
        pending_[slot & 7] = true;
    }
    // Staged: the RESIDENT slots first, straight from the table, while the
    // side stream is still copying the missed ones (L5 step 3c); then the join
    // and the missed slots from their staged copies. Every slot runs the same
    // kernel body on the same bytes as before -- only which launch it is in
    // changes -- so the result is bit-identical.
    void moe_gate_up(const D::ExpertTable & t, const int * ids, const float * x, float * yg,
                     float * yu, int slot, int T) override {
        if (T > 1) {
            // a chunk: in place through the table (a missed row crosses PCIe
            // once for all of its assignments when expert-major)
            const Items it = items(ids, T);
            launch_moe(t, t.type_gu, t.up,   ids, nullptr, 1, t.row_gu, t.rows_gu, t.K_gu, x, 0, yu, PC_DS4_EXP_GU, it);
            launch_moe(t, t.type_gu, t.gate, ids, nullptr, 0, t.row_gu, t.rows_gu, t.K_gu, x, 0, yg, PC_DS4_EXP_GU, it);
            return;
        }
        if (slot < 0 || slot >= n_slots_) {
            launch_moe(t, t.type_gu, t.up,   ids, nullptr, 1, t.row_gu, t.rows_gu, t.K_gu, x, 0, yu, PC_DS4_EXP_GU);
            launch_moe(t, t.type_gu, t.gate, ids, nullptr, 0, t.row_gu, t.rows_gu, t.K_gu, x, 0, yg, PC_DS4_EXP_GU);
            return;
        }
        launch_moe_f(t, t.type_gu, t.up,   ids, nullptr, 1, 1, t.row_gu, t.rows_gu, t.K_gu, x, 0, yu, PC_DS4_EXP_GU);
        launch_moe_f(t, t.type_gu, t.gate, ids, nullptr, 0, 1, t.row_gu, t.rows_gu, t.K_gu, x, 0, yg, PC_DS4_EXP_GU);
        const void * const * sp = join(slot);
        launch_moe_f(t, t.type_gu, t.up,   ids, sp, 1, 2, t.row_gu, t.rows_gu, t.K_gu, x, 0, yu, PC_DS4_EXP_GU);
        launch_moe_f(t, t.type_gu, t.gate, ids, sp, 0, 2, t.row_gu, t.rows_gu, t.K_gu, x, 0, yg, PC_DS4_EXP_GU);
    }
    void moe_down(const D::ExpertTable & t, const int * ids, const float * h, float * y, int slot,
                  int T) override {
        if (T > 1) {
            // the sort of moe_gate_up is still valid: same ids, same stream
            launch_moe(t, t.type_d, t.down, ids, nullptr, 2, t.row_d, t.rows_d, t.K_d, h, 1, y,
                       PC_DS4_EXP_DOWN, items_last_);
            return;
        }
        // the join happened in moe_gate_up: the staged pointers are all valid
        const void * const * sp = (slot >= 0 && slot < n_slots_) ? sp_ + (size_t) slot * D::N_EXPERT_USED * 3 : nullptr;
        launch_moe(t, t.type_d, t.down, ids, sp, 2, t.row_d, t.rows_d, t.K_d, h, 1, y, PC_DS4_EXP_DOWN, Items());
    }
    void swiglu_clamp(const float * g, const float * u, float * h, int n, float limit) override {
        g_.mark(PC_DS4_MOE_TAIL); g_.last_op_ = "ds4_swiglu_clamp";
        k_ds4_swiglu_clamp<<<ceil_div(n, BLK), BLK, 0, g_.stream_>>>(g, u, h, n, limit);
    }
    void moe_accum(const float * y, const float * w, int n_used, int n, float * weighted,
                   float * out, int T) override {
        g_.mark(PC_DS4_MOE_TAIL); g_.last_op_ = "ds4_moe_accum";
        k_ds4_moe_accum<<<ceil_div(n * T, BLK), BLK, 0, g_.stream_>>>(y, w, n_used, n, weighted, out, T);
    }

    // ---- profiling ----------------------------------------------------------------
    void set_profile(bool on) override { prof_ = on; }
    void set_staged_loads(int on) override { staged_ok_ = on != 0; }
    void set_stage_wgs(int n) override { if (n > 0) stage_wgs_ = n; }
    void prof_gap() override { if (prof_) { g_.mark(PC_GAP); g_.last_op_ = "ds4_segment_end"; } }
    void upstream_wait(D::Ds4Ops & prev) override {
        if (!prof_) return;
        Ds4GpuOps & p = static_cast<Ds4GpuOps &>(prev);
        if (!up_ev_) { dev_ensure(p.g_.dev_); HIP_CHECK(hipEventCreateWithFlags(&up_ev_, hipEventDisableTiming)); }
        dev_ensure(p.g_.dev_);                       // the event belongs to the upstream card
        HIP_CHECK(hipEventRecord(up_ev_, p.g_.stream_));
        g_.mark(PC_DS4_UPSTREAM); g_.last_op_ = "ds4_upstream_wait";   // ensures THIS card
        HIP_CHECK(hipStreamWaitEvent(g_.stream_, up_ev_, 0));
    }
    void count_misses(const D::ExpertTable & t, const int * ids, int T) override {
        if (!prof_) return;
        g_.mark(PC_ROUTER); g_.last_op_ = "ds4_miss_count";
        const int n = T * D::N_EXPERT_USED;
        k_ds4_miss_count<<<ceil_div(n, 256), 256, 0, g_.stream_>>>(ids, t.miss_bytes, miss_ctr_, n);
    }
    void set_expert_gather(int on) override { expert_gather_ = on != 0; }
    void set_attn_mb(int mb) override { if (mb > 0) attn_mb_ = mb; }
    double miss_bytes_total() override {
        unsigned long long v = 0;
        g_.download(&v, miss_ctr_, sizeof(v));
        return (double) v;
    }
    void reset_miss_count() override {
        dev_ensure(g_.dev_);
        HIP_CHECK(hipMemsetAsync(miss_ctr_, 0, sizeof(unsigned long long), g_.stream_));
    }

    // Pinned host memory, mapped into this card's address space: the miss
    // path's expert copies (DEEPSEEK4.md section 5).
    const void * alloc_host_mapped(size_t bytes, void ** host) override {
        dev_ensure(g_.dev_);
        void * h = nullptr;
        HIP_CHECK(hipHostMalloc(&h, bytes, hipHostMallocMapped | hipHostMallocPortable));
        void * d = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&d, h, 0));
        host_.push_back(h);
        *host = h;
        return d;
    }

private:
    // While a graph is captured: the device position the graph reads, and the
    // last position of the class for sizing grids. Eagerly: none, and `pos`.
    const int * dp() const { return g_.capturing_ ? g_.gpos_ : nullptr; }
    int bpos(int pos) const { return g_.capturing_ ? g_.graph_bound_ - 1 : pos; }
    // The main stream joins the side stream's copy for `slot` (once). The
    // interval charged to PC_DS4_MISS_WAIT is the time the card sat waiting.
    const void * const * join(int slot) {
        if (slot < 0 || slot >= n_slots_) return nullptr;
        if (pending_[slot & 7]) {
            g_.mark(PC_DS4_MISS_WAIT); g_.last_op_ = "ds4_stage_join";
            HIP_CHECK(hipStreamWaitEvent(g_.stream_, ev_done_[(size_t) slot], 0));
            pending_[slot & 7] = false;
        }
        return sp_ + (size_t) slot * D::N_EXPERT_USED * 3;
    }
    static bool aligned16(const void * p, size_t rb) {
        return ((uintptr_t) p % 16 == 0) && (rb % 16 == 0);
    }
    template <int TY>
    void launch_gemv(const Mat & W, const float * x, float * y, int T) {
        g_.mark(PC_DS4_TRUNK_KQ, -1, W.row_bytes * (size_t) W.rows);
        g_.last_op_ = "ds4_gemv";
        const int staged = staged_ok_ && aligned16(W.base, W.row_bytes) && W.row_bytes <= (size_t) ds4_row_max<TY>();
        k_ds4_gemv<TY><<<ceil_div((int) W.rows, DS4_GEMV_ROWS), DS4_GEMV_THREADS, 0, g_.stream_>>>(
            (const unsigned char *) W.base, W.row_bytes, (int) W.rows, (int) W.K, x, y, staged, T < 1 ? 1 : T);
    }
    using Items = Ds4MoeItems;
    Items items_last_;
    Items items(const int * ids, int T) {
        Items it;
        it.n = T * D::N_EXPERT_USED;
        if (expert_gather_ && sort_order_ && it.n <= DS4_MOE_SORT_MAX) {
            g_.mark(PC_DS4_EXP_SORT); g_.last_op_ = "ds4_moe_sort";
            k_ds4_moe_sort<<<1, D::N_EXPERT, 0, g_.stream_>>>(ids, it.n, sort_cnt_, sort_off_, sort_order_);
            it.n = D::N_EXPERT; it.order = sort_order_; it.cnt = sort_cnt_; it.off = sort_off_;
        }
        items_last_ = it;
        return it;
    }
    template <int TY>
    void moe_launch(const D::ExpertTable & t, const void * const * tab, const int * ids,
                    const void * const * sp, int which, int filter, size_t rb, int rows, int K,
                    const float * x, int x_per_slot, float * y, const Items & it) {
        // every table entry and staging pointer is 16-byte aligned by
        // construction (hipMalloc / hipHostMalloc bases, slab offsets that are
        // multiples of 16: ds4_model.cpp place_experts), so the row size decides
        const int staged = staged_ok_ && (rb % 16 == 0) && rb <= (size_t) ds4_row_max<TY>();
        const dim3 grid((unsigned) ceil_div(rows, DS4_GEMV_ROWS), (unsigned) it.n);
        k_ds4_moe<TY><<<grid, DS4_GEMV_THREADS, 0, g_.stream_>>>(tab, ids, sp, which, t.miss_bytes,
            filter, rb, rows, K, x, x_per_slot, y, staged, it.order, it.cnt, it.off);
    }
    // one launch with an explicit filter (1 resident slots, 2 missed slots)
    void launch_moe_f(const D::ExpertTable & t, int type, const void * const * tab, const int * ids,
                      const void * const * sp, int which, int filter, size_t rb, int rows, int K,
                      const float * x, int x_per_slot, float * y, int cls) {
        const Items it;                              // the decode token: six direct items
        g_.mark(cls); g_.last_op_ = "ds4_moe";
        switch (type) {
            case FK_Q_IQ2_XXS: moe_launch<FK_Q_IQ2_XXS>(t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
            case FK_Q_IQ2_S:   moe_launch<FK_Q_IQ2_S>  (t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
            case FK_Q_IQ3_XXS: moe_launch<FK_Q_IQ3_XXS>(t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
            case FK_Q_MXFP4:   moe_launch<FK_Q_MXFP4>  (t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
            default:
                std::fprintf(stderr, "ds4 moe: expert format %d has no kernel\n", type);
                std::abort();
        }
    }
    void launch_moe(const D::ExpertTable & t, int type, const void * const * tab, const int * ids,
                    const void * const * sp, int which, size_t rb, int rows, int K,
                    const float * x, int x_per_slot, float * y, int cls, const Items & it = Items()) {
        // In place (no staging) under --profile, the resident and the missed
        // slots go in two launches so their device times separate; the work
        // per slot is identical either way.
        const bool split = prof_ && !sp;
        for (int pass = 0; pass < (split ? 2 : 1); ++pass) {
            const int filter = split ? pass + 1 : 0;
            g_.mark(filter == 2 ? PC_DS4_EXP_MISS : cls); g_.last_op_ = "ds4_moe";
            switch (type) {
                case FK_Q_IQ2_XXS: moe_launch<FK_Q_IQ2_XXS>(t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
                case FK_Q_IQ2_S:   moe_launch<FK_Q_IQ2_S>  (t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
                case FK_Q_IQ3_XXS: moe_launch<FK_Q_IQ3_XXS>(t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
                case FK_Q_MXFP4:   moe_launch<FK_Q_MXFP4>  (t, tab, ids, sp, which, filter, rb, rows, K, x, x_per_slot, y, it); break;
                default:
                    std::fprintf(stderr, "ds4 moe: expert format %d has no kernel\n", type);
                    std::abort();
            }
        }
    }
    GpuBackend & g_;
    float * po_ = nullptr, * pm_ = nullptr, * pl_ = nullptr;
    int cap_chunks_ = 0, rows_blk_ = 0;
    int attn_mb_ = 256;                              // --ds4-attn-mb: partials for a row block
    // a chunk's expert sort (--ds4-expert-gather, on by default for chunks)
    bool expert_gather_ = true;
    int * sort_order_ = nullptr, * sort_cnt_ = nullptr, * sort_off_ = nullptr;
    std::vector<void *> host_;
    // the miss path
    hipStream_t side_ = nullptr;
    std::vector<hipEvent_t> ev_route_, ev_done_;
    unsigned char * stage_ = nullptr;
    const void ** sp_ = nullptr;
    size_t slot_bytes_ = 0, staged_bytes_ = 0;
    int n_slots_ = 0;
    bool pending_[8] = {false, false, false, false, false, false, false, false};
    bool staged_ok_ = true;
    int  stage_wgs_ = 64;
    hipEvent_t up_ev_ = nullptr;                     // created on the UPSTREAM card
    // profiling
    bool prof_ = false;
    unsigned long long * miss_ctr_ = nullptr;
    // --adapt
    uint32_t * stats_ = nullptr;
    uint32_t * snap_host_ = nullptr;
    hipStream_t copy_ = nullptr;
    hipEvent_t ev_snap_ = nullptr, ev_fence_ = nullptr, ev_copied_ = nullptr;
    bool snap_pending_ = false, copy_pending_ = false;
    unsigned long long * cmp_ctr_ = nullptr;
};
