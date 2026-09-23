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

template <int TY>
__global__ __launch_bounds__(DS4_GEMV_THREADS)
void k_ds4_gemv(const unsigned char * __restrict__ W, size_t row_bytes, int rows, int K,
                const float * __restrict__ x, float * __restrict__ y) {
    const int lane = threadIdx.x & (WAVE - 1);
    const int r = blockIdx.x * DS4_GEMV_ROWS + (threadIdx.x / WAVE);
    if (r >= rows) return;                                   // wave-uniform
    float a = ds4_row_lane<TY>(W + (size_t) r * row_bytes, x, K, lane);
    a = wave_sum(a);
    if (lane == 0) y[r] = a;
}

// The routed experts: grid.y is the slot k, the expert is ids[k], and its
// address comes from the layer's TABLE -- VRAM when resident, the
// device-mapped pinned host copy when not. Nothing in the kernel knows which.
template <int TY>
__global__ __launch_bounds__(DS4_GEMV_THREADS)
void k_ds4_moe(const void * const * __restrict__ tab, const int * __restrict__ ids,
               size_t row_bytes, int rows, int K, const float * __restrict__ x, int x_per_slot,
               float * __restrict__ y) {
    const int slot = blockIdx.y;
    const int lane = threadIdx.x & (WAVE - 1);
    const int r = blockIdx.x * DS4_GEMV_ROWS + (threadIdx.x / WAVE);
    if (r >= rows) return;
    const unsigned char * W = (const unsigned char *) tab[ids[slot]];
    const float * xs = x + (x_per_slot ? (size_t) slot * K : 0);
    float a = ds4_row_lane<TY>(W + (size_t) r * row_bytes, xs, K, lane);
    a = wave_sum(a);
    if (lane == 0) y[(size_t) slot * rows + r] = a;
}

// -------------------------------------------------------- hyper-connections --
// One thread: 24 inputs and a 4x4 Sinkhorn loop, the CPU op verbatim.
__global__ void k_ds4_hc_split(const float * __restrict__ m, const float * __restrict__ s,
                               const float * __restrict__ b, float * __restrict__ pre,
                               float * __restrict__ post, float * __restrict__ comb,
                               float eps, int iters) {
    if (threadIdx.x != 0) return;
    constexpr int H = D::HC;
    for (int h = 0; h < H; ++h) {
        pre[h]  = ds4_sigmoid(m[h] * s[0] + b[h]) * 1.0f + eps;
        post[h] = ds4_sigmoid(m[H + h] * s[1] + b[H + h]) * 2.0f;
    }
    float c[H * H];
    for (int isrc = 0; isrc < H; ++isrc) {
        float mx = -INFINITY;
        for (int idst = 0; idst < H; ++idst) {
            const int idx = idst + H * isrc;
            c[idx] = m[2 * H + idx] * s[2] + b[2 * H + idx];
            mx = fmaxf(mx, c[idx]);
        }
        float sum = 0.0f;
        for (int idst = 0; idst < H; ++idst) { const int idx = idst + H * isrc; c[idx] = expf(c[idx] - mx); sum += c[idx]; }
        const float inv = 1.0f / sum;
        for (int idst = 0; idst < H; ++idst) { const int idx = idst + H * isrc; c[idx] = c[idx] * inv + eps; }
    }
    for (int it = 0; it < 2 * iters - 1; ++it) {
        const bool cols = (it % 2 == 0);     // cols, then (rows, cols) x (iters-1)
        for (int o = 0; o < H; ++o) {
            float sum = eps;
            for (int i = 0; i < H; ++i) sum += cols ? c[o + H * i] : c[i + H * o];
            const float inv = 1.0f / sum;
            for (int i = 0; i < H; ++i) { if (cols) c[o + H * i] *= inv; else c[i + H * o] *= inv; }
        }
    }
    for (int i = 0; i < H * H; ++i) comb[i] = c[i];
}

__global__ void k_ds4_hc_head_pre(const float * __restrict__ m, const float * __restrict__ s,
                                  const float * __restrict__ b, float * __restrict__ pre, float eps) {
    const int h = threadIdx.x;
    if (h < D::HC) pre[h] = ds4_sigmoid(m[h] * s[0] + b[h]) * 1.0f + eps;
}

__global__ void k_ds4_hc_wsum(const float * __restrict__ H, const float * __restrict__ w,
                              float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D::N_EMBD) return;
    float sum = 0.0f;
    for (int h = 0; h < D::HC; ++h) sum += H[(size_t) h * D::N_EMBD + i] * w[h];
    out[i] = sum;
}

__global__ void k_ds4_hc_post(const float * __restrict__ x, const float * __restrict__ H,
                              const float * __restrict__ post, const float * __restrict__ comb,
                              float * __restrict__ Ho) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D::HC_DIM) return;
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
__global__ void k_ds4_rope_tail(float * __restrict__ x, int row_len, float pos, float theta_scale,
                                float freq_scale, float ext, float mscale, float c0, float c1,
                                float sin_sign) {
    const int i = threadIdx.x;                  // pair, 0..N_ROT/2-1
    float theta = pos;
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

__global__ void k_ds4_to_f16(const float * __restrict__ x, unsigned short * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half_as_ushort(__float2half_rn(x[i]));
}

// ------------------------------------------------------ compressor pools --
// One thread a dimension: softmax over the candidates, then the weighted sum
// (the CPU arm sums in double; here in float -- a summation-order difference).
__global__ void k_ds4_comp_pool(const float * __restrict__ rk, const float * __restrict__ rs,
                                int ring, int R, int d_out, int blk, int overlap,
                                float * __restrict__ out) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_out) return;
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
// dot: 32 FMAs, a two-step shuffle within the four, relu, times w_h. q (32 KB)
// and w are in LDS; each group of four reads four distinct q addresses, the
// rest of the wave the same four (broadcast). Four consecutive lanes read the
// key's 256 contiguous bytes, so the key stream is fully coalesced. The first
// shape tried -- one thread a key, all 64 head sums in registers -- asked for
// 256 VGPR and spilled 404 B a lane; so did 16 heads a pass (246 VGPR, the
// compiler hoisted the unrolled LDS reads). Heads are summed in ascending
// order; a dot is ((p0+p1)+(p2+p3)) of the four partials.
constexpr int DS4_LID_THREADS = 256;
constexpr int DS4_LID_LANES   = 4;                                  // lanes a key
constexpr int DS4_LID_KEYS    = DS4_LID_THREADS / DS4_LID_LANES;    // 64 keys a workgroup
__global__ __launch_bounds__(DS4_LID_THREADS)
void k_ds4_lid_scores(const float * __restrict__ q, const float * __restrict__ w,
                      const unsigned short * __restrict__ keys, int n, float * __restrict__ scores) {
    __shared__ float sq[D::IDX_N_HEAD * D::IDX_DIM];
    __shared__ float sw[D::IDX_N_HEAD];
    for (int i = threadIdx.x; i < D::IDX_N_HEAD * D::IDX_DIM; i += DS4_LID_THREADS) sq[i] = q[i];
    if (threadIdx.x < D::IDX_N_HEAD) sw[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    const int part = threadIdx.x & (DS4_LID_LANES - 1);
    const int b = blockIdx.x * DS4_LID_KEYS + threadIdx.x / DS4_LID_LANES;
    if (b >= n) return;                         // uniform across the key's four lanes
    constexpr int PD = D::IDX_DIM / DS4_LID_LANES;                  // 32 dims a lane
    const uint4 * k4 = (const uint4 *) (keys + (size_t) b * D::IDX_DIM + part * PD);
    float kf[PD];
#pragma unroll
    for (int j = 0; j < PD / 8; ++j) {
        const uint4 u = k4[j];
        const unsigned int w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) { kf[8 * j + 2 * i] = ds4_h2f(w4[i] & 0xFFFFu); kf[8 * j + 2 * i + 1] = ds4_h2f(w4[i] >> 16); }
    }
    float sc = 0.0f;
    for (int h = 0; h < D::IDX_N_HEAD; ++h) {
        const float * qh = sq + h * D::IDX_DIM + part * PD;
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < PD; ++i) a = fmaf(qh[i], kf[i], a);
        a += __shfl_xor(a, 1, WAVE);
        a += __shfl_xor(a, 2, WAVE);
        sc += fmaxf(a, 0.0f) * sw[h];
    }
    if (part == 0) scores[b] = sc;
}

// Top-k as a SET: one workgroup, a 4-pass 8-bit MSB-first radix select on
// the order-preserving uint image of the scores (n <= 65 536 at 256k), then
// everything above the threshold plus as many ties as the count needs.
constexpr int DS4_TOPK_THREADS = 1024;
__device__ __forceinline__ unsigned ds4_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ __launch_bounds__(DS4_TOPK_THREADS)
void k_ds4_topk(const float * __restrict__ s, int n, int k, int * __restrict__ out) {
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_prefix, sh_mask, sh_wr, sh_tie;
    __shared__ int sh_kr;
    const int t = threadIdx.x;
    unsigned prefix = 0, mask = 0;
    int kr = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = t; i < 256; i += DS4_TOPK_THREADS) hist[i] = 0;
        __syncthreads();
        for (int i = t; i < n; i += DS4_TOPK_THREADS) {
            const unsigned u = ds4_key(s[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (t == 0) {
            unsigned cum = 0; int sel = 0;
            for (int bb = 255; bb >= 0; --bb) {
                if (cum + hist[bb] >= (unsigned) kr) { sel = bb; break; }
                cum += hist[bb];
            }
            sh_kr = kr - (int) cum;
            sh_prefix = prefix | ((unsigned) sel << shift);
            sh_mask = mask | (255u << shift);
        }
        __syncthreads();
        prefix = sh_prefix; mask = sh_mask; kr = sh_kr;
        __syncthreads();
    }
    if (t == 0) { sh_wr = 0; sh_tie = 0; }
    __syncthreads();
    for (int i = t; i < n; i += DS4_TOPK_THREADS) {
        const unsigned u = ds4_key(s[i]);
        if (u > prefix) out[atomicAdd(&sh_wr, 1u)] = i;
        else if (u == prefix && atomicAdd(&sh_tie, 1u) < (unsigned) kr) out[atomicAdd(&sh_wr, 1u)] = i;
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
__global__ __launch_bounds__(256)
void k_ds4_attn_part(const float * __restrict__ q, const unsigned short * __restrict__ raw,
                     int raw0, int n_raw, const unsigned short * __restrict__ comp,
                     const int * __restrict__ ids, int n_comp, float scale,
                     float * __restrict__ po, float * __restrict__ pm, float * __restrict__ pl) {
    __shared__ unsigned kl[DS4_ATT_CHUNK * DS4_ATT_STRIDE];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = threadIdx.x;
    const int n_keys = n_raw + n_comp;
    const int j0 = chunk * DS4_ATT_CHUNK;
    const int nk = min(DS4_ATT_CHUNK, n_keys - j0);
    for (int idx = t; idx < DS4_ATT_CHUNK * (D::HEAD_DIM / 2); idx += 256) {
        const int r = idx / (D::HEAD_DIM / 2), c = idx % (D::HEAD_DIM / 2);
        if (r < nk) {
            const int g = j0 + r;
            const unsigned short * row = (g < n_raw)
                ? raw + (size_t) ((raw0 + g) % D::N_SWA) * D::HEAD_DIM
                : comp + (size_t) (ids ? ids[g - n_raw] : g - n_raw) * D::HEAD_DIM;
            kl[r * DS4_ATT_STRIDE + c] = ((const unsigned *) row)[c];
        }
    }
    __syncthreads();
    const int wave = t / WAVE, lane = t & (WAVE - 1);
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
                        const float * __restrict__ pl, int n_chunks,
                        const float * __restrict__ sinks, float * __restrict__ out) {
    const int h = blockIdx.x, d = threadIdx.x;
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
                  float * __restrict__ wn, float * __restrict__ ws) {
    __shared__ float sp[D::N_EXPERT], sv[D::N_EXPERT];
    __shared__ int   si[D::N_EXPERT], sel[D::N_EXPERT_USED];
    const int e = threadIdx.x;
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
    }
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
                                int n_used, int n, float * __restrict__ weighted, float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const float v = y[(size_t) e * n + i] * w[e];
        weighted[(size_t) e * n + i] = v;
        a = (e == 0) ? v : a + v;
    }
    out[i] = a;
}

// ------------------------------------------------------------- the host side --
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
    }
    ~Ds4GpuOps() override {
        dev_ensure(g_.dev_);
        HIP_IGNORE(hipFree(po_)); HIP_IGNORE(hipFree(pm_)); HIP_IGNORE(hipFree(pl_));
        for (void * h : host_) HIP_IGNORE(hipHostFree(h));
    }

    void gemv(const Mat & W, const float * x, float * y) override {
        switch (W.type) {
            case FK_Q_Q4_K:    return launch_gemv<FK_Q_Q4_K>(W, x, y);
            case FK_Q_Q5_K:    return launch_gemv<FK_Q_Q5_K>(W, x, y);
            case FK_Q_IQ2_XXS: return launch_gemv<FK_Q_IQ2_XXS>(W, x, y);
            case FK_Q_IQ2_S:   return launch_gemv<FK_Q_IQ2_S>(W, x, y);
            case FK_Q_IQ3_XXS: return launch_gemv<FK_Q_IQ3_XXS>(W, x, y);
            case FK_Q_MXFP4:   return launch_gemv<FK_Q_MXFP4>(W, x, y);
            default:           return g_.gemv(W, x, y);   // Q8_0, Q6_K, BF16, F32: the tuned L0 kernel
        }
    }

    void hc_split(const float * m, const float * s, const float * b, float * pre, float * post,
                  float * comb, float eps, int iters) override {
        g_.mark(PC_ELEM); g_.last_op_ = "ds4_hc_split";
        k_ds4_hc_split<<<1, WAVE, 0, g_.stream_>>>(m, s, b, pre, post, comb, eps, iters);
    }
    void hc_head_pre(const float * m, const float * s, const float * b, float * pre, float eps) override {
        g_.mark(PC_ELEM); g_.last_op_ = "ds4_hc_head_pre";
        k_ds4_hc_head_pre<<<1, WAVE, 0, g_.stream_>>>(m, s, b, pre, eps);
    }
    void hc_weighted_sum(const float * H, const float * w, float * out) override {
        g_.mark(PC_ELEM); g_.last_op_ = "ds4_hc_wsum";
        k_ds4_hc_wsum<<<ceil_div(D::N_EMBD, BLK), BLK, 0, g_.stream_>>>(H, w, out);
    }
    void hc_post(const float * x, const float * H, const float * post, const float * comb,
                 float * Ho) override {
        g_.mark(PC_ELEM); g_.last_op_ = "ds4_hc_post";
        k_ds4_hc_post<<<ceil_div(D::HC_DIM, BLK), BLK, 0, g_.stream_>>>(x, H, post, comb, Ho);
    }

    void rope_tail(float * x, int n_rows, int row_len, int pos, const D::RopeParams & rp,
                   bool inverse) override {
        // the host half of ggml's rope cache, with the CPU arm's own libm
        const float theta_scale = powf(rp.freq_base, -2.0f / D::N_ROT);
        auto corr = [&](float beta) {
            return D::N_ROT * logf(rp.n_ctx_orig / (beta * 2 * (float) M_PI)) / (2 * logf(rp.freq_base));
        };
        const float c0 = std::max(0.0f, floorf(corr(rp.beta_fast)));
        const float c1 = std::min((float) (D::N_ROT - 1), ceilf(corr(rp.beta_slow)));
        float ms = rp.attn_factor;
        if (rp.ext_factor != 0.0f) ms *= 1.0f + 0.1f * logf(1.0f / rp.freq_scale);
        g_.mark(PC_QSA); g_.last_op_ = "ds4_rope_tail";
        k_ds4_rope_tail<<<n_rows, D::N_ROT / 2, 0, g_.stream_>>>(x, row_len, (float) pos, theta_scale,
            rp.freq_scale, rp.ext_factor, ms, c0, c1, inverse ? -1.0f : 1.0f);
    }
    void fwht(float * x, int n_rows, int n) override {
        if (n != D::IDX_DIM) { std::fprintf(stderr, "ds4 fwht: n=%d, the kernel is 128-wide\n", n); std::abort(); }
        g_.mark(PC_QSA); g_.last_op_ = "ds4_fwht";
        k_ds4_fwht<<<n_rows, D::IDX_DIM, 0, g_.stream_>>>(x, 1.0f / sqrtf((float) n));
    }
    void to_f16(const float * x, uint16_t * y, int n) override {
        g_.mark(PC_QSA); g_.last_op_ = "ds4_to_f16";
        k_ds4_to_f16<<<ceil_div(n, BLK), BLK, 0, g_.stream_>>>(x, (unsigned short *) y, n);
    }

    void comp_pool(const float * rk, const float * rs, int ring, int R, int d_out, int blk,
                   float * out) override {
        g_.mark(PC_QSA); g_.last_op_ = "ds4_comp_pool";
        k_ds4_comp_pool<<<ceil_div(d_out, 128), 128, 0, g_.stream_>>>(rk, rs, ring, R, d_out, blk, 0, out);
    }
    void comp_pool_overlap(const float * rk, const float * rs, int ring, int R, int d_out, int blk,
                           float * out) override {
        g_.mark(PC_QSA); g_.last_op_ = "ds4_comp_pool_overlap";
        k_ds4_comp_pool<<<ceil_div(d_out, 128), 128, 0, g_.stream_>>>(rk, rs, ring, R, d_out, blk, 1, out);
    }

    void lid_scores(const float * q, const float * w, const uint16_t * keys, int n,
                    float * scores) override {
        g_.mark(PC_QSA); g_.last_op_ = "ds4_lid_scores";
        k_ds4_lid_scores<<<ceil_div(n, DS4_LID_KEYS), DS4_LID_THREADS, 0, g_.stream_>>>(
            q, w, (const unsigned short *) keys, n, scores);
    }
    void topk(const float * s, int n, int k, int * out) override {
        g_.mark(PC_QSA); g_.last_op_ = "ds4_topk";
        k_ds4_topk<<<1, DS4_TOPK_THREADS, 0, g_.stream_>>>(s, n, k, out);
    }

    void attn(const float * q, const uint16_t * raw, int raw0, int n_raw, const uint16_t * comp,
              const int * ids, int n_comp, const float * sinks, float scale, float * out) override {
        const int n_chunks = ceil_div(n_raw + n_comp, DS4_ATT_CHUNK);
        ensure_attn(n_chunks);
        g_.mark(PC_QSA_ROWS); g_.last_op_ = "ds4_attn_part";
        k_ds4_attn_part<<<dim3((unsigned) n_chunks, DS4_ATT_HGROUPS), 256, 0, g_.stream_>>>(
            q, (const unsigned short *) raw, raw0, n_raw, (const unsigned short *) comp, ids, n_comp,
            scale, po_, pm_, pl_);
        g_.mark(PC_QSA_ROWS); g_.last_op_ = "ds4_attn_combine";
        k_ds4_attn_combine<<<D::N_HEAD, D::HEAD_DIM, 0, g_.stream_>>>(po_, pm_, pl_, n_chunks, sinks, out);
    }

    void router(const float * logits, const float * bias, const int32_t * hash, float * probs,
                float * pb, int * ids, float * wr, float * wn, float * ws) override {
        g_.mark(PC_ROUTER); g_.last_op_ = "ds4_router";
        k_ds4_router<<<1, D::N_EXPERT, 0, g_.stream_>>>(logits, bias, hash, probs, pb, ids, wr, wn, ws);
    }
    void moe_gate_up(const D::ExpertTable & t, const int * ids, const float * x, float * yg,
                     float * yu) override {
        launch_moe(t.type_gu, t.up,   ids, t.row_gu, t.rows_gu, t.K_gu, x, 0, yu, PC_EXP_GATE_UP);
        launch_moe(t.type_gu, t.gate, ids, t.row_gu, t.rows_gu, t.K_gu, x, 0, yg, PC_EXP_GATE_UP);
    }
    void moe_down(const D::ExpertTable & t, const int * ids, const float * h, float * y) override {
        launch_moe(t.type_d, t.down, ids, t.row_d, t.rows_d, t.K_d, h, 1, y, PC_EXP_DOWN);
    }
    void swiglu_clamp(const float * g, const float * u, float * h, int n, float limit) override {
        g_.mark(PC_ELEM); g_.last_op_ = "ds4_swiglu_clamp";
        k_ds4_swiglu_clamp<<<ceil_div(n, BLK), BLK, 0, g_.stream_>>>(g, u, h, n, limit);
    }
    void moe_accum(const float * y, const float * w, int n_used, int n, float * weighted,
                   float * out) override {
        g_.mark(PC_EXP_COMBINE); g_.last_op_ = "ds4_moe_accum";
        k_ds4_moe_accum<<<ceil_div(n, BLK), BLK, 0, g_.stream_>>>(y, w, n_used, n, weighted, out);
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
    template <int TY>
    void launch_gemv(const Mat & W, const float * x, float * y) {
        g_.mark(PC_TRUNK, -1, W.row_bytes * (size_t) W.rows);
        g_.last_op_ = "ds4_gemv";
        k_ds4_gemv<TY><<<ceil_div((int) W.rows, DS4_GEMV_ROWS), DS4_GEMV_THREADS, 0, g_.stream_>>>(
            (const unsigned char *) W.base, W.row_bytes, (int) W.rows, (int) W.K, x, y);
    }
    void launch_moe(int type, const void * const * tab, const int * ids, size_t rb, int rows, int K,
                    const float * x, int x_per_slot, float * y, int cls) {
        g_.mark(cls); g_.last_op_ = "ds4_moe";
        const dim3 grid((unsigned) ceil_div(rows, DS4_GEMV_ROWS), (unsigned) D::N_EXPERT_USED);
        switch (type) {
            case FK_Q_IQ2_XXS: k_ds4_moe<FK_Q_IQ2_XXS><<<grid, DS4_GEMV_THREADS, 0, g_.stream_>>>(tab, ids, rb, rows, K, x, x_per_slot, y); break;
            case FK_Q_IQ2_S:   k_ds4_moe<FK_Q_IQ2_S>  <<<grid, DS4_GEMV_THREADS, 0, g_.stream_>>>(tab, ids, rb, rows, K, x, x_per_slot, y); break;
            case FK_Q_IQ3_XXS: k_ds4_moe<FK_Q_IQ3_XXS><<<grid, DS4_GEMV_THREADS, 0, g_.stream_>>>(tab, ids, rb, rows, K, x, x_per_slot, y); break;
            case FK_Q_MXFP4:   k_ds4_moe<FK_Q_MXFP4>  <<<grid, DS4_GEMV_THREADS, 0, g_.stream_>>>(tab, ids, rb, rows, K, x, x_per_slot, y); break;
            default:
                std::fprintf(stderr, "ds4 moe: expert format %d has no kernel\n", type);
                std::abort();
        }
    }
    // Grown, never shrunk: a hipFree synchronises the device (CLAUDE.md). The
    // chunk count only grows with the position, and ctx fixes its maximum.
    void ensure_attn(int n_chunks) {
        if (n_chunks <= cap_chunks_) return;
        dev_ensure(g_.dev_);
        int want = cap_chunks_ ? cap_chunks_ : 8;
        while (want < n_chunks) want *= 2;
        HIP_IGNORE(hipFree(po_)); HIP_IGNORE(hipFree(pm_)); HIP_IGNORE(hipFree(pl_));
        HIP_CHECK(hipMalloc(&po_, (size_t) want * D::N_HEAD * D::HEAD_DIM * sizeof(float)));
        HIP_CHECK(hipMalloc(&pm_, (size_t) want * D::N_HEAD * sizeof(float)));
        HIP_CHECK(hipMalloc(&pl_, (size_t) want * D::N_HEAD * sizeof(float)));
        cap_chunks_ = want;
    }

    GpuBackend & g_;
    float * po_ = nullptr, * pm_ = nullptr, * pl_ = nullptr;
    int cap_chunks_ = 0;
    std::vector<void *> host_;
};
