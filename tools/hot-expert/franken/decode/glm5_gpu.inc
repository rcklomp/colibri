// tools/hot-expert/franken/decode/glm5_gpu.inc
//
// Glm5GpuOps: the GLM-5.3-Flash operations of glm5_ops.h as HIP kernels (L5
// GLM step 4). #included by decode_gpu.hip at the end of its anonymous
// namespace, after GpuBackend and ds4_gpu.inc, so every main-stream launch
// goes to the backend's own stream through its own mark() (device selection,
// launch count, --profile, --sync-debug). Nothing above this include reads
// anything in it.
//
// First cut, correctness first: every kernel a transcription of the CPU op it
// replaces (glm5_cpu.cpp); where the order of a sum differs it says so.
// HIP multi-GPU rule (CLAUDE.md): every entry point that allocates, creates
// an event or stream, or launches selects its card first -- mark() for the
// main stream, dev_ensure() for the side, helper and copy streams.
//
// No --hip-graph for GLM yet: every op takes the position as an argument.

namespace G5 = ::fk::glm5;

// ------------------------------------------------------------------ KDA ----
// One thread a channel of q|k|v: the 4-tap causal conv over [3 history rows |
// the new row] (float, tap order, as ops.cpp ssm_conv), SiLU, and the window
// advanced in place (every channel is independent, T == 1).
__global__ void k_glm5_kda_conv(const float * __restrict__ qkv, float * __restrict__ st,
                                const float * __restrict__ wq, const float * __restrict__ wk,
                                const float * __restrict__ wv, float * __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= G5::KDA_QKV) return;
    const float * w = c < G5::KDA_INNER ? wq + (size_t) c * G5::KDA_CONV
                    : c < 2 * G5::KDA_INNER ? wk + (size_t) (c - G5::KDA_INNER) * G5::KDA_CONV
                    : wv + (size_t) (c - 2 * G5::KDA_INNER) * G5::KDA_CONV;
    const float s0 = st[c], s1 = st[G5::KDA_QKV + c], s2 = st[2 * G5::KDA_QKV + c], x = qkv[c];
    float sum = 0.0f;
    sum += s0 * w[0];
    sum += s1 * w[1];
    sum += s2 * w[2];
    sum += x * w[3];
    out[c] = sum / (1.0f + expf(-sum));
    st[c] = s1; st[G5::KDA_QKV + c] = s2; st[2 * G5::KDA_QKV + c] = x;
}

// g = lb * sigmoid(-((fb + dt) * a[h])), the ggml op order
__global__ void k_glm5_kda_gate(const float * __restrict__ fb, const float * __restrict__ dt,
                                const float * __restrict__ a, float lb, float * __restrict__ g) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= G5::KDA_INNER) return;
    float x = fb[c] + dt[c];
    x = x * a[c / G5::KDA_DIM];
    x = x * -1.0f;
    x = 1.0f / (1.0f + expf(-x));
    g[c] = x * lb;
}

// The KDA recurrence, one workgroup a head (64), 256 threads: thread t owns
// half of state row j = t/2 (64 keys, in registers), the partner the other
// half; the two partial dots meet by a lane exchange. Per row: decay by
// exp(g[i]) per KEY channel, the prediction, the delta, the rank-1 update, the
// output -- ggml's order. Sums are float over two halves (the CPU arm: double
// over 128), a summation-order difference.
constexpr int GLM5_KDA_THREADS = 256;
__global__ __launch_bounds__(GLM5_KDA_THREADS)
void k_glm5_kda_step(float * __restrict__ state, const float * __restrict__ q, const float * __restrict__ k,
                     const float * __restrict__ v, const float * __restrict__ g,
                     const float * __restrict__ beta, float * __restrict__ out) {
    constexpr int D = G5::KDA_DIM, HALF = G5::KDA_DIM / 2;
    __shared__ float sk[D], sq[D], sdec[D];
    const int h = blockIdx.x, t = threadIdx.x;
    if (t < D) {
        sk[t] = k[h * D + t];
        sq[t] = q[h * D + t];
        sdec[t] = expf(g[h * D + t]);
    }
    __syncthreads();
    const int j = t >> 1, part = t & 1, i0 = part * HALF;
    float * row = state + ((size_t) h * D + j) * D + i0;
    float r[HALF];
#pragma unroll
    for (int i = 0; i < HALF; i += 4) {
        const float4 u = *(const float4 *) (row + i);
        r[i] = u.x; r[i + 1] = u.y; r[i + 2] = u.z; r[i + 3] = u.w;
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < HALF; ++i) { r[i] = r[i] * sdec[i0 + i]; s = fmaf(r[i], sk[i0 + i], s); }
    s += __shfl_xor(s, 1, WAVE);
    const float d = (v[h * D + j] - s) * beta[h];
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < HALF; ++i) { r[i] = fmaf(sk[i0 + i], d, r[i]); o = fmaf(r[i], sq[i0 + i], o); }
    o += __shfl_xor(o, 1, WAVE);
    if (part == 0) out[h * D + j] = o * (1.0f / sqrtf((float) D));
#pragma unroll
    for (int i = 0; i < HALF; i += 4) *(float4 *) (row + i) = make_float4(r[i], r[i + 1], r[i + 2], r[i + 3]);
}

// ------------------------------------------------------------- the indexer --
// ggml_norm then * w + b over one row of ne0 <= 1024 (one workgroup a row)
__global__ __launch_bounds__(128)
void k_glm5_layer_norm(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ b,
                       float * __restrict__ y, int ne0, float eps) {
    __shared__ float red[128 / WAVE];
    x += (size_t) blockIdx.x * ne0; y += (size_t) blockIdx.x * ne0;
    const int t = threadIdx.x, lane = t & (WAVE - 1), wv = t / WAVE;
    float a = 0.0f;
    for (int i = t; i < ne0; i += blockDim.x) a += x[i];
    a = wave_sum(a);
    if (lane == 0) red[wv] = a;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < (int) blockDim.x / WAVE; ++i) sum += red[i];
    const float mean = sum / ne0;
    __syncthreads();
    float q = 0.0f;
    for (int i = t; i < ne0; i += blockDim.x) { const float d = x[i] - mean; q += d * d; }
    q = wave_sum(q);
    if (lane == 0) red[wv] = q;
    __syncthreads();
    float var = 0.0f;
    for (int i = 0; i < (int) blockDim.x / WAVE; ++i) var += red[i];
    const float scale = 1.0f / sqrtf(var / ne0 + eps);
    for (int i = t; i < ne0; i += blockDim.x) {
        float o = (x[i] - mean) * scale;
        if (w) o = o * w[i];
        if (b) o = o + b[i];
        y[i] = o;
    }
}

__global__ void k_glm5_store_f16(const float * __restrict__ src, unsigned short * __restrict__ dst,
                                 int dst_stride, int dst_off, int width, int ring_rows, int pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int row = ring_rows > 0 ? pos % ring_rows : pos;
    dst[(size_t) row * dst_stride + dst_off + i] = __half_as_ushort(__float2half_rn(src[i]));
}

// The pool of four cells a completed position closes: per dim, softmax over
// the four slots of (gate + ape), then the probability-weighted key (float;
// the CPU arm sums in double).
__global__ __launch_bounds__(G5::IDX_DIM)
void k_glm5_idx_pool(const unsigned short * __restrict__ kg, int kg_rows, const float * __restrict__ ape,
                     unsigned short * __restrict__ pooled, int pos) {
    if ((pos + 1) % G5::KPOOL != 0) return;
    const int d = threadIdx.x, b = pos / G5::KPOOL;
    float key[G5::KPOOL], sc[G5::KPOOL], mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < G5::KPOOL; ++j) {
        const unsigned short * cell = kg + (size_t) ((b * G5::KPOOL + j) % kg_rows) * (2 * G5::IDX_DIM);
        key[j] = ds4_h2f(cell[d]);
        sc[j]  = ds4_h2f(cell[G5::IDX_DIM + d]) + ape[j * G5::IDX_DIM + d];
        mx = fmaxf(mx, sc[j]);
    }
    float e[G5::KPOOL], sum = 0.0f;
#pragma unroll
    for (int j = 0; j < G5::KPOOL; ++j) { e[j] = expf(sc[j] - mx); sum += e[j]; }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < G5::KPOOL; ++j) acc += key[j] * (e[j] * inv);
    pooled[(size_t) b * G5::IDX_DIM + d] = __half_as_ushort(__float2half_rn(acc));
}

// The pool scan: DeepSeek's k_ds4_lid_scores shape (four lanes a key, four
// keys a group, q in LDS, DPP exchange, a persistent grid) at 32 heads and
// the (p + 1) / 4 pooled f16 keys, no FWHT. Heads summed in ascending order,
// fused `sc = fma(relu, w_h, sc)` (the CPU arm: double over the heads).
constexpr int GLM5_IDX_THREADS = 256;
constexpr int GLM5_IDX_LANES   = 4;
constexpr int GLM5_IDX_GROUPS  = GLM5_IDX_THREADS / GLM5_IDX_LANES;
constexpr int GLM5_IDX_KPT     = 4;
constexpr int GLM5_IDX_TILE    = GLM5_IDX_GROUPS * GLM5_IDX_KPT;       // 256 keys a tile
constexpr int GLM5_IDX_MAX_WGS = 384;
__global__ __launch_bounds__(GLM5_IDX_THREADS)
void k_glm5_idx_scores(const float * __restrict__ q, const float * __restrict__ w,
                       const unsigned short * __restrict__ keys, int pos, float * __restrict__ scores) {
    const int n = (pos + 1) / G5::KPOOL;
    if ((int) blockIdx.x * GLM5_IDX_TILE >= n) return;
    __shared__ float sq[G5::IDX_N_HEAD * G5::IDX_DIM];
    __shared__ float sw[G5::IDX_N_HEAD];
    for (int i = threadIdx.x; i < G5::IDX_N_HEAD * G5::IDX_DIM / 4; i += GLM5_IDX_THREADS)
        ((float4 *) sq)[i] = ((const float4 *) q)[i];
    if (threadIdx.x < G5::IDX_N_HEAD) sw[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    constexpr int PD = G5::IDX_DIM / GLM5_IDX_LANES;                   // 32 dims a lane
    const int part = threadIdx.x & (GLM5_IDX_LANES - 1);
    const int g = threadIdx.x / GLM5_IDX_LANES;
    for (int tile = blockIdx.x; tile * GLM5_IDX_TILE < n; tile += gridDim.x) {
        unsigned kp[GLM5_IDX_KPT][PD / 2];
        int b[GLM5_IDX_KPT];
#pragma unroll
        for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) {
            b[kk] = tile * GLM5_IDX_TILE + kk * GLM5_IDX_GROUPS + g;
            const uint4 * k4 = (const uint4 *) (keys + (size_t) b[kk] * G5::IDX_DIM + part * PD);
#pragma unroll
            for (int j = 0; j < PD / 8; ++j) {
                const uint4 u = (b[kk] < n) ? k4[j] : make_uint4(0u, 0u, 0u, 0u);
                kp[kk][4 * j] = u.x; kp[kk][4 * j + 1] = u.y; kp[kk][4 * j + 2] = u.z; kp[kk][4 * j + 3] = u.w;
            }
        }
        float sc[GLM5_IDX_KPT];
#pragma unroll
        for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) sc[kk] = 0.0f;
        for (int hh = 0; hh < G5::IDX_N_HEAD; ++hh) {
            const float * qh = sq + hh * G5::IDX_DIM + part * PD;
            float a[GLM5_IDX_KPT];
#pragma unroll
            for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) a[kk] = 0.0f;
#pragma unroll
            for (int i = 0; i < PD / 2; ++i) {
                const float q0 = qh[2 * i], q1 = qh[2 * i + 1];
#pragma unroll
                for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) {
                    a[kk] = fmaf(q0, ds4_h2f(kp[kk][i] & 0xFFFFu), a[kk]);
                    a[kk] = fmaf(q1, ds4_h2f(kp[kk][i] >> 16), a[kk]);
                }
            }
            const float wh = sw[hh];
#pragma unroll
            for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) {
                a[kk] = a[kk] + ds4_quad_xor1(a[kk]);
                a[kk] = a[kk] + ds4_quad_xor2(a[kk]);
                sc[kk] = fmaf(fmaxf(a[kk], 0.0f), wh, sc[kk]);
            }
        }
        if (part == 0) {
#pragma unroll
            for (int kk = 0; kk < GLM5_IDX_KPT; ++kk) if (b[kk] < n) scores[b[kk]] = sc[kk];
        }
    }
}

// ---------------------------------------------------------- absorbed MLA ----
// Every head's slice of a [K, rows, n_heads] Q8_0 tensor in one launch: one
// wave a row (8 a workgroup), row r of head r / rows_per_head against that
// head's x. The lane decoder is decode_quant.h's Q8_0 one (bit-exact to ggml,
// ds4_quant_emul --glm-model); the wave sum is float.
constexpr int GLM5_HG_THREADS = 256;
__global__ __launch_bounds__(GLM5_HG_THREADS)
void k_glm5_head_gemv_q8(const unsigned char * __restrict__ W, size_t row_bytes, int rows_total,
                         int rows_per_head, int K, const float * __restrict__ x, int x_stride,
                         float * __restrict__ y) {
    const int lane = threadIdx.x & (WAVE - 1);
    const int r = blockIdx.x * (GLM5_HG_THREADS / WAVE) + threadIdx.x / WAVE;
    if (r >= rows_total) return;
    const unsigned char * row = W + (size_t) r * row_bytes;
    const float * xh = x + (size_t) (r / rows_per_head) * x_stride;
    float a = 0.0f;
    for (int b = 0; b < K / FK_Q8_0_BLOCK_WEIGHTS; ++b)
        a += fk_q8_0_lane_dot(row + (size_t) b * FK_Q8_0_BLOCK_BYTES, xh + b * FK_Q8_0_BLOCK_WEIGHTS, lane);
    a = wave_sum(a);
    if (lane == 0) y[r] = a;
}

// The attended cells of the row at p (glm5_ops.h mla_attn): every cell <= p
// while the visible pools are <= 512 (or no selection), else the 2 048 cells
// of the selected pools (ascending) then the tail.
__device__ __forceinline__ int glm5_n_keys(int p, const int * sel) {
    const int n_vis = (p + 1) / G5::KPOOL;
    return (!sel || n_vis <= G5::IDX_TOP_POOLS) ? p + 1 : G5::IDX_TOP_K + (p + 1 - n_vis * G5::KPOOL);
}
__device__ __forceinline__ int glm5_cell(int p, const int * sel, int j) {
    const int n_vis = (p + 1) / G5::KPOOL;
    if (!sel || n_vis <= G5::IDX_TOP_POOLS) return j;
    if (j < G5::IDX_TOP_K) return sel[j / G5::KPOOL] * G5::KPOOL + (j % G5::KPOOL);
    return n_vis * G5::KPOOL + (j - G5::IDX_TOP_K);
}

// DeepSeek's flash-style split (k_ds4_attn_part / _combine): 32 keys a
// workgroup into LDS (rows padded to 257 u32), 16 of the 64 heads (grid.y 4
// head groups), lane j scores key j, V = K reused from LDS; the combine folds
// the chunks. GLM: the keys are the latent rows of the attended cells, 512
// wide exactly as DeepSeek's head, and there is no sink.
constexpr int GLM5_ATT_CHUNK   = 32;
constexpr int GLM5_ATT_HGROUPS = 4;
constexpr int GLM5_ATT_STRIDE  = G5::KV_LORA / 2 + 1;
constexpr int GLM5_ATT_MAX_KEYS = G5::IDX_N_SELECT;                  // 2 051: dense below, sparse above
constexpr int GLM5_ATT_CHUNKS  = (GLM5_ATT_MAX_KEYS + GLM5_ATT_CHUNK - 1) / GLM5_ATT_CHUNK;
__global__ __launch_bounds__(256)
void k_glm5_attn_part(const float * __restrict__ q, const unsigned short * __restrict__ kv, int pos,
                      const int * __restrict__ sel, float scale, float * __restrict__ po,
                      float * __restrict__ pm, float * __restrict__ pl) {
    __shared__ unsigned kl[GLM5_ATT_CHUNK * GLM5_ATT_STRIDE];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = threadIdx.x;
    const int n_keys = glm5_n_keys(pos, sel);
    const int j0 = chunk * GLM5_ATT_CHUNK;
    if (j0 >= n_keys) return;
    const int nk = min(GLM5_ATT_CHUNK, n_keys - j0);
    for (int idx = t; idx < GLM5_ATT_CHUNK * (G5::KV_LORA / 2); idx += 256) {
        const int r = idx / (G5::KV_LORA / 2), c = idx % (G5::KV_LORA / 2);
        if (r < nk) {
            const unsigned short * row = kv + (size_t) glm5_cell(pos, sel, j0 + r) * G5::KV_LORA;
            kl[r * GLM5_ATT_STRIDE + c] = ((const unsigned *) row)[c];
        }
    }
    __syncthreads();
    const int wave = __builtin_amdgcn_readfirstlane(t / WAVE), lane = t & (WAVE - 1);
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * (G5::N_HEAD / GLM5_ATT_HGROUPS) + wave * 2 + hh;
        const float * qh = q + (size_t) h * G5::KV_LORA;
        float sc = -INFINITY;
        if (lane < nk) {
            float a = 0.0f;
            for (int c = 0; c < G5::KV_LORA / 2; ++c) {
                const unsigned u = kl[lane * GLM5_ATT_STRIDE + c];
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
                const unsigned u = kl[j * GLM5_ATT_STRIDE + lane + 32 * tt];
                acc[2 * tt]     = fmaf(pj, ds4_h2f(u & 0xFFFFu), acc[2 * tt]);
                acc[2 * tt + 1] = fmaf(pj, ds4_h2f(u >> 16),     acc[2 * tt + 1]);
            }
        }
        float * o = po + ((size_t) chunk * G5::N_HEAD + h) * G5::KV_LORA;
#pragma unroll
        for (int tt = 0; tt < 8; ++tt) {
            o[2 * (lane + 32 * tt)]     = acc[2 * tt];
            o[2 * (lane + 32 * tt) + 1] = acc[2 * tt + 1];
        }
        if (lane == 0) { pm[chunk * G5::N_HEAD + h] = m; pl[chunk * G5::N_HEAD + h] = l; }
    }
}

__global__ __launch_bounds__(G5::KV_LORA)
void k_glm5_attn_combine(const float * __restrict__ po, const float * __restrict__ pm,
                         const float * __restrict__ pl, int pos, const int * __restrict__ sel,
                         float * __restrict__ out) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int n_chunks = (glm5_n_keys(pos, sel) + GLM5_ATT_CHUNK - 1) / GLM5_ATT_CHUNK;
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, pm[c * G5::N_HEAD + h]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float f = expf(pm[c * G5::N_HEAD + h] - M);
        L += pl[c * G5::N_HEAD + h] * f;
        o += po[((size_t) c * G5::N_HEAD + h) * G5::KV_LORA + d] * f;
    }
    out[(size_t) h * G5::KV_LORA + d] = o / L;
}

// ------------------------------------------------------------------ MoE ----
// sigmoid, + exp_probs_b for the SELECTION, top-8 by eight block argmax
// passes (value descending, lower index first on a tie: the CPU arm's
// partial_sort rule), weights = probs[ids] / max(sum, 6.1e-5) x 2.5, and the
// --adapt route counters (one thread: nothing above reads them).
constexpr int GLM5_ROUTER_THREADS = 512;
__global__ __launch_bounds__(GLM5_ROUTER_THREADS)
void k_glm5_router(const float * __restrict__ logits, const float * __restrict__ bias,
                   float * __restrict__ probs, float * __restrict__ pb, int * __restrict__ ids,
                   float * __restrict__ wr, float * __restrict__ wn, float * __restrict__ ws,
                   const int * __restrict__ miss, unsigned * __restrict__ st) {
    __shared__ float sp[G5::N_EXPERT], sv[GLM5_ROUTER_THREADS];
    __shared__ int   si[GLM5_ROUTER_THREADS], sel[G5::N_EXPERT_USED];
    const int e = threadIdx.x;
    const bool valid = e < G5::N_EXPERT;
    float v = -INFINITY;
    if (valid) {
        const float p = 1.0f / (1.0f + expf(-logits[e]));
        sp[e] = p; probs[e] = p;
        v = p + bias[e];
        pb[e] = v;
    }
    bool taken = !valid;
    for (int k = 0; k < G5::N_EXPERT_USED; ++k) {
        sv[e] = taken ? -INFINITY : v; si[e] = e;
        __syncthreads();
        for (int s2 = GLM5_ROUTER_THREADS / 2; s2 > 0; s2 >>= 1) {
            if (e < s2) {
                const float a = sv[e], b = sv[e + s2];
                if (b > a || (b == a && si[e + s2] < si[e])) { sv[e] = b; si[e] = si[e + s2]; }
            }
            __syncthreads();
        }
        if (e == 0) sel[k] = si[0];
        __syncthreads();
        if (e == sel[k]) taken = true;
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < G5::N_EXPERT_USED; ++k) sum += sp[sel[k]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int k = 0; k < G5::N_EXPERT_USED; ++k) {
            const float w = sp[sel[k]];
            ids[k] = sel[k]; wr[k] = w; wn[k] = w / sum; ws[k] = (w / sum) * G5::EXPERT_WEIGHTS_SCALE;
        }
        if (st)
            for (int k = 0; k < G5::N_EXPERT_USED; ++k) {
                const int x = sel[k];
                atomicAdd(&st[x], 1u);
                if (miss[x] != 0) atomicAdd(&st[G5::G_ADAPT_MISS], 1u);
                atomicAdd(&st[G5::G_ADAPT_PICKS], 1u);
            }
    }
}

// THE PLAN (the three-link miss path, glm5_ops.h LinkPlan): the owner's
// packet -- x, the 8 ids, per slot missed or not (the table as it stands) and
// the card that computes it. `pat` is the 11-long pattern already mapped to
// card indices on the host.
struct Glm5Pk {
    float x[G5::N_EMBD];
    int ids[G5::N_EXPERT_USED], mk[G5::N_EXPERT_USED], assign[G5::N_EXPERT_USED], pad[G5::N_EXPERT_USED];
};
struct Glm5Pat { int card[11]; };
__global__ __launch_bounds__(256)
void k_glm5_plan(const int * __restrict__ ids, const int * __restrict__ miss, const float * __restrict__ x,
                 Glm5Pk * __restrict__ pk, int me, int n_links, int salt, const Glm5Pat pat) {
    for (int i = threadIdx.x; i < G5::N_EMBD; i += blockDim.x) pk->x[i] = x[i];
    if (threadIdx.x == 0) {
        int j = 0;
        for (int k = 0; k < G5::N_EXPERT_USED; ++k) {
            const int e = ids[k];
            const int m = miss[e] != 0;
            pk->ids[k] = e; pk->mk[k] = m;
            pk->assign[k] = (m && n_links > 1) ? pat.card[(salt + j++) % 11] : me;
        }
    }
}

// The staging copy, grid (wgs, 8): slot k's expert, if it is this card's to
// compute: a resident one (owner only) is POINTED at, a missed one is copied
// gate | up | down into dst + k * slot_bytes with 16-byte loads over THIS
// card's host link (DeepSeek's k_ds4_stage, keyed by the plan).
constexpr int GLM5_STAGE_THREADS = 256;
__global__ __launch_bounds__(GLM5_STAGE_THREADS)
void k_glm5_stage(const void * const * __restrict__ tg, const void * const * __restrict__ tu,
                  const void * const * __restrict__ td, const Glm5Pk * __restrict__ pk, int me,
                  unsigned long long sg, unsigned long long su, unsigned long long sd,
                  unsigned char * __restrict__ dst, unsigned long long slot_bytes, const void ** __restrict__ sp) {
    const int k = blockIdx.y;
    if (pk->assign[k] != me) return;
    const int e = pk->ids[k];
    const bool missed = pk->mk[k] != 0;
    unsigned char * d = dst + (size_t) k * slot_bytes;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sp[k * 3 + 0] = missed ? (const void *) d             : tg[e];
        sp[k * 3 + 1] = missed ? (const void *) (d + sg)      : tu[e];
        sp[k * 3 + 2] = missed ? (const void *) (d + sg + su) : td[e];
    }
    if (!missed) return;
    const unsigned long long total = sg + su + sd;
    const unsigned long long step = 16ull * GLM5_STAGE_THREADS * gridDim.x;
    for (unsigned long long o = 16ull * ((unsigned long long) blockIdx.x * GLM5_STAGE_THREADS + threadIdx.x);
         o < total; o += step) {
        const unsigned char * s = o < sg      ? (const unsigned char *) tg[e] + o
                                : o < sg + su ? (const unsigned char *) tu[e] + (o - sg)
                                              : (const unsigned char *) td[e] + (o - sg - su);
        *(uint4 *) (d + o) = *(const uint4 *) s;
    }
}

// The routed-expert GEMV, grid (rows / 8, 8 slots): one wave a row, the row
// staged into LDS with 16-byte loads (bit-identical to reading it in place,
// DEEPSEEK4.md section 10), then the lane decoder (decode_quant.h /
// m1_native_decode.h, bit-exact to ggml). Slot k runs here only if the plan
// gave it to this card; pass 1 = its resident experts (from the table),
// pass 2 = its missed experts (from the staged pointers). The same compiled
// body on the same bytes on any card: where a slot runs changes no bit.
template <int TY> constexpr int glm5_row_max() {
    return TY == FK_Q_IQ4_XS ? 2176 : TY == FK_Q_IQ3_S ? 1760 : 1680;   // K <= 4096 (Q6_K: K 2048)
}
template <int TY>
__device__ __forceinline__ float glm5_row_lane(const unsigned char * __restrict__ row,
                                               const float * __restrict__ x, int K, int lane) {
    float a = 0.0f;
    const int nb = K / 256;
    for (int b = 0; b < nb; ++b) {
        const float * xb = x + b * 256 + 8 * lane;
        if constexpr (TY == FK_Q_IQ3_S) {
            const int l = lane & 3;
            a += m1n_iq3s_block_dot(row + (size_t) b * IQ3S_BLOCK_BYTES, c_iq3s_grid, xb, lane, lane >> 2,
                                    8 - 2 * l, 7 - 2 * l);
        } else if constexpr (TY == FK_Q_IQ4_XS) {
            a += fk_iq4xs_block_dot(row + (size_t) b * FK_IQ4XS_BLOCK_BYTES, c_kvals_iq4nl, xb, lane);
        } else {
            a += fk_q6k_block_dot(row + (size_t) b * FK_Q6K_BLOCK_BYTES, xb, lane);
        }
    }
    return a;
}
template <int TY>
__global__ __launch_bounds__(DS4_GEMV_THREADS)
void k_glm5_moe(const void * const * __restrict__ tab, const void * const * __restrict__ sp,
                const Glm5Pk * __restrict__ pk, int me, int pass, int which, size_t row_bytes, int rows, int K,
                const float * __restrict__ x, int x_per_slot, float * __restrict__ y, int staged) {
    __shared__ uint4 lds[DS4_GEMV_ROWS][glm5_row_max<TY>() / 16];
    const int k = blockIdx.y;
    const int lane = threadIdx.x & (WAVE - 1);
    const int wv = threadIdx.x / WAVE;
    const int r = blockIdx.x * DS4_GEMV_ROWS + wv;
    if (r >= rows || pk->assign[k] != me) return;
    const int mk = pk->mk[k];
    if ((pass == 1 && mk) || (pass == 2 && !mk)) return;
    const unsigned char * W = (const unsigned char *) (pass == 1 ? tab[pk->ids[k]] : sp[k * 3 + which]);
    const unsigned char * src = ds4_stage_row(W + (size_t) r * row_bytes, row_bytes, staged, lds[wv], lane);
    const float * xs = x_per_slot ? x + (size_t) k * K : x;
    float acc = glm5_row_lane<TY>(src, xs, K, lane);
    acc = wave_sum(acc);
    if (lane == 0) y[(size_t) k * rows + r] = acc;
}

// swiglu_clamp on the slots this card computes (k_ds4_swiglu_clamp's
// expression), grid (rows / 256, 8)
__global__ void k_glm5_swiglu_rows(const float * __restrict__ yg, const float * __restrict__ yu,
                                   float * __restrict__ yh, int rows, const Glm5Pk * __restrict__ pk, int me,
                                   float limit) {
    const int k = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || pk->assign[k] != me) return;
    const size_t o = (size_t) k * rows + i;
    const float gg = fminf(yg[o], limit);
    const float uu = fminf(fmaxf(yu[o], -limit), limit);
    yh[o] = gg / (1.f + expf(-gg)) * uu;
}

// A helper card's finished rows into the OWNER's buffers (peer stores; the
// owner joins on the helper's event before it reads them). grid (rows_d/256, 8)
__global__ void k_glm5_scatter(const float * __restrict__ lg, const float * __restrict__ lu,
                               const float * __restrict__ lh, const float * __restrict__ ld,
                               float * __restrict__ og, float * __restrict__ ou, float * __restrict__ oh,
                               float * __restrict__ od, int rows_gu, int rows_d,
                               const Glm5Pk * __restrict__ pk, int me) {
    const int k = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (pk->assign[k] != me) return;
    if (i < rows_gu) {
        const size_t o = (size_t) k * rows_gu + i;
        og[o] = lg[o]; ou[o] = lu[o]; oh[o] = lh[o];
    }
    if (i < rows_d) od[(size_t) k * rows_d + i] = ld[(size_t) k * rows_d + i];
}

// build_hc_mean: ((H0 + H1) + H2) + H3, times 1/4
__global__ void k_glm5_hc_mean(const float * __restrict__ H, float * __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= G5::N_EMBD) return;
    float a = H[i];
    for (int s = 1; s < G5::HC; ++s) a = a + H[(size_t) s * G5::N_EMBD + i];
    out[i] = a * (1.0f / G5::HC);
}

// ------------------------------------------------------------- the host side --

class Glm5GpuOps : public G5::Glm5Ops {
public:
    explicit Glm5GpuOps(GpuBackend & g) : g_(g) {}
    ~Glm5GpuOps() override {
        dev_ensure(g_.dev_);
        for (void * p : {(void *) po_, (void *) pm_, (void *) pl_, (void *) own_ring_, (void *) help_ring_,
                         (void *) own_sp_, (void *) help_sp_, (void *) pk_, (void *) hpk_,
                         (void *) lg_, (void *) lu_, (void *) lh_, (void *) ld_})
            if (p) HIP_IGNORE(hipFree(p));
        for (hipEvent_t e : {ev_plan_[0], ev_plan_[1], ev_fork_, ev_own_[0], ev_own_[1], ev_help_})
            if (e) HIP_IGNORE(hipEventDestroy(e));
        if (side_) HIP_IGNORE(hipStreamDestroy(side_));
        if (help_) { HIP_IGNORE(hipStreamSynchronize(help_)); HIP_IGNORE(hipStreamDestroy(help_)); }
    }

    // ---- KDA ------------------------------------------------------------------
    void kda_conv(const float * qkv, float * st, const float * wq, const float * wk, const float * wv,
                  float * out, int T) override {
        if (T != 1) die("kda_conv: T > 1 not built");
        g_.mark(PC_GLM_KDA); g_.last_op_ = "glm5_kda_conv";
        k_glm5_kda_conv<<<ceil_div(G5::KDA_QKV, BLK), BLK, 0, g_.stream_>>>(qkv, st, wq, wk, wv, out);
    }
    void kda_gate(const float * fb, const float * dt, const float * a, float lb, float * g, int T) override {
        if (T != 1) die("kda_gate: T > 1 not built");
        g_.mark(PC_GLM_KDA); g_.last_op_ = "glm5_kda_gate";
        k_glm5_kda_gate<<<ceil_div(G5::KDA_INNER, BLK), BLK, 0, g_.stream_>>>(fb, dt, a, lb, g);
    }
    void kda_step(float * st, const float * q, const float * k, const float * v, const float * g,
                  const float * beta, float * out, int stride, int T) override {
        if (T != 1) die("kda_step: T > 1 not built");
        (void) stride;                               // T == 1: one row, q/k/v at their own addresses
        g_.mark(PC_GLM_KDA); g_.last_op_ = "glm5_kda_step";
        k_glm5_kda_step<<<G5::N_HEAD, GLM5_KDA_THREADS, 0, g_.stream_>>>(st, q, k, v, g, beta, out);
    }

    // ---- the indexer ------------------------------------------------------------
    void layer_norm(const float * x, const float * w, const float * b, float * y, int ne0, int n_rows,
                    float eps) override {
        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_layer_norm";
        k_glm5_layer_norm<<<n_rows, 128, 0, g_.stream_>>>(x, w, b, y, ne0, eps);
    }
    void store_f16(const float * src, int src_stride, uint16_t * dst, int dst_stride, int dst_off, int width,
                   int ring_rows, int pos, int T) override {
        (void) src_stride;
        if (T != 1) die("store_f16: T > 1 not built");
        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_store_f16";
        k_glm5_store_f16<<<ceil_div(width, BLK), BLK, 0, g_.stream_>>>(src, (unsigned short *) dst, dst_stride,
                                                                       dst_off, width, ring_rows, pos);
    }
    void idx_pool(const uint16_t * kg, int kg_rows, const float * ape, uint16_t * pooled, int pos, int T) override {
        if (T != 1) die("idx_pool: T > 1 not built");
        if ((pos + 1) % G5::KPOOL != 0) return;      // eager: no pool closes here
        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_idx_pool";
        k_glm5_idx_pool<<<1, G5::IDX_DIM, 0, g_.stream_>>>((const unsigned short *) kg, kg_rows, ape,
                                                           (unsigned short *) pooled, pos);
    }
    void idx_scores(const float * q, const float * w, const uint16_t * pooled, int pos, float * scores,
                    int stride, int T) override {
        (void) stride;
        if (T != 1) die("idx_scores: T > 1 not built");
        const int n = (pos + 1) / G5::KPOOL;
        if (n <= 0) return;
        const int wgs = std::min(ceil_div(n, GLM5_IDX_TILE), GLM5_IDX_MAX_WGS);
        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_idx_scores";
        k_glm5_idx_scores<<<wgs, GLM5_IDX_THREADS, 0, g_.stream_>>>(q, w, (const unsigned short *) pooled, pos, scores);
    }
    // DeepSeek's radix top-k: its visible count is (pos + 1) / 4 -- CSA_RATIO
    // is GLM's KPOOL -- and its output ascending: exactly the pool top-512.
    void idx_topk(const float * s, int stride, int pos, int * out, int T) override {
        static_assert(D::CSA_RATIO == G5::KPOOL, "k_ds4_topk counts visible pools as (p+1)/CSA_RATIO");
        g_.mark(PC_GLM_IDX); g_.last_op_ = "glm5_topk";
        k_ds4_topk<<<T, DS4_TOPK_THREADS, 0, g_.stream_>>>(s, pos, nullptr, G5::IDX_TOP_POOLS, out, stride);
    }

    // ---- absorbed MLA -------------------------------------------------------------
    void head_gemv(const Mat & W, int n_heads, const float * x, int x_stride, float * y) override {
        if (W.type != FK_Q_Q8_0) die("head_gemv: the batched per-head kernel is Q8_0 only");
        const int rows = (int) W.rows * n_heads;
        g_.mark(PC_GLM_ATTN, -1, W.row_bytes * (size_t) rows); g_.last_op_ = "glm5_head_gemv";
        k_glm5_head_gemv_q8<<<ceil_div(rows, GLM5_HG_THREADS / WAVE), GLM5_HG_THREADS, 0, g_.stream_>>>(
            (const unsigned char *) W.base, W.row_bytes, rows, (int) W.rows, (int) W.K, x, x_stride, y);
    }
    void mla_attn(const float * q, const uint16_t * kv, int pos, const int * sel, float scale, float * out,
                  int T) override {
        if (T != 1) die("mla_attn: T > 1 not built");
        if (!po_) die("mla_attn before reserve()");
        const int n_vis = (pos + 1) / G5::KPOOL;
        const int n_keys = (!sel || n_vis <= G5::IDX_TOP_POOLS) ? pos + 1 : G5::IDX_TOP_K + (pos + 1 - n_vis * G5::KPOOL);
        const int n_chunks = ceil_div(n_keys, GLM5_ATT_CHUNK);
        g_.mark(PC_GLM_ATTN); g_.last_op_ = "glm5_attn_part";
        k_glm5_attn_part<<<dim3((unsigned) n_chunks, GLM5_ATT_HGROUPS), 256, 0, g_.stream_>>>(
            q, (const unsigned short *) kv, pos, sel, scale, po_, pm_, pl_);
        g_.mark(PC_GLM_ATTN); g_.last_op_ = "glm5_attn_combine";
        k_glm5_attn_combine<<<G5::N_HEAD, G5::KV_LORA, 0, g_.stream_>>>(po_, pm_, pl_, pos, sel, out);
    }
    void reserve(int ctx) override {
        (void) ctx;                                  // at most 2 051 keys at any depth
        if (po_) return;
        dev_ensure(g_.dev_);
        HIP_CHECK(hipMalloc(&po_, (size_t) GLM5_ATT_CHUNKS * G5::N_HEAD * G5::KV_LORA * sizeof(float)));
        HIP_CHECK(hipMalloc(&pm_, (size_t) GLM5_ATT_CHUNKS * G5::N_HEAD * sizeof(float)));
        HIP_CHECK(hipMalloc(&pl_, (size_t) GLM5_ATT_CHUNKS * G5::N_HEAD * sizeof(float)));
    }

    // ---- MoE ------------------------------------------------------------------------
    void router(const float * logits, const float * bias, float * probs, float * pb, int * ids, float * wr,
                float * wn, float * ws, const int * miss, uint32_t * stats, int T) override {
        if (T != 1) die("router: T > 1 not built");
        g_.mark(PC_ROUTER); g_.last_op_ = "glm5_router";
        k_glm5_router<<<1, GLM5_ROUTER_THREADS, 0, g_.stream_>>>(logits, bias, probs, pb, ids, wr, wn, ws,
                                                                 miss, (unsigned *) stats);
    }

    void moe_reserve(size_t max_slab) override {
        if (pk_) return;
        dev_ensure(g_.dev_);
        slot_bytes_ = (max_slab + 255) & ~(size_t) 255;
        const size_t ring = 2 * (size_t) G5::N_EXPERT_USED * slot_bytes_;     // two banks x 8 slots
        HIP_CHECK(hipMalloc(&own_ring_, ring));
        HIP_CHECK(hipMalloc(&help_ring_, ring));
        HIP_CHECK(hipMalloc(&own_sp_, 2 * (size_t) G5::N_EXPERT_USED * 3 * sizeof(void *)));
        HIP_CHECK(hipMalloc(&help_sp_, 2 * (size_t) G5::N_EXPERT_USED * 3 * sizeof(void *)));
        HIP_CHECK(hipMalloc(&pk_, 2 * sizeof(Glm5Pk)));
        HIP_CHECK(hipMalloc(&hpk_, 2 * sizeof(Glm5Pk)));
        HIP_CHECK(hipMalloc(&lg_, (size_t) G5::N_EXPERT_USED * G5::N_FF_EXP * sizeof(float)));
        HIP_CHECK(hipMalloc(&lu_, (size_t) G5::N_EXPERT_USED * G5::N_FF_EXP * sizeof(float)));
        HIP_CHECK(hipMalloc(&lh_, (size_t) G5::N_EXPERT_USED * G5::N_FF_EXP * sizeof(float)));
        HIP_CHECK(hipMalloc(&ld_, (size_t) G5::N_EXPERT_USED * G5::N_EMBD * sizeof(float)));
        HIP_CHECK(hipMemset(pk_, 0, 2 * sizeof(Glm5Pk)));
        // streams and events, created with THIS card current
        HIP_CHECK(hipStreamCreateWithFlags(&side_, hipStreamNonBlocking));
        HIP_CHECK(hipStreamCreateWithFlags(&help_, hipStreamNonBlocking));
        for (hipEvent_t * e : {&ev_plan_[0], &ev_plan_[1], &ev_fork_, &ev_own_[0], &ev_own_[1], &ev_help_})
            HIP_CHECK(hipEventCreateWithFlags(e, hipEventDisableTiming));
    }
    void moe_plan(const D::ExpertTable & t, const int * ids, const float * x, int me, const G5::LinkPlan & lp,
                  int salt, int bank) override {
        Glm5Pat pat;
        for (int i = 0; i < 11; ++i) pat.card[i] = G5::glm_link_card(lp, me, i, 0);
        g_.mark(PC_GLM_XLINK); g_.last_op_ = "glm5_plan";
        k_glm5_plan<<<1, 256, 0, g_.stream_>>>(ids, t.miss_bytes, x, pk_ + (bank & 1), me, lp.n_links, salt, pat);
        HIP_CHECK(hipEventRecord(ev_plan_[bank & 1], g_.stream_));
    }
    // On THIS (helper) card's own stream: the owner's plan, its packet over
    // P2P (16 KB), this card's missed slots over this card's host link, the
    // GEMVs, and the rows back into the owner's buffers.
    void moe_help(G5::Glm5Ops & owner_ops, const D::ExpertTable & v, int me, int bank, float * yg, float * yu,
                  float * yh, float * yd, float limit) override {
        Glm5GpuOps & o = static_cast<Glm5GpuOps &>(owner_ops);
        const int b = bank & 1;
        dev_ensure(g_.dev_);
        HIP_CHECK(hipStreamWaitEvent(help_, o.ev_plan_[b], 0));
        HIP_CHECK(hipMemcpyPeerAsync(hpk_ + b, g_.dev_, o.pk_ + b, o.g_.dev_, sizeof(Glm5Pk), help_));
        const void ** sp = help_sp_ + (size_t) b * G5::N_EXPERT_USED * 3;
        k_glm5_stage<<<dim3((unsigned) std::max(1, stage_wgs_ / G5::N_EXPERT_USED), G5::N_EXPERT_USED),
                       GLM5_STAGE_THREADS, 0, help_>>>(v.gate, v.up, v.down, hpk_ + b, me, v.sz_g, v.sz_u, v.sz_d,
            help_ring_ + (size_t) b * G5::N_EXPERT_USED * slot_bytes_, slot_bytes_, sp);
        run_slots(v, hpk_ + b, sp, me, /*pass*/ 2, lg_, lu_, lh_, ld_, limit, help_, false);
        const int rmax = std::max(v.rows_gu, v.rows_d);
        k_glm5_scatter<<<dim3((unsigned) ceil_div(rmax, BLK), G5::N_EXPERT_USED), BLK, 0, help_>>>(
            lg_, lu_, lh_, ld_, yg, yu, yh, yd, v.rows_gu, v.rows_d, hpk_ + b, me);
        HIP_CHECK(hipEventRecord(ev_help_, help_));
    }
    // The owner's own misses: fork a side stream behind the plan and copy them
    // into this card's ring while the main stream runs the shared expert.
    void moe_stage_own(const D::ExpertTable & t, int me, int bank) override {
        const int b = bank & 1;
        g_.mark(PC_GLM_XLINK); g_.last_op_ = "glm5_stage_fork";
        HIP_CHECK(hipEventRecord(ev_fork_, g_.stream_));
        HIP_CHECK(hipStreamWaitEvent(side_, ev_fork_, 0));
        k_glm5_stage<<<dim3((unsigned) std::max(1, stage_wgs_ / G5::N_EXPERT_USED), G5::N_EXPERT_USED),
                       GLM5_STAGE_THREADS, 0, side_>>>(t.gate, t.up, t.down, pk_ + b, me, t.sz_g, t.sz_u, t.sz_d,
            own_ring_ + (size_t) b * G5::N_EXPERT_USED * slot_bytes_, slot_bytes_,
            own_sp_ + (size_t) b * G5::N_EXPERT_USED * 3);
        HIP_CHECK(hipEventRecord(ev_own_[b], side_));
    }
    // Resident slots first (from the table) while the copy runs, then the join,
    // then this card's missed slots from the ring; swiglu and down on this
    // card's slots only (the helpers write theirs).
    void moe_compute_own(const D::ExpertTable & t, int me, int bank, float * yg, float * yu, float * yh,
                         float * yd, float limit) override {
        const int b = bank & 1;
        const void * const * sp = own_sp_ + (size_t) b * G5::N_EXPERT_USED * 3;
        Glm5Pk * pk = pk_ + b;
        launch_moe(t, t.type_gu, t.up,   sp, pk, me, 1, 1, t.row_gu, t.rows_gu, t.K_gu, pk->x, 0, yu, g_.stream_, true);
        launch_moe(t, t.type_gu, t.gate, sp, pk, me, 1, 0, t.row_gu, t.rows_gu, t.K_gu, pk->x, 0, yg, g_.stream_, true);
        g_.mark(PC_GLM_XLINK); g_.last_op_ = "glm5_own_join";
        HIP_CHECK(hipStreamWaitEvent(g_.stream_, ev_own_[b], 0));
        run_slots(t, pk, sp, me, 2, yg, yu, yh, yd, limit, g_.stream_, true, /*gate_up_resident_done*/ true);
    }
    void moe_join(G5::Glm5Ops & helper) override {
        Glm5GpuOps & h = static_cast<Glm5GpuOps &>(helper);
        g_.mark(PC_GLM_XLINK); g_.last_op_ = "glm5_helper_join";   // ensures THIS card
        HIP_CHECK(hipStreamWaitEvent(g_.stream_, h.ev_help_, 0));
    }

    void hc_mean(const float * H, float * out, int T) override {
        if (T != 1) die("hc_mean: T > 1 not built");
        g_.mark(PC_ELEM); g_.last_op_ = "glm5_hc_mean";
        k_glm5_hc_mean<<<ceil_div(G5::N_EMBD, BLK), BLK, 0, g_.stream_>>>(H, out);
    }

    const void * host_view(const void * host_base) override {
        dev_ensure(g_.dev_);
        void * d = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&d, const_cast<void *>(host_base), 0));
        return d;
    }
    void set_profile(bool on) override { prof_ = on; }

private:
    [[noreturn]] static void die(const char * m) { std::fprintf(stderr, "glm5 gpu: %s\n", m); std::abort(); }

    template <int TY>
    void moe_launch(const void * const * tab, const void * const * sp, const Glm5Pk * pk, int me, int pass,
                    int which, size_t rb, int rows, int K, const float * x, int x_per_slot, float * y,
                    hipStream_t s) {
        // every table entry, ring slot and slab offset is 16-byte aligned
        // (hipMalloc / hipHostMalloc bases, slices of multiples of 16: glm5_model.cpp)
        const int staged = (rb % 16 == 0) && rb <= (size_t) glm5_row_max<TY>();
        k_glm5_moe<TY><<<dim3((unsigned) ceil_div(rows, DS4_GEMV_ROWS), G5::N_EXPERT_USED), DS4_GEMV_THREADS, 0, s>>>(
            tab, sp, pk, me, pass, which, rb, rows, K, x, x_per_slot, y, staged);
    }
    void launch_moe(const D::ExpertTable & t, int type, const void * const * tab, const void * const * sp,
                    const Glm5Pk * pk, int me, int pass, int which, size_t rb, int rows, int K, const float * x,
                    int x_per_slot, float * y, hipStream_t s, bool main) {
        (void) t;
        if (main) { g_.mark(PC_GLM_EXP); g_.last_op_ = "glm5_moe"; } else dev_ensure(g_.dev_);
        switch (type) {
            case FK_Q_IQ3_S:  moe_launch<FK_Q_IQ3_S> (tab, sp, pk, me, pass, which, rb, rows, K, x, x_per_slot, y, s); break;
            case FK_Q_IQ4_XS: moe_launch<FK_Q_IQ4_XS>(tab, sp, pk, me, pass, which, rb, rows, K, x, x_per_slot, y, s); break;
            case FK_Q_Q6_K:   moe_launch<FK_Q_Q6_K>  (tab, sp, pk, me, pass, which, rb, rows, K, x, x_per_slot, y, s); break;
            default: die("routed-expert format without a kernel (GLM: IQ3_S, IQ4_XS, Q6_K)");
        }
    }
    // gate/up (unless the resident pass already ran them), swiglu on this
    // card's slots, down: pass 2 = the missed slots from `sp`. On the owner
    // (`owner_tail`) the down also runs the resident slots (pass 1).
    void run_slots(const D::ExpertTable & v, Glm5Pk * pk, const void * const * sp, int me, int pass,
                   float * yg, float * yu, float * yh, float * yd, float limit, hipStream_t s, bool main,
                   bool owner_tail = false) {
        launch_moe(v, v.type_gu, v.up,   sp, pk, me, pass, 1, v.row_gu, v.rows_gu, v.K_gu, pk->x, 0, yu, s, main);
        launch_moe(v, v.type_gu, v.gate, sp, pk, me, pass, 0, v.row_gu, v.rows_gu, v.K_gu, pk->x, 0, yg, s, main);
        if (main) { g_.mark(PC_GLM_EXP); g_.last_op_ = "glm5_swiglu_rows"; } else dev_ensure(g_.dev_);
        k_glm5_swiglu_rows<<<dim3((unsigned) ceil_div(v.rows_gu, BLK), G5::N_EXPERT_USED), BLK, 0, s>>>(
            yg, yu, yh, v.rows_gu, pk, me, limit);
        if (owner_tail)
            launch_moe(v, v.type_d, v.down, sp, pk, me, 1, 2, v.row_d, v.rows_d, v.K_d, yh, 1, yd, s, main);
        launch_moe(v, v.type_d, v.down, sp, pk, me, pass, 2, v.row_d, v.rows_d, v.K_d, yh, 1, yd, s, main);
    }

    GpuBackend & g_;
    float * po_ = nullptr, * pm_ = nullptr, * pl_ = nullptr;
    // the miss path: rings, staged pointers, packets, a helper's local rows
    size_t slot_bytes_ = 0;
    unsigned char * own_ring_ = nullptr, * help_ring_ = nullptr;
    const void ** own_sp_ = nullptr, ** help_sp_ = nullptr;
    Glm5Pk * pk_ = nullptr, * hpk_ = nullptr;
    float * lg_ = nullptr, * lu_ = nullptr, * lh_ = nullptr, * ld_ = nullptr;
    hipStream_t side_ = nullptr, help_ = nullptr;
    hipEvent_t ev_plan_[2] = {nullptr, nullptr}, ev_fork_ = nullptr, ev_own_[2] = {nullptr, nullptr};
    hipEvent_t ev_help_ = nullptr;
    int stage_wgs_ = 64;
    bool prof_ = false;
};
