// Attention over ragged sequences: each atom's central token and its kept edges,
// with no padding slots, on CUDA in an fp32 network. The kernels are mlip.cpp's
// (perf/gpu 1e328da, ggml-pet-ragged.patch: pet-ragged.cu) with the row layout
// and the bias source changed to pet-kokkos's.
//
// Token t of atom n is one row of a [T, *] array:
//   packed (TokenLayout::S0 = 0):  t = 0 -> row n, t > 0 -> row N + off(n) + t - 1,
//                                   T = N + E: the central rows, then the packed edges;
//   padded (S0 = S):               row n S + t, T = N S (pet-kokkos's [N, M+1] tokens).
// Atom n has L = 1 + off(n+1) - off(n) tokens. Key t > 0 carries the bias
// lcf(off(n) + t - 1) = log(cf) of its edge (-inf when cf <= 0: no query sees it),
// the central key none. The forward saves the merged heads and each row's
// log-sum-exp per head; the backward recomputes the weights from them.
//
// Short sequences (max L <= 64, head size <= 32) take one block per (atom, head):
//   split  (default) a thread per row (PET_RAGGED_SPLIT=S[,R]: S lanes a row, R
//          rows a thread), all of the head's rows staged at once with cp.async; the
//          backward's two passes (queries: dQ; keys: dK, dV, the bias adjoint) side
//          by side in one block;
//   mma    on TF32 tensor cores, FlashAttention-2 style (PET_RAGGED_MMA=1), or
//          3xTF32, close to fp32 (=3): by default 3xTF32 for head size 32 when the
//          GEMMs run in TF32, never in the native (fp32) mode;
// longer ones the tiled kernels (a block per (atom, query tile), all heads).
#include "ops.hpp"

#include "pet/gemm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace pet {

#if defined(KOKKOS_ENABLE_CUDA) && !defined(PET_KOKKOS_FP64)
#define PET_RAGGED 1
#endif

bool attention_ragged_supported(int heads, int head_dim) {
#if defined(PET_RAGGED)
  static const bool off = [] {
    const char* e = std::getenv("PET_ATTN");
    return e && std::strcmp(e, "kokkos") == 0;
  }();
  return !off && heads >= 1 && heads <= 256 &&
         (head_dim == 8 || head_dim == 16 || head_dim == 32 || head_dim == 64);
#else
  (void) heads, (void) head_dim;
  return false;
#endif
}

#if defined(PET_RAGGED)

namespace {

// ---- sequences ----

struct Seq {
  long long r0, e0;  // the central row, the first edge row
  int a, L;          // the first edge's packed index, the length
  __device__ __forceinline__ long long row(const int t) const { return t == 0 ? r0 : e0 + t - 1; }
  __device__ __forceinline__ float bias(const float* __restrict__ lcf, const int t) const {
    return t == 0 ? 0.0f : lcf[a + t - 1];
  }
};

__device__ __forceinline__ Seq seq_of(const int* __restrict__ off, const int n, const int N, const int S0) {
  const int a = off[n], b = off[n + 1];
  const long long r0 = S0 ? (long long) n * S0 : n;
  return {r0, S0 ? r0 + 1 : (long long) N + a, a, 1 + b - a};
}

// ---- long sequences: tiled ----
// A thread owns R rows of one head; a block holds G row groups x H heads and
// streams the other side of its sequence in tiles of TS rows through shared
// memory, rows padded to d + 4 floats. Block (n, y) holds rows [y TP, (y+1) TP).

constexpr int kMaxSmem = 48 * 1024;

template <int d>
__device__ __forceinline__ void load_row(const float* __restrict__ s, float (&r)[d]) {
#pragma unroll
  for (int c = 0; c < d; c += 4) {
    const float4 v = *reinterpret_cast<const float4*>(s + c);
    r[c] = v.x, r[c + 1] = v.y, r[c + 2] = v.z, r[c + 3] = v.w;
  }
}

// tokens [k0, k0 + nk) of q, columns [col, col + D) of rows of `stride` floats, into shared [nk][H][P]
__device__ __forceinline__ void stage_tokens(float* __restrict__ dst, const float* __restrict__ src, const Seq& q,
                                             const int k0, const int nk, const long long stride, const int col,
                                             const int D, const int d, const int P, const int t, const int nt) {
  const int H = D / d, D4 = D / 4;
  for (int e = t; e < nk * D4; e += nt) {
    const int r = e / D4, c = (e % D4) * 4;
    *reinterpret_cast<float4*>(dst + (r * H + c / d) * P + c % d) =
        *reinterpret_cast<const float4*>(src + q.row(k0 + r) * stride + col + c);
  }
}

template <int d, int R>
__global__ void fwd_tiled(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
                          float* __restrict__ out, float* __restrict__ lse, const int N, const int S0, const int H,
                          const int TP, const int TS, const float scale) {
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float* Ks = sm;
  float* Vs = Ks + TS * H * P;
  float* Bs = Vs + TS * H * P;
  const Seq sq = seq_of(off, blockIdx.x, N, S0);
  const int L = sq.L, q0 = blockIdx.y * TP;
  if (q0 >= L) return;
  const int t = threadIdx.x, h = t % H, g = t / H;
  float q[R][d], acc[R][d], m[R], l[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    m[r] = -INFINITY, l[r] = 0.0f;
    const float* src = qkv + sq.row(i < L ? i : 0) * 3 * D + h * d;
#pragma unroll
    for (int c = 0; c < d; c++) q[r][c] = i < L ? scale * src[c] : 0.0f, acc[r][c] = 0.0f;
  }
  for (int k0 = 0; k0 < L; k0 += TS) {
    const int nk = min(TS, L - k0);
    stage_tokens(Ks, qkv, sq, k0, nk, 3 * D, D, D, d, P, t, blockDim.x);
    stage_tokens(Vs, qkv, sq, k0, nk, 3 * D, 2 * D, D, d, P, t, blockDim.x);
    for (int e = t; e < nk; e += blockDim.x) Bs[e] = sq.bias(lcf, k0 + e);
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
      float kr[d], vr[d];
      load_row<d>(Ks + (kk * H + h) * P, kr);
      load_row<d>(Vs + (kk * H + h) * P, vr);
      const float b = Bs[kk];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = b;
#pragma unroll
        for (int c = 0; c < d; c++) s += q[r][c] * kr[c];
        if (s > m[r]) {
          const float corr = __expf(m[r] - s);
          l[r] *= corr;
#pragma unroll
          for (int c = 0; c < d; c++) acc[r][c] *= corr;
          m[r] = s;
        }
        const float p = __expf(s - m[r]);
        l[r] += p;
#pragma unroll
        for (int c = 0; c < d; c++) acc[r][c] += p * vr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    if (i >= L) continue;
    const long long row = sq.row(i);
    float* o = out + row * D + h * d;
#pragma unroll
    for (int c = 0; c < d; c++) o[c] = acc[r][c] / l[r];
    lse[row * H + h] = m[r] + logf(l[r]);
  }
}

template <int d, int R>
__global__ void bwd_dq_tiled(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
                             const float* __restrict__ res, const float* __restrict__ lse,
                             const float* __restrict__ grad, float* __restrict__ dx, const int N, const int S0,
                             const int H, const int TP, const int TS, const float scale) {
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float* Ks = sm;
  float* Vs = Ks + TS * H * P;
  float* Bs = Vs + TS * H * P;
  const Seq sq = seq_of(off, blockIdx.x, N, S0);
  const int L = sq.L, q0 = blockIdx.y * TP;
  if (q0 >= L) return;
  const int t = threadIdx.x, h = t % H, g = t / H;
  float q[R][d], dO[R][d], dq[R][d], Di[R], ls[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    const bool ok = i < L;
    const long long row = sq.row(ok ? i : 0);
    Di[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < d; c++) {
      q[r][c] = ok ? qkv[row * 3 * D + h * d + c] : 0.0f;
      dO[r][c] = ok ? grad[row * D + h * d + c] : 0.0f;
      Di[r] += dO[r][c] * (ok ? res[row * D + h * d + c] : 0.0f);
      dq[r][c] = 0.0f;
    }
    ls[r] = ok ? lse[row * H + h] : 0.0f;
  }
  for (int k0 = 0; k0 < L; k0 += TS) {
    const int nk = min(TS, L - k0);
    stage_tokens(Ks, qkv, sq, k0, nk, 3 * D, D, D, d, P, t, blockDim.x);
    stage_tokens(Vs, qkv, sq, k0, nk, 3 * D, 2 * D, D, d, P, t, blockDim.x);
    for (int e = t; e < nk; e += blockDim.x) Bs[e] = sq.bias(lcf, k0 + e);
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
      float kr[d], vr[d];
      load_row<d>(Ks + (kk * H + h) * P, kr);
      load_row<d>(Vs + (kk * H + h) * P, vr);
      const float b = Bs[kk];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int c = 0; c < d; c++) s += q[r][c] * kr[c], dp += dO[r][c] * vr[c];
        const float p = __expf(scale * s + b - ls[r]);
        const float ds = p * (dp - Di[r]);
#pragma unroll
        for (int c = 0; c < d; c++) dq[r][c] += ds * kr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    if (i >= L) continue;
    float* o = dx + sq.row(i) * 3 * D + h * d;
#pragma unroll
    for (int c = 0; c < d; c++) o[c] = scale * dq[r][c];
  }
}

template <int d, int R>
__global__ void bwd_dkv_tiled(const float* __restrict__ qkv, const float* __restrict__ lcf,
                              const int* __restrict__ off, const float* __restrict__ res,
                              const float* __restrict__ lse, const float* __restrict__ grad, float* __restrict__ dx,
                              float* __restrict__ db, const int N, const int S0, const int H, const int TP,
                              const int TS, const float scale) {
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float* Qs = sm;
  float* dOs = Qs + TS * H * P;
  float* Ls = dOs + TS * H * P;
  float* Ds = Ls + TS * H;
  const Seq sq = seq_of(off, blockIdx.x, N, S0);
  const int L = sq.L, k0 = blockIdx.y * TP;
  if (k0 >= L) return;
  const int t = threadIdx.x, h = t % H, g = t / H;
  float k[R][d], v[R][d], dk[R][d], dv[R][d], bj[R], dbj[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int j = k0 + g + r * G;
    const bool ok = j < L;
    const long long row = sq.row(ok ? j : 0);
#pragma unroll
    for (int c = 0; c < d; c++) {
      k[r][c] = ok ? qkv[row * 3 * D + D + h * d + c] : 0.0f;
      v[r][c] = ok ? qkv[row * 3 * D + 2 * D + h * d + c] : 0.0f;
      dk[r][c] = dv[r][c] = 0.0f;
    }
    bj[r] = ok ? sq.bias(lcf, j) : 0.0f, dbj[r] = 0.0f;
  }
  for (int q0 = 0; q0 < L; q0 += TS) {
    const int nq = min(TS, L - q0);
    stage_tokens(Qs, qkv, sq, q0, nq, 3 * D, 0, D, d, P, t, blockDim.x);
    stage_tokens(dOs, grad, sq, q0, nq, D, 0, D, d, P, t, blockDim.x);
    for (int e = t; e < nq * H; e += blockDim.x) Ls[e] = lse[sq.row(q0 + e / H) * H + e % H];
    __syncthreads();
    for (int e = t; e < nq * H; e += blockDim.x) {
      const int qq = e / H, hh = e % H;
      const float* o = res + sq.row(q0 + qq) * D + hh * d;
      float s = 0.0f;
      for (int c = 0; c < d; c++) s += dOs[(qq * H + hh) * P + c] * o[c];
      Ds[e] = s;
    }
    __syncthreads();
    for (int qq = 0; qq < nq; qq++) {
      float qr[d], gr[d];
      load_row<d>(Qs + (qq * H + h) * P, qr);
      load_row<d>(dOs + (qq * H + h) * P, gr);
      const float lq = Ls[qq * H + h], dq = Ds[qq * H + h];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int c = 0; c < d; c++) s += qr[c] * k[r][c], dp += gr[c] * v[r][c];
        const float p = __expf(scale * s + bj[r] - lq);
        const float ds = p * (dp - dq);
        dbj[r] += ds;
#pragma unroll
        for (int c = 0; c < d; c++) dv[r][c] += p * gr[c], dk[r][c] += ds * qr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int j = k0 + g + r * G;
    if (j >= L) continue;
    const long long row = sq.row(j);
    float* o = dx + row * 3 * D;
#pragma unroll
    for (int c = 0; c < d; c++) o[D + h * d + c] = scale * dk[r][c], o[2 * D + h * d + c] = dv[r][c];
    db[row * H + h] = dbj[r];
  }
}

struct TiledShape {
  int TP, TS;
  size_t smem;
};
TiledShape tiled_shape(int H, int d, int R, int L, bool dkv) {
  const int G = std::max(1, std::min(256 / H, (L + R - 1) / R)), TP = G * R;
  auto smem = [&](int TS) {
    return (2 * (size_t) TS * H * (d + 4) + (dkv ? 2 * (size_t) TS * H : (size_t) TS)) * sizeof(float);
  };
  int TS = 32;
  while (TS > 1 && (TS / 2 >= L || smem(TS) > kMaxSmem)) TS /= 2;
  return {TP, TS, smem(TS)};
}

// ---- short sequences, split rows ----
// S lanes share a row (C = d / S channels each, the dot products reduced over the
// S lanes by shuffles), R rows a thread. Block n H + h: atom n, head h, so the
// heads of an atom are consecutive blocks and read whole token rows together.

constexpr int kShortL = 64;

__device__ __forceinline__ void cp_async16(float* dst, const float* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned s = (unsigned) __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
#else
  *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
#endif
}
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// columns [col, col + d) of the L rows of q into dst [L][d], asynchronously
template <int d>
__device__ __forceinline__ void stage_head_async(float* __restrict__ dst, const float* __restrict__ src, const Seq& q,
                                                 const int L, const long long stride, const int col) {
  constexpr int d4 = d / 4;
  for (int e = threadIdx.x; e < L * d4; e += blockDim.x) {
    const int r = e / d4, c = (e % d4) * 4;
    cp_async16(dst + r * d + c, src + q.row(r) * stride + col + c);
  }
}

template <int C>
__device__ __forceinline__ void load4(const float* __restrict__ s, float (&r)[C]) {
#pragma unroll
  for (int c = 0; c < C; c += 4) {
    const float4 v = *reinterpret_cast<const float4*>(s + c);
    r[c] = v.x, r[c + 1] = v.y, r[c + 2] = v.z, r[c + 3] = v.w;
  }
}
template <int C>
__device__ __forceinline__ void store4(float* __restrict__ s, const float (&r)[C], const float f) {
#pragma unroll
  for (int c = 0; c < C; c += 4)
    *reinterpret_cast<float4*>(s + c) = make_float4(f * r[c], f * r[c + 1], f * r[c + 2], f * r[c + 3]);
}

// sum over the S lanes of a row (S consecutive lanes of a warp, active together)
template <int S>
__device__ __forceinline__ float row_sum(float v) {
  if constexpr (S > 1) {
    const unsigned mask = ((1u << S) - 1u) << ((threadIdx.x % 32) & ~(S - 1));
#pragma unroll
    for (int o = S / 2; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
  }
  return v;
}

__host__ __device__ __forceinline__ int split_groups(const int Lm, const int R) { return (Lm + R - 1) / R; }
int split_half(const int Lm, const int S, const int R) { return (split_groups(Lm, R) * S + 31) / 32 * 32; }

template <int d, int S, int R>
__global__ void fwd_split(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
                          float* __restrict__ out, float* __restrict__ lse, const int N, const int S0, const int H,
                          const float scale, const int Lm) {
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  constexpr int C = d / S;
  const int n = blockIdx.x / H, h = blockIdx.x % H;
  const Seq sq = seq_of(off, n, N, S0);
  const int L = sq.L, D = H * d, t = threadIdx.x, p = t % S, G = split_groups(Lm, R), g = t / S;
  float *Ks = sm, *Vs = Ks + Lm * d, *Bs = Vs + Lm * d;
  stage_head_async<d>(Ks, qkv, sq, L, 3 * D, D + h * d);
  stage_head_async<d>(Vs, qkv, sq, L, 3 * D, 2 * D + h * d);
  for (int u = t; u < L; u += blockDim.x) Bs[u] = sq.bias(lcf, u);
  const bool active = g < min(G, L);
  float q[R][C], acc[R][C], m[R], l[R];
  if (active) {
#pragma unroll
    for (int r = 0; r < R; r++) {
      load4<C>(qkv + sq.row(min(g + r * G, L - 1)) * 3 * D + h * d + p * C, q[r]);
      m[r] = -INFINITY, l[r] = 0.0f;
#pragma unroll
      for (int c = 0; c < C; c++) q[r][c] *= scale, acc[r][c] = 0.0f;
    }
  }
  cp_async_wait();
  __syncthreads();
  if (!active) return;
  for (int u = 0; u < L; u++) {
    float k[C], v[C];
    load4<C>(Ks + u * d + p * C, k);
    load4<C>(Vs + u * d + p * C, v);
    const float b = Bs[u];
#pragma unroll
    for (int r = 0; r < R; r++) {
      float s = 0.0f;
#pragma unroll
      for (int c = 0; c < C; c++) s += q[r][c] * k[c];
      s = row_sum<S>(s) + b;
      if (s > m[r]) {
        const float corr = __expf(m[r] - s);
        l[r] *= corr;
#pragma unroll
        for (int c = 0; c < C; c++) acc[r][c] *= corr;
        m[r] = s;
      }
      const float e = __expf(s - m[r]);
      l[r] += e;
#pragma unroll
      for (int c = 0; c < C; c++) acc[r][c] += e * v[c];
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = g + r * G;
    if (i >= L) continue;
    const long long row = sq.row(i);
    store4<C>(out + row * D + h * d + p * C, acc[r], 1.0f / l[r]);
    if (p == 0) lse[row * H + h] = m[r] + logf(l[r]);
  }
}

template <int d, int S, int R>
__global__ void bwd_split(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
                          const float* __restrict__ res, const float* __restrict__ lse, const float* __restrict__ grad,
                          float* __restrict__ dx, float* __restrict__ db, const int N, const int S0, const int H,
                          const float scale, const int Lm, const int half) {
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  constexpr int C = d / S;
  const int n = blockIdx.x / H, h = blockIdx.x % H;
  const Seq sq = seq_of(off, n, N, S0);
  const int L = sq.L, D = H * d, G = split_groups(Lm, R);
  const bool keys = threadIdx.x >= half;  // first half: query rows, second half: key rows
  const int t = threadIdx.x - (keys ? half : 0), p = t % S, g = t / S;
  const bool active = g < min(G, L);
  float *Ks = sm, *Vs = Ks + Lm * d, *Qs = Vs + Lm * d, *Gs = Qs + Lm * d;
  float *Bs = Gs + Lm * d, *Ls = Bs + Lm, *Ds = Ls + Lm;
  stage_head_async<d>(Ks, qkv, sq, L, 3 * D, D + h * d);
  stage_head_async<d>(Vs, qkv, sq, L, 3 * D, 2 * D + h * d);
  stage_head_async<d>(Qs, qkv, sq, L, 3 * D, h * d);
  stage_head_async<d>(Gs, grad, sq, L, D, h * d);
  for (int u = threadIdx.x; u < L; u += blockDim.x) Bs[u] = sq.bias(lcf, u);
  // query rows: O and the lse from global while the stage is in flight
  float o[R][C], li[R];
  if (!keys && active) {
#pragma unroll
    for (int r = 0; r < R; r++) {
      const long long row = sq.row(min(g + r * G, L - 1));
      load4<C>(res + row * D + h * d + p * C, o[r]);
      li[r] = lse[row * H + h];
    }
  }
  cp_async_wait();
  __syncthreads();
  float a[R][C], b[R][C], acc[R][C], Di[R];  // query: q, dO, dq; key: k, v, dk (+ dv)
  if (active) {
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int i = min(g + r * G, L - 1);
      load4<C>((keys ? Ks : Qs) + i * d + p * C, a[r]);
      load4<C>((keys ? Vs : Gs) + i * d + p * C, b[r]);
#pragma unroll
      for (int c = 0; c < C; c++) acc[r][c] = 0.0f;
      if (!keys) {
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < C; c++) s += b[r][c] * o[r][c];
        Di[r] = row_sum<S>(s);
        if (p == 0 && g + r * G < L) Ls[i] = li[r], Ds[i] = Di[r];
      }
    }
  }
  __syncthreads();
  if (!active) return;
  if (!keys) {
    for (int u = 0; u < L; u++) {
      float k[C], v[C];
      load4<C>(Ks + u * d + p * C, k);
      load4<C>(Vs + u * d + p * C, v);
      const float bu = Bs[u];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int c = 0; c < C; c++) s += a[r][c] * k[c], dp += b[r][c] * v[c];
        s = row_sum<S>(s), dp = row_sum<S>(dp);
        const float ds = __expf(scale * s + bu - li[r]) * (dp - Di[r]);
#pragma unroll
        for (int c = 0; c < C; c++) acc[r][c] += ds * k[c];
      }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
      const int i = g + r * G;
      if (i < L) store4<C>(dx + sq.row(i) * 3 * D + h * d + p * C, acc[r], scale);
    }
    return;
  }
  float dv[R][C], bj[R], dbj[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    bj[r] = Bs[min(g + r * G, L - 1)], dbj[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < C; c++) dv[r][c] = 0.0f;
  }
  for (int i = 0; i < L; i++) {
    float q[C], gi[C];
    load4<C>(Qs + i * d + p * C, q);
    load4<C>(Gs + i * d + p * C, gi);
    const float l = Ls[i], Dq = Ds[i];
#pragma unroll
    for (int r = 0; r < R; r++) {
      float s = 0.0f, dp = 0.0f;
#pragma unroll
      for (int c = 0; c < C; c++) s += q[c] * a[r][c], dp += gi[c] * b[r][c];
      s = row_sum<S>(s), dp = row_sum<S>(dp);
      const float e = __expf(scale * s + bj[r] - l), ds = e * (dp - Dq);
      dbj[r] += ds;
#pragma unroll
      for (int c = 0; c < C; c++) acc[r][c] += ds * q[c], dv[r][c] += e * gi[c];
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int j = g + r * G;
    if (j >= L) continue;
    const long long row = sq.row(j);
    store4<C>(dx + row * 3 * D + D + h * d + p * C, acc[r], scale);
    store4<C>(dx + row * 3 * D + 2 * D + h * d + p * C, dv[r], 1.0f);
    if (p == 0) db[row * H + h] = dbj[r];
  }
}

// ---- short sequences on TF32 tensor cores ----
// One block per (atom, head), the sequence padded to Lp = 16 NW rows (zeros;
// padded keys get probability 0), Q, K, V (and dO) staged in shared memory.
// Forward: warp w owns query rows 16 w ..: S = Q K^T with mma.m16n8k8 (fp32
// accumulation), the softmax on the accumulators, O = P V. Backward (2 NW warps,
// P and dS never stored): warp w < NW owns query rows (S, dP -> dS, dQ = dS K),
// warp NW + w key rows (S^T = K Q^T, dP^T = V dO^T -> P^T, dS^T, dV = P^T dO,
// dK = dS^T Q, the bias adjoint as row sums of dS^T). An accumulator tile is the
// A operand of the next product as it is, its k slots q, q + 4 taken as the
// columns 2 q, 2 q + 1 it holds (the B operand loaded in the same order). X3:
// 3xTF32 (a = a_hi + a_lo; a_hi b_hi + a_hi b_lo + a_lo b_hi), close to fp32.

__device__ __forceinline__ uint32_t to_tf32(const float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}

template <bool X3>
struct FragA {
  uint32_t hi[4], lo[X3 ? 4 : 1];
};
template <bool X3>
struct FragB {
  uint32_t hi[2], lo[X3 ? 2 : 1];
};

template <bool X3, int NR>
__device__ __forceinline__ void split_tf32(const float (&x)[NR], uint32_t (&hi)[NR], uint32_t (&lo)[X3 ? NR : 1]) {
#pragma unroll
  for (int i = 0; i < NR; i++) {
    hi[i] = to_tf32(x[i]);
    if constexpr (X3) lo[i] = to_tf32(x[i] - __uint_as_float(hi[i]));
  }
}

__device__ __forceinline__ void mma_tf32(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, "
      "%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void) c, (void) a, (void) b;
#endif
}

template <bool X3>
__device__ __forceinline__ void mma(float (&c)[4], const FragA<X3>& a, const FragB<X3>& b) {
  if constexpr (X3) {
    uint32_t al[4] = {a.lo[0], a.lo[1], a.lo[2], a.lo[3]}, bl[2] = {b.lo[0], b.lo[1]};
    mma_tf32(c, al, b.hi);
    mma_tf32(c, a.hi, bl);
  }
  mma_tf32(c, a.hi, b.hi);
}

// A (16 x 8) at (m0, k0) of X [m][k] (leading dimension ld)
template <bool X3>
__device__ __forceinline__ FragA<X3> load_a(const float* __restrict__ X, const int ld, const int m0, const int k0) {
  const int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const float x[4] = {X[(m0 + g) * ld + k0 + q], X[(m0 + g + 8) * ld + k0 + q], X[(m0 + g) * ld + k0 + q + 4],
                      X[(m0 + g + 8) * ld + k0 + q + 4]};
  FragA<X3> f;
  split_tf32<X3, 4>(x, f.hi, f.lo);
  return f;
}
// B (8 x 8) at (k0, n0) with B(k, n) = X [n][k]
template <bool X3>
__device__ __forceinline__ FragB<X3> load_b(const float* __restrict__ X, const int ld, const int k0, const int n0) {
  const int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const float x[2] = {X[(n0 + g) * ld + k0 + q], X[(n0 + g) * ld + k0 + q + 4]};
  FragB<X3> f;
  split_tf32<X3, 2>(x, f.hi, f.lo);
  return f;
}
// B for the k-tile k0 of an accumulator used as A: slots q, q + 4 <- rows k0 + 2 q, k0 + 2 q + 1 of X [k][n]
template <bool X3>
__device__ __forceinline__ FragB<X3> load_b_acc(const float* __restrict__ X, const int ld, const int k0, const int n0) {
  const int lane = threadIdx.x % 32, g = lane / 4, q = lane % 4;
  const float x[2] = {X[(k0 + 2 * q) * ld + n0 + g], X[(k0 + 2 * q + 1) * ld + n0 + g]};
  FragB<X3> f;
  split_tf32<X3, 2>(x, f.hi, f.lo);
  return f;
}
// an accumulator tile (rows g, g + 8, columns 2 q, 2 q + 1) as the A operand of a product over its columns
template <bool X3>
__device__ __forceinline__ FragA<X3> acc_a(const float (&c)[4]) {
  const float x[4] = {c[0], c[2], c[1], c[3]};
  FragA<X3> f;
  split_tf32<X3, 4>(x, f.hi, f.lo);
  return f;
}

// rows [0, L) of columns [col, col + d) into dst [Lp][d + 4] asynchronously, rows [L, Lp) zeroed
template <int d>
__device__ __forceinline__ void stage_head_pad(float* __restrict__ dst, const float* __restrict__ src, const Seq& q,
                                               const int L, const int Lp, const long long stride, const int col) {
  constexpr int d4 = d / 4, ld = d + 4;
  for (int e = threadIdx.x; e < Lp * d4; e += blockDim.x) {
    const int r = e / d4, c = (e % d4) * 4;
    if (r < L) cp_async16(dst + r * ld + c, src + q.row(r) * stride + col + c);
    else *reinterpret_cast<float4*>(dst + r * ld + c) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// acc[n] += A B[n] over k in [0, K): A (16 x K) at row m0 of X [m][k], B[n] (K x 8) = columns 8 n of Y^T, Y [n][k]
template <bool X3, int K, int NN>
__device__ __forceinline__ void mma_abt(float (&acc)[NN][4], const float* __restrict__ X, const float* __restrict__ Y,
                                        const int ld, const int m0) {
#pragma unroll
  for (int k0 = 0; k0 < K; k0 += 8) {
    const FragA<X3> a = load_a<X3>(X, ld, m0, k0);
#pragma unroll
    for (int n = 0; n < NN; n++) mma<X3>(acc[n], a, load_b<X3>(Y, ld, k0, 8 * n));
  }
}
// acc[n] += P Y[:, 8 n ..] with P (16 x 8 NK) the accumulator tiles p, Y [k][n]
template <bool X3, int NK, int NN>
__device__ __forceinline__ void mma_acc(float (&acc)[NN][4], const float (&p)[NK][4], const float* __restrict__ Y,
                                        const int ld) {
#pragma unroll
  for (int kk = 0; kk < NK; kk++) {
    const FragA<X3> a = acc_a<X3>(p[kk]);
#pragma unroll
    for (int n = 0; n < NN; n++) mma<X3>(acc[n], a, load_b_acc<X3>(Y, ld, 8 * kk, 8 * n));
  }
}
template <int NN>
__device__ __forceinline__ void zero(float (&acc)[NN][4]) {
#pragma unroll
  for (int n = 0; n < NN; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
}

template <int d, int NW, bool X3>
__global__ void __launch_bounds__(32 * NW)
    fwd_mma(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
            float* __restrict__ out, float* __restrict__ lse, const int N, const int S0, const int H,
            const float scale) {
  constexpr int Lp = 16 * NW, ld = d + 4, NT = Lp / 8;
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  float *Qs = sm, *Ks = Qs + Lp * ld, *Vs = Ks + Lp * ld, *Bs = Vs + Lp * ld;
  const int n = blockIdx.x / H, h = blockIdx.x % H;
  const Seq sq = seq_of(off, n, N, S0);
  const int L = sq.L, D = H * d;
  stage_head_pad<d>(Qs, qkv, sq, L, Lp, 3 * D, h * d);
  stage_head_pad<d>(Ks, qkv, sq, L, Lp, 3 * D, D + h * d);
  stage_head_pad<d>(Vs, qkv, sq, L, Lp, 3 * D, 2 * D + h * d);
  for (int u = threadIdx.x; u < Lp; u += blockDim.x) Bs[u] = u < L ? sq.bias(lcf, u) : -INFINITY;
  cp_async_wait();
  __syncthreads();
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, q = lane % 4, m0 = 16 * w;
  if (m0 >= L) return;
  float s[NT][4];
  zero(s);
  mma_abt<X3, d>(s, Qs, Ks, ld, m0);
  // c0, c1 in row g, c2, c3 in row g + 8, columns 8 n + 2 q (+1)
  float mx[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int e = 0; e < 4; e++) {
      s[n][e] = scale * s[n][e] + Bs[8 * n + 2 * q + e % 2];
      mx[e / 2] = fmaxf(mx[e / 2], s[n][e]);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; r++)
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1)),
    mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int e = 0; e < 4; e++) s[n][e] = __expf(s[n][e] - mx[e / 2]), l[e / 2] += s[n][e];
  }
#pragma unroll
  for (int r = 0; r < 2; r++)
    l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1), l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
  float o[d / 8][4];
  zero(o);
  mma_acc<X3>(o, s, Vs, ld);
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int i = m0 + g + 8 * r;
    if (i >= L) continue;
    const long long row = sq.row(i);
    float* dst = out + row * D + h * d;
#pragma unroll
    for (int n = 0; n < d / 8; n++)
      *reinterpret_cast<float2*>(dst + 8 * n + 2 * q) = make_float2(o[n][2 * r] / l[r], o[n][2 * r + 1] / l[r]);
    if (q == 0) lse[row * H + h] = mx[r] + logf(l[r]);
  }
}

template <int d, int NW, bool X3>
__global__ void __launch_bounds__(64 * NW)
    bwd_mma(const float* __restrict__ qkv, const float* __restrict__ lcf, const int* __restrict__ off,
            const float* __restrict__ res, const float* __restrict__ lse, const float* __restrict__ grad,
            float* __restrict__ dx, float* __restrict__ db, const int N, const int S0, const int H,
            const float scale) {
  constexpr int Lp = 16 * NW, ld = d + 4, NT = Lp / 8;
  extern __shared__ float4 sm4[];
  float* sm = reinterpret_cast<float*>(sm4);
  float *Qs = sm, *Ks = Qs + Lp * ld, *Vs = Ks + Lp * ld, *Gs = Vs + Lp * ld, *Bs = Gs + Lp * ld, *Ls = Bs + Lp,
        *Ds = Ls + Lp;
  const int n = blockIdx.x / H, h = blockIdx.x % H;
  const Seq sq = seq_of(off, n, N, S0);
  const int L = sq.L, D = H * d, t = threadIdx.x;
  stage_head_pad<d>(Qs, qkv, sq, L, Lp, 3 * D, h * d);
  stage_head_pad<d>(Ks, qkv, sq, L, Lp, 3 * D, D + h * d);
  stage_head_pad<d>(Vs, qkv, sq, L, Lp, 3 * D, 2 * D + h * d);
  stage_head_pad<d>(Gs, grad, sq, L, Lp, D, h * d);
  for (int u = t; u < Lp; u += blockDim.x) {
    Bs[u] = u < L ? sq.bias(lcf, u) : 0.0f;
    Ls[u] = u < L ? lse[sq.row(u) * H + h] : INFINITY;  // padded queries: p = 0
  }
  // Di = dO_i . O_i, 4 threads per row (blockDim = 4 Lp): O from global while the stage is in flight
  const int r4 = t / 4, c4 = (t % 4) * (d / 4);
  float o4[d / 4];
  if (r4 < L) {
#pragma unroll
    for (int c = 0; c < d / 4; c += 4) {
      const float4 v = *reinterpret_cast<const float4*>(res + sq.row(r4) * D + h * d + c4 + c);
      o4[c] = v.x, o4[c + 1] = v.y, o4[c + 2] = v.z, o4[c + 3] = v.w;
    }
  }
  cp_async_wait();
  __syncthreads();
  {
    float s = 0.0f;
    if (r4 < L) {
#pragma unroll
      for (int c = 0; c < d / 4; c++) s += o4[c] * Gs[r4 * ld + c4 + c];
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (t % 4 == 0) Ds[r4] = s;
  }
  __syncthreads();
  const int w = t / 32, lane = t % 32, g = lane / 4, q = lane % 4;
  if (w < NW) {
    // query rows m0 ..: dS = P (dP - D), dQ = dS K
    const int m0 = 16 * w;
    if (m0 >= L) return;
    float s[NT][4], dp[NT][4];
    zero(s), zero(dp);
    mma_abt<X3, d>(s, Qs, Ks, ld, m0);
    mma_abt<X3, d>(dp, Gs, Vs, ld, m0);
    const float li[2] = {Ls[m0 + g], Ls[m0 + g + 8]}, Di[2] = {Ds[m0 + g], Ds[m0 + g + 8]};
#pragma unroll
    for (int n = 0; n < NT; n++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const float p = __expf(scale * s[n][e] + Bs[8 * n + 2 * q + e % 2] - li[e / 2]);
        dp[n][e] = 8 * n + 2 * q + e % 2 < L ? p * (dp[n][e] - Di[e / 2]) : 0.0f;
      }
    }
    float dq[d / 8][4];
    zero(dq);
    mma_acc<X3>(dq, dp, Ks, ld);
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int i = m0 + g + 8 * r;
      if (i >= L) continue;
      float* dst = dx + sq.row(i) * 3 * D + h * d;
#pragma unroll
      for (int n = 0; n < d / 8; n++)
        *reinterpret_cast<float2*>(dst + 8 * n + 2 * q) = make_float2(scale * dq[n][2 * r], scale * dq[n][2 * r + 1]);
    }
    return;
  }
  // key rows j0 ..: P^T, dS^T = P^T (dP^T - D), dV = P^T dO, dK = dS^T Q, the bias adjoint
  const int j0 = 16 * (w - NW);
  if (j0 >= L) return;
  float s[NT][4], dp[NT][4];
  zero(s), zero(dp);
  mma_abt<X3, d>(s, Ks, Qs, ld, j0);
  mma_abt<X3, d>(dp, Vs, Gs, ld, j0);
  const float bj[2] = {Bs[j0 + g], Bs[j0 + g + 8]};
  float dbj[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n = 0; n < NT; n++) {
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int i = 8 * n + 2 * q + e % 2;
      s[n][e] = __expf(scale * s[n][e] + bj[e / 2] - Ls[i]);
      dp[n][e] = s[n][e] * (dp[n][e] - Ds[i]);
      dbj[e / 2] += dp[n][e];
    }
  }
#pragma unroll
  for (int r = 0; r < 2; r++)
    dbj[r] += __shfl_xor_sync(0xffffffffu, dbj[r], 1), dbj[r] += __shfl_xor_sync(0xffffffffu, dbj[r], 2);
  float dk[d / 8][4], dv[d / 8][4];
  zero(dk), zero(dv);
  mma_acc<X3>(dv, s, Gs, ld);
  mma_acc<X3>(dk, dp, Qs, ld);
#pragma unroll
  for (int r = 0; r < 2; r++) {
    const int j = j0 + g + 8 * r;
    if (j >= L) continue;
    const long long row = sq.row(j);
    float* dst = dx + row * 3 * D;
#pragma unroll
    for (int n = 0; n < d / 8; n++) {
      *reinterpret_cast<float2*>(dst + D + h * d + 8 * n + 2 * q) =
          make_float2(scale * dk[n][2 * r], scale * dk[n][2 * r + 1]);
      *reinterpret_cast<float2*>(dst + 2 * D + h * d + 8 * n + 2 * q) = make_float2(dv[n][2 * r], dv[n][2 * r + 1]);
    }
    if (q == 0) db[row * H + h] = dbj[r];
  }
}

// ---- dispatch ----

struct Args {
  const float *qkv, *lcf;
  const int* off;
  int N, S0, H, Lm;
  float scale;
  cudaStream_t st;
};

template <int d, int S, int R>
void split_launch_sr(const Args& a, bool back, const float* res, const float* lse_in, const float* grad, float* out,
                  float* aux) {
  const unsigned grid = (unsigned) a.N * (unsigned) a.H;
  const int half = split_half(a.Lm, S, R);
  if (!back)
    fwd_split<d, S, R><<<grid, half, (2 * (size_t) a.Lm * d + a.Lm) * sizeof(float), a.st>>>(
        a.qkv, a.lcf, a.off, out, aux, a.N, a.S0, a.H, a.scale, a.Lm);
  else
    bwd_split<d, S, R><<<grid, 2 * half, (4 * (size_t) a.Lm * d + 3 * a.Lm) * sizeof(float), a.st>>>(
        a.qkv, a.lcf, a.off, res, lse_in, grad, out, aux, a.N, a.S0, a.H, a.scale, a.Lm, half);
}

// S lanes a row, R rows a thread; false if not instantiated for d
template <int d>
bool split_launch(int S, int R, const Args& a, bool back, const float* res, const float* lse_in, const float* grad,
                  float* out, float* aux) {
  auto go = [&](auto s) {
    constexpr int SS = decltype(s)::value;
    if constexpr (d / SS >= 4) {
      if (R == 1) return split_launch_sr<d, SS, 1>(a, back, res, lse_in, grad, out, aux), true;
      if (R == 2) return split_launch_sr<d, SS, 2>(a, back, res, lse_in, grad, out, aux), true;
    }
    return false;
  };
  switch (S) {
    case 1: return go(std::integral_constant<int, 1>{});
    case 2: return go(std::integral_constant<int, 2>{});
    case 4: return go(std::integral_constant<int, 4>{});
    default: return false;
  }
}

template <int d, int NW, bool X3>
void mma_launch_nw(const Args& a, bool back, const float* res, const float* lse_in, const float* grad, float* out,
                float* aux) {
  constexpr int Lp = 16 * NW, ld = d + 4;
  const unsigned grid = (unsigned) a.N * (unsigned) a.H;
  if (!back)
    fwd_mma<d, NW, X3><<<grid, 32 * NW, (3 * (size_t) Lp * ld + Lp) * sizeof(float), a.st>>>(
        a.qkv, a.lcf, a.off, out, aux, a.N, a.S0, a.H, a.scale);
  else
    bwd_mma<d, NW, X3><<<grid, 64 * NW, (4 * (size_t) Lp * ld + 3 * Lp) * sizeof(float), a.st>>>(
        a.qkv, a.lcf, a.off, res, lse_in, grad, out, aux, a.N, a.S0, a.H, a.scale);
}

// head sizes 16 and 32, Lm <= 64: false if not available
template <int d>
bool mma_launch(bool x3, const Args& a, bool back, const float* res, const float* lse_in, const float* grad,
                float* out, float* aux) {
  if constexpr (d == 16 || d == 32) {
    if (a.Lm > kShortL) return false;
    switch ((a.Lm + 15) / 16) {
#define PET_MMA_CASE(nw)                                                                      \
  case nw:                                                                                    \
    (x3 ? mma_launch_nw<d, nw, true> : mma_launch_nw<d, nw, false>)(a, back, res, lse_in, grad, out, aux); \
    return true;
      PET_MMA_CASE(1)
      PET_MMA_CASE(2)
      PET_MMA_CASE(3)
      PET_MMA_CASE(4)
#undef PET_MMA_CASE
      default: return false;
    }
  }
  return false;
}

// PET_RAGGED_SPLIT=S[,R] (0: the tiled kernels for short sequences too)
int2 split_env() {
  const char* e = std::getenv("PET_RAGGED_SPLIT");
  int S = 1, R = 1;
  if (e && *e && std::sscanf(e, "%d,%d", &S, &R) < 1) S = 1, R = 1;
  return make_int2(S, R);
}
// 0 (off), 1 (TF32) or 3 (3xTF32); only while the GEMMs run in TF32
int mma_mode(int d) {
  if (gemm_mode() != GemmMode::TF32) return 0;
  const char* e = std::getenv("PET_RAGGED_MMA");
  const int v = e && *e ? std::atoi(e) : (d == 32 ? 3 : 0);
  return v == 1 || v == 3 ? v : 0;
}

template <int d>
void launch(const Args& a, bool back, const float* res, const float* lse_in, const float* grad, float* out,
            float* aux) {
  static const int2 sp = split_env();
  const int mm = mma_mode(d);
  if (a.N == 0) return;
  if constexpr (d <= 32) {
    if (a.Lm <= kShortL) {
      if (mm && mma_launch<d>(mm == 3, a, back, res, lse_in, grad, out, aux)) return;
      if (sp.x && split_launch<d>(sp.x, sp.y, a, back, res, lse_in, grad, out, aux)) return;
    }
  }
  constexpr int RF = 64 / d > 1 ? 64 / d : 1;  // forward: q, acc in registers
  constexpr int RB = 32 / d > 1 ? 32 / d : 1;  // backward: three row arrays
  if (!back) {
    const TiledShape s = tiled_shape(a.H, d, RF, a.Lm, false);
    fwd_tiled<d, RF><<<dim3((unsigned) a.N, (a.Lm + s.TP - 1) / s.TP), s.TP / RF * a.H, s.smem, a.st>>>(
        a.qkv, a.lcf, a.off, out, aux, a.N, a.S0, a.H, s.TP, s.TS, a.scale);
  } else {
    const TiledShape sq = tiled_shape(a.H, d, RB, a.Lm, false), sk = tiled_shape(a.H, d, RB, a.Lm, true);
    bwd_dq_tiled<d, RB><<<dim3((unsigned) a.N, (a.Lm + sq.TP - 1) / sq.TP), sq.TP / RB * a.H, sq.smem, a.st>>>(
        a.qkv, a.lcf, a.off, res, lse_in, grad, out, a.N, a.S0, a.H, sq.TP, sq.TS, a.scale);
    bwd_dkv_tiled<d, RB><<<dim3((unsigned) a.N, (a.Lm + sk.TP - 1) / sk.TP), sk.TP / RB * a.H, sk.smem, a.st>>>(
        a.qkv, a.lcf, a.off, res, lse_in, grad, out, aux, a.N, a.S0, a.H, sk.TP, sk.TS, a.scale);
  }
}

void dispatch(int d, const Args& a, bool back, const float* res, const float* lse_in, const float* grad, float* out,
              float* aux) {
  switch (d) {
    case 8: return launch<8>(a, back, res, lse_in, grad, out, aux);
    case 16: return launch<16>(a, back, res, lse_in, grad, out, aux);
    case 32: return launch<32>(a, back, res, lse_in, grad, out, aux);
    case 64: return launch<64>(a, back, res, lse_in, grad, out, aux);
    default: throw std::runtime_error("attention_ragged: head size " + std::to_string(d) + " not instantiated");
  }
}

// Rows [N + off(N), T) of a packed layout with an edge capacity belong to no
// atom: zeroed (w floats of a, hb of b), so they stay finite.
__global__ void zero_tail(const int* __restrict__ off, const int N, const long long T, float* __restrict__ a,
                          const int w, float* __restrict__ b, const int hb) {
  for (long long r = (long long) N + off[N] + blockIdx.x; r < T; r += gridDim.x) {
    for (int c = threadIdx.x; c < w; c += blockDim.x) a[r * w + c] = 0.0f;
    for (int c = threadIdx.x; c < hb; c += blockDim.x) b[r * hb + c] = 0.0f;
  }
}

// Launched on Kokkos's stream (so a recorded graph holds it), reported to Kokkos
// Tools as a kernel.
template <class F>
void run(const char* name, F&& f) {
  uint64_t id = 0;
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::beginParallelFor(name, 0, &id);
  f(ExecSpace().cuda_stream());
  if (Kokkos::Tools::profileLibraryLoaded()) Kokkos::Tools::endParallelFor(id);
}

void check_dense(const View2D& v, int cols, const char* what) {
  if ((int) v.extent(1) != cols || (v.extent(0) > 1 && (int) v.stride(0) != cols))
    throw std::runtime_error(std::string("attention_ragged: ") + what + " is not a dense [T, " +
                             std::to_string(cols) + "] array");
}

}  // namespace

void attention_ragged(Workspace& ws, const std::string& key, View2D out, View2D qkv, const TokenLayout& tl,
                      const WeightRef& w_out, View1D b_out, int H, int hd, double temperature, bool save) {
  const int D = H * hd, T = tl.T;
  const float sc = float(1.0 / (std::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", T, D) : ws.tmp(T, D);
  View2D lse = save ? ws.n2(key + ":lse", T, H) : ws.tmp(T, H);
  check_dense(qkv, 3 * D, "qkv");
  const Args a{qkv.data(), tl.lcf.data(), tl.off.data(), tl.N, tl.S0, H, tl.max_len, sc, ExecSpace().cuda_stream()};
  run("attn_ragged", [&](cudaStream_t st) {
    if (tl.tail)
      zero_tail<<<1024, 128, 0, st>>>(tl.off.data(), tl.N, T, merged.data(), D, lse.data(), H);
    dispatch(hd, a, false, nullptr, nullptr, nullptr, merged.data(), lse.data());
  });
  linear(out, merged, w_out, b_out);
}

void attention_ragged_bwd(Workspace& ws, const std::string& key, View2D in_adj, View1D cf_adj, View2D out_adj,
                          View2D qkv, const TokenLayout& tl, const WeightRef& w_in, const WeightRef& w_out, int H,
                          int hd, double temperature, Net beta) {
  const int D = H * hd, T = tl.T;
  const float sc = float(1.0 / (std::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(T, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(T, 3 * D), db = ws.tmp(T, H);
  View2D merged = ws.peek2(key + ":merged"), lse = ws.peek2(key + ":lse");
  check_dense(qkv, 3 * D, "qkv");
  const Args a{qkv.data(), tl.lcf.data(), tl.off.data(), tl.N, tl.S0, H, tl.max_len, sc, ExecSpace().cuda_stream()};
  run("attn_ragged_bwd", [&](cudaStream_t st) {
    if (tl.tail)
      zero_tail<<<1024, 128, 0, st>>>(tl.off.data(), tl.N, T, qkv_adj.data(), 3 * D, db.data(), H);
    dispatch(hd, a, true, merged.data(), lse.data(), dmerged.data(), qkv_adj.data(), db.data());
  });
  // Each edge's bias adjoint, summed over heads, into its cutoff factor's: the
  // bias is log(cf).
  auto off = tl.off, center = tl.center;
  auto cut = tl.cut;
  const int N = tl.N, S0 = tl.S0, E = tl.E;
  Kokkos::parallel_for(
      "attn_bwd_cf", RangePolicy(0, E), KOKKOS_LAMBDA(int e) {
        if (e >= off(N)) return;  // a capacity's dead row
        const Net cf = cut(e);
        if (cf <= Net(0)) return;
        const int n = center(e);
        const long long row = S0 ? (long long) n * S0 + 1 + (e - off(n)) : (long long) N + e;
        Net total = Net(0);
        for (int h = 0; h < H; ++h) total += db(row, h);
        cf_adj(e) += total / cf;
      });
  linear_bwd(in_adj, qkv_adj, w_in, beta);
}

#else  // no ragged kernels here: attention_ragged_supported() is false

void attention_ragged(Workspace&, const std::string&, View2D, View2D, const TokenLayout&, const WeightRef&, View1D,
                      int, int, double, bool) {
  throw std::runtime_error("attention_ragged: not built for this backend");
}
void attention_ragged_bwd(Workspace&, const std::string&, View2D, View1D, View2D, View2D, const TokenLayout&,
                          const WeightRef&, const WeightRef&, int, int, double, Net) {
  throw std::runtime_error("attention_ragged_bwd: not built for this backend");
}

#endif

// Each kept edge's bias, log(cf), -inf when cf <= 0 (a key no query sees).
View1D attention_log_cf(Workspace& ws, const PackedEdges& pk) {
  View1D lcf = ws.n1("attn:lcf", pk.E);
  auto cut = pk.cut;
  Kokkos::parallel_for(
      "attn_lcf", RangePolicy(0, pk.E), KOKKOS_LAMBDA(int e) {
        const Net c = cut(e);
        lcf(e) = c > Net(0) ? fast_log(c) : -Kokkos::Experimental::infinity_v<Net>;
      });
  return lcf;
}

}  // namespace pet
