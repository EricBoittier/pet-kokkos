// Multi-head attention over each atom's tokens, forward and backward.
//
// Atom n has S = M+1 tokens: its own (s = 0) and one per neighbour slot. Per
// head, query q attends to key k with weight softmax_k(scale * q.k + log cf_k),
// where cf is the key's smooth cutoff factor; a key with cf <= 0 (padding, or an
// edge exactly at its cutoff) is left out, and a padding query (s > 0, cf <= 0)
// produces zero.
//
// Thread order: one thread per (atom, head, query), query fastest. A warp is
// then 32 queries of the same atom and head, which read the same key and value
// at the same time -- one broadcast from L1.
//
// Each thread holds q and its output accumulator in registers, sized exactly by
// a compile-time head_dim (16, 32, 48, 64 and 80, the sizes the published models
// use; anything else takes a generic path sized kMaxHeadDim).
#include "ops.hpp"

#include <algorithm>
#include <type_traits>

namespace pet {

namespace {

// Call f with head_dim as a compile-time constant, 0 meaning "generic".
template <class F>
void with_head_dim(int hd, F&& f) {
  switch (hd) {
    case 16: return f(std::integral_constant<int, 16>{});
    case 32: return f(std::integral_constant<int, 32>{});
    case 48: return f(std::integral_constant<int, 48>{});
    case 64: return f(std::integral_constant<int, 64>{});
    case 80: return f(std::integral_constant<int, 80>{});
    default: return f(std::integral_constant<int, 0>{});
  }
}

// Four consecutive values as one load: a head's slice of a row starts on a
// multiple of 4 (head_dim is, and so are the widths), in buffers the pool
// aligns, so the load is one 16-byte instruction rather than four. The loops are
// limited by load instructions, not arithmetic.
struct alignas(4 * sizeof(Net)) Net4 {
  Net v[4];
};
KOKKOS_INLINE_FUNCTION Net4 ld4(const Net* p) { return *reinterpret_cast<const Net4*>(p); }
// Between a row's two uses in the keys pass: read it again (from L1) rather
// than keep it in registers.
KOKKOS_INLINE_FUNCTION void reread() { asm volatile("" ::: "memory"); }

// Online softmax, one pass over the keys. Saves the output (key:merged) and each
// query's running max and 1/sum (key:ml) when `save`.
template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
                    const WeightRef& w_out, View1D b_out, int N, int S, int H, int hd,
                    double temperature, bool save) {
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", N * S, D) : ws.tmp(N * S, D);
  View2D ml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  Kokkos::parallel_for(
      "attention", RangePolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
        const int sq = i % S, h = (i / S) % H, n = i / (S * H);
        const int nd = HD > 0 ? HD : hd;
        const int row_q = n * S + sq, si = (n * H + h) * S + sq;
        if (sq > 0 && cf_seq(n, sq) <= Net(0)) {  // padding query
          for (int d = 0; d < nd; ++d) merged(row_q, h * hd + d) = Net(0);
          if (save) ml(si, 0) = -Net(1e30), ml(si, 1) = Net(0);
          return;
        }
        const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
        Net q[CAP], acc[CAP];
        for (int d = 0; d < nd; ++d) q[d] = qkv(row_q, qo + d), acc[d] = Net(0);
        Net m = -Net(1e30), l = Net(0);
        for (int sk = 0; sk < S; ++sk) {
          const Net cf = cf_seq(n, sk);
          if (cf <= Net(0)) continue;
          const int row_k = n * S + sk;
          const Net* kr = &qkv(row_k, ko);
          const Net* vr = &qkv(row_k, vo);
          Net dot = Net(0);
          if constexpr (HD > 0 && HD % 4 == 0) {
            for (int d = 0; d < HD; d += 4) {
              const Net4 k4 = ld4(kr + d);
              dot += q[d] * k4.v[0] + q[d + 1] * k4.v[1] + q[d + 2] * k4.v[2] + q[d + 3] * k4.v[3];
            }
          } else {
            for (int d = 0; d < nd; ++d) dot += q[d] * kr[d];
          }
          const Net s = dot * sc + fast_log(cf);
          const Net new_m = (s > m) ? s : m;
          const Net c = fast_exp(m - new_m), p = fast_exp(s - new_m);
          l = l * c + p;
          if constexpr (HD > 0 && HD % 4 == 0) {
            for (int d = 0; d < HD; d += 4) {
              const Net4 v4 = ld4(vr + d);
              for (int j = 0; j < 4; ++j) acc[d + j] = acc[d + j] * c + p * v4.v[j];
            }
          } else {
            for (int d = 0; d < nd; ++d) acc[d] = acc[d] * c + p * vr[d];
          }
          m = new_m;
        }
        const Net invl = (l > Net(0)) ? Net(1) / l : Net(0);
        for (int d = 0; d < nd; ++d) merged(row_q, h * hd + d) = acc[d] * invl;
        if (save) ml(si, 0) = m, ml(si, 1) = invl;
      });
  linear(out, merged, w_out, b_out);
}

// The backward, as the forward runs: nothing S x S is ever stored. With
// A = softmax weights (recomputed from the saved max m and 1/sum per query) and
// dS = A (dO.v - dO.out), the score adjoint before the scale:
//   queries, per (atom, head, query): dQ = sc sum_k dS K, and dO.out (Dq), kept;
//   keys, per (atom, head, key): dK = sc sum_q dS Q, dV = sum_q A dO, and the
//     key's dS summed over queries, the cf adjoint's share from this head;
//   cf, per (atom, key): those shares, summed over heads.
// Each pass recomputes q.k and dO.v for its pairs, in registers; a warp is
// consecutive queries (or keys) of one atom and head, so the rows it reads are
// broadcasts. Every sum has one owner and a fixed order: nothing is atomic.
// The passes hold several head-sized arrays per thread. Bounding the block at
// 256 threads lets the compiler give each thread the registers for them: under
// HIP's default bound of 1024 an MI250X GCD caps a thread at 128 and they spill
// (a pbe0-pet relaxation step 1.9x slower).
using BwdPolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                      Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;


// The keys pass, per (atom, head, key): dK and the key's cf share (DK), and dV
// (DV), from every query of its atom and head.
template <int HD, bool DK, bool DV>
struct KeysPass {
  View2D qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh;
  int S, H, hd, D;
  Net sc;

  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    constexpr bool vec = HD > 0 && HD % 4 == 0;
    const int sk = i % S, h = (i / S) % H, n = i / (S * H);
    const int nd = HD > 0 ? HD : hd;
    const int row_k = n * S + sk;
    const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
    Net k[CAP], v[DK ? CAP : 1], dk[DK ? CAP : 1], dv[DV ? CAP : 1];
    Net cf_acc = Net(0);
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) dk[d] = Net(0);
      if constexpr (DV) dv[d] = Net(0);
    }
    const Net cf = cf_seq(n, sk);
    if (cf > Net(0)) {  // else a padding key: no query sees it
      const Net lcf = fast_log(cf);
      for (int d = 0; d < nd; ++d) {
        k[d] = qkv(row_k, ko + d);
        if constexpr (DK) v[d] = qkv(row_k, vo + d);
      }
      for (int sq = 0; sq < S; ++sq) {
        const int si = (n * H + h) * S + sq, row_q = n * S + sq;
        const Net invl = ml(si, 1);
        if (invl <= Net(0)) continue;  // padding query
        const Net* qr = &qkv(row_q, qo);
        const Net* gr = &dmerged(row_q, qo);
        Net dot = Net(0), dA = Net(0);
        if constexpr (vec) {
          for (int d = 0; d < HD; d += 4) {
            const Net4 q4 = ld4(qr + d);
            for (int j = 0; j < 4; ++j) dot += q4.v[j] * k[d + j];
            if constexpr (DK) {
              const Net4 g4 = ld4(gr + d);
              for (int j = 0; j < 4; ++j) dA += g4.v[j] * v[d + j];
            }
          }
        } else {
          for (int d = 0; d < nd; ++d) {
            dot += qr[d] * k[d];
            if constexpr (DK) dA += gr[d] * v[d];
          }
        }
        const Net a = fast_exp(dot * sc + lcf - ml(si, 0)) * invl;
        const Net ds = DK ? a * (dA - dq_out(si, 0)) : Net(0);
        reread();
        if constexpr (vec) {
          for (int d = 0; d < HD; d += 4) {
            if constexpr (DK) {
              const Net4 q4 = ld4(qr + d);
              for (int j = 0; j < 4; ++j) dk[d + j] += ds * sc * q4.v[j];
            }
            if constexpr (DV) {
              const Net4 g4 = ld4(gr + d);
              for (int j = 0; j < 4; ++j) dv[d + j] += a * g4.v[j];
            }
          }
        } else {
          for (int d = 0; d < nd; ++d) {
            if constexpr (DK) dk[d] += ds * sc * qr[d];
            if constexpr (DV) dv[d] += a * gr[d];
          }
        }
        cf_acc += ds;
      }
    }
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) qkv_adj(row_k, ko + d) = dk[d];
      if constexpr (DV) qkv_adj(row_k, vo + d) = dv[d];
    }
    if constexpr (DK) cfh(row_k, h) = cf_acc;
  }
};

template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                        View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                        const WeightRef& w_out, int N, int S, int H, int hd, double temperature,
                        Net beta) {
  constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(N * S, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(N * S, 3 * D);
  View2D cfh = ws.tmp(N * S, H);  // per key and head: the cf adjoint's share
  View2D dq_out = ws.tmp(N * H * S, 1);  // per query and head: dO.out
  View2D merged = ws.peek2(key + ":merged"), ml = ws.peek2(key + ":ml");

  Kokkos::parallel_for(
      "attn_bwd_queries", BwdPolicy(0, N * H * S), KOKKOS_LAMBDA(int i) {
        const int sq = i % S, h = (i / S) % H, n = i / (S * H);
        const int nd = HD > 0 ? HD : hd;
        const int row_q = n * S + sq;
        const int qo = h * hd, ko = D + h * hd, vo = 2 * D + h * hd;
        const Net m = ml(i, 0), invl = ml(i, 1);
        Net q[CAP], dout[CAP], dq[CAP];
        Net Dq = Net(0);
        for (int d = 0; d < nd; ++d) {
          q[d] = qkv(row_q, qo + d);
          dout[d] = dmerged(row_q, qo + d);
          Dq += dout[d] * merged(row_q, qo + d);
          dq[d] = Net(0);
        }
        dq_out(i, 0) = Dq;
        if (invl > Net(0)) {  // else a padding query: no weights, no adjoints
          for (int sk = 0; sk < S; ++sk) {
            const Net cf = cf_seq(n, sk);
            if (cf <= Net(0)) continue;  // padding key
            const int row_k = n * S + sk;
            const Net* kr = &qkv(row_k, ko);
            const Net* vr = &qkv(row_k, vo);
            Net dot = Net(0), dA = Net(0);
            if constexpr (HD > 0 && HD % 4 == 0) {
              for (int d = 0; d < HD; d += 4) {
                const Net4 k4 = ld4(kr + d), v4 = ld4(vr + d);
                for (int j = 0; j < 4; ++j) dot += q[d + j] * k4.v[j], dA += dout[d + j] * v4.v[j];
              }
            } else {
              for (int d = 0; d < nd; ++d) dot += q[d] * kr[d], dA += dout[d] * vr[d];
            }
            const Net a = fast_exp(dot * sc + fast_log(cf) - m) * invl;
            const Net ds = a * (dA - Dq) * sc;
            if constexpr (HD > 0 && HD % 4 == 0) {
              for (int d = 0; d < HD; d += 4) {
                const Net4 k4 = ld4(kr + d);
                for (int j = 0; j < 4; ++j) dq[d + j] += ds * k4.v[j];
              }
            } else {
              for (int d = 0; d < nd; ++d) dq[d] += ds * kr[d];
            }
          }
        }
        for (int d = 0; d < nd; ++d) qkv_adj(row_q, qo + d) = dq[d];
      });

  // From head_dim 32, one pass holding k, v, dk and dv takes 374 registers on an
  // MI250X GCD and runs one wave per SIMD; as two passes (dK with the cf share,
  // then dV) they run two and four, the keys' share of a pet-mols-s step 40%
  // less. At 16 the one pass already runs two, and on NVIDIA it fits.
#if defined(KOKKOS_ENABLE_HIP)
  constexpr bool split = HD >= 32;
#else
  constexpr bool split = false;
#endif
  auto keys = [&](auto tag) { return decltype(tag){qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh, S, H, hd, D, sc}; };
  if constexpr (split) {
    Kokkos::parallel_for("attn_bwd_keys", BwdPolicy(0, N * H * S), keys(KeysPass<HD, true, false>{}));
    Kokkos::parallel_for("attn_bwd_values", BwdPolicy(0, N * H * S), keys(KeysPass<HD, false, true>{}));
  } else {
    Kokkos::parallel_for("attn_bwd_keys", BwdPolicy(0, N * H * S), keys(KeysPass<HD, true, true>{}));
  }

  Kokkos::parallel_for(
      "attn_bwd_cf", RangePolicy(0, N * S), KOKKOS_LAMBDA(int i) {
        const int sk = i % S, n = i / S;
        const Net cf = cf_seq(n, sk);
        if (cf <= Net(0)) return;
        Net total = Net(0);
        for (int h = 0; h < H; ++h) total += cfh(n * S + sk, h);
        cf_seq_adj(n, sk) += total / cf;
      });
  linear_bwd(in_adj, qkv_adj, w_in, beta);
}

}  // namespace

void attention(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
               const WeightRef& w_out, View1D b_out, int N, int S, int heads, int head_dim,
               double temperature, bool save) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_impl<decltype(hd)::value>(ws, key, out, qkv, cf_seq, w_out, b_out, N, S, heads,
                                        head_dim, temperature, save);
  });
}

void attention_bwd(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                   View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                   const WeightRef& w_out, int N, int S, int heads, int head_dim,
                   double temperature, Net beta) {
  with_head_dim(head_dim, [&](auto hd) {
    attention_bwd_impl<decltype(hd)::value>(ws, key, in_adj, cf_seq_adj, out_adj, qkv, cf_seq,
                                            w_in, w_out, N, S, heads, head_dim, temperature, beta);
  });
}

}  // namespace pet
