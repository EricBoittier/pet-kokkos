// Multi-head attention over each atom's tokens, forward and backward.
//
// Atom n has S = M+1 tokens: its own (s = 0) and one per neighbour slot. Per
// head, query q attends to key k with weight softmax_k(scale * q.k + log cf_k),
// where cf is the key's smooth cutoff factor; a key with cf <= 0 (padding, or an
// edge exactly at its cutoff) is left out, and a padding query (s > 0, cf <= 0)
// produces zero.
//
// Each pass runs a team per (atom, head), a thread per token. The team first
// copies the rows every thread reads -- the keys and values, or in the keys pass
// the queries and output adjoints -- into scratch, so they are read from global
// memory once and not by each thread; then each thread holds its own row and
// accumulator in registers, sized exactly by a compile-time head_dim (16, 32,
// 48, 64 and 80, the sizes the published models use; anything else takes a
// generic path sized kMaxHeadDim).
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
// multiple of 4 (head_dim is, and so are the widths), in buffers the pool and
// the scratch align, so the load is one 16-byte instruction rather than four.
// The loops are limited by load instructions, not arithmetic, and the compiler
// cannot see the alignment to do this itself.
struct alignas(4 * sizeof(Net)) Net4 {
  Net v[4];
};
KOKKOS_INLINE_FUNCTION Net4 ld4(const Net* p) { return *reinterpret_cast<const Net4*>(p); }
// Between a row's two uses in the dK pass: read it again rather than keep it in
// registers.
KOKKOS_INLINE_FUNCTION void reread() { asm volatile("" ::: "memory"); }

constexpr Net kOff = -Net(1e30);  // the log cf of a key no query sees
KOKKOS_INLINE_FUNCTION bool off(Net lc) { return lc < -Net(1e29); }

// The passes hold several head-sized arrays per thread. On HIP, bounding the
// team at 256 threads lets the compiler give each thread the registers for them:
// under the default bound of 1024 an MI250X GCD caps a thread at 128 and they
// spill (a pbe0-pet relaxation step 1.9x slower). On CUDA the same bound lets
// nvcc spend registers that occupancy needs more.
#if defined(KOKKOS_ENABLE_HIP)
using AttnPolicy = Kokkos::TeamPolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                      Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
#else
using AttnPolicy = TeamPolicy;
#endif
using Member = AttnPolicy::member_type;

// A team per (atom, head), whole warps (wavefronts) enough for its S tokens, and
// `values` Nets of scratch: fast scratch where they fit, else global (level 1).
struct Teams {
  AttnPolicy policy;
  int level;
  std::size_t bytes;
};
Teams teams(int N, int H, int S, std::size_t values) {
  const std::size_t bytes = sizeof(Net) * values + 16;
  const int T = std::min(256, (S + kLanes - 1) / kLanes * kLanes);
  const int level = bytes <= std::size_t(AttnPolicy::scratch_size_max(0)) ? 0 : 1;
  return {AttnPolicy(N * H, T).set_scratch_size(level, Kokkos::PerTeam(bytes)), level, bytes};
}
KOKKOS_INLINE_FUNCTION Net* scratch(const Member& m, int level, std::size_t bytes) {
  return (Net*) m.team_scratch(level).get_shmem_aligned(bytes - 16, 16);
}

// Rows n*S .. n*S+S-1 of src, columns [col, col+hd), into dst [S][hd].
template <int HD>
KOKKOS_INLINE_FUNCTION void stage(const Member& m, Net* dst, const View2D& src, int n, int S, int col, int hd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * (HD / 4)), [&](int t) {
      const int r = t / (HD / 4), c = (t % (HD / 4)) * 4;
      *reinterpret_cast<Net4*>(dst + r * HD + c) = ld4(&src(n * S + r, col + c));
    });
  } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S * hd), [&](int t) {
      dst[t] = src(n * S + t / hd, col + t % hd);
    });
  }
}
// A thread's own slice of a row into registers.
template <int HD>
KOKKOS_INLINE_FUNCTION void load_row(Net* reg, const Net* row, int nd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      for (int j = 0; j < 4; ++j) reg[d + j] = x.v[j];
    }
  } else {
    for (int d = 0; d < nd; ++d) reg[d] = row[d];
  }
}
// Each token's log cf, kOff for a key no query sees.
KOKKOS_INLINE_FUNCTION void stage_log_cf(const Member& m, Net* dst, const View2D& cf_seq, int n, int S) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
    const Net cf = cf_seq(n, r);
    dst[r] = cf > Net(0) ? fast_log(cf) : kOff;
  });
}

// row . reg over a head's slice: row (in memory) four at a time where head_dim
// allows, reg a thread's own registers.
template <int HD>
KOKKOS_INLINE_FUNCTION Net dot_row(const Net* row, const Net* reg, int nd) {
  Net s = Net(0);
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      s += x.v[0] * reg[d] + x.v[1] * reg[d + 1] + x.v[2] * reg[d + 2] + x.v[3] * reg[d + 3];
    }
  } else {
    for (int d = 0; d < nd; ++d) s += row[d] * reg[d];
  }
  return s;
}
// acc = acc * c + p * row, over a head's slice.
template <int HD>
KOKKOS_INLINE_FUNCTION void axpby_row(Net* acc, Net c, Net p, const Net* row, int nd) {
  if constexpr (HD > 0 && HD % 4 == 0) {
    for (int d = 0; d < HD; d += 4) {
      const Net4 x = ld4(row + d);
      for (int j = 0; j < 4; ++j) acc[d + j] = acc[d + j] * c + p * x.v[j];
    }
  } else {
    for (int d = 0; d < nd; ++d) acc[d] = acc[d] * c + p * row[d];
  }
}

// Where a pass reads the rows its threads share. On CUDA a team per (atom,
// head) copies them into scratch first (staged), so each is read from global
// memory once rather than by every thread: on an RTX 4080 the passes 15-35%
// faster. An MI250X CU has 64 KiB of LDS, which holds six such teams of one
// wavefront -- too few -- so there (and on the host) a flat range of threads
// reads them through L1 instead.
#if defined(KOKKOS_ENABLE_CUDA)
constexpr bool kStaged = true;
#else
constexpr bool kStaged = false;
#endif
#if defined(KOKKOS_ENABLE_HIP)
using FlatPolicy = Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<256, 1>,
                                       Kokkos::Experimental::WorkItemProperty::HintLightWeight_t>;
#else
using FlatPolicy = RangePolicy;
#endif

// A head's slice of S consecutive rows, in scratch or in a global array.
struct Rows {
  const Net* base;
  std::size_t stride;
  KOKKOS_INLINE_FUNCTION const Net* operator()(int r) const { return base + r * stride; }
};
KOKKOS_INLINE_FUNCTION Rows rows(const View2D& v, int row0, int col) { return {&v(row0, col), v.stride(0)}; }
// Each key's log cf: staged, or from the cutoff factors.
struct LogCf {
  const Net* staged;
  const Net* cf;
  KOKKOS_INLINE_FUNCTION Net operator()(int k) const {
    if (staged) return staged[k];
    return cf[k] > Net(0) ? fast_log(cf[k]) : kOff;
  }
};

// Runs pass f over every (atom, head, token): staged teams, or a flat range.
template <class F>
void launch(const char* name, F f, int N, int H, int S, std::size_t scratch_values) {
  if constexpr (kStaged) {
    const Teams t = teams(N, H, S, scratch_values);
    f.level = t.level, f.bytes = t.bytes;
    Kokkos::parallel_for(name, t.policy, f);
  } else {
    Kokkos::parallel_for(name, FlatPolicy(0, N * H * S), f);
  }
}

// The forward: online softmax, one pass over the keys. Saves the output
// (key:merged) and each query's running max and 1/sum (key:ml) when `save`.
template <int HD>
struct Forward {
  View2D qkv, cf_seq, merged, ml;
  int S, H, hd;
  Net sc;
  bool save;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd;
    const int row_q = n * S + sq, si = (n * H + h) * S + sq;
    if (sq > 0 && off(lcf(sq))) {  // padding query
      for (int d = 0; d < nd; ++d) merged(row_q, qo + d) = Net(0);
      if (save) ml(si, 0) = kOff, ml(si, 1) = Net(0);
      return;
    }
    Net q[CAP], acc[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    for (int d = 0; d < nd; ++d) acc[d] = Net(0);
    Net mx = kOff, l = Net(0);
    for (int k = 0; k < S; ++k) {
      const Net lc = lcf(k);
      if (off(lc)) continue;
      const Net s = dot_row<HD>(K(k), q, nd) * sc + lc;
      const Net nm = s > mx ? s : mx;
      const Net c = fast_exp(mx - nm), p = fast_exp(s - nm);
      l = l * c + p;
      axpby_row<HD>(acc, c, p, V(k), nd);
      mx = nm;
    }
    const Net invl = l > Net(0) ? Net(1) / l : Net(0);
    for (int d = 0; d < nd; ++d) merged(row_q, qo + d) = acc[d] * invl;
    if (save) ml(si, 0) = mx, ml(si, 1) = invl;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd, D = H * hd;
    Net* sk = scratch(m, level, bytes);
    Net* sv = sk + S * nd;
    Net* slc = sv + S * nd;
    stage<HD>(m, sk, qkv, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, Rows{sk, std::size_t(nd)}, Rows{sv, std::size_t(nd)}, LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, n * S, D + h * hd), rows(qkv, n * S, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
  }
};

template <int HD>
void attention_impl(Workspace& ws, const std::string& key, View2D out, View2D qkv, View2D cf_seq,
                    const WeightRef& w_out, View1D b_out, int N, int S, int H, int hd,
                    double temperature, bool save) {
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D merged = save ? ws.n2(key + ":merged", N * S, D) : ws.tmp(N * S, D);
  View2D ml = save ? ws.n2(key + ":ml", N * H * S, 2) : View2D();
  launch("attention", Forward<HD>{qkv, cf_seq, merged, ml, S, H, hd, sc, save}, N, H, S, 2 * std::size_t(S) * hd + S);
  linear(out, merged, w_out, b_out);
}

// The backward, as the forward runs: nothing S x S is ever stored. With
// A = softmax weights (recomputed from the saved max m and 1/sum per query) and
// dS = A (dO.v - dO.out), the score adjoint before the scale:
//   queries, per (atom, head, query): dQ = sc sum_k dS K, and dO.out (Dq), kept;
//   keys, per (atom, head, key): dK = sc sum_q dS Q, dV = sum_q A dO, and the
//     key's dS summed over queries, the cf adjoint's share from this head;
//   cf, per (atom, key): those shares, summed over heads.
// Each pass recomputes q.k and dO.v for its pairs. Every sum has one owner and a
// fixed order: nothing is atomic.

template <int HD>
struct Queries {
  View2D qkv, dmerged, merged, ml, dq_out, cf_seq, qkv_adj;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sq, Rows K, Rows V, LogCf lcf) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, qo = h * hd;
    const int row_q = n * S + sq, si = (n * H + h) * S + sq;
    const Net m = ml(si, 0), invl = ml(si, 1);
    Net q[CAP], dout[CAP], dq[CAP];
    load_row<HD>(q, &qkv(row_q, qo), nd);
    load_row<HD>(dout, &dmerged(row_q, qo), nd);
    const Net Dq = dot_row<HD>(&merged(row_q, qo), dout, nd);
    for (int d = 0; d < nd; ++d) dq[d] = Net(0);
    dq_out(si, 0) = Dq;
    if (invl > Net(0))  // else a padding query: no weights, no adjoints
      for (int k = 0; k < S; ++k) {
        const Net lc = lcf(k);
        if (off(lc)) continue;  // padding key
        const Net a = fast_exp(dot_row<HD>(K(k), q, nd) * sc + lc - m) * invl;
        const Net ds = a * (dot_row<HD>(V(k), dout, nd) - Dq) * sc;
        axpby_row<HD>(dq, Net(1), ds, K(k), nd);
      }
    for (int d = 0; d < nd; ++d) qkv_adj(row_q, qo + d) = dq[d];
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd, D = H * hd;
    Net* sk = scratch(m, level, bytes);
    Net* sv = sk + S * nd;
    Net* slc = sv + S * nd;
    stage<HD>(m, sk, qkv, n, S, D + h * hd, nd);
    stage<HD>(m, sv, qkv, n, S, 2 * D + h * hd, nd);
    stage_log_cf(m, slc, cf_seq, n, S);
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sq) {
      token(n, h, sq, Rows{sk, std::size_t(nd)}, Rows{sv, std::size_t(nd)}, LogCf{slc, nullptr});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sq = i % S, h = (i / S) % H, n = i / (S * H), D = H * hd;
    token(n, h, sq, rows(qkv, n * S, D + h * hd), rows(qkv, n * S, 2 * D + h * hd), LogCf{nullptr, &cf_seq(n, 0)});
  }
};

// The keys pass: dK and the key's cf share (DK), and dV (DV), from every query
// of its atom and head (their rows, output adjoints, max, 1/sum and Dq).
template <int HD, bool DK, bool DV>
struct Keys {
  View2D qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh;
  int S, H, hd;
  Net sc;
  int level = 0;
  std::size_t bytes = 0;

  // Per query: max, 1/sum and Dq, each at its stride.
  struct Stats {
    const Net *m, *l, *d;
    std::size_t sm, sd;
  };
  KOKKOS_INLINE_FUNCTION void token(int n, int h, int sk, Rows Q, Rows G, Stats st) const {
    constexpr int CAP = HD > 0 ? HD : kMaxHeadDim;
    const int nd = HD > 0 ? HD : hd, D = H * hd, ko = D + h * hd, vo = 2 * D + h * hd;
    const int row_k = n * S + sk;
    Net k[CAP], v[DK ? CAP : 1], dk[DK ? CAP : 1], dv[DV ? CAP : 1];
    Net cf_acc = Net(0);
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) dk[d] = Net(0);
      if constexpr (DV) dv[d] = Net(0);
    }
    const Net cf = cf_seq(n, sk);
    if (cf > Net(0)) {  // else a padding key: no query sees it
      const Net lcf = fast_log(cf);
      load_row<HD>(k, &qkv(row_k, ko), nd);
      if constexpr (DK) load_row<HD>(v, &qkv(row_k, vo), nd);
      for (int q = 0; q < S; ++q) {
        const Net invl = st.l[q * st.sm];
        if (invl <= Net(0)) continue;  // padding query
        const Net a = fast_exp(dot_row<HD>(Q(q), k, nd) * sc + lcf - st.m[q * st.sm]) * invl;
        Net ds = Net(0);
        if constexpr (DK) ds = a * (dot_row<HD>(G(q), v, nd) - st.d[q * st.sd]);
        if constexpr (!DV) reread();
        if constexpr (DK) axpby_row<HD>(dk, Net(1), ds * sc, Q(q), nd);
        if constexpr (DV) axpby_row<HD>(dv, Net(1), a, G(q), nd);
        cf_acc += ds;
      }
    }
    for (int d = 0; d < nd; ++d) {
      if constexpr (DK) qkv_adj(row_k, ko + d) = dk[d];
      if constexpr (DV) qkv_adj(row_k, vo + d) = dv[d];
    }
    if constexpr (DK) cfh(row_k, h) = cf_acc;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const int n = m.league_rank() / H, h = m.league_rank() % H, nd = HD > 0 ? HD : hd;
    Net* sq_ = scratch(m, level, bytes);
    Net* sg = sq_ + S * nd;
    Net* sm = sg + S * nd;
    Net* sl = sm + S;
    Net* sd = sl + S;
    stage<HD>(m, sq_, qkv, n, S, h * hd, nd);
    stage<HD>(m, sg, dmerged, n, S, h * hd, nd);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int r) {
      const int si = (n * H + h) * S + r;
      sm[r] = ml(si, 0), sl[r] = ml(si, 1), sd[r] = dq_out(si, 0);
    });
    m.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(m, S), [&](int sk) {
      token(n, h, sk, Rows{sq_, std::size_t(nd)}, Rows{sg, std::size_t(nd)}, Stats{sm, sl, sd, 1, 1});
    });
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    const int sk = i % S, h = (i / S) % H, n = i / (S * H), b = (n * H + h) * S;
    token(n, h, sk, rows(qkv, n * S, h * hd), rows(dmerged, n * S, h * hd),
          Stats{&ml(b, 0), &ml(b, 1), &dq_out(b, 0), ml.stride(0), dq_out.stride(0)});
  }
};

template <int HD>
void attention_bwd_impl(Workspace& ws, const std::string& key, View2D in_adj, View2D cf_seq_adj,
                        View2D out_adj, View2D qkv, View2D cf_seq, const WeightRef& w_in,
                        const WeightRef& w_out, int N, int S, int H, int hd, double temperature,
                        Net beta) {
  const int D = H * hd;
  const Net sc = Net(1.0 / (Kokkos::sqrt((double) hd) * temperature));
  Workspace::Scope scope(ws);
  View2D dmerged = ws.tmp(N * S, D);
  linear_bwd(dmerged, out_adj, w_out, 0);
  View2D qkv_adj = ws.tmp(N * S, 3 * D);
  View2D cfh = ws.tmp(N * S, H);  // per key and head: the cf adjoint's share
  View2D dq_out = ws.tmp(N * H * S, 1);  // per query and head: dO.out
  View2D merged = ws.peek2(key + ":merged"), ml = ws.peek2(key + ":ml");

  launch("attn_bwd_queries", Queries<HD>{qkv, dmerged, merged, ml, dq_out, cf_seq, qkv_adj, S, H, hd, sc}, N, H, S,
         2 * std::size_t(S) * hd + S);

  // From head_dim 32, one keys pass holding k, v, dk and dv takes 374 registers
  // on an MI250X GCD and runs one wave per SIMD; as two passes (dK with the cf
  // share, then dV) they run two and four, the keys' share of a pet-mols-s step
  // 40% less. At 16 the one pass already runs two, and on NVIDIA it fits.
#if defined(KOKKOS_ENABLE_HIP)
  constexpr bool split = HD >= 32;
#else
  constexpr bool split = false;
#endif
  const std::size_t kv = 2 * std::size_t(S) * hd + 3 * S;
  auto keys = [&](auto tag) { return decltype(tag){qkv, dmerged, ml, dq_out, cf_seq, qkv_adj, cfh, S, H, hd, sc}; };
  if constexpr (split) {
    launch("attn_bwd_keys", keys(Keys<HD, true, false>{}), N, H, S, kv);
    launch("attn_bwd_values", keys(Keys<HD, false, true>{}), N, H, S, kv);
  } else {
    launch("attn_bwd_keys", keys(Keys<HD, true, true>{}), N, H, S, kv);
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
