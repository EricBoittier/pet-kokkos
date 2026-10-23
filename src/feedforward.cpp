// The feedforward featurizer (PreLN, RMSNorm, SwiGLU, expanded central token):
// one GNN layer forward and backward, and the whole evaluation.
//
// A GNN layer:
//   et        = compress(input_edge, geometry, species)          [E, D]
//   A blocks  of centre contraction, attention over each atom's tokens [node; et],
//             centre expansion + MLP, edge residual + MLP        -> node, out_edge
//   input_edge += out_edge + comb_mlp(layernorm([out_edge | out_edge reversed]))
// The backward walks the same steps in reverse, turning node_adj and edge_adj --
// the adjoints of a step's outputs -- into those of its inputs, in place: a
// residual connection is then just leaving the value where it is.
#include "ops.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace pet {

namespace {
// Rows [r0, r0 + n) of a row-major array, as an array of its own.
View2D rows(const View2D& v, int r0, int n) { return View2D(v.data() + std::size_t(r0) * v.extent(1), n, v.extent(1)); }
}  // namespace

// The tokens of a block, three ways (ff_pass picks):
//   padded, the attention kernels in attention.cpp: [N S, D], atom n's tokens at
//     rows n S .. n S + M, its central token first, then its kept edges, then zeros;
//   padded, the ragged kernels (tl->S0 = S): the same rows, the padding skipped;
//   packed (tl->S0 = 0): [N + E, D], the central tokens, then the packed edges --
//     no padding row anywhere, and the two halves are the node and edge arrays
//     themselves: the centre contraction writes the central rows, the compress
//     MLP the edge rows, the attention output's halves feed the centre expansion
//     and the edge residual as they are, and the residual sum eps = attention +
//     edge_in is taken by the edge MLP's norm on its way through.
void PetModel::ff_layer(const DeviceEdgeData& dev, const PackedEdges& pk, int L, View2D& node,
                        View2D input_edge, View2D cond, LayerSaves* sav, bool save_wide, View2D remote_in,
                        bool share, const TokenLayout* tl) {
  const int N = dev.centres(), S = dev.max_neighbors + 1, E = pk.E;
  const int D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  const bool save = sav != nullptr, packed = tl && tl->S0 == 0;
  const int T = packed ? N + E : N * S;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  auto off = pk.off, rev = pk.reverse;
  if (save)
    for (auto* v : {&sav->tokens, &sav->qkv, &sav->node_new, &sav->tmp_center, &sav->eps, &sav->tmp_edge})
      v->assign(A, View2D());
  // A saved activation: named, per block, under the layer's tag.
  auto keep = [&](View2D& slot, const std::string& name, int a, int r, int c) {
    return slot = ws_.n2("s_" + name + (a < 0 ? "" : std::to_string(a)) + "_" + sav->tag, r, c);
  };
  Workspace::Scope layer_scope(ws_);

  // Packed: block 0's tokens are taken first, so that et lands in their edge rows.
  auto tokens_of = [&](int a) { return save ? keep(sav->tokens[a], "tok", a, T, D) : ws_.tmp(T, D); };
  View2D tokens0 = packed ? tokens_of(0) : View2D();
  View2D cpre = ws_.tmp(E, D), et = packed ? rows(tokens0, N, E) : ws_.tmp(E, D);
  compress_fwd(cpre, save ? keep(sav->cpre, "cpre", -1, E, D) : View2D(), compress_fold(L), input_edge, pk);
  linear(et, cpre, mat(g + ".compress.2.weight"), vec(g + ".compress.2.bias"));

  View2D node_cur = node, edge_cur = et;
  for (int a = 0; a < A; ++a) {
    const std::string tl_ = g + ".trans.layers." + std::to_string(a);
    const std::string key = "attn_" + (save ? sav->tag : ls) + "_" + std::to_string(a);
    const WeightRef w_in = mat(tl_ + ".attention.input_linear.weight"), w_out = mat(tl_ + ".attention.output_linear.weight");
    const View1D &b_in = vec(tl_ + ".attention.input_linear.bias"), &b_out = vec(tl_ + ".attention.output_linear.bias");
    // The node output leaves the layer, so it is named: two names, alternating
    // block to block. The edge output lives in the layer's scope. Packed, the
    // attention's output [N + E, D] does too: its edge rows become eps.
    View2D node_next = ws_.n2("node_new_" + std::to_string((L * A + a) % 2), N, Dn);
    View2D attn_out = packed ? (save ? keep(sav->eps[a], "eps", a, T, D) : ws_.tmp(T, D)) : View2D();
    View2D eps = packed ? rows(attn_out, N, E) : View2D();
    if (packed && save) sav->eps[a] = eps;
    // The block's edge output is eps + mlp(norm(eps)), accumulated onto a copy of
    // eps -- or, when nothing is saved, onto eps itself.
    View2D edge_next = packed && !save ? eps : ws_.tmp(E, D);
    Workspace::Scope block_scope(ws_);

    View2D out_node;
    if (packed) {
      View2D tokens = a == 0 ? tokens0 : tokens_of(a);
      if (a > 0) copy(rows(tokens, N, E), edge_cur);
      linear(rows(tokens, 0, N), node_cur, mat(tl_ + ".center_contraction.weight"),
             vec(tl_ + ".center_contraction.bias"));
      {
        Workspace::Scope qkv_scope(ws_);
        View2D attn_in = ws_.tmp(T, D);
        norm(attn_in, tokens, tl_ + ".norm_attention");
        View2D qkv = save && save_wide ? keep(sav->qkv[a], "qkv", a, T, 3 * D) : ws_.tmp(T, 3 * D);
        linear(qkv, attn_in, w_in, b_in);
        attention_ragged(ws_, key, attn_out, qkv, *tl, w_out, b_out, h_.num_heads, h_.head_dim,
                         h_.attention_temperature, save);
      }
      out_node = rows(attn_out, 0, N);
    } else {
      View2D input_node = ws_.tmp(N, D);
      linear(input_node, node_cur, mat(tl_ + ".center_contraction.weight"), vec(tl_ + ".center_contraction.bias"));
      View2D tokens = save ? keep(sav->tokens[a], "tok", a, N * S, D) : ws_.tmp(N * S, D);
      Kokkos::parallel_for(
          "tok", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
            const int row = i / D, d = i % D, n = row / S, e = off(n) + row % S - 1;
            tokens(row, d) = row % S == 0 ? input_node(n, d) : e < off(n + 1) ? edge_cur(e, d) : Net(0);
          });

      // Attention, then its output split: the central row to the centre
      // expansion, the edge rows onto the edge residual, eps = attn + edge_in.
      out_node = ws_.tmp(N, D);
      eps = save ? keep(sav->eps[a], "eps", a, E, D) : edge_next;
      Workspace::Scope attn_scope(ws_);
      View2D attn_out_p = ws_.tmp(N * S, D);
      {
        Workspace::Scope qkv_scope(ws_);
        View2D attn_in = ws_.tmp(N * S, D);
        norm(attn_in, tokens, tl_ + ".norm_attention");
        View2D qkv = save && save_wide ? keep(sav->qkv[a], "qkv", a, N * S, 3 * D) : ws_.tmp(N * S, 3 * D);
        linear(qkv, attn_in, w_in, b_in);
        if (tl)
          attention_ragged(ws_, key, attn_out_p, qkv, *tl, w_out, b_out, h_.num_heads, h_.head_dim,
                           h_.attention_temperature, save);
        else
          attention(ws_, key, attn_out_p, qkv, dev.cf_seq, w_out, b_out, N, S, h_.num_heads, h_.head_dim,
                    h_.attention_temperature, save);
      }
      View2D on = out_node, ep = eps, en = edge_next, ec = edge_cur;
      Kokkos::parallel_for(
          "split_res", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
            const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
            if (row % S == 0) return (void) (on(n, d) = attn_out_p(row, d));
            if (k >= off(n + 1)) return;
            ep(k, d) = attn_out_p(row, d) + ec(k, d);
            if (save) en(k, d) = ep(k, d);
          });
    }

    // The node side the same way: node_res = node_in + expansion(central row), the
    // block's node output node_res + mlp(norm(node_res)).
    View2D node_exp = ws_.tmp(N, Dn);
    linear(node_exp, out_node, mat(tl_ + ".center_expansion.weight"), vec(tl_ + ".center_expansion.bias"));
    View2D node_res = save ? keep(sav->node_new[a], "nn", a, N, Dn) : node_next;
    Kokkos::parallel_for(
        "nres", RangePolicy(0, N * Dn), KOKKOS_LAMBDA(int i) {
          const int n = i / Dn, d = i % Dn;
          node_res(n, d) = node_cur(n, d) + node_exp(n, d);
          if (save) node_next(n, d) = node_res(n, d);
        });
    const int cw = ffn_pre_width(tl_ + ".center_mlp"), ew = ffn_pre_width(tl_ + ".mlp");
    View2D node_norm = ws_.tmp(N, Dn);
    norm(node_norm, node_res, tl_ + ".norm_center_features");
    View2D pre_center = save ? keep(sav->tmp_center[a], "tc", a, N, cw) : ws_.tmp(N, cw);
    feedforward(node_next, node_norm, tl_ + ".center_mlp", pre_center, save, 1);
    {
      Workspace::Scope mlp_scope(ws_);
      View2D edge_norm = ws_.tmp(E, D);
      // Packed: eps = attention + edge_in here, in place, and into edge_next when
      // that is a copy.
      if (packed) norm(edge_norm, eps, tl_ + ".norm_mlp", edge_cur, save ? edge_next : View2D());
      else norm(edge_norm, eps, tl_ + ".norm_mlp");
      View2D pre_edge = save && save_wide ? keep(sav->tmp_edge[a], "te", a, E, ew) : ws_.tmp(E, ew);
      feedforward(edge_next, edge_norm, tl_ + ".mlp", pre_edge, save, 1);
    }
    node_cur = node_next;
    edge_cur = edge_next;
  }

  if (cond.extent(0) == (std::size_t) N)
    Kokkos::parallel_for(
        "cond_add", RangePolicy(0, N * Dn), KOKKOS_LAMBDA(int i) { node_cur(i / Dn, i % Dn) += cond(i / Dn, i % Dn); });
  node = node_cur;

  // input_edge += out_edge + comb_mlp(layernorm(concat)), concat = [out_edge |
  // out_edge of the reverse edge]; the MLP's last linear accumulates onto it. An
  // edge to a ghost reads its reverse from the rank owning the ghost.
  const View2D out_edge = edge_cur;
  const bool remote = dev.exchange != nullptr;  // every rank, rows or not: the exchange is collective
  auto ro = pk.remote_of, lp = pk.live_packed;
  if (remote && share) {
    View2D out = ws_.tmp(pk.n_live, D);
    Kokkos::parallel_for(
        "remote_rows", RangePolicy(0, pk.n_live * D), KOKKOS_LAMBDA(int i) {
          const int k = lp(i / D);  // -1: a capacity's padding
          out(i / D, i % D) = k >= 0 ? out_edge(k, i % D) : Net(0);
        });
    dev.exchange->edges(out, remote_in);
  }
  View2D concat = save ? keep(sav->concat, "cc", -1, E, 2 * D) : ws_.tmp(E, 2 * D);
  Kokkos::parallel_for(
      "cat_rev", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
        const int k = i / D, d = i % D, r = rev(k);
        concat(k, d) = out_edge(k, d);
        concat(k, D + d) = r >= 0 ? out_edge(r, d) : remote && ro(k) >= 0 ? remote_in(ro(k), d) : Net(0);
        input_edge(k, d) += out_edge(k, d);
      });
  View2D comb_norm = ws_.tmp(E, 2 * D), cph = ws_.tmp(E, 2 * D);
  norm_fwd(comb_norm, concat, vec("combination_norms." + ls + ".weight"), vec("combination_norms." + ls + ".bias"));
  linear_silu(cph, save ? keep(sav->cph, "cph", -1, E, 2 * D) : View2D(), comb_norm,
              mat("combination_mlps." + ls + ".0.weight"), vec("combination_mlps." + ls + ".0.bias"));
  linear(input_edge, cph, mat("combination_mlps." + ls + ".2.weight"), vec("combination_mlps." + ls + ".2.bias"), 1);
}

void PetModel::ff_layer_bwd(const DeviceEdgeData& dev, const PackedEdges& pk, int L, const LayerSaves& sav,
                            bool kept_wide, View2D node_adj, View2D input_edge_adj, View2D x4_adj,
                            View2D cf_seq_adj, View1D cf_adj, const TokenLayout* tl) {
  const int N = dev.centres(), S = dev.max_neighbors + 1, E = pk.E;
  const bool packed = tl && tl->S0 == 0;
  const int T = packed ? N + E : N * S;
  const int D = h_.d_pet, Dn = h_.d_node, A = h_.num_attention_layers;
  const std::string g = "gnn_layers." + std::to_string(L), ls = std::to_string(L);
  auto off = pk.off, rev = pk.reverse;
  // lin(norm(in)): a wide activation the forward did not keep, by the forward's
  // own route.
  auto rebuild = [&](View2D in, const std::string& norm_key, const std::string& lin, bool swiglu_pre) {
    View2D x = ws_.tmp(in.extent(0), in.extent(1));
    norm(x, in, norm_key);
    View2D out = ws_.tmp(in.extent(0), mat(lin + ".weight").extent(0));
    if (swiglu_pre) swiglu_in(out, {}, x, mat(lin + ".weight"), vec(lin + ".bias"));
    else linear(out, x, mat(lin + ".weight"), vec(lin + ".bias"));
    return out;
  };
  Workspace::Scope layer_scope(ws_);
  // Packed: edge_adj is the edge half of the attention output's adjoint, which
  // the forward's eps = attention + edge_in makes the same thing.
  View2D ao_adj = packed ? ws_.tmp(T, D) : View2D();
  View2D edge_adj = packed ? rows(ao_adj, N, E) : ws_.tmp(E, D);

  // The combination MLP. input_edge_adj passes through to the layer's input
  // unchanged, and out_edge gathers its own concat slot and its reverse's.
  {
    Workspace::Scope scope(ws_);
    View2D cph_adj = ws_.tmp(E, 2 * D), cnorm_adj = ws_.tmp(E, 2 * D), concat_adj = ws_.tmp(E, 2 * D);
    linear_silu_bwd(cph_adj, input_edge_adj, mat("combination_mlps." + ls + ".2.weight"), sav.cph);
    linear_bwd(cnorm_adj, cph_adj, mat("combination_mlps." + ls + ".0.weight"), 0);
    pet::norm_bwd(concat_adj, cnorm_adj, sav.concat, vec("combination_norms." + ls + ".weight"), true, false);
    // An edge to a ghost: its partner's rank has the adjoint of this edge's row,
    // and wants the adjoint of the row this edge read.
    const bool remote = dev.exchange != nullptr;
    auto ro = pk.remote_of, lp = pk.live_packed;
    View2D remote_adj = ws_.tmp(remote ? pk.n_live : 0, D);
    if (remote) {
      View2D out = ws_.tmp(pk.n_live, D);
      Kokkos::parallel_for(
          "remote_adj", RangePolicy(0, pk.n_live * D),
          KOKKOS_LAMBDA(int i) {
            const int k = lp(i / D);
            out(i / D, i % D) = k >= 0 ? concat_adj(k, D + i % D) : Net(0);
          });
      dev.exchange->edges(out, remote_adj);
    }
    Kokkos::parallel_for(
        "bw_concat", RangePolicy(0, E * D), KOKKOS_LAMBDA(int i) {
          const int k = i / D, d = i % D, r = rev(k);
          const Net back = r >= 0 ? concat_adj(r, D + d) : remote && ro(k) >= 0 ? remote_adj(ro(k), d) : Net(0);
          edge_adj(k, d) = input_edge_adj(k, d) + concat_adj(k, d) + back;
        });
  }

  for (int a = A - 1; a >= 0; --a) {
    const std::string tl_ = g + ".trans.layers." + std::to_string(a);
    const std::string key = "attn_" + sav.tag + "_" + std::to_string(a);
    // The two MLPs: afterwards edge_adj and node_adj are eps's and node_res's.
    {
      Workspace::Scope scope(ws_);
      View2D enorm_adj = ws_.tmp(E, D), cnorm_adj = ws_.tmp(N, Dn);
      feedforward_bwd(enorm_adj, edge_adj, tl_ + ".mlp",
                      kept_wide ? sav.tmp_edge[a] : rebuild(sav.eps[a], tl_ + ".norm_mlp", tl_ + ".mlp.w_in", true), 0);
      norm_bwd(edge_adj, enorm_adj, sav.eps[a], tl_ + ".norm_mlp");
      feedforward_bwd(cnorm_adj, node_adj, tl_ + ".center_mlp", sav.tmp_center[a], 0);
      norm_bwd(node_adj, cnorm_adj, sav.node_new[a], tl_ + ".norm_center_features");
    }
    Workspace::Scope block_scope(ws_);
    const WeightRef w_in = mat(tl_ + ".attention.input_linear.weight"), w_out = mat(tl_ + ".attention.output_linear.weight");
    auto qkv_of = [&] {
      return kept_wide ? sav.qkv[a] : rebuild(sav.tokens[a], tl_ + ".norm_attention", tl_ + ".attention.input_linear", false);
    };

    if (packed) {
      // The central rows' adjoint from the expansion; the edge rows' is edge_adj.
      linear_bwd(rows(ao_adj, 0, N), node_adj, mat(tl_ + ".center_expansion.weight"), 0);
      View2D input_node_adj = ws_.tmp(N, D);
      {
        Workspace::Scope scope(ws_);
        View2D attn_in_adj = ws_.tmp(T, D);
        attention_ragged_bwd(ws_, key, attn_in_adj, cf_adj, ao_adj, qkv_of(), *tl, w_in, w_out, h_.num_heads,
                             h_.head_dim, h_.attention_temperature, 0);
        // The norm row by row: the central rows to the contraction, the edge rows
        // onto edge_in's residual share.
        const View2D tok = sav.tokens[a];
        norm_bwd(input_node_adj, rows(attn_in_adj, 0, N), rows(tok, 0, N), tl_ + ".norm_attention", false);
        norm_bwd(edge_adj, rows(attn_in_adj, N, E), rows(tok, N, E), tl_ + ".norm_attention", true);
      }
      linear_bwd(node_adj, input_node_adj, mat(tl_ + ".center_contraction.weight"));
      continue;
    }

    // Attention: its output's adjoint is the expansion's input adjoint on the
    // central row and eps's on the edge rows.
    View2D tokens_adj = ws_.tmp(N * S, D);
    {
      Workspace::Scope scope(ws_);
      View2D out_node_adj = ws_.tmp(N, D), attn_out_adj = ws_.tmp(N * S, D), attn_in_adj = ws_.tmp(N * S, D);
      linear_bwd(out_node_adj, node_adj, mat(tl_ + ".center_expansion.weight"), 0);
      Kokkos::parallel_for(
          "bw_ao", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
            const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
            attn_out_adj(row, d) = row % S == 0 ? out_node_adj(n, d) : k < off(n + 1) ? edge_adj(k, d) : Net(0);
          });
      if (tl)
        attention_ragged_bwd(ws_, key, attn_in_adj, cf_adj, attn_out_adj, qkv_of(), *tl, w_in, w_out, h_.num_heads,
                             h_.head_dim, h_.attention_temperature, 0);
      else
        attention_bwd(ws_, key, attn_in_adj, cf_seq_adj, attn_out_adj, qkv_of(), dev.cf_seq, w_in, w_out, N, S,
                      h_.num_heads, h_.head_dim, h_.attention_temperature, 0);
      norm_bwd(tokens_adj, attn_in_adj, sav.tokens[a], tl_ + ".norm_attention", false);
    }

    // tokens = [input_node ; edge_in]: edge rows add onto edge_in's residual
    // share, the central row goes back through the contraction onto node_in's.
    View2D input_node_adj = ws_.tmp(N, D);
    Kokkos::parallel_for(
        "bw_tok", RangePolicy(0, N * S * D), KOKKOS_LAMBDA(int i) {
          const int row = i / D, d = i % D, n = row / S, k = off(n) + row % S - 1;
          if (row % S == 0) input_node_adj(n, d) = tokens_adj(row, d);
          else if (k < off(n + 1)) edge_adj(k, d) += tokens_adj(row, d);
        });
    linear_bwd(node_adj, input_node_adj, mat(tl_ + ".center_contraction.weight"));
  }

  // edge_adj is now et's.
  View2D cpre_adj = ws_.tmp(E, D);
  linear_bwd(cpre_adj, edge_adj, mat(g + ".compress.2.weight"), 0);
  compress_bwd(cpre_adj, sav.cpre, compress_fold(L), x4_adj, input_edge_adj);
}

DeviceOut PetModel::ff_pass(const DeviceEdgeData& dev, bool grad) {
  const int N = dev.centres(), S = dev.max_neighbors + 1;  // the model's rows
  const int D = h_.d_pet, Dn = h_.d_node, G = h_.num_gnn_layers;
  Workspace::Scope scope(ws_);
  // Every buffer's first writer overwrites it; the few accumulators are zeroed
  // where they are taken.
  ws_.set_zero(false);

  const PackedEdges pk = pack_edges(dev);
  const int E = pk.E;
  // The tokens' layout (see ff_layer): packed with the ragged attention kernels
  // where they run, unless PET_TOKENS=padded; PET_ATTN=kokkos keeps
  // attention.cpp's (padded) kernels.
  static const bool padded_tokens = [] {
    const char* e = std::getenv("PET_TOKENS");
    return e && std::strcmp(e, "padded") == 0;
  }();
  TokenLayout layout;
  const TokenLayout* tl = nullptr;
  if (attention_ragged_supported(h_.num_heads, h_.head_dim)) {
    layout.N = N, layout.E = E, layout.S0 = padded_tokens ? S : 0;
    layout.T = padded_tokens ? N * S : N + E;
    layout.max_len = S, layout.tail = !padded_tokens && dev.edge_capacity;
    layout.off = pk.off, layout.center = pk.center, layout.cut = pk.cut;
    layout.lcf = attention_log_cf(ws_, pk);
    tl = &layout;
  }
  const bool packed = tl && tl->S0 == 0;
  View2D node = ws_.n2("node", N, Dn), input_edge = ws_.n2("input_edge", E, D);
  gather(node, mat("node_embedders.0.weight").v, dev.species);
  gather(input_edge, mat("edge_embedder.weight").v, pk.species);
  const View2D cond = conditioning(dev);

  // What the backward will have: every layer's saves; the same without the two
  // widest (rebuilt when needed); or only each layer's inputs, the backward then
  // re-running one layer's forward at a time.
  const Recompute rc = recompute_tier(N, packed ? (N + E + N - 1) / std::max(N, 1) : S, E);
  const bool keep_wide = rc == Recompute::Never, ckpt = grad && rc == Recompute::Layers;
  std::vector<LayerSaves> sav(grad && !ckpt ? G : 0);
  std::vector<std::pair<View2D, View2D>> inputs(ckpt ? G : 0);
  // Over several ranks, each layer's rows from the partners' ranks, kept so a
  // recompute need not ask again.
  std::vector<View2D> remote_in(dev.exchange ? G : 0);
  for (int L = 0; L < (int) remote_in.size(); ++L) remote_in[L] = ws_.n2("remote_in_" + std::to_string(L), pk.n_live, D);
  for (int L = 0; L < G; ++L) {
    if (ckpt) {
      inputs[L] = {ws_.n2("ck_node_" + std::to_string(L), N, Dn), ws_.n2("ck_ie_" + std::to_string(L), E, D)};
      copy(inputs[L].first, node);
      copy(inputs[L].second, input_edge);
    }
    if (!sav.empty()) sav[L].tag = std::to_string(L);
    ff_layer(dev, pk, L, node, input_edge, cond, sav.empty() ? nullptr : &sav[L], grad && keep_wide,
             remote_in.empty() ? View2D() : remote_in[L], true, tl);
  }

  View1D net = ws_.n1("per_atom_net", N);
  ReadoutSaves rs;
  readout({node}, {input_edge}, net, pk, grad ? &rs : nullptr, "");
  RView1D per_atom = ws_.r1("per_atom", dev.n_atoms);  // every atom's, 0 past n_local
  assemble_energy(per_atom, net, dev.species, comp_view_, energy_scale_, pk.n_local);
  if (!grad) return {per_atom, {}, {}, {}};

  // The attention bias's adjoint: per (atom, slot) from attention.cpp's kernels,
  // per packed edge from the ragged ones.
  View2D x4_adj = ws_.n2("x4_adj", E, 4), cf_seq_adj = tl ? View2D() : ws_.n2("cf_seq_adj", N, S);
  View1D cf_adj = tl ? ws_.n1("cf_adj", E) : View1D();
  Kokkos::deep_copy(ExecSpace(), x4_adj, Net(0));
  if (tl) Kokkos::deep_copy(ExecSpace(), cf_adj, Net(0));
  else Kokkos::deep_copy(ExecSpace(), cf_seq_adj, Net(0));
  View2D node_adj = ws_.n2("node_adj", N, Dn), input_edge_adj = ws_.n2("ie_adj", E, D);
  View1D cutoff_adj = ws_.n1("cutoff_adj", E);
  readout_bwd(rs, 0, pk, N, node_adj, input_edge_adj, cutoff_adj, false);
  for (int L = G - 1; L >= 0; --L) {
    Workspace::Scope layer_scope(ws_);
    LayerSaves one;
    if (ckpt) {
      View2D n = ws_.tmp(N, Dn), ie = ws_.tmp(E, D);
      copy(n, inputs[L].first);
      copy(ie, inputs[L].second);
      one.tag = "c";  // one shared set of saves
      ff_layer(dev, pk, L, n, ie, cond, &one, false, remote_in.empty() ? View2D() : remote_in[L], false, tl);
    }
    ff_layer_bwd(dev, pk, L, ckpt ? one : sav[L], keep_wide, node_adj, input_edge_adj, x4_adj, cf_seq_adj, cf_adj,
                 tl);
  }

  DeviceOut out{per_atom, {}, {}, {}};
  forces_and_virial(ws_, dev, pk, h_, probes_, n_probes_, energy_scale_, x4_adj, cutoff_adj, cf_seq_adj,
                    out.forces, out.vir9, out.edge_grad, cf_adj);
  return out;
}

}  // namespace pet
