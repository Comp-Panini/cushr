// cushr/lattice.hpp
//
// In-memory lattice store for the SHR k-best decoder.
//
// Backing arrays are loaded from the .npz produced by the week-2 ingest
// script. Layout is flat-CSR over a *batch* of sentences: a sentence is just
// a contiguous slice of node ids. Edges never cross sentence boundaries.
//
// Coordinate system:
//   nodes  : global ids, [0, num_nodes_)
//   edges  : global ids, [0, num_edges_)
//   row_ptr_[v]..row_ptr_[v+1]   -> outgoing edges of v       (forward CSR)
//   in_row_ptr_[v]..in_row_ptr_[v+1] -> incoming edges of v   (reverse CSR)
//   col_idx_[e]                  -> destination node of edge e
//   in_col_idx_[e_rev]           -> source node of edge e_rev
//   in_edge_id_[e_rev]           -> forward-edge id corresponding to e_rev
//
// The reverse CSR is built lazily on load. We need it for the decoder's
// per-node "scan in-edges" loop.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cushr {

struct SentenceView {
    int sentence_id;
    int node_begin;   // global node id range for this sentence
    int node_end;     // exclusive
    int source_node;  // designated source within the sentence (typically node_begin)
    int sink_node;    // designated sink (typically node_end - 1)
};

class Lattice {
public:
    Lattice() = default;

    // Load from the .npz produced by week-2's ingest.py.
    // Expected arrays (all int32 except node_features which is float32):
    //   node_features    [num_nodes, feat_dim]
    //   row_ptr          [num_nodes + 1]
    //   col_idx          [num_edges]
    //   topo_level       [num_nodes]
    //   sentence_offsets [num_sentences + 1]   // node-id boundaries
    //   gold_path_mask   [num_nodes]           // 0/1
    //
    // Optional arrays (used by some scorers / metrics):
    //   node_word_length [num_nodes]           // surface form length in chars
    //   node_surface_id  [num_nodes]           // dictionary id of surface form
    //   edge_features    [num_edges, edge_feat_dim]  (if absent, edges have no
    //                                                  own features)
    static Lattice load_npz(const std::string& path);

    // In-memory constructor used by unit tests. Caller owns nothing afterward;
    // we copy.
    Lattice(std::vector<int> row_ptr,
            std::vector<int> col_idx,
            std::vector<int> topo_level,
            std::vector<int> sentence_offsets,
            std::vector<float> node_features,
            int feat_dim,
            std::vector<uint8_t> gold_path_mask = {});

    // ----------- shape -----------
    int num_nodes()      const { return (int)topo_level_.size(); }
    int num_edges()      const { return (int)col_idx_.size(); }
    int num_sentences()  const { return (int)sentence_offsets_.size() - 1; }
    int feat_dim()       const { return feat_dim_; }

    // ----------- per-node queries -----------
    int topo_level(int v) const { return topo_level_[v]; }
    int out_degree(int v) const { return row_ptr_[v + 1] - row_ptr_[v]; }
    int in_degree(int v)  const { return in_row_ptr_[v + 1] - in_row_ptr_[v]; }
    bool is_gold(int v)   const {
        return !gold_path_mask_.empty() && gold_path_mask_[v] != 0;
    }

    // ----------- edge iteration -----------
    // Forward CSR: outgoing edges of v are [row_ptr_[v], row_ptr_[v+1])
    int out_edge_begin(int v) const { return row_ptr_[v]; }
    int out_edge_end(int v)   const { return row_ptr_[v + 1]; }
    int edge_dst(int e)       const { return col_idx_[e]; }

    // Reverse CSR: incoming edges of v are [in_row_ptr_[v], in_row_ptr_[v+1])
    // and each entry yields a (src_node, forward_edge_id) pair.
    int in_edge_begin(int v) const { return in_row_ptr_[v]; }
    int in_edge_end(int v)   const { return in_row_ptr_[v + 1]; }
    int in_edge_src(int e_rev)       const { return in_col_idx_[e_rev]; }
    int in_edge_forward_id(int e_rev) const { return in_edge_id_[e_rev]; }

    // ----------- features -----------
    // node_feature_ptr(v) -> contiguous block of length feat_dim_
    const float* node_feature_ptr(int v) const {
        return node_features_.data() + (size_t)v * feat_dim_;
    }
    int node_word_length(int v) const {
        return node_word_length_.empty() ? 0 : node_word_length_[v];
    }

    // ----------- sentences -----------
    SentenceView sentence(int s) const;
    const std::vector<int>& sentence_offsets() const { return sentence_offsets_; }

    // ----------- topological iteration helpers -----------
    // Within a single sentence, return the nodes in topological order.
    // Currently this just returns [node_begin, node_end) on the assumption
    // that the ingest script already lays nodes out in topo order. We assert
    // that invariant.
    std::vector<int> topo_order_for_sentence(int s) const;

    // Sanity check: every edge (u, v) must satisfy topo_level[u] < topo_level[v].
    // Returns true on success; on failure, sets *err if non-null.
    bool validate(std::string* err = nullptr) const;

private:
    // Forward CSR
    std::vector<int>   row_ptr_;
    std::vector<int>   col_idx_;
    // Reverse CSR (built in build_reverse_csr_)
    std::vector<int>   in_row_ptr_;
    std::vector<int>   in_col_idx_;
    std::vector<int>   in_edge_id_;
    // Per-node
    std::vector<int>   topo_level_;
    std::vector<float> node_features_;
    int                feat_dim_ = 0;
    std::vector<int>   node_word_length_;
    std::vector<uint8_t> gold_path_mask_;
    // Sentences
    std::vector<int>   sentence_offsets_;

    void build_reverse_csr_();
};

}  // namespace cushr
