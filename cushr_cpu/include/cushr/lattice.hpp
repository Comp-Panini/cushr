#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace cushr {

struct SentenceView {
    int sentence_id;
    int node_begin; // starting node ID
    int node_end;  // ending node ID, exclusive
};

class Lattice {
public:
    Lattice() = default;
    static Lattice load_npz(const std::string& path);

    // in-memory constructor used by unit tests
    Lattice(std::vector<int> row_ptr,
            std::vector<int> col_idx,
            std::vector<int> topo_level,
            std::vector<int> sentence_offsets,
            std::vector<float> node_features,
            int feat_dim,
            std::vector<uint8_t> gold_path_mask = {});

    // shape
    int num_nodes() const { return (int)topo_level_.size(); }
    int num_edges() const { return (int)col_idx_.size(); }
    int num_sentences() const { return (int)sentence_offsets_.size() - 1; }
    int feat_dim() const { return feat_dim_; }

    // per-node queries
    int topo_level(int v) const { 
        return topo_level_[v]; 
    }
    int out_degree(int v) const { 
        return row_ptr_[v + 1] - row_ptr_[v]; 
    }
    int in_degree(int v) const { 
        return in_row_ptr_[v + 1] - in_row_ptr_[v]; 
    }
    bool is_gold(int v) const {
        return !gold_path_mask_.empty() && gold_path_mask_[v] != 0;
    }

    // True if the npz carried an explicit, pre-resolved gold path per sentence
    // (gold_path_nodes / gold_path_offsets). Newer ingest emits this; older
    // archives only have the per-node mask.
    bool has_explicit_gold() const { return !gold_path_offsets_.empty(); }

    // Explicit gold path (ordered word node ids) for sentence s. Empty when the
    // sentence has no resolved gold path. Boundary super-source/sink nodes are
    // NOT included.
    std::vector<int> gold_path(int s) const {
        if (gold_path_offsets_.empty()) return {};
        const int b = gold_path_offsets_[s];
        const int e = gold_path_offsets_[s + 1];
        return std::vector<int>(gold_path_nodes_.begin() + b,
                                gold_path_nodes_.begin() + e);
    }

    // edge iteration
    // forward CSR: outgoing edges of v are [row_ptr_[v], row_ptr_[v+1])
    int out_edge_begin(int v) const { 
        return row_ptr_[v]; 
    }
    int out_edge_end(int v) const { 
        return row_ptr_[v + 1]; 
    }
    int edge_dst(int e) const {
        return col_idx_[e];
    }
    // Source node of forward edge e. The CSR does not store it, so we binary
    // search row_ptr_. Only the biaffine scorer needs this; the decoder always
    // has u in hand already.
    int edge_src(int e) const {
        return (int)(std::upper_bound(row_ptr_.begin(), row_ptr_.end(), e)
                     - row_ptr_.begin()) - 1;
    }

    // reverse CSR: incoming edges of v are [in_row_ptr_[v], in_row_ptr_[v+1])
    int in_edge_begin(int v) const { 
        return in_row_ptr_[v]; 
    }
    int in_edge_end(int v) const { 
        return in_row_ptr_[v + 1]; 
    }
    int in_edge_src(int e_rev) const { 
        return in_col_idx_[e_rev]; 
    }
    int in_edge_forward_id(int e_rev) const { 
        return in_edge_id_[e_rev]; 
    }

    // features 
    const float* node_feature_ptr(int v) const {
        return node_features_.data() + (size_t)v * feat_dim_;
    }
    int node_word_length(int v) const {
        return node_word_length_.empty() ? 0 : node_word_length_[v];
    }

    // sentences
    SentenceView sentence(int s) const;
    const std::vector<int>& sentence_offsets() const { 
        return sentence_offsets_; 
    }

    // within sentence, return the nodes in topological order
    std::vector<int> topo_order_for_sentence(int s) const;

    // sanity check: every edge (u, v) must satisfy topo_level[u] < topo_level[v].
    bool validate(std::string* err = nullptr) const;

private:
    // forward CSR
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;

    // reverse CSR (built in build_reverse_csr_)
    std::vector<int> in_row_ptr_;
    std::vector<int> in_col_idx_;
    std::vector<int> in_edge_id_;

    // per-node
    std::vector<int> topo_level_;
    std::vector<float> node_features_;
    int feat_dim_ = 0;
    std::vector<int> node_word_length_;
    std::vector<uint8_t> gold_path_mask_;

    // explicit gold path: gold_path_nodes_ is a flat list of word node ids;
    // gold_path_offsets_[s..s+1] slices out sentence s's path (CSR-style).
    std::vector<int> gold_path_nodes_;
    std::vector<int> gold_path_offsets_;

    // sentences
    std::vector<int> sentence_offsets_;

    void build_reverse_csr_();
};

}  
