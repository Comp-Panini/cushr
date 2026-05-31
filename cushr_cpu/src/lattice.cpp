// cushr/lattice.cpp

#include "cushr/lattice.hpp"

#include <algorithm>
#include <cassert>
#include <cnpy.h>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cushr {

namespace {

template <typename T>
std::vector<T> load_required(cnpy::npz_t& z, const std::string& key) {
    auto it = z.find(key);
    if (it == z.end()) {
        throw std::runtime_error("cushr: missing required array '" + key + "' in npz");
    }
    const cnpy::NpyArray& arr = it->second;
    const T* p = arr.data<T>();
    return std::vector<T>(p, p + arr.num_vals);
}

template <typename T>
std::vector<T> load_optional(cnpy::npz_t& z, const std::string& key) {
    auto it = z.find(key);
    if (it == z.end()) return {};
    const cnpy::NpyArray& arr = it->second;
    const T* p = arr.data<T>();
    return std::vector<T>(p, p + arr.num_vals);
}

}  // namespace

Lattice Lattice::load_npz(const std::string& path) {
    cnpy::npz_t z = cnpy::npz_load(path);

    Lattice lat;

    lat.row_ptr_          = load_required<int>(z, "row_ptr");
    lat.col_idx_          = load_required<int>(z, "col_idx");
    lat.topo_level_       = load_required<int>(z, "topo_level");
    lat.sentence_offsets_ = load_required<int>(z, "sentence_offsets");

    // node_features is 2-D, [num_nodes, feat_dim]; cnpy gives us shape info.
    {
        auto it = z.find("node_features");
        if (it == z.end()) {
            throw std::runtime_error("cushr: missing 'node_features' in npz");
        }
        const auto& arr = it->second;
        if (arr.shape.size() != 2) {
            throw std::runtime_error("cushr: node_features must be 2-D");
        }
        lat.feat_dim_ = (int)arr.shape[1];
        const float* p = arr.data<float>();
        lat.node_features_.assign(p, p + arr.num_vals);
    }

    lat.gold_path_mask_   = load_optional<uint8_t>(z, "gold_path_mask");
    lat.node_word_length_ = load_optional<int>(z, "node_word_length");

    lat.build_reverse_csr_();

    std::string err;
    if (!lat.validate(&err)) {
        throw std::runtime_error("cushr: lattice failed validation: " + err);
    }
    return lat;
}

Lattice::Lattice(std::vector<int> row_ptr,
                 std::vector<int> col_idx,
                 std::vector<int> topo_level,
                 std::vector<int> sentence_offsets,
                 std::vector<float> node_features,
                 int feat_dim,
                 std::vector<uint8_t> gold_path_mask)
    : row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)),
      topo_level_(std::move(topo_level)),
      node_features_(std::move(node_features)),
      feat_dim_(feat_dim),
      gold_path_mask_(std::move(gold_path_mask)),
      sentence_offsets_(std::move(sentence_offsets)) {
    build_reverse_csr_();
}

void Lattice::build_reverse_csr_() {
    const int N = (int)topo_level_.size();
    const int E = (int)col_idx_.size();
    in_row_ptr_.assign(N + 1, 0);

    // count in-degrees
    for (int e = 0; e < E; ++e) {
        int v = col_idx_[e];
        in_row_ptr_[v + 1]++;
    }
    // prefix sum
    for (int v = 0; v < N; ++v) {
        in_row_ptr_[v + 1] += in_row_ptr_[v];
    }
    // scatter
    in_col_idx_.assign(E, 0);
    in_edge_id_.assign(E, 0);
    std::vector<int> cursor(N, 0);
    for (int u = 0; u < N; ++u) {
        for (int e = row_ptr_[u]; e < row_ptr_[u + 1]; ++e) {
            int v = col_idx_[e];
            int slot = in_row_ptr_[v] + cursor[v]++;
            in_col_idx_[slot] = u;
            in_edge_id_[slot] = e;
        }
    }
}

SentenceView Lattice::sentence(int s) const {
    SentenceView v;
    v.sentence_id = s;
    v.node_begin  = sentence_offsets_[s];
    v.node_end    = sentence_offsets_[s + 1];
    // By convention the source is the first node (in-degree 0) and the sink
    // is the last node (out-degree 0). We verify this in validate().
    v.source_node = v.node_begin;
    v.sink_node   = v.node_end - 1;
    return v;
}

std::vector<int> Lattice::topo_order_for_sentence(int s) const {
    const auto sv = sentence(s);
    std::vector<int> nodes;
    nodes.reserve(sv.node_end - sv.node_begin);
    for (int v = sv.node_begin; v < sv.node_end; ++v) {
        nodes.push_back(v);
    }
    // Stable sort by topo_level. Should already be sorted post-ingest, but
    // we don't trust the input file.
    std::stable_sort(nodes.begin(), nodes.end(),
        [this](int a, int b){ return topo_level_[a] < topo_level_[b]; });
    return nodes;
}

bool Lattice::validate(std::string* err) const {
    auto fail = [&](const std::string& m){
        if (err) *err = m;
        return false;
    };

    if (row_ptr_.size() != (size_t)num_nodes() + 1) {
        return fail("row_ptr size mismatch");
    }
    if ((int)row_ptr_.back() != num_edges()) {
        return fail("row_ptr tail != num_edges");
    }
    if (sentence_offsets_.empty() || sentence_offsets_.front() != 0
        || sentence_offsets_.back() != num_nodes()) {
        return fail("sentence_offsets does not span [0, num_nodes]");
    }
    if ((int)node_features_.size() != num_nodes() * feat_dim_) {
        return fail("node_features size mismatch");
    }

    // Edge topological consistency.
    for (int u = 0; u < num_nodes(); ++u) {
        for (int e = row_ptr_[u]; e < row_ptr_[u + 1]; ++e) {
            int v = col_idx_[e];
            if (v < 0 || v >= num_nodes()) return fail("edge to OOB node");
            if (topo_level_[u] >= topo_level_[v]) {
                std::ostringstream oss;
                oss << "edge " << e << " (" << u << "->" << v
                    << ") violates topo order: level[u]=" << topo_level_[u]
                    << ", level[v]=" << topo_level_[v];
                return fail(oss.str());
            }
        }
    }

    // Each sentence: first node should have in-degree 0, last out-degree 0.
    // (We can only check the global in/out-degree, which is equivalent since
    // no edges cross sentence boundaries.)
    for (int s = 0; s < num_sentences(); ++s) {
        auto sv = sentence(s);
        if (in_degree(sv.source_node) != 0) {
            return fail("sentence source has non-zero in-degree");
        }
        if (out_degree(sv.sink_node) != 0) {
            return fail("sentence sink has non-zero out-degree");
        }
        // Edges must stay within the sentence.
        for (int v = sv.node_begin; v < sv.node_end; ++v) {
            for (int e = row_ptr_[v]; e < row_ptr_[v + 1]; ++e) {
                int dst = col_idx_[e];
                if (dst < sv.node_begin || dst >= sv.node_end) {
                    return fail("edge crosses sentence boundary");
                }
            }
        }
    }

    return true;
}

}  // namespace cushr
