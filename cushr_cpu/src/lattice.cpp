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

// USED WHEN DECODER ABSOLUTELY NEEDS THIS ARR
// load a named array from an open .npz file as a std::vector<T>.
// arr.data<T>() gives a raw pointer to the array bytes interpreted as T.
template <typename T> std::vector<T> load_required(cnpy::npz_t& z, const std::string& key) {
// template <typename T> means it will be generated differently based on what type T is
// cnpy lib opens ZIP of .npy files and loads into C++ dictionary which is referred to by z
// key is the array you want

    auto it = z.find(key); 
    // search dict for key, it is a ptr to where key/val lives in dict
    
    if (it == z.end()) { // if no value for given key, it returns special marker marking end of dict
        throw std::runtime_error("cushr: missing required array '" + key + "' in npz");
    } // garbage memory

    const cnpy::NpyArray& arr = it->second; 
    // it->second is the value/numpy arr obj

    const T* p = arr.data<T>(); 
    // copy the raw bytes into a vector so the Lattice owns its memory independently of the cnpy map (which we discard after load_npz returns).
    
    return std::vector<T>(p, p + arr.num_vals);
    // executes a range constructor, deep copies data so when dict deleted you have data permanently
}

// USED WHEN IF ARR NOT THERE ITS OK
template <typename T> std::vector<T> load_optional(cnpy::npz_t& z, const std::string& key) {
    auto it = z.find(key);
    if (it == z.end()) return {};   // key absent -> caller gets empty vector
    const cnpy::NpyArray& arr = it->second;
    const T* p = arr.data<T>();
    return std::vector<T>(p, p + arr.num_vals);
}

}  // namespace


// opens the .npz file produced by ingest script
// loads each named array into the corresponding member vector
// builds the reverse CSR
// validates the result
Lattice Lattice::load_npz(const std::string& path) {

    // returns a map from array name to NpyArray.
    cnpy::npz_t z = cnpy::npz_load(path);

    Lattice lat;

    // forward CSR
    lat.row_ptr_ = load_required<int>(z, "row_ptr");
    lat.col_idx_ = load_required<int>(z, "col_idx");
    lat.topo_level_ = load_required<int>(z, "topo_level");
    lat.sentence_offsets_ = load_required<int>(z, "sentence_offsets");

    // node_features is [num_nodes, feat_dim] or a 2D arr, flatten to 1D
    // flatten it to a 1-D vector and store feat_dim_ separately so that
    //   node_feature_ptr(v) = node_features_.data() + v * feat_dim_
    // gives the feature vector for node v.
    {
        auto it = z.find("node_features");
        if (it == z.end()) {
            throw std::runtime_error("cushr: missing 'node_features' in npz");
        }
        const auto& arr = it->second;
        if (arr.shape.size() != 2) {
            throw std::runtime_error("cushr: node_features must be 2-D");
        }
        lat.feat_dim_ = (int)arr.shape[1]; // stride
        // to get node v's feature, do pts arithmetic.
        const float* p = arr.data<float>();
        lat.node_features_.assign(p, p + arr.num_vals); // store deep copy
    }


    lat.gold_path_mask_ = load_optional<uint8_t>(z, "gold_path_mask");
    lat.node_word_length_ = load_optional<int>(z, "node_word_length");

    // explicit pre-resolved gold path (optional; newer ingest emits it)
    lat.gold_path_nodes_ = load_optional<int>(z, "gold_path_nodes");
    lat.gold_path_offsets_ = load_optional<int>(z, "gold_path_offsets");

    // build the reverse CSR
    lat.build_reverse_csr_();

    // check graph structure before returning
    std::string err;
    if (!lat.validate(&err)) {
        throw std::runtime_error("cushr: lattice failed validation: " + err);
    }
    return lat;
}

// in-memory constructor used by unit tests only.
// takes pre-built arrays directly instead of reading from disk
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
    const int N = (int)topo_level_.size(); // num_nodes
    const int E = (int)col_idx_.size(); // num_edges

    // count how many incoming edges each node has
    in_row_ptr_.assign(N + 1, 0);
    for (int e = 0; e < E; ++e) {
        int v = col_idx_[e]; // destination of forward edge e
        in_row_ptr_[v + 1]++;
    }

    // prefix-sum to convert counts to start offsets.
    // in_row_ptr_[v] = index of the first incoming edge of v
    for (int v = 0; v < N; ++v) {
        in_row_ptr_[v + 1] += in_row_ptr_[v];
    }

    // put source nodes and forward edge ids into their slots.
    // cursor[v] tracks how many incoming edges of v we've placed so far.
    in_col_idx_.assign(E, 0); // remembers source nodes
    in_edge_id_.assign(E, 0); // remembers original edge ids
    std::vector<int> cursor(N, 0); // create a counter for every node that counts incoming edges
    for (int u = 0; u < N; ++u) { // loop over every node
        for (int e = row_ptr_[u]; e < row_ptr_[u + 1]; ++e) { // look at rowptr to see start and stop, loop thru those edges
            int v = col_idx_[e]; // look at destination array to see destination of edge e from node v
            int slot = in_row_ptr_[v] + cursor[v]++; // node v starts at in_row_ptr[v] and ends at the amount of edges already there, increment
            in_col_idx_[slot] = u; // source of this incoming edge is u, write in that box
            in_edge_id_[slot] = e; // original forward edge id is also written
        }
    }
}

// returns a SentenceView which is a lightweight struct describing which global node ids
// belong to sentence s. Nodes [node_begin, node_end) are the entire graph for
// that sentence.
// starts path reconstruction from the sink and follows backpointers to source
SentenceView Lattice::sentence(int s) const {
    SentenceView v;
    v.sentence_id = s;
    v.node_begin = sentence_offsets_[s];
    v.node_end = sentence_offsets_[s + 1];
    return v;
}

// returns the nodes of sentence s sorted by topo_level (ascending).
// decoder must visit nodes in this order so that when it processes node v,
// all predecessors of v have already been fully scored.

// ingest script expected to write nodes in topo order already, so safety net
std::vector<int> Lattice::topo_order_for_sentence(int s) const {
    const auto sv = sentence(s);
    std::vector<int> nodes;
    nodes.reserve(sv.node_end - sv.node_begin);
    for (int v = sv.node_begin; v < sv.node_end; ++v) {
        nodes.push_back(v);
    }
    std::stable_sort(nodes.begin(), nodes.end(),
        [this](int a, int b){ return topo_level_[a] < topo_level_[b]; });
    return nodes;
}


// checks that loaded lattice is internally consistent
bool Lattice::validate(std::string* err) const {
    auto fail = [&](const std::string& m){
        if (err) *err = m;
        return false;
    };

    // row_ptr must have exactly num_nodes+1 entries
    if (row_ptr_.size() != (size_t)num_nodes() + 1) {
        return fail("row_ptr size mismatch");
    }
    // The last entry of row_ptr must equal num_edges
    if ((int)row_ptr_.back() != num_edges()) {
        return fail("row_ptr tail != num_edges");
    }

    // sentence_offsets must start at 0 and end at num_nodes 
    if (sentence_offsets_.empty() || sentence_offsets_.front() != 0
        || sentence_offsets_.back() != num_nodes()) {
        return fail("sentence_offsets does not span [0, num_nodes]");
    }

    // node_features is stored flat: total elements must be num_nodes * feat_dim.
    if ((int)node_features_.size() != num_nodes() * feat_dim_) {
        return fail("node_features size mismatch");
    }

    // for every edge (u->v), level[u] < level[v] (topological stuff)
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

    // explicit gold path arrays (optional) must be internally consistent
    if (!gold_path_offsets_.empty()) {
        if ((int)gold_path_offsets_.size() != num_sentences() + 1) {
            return fail("gold_path_offsets size != num_sentences+1");
        }
        if (gold_path_offsets_.front() != 0
            || gold_path_offsets_.back() != (int)gold_path_nodes_.size()) {
            return fail("gold_path_offsets does not span gold_path_nodes");
        }
        for (int g : gold_path_nodes_) {
            if (g < 0 || g >= num_nodes()) return fail("gold_path node OOB");
        }
    }

    // each sentence must have one source (in-degree 0) and one sink (out-degree 0)
    for (int s = 0; s < num_sentences(); ++s) {
        auto sv = sentence(s);
        if (in_degree(sv.node_begin) != 0) {
            return fail("sentence source has non-zero in-degree");
        }
        if (out_degree(sv.node_end - 1) != 0) {
            return fail("sentence sink has non-zero out-degree");
        }
        // no edge may connect a node in sentence s to a node outside sentence s.
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

}  
