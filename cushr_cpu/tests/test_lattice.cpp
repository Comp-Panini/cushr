// tests/test_lattice.cpp
#include <doctest/doctest.h>

#include "cushr/lattice.hpp"

using cushr::Lattice;

// Build a tiny single-sentence lattice:
//   0 -> 1 -> 3
//   0 -> 2 -> 3
// topo levels: 0, 1, 1, 2
static Lattice make_diamond() {
    std::vector<int> row_ptr = {0, 2, 3, 4, 4};      // out-edges per node
    std::vector<int> col_idx = {1, 2, 3, 3};
    std::vector<int> topo    = {0, 1, 1, 2};
    std::vector<int> off     = {0, 4};               // one sentence, nodes [0,4)
    std::vector<float> feats(4 * 2, 0.0f);            // feat_dim = 2
    return Lattice(row_ptr, col_idx, topo, off, feats, /*feat_dim=*/2);
}

TEST_CASE("lattice basic shape") {
    auto lat = make_diamond();
    CHECK(lat.num_nodes() == 4);
    CHECK(lat.num_edges() == 4);
    CHECK(lat.num_sentences() == 1);
    CHECK(lat.feat_dim() == 2);

    CHECK(lat.out_degree(0) == 2);
    CHECK(lat.out_degree(3) == 0);
    CHECK(lat.in_degree(3)  == 2);
    CHECK(lat.in_degree(0)  == 0);
}

TEST_CASE("reverse CSR is consistent with forward CSR") {
    auto lat = make_diamond();
    // node 3 has incoming edges from 1 and 2
    std::vector<int> srcs;
    for (int er = lat.in_edge_begin(3); er < lat.in_edge_end(3); ++er) {
        srcs.push_back(lat.in_edge_src(er));
        int fid = lat.in_edge_forward_id(er);
        CHECK(lat.edge_dst(fid) == 3);
    }
    std::sort(srcs.begin(), srcs.end());
    CHECK(srcs == std::vector<int>{1, 2});
}

TEST_CASE("validate rejects topo violations") {
    // Edge 0->2 OK, but flip topo so level[2] < level[0]
    std::vector<int> row_ptr = {0, 1, 1, 1};
    std::vector<int> col_idx = {2};
    std::vector<int> topo    = {5, 0, 0};   // 5 -> 0 violates
    std::vector<int> off     = {0, 3};
    std::vector<float> feats(3 * 1, 0.0f);
    Lattice lat;
    bool threw = false;
    try {
        lat = Lattice(row_ptr, col_idx, topo, off, feats, 1);
        // validate is called inside load_npz, not the in-memory ctor.
        std::string err;
        CHECK_FALSE(lat.validate(&err));
    } catch (...) {
        threw = true;
    }
    CHECK_FALSE(threw);   // ctor itself doesn't throw, validate() reports
}

TEST_CASE("sentence boundaries are reported correctly") {
    // Two sentences, both diamonds, glued together
    std::vector<int> row_ptr = {0,2,3,4,4, 6,7,8,8};
    std::vector<int> col_idx = {1,2,3,3, 5,6,7,7};
    std::vector<int> topo    = {0,1,1,2, 0,1,1,2};
    std::vector<int> off     = {0,4,8};
    std::vector<float> feats(8 * 1, 0.0f);
    Lattice lat(row_ptr, col_idx, topo, off, feats, 1);

    CHECK(lat.num_sentences() == 2);
    auto s0 = lat.sentence(0);
    auto s1 = lat.sentence(1);
    CHECK(s0.node_begin == 0); CHECK(s0.node_end == 4);
    CHECK(s0.source_node == 0); CHECK(s0.sink_node == 3);
    CHECK(s1.node_begin == 4); CHECK(s1.node_end == 8);
    CHECK(s1.source_node == 4); CHECK(s1.sink_node == 7);

    std::string err;
    CHECK(lat.validate(&err));
}
