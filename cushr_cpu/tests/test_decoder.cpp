// tests/test_decoder.cpp
#include <doctest/doctest.h>

#include "cushr/decoder.hpp"
#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using cushr::Entry;
using cushr::Lattice;
using cushr::LogLinearScorer;
using cushr::TopKDecoder;
using cushr::UniformScorer;

// Same diamond used in test_lattice.cpp:
//   0 -> 1 -> 3
//   0 -> 2 -> 3
// Two source->sink paths: 0-1-3 and 0-2-3.
static Lattice make_diamond_with_features(
    const std::vector<float>& per_node_feat0) {
    std::vector<int> row_ptr = {0, 2, 3, 4, 4};
    std::vector<int> col_idx = {1, 2, 3, 3};
    std::vector<int> topo    = {0, 1, 1, 2};
    std::vector<int> off     = {0, 4};
    std::vector<float> feats = per_node_feat0;  // [4] -> feat_dim = 1
    return Lattice(row_ptr, col_idx, topo, off, feats, /*feat_dim=*/1);
}

TEST_CASE("empty lattice (single source/sink, no edges)") {
    std::vector<int> row_ptr = {0, 0};
    std::vector<int> col_idx = {};
    std::vector<int> topo    = {0};
    std::vector<int> off     = {0, 1};
    std::vector<float> feats = {0.0f};
    Lattice lat(row_ptr, col_idx, topo, off, feats, 1);

    TopKDecoder dec;
    UniformScorer scorer;
    auto results = dec.decode(lat, 4, scorer);
    REQUIRE(results.size() == 1);
    // Single node, zero edges -> one trivial path of length 1, score 0.
    REQUIRE(results[0].size() == 1);
    CHECK(results[0][0].score == 0.0f);
    CHECK(results[0][0].nodes == std::vector<int>{0});
}

TEST_CASE("single-edge lattice") {
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats = {0.0f, 0.0f};
    Lattice lat(row_ptr, col_idx, topo, off, feats, 1);

    TopKDecoder dec;
    UniformScorer scorer;
    auto r = dec.decode(lat, 4, scorer);
    REQUIRE(r[0].size() == 1);
    CHECK(r[0][0].score == 1.0f);  // one edge of weight 1
    CHECK(r[0][0].nodes == std::vector<int>{0, 1});
}

TEST_CASE("diamond at K=2 with uniform scorer: two equal-score paths") {
    auto lat = make_diamond_with_features({0, 0, 0, 0});
    TopKDecoder dec;
    UniformScorer scorer;
    auto r = dec.decode(lat, 4, scorer);
    REQUIRE(r[0].size() == 2);
    // Both paths score 2.0
    CHECK(r[0][0].score == doctest::Approx(2.0f));
    CHECK(r[0][1].score == doctest::Approx(2.0f));
    // Their node sets are {0,1,3} and {0,2,3} in *some* order.
    auto p1 = r[0][0].nodes;
    auto p2 = r[0][1].nodes;
    CHECK(p1.size() == 3);
    CHECK(p2.size() == 3);
    // Middle nodes must be 1 and 2 (in some order under our tie-break:
    // parent_node ascending, so the entry with parent 1 comes before parent 2)
    CHECK(((p1[1] == 1 && p2[1] == 2) || (p1[1] == 2 && p2[1] == 1)));
}

TEST_CASE("diamond with log-linear scorer: scorer picks the right branch") {
    // feat[v] is one float. weight = +1, bias = 0. Edge score = feat[dst].
    // Path 0-1-3: edge to 1 (feat 5.0) + edge to 3 (feat 2.0) = 7.0
    // Path 0-2-3: edge to 2 (feat 1.0) + edge to 3 (feat 2.0) = 3.0
    auto lat = make_diamond_with_features({0.0f, 5.0f, 1.0f, 2.0f});
    TopKDecoder dec;
    LogLinearScorer scorer({1.0f}, 0.0f);
    auto r = dec.decode(lat, 2, scorer);
    REQUIRE(r[0].size() == 2);
    CHECK(r[0][0].score == doctest::Approx(7.0f));
    CHECK(r[0][1].score == doctest::Approx(3.0f));
    CHECK(r[0][0].nodes == std::vector<int>{0, 1, 3});
    CHECK(r[0][1].nodes == std::vector<int>{0, 2, 3});
}

TEST_CASE("K=1 special case: decoder collapses to single-best Viterbi") {
    auto lat = make_diamond_with_features({0.0f, 5.0f, 1.0f, 2.0f});
    TopKDecoder dec;
    LogLinearScorer scorer({1.0f}, 0.0f);
    auto r = dec.decode(lat, 1, scorer);
    REQUIRE(r[0].size() == 1);
    CHECK(r[0][0].score == doctest::Approx(7.0f));
    CHECK(r[0][0].nodes == std::vector<int>{0, 1, 3});
}

TEST_CASE("high branching factor: 1 source -> 20 mids -> 1 sink") {
    const int B = 20;
    std::vector<int>   row_ptr;
    std::vector<int>   col_idx;
    std::vector<int>   topo;
    std::vector<float> feats;

    // node 0 = source, nodes 1..B = mids, node B+1 = sink
    // edges: 0->1..B, then 1..B -> B+1
    row_ptr.push_back(0);                       // node 0 row start
    for (int i = 1; i <= B; ++i) col_idx.push_back(i);
    row_ptr.push_back((int)col_idx.size());     // end of node 0

    for (int m = 1; m <= B; ++m) {
        col_idx.push_back(B + 1);
        row_ptr.push_back((int)col_idx.size()); // end of node m
    }
    row_ptr.push_back((int)col_idx.size());     // sink has no outgoing

    topo.push_back(0);
    for (int i = 0; i < B; ++i) topo.push_back(1);
    topo.push_back(2);

    feats.resize(B + 2);
    feats[0] = 0.0f;
    for (int i = 1; i <= B; ++i) feats[i] = (float)i;  // distinct per mid
    feats[B + 1] = 0.0f;

    std::vector<int> off = {0, B + 2};
    Lattice lat(row_ptr, col_idx, topo, off, feats, /*feat_dim=*/1);

    TopKDecoder dec;
    LogLinearScorer scorer({1.0f}, 0.0f);
    const int K = 32;
    auto r = dec.decode(lat, K, scorer);

    REQUIRE((int)r[0].size() == std::min(K, B));   // 20 paths only
    // Top path goes through mid B (highest feat -> highest edge score)
    CHECK(r[0][0].nodes[1] == B);
    CHECK(r[0][0].score == doctest::Approx((float)B));
    // Scores should be strictly decreasing
    for (size_t i = 1; i < r[0].size(); ++i) {
        CHECK(r[0][i].score < r[0][i - 1].score);
    }
}

TEST_CASE("k=1 result equals partial result of k=N for the same lattice") {
    // Important property: the GPU's K=1 kernel (week 4) must agree with
    // the K2 kernel's rank-0 output (week 6). Verify the property on CPU.
    auto lat = make_diamond_with_features({0.0f, 5.0f, 1.0f, 2.0f});

    TopKDecoder dec1, dec5;
    LogLinearScorer scorer({1.0f}, 0.0f);
    auto r1 = dec1.decode(lat, 1, scorer);
    auto r5 = dec5.decode(lat, 5, scorer);

    REQUIRE(!r1[0].empty());
    REQUIRE(!r5[0].empty());
    CHECK(r1[0][0].score == doctest::Approx(r5[0][0].score));
    CHECK(r1[0][0].nodes == r5[0][0].nodes);
}

TEST_CASE("backpointer reconstruction round-trips through topk_table") {
    auto lat = make_diamond_with_features({0.0f, 5.0f, 1.0f, 2.0f});
    TopKDecoder dec;
    LogLinearScorer scorer({1.0f}, 0.0f);
    auto r = dec.decode(lat, 2, scorer);

    const auto& table = dec.topk_table();
    REQUIRE(table.size() == 4);
    // Sink node = 3, two entries
    REQUIRE(table[3].size() == 2);
    // Entry 0 should backpoint to node 1 (the better branch), rank 0
    CHECK(table[3][0].parent_node == 1);
    CHECK(table[3][0].parent_rank == 0);
    // Entry 1 should backpoint to node 2, rank 0
    CHECK(table[3][1].parent_node == 2);
    CHECK(table[3][1].parent_rank == 0);

    // Manual reconstruction matches decode()'s output
    auto path0 = dec.reconstruct(3, 0);
    auto path1 = dec.reconstruct(3, 1);
    CHECK(path0 == r[0][0].nodes);
    CHECK(path1 == r[0][1].nodes);
}
