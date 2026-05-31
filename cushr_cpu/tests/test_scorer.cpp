// tests/test_scorer.cpp
#include <doctest/doctest.h>

#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"

#include <cmath>

using cushr::Lattice;
using cushr::LengthScorer;
using cushr::LogLinearScorer;
using cushr::UniformScorer;

TEST_CASE("uniform scorer returns 1.0 for every edge") {
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats(2, 0.0f);
    Lattice lat(row_ptr, col_idx, topo, off, feats, 1);
    UniformScorer s;
    CHECK(s.score(lat, 0) == 1.0f);
}

TEST_CASE("log linear scorer dot products correctly") {
    // 2 nodes, feat_dim=3, single edge 0->1
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats = {
        0.0f, 0.0f, 0.0f,    // node 0
        1.0f, 2.0f, 3.0f,    // node 1
    };
    Lattice lat(row_ptr, col_idx, topo, off, feats, 3);
    LogLinearScorer s({0.5f, -1.0f, 2.0f}, /*bias=*/0.25f);
    // 0.5*1 + (-1)*2 + 2*3 + 0.25 = 4.75
    CHECK(s.score(lat, 0) == doctest::Approx(4.75f));
}

TEST_CASE("log linear scorer rejects mismatched weight dims") {
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats(2 * 2, 0.0f);
    Lattice lat(row_ptr, col_idx, topo, off, feats, 2);
    LogLinearScorer s({1.0f, 2.0f, 3.0f}, 0.0f);  // 3 != 2
    CHECK_THROWS(s.score(lat, 0));
}
