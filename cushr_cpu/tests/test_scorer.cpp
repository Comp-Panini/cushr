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

// ---- week 9: biaffine ---------------------------------------------------
//
// The in-memory Lattice ctor carries no word_length array, so
// node_word_length() is 0 and the appended log1p(len) feature is log1p(0) = 0.
// That makes the expected value exactly the 3-feature biaffine dot product.

#include "cushr/scorer.hpp"

#include <cstdio>
#include <cstdint>
#include <fstream>

using cushr::BiaffineScorer;

namespace {

// Write the blob export_weights.py --bin produces.
std::string write_test_model(const std::string& path,
                             int feat_dim, int hidden, float bias,
                             const std::vector<float>& src,
                             const std::vector<float>& dst,
                             int32_t magic = 0x43534233) {  // 'CSB3'
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&feat_dim), 4);
    out.write(reinterpret_cast<const char*>(&hidden), 4);
    out.write(reinterpret_cast<const char*>(&bias), 4);
    out.write(reinterpret_cast<const char*>(src.data()), src.size() * 4);
    out.write(reinterpret_cast<const char*>(dst.data()), dst.size() * 4);
    return path;
}

}  // namespace

TEST_CASE("biaffine scorer matches the hand-computed bilinear form") {
    // 2 nodes, lattice feat_dim = 3. The model's feat_dim must equal it
    // exactly: the featurizer emits every column and the scorer appends none.
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats = {
        1.0f, 0.0f, 2.0f,   // node 0
        0.0f, 3.0f, 1.0f,   // node 1
    };
    Lattice lat(row_ptr, col_idx, topo, off, feats, 3);

    // hidden = 2, feat_dim = 3
    std::vector<float> src = {1.0f, 0.0f, 0.5f,
                              0.0f, 1.0f, 0.0f};
    std::vector<float> dst = {2.0f, 0.0f, 1.0f,
                              0.0f, 1.0f, 1.0f};
    const std::string path = "test_model_biaffine.bin";
    write_test_model(path, 3, 2, /*bias=*/0.5f, src, dst);

    auto s = BiaffineScorer::load(path);
    CHECK(s.feat_dim() == 3);
    CHECK(s.hidden() == 2);

    // x_u = [1,0,2] -> W_s x_u = [1*1 + 0.5*2, 0] = [2, 0]
    // x_v = [0,3,1] -> W_d x_v = [1*1, 3*1 + 1*1] = [1, 4]
    // <[2,0],[1,4]> + 0.5 = 2.5
    CHECK(s.score(lat, 0) == doctest::Approx(2.5f));
    std::remove(path.c_str());
}

TEST_CASE("biaffine scorer rejects a stale CSB2 weight file") {
    // CSB2 weights assume the scorer appends log1p(word_length) itself. Loading
    // them against a CSB3 scorer would not fail on shape -- it would quietly
    // score every edge with the wrong feature vector -- so the version check
    // has to be the thing that catches it.
    std::vector<float> src = {1.0f, 0.0f, 0.5f, 7.0f};
    std::vector<float> dst = {2.0f, 0.0f, 1.0f, 9.0f};
    const std::string path = "test_model_csb2.bin";
    write_test_model(path, 4, 1, /*bias=*/0.0f, src, dst, /*magic=*/0x43534232);
    CHECK_THROWS(BiaffineScorer::load(path));
    std::remove(path.c_str());
}

TEST_CASE("biaffine scorer rejects a feat_dim that disagrees with the lattice") {
    // The featurizer that built the .npz and the one the model was trained on
    // must be the same; a width mismatch is the cheapest detectable symptom.
    std::vector<int> row_ptr = {0, 1, 1};
    std::vector<int> col_idx = {1};
    std::vector<int> topo    = {0, 1};
    std::vector<int> off     = {0, 2};
    std::vector<float> feats = {1.0f, 0.0f, 2.0f,
                                0.0f, 3.0f, 1.0f};
    Lattice lat(row_ptr, col_idx, topo, off, feats, 3);

    std::vector<float> src = {1.0f, 0.0f, 0.5f, 7.0f};
    std::vector<float> dst = {2.0f, 0.0f, 1.0f, 9.0f};
    const std::string path = "test_model_widthmismatch.bin";
    write_test_model(path, 4, 1, /*bias=*/0.0f, src, dst);
    auto s = BiaffineScorer::load(path);
    CHECK_THROWS(s.score(lat, 0));
    std::remove(path.c_str());
}

TEST_CASE("biaffine scorer rejects a non-model file") {
    const std::string path = "test_model_garbage.bin";
    { std::ofstream out(path, std::ios::binary); out << "not a model at all"; }
    CHECK_THROWS(BiaffineScorer::load(path));
    std::remove(path.c_str());
}

TEST_CASE("edge_src recovers the CSR source node") {
    std::vector<int> row_ptr = {0, 2, 3, 3};
    std::vector<int> col_idx = {1, 2, 2};
    std::vector<int> topo    = {0, 1, 2};
    std::vector<int> off     = {0, 3};
    std::vector<float> feats(3, 0.0f);
    Lattice lat(row_ptr, col_idx, topo, off, feats, 1);
    CHECK(lat.edge_src(0) == 0);
    CHECK(lat.edge_src(1) == 0);
    CHECK(lat.edge_src(2) == 1);
}
