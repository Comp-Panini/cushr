// cushr/scorer.cpp

#include "cushr/scorer.hpp"

#include "cushr/lattice.hpp"

#include <cmath>
#include <stdexcept>

namespace cushr {

// LengthScorer ---------------------------------------------------------------
float LengthScorer::score(const Lattice& lat, int edge_id) const {
    const int v = lat.edge_dst(edge_id);
    const int len = lat.node_word_length(v);
    // log(1 + len) keeps the scorer well-defined when word_length is unknown
    // (returns 0). Bias toward longer words follows from monotonicity.
    return std::log(1.0f + (float)len);
}

// LogLinearScorer ------------------------------------------------------------
LogLinearScorer::LogLinearScorer(std::vector<float> weights, float bias)
    : weights_(std::move(weights)), bias_(bias) {}

float LogLinearScorer::score(const Lattice& lat, int edge_id) const {
    const int v = lat.edge_dst(edge_id);
    if ((int)weights_.size() != lat.feat_dim()) {
        throw std::runtime_error(
            "LogLinearScorer: weights dim does not match lattice feat_dim");
    }
    const float* f = lat.node_feature_ptr(v);
    float acc = bias_;
    for (int i = 0; i < (int)weights_.size(); ++i) {
        acc += weights_[i] * f[i];
    }
    return acc;
}

}  // namespace cushr
