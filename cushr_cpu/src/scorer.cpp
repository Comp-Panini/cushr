#include "cushr/scorer.hpp"

#include "cushr/lattice.hpp"

#include <cmath>
#include <stdexcept>

namespace cushr {

float LengthScorer::score(const Lattice& lat, int edge_id) const {
    const int v = lat.edge_dst(edge_id); // v = destination node of the given edge

    const int len = lat.node_word_length(v);

    return std::log(1.0f + (float)len); // 1 + log(len) function 
}

LogLinearScorer::LogLinearScorer(std::vector<float> weights, float bias)
    : weights_(std::move(weights)), bias_(bias) {}

float LogLinearScorer::score(const Lattice& lat, int edge_id) const {
    const int v = lat.edge_dst(edge_id); // v = destination node of the given edge

    // if it doesnt have all 43 features/weights, bad
    if ((int)weights_.size() != lat.feat_dim()) {
        throw std::runtime_error(
            "LogLinearScorer: weights dim does not match lattice feat_dim");
    }

    // f points to the contiguous feature row for node v in the flat
    // node_features array (length feat_dim). acc accumulates bias + w · f,
    const float* f = lat.node_feature_ptr(v);
    float acc = bias_;
    for (int i = 0; i < (int)weights_.size(); ++i) {
        acc += weights_[i] * f[i];
    }
    return acc;
}

}  // namespace cushr
