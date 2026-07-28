#include "cushr/scorer.hpp"

#include "cushr/lattice.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

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
    const float* f = lat.node_feature_ptr(v);
    float acc = bias_;
    for (int i = 0; i < (int)weights_.size(); ++i) {
        acc += weights_[i] * f[i];
    }
    return acc;
}

// ---------------------------------------------------------------- biaffine

namespace {
// 'CSB3'. Bumped from 'CSB2' when featurization moved into
// cushr_train/featurizers.py. Under CSB2 this scorer appended log1p(word
// length) itself as the final feature column; a CSB3 featurizer emits every
// column it wants, including that one. The two conventions produce silently
// different scores from the same weights, so the magic changes to make an
// old/new mismatch a load-time error instead of a numerical mystery.
constexpr int kBiaffineMagic = 0x43534233;  // 'CSB3'
constexpr int kBiaffineMagicV2 = 0x43534232;  // 'CSB2', no longer supported
}

BiaffineScorer BiaffineScorer::load(const std::string& bin_path) {
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("BiaffineScorer: cannot open " + bin_path);
    }

    int32_t magic = 0, feat_dim = 0, hidden = 0;
    float bias = 0.0f;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&feat_dim), 4);
    in.read(reinterpret_cast<char*>(&hidden), 4);
    in.read(reinterpret_cast<char*>(&bias), 4);
    if (!in) {
        throw std::runtime_error("BiaffineScorer: cannot read header of " + bin_path);
    }
    if (magic == kBiaffineMagicV2) {
        throw std::runtime_error(
            "BiaffineScorer: " + bin_path + " is a CSB2 file. Those weights "
            "expect this scorer to append log1p(word_length) as the last "
            "feature; featurization now lives in cushr_train/featurizers.py. "
            "Re-export from a model retrained on a featurized archive.");
    }
    if (magic != kBiaffineMagic) {
        throw std::runtime_error("BiaffineScorer: bad magic in " + bin_path +
                                 " (not an export_weights.py --bin file?)");
    }

    BiaffineScorer s;
    s.feat_dim_ = feat_dim;
    s.hidden_   = hidden;
    s.bias_     = bias;
    const size_t n = (size_t)hidden * (size_t)feat_dim;
    s.src_proj_.resize(n);
    s.dst_proj_.resize(n);
    in.read(reinterpret_cast<char*>(s.src_proj_.data()), n * sizeof(float));
    in.read(reinterpret_cast<char*>(s.dst_proj_.data()), n * sizeof(float));
    if (!in) {
        throw std::runtime_error("BiaffineScorer: truncated weight file " + bin_path);
    }
    return s;
}

// Return the cached row for node v: [ W_s x_v | W_d x_v ], 2*hidden_ floats.
const float* BiaffineScorer::project_(const Lattice& lat, int v) const {
    const int span = 2 * kWindow;
    if (proj_cache_.empty()) {
        proj_cache_.assign((size_t)span * 2 * hidden_, 0.0f);
        proj_valid_.assign(span, 0);
    }
    // Recentre the window when v falls outside it; costs one memset per
    // ~16K nodes of forward progress.
    if (win_base_ < 0 || v < win_base_ || v >= win_base_ + span) {
        win_base_ = (v / kWindow) * kWindow - kWindow;
        if (win_base_ < 0) win_base_ = 0;
        std::memset(proj_valid_.data(), 0, proj_valid_.size());
    }
    const int slot = v - win_base_;
    float* out = proj_cache_.data() + (size_t)slot * 2 * hidden_;
    if (proj_valid_[slot]) return out;

    // x(v) is the featurizer's output verbatim -- nothing is appended here.
    // Any length, position or frequency signal the model relies on is already
    // a column, put there by cushr_train/featurizers.py.
    const float* f = lat.node_feature_ptr(v);
    if (feat_dim_ != lat.feat_dim()) {
        throw std::runtime_error(
            "BiaffineScorer: model feat_dim (" + std::to_string(feat_dim_) +
            ") does not match lattice feat_dim (" +
            std::to_string(lat.feat_dim()) + "); the model and the .npz were "
            "built by different featurizers");
    }

    for (int h = 0; h < hidden_; ++h) {
        const float* ws = src_proj_.data() + (size_t)h * feat_dim_;
        const float* wd = dst_proj_.data() + (size_t)h * feat_dim_;
        float as = 0.0f;
        float ad = 0.0f;
        for (int i = 0; i < feat_dim_; ++i) {
            as += ws[i] * f[i];
            ad += wd[i] * f[i];
        }
        out[h]           = as;
        out[hidden_ + h] = ad;
    }
    proj_valid_[slot] = 1;
    return out;
}

float BiaffineScorer::score(const Lattice& lat, int edge_id) const {
    const int u = lat.edge_src(edge_id);
    const int v = lat.edge_dst(edge_id);
    // Both endpoints normally live in the same window, but a recentre between
    // the two project_ calls would dangle the first pointer -- so copy W_s x_u
    // into scratch before asking for v.
    src_scratch_.resize(hidden_);
    std::memcpy(src_scratch_.data(), project_(lat, u), hidden_ * sizeof(float));
    const float* dv = project_(lat, v) + hidden_;  // W_d x_v
    float acc = bias_;
    for (int h = 0; h < hidden_; ++h) acc += src_scratch_[h] * dv[h];
    return acc;
}

}
