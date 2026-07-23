// cushr/scorer.hpp
//
// Edge-scoring functions. The decoder treats scores as additive log-probs:
// higher is better, path score = sum of edge scores.
//
// All three implementations in the PDF are here:
//   - UniformScorer     : every edge scores 1.0 (sanity check)
//   - LengthScorer      : log-prob proportional to dest word length
//   - LogLinearScorer   : hand-tuned dot product over edge features
//
// In week 9 we add a fourth (the trained biaffine) which trivially fits
// behind this interface.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cushr {

class Lattice;  // fwd

class EdgeScorer {
public:
    virtual ~EdgeScorer() = default;
    virtual float score(const Lattice& lat, int edge_id) const = 0;
    virtual std::string name() const = 0;
};

// Every edge scores 1.0. Used to verify topology: at K=1, every path from
// source to sink scores out_path_length, and the top-1 path under this
// scorer simply selects the path with the most nodes.
class UniformScorer : public EdgeScorer {
public:
    float score(const Lattice&, int) const override { return 1.0f; }
    std::string name() const override { return "uniform"; }
};

// Score proportional to log(word_length(dst)). Fewer words -> longer words
// per word -> higher path score, which matches the PDF's stated bias.
class LengthScorer : public EdgeScorer {
public:
    float score(const Lattice& lat, int edge_id) const override;
    std::string name() const override { return "length"; }
};

// Hand-tuned linear combination of edge features. For week 3 we synthesise
// edge features from the dst node's feature vector (the lattice carries no
// per-edge features yet). Caller supplies the weight vector; its length must
// match lattice.feat_dim() exactly.
class LogLinearScorer : public EdgeScorer {
public:
    LogLinearScorer(std::vector<float> weights, float bias = 0.0f);
    float score(const Lattice& lat, int edge_id) const override;
    std::string name() const override { return "log_linear"; }

    const std::vector<float>& weights() const { return weights_; }
    float bias() const { return bias_; }

private:
    std::vector<float> weights_;
    float              bias_;
};

// Week-9 trained scorer:  score(u -> v) = <W_s x_u, W_d x_v> + b, where
//   x(v) = [ node_features[v] (43 morph one-hots), log1p(word_length[v]) ]
// so feat_dim is lattice.feat_dim() + 1. Weights come from
// cushr_train/export_weights.py --bin.
//
// Projections are cached lazily per node: a node is the source of ~14 edges on
// average, so projecting on every score() call would redo that work 14 times.
class BiaffineScorer : public EdgeScorer {
public:
    static BiaffineScorer load(const std::string& bin_path);

    float score(const Lattice& lat, int edge_id) const override;
    std::string name() const override { return "biaffine"; }

    int feat_dim() const { return feat_dim_; }
    int hidden() const { return hidden_; }

private:
    int feat_dim_ = 0;
    int hidden_   = 0;
    float bias_   = 0.0f;
    std::vector<float> src_proj_;  // [hidden_ * feat_dim_], row-major
    std::vector<float> dst_proj_;

    // Projection cache over a sliding window of node ids. Caching all 4.5M
    // nodes would cost 2*hidden*4 B each (4.6 GB at hidden=128); the decoder
    // walks sentences in node-id order and no sentence exceeds ~400 nodes, so
    // a window this size gets essentially every hit for ~8 MB.
    static constexpr int kWindow = 1 << 13;  // nodes covered = 2 * kWindow

    mutable std::vector<float>   proj_cache_;  // [2 * kWindow][2 * hidden_]
    mutable std::vector<uint8_t> proj_valid_;
    mutable int win_base_ = -1;
    mutable std::vector<float> src_scratch_;  // [hidden_], see score()

    const float* project_(const Lattice& lat, int v) const;
};

}  // namespace cushr
