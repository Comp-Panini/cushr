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

}  // namespace cushr
