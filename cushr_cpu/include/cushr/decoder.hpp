// cushr/decoder.hpp
//
// Top-K lattice decoder, CPU reference oracle for the GPU work in weeks 4+.
//
// Representation of paths is via backpointers, NOT via stored vector<int>:
//
//   struct Entry { float score; int parent_node; int parent_rank; };
//
// At each node v we store at most K entries sorted descending by score.
// Entry i at node v says: "to reach v with the i-th best score, take entry
// `parent_rank` at node `parent_node` and follow the edge to v."
//
// This is *exactly* the representation the GPU kernel will produce in
// week 6, which makes the regression test in week 4 a direct triple-by-triple
// comparison.
//
// Tie-break policy: when two candidates tie on score, we break by
//   (parent_node, parent_rank)
// in ascending order. This is deterministic and easy to mirror on the GPU.
// See decoder.cpp for the comparator.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace cushr {

class Lattice;
class EdgeScorer;

struct Entry {
    float score;
    int   parent_node;   // -1 for source nodes
    int   parent_rank;   // -1 for source nodes

    // Score-then-(parent_node, parent_rank) ordering. Strictly speaking we
    // want descending by score for partial_sort, so the comparator passed
    // there is the inverse of this; this operator is exposed for equality
    // tests.
    bool operator==(const Entry& o) const {
        return score == o.score
            && parent_node == o.parent_node
            && parent_rank == o.parent_rank;
    }
};

// A fully reconstructed path: list of node ids from source to sink.
struct DecodedPath {
    float            score;
    std::vector<int> nodes;  // includes both source and sink
};

class TopKDecoder {
public:
    // Decode every sentence in `lat` independently. Returned vector has one
    // element per sentence. Each sentence's vector has up to K paths,
    // sorted descending by score.
    //
    // We do NOT materialise paths for every node -- only for the sink of
    // each sentence. Intermediate nodes keep only their (score, parent,
    // rank) triples in `topk_`.
    std::vector<std::vector<DecodedPath>>
    decode(const Lattice& lat, int K, const EdgeScorer& scorer);

    // After decode(), the full triple table is available for regression
    // tests (and for GPU bit-identity comparison in week 4).
    // topk_[v] holds up to K entries sorted descending by score.
    const std::vector<std::vector<Entry>>& topk_table() const { return topk_; }

    // Reconstruct path i ending at node v by walking backpointers. Returns
    // the node sequence from the sentence source to v (inclusive). Useful
    // for tests and for downstream consumers that ask for a specific rank.
    std::vector<int> reconstruct(int sink_node, int rank) const;

private:
    // topk_[v] : up to K entries, sorted descending by score.
    std::vector<std::vector<Entry>> topk_;
};

}  // namespace cushr
