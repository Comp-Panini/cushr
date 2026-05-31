// cushr/metrics.hpp
//
// Word-level Precision/Recall/F1 and sentence-level Perfect Match.
//
// "Word" here means a (start, length, surface_id) triple identifying a
// specific node in the lattice. Two segmentations match iff they select the
// same set of nodes -- order is determined by topology, so set equality is
// enough. We use the node id itself as the equality key; it encodes
// position+surface uniquely within a sentence.

#pragma once

#include <vector>

namespace cushr {

class Lattice;

struct WordMetrics {
    double precision;
    double recall;
    double f1;
    long   tp;
    long   fp;
    long   fn;
    long   perfect_match;   // sentences with PM == 1
    long   num_sentences;
};

// gold_paths_per_sentence[s] : list of node ids on the gold path (or empty
//   to skip that sentence).
// pred_paths_per_sentence[s] : list of node ids on the predicted top-1 path.
WordMetrics evaluate_word_metrics(
    const std::vector<std::vector<int>>& gold_paths_per_sentence,
    const std::vector<std::vector<int>>& pred_paths_per_sentence);

// Extract the gold path for sentence s as a list of node ids by walking
// gold_path_mask in topological order. The mask is expected to form a
// connected source-to-sink path; if it does not, the returned vector is
// empty.
std::vector<int> gold_path_nodes(const Lattice& lat, int sentence_idx);

}  // namespace cushr
