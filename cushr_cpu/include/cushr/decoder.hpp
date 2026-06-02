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

    bool operator==(const Entry& o) const {
        return score == o.score
            && parent_node == o.parent_node
            && parent_rank == o.parent_rank;
    }
};

// list of node ids from source to sink and including both
struct DecodedPath {
    float score;
    std::vector<int> nodes; 
};

class TopKDecoder {
public:
    // decode every sentence in lattice independently
    // output: results which is vector < vector <DecodedPath>> which has S vectors each with K decoded paths
    // outer vector = 1 slot per sentence
    // inner vector has <= K DecodedPath structs which contain score and vector<int> nodes
    // also populates topk
    std::vector<std::vector<DecodedPath>> decode(const Lattice& lat, int K, const EdgeScorer& scorer);

    // takes the backpointer table as input
    // output is a vector<vector<entry>> of size N
    // for every node it contains up to K Entry structs containing 3 pieces of info
    const std::vector<std::vector<Entry>>& topk_table() const { 
        return topk_; 
    }

    // given the nodeID of last word (sink) of sentence and K
    // output is a vector<int> of the path
    std::vector<int> reconstruct(int sink_node, int rank) const;

private:
    // topk_[v] has up to K entries, sorted descending by score
    std::vector<std::vector<Entry>> topk_;
};

}  
