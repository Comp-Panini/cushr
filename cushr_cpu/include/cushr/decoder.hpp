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

// per-decode statistics on how many candidates each node actually kept.
// "keep" = min(K, buf.size()); keep < K means the beam was under-full (fewer
// than K candidate paths reached that node) -- the divergence from K.
struct KeepStats {
    int K = 0;
    std::vector<long long> hist;   // hist[k] = # nodes whose keep == k, size K+1
    long long total_nodes = 0;     // non-source nodes that ran the keep step
    long long full = 0;            // keep == K
    long long under = 0;           // keep <  K
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

    // keep-value stats gathered during the most recent decode() call
    const KeepStats& keep_stats() const { return keep_stats_; }

private:
    // topk_[v] has up to K entries, sorted descending by score
    std::vector<std::vector<Entry>> topk_;

    // populated by decode(): distribution of per-node keep values
    KeepStats keep_stats_;
};

}  
