#include "cushr/decoder.hpp"

#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"

#include <algorithm>
#include <cassert>

namespace cushr {

namespace {

// level of priority of ranking best paths
struct EntryGreater {
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.score != b.score) return a.score > b.score; // 1 score
        if (a.parent_node != b.parent_node) return a.parent_node < b.parent_node; // 2 parent_node is ranked lower (better)
        return a.parent_rank < b.parent_rank; // 3 parent_rank is ranked lower (better)
    }
};

}  

// the actual decoder
std::vector<std::vector<DecodedPath>> TopKDecoder::decode(const Lattice& lat, int K, const EdgeScorer& scorer) {
    assert(K >= 1); // k must be valid value

    const int N = lat.num_nodes(); // N = total number of nodes

    topk_.assign(N, {}); // vector of vectors, there are N internal vectors that are all empty

    //one slot per sentence (out of 119k)
    // results[s] = list of up to K decodedPath structs for sentence sorted best-first
    std::vector<std::vector<DecodedPath>> results(lat.num_sentences());

    // process sentnence one at a time
    std::vector<Entry> buf;
    // 64 is a heuristic upper bound on average in-degree per node
    // 64*k is max degree per node x # things to be ranked
    // avoids repeated small reallocations
    buf.reserve(64 * K);

    for (int s = 0; s < lat.num_sentences(); ++s) {
        const auto sv  = lat.sentence(s); // sv = sentence
        const auto ord = lat.topo_order_for_sentence(s); // ord is all nodes in each topo level

        for (int v : ord) {
            if (lat.in_degree(v) == 0) { // if src node
                topk_[v].push_back({0.0f, -1, -1}); // score 0, no parent, no parent rank
                continue;
            }

            // for each incoming edge e=(u,v)
            // for each rank r in u's top-K
            // produce candidate (parent.score + s_{u,v}, u, r).
            buf.clear();

            // for each incoming edge
            for (int er = lat.in_edge_begin(v); er < lat.in_edge_end(v); ++er) {
                const int u = lat.in_edge_src(er); // u = src for this edge
                const int eid = lat.in_edge_forward_id(er); // edge id
                const float es = scorer.score(lat, eid); // score of the edge
                
                // parent is the whole array of topk at its index
                const auto& parent = topk_[u];

                // for each rank r in u's top-K
                for (int r = 0; r < (int)parent.size(); ++r) {

                    // constructs a candidate entry, appends to the buffer
                    buf.push_back({parent[r].score + es, u, r});
                }
            }

            if (buf.empty()) {
                continue;
            }

            // number of things to keep is the min of K or size of buf
            const int keep = std::min<int>(K, (int)buf.size());

            // sort the top K, using entryGreater operator defined above
            std::partial_sort(buf.begin(), buf.begin() + keep, buf.end(), EntryGreater{});

            // assign this new node-specific topK ranking to that node's vector in the main topk
            topk_[v].assign(buf.begin(), buf.begin() + keep);
        }

        // create paths only at the sink
        const auto& sink_entries = topk_[sv.sink_node];

        results[s].reserve(sink_entries.size()); // the amount of sink entries is how many results
        
        for (int r = 0; r < (int)sink_entries.size(); ++r) {
            DecodedPath p;
            p.score = sink_entries[r].score; // for each path, get score and nodes and sort
            p.nodes = reconstruct(sv.sink_node, r);
            results[s].push_back(std::move(p));
        }
    }

    return results;
}


std::vector<int> TopKDecoder::reconstruct(int sink_node, int rank) const {
    std::vector<int> rev; // this is a arr of node IDs
    int v = sink_node;
    int r = rank;
    while (v >= 0) { // walk backpointer chain back, appending nodes in reverse order
        // keep walking back until there is no parent node meaning source (-ve value for v)
        rev.push_back(v); 
        if (r < 0 || r >= (int)topk_[v].size()) break; // if rank is invalid break
        const Entry& e = topk_[v][r]; // the node
        v = e.parent_node; // replace
        r = e.parent_rank;
    }

    std::reverse(rev.begin(), rev.end()); // reverse to return correct path
    return rev;
}

}  
