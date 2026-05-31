// cushr/decoder.cpp

#include "cushr/decoder.hpp"

#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"

#include <algorithm>
#include <cassert>

namespace cushr {

namespace {

// Strict-weak ordering for partial_sort: best score first; ties broken by
// (parent_node, parent_rank) ascending. This must be mirrored on the GPU.
struct EntryGreater {
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.score != b.score) return a.score > b.score;
        if (a.parent_node != b.parent_node) return a.parent_node < b.parent_node;
        return a.parent_rank < b.parent_rank;
    }
};

}  // namespace

std::vector<std::vector<DecodedPath>>
TopKDecoder::decode(const Lattice& lat, int K, const EdgeScorer& scorer) {
    assert(K >= 1);

    const int N = lat.num_nodes();
    topk_.assign(N, {});

    std::vector<std::vector<DecodedPath>> results(lat.num_sentences());

    // We process sentences independently. Within a sentence we iterate nodes
    // in topological order. We *could* sweep all sentences in one pass since
    // edges don't cross boundaries -- that's exactly what the GPU will do
    // in week 8 -- but for the CPU oracle, sentence-at-a-time is easier to
    // reason about and equivalently correct.
    std::vector<Entry> buf;
    buf.reserve(64 * K);

    for (int s = 0; s < lat.num_sentences(); ++s) {
        const auto sv  = lat.sentence(s);
        const auto ord = lat.topo_order_for_sentence(s);

        for (int v : ord) {
            // Source nodes start with one entry: score 0, no parent.
            if (lat.in_degree(v) == 0) {
                topk_[v].push_back({0.0f, -1, -1});
                continue;
            }

            // Collect: for each incoming edge e=(u,v), for each rank r in
            // u's top-K, produce candidate (parent.score + s_{u,v}, u, r).
            buf.clear();
            for (int er = lat.in_edge_begin(v); er < lat.in_edge_end(v); ++er) {
                const int u   = lat.in_edge_src(er);
                const int eid = lat.in_edge_forward_id(er);
                const float es = scorer.score(lat, eid);
                const auto& parent = topk_[u];
                // parent could be empty if u was unreachable from the source.
                // That can happen with malformed lattices; treat u as
                // contributing nothing in that case.
                for (int r = 0; r < (int)parent.size(); ++r) {
                    buf.push_back({parent[r].score + es, u, r});
                }
            }

            if (buf.empty()) {
                // unreachable; topk_[v] stays empty
                continue;
            }

            const int keep = std::min<int>(K, (int)buf.size());
            std::partial_sort(buf.begin(), buf.begin() + keep, buf.end(),
                              EntryGreater{});
            topk_[v].assign(buf.begin(), buf.begin() + keep);
        }

        // Materialise paths only at the sink.
        const auto& sink_entries = topk_[sv.sink_node];
        results[s].reserve(sink_entries.size());
        for (int r = 0; r < (int)sink_entries.size(); ++r) {
            DecodedPath p;
            p.score = sink_entries[r].score;
            p.nodes = reconstruct(sv.sink_node, r);
            results[s].push_back(std::move(p));
        }
    }

    return results;
}

std::vector<int> TopKDecoder::reconstruct(int sink_node, int rank) const {
    std::vector<int> rev;
    int v = sink_node;
    int r = rank;
    while (v >= 0) {
        rev.push_back(v);
        if (r < 0 || r >= (int)topk_[v].size()) break;
        const Entry& e = topk_[v][r];
        v = e.parent_node;
        r = e.parent_rank;
    }
    std::reverse(rev.begin(), rev.end());
    return rev;
}

}  // namespace cushr
