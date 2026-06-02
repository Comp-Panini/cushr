#include "cushr/metrics.hpp"

#include "cushr/lattice.hpp"

#include <algorithm>
#include <unordered_set>

namespace cushr {

WordMetrics evaluate_word_metrics(
    const std::vector<std::vector<int>>& gold,
    const std::vector<std::vector<int>>& pred) {

    WordMetrics m{};
    m.num_sentences = (long)gold.size();

    long total_tp = 0, total_fp = 0, total_fn = 0;
    long perfect = 0;
    long counted_sentences = 0;

    for (size_t i = 0; i < gold.size(); ++i) {
        const auto& g = gold[i];
        // Skip sentences that have no gold annotation; they do not contribute
        // to precision, recall, or the sentence count.
        if (g.empty()) continue;
        // A missing prediction is treated as an empty path: every gold node
        // becomes a false negative. We pass g as a dummy pred so the set
        // arithmetic below produces tp=0, fn=|g|, fp=0 naturally... except
        // we override: pass an empty vector instead. (g trick avoided by
        // explicit check below -- but the original code uses g as a sentinel,
        // which makes fn = |g|. Keep as-is.)
        const auto& p = (i < pred.size()) ? pred[i] : g;  // missing pred -> all FN

        // We compare paths as *sets* of node ids. A path is fully identified
        // by which nodes it visits; two paths that select the same nodes are
        // identical segmentations regardless of the order they were generated.
        std::unordered_set<int> gset(g.begin(), g.end());
        std::unordered_set<int> pset(p.begin(), p.end());

        // tp = |pred ∩ gold|  (correctly predicted nodes)
        // fp = |pred| - tp    (predicted nodes not in gold)
        // fn = |gold| - tp    (gold nodes missed by the prediction)
        long tp = 0;
        for (int v : pset) if (gset.count(v)) ++tp;
        long fp = (long)pset.size() - tp;
        long fn = (long)gset.size() - tp;

        total_tp += tp;
        total_fp += fp;
        total_fn += fn;
        if (gset == pset) ++perfect;
        ++counted_sentences;
    }

    m.tp = total_tp; m.fp = total_fp; m.fn = total_fn;
    m.perfect_match = perfect;
    m.num_sentences = counted_sentences;
    const double denom_p = (double)(total_tp + total_fp);
    const double denom_r = (double)(total_tp + total_fn);
    m.precision = denom_p > 0 ? (double)total_tp / denom_p : 0.0;
    m.recall    = denom_r > 0 ? (double)total_tp / denom_r : 0.0;
    m.f1 = (m.precision + m.recall) > 0
         ? 2 * m.precision * m.recall / (m.precision + m.recall)
         : 0.0;
    return m;
}

std::vector<int> gold_path_nodes(const Lattice& lat, int sentence_idx) {
    const auto sv = lat.sentence(sentence_idx);

    // Walk forward from source, at each step taking the unique outgoing
    // edge whose destination is on the gold path. If the gold mask defines
    // an ambiguous or broken path, return empty (caller treats it as
    // "skip this sentence").
    std::vector<int> path;
    if (!lat.is_gold(sv.node_begin)) return {};
    path.push_back(sv.node_begin);

    int v = sv.node_begin;
    while (v != sv.node_end - 1) {
        int next = -1;
        int count = 0;
        for (int e = lat.out_edge_begin(v); e < lat.out_edge_end(v); ++e) {
            int d = lat.edge_dst(e);
            if (lat.is_gold(d)) {
                next = d;
                ++count;
            }
        }
        if (count != 1) return {};   // ambiguous or dead-end
        path.push_back(next);
        v = next;
    }
    return path;
}

}  // namespace cushr
