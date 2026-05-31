// apps/evaluate.cpp
//
// Run the CPU reference decoder over the SIGHUM lattice batch and report
// word-level F1 + Perfect Match. Optionally dump a golden-output JSON for
// the first N sentences (default 500), which becomes the regression oracle
// for week 4's GPU port.
//
// Usage:
//   cushr_evaluate <lattice.npz>
//                  [--scorer uniform|length|log_linear]
//                  [--weights w0,w1,...]
//                  [--bias b]
//                  [--K 10]
//                  [--golden golden_outputs.json]
//                  [--golden-n 500]

#include "cushr/decoder.hpp"
#include "cushr/json_io.hpp"
#include "cushr/lattice.hpp"
#include "cushr/metrics.hpp"
#include "cushr/scorer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> parse_csv_floats(const std::string& s) {
    std::vector<float> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stof(item));
    }
    return out;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <lattice.npz> [--scorer uniform|length|log_linear]\n"
        "           [--weights w0,w1,...] [--bias b]\n"
        "           [--K 10] [--golden out.json] [--golden-n 500]\n", prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string npz_path  = argv[1];
    std::string scorer_kind = "log_linear";
    std::string weights_csv;
    float bias = 0.0f;
    int   K    = 10;
    std::string golden_path;
    int   golden_n = 500;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--scorer")        scorer_kind = need("--scorer");
        else if (a == "--weights")  weights_csv = need("--weights");
        else if (a == "--bias")     bias = std::stof(need("--bias"));
        else if (a == "--K")        K = std::stoi(need("--K"));
        else if (a == "--golden")   golden_path = need("--golden");
        else if (a == "--golden-n") golden_n = std::stoi(need("--golden-n"));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    std::cout << "Loading lattice from " << npz_path << " ...\n";
    auto t0 = std::chrono::steady_clock::now();
    cushr::Lattice lat = cushr::Lattice::load_npz(npz_path);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "  nodes="     << lat.num_nodes()
              << " edges="      << lat.num_edges()
              << " sentences="  << lat.num_sentences()
              << " feat_dim="   << lat.feat_dim()
              << " load_ms="
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << "\n";

    // Build scorer
    std::unique_ptr<cushr::EdgeScorer> scorer;
    if (scorer_kind == "uniform") {
        scorer = std::make_unique<cushr::UniformScorer>();
    } else if (scorer_kind == "length") {
        scorer = std::make_unique<cushr::LengthScorer>();
    } else if (scorer_kind == "log_linear") {
        std::vector<float> w = parse_csv_floats(weights_csv);
        if ((int)w.size() != lat.feat_dim()) {
            // Default: first feature gets weight 1, rest 0. Caller should
            // override with a proper hand-tuned vector once they know what
            // their features are.
            std::cerr << "[warn] no --weights given (or wrong length); using e_0\n";
            w.assign(lat.feat_dim(), 0.0f);
            if (!w.empty()) w[0] = 1.0f;
        }
        scorer = std::make_unique<cushr::LogLinearScorer>(std::move(w), bias);
    } else {
        std::fprintf(stderr, "unknown scorer: %s\n", scorer_kind.c_str());
        return 1;
    }
    std::cout << "Scorer: " << scorer->name() << "\n";

    // Decode
    cushr::TopKDecoder dec;
    t0 = std::chrono::steady_clock::now();
    auto results = dec.decode(lat, K, *scorer);
    t1 = std::chrono::steady_clock::now();
    const double wall_sec = std::chrono::duration<double>(t1 - t0).count();
    const double per_sec  = lat.num_sentences() / wall_sec;
    std::cout << "Decoded " << lat.num_sentences() << " sentences in "
              << wall_sec << " s  (" << per_sec << " sent/s)\n";

    // Gather top-1 predictions and gold paths
    std::vector<std::vector<int>> gold(lat.num_sentences());
    std::vector<std::vector<int>> pred(lat.num_sentences());
    for (int s = 0; s < lat.num_sentences(); ++s) {
        gold[s] = cushr::gold_path_nodes(lat, s);
        if (!results[s].empty()) {
            pred[s] = results[s][0].nodes;   // top-1
        }
    }

    auto m = cushr::evaluate_word_metrics(gold, pred);
    std::cout << "\n--- Word-level metrics (top-1) ---\n";
    std::cout << "  sentences (with gold) : " << m.num_sentences << "\n";
    std::cout << "  TP/FP/FN              : " << m.tp << " / " << m.fp << " / " << m.fn << "\n";
    std::cout << "  precision             : " << m.precision << "\n";
    std::cout << "  recall                : " << m.recall    << "\n";
    std::cout << "  F1                    : " << m.f1        << "\n";
    std::cout << "  perfect match         : " << m.perfect_match
              << " / " << m.num_sentences
              << "  (" << (m.num_sentences ? 100.0 * m.perfect_match / m.num_sentences : 0.0)
              << "%)\n";

    // Dump golden outputs
    if (!golden_path.empty()) {
        std::vector<cushr::GoldenSentence> sents;
        const int n = std::min(golden_n, lat.num_sentences());
        sents.reserve(n);
        for (int s = 0; s < n; ++s) {
            cushr::GoldenSentence gs;
            gs.sentence_id = s;
            gs.paths = results[s];   // already at most K
            sents.push_back(std::move(gs));
        }
        cushr::write_golden_outputs_json(golden_path, scorer->name(), K, sents);
        std::cout << "Wrote golden outputs for " << n
                  << " sentences to " << golden_path << "\n";
    }

    return 0;
}
