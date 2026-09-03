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
//                  [--keep-report keep_report.csv]
//                  [--csv cpu_bench.csv]      one machine-readable timing row

#include "cushr/decoder.hpp"
#include "cushr/json_io.hpp"
#include "cushr/lattice.hpp"
#include "cushr/metrics.hpp"
#include "cushr/scorer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
        "Usage: %s <lattice.npz> [--scorer uniform|length|log_linear|biaffine]\n"
        "           [--weights w0,w1,...] [--bias b] [--model model.bin]\n"
        "           [--K 10] [--golden out.json] [--golden-n 500]\n"
        "           [--keep-report keep_report.csv] [--csv cpu_bench.csv]\n", prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    std::string npz_path  = argv[1];
    std::string scorer_kind = "log_linear";
    std::string weights_csv;
    std::string model_bin;
    float bias = 0.0f;
    int   K    = 10;
    std::string golden_path;
    int   golden_n = 500;
    std::string keep_report_path;
    std::string csv_path;

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
        else if (a == "--model")    model_bin = need("--model");
        else if (a == "--weights")  weights_csv = need("--weights");
        else if (a == "--bias")     bias = std::stof(need("--bias"));
        else if (a == "--K")        K = std::stoi(need("--K"));
        else if (a == "--golden")   golden_path = need("--golden");
        else if (a == "--golden-n") golden_n = std::stoi(need("--golden-n"));
        else if (a == "--keep-report") keep_report_path = need("--keep-report");
        else if (a == "--csv")      csv_path = need("--csv");
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
    } else if (scorer_kind == "biaffine") {
        if (model_bin.empty()) {
            std::fprintf(stderr,
                "--scorer biaffine requires --model <model_biaffine.bin>\n"
                "  (produced by cushr_train/export_weights.py --bin)\n");
            return 1;
        }
        auto b = cushr::BiaffineScorer::load(model_bin);
        std::cout << "  loaded biaffine: feat_dim=" << b.feat_dim()
                  << " hidden=" << b.hidden() << "\n";
        scorer = std::make_unique<cushr::BiaffineScorer>(std::move(b));
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

    // Keep-value divergence report: how many nodes kept the full K candidates
    // vs fewer (keep < K = under-full beam).
    if (!keep_report_path.empty()) {
        const auto& ks = dec.keep_stats();
        const double full_pct  = ks.total_nodes ? 100.0 * ks.full  / ks.total_nodes : 0.0;
        const double under_pct = ks.total_nodes ? 100.0 * ks.under / ks.total_nodes : 0.0;

        char summary[256];
        std::snprintf(summary, sizeof(summary),
            "# K=%d  total_nodes=%lld  full(keep==K)=%lld (%.2f%%)  under(keep<K)=%lld (%.2f%%)",
            ks.K, ks.total_nodes, ks.full, full_pct, ks.under, under_pct);

        std::ofstream out(keep_report_path);
        if (!out) {
            std::fprintf(stderr, "could not open keep-report file: %s\n",
                         keep_report_path.c_str());
            return 1;
        }
        out << summary << "\n";
        out << "keep,num_nodes\n";
        for (int k = 0; k <= ks.K; ++k) out << k << "," << ks.hist[k] << "\n";

        std::cout << "\n--- Keep-value divergence (K=" << ks.K << ") ---\n";
        std::cout << summary << "\n";
        std::cout << "Wrote keep report to " << keep_report_path << "\n";
    }

    // Gather top-1 predictions and gold paths
    std::vector<std::vector<int>> gold(lat.num_sentences());
    std::vector<std::vector<int>> pred(lat.num_sentences());
    for (int s = 0; s < lat.num_sentences(); ++s) {
        gold[s] = cushr::gold_path_nodes(lat, s);
        if (!results[s].empty()) {
            // top-1 decoded path runs super-source -> words... -> super-sink.
            // The explicit gold path contains word nodes only, so drop the two
            // boundary nodes (front/back) before comparing.
            const auto& nodes = results[s][0].nodes;
            if (nodes.size() > 2) {
                pred[s].assign(nodes.begin() + 1, nodes.end() - 1);
            }
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

    // One machine-readable row, so the CPU baseline can appear in the results
    // matrix. Until this existed the only CPU throughput figure anywhere was
    // the "~100 sentences/sec" design target in README.md -- prose, never a
    // measurement. Column names are deliberately NOT the GPU CSV's
    // `sent_per_sec_kernel`: there is no kernel here, and reusing the name
    // would invite a reader to compare a wall-clock number against a
    // kernel-only one.
    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out) {
            std::fprintf(stderr, "could not open csv file: %s\n",
                         csv_path.c_str());
            return 1;
        }
        out << "K,n_sentences,n_gold,wall_sec,us_per_sent,sent_per_sec,"
               "precision,recall,f1,perfect_match,scorer\n";
        out << K << ","
            << lat.num_sentences() << ","
            << m.num_sentences << ","
            << wall_sec << ","
            << (lat.num_sentences() ? 1e6 * wall_sec / lat.num_sentences() : 0.0) << ","
            << per_sec << ","
            << m.precision << ","
            << m.recall << ","
            << m.f1 << ","
            << m.perfect_match << ","
            << scorer_kind << "\n";
        std::cout << "Wrote csv to " << csv_path << "\n";
    }

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
