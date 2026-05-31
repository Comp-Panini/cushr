// cushr/json_io.cpp

#include "cushr/json_io.hpp"

#include "cushr/decoder.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace cushr {

namespace {

// Format a float with enough digits to round-trip a float32 exactly.
// 9 digits suffice for IEEE-754 binary32.
void write_float(std::ostream& os, float x) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", (double)x);
    os << buf;
}

}  // namespace

void write_golden_outputs_json(
    const std::string& path,
    const std::string& scorer_name,
    int K,
    const std::vector<GoldenSentence>& sentences) {

    std::ofstream os(path);
    if (!os) throw std::runtime_error("cushr: cannot open " + path + " for writing");

    os << "{\n";
    os << "  \"schema_version\": 1,\n";
    os << "  \"scorer\": \"" << scorer_name << "\",\n";
    os << "  \"K\": " << K << ",\n";
    os << "  \"sentences\": [\n";

    for (size_t i = 0; i < sentences.size(); ++i) {
        const auto& s = sentences[i];
        os << "    {\n";
        os << "      \"sentence_id\": " << s.sentence_id << ",\n";
        os << "      \"paths\": [\n";
        for (size_t j = 0; j < s.paths.size(); ++j) {
            const auto& p = s.paths[j];
            os << "        { \"score\": ";
            write_float(os, p.score);
            os << ", \"nodes\": [";
            for (size_t k = 0; k < p.nodes.size(); ++k) {
                if (k) os << ", ";
                os << p.nodes[k];
            }
            os << "] }";
            if (j + 1 < s.paths.size()) os << ",";
            os << "\n";
        }
        os << "      ]\n";
        os << "    }" << (i + 1 < sentences.size() ? "," : "") << "\n";
    }

    os << "  ]\n";
    os << "}\n";
}

}  // namespace cushr
