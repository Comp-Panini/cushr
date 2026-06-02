#pragma once

#include <string>
#include <vector>

namespace cushr {

struct DecodedPath;

struct GoldenSentence {
    int sentence_id;
    std::vector<DecodedPath> paths;
};

void write_golden_outputs_json(
    const std::string& path, // dest path
    const std::string& scorer_name, // which scorer used
    int K,
    const std::vector<GoldenSentence>& sentences); // one entry per input sentence
    // contains sentence ID, topk decoded paths (score + nodes)

}  
