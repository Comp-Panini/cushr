// cushr/json_io.hpp
//
// Tiny dependency-free JSON writer for the golden-output file. We only ever
// emit -- never parse -- so this stays minimal.
//
// Output schema (golden_outputs.json):
//   {
//     "schema_version": 1,
//     "scorer": "log_linear",
//     "K": 10,
//     "sentences": [
//        {
//           "sentence_id": 0,
//           "paths": [
//              { "score": 12.345, "nodes": [0, 3, 7, 9] },
//              ...
//           ]
//        },
//        ...
//     ]
//   }

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
    const std::string& path,
    const std::string& scorer_name,
    int K,
    const std::vector<GoldenSentence>& sentences);

}  // namespace cushr
