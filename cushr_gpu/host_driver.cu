#include "gpu_lattice.cuh"

#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"
#include "cushr/decoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <string>
#include <vector>


namespace cushr {
__global__ void init_best(GpuLattice lat);
__global__ void viterbi_relax_level(GpuLattice lat, const int* nodes_at_level, int n_nodes_at_level);
}

using namespace cushr;

// CUDA error-check helper - aborts if needed
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #call,      \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Build the EdgeScorer named on the command line.
// same edge scoring method as CPU to make consistent
static std::unique_ptr<EdgeScorer> make_scorer(const std::string& name,
                                               const Lattice& lat,
                                               const std::vector<float>& weights,
                                               float bias) {
    if (name == "uniform") return std::make_unique<UniformScorer>();
    if (name == "length")  return std::make_unique<LengthScorer>();
    if (name == "log_linear") {
        std::vector<float> w = weights;
        if (w.empty()) w.assign(lat.feat_dim(), 1.0f);
        return std::make_unique<LogLinearScorer>(std::move(w), bias);
    }
    std::fprintf(stderr, "unknown scorer '%s'\n", name.c_str());
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <lattice.npz> [--scorer uniform|length|log_linear]\n"
            "          [--weights w0,w1,...] [--bias b] [--limit N]\n", argv[0]);
        return 1;
    }

    // parse args
    std::string npz_path = argv[1];
    std::string scorer_name = "log_linear";
    std::vector<float> weights;
    float bias = 0.0f;
    int   limit = 1000;  // number of sentences to check 

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scorer" && i + 1 < argc) scorer_name = argv[++i];
        else if (a == "--bias" && i + 1 < argc) bias = std::atof(argv[++i]);
        else if (a == "--limit" && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (a == "--weights" && i + 1 < argc) {
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                weights.push_back((float)std::atof(tok));
        }
    }

    // load lattice (CPU loader: cnpy + reverse CSR + validation)
    std::printf("loading %s ...\n", npz_path.c_str());
    Lattice lat = Lattice::load_npz(npz_path);
    const int N = lat.num_nodes();
    const int E = lat.num_edges(); // total num of fwd edges
    const int S = lat.num_sentences();
    std::printf("  nodes=%d edges=%d sentences=%d feat_dim=%d\n", N, E, S, lat.feat_dim());

    // precompute one float per forward edge with the EdgeScorer
    auto scorer = make_scorer(scorer_name, lat, weights, bias);
    std::printf("  scorer=%s\n", scorer->name().c_str());

    // precomputing every edge's score on CPU before going to GPU
    std::vector<float> h_edge_score(E); //h edge score is host buffer that gets copied
    // it gets copied and is read later on
    for (int e = 0; e < E; ++e) h_edge_score[e] = scorer->score(lat, e);

    // rebuild the reverse CSR on the host so we can upload it 
    // The Lattice keeps its reverse CSR private, so we reconstruct the same three arrays here (identical logic to Lattice::build_reverse_csr_)
    std::vector<int> h_in_row_ptr(N + 1, 0);
    for (int v = 0; v < N; ++v)
        h_in_row_ptr[v + 1] = h_in_row_ptr[v] + lat.in_degree(v);

    std::vector<int> h_in_col_idx(E);
    std::vector<int> h_in_edge_id(E);
    for (int v = 0; v < N; ++v) {
        int w = h_in_row_ptr[v];
        for (int er = lat.in_edge_begin(v); er < lat.in_edge_end(v); ++er, ++w) {
            h_in_col_idx[w] = lat.in_edge_src(er);
            h_in_edge_id[w] = lat.in_edge_forward_id(er);
        }
    }

    //copies topo leel into host array and finds maximum level as it is private in Lattice obj
    std::vector<int> h_topo(N);
    int max_level = 0; // how many lelve-sweep iterations exist at max, bound
    for (int v = 0; v < N; ++v) {
        h_topo[v] = lat.topo_level(v);
        if (h_topo[v] > max_level) max_level = h_topo[v];
    }

    // upload everything once
    GpuLattice d{};
    d.num_nodes = N;
    d.num_edges = E;
    d.max_level = max_level;

    CUDA_CHECK(cudaMalloc(&d.in_row_ptr,  sizeof(int) * (N + 1)));
    int* d_in_col; int* d_in_eid; float* d_escore; int* d_topo;
    CUDA_CHECK(cudaMalloc(&d_in_col, sizeof(int) * E));
    CUDA_CHECK(cudaMalloc(&d_in_eid, sizeof(int) * E));
    CUDA_CHECK(cudaMalloc(&d_escore, sizeof(float) * E));
    CUDA_CHECK(cudaMalloc(&d_topo,   sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d.best_score,  sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d.best_parent, sizeof(int) * N));

    CUDA_CHECK(cudaMemcpy((void*)d.in_row_ptr, h_in_row_ptr.data(),
                          sizeof(int) * (N + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_col, h_in_col_idx.data(), sizeof(int) * E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_eid, h_in_edge_id.data(), sizeof(int) * E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_escore, h_edge_score.data(), sizeof(float) * E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topo,   h_topo.data(),       sizeof(int) * N, cudaMemcpyHostToDevice));
    d.in_col_idx = d_in_col;
    d.in_edge_id = d_in_eid;
    d.edge_score = d_escore;
    d.topo_level = d_topo;

    // initialize best_score/best_parent for the whole lattice once
    {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        init_best<<<blocks, threads>>>(d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int* d_level_nodes = nullptr;
    int  level_cap = 0;

    // host-side outputs we read back per sentence
    std::vector<float> h_best_score(N);
    std::vector<int>   h_best_parent(N);

    // per-sentence level sweep + compare to CPU 
    // Reference CPU decode at K=1, once over the whole lattice. 
    // results[s][0] is the CPU's top-1 path for sentence s.
    TopKDecoder cpu;
    auto cpu_results = cpu.decode(lat, /*K=*/1, *scorer);

    const int n_check = (limit < S) ? limit : S;

    int   mismatches = 0;
    double total_us = 0.0;

    for (int s = 0; s < n_check; ++s) {
        const auto sv = lat.sentence(s);
        const int node_begin = sv.node_begin;
        const int node_end   = sv.node_end; 
        const int sink = node_end - 1;  

        // group this sentence's nodes by topo level
        int min_level = lat.topo_level(node_begin);
        for (int v = node_begin; v < node_end; ++v)
            min_level = std::min(min_level, lat.topo_level(v));
        int loc_max = 0;
        for (int v = node_begin; v < node_end; ++v)
            loc_max = std::max(loc_max, lat.topo_level(v) - min_level);
        std::vector<std::vector<int>> buckets(loc_max + 1);
        for (int v = node_begin; v < node_end; ++v)
            buckets[lat.topo_level(v) - min_level].push_back(v);

        int biggest = 0;
        for (auto& b : buckets) if ((int)b.size() > biggest) biggest = (int)b.size();
        if (biggest > level_cap) {
            if (d_level_nodes) CUDA_CHECK(cudaFree(d_level_nodes));
            CUDA_CHECK(cudaMalloc(&d_level_nodes, sizeof(int) * biggest));
            level_cap = biggest;
        }

        // time the level sweep for this sentence.
        auto t0 = std::chrono::high_resolution_clock::now();


        for (int L = 0; L <= loc_max; ++L) {
            auto& nodes = buckets[L];
            if (nodes.empty()) continue;
            if (lat.in_degree(nodes[0]) == 0) continue;

            CUDA_CHECK(cudaMemcpy(d_level_nodes, nodes.data(),
                                  sizeof(int) * nodes.size(), cudaMemcpyHostToDevice));
            const int threads = 256;                 // 8 warps per block
            const int warps_needed = (int)nodes.size();
            const int blocks = (warps_needed * 32 + threads - 1) / threads;
            viterbi_relax_level<<<blocks, threads>>>(d, d_level_nodes, (int)nodes.size());
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // copy back just this sentence's slice of the outputs
        CUDA_CHECK(cudaMemcpy(h_best_score.data() + node_begin,
                              d.best_score + node_begin,
                              sizeof(float) * (node_end - node_begin),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_best_parent.data() + node_begin,
                              d.best_parent + node_begin,
                              sizeof(int) * (node_end - node_begin),
                              cudaMemcpyDeviceToHost));

        // walk best_parent from sink to source
        std::vector<int> gpu_path;
        for (int v = sink; v >= 0; v = h_best_parent[v]) {
            gpu_path.push_back(v);
            if (h_best_parent[v] < 0) break;
        }
        std::reverse(gpu_path.begin(), gpu_path.end());
        const float gpu_score = h_best_score[sink];

        // CPU reference top-1
        const auto& cpu_paths = cpu_results[s];
        const float cpu_score = cpu_paths.empty() ? CUSHR_NEG_INF : cpu_paths[0].score;
        const std::vector<int>& cpu_path = cpu_paths.empty() ? std::vector<int>{} : cpu_paths[0].nodes;

        // check that bits are same, verify alg was correct
        unsigned gi, ci;
        std::memcpy(&gi, &gpu_score, 4);
        std::memcpy(&ci, &cpu_score, 4);
        const bool score_ok = (gi == ci);
        const bool path_ok = (gpu_path == cpu_path);
        if (!score_ok || !path_ok) {
            if (mismatches < 10) {  // print first few
                std::printf("MISMATCH sentence %d: gpu_score=%.9g cpu_score=%.9g "
                            "score_ok=%d path_ok=%d (gpu_len=%zu cpu_len=%zu)\n",
                            s, gpu_score, cpu_score, (int)score_ok, (int)path_ok,
                            gpu_path.size(), cpu_path.size());
            }
            ++mismatches;
        }
    }

    // summary
    std::printf("\n=== Week-4 K=1 GPU vs CPU ===\n");
    std::printf("checked %d sentences\n", n_check);
    if (mismatches == 0)
        std::printf("BIT-IDENTICAL: %d/%d sentences match CPU\n", n_check, n_check);
    else
        std::printf("FAILED: %d/%d sentences mismatched\n", mismatches, n_check);
    std::printf("avg GPU level-sweep latency: %.1f us/sentence\n", total_us / n_check);

    // cleanup 
    if (d_level_nodes) cudaFree(d_level_nodes);
    cudaFree((void*)d.in_row_ptr); cudaFree(d_in_col); cudaFree(d_in_eid);
    cudaFree(d_escore); cudaFree(d_topo);
    cudaFree(d.best_score); cudaFree(d.best_parent);

    return mismatches == 0 ? 0 : 2;
}
