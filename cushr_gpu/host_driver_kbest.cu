//   1. loads a lattice .npz with the SAME CPU loader/scorer used by the oracle,
//   2. uploads the graph (reverse CSR + edge scores + topo levels) once,
//   3. for each requested K, runs the K2 kernels over every sentence and
//      reconstructs the top-K paths on the host,
//   4. compares those against the CPU TopKDecoder from Week 3 and reports how
//      many sentences are score-equivalent (the Week-6/7 correctness contract).
//
// This mirrors host_driver.cu (the K=1 driver) closely on purpose: same upload
// path, same per-sentence level bucketing. The only new parts are the GpuKBest
// table, the templated merge launch, and top-K path reconstruction.
//
// Build/run (see Makefile target proposed in the README notes):
//   ./cushr_kbest cushr_data.npz --scorer log_linear --K 1,8,16,32,64 --limit 1000

#include "gpu_lattice.cuh"
#include "gpu_kbest.cuh"

#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"
#include "cushr/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace cushr {
__global__ void init_kbest(GpuLattice lat, GpuKBest kb);
void launch_kbest_merge(const GpuLattice& lat, const GpuKBest& kb,
                        const int* d_nodes, int n_nodes,
                        int threads_per_block, cudaStream_t stream);
}
using namespace cushr;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #call,      \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Reconstruct one GPU top-K path by walking the (pnode, prank) back-pointer
// chain from the sink, mirroring TopKDecoder::reconstruct in decoder.cpp. The
// per-node table slices (h_pnode/h_prank/h_count) must already hold this
// sentence's data. Returns node ids in source->sink order, INCLUDING the
// boundary super-source/super-sink (caller strips them to match Lattice::gold_path).
static std::vector<int> reconstruct_gpu_path(
        const std::vector<int>& h_pnode,
        const std::vector<int>& h_prank,
        const std::vector<int>& h_count,
        int K, int sink_node, int rank) {
    std::vector<int> rev;
    int v = sink_node;
    int r = rank;
    while (v >= 0) {
        rev.push_back(v);
        if (r < 0 || r >= h_count[v]) break;   // invalid rank -> reached a source
        const int pn = h_pnode[(size_t)v * K + r];
        const int pr = h_prank[(size_t)v * K + r];
        v = pn;
        r = pr;
    }
    std::reverse(rev.begin(), rev.end());
    return rev;
}

// Strip the boundary super-source (front) and super-sink (back) so the sequence
// lines up with Lattice::gold_path, which excludes both boundary nodes.
static std::vector<int> strip_boundaries(const std::vector<int>& path) {
    if (path.size() <= 2) return {};
    return std::vector<int>(path.begin() + 1, path.end() - 1);
}

// Same scorer factory as host_driver.cu so GPU edge scores == CPU edge scores.
static std::unique_ptr<EdgeScorer> make_scorer(const std::string& name,
                                               const Lattice& lat,
                                               const std::vector<float>& weights,
                                               float bias,
                                               const std::string& model_path) {
    if (name == "uniform") return std::make_unique<UniformScorer>();
    if (name == "length")  return std::make_unique<LengthScorer>();
    if (name == "log_linear") {
        std::vector<float> w = weights;
        if (w.empty()) w.assign(lat.feat_dim(), 1.0f);
        return std::make_unique<LogLinearScorer>(std::move(w), bias);
    }
    // Week 10 parity with host_driver_batched.cu. K2 has no K4: it scores
    // on the host and uploads, which is exactly what makes it the reference
    // the device path is compared against.
    if (name == "biaffine") {
        if (model_path.empty()) {
            std::fprintf(stderr, "--scorer biaffine requires --model <CSB3 .bin>\n");
            std::exit(1);
        }
        return std::make_unique<BiaffineScorer>(BiaffineScorer::load(model_path));
    }
    std::fprintf(stderr, "unknown scorer '%s'\n", name.c_str());
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <lattice.npz> [--scorer uniform|length|log_linear|biaffine]\n"
            "          [--model m.bin]   (required by --scorer biaffine)\n"
            "          [--weights w0,w1,...] [--bias b] [--K 1,5,16,32,64]\n"
            "          [--limit N | -1 for all] [--csv out.csv]\n", argv[0]);
        return 1;
    }

    // ---- parse args ---------------------------------------------------------
    std::string npz_path = argv[1];
    std::string scorer_name = "log_linear";
    std::vector<float> weights;
    float bias = 0.0f;
    int   limit = 1000;
    std::string csv_path;                   // when set, append one row per K
    std::string model_path;                 // CSB3 weights for --scorer biaffine
    std::vector<int> Ks = {1, 5, 16, 32, 64};   // default beam widths to check

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--scorer" && i + 1 < argc) scorer_name = argv[++i];
        else if (a == "--bias"   && i + 1 < argc) bias = std::atof(argv[++i]);
        else if (a == "--limit"  && i + 1 < argc) limit = std::atoi(argv[++i]);
        else if (a == "--csv"    && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--weights" && i + 1 < argc) {
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                weights.push_back((float)std::atof(tok));
        } else if (a == "--K" && i + 1 < argc) {
            Ks.clear();
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                Ks.push_back(std::atoi(tok));
        }
    }

    // ---- load lattice + precompute edge scores (identical to the K=1 driver) --
    std::printf("loading %s ...\n", npz_path.c_str());
    Lattice lat = Lattice::load_npz(npz_path);
    const int N = lat.num_nodes();
    const int E = lat.num_edges();
    const int S = lat.num_sentences();
    std::printf("  nodes=%d edges=%d sentences=%d feat_dim=%d\n", N, E, S, lat.feat_dim());

    auto scorer = make_scorer(scorer_name, lat, weights, bias, model_path);
    std::printf("  scorer=%s\n", scorer->name().c_str());
    if (!lat.has_explicit_gold())
        std::fprintf(stderr, "  WARNING: npz has no explicit gold paths; "
                             "recall@K will be reported as NA.\n");

    std::vector<float> h_edge_score(E);
    for (int e = 0; e < E; ++e) h_edge_score[e] = scorer->score(lat, e);

    // rebuild reverse CSR on the host to upload (same as host_driver.cu).
    std::vector<int> h_in_row_ptr(N + 1, 0);
    for (int v = 0; v < N; ++v)
        h_in_row_ptr[v + 1] = h_in_row_ptr[v] + lat.in_degree(v);
    std::vector<int> h_in_col_idx(E), h_in_edge_id(E);
    for (int v = 0; v < N; ++v) {
        int w = h_in_row_ptr[v];
        for (int er = lat.in_edge_begin(v); er < lat.in_edge_end(v); ++er, ++w) {
            h_in_col_idx[w] = lat.in_edge_src(er);
            h_in_edge_id[w] = lat.in_edge_forward_id(er);
        }
    }
    std::vector<int> h_topo(N);
    int max_level = 0;
    for (int v = 0; v < N; ++v) {
        h_topo[v] = lat.topo_level(v);
        max_level = std::max(max_level, h_topo[v]);
    }

    // ---- upload graph once (outputs table is (re)allocated per K below) ------
    GpuLattice d{};
    d.num_nodes = N; d.num_edges = E; d.max_level = max_level;
    int *d_irp=nullptr,*d_ici=nullptr,*d_iei=nullptr,*d_topo=nullptr; float* d_es=nullptr;
    CUDA_CHECK(cudaMalloc(&d_irp, sizeof(int)*(N+1)));
    CUDA_CHECK(cudaMalloc(&d_ici, sizeof(int)*E));
    CUDA_CHECK(cudaMalloc(&d_iei, sizeof(int)*E));
    CUDA_CHECK(cudaMalloc(&d_es,  sizeof(float)*E));
    CUDA_CHECK(cudaMalloc(&d_topo,sizeof(int)*N));
    CUDA_CHECK(cudaMemcpy(d_irp, h_in_row_ptr.data(), sizeof(int)*(N+1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ici, h_in_col_idx.data(), sizeof(int)*E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_iei, h_in_edge_id.data(), sizeof(int)*E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_es,  h_edge_score.data(), sizeof(float)*E, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topo,h_topo.data(),       sizeof(int)*N, cudaMemcpyHostToDevice));
    d.in_row_ptr=d_irp; d.in_col_idx=d_ici; d.in_edge_id=d_iei; d.edge_score=d_es; d.topo_level=d_topo;
    d.best_score=nullptr; d.best_parent=nullptr;   // unused by K2

    int* d_level_nodes = nullptr; int level_cap = 0;

    // Open the CSV once (truncate) and write the header, if requested.
    FILE* csv = nullptr;
    if (!csv_path.empty()) {
        csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) { std::fprintf(stderr, "cannot open --csv %s\n", csv_path.c_str()); return 1; }
        std::fprintf(csv,
            "K,n_sentences,n_gold,recall_at_K,us_per_sent_loop,us_per_sent_kernel,"
            "sent_per_sec_kernel,gpu_table_MB,gpu_used_MB,score_mismatch,count_mismatch\n");
    }

    // ---- run each K, comparing to the CPU reference at that same K ----------
    for (int K : Ks) {
        // CPU oracle for this K.
        TopKDecoder cpu;
        auto cpu_results = cpu.decode(lat, K, *scorer);

        // Allocate the GPU top-K table for this K. Measure device memory used by
        // the table via cudaMemGetInfo before/after the four allocations.
        size_t mem_free_before = 0, mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free_before, &mem_total));

        GpuKBest kb{};
        kb.K = K;
        kb.cap = (K <= 32) ? 32 : 64;
        CUDA_CHECK(cudaMalloc(&kb.score, sizeof(float)*(size_t)N*K));
        CUDA_CHECK(cudaMalloc(&kb.pnode, sizeof(int)  *(size_t)N*K));
        CUDA_CHECK(cudaMalloc(&kb.prank, sizeof(int)  *(size_t)N*K));
        CUDA_CHECK(cudaMalloc(&kb.count, sizeof(int)  *(size_t)N));

        size_t mem_free_after = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free_after, &mem_total));
        // Analytic table size (score+pnode+prank are N*K*4 each, count is N*4).
        const double table_MB = ((double)(size_t)N * K * (4 + 4 + 4)
                                 + (double)(size_t)N * 4) / (1024.0 * 1024.0);
        // Actual driver-reported delta (rounds up to allocation granularity).
        const double used_MB = (mem_free_before >= mem_free_after)
                             ? (double)(mem_free_before - mem_free_after) / (1024.0 * 1024.0)
                             : 0.0;

        // seed sources.
        {
            int t = 256, b = (N + t - 1) / t;
            init_kbest<<<b, t>>>(d, kb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // host-side readback buffers for the whole table (reused across sentences).
        std::vector<float> h_score((size_t)N*K);
        std::vector<int>   h_pnode((size_t)N*K), h_prank((size_t)N*K), h_count(N);

        const int n_check = (limit < 0) ? S : std::min(limit, S);
        int   score_mismatch = 0;    // sentences whose top-K score list differs
        int   count_mismatch = 0;    // sentences whose #paths differs
        long  gold_total = 0;        // sentences with a resolved gold path
        long  gold_hits  = 0;        // ...where gold appears in the GPU top-K
        double total_us = 0.0;        // wall time of the whole per-sentence level loop
        double total_kernel_ms = 0.0; // GPU-only time inside the merge kernels

        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));

        for (int s = 0; s < n_check; ++s) {
            const auto sv = lat.sentence(s);
            const int nb = sv.node_begin, ne = sv.node_end;
            const int sink = ne - 1;

            // bucket this sentence's nodes by (level - min_level).
            int min_level = lat.topo_level(nb);
            for (int v = nb; v < ne; ++v) min_level = std::min(min_level, lat.topo_level(v));
            int loc_max = 0;
            for (int v = nb; v < ne; ++v) loc_max = std::max(loc_max, lat.topo_level(v) - min_level);
            std::vector<std::vector<int>> buckets(loc_max + 1);
            for (int v = nb; v < ne; ++v) buckets[lat.topo_level(v) - min_level].push_back(v);

            int biggest = 0;
            for (auto& b : buckets) biggest = std::max(biggest, (int)b.size());
            if (biggest > level_cap) {
                if (d_level_nodes) CUDA_CHECK(cudaFree(d_level_nodes));
                CUDA_CHECK(cudaMalloc(&d_level_nodes, sizeof(int)*biggest));
                level_cap = biggest;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int L = 0; L <= loc_max; ++L) {
                auto& nodes = buckets[L];
                if (nodes.empty()) continue;
                if (lat.in_degree(nodes[0]) == 0) continue;   // source level: seeded already
                CUDA_CHECK(cudaMemcpy(d_level_nodes, nodes.data(),
                                      sizeof(int)*nodes.size(), cudaMemcpyHostToDevice));
                // GPU-only timing of just this merge launch.
                CUDA_CHECK(cudaEventRecord(ev0, 0));
                launch_kbest_merge(d, kb, d_level_nodes, (int)nodes.size(),
                                   /*threads_per_block=*/256, /*stream=*/0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(ev1, 0));
                CUDA_CHECK(cudaEventSynchronize(ev1));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                total_kernel_ms += ms;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

            // copy back this sentence's slice of the table.
            CUDA_CHECK(cudaMemcpy(h_score.data()+(size_t)nb*K, kb.score+(size_t)nb*K,
                                  sizeof(float)*(size_t)(ne-nb)*K, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_pnode.data()+(size_t)nb*K, kb.pnode+(size_t)nb*K,
                                  sizeof(int)*(size_t)(ne-nb)*K, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_prank.data()+(size_t)nb*K, kb.prank+(size_t)nb*K,
                                  sizeof(int)*(size_t)(ne-nb)*K, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_count.data()+nb, kb.count+nb,
                                  sizeof(int)*(ne-nb), cudaMemcpyDeviceToHost));

            // GPU top-K scores at the sink (already sorted best-first by the kernel).
            const int gpu_cnt = h_count[sink];
            std::vector<float> gpu_scores(gpu_cnt);
            for (int r = 0; r < gpu_cnt; ++r) gpu_scores[r] = h_score[(size_t)sink*K + r];

            // CPU top-K scores.
            const auto& cpu_paths = cpu_results[s];
            std::vector<float> cpu_scores(cpu_paths.size());
            for (size_t r = 0; r < cpu_paths.size(); ++r) cpu_scores[r] = cpu_paths[r].score;

            // Compare. Multiple equally-scored paths may be ordered differently,
            // so we compare the *sorted score multisets* within a float epsilon.
            if ((int)cpu_scores.size() != gpu_cnt) { ++count_mismatch; }
            std::vector<float> a = gpu_scores, b = cpu_scores;
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            bool ok = (a.size() == b.size());
            for (size_t r = 0; ok && r < a.size(); ++r)
                if (std::fabs(a[r] - b[r]) > 1e-4f * (1.0f + std::fabs(b[r]))) ok = false;
            if (!ok) {
                if (score_mismatch < 10)
                    std::printf("  [K=%d] MISMATCH sentence %d: gpu#=%d cpu#=%d "
                                "top1 gpu=%.6g cpu=%.6g\n", K, s, gpu_cnt,
                                (int)cpu_scores.size(),
                                gpu_cnt ? gpu_scores[0] : 0.0f,
                                cpu_scores.empty() ? 0.0f : cpu_scores[0]);
                ++score_mismatch;
            }

            // ---- top-K recall vs gold (only sentences with a resolved gold) ----
            if (lat.has_explicit_gold()) {
                std::vector<int> gold = lat.gold_path(s);
                if (!gold.empty()) {
                    ++gold_total;
                    bool hit = false;
                    for (int r = 0; r < gpu_cnt && !hit; ++r) {
                        std::vector<int> pred = strip_boundaries(
                            reconstruct_gpu_path(h_pnode, h_prank, h_count, K, sink, r));
                        if (pred == gold) hit = true;
                    }
                    if (hit) ++gold_hits;
                }
            }
        }

        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        const double us_loop   = total_us / std::max(1, n_check);
        const double us_kernel = (total_kernel_ms * 1000.0) / std::max(1, n_check);
        const double sent_per_sec = us_kernel > 0.0 ? 1e6 / us_kernel : 0.0;
        const double recall = gold_total > 0 ? (double)gold_hits / (double)gold_total : -1.0;

        std::printf("=== K=%2d ===  checked %d sentences: %s"
                    "  (score_mismatch=%d, count_mismatch=%d)"
                    "  loop %.1f us/sent  kernel %.1f us/sent"
                    "  recall@K=%s (%ld/%ld)  table=%.1f MB used=%.1f MB\n",
                    K, n_check,
                    (score_mismatch == 0 ? "SCORE-EQUIVALENT to CPU" : "FAILED"),
                    score_mismatch, count_mismatch, us_loop, us_kernel,
                    (gold_total > 0 ? "" : "NA"), gold_hits, gold_total,
                    table_MB, used_MB);
        if (gold_total > 0)
            std::printf("             recall@%d = %.4f\n", K, recall);

        if (csv) {
            if (gold_total > 0)
                std::fprintf(csv, "%d,%d,%ld,%.6f,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d\n",
                             K, n_check, gold_total, recall, us_loop, us_kernel,
                             sent_per_sec, table_MB, used_MB, score_mismatch, count_mismatch);
            else
                std::fprintf(csv, "%d,%d,%ld,NA,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d\n",
                             K, n_check, gold_total, us_loop, us_kernel,
                             sent_per_sec, table_MB, used_MB, score_mismatch, count_mismatch);
            std::fflush(csv);
        }

        cudaFree(kb.score); cudaFree(kb.pnode); cudaFree(kb.prank); cudaFree(kb.count);
    }

    if (csv) std::fclose(csv);

    if (d_level_nodes) cudaFree(d_level_nodes);
    cudaFree(d_irp); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_es); cudaFree(d_topo);
    return 0;
}
