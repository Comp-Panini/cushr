// Build/run:
//   make cushr_batched
//   ./cushr_batched cushr_data.npz --scorer log_linear --K 1,5,16,32,64 \
//                   --batch 1024 --check 1000 --csv batched_bench.csv

#include "gpu_lattice.cuh"
#include "gpu_kbest.cuh"
#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"
#include "cushr/decoder.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>


namespace cushr {
__global__ void init_kbest(GpuLattice lat, GpuKBest kb);
__global__ void init_kbest_seed(GpuKBest kb, const int* src_nodes, int n);
void launch_kbest_merge(const GpuLattice& lat, const GpuKBest& kb, const int* d_nodes, int n_nodes, int threads_per_block, cudaStream_t stream);
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


// K5: reconstruct one GPU topK path by walking (pnode,prank) back-pointers from the sink
// Copied from host_driver_kbest.cu
// mirrors TopKDecoder::reconstruct
// Returns node ids source->sink INCLUDING boundaries.
static std::vector<int> reconstruct_gpu_path(const std::vector<int>& h_pnode, const std::vector<int>& h_prank, 
    const std::vector<int>& h_count, int K, int sink_node, int rank) {
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

// strip the boundary super-source/super-sink so the sequence lines up with Lattice::gold_path, which excludes both. 
// Copied from host_driver_kbest.cu
static std::vector<int> strip_boundaries(const std::vector<int>& path) {
    if (path.size() <= 2) return {};
    return std::vector<int>(path.begin() + 1, path.end() - 1);
}


static std::unique_ptr<EdgeScorer> make_scorer(const std::string& name, const Lattice& lat, const std::vector<float>& weights, float bias) {
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
            "          [--weights w0,w1,...] [--bias b] [--K 1,5,16,32,64]\n"
            "          [--batch N] [--check M] [--csv out.csv]\n"
            "  --batch N : sentences per memory-bounded chunk (<=0: whole corpus)\n"
            "  --check M : verify the first M sentences against the CPU decoder\n",
            argv[0]);
        return 1;
    }


    std::string npz_path = argv[1];
    std::string scorer_name = "log_linear";
    std::vector<float> weights;
    float bias = 0.0f;
    int   batch = -1;       // -1 => whole dataset in one chunk
    int   check = 1000;     // CPU verification count (0 to skip)
    std::string csv_path;
    std::vector<int> Ks = {1, 5, 16, 32, 64};


    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scorer" && i + 1 < argc) scorer_name = argv[++i];
        else if (a == "--bias" && i + 1 < argc) bias = std::atof(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (a == "--check" && i + 1 < argc) check = std::atoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--weights" && i + 1 < argc) {
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                weights.push_back((float)std::atof(tok));
        } 
        else if (a == "--K" && i + 1 < argc) {
            Ks.clear();
            char* s = argv[++i];
            for (char* tok = std::strtok(s, ","); tok; tok = std::strtok(nullptr, ","))
                Ks.push_back(std::atoi(tok));
        }
    }


    // load + score edges (identical to the K2 driver)
    std::printf("loading %s ...\n", npz_path.c_str());
    Lattice lat = Lattice::load_npz(npz_path);
    const int N = lat.num_nodes();
    const int E = lat.num_edges();
    const int S = lat.num_sentences();
    std::printf("  nodes=%d edges=%d sentences=%d feat_dim=%d\n", N, E, S, lat.feat_dim());


    auto scorer = make_scorer(scorer_name, lat, weights, bias);
    std::printf("  scorer=%s\n", scorer->name().c_str());
    if (!lat.has_explicit_gold())
        std::fprintf(stderr, "  WARNING: npz has no explicit gold paths; "
                             "recall@K will be reported as NA.\n");


    std::vector<float> h_edge_score(E);
    for (int e = 0; e < E; ++e) h_edge_score[e] = scorer->score(lat, e);


    // reverse CSR (host copy to upload).
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


    // upload the global graph once (shared, read-only across chunks)
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
    d.best_score=nullptr; d.best_parent=nullptr;


    const int bs = (batch <= 0) ? S : batch;
    const int n_chunks = (S + bs - 1) / bs;
    std::printf("  batch=%d  chunks=%d  (whole-corpus sweep = batch<=0)\n", bs, n_chunks);


    FILE* csv = nullptr;
    if (!csv_path.empty()) {
        csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) { std::fprintf(stderr, "cannot open --csv %s\n", csv_path.c_str()); return 1; }
        // Same header the K2 driver emits, so make_benchmark_md.py parses it.
        std::fprintf(csv,
            "K,n_sentences,n_gold,recall_at_K,us_per_sent_loop,us_per_sent_kernel,"
            "sent_per_sec_kernel,gpu_table_MB,gpu_used_MB,score_mismatch,count_mismatch\n");
    }


    //  per K: chunked batched sweep over the whole corpus 
    for (int K : Ks) {
        // CPU oracle for the checked sentences (decodes all S; we only read the
        // first `check`). Skip entirely when --check 0.
        std::vector<std::vector<DecodedPath>> cpu_results;
        if (check > 0) { TopKDecoder cpu; cpu_results = cpu.decode(lat, K, *scorer); }
        const int n_check = std::min(check, S);


        // cross-chunk accumulators
        double total_kernel_ms = 0.0;   // GPU-only merge time
        double total_loop_us   = 0.0;   // host sweep wall time
        int score_mismatch  = 0;
        int count_mismatch  = 0;
        long gold_total = 0, gold_hits = 0;
        double max_table_MB = 0.0, max_used_MB = 0.0;
        int total_launches = 0;


        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));


        for (int c = 0; c < n_chunks; ++c) {
            const int s0 = c * bs;
            const int s1 = std::min(s0 + bs, S);
            if (s0 >= s1) break;
            const int nb0 = lat.sentence(s0).node_begin;
            const int ne1 = lat.sentence(s1 - 1).node_end;
            const int span = ne1 - nb0;


            // allocate the chunk's k-best table (sized to `span`, not N)
            size_t mem_free_before = 0, mem_total = 0;
            CUDA_CHECK(cudaMemGetInfo(&mem_free_before, &mem_total));


            GpuKBest kb{};
            kb.K = K;
            kb.cap = (K <= 32) ? 32 : 64;
            float* base_score=nullptr; int *base_pnode=nullptr,*base_prank=nullptr,*base_count=nullptr;
            CUDA_CHECK(cudaMalloc(&base_score, sizeof(float)*(size_t)span*K));
            CUDA_CHECK(cudaMalloc(&base_pnode, sizeof(int)  *(size_t)span*K));
            CUDA_CHECK(cudaMalloc(&base_prank, sizeof(int)  *(size_t)span*K));
            CUDA_CHECK(cudaMalloc(&base_count, sizeof(int)  *(size_t)span));


            size_t mem_free_after = 0;
            CUDA_CHECK(cudaMemGetInfo(&mem_free_after, &mem_total));
            const double table_MB = ((double)(size_t)span * K * 12 + (double)(size_t)span * 4)
                                    / (1024.0 * 1024.0);
            const double used_MB = (mem_free_before >= mem_free_after)
                                 ? (double)(mem_free_before - mem_free_after) / (1024.0 * 1024.0)
                                 : 0.0;
            max_table_MB = std::max(max_table_MB, table_MB);
            max_used_MB  = std::max(max_used_MB, used_MB);


            // Offset device pointers by the chunk base so the global-node-indexed
            // kernels (init_kbest_seed / kbest_merge_level) address into the chunk
            // buffer: kb.score[v*K] == base_score[(v-nb0)*K] for v in [nb0, ne1).
            kb.score = base_score - (size_t)nb0 * K;
            kb.pnode = base_pnode - (size_t)nb0 * K;
            kb.prank = base_prank - (size_t)nb0 * K;
            kb.count = base_count - (size_t)nb0;


            // count starts at 0 for every node in the chunk; merged nodes get it
            // overwritten by the kernel, sources by the seed below.
            CUDA_CHECK(cudaMemset(base_count, 0, sizeof(int)*(size_t)span));


            //  build this chunk's source list + level-binned worklist 
            std::vector<int> src_nodes;
            int min_level = h_topo[nb0], loc_max = 0;
            for (int v = nb0; v < ne1; ++v) min_level = std::min(min_level, h_topo[v]);
            for (int v = nb0; v < ne1; ++v) loc_max = std::max(loc_max, h_topo[v] - min_level);
            std::vector<std::vector<int>> buckets(loc_max + 1);
            for (int v = nb0; v < ne1; ++v) {
                if (lat.in_degree(v) == 0) src_nodes.push_back(v);      // seeded
                else buckets[h_topo[v] - min_level].push_back(v);       // merged
            }
            std::vector<int> flat; flat.reserve(span);
            std::vector<int> level_ptr(loc_max + 2, 0);
            for (int L = 0; L <= loc_max; ++L) {
                for (int v : buckets[L]) flat.push_back(v);
                level_ptr[L + 1] = (int)flat.size();
            }


            // upload src + worklist for this chunk.
            int* d_src = nullptr;
            if (!src_nodes.empty()) {
                CUDA_CHECK(cudaMalloc(&d_src, sizeof(int)*src_nodes.size()));
                CUDA_CHECK(cudaMemcpy(d_src, src_nodes.data(),
                                      sizeof(int)*src_nodes.size(), cudaMemcpyHostToDevice));
            }
            int* d_flat = nullptr;
            if (!flat.empty()) {
                CUDA_CHECK(cudaMalloc(&d_flat, sizeof(int)*flat.size()));
                CUDA_CHECK(cudaMemcpy(d_flat, flat.data(),
                                      sizeof(int)*flat.size(), cudaMemcpyHostToDevice));
            }


            //  seed sources 
            if (!src_nodes.empty()) {
                int t = 256, b = ((int)src_nodes.size() + t - 1) / t;
                init_kbest_seed<<<b, t>>>(kb, d_src, (int)src_nodes.size());
                CUDA_CHECK(cudaGetLastError());
            }


            // timed batched sweep (kernel-only via events; loop via chrono)-
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev0, 0));
            for (int L = 0; L <= loc_max; ++L) {
                const int off = level_ptr[L];
                const int cnt = level_ptr[L + 1] - off;
                if (cnt <= 0) continue;
                launch_kbest_merge(d, kb, d_flat + off, cnt, /*tpb=*/256, /*stream=*/0);
                ++total_launches;
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ev1, 0));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            auto t1 = std::chrono::high_resolution_clock::now();
            float chunk_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&chunk_ms, ev0, ev1));
            total_kernel_ms += chunk_ms;
            total_loop_us  += std::chrono::duration<double, std::micro>(t1 - t0).count();


            //  copy back the chunk table + K5 reconstruct / verify 
            const int chk_lo = std::max(s0, 0);
            const int chk_hi = std::min(s1, n_check);
            if (chk_hi > chk_lo) {
                // pull back the whole chunk slice once (indexed by global v below).
                std::vector<float> h_score((size_t)span*K);
                std::vector<int>   h_pnode((size_t)span*K), h_prank((size_t)span*K), h_count(span);
                CUDA_CHECK(cudaMemcpy(h_score.data(), base_score, sizeof(float)*(size_t)span*K, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_pnode.data(), base_pnode, sizeof(int)*(size_t)span*K, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_prank.data(), base_prank, sizeof(int)*(size_t)span*K, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_count.data(), base_count, sizeof(int)*(size_t)span, cudaMemcpyDeviceToHost));


                // reconstruct_gpu_path() indexes h_pnode/h_prank/h_count by GLOBAL
                // node id, so build global-indexed views [0, ne1) whose [nb0, ne1)
                // slice holds this chunk's data (reconstruction only ever walks
                // nodes within the sentence, i.e. within the chunk). Cheap: ne1
                // ints, reused for every checked sentence in the chunk.
                std::vector<int> g_pnode((size_t)ne1*K), g_prank((size_t)ne1*K), g_count(ne1);
                std::memcpy(g_pnode.data()+(size_t)nb0*K, h_pnode.data(), sizeof(int)*(size_t)span*K);
                std::memcpy(g_prank.data()+(size_t)nb0*K, h_prank.data(), sizeof(int)*(size_t)span*K);
                std::memcpy(g_count.data()+nb0, h_count.data(), sizeof(int)*(size_t)span);


                for (int s = chk_lo; s < chk_hi; ++s) {
                    const int sink = lat.sentence(s).node_end - 1;
                    const int gpu_cnt = h_count[sink - nb0];


                    // score-multiset comparison vs CPU oracle
                    std::vector<float> a(gpu_cnt);
                    for (int r = 0; r < gpu_cnt; ++r) a[r] = h_score[(size_t)(sink - nb0)*K + r];
                    const auto& cp = cpu_results[s];
                    std::vector<float> b(cp.size());
                    for (size_t r = 0; r < cp.size(); ++r) b[r] = cp[r].score;
                    if ((int)b.size() != gpu_cnt) ++count_mismatch;
                    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                    bool ok = (a.size() == b.size());
                    for (size_t r = 0; ok && r < a.size(); ++r)
                        if (std::fabs(a[r] - b[r]) > 1e-4f * (1.0f + std::fabs(b[r]))) ok = false;
                    if (!ok) ++score_mismatch;


                    //  K5 reconstruct + top-K recall vs gold 
                    if (lat.has_explicit_gold()) {
                        std::vector<int> gold = lat.gold_path(s);
                        if (!gold.empty()) {
                            ++gold_total;
                            bool hit = false;
                            for (int r = 0; r < gpu_cnt && !hit; ++r) {
                                std::vector<int> pred = strip_boundaries(
                                    reconstruct_gpu_path(g_pnode, g_prank, g_count, K, sink, r));
                                if (pred == gold) hit = true;
                            }
                            if (hit) ++gold_hits;
                        }
                    }
                }
            }


            if (d_src)  cudaFree(d_src);
            if (d_flat) cudaFree(d_flat);
            cudaFree(base_score); cudaFree(base_pnode); cudaFree(base_prank); cudaFree(base_count);
        }


        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));


        const double us_kernel = total_kernel_ms * 1000.0 / std::max(1, S);
        const double us_loop   = total_loop_us / std::max(1, S);
        const double sent_per_sec = total_kernel_ms > 0.0 ? (double)S / (total_kernel_ms / 1000.0) : 0.0;
        const double recall = gold_total > 0 ? (double)gold_hits / (double)gold_total : -1.0;


        std::printf("=== K=%2d ===  %d chunks, %d launches, sweep %.2f ms"
                    "  %.2f us/sent (kernel)  %.0f sent/sec"
                    "  table=%.1f MB used=%.1f MB"
                    "  check(%d): %s (score_mm=%d count_mm=%d)  recall@K=%s\n",
                    K, n_chunks, total_launches, total_kernel_ms, us_kernel, sent_per_sec,
                    max_table_MB, max_used_MB, n_check,
                    (n_check == 0 ? "skipped" : score_mismatch == 0 ? "SCORE-EQUIVALENT" : "FAILED"),
                    score_mismatch, count_mismatch,
                    (gold_total > 0 ? "" : "NA"));
        if (gold_total > 0)
            std::printf("             recall@%d = %.4f (%ld/%ld)\n", K, recall, gold_hits, gold_total);


        if (csv) {
            if (gold_total > 0)
                std::fprintf(csv, "%d,%d,%ld,%.6f,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d\n",
                             K, S, gold_total, recall, us_loop, us_kernel,
                             sent_per_sec, max_table_MB, max_used_MB, score_mismatch, count_mismatch);
            else
                std::fprintf(csv, "%d,%d,%ld,NA,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d\n",
                             K, S, gold_total, us_loop, us_kernel,
                             sent_per_sec, max_table_MB, max_used_MB, score_mismatch, count_mismatch);
            std::fflush(csv);
        }
    }


    if (csv) std::fclose(csv);
    cudaFree(d_irp); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_es); cudaFree(d_topo);
    return 0;
}
