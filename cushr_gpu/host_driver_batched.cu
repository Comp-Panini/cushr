// Build/run:
//   make cushr_batched
//   ./cushr_batched cushr_data.npz --scorer log_linear --K 1,5,16,32,64 \
//                   --batch 1024 --check 1000 --csv batched_bench.csv
//
// Week 10 (K4): with --scorer biaffine --model model95_ctx.bin the edge scores
// are computed ON THE DEVICE by score_edges.cu instead of being handed over
// pre-scored by the host. The per-chunk pipeline becomes
//     load batch -> K4 (score edges) -> K3 (topo sweep) -> K5 (reconstruct)
// and the K4 and K3 halves are timed separately. --k4 host keeps the old
// behaviour (host-scored, uploaded once) for an apples-to-apples comparison.
//
//   ./cushr_batched g95_ctx_mat.npz --scorer biaffine --model model95_ctx.bin \
//                   --K 32 --check -1 --dump-paths gpu_k32.npz

#include "gpu_lattice.cuh"
#include "gpu_kbest.cuh"
#include "score_edges.cuh"
#include "cushr/lattice.hpp"
#include "cushr/scorer.hpp"
#include "cushr/decoder.hpp"
#include "cnpy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
// Returns node ids source->sink including boundaries.
static std::vector<int> reconstruct_gpu_path(const std::vector<int>& h_pnode, const std::vector<int>& h_prank, 
    const std::vector<int>& h_count, int K, int sink_node, int rank) {
    std::vector<int> rev;
    int v = sink_node;
    int r = rank;
    while (v >= 0) {
        rev.push_back(v);
        if (r < 0 || r >= h_count[v]) break;   // invalid rank means reached a source
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


static std::unique_ptr<EdgeScorer> make_scorer(const std::string& name, const Lattice& lat, const std::vector<float>& weights, float bias, const std::string& model_path) {
    if (name == "uniform") return std::make_unique<UniformScorer>();
    if (name == "length")  return std::make_unique<LengthScorer>();
    if (name == "log_linear") {
        std::vector<float> w = weights;
        if (w.empty()) w.assign(lat.feat_dim(), 1.0f);
        return std::make_unique<LogLinearScorer>(std::move(w), bias);
    }
    // Week 10: the trained biaffine, previously reachable only from the CPU
    // evaluator. The GPU needs it here too -- not to score the edges it decodes
    // (K4 does that on the device) but so the --check oracle compares against
    // the SAME model. Checking a K4 decode against a log_linear oracle would
    // report a mismatch on every sentence and mean nothing.
    if (name == "biaffine") {
        if (model_path.empty()) {
            std::fprintf(stderr, "--scorer biaffine requires --model <CSB3 .bin> "
                                 "(cushr_train/export_weights.py --bin)\n");
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
            "          [--weights w0,w1,...] [--bias b] [--K 1,5,16,32,64]\n"
            "          [--batch N] [--check M] [--csv out.csv]\n"
            "          [--model m.bin] [--k4 twopass|fused|host] [--k4-scratch MB]\n"
            "          [--tol t] [--dump-paths out.npz]\n"
            "  --batch N : sentences per memory-bounded chunk (<=0: whole corpus)\n"
            "  --check M : verify the first M sentences against the CPU decoder\n"
            "              (-1: all sentences -- full-corpus recall, comparable to\n"
            "               the K2 number; 0: skip verification entirely)\n"
            "  --model   : CSB3 weight file; required by --scorer biaffine\n"
            "  --k4      : where edge scores come from. twopass/fused run K4 on the\n"
            "              device (biaffine only); host uploads CPU-scored edges,\n"
            "              which is the pre-week-10 behaviour and the timing baseline\n"
            "  --k4-scratch : cap on the K4a projection scratch (default 512 MB);\n"
            "              sets how many nodes one K4 tile covers\n"
            "  --tol     : relative tolerance for the score comparison against the\n"
            "              CPU oracle (default 1e-4). K4 reduces in a different\n"
            "              order than the CPU scorer, so this is not bit-exact --\n"
            "              the path and count comparisons still are\n"
            "  --dump-paths : write the top-K reconstructions for EVERY sentence as\n"
            "              an npz in make_rerank_data.py's layout, for eval_slm.py\n",
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
    std::string model_path;
    std::string k4_name = "twopass";
    std::string dump_path;
    double k4_scratch_MB = 512.0;
    float tol = 1e-4f;


    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scorer" && i + 1 < argc) scorer_name = argv[++i];
        else if (a == "--bias" && i + 1 < argc) bias = std::atof(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (a == "--check" && i + 1 < argc) check = std::atoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--k4" && i + 1 < argc) k4_name = argv[++i];
        else if (a == "--k4-scratch" && i + 1 < argc) k4_scratch_MB = std::atof(argv[++i]);
        else if (a == "--tol" && i + 1 < argc) tol = (float)std::atof(argv[++i]);
        else if (a == "--dump-paths" && i + 1 < argc) dump_path = argv[++i];
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


    K4Mode k4 = K4Mode::TwoPass;
    if      (k4_name == "host")    k4 = K4Mode::Host;
    else if (k4_name == "fused")   k4 = K4Mode::Fused;
    else if (k4_name == "twopass") k4 = K4Mode::TwoPass;
    else { std::fprintf(stderr, "unknown --k4 '%s'\n", k4_name.c_str()); return 1; }
    // Only the biaffine has a device implementation. Everything else is a
    // handful of host flops per edge and is not worth a kernel.
    if (scorer_name != "biaffine") k4 = K4Mode::Host;

    auto scorer = make_scorer(scorer_name, lat, weights, bias, model_path);
    std::printf("  scorer=%s  k4=%s\n", scorer->name().c_str(),
                k4 == K4Mode::Host ? "host" : k4 == K4Mode::Fused ? "fused" : "twopass");
    if (!lat.has_explicit_gold())
        std::fprintf(stderr, "  WARNING: npz has no explicit gold paths; "
                             "recall@K will be reported as NA.\n");


    // Host scoring is what K4 replaces, so only pay for it when the device is
    // not going to do the job. It is ~E scorer calls; at 60M+ edges with the
    // biaffine that is minutes of single-threaded work.
    std::vector<float> h_edge_score;
    if (k4 == K4Mode::Host) {
        h_edge_score.assign(E, 0.0f);
        for (int e = 0; e < E; ++e) h_edge_score[e] = scorer->score(lat, e);
    }


    // reverse CSR (host copy to upload). h_in_dst is new in week 10: K4
    // iterates over reverse-CSR SLOTS rather than forward edges, so it needs
    // the row that owns each slot. Building it costs E ints, the same as the
    // two arrays either side of it, and saves uploading a second (src,dst) edge
    // list -- the reverse CSR already carries the src and the forward edge id.
    std::vector<int> h_in_row_ptr(N + 1, 0);
    for (int v = 0; v < N; ++v)
        h_in_row_ptr[v + 1] = h_in_row_ptr[v] + lat.in_degree(v);
    std::vector<int> h_in_col_idx(E), h_in_edge_id(E), h_in_dst;
    if (k4 != K4Mode::Host) h_in_dst.assign(E, 0);
    for (int v = 0; v < N; ++v) {
        int w = h_in_row_ptr[v];
        for (int er = lat.in_edge_begin(v); er < lat.in_edge_end(v); ++er, ++w) {
            h_in_col_idx[w] = lat.in_edge_src(er);
            h_in_edge_id[w] = lat.in_edge_forward_id(er);
            if (!h_in_dst.empty()) h_in_dst[w] = v;
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
    if (k4 == K4Mode::Host)
        CUDA_CHECK(cudaMemcpy(d_es,  h_edge_score.data(), sizeof(float)*E, cudaMemcpyHostToDevice));
    else
        CUDA_CHECK(cudaMemset(d_es, 0, sizeof(float)*E));  // every edge is rewritten by K4
    CUDA_CHECK(cudaMemcpy(d_topo,h_topo.data(),       sizeof(int)*N, cudaMemcpyHostToDevice));
    d.in_row_ptr=d_irp; d.in_col_idx=d_ici; d.in_edge_id=d_iei; d.edge_score=d_es; d.topo_level=d_topo;
    d.best_score=nullptr; d.best_parent=nullptr;


    //  K4 setup: weights, node features and the projection scratch
    //
    // Everything here is allocated once and reused by every chunk and every K.
    // The scratch is the only size decision, and it scales with `hidden`, not
    // with feat_dim: at hidden=128 a node costs 2*128*4 = 1 KB of S+D, so a
    // 512 MB cap covers ~512K nodes and K4 walks the chunk in tiles that big.
    // The longest sentence in the g95 corpus is 400 nodes, so a tile holds
    // ~1,300 of the longest sentences and the guard below has vast headroom.
    //
    // The node-feature upload is the large allocation, and it DOES scale with
    // feat_dim: 4,488,155 nodes * 192 * 4 B = 3.21 GiB for the headline model,
    // plus 1.17 GiB for in_col_idx/in_edge_id/in_dst/edge_score over the
    // 78,847,461 edges. ~4.4 GiB resident before the k-best table. Tiles are sentence-aligned, and no edge crosses a sentence, so
    // every edge of a tile has both endpoints projected.
    GpuBiaffine bf{};
    int *d_in_dst = nullptr;
    float *d_nf = nullptr, *d_wsrc = nullptr, *d_wdst = nullptr;
    float *d_S = nullptr, *d_D = nullptr;
    int nodes_per_tile = 0;
    if (k4 != K4Mode::Host) {
        BiaffineWeights w = BiaffineWeights::load(model_path);
        if (w.feat_dim != lat.feat_dim()) {
            std::fprintf(stderr,
                "K4: model feat_dim=%d but lattice feat_dim=%d -- the .bin and "
                "the .npz came from different featurizers\n", w.feat_dim, lat.feat_dim());
            return 1;
        }
        bf.feat_dim = w.feat_dim; bf.hidden = w.hidden; bf.bias = w.bias;
        // Upload the projections TRANSPOSED, [feat_dim][hidden] instead of the
        // file's [hidden][feat_dim]. The kernels assign lane l the hidden dims
        // l, l+32, ..., so with the row-major layout the 32 lanes of a warp
        // read 32 rows 768 B apart -- 32 sectors per load instruction. ncu
        // measured project_nodes pinned at 99.24% of peak L1/TEX throughput
        // with compute at 5.98%. Transposed, consecutive lanes read consecutive
        // floats: 4 sectors instead of 32, same arithmetic, bit-identical
        // results. The CSB3 file stays row-major -- cushr_cpu's BiaffineScorer
        // reads it and model95_ctx.bin needs no re-export.
        const size_t wn = (size_t)w.hidden * w.feat_dim;
        const std::vector<float> wsT = transpose_proj(w.src_proj, w.hidden, w.feat_dim);
        const std::vector<float> wdT = transpose_proj(w.dst_proj, w.hidden, w.feat_dim);
        CUDA_CHECK(cudaMalloc(&d_wsrc, sizeof(float)*wn));
        CUDA_CHECK(cudaMalloc(&d_wdst, sizeof(float)*wn));
        CUDA_CHECK(cudaMemcpy(d_wsrc, wsT.data(), sizeof(float)*wn, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_wdst, wdT.data(), sizeof(float)*wn, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_nf, sizeof(float)*(size_t)N*w.feat_dim));
        CUDA_CHECK(cudaMemcpy(d_nf, lat.node_feature_ptr(0),
                              sizeof(float)*(size_t)N*w.feat_dim, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_in_dst, sizeof(int)*E));
        CUDA_CHECK(cudaMemcpy(d_in_dst, h_in_dst.data(), sizeof(int)*E, cudaMemcpyHostToDevice));

        bf.d_node_feat = d_nf; bf.d_src_projT = d_wsrc; bf.d_dst_projT = d_wdst;

        if (k4 == K4Mode::TwoPass) {
            const size_t per_node = (size_t)2 * w.hidden * sizeof(float);
            size_t cap = (size_t)(k4_scratch_MB * 1024.0 * 1024.0);
            nodes_per_tile = (int)std::min<size_t>(cap / per_node, (size_t)N);
            if (nodes_per_tile < 1) nodes_per_tile = 1;
            CUDA_CHECK(cudaMalloc(&d_S, sizeof(float)*(size_t)nodes_per_tile*w.hidden));
            CUDA_CHECK(cudaMalloc(&d_D, sizeof(float)*(size_t)nodes_per_tile*w.hidden));
        } else {
            nodes_per_tile = N;   // fused needs no scratch; one tile is enough
        }
        // A tile must hold at least one whole sentence, or the fallback in the
        // tiling loop below ("take it anyway") would project past the end of
        // the scratch and corrupt d_S/d_D silently. The longest sentence is a
        // few hundred nodes and a tile is ~512K, so this only fires if
        // --k4-scratch was set absurdly low -- but silent corruption is the
        // wrong way to find that out.
        int max_sent_span = 0;
        for (int s = 0; s < S; ++s)
            max_sent_span = std::max(max_sent_span,
                                     lat.sentence(s).node_end - lat.sentence(s).node_begin);
        if (nodes_per_tile < max_sent_span) {
            std::fprintf(stderr,
                "K4: --k4-scratch %.0f MB gives %d nodes/tile but the longest "
                "sentence has %d nodes; raise it to at least %.0f MB\n",
                k4_scratch_MB, nodes_per_tile, max_sent_span,
                (double)max_sent_span * 2 * bf.hidden * 4 / 1048576.0);
            return 1;
        }
        std::printf("  K4: feat_dim=%d hidden=%d bias=%.6f  feats=%.0f MB  "
                    "scratch=%.0f MB (%d nodes/tile)\n",
                    w.feat_dim, w.hidden, w.bias,
                    (double)N*w.feat_dim*4/1048576.0,
                    d_S ? (double)nodes_per_tile*w.hidden*8/1048576.0 : 0.0,
                    nodes_per_tile);
    }


    const int bs = (batch <= 0) ? S : batch;
    const int n_chunks = (S + bs - 1) / bs;
    std::printf("  batch=%d  chunks=%d  (whole-corpus sweep = batch<=0)\n", bs, n_chunks);


    FILE* csv = nullptr;
    if (!csv_path.empty()) {
        csv = std::fopen(csv_path.c_str(), "w");
        if (!csv) { std::fprintf(stderr, "cannot open --csv %s\n", csv_path.c_str()); return 1; }
        // The K2 driver's header, plus batched-only columns appended at the end
        // (n_check/n_chunks/n_launches). make_benchmark_md.py looks columns up by
        // name and tolerates missing ones, so the frozen K2 CSV still parses.
        std::fprintf(csv,
            "K,n_sentences,n_gold,recall_at_K,us_per_sent_loop,us_per_sent_kernel,"
            "sent_per_sec_kernel,gpu_table_MB,gpu_used_MB,score_mismatch,count_mismatch,"
            "n_check,n_chunks,n_launches,us_per_sent_k4,us_per_sent_k3,k4_mode,scorer\n");
    }


    //  per K: chunked batched sweep over the whole corpus 
    for (int K : Ks) {
        // CPU oracle for the checked sentences. decode() always does all S, so
        // --check only bounds the comparison loop below, not this cost: raising
        // --check to S is nearly free. Skip entirely when --check 0.
        std::vector<std::vector<DecodedPath>> cpu_results;
        if (check != 0) { TopKDecoder cpu; cpu_results = cpu.decode(lat, K, *scorer); }
        const int n_check = (check < 0) ? S : std::min(check, S);   // --check -1 => all


        // cross-chunk accumulators
        double total_kernel_ms = 0.0;   // GPU-only merge time (K3)
        double total_k4_ms     = 0.0;   // GPU-only scoring time (K4), 0 when --k4 host
        double total_loop_us   = 0.0;   // host sweep wall time
        int total_k4_launches  = 0;
        int score_mismatch  = 0;
        int count_mismatch  = 0;
        long gold_total = 0, gold_hits = 0;
        double max_table_MB = 0.0, max_used_MB = 0.0;
        int total_launches = 0;


        // --dump-paths accumulators, in make_rerank_data.py's flat-candidate
        // layout: cand_nodes is every candidate's node sequence concatenated,
        // cand_off[i]..cand_off[i+1] slices out candidate i, cand_sent indexes
        // sent_ids. Node ids are GLOBAL and in path (topological) order; the
        // form filter, span-start sort and gold labelling that the Python
        // pipeline applies live in cushr_train/gpu_paths_to_rerank.py, which
        // has the cache this driver does not.
        std::vector<int>     dump_nodes, dump_sent, dump_sent_ids;
        std::vector<int64_t> dump_off{0};
        std::vector<float>   dump_score;

        cudaEvent_t ev0, ev1, ev_k4a, ev_k4b;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev_k4a));
        CUDA_CHECK(cudaEventCreate(&ev_k4b));


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


            //  K4: score this chunk's edges on the device
            //
            // Tiles are cut on sentence boundaries so that both endpoints of
            // every edge land in the same tile (edges never cross sentences),
            // which is what makes the projection scratch reusable. The scratch
            // is addressed with the pointer biased by the tile base, exactly as
            // the k-best table is, so the kernels stay global-node-indexed.
            float chunk_k4_ms = 0.0f;
            if (k4 != K4Mode::Host) {
                CUDA_CHECK(cudaEventRecord(ev_k4a, 0));
                int ts = s0;
                while (ts < s1) {
                    int te = ts;
                    const int tb = lat.sentence(ts).node_begin;
                    while (te < s1 && lat.sentence(te).node_end - tb <= nodes_per_tile) ++te;
                    if (te == ts) ++te;   // one sentence larger than a tile: take it anyway
                    const int tn = lat.sentence(te - 1).node_end;

                    bf.d_S = d_S ? d_S - (size_t)tb * bf.hidden : nullptr;
                    bf.d_D = d_D ? d_D - (size_t)tb * bf.hidden : nullptr;

                    const int slot0   = h_in_row_ptr[tb];
                    const int n_slots = h_in_row_ptr[tn] - slot0;
                    launch_score_edges(bf, k4,
                                       d_ici + slot0, d_iei + slot0, d_in_dst + slot0,
                                       /*slot0=*/0, n_slots, tb, tn, d_es, /*stream=*/0);
                    ++total_k4_launches;
                    ts = te;
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(ev_k4b, 0));
                CUDA_CHECK(cudaEventSynchronize(ev_k4b));
                CUDA_CHECK(cudaEventElapsedTime(&chunk_k4_ms, ev_k4a, ev_k4b));
                total_k4_ms += chunk_k4_ms;
            }


            // K3 STUFF. looping over topological levels and not sentences.
            // loc_max = depth of longest sentence in the chunk.
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(ev0, 0));
            for (int L = 0; L <= loc_max; ++L) {
                const int off = level_ptr[L]; // offset where level L's nodes begin in the flat array
                const int cnt = level_ptr[L + 1] - off; // cnt is number of nodes at level L
                if (cnt <= 0) continue;
                // d=device side lattice, kb= k best table
                // d_flat + off = all levels node list, off is where level L's nodes start
                // cnt = num nodes in that slice
                // 256 = threads per blok, 8 nodes per block
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
            // Wall time for the whole per-chunk pipeline, K4 included -- the
            // "2x slowdown vs. the unscored decoder" target is about the
            // pipeline, not about K3 in isolation.
            total_loop_us  += std::chrono::duration<double, std::micro>(t1 - t0).count()
                            + chunk_k4_ms * 1000.0;


            //  copy back the chunk table + K5 reconstruct / verify 
            
            const int chk_lo = std::max(s0, 0);
            const int chk_hi = std::min(s1, n_check);
            // --dump-paths needs every sentence reconstructed, not just the
            // verified prefix, so it widens this block's trigger and its inner
            // loop; the CPU-oracle comparison still only runs below n_check.
            if (chk_hi > chk_lo || !dump_path.empty()) {
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


                const int lo = dump_path.empty() ? chk_lo : s0;
                const int hi = dump_path.empty() ? chk_hi : s1;
                for (int s = lo; s < hi; ++s) {
                    const int sink = lat.sentence(s).node_end - 1;
                    const int gpu_cnt = h_count[sink - nb0];


                    if (s >= chk_lo && s < chk_hi) {
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
                            if (std::fabs(a[r] - b[r]) > tol * (1.0f + std::fabs(b[r]))) ok = false;
                        if (!ok) ++score_mismatch;
                    }


                    //  K5 reconstruct + top-K recall vs gold
                    // With --dump-paths every rank is reconstructed anyway, so
                    // reconstruct once and reuse the result for both the recall
                    // check and the dump rather than walking the back-pointers
                    // twice.
                    const bool want_gold = (s >= chk_lo && s < chk_hi) && lat.has_explicit_gold();
                    std::vector<int> gold;
                    if (want_gold) gold = lat.gold_path(s);
                    const bool count_gold = want_gold && !gold.empty();
                    if (count_gold) ++gold_total;
                    bool hit = false;

                    const int si = (int)dump_sent_ids.size();
                    if (!dump_path.empty()) dump_sent_ids.push_back(s);

                    for (int r = 0; r < gpu_cnt; ++r) {
                        if (hit && dump_path.empty()) break;   // recall only needs the first hit
                        std::vector<int> pred = strip_boundaries(
                            reconstruct_gpu_path(g_pnode, g_prank, g_count, K, sink, r));
                        if (count_gold && !hit && pred == gold) hit = true;
                        if (!dump_path.empty()) {
                            dump_nodes.insert(dump_nodes.end(), pred.begin(), pred.end());
                            dump_off.push_back((int64_t)dump_nodes.size());
                            dump_sent.push_back(si);
                            dump_score.push_back(h_score[(size_t)(sink - nb0)*K + r]);
                        }
                    }
                    if (count_gold && hit) ++gold_hits;
                }
            }


            if (d_src)  cudaFree(d_src);
            if (d_flat) cudaFree(d_flat);
            cudaFree(base_score); cudaFree(base_pnode); cudaFree(base_prank); cudaFree(base_count);
        }


        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaEventDestroy(ev_k4a));
        CUDA_CHECK(cudaEventDestroy(ev_k4b));


        // us_per_sent_kernel / sent_per_sec_kernel now cover K4+K3 together, so
        // a --k4 host row and a --k4 twopass row from the same corpus are
        // directly comparable and their ratio is the cost of learned scoring.
        // us_per_sent_k4 breaks out the K4 half on its own.
        const double total_gpu_ms = total_kernel_ms + total_k4_ms;
        const double us_k4     = total_k4_ms * 1000.0 / std::max(1, S);
        const double us_k3     = total_kernel_ms * 1000.0 / std::max(1, S);
        const double us_kernel = total_gpu_ms * 1000.0 / std::max(1, S);
        const double us_loop   = total_loop_us / std::max(1, S);
        const double sent_per_sec = total_gpu_ms > 0.0 ? (double)S / (total_gpu_ms / 1000.0) : 0.0;
        const double recall = gold_total > 0 ? (double)gold_hits / (double)gold_total : -1.0;


        std::printf("=== K=%2d ===  %d chunks, %d K3 + %d K4 launches,"
                    " K4 %.2f ms + K3 %.2f ms"
                    "  %.2f us/sent (kernel)  %.0f sent/sec"
                    "  table=%.1f MB used=%.1f MB"
                    "  check(%d): %s (score_mm=%d count_mm=%d)  recall@K=%s\n",
                    K, n_chunks, total_launches, total_k4_launches,
                    total_k4_ms, total_kernel_ms, us_kernel, sent_per_sec,
                    max_table_MB, max_used_MB, n_check,
                    (n_check == 0 ? "skipped" : score_mismatch == 0 ? "SCORE-EQUIVALENT" : "FAILED"),
                    score_mismatch, count_mismatch,
                    (gold_total > 0 ? "" : "NA"));
        if (gold_total > 0)
            std::printf("             recall@%d = %.4f (%ld/%ld)\n", K, recall, gold_hits, gold_total);


        if (csv) {
            // n_sentences is S (the throughput denominator: every sentence is
            // swept); n_check is how many were verified against the CPU decoder.
            // They are different numbers -- do not report S as "checked".
            if (gold_total > 0)
                std::fprintf(csv, "%d,%d,%ld,%.6f,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d,%d,%d,%d,%.3f,%.3f,%s,%s\n",
                             K, S, gold_total, recall, us_loop, us_kernel,
                             sent_per_sec, max_table_MB, max_used_MB, score_mismatch, count_mismatch,
                             n_check, n_chunks, total_launches,
                             us_k4, us_k3, k4_name.c_str(), scorer_name.c_str());
            else
                std::fprintf(csv, "%d,%d,%ld,NA,%.3f,%.3f,%.1f,%.3f,%.3f,%d,%d,%d,%d,%d,%.3f,%.3f,%s,%s\n",
                             K, S, gold_total, us_loop, us_kernel,
                             sent_per_sec, max_table_MB, max_used_MB, score_mismatch, count_mismatch,
                             n_check, n_chunks, total_launches,
                             us_k4, us_k3, k4_name.c_str(), scorer_name.c_str());
            std::fflush(csv);
        }


        if (!dump_path.empty()) {
            // One file per K: a sweep over several Ks would otherwise overwrite
            // itself, and the candidate lists are not interchangeable.
            std::string out = dump_path;
            const size_t dot = out.rfind(".npz");
            const std::string suffix = "_K" + std::to_string(K);
            if (dot == std::string::npos) out += suffix + ".npz";
            else out.insert(dot, suffix);
            cnpy::npz_save(out, "cand_nodes", dump_nodes, "w");
            cnpy::npz_save(out, "cand_off",   dump_off,   "a");
            cnpy::npz_save(out, "cand_sent",  dump_sent,  "a");
            cnpy::npz_save(out, "cand_score", dump_score, "a");
            cnpy::npz_save(out, "sent_ids",   dump_sent_ids, "a");
            const std::vector<int> kk{K};
            cnpy::npz_save(out, "K", kk, "a");
            std::printf("             wrote %s  (%zu sentences, %zu candidates)\n",
                        out.c_str(), dump_sent_ids.size(), dump_sent.size());
        }
    }


    if (csv) std::fclose(csv);
    cudaFree(d_irp); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_es); cudaFree(d_topo);
    if (d_in_dst) cudaFree(d_in_dst);
    if (d_nf)     cudaFree(d_nf);
    if (d_wsrc)   cudaFree(d_wsrc);
    if (d_wdst)   cudaFree(d_wdst);
    if (d_S)      cudaFree(d_S);
    if (d_D)      cudaFree(d_D);
    return 0;
}
