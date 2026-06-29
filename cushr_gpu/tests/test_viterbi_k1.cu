// cushr_gpu/tests/test_viterbi_k1.cu
//
// Week-5 deliverable: hardening unit tests for the K=1 GPU Viterbi kernel.
//
// These tests build tiny lattices BY HAND in reverse-CSR form, run init_best +
// viterbi_relax_level level by level, copy the results back, and compare to a
// hand-computed expected answer. They are intentionally independent of the CPU
// library and the .npz format so a kernel regression fails here in isolation.
//
// Covered edge cases (from the plan):
//   1. empty lattice                  (0 nodes)
//   2. single-node lattice            (a lone source = sink)
//   3. single-edge lattice            (source -> sink)
//   4. high-branching lattice         (one node with many incoming edges,
//                                       exercising the multi-iteration strided
//                                       scan AND the warp-shuffle reduction)
//   + a tie-break test (two equal-score parents must resolve to the SMALLER
//     parent index, matching the CPU EntryGreater rule).
//
// This is a write-only file: compile and run it on the lab A100 box with
//   make test
// No external test framework — just asserts and a pass/fail tally.

#include "../gpu_lattice.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cushr {
__global__ void init_best(GpuLattice lat);
__global__ void viterbi_relax_level(GpuLattice lat, const int* nodes_at_level,
                                    int n_nodes_at_level);
}
using namespace cushr;

#define CK(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(_e));                    \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static int g_failed = 0;
#define CHECK_EQ(a, b, msg)                                                    \
    do {                                                                       \
        if (!((a) == (b))) {                                                   \
            std::printf("  FAIL: %s (got %g, want %g)\n", msg,                  \
                        (double)(a), (double)(b));                             \
            ++g_failed;                                                        \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Host harness: take reverse-CSR arrays + per-edge scores + a list of nodes
// grouped by level, run the kernels, return best_score / best_parent.
// ---------------------------------------------------------------------------
struct Result { std::vector<float> score; std::vector<int> parent; };

static Result run(int N, int E,
                  const std::vector<int>& in_row_ptr,
                  const std::vector<int>& in_col_idx,
                  const std::vector<int>& in_edge_id,
                  const std::vector<float>& edge_score,
                  const std::vector<int>& topo_level,
                  const std::vector<std::vector<int>>& levels) {
    GpuLattice d{};
    d.num_nodes = N;
    d.num_edges = E;
    d.max_level = (int)levels.size();

    int *d_irp = nullptr, *d_ici = nullptr, *d_iei = nullptr, *d_topo = nullptr;
    float* d_es = nullptr;
    CK(cudaMalloc(&d_irp, sizeof(int) * (N + 1)));
    CK(cudaMalloc(&d_ici, sizeof(int) * (E ? E : 1)));
    CK(cudaMalloc(&d_iei, sizeof(int) * (E ? E : 1)));
    CK(cudaMalloc(&d_es,  sizeof(float) * (E ? E : 1)));
    CK(cudaMalloc(&d_topo, sizeof(int) * (N ? N : 1)));
    CK(cudaMalloc(&d.best_score,  sizeof(float) * (N ? N : 1)));
    CK(cudaMalloc(&d.best_parent, sizeof(int) * (N ? N : 1)));

    CK(cudaMemcpy(d_irp, in_row_ptr.data(), sizeof(int) * (N + 1), cudaMemcpyHostToDevice));
    if (E) {
        CK(cudaMemcpy(d_ici, in_col_idx.data(), sizeof(int) * E, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_iei, in_edge_id.data(), sizeof(int) * E, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_es,  edge_score.data(), sizeof(float) * E, cudaMemcpyHostToDevice));
    }
    if (N) CK(cudaMemcpy(d_topo, topo_level.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    d.in_row_ptr = d_irp; d.in_col_idx = d_ici; d.in_edge_id = d_iei;
    d.edge_score = d_es;  d.topo_level = d_topo;

    if (N) {
        int t = 256, b = (N + t - 1) / t;
        init_best<<<b, t>>>(d);
        CK(cudaGetLastError());
    }

    int *d_nodes = nullptr; int cap = 0;
    for (auto& lvl : levels) {
        if (lvl.empty()) continue;
        if (in_row_ptr[lvl[0]] == in_row_ptr[lvl[0] + 1]) continue; // source level
        if ((int)lvl.size() > cap) {
            if (d_nodes) CK(cudaFree(d_nodes));
            CK(cudaMalloc(&d_nodes, sizeof(int) * lvl.size()));
            cap = (int)lvl.size();
        }
        CK(cudaMemcpy(d_nodes, lvl.data(), sizeof(int) * lvl.size(), cudaMemcpyHostToDevice));
        int t = 256, b = ((int)lvl.size() * 32 + t - 1) / t;
        viterbi_relax_level<<<b, t>>>(d, d_nodes, (int)lvl.size());
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());

    Result r;
    r.score.resize(N); r.parent.resize(N);
    if (N) {
        CK(cudaMemcpy(r.score.data(), d.best_score, sizeof(float) * N, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r.parent.data(), d.best_parent, sizeof(int) * N, cudaMemcpyDeviceToHost));
    }

    if (d_nodes) cudaFree(d_nodes);
    cudaFree(d_irp); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_es); cudaFree(d_topo);
    cudaFree(d.best_score); cudaFree(d.best_parent);
    return r;
}

// ---- test 1: empty lattice (0 nodes) — must not crash ----
static void test_empty() {
    std::printf("test_empty\n");
    run(0, 0, {0}, {}, {}, {}, {}, {});
    std::printf("  ok (no crash)\n");
}

// ---- test 2: single node (lone source = sink) ----
static void test_single_node() {
    std::printf("test_single_node\n");
    // N=1, no edges. in_row_ptr = [0,0] => node 0 is a source.
    auto r = run(1, 0, {0, 0}, {}, {}, {}, {0}, {{0}});
    CHECK_EQ(r.score[0], 0.0f, "source score");
    CHECK_EQ(r.parent[0], -1, "source parent");
}

// ---- test 3: single edge 0 -> 1 with score 2.5 ----
static void test_single_edge() {
    std::printf("test_single_edge\n");
    // forward edge 0: 0->1. reverse CSR:
    //   node 0: no incoming  -> in_row_ptr[0..1] = 0,0
    //   node 1: one incoming -> in_row_ptr[1..2] = 0,1, src=0, edge_id=0
    std::vector<int> irp = {0, 0, 1};
    std::vector<int> ici = {0};
    std::vector<int> iei = {0};
    std::vector<float> es = {2.5f};
    std::vector<int> topo = {0, 1};
    auto r = run(2, 1, irp, ici, iei, es, topo, {{0}, {1}});
    CHECK_EQ(r.score[0], 0.0f, "src score");
    CHECK_EQ(r.score[1], 2.5f, "sink score");
    CHECK_EQ(r.parent[1], 0, "sink parent");
}

// ---- test 4: high branching — 40 parents all feeding node 40 ----
// Forces >1 strided iteration per lane (40 > 32) and a full warp reduction.
static void test_high_branching() {
    std::printf("test_high_branching\n");
    const int P = 40;           // parents 0..39, all sources at level 0
    const int sink = P;         // node 40 at level 1
    const int N = P + 1;
    const int E = P;            // one edge from each parent to the sink

    // reverse CSR: only the sink has incoming edges.
    std::vector<int> irp(N + 1, 0);
    for (int v = 0; v <= sink; ++v) irp[v + 1] = irp[v];     // 0 in-edges for parents
    irp[sink + 1] = irp[sink] + P;                            // P in-edges for sink

    std::vector<int> ici(E), iei(E);
    std::vector<float> es(E);
    // Make parent 17 the unique winner with the largest edge score.
    for (int i = 0; i < P; ++i) { ici[i] = i; iei[i] = i; es[i] = (float)i; }
    es[17] = 100.0f;

    std::vector<int> topo(N, 0); topo[sink] = 1;
    std::vector<int> parents_level; for (int i = 0; i < P; ++i) parents_level.push_back(i);
    auto r = run(N, E, irp, ici, iei, es, topo, {parents_level, {sink}});
    CHECK_EQ(r.score[sink], 100.0f, "sink best score");
    CHECK_EQ(r.parent[sink], 17, "sink best parent");
}

// ---- test 5: tie-break — two parents with equal winning score ----
// Parents 3 and 5 both yield score 9.0; the SMALLER parent (3) must win,
// matching the CPU EntryGreater rule.
static void test_tie_break() {
    std::printf("test_tie_break\n");
    const int P = 6, sink = P, N = P + 1, E = P;
    std::vector<int> irp(N + 1, 0);
    irp[sink + 1] = P;
    std::vector<int> ici(E), iei(E);
    std::vector<float> es(E);
    for (int i = 0; i < P; ++i) { ici[i] = i; iei[i] = i; es[i] = 1.0f; }
    es[3] = 9.0f; es[5] = 9.0f;     // tie between parent 3 and parent 5
    std::vector<int> topo(N, 0); topo[sink] = 1;
    std::vector<int> pl; for (int i = 0; i < P; ++i) pl.push_back(i);
    auto r = run(N, E, irp, ici, iei, es, topo, {pl, {sink}});
    CHECK_EQ(r.score[sink], 9.0f, "tie score");
    CHECK_EQ(r.parent[sink], 3, "tie resolves to smaller parent");
}

int main() {
    test_empty();
    test_single_node();
    test_single_edge();
    test_high_branching();
    test_tie_break();

    if (g_failed == 0) { std::printf("\nALL TESTS PASSED\n"); return 0; }
    std::printf("\n%d CHECK(S) FAILED\n", g_failed);
    return 1;
}
