// unit tests for k best merge kernel

// builds lattices in reverse CSR form and runs init_kbest, kbest_merge_level level by level
// copy table back , compare to CPU brute force top-K reference

// k=1 case: K2=top-Klist per node must produce identical output to k1
// K=8,16,32 against top-K must match within some level
// stress tests: branching factor 20, 100 levels
// k=64, work but slower

#include "../gpu_lattice.cuh"
#include "../gpu_kbest.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cushr {
__global__ void init_kbest(GpuLattice lat, GpuKBest kb);
void launch_kbest_merge(const GpuLattice& lat, const GpuKBest& kb,
                        const int* d_nodes, int n_nodes,
                        int threads_per_block, cudaStream_t stream);
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
#define CHECK(cond, msg)                                                       \
    do { if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++g_failed; } } while (0)

// -----------------------------------------------------------------------------
// A candidate triple + the SAME total order the kernel uses (EntryGreater).
// -----------------------------------------------------------------------------
struct Ent { float score; int pnode; int prank; };
static bool cpu_better(const Ent& a, const Ent& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.pnode != b.pnode) return a.pnode < b.pnode;
    return a.prank < b.prank;
}

// -----------------------------------------------------------------------------
// CPU brute-force reference: fill a per-node top-K table exactly like the GPU
// should, walking nodes in the given level order. Returns table[v] = sorted beam.
// -----------------------------------------------------------------------------
static std::vector<std::vector<Ent>>
cpu_reference(int N, const std::vector<int>& irp, const std::vector<int>& ici,
              const std::vector<int>& iei, const std::vector<float>& es,
              const std::vector<std::vector<int>>& levels, int K) {
    std::vector<std::vector<Ent>> tk(N);
    for (auto& lvl : levels) {
        for (int v : lvl) {
            if (irp[v] == irp[v + 1]) {           // source
                tk[v] = {{0.0f, -1, -1}};
                continue;
            }
            std::vector<Ent> buf;
            for (int er = irp[v]; er < irp[v + 1]; ++er) {
                int u = ici[er]; float w = es[iei[er]];
                for (int r = 0; r < (int)tk[u].size(); ++r)
                    buf.push_back({tk[u][r].score + w, u, r});
            }
            int keep = std::min((int)buf.size(), K);
            std::partial_sort(buf.begin(), buf.begin() + keep, buf.end(), cpu_better);
            tk[v].assign(buf.begin(), buf.begin() + keep);
        }
    }
    return tk;
}

// -----------------------------------------------------------------------------
// Device harness: upload, seed, sweep levels, read the table back.
// -----------------------------------------------------------------------------
struct GpuTable { std::vector<float> score; std::vector<int> pnode, prank, count; int K; };

static GpuTable run(int N, int E,
                    const std::vector<int>& irp, const std::vector<int>& ici,
                    const std::vector<int>& iei, const std::vector<float>& es,
                    const std::vector<int>& topo,
                    const std::vector<std::vector<int>>& levels, int K) {
    GpuLattice d{};
    d.num_nodes = N; d.num_edges = E; d.max_level = (int)levels.size();
    int *d_irp=nullptr,*d_ici=nullptr,*d_iei=nullptr,*d_topo=nullptr; float* d_es=nullptr;
    CK(cudaMalloc(&d_irp, sizeof(int)*(N+1)));
    CK(cudaMalloc(&d_ici, sizeof(int)*(E?E:1)));
    CK(cudaMalloc(&d_iei, sizeof(int)*(E?E:1)));
    CK(cudaMalloc(&d_es,  sizeof(float)*(E?E:1)));
    CK(cudaMalloc(&d_topo,sizeof(int)*(N?N:1)));
    CK(cudaMemcpy(d_irp, irp.data(), sizeof(int)*(N+1), cudaMemcpyHostToDevice));
    if (E) {
        CK(cudaMemcpy(d_ici, ici.data(), sizeof(int)*E, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_iei, iei.data(), sizeof(int)*E, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_es,  es.data(),  sizeof(float)*E, cudaMemcpyHostToDevice));
    }
    if (N) CK(cudaMemcpy(d_topo, topo.data(), sizeof(int)*N, cudaMemcpyHostToDevice));
    d.in_row_ptr=d_irp; d.in_col_idx=d_ici; d.in_edge_id=d_iei; d.edge_score=d_es; d.topo_level=d_topo;
    d.best_score=nullptr; d.best_parent=nullptr;

    GpuKBest kb{};
    kb.K = K; kb.cap = (K <= 32) ? 32 : 64;
    CK(cudaMalloc(&kb.score, sizeof(float)*(size_t)(N?N:1)*K));
    CK(cudaMalloc(&kb.pnode, sizeof(int)  *(size_t)(N?N:1)*K));
    CK(cudaMalloc(&kb.prank, sizeof(int)  *(size_t)(N?N:1)*K));
    CK(cudaMalloc(&kb.count, sizeof(int)  *(size_t)(N?N:1)));

    if (N) { int t=256,b=(N+t-1)/t; init_kbest<<<b,t>>>(d,kb); CK(cudaGetLastError()); }

    int* d_nodes=nullptr; int cap=0;
    for (auto& lvl : levels) {
        if (lvl.empty()) continue;
        if (irp[lvl[0]] == irp[lvl[0]+1]) continue;   // source level: seeded
        if ((int)lvl.size() > cap) {
            if (d_nodes) CK(cudaFree(d_nodes));
            CK(cudaMalloc(&d_nodes, sizeof(int)*lvl.size())); cap=(int)lvl.size();
        }
        CK(cudaMemcpy(d_nodes, lvl.data(), sizeof(int)*lvl.size(), cudaMemcpyHostToDevice));
        launch_kbest_merge(d, kb, d_nodes, (int)lvl.size(), 256, 0);
        CK(cudaGetLastError());
    }
    CK(cudaDeviceSynchronize());

    GpuTable g; g.K = K;
    g.score.resize((size_t)N*K); g.pnode.resize((size_t)N*K);
    g.prank.resize((size_t)N*K); g.count.resize(N);
    if (N) {
        CK(cudaMemcpy(g.score.data(), kb.score, sizeof(float)*(size_t)N*K, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g.pnode.data(), kb.pnode, sizeof(int)*(size_t)N*K, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g.prank.data(), kb.prank, sizeof(int)*(size_t)N*K, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(g.count.data(), kb.count, sizeof(int)*N, cudaMemcpyDeviceToHost));
    }
    if (d_nodes) cudaFree(d_nodes);
    cudaFree(d_irp); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_es); cudaFree(d_topo);
    cudaFree(kb.score); cudaFree(kb.pnode); cudaFree(kb.prank); cudaFree(kb.count);
    return g;
}

// Compare the GPU beam of every node against the CPU reference. Scores must
// match within epsilon; when all scores are distinct we also demand identical
// (pnode, prank) ordering.
static void compare_all(const char* name, int N, int K,
                        const GpuTable& g,
                        const std::vector<std::vector<Ent>>& ref) {
    for (int v = 0; v < N; ++v) {
        const auto& r = ref[v];
        CHECK(g.count[v] == (int)r.size(), name);
        int n = std::min<int>(g.count[v], (int)r.size());
        for (int i = 0; i < n; ++i) {
            float gs = g.score[(size_t)v*K + i];
            bool score_ok = std::fabs(gs - r[i].score) <= 1e-4f*(1.0f+std::fabs(r[i].score));
            CHECK(score_ok, name);
            // parent identity check (exact ordering) when scores are distinct.
            bool distinct = true;
            for (int j = 0; j < (int)r.size(); ++j)
                if (j != i && r[j].score == r[i].score) distinct = false;
            if (distinct) {
                CHECK(g.pnode[(size_t)v*K+i] == r[i].pnode, name);
                CHECK(g.prank[(size_t)v*K+i] == r[i].prank, name);
            }
        }
    }
}

// ---- test 1: K=1 single-best over a small branching node ----
static void test_k1() {
    std::printf("test_k1 (K=1 == argmax)\n");
    const int P=5, sink=P, N=P+1, E=P;
    std::vector<int> irp(N+1,0); irp[sink+1]=P;
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    for (int i=0;i<P;++i){ici[i]=i;iei[i]=i;es[i]=(float)(i+1);}   // parent 4 best (es=5)
    std::vector<int> topo(N,0); topo[sink]=1;
    std::vector<int> pl; for(int i=0;i<P;++i) pl.push_back(i);
    auto g = run(N,E,irp,ici,iei,es,topo,{pl,{sink}},1);
    auto ref = cpu_reference(N,irp,ici,iei,es,{pl,{sink}},1);
    compare_all("k1", N, 1, g, ref);
}

// ---- test 2: single edge ----
static void test_single_edge() {
    std::printf("test_single_edge\n");
    std::vector<int> irp={0,0,1}, ici={0}, iei={0}; std::vector<float> es={2.5f};
    std::vector<int> topo={0,1};
    auto g = run(2,1,irp,ici,iei,es,topo,{{0},{1}},8);
    auto ref = cpu_reference(2,irp,ici,iei,es,{{0},{1}},8);
    compare_all("single_edge", 2, 8, g, ref);
    CHECK(g.count[1]==1, "single edge sink has 1 path");
}

// ---- test 3: diamond, K=8. 0 -> {1,2} -> 3, with 1 and 2 carrying beams ----
// Actually build a 2-level fan so the sink merges two non-trivial beams.
static void test_diamond_k8() {
    std::printf("test_diamond_k8\n");
    // Level0: sources 0,1,2,3 (4 sources)
    // Level1: nodes 4,5 each take all four sources (so each has a beam of 4)
    // Level2: sink 6 takes 4 and 5 -> merges two beams of up to 8 candidates.
    const int N=7, E=4+4+2;
    std::vector<int> irp(N+1,0);
    // in-degrees: 0..3:0 ; 4:4 ; 5:4 ; 6:2
    int deg[7]={0,0,0,0,4,4,2};
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int u=0;u<4;++u){ici[w]=u;iei[w]=eid;es[eid]=1.0f+0.1f*u; ++w;++eid;} // ->4
    for (int u=0;u<4;++u){ici[w]=u;iei[w]=eid;es[eid]=2.0f+0.1f*u; ++w;++eid;} // ->5
    ici[w]=4;iei[w]=eid;es[eid]=0.5f;++w;++eid;   // 4->6
    ici[w]=5;iei[w]=eid;es[eid]=0.7f;++w;++eid;   // 5->6
    std::vector<int> topo={0,0,0,0,1,1,2};
    std::vector<std::vector<int>> levels={{0,1,2,3},{4,5},{6}};
    auto g = run(N,E,irp,ici,iei,es,topo,levels,8);
    auto ref = cpu_reference(N,irp,ici,iei,es,levels,8);
    compare_all("diamond_k8", N, 8, g, ref);
}

// ---- test 4: tie-break — two parents give equal score; smaller parent first --
static void test_tie_break() {
    std::printf("test_tie_break\n");
    const int P=6, sink=P, N=P+1, E=P;
    std::vector<int> irp(N+1,0); irp[sink+1]=P;
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    for (int i=0;i<P;++i){ici[i]=i;iei[i]=i;es[i]=1.0f;}
    es[3]=9.0f; es[5]=9.0f;      // tie between parent 3 and 5
    std::vector<int> topo(N,0); topo[sink]=1;
    std::vector<int> pl; for(int i=0;i<P;++i) pl.push_back(i);
    auto g = run(N,E,irp,ici,iei,es,topo,{pl,{sink}},8);
    // best two must be parent 3 then parent 5 (smaller parent first).
    CHECK(g.count[sink] >= 2, "tie: at least 2 paths");
    CHECK(std::fabs(g.score[(size_t)sink*8+0]-9.0f) < 1e-5f, "tie: best score 9");
    CHECK(g.pnode[(size_t)sink*8+0]==3, "tie: 1st parent is 3");
    CHECK(g.pnode[(size_t)sink*8+1]==5, "tie: 2nd parent is 5");
}

// ---- test 5: branching stress, K=16 and K=32 over a 3-level chain ----
static void test_stress() {
    std::printf("test_stress (K=16, K=32)\n");
    // Level0: 20 sources. Level1: 3 hubs each taking all 20 sources.
    // Level2: sink taking all 3 hubs -> merges 3 beams of up to K.
    const int SRC=20, HUB=3;
    const int N = SRC + HUB + 1;
    const int sink = N - 1;
    const int E = SRC*HUB + HUB;
    std::vector<int> irp(N+1,0);
    std::vector<int> deg(N,0);
    for (int h=0;h<HUB;++h) deg[SRC+h]=SRC;
    deg[sink]=HUB;
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int h=0;h<HUB;++h)
        for (int u=0;u<SRC;++u){ici[w]=u;iei[w]=eid;es[eid]=0.01f*eid+0.5f;++w;++eid;}
    for (int h=0;h<HUB;++h){ici[w]=SRC+h;iei[w]=eid;es[eid]=0.3f+0.05f*h;++w;++eid;}
    std::vector<int> topo(N,0);
    for (int h=0;h<HUB;++h) topo[SRC+h]=1;
    topo[sink]=2;
    std::vector<int> l0; for(int i=0;i<SRC;++i) l0.push_back(i);
    std::vector<int> l1; for(int h=0;h<HUB;++h) l1.push_back(SRC+h);
    std::vector<std::vector<int>> levels={l0,l1,{sink}};
    for (int K : {16, 32}) {
        auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
        auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
        compare_all(K==16?"stress_k16":"stress_k32", N, K, g, ref);
    }
}

// ---- test 6: full-capacity real merge. Several parents each carrying a FULL
// beam of K distinct real paths, so the sink must truncate genuine competitors
// (no -inf padding involved in the winners). Run at K=32 and K=64 to exercise
// both the max=32 and max=64 templates and the 2-slots-per-lane load path. ----
static void test_full_capacity() {
    std::printf("test_full_capacity (K=32, K=64)\n");
    for (int K : {32, 64}) {
        // Level0: K sources (so each hub can build a beam of exactly K).
        // Level1: HUB hubs, each taking all K sources -> full beam of K.
        // Level2: sink taking all hubs -> merges HUB full beams of K, must keep top K.
        const int SRC = K, HUB = 3;
        const int N = SRC + HUB + 1;
        const int sink = N - 1;
        const int E = SRC*HUB + HUB;
        std::vector<int> irp(N+1,0), deg(N,0);
        for (int h=0;h<HUB;++h) deg[SRC+h]=SRC;
        deg[sink]=HUB;
        for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
        std::vector<int> ici(E), iei(E); std::vector<float> es(E);
        int w=0, eid=0;
        // distinct edge weights so every candidate score is distinct.
        for (int h=0;h<HUB;++h)
            for (int u=0;u<SRC;++u){ici[w]=u;iei[w]=eid;es[eid]=1.0f+0.013f*u+0.7f*h;++w;++eid;}
        for (int h=0;h<HUB;++h){ici[w]=SRC+h;iei[w]=eid;es[eid]=0.31f+0.017f*h;++w;++eid;}
        std::vector<int> topo(N,0);
        for (int h=0;h<HUB;++h) topo[SRC+h]=1;
        topo[sink]=2;
        std::vector<int> l0; for(int i=0;i<SRC;++i) l0.push_back(i);
        std::vector<int> l1; for(int h=0;h<HUB;++h) l1.push_back(SRC+h);
        std::vector<std::vector<int>> levels={l0,l1,{sink}};
        auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
        auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
        compare_all(K==32?"full_cap_k32":"full_cap_k64", N, K, g, ref);
        // sink must be saturated at exactly K (HUB*K >= K candidates available).
        CHECK(g.count[sink]==K, K==32?"full_cap_k32 sink saturated":"full_cap_k64 sink saturated");
    }
}

// ---- test 7: K=64 diamond. Smaller than full-capacity but specifically drives
// the max=64 template with a non-trivial two-beam merge and under-K counts. ----
static void test_k64_diamond() {
    std::printf("test_k64_diamond (K=64)\n");
    const int K=64;
    // 40 sources -> 2 hubs (beam of 40 each) -> sink merges two beams of 40 = 80
    // candidates, truncated to 64.
    const int SRC=40, HUB=2, N=SRC+HUB+1, sink=N-1, E=SRC*HUB+HUB;
    std::vector<int> irp(N+1,0), deg(N,0);
    for (int h=0;h<HUB;++h) deg[SRC+h]=SRC;
    deg[sink]=HUB;
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int h=0;h<HUB;++h)
        for (int u=0;u<SRC;++u){ici[w]=u;iei[w]=eid;es[eid]=0.5f+0.011f*u+0.37f*h;++w;++eid;}
    for (int h=0;h<HUB;++h){ici[w]=SRC+h;iei[w]=eid;es[eid]=0.2f+0.09f*h;++w;++eid;}
    std::vector<int> topo(N,0);
    for (int h=0;h<HUB;++h) topo[SRC+h]=1;
    topo[sink]=2;
    std::vector<int> l0; for(int i=0;i<SRC;++i) l0.push_back(i);
    std::vector<int> l1; for(int h=0;h<HUB;++h) l1.push_back(SRC+h);
    std::vector<std::vector<int>> levels={l0,l1,{sink}};
    auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
    auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
    compare_all("k64_diamond", N, K, g, ref);
    CHECK(g.count[sink]==K, "k64_diamond sink truncated to 64");
}

// ---- test 8: empty lattice (N=0, no nodes/edges). Must not crash; nothing to
// compare. Exercises the run() degenerate-size guards. ----
static void test_empty_lattice() {
    std::printf("test_empty_lattice (N=0)\n");
    std::vector<int> irp={0}, ici, iei; std::vector<float> es; std::vector<int> topo;
    auto g = run(0,0,irp,ici,iei,es,topo,{},8);
    CHECK(g.count.empty(), "empty lattice has no counts");
}

// ---- test 9: single node, no edges. The lone node is a source, so its beam is
// exactly one path (score 0, parent -1). ----
static void test_single_node() {
    std::printf("test_single_node (N=1)\n");
    std::vector<int> irp={0,0}, ici, iei; std::vector<float> es; std::vector<int> topo={0};
    auto g = run(1,0,irp,ici,iei,es,topo,{{0}},8);
    auto ref = cpu_reference(1,irp,ici,iei,es,{{0}},8);
    compare_all("single_node", 1, 8, g, ref);
    CHECK(g.count[0]==1, "single node is a source with 1 path");
    CHECK(std::fabs(g.score[0]) < 1e-6f, "single node source score is 0");
    CHECK(g.pnode[0]==-1, "single node source has no parent");
}

// ---- test 10: deep lattice — branching factor 20, 100 levels deep. Every node
// in level L takes all 20 nodes of level L-1 (in-degree 20), so back-pointer
// chains are 100 hops long and any per-level propagation bug compounds. Run at
// K=16 and K=32 against the CPU reference. ----
static void test_deep_stress() {
    std::printf("test_deep_stress (branching 20, 100 levels; K=16, K=32)\n");
    const int LEVELS = 100, W = 20;
    const int N = LEVELS * W + 1;      // +1 sink
    const int sink = N - 1;
    const int E = (LEVELS - 1) * W * W + W;   // inter-level fans + sink's in-edges

    std::vector<int> irp(N+1,0), deg(N,0);
    for (int L = 1; L < LEVELS; ++L)          // levels 1..99: each node in-degree W
        for (int j = 0; j < W; ++j) deg[L*W + j] = W;
    deg[sink] = W;                            // sink takes all of the last level
    for (int v = 0; v < N; ++v) irp[v+1] = irp[v] + deg[v];

    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int L = 1; L < LEVELS; ++L) {
        for (int j = 0; j < W; ++j) {         // dst node L*W+j takes all of level L-1
            for (int u = 0; u < W; ++u) {
                ici[w] = (L-1)*W + u; iei[w] = eid;
                // distinct-ish weights so the merge has real winners to order.
                es[eid] = 0.5f + 0.001f*eid + 0.01f*u + 0.007f*j;
                ++w; ++eid;
            }
        }
    }
    for (int u = 0; u < W; ++u) {             // sink takes all of level LEVELS-1
        ici[w] = (LEVELS-1)*W + u; iei[w] = eid; es[eid] = 0.3f + 0.013f*u; ++w; ++eid;
    }

    std::vector<int> topo(N,0);
    for (int L = 0; L < LEVELS; ++L)
        for (int j = 0; j < W; ++j) topo[L*W + j] = L;
    topo[sink] = LEVELS;

    std::vector<std::vector<int>> levels(LEVELS + 1);
    for (int L = 0; L < LEVELS; ++L)
        for (int j = 0; j < W; ++j) levels[L].push_back(L*W + j);
    levels[LEVELS].push_back(sink);

    for (int K : {16, 32}) {
        auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
        auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
        compare_all(K==16?"deep_stress_k16":"deep_stress_k32", N, K, g, ref);
        CHECK(g.count[sink]==K, K==16?"deep_stress_k16 sink saturated"
                                     :"deep_stress_k32 sink saturated");
    }
}

// ---- test 11: NEGATIVE edge weights (log-probs). Every edge score is < 0, so
// "best" = least-negative and the -inf padding must still sort to the tail. The
// real LogLinearScorer emits negative scores, so this is the production regime.
// Same diamond shape as test 3, run at K=8 and K=32. ----
static void test_negative_weights() {
    std::printf("test_negative_weights (K=8, K=32)\n");
    const int N=7, E=4+4+2;
    std::vector<int> irp(N+1,0);
    int deg[7]={0,0,0,0,4,4,2};
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int u=0;u<4;++u){ici[w]=u;iei[w]=eid;es[eid]=-(1.0f+0.1f*u); ++w;++eid;} // ->4
    for (int u=0;u<4;++u){ici[w]=u;iei[w]=eid;es[eid]=-(2.0f+0.1f*u); ++w;++eid;} // ->5
    ici[w]=4;iei[w]=eid;es[eid]=-0.5f;++w;++eid;   // 4->6
    ici[w]=5;iei[w]=eid;es[eid]=-0.7f;++w;++eid;   // 5->6
    std::vector<int> topo={0,0,0,0,1,1,2};
    std::vector<std::vector<int>> levels={{0,1,2,3},{4,5},{6}};
    for (int K : {8, 32}) {
        auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
        auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
        compare_all(K==8?"neg_k8":"neg_k32", N, K, g, ref);
    }
}

// ---- test 12: tie-break AT THE TRUNCATION BOUNDARY. A sink has P parents that
// ALL yield the identical score, and K < P, so the beam cut falls in the middle
// of a fully-tied group. The strict total order (score desc, then parent_node
// asc, then parent_rank asc) must keep exactly the K smallest parent ids and
// DROP the rest -- deterministically, matching the CPU. compare_all skips
// (pnode,prank) when scores tie, so this test asserts the kept parents directly. --
static void test_tie_break_boundary() {
    std::printf("test_tie_break_boundary (K cuts a fully-tied group)\n");
    const int P=10, sink=P, N=P+1, E=P, K=4;
    std::vector<int> irp(N+1,0); irp[sink+1]=P;
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    for (int i=0;i<P;++i){ici[i]=i;iei[i]=i;es[i]=5.0f;}   // every parent ties at 5.0
    std::vector<int> topo(N,0); topo[sink]=1;
    std::vector<int> pl; for(int i=0;i<P;++i) pl.push_back(i);
    auto g = run(N,E,irp,ici,iei,es,topo,{pl,{sink}},K);
    auto ref = cpu_reference(N,irp,ici,iei,es,{pl,{sink}},K);
    compare_all("tie_boundary", N, K, g, ref);
    CHECK(g.count[sink]==K, "tie boundary: sink saturated at K");
    // the K survivors must be exactly the K smallest parent ids, in order.
    for (int i=0;i<K;++i) {
        CHECK(std::fabs(g.score[(size_t)sink*K+i]-5.0f) < 1e-5f, "tie boundary: score 5");
        CHECK(g.pnode[(size_t)sink*K+i]==i, "tie boundary: keeps smallest parent ids in order");
    }
}

// ---- test 13: NON-power-of-two K. K in {5, 24, 48} exercises the "emit only the
// first K" masking for K that is neither a power of two nor the template cap
// (5,24 -> cap 32; 48 -> cap 64). K=5 is used by the actual benchmark. Sources
// feed 3 hubs with full beams so the sink truncates 3*K genuine candidates. ----
static void test_odd_k() {
    std::printf("test_odd_k (K=5, K=24, K=48)\n");
    const int SRC=64, HUB=3;
    const int N = SRC + HUB + 1;
    const int sink = N - 1;
    const int E = SRC*HUB + HUB;
    std::vector<int> irp(N+1,0), deg(N,0);
    for (int h=0;h<HUB;++h) deg[SRC+h]=SRC;
    deg[sink]=HUB;
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int h=0;h<HUB;++h)
        for (int u=0;u<SRC;++u){ici[w]=u;iei[w]=eid;es[eid]=1.0f+0.013f*u+0.7f*h;++w;++eid;}
    for (int h=0;h<HUB;++h){ici[w]=SRC+h;iei[w]=eid;es[eid]=0.31f+0.017f*h;++w;++eid;}
    std::vector<int> topo(N,0);
    for (int h=0;h<HUB;++h) topo[SRC+h]=1;
    topo[sink]=2;
    std::vector<int> l0; for(int i=0;i<SRC;++i) l0.push_back(i);
    std::vector<int> l1; for(int h=0;h<HUB;++h) l1.push_back(SRC+h);
    std::vector<std::vector<int>> levels={l0,l1,{sink}};
    for (int K : {5, 24, 48}) {
        auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
        auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
        const char* nm = K==5?"odd_k5":K==24?"odd_k24":"odd_k48";
        compare_all(nm, N, K, g, ref);
        CHECK(g.count[sink]==K, "odd_k: sink saturated at K");
    }
}

// ---- test 14: pure PROPAGATION chain, in-degree 1 for 50 levels. No merging
// ever happens; the single beam of size 1 must propagate untouched and the final
// score must equal the exact sum of edge weights. Isolates the "carry one path
// forward" path from any truncation logic. ----
static void test_chain_propagation() {
    std::printf("test_chain_propagation (in-degree 1, 50 levels)\n");
    const int L=50, N=L+1, E=L, K=8;
    std::vector<int> irp(N+1,0), deg(N,0);
    for (int v=1;v<N;++v) deg[v]=1;              // node v takes only node v-1
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    double expect=0.0;
    for (int e=0;e<E;++e){ici[e]=e;iei[e]=e;es[e]=0.1f+0.03f*e; expect+=es[e];}
    std::vector<int> topo(N,0); for (int v=0;v<N;++v) topo[v]=v;
    std::vector<std::vector<int>> levels(N);
    for (int v=0;v<N;++v) levels[v].push_back(v);
    auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
    auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
    compare_all("chain", N, K, g, ref);
    CHECK(g.count[L]==1, "chain: sink has exactly one path");
    CHECK(std::fabs(g.score[(size_t)L*K+0]-(float)expect) < 1e-3f*(1.0f+std::fabs((float)expect)),
          "chain: sink score == sum of edge weights");
}

// ---- test 15: INTERLEAVED node ids (id order != topo order). Sink is node 0,
// hub is node 1, sources are nodes 2,3,4 -- so global node ids run OPPOSITE to
// topological order. Confirms every kernel index (kb table, reverse-CSR, count)
// is keyed by global node id, not by any implicit id==level assumption. ----
static void test_interleaved_levels() {
    std::printf("test_interleaved_levels (node id order reversed vs topo)\n");
    const int N=5, K=8;                  // 0=sink, 1=hub, 2,3,4=sources
    const int E=3+1;                     // hub<-3 sources, sink<-hub
    std::vector<int> deg(N,0); deg[1]=3; deg[0]=1;
    std::vector<int> irp(N+1,0);
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    // node 0 (sink) <- node 1 (hub)
    ici[w]=1; iei[w]=eid; es[eid]=0.4f; ++w; ++eid;
    // node 1 (hub) <- sources 2,3,4
    for (int u=2;u<=4;++u){ici[w]=u; iei[w]=eid; es[eid]=1.0f+0.2f*(u-2); ++w; ++eid;}
    std::vector<int> topo(N,0); topo[2]=topo[3]=topo[4]=0; topo[1]=1; topo[0]=2;
    std::vector<std::vector<int>> levels={{2,3,4},{1},{0}};
    auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
    auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
    compare_all("interleaved", N, K, g, ref);
    CHECK(g.count[0]==3, "interleaved: sink beam merges 3 source paths");
}

// ---- test 16: PARTIAL-WARP level. A single level holds 47 independent sink
// nodes (47 warps => the last block/warp is partially filled). Exercises the
// warp_id >= n_nodes_at_level guard and the per-warp count reduction when the
// launch grid does not tile the node count evenly. Each sink merges 2 parents. --
static void test_partial_warp_level() {
    std::printf("test_partial_warp_level (47 sinks in one level)\n");
    const int M=47, K=8;
    const int N = 2*M + M;               // 2M sources + M sinks
    const int E = 2*M;                   // each sink takes 2 sources
    std::vector<int> deg(N,0);
    for (int j=0;j<M;++j) deg[2*M+j]=2;
    std::vector<int> irp(N+1,0);
    for (int v=0;v<N;++v) irp[v+1]=irp[v]+deg[v];
    std::vector<int> ici(E), iei(E); std::vector<float> es(E);
    int w=0, eid=0;
    for (int j=0;j<M;++j){
        ici[w]=2*j;   iei[w]=eid; es[eid]=1.0f+0.01f*j;   ++w;++eid;
        ici[w]=2*j+1; iei[w]=eid; es[eid]=0.5f+0.02f*j;   ++w;++eid;
    }
    std::vector<int> topo(N,0);
    for (int j=0;j<M;++j) topo[2*M+j]=1;
    std::vector<int> src; for (int u=0;u<2*M;++u) src.push_back(u);
    std::vector<int> snk; for (int j=0;j<M;++j) snk.push_back(2*M+j);
    std::vector<std::vector<int>> levels={src, snk};
    auto g = run(N,E,irp,ici,iei,es,topo,levels,K);
    auto ref = cpu_reference(N,irp,ici,iei,es,levels,K);
    compare_all("partial_warp", N, K, g, ref);
    for (int j=0;j<M;++j)
        CHECK(g.count[2*M+j]==2, "partial_warp: each sink has 2 paths");
}

int main() {
    test_k1();
    test_single_edge();
    test_diamond_k8();
    test_tie_break();
    test_stress();
    test_full_capacity();
    test_k64_diamond();
    test_empty_lattice();
    test_single_node();
    test_deep_stress();
    test_negative_weights();
    test_tie_break_boundary();
    test_odd_k();
    test_chain_propagation();
    test_interleaved_levels();
    test_partial_warp_level();
    if (g_failed == 0) { std::printf("\nALL K-BEST TESTS PASSED\n"); return 0; }
    std::printf("\n%d CHECK(S) FAILED\n", g_failed);
    return 1;
}
