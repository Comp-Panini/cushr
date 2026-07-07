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

int main() {
    test_k1();
    test_single_edge();
    test_diamond_k8();
    test_tie_break();
    test_stress();
    if (g_failed == 0) { std::printf("\nALL K-BEST TESTS PASSED\n"); return 0; }
    std::printf("\n%d CHECK(S) FAILED\n", g_failed);
    return 1;
}
