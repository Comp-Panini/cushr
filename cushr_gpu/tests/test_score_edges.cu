// cushr_gpu/tests/test_score_edges.cu
//
// K4 unit tests. Synthetic reverse-CSR graphs and random weights, no .npz and
// no CPU library -- the point is the arithmetic, not the loader.
//
// Three properties, in order of how much they would hurt to get wrong:
//
//   1. Both kernels agree with a straight-line host reference. This is the
//      correctness claim: edge_score[e] == <W_s x_u, W_d x_v> + b.
//   2. fused and twopass agree BITWISE. They split `hidden` across lanes the
//      same way and reduce in the same order, so any drift between them means
//      one of them was restructured without the other -- the failure that would
//      otherwise show up as an unreproducible benchmark months later.
//   3. Dimensions that are not multiples of 32 work. feat_dim is read from the
//      weight file and never hardcoded, so the cases below sweep several
//      widths: 192 (the headline model), 43 (the old lattice the log_linear
//      weights belong to) and hidden=100, which exercises the ragged tail of
//      every lane-strided loop in the file.

#include "../score_edges.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cushr;

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::printf("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,     \
                        cudaGetErrorString(_e));                               \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
    if (!ok) ++g_fail;
}

// A chain-plus-skips lattice: node v has in-edges from v-1, v-2, v-3 (clipped
// at 0). Not a real Sanskrit lattice, but it gives every node several parents
// and every parent several children, which is the reuse pattern K4a exists for.
struct Graph {
    int n_nodes = 0, n_edges = 0;
    std::vector<int> in_row_ptr, in_col_idx, in_edge_id, in_dst;
};

static Graph make_graph(int n_nodes, int fan) {
    Graph g;
    g.n_nodes = n_nodes;
    g.in_row_ptr.assign(n_nodes + 1, 0);
    int e = 0;
    for (int v = 0; v < n_nodes; ++v) {
        for (int d = 1; d <= fan; ++d) {
            const int u = v - d;
            if (u < 0) continue;
            g.in_col_idx.push_back(u);
            g.in_edge_id.push_back(e++);
            g.in_dst.push_back(v);
        }
        g.in_row_ptr[v + 1] = (int)g.in_col_idx.size();
    }
    g.n_edges = e;
    return g;
}

// The reference: exactly the formula in scorer.hpp, summed in the obvious
// order. Deliberately NOT the kernel's order -- that is what the tolerance in
// property 1 is for.
static std::vector<float> host_scores(const Graph& g, const std::vector<float>& x,
                                      const std::vector<float>& ws,
                                      const std::vector<float>& wd,
                                      int feat_dim, int hidden, float bias) {
    std::vector<float> S((size_t)g.n_nodes * hidden), D((size_t)g.n_nodes * hidden);
    for (int v = 0; v < g.n_nodes; ++v) {
        for (int h = 0; h < hidden; ++h) {
            double as = 0.0, ad = 0.0;
            for (int i = 0; i < feat_dim; ++i) {
                as += (double)ws[(size_t)h * feat_dim + i] * x[(size_t)v * feat_dim + i];
                ad += (double)wd[(size_t)h * feat_dim + i] * x[(size_t)v * feat_dim + i];
            }
            S[(size_t)v * hidden + h] = (float)as;
            D[(size_t)v * hidden + h] = (float)ad;
        }
    }
    std::vector<float> out(g.n_edges, 0.0f);
    for (int slot = 0; slot < g.n_edges; ++slot) {
        const int u = g.in_col_idx[slot], v = g.in_dst[slot], e = g.in_edge_id[slot];
        double acc = 0.0;
        for (int h = 0; h < hidden; ++h)
            acc += (double)S[(size_t)u * hidden + h] * D[(size_t)v * hidden + h];
        out[e] = (float)(acc + bias);
    }
    return out;
}

// Run one configuration through both kernels and check all three properties.
static void run_case(const std::string& label, int n_nodes, int fan,
                     int feat_dim, int hidden, float bias, unsigned seed) {
    std::printf("%s (nodes=%d fan=%d feat_dim=%d hidden=%d)\n",
                label.c_str(), n_nodes, fan, feat_dim, hidden);

    Graph g = make_graph(n_nodes, fan);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x((size_t)n_nodes * feat_dim), ws((size_t)hidden * feat_dim),
                       wd((size_t)hidden * feat_dim);
    for (auto& t : x)  t = u(rng);
    for (auto& t : ws) t = u(rng) * 0.1f;   // keep the length-`hidden` dot in a
    for (auto& t : wd) t = u(rng) * 0.1f;   // range where fp32 is not the story

    const std::vector<float> ref = host_scores(g, x, ws, wd, feat_dim, hidden, bias);

    GpuBiaffine bf{};
    bf.feat_dim = feat_dim; bf.hidden = hidden; bf.bias = bias;
    float *d_x, *d_ws, *d_wd, *d_S, *d_D, *d_out;
    int *d_ici, *d_iei, *d_dst;
    CHECK(cudaMalloc(&d_x,  sizeof(float) * x.size()));
    CHECK(cudaMalloc(&d_ws, sizeof(float) * ws.size()));
    CHECK(cudaMalloc(&d_wd, sizeof(float) * wd.size()));
    CHECK(cudaMalloc(&d_S,  sizeof(float) * (size_t)n_nodes * hidden));
    CHECK(cudaMalloc(&d_D,  sizeof(float) * (size_t)n_nodes * hidden));
    CHECK(cudaMalloc(&d_out, sizeof(float) * g.n_edges));
    CHECK(cudaMalloc(&d_ici, sizeof(int) * g.n_edges));
    CHECK(cudaMalloc(&d_iei, sizeof(int) * g.n_edges));
    CHECK(cudaMalloc(&d_dst, sizeof(int) * g.n_edges));
    CHECK(cudaMemcpy(d_x,  x.data(),  sizeof(float) * x.size(),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws, ws.data(), sizeof(float) * ws.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wd, wd.data(), sizeof(float) * wd.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ici, g.in_col_idx.data(), sizeof(int) * g.n_edges, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_iei, g.in_edge_id.data(), sizeof(int) * g.n_edges, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dst, g.in_dst.data(),     sizeof(int) * g.n_edges, cudaMemcpyHostToDevice));
    bf.d_node_feat = d_x; bf.d_src_proj = d_ws; bf.d_dst_proj = d_wd;
    bf.d_S = d_S; bf.d_D = d_D;

    std::vector<float> got_two(g.n_edges), got_fused(g.n_edges);

    CHECK(cudaMemset(d_out, 0, sizeof(float) * g.n_edges));
    launch_score_edges(bf, K4Mode::TwoPass, d_ici, d_iei, d_dst, 0, g.n_edges,
                       0, n_nodes, d_out, 0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got_two.data(), d_out, sizeof(float) * g.n_edges, cudaMemcpyDeviceToHost));

    CHECK(cudaMemset(d_out, 0, sizeof(float) * g.n_edges));
    launch_score_edges(bf, K4Mode::Fused, d_ici, d_iei, d_dst, 0, g.n_edges,
                       0, n_nodes, d_out, 0);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got_fused.data(), d_out, sizeof(float) * g.n_edges, cudaMemcpyDeviceToHost));

    double worst_two = 0.0, worst_fused = 0.0;
    int bitwise_diff = 0;
    for (int e = 0; e < g.n_edges; ++e) {
        const double scale = 1.0 + std::fabs((double)ref[e]);
        worst_two   = std::max(worst_two,   std::fabs((double)got_two[e]   - ref[e]) / scale);
        worst_fused = std::max(worst_fused, std::fabs((double)got_fused[e] - ref[e]) / scale);
        if (got_two[e] != got_fused[e]) ++bitwise_diff;
    }
    expect(worst_two   < 1e-4, "twopass matches the host reference (worst rel "
                               + std::to_string(worst_two) + ")");
    expect(worst_fused < 1e-4, "fused matches the host reference (worst rel "
                               + std::to_string(worst_fused) + ")");
    expect(bitwise_diff == 0, "fused == twopass bitwise (" +
                              std::to_string(bitwise_diff) + " differing edges)");

    cudaFree(d_x); cudaFree(d_ws); cudaFree(d_wd); cudaFree(d_S); cudaFree(d_D);
    cudaFree(d_out); cudaFree(d_ici); cudaFree(d_iei); cudaFree(d_dst);
}

// K4 must reject the same files the CPU scorer rejects, or a stale export would
// be caught on one path and silently mis-scored on the other.
static void test_weight_loader() {
    std::printf("weight loader\n");
    const std::string path = "test_score_edges_tmp.bin";
    const int feat_dim = 8, hidden = 4;
    const float bias = 0.5f;

    auto write_file = [&](int magic) {
        FILE* f = std::fopen(path.c_str(), "wb");
        std::fwrite(&magic, 4, 1, f);
        std::fwrite(&feat_dim, 4, 1, f);
        std::fwrite(&hidden, 4, 1, f);
        std::fwrite(&bias, 4, 1, f);
        std::vector<float> w((size_t)hidden * feat_dim);
        for (size_t i = 0; i < w.size(); ++i) w[i] = (float)i;
        std::fwrite(w.data(), 4, w.size(), f);   // src_proj
        std::fwrite(w.data(), 4, w.size(), f);   // dst_proj
        std::fclose(f);
    };

    write_file(0x43534233);  // CSB3
    bool ok = false;
    try {
        BiaffineWeights w = BiaffineWeights::load(path);
        ok = (w.feat_dim == feat_dim && w.hidden == hidden && w.bias == bias &&
              w.src_proj.size() == (size_t)hidden * feat_dim &&
              w.src_proj[7] == 7.0f && w.dst_proj[7] == 7.0f);
    } catch (const std::exception&) { ok = false; }
    expect(ok, "CSB3 file round-trips");

    write_file(0x43534232);  // CSB2
    bool threw = false;
    try { BiaffineWeights::load(path); } catch (const std::exception&) { threw = true; }
    expect(threw, "CSB2 file is rejected (matches cushr_cpu/src/scorer.cpp)");

    write_file(0xdeadbeef);  // neither magic
    threw = false;
    try { BiaffineWeights::load(path); } catch (const std::exception&) { threw = true; }
    expect(threw, "a file with no recognised magic is rejected");

    std::remove(path.c_str());
}

int main() {
    std::printf("=== K4 score_edges tests ===\n");

    // The headline dims: model95_ctx_ex4200's materialized features are 192
    // wide (96 hybrid_tag + 96 char_bilstm) at hidden=128. Both are multiples
    // of 32, so this case exercises the common path with no ragged tails.
    run_case("headline dims", 4096, 3, 192, 128, 0.25f, 1234);
    // feat_dim=43 is the old lattice; hidden=100 leaves a ragged lane tail
    // (100 = 3*32 + 4, so lanes 4..31 sit out the last pass).
    run_case("ragged dims", 1000, 5, 43, 100, -1.5f, 99);
    // hidden below one warp: every lane past `hidden` must sit out the loop
    // and still contribute a zero to the butterfly reduction.
    run_case("narrow hidden", 300, 2, 16, 8, 0.0f, 7);
    // A single node has no in-edges at all -- n_slots is 0 for that row and the
    // launcher must not emit a zero-block grid.
    run_case("tiny", 1, 3, 32, 32, 3.0f, 11);

    test_weight_loader();

    std::printf(g_fail == 0 ? "=== all passed ===\n" : "=== %d FAILED ===\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
