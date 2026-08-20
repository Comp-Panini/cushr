// cushr_gpu/score_edges.cuh
//
// K4: the trained biaffine edge scorer, running on the device.
//
// Until week 10 the GPU never scored an edge: host_driver_batched.cu filled a
// flat float edge_score[num_edges] with EdgeScorer::score() and uploaded it,
// so K3 only ever read a number the CPU had already computed. K4 computes that
// number on the device instead, from the same 'CSB3' weight file the CPU
// BiaffineScorer loads (cushr_train/export_weights.py --bin):
//
//     score(u -> v) = < W_s x_u , W_d x_v > + b
//
// x_v is node_features[v] verbatim -- feat_dim columns, whatever the featurizer
// in cushr_train/featurizers.py emitted. A contextual model must be frozen with
// train.py --materialize first; a char-BiLSTM cannot run inside this kernel.
//
// Two implementations, same arithmetic:
//
//   K4a project_nodes + K4b score_edges_twopass  (default)
//       Projects every node once into S = W_s x and D = W_d x, then one warp
//       per edge does a length-`hidden` dot product. A node is the source of
//       17.6 edges on average (78,847,461 edges over 4,488,155 nodes), so the
//       projection -- which is all of the FLOPs -- happens 17.6x less often
//       than in the fused kernel.
//
//   K4 score_edges_fused                          (--k4 fused)
//       The literal one-warp-per-edge form: each warp loads node_feat[src] and
//       node_feat[dst] and multiplies them against W_src/W_dst inline, with no
//       reuse between the edges that share an endpoint. Kept so the cost of
//       that redundancy is a measured number rather than an assertion.
//
// Both accumulate in the same order (lane l owns hidden dims l, l+32, ... and
// the warp finishes with a shfl_down butterfly), so they agree BITWISE with
// each other. Neither agrees bitwise with the CPU scorer, which sums h in
// order 0..hidden -- compare those with a tolerance.
//
// Tensor cores are deliberately not used, on the expectation that these kernels
// are bound by streaming node features and edge scores rather than by math.
// Note that the usual supporting argument -- "the projection matrices are tiny
// and live in L1" -- does NOT hold at the headline dims: feat_dim=192,
// hidden=128 puts W_s and W_d at 96 KiB each, 192 KiB together, which is the
// A100's entire per-SM combined L1/shared budget. The expectation is therefore
// untested; K4_BENCHMARK.md records what the profiler actually says.

#pragma once

#include <string>
#include <vector>

namespace cushr {

// Device-side biaffine weights + the per-tile projection scratch.
//
// d_src_proj / d_dst_proj are [hidden * feat_dim] row-major, uploaded once.
// d_S / d_D are [tile_span * hidden] and are re-filled per tile; like the
// k-best table in host_driver_batched.cu they are addressed with the pointer
// biased by the tile base, so kernels stay global-node-indexed:
//     d_S[(size_t)v * hidden + h]   for v in [tile_begin, tile_end)
struct GpuBiaffine {
    int   feat_dim = 0;   // 192 for model95_ctx_ex4200: 96 hybrid_tag + 96 ctx
    int   hidden   = 0;   // 128
    float bias     = 0.0f;

    const float* d_node_feat = nullptr;  // [num_nodes * feat_dim], global
    const float* d_src_proj  = nullptr;  // [hidden * feat_dim]
    const float* d_dst_proj  = nullptr;  // [hidden * feat_dim]

    float* d_S = nullptr;   // [.. * hidden], bias-adjusted (see above)
    float* d_D = nullptr;
};

// Host-side weights read from a 'CSB3' file. Parsed by the CPU
// BiaffineScorer::load path in cushr_cpu/src/scorer.cpp; this mirrors the same
// header so the GPU driver can upload without constructing a CPU scorer's
// projection cache. Throws std::runtime_error on a bad/CSB2/truncated file.
struct BiaffineWeights {
    int   feat_dim = 0;
    int   hidden   = 0;
    float bias     = 0.0f;
    std::vector<float> src_proj;  // [hidden * feat_dim]
    std::vector<float> dst_proj;

    static BiaffineWeights load(const std::string& bin_path);
};

// K4a. One warp per node over [tile_begin, tile_end). Fills d_S and d_D.
__global__ void project_nodes(GpuBiaffine bf, int tile_begin, int tile_end);

// K4b. One warp per entry of the tile's reverse-CSR slot range. For slot i:
//   v = tile_dst[i]           (dst node, host-built: the row that owns slot i)
//   u = in_col_idx[slot0 + i] (src node)
//   e = in_edge_id[slot0 + i] (forward edge id -- what K3 indexes)
// Writes edge_score[e]. Requires project_nodes to have run over the tile.
__global__ void score_edges_twopass(GpuBiaffine bf,
                                    const int* in_col_idx, const int* in_edge_id,
                                    const int* tile_dst, int slot0, int n_slots,
                                    float* edge_score);

// K4, fused. Same slot iteration, but projects both endpoints inline and
// touches neither d_S nor d_D.
__global__ void score_edges_fused(GpuBiaffine bf,
                                  const int* in_col_idx, const int* in_edge_id,
                                  const int* tile_dst, int slot0, int n_slots,
                                  float* edge_score);

enum class K4Mode { Host, TwoPass, Fused };

// Score one tile's edges into edge_score (a device pointer into the global
// [num_edges] array). `stream` is honoured; no synchronisation is performed.
void launch_score_edges(const GpuBiaffine& bf, K4Mode mode,
                        const int* in_col_idx, const int* in_edge_id,
                        const int* tile_dst, int slot0, int n_slots,
                        int tile_begin, int tile_end,
                        float* edge_score, cudaStream_t stream);

}  // namespace cushr
