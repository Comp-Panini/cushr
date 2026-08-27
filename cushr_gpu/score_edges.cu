
#include "score_edges.cuh"

#include <cstdint>    
#include <cstdio>
#include <fstream>    
#include <stdexcept>  

namespace cushr {

// weights

namespace {
// must match cushr_cpu/src/scorer.cpp
// stale CSB2 export fails identically whichever scorer opens it
constexpr int kBiaffineMagic   = 0x43534233;  // CSB3
constexpr int kBiaffineMagicV2 = 0x43534232;  // CSB2, no longer supported
}

// opens file written by export_weights.py
// containing trained neural network's parameters including weights/biases
BiaffineWeights BiaffineWeights::load(const std::string& bin_path) {

    // error handling
    std::ifstream in(bin_path, std::ios::binary);
    if (!in) throw std::runtime_error("BiaffineWeights: cannot open " + bin_path);
    int32_t magic = 0, feat_dim = 0, hidden = 0;
    float bias = 0.0f;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&feat_dim), 4);
    in.read(reinterpret_cast<char*>(&hidden), 4);
    in.read(reinterpret_cast<char*>(&bias), 4);
    if (!in) throw std::runtime_error("BiaffineWeights: cannot read header of " + bin_path);
    if (magic == kBiaffineMagicV2)
        throw std::runtime_error(
            "BiaffineWeights: " + bin_path + " is a CSB2 file; re-export with "
            "cushr_train/export_weights.py from a featurized model");
    if (magic != kBiaffineMagic)
        throw std::runtime_error("BiaffineWeights: bad magic in " + bin_path +
                                 " (not an export_weights.py --bin file?)");
    if (feat_dim <= 0 || hidden <= 0)
        throw std::runtime_error("BiaffineWeights: nonsensical dims in " + bin_path);



    BiaffineWeights w;

    // width of raw feature vector for a single node
    w.feat_dim = feat_dim; // 192 = 96 for hybrid_tag + 96 for char_bilstm (concat)

    // length of the projected vector that biaffine scorere actually uses to do dot product
    // main "bottleneck"
    // W_s and W_d are two 128 x 192 matrices
    // these are the globally shared parameters / weights 
    // they project a node's 192 length feature vector down to length of 128
    w.hidden = hidden; 

    w.bias = bias;

    const size_t n = (size_t)hidden * (size_t)feat_dim;
    // resize() allocates heap memory for 24576 floats before calling the in.read()
    // the raw bytes are put into src_proj and dest_proj
    w.src_proj.resize(n); 
    w.dst_proj.resize(n);

    // each matrix is a single read directly into the vector's storage as row-major format

    // data() returns a pointer to vector's internal buffer
    // read() puts bytes into a location in heap
    in.read(reinterpret_cast<char*>(w.src_proj.data()), n * sizeof(float));
    in.read(reinterpret_cast<char*>(w.dst_proj.data()), n * sizeof(float));
    if (!in) throw std::runtime_error("BiaffineWeights: truncated weight file " + bin_path);
    return w;  
}

// Row-major [hidden][feat_dim] -> transposed [feat_dim][hidden].
//
// The CSB3 file on disk stays row-major -- that is what cushr_cpu's
// BiaffineScorer reads, and model95_ctx.bin is already exported that way. Only
// the GPU copy is transposed, because only the GPU cares: see the coalescing
// note on row_dot above. Done once per run on the host, over hidden*feat_dim =
// 24,576 floats at the headline dims, so the cost is nil against a 3.21 GiB
// node-feature upload.
std::vector<float> transpose_proj(const std::vector<float>& m, int hidden, int feat_dim) {
    if (m.size() != (size_t)hidden * (size_t)feat_dim)
        throw std::runtime_error("transpose_proj: size does not match hidden*feat_dim");
    std::vector<float> t((size_t)feat_dim * (size_t)hidden);
    for (int h = 0; h < hidden; ++h)
        for (int i = 0; i < feat_dim; ++i)
            t[(size_t)i * hidden + h] = m[(size_t)h * feat_dim + i];
    return t;
}

// kernels

namespace {

// sum across a warp with lane 0 holding total
// step of 16,8,4,2,1 (how nany lanes above me should I fetch from)
// useful lanes = 0-15, then 0-7, then... 0-1, then 0
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        // the participating mask is 0xffffffff
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// dot(W[row], x) with x already in shared memory, W stored TRANSPOSED.
//
// WT is [feat_dim][hidden], NOT [hidden][feat_dim]. That transpose is the whole
// point of this function's shape and it was put there to fix a measured
// bottleneck, so the reasoning is worth spelling out.
//
// Callers assign lane l the hidden dims l, l+32, l+64, ... So at any instant the
// 32 lanes of a warp are evaluating 32 DIFFERENT rows of W at the same column i.
//
//   Row-major W (the original code, and still the on-disk CSB3 layout):
//       lane h reads W[h*feat_dim + i]
//   Consecutive lanes are then feat_dim*4 = 768 B apart. One warp load touches
//   32 separate 32-byte sectors -- fully uncoalesced. Nsight Compute measured
//   the consequence on an A100: project_nodes ran at 99.24% of peak L1/TEX
//   throughput with compute at 5.98% and DRAM at 0.14%, stalling ~211 cycles
//   per warp on MIO throttle out of ~473 between issues. The data was cached
//   (98.3% L1 hit) -- the wall was the REQUEST rate, not the bytes.
//
//   Transposed WT (what this reads):
//       lane h reads WT[i*hidden + h]
//   Consecutive lanes are now 4 B apart, so a warp's 32 loads fall in 128
//   contiguous bytes = 4 sectors instead of 32. Same arithmetic, same operand
//   values, 8x fewer sector requests per instruction.
//
// The transpose happens once on the host at upload (transpose_proj below), so
// the CSB3 file and the CPU BiaffineScorer are untouched.
//
// Bitwise note: i still runs 0..feat_dim in order and the summation order is
// unchanged, so this produces bit-identical results to the row-major version.
// The fused-vs-twopass bitwise equality asserted in tests/test_score_edges.cu
// therefore survives, as does agreement with the pre-transpose scores.
__device__ __forceinline__ float row_dot(const float* __restrict__ WT, int row,
                                         const float* __restrict__ x,
                                         int feat_dim, int hidden) {

    const float* w = WT + row;        // column `row`, walked with stride `hidden`
    float acc = 0.0f;
    for (int i = 0; i < feat_dim; i++) {
        acc += w[(size_t)i * hidden] * x[i];
    }
    return acc;
}

}  // namespace

// K4a: node calculator. takes the 192 feature representation and compresses it into two smaller 128-feature vectors
// one of them is the source and one is the destination
// one warp per node, lane l owns hidden dims l, l+32, ...
// each warp of 32 threads processes 1 node
// thread 0 owns the hidden dimensions 0, 32, 64, 96
// the warp loads the node's 192-length feature vector x into fast shared memory
// the threads split up the wordload calculating the 128 rows of the output vector
// write the resulting 128 length vector to global memory d_s and d_d
__global__ void project_nodes(GpuBiaffine bf, int tile_begin, int tile_end) {
    // dynamically sized shared memory
    extern __shared__ float smem[];
    // threadIdx.x calculation
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_nodes = tile_end - tile_begin; // n_nodes = total number of nodes that GPU must process in this batch

    // gwarp = warp's index across the grid
    // blockDim.x >> 5 is warps-per-block
    const int gwarp = (blockIdx.x * (blockDim.x >> 5)) + warp;
    if (gwarp >= n_nodes) return;

    // v = global node ID, add the starting point of the batch to the warp num
    const int v = tile_begin + gwarp;

    // x = memory address pointing to fast shared memory
    float* x = smem + (size_t)warp * bf.feat_dim;

    // src = memory address pointing to slow main memory where the 192-length feature vector is stored
    const float* src = bf.d_node_feat + (size_t)v * bf.feat_dim;
    for (int i = lane; i < bf.feat_dim; i += 32) {
        x[i] = src[i];
    }
    __syncwarp();

    // output rows for this node
    float* S = bf.d_S + (size_t)v * bf.hidden;
    float* D = bf.d_D + (size_t)v * bf.hidden;

    // lane-strided again across the hidden output dims: lane 0 computes dims 0, 32, 64, 96, etc.
    for (int h = lane; h < bf.hidden; h += 32) {
        S[h] = row_dot(bf.d_src_projT, h, x, bf.feat_dim, bf.hidden);
        D[h] = row_dot(bf.d_dst_projT, h, x, bf.feat_dim, bf.hidden);
    }
}

// K4b: edge scorer, scoring edges between words, relies on K4a
// looks up the source vector and dest vector for the start and end nodes, multiply, add bias, output final edge score
// one warp per reverse-CSR slot
// tile_dst is an array containing IDs of destination nodes
// slot0 tells kernel where to start readinf from arrays if batched
// n_slots = total num edges this kernel will process
//edge_scores = blank output array in global memory where final scores will be written
__global__ void score_edges_twopass(GpuBiaffine bf, const int* __restrict__ in_col_idx, const int* __restrict__ in_edge_id, const int* __restrict__ tile_dst,
                                    int slot0, int n_slots, float* __restrict__ edge_score) {
    // unit of work is now an edge
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gwarp = (blockIdx.x * (blockDim.x >> 5)) + warp;
    if (gwarp >= n_slots) return;

    // v = ID of destination node
    const int v = tile_dst[gwarp];

    // u = ID of source node
    const int u = in_col_idx[slot0 + gwarp];

    // global ID of the edge
    const int e = in_edge_id[slot0 + gwarp];


    // S = memory pointer to start of 128-len src vector for u
    // D = same but dest vector for v
    const float* S = bf.d_S + (size_t)u * bf.hidden;
    const float* D = bf.d_D + (size_t)v * bf.hidden;
    float acc = 0.0f;

    // lane 0 handles dims 0,32,64,96
    // multiply the assigned elements from src and dest vectors and add result to acc
    for (int h = lane; h < bf.hidden; h += 32) {
        acc += S[h] * D[h];
    }

    // all threads add up their individual products, at the end lane 0 holds the total of 128 multiplications
    acc = warp_sum(acc);

    // adds the bias (globally shared num), write final score to output array at position e
    if (lane == 0) {
        edge_score[e] = acc + bf.bias;
    }
}

// K4, does both steps at once
// for every edge, grabs the 192 feature data for both endpoints, does all math and writes final score
// lot of extra math
// skips intermediate steps, needs 192-len feature vectors for both endpoints
__global__ void score_edges_fused(GpuBiaffine bf, const int* __restrict__ in_col_idx, const int* __restrict__ in_edge_id,
                                  const int* __restrict__ tile_dst, int slot0, int n_slots, float* __restrict__ edge_score) {
    
    // shared memory fast
    extern __shared__ float smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int gwarp = (blockIdx.x * (blockDim.x >> 5)) + warp;
    if (gwarp >= n_slots) return;

    // edge decoding
    const int v = tile_dst[gwarp];
    const int u = in_col_idx[slot0 + gwarp];
    const int e = in_edge_id[slot0 + gwarp];

    // needs both endpoints' raw features 192-len
    // xu points to start of this warp's portion of smem
    // xv points to second half of that slice
    float* xu = smem + (size_t)warp * 2 * bf.feat_dim;
    float* xv = xu + bf.feat_dim;

    // fu and fv are pointers to actual feature data in main memory
    const float* fu = bf.d_node_feat + (size_t)u * bf.feat_dim;
    const float* fv = bf.d_node_feat + (size_t)v * bf.feat_dim;

    // 32 threads work together to copy 192 floats from fu to xu and fv to xv
    // coalasced reads for contiguous mem chunks
    for (int i = lane; i < bf.feat_dim; i += 32) { 
        xu[i] = fu[i]; xv[i] = fv[i]; 
    }
    __syncwarp();

    float acc = 0.0f;
    for (int h = lane; h < bf.hidden; h += 32) {
        // call row_dot helper taking one row of global 128x192 src matrix
        // dot product it with the 192-len feature vector in shared memory
        // produces a single scalar

        // this math is repeated ~17.6 times (as every node is connected to ~17.6 edges)
        // tradeoff: more math, but less memory bandwidth
        const float s = row_dot(bf.d_src_projT, h, xu, bf.feat_dim, bf.hidden);
        const float d = row_dot(bf.d_dst_projT, h, xv, bf.feat_dim, bf.hidden);

        // multiply the scalars for the src and dest nodes and add them together
        acc += s * d;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        edge_score[e] = acc + bf.bias;
    }
}

// launcher


void launch_score_edges(const GpuBiaffine& bf, K4Mode mode, const int* in_col_idx, const int* in_edge_id, const int* tile_dst, int slot0, int n_slots,
                        int tile_begin, int tile_end, float* edge_score, cudaStream_t stream) {

    if (mode == K4Mode::Host) return;   

    const int tpb = 256; // 8 warps/block
    const int warps = tpb >> 5;

    if (mode == K4Mode::Fused) {
        if (n_slots <= 0) return;
        const int blocks = (n_slots + warps - 1) / warps;
        const size_t sh  = (size_t)warps * 2 * bf.feat_dim * sizeof(float);
        score_edges_fused<<<blocks, tpb, sh, stream>>>(
            bf, in_col_idx, in_edge_id, tile_dst, slot0, n_slots, edge_score);
        return; 
    }

    // two-pass: project every node in the tile, then score every edge
    const int n_nodes = tile_end - tile_begin;
    if (n_nodes > 0) {
        const int blocks = (n_nodes + warps - 1) / warps;
        // Half the fused kernel's shared memory -- one feat_dim slice per warp,
        // 6 KiB at the headline dims, because only one node is staged.
        const size_t sh  = (size_t)warps * bf.feat_dim * sizeof(float);
        project_nodes<<<blocks, tpb, sh, stream>>>(bf, tile_begin, tile_end);
    }
    if (n_slots > 0) {
        const int blocks = (n_slots + warps - 1) / warps;
        score_edges_twopass<<<blocks, tpb, 0, stream>>>(
            bf, in_col_idx, in_edge_id, tile_dst, slot0, n_slots, edge_score);
    }
}

}  // namespace cushr
