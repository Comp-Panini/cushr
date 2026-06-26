// produce best_score / best_parent that are bit-identical to the CPU
// this is the device side


#include "gpu_lattice.cuh"

namespace cushr {

// set the initial best_score/best_parent for every node
// best score is 0, best parent is -1
// all other nodes are at -inf

// detect a source from the reverse CSR

__global__ void init_best(GpuLattice lat) {
    int v = blockIdx.x * blockDim.x + threadIdx.x; // one thread per node
    if (v >= lat.num_nodes) {
        return;
    }

    const bool is_source = (lat.in_row_ptr[v] == lat.in_row_ptr[v + 1]);
    // in_row_ptr[v] == in_row_ptr[v+1] means zero incoming edges
    if (is_source) {
        lat.best_score[v] = 0.0f;
    }
    else {
        lat.best_score[v] = -3.4028235e38f; // this is -(float max)
    }
    lat.best_parent[v] = -1;  // sources have no parent
}


// candidate comparison, identical to EntryGreater in cpu.
__device__ bool better(float a_score, int a_parent, float b_score, int b_parent) {
    if (a_score != b_score) return a_score > b_score;   // higher score
    return a_parent < b_parent; // smaller parent wins ties
}

// called once per topo level
// relax every node whose topo_level == `level`.
__global__ void viterbi_relax_level(GpuLattice lat, const int* nodes_at_level, int n_nodes_at_level) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread / 32; // each warp responsible for one node on level
    const int lane = global_thread % 32; // each thread is responsible for a subset of incoming nodes

    if (warp_id >= n_nodes_at_level) return;  // there is no node at this warp
    // we usually return on a per-warp (per-node) basis in case shuffling being done

    const int v = nodes_at_level[warp_id]; // num nodes at this level
    const int beg = lat.in_row_ptr[v]; // first incoming edge index
    const int end = lat.in_row_ptr[v + 1]; // last incoming edge index + 1

    float my_score = -3.4028235e38f;
    int my_parent = -1;

    // strided scan across incoming edges to the various nodes in this level
    // lane handles slots beg+lane, beg+lane+32, beg+lane+64...
    for (int er = beg + lane; er < end; er += 32) {
        const int u = lat.in_col_idx[er]; // source node of this edge
        const int eid = lat.in_edge_id[er];  // forward edge id
        const float es = lat.edge_score[eid];  // precomputed edge score

        // candidate path score through parent u
        const float cand = lat.best_score[u] + es;

        if (better(cand, u, my_score, my_parent)) {
            my_score  = cand;
            my_parent = u;
        }
    }

    // combine the 32 best per lane into lane 0
    // use __shfl_down_sync to get highest score in lane 0
    const unsigned FULL = 0xffffffffu; // bitmask for participating lanes
    for (int offset = 16; offset > 0; offset >>= 1) {

        // move score and parent together to keep values together
        const float o_score = __shfl_down_sync(FULL, my_score, offset);
        const int o_parent = __shfl_down_sync(FULL, my_parent, offset);
        if (better(o_score, o_parent, my_score, my_parent)) {
            my_score  = o_score;
            my_parent = o_parent;
        }
    }

    if (lane == 0) {
        lat.best_score[v]  = my_score;
        lat.best_parent[v] = my_parent;
    }
}

} 
