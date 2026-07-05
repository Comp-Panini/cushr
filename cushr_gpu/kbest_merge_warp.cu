// loop over incoming edges of v
// each lane reads one parent candidate, add edge score, bitonic sort

#include "gpu_lattice.cuh"
#include "gpu_kbest.cuh"

namespace cushr {

// comparator
__device__ bool better(float sa, int na, int ra, float sb, int nb, int rb) {
    if (sa != sb) return sa > sb; // first, higher score wins
    if (na != nb) return na < nb; // next, smaller parent_node wins
    return ra < rb; // finally, smaller parent_rank wins
}

// given 2 bitonic arrays with K candidates each, sort it fully ascending.
// max is a power of 2 and a multiple of 32 in order for algo to work properly
template<int max> __device__ void warp_bitonic_merge(float* s, int* pn, int* pr, int lane) {
    const int total_combined_elements = 2*max;
    const int slots_per_lane = total_combined_elements/32;
    const unsigned mask = 0xffffffffu;

    for (int i = 2*total_combined_elements; i > 0; i /= 2) {
        if (i >= 32) {
            const int slot_dist = i/32;
            for (int slot = 0; slot < slots_per_lane; slot++) {
                const int my_idx = (slot*32) | lane;
                const int partner_idx = idx ^ i;
                if (partner_idx <= my_idx) {
                    continue;
                }

                const int partner_slot = slot ^ slot_dist;

                const bool b_better = better(s[partner_slot], pn[partner_slot], pr[partner_slot], s[slot], pn[slot], pr[slot]);
                if (b_better) {
                    // swap them
                    float temp_score = s[slot]; 
                    s[slot] = s[partner_slot];
                    s[partner_slot] = temp_score;
                    float temp_parent_node = pn[slot]; 
                    pn[slot] = pn[partner_slot];
                    pn[partner_slot] = temp_parent_node;
                    float temp_rank = pr[slot]; 
                    pr[slot] = pr[partner_slot];
                    pr[partner_slot] = temp_rank;
                }
            }
        }

        else { // exchange cross lane, partner is lane^j, same slot
            for (int slot = 0; slot < slots_per_lane; slot++) {
                const int my_idx = slot*32 | lane;
                const int partner_idx = my_idx ^ i;

                const float parent_score = __shfl_xor_sync(mask, s[slot], i);
                const float parent_node_2 = __shfl_xor_sync(mask, pn[slot], i);
                const float parent_rank_2 = __shfl_xor_sync(mask, pr[slot], i);

                const bool idx_is_low = (my_idx < partner_idx);

                bool take_partner;
                if (idx_is_low) {
                    take_partner = better(parent_score, parent_node_2, parent_rank_2, s[slot], pn[slot], pr[slot]);
                }
                else {
                    take_partner = better(s[slot], pn[slot], pr[slot], parent_score, parent_node_2, parent_rank_2);
                }

                if (take_partner) {
                    s[slot] = parent_score;
                    pn[slot] = parent_node_2;
                    pr[slot] = parent_rank_2;
                }
            }

        }
        __syncwarp(mask);
    }
}

// starts off the top-K table with source node
// all other nodes are at 0 count
__global__ void init_kbest(GpuLattice lat, GpuKBest kb) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (v >= lat.num_nodes) return;

    const bool is_source;
    if(lat.in_row_ptr[v] == lat.in_row_ptr[v + 1]) {
        is_source = true;
    }
    else {
        is_source = false;
    }

    if (is_source) {
        kb.score[v*kb.K + 0] = 0.0f;
        kb.pnode[v*kb.K + 0] = -1;
        kb.prank[v*kb.K + 0] = -1;
        kb.count[v] = 1;
    } 
    
    else {
        kb.count[v] = 0;
    }
}

// relax every node who is at this topo level
// clear running top-K, load parents beam as new chunk, ad edge score, resort
// write topK entries
template<int max> __global__ void kbest_merge_level(GpuLattice lat, GpuKBest kb, const int* nodes_at_level, int n_nodes_at_level) {
    const int slots_per_lane_each_half = max / 32;    // slots per lane occupied by each half (1 or 2)
    const int slots_per_lane  = slots_per_lane_each_half * 2;    // total slots per lane in the combined array

    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_thread >> 5;
    const int lane = global_thread & 31;
    if (warp_id >= n_nodes_at_level) return;   // whole warp returns together

    const int v = nodes_at_level[warp_id];
    const int beg = lat.in_row_ptr[v];
    const int end = lat.in_row_ptr[v + 1];

    // Combined scratch: slots [0, slots_per_lane_each_half) = running top-K, [HALF, 2*HALF) = new chunk.
    float s[slots_per_lane];
    int pn[slots_per_lane];
    int pr[slots_per_lane];

    // (1) running list starts empty (all -inf sentinels). An all-equal list is
    //     trivially sorted, so the bitonic invariant holds from the first edge.
    for (int t = 0; t < slots_per_lane_each_half; t++) {
        s[t] = CUSHR_NEG_INF;
        pn[t] = -1;
        pr[t] = -1;
    }

    // (2) fold in every incoming edge with a bitonic MERGE.
    for (int er = beg; er < end; er++) {
        const int u = lat.in_col_idx[er];   // parent node
        const int eid = lat.in_edge_id[er];   // forward edge id
        const float es = lat.edge_score[eid];  // precomputed edge score
        const int cu = kb.count[u];           // number of candidates in parent's beam

        // Load parent u's (already best-first sorted) beam into the chunk half,
        // but REVERSED so that [running ascending || chunk descending] is bitonic.
        // Global index g = cslot*32 + lane in [max, 2*max); its reversed rank is
        //     r = (2*max - 1) - g   in [0, max).
        // Adding the constant es preserves the beam's order, so no chunk sort is
        // needed. Ranks >= cu are padded with -inf (worst -> sinks to the tail).
        for (int t = 0; t < slots_per_lane_each_half; ++t) {
            const int cslot = slots_per_lane_each_half + t;                 // destination slot in the chunk half
            const int g = (cslot << 5) | lane;      // global index in [max, 2*max)
            const int r = (2 * max - 1) - g;        // reversed parent rank in [0, max)
            if (r < cu) {
                s[cslot] = kb.score[u * kb.K + r] + es;  // extend the parent path
                pn[cslot] = u;                            // candidate's parent node
                pr[cslot] = r;                            // candidate's parent rank
            } 
            
            else {
                s[cslot] = CUSHR_NEG_INF;
                pn[cslot] = -1;
                pr[cslot] = -1;
            }
        }

        // Merge the bitonic 2*max array; the top max land ascending in slots
        // [0, HALF) and become the running list for the next edge. Slots
        // [HALF, 2*HALF) hold the discarded tail and are overwritten next edge.
        warp_bitonic_merge<max>(s, pn, pr, lane);
    }

    // (3) write the top K entries back. Global rank p = t*32 + lane occupies
    // running slot t (only slots [0,HALF) are the kept top-max). We only emit the
    // first K ranks and only those that are real (score > -inf).
    int my_count = 0;                            // valid slots this lane writes
    for (int t = 0; t < slots_per_lane_each_half; t++) {
        const int p = t*32 | lane;           // this element's global rank (0=best)
        if (p < kb.K && s[t] > CUSHR_NEG_INF) {
            kb.score[v*kb.K + p] = s[t];
            kb.pnode[v*kb.K + p] = pn[t];
            kb.prank[v*kb.K + p] = pr[t];
            my_count++;
        }
    }

    // count[v] = total valid entries among the first K: warp-reduce the per-lane
    // counts into lane 0.
    const unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1)
        my_count += __shfl_down_sync(mask, my_count, off);
    if (lane == 0) {
        kb.count[v] = my_count;   // total valid entries for node v
    }
}

// Explicit template instantiations so the host driver / tests can link them.
template __global__ void kbest_merge_level<32>(GpuLattice, GpuKBest, const int*, int);
template __global__ void kbest_merge_level<64>(GpuLattice, GpuKBest, const int*, int);

// pick the max that covers requested K and launch matching template
// K<=32 used cap = 32, 32 < K <= 64 uses CAP=64
// one warp per node in the level
void launch_kbest_merge(const GpuLattice& lat, const GpuKBest& kb, const int* d_nodes, int n_nodes, int threads_per_block, cudaStream_t stream) {
    
    if (n_nodes <= 0) return;

    const int warps  = n_nodes;
    const int blocks = (warps * 32 + threads_per_block - 1) / threads_per_block;

    if (kb.cap <= 32) {
        kbest_merge_level<32><<<blocks, threads_per_block, 0, stream>>>(lat, kb, d_nodes, n_nodes);
    }
    else {
        kbest_merge_level<64><<<blocks, threads_per_block, 0, stream>>>(lat, kb, d_nodes, n_nodes);
    }
}

}  