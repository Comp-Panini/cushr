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
// cap is a power of 2 and a multiple of 32 in order for algo to work properly
template<int cap> __device__ void warp_bitonic_merge(float* s, int* pn, int* pr, int lane) {
    const int total_combined_elements = 2*cap;
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