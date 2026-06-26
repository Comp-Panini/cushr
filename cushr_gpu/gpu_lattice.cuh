#pragma once

#include <cstdint>

namespace cushr {

#ifndef CUSHR_NEG_INF
#define CUSHR_NEG_INF (-3.4028235e38f)  
#endif

struct GpuLattice {
    // graph shape 
    int num_nodes; // total nodes across all sentences in this upload
    int num_edges; // total forward edges
    int max_level; // max topo_level over all nodes (host computed once)


    // reverse CSR, same logic as in CPU
    const int* in_row_ptr; // num_nodes + 1
    const int* in_col_idx; // num_edges 
    const int* in_edge_id; // num_edges


    // per-edge
    const float* edge_score; // num_edges, precomputed on host by EdgeScorer
    const int* topo_level; // num_nodes, topological rank

    // outputs written by kernel
    // these r the K=1 backpointer table
    float* best_score;  // num_nodes, arr stores score of the best path from a source to v
    int* best_parent;  // num_nodes, arr stores the parent node u on that best path 
};

}  
