#pragma once

#include <cstdint>

namespace cushr {

#ifndef CUSHR_NEG_INF
#define CUSHR_NEG_INF (-3.4028235e38f)   
#endif

struct GpuKBest {
    int K; // 1,4,8,16,..64
    int cap; // working capacity per node, 32 for K<=32, else 64.
             // affects number of registers used, decoder still keeps K candidates
    float* score; // total path score of node v's r'th best path: score[v*K + r]
    int* pnode; // parent node of that path: pnode[v*K + r]
    int* prank; // parent's rank, which path does the parent extend: prank[v*K + r]
    int* count; // count of number of valid entries in [v*K, v*K+K) is count[v]

}; 

}