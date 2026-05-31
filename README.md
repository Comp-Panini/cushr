# cushr
cuSHR: A CUDA-accelerated top-K lattice decoder for Sanskrit morphological segmentation. Built to optimize the Sanskrit Heritage Reader (SHR), it leverages warp-privatized dynamic programming, register-level shuffle primitives, and backpointer graphs to drastically reduce memory latency and compute time on complex linguistic DAGs.
