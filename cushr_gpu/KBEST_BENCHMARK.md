# cuSHR K-best (K2) Benchmark — Week 7 (CP-4)

Warp-level k-best merge kernel (`kbest_merge_level`) benchmarked over the SIGHUM dataset. Kernel-only timing uses CUDA events around each per-level merge launch (no H2D / reconstruction overhead). Correctness is checked against the Week-3 CPU `TopKDecoder` at the same K.

**Correctness:** SCORE-EQUIVALENT to CPU at every K (checked 119503 sentences per K).

## Throughput and memory vs K

| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | table MB | used MB |
|---|------------------:|-----------------:|---------------:|---------:|--------:|
| 1 | 2706 | 369.6 | 499.8 | 68.5 | 72.0 |
| 5 | 2625 | 381.0 | 510.7 | 273.9 | 276.0 |
| 16 | 2757 | 362.7 | 491.0 | 838.9 | 840.0 |
| 32 | 2716 | 368.2 | 495.5 | 1660.7 | 1662.0 |
| 64 | 1202 | 831.8 | 958.3 | 3304.3 | 3306.0 |

## Top-K recall vs gold

Measured over the 59092 sentences that carry a resolved gold path (~50% of the corpus).

| K | recall@K |
|---|---------:|
| 1 | 0.0551 |
| 5 | 0.0941 |
| 16 | 0.1427 |
| 32 | 0.1848 |
| 64 | 0.2293 |

![Top-K recall vs K](recall_vs_k.png)

![Throughput vs K](throughput_vs_k.png)

## Nsight Compute summary

| K | occupancy % | DRAM throughput % | top warp-stall reasons |
|---|------------:|------------------:|------------------------|
| 32 | 4.7 | 0.0 | wait (3.32); long_scoreboard (2.81) |
| 64 | 4.5 | 0.0 | wait (3.31); long_scoreboard (1.26) |

_Occupancy = `sm__warps_active` % of peak; DRAM % = `dram__throughput` % of peak sustained. Stall values are avg warps stalled per active cycle for that reason._
