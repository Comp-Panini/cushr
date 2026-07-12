# cuSHR K-best (K2) Benchmark — Week 7 (CP-4)

Warp-level k-best merge kernel (`kbest_merge_level`) benchmarked over the SIGHUM dataset. Kernel-only timing uses CUDA events around each per-level merge launch (no H2D / reconstruction overhead). Correctness is checked against the Week-3 CPU `TopKDecoder` at the same K.

**Correctness:** SCORE-EQUIVALENT to CPU at every K (checked 119503 sentences per K).

## Throughput and memory vs K

| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | table MB | used MB |
|---|------------------:|-----------------:|---------------:|---------:|--------:|
| 1 | 2717 | 368.0 | 497.4 | 68.5 | 72.0 |
| 5 | 2631 | 380.1 | 508.4 | 273.9 | 276.0 |
| 16 | 2763 | 361.9 | 488.6 | 838.9 | 840.0 |
| 32 | 2726 | 366.9 | 492.9 | 1660.7 | 1662.0 |
| 64 | 1204 | 830.3 | 956.0 | 3304.3 | 3306.0 |

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

_Profiling CSVs (`ncu_kbest_K*.csv`) not found. Run `profile_kbest.slurm`, copy the CSVs next to this script, and re-run `make_benchmark_md.py`._ **TODO**
