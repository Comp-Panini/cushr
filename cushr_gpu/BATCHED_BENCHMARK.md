# cuSHR Batched (K3 + K5) Benchmark — Week 8

## Results

<!-- BEGIN AUTO-GENERATED RESULTS (make_benchmark_md.py --inject) -->

**Correctness:** SCORE-EQUIVALENT to CPU at every K (checked 119503 sentences per K).

## Throughput and memory vs K

| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | table MB | used MB |
|---|------------------:|-----------------:|---------------:|---------:|--------:|
| 1 | 939934 | 1.1 | 1.1 | 68.5 | 72.0 |
| 5 | 898524 | 1.1 | 1.1 | 273.9 | 276.0 |
| 16 | 979102 | 1.0 | 1.0 | 838.9 | 840.0 |
| 32 | 954268 | 1.0 | 1.0 | 1660.7 | 1662.0 |
| 64 | 465990 | 2.1 | 2.1 | 3304.3 | 3306.0 |

## Top-K recall vs gold

Measured over the 485 checked sentences that carry a resolved gold path.

> **Recall is a correctness invariant, not a Week-8 result.** The batched decode is score-equivalent to K2, so recall is identical by construction; this is only a spot-check over the first `--check` sentences. For the full-corpus accuracy use the K2 number in `KBEST_BENCHMARK.md` (recall@32 ≈ 0.31).

| K | recall@K |
|---|---------:|
| 1 | 0.0495 |
| 5 | 0.0969 |
| 16 | 0.1588 |
| 32 | 0.2268 |
| 64 | 0.3072 |

![Top-K recall vs K](batched_recall_vs_k.png)

![Throughput vs K](batched_throughput_vs_k.png)

## Nsight Compute summary

_Profiling CSVs (`ncu_batched_K*.csv`) not found. Run `profile_batched.slurm`, copy the CSVs next to this script, and re-run `make_benchmark_md.py`._ **TODO**

<!-- END AUTO-GENERATED RESULTS -->
