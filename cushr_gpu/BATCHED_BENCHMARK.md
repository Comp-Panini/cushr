# cuSHR Batched (K3 + K5) Benchmark — Week 8

## Results

<!-- BEGIN AUTO-GENERATED RESULTS (make_benchmark_md.py --inject) -->

**Correctness:** SCORE-EQUIVALENT to CPU at every K (spot-checked on the first 1000 of 119503 sentences per K).

## Throughput and memory vs K

| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | table MB | used MB |
|---|------------------:|-----------------:|---------------:|---------:|--------:|
| 1 | 939934 | 1.1 | 1.1 | 68.5 | 72.0 |
| 5 | 898524 | 1.1 | 1.1 | 273.9 | 276.0 |
| 16 | 979102 | 1.0 | 1.0 | 838.9 | 840.0 |
| 32 | 954268 | 1.0 | 1.0 | 1660.7 | 1662.0 |
| 64 | 465990 | 2.1 | 2.1 | 3304.3 | 3306.0 |

## Launch count (batched sweep)

One `kbest_merge_level` launch per topo level per chunk, covering that level's nodes across *all* sentences in the chunk. Launches therefore scale with depth and chunk count, not with sentence count (119503 sentences).

| K | chunks | launches | sentences / launch |
|---|-------:|---------:|-------------------:|
| 1 | 1 | 50 | 2390 |
| 5 | 1 | 50 | 2390 |
| 16 | 1 | 50 | 2390 |
| 32 | 1 | 50 | 2390 |
| 64 | 1 | 50 | 2390 |

## Batch-size sweep (memory vs speed)

`--batch N` sizes each chunk's k-best table to that chunk's node span instead of the whole corpus, so peak device memory scales with N. Smaller chunks cost more launches (each chunk repeats the level loop) and lose cross-sentence parallelism per launch. Throughput is over all 119503 sentences in every row.

### Peak device memory (used MB)

Whole-corpus (`--batch -1`) allocates the full table; this is the column the K2 driver has no answer to.

| batch | chunks | launches | K=1 | K=5 | K=16 | K=32 | K=64 |
|---|-------:|---------:|------:|------:|------:|------:|------:|
| 256 | 467 | 10126 | 2 | 2 | 4 | 6 | 14 |
| 1024 | 117 | 3139 | 2 | 4 | 14 | 20 | 38 |
| 4096 | 30 | 1003 | 4 | 14 | 38 | 68 | 128 |
| -1 (whole corpus) | 1 | 50 | 72 | 276 | 840 | 1662 | 3306 |

### Throughput (sent/sec, kernel)

Same total work in every row — only the chunking differs.

| batch | chunks | launches | K=1 | K=5 | K=16 | K=32 | K=64 |
|---|-------:|---------:|------:|------:|------:|------:|------:|
| 256 | 467 | 10126 | 76446 | 75147 | 77500 | 78327 | 31993 |
| 1024 | 117 | 3139 | 173535 | 170815 | 184688 | 180257 | 76045 |
| 4096 | 30 | 1003 | 342212 | 345362 | 356706 | 357327 | 162726 |
| -1 (whole corpus) | 1 | 50 | 939934 | 898524 | 979102 | 954268 | 465990 |

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
