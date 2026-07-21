# cuSHR K-best (K2) Benchmark — Week 7 (CP-4)

Warp-level k-best merge kernel (`kbest_merge_level`) benchmarked over the SIGHUM dataset. Kernel-only timing uses CUDA events around each per-level merge launch (no H2D / reconstruction overhead). Correctness is checked against the Week-3 CPU `TopKDecoder` at the same K.

**Correctness:** SCORE-EQUIVALENT to CPU at every K (checked 119503 sentences per K).

## Throughput and memory vs K

| K | sent/sec (kernel) | µs/sent (kernel) | µs/sent (loop) | table MB | used MB |
|---|------------------:|-----------------:|---------------:|---------:|--------:|
| 1 | 2730 | 366.3 | 496.4 | 68.5 | 72.0 |
| 5 | 2698 | 370.7 | 499.9 | 273.9 | 276.0 |
| 16 | 2730 | 366.3 | 495.7 | 838.9 | 840.0 |
| 32 | 2775 | 360.4 | 489.3 | 1660.7 | 1662.0 |
| 64 | 1210 | 826.6 | 954.3 | 3304.3 | 3306.0 |

## Top-K recall vs gold

Measured over the 59092 sentences that carry a resolved gold path (~50% of the corpus).

| K | recall@K |
|---|---------:|
| 1 | 0.0831 |
| 5 | 0.1663 |
| 16 | 0.2529 |
| 32 | 0.3104 |
| 64 | 0.3679 |

![Top-K recall vs K](recall_vs_k.png)

![Throughput vs K](throughput_vs_k.png)

## Nsight Compute summary

| K | occupancy % | DRAM throughput % | top warp-stall reasons |
|---|------------:|------------------:|------------------------|
| 32 | 4.7 | 0.0 | wait (3.32); long_scoreboard (2.81) |
| 64 | 4.5 | 0.0 | wait (3.31); long_scoreboard (1.26) |

_Occupancy = `sm__warps_active` % of peak; DRAM % = `dram__throughput` % of peak sustained. Stall values are avg warps stalled per active cycle for that reason._

> **Note:** These are the frozen Week-7 per-sentence (K2) numbers. The Week-8 batched (K3+K5) results are kept **separately** in `BATCHED_BENCHMARK.md` so the two can be compared side by side — this file is never overwritten by the batched benchmark.

## Register / spill / occupancy tuning (`kbest_merge_level`)

The merge kernel is **latency-bound** (DRAM ~0%, dominant stalls are `long_scoreboard`
and `wait`), so resident-warp count — i.e. occupancy — is the primary lever for hiding
latency. The candidate arrays `s[]`/`pn[]`/`pr[]` are per-lane register arrays whose size
scales with `slots_per_lane = 2*max/32` (K=32 → 2 slots, K=64 → 4 slots). This makes the
register/spill/occupancy trade behave very differently for the two instantiations.

Measured on A100 / sm_80, block = 256 threads (8 warps), `nvcc -Xptxas -v`:

| variant | K=32 regs / spill | K=64 regs / spill | K=64 theoretical occ |
|---|---|---|---|
| baseline (branchy, no unroll) | 41 / 48 B spill | **30 / 48 B spill** | **100%** |
| branchless compare-exchange | 39 / 0 | 61 / 0 | 50% |
| `#pragma unroll` both loops | 38 / 0 | 79 / 0 | 37.5% |
| **final (per-K unroll, below)** | **38 / 0** | **30 / 48 B spill** | **100%** |

**Findings:**

- **K=32 — unroll + bitwise comparator is a clean win.** Full-unrolling the bitonic
  loops makes every array index a compile-time constant, so the candidate arrays stay in
  registers: **38 regs, 0 spills** (down from 41 regs + a 48 B spill). The bitwise
  comparator (`better()` with `|`/`&` instead of short-circuit `||`/`&&`) collapses the
  merge-body branches (PTX `bra` 148 → ~35) and is register-neutral. Kept.

- **K=64 — leave it at the baseline.** Every attempt to remove the 48 B spill made it
  worse: eliminating the spill forces the 4-slot working set entirely into registers
  (branchless → 61 regs / 50% occ; unroll → 79 regs / 37.5% occ). For a latency-bound
  kernel that is a bad trade — the 48 B spill is small and L1-cached, and the baseline's
  100% occupancy already hides that latency. **The original 30 regs / 48 B spill / 100%
  occupancy is the best state for K=64.**

- The Nsight data shows **K=32 is the faster, better-occupied kernel**; K=64's larger
  per-lane working set is a structural limit, not something micro-optimization fixes.

**Implementation.** The unroll is gated on the template size so K=32 gets full unrolling
while K=64 stays non-unrolled (branchy, 30 regs):

```cuda
#pragma unroll (max <= 32 ? 32 : 1)   // full unroll for K<=32, factor 1 = no unroll for K=64
```

applied to all three bitonic-merge loops. The bitwise `better()` comparator is kept for
both instantiations (register-neutral, fewer branches).
