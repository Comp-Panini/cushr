# cuSHR Batched (K3 + K5) Benchmark — Week 8

Companion to `KBEST_BENCHMARK.md` (the frozen Week-7 per-sentence K2 numbers). This file holds the **batched** results and is generated/updated independently, so the two versions can be compared without overwriting each other.

The Week-7 numbers come from the per-sentence K2 driver (`host_driver_kbest.cu`): one CPU-side level loop and its own kernel launches **per sentence**, so launch overhead dominates (≈2,775 sent/sec at K=32). Week 8 removes that overhead with the batched driver (`host_driver_batched.cu`, target `make cushr_batched`):

- **K3 — global batched topo sweep.** The whole graph is uploaded once. Sentences are contiguous in node id, and no edge crosses a sentence boundary, so a single global topo sweep relaxes **all sentences at once**: one `kbest_merge_level` launch per topo level covers a coalesced slice of nodes drawn from every sentence at that level. Total launches drop from `O(sum of per-sentence depths)` to `O(max_level)`.
- **Memory-bounded batching (`--batch N`).** The corpus is processed in chunks of `N` sentences. Each chunk allocates a `GpuKBest` table sized to just that chunk's contiguous node span `[nb0, ne1)` — **not** the whole corpus. Device pointers are offset by the chunk base (`kb.score = base - nb0*K`, etc.) so the unchanged global-node-indexed kernels address into the chunk buffer. Peak device memory therefore scales with `--batch`, letting the full corpus run within a fixed footprint. `--batch -1` = whole corpus in one chunk (max batch, headline throughput).
- **K5 — top-K path reconstruction.** After each chunk's sweep, the `(pnode, prank)` back-pointer chains are walked from each sentence sink (`reconstruct_gpu_path` / `strip_boundaries`, shared with the K2 driver) to materialize the top-K paths, which feed the top-K-recall-vs-gold metric and the CPU score-equivalence check.

Source seeding uses a new additive kernel `init_kbest_seed` (only touches a chunk's source nodes, so it is safe with the offset chunk table); `kbest_merge_level` is unchanged, so the Week-6/7 correctness contract and unit tests (`make test-kbest`) still hold.

## Reproduce

```bash
make cushr_batched
sbatch bench_batched.slurm      # throughput/memory/recall -> batched_bench*.csv
sbatch profile_batched.slurm    # Nsight Compute counters   -> ncu_batched_K*.csv

# Inject the generated tables + plots + Nsight section INTO this file, between the
# AUTO-GENERATED markers in the Results section below. The rest of this narrative
# is preserved, and no Week-7 artifact is touched.
python make_benchmark_md.py --bench batched_bench.csv \
    --title "cuSHR Batched (K3 + K5) Results — Week 8" \
    --md BATCHED_BENCHMARK.md --inject \
    --recall-png batched_recall_vs_k.png \
    --thru-png   batched_throughput_vs_k.png \
    --ncu-prefix ncu_batched \
    --recall-note "> **Recall is a correctness invariant, not a Week-8 result.** The batched decode is score-equivalent to K2, so recall is identical by construction; this is only a spot-check over the first \`--check\` sentences. For the full-corpus accuracy use the K2 number in \`KBEST_BENCHMARK.md\` (recall@32 ≈ 0.31)."
```

> On **PowerShell** put the whole command on one line (backtick `` ` `` for continuations, not `\`).

### Measuring the K2 → K3/K5 improvement

Run the K2 job (`bench_kbest.slurm` → `kbest_bench.csv`) and the batched job (`bench_batched.slurm` → `batched_bench*.csv`), then:

```bash
python compare_k2_k3.py     # -> COMPARISON.md + comparison_throughput.png + comparison_memory.png
```

This joins both CSVs on K and reports the improvement two ways:

- **Wall-clock throughput** (`us_per_sent_loop`) — the headline. K2 pays per-sentence launch/sync/memcpy overhead; K3 amortizes it across the batch. This is where the big speedup lives.
- **Kernel-only throughput** (`us_per_sent_kernel`) — also improves, for a *different* reason: K2 launches one tiny grid per sentence (~4.7% A100 occupancy in the Week-7 profile), while K3's cross-sentence launches saturate the GPU.
- **Memory** — K2 always allocates the full-corpus table; K3's `--batch N` sizes it to one chunk, so `gpu_used_MB` scales down with batch (the `batched_bench_b*.csv` sweep), plotted against K2's fixed baseline.

> Note: do **not** compare kernel-only numbers alone and conclude batching did nothing — the launch-overhead win is invisible there by construction. Lead with wall clock.

Every batched output uses a distinct filename from the Week-7 ones, so no previous numbers are lost:

| purpose | Week-7 (frozen) | Week-8 batched |
|---|---|---|
| raw CSV | `kbest_bench.csv` | `batched_bench.csv`, `batched_bench_b<N>.csv` |
| tables (auto) | `KBEST_BENCHMARK.md` (whole file) | injected into `BATCHED_BENCHMARK.md` (this file) |
| recall plot | `recall_vs_k.png` | `batched_recall_vs_k.png` |
| throughput plot | `throughput_vs_k.png` | `batched_throughput_vs_k.png` |
| Nsight CSV / report | `ncu_kbest_K*.csv`, `*.ncu-rep` | `ncu_batched_K*.csv`, `*.ncu-rep` |
| hand-written narrative | — | `BATCHED_BENCHMARK.md` (this file) |

**Correctness:** the batched sweep is checked `SCORE-EQUIVALENT` against the Week-3 CPU `TopKDecoder` over the first `--check` sentences at every K; `--batch` chunking must not change results (a chunked run and `--batch -1` agree sentence-for-sentence).

## Results

The tables, plots, and Nsight summary below are filled in automatically by the
`make_benchmark_md.py --inject` command in the Reproduce section. Everything
between the two markers is regenerated on each run; text outside them (this whole
narrative) is preserved. Until the first A100 run, the block is empty.

For the K2 → K3/K5 improvement (speedup + memory-vs-batch), run `compare_k2_k3.py`
→ `COMPARISON.md`.

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
