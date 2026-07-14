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

# Generate the auto tables + batched plots + Nsight section WITHOUT touching any
# Week-7 artifact. Note the distinct --md target (BATCHED_RESULTS.md), so this
# narrative file (BATCHED_BENCHMARK.md) is NOT overwritten either:
python make_benchmark_md.py --bench batched_bench.csv \
    --title "cuSHR Batched (K3 + K5) Results — Week 8" \
    --md BATCHED_RESULTS.md \
    --recall-png batched_recall_vs_k.png \
    --thru-png   batched_throughput_vs_k.png \
    --ncu-prefix ncu_batched
```

Every batched output uses a distinct filename from the Week-7 ones, so no previous numbers are lost:

| purpose | Week-7 (frozen) | Week-8 batched |
|---|---|---|
| raw CSV | `kbest_bench.csv` | `batched_bench.csv`, `batched_bench_b<N>.csv` |
| auto-generated tables | `KBEST_BENCHMARK.md` | `BATCHED_RESULTS.md` |
| recall plot | `recall_vs_k.png` | `batched_recall_vs_k.png` |
| throughput plot | `throughput_vs_k.png` | `batched_throughput_vs_k.png` |
| Nsight CSV / report | `ncu_kbest_K*.csv`, `*.ncu-rep` | `ncu_batched_K*.csv`, `*.ncu-rep` |
| hand-written narrative | — | `BATCHED_BENCHMARK.md` (this file) |

**Correctness:** the batched sweep is checked `SCORE-EQUIVALENT` against the Week-3 CPU `TopKDecoder` over the first `--check` sentences at every K; `--batch` chunking must not change results (a chunked run and `--batch -1` agree sentence-for-sentence).

## Results

_Pending the A100 run (`bench_batched.slurm`)._ Expected: throughput ≫ the Week-7 per-sentence numbers (plan target up to ~50k sent/sec at K=32), with peak memory bounded by `--batch` (~4 GB envelope at 1,024 sentences / K=32). Populate from `batched_bench*.csv`:

| K | batch | sent/sec (kernel) | µs/sent (kernel) | table MB | used MB | recall@K | check |
|---|------:|------------------:|-----------------:|---------:|--------:|---------:|-------|
| _TODO — populate from batched_bench*.csv_ | | | | | | | |

### Batched vs per-sentence (headline comparison)

Fill this in from `batched_bench.csv` (whole-corpus sweep) against the Week-7 table in `KBEST_BENCHMARK.md`:

| K | K2 sent/sec (Week 7) | K3 batched sent/sec (Week 8) | speedup |
|---|---------------------:|-----------------------------:|--------:|
| 1  | 2730 | _TODO_ | _TODO_ |
| 5  | 2698 | _TODO_ | _TODO_ |
| 16 | 2730 | _TODO_ | _TODO_ |
| 32 | 2775 | _TODO_ | _TODO_ |
| 64 | 1210 | _TODO_ | _TODO_ |
