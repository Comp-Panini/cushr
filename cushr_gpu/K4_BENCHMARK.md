# cuSHR K4 — the learned scorer inside the pipeline (Week 10 / CP-5)

**Status: measured.** All tables below are filled from jobs run on a Lonestar6
A100 (`gpu-a100-small`): `bench_k4.slurm` job 3378992 (13 m 43 s wall) for
throughput and correctness, and `profile_k4.slurm` for the Nsight Compute
counters. Every number is copied from `k4_bench_*.csv`, `k4_tests_results.txt`,
or the `ncu_k4_*.ncu-rep` reports, all of which are committed beside this file.
Nothing here is estimated except where a line says so explicitly.

**Headline:** correctness is exact — zero score and count mismatches against the
CPU decoder over all 119,503 sentences, recall@32 = **0.973053**. Throughput
clears the absolute target (**72,270** sent/sec at K=32, against ≥25,000) but
**misses the ≤2× target at 13.8×**. The profiler found the reason, and it is not
the one this document originally predicted: see
[The transpose fix](#the-transpose-fix-and-what-the-profiler-actually-found).

## What changed, and why it needed changing

Before this, the GPU never scored an edge. `host_driver_batched.cu` filled a
flat `float edge_score[num_edges]` on the host via `EdgeScorer::score()` and
uploaded it; K3 only ever read a number the CPU had already computed. Worse,
`make_scorer()` in the GPU drivers knew only `uniform | length | log_linear` —
it could not construct the trained `BiaffineScorer` at all. So every GPU
benchmark in this repo, including `BATCHED_BENCHMARK.md` and its recall@64 of
0.3679, ran on 43 hand-tuned log-linear weights and **not** on the trained
model. Those numbers and the trained model's F1 have never belonged in the same
sentence.

K4 closes that. The per-batch pipeline is now

```
load batch  ->  K4 (score edges)  ->  K3 (batched topo relax, K-best merge)  ->  K5 (reconstruct top-K)
```

with `score(u -> v) = <W_s x_u, W_d x_v> + b` computed on the device from the
same `'CSB3'` weight file the CPU scorer reads
(`cushr_train/export_weights.py --bin`).

## The kernels

`score_edges.cu`. Two implementations of the same arithmetic, selected with
`--k4`:

**`twopass` (default)** — K4a `project_nodes` runs one warp per node and writes
`S = W_s x` and `D = W_d x`; K4b `score_edges_twopass` runs one warp per edge
and does a length-`hidden` dot product. The g95 corpus has 78,847,461 edges over
4,488,155 nodes, so a node is the source of **17.6** edges on average and the
projection — which is essentially all of the FLOPs — happens 17.6× less often
than in the fused kernel.

**`fused`** — `score_edges_fused`, the literal one-warp-per-edge form from the
spec: each warp loads `node_feat[src]` and `node_feat[dst]` and multiplies them
against `W_src`/`W_dst` inline, with no reuse between edges that share an
endpoint. Kept so the price of that redundancy is a measured number rather than
an assertion.

Both split `hidden` the same way across lanes (lane *l* owns dims *l*, *l*+32, …)
and close with the same `__shfl_down_sync` butterfly, so **they agree bitwise**;
`cushr_score_tests` asserts it. Neither agrees bitwise with the CPU scorer,
which sums *h* in order 0..hidden — hence `--tol` on the oracle comparison.

**No tensor cores** — and this is now a settled call rather than an expectation,
though for a different reason than the one first written here. The profiler
found compute utilisation at **5.4–6.0%** in both slow kernels; WMMA would
accelerate a unit that is already ~95% idle. The worry originally recorded on
this line — that `W_s`+`W_d` at 192 KiB would thrash the A100's 192 KB per-SM
L1/shared budget — **did not happen**: L1 hit rate is 98.3% in `project_nodes`
and DRAM throughput is 0.14%. The bytes were always cached. What saturates is
the *number* of L1 sector requests, which is a different problem with a
different fix.

### Memory

Uploaded once for the whole corpus, at the headline dims:

| buffer | size |
|---|---|
| `node_features` (4,488,155 × 192 × 4 B) | 3.21 GiB |
| `in_col_idx`, `in_edge_id`, `in_dst`, `edge_score` (4 × 78,847,461 × 4 B) | 1.17 GiB |
| **resident before the k-best table** | **≈ 4.4 GiB** |

`in_dst` is new — the reverse-CSR row owning each slot, which K4 needs and K3
did not. Per tile: `S` and `D`, 2 × hidden × 4 B = 1 KB per node, capped by
`--k4-scratch` (default 512 MB ≈ 512K nodes). That scales with `hidden`, not
`feat_dim`. Tiles are cut on sentence boundaries so both endpoints of every edge
land in the same tile; the longest sentence in g95 is 400 nodes against a
~512K-node tile, and the driver refuses to run if a tile cannot hold it.

## Prerequisite: materialization — already done

`model95_ctx_ex4200` scores over a char-BiLSTM (`cushr_train/context.py`), which
no CUDA kernel can run, so the model has to be frozen into dense node vectors.
**That already happened.** `PAPER_COMPARISON.md:693–697` shows the model was
trained with `--materialize ../data/g95_ctx_ex4200.npz`, and
`cache95_ctx_ex4200` was then built from that file with `prepare.py`. Its
`node_features` are therefore 192 dense columns — 96 from the `hybrid_tag`
featurizer plus 96 from the char-BiLSTM, concatenated in that order
(`train.py:275`, `context.py:103`).

Verified numerically rather than assumed. Applying the model's projections
directly to those columns and running a top-1 Viterbi gives **63.39%** exact
gold-path match on the full 5,681-sentence test split, against the documented
base of **64.37** (`rerank_train.log:60`); the ~1-point gap is the form-filter
and span-start sort that log applies via `canon()` and this check does not.
Zeroing either 96-column block collapses it — 2.27% without cols 96–191, 21.33%
without cols 0–95 — so both halves are load-bearing and the contextual half is
demonstrably present.

The intermediate `data/g95_ctx_ex4200.npz` was deleted (≈3.3 GiB), but the cache
built from it survives and holds every array `Lattice::load_npz` wants, under
the same names and already in the right dtypes. So the only prerequisite is a
**repack**, not a re-materialization:

```
cd cushr_train
python cache_to_lattice_npz.py --cache ./cache95_ctx_ex4200 --out ../data/g95_ctx_mat.npz
python export_weights.py --model model95_ctx_ex4200.npz --bin model95_ctx.bin
```

Dtypes matter: `load_npz` reads the integer arrays with cnpy's `arr.data<int>()`,
a raw reinterpret with no width conversion, so an int64 column would be read as
garbage rather than rejected. `cache_to_lattice_npz.py` asserts every dtype
before writing.

## The four rows

`bench_k4.slurm` produces these in one job on one node, so the ratios are
measured rather than compared across jobs.

| Row | Scorer | `--k4` | What it isolates |
|---|---|---|---|
| A | `uniform` | `host` | The **unscored** decoder — the baseline the 2× target names |
| B | `biaffine` | `host` | Cost of the learned *model* to the decoder, with no K4 |
| C | `biaffine` | `twopass` | **The deliverable** |
| D | `biaffine` | `fused` | Cost of projecting per-edge instead of per-node (C/D) |

Row A is `uniform`, not `log_linear`. The 43 hand-tuned weights in
`bench_batched.slurm` belong to the old feat_dim=43 lattice, and
`LogLinearScorer` throws when `weights.size() != lat.feat_dim()`
(`cushr_cpu/src/scorer.cpp:29`) — this corpus is feat_dim=192, so those weights
would abort the job, and 192 invented ones would measure nothing. `uniform` is
also the more faithful baseline: the target is stated against the *unscored*
decoder.

Targets: **C within 2× of A** on `sent_per_sec_kernel`, and **≥ 25,000
sentences/sec at K = 32**. C and D must give identical recall@K — any difference
is a bug, not a tradeoff.

## Results

<!-- BEGIN AUTO-GENERATED RESULTS (make_benchmark_md.py --inject) -->

Source: `bench_k4.slurm` job 3378992, one A100 node, whole corpus (119,503
sentences / 4,488,155 nodes / 78,847,461 edges), `--batch -1`.

### Throughput and memory (whole-corpus sweep, `--batch -1`)

At **K = 32**, the target point:

| Row | µs/sent (K4) | µs/sent (K3) | µs/sent (total) | sent/sec | table MB |
|---|---|---|---|---|---|
| A `uniform`, `--k4 host` (unscored) | 0.000 | 1.002 | 1.002 | **997,582** | 1660.7 |
| B `biaffine`, `--k4 host` | 0.000 | 1.012 | 1.012 | 988,122 | 1660.7 |
| **C `biaffine`, `--k4 twopass`** | **12.826** | 1.010 | **13.837** | **72,270** | 1660.7 |
| D `biaffine`, `--k4 fused` | 219.274 | 1.018 | 220.292 | 4,539 | 1660.7 |

Full sweep, `sent/sec`:

| K | A unscored | B host-scored | C twopass | D fused |
|---|---|---|---|---|
| 1 | 603,243 | 622,199 | 72,262 | 4,540 |
| 5 | 1,001,520 | 993,946 | 72,263 | 4,540 |
| 16 | 993,641 | 986,218 | 72,216 | 4,539 |
| 24 | 990,815 | 985,818 | 72,184 | 4,539 |
| 32 | 997,582 | 988,122 | 72,270 | 4,539 |
| 48 | 234,181 | 232,135 | 58,396 | 4,472 |
| 64 | 234,015 | 226,035 | 58,052 | 4,470 |

**Against the targets:**

| target | result |
|---|---|
| ≥ 25,000 sent/sec at K=32 | **PASS** — 72,270, 2.9× the target |
| C within 2× of A | **FAIL** — 13.8× at K=32 (4.0× at K=64) |

Three things make that failure readable rather than just a red mark.

**B ≈ A at every K.** The learned model costs the *decoder* nothing; 988,122 vs
997,582 sent/sec is under 1%. All of the cost is device-side scoring, which is
what row B was built to isolate.

**K4's cost is flat in K** — 12.817 to 12.839 µs across K = 1…64, a 0.2% spread.
That is the designed behaviour: K4 writes `edge_score`, K3 consumes it, and K
never enters K4's work. So the C/A ratio moves only because the *baseline*
moves, which is why the same run is 13.8× at K=32 and 4.0× at K=64.

**The K=48 cliff is K3's, not K4's.** Rows A and B drop 4.3× between K=32 and
K=48 (1.002 → 4.270 µs) while K4 holds flat. That is the k-best merge's
register/occupancy cliff, already on record for this kernel, and it is untouched
by this work.

**D/C = 17.10×**, against a predicted 17.6× from the mean out-degree
(78,847,461 / 4,488,155). The spec-literal fused kernel costs almost exactly its
redundancy factor. But see the profiler section — the *mechanism* is not the one
that prediction assumed.

### Correctness

| Run | `--k4` | score_mismatch | count_mismatch | n_check | recall@K |
|---|---|---|---|---|---|
| smoke, K=32, batch 8192 | twopass | **0** | **0** | 2,000 | 0.972595 |
| smoke, K=32, batch 8192 | fused | **0** | **0** | 2,000 | 0.972595 |
| **E, K=32, batch −1** | twopass | **0** | **0** | **119,503** | **0.973053** |

Row E checked **every sentence in the corpus** against the CPU `TopKDecoder`
running the same biaffine model: zero score mismatches within `--tol`, and zero
count mismatches — the path/count comparison is exact, not toleranced. The two
smoke rows agree to six decimal places on recall, which is the corpus-scale
expression of the bitwise property the unit tests assert.

For scale: the old log-linear GPU path recorded recall@64 = 0.3679
(`BATCHED_BENCHMARK.md`). This is 0.973 at K=32. Those two numbers measure
different scorers, which was the entire point of the exercise.

Unit tests (`k4_tests_results.txt`, `cushr_score_tests`) — all 15 pass:

| case | twopass vs host ref | fused vs host ref | fused vs twopass |
|---|---|---|---|
| headline (4096 nodes, fan 3, feat_dim 192, hidden 128) | rel 2e-6 | rel 2e-6 | **0 differing edges** |
| ragged (1000, fan 5, feat_dim 43, hidden 100) | rel 0.000000 | rel 0.000000 | **0 differing edges** |
| narrow hidden (300, fan 2, feat_dim 16, hidden 8) | rel 0.000000 | rel 0.000000 | **0 differing edges** |
| tiny (1 node, fan 3, feat_dim 32, hidden 32) | rel 0.000000 | rel 0.000000 | **0 differing edges** |

Plus the loader: CSB3 round-trips, CSB2 is rejected, unrecognised magic is
rejected.

### Kernel bottlenecks (Nsight Compute)

`profile_k4.slurm`, A100, `--set full`, per-launch means over 100 (twopass) and
115 (fused) launches at `--K 32 --batch 1024`:

| kernel | duration | **SM %peak** | **L1/TEX %peak** | **DRAM %peak** | occupancy % | L1 hit % | L2 hit % |
|---|---|---|---|---|---|---|---|
| `project_nodes` | 24.12 ms | **5.98** | **99.24** | **0.14** | 92.8 | 98.3 | 95.8 |
| `score_edges_twopass` | 0.35 ms | 59.61 | 45.11 | 9.25 | 85.8 | 50.1 | 83.4 |
| `score_edges_fused` | 413.42 ms | **5.38** | **99.80** | **0.01** | 91.6 | 82.6 | 99.9 |

**Every prediction this document recorded was wrong.** They are kept below,
struck through, because the point of writing them down was to let the run
contradict them:

- ~~`project_nodes` compute bound (it is the matmul)~~ — it is at **5.98%**
  compute. It achieved 1% of the device's fp32 peak.
- ~~`score_edges_twopass` bandwidth bound (it streams S and D per edge)~~ — it
  is the *most* compute-heavy kernel of the three, 59.6% SM against 9.25% DRAM,
  and it is the only one that is reasonably balanced.
- ~~`score_edges_fused` compute bound, possibly plus W cache thrashing~~ —
  compute is **5.38%** and DRAM is **0.01%**. Nothing thrashes to memory.

The real bottleneck in both slow kernels is **L1/TEX request throughput at
99.2–99.8% of peak**, with the FMA pipes idle and DRAM untouched. The warp state
counters agree: `project_nodes` warps spend ~211 cycles on MIO throttle and ~206
on LG throttle out of ~473 cycles between issues, and only 0.17 warps per
scheduler are eligible per cycle despite 92.8% occupancy. The kernel is not
short of warps; every warp it has is queued behind the load/store unit.

**`project_nodes` is 98.6% of twopass's K4 time** (24.12 of 24.46 ms per
launch). The per-edge dot product is effectively free. The two-pass design was
the right call — the projection kernel's memory *access pattern* is what costs,
not the projection itself.

### SIGHUM-test F1 vs TransLIST

Headline is **decoder top-1**; the reranker row is the CP-5 stretch.

| System | S | L | S+M | L+M | S+L+M |
|---|---|---|---|---|---|
| TransLIST (published) | | | | | |
| cuSHR K4 top-1 (GPU) | | | | | |
| cuSHR K4 + reranker | | | | | |
| ORACLE @ K=32 | | | | | |

**Still pending** — this is the one table with no data. `--dump-paths` wrote
`gpu_k32_K32.npz` (155 MB, top-32 paths for all 119,503 sentences; kept out of
git, regenerate from row E), but the conversion and evaluation have not been
run:

```
cd ../cushr_train
python gpu_paths_to_rerank.py --gpu ../cushr_gpu/gpu_k32_K32.npz     --cache ./cache95_ctx_ex4200 --out gpu_rerank_k32.npz
python eval_slm.py ...                                  # top-1
python eval_slm.py --rerank reranker_full.pt ...        # stretch row
```

`gpu_paths_to_rerank.py` prints its own recall@32; it must equal **0.973053**,
the figure `cushr_batched` printed for the same run. A disagreement means the
form filter or the span-start sort was applied differently on the two sides, and
the candidate lists are not comparable to `make_rerank_data.py`'s.

<!-- END AUTO-GENERATED RESULTS -->

## The transpose fix, and what the profiler actually found

**Status: implemented, compiles, not yet re-measured on hardware.** The numbers
in this section that describe the *problem* are measured; the numbers that
describe the *expected improvement* are arithmetic and are labelled as such. No
row above has been updated for it — the tables still report the pre-transpose
run, and they stay that way until a new job replaces them.

### The cause

`row_dot` evaluates one row of a projection matrix against a node's feature
vector, and callers give lane *l* the hidden dims *l*, *l*+32, *l*+64, … So at
any instant the 32 lanes of a warp are working on 32 **different rows** of `W`
at the same column *i*.

With `W` stored row-major `[hidden][feat_dim]` — the CSB3 file's layout, and
what the kernels originally read — lane *h* loads `W[h*feat_dim + i]`.
Consecutive lanes are `feat_dim * 4 = 768 B` apart, so a single warp load
instruction touches **32 separate sectors**. That is a fully uncoalesced access,
issued in the innermost loop of the kernel that accounts for 98.6% of K4's time.

The counters say exactly this. L1 hit rate is 98.3% — the data *is* cached, W is
not being re-fetched from DRAM (0.14% throughput) — yet L1/TEX sits at 99.24% of
peak. The wall is the request *rate*, not the bytes. Compute idles at 5.98%
while warps queue on MIO and LG throttle.

### The fix

Store the projections transposed, `[feat_dim][hidden]`, so lane *h* loads
`WT[i*hidden + h]`. Consecutive lanes are then 4 B apart and a warp's 32 loads
fall inside 128 contiguous bytes: **4 sectors instead of 32**.

- The **on-disk CSB3 format is unchanged.** `cushr_cpu`'s `BiaffineScorer` reads
  it, and `model95_ctx.bin` needs no re-export. The transpose is applied on the
  host at upload time by `transpose_proj()` (`score_edges.cu`), once per run over
  24,576 floats — nothing against a 3.21 GiB node-feature upload.
- The struct members are renamed `d_src_projT` / `d_dst_projT` deliberately.
  Passing a row-major buffer would compile and silently produce wrong scores, so
  the type system gets to catch it instead.
- **Bit-identical results.** `i` still runs `0..feat_dim` in order and the
  summation order is untouched; only the address arithmetic changed. So the
  fused-vs-twopass bitwise equality and the `--check` agreement both carry over
  unchanged, and `tests/test_score_edges.cu` now also covers `transpose_proj`
  itself — its host reference still uses the row-major matrices, so a wrong
  transpose fails the existing comparison rather than passing quietly.

### What to expect, and what not to

Arithmetic, not measurement: an 8× reduction in sector requests, if the kernel
scales with it perfectly, takes K4 from 12.826 µs to ≈1.60 µs and the pipeline
to ≈2.61 µs/sent — about **2.6× the unscored baseline**, down from 13.8×.

That would be a large win and **still short of the ≤2× target.** Hitting 2×
requires K4 under 0.994 µs, i.e. a ~12.9× speedup, and the transpose's ceiling is
8×. Perfect scaling is also unlikely: as requests drop, some other limiter — the
`x` shared-memory reads, or compute finally becoming relevant — takes over. The
honest expectation is "materially better, target still missed," and the next
profile decides what the new bottleneck is.

Verification order when the hardware is next available:

```
sbatch test_k4.slurm     # bitwise + host-reference agreement must be unchanged
sbatch bench_k4.slurm    # rows A-E; recall@32 must still be 0.973053 exactly
sbatch profile_k4.slurm  # is project_nodes still L1-request bound?
```

Recall@32 changing *at all* would mean the transpose altered the arithmetic,
which it must not.

## If top-1 F1 lands below TransLIST

Week-11 recovery, in diagnostic order:

1. **recall@K from the checked run.** If the gold path is not in the beam at
   K=32, no scoring or reranking change can help and the problem is upstream.
2. **The reranker row.** If top-1 is weak but the reranker row is strong, the
   scores are ordering badly rather than being wrong.
3. **Only then, the scoring model.**

Do not tune the kernel for F1. K4 is exact-by-construction against the CPU
scorer, and `--check` says so; any F1 movement from touching it would be a bug
being introduced, not a model being improved.

## Reproducing

```
sbatch test_k4.slurm     # unit tests: both kernels vs host ref, and vs each other
sbatch bench_k4.slurm    # the four rows + the checked/dumped K=32 run
sbatch profile_k4.slurm  # ncu counters, per kernel -- the compute/bandwidth split
```

`bench_k4.slurm` and `profile_k4.slurm` answer different questions and neither
substitutes for the other. The bench gives wall-clock, but its `us_per_sent_k4`
column wraps BOTH twopass kernels in one cudaEvent span, so it cannot separate
`project_nodes` from `score_edges_twopass` — and it carries no counters at all.
The profile gives per-kernel counters but not end-to-end throughput. The profile
run uses `--K 32` alone, since K4's output does not depend on K.

Prerequisites and the exact materialize/export commands are in the header
comment of `bench_k4.slurm`.
