# cuSHR K4 — the learned scorer inside the pipeline (Week 10 / CP-5)

**Status: implemented, not yet measured.** The kernels, the driver integration
and the unit tests are in the tree; the numbers below are blank because the
Lonestar6 jobs have not run. Nothing in this file is estimated or predicted —
each table gets filled from the CSVs `bench_k4.slurm` writes, and until then the
cells stay empty. Do not quote this document as a result.

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

**No tensor cores** — but the usual justification does *not* apply here, so this
is an expectation awaiting the profiler rather than a settled call. At the
headline dims (feat_dim **192**, hidden 128) `W_s` and `W_d` are 96 KiB each,
**192 KiB together**, which is the A100's entire per-SM combined L1/shared
budget — and each block is already holding 6–12 KiB of that as shared memory. So
the "the matrices are tiny and live in L1" argument that would normally make
`row_dot`'s strided row access free is unavailable. The expectation is still
that these kernels are bound by streaming node features and edge scores rather
than by math, but the C-vs-D rows and the ncu profile are what decide it.

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

### Throughput and memory (whole-corpus sweep, `--batch -1`)

| K | Row | us/sent (K4) | us/sent (K3) | us/sent (total) | sent/sec | table MB |
|---|---|---|---|---|---|---|
| | | | | | | |

*Pending `sbatch bench_k4.slurm` → `k4_bench_{A,B,C,D}_*.csv`.*

### Correctness

| Run | score_mismatch | count_mismatch | n_check | recall@K |
|---|---|---|---|---|
| | | | | |

*Pending `k4_bench_smoke.csv`, `k4_bench_smoke_fused.csv`, `k4_bench_E_k32_checked.csv`.*

### Kernel bottlenecks (Nsight Compute)

The C-vs-D wall-clock rows say which variant wins; these say why. Whichever of
`sm__throughput` / `dram__throughput` sits closer to peak is that kernel's
bottleneck — both low means latency bound, and the occupancy and stall columns
in the wide CSV explain it.

| kernel | sm__throughput %peak | dram__throughput %peak | occupancy % | L1 hit % | L2 hit % | verdict |
|---|---|---|---|---|---|---|
| `project_nodes` | | | | | | |
| `score_edges_twopass` | | | | | | |
| `score_edges_fused` | | | | | | |

*Pending `sbatch profile_k4.slurm` → `ncu_k4_{twopass,fused}_<TAG>.csv`.*

The prediction on record, so the run can contradict it: `project_nodes` compute
bound (it is the matmul), `score_edges_twopass` bandwidth bound (it streams S
and D per edge), `score_edges_fused` compute bound. The open question is the L1
hit rate on the fused row — at feat_dim=192 `W_s`+`W_d` is 192 KiB against the
A100's 192 KB per-SM L1/shared, so if that rate is poor then the 17.6× FLOP
redundancy is not the whole story and cache thrashing on `W` is a second,
separable cost. That distinction is invisible to wall-clock timing.

### SIGHUM-test F1 vs TransLIST

Headline is **decoder top-1**; the reranker row is the CP-5 stretch.

| System | S | L | S+M | L+M | S+L+M |
|---|---|---|---|---|---|
| TransLIST (published) | | | | | |
| cuSHR K4 top-1 (GPU) | | | | | |
| cuSHR K4 + reranker | | | | | |
| ORACLE @ K=32 | | | | | |

*Pending the `--dump-paths` run, then:*

```
cd ../cushr_train
python gpu_paths_to_rerank.py --gpu ../cushr_gpu/gpu_k32_K32.npz \
    --cache ./cache95_ctx_ex4200 --out gpu_rerank_k32.npz
python eval_slm.py ...                                  # top-1
python eval_slm.py --rerank reranker_full.pt ...        # stretch row
```

`gpu_paths_to_rerank.py` prints its own recall@32; it must equal the number
`cushr_batched` printed for the same run. A disagreement means the form filter
or the span-start sort was applied differently on the two sides, and the
candidate lists are not comparable to `make_rerank_data.py`'s.

<!-- END AUTO-GENERATED RESULTS -->

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
