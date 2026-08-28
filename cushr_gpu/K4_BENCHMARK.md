# cuSHR K4 — the learned scorer inside the pipeline (Week 10 / CP-5)

**Status: measured, post-transpose.** All tables below are filled from jobs run
on a Lonestar6 A100 (`gpu-a100-small`) from
`/home1/11503/njhavar/cushr/cushr_gpu`. The post-transpose set is **job
3393407** (`test_k4`, 7 s), **job 3394531** (`bench_k4`, 10 m 26 s) and **job
3395099** (`k4_ncu`, 35 m 25 s); the pre-transpose baseline is job 3378992 with
its companion profile (`ncu_k4_*_20260820_1516`). Every number is copied
from `k4_bench_*.csv`,
`k4_tests_results.txt`, or the `ncu_k4_*.ncu-rep` reports, all committed beside
this file. Nothing here is estimated except where a line says so explicitly.

**Headline:** correctness is exact — zero score and count mismatches against the
CPU decoder over all 119,503 sentences, recall@32 = **0.973053**, *unchanged to
six decimals by the transpose*, which is the property that fix had to preserve.
Throughput clears the absolute target by 18.6× (**464,995** sent/sec at K=32,
against ≥25,000) and now sits at **2.14× the unscored baseline**, down from
13.8× before the fix — passing the ≤2× target at K=1, 5, 48 and 64 and missing
it by 7% at K=16/24/32. See
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

**No tensor cores** — settled, though the reason changed twice. The worry
originally recorded here, that `W_s`+`W_d` at 192 KiB would thrash the A100's
192 KB per-SM L1/shared budget, **never happened**: L1 hit rate was 98.3% and
DRAM throughput 0.14%, so the bytes were always cached. The first profile then
found compute at 5.4–6.0%, which made WMMA look pointless for the opposite
reason — it would accelerate an idle unit. After the transpose, `project_nodes`
sits at **87.89% of peak SM throughput**, so that unit is no longer idle either.
The call still stands, now on the only ground that survives: TF32 would change
the reduction order and cost the bit-exactness the `--check` and unit-test gates
depend on, to chase ~1.14× of remaining arithmetic headroom.

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

Source: `bench_k4.slurm` job **3394531** (10 m 26 s), one A100 node, whole
corpus (119,503 sentences / 4,488,155 nodes / 78,847,461 edges), `--batch -1`.
The pre-transpose figures from job 3378992 (13 m 43 s) are kept alongside, since
the delta is the result of this week's second half.

### Throughput and memory (whole-corpus sweep, `--batch -1`)

At **K = 32**, the target point:

| Row | µs/sent (K4) | µs/sent (K3) | µs/sent (total) | sent/sec | table MB |
|---|---|---|---|---|---|
| A `uniform`, `--k4 host` (unscored) | 0.000 | 1.001 | 1.001 | **998,666** | 1660.7 |
| B `biaffine`, `--k4 host` | 0.000 | 1.011 | 1.011 | 988,708 | 1660.7 |
| **C `biaffine`, `--k4 twopass`** | **1.092** | 1.058 | **2.151** | **464,995** | 1660.7 |
| D `biaffine`, `--k4 fused` | 14.360 | 1.056 | 15.417 | 64,865 | 1660.7 |

What the transpose bought, same job structure, K = 32:

| | before | after | speedup |
|---|---|---|---|
| K4 `twopass` | 12.826 µs | **1.092 µs** | **11.75×** |
| K4 `fused` | 219.274 µs | **14.360 µs** | **15.27×** |
| Row C end-to-end | 13.837 µs | **2.151 µs** | **6.43×** |
| Row C throughput | 72,270 sent/sec | **464,995 sent/sec** | **6.43×** |

Full sweep, `sent/sec`:

| K | A unscored | B host-scored | C twopass | D fused |
|---|---|---|---|---|
| 1 | 727,688 | 659,521 | 471,642 | 64,961 |
| 5 | 923,649 | 994,886 | 469,659 | 65,081 |
| 16 | 991,547 | 990,798 | 466,083 | 65,043 |
| 24 | 992,728 | 991,227 | 467,182 | 65,052 |
| 32 | 998,666 | 988,708 | 464,995 | 64,865 |
| 48 | 234,241 | 233,679 | 185,587 | 53,392 |
| 64 | 234,064 | 227,706 | 182,091 | 53,175 |

**Against the targets:**

| target | result |
|---|---|
| ≥ 25,000 sent/sec at K=32 | **PASS** — 464,995, 18.6× the target |
| C within 2× of A | **PASS at K=1, 5, 48, 64; MISS by ~7% at K=16, 24, 32** |

The C/A ratio by K, which is the honest way to report a target that is no longer
uniformly missed:

| K | 1 | 5 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| C/A | 1.54× | 1.97× | 2.13× | 2.13× | **2.15×** | 1.26× | 1.29× |

Against row B — the fairer comparison, since it holds the scorer fixed and
removes only the device-side scoring — K=32 is **2.13×**. Either way the target
point misses by about 7%, against 13.8× before the fix.

Four things make that readable.

**B ≈ A at every K.** The learned model costs the *decoder* nothing; 988,708 vs
998,666 sent/sec is under 1%. All of the remaining cost is device-side scoring,
which is what row B was built to isolate.

**K4's cost is still flat in K** — 1.085 to 1.098 µs across K = 1…64, a 1.2%
spread. That is the designed behaviour: K4 writes `edge_score`, K3 consumes it,
and K never enters K4's work. So the C/A ratio moves only because the *baseline*
moves, which is why the same run is 2.15× at K=32 and 1.29× at K=64.

**The K=48 cliff is K3's, not K4's.** Rows A and B drop 4.2× between K=32 and
K=48 (1.001 → 4.269 µs) while K4 holds flat. That is the k-best merge's
register/occupancy cliff, already on record for this kernel, untouched by this
work — and it is also why the 2× target *passes* at K=48 and 64: the baseline
got slower, not K4.

**D/C = 13.15×**, against 17.10× before the transpose and a 17.6× prediction
from the mean out-degree (78,847,461 / 4,488,155). The redundancy argument for
the two-pass design still holds, but the fused kernel gained slightly more from
coalescing than twopass did, so the gap narrowed.

### Correctness

| Run | `--k4` | score_mismatch | count_mismatch | n_check | recall@K |
|---|---|---|---|---|---|
| smoke, K=32, batch 8192 | twopass | **0** | **0** | 2,000 | 0.972595 |
| smoke, K=32, batch 8192 | fused | **0** | **0** | 2,000 | 0.972595 |
| **E, K=32, batch −1** | twopass | **0** | **0** | **119,503** | **0.973053** |

All three rows are from job 3394531.

Row E checked **every sentence in the corpus** against the CPU `TopKDecoder`
running the same biaffine model: zero score mismatches within `--tol`, and zero
count mismatches — the path/count comparison is exact, not toleranced. The two
smoke rows agree to six decimal places on recall, which is the corpus-scale
expression of the bitwise property the unit tests assert.

**All three recall figures are identical to the pre-transpose run**, digit for
digit: 0.973053 on the full corpus and 0.972595 on both smoke rows. That is the
gate the transpose had to pass. It only changed address arithmetic, never the
summation order, so any movement at all in these numbers would have meant the
rewrite was wrong; there is none.

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

`profile_k4.slurm`, A100, `--set full`, per-launch means at `--K 32 --batch
1024`. Both runs are shown: `ncu_k4_*_20260820_1516` (pre-transpose, 100/115
launches) against `ncu_k4_*_20260827_1301` (job **3395099**, 35 m 25 s,
post-transpose, 100/117 launches):

| kernel | run | duration | **SM %peak** | **L1/TEX %peak** | occupancy % | L1 hit % | issue active % |
|---|---|---|---|---|---|---|---|
| `project_nodes` | before | 24.12 ms | **5.98** | **99.24** | 92.8 | 98.3 | 3.1 |
| `project_nodes` | **after** | **1.64 ms** | **87.89** | 89.62 | 93.9 | 87.8 | **61.3** |
| `score_edges_twopass` | before | 0.349 ms | 59.61 | 45.11 | 85.8 | 50.1 | 60.3 |
| `score_edges_twopass` | after | 0.346 ms | 60.11 | 45.49 | 85.8 | 50.1 | 60.8 |
| `score_edges_fused` | before | 413.42 ms | **5.38** | **99.80** | 91.6 | 82.6 | 3.0 |
| `score_edges_fused` | **after** | **24.00 ms** | **92.52** | 92.68 | 99.5 | 87.0 | **69.4** |

**`project_nodes` flipped from memory-bound to compute-bound.** It is 14.7×
faster (24.12 → 1.64 ms) and the two counters traded places: SM went 5.98 →
**87.89%** while L1/TEX came off its 99.24% ceiling. The stall reasons collapsed
with it — MIO throttle 210.7 → **8.4** cycles per issue-active, LG throttle
206.2 → **2.1** — and issue rate went 3.1 → **61.3%**. The kernel was never
short of warps (92.8% occupancy before, 93.9% after); every warp it had was
queued behind the load/store unit, and now they issue.

`score_edges_fused` moved the same way, 413.42 → 24.00 ms at 92.52% SM.
`score_edges_twopass` is unchanged within noise on every counter, which is the
expected control: it reads `S` and `D`, never the projection matrices, so the
transpose could not touch it.

**`project_nodes` is now 82.6% of twopass K4 time** (1.642 of 1.988 ms), down
from 98.6%.

### Reading "5% compute, 99% memory"

That pairing is the most diagnostic signature in the table, and the obvious
reading of it is wrong. Recording the correct one here because it is the thing
that generalises past this kernel.

**The two numbers are not two halves of one pie.** `sm__throughput` and
`l1tex__throughput` are each a percentage of peak for a *different pipeline*;
they do not sum to 100. 5.98 / 99.24 does not mean "6% of the work is math and
94% is memory". It means the L1/TEX pipe was pinned at its own ceiling while the
math units sat idle. Whichever pipe reads ~100% is the limiter; everything else
is downstream of it.

**L1/TEX throughput is measured in requests, not bytes.** This is the part that
misleads. It saturates on *sector transactions per cycle*, not data volume, and
for pre-transpose `project_nodes` the byte traffic was negligible:

| `project_nodes` | before | after |
|---|---:|---:|
| L1/TEX throughput | **99.241%** | 89.616% |
| L1 hit rate | 98.342% | 87.791% |
| **DRAM throughput** (`gpu__dram_throughput`) | **0.136%** | 2.012% |

DRAM was doing essentially nothing — **0.136% of peak** — and 98.3% of accesses
hit in L1. The data was already resident; the memory system was barely moving
bytes. (`score_edges_fused` was more extreme still, at **0.006%**.) So 99%
"memory throughput" did **not** mean bandwidth-bound — the L1 was saturated *issuing
transactions for data it already held*. Row-major `W` put 32 lanes on 32 rows
768 B apart, so one warp-wide load became **32 sector requests instead of 4**:
identical bytes, 8× the transactions. The transpose changed only address
arithmetic and moved not one byte more or less.

DRAM throughput *rising* afterwards (0.136 → 2.012%) is the expected direction
and not a regression: the same byte traffic compressed into a 14.7× shorter
kernel is a higher rate by definition. It is still 2% of peak, which is why the
post-transpose kernel is compute-bound rather than newly bandwidth-bound.

**The 5% compute figure is an effect, not an independent fact.** The chain runs:
LSU queue backs up (MIO throttle **210.7** warps stalled per issue-active cycle)
→ warps cannot issue (issue rate **3.1%**) → the SM has nothing to execute (SM
throughput **5.98%**). The math units were never the problem. They were starved.

**Occupancy was a red herring throughout.** 92.8% before, 93.9% after — nearly
unchanged across a 14.7× speedup. There were always plenty of resident warps;
they simply could not *issue*. Occupancy counts warps present, not warps making
progress, and tuning for it here would have moved nothing.

The rule worth carrying to K3 or any future kernel:

| SM % | L1 % | **DRAM %** | reading |
|---|---|---|---|
| low | high | **low** | access-pattern problem — fix coalescing, same bytes |
| low | high | **high** | genuinely bandwidth-bound — reduce bytes or improve reuse |
| high | low | low | compute-bound — where `project_nodes` is now |

**Always check DRAM before concluding "memory-bound."** Low SM% with high L1%
and low DRAM% means paying for transactions that are not needed, which is
usually cheap to fix and — as here — bit-preserving. The high-DRAM version is
the expensive one, and it is the one that would have justified the algorithmic
restructuring this document originally proposed.

### What the first profile said (pre-transpose)

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

The real bottleneck in both slow kernels was **L1/TEX request throughput at
99.2–99.8% of peak**, with the FMA pipes idle and DRAM untouched (0.14% and
0.01%). Nothing was thrashing to memory — the bytes were cached; the *request
rate* was the wall. That diagnosis is what the transpose acted on, and the
post-transpose column above confirms it: removing the 32-sector loads moved
`project_nodes` to 87.89% SM.

**`project_nodes` is 98.6% of twopass's K4 time** (24.12 of 24.46 ms per
launch). The per-edge dot product is effectively free. The two-pass design was
the right call — the projection kernel's memory *access pattern* is what costs,
not the projection itself.

### Where that leaves K4

`project_nodes` now runs at **87.89% of peak SM throughput**. There is no
memory-side headroom left to recover and roughly 1.14× of arithmetic headroom in
principle, which no real kernel reaches. The remaining 0.149 µs between row C
(2.151 µs) and the 2.002 µs target is therefore **not available from K4 by
bit-preserving means**. The only lever that would move it further is TF32 tensor
cores, which change the reduction order and would cost the exact-by-construction
property — a bad trade for 7%.

Two consequences worth stating plainly:

- The `## No tensor cores` reasoning earlier in this document is now obsolete on
  its own terms. It argued WMMA would accelerate an idle unit; that unit is no
  longer idle. The conclusion stands, but for the opposite reason — the kernel
  is close enough to fp32 peak that only a different numeric format would help.
- ~38% of the remaining gap is not K4's at all. K3 costs **1.058 µs in row C
  against 1.001 in row A** at identical K and identical table size. That
  difference, not the projection kernel, is the cheaper thing left to explain.

### SIGHUM-test F1 vs TransLIST

Headline is **decoder top-1**; the reranker row is the CP-5 stretch.

| System | S | L | S+M | L+M | S+L+M |
|---|---|---|---|---|---|
| TransLIST (published) | **93.97** | — | — | — | — |
| cuSHR K4 top-1 (GPU) | 91.52 | 65.62 | 45.69 | 45.45 | 45.29 |
| cuSHR K4 + reranker | 91.98 | 65.98 | **50.29** | **50.02** | **49.86** |
| ORACLE @ K=32 | 98.40 | 71.71 | 66.00 | 66.29 | 66.00 |

Sentence-level perfect match over the 4,200 SIGHUM test sentences, no
convention maps applied. TransLIST reports only word-level segmentation
(its PM = 93.97, `PAPER_COMPARISON.md`), so its lemma and morph cells are
blank because the paper has no number there, not because we did not measure.

**The top-1 row is digit-identical to the CPU decoder** — 91.52 / 65.62 /
45.69 / 45.45 / 45.29, the same five figures `eval_slm.py` produces with no
`--cands` at all. That is the result this table exists to report: K4 scoring
edges on the device changes throughput and nothing else. The two columns run
through one `analysis()`, one `score()` and one reference; only the origin of
the paths differs.

The reranker moved the top-1 path on **641 / 4,200 sentences (15.3%)** and is
worth **+4.6 points** at S+M. Its ceiling is the K=32 row, not the gold-path
oracle — it can only choose among the 32 candidates K3 produced.

cuSHR remains **2.45 points behind TransLIST** on segmentation (91.52 vs 93.97;
91.98 reranked). The gap is real and not a kernel problem: the GPU reproduces the
CPU decoder exactly, so closing it is a scoring-model question, not a K4 one.

Provenance: `gpu_k32_K32.npz` from row E (job 3394531), converted by
`gpu_paths_to_rerank.py`, which reported **recall@32 = 97.3053%** — matching
row E's `recall_at_K` of `0.973053` to all six decimals, so the candidate lists
are confirmed comparable to `make_rerank_data.py`'s.

### The same pipeline vs ByT5-Sanskrit

`eval_slm.py --pred-jsonl` scores ByT5's own predictions through **this same
reference and this same `score()`**, so the only difference between the columns
is where the predictions came from. Both rows below are the GPU pipeline's
paths.

| level | cuSHR K4 (GPU, +rerank +maps) | ByT5-Sanskrit (measured, same 4,200) | ORACLE +maps |
|---|---:|---:|---:|
| S | **91.98** | 81.38 | 98.00 |
| L | 74.52 | **90.55** | 79.27 |
| S+M | 57.40 | **67.79** | 77.65 |
| L+M | 57.48 | **76.50** | 77.86 |
| S+L+M | 57.00 | **66.90** | 77.36 |

Without the convention maps and the reranker the cuSHR column is 91.52 / 65.62 /
45.69 / 45.45 / 45.29; ByT5's column is unaffected by either, since the maps are
never applied to it and it emits DCS conventions natively.

**The split is clean and it is not a wash.** cuSHR wins segmentation by **+10.6
points** (91.98 vs 81.38) — it is constrained to SHR's lexicon, so it proposes
only analysable splits. ByT5 wins every level that involves a lemma or a
morphological tag, by 11 to 19 points.

The reason is visible in the oracle column: **ByT5 scores 90.55 on L against our
own gold path's 79.27.** A system cannot beat our ceiling by being a better
decoder — it beats it because the ceiling is a convention artifact. Our gold
path carries SHR's participial stems where DCS wants the verbal root, and the
163-rule convention map recovers only part of that (L 65.62 → 74.52). ByT5 was
trained on DCS and has no gap to close. So L / S+M / L+M / S+L+M here measure
convention agreement at least as much as model quality, and the L row in
particular should not be read as ByT5 analysing Sanskrit 16 points better than
cuSHR does.

The honest summary: **cuSHR segments better, ByT5 labels better**, and the
labelling gap is substantially — not entirely — a vocabulary-convention gap that
the maps have not finished closing. None of it is a K4 result; the GPU's top-1
is identical to the CPU decoder's, so every number here is a scoring-model and
convention finding that would read the same on either backend.

<!-- END AUTO-GENERATED RESULTS -->

## The transpose fix, and what the profiler actually found

**Status: implemented and measured.** Every table above reports the
post-transpose code. The prediction this section originally made is kept below
next to what actually happened.

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

### What was predicted, and what happened

The prediction written here before the run, from sector arithmetic alone: an 8×
reduction in requests, scaling perfectly, would take K4 from 12.826 µs to
≈1.60 µs and the pipeline to ≈2.61 µs/sent — about **2.6× the unscored
baseline**, down from 13.8×.

**The measurement beat the prediction.** K4 went to **1.092 µs**, not 1.60, and
the pipeline to **2.151 µs/sent**, not 2.61 — a **2.14×** ratio to unscored
rather than the predicted 2.6×. The 8× sector count was a floor, not a ceiling:
the realised gain is 11.75×, so the fix also improved cache and issue behaviour
beyond the request count itself. The refreshed profile will say which.

The prediction was right about the *shape* of the outcome and wrong about the
magnitude in the conservative direction. It was also right that the ≤2× target
would still be missed at the target point — 2.14× against 2.00× — though it is
now missed by 7% rather than by 590%, and the target passes outright at K=1, 5,
48 and 64. Closing that last 7% needs K4 under 0.994 µs, another 1.10×.

Verification actually run, and its results:

| | check | result |
|---|---|---|
| `test_k4.slurm` | bitwise + host-reference agreement unchanged | **15/15 pass**, `fused == twopass` 0 differing edges, worst rel 2e-6 |
| `bench_k4.slurm` | recall@32 still exactly 0.973053 | **0.973053**, `score_mismatch` 0, `count_mismatch` 0, `n_check` 119,503 |
| `profile_k4.slurm` | is `project_nodes` still L1-request bound? | **running** |

Recall@32 changing *at all* would have meant the transpose altered the
arithmetic. It did not move.

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

One reporting caveat. Row E's CSV lists `us_per_sent_k4 = 1.805` where row C
lists 1.092 at the same K=32. They are different processes: row C runs K4 seven
times in one process during the sweep, so its later K values profit from warmed
GPU clocks, while row E is a single cold K with `--check` and `--dump-paths`
attached. **Row C is the number to quote for throughput; row E is the number to
quote for correctness.** They are not two measurements of one quantity.

`bench_k4.slurm` and `profile_k4.slurm` answer different questions and neither
substitutes for the other. The bench gives wall-clock, but its `us_per_sent_k4`
column wraps BOTH twopass kernels in one cudaEvent span, so it cannot separate
`project_nodes` from `score_edges_twopass` — and it carries no counters at all.
The profile gives per-kernel counters but not end-to-end throughput. The profile
run uses `--K 32` alone, since K4's output does not depend on K.

Prerequisites and the exact materialize/export commands are in the header
comment of `bench_k4.slurm`.
