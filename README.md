# cuSHR

A GPU-accelerated lattice decoder and neural scorer for **Sanskrit word
segmentation** — splitting sandhi-fused text into words, and choosing each
word's lemma and morphological analysis.

The task is structured prediction over a DAG, not sequence labelling. The
Sanskrit Heritage Reader (SHR) proposes candidate analyses as a lattice; cuSHR
learns to score its edges and decodes the best path with Viterbi. That makes it
a *k-best Viterbi decoder with a learned biaffine scorer* — closer to
graph-based dependency parsing than to BIO tagging.

For joint lemma and morphology an optional second pass rescores the top-16
paths with a BiLSTM, which is what a first-order edge scorer structurally
cannot do: judge a candidate analysis as a whole.

---

## Results

**Segmentation on the SIGHUM 4,200-sentence test set: 91.52 PM / 98.32 word F,
third of twelve published systems** — 0.54 F behind TransLIST, at 1/210th of
ByT5's parameters and with no pretraining. Segmentation-specific models reach
93.43 ± 0.13 with the reranker. On joint lemma + morphology ByT5 still leads.

Batching the k-best merge collapses **1,205,796 kernel launches into 50** and
decodes the 119,503-sentence corpus **455× faster**, with bit-identical scores.

Full tables, the four findings behind them, and the known limits:
**[`RESULTS.md`](RESULTS.md)**. Current per-task comparison against TransLIST
and ByT5: [`papers/COMPARISON_OCTOBER.md`](papers/COMPARISON_OCTOBER.md).

---

## Repository

| directory | contents |
|---|---|
| `ingest/` | graphml + DCS pickles → lattice archive; gold-path resolution |
| `cushr_train/` | featurizers, biaffine scorer, contextual encoder, evaluation |
| `cushr_cpu/` | C++17 reference decoder — the correctness oracle |
| `cushr_gpu/` | CUDA k-best merge kernels, benchmarks, Nsight profiles |
| `viz/` | lattice visualiser |
| `papers/` | TransLIST and ByT5-Sanskrit, for the comparison |

## Documentation

**Accuracy** — [`PAPER_COMPARISON.md`](cushr_train/PAPER_COMPARISON.md) ·
[`CONTEXTUAL_ENCODER.md`](cushr_train/CONTEXTUAL_ENCODER.md) ·
[`FEATURIZER_COMPARISON_gold75.md`](cushr_train/FEATURIZER_COMPARISON_gold75.md) ·
[`GOLD49_VS_GOLD75.md`](cushr_train/GOLD49_VS_GOLD75.md)

**Data** — [`INGEST_METHODOLOGY.md`](ingest/INGEST_METHODOLOGY.md) ·
[`GOLD94_EDGE_ORDER.md`](cushr_train/GOLD94_EDGE_ORDER.md)

**Performance** — [`COMPARISON.md`](cushr_gpu/COMPARISON.md) ·
[`BATCHED_BENCHMARK.md`](cushr_gpu/BATCHED_BENCHMARK.md) ·
[`KBEST_BENCHMARK.md`](cushr_gpu/KBEST_BENCHMARK.md)

---

## Known limits

The lattice bounds accuracy (oracle 98.0 segmentation, 77.65 joint); the
reranker is bounded by the beam rather than that oracle; lemma and tag errors
are largely one error; case syncretism is the residual; contamination is
asymmetric, since ByT5 saw every benchmark sentence in pretraining while cuSHR
excludes all 4,200. Each of these is quantified in [`RESULTS.md`](RESULTS.md).
