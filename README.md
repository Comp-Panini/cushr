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

**Segmentation on the SIGHUM 4,200-sentence test set**, against every published
system on that benchmark:

| Model | Params | Pretraining | PM | Word F |
|---|---:|---|---:|---:|
| TransLIST | Transformer + SHR lexicon | none | **93.97** | **98.86** |
| ByT5-Sanskrit | 582M | 6.5B tokens | 93.83 | – |
| **cuSHR** | **2.77M** | **none** | **91.52** | **98.32** |
| rcNN-SS | – | none | 87.08 | 96.84 |
| FLAT-Lattice | – | none | 85.65 | 96.72 |
| *(7 further systems, 83.88 down to 38.64 PM)* | | | | |

**Third of twelve on perfect match, third on word F — 0.54 F behind the
leader, at 1/210th of ByT5's parameters and with no pretraining.** Inference is
a deterministic C++/CUDA decoder that needs no GPU to serve.

Full table, contamination audit and caveats: [`cushr_train/PAPER_COMPARISON.md`](cushr_train/PAPER_COMPARISON.md).

### Decoder throughput

Batching the k-best merge across sentences collapses **1,205,796 kernel
launches into 50**, raising occupancy from 4.7% to saturation. End-to-end decode
of the 119,503-sentence corpus: **455× faster** wall-clock, with **bit-identical
scores** verified against a C++ reference at every K.

The honest framing: the arithmetic did not get 455× faster. A launch-bound
baseline left the GPU idle ~99.8% of the time, and the win is removing that.
[`cushr_gpu/COMPARISON.md`](cushr_gpu/COMPARISON.md).

---

## Four things this project found

### 1. Half the corpus was unusable, and it was a wiring bug

Only **49.45%** of sentences had a resolvable gold path. Not missing
annotation — every sentence has one, and the DCS annotation set is a 3.7×
superset of the lattices. SHR attaches auxiliary analyses (the verbal root of a
participle) through edge types the forced-DAG filter drops, stranding the exact
nodes DCS names. Four fixes took coverage to **96.61%**:

| | coverage |
|---|---:|
| baseline | 49.45% |
| orphan redirect | 74.99% |
| \+ edge reorientation | 93.89% |
| \+ `key=2` sandhi edges, lemma families | **96.61%** |

[`ingest/INGEST_METHODOLOGY.md`](ingest/INGEST_METHODOLOGY.md).

### 2. Sentence context beat model capacity by 170×

Two changes were tried against the same baseline:

| change | added params | Δ test F1 |
|---|---:|---:|
| remove the 156→96 projection bottleneck | +25K | **+0.0004** |
| character BiLSTM over the sentence | +619K | **+0.0695** |

More capacity over the same per-node features bought nothing. A character
encoder — which supplies the *sandhi-fused surface text*, information every
prior feature had discarded in favour of resolved forms — lifted perfect match
from 43.7% to 67.3%. [`cushr_train/CONTEXTUAL_ENCODER.md`](cushr_train/CONTEXTUAL_ENCODER.md).

### 3. A measured head-to-head against ByT5-Sanskrit

The released `chronbmm/sanskrit5-multitask` was run on the *same* 4,200
sentences and scored through the *same* reference and code path — not compared
against its published numbers on a different corpus.

| level | cuSHR | +maps | +maps +rerank | ORACLE | ByT5 |
|---|---:|---:|---:|---:|---:|
| S | **91.52** | 91.52 | **92.60** | 98.00 | 81.38 |
| L | 65.62 | 85.40 | **86.81** | 91.75 | **90.55** |
| S+M | 45.69 | 52.26 | **61.21** | 77.65 | **67.79** |
| S+L+M | 45.29 | 51.95 | **60.98** | 77.50 | **66.90** |

Two findings worth stating plainly:

- **cuSHR's S lead is not real.** ByT5's failures are DCS-vs-SIGHUM compound
  conventions, not mis-segmentation — only 0.2% over-generate. No SIGHUM-tuned
  checkpoint was ever released, so this is the closest available comparison.
- **On joint analysis cuSHR still loses, and it is not a scoring artefact.**
  Restricted to sentences where both segment identically, ByT5 reaches **82.21**
  at S+L+M. The gap has closed substantially — most of what once looked like a
  ceiling deficit was untranslated vocabulary — but ByT5 still leads by ~6
  points on the full set.

### 4. The joint-analysis gap was two independent problems

M sat at 45.69 against an oracle of 68.23. That turned out to be a *vocabulary*
failure and a *ranking* failure stacked on top of each other, and separating
them took M to **61.21**:

| fix | what it addresses | S+M |
|---|---|---:|
| baseline | | 45.69 |
| \+ convention tables | SHR and DCS name the same analysis differently | 52.26 |
| \+ BiLSTM reranker | the right analysis was in the lattice, ranked too low | **61.21** |

**They are independent, and the evidence is that the tables moved the floor and
the ceiling by the same factor** — %-of-ceiling stayed pinned at 67% before and
after, which is what proved the remaining deficit belonged to the scorer.

The reranker was gated on measurement rather than intuition. An exact K-best
decoder (verified: K=1 reproduces `viterbi()` on 5,681/5,681 sentences, and
top-K matches exhaustive path enumeration) showed **recall@64 = 76.52 against an
oracle of 77.65** — the correct analysis is nearly always *in* the beam, so this
was a ranking problem, not a coverage problem.

It also killed the obvious design before it was built: only **20.1%** of that
headroom is separable by role counts (nominatives, finite verbs), capping a
count-based reranker at 56.57. Reading each candidate as a *sequence* instead
reaches **61.21** — past the ceiling counts could ever have hit.

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

- **The lattice bounds accuracy.** cuSHR can only select analyses SHR proposed.
  Where DCS's analysis is unreachable, no scorer recovers it — the oracle
  ceiling is 98.0 for segmentation and 77.65 for joint analysis.
- **The reranker is bounded by the beam, not the oracle.** It can only choose
  among the 16 paths the base decoder produced, so its ceiling is recall@16 =
  73.67, not 77.65. Closing the rest means moving the structure into the scorer
  itself (second-order Viterbi), which would break `cushr_gpu`'s bit-exact
  decoder — deliberately not attempted.
- **Reranking is a second decode pass.** The headline 2.77M-parameter,
  no-pretraining figures describe the base model; `+rerank` adds 866K
  parameters and a K=16 decode, and the segmentation numbers in §1 are the base
  model's.
- **Lemma and tag errors are one error.** 93% of tag mistakes coincide with a
  lemma mistake, and all of those are DCS reading a word as a participle where
  the path chose the nominal reading. One node choice, two metric failures —
  which is why the convention table has to rewrite the *pair* and a lemma-only
  table could not move M at all.
- **Case syncretism is the residual.** 74.3% of the model's own node errors are
  same-form, same-lemma, wrong `cng`; the largest single confusion is
  nom.sg.n. vs acc.sg.n. (494 cases), which are *always* homographic in
  Sanskrit and half of which sit in sentences with no finite verb to
  disambiguate against.
- **Test sets overlap 97%, not 100%.** Our SIGHUM copy and the published split
  differ on 125 of 4,200 sentences.
- **Contamination is asymmetric.** cuSHR excludes all 4,200 benchmark sentences
  from training; ByT5 saw every one during pretraining, unavoidably.
- **Single seed throughout.** No variance estimates or significance tests.
