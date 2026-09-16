# Comparing the segmentation models

Three cuSHR models and TransLIST, scored on the same 4,200 SIGHUM sentences
through the same script (`eval_surface.py`) and the same reference.

- **cuSHR-node** — `model95_ctx_ex4200`, the original model, trained to pick the
  exact gold lattice node (form + lemma + morph tag).
- **cuSHR-seg** — `model95_seg_ex4200`, identical architecture, trained to pick
  the right *segmentation* and to ignore lemma and tag.
- **cuSHR-seg + reranker** — cuSHR-seg plus `segrr_k16_featseq_loss_s{0,1,2}.pt`,
  a second-pass model that rescores the top 16 distinct segmentations.
- **TransLIST** — Sandhan et al., Findings of EMNLP 2022, as published.

Everything here is segmentation: the output word string, no lemma, no tag. For
lemma and morphology see `PAPER_COMPARISON.md` §3 — and §6 below, because
cuSHR-seg is *worse* at those by construction.

## 1. Results

Word P/R/F are MACRO (per sentence, then averaged), which is TransLIST's
protocol. PM = perfect match of the whole word sequence.

| model | PM | word P | word R | word F |
|---|---:|---:|---:|---:|
| cuSHR-node | 91.52 | 98.36 | 98.33 | 98.32 |
| cuSHR-seg | 92.76 | 98.47 | 98.42 | 98.42 |
| cuSHR-seg + reranker (3-seed mean) | **93.43 ± 0.13** | 98.76 | 98.74 | **98.73** |
| TransLIST | 93.97 | 98.80 | 98.93 | 98.86 |

Per seed of the reranker (the base model is one seed throughout):

| reranker seed | PM | word F | best epoch | predictions changed |
|---|---:|---:|---:|---:|
| 0 | 93.55 | 98.72 | 7 | 90 / 4,200 |
| 1 | 93.45 | 98.77 | 8 | 94 / 4,200 |
| 2 | 93.29 | 98.70 | 3 | 64 / 4,200 |

Net effect: **+1.91 PM over cuSHR-node**, and the gap to TransLIST falls from
2.45 to **0.54** points (~23 sentences).

## 2. What the training data is

All three cuSHR models see exactly the same data.

| | cuSHR (all three) | TransLIST |
|---|---|---|
| corpus | SIGHUM, ingested by `ingest/parallel_ingest.py` into `cushr_data_g95.npz` (119,503 sentences, 96.61% with a resolved gold path) | SIGHUM |
| train / dev / test | 104,159 / 1,636 / 5,681 sentences (`splits95_ex4200.json`) | 99,260 / 4,322 / 4,200 (the split shipped in `new_LREC_data_complete.csv`) |
| benchmark held out | yes — all 4,200 benchmark sentences excluded from train and dev | no (the 4,200 are its official test split) |
| pretraining | none | none |
| lexicon | SHR lattice (candidate words + edges) | SHR lattice, or n-grams (a separate variant) |

**The test sets are identical.** `sighum_test_4200.tsv` and the `split=test`
rows of `LREC-Data/new_LREC_data_complete.csv` — the file TransLIST's
`sighum-ngram` / `sighum-shr` settings read
(`fastnlp-copy/core/dataset.py:798`, `constrained_inference.py:36`) — are the
same 4,200 DCS-IDs, with identical input sentences and identical gold
segmentations after normalising the word separator. Verify with
`python check_testset_overlap.py`.

> Earlier versions of this document, `PAPER_COMPARISON.md`,
> `RESULTS_MATRIX.md` and `WEEK11_EVALUATION_MATRIX.md` all said the two sets
> overlap by 97.02% (4,075 / 4,200) and caveated every TransLIST comparison
> accordingly. **That was wrong** — no code ever computed it, and measured
> against the file TransLIST actually loads the overlap is 4,200 / 4,200. The
> 97.02% most likely came from string-matching a differently transliterated or
> differently sourced copy, where 125 sentences failed to match as strings while
> being the same sentences. The consequence is that the gap to TransLIST is a
> difference between the systems, not between the data.

## 3. cuSHR architecture and parameters

cuSHR-node and cuSHR-seg are **the same 2,789,109-parameter network**, trained
with different objectives. Counted from the saved weights:

| block | params | what it is |
|---|---:|---|
| form embedding | 1,804,480 | 56,390 × 32 |
| lemma embedding | 276,304 | 17,269 × 16 |
| preverb embedding | 5,124 | 1,281 × 4 |
| morph-tag embedding | 20,352 | 848 × 24 |
| projection 156 → 96 | 15,072 | node vector |
| char BiLSTM encoder | 618,624 | char emb 32, 2 layers, hidden 128/direction, out 96 |
| biaffine edge scorer | 49,153 | 192-dim node vectors, hidden 128 |
| **total** | **2,789,109** | |

Node features are `ngrams80` (80 scalar columns) plus the learned identity
embeddings; the node vector the scorer sees is 192-dim (96 featurizer + 96
context). Decoding is Viterbi over the lattice DAG: the path score is the plain
sum of edge scores, with no learned transition parameters.

### Training configuration (identical except the objective)

| | cuSHR-node | cuSHR-seg |
|---|---|---|
| featurizer / encoder | `hybrid_tag` / `char_bilstm` | same |
| epochs, batch, lr | 8, 64, 1e-3 | same |
| weight decay, margin | 1e-5, 1.0 | same |
| word dropout | 0.1 | same |
| seed, device | 0, CPU | same |
| **cost** | **`node`** — cost 1 for any predicted node that is not the exact gold node | **`surface`** — cost 1 only for a wrong segmentation word or a jump that skips a gold word; a right word with the wrong lemma/tag costs `--morph-cost` (0.0) |
| **gold target** | **`exact`** — hinge against the gold node path | **`surface`** — hinge against the best path spelling the gold segmentation |
| best epoch chosen by | dev node F1 | dev surface PM |
| training time | ~70 min | ~70 min |

Both use the same structured hinge, `relu(cost(pred) + score(pred) - score(gold))`,
implemented in `losses.py`; the flags select which cost and which gold target.

**Why the objectives differ in practice.** A Sanskrit surface form is
morphologically ambiguous, so the lattice holds on average ~1 extra node per
gold word that spells the same word with a different lemma or tag (measured:
2,272 such nodes against 2,251 gold nodes on 256 training sentences). Under the
`node` cost those are penalised exactly as hard as a wrong split, so most of the
scorer's capacity goes to decisions that do not change the segmentation.

**One implementation note that cost a training run.** The `surface` gold target
must require the gold path to visit *every* gold word in order. A first version
only required each node to spell *some* gold word; because the lattice has edges
that jump past a word, 61% of those "gold" paths were shorter than the real one
(313 of 512 sentences), the model learned that skipping words was free, and the
run scored P 0.75 / R 0.23. `losses.py` now ranks the gold words and admits an
edge only between consecutive ranks.

## 4. The reranker

`rerank.py`, applied to the top **16 distinct segmentations** decoded from the
top 64 paths (`--dedup-surface`: without it the beam fills with morphological
variants of one split). 865,844 parameters:

| block | params |
|---|---:|
| lemma embedding (24,120 × 32) | 771,840 |
| morph-tag embedding (848 × 32) | 27,136 |
| BiLSTM (64 in, 64 hidden, bidirectional) | 66,560 |
| output projection + scalars | 131 |
| segmentation-feature head (8 → 16 → 1) | 161 |
| standardisation buffers | 16 |

Final score = `w_base * base_viterbi_score + BiLSTM_scalar + feature_scalar + b`,
with `w_base = 1` and both scalar branches zero-initialised, so **at epoch 0 the
reranker is exactly the base decoder** (asserted in the run).

The eight segmentation features (`seg_features.py`) carry no word identity:
word count; sum / mean / min of log(1 + count of each word as a gold word in
training); counts of unseen, rare (<3) and short (≤2 char) words; and the
candidate's base score relative to the beam's best. The gold-word counts come
from the training candidate lists (32,201 forms), with each training sentence's
own gold words subtracted before featurising it (leave-one-out).

| | |
|---|---|
| training data | candidate lists for 30,000 randomly sampled train sentences; dev 1,636; test 5,681 |
| label | a candidate is correct if its segmentation equals the gold path's |
| loss | listwise: softmax over the candidates of one sentence, `-log Σ p(correct)` |
| epochs / patience | up to 20, early stop after 6 epochs with no dev gain |
| checkpoint chosen by | **dev loss** (`--select-by loss`) |
| seeds | 0, 1, 2 |
| training time | ~5 min/seed on CPU |

**Why dev loss and not dev top-1.** Dev has 1,636 sentences, so top-1 moves in
steps of 0.06 points and a lucky early epoch can win by one sentence — seed 2
kept an epoch-1 checkpoint through 6 epochs of patience and scored lowest of the
three. Dev loss falls smoothly for 3–8 epochs and then rises while train loss
keeps falling. Switching the criterion moved the 3-seed mean from 93.32 to 93.43
with no other change.

### Ablations (dev top-1 against the gold path; base decoder = 93.58)

| reranker variant | params | dev | test |
|---|---:|---:|---:|
| morph only | 77,443 | +0.00 | +0.02 |
| morph + lemma | 865,667 | +0.43 | +0.25 |
| morph + lemma + form embedding | 1,893,091 | +0.31 | +0.16 |
| segmentation features only | 163 | +0.43 | +0.09 |
| **features + morph + lemma** | 865,844 | **+0.80** | **+0.37** |

The identity tables memorise (train loss kept falling while dev flattened) and
the 163-parameter feature model saturates after one epoch; together they gain
about twice what either gains alone.

## 5. TransLIST

Parameters counted from the released checkpoint
`translist/V0/saved_models/best_hack_ngram2` (6.85M):

| block | params | what it learns |
|---|---:|---|
| lattice/word embedding (126,355 × 50) | 6,317,750 | a vector per SHR / n-gram candidate word |
| bigram embedding | 129,553 | char-bigram identity |
| char embedding | 53,632 | character identity |
| 4-position fusion + rel-pos projections | ~65K | how two spans sit relative to each other |
| 1 transformer layer, 8 heads × 20 = 160 hidden | ~180K | token ↔ candidate-word attention |
| output linear 160 → 163 + CRF transitions (163²) | ~47K | tag emission + label-transition scores |

Method, and how it differs from cuSHR:

- **Input is characters *and* candidate words in one flat sequence**, so every
  candidate can attend to every other candidate and to every character. cuSHR's
  scorer is first-order over lattice edges: it scores one (node, node) pair at a
  time and cannot express an interaction between two non-adjacent candidates.
- **Four-position encoding.** Each item carries a start and an end, and every
  pair yields four signed distances (ss, se, es, ee), each embedded and fused.
  cuSHR has no positional parameters at all; adjacency is implicit in the DAG.
- **CRF transitions** (163²) — learned label-to-label compatibility. cuSHR's
  Viterbi sums independent edge scores with zero learned transition parameters.
- **Constrained Inference / PRCP**, a post-hoc repair step: when the tagger's
  output is not a valid lattice path, it re-picks the best valid path by
  character logits divided by word count
  (`translist/V0/interactive_module.py:200`). cuSHR cannot produce an invalid
  path — it decodes inside the lattice — so it has no analogue and needs none.
- **Segmentation only.** TransLIST has no lemma or morphology parameters; ~92%
  of its 6.85M is the word table, and its reasoning machinery is ~290K.
- **Variants.** The `SHR` variant (lattice from the Heritage Reader) is the
  93.97 row. The `ngrams` variant scores 79.28 PM and is a different setting.

## 6. The tradeoff: cuSHR-seg is worse at morphology

Measured on cuSHR's own 5,681-sentence test split, exact-node match (form +
lemma + tag, all correct):

| | node F1 | node PM |
|---|---:|---:|
| cuSHR-node | 0.923 | 0.644 |
| cuSHR-seg | 0.788 | 0.316 |

This is the objective working as intended, not a regression: cuSHR-seg is not
penalised for choosing the wrong lemma or tag, so it does not learn them.

**Keep both models.** Use cuSHR-seg (+ reranker) for segmentation, and
cuSHR-node for the S / L / M ladder in `PAPER_COMPARISON.md`. The old S+M
reranker belongs to cuSHR-node; the segmentation reranker here belongs to
cuSHR-seg.

## 7. Ceilings — what is still reachable

Surface recall@k for cuSHR-seg over *distinct segmentations*, against the SIGHUM
reference: is the correct segmentation anywhere in the top k?

| @1 | @2 | @4 | @8 | @16 | lattice ORACLE |
|---:|---:|---:|---:|---:|---:|
| 92.76 | 96.93 | 98.14 | 98.50 | 98.55 | 98.00 |

The reranker reaches 93.43 of a 98.55 ceiling, so **the beam almost always
contains the right answer and the remaining failure is ranking, not coverage**.
4 of the 4,200 sentences have no gold path in the lattice at all and score 0 by
construction.

## 8. Caveats

1. **One base-model seed.** cuSHR-node and cuSHR-seg are single runs (seed 0);
   only the reranker has a variance estimate (±0.13 over 3 seeds).
2. **The TransLIST row is reported, not reproduced.** Its 93.97 comes from the
   paper's Table 1, not from running their checkpoint through `eval_surface.py`.
   The test set is identical (§2), so the 0.54-point gap is a system
   difference — but it is a gap against a published number, and the two were
   produced by different scoring code. Running their released
   `best_sighum_shr2` through our scorer would remove that last asymmetry.
3. **Model selection used dev only.** Epochs were chosen on dev loss / dev
   surface PM, never on the SIGHUM result.
4. **The 91.52 and 92.76 rows have no saved JSON** — they were read from console
   output. The reranker rows are in `segrr_eval_loss_s{0,1,2}.json`.

## 9. Reproducing

```bash
# cuSHR-seg: same command as cuSHR-node plus the four objective flags
python train.py --cache ./cache95_ngrams80 --learned hybrid_tag \
    --encoder char_bilstm --word-dropout 0.1 --epochs 8 --seed 0 \
    --splits-override splits95_ex4200.json \
    --cost surface --gold-target surface --select-by surface_pm \
    --surface-raw ../data/cushr_data_g95.npz \
    --out model95_seg_ex4200.npz --log log95_seg_ex4200.json \
    --materialize ../data/g95_seg_ex4200.npz
python prepare.py --npz ../data/g95_seg_ex4200.npz --cache ./cache95_seg_ex4200 --force

# candidate lists (30k train sentences), then the reranker, per seed
python make_rerank_data.py --cache ./cache95_seg_ex4200 \
    --model model95_seg_ex4200.npz --splits-override splits95_ex4200.json \
    --label surface --dedup-surface --k 16 --k-decode 64 --limit 30000 \
    --out-prefix segrr_k16
python rerank.py --train segrr_k16_train.npz --dev segrr_k16_dev.npz \
    --test segrr_k16_test.npz --seg-features --epochs 20 --patience 6 \
    --select-by loss --seed 0 --out segrr_k16_featseq_loss_s0.pt

# the table above
python eval_surface.py --cache ./cache95_seg_ex4200 \
    --model model95_seg_ex4200.npz --index ../data/sentence_index_g95.json \
    --raw ../data/cushr_data_g95.npz                      # 92.76, no reranker
python eval_surface.py --cache ./cache95_seg_ex4200 \
    --model model95_seg_ex4200.npz --index ../data/sentence_index_g95.json \
    --raw ../data/cushr_data_g95.npz --kbest 16 --k-decode 64 --dedup-surface \
    --rerank segrr_k16_featseq_loss_s0.pt                 # 93.55
```

All flags added for this work are opt-in: omitting them reproduces cuSHR-node
and the original reranker exactly (`losses.py` defaults `--cost node
--gold-target exact`; `make_rerank_data.py` defaults `--label node` with no
dedup; `rerank.py` defaults `--select-by top1` with no features).

## 10. Built but not yet trained: lattice attention

The reranker's ceiling is recall@16 = 98.55, and it can only reorder what the
first-order scorer produced. Closing the last ~0.5 needs the scorer itself to see
more than two nodes at a time — TransLIST's actual advantage, once its 6.3M-row
word table is set aside.

`context.py` now carries two more encoders:

- **`lattice_attn`** — global self-attention over the candidate nodes of one
  sentence: every candidate attends to every other candidate.
- **`lattice_attn_char`** — the same, with the proven char-BiLSTM span vector as
  an additional input. ~306K new parameters (3,095,541 total vs 2,789,109).

Attention logits carry a learned bias built from the four signed distances
between two candidates' character spans (`ss`, `se`, `es`, `ee`), which is what
lets the model read *overlap* (mutually exclusive candidates), *precedence*, and
*shared head* (competing splits from one start point). One distance cannot
express that family; four can. The fusion matrix is folded into four per-head
tables, so nothing wider than `[group, nodes, nodes, heads]` is materialised.

**The decoder is untouched.** The encoder output is still a per-node vector that
depends on the sentence but never on the path, so `--materialize` freezes it and
`cushr_gpu/` consumes it unchanged.

```bash
python test_lattice_attn.py --cache ./cache95_ngrams80            # invariants
python test_lattice_attn.py --cache ./cache95_ngrams80 --double   # rounding vs bug
sbatch --export=ALL,STAGE=preflight train_attn.slurm              # then STAGE=train, STAGE=eval
```

`test_lattice_attn.py` is the regression suite, and it matters because every
failure mode here is silent — the loss still falls and the score is merely worse.
It checks padding invariance, chunk invariance, no cross-sentence leakage,
permutation equivariance, zeroed boundary rows, finiteness, that the attention is
not inert, that the span bias changes the output, that `char_bilstm` is
bit-identical under the widened signature, and that `materialize_contextual`
reproduces the training path. The invariances are exact in arithmetic but not
bitwise in float32 (~1e-6); `--double` drops them to ~1.7e-15, which is how a
rounding residual is told apart from a real bug.

**No accuracy numbers yet** — this machine is CPU-only (`torch 2.13.0+cpu`), so
training runs on the cluster. The gate is a base decoder above 92.76; the
most informative ablation is `--no-span-bias`, which separates "attention helps"
from "span geometry helps".

## Sources

- Sandhan, Singha, Rao, Samanta, Behera, Goyal. *TransLIST: A Transformer-Based
  Linguistically Informed Sanskrit Tokenizer.* Findings of EMNLP 2022,
  arXiv:2210.11753, Table 1.
- `PAPER_COMPARISON.md` — the S / L / M ladder, the ORACLE ceilings, and the
  ByT5-Sanskrit head-to-head.
