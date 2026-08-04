# Contextual encoding: `char_bilstm`

A character-level BiLSTM inserted between the node featurizer and the biaffine
scorer. Largest single improvement measured on this project: **perfect match
0.4369 → 0.6731** on the gold75 test split, a 54% relative gain.

- Corpus: `data/cushr_data_repaired.npz`, 89,611 resolved gold paths (75.0%),
  train 80,871 / dev 4,357 / test 4,383
- Baseline: `hybrid_tag`, 8 epochs, seed 0, same cache. Only `--encoder` differs.

---

## 1. The problem it addresses

`BiaffineEdgeScorer.edge_scores` (`model.py:36`) computes

```
score(u -> v) = src_proj(f)[u] · dst_proj(f)[v] + bias
```

so an edge's score is a function of exactly two node vectors. Viterbi recovers a
globally optimal *path*, but it can only combine the scores it is handed — it
cannot invent an interaction the scorer never expressed. If two segmentations
differ only in how they interact with a word ten positions away, the scorer
assigns them identical scores. The model was first-order over the lattice.

### Why the diagnosis was actionable, and how we knew

The preceding experiment is what made this a confident prediction rather than a
guess. `hybrid_tag_full` removed the 156→96 projection inside the featurizer,
roughly doubling scorer parameters:

| change | added params | Δ test F1 |
|---|---:|---:|
| `hybrid_tag_full` — remove the 156→96 bottleneck | +25K (scorer) | **+0.0004** |
| `char_bilstm` — sentence context | +619K (encoder) | **+0.0695** |

More capacity over the *same per-node features* bought nothing. That is evidence
the ceiling was missing information, not representational power — so the next
change had to introduce something the model could not previously see.

---

## 2. Why characters and not nodes

The obvious implementation is a BiLSTM over the node vectors. **Nodes are not a
sequence.** Measured over the corpus:

| | |
|---|---:|
| nodes per sentence | 37.6 mean, 400 max |
| gold words per sentence | 6.5 |
| **candidate analyses per real word** | **5.8×** |
| sentences whose nodes are ordered by `(chunk, position)` | **24.4%** |

A recurrence over `node_features` would read a positionally scrambled bag of
mutually exclusive hypotheses. Characters *are* a sequence — 41.4 per sentence
over a 53-symbol alphabet — and every node covers a span of them.

### Context supplies information, not just structure

Every pre-existing feature derives from a node's **resolved** form. The written
text is sandhi-fused and differs. Checking each node's span against its surface
form through `collate`:

```
span == surface form:            70.6%
span length != node_word_length:  0 of 10,379   (alignment is exact)
```

The 29.4% that differ are same-length orthographic variants:

```
start=34 end=40   span='SAstfa'   node word='SAstra'
start=14 end=19   span='daRqe'    node word='daRqa'
start=18 end=21   span='ena'      node word='ina'
```

The model had never seen the characters actually written — only the citation
forms recovered from them. Since resolving sandhi *is* the task, this is the
most relevant information available.

---

## 3. Architecture

```
chars [B, L]  -> Embedding(257, 32, padding_idx=0)
              -> BiLSTM(2 layers, hidden 128, bidirectional)   -> [B, L, 256]
node i (sentence b, span [s, e))
              -> concat(H[b, s], H[b, e-1])                    -> [n, 512]
              -> Linear(512, 96)                               -> ctx [n, 96]
node vector    = concat(featurizer_out [96], ctx [96])         -> [n, 192]
              -> BiaffineEdgeScorer(192, 128)
```

Endpoint readout rather than pooling: in a bidirectional encoder the forward
state at the span's end already summarises everything to its left and the
backward state at its start everything to its right, so the pair carries the
span's content *and* its two-sided context.

| | params |
|---|---:|
| encoder (`char_bilstm`) | 618,624 |
| featurizer embeddings (`hybrid_tag`) | 1,638,876 |
| **total model** | **2,321,725** |

**The decoder is unaffected.** Context depends on the sentence, never on the
path chosen through it, so each node still has exactly one vector at inference.
`context.materialize_contextual` freezes them into the same dense `[N, 192]`
array the precomputed path produces, so `export_weights.py`, the `.bin` header
and the C++/CUDA decoders are untouched — only `feat_dim` changes. Verified:
`model75_ctx.npz["src_proj"].shape == (128, 192)`, name
`ngrams80+hybrid_tag+char_bilstm`.

---

## 4. Results (8 epochs, seed 0)

| test subset | F1 | PM | baseline F1 | baseline PM | ΔF1 | ΔPM |
|---|---:|---:|---:|---:|---:|---:|
| **all** (n=4,383) | **0.9282** | **0.6731** | 0.8587 | 0.4369 | **+0.0695** | **+0.2362** |
| pre-repair (n=2,885) | 0.9365 | 0.7300 | 0.8827 | 0.5324 | +0.0538 | +0.1976 |
| recovered (n=1,498) | 0.9144 | 0.5634 | 0.8185 | 0.2530 | +0.0959 | **+0.3104** |

Full metrics on the whole split: P 0.9289, R 0.9275.

### Dev by epoch

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| dev F1 | 0.8755 | 0.9015 | 0.9133 | 0.9225 | 0.9257 | 0.9277 | 0.9298 | 0.9302 |
| dev PM | 0.4919 | 0.5731 | 0.6123 | 0.6431 | 0.6564 | 0.6610 | 0.6711 | 0.6725 |
| train loss | 2.6479 | 1.9435 | 1.6589 | 1.4778 | 1.3568 | 1.2557 | 1.1764 | 1.0986 |
| active frac | 0.878 | 0.815 | 0.772 | 0.739 | 0.715 | 0.692 | 0.675 | 0.656 |

**Epoch 1 alone (dev F1 0.8755, PM 0.4919) already beat the baseline's best of
eight epochs** (0.8591 / 0.4306). Dev and test agree closely (0.9302/0.6725 vs
0.9282/0.6731), so this is not a dev-set artifact.

`active` — the fraction of sentences still producing a margin violation — falls
from 0.878 to 0.656, so by the end a third of the training set decodes correctly
with margin to spare.

---

## 4a. 16 epochs buys nothing — the budget is 8

Rerun identical except `--epochs 16`. Both rows report their **best-dev
checkpoint**, since `train.py` restores `best_state` before the test pass, so
this compares epoch 10 against epoch 8 rather than epoch 16 against epoch 8.

| | 8 epochs | 16 epochs | Δ |
|---|---:|---:|---:|
| test F1 | 0.9282 | 0.9287 | +0.0005 |
| test PM | **0.6731** | 0.6724 | −0.0007 |
| best dev F1 | 0.9302 (ep 8) | 0.9312 (ep 10) | |
| best dev PM | 0.6725 (ep 8) | 0.6801 (ep 10) | |

Both deltas are inside noise. This is a sharp break from the featurizer-only
runs, where 8→16 was worth +0.006 to +0.009 F1 and +0.02 to +0.03 PM
(`HYBRID_LONGER_TRAINING_gold75.md`).

### Dev by epoch, 16-epoch run

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev F1 | 0.8755 | 0.9015 | 0.9133 | 0.9225 | 0.9257 | 0.9277 | 0.9298 | 0.9302 | 0.9311 | **0.9312** | 0.9286 | 0.9283 | 0.9297 | 0.9295 | 0.9282 | 0.9284 |
| dev PM | 0.4919 | 0.5731 | 0.6123 | 0.6431 | 0.6564 | 0.6610 | 0.6711 | 0.6725 | 0.6787 | **0.6801** | 0.6693 | 0.6661 | 0.6791 | 0.6725 | 0.6651 | 0.6677 |
| train loss | 2.6479 | 1.9435 | 1.6589 | 1.4778 | 1.3568 | 1.2557 | 1.1764 | 1.0986 | 1.0276 | 0.9646 | 0.9052 | 0.8562 | 0.8099 | 0.7683 | 0.7326 | 0.7027 |
| active | 0.878 | 0.815 | 0.772 | 0.739 | 0.715 | 0.692 | 0.675 | 0.656 | 0.633 | 0.618 | 0.599 | 0.583 | 0.565 | 0.550 | 0.534 | 0.527 |

**Peak at epoch 10, then six epochs of decline while train loss falls
monotonically** (1.0986 → 0.7027). Textbook overfitting, and plausible given the
encoder's 619K extra parameters — every earlier model on this project had far
less capacity to memorise with, which is why none of them ever turned over.

Epochs 1–8 reproduce the 8-epoch run **exactly** (identical loss, `active`, dev
F1 and dev PM at every epoch), which incidentally confirms the seeding and the
`--resume` RNG round-trip are deterministic.

**Practical conclusion: train this architecture for 8 epochs.** 96.5 min for 16
epochs on CPU buys a checkpoint no better than the 47-minute one. If more is
wanted from it, the levers are regularisation — `nn.LSTM(dropout=...)` is
currently **0.0**, and `--ctx-dim`/`--ctx-hidden` are untuned — or a
learning-rate schedule. Not more epochs.

---

### The recovered sentences gain most

The largest single effect is **+0.3104 perfect match on the recovered subset**,
25.3% → 56.3%. Those are the sentences the orphan-gold repair rescued
(`ingest/INGEST_METHODOLOGY.md`), whose gold paths run through analyses SHR files
as auxiliary — the sandhi-heavy cases. The two pieces of work compound: the
repair supplied the labels, the encoder supplied the information needed to use
them. Before this change the recovered sentences looked like a drag on the
averages; they were simply the ones most starved of context.

---

## 5. Caveats

- **One seed.** No variance estimate. Given effect sizes of +0.07 F1 and +0.24
  PM this is unlikely to be noise, but it is unmeasured.
- **~~Still improving at epoch 8.~~** Resolved by §4a: the 16-epoch rerun peaks
  at epoch 10 and then overfits, and its best checkpoint tests no better than the
  8-epoch one (+0.0005 F1, −0.0007 PM). 8 epochs is the right budget.
- **No capacity control.** The encoder adds 619K parameters, so strictly the gain
  is confounded with model size. The `hybrid_tag_full` null result (+0.0004 for
  +25K scorer params over the same features) argues strongly against capacity as
  the explanation, but a same-size non-contextual control was not run.
- **Not compared to Sanskrit segmentation SOTA.** These numbers are internal to
  this pipeline and its splits.

---

## Reproducing

```bash
cd cushr_train
# the cache must carry surface_text / surface_text_offsets / node_char_start;
# prepare.py picks them up automatically from a build_features.py archive
python build_features.py --featurizer ngrams80 --raw ../data/cushr_data_repaired.npz \
    --out ../data/g75_ngrams80.npz --vocab-dir ../data --emit-ids --min-count 3
python prepare.py --npz ../data/g75_ngrams80.npz --cache ./cache75_ngrams80 --force
rm ../data/g75_ngrams80.npz

python train.py --cache ./cache75_ngrams80 --learned hybrid_tag \
    --encoder char_bilstm --word-dropout 0.1 --epochs 8 --seed 0 --resume \
    --out model75_ctx.npz --log log75_ctx.json --materialize ../data/g75_ctx.npz
python prepare.py --npz ../data/g75_ctx.npz --cache ./cache75_ctx --force
rm ../data/g75_ctx.npz
python eval_gold_subset.py --cache ./cache75_ctx --model model75_ctx.npz
```

~6 min/epoch on CPU (440 s first epoch, ~355 s after); 47 min for 8 epochs, 96.5
for 16. `--resume` restarts from `<--out>.ckpt`, written after every epoch.

The 16-epoch variant in §4a is the same command with `--epochs 16 --out
model75_16_ctx.npz --log log75_16_ctx.json`; it is documented as a negative
result and is not the recommended configuration.

### Invariants worth re-checking after any change here

1. **Span alignment** — `span_end - span_start == node_word_length` for every
   node with `char_ok`; content matches the surface form ~70% of the time, the
   rest same-length sandhi variants.
2. **Padding invariance** — a short sentence must get identical context vectors
   alone and when batched with a long one (measured max abs diff 1.9e-08). This
   is the classic BiLSTM padding bug and it fails silently; `pack_padded_sequence`
   is what prevents it.
3. **Boundary invariant** — context rows for `~char_ok` nodes are exactly zero,
   matching every other featurizer.
