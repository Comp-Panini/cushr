# 93.9%-gold corpus: does the edge-orientation fix help downstream?

`ingest/INGEST_METHODOLOGY.md` §4e raised gold-path coverage from 74.99% to
93.89% by orienting SHR's symmetric `key=1` edges along sentence order rather
than node id. This file answers the only question that matters about that: are
the 22,616 new labels signal or noise?

- **gold49** — `data/new_cushr_data_fixed_USE_THIS.npz`, 59,092 paths (49.45%)
- **gold75** — `data/cushr_data_repaired.npz`, 89,611 paths (74.99%)
- **gold94** — `data/cushr_data_posorder.npz`, 112,200 paths (**93.89%**),
  train 101,223 / dev 5,447 / test 5,530

All numbers below are `hybrid_tag`, 8 epochs, seed 0, `--node-dim 96
--word-dropout 0.1` — identical hyperparameters to the two earlier reports.

---

## 1. Headline — DIFFERENT test sets, not comparable

| corpus | test F1 | test PM | n |
|---|---:|---:|---:|
| gold49 | 0.8831 | 0.5450 | 2,886 |
| gold75 | 0.8587 | 0.4369 | 4,383 |
| **gold94** | **0.8609** | **0.4492** | **5,530** |

Each column is measured on its own split, and every split is a superset of the
one above it. The added sentences are the ones that used to be unlabellable, so
they are harder on average and drag the mean down. **These deltas say nothing
about model quality.** §2 is the measurement that does.

## 2. Same-test-set: gold94 model on the gold49 subset

`eval_gold_subset.py` splits the gold94 test set into sentences that already had
a gold path in gold49 and the rest, then evaluates the gold94-trained model on
each. The first subset is the gold49 test split less one sentence — 2,885 of
2,886 — the missing one being among the four dropped by the orphan repair.

**On the same 2,885 sentences:**

| model | F1 | precision | recall | PM |
|---|---:|---:|---:|---:|
| gold49-trained (own split, n=2,886) | 0.8831 | 0.8863 | 0.8800 | 0.5450 |
| gold75-trained (pre-repair subset) | 0.8827 | 0.8862 | 0.8792 | 0.5324 |
| **gold94-trained (pre-repair subset)** | **0.8865** | **0.8908** | **0.8823** | **0.5529** |
| gold94 − gold49 | **+0.0034** | +0.0045 | +0.0023 | **+0.0079** |
| gold94 − gold75 | **+0.0038** | +0.0046 | +0.0031 | **+0.0205** |

**The added labels help.** On identical sentences the gold94 model beats both
predecessors. It is a small gain and a single seed, so the right reading is "no
harm, plausibly a modest gain" rather than a demonstrated improvement — but it is
in the opposite direction from the risk §5 was written to catch.

Worth noting: gold75's one hint of a cost was PM on this subset falling 0.0126
against gold49, which that report flagged as worth rechecking on another seed.
gold94 recovers it and adds to it (+0.0079 over gold49). That is weak evidence
the earlier dip was noise, not a real cost.

## 3. Same-test-set: isolating the reorientation increment

Same model, split instead by what gold75 could already label:

| subset | F1 | PM | n |
|---|---:|---:|---:|
| all | 0.8609 | 0.4492 | 5,530 |
| already in gold75 | 0.8621 | 0.4511 | 4,383 |
| **recovered by reorientation** | **0.8565** | **0.4420** | **1,147** |
| gap | −0.0056 | −0.0091 | |

Compare the equivalent split for the orphan repair (from `GOLD49_VS_GOLD75.md`):

| increment | its F1 | baseline F1 | gap |
|---|---:|---:|---:|
| orphan repair (gold75) | 0.8185 | 0.8827 | **−0.0642** |
| **reorientation (gold94)** | **0.8565** | 0.8621 | **−0.0056** |

**The two increments are not alike.** The orphan-repair sentences are markedly
harder than the corpus they joined; the reorientation sentences are nearly
indistinguishable from it. That matches what the two changes actually do. The
orphan redirect substitutes SHR's positioned analysis for the DCS one, genuinely
redefining the label (§5 of the methodology). Reorientation changes no label: all
20 paths it altered describe an identical `(chunk, position, surface)` sequence,
and its 22,616 new labels are ordinary gold paths that were simply unreachable
because their edges pointed the wrong way.

The gold94 model also scores 0.8621 on gold75's own test split, against the
gold75 model's 0.8587 — same sentences, +0.0034.

## 4. Caveats

- **Single seed (0), single featurizer (`hybrid_tag`), 8 epochs.** The other five
  featurizers and all 16-epoch runs have no gold94 number.
- **No gold49 or gold75 model was re-evaluated here.** Their columns are the
  published figures from the two earlier reports. The comparison assumes those
  runs are reproducible, which is reasonable — training is deterministic at fixed
  seed, confirmed incidentally when an interrupted gold94 run reproduced its
  epoch-1 dev F1 exactly on restart — but it was not re-verified.
- **The gains are small relative to seed noise** and should not be quoted as a
  demonstrated improvement without a multi-seed run. The defensible claim is that
  **93.9% coverage costs nothing on the original distribution**, which is what §5
  of the methodology asked.
- **27 sentences were dropped** relative to gold75. All 27 are DCS annotations
  that do not cover their sentence; see §4e of the methodology. They are absent
  from the gold94 test split, so they affect no number here.

## 5. Reproduce

```bash
cd ingest
python parallel_ingest.py --shards 24 --workers 4 --resume --keep-shards \
    --repair-orphan-gold --edge-order position \
    --out ../data/cushr_data_posorder.npz --index ../data/sentence_index_posorder.json

cd ../cushr_train
python build_features.py --featurizer ngrams80 --raw ../data/cushr_data_posorder.npz \
    --out ../data/g94_ngrams80.npz --vocab-dir ../data --emit-ids --min-count 3
python prepare.py --npz ../data/g94_ngrams80.npz --cache ./cache94_ngrams80 --force
python train.py --cache ./cache94_ngrams80 --learned hybrid_tag --node-dim 96 \
    --word-dropout 0.1 --epochs 8 --out model94_hybrid_tag.npz \
    --log log94_hybrid_tag.json --materialize ../data/g94_hybrid_tag.npz
python prepare.py --npz ../data/g94_hybrid_tag.npz --cache ./cache94_hybrid_tag --force

python eval_gold_subset.py --cache ./cache94_hybrid_tag --model model94_hybrid_tag.npz \
    --raw ../data/cushr_data_posorder.npz --old-npz ../data/new_cushr_data_fixed_USE_THIS.npz
python eval_gold_subset.py --cache ./cache94_hybrid_tag --model model94_hybrid_tag.npz \
    --raw ../data/cushr_data_posorder.npz --old-npz ../data/cushr_data_repaired.npz
```

`--materialize` is required: without it the learned embedding tables are never
persisted, `model*.npz` holds only the scorer, and `eval_gold_subset.py` fails
with a shape mismatch (96-dim learned features vs the cache's 80 raw ones).
