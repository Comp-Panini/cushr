# Featurizer comparison (75%-gold corpus): `morph43` → `scalars64` → `ngrams80` → `hybrid` → `hybrid_tag`

- Corpus: all 119,503 SIGHUM sentences (4,488,155 lattice nodes)
- Source archive: `data/cushr_data_repaired.npz` (orphan-gold repair; see
  `ingest/INGEST_METHODOLOGY.md`)
- Sentences with a resolved gold path: **89,611 (75.0%)** → train 80,871 / dev
  4,357 / test 4,383
- Identical training for all six: 8 epochs, batch 64, AdamW lr 1e-3, structured hinge

> **Do not compare the headline F1 below against
> `FEATURIZER_COMPARISON_gold49.md` directly.** That report's test split is a
> *strict subset* of this one — the repair only ever adds resolved sentences —
> so the two evaluate different sentence populations. Every number here is lower
> than its 49% counterpart, and §"Is the model actually worse?" shows that is
> entirely test-set composition, not model quality.

Word-level top-1 accuracy on the **test** split, decoded with Viterbi over the learned biaffine edge scores.


## Headline

| featurizer | F1 | precision | recall | perfect match | sentences | params |
|---|---|---|---|---|---|---|
| `morph43` | **0.7526** | 0.7854 | 0.7225 | 0.2136 | 4,383 | 16,385 |
| `scalars64` | **0.8181** | 0.8239 | 0.8122 | 0.3447 | 4,383 | 16,385 |
| `ngrams80` | **0.8217** | 0.8269 | 0.8167 | 0.3543 | 4,383 | 20,481 |
| `hybrid` | **0.8486** | 0.8533 | 0.8439 | 0.4091 | 4,383 | 24,577 |
| `hybrid_tag_only` | **0.8585** | 0.8626 | 0.8544 | 0.4337 | 4,383 | 24,577 |
| `hybrid_tag` | **0.8587** | 0.8623 | 0.8552 | 0.4369 | 4,383 | 24,577 |

## Recall by training frequency of the gold word

Each gold word is bucketed by how often its surface form occurs in the training split. This isolates whether an advantage comes from the frequency feature or from the rest of the vector.

| featurizer | unseen (n=886) | rare (n=1,236) | mid (n=5,715) | common (n=20,886) |
|---|---|---|---|---|
| `morph43` | 0.6828 | 0.8293 | 0.7946 | 0.6981 |
| `scalars64` | 0.7314 | 0.8455 | 0.8194 | 0.8117 |
| `ngrams80` | 0.7348 | 0.8439 | 0.8170 | 0.8184 |
| `hybrid` | 0.7302 | 0.8600 | 0.8318 | 0.8511 |
| `hybrid_tag_only` | 0.7427 | 0.8649 | 0.8469 | 0.8606 |
| `hybrid_tag` | 0.7506 | 0.8568 | 0.8481 | 0.8614 |

Delta vs `morph43`:

| featurizer | unseen | rare | mid | common |
|---|---|---|---|---|
| `scalars64` | +0.0485 | +0.0162 | +0.0248 | +0.1137 |
| `ngrams80` | +0.0519 | +0.0146 | +0.0224 | +0.1204 |
| `hybrid` | +0.0474 | +0.0307 | +0.0373 | +0.1530 |
| `hybrid_tag_only` | +0.0598 | +0.0356 | +0.0523 | +0.1625 |
| `hybrid_tag` | +0.0677 | +0.0275 | +0.0535 | +0.1634 |

Step-to-step delta (each rung vs the one before):

| step | unseen | rare | mid | common |
|---|---|---|---|---|
| `morph43` -> `scalars64` | +0.0485 | +0.0162 | +0.0248 | +0.1137 |
| `scalars64` -> `ngrams80` | +0.0034 | -0.0016 | -0.0024 | +0.0067 |
| `ngrams80` -> `hybrid` | -0.0045 | +0.0162 | +0.0149 | +0.0327 |
| `hybrid` -> `hybrid_tag_only` | +0.0124 | +0.0049 | +0.0150 | +0.0095 |
| `hybrid_tag_only` -> `hybrid_tag` | +0.0079 | -0.0081 | +0.0012 | +0.0009 |

## Error counts

| featurizer | TP | FP | FN |
|---|---|---|---|
| `morph43` | 20,751 | 5,671 | 7,972 |
| `scalars64` | 23,330 | 4,985 | 5,393 |
| `ngrams80` | 23,457 | 4,912 | 5,266 |
| `hybrid` | 24,240 | 4,169 | 4,483 |
| `hybrid_tag_only` | 24,541 | 3,908 | 4,182 |
| `hybrid_tag` | 24,563 | 3,924 | 4,160 |

## Training behaviour

Dev F1 by epoch:

| featurizer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `morph43` | 0.7305 | 0.7315 | 0.7455 | 0.7489 | 0.7361 | 0.7401 | 0.7486 | 0.7442 |
| `scalars64` | 0.8022 | 0.8050 | 0.8138 | 0.8146 | 0.8067 | 0.8107 | 0.8140 | 0.8156 |
| `ngrams80` | 0.8041 | 0.8126 | 0.8147 | 0.8171 | 0.8136 | 0.8143 | 0.8159 | 0.8182 |
| `hybrid` | 0.8119 | 0.8266 | 0.8390 | 0.8432 | 0.8430 | 0.8450 | 0.8495 | 0.8504 |
| `hybrid_tag_only` | 0.8153 | 0.8366 | 0.8426 | 0.8493 | 0.8502 | 0.8556 | 0.8563 | 0.8583 |
| `hybrid_tag` | 0.8174 | 0.8376 | 0.8468 | 0.8524 | 0.8525 | 0.8566 | 0.8591 | 0.8583 |

Final training loss and wall-clock (CPU, 8 epochs):

| featurizer | final train loss | total min |
|---|---|---|
| `morph43` | 3.7451 | 6.1 |
| `scalars64` | 2.9602 | 6.5 |
| `ngrams80` | 2.9233 | 7.4 |
| `hybrid` | 2.5227 | 7.7 |
| `hybrid_tag_only` | 2.3449 | 7.3 |
| `hybrid_tag` | 2.3096 | 9.6 |

The three learned rungs are still climbing at epoch 8, exactly as on the 49%
corpus; `HYBRID_LONGER_TRAINING_gold75.md` reruns them at 16 epochs. Those
numbers are not interchangeable with the tables here, which share an 8-epoch
budget.


## Is the model actually worse? No — the test set is harder

Every headline number here is below its `FEATURIZER_COMPARISON_gold49.md`
counterpart. That is a composition effect, and it is measurable rather than a
matter of opinion, because the old test split is a strict subset of the new one.

`eval_gold_subset.py` splits this test set into the sentences that already had a
gold path before the orphan repair and the ones the repair recovered, and
evaluates the *same* 75%-trained `hybrid_tag` model on each:

| test subset | F1 | precision | recall | perfect match | n |
|---|---|---|---|---|---|
| all (this report's headline) | 0.8587 | 0.8623 | 0.8552 | 0.4369 | 4,383 |
| **pre-repair sentences** | **0.8827** | 0.8862 | 0.8792 | 0.5324 | 2,885 |
| recovered sentences | 0.8185 | 0.8221 | 0.8150 | 0.2530 | 1,498 |

The 49%-trained `hybrid_tag` scored **0.8831** on its test split of 2,886
sentences. This model scores **0.8827** on the 2,885-sentence pre-repair subset —
a difference of **−0.0004**, which is noise.

Three conclusions follow:

1. **Training on 52% more sentences neither helped nor hurt on the original
   distribution.** The extra data is not corrupting the model, which was the open
   risk flagged in `INGEST_METHODOLOGY.md` §5. That risk is now closed.
2. **The recovered sentences are genuinely harder** — 0.8185 F1 and, far more
   starkly, 0.2530 perfect match against 0.5324. They are the sentences whose
   gold path runs through analyses SHR files as auxiliary, so this is expected
   rather than alarming.
3. **The headline drop is entirely reweighting.** A third of the new test set is
   drawn from that harder population.

The absence of a *gain* is worth stating plainly: 52% more training data bought
no measurable accuracy on the original distribution. What it bought is coverage —
the model is now trained and evaluated on a population 1.5× larger, including
sentence types it previously never saw supervised.

Caveat: `eval_gold_subset.py` was run for `hybrid_tag` only. The other five rungs
are assumed to behave the same way, which is untested.


## Reproducing

```bash
# 1. ingest with the orphan-gold repair (see ingest/INGEST_METHODOLOGY.md)
cd ingest
python parallel_ingest.py --shards 24 --workers 4 --keep-shards --resume \
    --repair-orphan-gold --out ../data/cushr_data_repaired.npz \
    --index ../data/sentence_index_repaired.json

cd ../cushr_train
# 2. precomputed featurizers.  --vocab-dir is required: ingest writes the
#    vocabularies beside its --out, i.e. into data/, not into ingest/.
for F in morph43 scalars64; do
  python build_features.py --featurizer $F --raw ../data/cushr_data_repaired.npz \
      --out ../data/g75_$F.npz --vocab-dir ../data
  python prepare.py --npz ../data/g75_$F.npz --cache ./cache75_$F --force
  python train.py --cache ./cache75_$F --epochs 8 \
      --out model75_$F.npz --log log75_$F.json
  rm ../data/g75_$F.npz        # 1.6 GB each; the cache is what training reads
done

# 3. ngrams80, emitting the id columns hybrid will reuse
python build_features.py --featurizer ngrams80 --raw ../data/cushr_data_repaired.npz \
    --out ../data/g75_ngrams80.npz --vocab-dir ../data --emit-ids --min-count 3
python prepare.py --npz ../data/g75_ngrams80.npz --cache ./cache75_ngrams80 --force
python train.py --cache ./cache75_ngrams80 --learned none --epochs 8 \
    --out model75_ngrams80.npz --log log75_ngrams80.json

# 4. hybrid variants: same cache, learned tables, frozen to a dense archive
for V in hybrid hybrid_tag_only hybrid_tag; do
  python train.py --cache ./cache75_ngrams80 --learned $V --node-dim 96 \
      --word-dropout 0.1 --epochs 8 \
      --out model75_$V.npz --log log75_$V.json --materialize ../data/g75_$V.npz
  python prepare.py --npz ../data/g75_$V.npz --cache ./cache75_$V --force
  rm ../data/g75_$V.npz
done

# 5. comparison report (this file)
python compare_featurizers.py --raw ../data/cushr_data_repaired.npz \
    --run morph43=./cache75_morph43=model75_morph43.npz \
    --run scalars64=./cache75_scalars64=model75_scalars64.npz \
    --run ngrams80=./cache75_ngrams80=model75_ngrams80.npz \
    --run hybrid=./cache75_hybrid=model75_hybrid.npz \
    --run hybrid_tag_only=./cache75_hybrid_tag_only=model75_hybrid_tag_only.npz \
    --run hybrid_tag=./cache75_hybrid_tag=model75_hybrid_tag.npz \
    --out FEATURIZER_COMPARISON_gold75.md

# 6. the composition check above
python eval_gold_subset.py --cache ./cache75_hybrid_tag \
    --model model75_hybrid_tag.npz
```

Budget ~3 GB of disk per featurizer while it is being built (1.6 GB archive +
1.4 GB cache); deleting each archive after `prepare.py` keeps the peak near 3 GB
instead of 18 GB. All corpus statistics are fitted on the training split only.

Column-by-column feature definitions are unchanged from the 49% run and are not
repeated here — see `FEATURIZER_COMPARISON_gold49.md`, "Feature reference".
