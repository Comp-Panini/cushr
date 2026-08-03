# Hybrid featurizers at 16 epochs (75%-gold corpus)

Source archive `data/cushr_data_repaired.npz` — 89,611 resolved gold paths
(75.0%), train 80,871 / dev 4,357 / test 4,383. See
`ingest/INGEST_METHODOLOGY.md`.

> Not comparable to `HYBRID_LONGER_TRAINING_gold49.md` by headline: that report's
> test split is a strict subset of this one. See
> `FEATURIZER_COMPARISON_gold75.md`, "Is the model actually worse?".

## Headline: 8 epochs vs 16

| featurizer | epochs | F1 | precision | recall | perfect match |
|---|---|---|---|---|---|
| `hybrid` | 8 | 0.8486 | 0.8533 | 0.8439 | 0.4091 |
| `hybrid` | 16 | 0.8580 | 0.8603 | 0.8557 | 0.4413 |
| `hybrid_tag_only` | 8 | 0.8585 | 0.8626 | 0.8544 | 0.4337 |
| `hybrid_tag_only` | 16 | 0.8646 | 0.8677 | 0.8615 | 0.4552 |
| `hybrid_tag` | 8 | 0.8587 | 0.8623 | 0.8552 | 0.4369 |
| `hybrid_tag` | 16 | **0.8675** | **0.8703** | **0.8648** | **0.4652** |

| featurizer | F1 @8 | F1 @16 | delta | PM @8 | PM @16 | delta |
|---|---|---|---|---|---|---|
| `hybrid` | 0.8486 | 0.8580 | +0.0094 | 0.4091 | 0.4413 | +0.0322 |
| `hybrid_tag_only` | 0.8585 | 0.8646 | +0.0061 | 0.4337 | 0.4552 | +0.0214 |
| `hybrid_tag` | 0.8587 | 0.8675 | +0.0088 | 0.4369 | 0.4652 | +0.0283 |

Doubling the budget is worth **+0.006 to +0.009 F1**, and considerably more on
perfect match (**+0.021 to +0.032**) — the same pattern as on the 49% corpus,
where the gains were +0.007 to +0.008 F1 and +0.018 to +0.020 PM. Perfect match
benefits disproportionately in both, which is what you would expect if the extra
epochs are fixing the last one or two words in sentences that are otherwise
already correct.

Ranking is unchanged at both budgets and on both corpora:
`hybrid_tag` > `hybrid_tag_only` > `hybrid`.

## Dev F1 by epoch

| epoch | `hybrid` | `hybrid_tag_only` | `hybrid_tag` |
|---|---|---|---|
| 1 | 0.8119 | 0.8153 | 0.8174 |
| 2 | 0.8266 | 0.8366 | 0.8376 |
| 3 | 0.8390 | 0.8426 | 0.8468 |
| 4 | 0.8432 | 0.8493 | 0.8524 |
| 5 | 0.8430 | 0.8502 | 0.8525 |
| 6 | 0.8450 | 0.8556 | 0.8566 |
| 7 | 0.8495 | 0.8563 | 0.8591 |
| 8 | 0.8504 | 0.8583 | 0.8583 |
| 9 | 0.8510 | 0.8592 | 0.8619 |
| 10 | 0.8526 | 0.8639 | 0.8643 |
| 11 | 0.8536 | 0.8600 | 0.8623 |
| 12 | 0.8535 | 0.8634 | 0.8657 |
| 13 | 0.8568 | 0.8654 | 0.8671 |
| 14 | 0.8582 | 0.8659 | 0.8685 |
| 15 | **0.8608** | **0.8676** | **0.8688** |
| 16 | 0.8606 | 0.8667 | 0.8677 |

All three dip together at epoch 11 and peak together at epoch 15. The
synchronisation is expected: the runs share `--seed 0`, so they see identical
shuffle orders, and a batch ordering that is awkward for one is awkward for all.
It is a property of the schedule, not evidence about the featurizers.

## Convergence: peaked at 15, unlike the 49% corpus

| featurizer | best dev F1 | at epoch | epochs after best |
|---|---|---|---|
| `hybrid` | 0.8608 | 15 | 1 |
| `hybrid_tag_only` | 0.8676 | 15 | 1 |
| `hybrid_tag` | 0.8688 | 15 | 1 |

**This differs from the 49% run, where all three peaked at epoch 16 and were
still climbing.** Here every variant peaks at 15 and gives back a little at 16.
That is a single epoch of decline on one seed and should not be read as
convergence — but combined with the shrinking marginal gains below, 16 epochs is
no longer obviously too few, whereas on the 49% corpus it clearly was.

| featurizer | gain over epochs 1-8 | gain over epochs 9-16 | gain over last 3 |
|---|---|---|---|
| `hybrid` | +0.0385 | +0.0102 | +0.0037 |
| `hybrid_tag_only` | +0.0430 | +0.0084 | +0.0013 |
| `hybrid_tag` | +0.0409 | +0.0095 | +0.0006 |

The second half of training is worth roughly a quarter of the first, and the last
three epochs are worth almost nothing for the two tag variants.

## Train loss by epoch

| epoch | `hybrid` | `hybrid_tag_only` | `hybrid_tag` |
|---|---|---|---|
| 1 | 3.3509 | 3.4028 | 3.2274 |
| 2 | 2.9070 | 2.7634 | 2.7022 |
| 4 | 2.7003 | 2.5293 | 2.4796 |
| 6 | 2.5964 | 2.4186 | 2.3789 |
| 8 | 2.5227 | 2.3449 | 2.3096 |
| 10 | 2.4649 | 2.2861 | 2.2542 |
| 12 | 2.4173 | 2.2375 | 2.2075 |
| 14 | 2.3791 | 2.1969 | 2.1719 |
| 16 | 2.3425 | 2.1626 | 2.1382 |

Training loss is still falling monotonically at epoch 16 while dev F1 has
flattened — the usual signal that further epochs buy fit, not generalisation.

Wall-clock, CPU: `hybrid` 14.2 min, `hybrid_tag_only` 10.9 min, `hybrid_tag`
11.3 min for 16 epochs.

## Reproducing

```bash
cd cushr_train
for V in hybrid hybrid_tag_only hybrid_tag; do
  python train.py --cache ./cache75_ngrams80 --learned $V --node-dim 96 \
      --word-dropout 0.1 --epochs 16 --seed 0 \
      --out model75_16_$V.npz --log log75_16_$V.json
done
```

Add `--materialize ../data/g75_16_$V.npz` to each line for the frozen dense
archives needed by the C++/GPU path, the per-frequency breakdown, and
`eval_gold_subset.py`.

> **These runs were made without `--materialize`,** so there is no cache matching
> their learned tables and the pre-repair/recovered subset split in
> `FEATURIZER_COMPARISON_gold75.md` could **not** be repeated at 16 epochs. An
> attempt to evaluate `model75_16_hybrid_tag.npz` against `cache75_hybrid_tag`
> returned 0.8542 rather than the logged 0.8675, because that cache holds vectors
> materialised from the *8-epoch* featurizer — a silent featurizer/scorer
> mismatch, not a real result. It is recorded here only as a warning: a model
> trained with `--learned` is only valid against a cache materialised from the
> same run.
