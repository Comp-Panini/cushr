# Hybrid featurizers at 16 epochs

## Headline: 8 epochs vs 16

| featurizer | epochs | F1 | precision | recall | perfect match |
|---|---|---|---|---|---|
| `hybrid` | 8 | 0.8771 | 0.8795 | 0.8748 | 0.5246 |
| `hybrid` | 16 | 0.8838 | 0.8874 | 0.8803 | 0.5423 |
| `hybrid_tag_only` | 8 | 0.8806 | 0.8838 | 0.8774 | 0.5333 |
| `hybrid_tag_only` | 16 | 0.8881 | 0.8917 | 0.8845 | 0.5537 |
| `hybrid_tag` | 8 | 0.8831 | 0.8863 | 0.8800 | 0.5450 |
| `hybrid_tag` | 16 | **0.8908** | **0.8940** | **0.8876** | **0.5634** |

| featurizer | F1 @8 | F1 @16 | delta | PM @8 | PM @16 | delta |
|---|---|---|---|---|---|---|
| `hybrid` | 0.8771 | 0.8838 | +0.0067 | 0.5246 | 0.5423 | +0.0177 |
| `hybrid_tag_only` | 0.8806 | 0.8881 | +0.0075 | 0.5333 | 0.5537 | +0.0204 |
| `hybrid_tag` | 0.8831 | 0.8908 | +0.0077 | 0.5450 | 0.5634 | +0.0184 |

## Dev F1 by epoch

| epoch | `hybrid` | `hybrid_tag_only` | `hybrid_tag` |
|---|---|---|---|
| 1 | 0.8333 | 0.8264 | 0.8419 |
| 2 | 0.8500 | 0.8430 | 0.8527 |
| 3 | 0.8598 | 0.8595 | 0.8681 |
| 4 | 0.8666 | 0.8615 | 0.8685 |
| 5 | 0.8636 | 0.8722 | 0.8753 |
| 6 | 0.8685 | 0.8712 | 0.8766 |
| 7 | 0.8740 | 0.8748 | 0.8826 |
| 8 | 0.8742 | 0.8816 | 0.8857 |
| 9 | 0.8757 | 0.8817 | 0.8843 |
| 10 | 0.8751 | 0.8810 | 0.8861 |
| 11 | 0.8755 | 0.8768 | 0.8816 |
| 12 | 0.8804 | 0.8842 | 0.8882 |
| 13 | 0.8789 | 0.8861 | 0.8887 |
| 14 | 0.8797 | 0.8835 | 0.8882 |
| 15 | 0.8795 | 0.8863 | 0.8877 |
| 16 | **0.8805** | **0.8865** | **0.8904** |


## Convergence: still not there at 16

| featurizer | best dev F1 | at epoch | epochs after best |
|---|---|---|---|
| `hybrid` | 0.8805 | 16 | 0 |
| `hybrid_tag_only` | 0.8865 | 16 | 0 |
| `hybrid_tag` | 0.8904 | 16 | 0 |


| featurizer | gain over epochs 1-8 | gain over epochs 9-16 | gain over last 3 |
|---|---|---|---|
| `hybrid` | +0.0409 | +0.0062 | +0.0015 |
| `hybrid_tag_only` | +0.0552 | +0.0050 | +0.0004 |
| `hybrid_tag` | +0.0438 | +0.0047 | +0.0017 |


## Train loss by epoch

| epoch | `hybrid` | `hybrid_tag_only` | `hybrid_tag` |
|---|---|---|---|
| 1 | 3.0171 | 3.1266 | 2.9260 |
| 2 | 2.4701 | 2.4401 | 2.3371 |
| 4 | 2.2340 | 2.1601 | 2.0882 |
| 6 | 2.1189 | 2.0224 | 1.9651 |
| 8 | 2.0459 | 1.9450 | 1.8889 |
| 10 | 1.9870 | 1.8792 | 1.8318 |
| 12 | 1.9370 | 1.8295 | 1.7859 |
| 14 | 1.8973 | 1.7876 | 1.7448 |
| 16 | 1.8607 | 1.7451 | 1.7044 |


## Reproducing

```bash
cd cushr_train
for V in hybrid hybrid_tag_only hybrid_tag; do
  python train.py --cache ./cache_ngrams80 --learned $V --node-dim 96 \
      --word-dropout 0.1 --epochs 16 --seed 0 \
      --out model16_$V.npz --log log16_$V.json
done
```

Add `--materialize ../data/f16_$V.npz` to each line if you also want the frozen
dense archives for the C++/GPU path and the frequency breakdown above.
