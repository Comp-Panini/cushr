# Contextual encoding: `char_bilstm`

A character-level BiLSTM inserted between the node featurizer and the biaffine
scorer. 

- Corpus: `data/cushr_data_repaired.npz`, 89,611 resolved gold paths (75.0%),
  train 80,871 / dev 4,357 / test 4,383
- Baseline: `hybrid_tag`, 8 epochs, seed 0, same cache. Only `--encoder` differs.

---

`BiaffineEdgeScorer.edge_scores` (`model.py:36`) computes

```
score(u -> v) = src_proj(f)[u] · dst_proj(f)[v] + bias
```

so an edge's score is a function of exactly two node vectors. Viterbi recovers a
globally optimal *path*, but it can only combine the scores it is handed — it
cannot invent an interaction the scorer never expressed. If two segmentations
differ only in how they interact with a word ten positions away, the scorer
assigns them identical scores. The model was first-order over the lattice.

| change | added params | Δ test F1 |
|---|---:|---:|
| `hybrid_tag_full` — remove the 156→96 bottleneck | +25K (scorer) | **+0.0004** |
| `char_bilstm` — sentence context | +619K (encoder) | **+0.0695** |


| | params |
|---|---:|
| encoder (`char_bilstm`) | 618,624 |
| featurizer embeddings (`hybrid_tag`) | 1,638,876 |
| **total model** | **2,321,725** |


## Results (8 epochs, seed 0)

| test subset | F1 | PM | baseline F1 | baseline PM | ΔF1 | ΔPM |
|---|---:|---:|---:|---:|---:|---:|
| **all** (n=4,383) | **0.9282** | **0.6731** | 0.8587 | 0.4369 | **+0.0695** | **+0.2362** |
| pre-repair (n=2,885) | 0.9365 | 0.7300 | 0.8827 | 0.5324 | +0.0538 | +0.1976 |
| recovered (n=1,498) | 0.9144 | 0.5634 | 0.8185 | 0.2530 | +0.0959 | **+0.3104** |

### Dev by epoch

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| dev F1 | 0.8755 | 0.9015 | 0.9133 | 0.9225 | 0.9257 | 0.9277 | 0.9298 | 0.9302 |
| dev PM | 0.4919 | 0.5731 | 0.6123 | 0.6431 | 0.6564 | 0.6610 | 0.6711 | 0.6725 |
| train loss | 2.6479 | 1.9435 | 1.6589 | 1.4778 | 1.3568 | 1.2557 | 1.1764 | 1.0986 |
| active frac | 0.878 | 0.815 | 0.772 | 0.739 | 0.715 | 0.692 | 0.675 | 0.656 |

---

## 16 epochs vs 8 epochs

| | 8 epochs | 16 epochs | Δ |
|---|---:|---:|---:|
| test F1 | 0.9282 | 0.9287 | +0.0005 |
| test PM | **0.6731** | 0.6724 | −0.0007 |
| best dev F1 | 0.9302 (ep 8) | 0.9312 (ep 10) | |
| best dev PM | 0.6725 (ep 8) | 0.6801 (ep 10) | |


### Dev by epoch, 16-epoch run

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev F1 | 0.8755 | 0.9015 | 0.9133 | 0.9225 | 0.9257 | 0.9277 | 0.9298 | 0.9302 | 0.9311 | **0.9312** | 0.9286 | 0.9283 | 0.9297 | 0.9295 | 0.9282 | 0.9284 |
| dev PM | 0.4919 | 0.5731 | 0.6123 | 0.6431 | 0.6564 | 0.6610 | 0.6711 | 0.6725 | 0.6787 | **0.6801** | 0.6693 | 0.6661 | 0.6791 | 0.6725 | 0.6651 | 0.6677 |
| train loss | 2.6479 | 1.9435 | 1.6589 | 1.4778 | 1.3568 | 1.2557 | 1.1764 | 1.0986 | 1.0276 | 0.9646 | 0.9052 | 0.8562 | 0.8099 | 0.7683 | 0.7326 | 0.7027 |
| active | 0.878 | 0.815 | 0.772 | 0.739 | 0.715 | 0.692 | 0.675 | 0.656 | 0.633 | 0.618 | 0.599 | 0.583 | 0.565 | 0.550 | 0.534 | 0.527 |

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
