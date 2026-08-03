# 49%-gold vs 75%-gold corpus: featurizer results side by side

Compares the two training corpora across all six featurizers.

- **gold49** — `data/new_cushr_data_fixed_USE_THIS.npz`, 59,092 resolved gold
  paths (49.45%), train 53,300 / dev 2,906 / test 2,886
- **gold75** — `data/cushr_data_repaired.npz`, 89,611 resolved gold paths
  (75.0%), train 80,871 / dev 4,357 / test 4,383

The repair that produced gold75 is documented in `ingest/INGEST_METHODOLOGY.md`.
It only ever *adds* resolved sentences, so **the gold49 test split is a strict
subset of the gold75 one** — which is exactly why the two headline columns below
cannot be read against each other.

---

## 1. Headline, 8 epochs — DIFFERENT test sets

| featurizer | gold49 F1 | gold75 F1 | Δ | gold49 PM | gold75 PM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `morph43` | 0.7894 | 0.7526 | −0.0368 | 0.3049 | 0.2136 | −0.0913 |
| `scalars64` | 0.8534 | 0.8181 | −0.0353 | 0.4612 | 0.3447 | −0.1165 |
| `ngrams80` | 0.8543 | 0.8217 | −0.0326 | 0.4733 | 0.3543 | −0.1190 |
| `hybrid` | 0.8771 | 0.8486 | −0.0285 | 0.5246 | 0.4091 | −0.1155 |
| `hybrid_tag_only` | 0.8806 | 0.8585 | −0.0221 | 0.5333 | 0.4337 | −0.0996 |
| `hybrid_tag` | **0.8831** | **0.8587** | −0.0244 | **0.5450** | **0.4369** | −0.1081 |

n = 2,886 (gold49) vs 4,383 (gold75).

**These deltas are not model regressions.** Each column is measured on its own
test split, and the gold75 split contains 1,498 sentences that gold49 could not
label at all. §3 shows the drop is entirely that composition change.

## 2. Headline, 16 epochs (hybrid variants only) — DIFFERENT test sets

| featurizer | gold49 F1 | gold75 F1 | Δ | gold49 PM | gold75 PM | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid` | 0.8838 | 0.8580 | −0.0258 | 0.5423 | 0.4413 | −0.1010 |
| `hybrid_tag_only` | 0.8881 | 0.8646 | −0.0235 | 0.5537 | 0.4552 | −0.0985 |
| `hybrid_tag` | **0.8908** | **0.8675** | −0.0233 | **0.5634** | **0.4652** | −0.0982 |

### What 8 → 16 epochs buys, within each corpus

This *is* a fair comparison — each side is internally consistent.

| featurizer | gold49 ΔF1 | gold75 ΔF1 | gold49 ΔPM | gold75 ΔPM |
|---|---:|---:|---:|---:|
| `hybrid` | +0.0067 | +0.0094 | +0.0177 | +0.0322 |
| `hybrid_tag_only` | +0.0075 | +0.0061 | +0.0204 | +0.0214 |
| `hybrid_tag` | +0.0077 | +0.0088 | +0.0184 | +0.0283 |

Doubling the epoch budget is worth roughly the same on both corpora — **+0.006 to
+0.009 F1** — and perfect match gains three to four times more than F1 in every
one of the six cases.

One difference: on gold49 all three variants peaked at **epoch 16** and were
still climbing; on gold75 all three peak at **epoch 15** and give back slightly
at 16.

---

## 3. The one same-test-set measurement

`eval_gold_subset.py` splits the gold75 test set into the sentences that already
had a gold path before the repair and the ones the repair recovered, then
evaluates the gold75-trained model on each. The "pre-repair" subset is the gold49
test split, less one sentence — 2,885 of 2,886, the missing one being among the
four the repair dropped (`INGEST_METHODOLOGY.md` §3).

**`hybrid_tag`, 8 epochs, on the same 2,885/2,886 sentences:**

| model | test population | F1 | precision | recall | PM | n |
|---|---|---:|---:|---:|---:|---:|
| gold49-trained | its own test split | 0.8831 | 0.8863 | 0.8800 | 0.5450 | 2,886 |
| **gold75-trained** | **pre-repair subset** | **0.8827** | 0.8862 | 0.8792 | 0.5324 | 2,885 |
| difference | | **−0.0004** | −0.0001 | −0.0008 | −0.0126 | |

For reference, the same gold75 model on the two halves of its own split:

| subset | F1 | PM | n |
|---|---:|---:|---:|
| all | 0.8587 | 0.4369 | 4,383 |
| pre-repair | 0.8827 | 0.5324 | 2,885 |
| recovered | 0.8185 | 0.2530 | 1,498 |

### Reading

**On identical sentences the two models are indistinguishable: −0.0004 F1.**
Training on 52% more data neither helped nor hurt on the original distribution.
The −0.0244 headline gap for `hybrid_tag` is therefore *entirely* the arrival of
1,498 harder sentences in the test set, not any change in the model.

The recovered sentences really are harder — 0.8185 F1, and perfect match of
0.2530 against 0.5324. That is expected: they are precisely the sentences whose
gold path runs through analyses SHR files as auxiliary rather than as ordinary
path nodes.

Perfect match on the pre-repair subset falls slightly more than F1 (−0.0126 vs
−0.0004). On a 2,885-sentence split that is about 36 sentences, and this is a
single seed, so it is not separable from noise — but it is the only number in
this document that hints at a cost, and it is worth rechecking on another seed
before being dismissed.

### What this comparison does *not* cover

- **Only `hybrid_tag` at 8 epochs was measured this way.** The other five
  featurizers and all 16-epoch runs have no same-test-set number; §1 and §2 are
  the only data for them, with the caveat that carries.
- **No gold49 model was re-evaluated.** The gold49 column throughout is its
  published number. Re-running those models on the exact intersection is not
  possible from the current working tree: the per-featurizer gold49 caches were
  deleted, and `new_cushr_data_fixed_USE_THIS.npz` predates the raw id fields
  (`node_position`, `node_chunk`, `node_form_id`) that `build_features.py`
  requires, so the caches cannot be rebuilt without re-ingesting the unrepaired
  corpus.

---

## 4. Bottom line

| question | answer |
|---|---|
| Did the extra 30,519 sentences make the model better? | No — **−0.0004 F1** on identical sentences. |
| Did they make it worse? | No, by the same measurement. |
| Why is every headline number lower? | The test set grew by 1,498 harder sentences. |
| What did the repair actually buy? | **Coverage**: 49.45% → 75.0% of the corpus is now trainable and evaluable, including sentence types never previously seen supervised. |
| Best configuration on either corpus | `hybrid_tag`, 16 epochs. |

Ranking is identical across both corpora and both epoch budgets:
`hybrid_tag` > `hybrid_tag_only` > `hybrid` > `ngrams80` > `scalars64` > `morph43`.

Sources: `FEATURIZER_COMPARISON_gold49.md`, `FEATURIZER_COMPARISON_gold75.md`,
`HYBRID_LONGER_TRAINING_gold49.md`, `HYBRID_LONGER_TRAINING_gold75.md`, and
`eval_gold_subset.py` output for §3.
