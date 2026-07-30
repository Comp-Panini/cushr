# Featurizer comparison: `morph43` → `scalars64` → `ngrams80` → `hybrid`

- Corpus: all 119,503 SIGHUM sentences (4,488,155 lattice nodes, 61,140,552 edges)
- Sentences with a resolved gold path: 59,092 (49.4%) → train 53,300 / dev 2,906 / test 2,886
- Identical training for all four: 8 epochs, batch 64, AdamW lr 1e-3, structured hinge

## What each featurization contains

| columns | block | `morph43` | `scalars64` | `ngrams80` | `hybrid` |
|---|---|---|---|---|---|
| 0–42 | morph presence bits (43) | yes | yes | yes | yes |
| 43 | `log1p(word length)` | yes | yes | yes | yes |
| 44–47 | length extras | zero | added | yes | yes |
| 48–51 | position in sentence and chunk | zero | added | yes | yes |
| 52–55 | corpus frequency of form and lemma | zero | added | yes | yes |
| 56–63 | character-class / phonotactic summary | zero | added | yes | yes |
| 64–79 | hashed character n-grams (16 buckets) | — | — | added | yes |
| — | learned form/lemma/preverb embeddings | — | — | — | added |
| | node vector width | 64 | 64 | 80 | 96 |
| | trainable parameters | 16,385 | 16,385 | 20,481 | 1,200,529 |

## Headline

| featurizer | F1 | precision | recall | perfect match | params |
|---|---|---|---|---|---|
| `morph43` | 0.7894 | 0.8176 | 0.7632 | 0.3049 | 16,385 |
| `scalars64` | 0.8534 | 0.8580 | 0.8487 | 0.4612 | 16,385 |
| `ngrams80` | 0.8543 | 0.8570 | 0.8515 | 0.4733 | 20,481 |
| `hybrid` | **0.8771** | **0.8795** | **0.8748** | **0.5246** | 1,200,529 |


## Recall by training frequency of the gold word

| featurizer | unseen (n=663) | rare 1–4 (n=1,027) | mid 5–49 (n=4,122) | common 50+ (n=12,163) |
|---|---|---|---|---|
| `morph43` | 0.7481 | 0.8578 | 0.8214 | 0.7362 |
| `scalars64` | 0.7903 | 0.8929 | 0.8559 | 0.8458 |
| `ngrams80` | 0.7677 | 0.8822 | 0.8510 | 0.8537 |
| `hybrid` | 0.7949 | 0.8861 | 0.8675 | 0.8806 |


| step | unseen | rare | mid | common |
|---|---|---|---|---|
| `morph43` → `scalars64` | +0.0422 | +0.0351 | +0.0344 | +0.1095 |
| `scalars64` → `ngrams80` | −0.0226 | −0.0107 | −0.0049 | +0.0079 |
| `ngrams80` → `hybrid` | +0.0271 | +0.0039 | +0.0165 | +0.0270 |

## Error counts

| featurizer | TP | FP | FN |
|---|---|---|---|
| `morph43` | 13,718 | 3,061 | 4,257 |
| `scalars64` | 15,256 | 2,524 | 2,719 |
| `ngrams80` | 15,306 | 2,553 | 2,669 |
| `hybrid` | 15,724 | 2,155 | 2,251 |


## Training behaviour

Dev F1 by epoch:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `morph43` | 0.7677 | 0.7812 | 0.7711 | 0.7727 | 0.7788 | 0.7792 | 0.7726 | 0.7814 |
| `scalars64` | 0.8299 | 0.8354 | 0.8425 | 0.8390 | 0.8413 | 0.8436 | 0.8437 | 0.8438 |
| `ngrams80` | 0.8308 | 0.8406 | 0.8472 | 0.8434 | 0.8452 | 0.8472 | 0.8478 | 0.8451 |
| `hybrid` | 0.8333 | 0.8500 | 0.8598 | 0.8666 | 0.8636 | 0.8685 | 0.8740 | 0.8742 |

## Feature reference: what every column is

Fires is the fraction of the nodes where the column is non-zero. 
Mean is its average over those nodes. 

### Columns 0–42: morphology (presence bits parsed from the SHR tag)

| # | token | meaning | fires |
|---|---|---|---|
| 0 | `nom` | nominative | 0.2527 |
| 1 | `acc` | accusative | 0.1859 |
| 2 | `dat` | dative | 0.0150 |
| 3 | `abl` | ablative | 0.0176 |
| 4 | `g` | genitive | 0.0383 |
| 5 | `loc` | locative | 0.0486 |
| 6 | `voc` | vocative | 0.0626 |
| 7 | `i` | instrumental | 0.0324 |
| 8 | `sg` | singular | 0.4999 |
| 9 | `du` | dual | 0.1235 |
| 10 | `pl` | plural | 0.1190 |
| 11 | `m` | masculine | 0.2176 |
| 12 | `f` | feminine | 0.1637 |
| 13 | `n` | neuter | 0.2522 |
| 14 | `*` | gender-indeterminate | 0.0197 |
| 15 | `pr` | present | 0.0124 |
| 16 | `pft` | perfect | 0.0303 |
| 17 | `impft` | imperfect | 0.0134 |
| 18 | `fut` | future | 0.0023 |
| 19 | `aor` | aorist | 0.0027 |
| 20 | `opt` | optative | 0.0039 |
| 21 | `imp` | imperative | 0.0227 |
| 22 | `per` | periphrastic (as in `per. fut.`) | 0.0011 |
| 23 | `ac` | active voice (parasmaipada) | 0.0949 |
| 24 | `md` | middle voice (ātmanepada) | 0.0091 |
| 25 | `ps` | passive | 0.0041 |
| 26 | `1` | 1st person | 0.0220 |
| 27 | `2` | 2nd person | 0.0377 |
| 28 | `3` | 3rd person | 0.0295 |
| 29 | `ppr` | present participle | 0.0178 |
| 30 | `pfp` | gerundive / future passive participle | 0.0150 |
| 31 | `pp` | past passive participle | 0.0351 |
| 32 | `ppa` | past active participle | 0.0002 |
| 33 | `adv` | adverb | 0.0124 |
| 34 | `ind` | indeclinable | 0.0039 |
| 35 | `abs` | absolutive (gerund) | 0.0037 |
| 36 | `inf` | infinitive | 0.0005 |
| 37 | `iic` | compound-initial stem (*in initio compositi*) | 0.1173 |
| 38 | `conj` | conjugation class marker | 0.0231 |
| 39 | `prep` | preposition / preverb | 0.0098 |
| 40 | `part` | particle | 0.0170 |
| 41 | `ca` | causative (as in `ca. pp.`) | 0.0065 |
| 42 | `UNKNOWN` | tag absent or unparsed | 0.0000 |

### Column 43: length (in all four featurizations)

| # | feature | definition | fires | mean |
|---|---|---|---|---|
| 43 | `log1p(length)` | `log(1 + surface char length)` | 1.0000 | 1.7095 |

### Columns 44–63: `scalars64` additions

| # | block | feature | definition | fires | mean |
|---|---|---|---|---|---|
| 44 | length | scaled length | `clip(len / 20, 0, 1)` | 1.0000 | 0.2447 |
| 45 | length | short | `len <= 3` | 0.2988 | 0.2988 |
| 46 | length | medium | `3 < len <= 8` | 0.6319 | 0.6319 |
| 47 | length | long | `len > 8` | 0.0693 | 0.0693 |
| 48 | position | start in sentence | `char_start / sentence_len` | 0.8551 | 0.4149 |
| 49 | position | end in sentence | `(char_start + len) / sentence_len` | 1.0000 | 0.5338 |
| 50 | position | distance from chunk end | `(chunk_width - reach) / chunk_width` | 0.5000 | 0.2090 |
| 51 | position | chunk index | `clip(chunk_no / 20, 0, 1)` | 1.0000 | 0.1504 |
| 52 | frequency | form frequency | `log1p(train count of surface form) / 10` | 0.9223 | 0.5503 |
| 53 | frequency | lemma frequency | `log1p(train count of lemma) / 10` | 0.9698 | 0.6893 |
| 54 | frequency | hapax / unseen | `form train count <= 1` | 0.0824 | 0.0824 |
| 55 | frequency | frequency rank | rank of form by train count, normalised | 0.9790 | 0.0874 |
| 56 | char class | vowel rate | vowels / length | 1.0000 | 0.4740 |
| 57 | char class | cluster rate | adjacent consonant pairs / (length-1) | 0.3380 | 0.0797 |
| 58 | char class | retroflex rate | retroflex chars / length | 0.1322 | 0.0300 |
| 59 | char class | visarga | contains `H` | 0.1656 | 0.1656 |
| 60 | char class | anusvāra | contains `M` | 0.0044 | 0.0044 |
| 61 | char class | ends in vowel | final char is a vowel | 0.6057 | 0.6057 |
| 62 | char class | starts with vowel | initial char is a vowel | 0.3249 | 0.3249 |
| 63 | char class | final long vowel | final char is a long vowel | 0.2940 | 0.2940 |

### Columns 64–79: `ngrams80` additions

| # | feature | definition |
|---|---|---|
| 64–79 | hashed character n-grams | count-normalised CRC32 hash of the form's character bigrams and trigrams into 16 buckets |

### `hybrid`: learned embeddings

| table | vocabulary | kept (min_count=3) | dim | params |
|---|---|---|---|---|
| surface form | 89,045 | 31,206 (+PAD/UNK) | 32 | 998,656 |
| lemma | 24,120 | 10,078 (+PAD/UNK) | 16 | 161,280 |
| preverb | 1,675 | 810 (+PAD/UNK, no threshold) | 4 | 3,248 |


## Reproducing

```bash
# 1. ingest the corpus (parallel; ~20 min on 11 cores)
cd ingest
python parallel_ingest.py --out ../data/raw.npz --workers 11

cd ../cushr_train
# 2. precomputed featurizers
for F in morph43 scalars64; do
  python build_features.py --featurizer $F --raw ../data/raw.npz --out ../data/f_$F.npz
  python prepare.py --npz ../data/f_$F.npz --cache ./cache_$F --force
  python train.py --cache ./cache_$F --epochs 8 --out model_$F.npz --log log_$F.json
done

# 3. ngrams80, emitting the id columns hybrid will reuse
python build_features.py --featurizer ngrams80 --raw ../data/raw.npz \
    --out ../data/f_ngrams80.npz --emit-ids --min-count 3
python prepare.py --npz ../data/f_ngrams80.npz --cache ./cache_ngrams80 --force
python train.py --cache ./cache_ngrams80 --learned none --epochs 8 \
    --out model_ngrams80.npz --log log_ngrams80.json

# 4. hybrid: same cache, learned tables, then freeze to a dense archive
python train.py --cache ./cache_ngrams80 --learned hybrid --node-dim 96 \
    --word-dropout 0.1 --epochs 8 \
    --out model_hybrid.npz --log log_hybrid.json --materialize ../data/f_hybrid.npz
python prepare.py --npz ../data/f_hybrid.npz --cache ./cache_hybrid --force

# 5. comparison report (this file)
python compare_featurizers.py --raw ../data/raw.npz \
    --run morph43=./cache_morph43=model_morph43.npz \
    --run scalars64=./cache_scalars64=model_scalars64.npz \
    --run ngrams80=./cache_ngrams80=model_ngrams80.npz \
    --run hybrid=./cache_hybrid=model_hybrid.npz \
    --out FEATURIZER_COMPARISON.md
```

All corpus statistics — frequency features and the `min_count` threshold alike —
are fitted on the training split only. `build_features.py` reproduces
`prepare.py`'s md5 bucketing to determine it, so no dev or test sentence
contributes.
