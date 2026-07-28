# Featurizer comparison: `morph43` vs `scalars64`

- Corpus: all 119,503 SIGHUM sentences (4,488,155 lattice nodes, 61,140,552 edges)
- Sentences with a resolved gold path: 59,092 (49.4%) → train 53,300 / dev 2,906 / test 2,886
- Both models are the same size (16,385 parameters). Only the input features differ

## What the two featurizations contain

| columns | block | `morph43` | `scalars64` |
|---|---|---|---|
| 0–42 | morph presence bits (43) | yes | yes |
| 43 | `log1p(word length)` | yes | yes |
| 44–47 | length extras: scaled length + short/medium/long buckets | zero | added |
| 48–51 | position in sentence and chunk | zero | added |
| 52–55 | corpus frequency of the form and lemma | zero | added |
| 56–63 | character-class / phonotactic summary | zero | added |
| | columns carrying signal | 43 of 64 | 63 of 64 |

1. **Length extras (4).** `morph43` already has `log1p(length)`, a single
   monotonic column. Adding a scaled copy plus three bucket indicators lets the
   model express a *non-monotonic* length preference instead of one slope.
2. **Position (4).** Where the candidate starts and ends within the sentence,
   how far it sits from the end of its chunk, and which chunk it is. Encodes
   compound-initial vs compound-final, and position-sensitive phenomena such as
   verse-final verbs and particle placement.
3. **Frequency (4).** How often the surface form and its lemma occur in the
   *training split*. This is the block that does most of the work (see the
   bucket analysis below): it is a parameter-free surrogate for word identity,
   letting the model prefer a common analysis over a rare-but-legal one.
4. **Character class (8).** Vowel/cluster/retroflex rates and what the form
   begins and ends with. Sandhi is conditioned on the phonemes at a word
   junction, so a form's final phoneme carries real information about which
   junctions are plausible.

Crucially, no learned parameters are added. All 20 columns are computed
during ingest/featurization and stored as plain floats. 

## Headline

| featurizer | F1 | precision | recall | perfect match | params |
|---|---|---|---|---|---|
| `morph43` | 0.7894 | 0.8176 | 0.7632 | 0.3049 | 16,385 |
| `scalars64` | 0.8534 | 0.8580 | 0.8487 | 0.4612 | 16,385 |


## Featurizer vs Rareness of word

| featurizer | unseen (n=663) | rare 1–4 (n=1,027) | mid 5–49 (n=4,122) | common 50+ (n=12,163) |
|---|---|---|---|---|
| `morph43` | 0.7481 | 0.8578 | 0.8214 | 0.7362 |
| `scalars64` | 0.7903 | 0.8929 | 0.8559 | 0.8458 |


### Error composition

| featurizer | True Pos | False Pos | False Neg |
|---|---|---|---|
| `morph43` | 13,718 | 3,061 | 4,257 |
| `scalars64` | 15,256 | 2,524 | 2,719 |

## Training behaviour

Dev F1 by epoch:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `morph43` | 0.7677 | 0.7812 | 0.7711 | 0.7727 | 0.7788 | 0.7792 | 0.7726 | 0.7814 |
| `scalars64` | 0.8299 | 0.8354 | 0.8425 | 0.8390 | 0.8413 | 0.8436 | 0.8437 | 0.8438 |

## Feature reference: what every column is

"Fires" is the fraction of the 4,249,149 non-boundary lattice nodes in the full corpus where the column is non-zero.

"Mean" is its average value over those nodes.

### Columns 0–42: morphology (presence bits parsed from the SHR tag)

Decompose 'one-hot' tags into their parts, so it's not one-hot and memory-heavy.

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


### Column 43: length (in both featurizations)

| # | feature | definition | fires | mean |
|---|---|---|---|---|
| 43 | `log1p(length)` | `log(1 + surface char length)` | 1.0000 | 1.7095 |

This one column is the entire hand-tuned `LengthScorer` baseline.

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

## Reproducing

```bash
# 1. ingest the corpus (parallel; ~20 min on 11 cores)
cd ingest
python parallel_ingest.py --out ../data/raw.npz --workers 11

# 2. featurize, cache, and train, per featurizer
cd ../cushr_train
for F in morph43 scalars64; do
  python build_features.py --featurizer $F --raw ../data/raw.npz --out ../data/f_$F.npz
  python prepare.py --npz ../data/f_$F.npz --cache ./cache_$F --force
  python train.py --cache ./cache_$F --epochs 8 --batch 64 \
      --out model_$F.npz --log log_$F.json
done

# 3. comparison report (this file)
python compare_featurizers.py --raw ../data/raw.npz \
    --run morph43=./cache_morph43=model_morph43.npz \
    --run scalars64=./cache_scalars64=model_scalars64.npz \
    --out FEATURIZER_COMPARISON.md
```
