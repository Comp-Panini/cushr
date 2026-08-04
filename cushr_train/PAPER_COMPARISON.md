# cuSHR vs TransLIST and ByT5-Sanskrit on SIGHUM test (4,200)

Every number below is measured on the **same 4,200 sentences** the two papers
report on (`sighum_test_4200.tsv`), with our metric matched to theirs level by
level. Where a comparison is not legitimate, it says so rather than pairing the
numbers anyway.

**Model**: `model75_ctx_ex4200` — `ngrams80 + hybrid_tag + char_bilstm`, 8
epochs, seed 0, 2.32M parameters, trained on CPU.

> **This model was retrained specifically for this comparison.** Our splits come
> from md5 bucketing on sentence index, so 3,241 of the benchmark's 4,200
> sentences had landed in our training set. The original `model75_ctx` scored
> **+7.5 PM points higher on sentences it had trained on** than on held-out ones.
> `make_clean_split.py` removes all 4,200 from training and backfills an equal
> number from dev, keeping the training set at 80,871 and the test split
> bit-identical. Verification that it worked: the retrained model now scores
> **0.6637** on the previously-contaminated group versus **0.7083** on truly
> held-out sentences — the memorisation advantage is gone.

---

## 1. Segmentation only — comparable to TransLIST's Table 1

Both papers evaluate Sanskrit Word Segmentation as recovery of the **unsandhied
word forms**: no lemma, no morphological tag. `eval_surface.py` reproduces that —
decode, map each predicted node to its surface form, compare the sequence to the
reference `output` column. P/R/F are **macro-averaged** (per sentence, then
averaged), which is TransLIST's stated protocol.

| Model | P | R | F | **PM** |
|---|---:|---:|---:|---:|
| TransLIST | **98.80** | **98.93** | **98.86** | **93.97** |
| ByT5-Sanskrit | – | – | – | 93.83 |
| rcNN-SS | 96.86 | 96.83 | 96.84 | 87.08 |
| FLAT-Lattice | 96.75 | 96.70 | 96.72 | 85.65 |
| Transformer | 96.52 | 96.21 | 96.36 | 83.88 |
| **cuSHR (ours)** | **97.37** | **97.22** | **97.17** | **82.60** |
| Lattice-GNN | 95.76 | 95.24 | 95.50 | 81.58 |
| TransLIST_ngrams | 96.97 | 96.77 | 96.87 | 79.28 |
| Cliq-EBM | 96.18 | 97.67 | 96.92 | 78.83 |
| Lattice-LSTM | 94.36 | 93.83 | 94.09 | 76.99 |
| TENER | 90.03 | 89.20 | 89.61 | 61.24 |
| SupPCRW | 76.30 | 79.47 | 77.85 | 38.64 |
| Seq2seq | 73.44 | 73.04 | 73.24 | 29.20 |

Baseline numbers are from Sandhan et al. (2022) Table 1 and Nehrdich et al.
(2024) Table 3. Sorted by PM.

**Where we land: 6th of 13 on PM, 2nd of 13 on F.**

That split is the interesting part. Our word-level F (97.17) beats every system
except TransLIST — including rcNN-SS, FLAT-Lattice and Transformer, all of which
beat us on PM. High F with low PM means our errors are **spread thinly across
many sentences** rather than concentrated: we get most words in most sentences
right, but slightly more sentences have at least one bad word, and PM is
all-or-nothing.

Our micro-averaged figures are 97.66 / 97.54 / 97.60 — reported only for
completeness, since the papers use macro.

---

## 2. Why our PM is capped, and roughly where

Splitting our own result by whether ingest could resolve a gold path for the
sentence:

| group | n | PM |
|---|---:|---:|
| gold path resolved | 3,633 | **91.74** |
| no gold path resolved | 567 | **23.99** |
| combined | 4,200 | 82.60 |

On the 86.5% of sentences our pipeline can fully represent, we score **91.74** —
between rcNN-SS (87.08) and TransLIST (93.97). The 567 sentences where ingest
never recovered a gold path score 23.99 and drag the headline down by ~9 points.

Those 567 are the residue documented in `ingest/INGEST_METHODOLOGY.md` §4: the
gold analysis is not reachable in the lattice we built. This is an **oracle-recall
ceiling, not a scorer failure** — no amount of training fixes a path that is not
in the search space. It bounds us at roughly **86–90% PM** as the pipeline
currently stands, which is *below TransLIST's 93.97*. Closing the gap to
TransLIST therefore requires ingest work, not model work.

TransLIST uses the same SHR candidate space but is not bounded the same way: its
PRCP module rectifies predictions that fall outside the candidate space, and
TransLIST_ngrams operates with no candidate space at all. ByT5-Sanskrit is
lexicon-free and so has no such ceiling by construction.

---

## 3. The full-analysis ladder — comparable to ByT5's Table 7 *in kind only*

`eval_slm.py` scores at the levels ByT5-Sanskrit uses. Our lattice nodes carry
form, lemma and `cng`, so selecting a path commits to all three at once and the
ladder falls out of how much of the node has to match.

| level | cuSHR (SIGHUM 4,200 subset) | ByT5-Sanskrit (DCS 2024, sentence) | ByT5 (DCS 2024, paragraph) |
|---|---:|---:|---:|
| S | **92.98** | 84.61 | 88.21 |
| S+L | **89.54** | – | – |
| S+M | **67.41** | 63.86 | 74.38 |
| S+L+M | **66.89** | 61.27 | 74.31 |
| L (alone) | – | 79.88 | 83.96 |
| L+M | – | 62.00 | 72.33 |

> **Do not read the columns against each other as a result.** ByT5's ladder is
> measured on *their own* DCS April-2024 split (601,403 sentences, 8,398 test),
> **not on SIGHUM**. The only ByT5 number on SIGHUM is the segmentation PM of
> 93.83 in §1. Different corpus, different split, different annotation
> vintage — this table shows that the *same shape* of degradation appears in both
> systems, nothing more.

Ours is measured on the 3,633 SIGHUM sentences with a resolved gold path
(reference is our reconstructed gold, since the TSV carries only segmentation),
so it is also a biased-easy subset — note S here is 92.98 against 82.60 in §1 for
exactly that reason.

**The shape that does transfer**: adding morphology costs far more than adding
lemma. For us S→S+L loses 3.4 points but S→S+M loses 25.6; for ByT5 S→S+M loses
20.8. Both systems agree that morphosyntactic tags are where joint accuracy
collapses, which matches ByT5's own error analysis (most disagreements are
lexical and morphological, not boundary placement) and the ambiguity of Sanskrit
nominal endings.

---

## 4. What our internal reports measure

Our headline PM elsewhere in this repo (e.g. `CONTEXTUAL_ENCODER.md`, 0.6731) is
**node-identity** perfect match — form *and* lemma *and* tag, i.e. the S+L+M
level. It is not comparable to a segmentation number and reads ~26 points lower
for that reason alone. On our own test split (4,383) this model scores F1 0.9297
/ PM 0.6735, consistent with the 66.89 S+L+M above.

Reference sanity check: our reconstructed gold path agrees with DCS's published
segmentation on **97.7%** of the 3,633 scorable sentences. The 2.3% that differ
are the orphan-repair redirect choosing an adjacent analysis (`iva`/`eva`,
`saH`/`sa`), so our internal gold is a faithful stand-in with a small known
drift.

---

## 5. Caveats that keep this from being a clean head-to-head

1. **Different training data.** SIGHUM's official split is 97,000 train / 3,000
   dev / 4,200 test. Ours is 80,871 train drawn from a 119,503-sentence ingest of
   the same corpus, with the 4,200 excluded. Our models see ~17% less data, and
   not the same data.
2. **Different candidate space in practice.** We ingest SHR graphml with a
   forced-DAG edge filter (`INGEST_METHODOLOGY.md` §1) and lose analyses SHR
   attaches through non-`key=1` edges. TransLIST consumes SHR's candidate space
   through its own pipeline.
3. **567 sentences unrepresentable.** See §2. Included in §1 (honest) but they
   depress PM by ~9 points.
4. **Single seed, no significance testing.** TransLIST reports p<0.05 t-tests
   against baselines; we have one run.
5. **Trained on CPU at 2.32M parameters** against TransLIST's 50-epoch GPU
   training and ByT5-Sanskrit's 582M-parameter pretrained byte-level LM. The
   comparison is informative about the approach, not a controlled experiment.

---

## Reproducing

```bash
cd cushr_train
python make_clean_split.py                    # writes splits_ex4200.json
python train.py --cache ./cache75_ngrams80 --learned hybrid_tag \
    --encoder char_bilstm --word-dropout 0.1 --epochs 8 --seed 0 --resume \
    --splits-override splits_ex4200.json \
    --out model75_ctx_ex4200.npz --log log75_ctx_ex4200.json \
    --materialize ../data/g75_ctx_ex4200.npz
python prepare.py --npz ../data/g75_ctx_ex4200.npz --cache ./cache75_ctx_ex4200 --force
rm ../data/g75_ctx_ex4200.npz

python eval_surface.py  --cache ./cache75_ctx_ex4200 --model model75_ctx_ex4200.npz  # §1, §2
python eval_slm.py      --cache ./cache75_ctx_ex4200 --model model75_ctx_ex4200.npz  # §3
python eval_id_list.py  --cache ./cache75_ctx_ex4200 --model model75_ctx_ex4200.npz  # contamination audit
```

**A bug worth knowing about if you extend these scorers.** `predicted_nodes`
returns the path as Viterbi backtracked it, which is **reversed**. The first run
of `eval_surface.py` reported PM 0.0081 with word F 98.10 — the words were right
and the order was backwards, and 85.8% of "errors" were order-only. A metric that
only looked at F1 would have shown a healthy 98 and hidden it completely. Both
scorers now sort predicted nodes by `node_char_start`.

## Sources

- Sandhan, Singha, Rao, Samanta, Behera, Goyal. *TransLIST: A Transformer-Based
  Linguistically Informed Sanskrit Tokenizer.* Findings of EMNLP 2022.
  arXiv:2210.11753. Table 1.
- Nehrdich et al. *ByT5-Sanskrit.* Findings of EMNLP 2024, pp. 13745–13750.
  Tables 3, 7, 8.
