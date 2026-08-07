# cuSHR vs TransLIST and ByT5-Sanskrit

Two distinct scientific claims, kept apart on purpose:

- **§1 — unconstrained system benchmark.** Every system at full capability, no
  handicaps. Answers *which performs better in practice*, and frames cuSHR as a
  lightweight alternative: 97.5% of ByT5's perfect match at 1/210th the
  parameters.
- **§5 — controlled ablation** *(planned)*. Fine-tunes ByT5 on cuSHR's exact
  split to answer *why* the leaders are ahead — architecture, or data and
  pretraining.

Conflating the two produces either a weak excuse ("they had more data") or a
false equivalence. §1 gives the leaders full credit; §5 is where the confound
gets measured rather than asserted.

Below §1, each section matches what the relevant paper actually measures. They do
not share a test set and are not meant to.

- **§1 TransLIST** — word-level segmentation on the SIGHUM 4,200 test set. Both
  papers report there, so this is a true head-to-head.
- **§3 ByT5-Sanskrit** — the S / L / S+M / L+M / S+L+M ladder. **Now a measured
  head-to-head**: the released `chronbmm/sanskrit5-multitask` was run on these
  same 4,200 sentences and scored through the identical reference and code path
  as cuSHR. cuSHR leads on segmentation (91.52 vs 81.38, though see the
  compound-convention caveat) and loses decisively on lemmatisation
  (65.62 vs 90.55, beyond cuSHR's own 70.07 ceiling). The paper's published
  ladder is retained beside it as literature context only, since it is a
  different corpus.

**Model**: `model95_ctx_ex4200` — `ngrams80 + hybrid_tag + char_bilstm`, 8
epochs, seed 0, CPU. Trained on `cushr_data_g95.npz` (96.61% gold coverage),
104,159 train sentences, with all 4,200 benchmark sentences excluded from
training (§4).

---

## 1. Segmentation — unconstrained system benchmark

**What this section claims.** How these systems perform *as they actually exist*,
at full capability. ByT5-Sanskrit is entitled to its 582M parameters, 6.5B
pretraining tokens and 601,403 fine-tuning sentences; TransLIST to its lexicon
and its own split. Nothing is held back from anyone. cuSHR is presented here as a
**lightweight alternative**, not as a like-for-like replacement.

The separate question — *why* the leaders are ahead, architecture versus data and
pretraining — is a different experiment and is answered in §5, not here.

### Cost of each system

| Model | Params | Sanskrit pretraining | Fine-tune sentences |
|---|---:|---|---:|
| TransLIST | Transformer + SHR lexicon | none | 97,000 |
| ByT5-Sanskrit | 582M | 6.5B tokens | 601,403 |
| **cuSHR (ours)** | **2,771,989** | **none** | 104,159 |

cuSHR's 2.77M is `hybrid_tag` embeddings 2,106,260 + `char_bilstm` encoder
616,576 + biaffine scorer 49,153.

### Accuracy

Recovery of the **unsandhied word forms**: no lemma, no morphological tag.
`eval_surface.py` decodes, maps each predicted node to its surface form, and
compares the sequence to the reference `output` column. P/R/F are
**macro-averaged** (per sentence, then averaged), which is TransLIST's protocol.
Every one of the 4,200 reference sentences is scored, whether or not ingest
resolved a gold path for it.

| Model | P | R | F | **PM** |
|---|---:|---:|---:|---:|
| TransLIST | **98.80** | **98.93** | **98.86** | **93.97** |
| ByT5-Sanskrit | – | – | – | 93.83 |
| **cuSHR (ours)** | **98.36** | **98.33** | **98.32** | **91.52** |
| rcNN-SS | 96.86 | 96.83 | 96.84 | 87.08 |
| FLAT-Lattice | 96.75 | 96.70 | 96.72 | 85.65 |
| Transformer | 96.52 | 96.21 | 96.36 | 83.88 |
| Lattice-GNN | 95.76 | 95.24 | 95.50 | 81.58 |
| TransLIST_ngrams | 96.97 | 96.77 | 96.87 | 79.28 |
| Cliq-EBM | 96.18 | 97.67 | 96.92 | 78.83 |
| Lattice-LSTM | 94.36 | 93.83 | 94.09 | 76.99 |
| TENER | 90.03 | 89.20 | 89.61 | 61.24 |
| SupPCRW | 76.30 | 79.47 | 77.85 | 38.64 |

Baselines from Sandhan et al. (2022) Table 1 and Nehrdich et al. (2024) Table 3.
Sorted by PM.

**Third on PM, third on F, behind only the two systems that top this task.**

Read against the cost table, the result is an efficiency claim rather than an
accuracy one:

| | cuSHR vs ByT5-Sanskrit | cuSHR vs TransLIST |
|---|---|---|
| perfect match | −2.31 (91.52 vs 93.83) = **97.5% of its score** | −2.45 (91.52 vs 93.97) |
| word-level F | – (not reported) | **−0.54** (98.32 vs 98.86) |
| parameters | **1/210th** (2.77M vs 582M) | not comparable — lexicon-driven |
| fine-tuning data | **1/5.8th** (104,159 vs 601,403) | **+7,159 more** than its 97,000 |
| pretraining | **none** vs 6.5B tokens | none either side |

Three things follow, and the third is a limit on the claim:

1. **Words are recovered at nearly the leading rate** — 0.54 F behind TransLIST.
   The larger PM gap is isolated errors scattered across sentences rather than
   systematic mis-segmentation; PM is all-or-nothing per sentence, so a thin
   spread of single-word errors costs far more there than in F.
2. **97.5% of ByT5's perfect match at 1/210th the parameters and no
   pretraining**, served by a deterministic C++/CUDA lattice decoder that needs
   no GPU at inference.
3. **cuSHR does not use less data than everyone.** It trains on 104,159
   sentences, *7,159 more* than TransLIST's 97,000. The data-efficiency claim
   holds against ByT5 only, and stating it more broadly is wrong.

### Test-set identity and contamination — read before quoting the table

Measured by `audit_byt5_overlap.py` against the published split
(`chronbmm/sanskrit-sandhi-split-sighum`), not assumed.

**The test sets are 97% identical, not 100%.** Our `sighum_test_4200.tsv` and the
published SIGHUM test split share **4,075 of 4,200 sentences (97.02%)** after
transliterating to a common scheme; **125 sentences differ on each side** and are
not near-variants — none has even a 0.90-similar counterpart, so they are
different sentences, presumably a different release of the split. Rows above are
therefore compared across test sets overlapping by 97%, not on identical data.
That is close enough for the ranking to be meaningful and too far to call it
exact.

*(Two encoding traps were resolved on the way to that number, both of which
make disjoint corpora look plausible: the published split is IAST where ours is
SLP1 — matching raw gives 19/4,200 — and `ingest.normalize_lemma` strips
avagraha, which only one side went through, costing a further 455.)*

**Contamination is asymmetric and favours ByT5:**

| | cuSHR | ByT5-Sanskrit |
|---|---|---|
| benchmark sentences in **training** | **0 of 4,200** (verified against `splits95_ex4200.json`) | **100 of 4,200 (2.38%)** in the SIGHUM fine-tuning split |
| benchmark sentences in dev | 0 | – |
| **pretraining exposure** | none — cuSHR has no pretraining | **all 4,200, unavoidable** |

ByT5-Sanskrit was pretrained on the entire DCS, which contains the SIGHUM
sentences as raw text. Every test sentence was therefore seen during
pretraining, regardless of how clean the fine-tuning split is. This is **not
quantifiable and not fixable**, it attaches equally to their published 93.83
quoted above, and it is stated here rather than in a caveats section because a
reader who discovers it later will reasonably read it as a concealed flaw.

## 2. The oracle ceiling that used to cap this, and what closed it

An earlier version of this document reported **82.60 PM** and attributed the gap
to an oracle-recall ceiling: 567 of the 4,200 sentences had no resolvable gold
path, scored 23.99 PM, and dragged the headline down ~9 points. It concluded
that closing the gap "requires ingest work, not model work."

That was correct, and the ingest work happened. Three separate things were being
discarded — all documented in `../ingest/INGEST_METHODOLOGY.md` §4e–§4f:

| corpus | gold coverage | test-4200 with no gold path | PM |
|---|---:|---:|---:|
| gold75 (orphan repair only) | 74.99% | 567 (13.5%) | 82.60 |
| gold94 (+ edge reorientation) | 93.89% | 90 (2.1%) | – |
| **g95 (+ key=2 sandhi edges, + lemma families)** | **96.61%** | **4 (0.10%)** | **91.52** |

The ceiling is effectively gone: the 4 remaining unrepresentable sentences score
0.00 PM and cost 0.09 points. The split by gold-path status is now

| group | n | PM |
|---|---:|---:|
| gold path resolved | 4,196 | 91.61 |
| no gold path | 4 | 0.00 |

so the headline and the resolved-subset number have converged (91.52 vs 91.61),
where they previously differed by 9 points. **cuSHR is no longer
oracle-limited on this benchmark**; the distance to TransLIST is now model
quality, not search-space coverage.

## 3. The S/L/M ladder — a measured head-to-head

`eval_slm.py` scores at ByT5-Sanskrit's task levels. A cuSHR lattice node is
`(form, lemma, cng)`, so choosing a path commits to all three at once and the
ladder falls out of how much of each node has to match.

**References are external and cover all 4,200 sentences**: forms from the
published TSV, lemma and `cng` from the DCS `.p` annotations. Neither comes from
cuSHR, and both exist whether or not ingest resolved a path. Verified: the
pickle's flattened `(lemmas, cng)` sequence matches the TSV word count for
4,200/4,200.

**ByT5 has now been run on these same 4,200 sentences** — the released
`chronbmm/sanskrit5-multitask`, decoded on an A100, scored through the identical
`score()` and reference as cuSHR. The column marked *measured* is a genuine
head-to-head; the *paper* column is retained only as literature context and is a
different corpus.

| level | cuSHR | **ORACLE** | **ByT5 (measured, same 4,200)** | ByT5 paper *(DCS 2024, Sen)* |
|---|---:|---:|---:|---:|
| S | **91.52** | 98.00 | 81.38 | 84.61 |
| L | 65.62 | 70.07 | **90.55** | 79.88 |
| L(tol) | 66.05 | 70.64 | **90.74** | – |
| S+M | 45.69 | 68.23 | n/a | 63.86 |
| L+M | 45.45 | 68.18 | n/a | 62.00 |
| S+L+M | 45.29 | 67.97 | n/a | 61.27 |

### L: a clean loss for cuSHR, and the ceiling explains it

**65.62 against 90.55.** ByT5 beats cuSHR's *oracle* (70.07) by more than 20
points, so no amount of better path selection closes this. Both sides are scored
against DCS lemmas and both use DCS conventions, so nothing here is confounded —
this is the architectural limit stated in §2 made quantitative. cuSHR can only
emit a lemma that exists on a reachable lattice node; a sequence model simply
generates the string. If one number in this document argues for the seq2seq
approach over lattice reranking, it is this one.

### S: cuSHR leads by 10 points, and the lead should not be claimed

91.52 against 81.38 — but inspecting all 782 ByT5 failures shows they are
overwhelmingly **compound-boundary conventions, not segmentation errors**:

```
ByT5: droRaputraH      gold: droRa putraH     (did not split)
ByT5: a SaNkamAnaH     gold: aSaNkamAnaH      (over-split)
ByT5: narADipa         gold: nara aDipa       (did not split)
```

| failure mode | share of all 4,200 |
|---|---:|
| same word count, different forms | 13.2% |
| predicted fewer words | 3.1% |
| same characters, different split | 2.0% |
| **predicted more words** | **0.2%** |

Only 0.2% over-generate. The released multitask checkpoint learned **DCS**
compound conventions; this reference uses **SIGHUM's**. Their published 93.83 on
SIGHUM comes from a model *fine-tuned on SIGHUM*, which learned that convention —
this is the off-the-shelf model, so the S column understates it. cuSHR meanwhile
is scored against the convention its pipeline was built around.

**The honest reading: cuSHR is competitive on segmentation and clearly behind on
lemmatisation.** Quoting the S row as a win would be the same error as §1's
earlier data-efficiency overreach.

### M is still unscored

ByT5 emits full UD bundles (`Case=Nom|Gender=Masc|Number=Sing`); the reference is
DCS `cng` integers (`29`, `-153`). Scoring them directly would read as total
tagging failure rather than an unmapped vocabulary, so those rows are `n/a`
rather than a misleading zero.

One useful surprise from the real output: it is **not** the compressed `SNM`
codes of the paper's Figure 1, so `data/sanskrit_tags.tsv` is not needed. The
remaining bridge is `cng` ↔ feature bundle, buildable from the training split
alone.

> **The ByT5 column is a different corpus.** Their ladder is measured on a DCS
> April-2024 split (601,403 sentences, 8,398 test), not SIGHUM; their only
> SIGHUM figure is the 93.83 segmentation PM in §1. Their test split also
> "does not contain any reconstructed forms and [is] therefore strongly biased
> towards Vedic texts" (§5.5). Read the column for category alignment, not as a
> result. The `Sen` column is the right one — their `Par` column feeds
> pseudo-paragraphs of up to 512 characters, a different input granularity from
> our per-sentence decode.
>
> That caveat applies to the **paper** column only. The *measured* column above
> already is a head-to-head — same 4,200 sentences, same reference, same
> `score()`. What §5 would add is a *matched-training* comparison: fine-tuning
> ByT5 on our split would remove the compound-convention mismatch that depresses
> its S column here.

**Reproducing the measured column:**

```bash
# on Lonestar6 -- preflight first, it exits non-zero if any check fails
sbatch --export=ALL,PREFLIGHT=1 byt5_infer.slurm
sbatch --export=ALL,LIMIT=20    byt5_infer.slurm     # 20-sentence self-test
sbatch                          byt5_infer.slurm     # full 4,200

# locally, once byt5_preds_*_sighum_test.jsonl are back
python eval_slm.py --cache ./cache95_ctx_ex4200 --model model95_ctx_ex4200.npz \
    --pred-jsonl byt5_preds_segmentation-lemma-morphosyntax_sighum_test.jsonl \
    --pred-name ByT5
```

The raw predictions are committed, so the table can be re-scored without
re-running the model. Two things the run pinned down that the paper's figures do
not: the output format is `form_lemma_features` (not `lemma_TAG`), and the
features are full UD bundles (not compressed `SNM` codes).

### ORACLE is the column that matters

**ORACLE** is our own gold path scored against the same external reference. No
model is involved. It is the ceiling every cuSHR number sits under, and it is
nowhere near 100:

- **L is capped at 70.07** — but *not* for the reason an earlier version of this
  document gave. It claimed the DCS lemma is absent from the lattice, SHR
  offering only the participial stem (`kfta` where DCS wants `kf`). **That is
  false.** Measured over 300 test sentences, a node carrying DCS's exact
  `(surface, lemma, cng)` exists for **100%** of gold words — the DCS analysis is
  present as the negative-`cng`, unwired orphan node:

  ```
  jIvatsu/jIvat/181   jIvatsu/jIvat/179   jIvatsu/jIv/-10   <- DCS wants jIv/-10
                                          syAt/as/-23       <- DCS wants as/-23
  ```

  The cap is **reachability, not representation**. Restricting the gold DP to
  DCS-matching nodes and asking whether a *connected* path exists gives:

  | level | node exists for every word | connected path exists | reported ORACLE |
  |---|---:|---:|---:|
  | S | 100% | **99.0%** | 98.00 |
  | L | 100% | **70.0%** | 70.07 |
  | M | 100% | 63.0% | – |
  | S+L+M | 100% | 62.7% | 67.97 |

  Path feasibility tracks ORACLE almost exactly, which means **gold-path
  resolution is already extracting everything the lattice permits** — 70.07 is
  not a resolution failure to be tuned away, it is the ceiling. In ~30% of
  sentences the DCS-lemma nodes exist but are not mutually connected, so no
  choice of gold path reaches them.

  L therefore still **measures agreement with SHR's reachable analyses rather
  than lemmatisation ability**, and reporting 65.62 against ByT5's 79.88 as a
  deficit would still be wrong — but the mechanism is edge structure, not a
  missing lemma.

- **S+M is capped at 68.23 while S is capped at 98.00.** Orphan repair
  substitutes a same-surface node without regard to `cng`, so repaired words
  carry SHR's tag rather than DCS's. Most of the gap between S and S+M at the
  oracle is corpus construction, not model error.

> **Methodological note.** The first version of this diagnostic reported S
> feasibility at 48.7% against an ORACLE of 98.00 — a contradiction, since a path
> the gold already follows must be findable. The cause was deriving
> `sources`/`sinks` from the key=1∪key=2 union instead of key=1 alone;
> `ingest.py:765` documents exactly this collapse ("400/400 resolved sentences
> drop to 98/400"). Corrected, the diagnostic reproduces ORACLE and the ceiling
> estimates above are trustworthy. Any future ceiling analysis must build
> endpoints the way `process_corpus` does, not from the traversable graph.

### What the model actually contributes

Reading each level against its own ceiling rather than across the table:

| level | cuSHR / ORACLE |
|---|---:|
| S | 91.52 / 98.00 = **93%** of ceiling |
| L | 65.62 / 70.07 = **94%** |
| S+M | 45.69 / 68.23 = **67%** |
| S+L+M | 45.29 / 67.97 = **67%** |

Segmentation and lemma selection run close to their ceilings; **morphology does
not**. Conditional on getting the segmentation right, the model picks the correct
`cng` for every word in only ~50% of sentences (45.69/91.52), where the gold path
manages ~70% (68.23/98.00). That is a real, unforced model weakness and the
clearest thing to work on next — it is not explained by the corpus.

The tolerant-lemma variants (`L(tol)` 66.05, `S+L+M(tol)` 45.48) match the lemma
through `ingest.lemma_candidates` rather than exactly. They move the numbers by
under half a point, and they scale with how wide the normaliser happens to be, so
the strict figures are the ones quoted above.

## 4. Documented differences between the systems

These are **not** apologies for §1 — §1 is deliberately an unconstrained
benchmark, and the leaders are entitled to their advantages. They are the axes on
which the systems differ, listed so a reader knows what §1 does and does not
control, and so §5 has a checklist of what it removes.

1. **Contamination control.** All 4,200 benchmark sentences are excluded from
   training by `splits95_ex4200.json`. Verified directly against the split file
   actually used: **0 of 4,200 in train**, 0 in dev, 225 in our test split, 3,975
   held out of all splits, no overlap between splits. (`eval_id_list.py`
   reconstructs md5 buckets and does *not* reflect a `--splits-override` run, so
   it is not the right audit for this model.)
2. **Different training data.** SIGHUM's official split is 97k train / 3k dev /
   4.2k test (ByT5 Table 3 lists 99k samples). We train on 104,159 sentences
   from our own 119,503-sentence ingest of the same corpus with the 4,200
   excluded. Comparable in scale — an earlier version of this document claimed we
   saw "~17% less data", which was true of the gold75 model and is now stale.
3. **`cng` ↔ morphosyntactic tag.** The M level assumes SHR's `cng` and DCS's are
   the same scheme. They are: ingest's gold resolution joins on
   `(chunk, lemma, cng)` by literal string comparison and succeeds for 96.6% of
   the corpus, which it could not otherwise. Note this is `cng`, **not** the
   `morph` string in `node_features` — those are not interchangeable (288 morph
   strings collapse onto 168 cng values), which is why `eval_slm.py` reads `cng`
   from the graphml.
4. **Single seed, single configuration.** No variance estimate.

### A note on this document's history

§3 previously scored against **our own reconstructed gold path**, restricted to
the sentences ingest could resolve. That was self-referential and kept only the
easy sentences: it reported S at 92.98 when the unbiased figure was 82.60. Both
faults are fixed above — external reference, all 4,200 scored. The check that
confirms it: `eval_slm.py`'s S and `eval_surface.py`'s PM are now **both 91.52**,
computed by different code paths against the same reference.

## 5. Controlled ablation — where does the remaining gap come from? *(planned)*

§1 answers *which system performs better in the wild*. It cannot answer *why*,
because the leaders differ from cuSHR on three axes at once: architecture,
fine-tuning data (601,403 vs 104,159) and pretraining (6.5B tokens vs none).

The experiment that separates them holds the data fixed and varies only the
model. Three systems, cuSHR's exact split, one scorer:

| system | params | Sanskrit pretraining | fine-tuned on |
|---|---:|---|---|
| cuSHR `model95_ctx_ex4200` | 2.77M | none | our 104,159 |
| `byt5-sanskrit-ft` | 582M | 6.5B tokens | our 104,159 |
| `byt5-base-ft` | 582M | **none** | our 104,159 |

Two quantities fall out:

- **`gap(cuSHR, byt5-base-ft)`** — architecture alone, at matched data with
  neither side pretrained.
- **`gap(byt5-sanskrit-ft, byt5-base-ft)`** — **the value of Sanskrit
  pretraining at matched fine-tuning data.** This number is not in the
  literature, and it is exactly the confound that makes ByT5's published
  segmentation results hard to interpret: its pretraining corpus (GRETIL +
  Sangraha) contains the DCS source texts, so the model has seen the test
  sentences as raw text.

Status: not yet run. Requires fine-tuning two 582M models on Lonestar6
(50–190 A100-hours). ByT5's own repository releases **inference code only** — the
README references a `training/` directory that does not exist — so the recipe is
ours, and it must first be validated by reproducing their published 93.83 on
their own split before any number from this section is trusted. An
under-trained ByT5 would produce a flattering result that is simply wrong.

## Reproducing

```bash
cd ingest
python parallel_ingest.py --shards 24 --workers 4 --resume --keep-shards \
    --repair-orphan-gold --edge-order position \
    --edge-keys 12 --lemma-variants extended --shard-dir <FRESH dir> \
    --out ../data/cushr_data_g95.npz --index ../data/sentence_index_g95.json

cd ../cushr_train
python build_features.py --featurizer ngrams80 --raw ../data/cushr_data_g95.npz \
    --out ../data/g95_ngrams80.npz --vocab-dir ../data --emit-ids --min-count 3
python prepare.py --npz ../data/g95_ngrams80.npz --cache ./cache95_ngrams80 --force
python make_clean_split.py --index ../data/sentence_index_g95.json \
    --raw ../data/cushr_data_g95.npz --out splits95_ex4200.json
python train.py --cache ./cache95_ngrams80 --learned hybrid_tag \
    --encoder char_bilstm --word-dropout 0.1 --epochs 8 --seed 0 \
    --splits-override splits95_ex4200.json \
    --out model95_ctx_ex4200.npz --log log95_ctx_ex4200.json \
    --materialize ../data/g95_ctx_ex4200.npz
python prepare.py --npz ../data/g95_ctx_ex4200.npz --cache ./cache95_ctx_ex4200 --force

python eval_surface.py --cache ./cache95_ctx_ex4200 --model model95_ctx_ex4200.npz \
    --index ../data/sentence_index_g95.json --raw ../data/cushr_data_g95.npz   # §1, §2
python eval_slm.py     --cache ./cache95_ctx_ex4200 --model model95_ctx_ex4200.npz \
    --index ../data/sentence_index_g95.json --raw ../data/cushr_data_g95.npz   # §3
```

`--materialize` is required: without it the learned embedding tables are never
persisted, `model*.npz` holds only the scorer, and both eval scripts fail with a
96-vs-80 dimension mismatch.

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
- Nehrdich, Hellwig, et al. *One Model is All You Need: ByT5-Sanskrit, a Unified
  Model for Sanskrit NLP Tasks.* Findings of EMNLP 2024. arXiv:2409.13920.
  Tables 3 and 7, §5.5.
