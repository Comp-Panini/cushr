# cuSHR vs TransLIST and ByT5-Sanskrit

- **TransLIST** — word-level segmentation on the SIGHUM 4,200 test set. 
- **ByT5-Sanskrit** — the S / L / S+M / L+M / S+L+M ladder. The released `chronbmm/sanskrit5-multitask` was run on these
  same 4,200 sentences and scored through the identical reference and code path
  as cuSHR. 

**Model**: `model95_ctx_ex4200` — `ngrams80 + hybrid_tag + char_bilstm`, 8
epochs, seed 0, CPU. Trained on `cushr_data_g95.npz` (96.61% gold coverage),
104,159 train sentences, with all 4,200 benchmark sentences excluded from
training (§4).

---

## 1. Segmentation

### Cost of each system

| Model | Params | Sanskrit pretraining | Fine-tune sentences |
|---|---:|---|---:|
| TransLIST | Transformer + SHR lexicon | none | 97,000 |
| ByT5-Sanskrit | 582M | 6.5B tokens | 601,403 |
| cuSH | 2,771,989 | none | 104,159 |

cuSHR's 2.77M is `hybrid_tag` embeddings 2,106,260 + `char_bilstm` encoder
616,576 + biaffine scorer 49,153.

### Accuracy

Recovery of the unsandhied word forms: no lemma, no morphological tag.

| Model | P | R | F | PM |
|---|---:|---:|---:|---:|
| TransLIST | 98.80 | 98.93 | 98.86 | 93.97 |
| ByT5-Sanskrit | – | – | – | 93.83 |
| cuSHR | 98.36 | 98.33 | 98.32 | 91.52 |
| rcNN-SS | 96.86 | 96.83 | 96.84 | 87.08 |
| FLAT-Lattice | 96.75 | 96.70 | 96.72 | 85.65 |
| Transformer | 96.52 | 96.21 | 96.36 | 83.88 |
| Lattice-GNN | 95.76 | 95.24 | 95.50 | 81.58 |
| TransLIST_ngrams | 96.97 | 96.77 | 96.87 | 79.28 |
| Cliq-EBM | 96.18 | 97.67 | 96.92 | 78.83 |
| Lattice-LSTM | 94.36 | 93.83 | 94.09 | 76.99 |
| TENER | 90.03 | 89.20 | 89.61 | 61.24 |
| SupPCRW | 76.30 | 79.47 | 77.85 | 38.64 |


### Test-set identity and contamination

`sighum_test_4200.tsv` and the
published SIGHUM test split share 4,075 of 4,200 sentences (97.02%) after
transliterating to a common scheme. Rows above are
therefore compared across test sets overlapping by 97%, not on identical data.


**Contamination is asymmetric and favours ByT5:**

| | cuSHR | ByT5-Sanskrit |
|---|---|---|
| benchmark sentences in training | 0 of 4,200 | 100 of 4,200 (2.38%) in the SIGHUM fine-tuning split |
| benchmark sentences in dev | 0 | – |
| pretraining exposure | none — cuSHR has no pretraining | all 4,200, unavoidable |

ByT5-Sanskrit was pretrained on the entire DCS, which contains the SIGHUM
sentences as raw text. Every test sentence was therefore seen during
pretraining, regardless of how clean the fine-tuning split is. 

## 2. The oracle ceiling that used to cap this, and what closed it

| corpus | gold coverage | test-4200 with no gold path | PM |
|---|---:|---:|---:|
| gold75 (orphan repair only) | 74.99% | 567 (13.5%) | 82.60 |
| gold94 (+ edge reorientation) | 93.89% | 90 (2.1%) | – |
| g95 (+ key=2 sandhi edges, + lemma families) | 96.61% | 4 (0.10%) | 91.52 |

The split by gold-path status is now

| group | n | PM |
|---|---:|---:|
| gold path resolved | 4,196 | 91.61 |
| no gold path | 4 | 0.00 |

## 3. The S/L/M ladder — a measured head-to-head

`eval_slm.py` scores at ByT5-Sanskrit's task levels. A cuSHR lattice node is
`(form, lemma, cng)`, so choosing a path commits to all three at once and the
ladder falls out of how much of each node has to match.

ByT5 has now been run on these same 4,200 sentences — the released
`chronbmm/sanskrit5-multitask`, decoded on an A100, scored through the identical
`score()` and reference as cuSHR. The column marked *measured* is a
head-to-head omparison.

The 'paper' column is only a literature context and is a
different corpus.

| level | cuSHR | +maps | +maps +rerank | ORACLE +maps | ByT5 (measured, same 4,200) | ByT5 paper *(DCS 2024)* |
|---|---:|---:|---:|---:|---:|---:|
| S | **91.52** | 91.52 | **92.60** | 98.00 | 81.38 | 84.61 |
| L | 65.62 | 85.40 | **86.81** | **91.75** | 90.55 | 79.88 |
| S+M | 45.69 | 52.26 | **61.21** | 77.65 | **67.79** | 63.86 |
| L+M | 45.45 | 52.38 | **61.43** | 78.05 | **76.50** | 62.00 |
| S+L+M | 45.29 | 51.95 | **60.98** | 77.50 | **66.90** | 61.27 |

**+maps** = the two convention tables described below, both derived from the
training split only and applied to cuSHR's output. They translate SHR's
analytical vocabulary into DCS's; they do not change which node the model
picked. ByT5 needs no such translation — it emits DCS conventions natively — so
its column is unaffected by them.

**+rerank** = a BiLSTM rescoring the top 16 lattice paths, trained on the
training split. Unlike the maps this *does* change which node is chosen. The
two are complementary and additive on M (+6.57 then +8.95), because they fix
different things: vocabulary and ranking. Together they take S+L+M from 45.29
to 60.98, against the ByT5 paper's 61.27 on its own corpus.

M is scored by translating ByT5's UD bundles into DCS `cng`
(`build_tag_bridge.py`), so all three columns sit in the reference's own space
and neither the reference nor cuSHR is weakened to make the comparison possible.
The direction was chosen by measurement: `bundle -> cng` is 96.69% pure where
`cng -> bundle` is 93.18%, because DCS's negative `cng` are underspecified —
one value such as -190 spans many case/gender/number bundles. The bridge is
built from 1,153 exact-word-aligned **training** sentences; 6.3% of its tokens
fall in bundles under 90% pure, so ByT5's M carries a point or so of bridge
noise that cuSHR's, scored natively, does not.

### L: cuSHR loses, but it is a convention mismatch, not an architectural limit

An earlier version of this section called the L gap "the architectural limit made
quantitative" and said "no amount of better path selection closes this."
**Measurement contradicts both claims.** What follows replaces them.

**First, a framing correction: ByT5 has no oracle.** 90.55 is its actual score.
70.07 is *cuSHR's ceiling*. So the comparison is ByT5's real performance against
the best cuSHR could do — which is why the gap looked structural.

**The per-word gap is small; sentence-level perfect match magnifies it.**

| | word-level lemma accuracy | sentences |
|---|---:|---:|
| cuSHR oracle | **94.83%** | 26,982 / 28,454 |
| ByT5 | **99.34%** | 26,797 / 26,975 |

4.5 points per word, not 20. Perfect match is all-or-nothing over a mean of
**6.78 words**, so the per-word rate compounds:

```
cuSHR oracle   0.9483 ^ 6.78 = 0.698     (observed 0.701)
ByT5           0.9934 ^ 6.78 = 0.956
               x (1 - 226/4200 length mismatches) = 0.905   (observed 0.9055)
```

Both predictions land within 0.005 of the measured values, so **the entire
20-point sentence-level gap is a 4.5-point per-word gap raised to the 6.78th
power.** That also explains ByT5's own 226 unscorable sentences: when it splits
differently from the reference, the lemma sequence length disagrees and the
sentence is lost outright.

**The disagreements are one regular pattern.** SHR gives the participial stem
where DCS gives the verbal root:

| SHR | DCS | count | | SHR | DCS | count |
|---|---|---:|---|---|---|---:|
| `ukta` | `vac` | 73 | | `sTita` | `sTA` | 38 |
| `gata` | `gam` | 67 | | `mfta` | `mf` | 26 |
| `kfta` | `kf` | 57 | | `jAta` | `jan` | 26 |
| `smfta` | `smf` | 45 | | `Sruta` | `Sru` | 18 |
| `yukta` | `yuj` | 42 | | `dfzwa` | `dfS` | 16 |

1,472 disagreeing words across 522 distinct pairs — and **4,424 of 4,484 SHR
lemma types (98.7%) map to exactly one DCS lemma.** The correspondence is
essentially a function, not a genuine ambiguity.

**So it is fixable by a lookup applied at output time.** Building the rewrite
table from the **training split only** (25,000 sentences, no test data touched),
keeping entries seen ≥3 times with ≥90% consistency, gives **589 rules** —
`smfta→smf`, `jita→ji`, `Binna→Bid`, `Bavizyat→BU`. Applied to the test set:

| ORACLE lemma | before | after | ByT5 |
|---|---:|---:|---:|
| word-level | 94.83 | **98.38** | 99.34 |
| sentence PM | 70.07 | **89.80** | 90.55 |

**+19.73 points, closing all but 0.75 of the gap to ByT5** — with no change to
the lattice, the model, or the search. The L deficit was cuSHR reporting SHR's
lemma convention while being scored against DCS's.

**Measured on the model, not just the ceiling.** `--lemma-map` gives cuSHR
L 65.62 → 84.00 and ORACLE 70.07 → 89.80. Sanity check: forcing the same table
onto ByT5 fires on 0.05% of its tokens and *costs* it 0.12 points, confirming it
encodes SHR's convention specifically rather than generically better lemmas.

**But the lemma table alone could not touch M** — it rewrites a *string* and
cannot change which node the path selected, so the tag stayed nominal and S+M
sat unmoved at 45.69. Since 93% of tag errors coincide with a lemma error and
100% of those want a negative `cng`, the two fields disagree *together* and want
one table over the pair.

### The joint convention table, and why the two compose

`build_convention_map.py` keys on **(SHR lemma, SHR cng) → (DCS lemma, DCS cng)**.
Measured on training data, the pair is *more* functional than the lemma alone,
because `cng` disambiguates the key:

```
SHR lemma        -> DCS lemma          99.26% functional over  4,748 keys
(SHR lemma, cng) -> (DCS lemma, cng)   99.75% functional over 10,259 keys
```

Typical rules — both fields moving together:

```
(ukta,  71)  ->  (vac, -190)        (Binna, 3)   ->  (Bid, -190)
(jita,  30)  ->  (ji,  -190)        (Sita, 101)  ->  (SA,  -190)
```

**The two tables compose by backoff rather than replacing each other**, and the
reason is count fragmentation. Keying on the pair splits a lemma's occurrences
across its `cng` values, so `kfta` — common overall — can fall below the ≥3
support threshold for every individual pair. Built on 6,000 sentences the joint
table therefore carried only 152 lemma rewrites against the lemma table's 589,
and **L fell from 84.00 to 73.83 even as M rose**. Backoff takes the joint rule
where the pair had support and the lemma-only rule otherwise:

| level | base | lemma only | joint only | **backoff** |
|---|---:|---:|---:|---:|
| L | 65.62 | 84.00 | 73.83 | **85.40** |
| S+M | 45.69 | 45.69 | 52.26 | **52.26** |
| S+L+M | 45.29 | 45.38 | 51.86 | **51.95** |

Backoff wins everywhere, and L exceeds the lemma-only table (85.40 vs 84.00)
because joint rules repair lemmas the lemma-only table missed.

**What this does and does not buy.** It is the first thing to move M at all:
**+6.57 on the model, +9.53 on the oracle**, and ORACLE L (91.75) now exceeds
ByT5's 90.55. But it translates the *reported analysis*, not the *node chosen* —
legitimate here because "nominative of the participial stem `dfzwa`" and "past
passive participle of `√dfS`" are the same claim in two traditions, not two
competing claims. It does nothing for the 74.3% of model errors that are
wrong-node syncretism, and the model↔oracle gap actually **widened** (52.26 vs
77.65) because the ceiling rose faster than the model did. That residual is the
scorer's problem, not the vocabulary's.

**Caveat on the size of the table.** The joint map is built on 6,000 training
sentences where the lemma map used 25,000 — two larger builds were killed by the
environment. More data yields more joint rules, so these are a **floor**, not a
converged number.

### Reranking: +5.10 on S+M, and why counts were skipped

The convention tables closed the *vocabulary* gap and left the ratios at 93 /
93 / 67 / 67, isolating the rest as the scorer's. Three measurements then
located it, in order of how much each one settled.

**1. The beam already contains the answer.** `kbest.py` decodes exact K-best
paths (K=1 reproduces `viterbi()` on 5,681/5,681 sentences; the top-K score
multiset matches exhaustive enumeration on 60 small lattices). Recall@k asks
whether the correct analysis is *anywhere* in the top k -- a hard bound on any
reranker:

| level | @1 | @8 | @16 | @64 | ORACLE |
|---|---:|---:|---:|---:|---:|
| S | 91.52 | 96.71 | 97.74 | **98.69** | 98.00 |
| L | 85.40 | 91.40 | 92.33 | **93.10** | 91.75 |
| S+M | 52.26 | 70.95 | 73.67 | **76.52** | 77.65 |

S+M recall@64 reaches 76.52 of a 77.65 ceiling. **The failure is ranking, not
coverage.** S and L recall@64 exceed the oracle outright: the beam holds
analyses better than our own gold path.

**2. Role counts cannot reach it.** The obvious reranker scores role counts
(nominatives, finite verbs) plus the base score. Measured before being built:
of the 21.40 points of headroom at K=16, only **20.1%** is separable by role
counts -- elsewhere a correct candidate carries the same count vector as an
incorrect one in the same beam, and no weighting breaks a tie. Ceiling 56.57.

**3. A sequence model beats that ceiling.** `rerank.py` embeds each word's
morph tag and lemma, runs a BiLSTM over the candidate path and rescores, so
`nom acc verb` and `acc nom verb` differ where the counts are identical.

| level | base | +reranker | recall@16 | ORACLE |
|---|---:|---:|---:|---:|
| S | 91.52 | **92.60** | 97.74 | 98.00 |
| L | 85.40 | **86.81** | 92.33 | 91.75 |
| S+M | 52.26 | **61.21** | 73.67 | 77.65 |
| S+L+M | 51.95 | **60.98** | 73.60 | 77.50 |

**61.21 far exceeds the 56.57 that counts could ever have reached**, which is
the result that justifies the architecture rather than merely reporting it. The
reranker moves top-1 on 20.5% of sentences, and closes **54%** of the gap
between the base decoder and its own recall@16 ceiling.

Two ablations, both decided on dev and confirmed on test rather than argued:

| variant | params | dev top-1 | test gold-path gain |
|---|---:|---:|---:|
| morph + lemma, 14 epochs | 865,667 | **74.82** | **+5.77** |
| morph + lemma, 6 epochs | 865,667 | 73.03 | +3.96 |
| morph only, 6 epochs | 77,443 | 72.18 | +2.89 |

Training length mattered more than anything else: six epochs looked converged
and was not, and the extra eight are worth **+3.85 S+M** (57.36 -> 61.21). The
lemma branch is worth its 11x parameter cost, but only after epoch 3 -- for the
first two epochs the morph-only model tracks it to two decimal places, so an
ablation stopped early would have concluded the opposite.

Three properties of the setup that make the number trustworthy:

- It is initialised as an **exact copy of the base decoder** (`w_base` = 1, the
  BiLSTM projection zeroed), asserted at epoch 0, so every point is measured
  against a real baseline. `|ctx|`, the mean absolute contribution of the
  BiLSTM branch, is logged each epoch -- a reranker that collapsed to "trust
  the base score" would show a falling loss and `|ctx|` at 0.
- It trains on the **train split's own candidate lists**, which is legitimate
  here because the base scorer has not memorised them: train recall@1 exceeds
  dev by 1.61 points. (An earlier probe reported 4.02 and was wrong -- it
  truncated each split to its lowest sentence ids, which are different source
  texts. Random sampling is the fix.)
- It is trained against **gold-path match**, not DCS. The reranker's job is to
  choose nodes; the convention maps translate them afterwards. The number above
  is nonetheless real S+M against the external reference, through the unchanged
  scorer.

Its ceiling is recall@16 = 73.67, not the ORACLE -- a reranker can only choose
among candidates the base decoder produced. Raising K, or moving the structure
into the scorer so it never proposes those candidates, is where the remaining
16 points live.

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

**The honest reading: both raw columns are convention artefacts pointing in
opposite directions.** cuSHR's S lead comes from being scored against SIGHUM's
compound convention, which its pipeline was built around; ByT5's L lead comes
from cuSHR reporting SHR's lemma convention while being scored against DCS's
(and a train-derived rewrite closes 19.73 of those 20 points — see above).
Quoting either raw column as a win would be the same error as §1's earlier
data-efficiency overreach.

### Removing the convention confound: the agreement subset

The S column above penalises ByT5 for a convention difference rather than an
error, and that penalty propagates into L and M — when the two systems split a
compound differently the word sequences stop corresponding, and the sentence is
lost outright at every level. The clean fix would be a SIGHUM-fine-tuned
checkpoint, but **none was ever released**: `chronbmm/sanskrit5-multitask` is
the multitask model, `chronbmm/byt5-sanskrit-analyzer-hackathon` targets the
*other* benchmark, and the per-dataset checkpoints behind the paper's Table 3
were not published.

So instead: score **both systems on the 3,418 of 4,200 sentences (81.4%) where
ByT5's segmentation already equals the reference**, with
`eval_slm.py --restrict-agreeing-seg`.

| level | cuSHR | cuSHR +maps | ORACLE | ORACLE +maps | ByT5 |
|---|---:|---:|---:|---:|---:|
| S | 94.21 | 94.21 | 99.24 | 99.24 | *100.00 — by construction* |
| L | 67.26 | **86.34** | 71.14 | 92.01 | **95.99** |
| S+M | 47.28 | **54.39** | 69.47 | 79.45 | **83.29** |
| L+M | 46.93 | **54.24** | 69.20 | 79.36 | **82.21** |
| S+L+M | 46.90 | **54.10** | 69.20 | 79.27 | **82.21** |

> **Read this table with three caveats.**
>
> 1. **ByT5's S of 100.00 is a tautology, not a result.** The subset is
>    *defined* as the sentences where its segmentation matches, so that cell
>    cannot be anything else. It is shown only to make the construction visible;
>    it must never be quoted.
> 2. **The subset is biased easy for both systems.** cuSHR's own numbers rise on
>    it (S 91.52 → 94.21, L 65.62 → 67.26, L+maps 85.40 → 86.34) purely because these are the
>    shorter, less compound-heavy sentences. Absolute values here are not
>    comparable to the full-set table above.
> 3. **The comparison between the two systems is still fair**, because both are
>    scored on the identical 3,418 sentences with the identical reference and
>    scorer. Only the cross-table comparison is invalid.

**What it shows.** Removing the convention penalty makes ByT5 *better*, not
worse — confirming the S column in the full table understates it. Conditional on
both systems seeing the same segmentation, ByT5 leads on lemma (95.99 vs 86.34)
and decisively on joint analysis (**82.21 vs 54.10** at S+L+M).

The sharpest number here: **ByT5's 82.21 still exceeds cuSHR's own ORACLE of
79.27.** On these sentences a perfect cuSHR path — one that selected the best
node available in the lattice at every position — would still lose to ByT5 on
joint segmentation, lemma and morphology. That is the clearest evidence in this
document for the sequence-generation approach over lattice reranking, and it is
not explained by conventions, contamination, or the scorer.

*Revised down by the convention tables.* Before them this gap read as 13 points
(82.21 vs an ORACLE of 69.29); it is now **2.9**. Most of what looked like a
ceiling deficit was untranslated vocabulary, and the remainder is a genuine but
much narrower lattice-reachability limit. The model↔ceiling gap on this subset
is far larger than the ceiling↔ByT5 gap — 25.2 points against 2.9 — which puts
the burden on the scorer, not the lattice.

### How M is scored across two tagsets

ByT5 emits full UD bundles (`Case=Nom|Gender=Masc|Number=Sing`); the reference
is DCS `cng` integers (`29`, `-153`). Compared directly these agree on nothing,
which would read as total tagging failure rather than an unmapped vocabulary —
so M was `n/a` until a validated bridge existed. `build_tag_bridge.py` builds
one and the M rows above are its output.

Three decisions in it are worth stating, because each could have quietly
distorted the numbers:

- **Direction, chosen by measurement.** `bundle -> cng` is 96.69% pure;
  `cng -> bundle` is 93.18%. The latter fails because DCS's *negative* `cng` are
  underspecified — `-190` ("participle read verbally") legitimately spans many
  case/gender/number bundles. Split by sign, `cng -> bundle` is 96.80% pure on
  positive `cng` and only 74.53% on negative.
- **Mapping into `cng`, not out of it.** The reference and cuSHR keep their
  native representation, so their published numbers stay valid and only ByT5 is
  translated. Mapping everything into bundle space would instead have collapsed
  distinct `cng` together and made M easier for *both* systems.
- **Alignment by exact word match, not word count.** Equal counts with different
  splits pair the wrong tag with the wrong `cng`. Requiring the words to match
  lifted purity from 87.52% to 93.18%.

Built from 1,153 exact-aligned **training** sentences. 6.3% of its tokens sit in
bundles under 90% pure, so ByT5's M carries roughly a point of bridge noise that
cuSHR's — scored natively — does not.

A useful surprise from the real output: it is **not** the compressed `SNM` codes
of the paper's Figure 1, so `data/sanskrit_tags.tsv` was never needed.

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

  L therefore **measures agreement with SHR's reachable analyses rather than
  lemmatisation ability** — the mechanism is edge structure, not a missing lemma.

  **This ceiling is real but it is not binding**, which §3's L subsection
  establishes: rewrite tables derived from the training split alone lift the
  oracle from 70.07 to **91.75** by translating SHR's participial stems into
  DCS's verbal roots at output time. The unreachable-node ceiling caps what the
  *lattice* can express in SHR's own convention; it does not cap what the system
  can *report* once that convention is mapped.

- **M is capped at ~68 while S is capped at 98.00 — and it is now the binding
  constraint.** (This bullet diagnoses the untranslated ceiling; §3's joint
  table then acts on the diagnosis and raises it to 77.65. The analysis below is
  what made that fix findable, so it is kept as measured.) An earlier version
  blamed orphan repair ("substitutes a
  same-surface node without regard to `cng`"). **Measured, that is false:
  0.0% of the 1,544 disagreeing words sit on a `position = -1` orphan node.**

  The real cause is node selection, and it is the *same* disagreement that
  drives the L gap. Per-word, our gold path's `cng` agrees with DCS 94.57% of
  the time, which compounds to 0.9457^6.78 = 0.685 against an observed 68.76 —
  so M behaves exactly like L, a small per-word rate raised to the sentence
  length. The disagreements are one-directional:

  | ours | DCS | n | | ours | DCS | n |
  |---|---|---:|---|---|---|---:|
  | 71 | **-190** | 293 | | 30 | **-190** | 60 |
  | 29 | **-190** | 268 | | 29 | **-10** | 47 |
  | 3 | **-190** | 132 | | 80 | **-190** | 45 |
  | 71 | **-210** | 81 | | 75 | **-190** | 37 |

  DCS wants a **negative** `cng` — its code for a participle read verbally —
  where our path selects the positive `cng` of the nominal reading. Crosstabbing
  lemma against tag makes the identity explicit:

  | | words | share |
  |---|---:|---:|
  | both lemma and `cng` wrong | 1,438 | 5.05% |
  | only `cng` wrong | 106 | 0.37% |
  | only lemma wrong | 34 | 0.12% |
  | both right | 26,876 | 94.45% |

  **93% of tag errors coincide with a lemma error, and 100% of those want a
  negative `cng`.** L and M are not two problems. They are one decision — which
  node the path selects — surfacing in two metrics.

  **This is what the joint `(lemma, cng)` table in §3 was built from**, and it
  is why a lemma-only table could not move M: the two fields fail together, so
  they had to be rewritten together. Acting on it took ORACLE S+M from 68.23 to
  **77.65** and the model from 45.69 to **52.26** — the first change in this
  document to move M at all.

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

| level | cuSHR / ORACLE | with the convention tables |
|---|---:|---:|
| S | 91.52 / 98.00 = **93%** | 91.52 / 98.00 = **93%** |
| L | 65.62 / 70.07 = **94%** | 85.40 / 91.75 = **93%** |
| S+M | 45.69 / 68.23 = **67%** | 52.26 / 77.65 = **67%** |
| S+L+M | 45.29 / 67.97 = **67%** | 51.95 / 77.50 = **67%** |

Segmentation and lemma selection run close to their ceilings; **morphology does
not**. Conditional on getting the segmentation right, the model picks the correct
`cng` for every word in only ~57% of sentences (52.26/91.52), where the gold path
manages ~79% (77.65/98.00). That is a real, unforced model weakness and the
clearest thing to work on next — it is not explained by the corpus.

**The tables do not change that diagnosis; they sharpen it.** The right-hand
column is the load-bearing observation: translating conventions raises floor and
ceiling by almost exactly the same factor, leaving the fraction-of-ceiling
figures unmoved at 93 / 93 / 67 / 67. So the convention gap and the model gap
are **independent, multiplicative** deficits. Fixing vocabulary was worth +6.6
absolute points on M and buys nothing further; the remaining 33% is entirely the
scorer's, and closing it is now worth **+25.4 points** on S+M rather than the
+22.5 estimated before the tables existed.

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

# §3 convention tables -- all three derive from splits95_ex4200.json's TRAIN ids
# only, and each asserts dev/test membership is absent before writing.
python build_lemma_map.py      --sample 25000 --out lemma_map.json
python build_convention_map.py --sample  6000 --out convention_map.json
python build_tag_bridge.py     --out tag_bridge.json

# the +maps columns, and the ByT5 head-to-head in the same run
python eval_slm.py --cache ./cache95_ctx_ex4200 --model model95_ctx_ex4200.npz \
    --pred-jsonl byt5_preds_segmentation-lemma-morphosyntax_sighum_test.jsonl \
    --pred-name ByT5 --lemma-map lemma_map.json \
    --convention-map convention_map.json --tag-bridge tag_bridge.json
# add --restrict-agreeing-seg for the 3,418-sentence agreement subset

# the +rerank column. Data dump is ~25 min, training ~3 min/epoch on CPU.
python test_kbest.py --k 64                     # verify the k-best decoder first
python make_rerank_data.py --k 16               # candidate lists + gold-path labels
python rerank.py --epochs 14 --patience 3 --out reranker_long.pt
python eval_slm.py --cache ./cache95_ctx_ex4200 --model model95_ctx_ex4200.npz     --lemma-map lemma_map.json --convention-map convention_map.json     --kbest 16 --rerank reranker_long.pt
```

Run these one at a time. Two concurrent CPU jobs of this size get killed in
this environment, and so does anything heavy launched in the background.

`--convention-map` and `--lemma-map` **compose by backoff, and both should be
passed**: the joint rule wins where the `(lemma, cng)` pair had support and the
lemma-only rule applies otherwise. Passing the joint map alone scores *worse* on
L than the lemma map alone (73.83 vs 84.00) — see §3.

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
