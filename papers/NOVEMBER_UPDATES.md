# cuSHR — November 2026 updates

**Headline: 94.43 perfect match / 98.94 word F1 (macro) on SIGHUM segmentation,
from a two-model ensemble. TransLIST is 93.97 / 98.86 on the identical 4,200
sentences.** That is +1.00 PM over our previous best (93.43) and puts cuSHR
first on the published benchmark. The gain over our own single models is
statistically solid. The lead over TransLIST is real on paper but inside
sampling noise; see §5 before quoting it.

Everything below was measured on Nov 10, 2026, on the SIGHUM test set (4,200
sentences, the published split, the same DCS-IDs and gold as TransLIST's; see
`check_testset_overlap.py`).

---

## 1. Segmentation standings

| System | PM | Word F1 macro | Word F1 micro | source |
|---|---:|---:|---:|---|
| **cuSHR-ensemble (seg + attn)** | **94.43** | **98.94** | **99.02** | `eval_union_sighum_k16.json` |
| TransLIST | 93.97† | 98.86† | — | Sandhan et al. 2022, Table 1 |
| ByT5-Sanskrit, fine-tuned on SIGHUM | 93.83† | — | — | Nehrdich et al. 2024, Table 3 |
| cuSHR-seg + reranker (previous best) | 93.43 | 98.73 | 98.85 | same run, single-model row |
| cuSHR-attn + reranker | 93.36 | 98.84 | 98.91 | same run, single-model row |
| cuSHR-lm + reranker | 92.87 | — | — | `eval_lm_ex4200_rr_maps_s*.json`, 3-seed mean |
| cuSHR-seg (no reranker) | 92.76 | 98.42 | — | `COMPARING_MODELS.md` |

† reported in the paper, not run by us. Macro F1 (per sentence, then averaged)
is TransLIST's protocol.

The two single-model rows are the three rerankers of each model combined
(probabilities averaged), as the ensemble uses them. They come out equal to the
previously reported 3-seed means (93.43, 93.36), which is also the check that
this run reproduces the earlier ones.

---

## 2. What the model is

The ensemble has no new architecture. It is two cuSHR models that already
existed, each with its three rerankers, combined at the very end. Neither
model and none of the six rerankers was retrained for this.

### The two base models

Both are the standard cuSHR three-stage stack: a learned featurizer over SHR's
candidate words, a contextual encoder, and a first-order biaffine edge scorer,
decoded by Viterbi over the SHR lattice. They differ **only in the encoder**.
They share the training data (104,159 sentences, all 4,200 benchmark sentences
held out) and the objective (`--cost surface --gold-target surface`: only a
wrong segmentation word costs anything, the model is never charged for lemma
or tag).

| | cuSHR-seg | cuSHR-attn |
|---|---|---|
| file | `model95_seg_ex4200.npz` | `model95_lattice_attn_char_s0.npz` |
| encoder | character BiLSTM, 2 layers, hidden 128 | the same char BiLSTM **plus** lattice self-attention: 8 heads, 2 blocks, d_model 128, four-position span bias |
| what the encoder sees | the sandhi-fused surface text, left to right | the surface text, **plus** every candidate word attending to every other candidate word, weighted by how their character spans overlap |
| encoder params | 618,624 | 925,056 |
| featurizer + scorer params | 2,170,485 | 2,170,485 |
| **total** | **2,789,109** | **3,095,541** |
| SIGHUM PM alone, base | 92.76 | 92.45 |

### The rerankers (three per base model, six in all)

`rerank.py`, the same design for both models. **865,844 params each.** Each
reranker takes the base model's top-16 *distinct segmentations* and rescores
each one as a whole path:

- a BiLSTM over the path's sequence of (lemma, morph tag) embeddings,
- eight segmentation features with no word identity in them (word count; how
  often each word was a gold word in training; counts of unseen, rare and short
  words; the candidate's base score relative to the best in the beam),
- the base Viterbi score itself, with weight 1.

Trained listwise on candidate lists for 30,000 training sentences, checkpoint
chosen by dev loss, seeds 0/1/2.

### Total size

| component | params |
|---|---:|
| cuSHR-seg | 2,789,109 |
| cuSHR-attn | 3,095,541 |
| 6 rerankers × 865,844 | 5,195,064 |
| **ensemble total** | **11,079,714** |

About **11.1M parameters with no pretraining**. ByT5-Sanskrit has 582M
(~52×) and was pretrained on 6.5B tokens. TransLIST's parameter count is not
reported.

---

## 3. How the two are combined

The hard part is that the two models' Viterbi scores are on unrelated scales.
A score of 40 from cuSHR-seg means nothing next to a score of 40 from
cuSHR-attn, so the beams cannot simply be merged and re-sorted. The fix is to
turn each model's beam into probabilities first (`eval_union.py`):

1. Each model decodes its own **top-16 distinct segmentations** of the
   sentence.
2. Each of its three rerankers scores those 16. The scores are softmaxed
   **within that model's beam**, and the three rerankers' probabilities are
   averaged. Each model now holds a probability distribution over its own
   candidates.
3. A candidate is identified by its **word sequence**, which is exactly what
   the metric scores. If both models propose the same segmentation (even
   through different lattice nodes) it is one candidate. A segmentation a model
   never proposed gets probability 0 from that model.
4. The ensemble score is the **plain mean of the two probabilities** (the sum
   rule), and the highest-scoring segmentation is the output.

**The recipe was fixed in advance.** K=16, all three rerankers per model and
equal weights were chosen on the g95 split before SIGHUM was run, and nothing
was adjusted after seeing the 94.43.

---

## 4. Why it gained a whole point

### The two models are wrong on different sentences

cuSHR-seg and cuSHR-attn score within 0.07 PM of each other, but they do not
make the same mistakes:

| on the 4,200 test sentences | sentences |
|---|---:|
| both right | 3,823 |
| **only cuSHR-seg right** | **101** |
| **only cuSHR-attn right** | **98** |
| both wrong | 178 |

On 199 sentences exactly one of the two models is right. Each single model
gets only its own ~100 of those. The ensemble gets **140 of the 199**, and
loses **none** of the 3,823 that both already had right. It also recovers 3 of
the 178 where both were wrong (both had the correct answer somewhere in their
beams, just not ranked first).

The arithmetic: 3,823 + 140 + 3 = 3,966 correct = **94.43%**, against cuSHR-seg's
3,823 + 101 = 3,924 = 93.43%. That +42 sentences is the whole point.

### Their beams barely overlap

Over all sentences, the two top-16 lists share **8,388** candidates. Another
**3,468** appear only in cuSHR-seg's list and **4,467** only in cuSHR-attn's.
With both lists combined, the correct segmentation is present for **98.24%**
of sentences, against 97.50% for cuSHR-seg's list alone. So the union
both reaches more sentences and, more importantly, lets two independent votes
settle the close calls.

### Why the errors differ: what attention added

cuSHR-attn alone did *not* improve on cuSHR-seg (92.45 vs 92.76 base, 93.36 vs
93.43 reranked). In October that looked like a dead end. What it did change is
*which* sentences the model gets right. The BiLSTM reads the sentence as a
left-to-right character string. The attention layer additionally lets each
candidate word look at every other candidate word in the lattice, weighted by
how their spans sit relative to each other (start-start, start-end, end-start,
end-end distances). That is a different view of the same evidence, so it fails
in different places. Each model's rerankers are confident where its own view
is informative, and averaging the two probability distributions picks the more
confident one where they disagree.

That is the new ingredient: **not a better single model, but a second model
that is about as accurate and wrong somewhere else**, plus a way (per-beam
softmax over reranker scores) to put two incompatible scorers on a common scale.

It matches the g95 prediction almost exactly: +1.02 there (93.94 vs 92.92, on
the reranker's own label metric), +1.00 here.

---

## 5. Is 94.43 reliable?

### What is solid

**The number is reproduced and the inputs are verified.** The attention
model's lattice cache was rebuilt locally from its training checkpoint. The
rebuilt model's weights are bit-for-bit identical to the cluster's, and both
single-model rows reproduce their published values exactly (93.43, 93.36).

**The gain over our own single models is statistically significant.** It is
measured on the same sentences, so a paired test applies
(`union_significance.py`):

| ensemble vs | only ensemble right | only other right | net | exact McNemar p |
|---|---:|---:|---:|---:|
| cuSHR-seg + reranker | 71 | 29 | +42 | 3.2 × 10⁻⁵ |
| cuSHR-attn + reranker | 75 | 30 | +45 | 1.3 × 10⁻⁵ |

Paired bootstrap 95% CI on the gain over cuSHR-seg: **+0.55 to +1.48 PM**.

**It was not tuned on the test set.** The recipe was fixed on g95 first, and
the g95 gain (+1.02) predicted the SIGHUM gain (+1.00).

### What is not solid

**The lead over TransLIST is within sampling noise.** We have no per-sentence
TransLIST predictions, so no paired test is possible. Unpaired, the bootstrap
95% CI on our 94.43 is **[93.71, 95.12]**, which contains TransLIST's 93.97.
In 8.8% of bootstrap resamples our PM falls at or below 93.97. The lead is 0.46
PM, about 19 sentences. The F1 lead (98.94 vs 98.86) is smaller still. The
defensible claim is **"matches or exceeds TransLIST"**, not "significantly
beats".

**It is an ensemble against a single model.** TransLIST's 93.97 is one model.
An ensemble of TransLIST models would likely gain too, so this is not an
architecture-to-architecture comparison in the way the single-model rows are.
Report it as a system result and keep the single-model row (93.43) beside it.

**There is no seed variance.** Both base models are single-seed (seed 0). The
ensemble is one configuration that already uses all six rerankers, so there is
no spread to report. A real ± needs both base models retrained with more seeds.

**The g95 pre-registration has no artifact.** The g95 result (93.94 vs 92.92)
is recorded only in comments in `eval_union.py` and `eval_union.slurm`, not in
an output file. The claim that the recipe was fixed before SIGHUM rests on
those comments and the commit history (`eval_union.py` was committed Oct 12 in
`fa7918f`, before this run).

**The SIGHUM test set has been looked at many times.** Over the project, many
models were evaluated on these 4,200 sentences, and the decision to try an
ensemble came after seeing that cuSHR-attn did not help on its own. This
recipe was not tuned on the test set, but the choice of which experiments to
run was informed by earlier test results. That is a mild, common form of
test-set reuse and worth acknowledging.

**Inference cost is higher.** The headline is no longer "one 2.8M-parameter
model with a deterministic decoder". It is two base decodes, 16 candidates per
model, and six reranker passes. The base decoders still run in the C++/CUDA
path. The reranking and the combination do not.

### Bottom line

94.43 is a correct, reproduced measurement, and the +1.00 over our previous
best is real (p ≈ 3 × 10⁻⁵). Quote it as **"94.43 PM, level with or above
TransLIST's 93.97"**, label it as an ensemble, and keep the single-model 93.43
alongside it.

---

## 6. Other November results

### The reranker on cuSHR-lm's beams (lemma + morphology)

The reranker staged in `rerank_lm.slurm` was evaluated. Scored with the
vocabulary maps, 3 seeds:

| | S | L | S+M | L+M | S+L+M |
|---|---:|---:|---:|---:|---:|
| **cuSHR-lm + reranker** | 92.87 | 86.67 | 57.52 | **57.52 ± 0.23** | 57.10 |
| cuSHR-lm, no reranker | 92.29 | 85.93 | 54.74 | 54.67 | 54.29 |
| cuSHR-node + reranker (previous best) | 91.98 | 86.10 | 57.40 | 57.57 | 57.10 |
| ByT5, off the shelf | 81.38 | 90.55 | 67.79 | 76.50 | 66.90 |
| right answer in top 16 (reranker ceiling) | 97.95 | 92.64 | 73.67 | 74.19 | 73.60 |

+2.85 L+M, but that only ties the previous best. The reranker used 2.85 of the
~19.5 points its top-16 list made available. ByT5 still leads L+M by 19 points.

`RESULTS.md` §3 quotes 61.21 S+M for cuSHR-node + reranker, while
`COMPARISON_OCTOBER.md` §2 gives 57.40 for what looks like the same setting.
The table above uses 57.40. Which is right still needs checking.

---

## 7. Next steps

1. **Seed variance for the ensemble.** Retrain cuSHR-seg and cuSHR-attn with
   seeds 1 and 2 (cluster job, ~2 h each on an A100) and report the ensemble
   as a mean ± spread.
2. **Ask the TransLIST authors for per-sentence predictions.** That would
   allow a paired test and settle whether the 0.46 lead is significant.
3. **A third ensemble member.** A third model with different errors would
   likely add more, but it has to be chosen and fixed on g95 first, never by
   trying candidates on SIGHUM.

---

## Provenance

| number | artifact |
|---|---|
| ensemble, seg alone, attn alone (PM, F1, beam overlap, 98.24) | `cushr_train/eval_union_sighum_k16.json`, `.log` |
| per-sentence predictions | `cushr_train/eval_union_sighum_k16_preds.json`, `eval_union_seg_alone_preds.json`, `eval_union_attn_alone_preds.json` |
| McNemar, bootstrap CIs, agreement table | `cushr_train/union_significance.py` (re-runnable) |
| cuSHR-seg alone list ceiling, 97.50 | `cushr_train/union_selftest.json` |
| base model params | `papers/COMPARISON_OCTOBER.md` §0 |
| reranker params, features, training | `cushr_train/COMPARING_MODELS.md` §4, checkpoint metadata |
| g95 ensemble, +1.02 | comments in `cushr_train/eval_union.py`, `eval_union.slurm` (no output file) |
| cuSHR-lm + reranker | `cushr_train/eval_lm_ex4200_rr_{raw,maps}_s{0,1,2}.json` |
| TransLIST | arXiv:2210.11753, Table 1, SIGHUM column |
| ByT5 | arXiv:2409.13920, Table 3 |

**How the run was done.** The attention model's lattice cache existed only on
the cluster, so it was rebuilt locally: `train.py --resume` from
`model95_lattice_attn_char_s0.npz.ckpt` (8 of 8 epochs done, so no training
runs, it only reloads the best epoch and writes the frozen node vectors), then
`prepare.py`, then `eval_union.py` with exactly the arguments in
`eval_union.slurm`. All on CPU. The rebuilt scorer weights were compared with
the committed `model95_lattice_attn_char_s0.npz` and are identical.
