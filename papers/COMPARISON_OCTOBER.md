# cuSHR vs TransLIST vs ByT5-Sanskrit — October 2026 snapshot

Best cuSHR model **per task**, against the two published competitors. Every
cuSHR number was measured here; every competitor number is either measured here
(`we ran it`) or cited (†). No cell is inferred, averaged across settings, or
carried over from a different model — where we have not run it, the cell reads
`not measured`.

**Test set.** SIGHUM test, 4,200 sentences, the published split. TransLIST's
reported numbers are on the *identical* data: the same 4,200 DCS-IDs with
identical inputs and gold segmentations (`check_testset_overlap.py`). ByT5's
published numbers are **not** — they are a DCS April-2024 split, 8,398 test
sentences — which is why we also ran ByT5 ourselves on these 4,200.

---

## 0. The models

### What every cuSHR model shares

All four are the same three-stage stack and differ **only in the encoder and the
training objective**. Nothing about the decoder, the export format or the
C++/CUDA consumer changes between them.

1. **Featurizer** `hybrid_tag` — 80 hand-built scalar/n-gram columns, plus
   learned embeddings over four identity vocabularies from the lattice: 56,390
   forms, 17,269 lemmas, 1,281 preverbs, 848 morph tags → a 96-dim node vector.
   **2,121,332 params**, ~76% of the model. Word dropout 0.1.
2. **Contextual encoder** — see each model below. Output is concatenated with
   the featurizer vector, giving the 192-dim vector the scorer reads.
3. **Biaffine edge scorer** — `score(u→v) = src_proj(f)[u] · dst_proj(f)[v] +
   bias`, hidden 128. **49,153 params.** First-order by construction: an edge
   score sees exactly two node vectors, never the path.

Decoding is Viterbi over the lattice DAG. The encoder output depends on the
sentence but never on the path, so `--materialize` freezes it into a dense
archive and `cushr_gpu/` consumes it unchanged.

**Training data** (identical for all four): the SHR lattice archive
`cushr_data_g95.npz`, 119,503 sentences, 115,447 (96.6%) with a resolved gold
path. Split by `splits95_ex4200.json` — **train 104,159 / dev 5,607 / test
5,681**, with all 4,200 SIGHUM benchmark sentences held out of training. 8
epochs, batch 64, lr 1e-3, margin 1.0, weight decay 1e-5, seed 0, structured
hinge loss.

### cuSHR-node — `model95_ctx_ex4200.npz`

| | |
|---|---|
| encoder | **character BiLSTM**, 2 layers, hidden 128, char emb 32 — 618,624 params |
| total params | **2,789,109** |
| objective | `--cost node --gold-target exact`: every node off the gold path costs the margin, lemma and tag included |
| epoch chosen by | dev node F1 |
| trained for | the whole analysis at once — segmentation, lemma and morph jointly |

The original model and the source of every number in `RESULTS_MATRIX.md`. The
BiLSTM is what reads the *sandhi-fused surface text* — information every
per-node feature had discarded — and it was worth +6.95 F1 / +23.6 PM over no
encoder, against +0.04 F1 for adding scorer capacity instead.

### cuSHR-seg — `model95_seg_ex4200.npz`

| | |
|---|---|
| encoder | character BiLSTM, as above — 618,624 params |
| total params | **2,789,109** |
| objective | `--cost surface --gold-target surface`: only a wrong *segmentation* word costs the margin; a right word with the wrong lemma/tag is free (`--morph-cost 0`) |
| epoch chosen by | dev surface PM |
| trained for | **segmentation only** |

Identical architecture to cuSHR-node; only what it is trained toward changes.
The latent-gold target matters as much as the cost: the hinge compares against
the best path *spelling* the gold segmentation, so a morph variant of the right
split no longer produces loss.

### cuSHR-attn — `model95_lattice_attn_char_s0.npz`

| | |
|---|---|
| encoder | **lattice self-attention + char BiLSTM** — 925,056 params |
| total params | **3,095,541** |
| attention | 8 heads, d_model 128, 2 blocks, FFN ×2, dropout 0.1 |
| positional encoding | **four-position span bias** — separate learned embeddings over relative distance (`max_rel` 64, so 129 buckets × 8 heads) for start-start, start-end, end-start and end-end |
| objective | same as cuSHR-seg (segmentation only) |
| trained for | **segmentation only** |

The TransLIST mechanism on cuSHR's lattice: self-attention over all candidate
words of a sentence, biased by how their spans sit relative to one another, so
the scorer can express "this candidate is wrong given a word ten characters
away". Attention is computed in span groups under a `--pair-budget` so nothing
wider than `[G, N, N, heads]` is ever materialised.

**Measured Oct 6 and it did not help:** 92.45 base / 93.36 ± 0.09 reranked,
against cuSHR-seg's 92.76 / 93.43 ± 0.13. Dev had promised +0.67 and none of it
transferred. See §1.

### cuSHR-lm — `model95_lm_ex4200.npz`

| | |
|---|---|
| encoder | lattice self-attention + char BiLSTM, as cuSHR-attn — 925,056 params |
| total params | **3,095,541** |
| objective | `--cost lm --gold-target lm`: node identity is `(char_start, word_len, lemma id, morph tag id)` — the form is dropped, so a sandhi/spelling variant carrying the gold analysis is free, and a wrong lemma or tag is charged |
| epoch chosen by | dev L+M PM |
| trained for | **lemma + morphology** |

Trained Oct 2026, the newest model. Its weakness is known and recorded in §2:
the identity still pins each word to its character span, so it only forgives
*form*, and form errors are 1.1% of words while tag errors are 5.75%. Dev
`lm_pm` tracked the full-analysis `perfect_match` to within 0.15 at every epoch,
which is the measurement showing this objective is close to cuSHR-node's.

### The rerankers — second pass, not part of the base model

`rerank.py` embeds each word's morph tag and lemma, runs a 1-layer BiLSTM over a
**candidate path** and rescores the base decoder's top-16. This is what a
first-order edge scorer structurally cannot do: judge an analysis as a whole, so
that `nom acc verb` and `acc nom verb` differ where role *counts* are identical.

| variant | params | what it reads | gain |
|---|---:|---|---|
| full (`reranker_full.pt`) | 865,667 | morph tag + lemma sequence | +3.96 gold-path match |
| morph-only (`reranker_morph.pt`) | 77,443 | morph tag sequence only | +2.89 |
| segmentation (`segrr_*.pt`) | — | `--seg-features --label surface` | +0.67 PM (3-seed mean) |

Bounded by the beam, not the oracle: it reorders candidates and can never add
one, so recall@16 (73.67 at S+M) is a hard ceiling. **No reranker has been
trained on cuSHR-lm's beams yet** — `rerank_lm.slurm` is staged for it.

### Competitors

| | params | pretraining | how it works |
|---|---:|---|---|
| **TransLIST** | not reported | none | Transformer over characters with a *lattice-aware* module: SHR's candidate words are injected as additional tokens, plus a soft-masked attention that respects candidate spans. Same SHR lexicon as cuSHR, so the comparison is architecture-to-architecture. |
| **ByT5-Sanskrit** | 582M | 6.5B tokens | Byte-level seq2seq, multitask (segmentation, lemma, morphosyntax) — generates the analysis as text rather than selecting from a lattice, so it has no beam, no oracle ceiling, and can emit analyses SHR never proposed. We ran the released `chronbmm/sanskrit5-multitask` off the shelf; their SIGHUM figures come from a checkpoint fine-tuned on SIGHUM that was never released. |

cuSHR is **~1/190th of ByT5's parameter count with no pretraining**, and its
inference is a deterministic C++/CUDA decoder that needs no GPU to serve.

---

## 1. Segmentation (S)

| System | Perfect match | Word F1 | source |
|---|---:|---:|---|
| TransLIST | **93.97**† | **98.86**† | Sandhan et al. 2022, Table 1 (SIGHUM) |
| ByT5-Sanskrit, fine-tuned on SIGHUM | 93.83† | — | Nehrdich et al. 2024, Table 3 |
| **cuSHR-seg + reranker** (our best) | **93.43 ± 0.13** | 98.73 | 3-seed mean, `COMPARING_MODELS.md` |
| cuSHR-attn + reranker | 93.36 ± 0.09 | **98.80** | 3-seed mean, `eval_lattice_attn_char_s0_rerank_s{0,1,2}.json` |
| cuSHR-seg | 92.76 | 98.42 | `model95_seg_ex4200` |
| cuSHR-attn | 92.45 | 98.46 | `eval_lattice_attn_char_s0_base.json` |
| cuSHR-lm | 92.29 | — | `eval_lm_raw.json`, Oct 6 |
| cuSHR-node (joint) | 91.52 | 98.32 | `RESULTS_MATRIX.md` |
| ByT5-Sanskrit, off-the-shelf multitask (we ran it) | 81.38 | — | `eval_lm_raw.json` |
| *ORACLE (our gold vs the reference)* | *98.00* | — | ceiling |

**Standing: third, by 0.54 against TransLIST and 0.40 against fine-tuned ByT5.**
The best S model is `cuSHR-seg` — the plain character BiLSTM. Lattice attention
was tried and did not beat it (below).

Two readings to keep straight. The 81.38 we measured for ByT5 is the *released
multitask checkpoint*, which uses DCS compound conventions while this reference
uses SIGHUM's; most of its errors are compound-boundary disagreements
(`droRaputraH` vs `droRa putraH`), not failures to segment. The fair ByT5
comparison is their fine-tuned 93.83. Our 11-point win over 81.38 is a
configuration artifact and should not be quoted as a result.

### Lattice attention did not help — a negative result (Oct 6)

`cuSHR-attn` adds TransLIST's own mechanism to cuSHR's lattice: 8-head
self-attention over all candidate words with a four-position span bias, 306K
parameters on top of the BiLSTM. On dev it looked decisive, **+0.67** (94.25 vs
cuSHR-seg's 93.58). **None of it transferred.**

| | base PM | + reranker (3 seeds) | F1 macro | F1 micro |
|---|---:|---:|---:|---:|
| cuSHR-seg | **92.76** | **93.43 ± 0.13** | 98.73 | 98.86 |
| cuSHR-attn | 92.45 | 93.36 ± 0.09 | **98.80** | **98.89** |

Per seed: 93.43 / 93.38 / 93.26. The base model is **0.31 worse**, and the
reranked mean is 0.07 lower — inside the seed spread either way. The only real
gain is F1, +0.07 macro, which does not move perfect match.

Dev and test differ in corpus, not just sample: dev is 5,607 g95 sentences,
test is the 4,200 published SIGHUM ones. The extra capacity fit dev-specific
structure. Read this as a warning about the dev metric as much as about the
encoder.

**What it rules out.** Porting TransLIST's attention onto a fixed SHR lattice
does not reproduce TransLIST's score. Their advantage is likely architectural in
a way this port does not capture: they inject candidate words as extra tokens
into a character transformer rather than scoring the edges of a pre-built
lattice, so candidate selection and context modelling are not separate stages.

Incidental: cuSHR-attn's beam is slightly *better* (recall@16 98.50 vs 97.74)
yet its reranker extracts less from it, moving only 71–82 of 4,200 sentences.

## 2. Lemma + morphology (L+M)

Measured at ByT5's task levels by `eval_slm.py`, with the convention maps
applied (they translate SHR's analytical vocabulary into DCS's, and are built
from train ids only). ByT5 needs no translation — it emits DCS natively — so
the maps are applied to cuSHR and its oracle, never to ByT5.

| System | S | L | S+M | L+M | S+L+M |
|---|---:|---:|---:|---:|---:|
| **ByT5-Sanskrit, off-the-shelf (we ran it)** | 81.38 | **90.55** | **67.79** | **76.50** | **66.90** |
| **cuSHR-lm** (our best) | **92.29** | 85.93 | 54.74 | 54.67 | 54.29 |
| cuSHR-node (joint) | 91.52 | 85.40 | 52.26 | 52.38 | 51.95 |
| cuSHR-node + reranker | 91.98 | 86.10 | 57.40 | 57.57 | 57.10 |
| *ORACLE (ceiling)* | *98.00* | *91.75* | *77.65* | *78.05* | *77.50* |
| ByT5 published (DIFFERENT CORPUS) | 84.61† | 79.88† | 63.86† | 62.00† | 61.27† |

**Standing: ByT5 leads L+M by 21.8 points (76.50 vs 54.67).** The gap is not a
convention artifact: the maps already put both sides in DCS's vocabulary, and
the 78.05 ceiling binds both. ByT5 sits 1.55 below that ceiling; cuSHR-lm sits
23.4 below it.

Note the single strongest L+M number we have is **cuSHR-node + reranker at
57.57** — a reranker over the *old* model's beams still beats the new
specialist's 54.67. Reranking cuSHR-lm's beams has not been run and is the
obvious next step (§4).

The published ByT5 row is retained only for category alignment. Their test split
is a different corpus, excludes reconstructed forms, and is biased toward Vedic
texts (their §5.5). Never read it as a SIGHUM result.

## 3. Throughput (cuSHR only; neither competitor was run on our hardware)

| K | sentences/sec | GPU MB | recall@K |
|---:|---:|---:|---:|
| 1 | 471,642 | 72 | — |
| 32 | 464,995 | 1,662 | 97.31 |
| 64 | 182,091 | 3,306 | 98.42 |

Lonestar6 A100, biaffine scorer, `k4=twopass`. Recall figures come from the
separate `--check -1` runs that cross-check every sentence on the CPU
(`k4_bench_E/F`), not from this sweep. TransLIST and ByT5 report no comparable
figure, and we did not run either on this hardware, so this table has no
competitor column by construction.

## 4. Where the remaining error is

Per-word error types, 13,430 words of the test split, decoded against **our own
gold path** (so no convention gap enters):

| | share of words |
|---|---:|
| exactly right | 92.42% |
| wrong span (segmentation) | 1.11% |
| right span, wrong lemma only | 0.13% |
| **right span, wrong morph tag only** | **5.75%** |
| right span, both wrong | 0.60% |

Tag-only errors outnumber lemma-only errors **45 to 1**. Lemmatisation is
effectively solved (L is 85.93 of a 91.75 ceiling); the morph tag carries ~85%
of all analysis error.

And it is a *ranking* failure, not a coverage one:

| level | @1 | @16 | @64 | ORACLE |
|---|---:|---:|---:|---:|
| S | 91.52 | 97.74 | 98.69 | 98.00 |
| L | 85.40 | 92.33 | 93.10 | 91.75 |
| S+M | 52.26 | 73.67 | 76.52 | 77.65 |

S+M recall@64 reaches 76.52 of a 77.65 ceiling — the beam almost always holds
the right analysis and the scorer does not pick it. `rerank_train.log` puts the
same point at sentence level: beam ceiling 92.66 gold-path match against a
64.37 base.

Already ruled out: role-count rerankers separate only **20.1%** of the K=16
headroom. The BiLSTM path reranker beats that, and is worth **+8.95 S+M** on the
old model (52.26 → 61.21).

---

## Provenance

| number | artifact |
|---|---|
| cuSHR-lm S / L+M, Oct 6 | `cushr_train/eval_lm_raw.json`, `eval_lm_maps.json` |
| cuSHR-lm training | `log95_lm_ex4200.json` (`--cost lm --gold-target lm --select-by lm_pm`) |
| cuSHR-seg + reranker | `COMPARING_MODELS.md` §1–2, 3-seed mean |
| cuSHR-node rows, throughput | `RESULTS_MATRIX.md`, `results_manifest.json` |
| recall@K, separability, reranker gain | `PAPER_COMPARISON.md`, `rerank_train.log` |
| per-word error split | measured Oct 6 on `cache95_ctx_ex4200` + `model95_ctx_ex4200` |
| TransLIST | arXiv:2210.11753, Table 1, SIGHUM column |
| ByT5 published | arXiv:2409.13920, Tables 3 and 7 |

**Known gaps in this snapshot.** The per-word error split in §4 and the
recall@K table were measured on the *joint* model, not cuSHR-lm; the direction
is almost certainly the same but the magnitudes are not cuSHR-lm's. The
attention model has no SIGHUM-test number. cuSHR-lm has never been reranked.
All cuSHR base models are single-seed (seed 0) while the reranker rows are
3-seed means.
