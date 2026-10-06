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

## 1. Segmentation (S)

| System | Perfect match | Word F1 | source |
|---|---:|---:|---|
| TransLIST | **93.97**† | **98.86**† | Sandhan et al. 2022, Table 1 (SIGHUM) |
| ByT5-Sanskrit, fine-tuned on SIGHUM | 93.83† | — | Nehrdich et al. 2024, Table 3 |
| **cuSHR-seg + reranker** (our best) | **93.43 ± 0.13** | 98.73 | 3-seed mean, `COMPARING_MODELS.md` |
| cuSHR-seg | 92.76 | 98.42 | `model95_seg_ex4200` |
| cuSHR-lm | 92.29 | — | `eval_lm_raw.json`, Oct 6 |
| cuSHR-node (joint) | 91.52 | 98.32 | `RESULTS_MATRIX.md` |
| ByT5-Sanskrit, off-the-shelf multitask (we ran it) | 81.38 | — | `eval_lm_raw.json` |
| *ORACLE (our gold vs the reference)* | *98.00* | — | ceiling |

**Standing: third, by 0.54 against TransLIST and 0.40 against fine-tuned ByT5.**

Two readings to keep straight. The 81.38 we measured for ByT5 is the *released
multitask checkpoint*, which uses DCS compound conventions while this reference
uses SIGHUM's; most of its errors are compound-boundary disagreements
(`droRaputraH` vs `droRa putraH`), not failures to segment. The fair ByT5
comparison is their fine-tuned 93.83. Our 11-point win over 81.38 is a
configuration artifact and should not be quoted as a result.

**Unmeasured and possibly decisive:** `model95_lattice_attn_char_s0`, the
lattice-attention model, has dev segmentation PM **94.25** against cuSHR-seg's
93.58 (+0.67) but no SIGHUM-test number yet. If that margin carries and the
reranker adds its usual +0.67, the headline becomes ≈94.1 — above TransLIST.
One job: `sbatch --export=ALL,STAGE=eval,TAG=lattice_attn_char_s0`.

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
