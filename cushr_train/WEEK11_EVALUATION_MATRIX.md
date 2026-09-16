# Week 11 / CP-6 — Full evaluation matrix

**Reproduce everything below with one command:**

```bash
cd cushr_train
python make_results_matrix.py --manifest results_manifest.json
```

That regenerates `RESULTS_MATRIX.md`, `results/results_matrix.json`, and all three
plots. It is deterministic: two consecutive runs produce byte-identical output.
Cells are cached by content hash, so a re-run costs seconds.

> **Reproduction is not repo-only.** `eval_slm.py` reads the out-of-repo trees
> `../../SIGHUM_database/After_graphml` and `../../SIGHUM_database_gold_path/DCS_pick`
> at runtime. A reader with only this repository cannot rebuild the table.

Every number below traces to a named artifact. **An empty cell means the number
was not produced — nothing here is inferred, interpolated, or carried across
cells.** Section 2 gives step-by-step instructions for each gap.

---

## 0. Status against the Week 11 specification

The spec predates the decision to cite rather than re-run baselines (§2). Rows are
marked against that policy, not against the original wording.

| Spec item | State |
|---|---|
| **Dataset 1** — SIGHUM-test, 4,200 | ✅ complete |
| **Dataset 2** — SIGHUM-hackathon-test | 🔨 reachable, spike defined — see §2.3 |
| **Dataset 3** — DCS held-out 1,000, disjoint from SIGHUM train | ⚠️ built with two documented deviations — see §1.2 |
| **Dataset 4** — GRETIL śāstra ~500 | ⏸ deferred for annotation cost — see §2.4 |
| **Baseline 1** — cushr_cpu | ✅ measured on a Lonestar6 compute node |
| **Baseline 2** — TransLIST on A100 | ✅ **cited by policy**, daggered — see §2 |
| **Baseline 3** — ByT5-Sanskrit on A100 | ✅ **measured by us** on dataset 1; deliberately not cited — see §2 |
| **Baseline 4** — SHR's own beam (optional) | ❌ not installed; optional in the spec |
| **Metric** — Word-level P / R / F1 | ✅ dataset 1; structurally impossible on dataset 3 (§3.2) |
| **Metric** — Sentence-level Perfect Match | ✅ same |
| **Metric** — Top-K recall @ 1, 5, 16, 32, 64 | ✅ complete |
| **Metric** — Wallclock throughput | ⚠️ CPU is true wallclock; GPU is kernel-only — see §2.2 |
| **Metric** — GPU memory | ✅ cuSHR complete; not reported for cited systems, by policy (§2.1) |
| **Deliverable** — one script, three plots | ✅ complete |

---

## 1. Results we have

### 1.1 SIGHUM-test (in-domain, published) — n = 4,200

#### Word-level segmentation

| System | P | R | F1 | Perfect match |
|---|---:|---:|---:|---:|
| cuSHR K4 top-1 (GPU) | 98.36 | 98.33 | 98.32 | 91.52 |
| cuSHR K4 + reranker | | | | 91.98 |
| cushr_cpu (week 3) | 98.36 | 98.33 | 98.32 | 91.52 |
| ByT5-Sanskrit *(measured by us)* | | | | 81.38 |
| TransLIST † | 98.80 | 98.93 | 98.86 | 93.97 |

† Reported by Sandhan et al., *TransLIST*, Findings of EMNLP 2022, arXiv:2210.11753,
Table 1 (SIGHUM column); **not reproduced here** — see §2. Their split is *identical*
to ours (same 4,200 DCS-IDs, identical input and gold strings; verify with
`check_testset_overlap.py`), so this is a comparison on the same data against a
reported number. Their Hackathon column is 97.78 / 97.44 / 97.61 / 85.47 and is a
different dataset; do not mix the two.

#### Sentence-level perfect match by annotation level

Reported in two conventions. **raw** is what the decoder emits; **+maps** is the same
output after SHR's analytical vocabulary is translated into DCS's by two tables
(`lemma_map.json`, `convention_map.json`) built from **train ids only**, with dev/test
membership asserted absent rather than assumed (`build_lemma_map.py:70-75` raises on
leakage).

| System | S | L raw | L +maps | S+M raw | S+M +maps | L+M raw | L+M +maps |
|---|---:|---:|---:|---:|---:|---:|---:|
| cuSHR K4 top-1 (GPU) | 91.52 | 65.62 | **85.40** | 45.69 | 52.26 | 45.45 | 52.38 |
| cuSHR K4 + reranker | 91.98 | 65.98 | **86.10** | 50.29 | 57.40 | 50.02 | 57.57 |
| cushr_cpu (week 3) | 91.52 | 65.62 | 85.40 | 45.69 | 52.26 | 45.45 | 52.38 |
| ByT5-Sanskrit | 81.38 | 90.55 | *n/a* | | *n/a* | | *n/a* |
| *ORACLE (ceiling)* | *98.00* | *70.07* | *91.75* | *68.23* | *77.65* | *68.18* | *78.05* |

S is identical in both conventions because the maps rewrite lemma and cng and never
touch form. The generator asserts this rather than trusting it.

**Read every cuSHR number against the ORACLE in the same convention, never across
them.** The maps are applied to cuSHR *and to its ORACLE*, so the +maps ladder is a
**change of measurement target, not a model gain**. They are never applied to ByT5,
which emits DCS conventions natively and needs no translation — so its column is
identical under both and is marked *n/a* rather than repeated.

This is what the raw ladder was hiding: ByT5's L = 90.55 exceeds our *raw* ceiling of
70.07, which is impossible as a quality claim — a system cannot beat our ceiling by
decoding better. Once the convention gap is removed the comparison is 85.40 against
90.55 with a ceiling of 91.75, which is a real and much smaller gap. DCS lemmatises
participles to the verbal root where SHR gives the participial stem; that is the whole
of the difference. See §4.

#### Top-K recall — beam width K = 64

| Level | @1 | @5 | @16 | @32 | @64 |
|---|---:|---:|---:|---:|---:|
| S | 91.52 | 95.81 | 97.74 | 98.40 | 98.69 |
| L | 65.62 | 69.57 | 71.19 | 71.71 | 72.21 |
| S+M | 45.69 | 59.62 | 64.40 | 66.00 | 66.81 |


### 1.2 g95 held-out (out-of-domain by text) — n = 1,000

1,000 DCS sentences, disjoint from the training split **and** from the published
4,200, checked by DCS id rather than row index so a duplicated id cannot leak.
Pool: 5,456 eligible of 5,681 test rows. Built by `make_heldout_set.py`.

**Two deviations from the Week 11 spec, both structural:**

1. The pool is restricted to sentences SHR has already parsed (§3.1).
2. There is no surface-segmentation reference, so S / P / R / F1 / PM are
   unmeasurable here **by construction, not merely unmeasured** (§3.2).

| System | S | L raw | L +maps | L+M raw | L+M +maps |
|---|---:|---:|---:|---:|---:|
| cuSHR K4 top-1 (GPU) | *n/a* | 57.60 | **83.80** | 39.50 | 48.00 |
| ByT5-Sanskrit | *n/a* | | *n/a* | | *n/a* |
| *ORACLE (ceiling)* | *n/a* | *61.90* | *90.80* | *60.70* | *73.40* |

| Level | @1 | @5 | @16 | @32 | @64 |
|---|---:|---:|---:|---:|---:|
| L | 57.60 | 61.50 | 62.80 | 63.20 | |

**The headline out-of-domain finding — and it survives both conventions.**

Do not read 57.60 against SIGHUM's 65.62 directly. L does fall, but the ORACLE falls
with it (70.07 → 61.90), so the comparison that carries information is each number
against its own ceiling:

| convention | in-domain | out-of-domain |
|---|---:|---:|
| raw | 65.62 / 70.07 = **93.7%** | 57.60 / 61.90 = **93.1%** |
| +maps | 85.40 / 91.75 = **93.1%** | 83.80 / 90.80 = **92.3%** |

cuSHR retains roughly **93% of what its own lattice makes reachable in both domains**.
The apparent drop is mostly the convention ceiling moving, not the model degrading —
and the ratio holding under two independent normalisations is a stronger claim than
either number alone.


### 1.3 Throughput and memory (Lonestar6 A100)

`cuSHR GPU, biaffine, k4=twopass, batch -1`

| K | sentences/sec (kernel) | GPU MB | µs/sent K4 | µs/sent K3 |
|---:|---:|---:|---:|---:|
| 1 | 471,642 | 72 | 1.090 | 1.030 |
| 5 | 469,658 | 276 | 1.092 | 1.037 |
| 16 | 466,083 | 840 | 1.095 | 1.050 |
| 24 | 467,182 | 1,254 | 1.098 | 1.043 |
| 32 | 464,995 | 1,662 | 1.092 | 1.058 |
| 48 | 185,587 | 2,484 | 1.085 | **4.303** |
| 64 | 182,091 | 3,306 | 1.087 | **4.405** |


#### Whole-corpus recall, verified against the CPU decoder

| K | recall@K | n_gold | sentences/sec | GPU MB | source |
|---:|---:|---:|---:|---:|---|
| 32 | 97.3053 | 115,447 | 352,024 | 1,662 | `k4_bench_E_k32_checked.csv` |
| 64 | 98.4192 | 115,447 | 155,772 | 3,306 | `k4_bench_F_k64_checked.csv` |


#### CPU baseline

**1,076.75 sentences/sec wall clock** (K=1, 119,503 sentences in 111.0 s,
biaffine, Lonestar6 compute node, `cpu_bench.csv`).

The resulting ratio is **~432×** at K=32 (464,995 / 1,077). Note it compares a
kernel-only GPU figure against an end-to-end CPU figure, so it is an **upper
bound on the achievable end-to-end speedup, not a measurement of one**. §2.3 is
how to close that gap.

### 1.4 Plots

| File | Content | Complete? |
|---|---|---|
| `results/pareto_accuracy_throughput.png` | F1 vs sentences/sec, log-x | 2 points; TransLIST and ByT5 absent for want of throughput |
| `results/recall_vs_k.png` | recall@K per dataset, one common level | ✅ |
| `results/memory_vs_k_batch.png` | `gpu_used_MB` vs K, and vs batch size | ✅ |

> The batch-size panel is drawn from the seven `batched_bench_b*.csv` files. Those
> use the 14-column schema, which has **no `scorer` column**, so the scorer for that
> sweep cannot be confirmed from the artifact. Treat the batch panel as a memory
> and scaling result only; do not attach an accuracy claim to it.

---
## 2. Baseline policy — what we measure, what we cite

We will cite their
published numbers, mark them, and never mix a cited accuracy with our throughput.
Re-running a baseline is only warranted when the split differs, when a head-to-head
speed claim on identical hardware is being made, or when the paper does not report the
metric needed. For accuracy, none of those applies.

| System | Policy | Why |
|---|---|---|
| cuSHR (GPU top-1, reranker, ORACLE) | measured | ours |
| `cushr_cpu` | measured | the only same-hardware speed claim in the paper |
| **ByT5-Sanskrit** | **measured — ours, not cited** | see below |
| **TransLIST** | **cited, daggered** | same task, 97% overlapping split |
| SHR's own beam | not run | optional in the spec; not installed |

**ByT5 is the exception, and deliberately so.** Its published results are measured on a
DCS April-2024 split (601,403 sentences, 8,398 test), *not* SIGHUM — see
`PAPER_COMPARISON.md:448`. Citing those figures in a SIGHUM-test row would silently
swap corpora mid-table. We already ran the released `chronbmm/sanskrit5-multitask` on
our exact 4,200 through the identical reference and `score()`, so the honest column is
the one we measured. The published ByT5 ladder may still be shown *as literature
context*, clearly labelled as a different corpus, never as a table row.

**TransLIST's split IS identical to ours** — measured, after this document
originally claimed the opposite. `sighum_test_4200.tsv` and the `split=test` rows of
the file TransLIST reads (`LREC-Data/new_LREC_data_complete.csv`) carry the same
4,200 DCS-IDs with identical input and gold strings; `check_testset_overlap.py`
reproduces it. The old "97.02% (4,075/4,200), after transliterating to a common
scheme" claim was never computed by any code and is wrong. The generated footnote
(`results_manifest.json` → `cells."sighum_test/translist".published.split_note`) now
says "identical", and the remaining asymmetry is only that their number is *reported*
rather than re-scored by us.

**Numbers verified at source.** TransLIST's row was read directly from Table 1 of
`papers/2210.11753v1.pdf`: SIGHUM 98.80 / 98.93 / 98.86 / 93.97, Hackathon
97.78 / 97.44 / 97.61 / 85.47. A secondary summary claimed 93.97 was the Hackathon
figure; it is not. Do not re-derive these from anything but the PDF.

**Contamination runs the other way, and is worth one sentence in the paper.**
`PAPER_COMPARISON.md` records 0 of 4,200 benchmark sentences in cuSHR's training data
against 100 of 4,200 (2.38%) in ByT5's SIGHUM fine-tuning split, plus unavoidable
pretraining exposure to all 4,200. The comparison is not tilted our way.

### 2.1 Throughput and memory are the one place re-running would matter

The spec asked for baselines "on the same hardware", which is a *throughput* claim, not
an accuracy one. Since we are not re-running anyone, the honest consequence is:

- Throughput and GPU memory are reported **only** for cuSHR and `cushr_cpu`.
- The accuracy-vs-throughput plot **excludes cited systems by construction** — this is
  enforced in `make_results_matrix.py`, not left to reviewer discipline, and the
  excluded system is named in the figure caption.
- ByT5's throughput was never recorded even though we ran it. If it is ever re-run,
  capture wall clock from the job log and `torch.cuda.max_memory_allocated()`; that
  would put a third point on the Pareto plot.

### 2.2 GPU wallclock throughput — deferred, not descoped

Our GPU column is kernel-only; the spec asked for wallclock. I checked whether the CSVs
already answer this: **they do not.** `us_per_sent_loop` and `us_per_sent_kernel` are
identical in every row of every sweep, so that column excludes host time too.

One job fixes it, and it is *our* system, so the cite-don't-rerun policy does not apply:

```bash
cd /home1/11503/njhavar/cushr/cushr_gpu
time ./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" \
    --k4 twopass --K 32 --batch -1 --check 0 --csv k4_bench_G_k32_wall.csv
```

Report both figures, labelled distinctly. Never rename one to the other. The job logs on
Lonestar6 also carry a `real` time for the runs already done, but that number includes
the npz load, the CPU cross-check and the 325 MB dump, so it understates streaming
throughput badly.

### 2.3 SIGHUM-hackathon-test — reachable, and previously descoped in error

I had descoped this on the grounds that cuSHR cannot decode sentences SHR has not
parsed. **That reasoning does not apply to this dataset.** From
`papers/2210.11753v1.pdf` §3, read directly:

> "Both datasets are made of DCS. These datasets also come with candidate solution
> space generated by SHR for SWS."

The SHR candidate space already exists for the Hackathon set — 90,000 train / 10,332 dev
/ **9,963 test** — and Sandhan et al. state *"We release our codebase and datasets
publicly under the Apache license 2.0."* Because it ships gold segmentation, this
dataset could carry **full word-level P/R/F1 and surface PM**, unlike `g95_heldout`.
TransLIST also publishes a Hackathon column (97.78 / 97.44 / 97.61 / 85.47), so the
cited comparison comes free.

**Spike, with a hard go/no-go at step 3:**

1. **Download.** `Hackathon_data/` is referenced by `github.com/rsingha108/TransLIST`;
   the SIGHUM original is `zenodo.org/record/803508` (footnote 7 of the paper).
2. **Characterise the candidate-space format — read only, no code yet.** Our loader
   consumes SHR `.graphml` with the `key=1` / `key=2` edge semantics documented in
   `ingest/INGEST_METHODOLOGY.md`. Their release may be graphml in the same shape,
   graphml in a different shape, or something else entirely. This is the unknown.
3. **Go/no-go: can it convert without inventing edges?** Validate by converting a
   SIGHUM sentence we *already* have graphml for and diffing node and edge sets against
   `../../SIGHUM_database/After_graphml`. If it does not round-trip, stop and descope
   with that as the stated reason.
4. If it converts: ingest, add `hackathon_test` to `results_manifest.json`, decode, and
   fill a complete row.

Do **not** hand-write a converter against a guessed schema.

### 2.4 GRETIL śāstra subset — deferred for annotation cost, not capability

Also not blocked on SHR the way I first claimed.
`github.com/SriramKrishnan8/sandhi_vicchedika` is a local, batch-capable wrapper around
the Heritage Engine's `interface2` binary (prereqs: ocaml, ocamlbuild, camlp4, python3,
devtrans; accepts SLP as the `SL` encoding; `-i input_file -o output_file`). Producing
SHR analyses for new sentences is roughly a day of work.

**The real blocker is gold.** A GRETIL śāstra text has no DCS annotation, so ~500
sentences need hand-correction by someone competent in śāstric Sanskrit. That is not an
engineering cost. Secondary risk: that tool's documented output is a segmentation and
morphology list, **not** graphml, so the same step-3 schema gate as above would apply.

Say in the paper that it is deferred for annotation cost. That is accurate; "we could
not run SHR" is not.
## 3. Structural limits — state these in the paper

### 3.1 cuSHR's coverage is bounded by SHR's, not by annotation availability

The lattice **is** a Sanskrit Heritage Reader `.graphml`, and nothing in this
repository invokes SHR. `ingest/INGEST_METHODOLOGY.md:150` states the corpus is
"limited by how many sentences were run through SHR, not by annotation."

Concretely: **119,503 sentences have graphml; 322,232 DCS pickles do not.**
A sentence SHR has never parsed cannot be decoded at all, at any K.

**State the bound precisely, because a loose version of it is wrong.** It constrains
expansion onto *arbitrary new text* — any sentence we would have to run through SHR
ourselves. It does **not** prevent us from using a released dataset that ships its own
SHR candidate space, which is exactly what the Hackathon set does (§2.3). I initially
descoped that dataset on this reasoning and was wrong to.

So the bound explains dataset 3's restriction to already-parsed sentences (§1.2), and
it explains why extending to arbitrary GRETIL text needs a generation step. It does not
by itself explain dataset 2. Nor is running SHR a capability we lack outright:
`sandhi_vicchedika` wraps the Heritage Engine's `interface2` locally (§2.4). The honest
statement is that we have not needed to, not that we could not.

It still deserves a paragraph in the paper's limitations section — cuSHR's coverage is
bounded by SHR's, not by annotation availability — but as a property of the pipeline,
scoped as above.

### 3.2 No surface-segmentation gold exists outside the published 4,200

Verified three independent ways:

- The DCS pickles expose only `cng, dcs_chunks, lemmas, sent_id, sentence`, and
  `lemmas` is IAST **lemmas** — `['kṣip','mad','suta','rājan',…]` where a surface
  reference needs `cikzepa me sutaH rAjan …`.
- npz `surface_text` is a lossy reconstruction of *our own gold path*
  (`aaApatat` for `aTa apatat`; literal `?` for unresolved characters). It is our
  output, not an independent reference.
- `eval_slm.py:21` names the source as the *published* `sighum_test_4200.tsv`
  `output` column. No larger copy exists in the repo or either external tree.

Hence `eval_slm.py --no-surface`, which drops the S levels outright.
**Do not synthesise an `output` column from the gold path to fill those cells** —
it would score cuSHR against its own lattice gold and drive ORACLE toward 100.

---

## 4. Meeting decisions

### Headline vs appendix

**Headline:** S-level F1 and Perfect Match; the **+maps** L/M ladder with its ORACLE and
the convention footnote; throughput and the CPU/GPU ratio; recall-vs-K; the
out-of-domain ceiling-relative result from §1.2.

**Appendix:** the raw ladder beside the +maps one; the memory sweeps;
fused-vs-twopass; the ncu profile.

Rationale, revised: leading with S alone was the safe choice while the only L figure
was the raw 65.62, which loses to ByT5's 90.55 by a margin that is convention, not
quality. With both conventions computed we can headline the ladder honestly — 85.40
against 90.55 under a 91.75 ceiling — provided the footnote travels with it. **Both
columns must appear somewhere**; publishing only +maps would look like a chosen
convention, and only raw understates by ~20 points on L for no substantive reason.


---

## 5. Provenance

| Cell | Produced by |
|---|---|
| `sighum_test/cushr_gpu_top1` | `eval_slm.py`, `eval_surface.py` |
| `sighum_test/cushr_gpu_rerank` | `eval_slm.py` |
| `sighum_test/byt5` | `eval_slm.py` |
| `sighum_test/cushr_cpu` | accuracy asserted identical to `cushr_gpu_top1` (`score_mismatch = 0`); throughput from `cpu_bench.csv` |
| `sighum_test/translist` | `papers/2210.11753v1.pdf` via `PAPER_COMPARISON.md` — **not reproduced** |
| `g95_heldout/cushr_gpu_top1` | `eval_slm.py --no-surface` |
| K sweep | `cushr_gpu/k4_bench_C_biaffine_twopass.csv` |
| Checked recall | `cushr_gpu/k4_bench_{E_k32,F_k64}_checked.csv` |
| Batch sweep | `cushr_gpu/batched_bench_b*.csv` |
| CPU | `cushr_cpu/cpu_bench.csv` |

### Verification performed

- `eval_slm.py` with no new flags reproduces the CP-5 anchors **91.52 / 65.62 /
  45.69 / 45.45 / 45.29**, unchanged.
- `--json-out` values equal the printed values: 8 levels, 0 mismatches.
- `eval_surface.py` PM (91.52) equals `eval_slm.py` S (91.52).
- `gpu_paths_to_rerank.py` on the K=64 dump prints recall@64 = **98.4192%**,
  equal to `recall_at_K` in `k4_bench_F_k64_checked.csv` over the same
  `n_gold = 115,447`.
- Aggregator run twice → byte-identical `results_matrix.json` **and**
  `RESULTS_MATRIX.md`.
- Held-out pool: 5,681 test rows − 225 published = 5,456 eligible; zero drops for
  train/dev/no-gold/no-pickle/no-graphml; **0 pool ids appear in any train row**.
