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

| Spec item | State |
|---|---|
| **Dataset 1** — SIGHUM-test, 4,200 | ✅ complete |
| **Dataset 2** — SIGHUM-hackathon-test | ❌ not attempted — see §2.5 |
| **Dataset 3** — DCS held-out 1,000, disjoint from SIGHUM train | ⚠️ built with two documented deviations — see §1.2, §2.6 |
| **Dataset 4** — GRETIL śāstra ~500 | ❌ not done — see §2.7 |
| **Baseline 1** — cushr_cpu | ✅ measured on a Lonestar6 compute node |
| **Baseline 2** — TransLIST on A100 | ❌ published figures only, not reproduced — see §2.2 |
| **Baseline 3** — ByT5-Sanskrit on A100 | ⚠️ run on dataset 1 only — see §2.1 |
| **Baseline 4** — SHR's own beam (optional) | ❌ not installed — see §2.8 |
| **Metric** — Word-level P / R / F1 | ✅ dataset 1; structurally impossible on dataset 3 (§3.2) |
| **Metric** — Sentence-level Perfect Match | ✅ same |
| **Metric** — Top-K recall @ 1, 5, 16, 32, 64 | ✅ complete |
| **Metric** — Wallclock throughput | ⚠️ CPU is true wallclock; GPU is kernel-only — see §2.3 |
| **Metric** — GPU memory | ⚠️ cuSHR complete; ByT5 never recorded — see §2.4 |
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
| ByT5-Sanskrit | | | | 81.38 |
| TransLIST *(published, not reproduced)* | 98.80 | 98.93 | 98.86 | 93.97 |

The `cushr_cpu` accuracy row is an **asserted identity, not a second
measurement**: same biaffine weights, same lattice, same top-1 Viterbi. Week 10
established digit-identity with the GPU, and `k4_bench_F_k64_checked.csv`
re-confirms it with `score_mismatch = count_mismatch = 0` across all 119,503
sentences. Only throughput differs between the two rows.

> **Trap, documented in `results_manifest.json`:** `cpu_bench.csv` carries its own
> `precision,recall,f1` columns. Those are **node-level over the whole 119,503-sentence
> corpus**, not surface-level over the 4,200. They are not the numbers in this table
> and must never be substituted into it.

#### Sentence-level perfect match by annotation level

| System | S | L | S+M | L+M | S+L+M |
|---|---:|---:|---:|---:|---:|
| cuSHR K4 top-1 (GPU) | 91.52 | 65.62 | 45.69 | 45.45 | 45.29 |
| cuSHR K4 + reranker | 91.98 | 65.98 | 50.29 | 50.02 | 49.86 |
| cushr_cpu (week 3) | 91.52 | 65.62 | 45.69 | 45.45 | 45.29 |
| ByT5-Sanskrit | 81.38 | 90.55 | | | |
| *ORACLE (ceiling)* | *98.00* | *70.07* | *68.23* | *68.18* | *67.97* |

**Read every cuSHR number against its ORACLE, not against ByT5.** ByT5's L =
90.55 exceeds our own ceiling of 70.07. A system cannot beat our ceiling by
decoding better, so that gap is a *convention* artifact: DCS lemmatises
participles to the verbal root where SHR gives the participial stem. The L / L+M
/ S+L+M columns measure convention agreement as much as model quality. See §4.

#### Top-K recall — beam width K = 64

This is the **beam's own** recall, so it is a hard ceiling on any reranker: a
reranker reorders these candidates and cannot add one.

| Level | @1 | @5 | @16 | @32 | @64 |
|---|---:|---:|---:|---:|---:|
| S | 91.52 | 95.81 | 97.74 | 98.40 | 98.69 |
| L | 65.62 | 69.57 | 71.19 | 71.71 | 72.21 |
| S+M | 45.69 | 59.62 | 64.40 | 66.00 | 66.81 |

Source: `eval_slm.py --cands gpu_rerank_k64.npz --kbest 64`, and **only** that.
The four `k4_bench_{A,B,C,D}` sweeps ran `--check 0` and wrote `recall_at_K=NA`;
the one bench CSV carrying a full curve (`batched_bench.csv`) is the hand-tuned
`log_linear` scorer with recall@1 = 8.31%, which must never be quoted beside the
trained model.

### 1.2 g95 held-out (out-of-domain by text) — n = 1,000

1,000 DCS sentences, disjoint from the training split **and** from the published
4,200, checked by DCS id rather than row index so a duplicated id cannot leak.
Pool: 5,456 eligible of 5,681 test rows. Built by `make_heldout_set.py`.

**Two deviations from the Week 11 spec, both structural:**

1. The pool is restricted to sentences SHR has already parsed (§3.1).
2. There is no surface-segmentation reference, so S / P / R / F1 / PM are
   unmeasurable here **by construction, not merely unmeasured** (§3.2).

| System | S | L | S+M | L+M | S+L+M |
|---|---:|---:|---:|---:|---:|
| cuSHR K4 top-1 (GPU) | *n/a* | 57.60 | *n/a* | 39.50 | *n/a* |
| ByT5-Sanskrit | *n/a* | | *n/a* | | *n/a* |
| *ORACLE (ceiling)* | *n/a* | *61.90* | *n/a* | *60.70* | *n/a* |

| Level | @1 | @5 | @16 | @32 | @64 |
|---|---:|---:|---:|---:|---:|
| L | 57.60 | 61.50 | 62.80 | 63.20 | |

**The headline out-of-domain finding.** L drops 65.62 → 57.60, but ORACLE drops
70.07 → 61.90 alongside it. cuSHR holds 93.7% of its ceiling in-domain and 93.1%
out-of-domain. **The loss is the convention ceiling moving, not the model
degrading** — a distinction the raw L number alone would hide.

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

**A throughput cliff sits between K=32 and K=48**, and it is entirely in K3: K4
stays flat at ~1.09 µs/sent while K3 jumps 1.058 → 4.303, a 4.1× step. This is
the k-best merge losing occupancy once the per-thread candidate array stops
fitting the register budget. It is a result worth reporting, not an anomaly.

#### Whole-corpus recall, verified against the CPU decoder

| K | recall@K | n_gold | sentences/sec | GPU MB | source |
|---:|---:|---:|---:|---:|---|
| 32 | 97.3053 | 115,447 | 352,024 | 1,662 | `k4_bench_E_k32_checked.csv` |
| 64 | 98.4192 | 115,447 | 155,772 | 3,306 | `k4_bench_F_k64_checked.csv` |

Both ran `--check -1` (every sentence decoded on the CPU too and compared) with
`score_mismatch = count_mismatch = 0`. The denominator is `n_gold`, not
`n_sentences`: 4,056 of the 119,503 corpus sentences have an empty gold span and
can never be hit by any decoder, so including them would understate recall.

These throughputs are lower than the same K in the sweep above (K=32: 352,024 vs
464,995; K=64: 155,772 vs 182,091) because they also paid for the path dump and
ran alongside the cross-check. **Quote the sweep rows as the throughput result.**

#### CPU baseline — and a correction

**1,076.75 sentences/sec wall clock** (K=1, 119,503 sentences in 111.0 s,
biaffine, Lonestar6 compute node, `cpu_bench.csv`).

This **supersedes** the "~100 sentences/sec single-threaded" figure at
`cushr_cpu/README.md:55`, which is a design target written before the decoder
existed and is **low by a factor of ~11**. Any speedup claim must use the
measured number.

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

## 2. What is missing, and exactly how to get it

Ordered by value per unit effort.

### 2.1 ByT5 on the held-out set — *one job, fills an entire empty row*

The inputs already exist locally but `byt5_in/` is git-ignored, so regenerate
them on Lonestar6 rather than copying.

```bash
cd /home1/11503/njhavar/cushr/cushr_train

# 1. Build the inputs from the tracked TSV.
python make_byt5_input.py --tsv heldout_1000.tsv \
    --out-dir byt5_in --stem heldout_1000

# 2. Teach the job about the new stem. Line 81-83 hardcodes STEM, and line 242
#    calls make_byt5_input.py without --tsv, so both need a branch.
sed -i '83a [ "$HELDOUT" = "1" ] \&\& STEM="heldout_1000"' byt5_infer.slurm
sed -i 's|--out-dir byt5_in --stem "$STEM" --limit "$LIMIT"|--out-dir byt5_in --stem "$STEM" --limit "$LIMIT" --tsv "${TSV:-sighum_test_4200.tsv}"|' byt5_infer.slurm

# 3. Run it.
sbatch --export=ALL,HELDOUT=1,TSV=heldout_1000.tsv byt5_infer.slurm
```

Copy back `byt5_preds_segmentation-lemma-morphosyntax_heldout_1000.jsonl`, then
set the manifest cell:

```json
"g95_heldout/byt5": {
  "slm": {"pred_jsonl": "byt5_preds_segmentation-lemma-morphosyntax_heldout_1000.jsonl",
          "pred_name": "ByT5"}
}
```

and rebuild. **While you are there, capture §2.4's numbers from the same job.**

Expect only the L and L+M columns to populate — the `--no-surface` rule applies
to every system on this dataset, ByT5 included.

### 2.2 TransLIST measured rather than transcribed — *~4 hours, timeboxed*

The table currently carries published figures with `"reproduced": false`. To
measure them on our data:

```bash
# On a Lonestar6 login node.
cd $WORK
git clone https://github.com/rsingha108/TransLIST
cd TransLIST
# Fetch saved_models/ from the Google Drive link in the repo README.
```

**The documented stack is Python 3.7.3 / PyTorch 1.5.0 / CUDA 9.2. An A100 is
sm_80 and requires CUDA 11+, so "run on A100" does not work as written.** Run it
on CPU instead — 4,200 sentences is tractable, and the accuracy column becomes a
measurement on our data rather than a transcription.

```bash
conda create -n translist python=3.7 -y && conda activate translist
pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Input is SLP1, one sentence per line -- the same format as byt5_in/.
python interactive_module.py < sighum_test_4200.slp1.txt > translist_preds.txt
```

**Timebox it.** If the checkpoint will not load inside four hours, stop and keep
the published row with its `reproduced: false` footnote. Do not attempt a
PyTorch port. The throughput cell stays empty either way, with the CUDA-version
reason stated.

Then replace the manifest's `published` block with a `pred_jsonl` cell in the
same shape as ByT5's.

### 2.3 GPU wallclock throughput — *no new job needed*

The spec asks for wallclock; the GPU column is kernel-only. I checked whether
the CSVs already answer this: **they do not.** `us_per_sent_loop` and
`us_per_sent_kernel` are identical in every row of every sweep, so that column
excludes host time too.

The number does exist — both SLURM scripts wrap their run in `time`, and the
job logs are still on Lonestar6:

```bash
grep -A3 "^real" /home1/11503/njhavar/cushr/cushr_gpu/k64_bench.o*
grep -A3 "^real" /home1/11503/njhavar/cushr/cushr_cpu/cpu_bench.o*
```

Divide 119,503 by the GPU job's `real` seconds for an end-to-end figure — but
note that number **includes** the npz load, the CPU cross-check, and the 325 MB
dump, so it understates streaming throughput badly. For a clean measurement,
re-run one row without `--check` or `--dump-paths` and time it:

```bash
cd /home1/11503/njhavar/cushr/cushr_gpu
time ./cushr_batched "$DATA" --scorer biaffine --model "$MODEL" \
    --k4 twopass --K 32 --batch -1 --check 0 --csv k4_bench_G_k32_wall.csv
```

Report both, labelled distinctly. Do **not** rename either to the other.

### 2.4 ByT5 throughput and GPU memory — *free if you run §2.1*

Never recorded. The preds jsonl carries only `id, mode, words, lemmas, tags,
raw` — no timing. From the job that runs §2.1:

```bash
# Wallclock: the job log already has it.
grep -E "^real|elapsed" byt5_infer.o<jobid>

# Peak GPU memory: add this to byt5_infer.slurm after inference, inside the
# same python process that ran the model.
#   import torch; print("peak_gpu_MB", torch.cuda.max_memory_allocated()/1e6)
```

That fills ByT5's throughput and memory cells and puts a third point on the
Pareto plot — currently the plot has only two, which is thin for a "frontier".

### 2.5 SIGHUM-hackathon-test — *blocked on a source*

Not present in this repo or either external tree, and no distribution URL has
been identified. Before spending time here, establish that the split is publicly
available at all:

1. Check the SIGHUM / Sanskrit hackathon shared-task page for a test release.
2. If it exists, confirm it ships **gold surface segmentations** — without them
   it lands in the same `--no-surface` bucket as the held-out set (§3.2).
3. Confirm the sentences have SHR graphml, or can be run through SHR (§3.1).

If any of the three fails, descope it explicitly in the paper rather than
leaving the row blank without explanation.

### 2.6 A true DCS held-out set — *blocked on SHR*

§1.2 is the closest reachable approximation. To build the set as specified —
1,000 DCS sentences with no SHR restriction — you must first give cuSHR the
ability to parse new sentences:

1. Install the Sanskrit Heritage Reader (`gitlab.inria.fr/huet/Heritage_Platform`).
2. Run the 322,232 DCS sentences that currently have no graphml through it.
3. Point `ingest.py` at the new graphml directory and re-ingest.
4. Re-run `make_heldout_set.py` with the enlarged corpus.

This is a multi-week capability, not a Week 11 task. **It also unblocks §2.7 and
§2.8.** Until then, §3.1 is a limitations paragraph, not a gap.

### 2.7 GRETIL śāstra subset — *blocked on SHR, plus annotation*

Same blocker as §2.6, plus: a GRETIL technical text has no DCS gold at all, so
the ~500 sentences would need hand-correction by someone competent in śāstric
Sanskrit. Sequence, once SHR is installed:

1. Select the text (e.g. a section of the Kāśikā or the Sāṅkhyakārikā).
2. Segment it into sentences; SLP1-transliterate.
3. Run through SHR to obtain graphml.
4. Obtain gold: match against DCS where possible, hand-correct the remainder.
5. Ingest, then add as a third dataset in `results_manifest.json`.

Step 4 is the expensive one and needs a human annotator.

### 2.8 SHR's own beam — *optional in the spec*

Requires the same Heritage Platform install as §2.6. Nothing in this repository
invokes SHR; `ingest.py` only *consumes* pre-existing graphml. Reasonable to
leave descoped and say so.

---

## 3. Structural limits — state these in the paper

### 3.1 cuSHR's coverage is bounded by SHR's, not by annotation availability

The lattice **is** a Sanskrit Heritage Reader `.graphml`, and nothing in this
repository invokes SHR. `ingest/INGEST_METHODOLOGY.md:150` states the corpus is
"limited by how many sentences were run through SHR, not by annotation."

Concretely: **119,503 sentences have graphml; 322,232 DCS pickles do not.**
A sentence SHR has never parsed cannot be decoded at all, at any K.

This single fact explains datasets 2, 3 (partially), and 4, and baseline 4. It
is the structural finding of the week and deserves a paragraph in the paper's
limitations section — it is a real property of the system, not an excuse.

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

**Headline:** S-level F1 and Perfect Match; throughput and the CPU/GPU ratio;
recall-vs-K; the out-of-domain ceiling-relative result from §1.2.

**Appendix:** the full S/L/M ladder with the convention caveat; the memory
sweeps; fused-vs-twopass; the ncu profile.

Rationale: the L/M columns invite a direct comparison against ByT5 that the
ORACLE row shows to be invalid. Leading with S puts the comparison on the axis
we actually measure well; burying the ladder without its caveat would let the
table argue against us.

### Three numbers to raise explicitly

1. **The CPU baseline is 11× faster than the README claimed.** The speedup
   headline shrinks accordingly. Better found now than in review, and
   `cushr_cpu/README.md:55` should be corrected.
2. **The K=32→48 throughput cliff is entirely in K3** — a concrete optimisation
   target, and consistent with the register/occupancy trade from the branchless
   merge rewrite. Evaluate with a batched `ncu` profile, not per-sentence.
3. **Out-of-domain, cuSHR holds ~93% of its oracle in both domains.** The
   apparent L drop is the ceiling moving. This is a stronger claim than the raw
   number and should be made in these terms.

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
