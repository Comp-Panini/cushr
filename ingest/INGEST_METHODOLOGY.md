# Gold-path ingestion: methodology, coverage, and residual failures

How the SIGHUM corpus becomes trainable data, why only half of it originally
carried a usable label, what was changed to raise that to three quarters, and
what is actually wrong with the quarter that still fails.

Every number here is either **measured** over the full 119,503-sentence corpus
or **sampled** with the sample size stated. Where a figure is an estimate, it
says so.

---

## 1. What ingestion has to produce

`ingest.py` turns each sentence into a lattice and a label.

**The lattice** comes from a Sanskrit Heritage Reader (SHR) `.graphml` file. Nodes
are candidate word analyses — a surface form with a lemma, a morphological tag
(`cng`), a chunk index, and a position. Edges mean *can directly follow*.
Segmenting a sentence is choosing a source-to-sink path.

**The label** comes from a Digital Corpus of Sanskrit (DCS) `.p` pickle: the
ordered sequence of words a human annotator says is correct. It is a *path*
description, not a set — first word, second word, and so on.

Training needs both. A lattice with no label is unusable for supervised
learning, and this is exactly what half the corpus was.

### Matching the two

DCS names words; SHR names nodes. They are joined on the triple

```
(chunk_no, lemma, cng)
```

`reconstruct_gold_path` (`ingest.py:302`) collects candidate nodes per gold word,
then runs a forward DP with backpointers. It returns `[]` — failure — at four
points:

| exit | meaning |
|---|---|
| `ingest.py:358` | a gold word matched no node at all |
| `ingest.py:369` | no candidate for gold word 0 is a source |
| `ingest.py:379` | no forward edge from position *i−1* to any candidate at *i* |
| `ingest.py:384` | chain complete but the last word is not a sink |

All four collapse to the same output: an empty path. `goldpathmask`
(`ingest.py:541`) is derived *from* the resolved path, so it is all-zero on
failure and **cannot distinguish these cases**. That is why diagnosing this
required re-running the resolution logic offline rather than reading the archive.

### The forced-DAG edge filter

Before anything else, `ingest.py:517-524` drops every edge that is not
`key == '1'` or not forward in node-id order. SHR uses `key` values of `-1`, `2`,
and `3` to attach auxiliary and alternative analyses; only `key == '1'` is the
traversable "can follow" relation. Keeping the others would make the graph
cyclic and the DP meaningless.

> **Methodological note.** An early version of this diagnosis omitted that filter
> and reported `n_sources: 0` for every sentence — impossible, since a finite DAG
> always has a source. That impossibility is what exposed the bug. Every
> diagnostic below replicates `ingest.py:517-524` exactly. Any analysis of this
> lattice that does not apply the filter is measuring a different graph.

---

## 2. Baseline: 49.45%

| | count |
|---|---:|
| sentences ingested | 119,503 |
| **gold path resolved** | **59,092 (49.45%)** |
| gold nodes flagged | 365,393 |

Nodes, edges, and features were produced for all 119,503. Only the *label*
failed. The ~46% figure sometimes quoted is the rate over the first 3,000
sentences (46.07%); the full-corpus rate is 49.45%.

### Ruling out the obvious explanation first

Before investigating the lattice, the annotation itself had to be cleared. A
250-sentence sample suggested missing gold files were not the cause; that sample
was too small to support the claim, so it was rechecked exhaustively:

| | |
|---|---:|
| `.graphml` on disk | 119,503 |
| `.p` on disk | 441,735 |
| graphml **without** a `.p` | **0** |
| unresolved sentences lacking a `.p` | **0** |
| sampled unresolved with parseable gold entries and lemmas | **400 / 400** |

Missing or corrupt annotation explains **none** of the loss. The `.p` directory
is a *superset*: 322,232 pickles have no corresponding graphml, so the corpus is
limited by how many sentences were run through SHR, not by annotation.

*(Caveat, stated because it is an inherited assumption rather than a proven one:
the graphml↔pickle join is by filename stem, which is what `ingest.py:502`
assumes. That DCS sentence *N* and SHR sentence *N* are the same text was not
independently verified. The circumstantial evidence is strong — 59,092 sentences
resolve to fully connected paths with chunks and lemmas agreeing — but it is not
proof.)*

### Where the failures actually were

Re-running the resolution logic offline on unresolved sentences, with the edge
filter applied:

| stage | share |
|---|---:|
| `broken_chain` | 77.0% |
| `no_source_start` | 13.3% |
| `word_unmatched` | 9.0% |
| `no_sink_end` | 0.7% |

### Root cause: orphan nodes

Inspecting individual sentences showed SHR attaching auxiliary morphological
analyses — typically the **verbal root of a participle**, carrying `position = -1`
and a negative `cng` — via non-`key=1` edges. The forced-DAG filter removes those
edges but keeps the nodes, so the nodes survive **with no edges at all**.

Verified directly in raw graphml: nodes 7 and 9 of sentence `115153` have **zero
in-edges before any filtering**.

DCS frequently names exactly those root analyses. So `(chunk, lemma, cng)`
matching *succeeds* — the word is found — but the node it points at lies on no
path, and the DP dies. The label was lost not because the word was unknown but
because the correct node was unreachable.

Measured over the corpus:

| | |
|---|---:|
| word nodes with `position = -1` | 292,028 / 4,249,149 (**6.87%**) |
| nodes whose only wiring is `SRC → n → SINK` | 11.3% |
| sentences containing at least one such node | 77.7% |

The second row is a side effect worth naming: `ingest.py:624-630` wires
super-source → every source and every sink → super-sink. An edge-less node is in
*both* sets, so it becomes a legal one-word "segmentation" of the entire
sentence. These degenerate paths are spurious labels.

---

## 3. The repair: 49.45% → 74.99%

### Rule

If a gold word's candidates are **all** edge-less, redirect it to nodes with the
**same `chunk_no` and byte-identical surface `word`** that do have edges
(`build_repair_index`, `ingest.py:278`; applied at `ingest.py:340`).

The rationale is that the orphan and its wired twin describe the same span of
text; SHR simply files the root analysis separately from the inflected one. The
redirect is conservative by construction:

- fires **only** when every candidate is unwired — a word with any usable node is
  untouched;
- targets **only** wired nodes, so a redirect can never land on another orphan;
- if no twin exists, candidates are left alone and the sentence fails as before.

### Result

| | original | repaired |
|---|---:|---:|
| resolved | 59,092 (49.45%) | **89,611 (74.99%)** |
| gold nodes | 365,393 | 584,604 |
| newly resolved | — | **+30,523** |
| lost | — | 4 |
| previously-resolved paths **altered** | — | **0** |

Redirect activity, summed independently from the 24 shard logs:

| | |
|---|---:|
| gold words redirected | 46,736 |
| sentences touched | 38,973 |
| sentences recovered | 30,523 (78.3% of touched) |

The shard-log total and the archive diff agree exactly on 30,523, which is a
useful cross-check: the two are computed by different code paths.

### The 4 lost sentences

`15665`, `204400`, `31867`, `35282`. Re-ingesting them both ways confirms the
repair caused it — they resolve without it and fail with it.

All four had **gold path length 1**: the degenerate `SRC → node → SINK` case
above. The lone node was simultaneously source and sink, so it formed a complete
one-word "segmentation" of the whole sentence. Redirecting to a wired twin, which
is not both source and sink, correctly breaks that. **These were spurious labels
being removed, not good labels being destroyed.**

### Validation

On a controlled 3,000-sentence comparison (same input, repair on vs off):

- all 14 lattice arrays — nodes, edges, `rowptr`/`colidx`, `topolevel`, features,
  surface text — **bit-identical**; only the three gold arrays differ;
- **0 paths lost, 0 previously-resolved paths altered**.

At full corpus scale, 0 altered holds and 4 are lost, all explained above. The
repaired archive is a **near-strict superset** of the original labelling.

The sharded/merged output was separately verified **bit-identical to a single
sequential run** across all 17 arrays including vocabulary ids — the merge
remaps ids, so this was the failure mode most likely to corrupt data silently.

### Operational notes

Getting this to run took four attempts and produced changes worth keeping:

- **`ingest.py`** had `repair_orphan_gold` as a parameter but nothing could reach
  it — no argparse, and `__main__` called `process_corpus` positionally. Added a
  CLI (`--repair-orphan-gold`, `--out`, `--index`, `--limit`).
- **Memory.** `process_corpus` accumulates the entire corpus in Python lists
  before packaging. On a 15.2 GB machine with ~900 MB actually available, a
  single-process run died at 66,000 / 119,503 after 42 minutes **with nothing
  saved** — packaging happens only after the loop.
- **Sharding alone does not fix that.** `parallel_ingest.py` tied shard count to
  worker count, so N workers still held the whole corpus between them at peak.
  Added `--shards`, decoupled from `--workers`, with `chunksize=1` so a worker
  writes one shard and frees it before taking the next. Peak RAM becomes
  `workers × shard-size`.
- **Resumability.** Added `--resume` (skip shards already on disk), `--merge-only`,
  and `--max-new N` (bound wall-clock per invocation). The merge is now skipped
  if any shard is missing rather than crashing on a partial set.

Reproduce with:

```bash
python parallel_ingest.py --shards 24 --workers 4 --max-new 5 \
    --keep-shards --resume --repair-orphan-gold \
    --out ../data/cushr_data_repaired.npz \
    --index ../data/sentence_index_repaired.json
```

Repeat until it merges. ~7 min per 5 shards, ~35 min total.

### Output

`data/cushr_data_repaired.npz`, `data/sentence_index_repaired.json`, and
`data/{form,lemma,morph,preverb}_vocabulary.txt`.

> **This is raw ingest output and not a drop-in replacement for
> `new_cushr_data_fixed_USE_THIS.npz`**, which is one pipeline stage later: it
> uses different key names (`col_idx` vs `colidx`, `topo_level` vs `topolevel`,
> `gold_path_mask` vs `goldpathmask`), carries reverse-CSR arrays ingest never
> emits (`in_row_ptr`, `in_col_idx`, `in_edge_id`, from `add_reverse_csr.py`), and
> has `node_word_length`. Post-processing must be re-run before training.

---

## 4. Why the remaining 25% still fails

29,892 sentences remain unresolved. Sample of 50, same offline replication of the
resolution logic:

| stage | count | share |
|---|---:|---:|
| `broken_chain` | 32 | **64%** |
| `no_source_start` | 12 | 24% |
| `word_unmatched` | 6 | 12% |

Median failing sentence has 6 gold words (range 2–21) — these are not degenerate
short cases.

### 4a. A hypothesis that was wrong

The traces show the DP reaching the end of one chunk and needing the next, which
suggests missing cross-chunk edges. **Measured, and false.** Unresolved sentences
average **610 cross-chunk edges** each; resolved ones average **331**. Zero
sentences in either group lack them entirely. The lattice is well connected. The
gold simply picks nodes that are not on a source-to-sink path.

### 4b. The real mechanism: dead ends

In **22 of 32** broken chains, every node the DP reached has **out-degree 0**:

```
[303363]  reached ('2', 6, 'aKilam')     needs ('3', 'tvayi')     reachable: []
[80506]   reached ('1', 8, 'aSvameDena') needs ('2', 'yat')       reachable: []
[76836]   reached ('2', 'kfzRalavaRam')  needs ('3', 'dAqime')    reachable: []
```

This is the orphan problem one level subtler. `build_repair_index` defines
*wired* as having **any** edge, in or out. A node with in-edges but no out-edges
passes that test, so the repair never fires — yet it is a dead end mid-sentence.

The `no_source_start` group is the mirror image: in **all 12**, the first gold
word's candidates have in-edges, so none can start a path. The traces show the
underlying disagreement plainly — DCS annotates a whole chunk as one word, SHR
wires only the sandhi split:

```
[307806] gold word 0: 'aDArmikAH'   sources: 'a'    + 'DArmikAH'
[325741] gold word 0: 'anaBimatam'  sources: 'an'   + 'aBimatam'
[181131] gold word 0: 'yUpAkzaH'    sources: 'yUpa' + 'akzaH'
[4568]   gold word 0: 'viqaNgA'     sources: 'viw'  ...
```

The unsplit form exists as a node but sits off the traversable path.

### 4c. `word_unmatched` (12%)

In **5 of 6**, the chunk exists and the `cng` matches but **the lemma does not**.
DCS and SHR disagree on lemmatization for that word. Not a wiring problem.

### 4d. The reachability repair: built, measured, recovers nothing

The natural extension is to stop treating *has any edge* as sufficient and
require the node to be usable as a path element. Implemented as
`--reach-repair` (`reconstruct_gold_path(expand_surface=True)`): on any sentence
the strict pass fails to resolve, retry with **every** gold word widened to all
wired nodes sharing its chunk and surface form. It runs strictly as a fallback,
so a sentence that already resolves is never re-derived and no existing label can
change.

**Result on 3,000 sentences: zero recovered.**

```
gold words widened:  2,053
fallback attempts:     860
fallback recovered:      0
```

The output archive is **bit-identical to the orphan-only archive across all 17
arrays**. The fallback fires on every unresolved sentence and does widen the
candidate sets; it simply never yields a connected path. No full-corpus run was
made, because it would reproduce `cushr_data_repaired.npz` byte for byte.

#### Why the projection was wrong

An earlier draft of this document predicted **+3,900 sentences (~78%)** from this
change. That estimate came from asking, for each stuck DP, whether a same-surface
twin with out-edges *exists* — true in 7 of 32 broken chains:

| group | twin exists | share |
|---|---:|---:|
| `broken_chain` (needs out-edges) | 7 / 32 | 22% |
| `no_source_start` (needs to be a source) | **0 / 12** | **0%** |

Twin existence is **necessary but not sufficient**, and treating it as predictive
was the error. Supplying a traversable node for gold word *i* only helps if the DP
can also *reach* it from gold word *i−1*; the twin has out-edges but typically no
in-edge from where the chain currently stands. Repairing one link does not make
the chain hold.

> **Methodological note.** The estimate for the *orphan* repair (§3) avoided this
> trap: `recovery_estimate.py` re-ran the full DP on redirected candidates and
> predicted 80.9% recovery against an actual 78.3%. The reachability estimate
> substituted a local existence check for that end-to-end test and was off by the
> entire effect. **Predict path-level outcomes by running the path algorithm, not
> by counting local properties of nodes on it.**

#### What the null result establishes

The residual failures are not "the right node was passed over." For these
sentences the gold path DCS describes has **no realization in SHR's lattice at
all**. The `no_source_start` evidence (0 of 12 twins) already pointed this way;
the null result generalises it to the broken-chain group.

This is consistent with the concrete traces in §4b: DCS annotates a whole chunk as
one word (`anaBimatam`), SHR only ever wires the sandhi split (`an` + `aBimatam`).
No redirection among existing nodes can produce a node that was never built.

Closing the gap therefore requires changing the lattice or the annotation, not the
resolution logic — adding DCS's segmentation as new nodes, or re-running SHR with
different sandhi settings. Neither is a change to `ingest.py`.

The `--reach-repair` code is retained despite recovering nothing: it is a cheap,
runnable refutation of the most obvious next idea.

---

## 5. Caveat on what the recovered labels mean

The redirect adopts **SHR's positioned analysis** in place of the DCS analysis for
30,523 sentences. For a participle, the model is now trained to prefer the
inflected surface node over the root node DCS named. That is a subtle
redefinition of the gold label, applied to roughly one training sentence in
three.

It is defensible — the two nodes describe the same span, and a segmenter's job is
to pick spans — but it has **not been validated against downstream accuracy**.

**The test that settles it:** evaluate a model trained on the repaired corpus
against the **original 59,092-sentence subset**. If F1 there drops relative to a
model trained only on those 59,092, the added labels are noise and the repair is
hurting. If it holds or improves, the extra 30,523 sentences are real signal.
Until that is run, treat 75% coverage as *more data of slightly different
provenance*, not simply *more of the same data*.

---

## 6. Summary

| stage | coverage | mechanism |
|---|---:|---|
| baseline | 49.45% | gold words match orphan nodes left edge-less by the forced-DAG filter |
| orphan redirect | **74.99%** | redirect to a same-surface wired twin (+30,523, −4) |
| reachability repair *(built, measured)* | **74.99%** | widen candidates to same-surface wired nodes — **+0**, see §4d |
| ceiling for resolution-side fixes | **~75%** | remainder has no realization in the lattice at all |

**75% is the ceiling for anything done inside `ingest.py`.** The two candidate
repairs are now separated by evidence rather than plausibility: the orphan
redirect gained 25.5 points, and the reachability repair — equally plausible
beforehand — gained nothing. What is left is not a resolution bug but a
disagreement between DCS's segmentation and the lattice SHR built.

Two routes past it, both outside this module:

1. **Change the lattice.** Re-run SHR with different sandhi settings, or inject
   DCS's unsplit word forms as nodes. Addresses the 24% `no_source_start` group
   directly.
2. **Change the corpus.** The corpus is not annotation-limited — every sentence
   has gold, and **322,232 pickles have no lattice at all**. Generating graphml
   for those would expand the corpus **3.7×**, against the ~25% still unresolved
   here. This is by far the larger prize, and it needs SHR runs, not ingest work.

Before either, the cheaper question is whether the 30,523 recovered sentences
actually help: run the §5 validation first.
