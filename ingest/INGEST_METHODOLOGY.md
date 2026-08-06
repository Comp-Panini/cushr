# Gold-path ingestion: methodology, coverage, and residual failures

How the SIGHUM corpus becomes trainable data, why only half of it originally
carried a usable label, the changes that raised that to **93.9%** and then
**94.5%**, and what is actually wrong with the remainder.

Several conclusions in earlier versions of this document were wrong and are
retracted in place rather than deleted — §4d's "no realization in the lattice",
§4b's reading of the `no_source_start` group, and §4c's reading of
`word_unmatched`. The retractions are kept visible because every one of them was
the same error: attributing to the corpus something the code was discarding.
See §6.

Every number here is either **measured** over the full 119,503-sentence corpus
or **sampled** with the sample size stated. Where a figure is an estimate, it
says so.

---

## 1. What ingestion has to produce

`ingest.py` turns each sentence into a lattice and a label.

**The lattice** comes from a Sanskrit Heritage Reader (SHR) `.graphml` file. Nodes
are candidate word analyses — a surface form with a lemma, a morphological tag
(`cng`), a chunk index, and a position. Edges mean *can be adjacent* — they are
undirected in the source file, and ingest orients them (see §1's filter).
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

`reconstruct_gold_path` (`ingest.py:366`) collects candidate nodes per gold word,
then runs a forward DP with backpointers. It returns `[]` — failure — at four
points:

| exit | meaning |
|---|---|
| `ingest.py:442` | a gold word matched no node at all |
| `ingest.py:452` | no candidate for gold word 0 is a source |
| `ingest.py:462` | no forward edge from position *i−1* to any candidate at *i* |
| `ingest.py:467` | chain complete but the last word is not a sink |

All four collapse to the same output: an empty path. `goldpathmask`
(`ingest.py:707`) is derived *from* the resolved path, so it is all-zero on
failure and **cannot distinguish these cases**. That is why diagnosing this
required re-running the resolution logic offline rather than reading the archive.

### The forced-DAG edge filter

Before anything else, the filter (`forward_edge_filter`) drops every edge that is
not in the selected `key` set or not forward in the chosen node order. Measured
over 300 graphml files:

| `key` | count | symmetric | what it is |
|---|---:|---:|---|
| `1` | 295,764 | **100%** | words that can be adjacent, spans not touching |
| `2` | 92,108 | **100%** | adjacent **through a sandhi merge**, spans overlap by one char |
| `-1` | 1,339 | 0% | `position = -1` participle-root orphans |
| `3` | 648 | 0% | (tiny) |

**`key=1` and `key=2` are both genuine adjacency**, and both are symmetric.
Earlier versions of this document said only `key=1` was traversable and the rest
were "auxiliary and alternative analyses". That was wrong for `key=2`: it is the
sandhi junction, where the final glyph of one word and the initial of the next
fuse into a single character (`tejaH` + `niDim` → `tejoniDim`), so the two spans
overlap by exactly that character. Dropping it deletes every transition across a
sandhi merge. See §4f.

`key=-1` and `key=3` are directed, tiny, and attach the auxiliary analyses
§3's repair exists to redirect away from; they stay excluded.

**The relation is symmetric, and that changes what this filter is.** Measured over
3,000 graphml files: all **2,837,548** `key=1` edges have a reciprocal partner —
100%, with zero self-loops; `key=2` is likewise 100% symmetric. Both `1 → 38` and
`38 → 1` are present. So it is not a directed *can follow* relation, as this
document previously claimed; it is an undirected compatibility relation, and the
second half of the filter does not remove spurious edges — it **chooses an
orientation**.

Which order it orients along is therefore a real modelling decision, not a
formality. Ingest offers two, via `--edge-order`:

| | orientation key | notes |
|---|---|---|
| `id` *(default)* | node id | original behaviour; reproduces every archive below |
| `position` | `(chunk_no, position, position+length_word, node_id)` | sentence order, the same key `surface.py` and `char_start` already use |

Node id is **not** sentence order — see §4b, which is where the consequences land.

Both keys are **injective** (node id is unique, and it is the last component of
the `position` key), so each is a strict total order and two nodes can never tie.
That is what makes the result a DAG by construction, and it means **exactly one
direction of every symmetric pair survives under either ordering** — no edge is
ever lost to the choice. Confirmed on the §4e A/B: word-to-word edge count is
identical at **1,418,774** either way. Only the super-source/sink wiring differs,
since which nodes are sources and sinks does change.

> **Methodological note.** An early version of this diagnosis omitted that filter
> and reported `n_sources: 0` for every sentence — impossible, since a finite DAG
> always has a source. That impossibility is what exposed the bug. Every
> diagnostic below replicates the filter exactly, which is now enforced by
> construction: `verify.py` and `viz/build_db.py` call the same
> `forward_edge_filter` rather than each keeping a hand-copied version. Any
> analysis of this lattice that does not apply the filter is measuring a
> different graph.

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
the graphml↔pickle join is by filename stem, which is what `ingest.py:591`
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

The second row is a side effect worth naming: `ingest.py:690-699` wires
super-source → every source and every sink → super-sink. An edge-less node is in
*both* sets, so it becomes a legal one-word "segmentation" of the entire
sentence. These degenerate paths are spurious labels.

---

## 3. The repair: 49.45% → 74.99%

### Rule

If a gold word's candidates are **all** edge-less, redirect it to nodes with the
**same `chunk_no` and byte-identical surface `word`** that do have edges
(`build_repair_index`, `ingest.py:340`; applied at `ingest.py:422`).

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

## 4. Why the remaining 25% still fails *(superseded by §4e)*

This section diagnoses the residual left by §3's orphan repair. Two of its
conclusions turned out to be wrong; §4e replaces them and cuts the residual from
25% to 6%. It is kept in full because the reasoning that misled it is the point.

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

#### Worked example: 100104, and why the dead ends exist

Four chunks: `pramattezu aBiGAtam / hi / kuryAt SAlvaH / naraDipaH`. All 7 gold
words match a node. The DP still dies:

```
i=0  'pramad/-190' -> n3,7,9 all orphans -> REDIRECTED to n1,2,4,5,6,8   ok
i=1  'aBiGAta/69'  -> n38, reached via 1->38                             ok
i=2  'hi/2'        -> n23, reachable from {38}: []                       FAIL
```

n38 is `aBiGAtam`, chunk 1, position 10 — mid-sentence — with **in-degree 30,
out-degree 0**. But the edge the DP wanted exists in the raw graphml:

```
38 -> 23  {'key': 1}      # aBiGAtam (ch1) -> hi (ch2)
```

It is deleted because `38 >= 23`. **Every** `key=1` out-edge of n38 goes to a
lower id, so the node loses all of them.

The cause is that node ids are not sentence order. SHR numbers the sandhi-split
alternatives first (n19 `aBi`, n21/22 `GAtam`, both chunk 1) and the merged
unsplit analysis last (n38/39 `aBiGAtam`) — *after* the chunk-2/3/4 nodes n23–n37.
Orienting by id then reverses genuine edges, and does so preferentially on
exactly the merged-form nodes DCS likes to annotate. Under `--edge-order position`
the sentence resolves: `1 → 38 → 23 → 24 → 25 → 26 → 27`.

This is measured in §4e, and it is why the conclusions in §4d needed revising.

The `no_source_start` group is the mirror image — more literally than was
realised when this was written; see §4e, which shows it is the *same* bug and
retracts the reading below. In **all 12**, the first gold word's candidates have
in-edges, so none can start a path:

```
[307806] gold word 0: 'aDArmikAH'   sources: 'a'    + 'DArmikAH'
[325741] gold word 0: 'anaBimatam'  sources: 'an'   + 'aBimatam'
[181131] gold word 0: 'yUpAkzaH'    sources: 'yUpa' + 'akzaH'
[4568]   gold word 0: 'viqaNgA'     sources: 'viw'  ...
```

The unsplit form exists as a node but sits off the traversable path. *(The
inference drawn here — that this is a DCS/SHR segmentation disagreement — was
wrong. It sits off the path because its edges were oriented backwards. §4e.)*

### 4c. `word_unmatched` (12%)

In **5 of 6**, the chunk exists and the `cng` matches but **the lemma does not**.
DCS and SHR disagree on lemmatization for that word. Not a wiring problem.

*(Retracted in part. The observation is right; "DCS and SHR disagree" was the
wrong inference. For a large share of these the two agree perfectly and
`lemma_candidates` just never generated the spelling — the anusvara and `f`-grade
families of §4f. What remains genuinely unmatched is the compound/suppletion
tail in §4g.)*

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

#### What the null result establishes — and what it does not

The null result is sound but was over-read. What it establishes is narrow: **no
redirection among the nodes present can fix these sentences.** The conclusion
drawn from it — that the gold path "has no realization in SHR's lattice at all,"
and hence that ~75% is a ceiling — was wrong for the `broken_chain` **and**
`no_source_start` groups, which together were 88% of the residual. §4e measures by
how much.

The error was scope. `--reach-repair` widens *candidate node sets*. If the failure
is a **deleted edge**, no choice of nodes can help, so the fallback was structurally
incapable of detecting the actual cause and its silence carried no information
about it. For 100104 the gold path is fully realizable in `key=1` edges — ingest
oriented two of them backwards.

What survives unchanged is only the narrow claim: no redirection produces a node
that was never built. The `no_source_start` reading in §4b — that DCS annotates
`anaBimatam` where SHR wires only `an` + `aBimatam` — does **not** survive. Those
nodes exist and are reachable; §4e shows that group all but disappears under a
corrected edge orientation.

The `--reach-repair` code is retained despite recovering nothing: it is a cheap,
runnable refutation of the most obvious next idea, and a standing reminder that a
null result bounds only the mechanism actually tested.

### 4e. Edge orientation: `--edge-order position`

Per §1, `key=1` is symmetric and the filter chooses a direction. Orienting by
`(chunk_no, position)` instead of node id stops genuine edges being reversed.

Controlled A/B, 3,000 sentences, `--repair-orphan-gold` on in both, nothing else
changed:

| | `id` | `position` |
|---|---:|---:|
| resolved | 2,140 (71.33%) | **2,779 (92.63%)** |
| newly resolved | — | **+639** |
| paths **lost** | — | **0** |
| paths **altered** | — | **3** |
| sentences dropped as cyclic | 0 | **0** |
| gold nodes flagged | 14,575 | 18,538 |

Edge counts, to make precise what "reorientation" costs: **word-to-word edges are
identical at 1,418,774** under both orderings, as §1 argues they must be. The
totals differ (1,465,888 vs 1,467,704) only in boundary wiring — `SRC→x` goes
22,454 → 23,354 and `x→SINK` 24,660 → 25,576, because the source and sink sets
change. Nothing is discarded; edges are pointed the other way.

Array-level diff of the two archives: all 11 node/surface arrays
(`node_features`, `node_position`, `node_chunk`, `node_form_id`, `node_lemma_id`,
`node_preverb_id`, `node_length`, `node_char_start`, `sentenceoffsets`,
`surface_text`, `surface_text_offsets`) **bit-identical**. `rowptr`/`colidx` differ
by construction — that is the change — as does `topolevel`, which is recomputed by
longest path from the surviving edges. The invariant the C++ loader asserts
(`lattice.cpp:216`, `topo_level[u] < topo_level[v]` on every edge) holds in both,
with 0 violations across 1.47M edges.

**The 3 altered paths.** All three differ only in the final node, and in every case
the two nodes are the same word: `samAhitaH`, same chunk, same position, same
`cng=29`, lemma spelled `samAhita` vs `samaahita` — one transliteration
convention against another, which `lemma_cng_candidates` already treats as
equivalent. SHR emits both as parallel nodes; `reconstruct_gold_path` breaks the
tie with `min(ends)`, and a changed sink set moves it to the lower-id twin. These
are duplicate-node tie-breaks, not relabellings.

Reproduce the A/B with:

```bash
python ingest.py --limit 3000 --repair-orphan-gold --edge-order id       --out /tmp/id.npz  --index /tmp/id_index.json
python ingest.py --limit 3000 --repair-orphan-gold --edge-order position --out /tmp/pos.npz --index /tmp/pos_index.json
```

and the full corpus with `parallel_ingest.py --edge-order position` (same shape as
the §3 command). **Shards from different `--edge-order` settings must not share a
`--shard-dir`** — unlike the repair flags, this one changes the emitted edges, and
`--resume` matches on filename only.

#### Full corpus

All 119,503 sentences, `--shards 24 --workers 4`, 22 minutes:

| | `id` (§3) | `position` |
|---|---:|---:|
| **gold path resolved** | 89,611 (74.99%) | **112,200 (93.89%)** |
| newly resolved | — | **+22,616** |
| lost | — | **27** |
| altered | — | **20** |
| gold nodes flagged | 584,604 | 732,669 |
| dropped as cyclic | 0 | **0** |

All 11 node/surface arrays are bit-identical to `cushr_data_repaired.npz`. Over
61,189,928 edges: **0** violations of `topo_level[u] < topo_level[v]`, and every
edge stays inside its sentence block. 1,293,970 edges now point backwards in node-id
space — expected, and consumed by nothing (every decoder sweeps by topo level).

**The 27 lost are spurious labels, as in §3.** In all 27, the DCS gold does not
cover the whole sentence: 18 stop before the last chunk, and the other 9 stop
partway through the final chunk, leaving text like `EH`, `na`, `SOdreRa`
unannotated. They resolved under id-order only because the last gold node's real
successors were all lower-id and got deleted, making it a **spurious sink** — the
same failure mode as §3's degenerate `SRC → node → SINK`, one level up. Under
`position` the node keeps its successors, is correctly not a sink, and the
incomplete annotation is correctly rejected.

```
[93167]  gold = 2 words, both chunk 1; sentence has chunks 1-2
         last gold node n31 'avyayIBAvam' (ch1 pos 9)
         id-order successors: none  -> "sink", path accepted
         position successors:  n9 'na'(ch2), n10 'api'(ch2), ... -> not a sink
```

**The 20 altered are not relabellings.** Every one covers an *identical*
`(chunk, position, surface)` sequence at every step — same segmentation, a
different duplicate node for it. These are SHR emitting the same analysis twice
under two transliteration spellings (`samAhita` / `samaahita`), which
`lemma_cng_candidates` already treats as equivalent; `min(ends)` breaks the tie
differently once the sink set changes. **0 of 112,200 resolved paths describe a
different segmentation.**

#### What is left, and a second correction to §4b

Residual failures after reorientation, 300-sentence sample of the 7,303 still
unresolved, classified the same way as §2:

| stage | before (of 29,892) | after (of 7,303) | absolute |
|---|---:|---:|---|
| `word_unmatched` | 12% | **64.3%** | ~3,600 → ~4,700 |
| `broken_chain` | 64% | 27.7% | ~19,100 → ~2,000 |
| `no_sink_end` | ~0% | 7.7% | — → ~560 |
| `no_source_start` | 24% | **0.3%** | ~7,200 → **~22** |

**The `no_source_start` group was also an orientation artifact.** §4b explained it
as DCS annotating a whole chunk (`anaBimatam`) where SHR only wires the sandhi
split (`an` + `aBimatam`) — a genuine annotation disagreement. That reading was
wrong, and reorientation eliminates the group almost entirely: ~7,200 sentences
down to roughly 22.

The mechanism is the exact mirror of the dead end, which §4b half-saw when it
called this group "the mirror image". The merged unsplit node is numbered high, so
under id-order every one of its `key=1` edges to lower-numbered nodes was kept as
an *in*-edge — giving it in-degree > 0 and disqualifying it as a source. Under
position order those same edges point outward from it, it correctly has in-degree
0, and it starts the path. The unsplit form was never "off the traversable path";
ingest had pointed its edges the wrong way.

What remains is dominated by `word_unmatched` — §4c's finding, that the chunk and
`cng` match but the lemma does not. That is a real DCS/SHR lemmatization
disagreement and no edge orientation can touch it.

> **Methodological note, again.** A 1,500-sentence pilot of this change measured
> **0 lost and 0 altered**. The full corpus gives 27 and 20. The pilot was not
> wrong, it was small — and both quantities are rare events, precisely what a
> pilot under-observes. The §4d note says to predict path-level outcomes by
> running the path algorithm; the corollary is to run it at full scale before
> quoting a zero. Note also that the pilot's zeros were *optimistic in appearance
> only*: inspecting the 27 showed them to be spurious labels being removed, which
> is a better outcome than the zero suggested.

### 4f. Two more things ingest was cutting

§4e left 7,303 unresolved, attributed largely to `word_unmatched` — "a real
DCS/SHR lemmatization disagreement that no edge orientation can touch." Wrong
again, in both halves. There were two further cuts.

**Cut 1: the lemma normalizer was missing two productive alternations.**

| gold (DCS) | lattice (SHR) | alternation |
|---|---|---|
| `saMjaya` | `saYjaya` | anusvara → homorganic nasal (palatal) |
| `puMgava` | `puNgava` | anusvara → homorganic nasal (velar) |
| `vAr` | `vf` | vrddhi `Ar ↔ f` |
| `mard` | `mfd` | guna `ar ↔ f` |

`_retroflex_variants` covered only `M→m`, `Md→nd`, `Mb→mb`; `_guna_variants`
covered `i/u → e/o` but **not `f`**, one of the commonest root vowels in the
language. The lemmas never disagreed — `lemma_candidates` simply never generated
the spelling the lattice uses. Added as `_anusvara_variants` and
`_r_grade_variants` under `--lemma-variants extended`.

**Cut 2: `key=2` edges were being deleted** (see §1). Every one of 92 sampled
`broken_chain` failures where all lemmas matched was a gold pair overlapping by
one character, with a `key=2` edge sitting in the raw file:

```
[413987] tejaH (ch4 p0+5) -> niDim (ch4 p4)    28->32 {'key': 2}   deleted
[135163] udyamaH (ch3 p0+7) -> arTa (ch3 p6)   20->21 {'key': 2}   deleted
```

#### The non-obvious part: where sources and sinks come from

Traversing `key=2` naively makes things **worse** — on a 400/400 sample,
resolution collapses from 400 to 98. Extra edges cannot break a path that already
existed; they break the *endpoint conditions*. A `key=2` edge is sandhi-internal,
so it gives interior nodes both in- and out-edges, and the first gold word stops
being a source while the last stops being a sink.

The fix is to keep the two graphs separate:

- **traversable** = `key ∈ {1,2}` — `succ`, `build_repair_index`, the gold DP,
  `topolevel`, and the **emitted** `rowptr`/`colidx`;
- **structural** = `key = 1` only — used for *nothing* but `sources`/`sinks`.

Sentence boundaries are a `key=1` property; sandhi joins are not.

| configuration | unresolved recovered | resolved kept |
|---|---:|---:|
| §4e baseline | 0 / 400 | 400 / 400 |
| + lemma families | 91 (22.8%) | 400 / 400 |
| + `key=2` naively | 49 (12.2%) | **98 / 400** |
| **+ `key=2`, endpoints from `key=1`** | **182 (45.5%)** | **400 / 400** |

The emitted lattice **must** carry the `key=2` edges: `cushr_train/dataset.py`
looks each consecutive gold pair up in the CSR and drops the sentence when the
edge is missing, so resolving gold over unemitted edges would recover nothing
downstream. Edge count grows ~29% (1.47M → 1.89M on 3,000 sentences).

#### Measured, 3,000-sentence A/B

| | §4e | + both fixes |
|---|---:|---:|
| resolved | 2,779 (92.63%) | **2,836 (94.53%)** |
| paths lost | — | **0** |
| paths altered | — | 9 |
| cyclic | 0 | **0** |
| node/surface arrays | — | **bit-identical** |
| topo invariant | 0 violations | **0 violations** |
| gold paths traversable in emitted CSR | 2,779 / 2,779 | **2,836 / 2,836** |

All 9 altered paths keep an **unchanged lemma sequence**; what moves is the span
boundary (`prayata` → `prayatam`), which is exactly the ambiguity `key=2` exists
to represent, plus the orphan redirect landing differently now that more nodes
count as wired.

#### Full corpus

119,503 sentences, ~25 min, 78,847,461 edges (+28.9%):

| | §4e | + both fixes |
|---|---:|---:|
| **gold path resolved** | 112,200 (93.89%) | **115,447 (96.61%)** |
| newly resolved | — | **+3,247** |
| lost | — | **0** |
| altered | — | 373 (0.33%) |
| cyclic | 0 | **0** |

All 11 node/surface arrays bit-identical; 0 topo violations over 78.8M edges;
every gold path traversable in the emitted CSR (115,447 / 115,447).

**The 373 altered paths, audited.** 87.9% keep the lemma sequence, but only 30%
keep the surface sequence — `key=2` genuinely re-cuts spans at sandhi junctions.
Scoring each altered path's token-level agreement with DCS's own `(lemma, cng)`:
**337 unchanged, 31 worse, 5 better**. So a net 26 sentences (0.02% of resolved)
agree slightly less with DCS than before, against +3,247 recovered. Stated
because it is a real cost, not because it changes the verdict.

> **Two bogus equivalences caught by the altered-path check, not by the coverage
> number.** Both raised coverage by *nothing* and silently mislabelled:
> composing `_r_grade_variants` with itself derives `arTa → fTa → ArTa`,
> asserting `ar ≡ Ar`, and it relabelled 19 gold paths from `arTaH` to `ArTaH`.
> Including samprasarana (`ra ↔ f`) equated `astra` "weapon" with `astf`
> "thrower". Fixes: apply the new families once, never to their own output, and
> keep guna/vrddhi only. **A widening that recovers nothing can still do damage,
> so "paths altered" has to be inspected even when coverage is flat.**

### 4g. What is left, and why 100% is not reachable

Attribution of 400 still-unresolved sentences after §4f:

| | share | fixable? |
|---|---:|---|
| lemma not found (compound / suppletion) | 39.8% | open-ended — `lokapAla` vs `lokapAlatva`, `dfS` vs suppletive `paS` |
| **gold does not cover its own sentence** | **7.2%** | **never** |
| `cng` placeholder `= 1` | 5.2% | deferred |
| lemma OK, `cng` mismatch | 1.5% | deferred |

The 7.2% is DCS annotating only part of the sentence: its own `sentence` field
holds more tokens than its `lemmas` list. Sentence `1001` is
`Cade hrasvas tElapuzpas tulyas tu rasavIryataH` — six chunks — and DCS annotates
five, leaving `rasavIryataH` unanalysed. No resolution logic can invent it. Those
sentences are correctly rejected, exactly like the 27 §4e drops.

Corpus-wide that hard floor is ~0.2–0.5%, so **~99.5% is the ceiling, not 100%.**

Measured after the full-corpus §4f run: **4,056 sentences (3.39%) remain
unresolved.** On the SIGHUM 4,200 benchmark the residue is **4 sentences
(0.10%)** — see `cushr_train/PAPER_COMPARISON.md` §2, where this stopped being
the binding constraint on the headline metric.

The two `cng` items are deferred on purpose: both require dropping part of the
match key, which is the first change in this line of work that could match the
*wrong* node, and §4f shows what that costs. They need their own paths-altered
measurement first.

---

## 5. Caveat on what the recovered labels mean

The redirect adopts **SHR's positioned analysis** in place of the DCS analysis for
30,523 sentences. For a participle, the model is now trained to prefer the
inflected surface node over the root node DCS named. That is a subtle
redefinition of the gold label, applied to roughly one training sentence in
three.

It is defensible — the two nodes describe the same span, and a segmenter's job is
to pick spans — but it has **not been validated against downstream accuracy**.

> **It has now been measured, and it does cost something — just not on
> segmentation.** `cushr_train/PAPER_COMPARISON.md` §3 scores our gold path
> against DCS's published annotation. Segmentation agreement is 98.00%, but
> **lemma agreement is only 70.07% and lemma+tag 68.18%**, precisely because the
> redirect substitutes SHR's analysis for DCS's on participles (`kf` → `kfta`).
> So the label redefinition is invisible to a segmentation metric and caps any
> lemma or morphology metric at ~70. If cuSHR is ever evaluated on
> lemmatisation or tagging as a task, this repair — not the model — is the first
> thing to revisit.

**The test that settles it:** evaluate a model trained on the repaired corpus
against the **original 59,092-sentence subset**. If F1 there drops relative to a
model trained only on those 59,092, the added labels are noise and the repair is
hurting. If it holds or improves, the extra 30,523 sentences are real signal.
Until that is run, treat the added coverage as *more data of slightly different
provenance*, not simply *more of the same data*.

This caveat applies with **less** force to §4e's increment than to §3's. The
orphan redirect substitutes SHR's positioned analysis for the DCS one, genuinely
redefining the label. Reorientation does not: it changes which edges exist, and
all 20 paths it altered describe an identical `(chunk, position, surface)`
sequence. Its 22,616 new labels are ordinary gold paths that were unreachable, not
relabelled ones.

**This has now been run** — `cushr_train/GOLD94_EDGE_ORDER.md`. `hybrid_tag`,
8 epochs, evaluated on the constant gold49 subset (2,885 sentences):

| model | F1 | PM |
|---|---:|---:|
| gold49-trained | 0.8831 | 0.5450 |
| gold75-trained | 0.8827 | 0.5324 |
| **gold94-trained** | **0.8865** | **0.5529** |

93.9% coverage costs nothing on the original distribution and is slightly ahead
of both predecessors. The two increments also behave differently on their own
recovered sentences: the orphan-repair set scores 0.0642 F1 below its baseline,
the reorientation set only 0.0056 below — consistent with the argument above that
one redefines labels and the other merely makes existing ones reachable. Single
seed, single featurizer; see that file's caveats.

---

## 6. Summary

| stage | coverage | mechanism |
|---|---:|---|
| baseline | 49.45% | gold words match orphan nodes left edge-less by the forced-DAG filter |
| orphan redirect | **74.99%** | redirect to a same-surface wired twin (+30,523, −4) |
| reachability repair *(built, measured)* | **74.99%** | widen candidates to same-surface wired nodes — **+0**, see §4d |
| `--edge-order position` | **93.89%** | orient the symmetric relation by sentence order instead of node id (+22,616, −27 spurious) — §4e |
| `--edge-keys 12 --lemma-variants extended` | **96.61%** | traverse `key=2` sandhi edges with endpoints still from `key=1`, and add the anusvara / `f`-grade lemma families (+3,247, −0) — §4f |

**The 75% ceiling claim was wrong, and so was the reasoning that replaced it.**
Four candidate repairs are now separated by evidence rather than plausibility: the
orphan redirect gained 25.5 points, the reachability repair — equally plausible
beforehand — gained nothing, the edge reorientation gained a further 18.9, and
§4f's two cuts gained 2.7 more. Coverage went 49.45% → **96.61%** without a
single change to the corpus.

The recurring error is worth naming once, because it has now happened three
times. Each residual was explained as **a property of the data** — "the gold path
has no realization in the lattice", "DCS annotates the whole chunk where SHR
wires the split", "DCS and SHR disagree on lemmatization". Each time it was
**something ingest was discarding**: an edge oriented backwards, an edge class
dropped wholesale, a spelling the normalizer never generated. The data was
mostly fine. Before attributing a residual to the corpus, check what the code
threw away.

The pattern across all three is worth stating, because it cost two wrong
conclusions. Both errors came from reasoning about **nodes** when the object that
determines a path is the **edge set**: §4d predicted recovery by counting node
properties, and then read a node-level null result as a statement about the
lattice as a whole. The failure was an edge the filter deleted, and neither
analysis was looking at edges.

What remained after §4e (7,303 sentences, 6.11%) was described here as genuine
DCS/SHR disagreement needing a different lattice. §4f cut it further still, and
§4g gives the current breakdown. The residue that is now genuinely unfixable is
small and specific: DCS annotations that do not cover their own sentence, ~0.2–0.5%
of the corpus. **~99.5% is the ceiling; 100% is not reachable.**

Two routes past it, both outside this module:

1. **Change the lattice or reconcile lemmas.** The residual is now 64%
   `word_unmatched` — chunk and `cng` agree, the lemma does not. A normalization
   table over DCS↔SHR lemma conventions is the cheap first probe; §4c's sample
   suggests it is a spelling/convention gap rather than a genuine analysis
   disagreement, but that has not been measured at scale. *(The
   `no_source_start` group this bullet used to target no longer exists — §4e.)*
2. **Change the corpus.** The corpus is not annotation-limited — every sentence
   has gold, and **322,232 pickles have no lattice at all**. Generating graphml
   for those would expand the corpus **3.7×**, against the ~25% still unresolved
   here. This is by far the larger prize, and it needs SHR runs, not ingest work.

The §5 validation has been run for the reorientation increment
(`cushr_train/GOLD94_EDGE_ORDER.md`): on the held-constant gold49 subset the
gold94 model scores 0.8865 F1 against gold49's 0.8831 and gold75's 0.8827, so the
added coverage costs nothing. That result also covers the orphan increment
transitively, since gold94 contains it.
