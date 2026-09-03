"""Carve a held-out evaluation set out of the g95 corpus.

Why this exists
---------------
The paper needs a second dataset. The three obvious candidates -- a DCS
held-out sample, a GRETIL sastra subset, the SIGHUM hackathon split -- are all
out of reach: cuSHR cannot decode a sentence the Sanskrit Heritage Reader has
not parsed, because the lattice IS an SHR .graphml, and nothing in this repo
invokes SHR (see ingest/INGEST_METHODOLOGY.md: the corpus is "limited by how
many sentences were run through SHR, not by annotation"). So the only sentences
we can decode at all are the 119,503 already ingested.

What this set can and cannot measure
------------------------------------
It is held out honestly: every sentence here is outside the training split AND
outside the published 4,200, checked by DCS id rather than by row, so a
duplicated id cannot leak.

But there is NO surface-segmentation reference for it, and none can be built:

  * the DCS pickles carry `lemmas` and `cng` only, in IAST -- for sentence
    100178 they give ['ksip','mad','suta',...] where a surface reference needs
    'cikzepa me sutaH ...';
  * npz `surface_text` is a lossy reconstruction of our own gold path
    ('aaApatat' for 'aTa apatat', literal '?' for unresolved characters), and
    is our output, not an independent reference;
  * the `output` column of sighum_test_4200.tsv is a PUBLISHED segmentation
    that exists for those 4,200 sentences and no others.

So this set is scored with `eval_slm.py --no-surface`: the L / L+M ladder,
recall@K, throughput and memory. Word-level P/R/F1 and surface perfect match
stay a SIGHUM-test-only result. Do not synthesise an `output` column from the
gold path to fill those cells -- it would score cuSHR against its own lattice
gold and drive ORACLE to ~100.

Usage
-----
    python make_heldout_set.py --n 1000
"""
import argparse
import csv
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits95_ex4200.json")
    ap.add_argument("--index", default="../data/sentence_index_g95.json")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--tsv", default="sighum_test_4200.tsv",
                    help="the published set to stay disjoint from")
    ap.add_argument("--p-dir", default="../../SIGHUM_database_gold_path/DCS_pick")
    ap.add_argument("--graphml-dir", default="../../SIGHUM_database/After_graphml")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="heldout_1000.tsv")
    ap.add_argument("--ids-out", default="heldout_1000.ids.json")
    args = ap.parse_args()

    splits = json.load(open(args.splits))
    index = json.load(open(args.index))
    published = {r["DCS-ID"].strip() for r in
                 csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                                delimiter="\t")}
    train_ids = {index[i] for i in splits["train"]}
    dev_ids = {index[i] for i in splits.get("dev", [])}

    z = np.load(args.raw)
    gold_len = np.diff(z["gold_path_offsets"])

    # Disjointness is checked on the DCS id, not the row index: the same
    # sentence can appear at more than one row, and a row-level check would
    # let it leak.
    pool = []
    drop = {"published": 0, "train": 0, "dev": 0, "no_gold": 0,
            "no_pickle": 0, "no_graphml": 0}
    for i in splits["test"]:
        sid = index[i]
        if sid in published:
            drop["published"] += 1
        elif sid in train_ids:
            drop["train"] += 1
        elif sid in dev_ids:
            drop["dev"] += 1
        elif gold_len[i] <= 0:
            # The 4,056 corpus sentences with an empty gold span can never be
            # hit by any decoder; including them would understate recall
            # exactly the way the n vs n_gold denominator did in CP-5.
            drop["no_gold"] += 1
        elif not os.path.exists(os.path.join(args.p_dir, f"{sid}.p")):
            drop["no_pickle"] += 1
        elif not os.path.exists(os.path.join(args.graphml_dir,
                                             f"{sid}.graphml")):
            # eval_slm.py reads cng straight from the graphml; without it the
            # sentence cannot be scored at the M levels.
            drop["no_graphml"] += 1
        else:
            pool.append(i)

    print(f"test split           : {len(splits['test']):,}")
    for k, v in drop.items():
        print(f"  dropped {k:<12}: {v:,}")
    print(f"eligible pool        : {len(pool):,}")
    if len(pool) < args.n:
        raise SystemExit(f"pool of {len(pool):,} is smaller than --n {args.n}")

    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(pool), size=args.n, replace=False).tolist())
    rows = [pool[j] for j in pick]
    ids = [index[i] for i in rows]
    assert len(set(ids)) == len(ids), "duplicate id in the sample"

    # Same column layout as sighum_test_4200.tsv so every --tsv consumer reads
    # it unchanged. `output` is deliberately EMPTY -- see the module docstring.
    # Only eval_slm.py --no-surface may consume this file.
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["DCS-ID", "input", "output"])
        for i, sid in zip(rows, ids):
            w.writerow([sid, "", ""])
    json.dump(ids, open(args.ids_out, "w"), indent=1)

    print(f"\nwrote {args.out}  ({args.n:,} sentences)")
    print(f"wrote {args.ids_out}")
    print("  `output` is empty by design: no surface reference exists outside "
          "the\n  published 4,200. Score this file with "
          "`eval_slm.py --no-surface` only.")


if __name__ == "__main__":
    main()
