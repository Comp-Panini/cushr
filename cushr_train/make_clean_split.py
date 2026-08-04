#!/usr/bin/env python3
"""Build a training split that excludes an external benchmark, same size as before.

`sighum_test_4200.tsv` is a published test set, but our splits come from md5
bucketing on sentence index, so 3,241 of its sentences landed in our TRAINING
set. Any score we report on it is therefore training accuracy, not
generalisation (measured: +7.5 PM points on seen vs held-out sentences).

This writes an explicit split that removes every benchmark sentence from
training while keeping the training set the same size, so the retrained model
differs from the original in exactly one respect: which sentences it saw.

Backfill comes from the DEV bucket, never from test. Test therefore stays
bit-identical to the split every existing result was measured on, and the new
model remains directly comparable to `model75_ctx`. The cost is a smaller dev
set for checkpoint selection, which is the cheaper thing to give up.
"""
import argparse
import csv
import json

import numpy as np

from prepare import bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="sighum_test_4200.tsv")
    ap.add_argument("--index", default="../data/sentence_index_repaired.json")
    ap.add_argument("--raw", default="../data/cushr_data_repaired.npz")
    ap.add_argument("--out", default="splits_ex4200.json")
    ap.add_argument("--dev-pct", type=int, default=5)
    ap.add_argument("--test-pct", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8-sig"),
                               delimiter="\t"))
    index = json.load(open(args.index))
    pos = {s: i for i, s in enumerate(index)}
    exclude = {pos[r["DCS-ID"].strip()] for r in rows
               if r["DCS-ID"].strip() in pos}

    z = np.load(args.raw)
    has_gold = np.diff(np.asarray(z["gold_path_offsets"], dtype=np.int64)) > 0

    train, devp, test = [], [], []
    for s in range(len(index)):
        if not has_gold[s]:
            continue
        # Must match prepare.py:87-92 exactly -- bucket < dev_pct is DEV, the
        # next slice is TEST. Inverting these silently swaps the two splits and
        # every comparison against existing results becomes wrong.
        b = bucket(s)
        if b < args.dev_pct:
            devp.append(s)
        elif b < args.dev_pct + args.test_pct:
            test.append(s)
        else:
            train.append(s)

    orig_n = len(train)
    train_clean = [s for s in train if s not in exclude]
    dev_clean = [s for s in devp if s not in exclude]
    need = orig_n - len(train_clean)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(dev_clean))
    moved = [dev_clean[i] for i in order[:need]]
    dev_final = sorted(dev_clean[i] for i in order[need:])
    train_final = sorted(train_clean + moved)

    print(f"benchmark sentences excluded from training: "
          f"{len(exclude & set(train)):,}")
    print(f"train  {orig_n:,} -> {len(train_final):,}  "
          f"(backfilled {len(moved):,} from dev)")
    print(f"dev    {len(devp):,} -> {len(dev_final):,}")
    print(f"test   {len(test):,} -> {len(test):,}  (unchanged, still comparable)")
    assert len(train_final) == orig_n, "training size must be preserved"
    assert not (set(train_final) & exclude), "benchmark leaked into training"
    assert not (set(train_final) & set(dev_final)), "train/dev overlap"
    assert not (set(train_final) & set(test)), "train/test overlap"
    print("checks passed: no benchmark sentence in training, no split overlap")

    json.dump({"train": train_final, "dev": dev_final, "test": test},
              open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
